"""Levenberg-Marquardt for PDE least-squares problems.

The solver is an ``eqx.Module`` whose configuration fields are static. Each
iteration:
  - solves the damped Gauss-Newton system in JAX with a selectable dense
    factorization;
  - evaluates the trial point's residual only (no Jacobian) to decide
    acceptance;
  - adapts the damping by Nielsen's rule with step-norm clipping.

The damping diagonal is supplied by an [XScale][diff_soup.inference.fitting.scales.XScale]
object (default [JacScale][diff_soup.inference.fitting.scales.JacScale]), and any extra
least-squares rows come from the problem's
[Regularizer][diff_soup.inference.fitting.regularization.Regularizer] via its ``residual``
method.  Construct the solver and call [LevenbergMarquardt.solve][diff_soup.inference.solvers.levenberg_marquardt.LevenbergMarquardt.solve]::

    solver = LevenbergMarquardt(max_steps=200)
    result = solver.solve(problem, callbacks=(my_callback,))
"""

from __future__ import annotations

import time
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._distributed import any_process, assert_decision_value, local_value
from diff_soup._errors import BackendError, ConfigurationError
from diff_soup.inference._abc import OptSolver
from diff_soup.inference.fitting.bounds import projected_gradient_norm
from diff_soup.inference.fitting.callbacks import StopFit
from diff_soup.inference.fitting.result import EvalCounts, OptResult, StepInfo, TerminationStatus
from diff_soup.inference.fitting.scales import JacScale, XScale
from diff_soup.inference.solvers._loop import cost_threshold, trial_valid
from diff_soup.inference.solvers._termination import (
    DEFAULT_COST_ATOL,
    DEFAULT_COST_TOL,
    DEFAULT_STEP_TOL,
    absolute_tolerance,
    gradient_tolerance,
    model_held_the_step,
    relative_tolerance,
    stall_reason_of,
    step_budget,
)
from diff_soup.inference.solvers.factorization import (
    cholesky_factor,
    solve_augmented_qr,
    solve_factorized_normal,
)
from diff_soup.inference.solvers.tangent import (
    jacfwd_from_pushforward,
    linearize_value,
)

# A failed evaluation of the trial predicate ``problem.is_valid``: the compiled program
# raised at run time, a diff_soup backend failed, or a floating-point trap fired.
# A predicate that breaks its contract (a ValueError) propagates.
_PREDICATE_FAILURES = (jax.errors.JaxRuntimeError, BackendError, FloatingPointError)

# A failed evaluation of the residual programs at a trial or accepted point.
# They also catch ValueError: the programs already ran at the current point with
# arrays of the same shapes, and JAX's fast dispatch path raises ValueError
# for a compiled program that fails on a later call. A TypeError,
# AttributeError or KeyError is a programming error and propagates.
_EVALUATION_FAILURES = _PREDICATE_FAILURES + (ValueError,)


# ---------------------------------------------------------------------------
# Gauss-Newton step (pure JAX, runs on GPU for small n_params)
# ---------------------------------------------------------------------------


def _solve_spd(matrix: jax.Array, rhs: jax.Array, linear_solver: str) -> jax.Array:
    """Solve a dense damped normal equation with the requested factorization."""
    if linear_solver == "solve":
        return jnp.linalg.solve(matrix, rhs)
    return solve_factorized_normal(cholesky_factor(matrix), rhs)


def _gn_step(
    J: jax.Array,
    r: jax.Array,
    lam: float,
    max_step_norm: float,
    d: jax.Array,
    linear_solver: str = "solve",
):
    """Compute damped Gauss-Newton step with optional clipping.

    Args:
        J: Jacobian of shape ``(m, n)``.
        r: Residual of shape ``(m,)``.
        lam: Levenberg-Marquardt damping parameter.
        max_step_norm: If positive, clip the step to this norm.
        d: Positive damping diagonal of length ``n`` (from the scale object).
            The damped normal equations use ``D = diag(d)``.
        linear_solver: ``"cholesky"`` or ``"solve"`` for the damped normal
            equations, or ``"qr"`` for the augmented least-squares system.

    Returns ``(delta, dnorm, pred_red, Jtr, grad_norm, JtJ)``.
    """
    Jtr = J.T @ r
    grad_norm = jnp.linalg.norm(Jtr)
    JtJ = J.T @ J

    D = jnp.diag(d)

    if linear_solver == "qr":
        damping_rows = jnp.sqrt(lam * d)[:, None] * jnp.eye(J.shape[1], dtype=J.dtype)
        augmented_jacobian = jnp.concatenate((J, damping_rows), axis=0)
        augmented_rhs = jnp.concatenate((-r, jnp.zeros(J.shape[1], dtype=r.dtype)))
        delta = solve_augmented_qr(augmented_jacobian, augmented_rhs)
    else:
        delta = _solve_spd(JtJ + lam * D, -Jtr, linear_solver)
    dnorm = jnp.linalg.norm(delta)

    # Step-norm clipping
    clip = jnp.where(
        (max_step_norm > 0) & (dnorm > max_step_norm),
        max_step_norm / dnorm,
        1.0,
    )
    delta = delta * clip
    dnorm = dnorm * clip

    pred_red = -(Jtr @ delta + 0.5 * delta @ (JtJ @ delta))

    return delta, dnorm, pred_red, Jtr, grad_norm, JtJ


def _gn_step_normal(
    JtJ: jax.Array,
    Jtr: jax.Array,
    lam: float,
    max_step_norm: float,
    d: jax.Array,
    linear_solver: str = "solve",
):
    """Damped Gauss-Newton step from the normal equations directly.

    The distributed path reduces ``J^T J`` and ``J^T r`` across the device mesh and
    never forms the full Jacobian, so the step is computed from those replicated
    matrices.  Mirrors
    [_gn_step][diff_soup.inference.solvers.levenberg_marquardt._gn_step]
    exactly given the same ``(J^T J, J^T r)``.

    Returns ``(delta, dnorm, pred_red, Jtr, grad_norm)``.
    """
    grad_norm = jnp.linalg.norm(Jtr)
    D = jnp.diag(d)

    delta = _solve_spd(JtJ + lam * D, -Jtr, linear_solver)
    dnorm = jnp.linalg.norm(delta)

    clip = jnp.where(
        (max_step_norm > 0) & (dnorm > max_step_norm),
        max_step_norm / dnorm,
        1.0,
    )
    delta = delta * clip
    dnorm = dnorm * clip

    pred_red = -(Jtr @ delta + 0.5 * delta @ (JtJ @ delta))

    return delta, dnorm, pred_red, Jtr, grad_norm


def _gauss_newton_decrease(gram, gradient, point, lb=None, ub=None) -> float:
    """The decrease the Gauss-Newton model predicts for its undamped step at ``point``.

    The step is the minimum-norm minimizer of ``g.T p + 0.5 p.T G p`` over the
    range of ``G`` (eigenvalues above ``n * eps`` of the largest), projected onto
    the box when bounds are given.
    """
    gram = 0.5 * (gram + gram.T)
    values, vectors = jnp.linalg.eigh(gram)
    cutoff = values.shape[0] * jnp.finfo(values.dtype).eps * jnp.maximum(values[-1], 0.0)
    coefficients = vectors.T @ gradient
    keep = values > cutoff
    safe = jnp.where(keep, values, 1.0)
    step = -vectors @ jnp.where(keep, coefficients / safe, 0.0)
    if lb is not None:
        step = jnp.clip(point + step, lb, ub) - point
    return float(local_value(-(gradient @ step + 0.5 * step @ (gram @ step))))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class LevenbergMarquardt(OptSolver):
    """The Levenberg-Marquardt method, damped Gauss-Newton steps on a least-squares problem.

    It needs the residual and its Jacobian, which a `LeastSquaresProblem` provides. Box bounds
    are kept.

    Args:
        max_steps: The step budget.
        lam_init: The initial damping.
        lam_max: The damping above which it restarts from `lam_init`.
        nu_init: The factor the damping grows by on a rejection after an accepted step, itself doubled on each further rejection.
        nu_start: The growth factor before the first accepted step, `nu_init` when `None`.
        accept_cutoff: The ratio of achieved to predicted decrease a step must exceed to be accepted.
        grad_tol: The gradient-norm tolerance, or `None` for the dtype's default.
        step_tol: The step-norm tolerance relative to the parameter norm, or `None` for the dtype's default.
        cost_tol: The tolerance on the change in the objective relative to the objective, or `None` for the dtype's default.
        cost_atol: An absolute floor under the cost-change test, zero when `None`.
        max_step_norm: The longest step allowed, or `0` for no limit.
        linear_solver: `"cholesky"`, `"qr"`, `"solve"`, or `"auto"` for QR in single precision with a full Jacobian and Cholesky otherwise.
        x_scale: The parameter scaling of the damping, `JacScale()` by default.
        callbacks: Callbacks run together with those passed to `ds.inference.optsolve`.
    """

    requires: ClassVar[frozenset] = frozenset({"residual_jacobian"})

    max_steps: int = eqx.field(static=True, default=30)
    lam_init: float = eqx.field(static=True, default=1.0)
    lam_max: float = eqx.field(static=True, default=1e6)
    nu_init: float = eqx.field(static=True, default=2.0)
    nu_start: float | None = eqx.field(static=True, default=None)
    accept_cutoff: float = eqx.field(static=True, default=1e-4)
    grad_tol: float | None = eqx.field(static=True, default=None)
    step_tol: float | None = eqx.field(static=True, default=None)
    cost_tol: float | None = eqx.field(static=True, default=None)
    cost_atol: float | None = eqx.field(static=True, default=None)
    max_step_norm: float = eqx.field(static=True, default=5.0)
    linear_solver: str = eqx.field(static=True, default="auto")
    x_scale: XScale = eqx.field(default_factory=JacScale)
    callbacks: tuple = eqx.field(static=True, default=())

    def solve(self, problem, *, callbacks=()) -> OptResult:
        """Run LM on the given problem.

        Args:
            problem: A [LeastSquaresProblem][diff_soup.inference._abc.LeastSquaresProblem]
                (structure ``"residual_jacobian"``). When its
                ``reduced_gauss_newton`` is set (the lanes of a lowered study,
                or a joint sharded over devices) the loop steps from its
                ``gauss_newton`` statistics and never forms the residual
                Jacobian; otherwise from ``residual`` and
                ``residual_and_jacobian``. Its ``regularization`` rows are
                appended, its ``bounds`` project every trial, its
                ``is_valid`` screens every trial, and ``fitted`` rebuilds
                the fitted model.
            callbacks: Extra callback objects merged with ``self.callbacks``.
                Each callback's ``on_start`` / ``on_step`` / ``on_end`` methods
                fire during the solve.  Must be passed as a keyword argument.

        Returns:
            A [OptResult][diff_soup.inference.fitting.result.OptResult].
        """
        cbs = tuple(self.callbacks) + tuple(callbacks)

        x0 = problem.x0
        n_params = len(x0)

        # --- Evaluation route ---------------------------------------------
        # A problem whose Gauss-Newton statistics are reduced without forming the
        # residual Jacobian (the lanes of a lowered study, or a joint sharded over
        # devices) drives the loop from (cost, J^T r, J^T J); every other problem
        # from its residual and Jacobian. On the lanes a trial is evaluated with the
        # statistics and an accepted trial keeps them, as the full path keeps a
        # trial's linearization; a sharded joint's trial evaluates its cost alone.
        use_reduced = bool(problem.reduced_gauss_newton)
        use_lanes = use_reduced and "batched_lanes" in problem.structure
        use_reduced = use_reduced and not use_lanes
        if use_reduced or use_lanes:
            _gauss_newton = problem.gauss_newton
            _value = problem.value
        else:
            _r_fn = problem.residual
            _rj_fn = problem.residual_and_jacobian
        if self.linear_solver not in ("auto", "cholesky", "qr", "solve"):
            raise ConfigurationError(
                "LevenbergMarquardt linear_solver must be 'auto', 'cholesky', "
                f"'qr', or 'solve'; got {self.linear_solver!r}."
            )
        resolved_linear_solver = self.linear_solver
        if resolved_linear_solver == "auto":
            single = np.dtype(np.asarray(problem.x0).dtype).itemsize <= 4
            reduced = use_reduced or use_lanes
            resolved_linear_solver = "qr" if single and not reduced else "cholesky"
        if use_reduced and resolved_linear_solver == "qr":
            raise ConfigurationError(
                "LevenbergMarquardt linear_solver='qr' requires the full residual "
                "Jacobian, but this sharded problem exposes only reduced J.T @ J "
                "statistics; use linear_solver='cholesky' or 'solve'."
            )
        if use_lanes and resolved_linear_solver == "qr":
            raise ConfigurationError(
                "LevenbergMarquardt linear_solver='qr' requires the full residual "
                "Jacobian, but this problem evaluates only J.T @ J and J.T @ r on its "
                "lanes; use linear_solver='cholesky' or 'solve'."
            )

        n_seg = int(problem.n_segments)

        # --- Regularisation ------------------------------------------------
        # The regularizer's ``residual(x)`` returns the extra least-squares
        # rows (length 0 for no regularization).  ``has_reg`` is decided from
        # the static row count at the initial point, so augmentation is a
        # plain Python branch (no traced control flow).
        reg = problem.regularization
        x = jnp.asarray(x0)
        max_steps = step_budget(self.max_steps, label="LevenbergMarquardt")
        grad_tolerance = gradient_tolerance(self.grad_tol, x.dtype, label="LevenbergMarquardt")
        grad_tol = None
        step_tol = relative_tolerance(
            self.step_tol, DEFAULT_STEP_TOL, x.dtype, name="step_tol", label="LevenbergMarquardt"
        )
        cost_tol = relative_tolerance(
            self.cost_tol, DEFAULT_COST_TOL, x.dtype, name="cost_tol", label="LevenbergMarquardt"
        )
        cost_atol = absolute_tolerance(
            self.cost_atol, DEFAULT_COST_ATOL, name="cost_atol", label="LevenbergMarquardt"
        )

        # --- Box bounds ---------------------------------------------------
        # When the problem declares bounds, project each trust-region trial onto
        # the box and use the projected step in the predicted-reduction / gain-ratio
        # arithmetic (projecting only the accepted point would corrupt the gain ratio
        # at active bounds). The initial point is clipped once.
        bnds = problem.bounds
        lb = ub = None
        if bnds is not None:
            lb, ub = (jnp.asarray(side) for side in bnds)
            x = jnp.clip(x, lb, ub)
        if not trial_valid(problem, x):
            raise ValueError("the initial optimization point failed the trial predicate.")

        has_reg = reg is not None and reg.residual(x).shape[0] > 0

        # --- Evaluation helpers (JAX arrays, no numpy mid-loop) -----------

        def eval_rj(x_jax):
            """Residual + Jacobian, with regularisation, all on device."""
            r_pde, J_pde = _rj_fn(x_jax)
            ok = jnp.all(jnp.isfinite(r_pde))
            if has_reg:
                r = jnp.concatenate([r_pde, reg.residual(x_jax)])
                J = jnp.concatenate([J_pde, jax.jacfwd(reg.residual)(x_jax)], axis=0)
            else:
                r, J = r_pde, J_pde
            return r, J, r_pde, ok

        def eval_r_linearized(x_jax):
            """Evaluate a trial once and retain its forward linearization."""
            r_pde, pushforward = linearize_value(_r_fn, x_jax)
            ok = jnp.all(jnp.isfinite(r_pde))
            r = jnp.concatenate([r_pde, reg.residual(x_jax)]) if has_reg else r_pde
            return 0.5 * jnp.sum(r**2), r_pde, ok, pushforward

        def eval_rj_linearized(x_jax, r_pde, pushforward):
            """Materialize an accepted Jacobian without repeating its primal."""
            J_pde = jacfwd_from_pushforward(pushforward, x_jax, r_pde)
            if has_reg:
                return (
                    jnp.concatenate([r_pde, reg.residual(x_jax)]),
                    jnp.concatenate([J_pde, jax.jacfwd(reg.residual)(x_jax)], axis=0),
                )
            return r_pde, J_pde

        def eval_cost_reduced(x_jax):
            """The trial cost (no derivatives), with regularisation, all on device."""
            cost_pde = _value(x_jax)
            ok = jnp.isfinite(cost_pde)
            if has_reg:
                cost = cost_pde + 0.5 * jnp.sum(reg.residual(x_jax) ** 2)
            else:
                cost = cost_pde
            return cost, ok

        def eval_lanes(x_jax):
            """``(cost, J^T r, J^T J, ok)`` from the problem's one evaluation, with regularisation.

            The statistics arrive already summed (across the lanes or the mesh);
            the small, replicated regularisation block is added on every device.
            """
            cost_pde, jtr, jtj, failed = _gauss_newton(x_jax)
            ok = failed == 0
            if has_reg:
                r_reg = reg.residual(x_jax)
                j_reg = jax.jacfwd(reg.residual)(x_jax)
                cost = cost_pde + 0.5 * jnp.sum(r_reg**2)
                jtr = jtr + j_reg.T @ r_reg
                jtj = jtj + j_reg.T @ j_reg
            else:
                cost = cost_pde
            return cost, jtr, jtj, ok

        # --- Initial evaluation -------------------------------------------
        # No hard fp64 cast on the param path: ``x0``'s dtype flows
        # through, so an fp32 params pytree survives an LM step without a
        # silent fp64 promotion of the param leaves.
        _param_dtype = jnp.asarray(x0).dtype
        t0_wall = time.perf_counter()

        # ``Jtr_red`` / ``JtJ_red`` carry the reduced normal-equation statistics on the
        # sharded path; ``r`` / ``J`` carry the full residual/Jacobian otherwise.
        Jtr_red = JtJ_red = None
        if use_reduced or use_lanes:
            cost_j, Jtr_red, JtJ_red, init_ok = eval_lanes(x)
            cost = float(local_value(cost_j))
            n_residuals = int(problem.n_residuals or 0)
        else:
            r, J, r_pde_init, init_ok = eval_rj(x)
            cost = 0.5 * float(local_value(jnp.sum(r**2)))
            n_residuals = int(local_value(r_pde_init.shape[0]))
        for cb in cbs:
            cb.on_start(n_params, n_residuals, self.max_steps)

        lam = float(self.lam_init)
        nu = float(self.nu_init if self.nu_start is None else self.nu_start)
        cost_history = []
        grad_history = []
        n_accepted = 0
        n_resid_evals, n_jac_evals = 1, 1
        trial_predicate_rejections = 0
        status = TerminationStatus.MAX_STEPS
        stall_reason = None
        # ``max_steps=0`` makes no step: the start point is returned.
        n_steps = 0

        # --- Main loop (Python outer, JAX inner math) ---------------------
        for step in range(max_steps):
            n_steps = step + 1
            # Damping diagonal from the scale object (length n_params), then the GN
            # step — from the reduced normal equations when sharded, else the full J.
            if use_reduced or use_lanes:
                d = self.x_scale.damping_from_gram(JtJ_red, x)
                delta, dnorm, pred_red, Jtr, grad_norm = _gn_step_normal(
                    JtJ_red,
                    Jtr_red,
                    lam,
                    self.max_step_norm,
                    d,
                    resolved_linear_solver,
                )
            else:
                d = self.x_scale.damping_diagonal(J, x)
                delta, dnorm, pred_red, Jtr, grad_norm, JtJ = _gn_step(
                    J,
                    r,
                    lam,
                    self.max_step_norm,
                    d,
                    resolved_linear_solver,
                )

            # The normal-equation math upcasts to the accumulate dtype (e.g. fp64),
            # so ``delta`` comes back in the Jacobian's dtype. Cast it back to the
            # param dtype at the x update so an fp32 param stream survives end-to-end.
            # For an fp64 stream this is a no-op.
            delta = delta.astype(_param_dtype)

            # Read scalars for convergence checks
            grad_norm_f = float(local_value(grad_norm))

            # Trial step, projected onto the box when bounds are active. Using the
            # projected step ``delta_proj = clip(x + delta, lb, ub) - x`` in the
            # predicted reduction keeps the gain ratio consistent with the step the
            # solver actually takes at an active bound; the model reduction is
            # ``-(g·delta + 0.5 delta·(JtJ)·delta)`` evaluated at that projected step.
            if bnds is not None:
                x_trial = jnp.clip(x + delta, lb, ub)
                delta_proj = x_trial - x
                gram = JtJ_red if use_reduced or use_lanes else JtJ
                pred_red = -(Jtr @ delta_proj + 0.5 * delta_proj @ (gram @ delta_proj))
                dnorm = jnp.linalg.norm(delta_proj)
                # At an active bound the ordinary gradient does not vanish; stop on
                # the projected-gradient norm instead.
                stop_grad = float(local_value(projected_gradient_norm(x, Jtr, lb, ub)))
            else:
                x_trial = x + delta
                stop_grad = grad_norm_f

            dnorm_f = float(local_value(dnorm))
            pred_red_f = float(local_value(pred_red))

            cost_history.append(cost)
            grad_history.append(grad_norm_f)

            if grad_tol is None:
                grad_tol = grad_tolerance.threshold(stop_grad)
            if assert_decision_value(stop_grad < grad_tol, label="Levenberg-Marquardt"):
                status = TerminationStatus.GRAD_TOL
                break

            # Trial: cost only (cheap — no Jacobian)
            try:
                trial_ok = trial_valid(problem, x_trial)
            except _PREDICATE_FAILURES:
                trial_ok = False
            if not trial_ok:
                cost_trial = 1e30
                trial_ok = False
                trial_predicate_rejections += 1
            else:
                try:
                    if use_reduced:
                        cost_trial_j, trial_ok = eval_cost_reduced(x_trial)
                    elif use_lanes:
                        cost_trial_j, trial_jtr, trial_jtj, trial_ok = eval_lanes(x_trial)
                    else:
                        cost_trial_j, trial_r, trial_ok, trial_pushforward = eval_r_linearized(
                            x_trial
                        )
                    cost_trial = float(local_value(cost_trial_j))
                    trial_ok = bool(local_value(trial_ok))
                except _EVALUATION_FAILURES:
                    # A trial whose evaluation failed is rejected.
                    cost_trial = 1e30
                    trial_ok = False
                n_resid_evals += 1

            actual_red = cost - cost_trial
            rho = actual_red / max(pred_red_f, 1e-30)
            # The model at the point this step was taken from, for the stall test.
            attempt_point = x
            attempt_gram = JtJ_red if use_reduced or use_lanes else JtJ
            threshold = cost_threshold(cost, cost_tol, cost_atol)

            if assert_decision_value(
                rho > self.accept_cutoff and trial_ok, label="Levenberg-Marquardt"
            ):
                x = x_trial
                cost = cost_trial
                accepted = True
                n_accepted += 1
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = self.nu_init

                # Recompute the Gauss-Newton statistics at the accepted point.
                try:
                    if use_reduced:
                        cost_j, Jtr_red, JtJ_red, _ = eval_lanes(x)
                        cost = float(local_value(cost_j))
                        n_resid_evals += 1
                    elif use_lanes:
                        Jtr_red, JtJ_red = trial_jtr, trial_jtj
                    else:
                        r, J = eval_rj_linearized(x, trial_r, trial_pushforward)
                except _EVALUATION_FAILURES:
                    # The step stays accepted and the next step's model uses the
                    # statistics of the previous point. On the dense path the
                    # result records a final residual and Jacobian only when
                    # their cost reproduces the returned cost, so stale ones are
                    # left out.
                    pass
                n_jac_evals += 1
            else:
                accepted = False
                lam *= nu
                nu *= 2.0

                if lam > self.lam_max:
                    lam = self.lam_init
                    nu = self.nu_init

            # Callbacks
            info = StepInfo(
                step=step,
                cost=cost,
                grad_norm=grad_norm_f,
                params=local_value(x),
                accepted=accepted,
                delta_norm=dnorm_f,
                wall_time=time.perf_counter() - t0_wall,
                rho=rho,
                lam=lam,
                nu=nu,
                accepted_count=n_accepted,
            )
            stop_requested = False
            for cb in cbs:
                try:
                    cb.on_step(info)
                except StopFit:
                    stop_requested = True
            if any_process(stop_requested):
                status = TerminationStatus.STOPPED
                break

            # Convergence checks
            x_norm = float(local_value(jnp.linalg.norm(x)))
            # A numerically stationary trial can have zero representable cost
            # reduction and therefore miss the gain-ratio acceptance cutoff.
            # Step/cost convergence describe that stationary trial itself, so
            # they require a valid evaluation but not formal acceptance.
            stop = None
            if assert_decision_value(
                trial_ok and dnorm_f < step_tol * (1.0 + x_norm), label="Levenberg-Marquardt"
            ):
                stop = TerminationStatus.STEP_TOL
            # The cost-change test is relative to the cost at the step's start, so
            # where a fit stops does not depend on the units of its residual rows;
            # an absolute floor is ``cost_atol``, which the caller sets. It needs
            # the model to agree with the decrease (SciPy's ratio), or a model that
            # predicts no decrease the tolerance can see.
            elif assert_decision_value(
                trial_ok
                and abs(actual_red) < threshold
                and (rho > 0.25 or pred_red_f <= threshold),
                label="Levenberg-Marquardt",
            ):
                stop = TerminationStatus.COST_TOL
            if stop is not None:
                status = stop
                full_decrease = _gauss_newton_decrease(attempt_gram, Jtr, attempt_point, lb, ub)
                if assert_decision_value(
                    model_held_the_step(full_decrease, pred_red_f, threshold),
                    label="Levenberg-Marquardt stall",
                ):
                    status = TerminationStatus.STALLED
                    stall_reason = stall_reason_of(
                        invalid=False, noise=not accepted and abs(actual_red) < threshold
                    )
                break

        wall_time = time.perf_counter() - t0_wall

        # ``grad_norm_f`` is evaluated at the beginning of an iteration. An
        # accepted step can then terminate on step/cost tolerance, so derive the
        # returned diagnostics from the final point's already-current normal
        # statistics instead of returning that preceding iterate's gradient.
        final_gradient = Jtr_red if use_reduced or use_lanes else J.T @ r
        grad_norm_f = float(local_value(jnp.linalg.norm(final_gradient)))
        projected_grad_norm_f = (
            grad_norm_f
            if bnds is None
            else float(local_value(projected_gradient_norm(x, final_gradient, lb, ub)))
        )

        # --- Reconstruct fitted problem -----------------------------------
        fitted = problem.fitted(x)

        # The dense path holds the residual and Jacobian of the returned point:
        # an accepted step re-linearizes at its new point and a rejected step
        # keeps the point together with its linearization. Rows are the
        # problem's residual followed by its regularization rows. They are
        # recorded only when their cost reproduces the returned cost, so an
        # accepted point whose re-linearization raised records nothing.
        final_arrays = {}
        if not (use_reduced or use_lanes):
            final_cost = 0.5 * float(local_value(jnp.sum(r**2)))
            if final_cost == cost:
                final_arrays = {
                    "final_residual": np.asarray(local_value(r)),
                    "final_jacobian": np.asarray(local_value(J)),
                }

        result = OptResult(
            opt_params=fitted,
            params=local_value(x),
            initial_params=local_value(x0),
            cost=cost,
            grad_norm=grad_norm_f,
            projected_grad_norm=projected_grad_norm_f,
            status=status,
            message=status.value,
            n_steps=n_steps,
            max_steps=self.max_steps,
            n_accepted=n_accepted,
            eval_counts=EvalCounts(
                residual_evals=n_resid_evals,
                jacobian_evals=n_jac_evals,
                pde_solves=n_resid_evals * n_seg + n_jac_evals * n_seg,
            ),
            cost_history=np.array(cost_history),
            grad_history=np.array(grad_history),
            wall_time=wall_time,
            info={
                "trial_predicate_rejections": trial_predicate_rejections,
                "linear_solver": resolved_linear_solver,
                **({} if stall_reason is None else {"stall_reason": stall_reason}),
                **final_arrays,
            },
        )

        # Fire on_end so a config-attached callback (e.g. a LogCallback
        # holding an open file) is properly released.
        for cb in cbs:
            cb.on_end(result)
        return result
