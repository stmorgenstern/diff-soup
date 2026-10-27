"""Inequality constraints for parameter fitting via augmented-Lagrangian penalties.

A user declares inequality constraints with
[InequalityConstraints][diff_soup.inference.fitting.constrained.InequalityConstraints]:
a
function ``g(params) -> (m,)`` on the *unflattened* learnables view, where
``g(params) <= 0`` means feasible.  The constraints are enforced with the
Powell-Hestenes-Rockafellar (PHR) augmented-Lagrangian form, expressed as extra
least-squares rows on the existing regularizer seam so every solver picks them
up without new machinery.

The penalty rows are

    r_i = max(0, lam_i + rho * g_i(x)) / sqrt(rho),

so ``0.5 * ||r||^2`` equals the PHR penalty
``sum(max(0, lam + rho*g)^2) / (2*rho)`` (up to a lam-only constant).  The
multiplier estimate ``lam`` and the penalty weight ``rho`` are runtime arrays
held on a *mutable* penalty object and read eagerly at evaluation time.  An
outer loop reassigns them between inner solves; because the rows are evaluated
outside any compiled region, reassignment recompiles nothing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.inference.fitting.bounds import projected_gradient_norm
from diff_soup.inference.fitting.regularization import NoReg
from diff_soup.inference.fitting.result import OptResult


@dataclasses.dataclass(frozen=True)
class InequalityConstraints:
    """Inequality constraints `g(params) <= 0` for `ds.inference.constrained_optsolve`.

    Attributes:
        g: A function of the fitted parameters returning one value per constraint, satisfied where every value is at most zero.
        n_constraints: The length of the array `g` returns.
    """

    g: Callable
    n_constraints: int


class _AugmentedLagrangianRows:
    """PHR penalty rows implementing the regularizer protocol.

    The rows are ``max(0, lam + rho*g(x)) / sqrt(rho)`` and the penalty is
    ``0.5 * ||rows||^2``.  ``lam`` (the multiplier estimate) and ``rho`` (the
    penalty weight) are plain `jax.numpy` arrays stored as attributes on
    this object.  An outer augmented-Lagrangian loop reassigns them between
    inner solves with an ordinary attribute write; because ``residual`` and
    ``penalty`` are evaluated eagerly on the host (outside the compiled PDE
    residual), reassigning ``lam`` or ``rho`` recompiles nothing.
    """

    def __init__(self, g_flat: Callable, m: int):
        self.g_flat = g_flat
        self.lam = jnp.zeros(m)
        self.rho = jnp.asarray(10.0)

    def residual(self, x):
        """Return the PHR penalty rows ``max(0, lam + rho*g(x)) / sqrt(rho)``."""
        gv = jnp.atleast_1d(self.g_flat(x))
        return jnp.maximum(0.0, self.lam + self.rho * gv) / jnp.sqrt(self.rho)

    def penalty(self, x):
        """Scalar penalty ``0.5 * ||residual(x)||^2``."""
        r = self.residual(x)
        return 0.5 * jnp.sum(r * r)


@dataclasses.dataclass(frozen=True)
class _ChainedRegularizer:
    """Two regularizers on one seam: concatenated rows, summed penalties.

    Chaining preserves a user-supplied regularizer when constraint penalty rows
    are added, so neither contribution is dropped.
    """

    base: Any
    extra: Any

    def residual(self, x):
        """Concatenate the base rows and the extra rows."""
        return jnp.concatenate(
            [
                jnp.reshape(self.base.residual(x), (-1,)),
                jnp.reshape(self.extra.residual(x), (-1,)),
            ]
        )

    def penalty(self, x):
        """Sum the base and extra penalties."""
        return self.base.penalty(x) + self.extra.penalty(x)


def augmented_lagrangian_penalty(problem, constraints: InequalityConstraints):
    """Build the PHR penalty rows for ``constraints`` on ``problem``.

    ``problem`` is an [OptimizationProblem][diff_soup.inference.OptimizationProblem]
    (``optsolve`` lowers an ``InferenceProblem`` to one). The constraint
    function receives the *unflattened* learnables view via
    ``problem.unflatten``.  In decision-window mode ``unflatten`` already
    slices the leading theta block off the flat vector, so the window-state
    tail is never passed to ``g``.  A joint problem has no single
    unflattened view; there ``g`` receives the flat joint parameter vector
    (the layout ``theta_names`` describes).

    ``g`` is probed once at ``problem.x0`` to fix the static constraint
    count: the probe must be a 1-D array of length ``constraints.n_constraints``
    with finite entries, otherwise a
    [ConfigurationError][diff_soup._errors.ConfigurationError]
    is raised stating the mismatch.

    If ``problem`` already carries a regularizer, the penalty rows are
    *chained* onto it (rows concatenated, penalties summed) so the user's
    regularizer is kept.  With no user regularizer the penalty object is
    returned on its own.

    Returns:
        A ``(regularizer, penalty)`` pair.  ``regularizer`` is what to attach to
        the problem's regularizer seam; ``penalty`` is the mutable
        [_AugmentedLagrangianRows][diff_soup.inference.fitting.constrained._AugmentedLagrangianRows]
        whose ``lam``/``rho`` the outer loop
        reassigns between inner solves.
    """
    m = int(constraints.n_constraints)
    g = constraints.g
    # A joint problem has no single unflattened view (its children
    # reconstruct per child); there g receives the flat joint parameter vector
    # ``[shared | locals...]`` — the same layout ``theta_names`` describes.
    unflatten = problem.unflatten
    if unflatten is None:
        unflatten = lambda x: x  # noqa: E731

    def g_flat(x):
        return jnp.atleast_1d(g(unflatten(x)))

    # Probe once at the starting point to fix and validate the static count.
    x0 = jnp.asarray(problem.x0)
    probe = np.asarray(g_flat(x0))
    if probe.ndim != 1 or probe.shape[0] != m:
        raise ConfigurationError(
            "InequalityConstraints.g must return a 1-D array of length "
            f"n_constraints={m}, but at the starting point it returned shape "
            f"{tuple(probe.shape)}. Make g return one value per declared "
            "constraint, or set n_constraints to match g's output length."
        )
    if not np.all(np.isfinite(probe)):
        raise ConfigurationError(
            "InequalityConstraints.g returned a non-finite value at the "
            "starting point. g must be finite everywhere it is evaluated; "
            "check for divisions by zero or logs of non-positive quantities."
        )

    penalty = _AugmentedLagrangianRows(g_flat=g_flat, m=m)

    base = getattr(problem, "regularization", None)
    if base is None or isinstance(base, NoReg):
        return penalty, penalty
    return _ChainedRegularizer(base=base, extra=penalty), penalty


class _ConstrainedView:
    """A duck view of a problem with the AL rows attached and a settable start.

    Everything except ``regularization``, ``x0`` and ``numeric_problem``
    delegates to the wrapped problem, so its cached compiled kernels are
    shared across the outer iterations and the user's problem object is never
    mutated.
    """

    def __init__(self, inner: Any, regularization: Any, x0: Any):
        self._inner = inner
        self.regularization = regularization
        self.x0 = np.asarray(x0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def numeric_problem(self, optimizer: Any) -> Any:
        """The wrapped problem lowered for ``optimizer``, at this view's start and regularization.

        ``optsolve`` lowers a statistical problem through its
        ``numeric_problem``; the wrapped problem's own method would lower it
        at its own ``x0`` and regularization, so the view replaces both on
        the lowered problem.
        """
        from diff_soup.inference.fitting._optsolve import _numeric_problem

        lowered = _numeric_problem(self._inner, optimizer)
        return dataclasses.replace(
            lowered, x0=np.asarray(self.x0), _regularization=self.regularization
        )


@dataclasses.dataclass(frozen=True)
class ConstrainedOptResult:
    """The result of [constrained_optsolve][diff_soup.inference.fitting.constrained.constrained_optsolve].

    Attributes:
        result: The final inner solve's
            [OptResult][diff_soup.inference.fitting.result.OptResult].
        multipliers: The augmented-Lagrangian multiplier estimates, one per
            constraint row (zero where the constraint is inactive).
        rho: The final penalty weight.
        kkt: First-order optimality measures at the returned point:
            ``stationarity`` (the (projected-)gradient norm of the Lagrangian),
            ``stationarity_scale`` (``max(1, |grad f|, |G^T lam|)`` — the size
            of the gradient terms the KKT condition balances, the scale the
            stationarity test is judged against), ``max_violation`` (the
            largest positive constraint value), and ``complementarity`` (the
            largest ``|multiplier * constraint|``).
        n_outer: How many outer iterations ran.
        converged: ``True`` when both ``max_violation <= feas_tol`` and
            ``stationarity <= stat_tol * stationarity_scale`` were reached
            within ``outer_max``. The stationarity test is relative to the
            gradient scale because a cancellation between large gradient terms
            cannot be resolved much below their own rounding; for problems
            with order-one gradients it reduces to the absolute tolerance.
        notes: Human-readable notes on abnormal exits (e.g. penalty growth
            freezing because it stopped reducing the violation, or the penalty
            weight hitting its cap).
    """

    result: OptResult
    multipliers: np.ndarray
    rho: float
    kkt: dict
    n_outer: int
    converged: bool
    notes: tuple = ()


#: Penalty growth targets genuine infeasibility. Within one order of magnitude
#: of ``feas_tol`` the measured violation is indistinguishable from the inner
#: solves' accuracy floor, where a larger weight only degrades conditioning.
_NEAR_FEASIBLE_FACTOR = 10.0

#: The penalty weight grows when an outer iteration leaves the violation above
#: this fraction of its previous value. Each multiplier update shrinks the
#: violation of an active constraint by about ``c / (c + rho)``, where ``c`` is
#: the objective's curvature along the constraint normal. When ``rho`` equals
#: ``c`` (the default weight on rows expressed in inverse-noise units with a
#: noise of 1e-3) that factor is exactly one half, so a test against one half
#: would decide on rounding. A quarter grows the weight until it is at least
#: three times the curvature, after which the violation falls by a factor of
#: four or more per iteration.
_STALL_RATIO = 0.25

#: The inner solves stop at this fraction of the outer stationarity tolerance,
#: so the outer test still passes when the gradient scale shifts slightly
#: between the inner solve and the outer measurement.
_INNER_STATIONARITY_MARGIN = 0.5


def _inner_optimizer(optimizer: Any, grad_target: float, dtype: Any) -> Any:
    """``optimizer`` with its unset termination tolerances tied to the outer test.

    The outer stationarity measure is the gradient of the inner objective (the
    data rows plus the penalty rows) at the point the inner solve returns. An
    inner solve that stops on a small cost change or step can return while that
    gradient is still far above the outer tolerance, and then no number of outer
    iterations certifies stationarity. An unset ``grad_tol`` therefore becomes
    ``grad_target``, and an unset ``step_tol`` or ``cost_tol`` becomes the
    smallest relative change the decision dtype resolves (ten machine
    epsilons), so an inner solve stops when its gradient passes the outer test
    or when its progress is lost to rounding. A tolerance the caller set is kept.

    Args:
        optimizer: The inner solver.
        grad_target: The absolute gradient norm an inner solve should reach.
        dtype: The decision vector's dtype.

    Returns:
        A copy of ``optimizer`` with the unset tolerances filled in, or
        ``optimizer`` itself when it has none unset.
    """
    from diff_soup.inference.solvers._termination import _resolution

    if not dataclasses.is_dataclass(optimizer):
        return optimizer
    names = {field.name for field in dataclasses.fields(optimizer)}
    changes: dict = {}
    if "grad_tol" in names and optimizer.grad_tol is None:
        changes["grad_tol"] = float(grad_target)
    for name in ("step_tol", "cost_tol"):
        if name in names and getattr(optimizer, name) is None:
            changes[name] = _resolution(dtype)
    return dataclasses.replace(optimizer, **changes) if changes else optimizer


def _lagrangian_gradient(view: Any, penalty: _AugmentedLagrangianRows, x, lam):
    """Gradient of the Lagrangian ``f + lam^T g`` at ``x``, with its two terms.

    Returns ``(grad, grad_f, grad_g)`` — the Lagrangian gradient and the
    objective / multiplier terms it sums, so the caller can judge the
    stationarity norm against the size of the terms being cancelled.

    The sum equals the gradient of the objective-plus-PHR-penalty at the inner
    optimum, because the PHR penalty's gradient is exactly ``G^T max(0,
    lam_prev + rho*g) = G^T lam``. ``view`` is the lowered
    [LeastSquaresProblem][diff_soup.inference.LeastSquaresProblem]. The
    objective gradient is ``J^T r`` from its Gauss-Newton statistics where
    they are reduced without forming the Jacobian (the lanes of a lowered
    study, a joint sharded over devices), and otherwise from its residual and
    Jacobian; a user regularizer that was chained under the penalty
    contributes through its scalar penalty.
    """
    x = jnp.asarray(x)
    if view.reduced_gauss_newton:
        grad_f = jnp.asarray(view.gauss_newton(x)[1])
    else:
        r, J = view.residual_and_jacobian(x)
        grad_f = J.T @ r

    base = getattr(view, "regularization", None)
    if isinstance(base, _ChainedRegularizer):
        # Only the USER regularizer is part of the objective; the AL rows enter
        # through the multiplier term below.
        grad_f = grad_f + jax.grad(lambda p: base.base.penalty(p))(x)

    _, g_vjp = jax.vjp(penalty.g_flat, x)
    (grad_g,) = g_vjp(jnp.asarray(lam))
    return grad_f + grad_g, grad_f, grad_g


def constrained_optsolve(
    problem: Any,
    optimizer: Any = None,
    *,
    constraints: InequalityConstraints,
    outer_max: int = 15,
    feas_tol: float = 1e-6,
    stat_tol: float = 1e-6,
    rho0: float = 1.0e6,
    callbacks: tuple = (),
) -> ConstrainedOptResult:
    """Fit a problem subject to inequality constraints with an augmented-Lagrangian outer loop.

    Each outer iteration runs `ds.inference.optsolve` on the objective plus a penalty on the
    constraints, then updates the multipliers. It needs JAX 64-bit mode.

    Args:
        problem: The problem to fit, as for `ds.inference.optsolve`.
        optimizer: The inner optimizer, `NewtonTrustRegion()` when `None`.
        constraints: The [InequalityConstraints][diff_soup.inference.InequalityConstraints] to satisfy.
        outer_max: The largest number of outer iterations.
        feas_tol: The largest constraint violation accepted at convergence.
        stat_tol: The tolerance on the norm of the Lagrangian gradient, relative to the gradient scale.
        rho0: The initial penalty weight.
        callbacks: Callbacks passed to every inner fit.

    Returns:
        A `ConstrainedOptResult` with the last inner fit as `result`, the `multipliers`, the `kkt` measures and `converged`.
    """
    from diff_soup.inference.fitting._optsolve import _numeric_problem, optsolve
    from diff_soup.utils.precision import warn_if_float32_default

    warn_if_float32_default()

    if not jax.config.jax_enable_x64:
        raise ConfigurationError(
            "constrained_optsolve requires JAX 64-bit (x64) mode, which is "
            "disabled: the augmented-Lagrangian multiplier and penalty "
            "updates lose the precision they need in float32 and the outer "
            "loop returns non-finite diagnostics instead of a solution. "
            "Enable x64 with jax.config.update('jax_enable_x64', True) or "
            "JAX_ENABLE_X64=1 before calling constrained_optsolve."
        )

    if optimizer is None:
        from diff_soup.inference.solvers.newton_trust_region import NewtonTrustRegion

        optimizer = NewtonTrustRegion()  # the default of optsolve
    regularizer, penalty = augmented_lagrangian_penalty(
        _numeric_problem(problem, optimizer), constraints
    )
    penalty.rho = jnp.asarray(float(rho0))
    view = _ConstrainedView(problem, regularizer, problem.x0)
    # The lowered view gives the residual, the Jacobian and the flat bounds the
    # outer loop reads; its regularization is the view's (the AL rows).
    lowered = _numeric_problem(view, optimizer)
    bnds = lowered.bounds
    dtype = view.x0.dtype
    # Each inner solve aims at the outer stationarity tolerance of the latest
    # gradient scale; before the first solve that is the objective's at x0.
    _, grad_f0, _ = _lagrangian_gradient(lowered, penalty, jnp.asarray(view.x0), penalty.lam)
    stat_scale = max(1.0, float(jnp.linalg.norm(grad_f0)))

    rho_max = 1e8
    notes: list = []
    prev_viol: Optional[float] = None
    grew_last = False
    growth_stopped = False
    converged = False
    result = None
    kkt: dict = {}
    n_outer = 0

    for n_outer in range(1, int(outer_max) + 1):
        inner = _inner_optimizer(
            optimizer, _INNER_STATIONARITY_MARGIN * stat_tol * stat_scale, dtype
        )
        result = optsolve(view, inner, callbacks=callbacks, diagnostics=False)
        x = jnp.asarray(result.params)

        gv = penalty.g_flat(x)
        lam = jnp.maximum(0.0, penalty.lam + penalty.rho * gv)
        penalty.lam = lam

        viol = float(jnp.max(jnp.maximum(0.0, gv)))
        grad_l, grad_f, grad_g = _lagrangian_gradient(lowered, penalty, x, lam)
        if bnds is not None:
            stationarity = float(projected_gradient_norm(x, grad_l, bnds[0], bnds[1]))
        else:
            stationarity = float(jnp.linalg.norm(grad_l))
        stat_scale = max(1.0, float(jnp.linalg.norm(grad_f)), float(jnp.linalg.norm(grad_g)))
        kkt = {
            "stationarity": stationarity,
            "stationarity_scale": stat_scale,
            "max_violation": viol,
            "complementarity": float(jnp.max(jnp.abs(lam * gv))),
        }

        if viol <= feas_tol and stationarity <= stat_tol * stat_scale:
            converged = True
            break

        # Grow the penalty only when feasibility stalled (violation not cut to
        # a quarter), the violation is more than an order of magnitude above
        # the tolerance (closer in, it is indistinguishable from the inner
        # solves' accuracy floor), and the previous growth actually helped. A
        # growth that did not even halve the violation proves the floor has
        # been reached — or the problem is infeasible at this tolerance — and a
        # larger weight would only degrade the inner solves' conditioning, so
        # growth freezes.
        stalled = prev_viol is not None and viol > _STALL_RATIO * prev_viol
        ineffective = prev_viol is not None and viol > 0.5 * prev_viol
        if ineffective and grew_last and not growth_stopped:
            growth_stopped = True
            notes.append(
                f"penalty growth froze at rho={float(penalty.rho):.0e}: raising "
                f"the weight did not reduce the constraint violation (stalled "
                f"at {viol:.3e} with feas_tol={feas_tol:.0e}). The violation "
                "is at the accuracy floor of the inner solves, or the problem "
                "is infeasible at this tolerance."
            )
        grew_last = False
        if stalled and viol > _NEAR_FEASIBLE_FACTOR * feas_tol and not growth_stopped:
            if float(penalty.rho) >= rho_max:
                notes.append(
                    f"penalty weight reached its cap ({rho_max:.0e}) with "
                    f"max violation {viol:.3e}; the constrained problem may be "
                    "infeasible or badly scaled."
                )
            penalty.rho = jnp.minimum(penalty.rho * 10.0, rho_max)
            grew_last = True
        prev_viol = viol

        # Warm start the next inner solve from this outer iterate (mandatory:
        # a cold restart re-pays the whole descent path every iteration).
        view.x0 = np.asarray(result.params)

    return ConstrainedOptResult(
        result=result,
        multipliers=np.asarray(penalty.lam),
        rho=float(penalty.rho),
        kkt=kkt,
        n_outer=n_outer,
        converged=converged,
        notes=tuple(notes),
    )
