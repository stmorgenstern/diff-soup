"""Shared consistent-initialization records and solver-view operators.

Defines the records both Euler-Newton consistent-IC modes use
([ICMode][diff_soup.timestepping.consistent_ic.pin_differential.ICMode], [ICResult][diff_soup.timestepping.consistent_ic.pin_differential.ICResult], [ICConfig][diff_soup.timestepping.consistent_ic.pin_differential.ICConfig], the step context)
and the operator adapters that present a solver view to the
nonlinear corrector as a KKT residual with its Schur blocks — including the
pinned operator, where selected state rows are held by identity and only
the remaining unknowns move. The projected root's sensitivities flow
through the corrector's standard implicit-function boundary, never through
the Newton iteration.

The default startup projection (pin the differential rows, solve the local
and global algebraic values) runs the capability-routed initialization
solve on the solver problem
([SolverView.consistent_initial_state][diff_soup.problem.implicit.solver_problem.SolverView.consistent_initial_state]),
not through this module.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.linear import Dense
from diff_soup.linear._api import SparseTopology
from diff_soup.nonlinear._kkt import LinearPlan
from diff_soup.nonlinear.solver import NonlinearConfig, nonlinear_solve_primal
from diff_soup.utils import safe_sqrt


class ICMode(enum.Enum):
    """Which unknowns an Euler-Newton consistent-IC solve determines.

    ``ALG_AND_YDOT`` solves the algebraic components of ``y`` and the
    differential components of ``ydot`` with leading coefficient ``cj = 1/hic``.
    ``ALL_Y_GIVEN_YDOT`` solves all of ``y`` at ``cj = 0`` given a fixed
    ``ydot``.
    """

    ALG_AND_YDOT = "alg_and_ydot"
    ALL_Y_GIVEN_YDOT = "all_y_given_ydot"


class ICResult(eqx.Module):
    """The consistent state, bordered globals, seed derivative, and residual.

    Attributes:
        y: The projected/consistent state ``u`` (length ``n_u``).
        lam: The bordered-global values (length ``n_lam``; ``(0,)`` when there
            are no global unknowns).
        ydot: The order-1 predictor seed. In ``ALG_AND_YDOT`` mode the
            differential rows are solved and the algebraic rows pass through
            from the input unchanged; in ``ALL_Y_GIVEN_YDOT`` mode the input
            ``ydot`` is returned unchanged. Filling the algebraic rows with
            the finite-difference recovery is opt-in via
            [ICConfig.recover_algebraic_ydot][diff_soup.timestepping.consistent_ic.pin_differential.ICConfig.recover_algebraic_ydot].
        alg_residual_norm: The achieved RMS norm over the local-algebraic and
            bordered-global residual rows at the root (a scalar array),
            reported for the solve statistics.
    """

    y: jax.Array
    lam: jax.Array
    ydot: jax.Array
    alg_residual_norm: jax.Array


@dataclasses.dataclass(frozen=True)
class ICConfig:
    """Configuration for the consistent-IC solves.

    Attributes:
        rtol: Relative tolerance fed to the error-weight law.
        atol: Absolute tolerance fed to the error-weight law.
        max_newton_iters: Newton iteration cap for the IC solve.
        ic_newton_tol: The IC Newton convergence-test constant (the weighted-RMS
            Newton-step norm must fall to or below this).
        hic_frac: The initial Euler-Newton step is ``hic_frac`` times the time
            scale ``tdist``.
        schur_eps: A small diagonal nudge for the bordered-global Schur solve.
        linesearch: Reserved. Line search is not available in this
            implementation; enabling it raises. The damped-free Newton default is
            the available path.
        recover_algebraic_ydot: When ``True``, fill the algebraic ``ydot``
            rows after the solve with the finite-difference recovery across
            two projections (it seeds the integrator's predictor; the
            converged solution does not depend on it beyond solver
            tolerance). The default ``False`` passes the algebraic rows
            through from the input.
    """

    rtol: float
    atol: float
    max_newton_iters: int = 10
    ic_newton_tol: float = 0.0033
    hic_frac: float = 0.001
    schur_eps: float = 1e-12
    linesearch: bool = False
    recover_algebraic_ydot: bool = False


# ---------------------------------------------------------------------------
# Step context and operator adapter
# ---------------------------------------------------------------------------
class ICStepContext(NamedTuple):
    """Time, derivative, and pin target the fully-implicit residual reads.

    The leaves carry gradients so the initial time ``t0``, the derivative
    ``ydot`` (for the all-state-given-derivative solve), and the pin target
    ``u_pin`` reach the residual through autodiff. The pin target rides here
    rather than on the pinned operator, so that operator holds only its static
    structural mask and stays free of any data-dependent array. That lets it be
    passed as the non-differentiable operator of the implicit-function corrector
    under ``jax.jit``: a data-dependent operator leaf would be a traced value in
    a position the corrector treats as non-differentiable, which JAX rejects.

    Every field is always an array, so the context has one fixed pytree
    structure under jit/vmap and across the autodiff boundary. Build it with
    [make][diff_soup.timestepping.consistent_ic.pin_differential.ICStepContext.make]
    (which fills the unused slots with array sentinels) rather than
    the raw constructor, so no call site materializes a ``None`` leaf whose
    absence would change the tree structure and let a tangent tree mismatch its
    primal.

    Attributes:
        t0: The initial time.
        ydot: The full-state derivative used in ``F = M(t, y) ydot - f`` (zero on
            the pin path and the algebraic-state solve; the fixed input on the
            all-state solve).
        u_pin: The state values the pinned rows are held at (length ``n_u``); a
            zero vector when no rows are pinned (the operator's pin mask decides
            whether it is read).
        t1: The integration endpoint used to size the Euler-Newton iteration's
            time scale; a non-finite sentinel (``+inf``) means "no endpoint",
            which consumers treat as the nominal unit time scale. Rides here —
            never in a closure crossing a non-differentiable argument position —
            because it may be a traced value.
    """

    t0: Any
    ydot: Any
    u_pin: Any
    t1: Any

    @classmethod
    def make(cls, *, t0, ydot, n_u, u_pin=None, t1=None):
        """Build a fixed-arity context, filling unused slots with array sentinels.

        ``u_pin`` defaults to a length-``n_u`` zero vector (unused rows) and
        ``t1`` to ``+inf`` (no endpoint); both stay arrays so the pytree
        structure never depends on which arguments were supplied.
        """
        dtype = jnp.asarray(ydot).dtype
        return cls(
            t0=jnp.asarray(t0, dtype=dtype),
            ydot=jnp.asarray(ydot),
            u_pin=jnp.zeros((n_u,), dtype=dtype) if u_pin is None else jnp.asarray(u_pin),
            t1=jnp.asarray(jnp.inf, dtype=dtype) if t1 is None else jnp.asarray(t1, dtype=dtype),
        )

    def replace_pin(self, u_pin):
        """Return a copy with the pin target replaced (arity preserved)."""
        return self._replace(u_pin=jnp.asarray(u_pin))


def _full_dense_topology(n: int) -> SparseTopology:
    """A dense ``n x n`` COO pattern (row-major) for the small IC Jacobian block."""
    idx = np.arange(n, dtype=np.int32)
    rows = np.repeat(idx, n)
    cols = np.tile(idx, n)
    return SparseTopology(rows=rows, cols=cols, n_dofs=n)


def _ic_topology(problem, n_u):
    """The state-block pattern for the consistent-init KKT solve.

    Returns the problem's declared sparse pattern when it has one, so a
    FEM-scale projection factors the true pattern via a sparse backend instead
    of a dense ``(n_u, n_u)`` block. A problem that declares no pattern (its
    Jacobian is a dense matrix) falls back to the full dense pattern.
    """
    topology = problem.state_topology
    return topology if topology is not None else _full_dense_topology(n_u)


def _ic_backend(backend):
    """The consistent-init linear backend: the caller's, or a dense default.

    Engines pass their own resolver-selected backend so the projection factors
    on the same sparse backend the per-step solves use; a standalone call
    defaults to the dense backend.
    """
    return backend if backend is not None else Dense()


def _ic_analysis(backend, topology, *, layout=None, precision=None):
    """The plan of ``topology`` under the caller's placement and dtype policy.

    The projection factors the same problem as the corrector, so an engine
    passes the layout its problem was lowered with and the precision its
    accepted-root derivatives are taken at; the factor then lands where and
    at the dtype the rest of the solve runs. Either policy left ``None``
    keeps the default for it (the default placement, full precision), which
    is what a standalone call that declares neither gets. The pattern is
    analysed on the factor contract when the first matrix is factored.
    """
    del backend
    return LinearPlan(topology=topology, layout=layout, precision=precision)


class ContractOperator(eqx.Module):
    """Present a solver problem to the corrector as residual + Schur blocks.

    The corrector consumes ``residual(u, lam, theta, ctx) -> (R, g)`` and
    ``jacobian_blocks(u, lam, theta, ctx) -> (A_values, B, C, D, topology)``. This
    adapter splits the residual and obtains the exact iteration
    matrix from the shared linearization plan.

    Attributes:
        problem: The solver-contract problem (static; not a differentiable leaf).
        n_u: The state length.
        n_lam: The bordered-global length.
        topology: The state-block pattern for ``A`` — the problem's declared
            sparse pattern, or the full dense pattern when it declares none.
        cj: Coefficient of ``F_ydot`` in the iteration matrix.
    """

    problem: Any = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_lam: int = eqx.field(static=True)
    topology: SparseTopology = eqx.field(static=True)
    cj: float = eqx.field(static=True)

    def state_size(self) -> int:
        """The state length ``n_u``."""
        return self.n_u

    def constraint_dim(self) -> int:
        """The bordered-global length ``n_lam``."""
        return self.n_lam

    def residual(self, u, lam, theta, ctx):
        """Return ``(R, g)`` = the fully-implicit residual split at ``n_u``."""
        y = jnp.concatenate([u, lam])
        resid = self.problem.residual(ctx.t0, y, ctx.ydot, theta)
        return resid[: self.n_u], resid[self.n_u :]

    def jacobian_blocks(self, u, lam, theta, ctx):
        """Return ``(A_values, B, C, D, topology)`` for the KKT step.

        The shared plan supplies ``F_y + cj F_ydot`` directly in the
        corrector's Schur convention. A dense state block is flattened
        row-major onto `topology`.
        """
        y = jnp.concatenate([u, lam])
        a, b, c, d = self.problem.state_jacobian_blocks(ctx.t0, y, ctx.ydot, theta, self.cj)
        a_values = jnp.reshape(a, (-1,)) if jnp.ndim(a) == 2 else jnp.asarray(a)
        return a_values, b, c, d, self.topology


class PinnedOperator(eqx.Module):
    """Wrap a contract operator to hold the pinned rows at their targets.

    See
    [ContractOperator][diff_soup.timestepping.consistent_ic.pin_differential.ContractOperator].
    The pinned state rows get an identity Jacobian row and a ``u - ctx.u_pin``
    residual, so the Newton solve leaves them at the user's initial condition and
    drives only the remaining unknowns onto the manifold. When the bordered
    global partition carries differential rows, ``lam_pin_mask`` extends the same
    treatment to those bordered rows (identity ``D`` row, zeroed ``C`` row,
    ``lam - ctx.lam_pin`` residual), so a full-state projection can hold
    differential globals bit-for-bit while solving the algebraic globals. Pin
    targets ride in the step context rather than on the operator, so the operator
    holds only static structural masks and can serve as the corrector's
    non-differentiable operator under ``jax.jit``.

    Attributes:
        base: The wrapped adapter.
        pin_mask: Boolean mask over the state, ``True`` on the pinned rows. Held
            as a host array (a genuine constant) so this operator stays free of
            traced leaves and can be the corrector's non-differentiable argument.
        lam_pin_mask: Optional boolean host mask over the bordered-global rows,
            ``True`` on pinned (differential-global) rows. ``None`` — the
            default, and the consistent-initialization configuration — pins no
            bordered-global rows and never reads ``ctx.lam_pin``, so contexts without
            that field remain valid.
    """

    base: ContractOperator
    pin_mask: Any
    lam_pin_mask: Any = None

    def state_size(self) -> int:
        """Forward the state length."""
        return self.base.state_size()

    def constraint_dim(self) -> int:
        """Forward the bordered-global length."""
        return self.base.constraint_dim()

    def residual(self, u, lam, theta, ctx):
        """Override the pinned rows with ``u - ctx.u_pin`` / ``lam - ctx.lam_pin``."""
        r, g = self.base.residual(u, lam, theta, ctx)
        r = jnp.where(self.pin_mask, u - ctx.u_pin, r)
        if self.lam_pin_mask is not None:
            g = jnp.where(jnp.asarray(self.lam_pin_mask), lam - ctx.lam_pin, g)
        return r, g

    def jacobian_blocks(self, u, lam, theta, ctx):
        """Give pinned rows identity Jacobian rows and zero their coupling rows."""
        a_values, b, c, d, topo = self.base.jacobian_blocks(u, lam, theta, ctx)
        rows = jnp.asarray(topo.rows)
        cols = jnp.asarray(topo.cols)
        # ``pin_mask`` may be a host array (kept a genuine constant so this
        # operator can serve as the corrector's non-differentiable argument); lift
        # it to a device array for the traced gather/broadcast below.
        mask = jnp.asarray(self.pin_mask)
        pinned_row = mask[rows]
        a_values = jnp.where(pinned_row, jnp.where(rows == cols, 1.0, 0.0), a_values)
        if b.shape[1] > 0:
            b = jnp.where(mask[:, None], 0.0, b)
        if self.lam_pin_mask is not None and d.shape[0] > 0:
            lam_mask = jnp.asarray(self.lam_pin_mask)
            c = jnp.where(lam_mask[:, None], 0.0, c)
            d = jnp.where(lam_mask[:, None], jnp.eye(d.shape[0], dtype=d.dtype), d)
        return a_values, b, c, d, topo


# ---------------------------------------------------------------------------
# Structural helpers (static: derived from id_mask / n_lam, never a traced mass)
# ---------------------------------------------------------------------------
def _structure(problem):
    """Return ``(n_u, n_lam, diff_mask, alg_u_index, needs_projection)`` (static).

    The differential/algebraic split and the "needs a projection at all" gate are
    structural facts read from ``id_mask`` and ``n_lam`` at build time -- never
    from a traced mass diagonal, so the pure-ODE self-gate is a host decision.
    ``diff_mask`` is a host boolean array so it stays a genuine constant when the
    pinned operator carrying it becomes the corrector's non-differentiable
    argument (a traced mask there is rejected under ``jax.jit``); it still drives
    ``jnp.where`` selections unchanged.
    """
    n_u = int(problem.n_u)
    n_lam = int(problem.n_lam)
    id_arr = np.asarray(problem.id_mask)
    diff_u = id_arr[:n_u] > 0
    alg_u_index = np.where(~diff_u)[0].astype(np.int64)
    has_alg = bool(alg_u_index.size > 0)
    needs_projection = has_alg or (n_lam > 0)
    return n_u, n_lam, diff_u, alg_u_index, needs_projection


def _nonlinear_config(cfg: ICConfig) -> NonlinearConfig:
    """Build the robust IC nonlinear config (sticky reuse, no early tooslow exit)."""
    return NonlinearConfig(
        maxiter=int(cfg.max_newton_iters),
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        schur_eps=float(cfg.schur_eps),
        jacobian_cache="reuse",
        relinearize_on_tooslow=True,
        newton_rate_guard=False,
    )


def _seed_ydot_diff(problem, u, lam, theta, t0, n_u, *, tol, backend, layout=None, precision=None):
    """Return the startup slope on the state rows.

    ``layout`` and ``precision`` reach the slope solve's linear analysis; each
    left ``None`` keeps the backend's own default for that policy.
    """
    y = jnp.concatenate([u, lam])
    return problem.startup_slope(
        t0,
        y,
        theta,
        ydot0=jnp.zeros_like(y),
        tol=tol,
        backend=backend,
        layout=layout,
        precision=precision,
    )[:n_u]


def _alg_residual_norm(problem, u, lam, ydot, theta, t0, n_u, alg_u_index):
    """RMS norm over local- and global-algebraic residual rows at the root."""
    y = jnp.concatenate([u, lam])
    ydot_aug = jnp.concatenate([ydot, jnp.zeros_like(lam)])
    residual = problem.residual(t0, y, ydot_aug, theta)
    parts = [residual[jnp.asarray(alg_u_index)]] if alg_u_index.size > 0 else []
    alg_lam_index = np.flatnonzero(np.asarray(problem.id_mask)[n_u:] == 0)
    if alg_lam_index.size > 0:
        parts.append(residual[n_u + jnp.asarray(alg_lam_index)])
    if not parts:
        return jnp.asarray(0.0, dtype=y.dtype)
    block = jnp.concatenate(parts)
    return safe_sqrt(jnp.mean(block**2))


def _pin_operator(problem, n_u, n_lam, diff_mask, backend=None, *, layout=None, precision=None):
    """Build the pinned pencil operator and its analysis for the state solve.

    The pin target is supplied through the step context at solve time (see
    [ICStepContext][diff_soup.timestepping.consistent_ic.pin_differential.ICStepContext]),
    so the operator carries only the static structural
    mask. The state pattern is the problem's declared sparse pattern (a dense
    fallback when it declares none); ``backend`` is the caller's linear solver,
    defaulting to the dense backend. ``layout`` and ``precision`` are the
    placement and dtype policy the analysis is built with; each left ``None``
    keeps the backend's own default for that policy.
    """
    topology = _ic_topology(problem, n_u)
    base = ContractOperator(
        problem=problem,
        n_u=n_u,
        n_lam=n_lam,
        topology=topology,
        cj=0.0,
    )
    operator = PinnedOperator(base=base, pin_mask=diff_mask)
    backend = _ic_backend(backend)
    analysis = _ic_analysis(backend, topology, layout=layout, precision=precision)
    return operator, backend, analysis


def _pin_state_primal(operator, backend, analysis, u0, lam0, theta, ctx, ncfg, diff_mask):
    """Pin the differential rows and solve the local and global algebraic values.

    Runs the nonlinear corrector over the pinned operator, then restores the
    differential rows exactly to their input values. The pin target is written
    into the step context so the operator reads it there.
    """
    ctx = ctx.replace_pin(u0)
    u_solved, lam_solved, _info = nonlinear_solve_primal(
        operator,
        backend,
        analysis,
        u0,
        lam0,
        theta,
        ctx,
        ncfg,
        early_exit=False,
    )
    u_ci = jnp.where(diff_mask, u0, u_solved)
    return u_ci, lam_solved
