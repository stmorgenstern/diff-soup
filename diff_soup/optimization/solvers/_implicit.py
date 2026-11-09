"""The implicit derivative of a constrained solution with respect to `args`, in two forms.

A compiled solver's `run` never differentiates its iterations. At the solution `(x*, lam*)`
it solves the KKT system

    [ H       J_E^T   J_A^T ] [dx   ]     [ d_args grad_x L ]
    [ J_E     0       0     ] [dlamE] = - [ d_args c_E      ]
    [ J_A     0       0     ] [dlamA]     [ d_args c_A      ]

with `H` the exact Lagrangian Hessian, computed once, and `A` the active inequalities; an
active bound fixes its coordinate (`dx_j = 0`) and its multiplier takes the stationarity row
(`diff_soup.optimization.solvers._kkt_system` builds the system, degenerate active sets
included).

A solver that steps on Gauss-Newton (`hessian="gauss-newton"`) on a least-squares problem
takes the Gauss-Newton form: `H = J_r^T J_r` of the residual rows, `d_args grad_x L =
J_r^T (d_args r)` from one forward derivative of the rows, the constraint curvature dropped.
It is first order throughout, and it equals the exact form at a zero-residual solution with
linear rows. `rows_tangent` and `rows_cotangent` map the rows themselves, `r(x*(args), args)`:
under the Gauss-Newton form their Jacobian is the projected one, `(I - P) J_args`.

The forward form, the default, is one `custom_jvp` linear in the tangents (one `jvp` of the
KKT residual in `args`, one solve), so JAX transposes it for reverse mode, as in
`diff_soup.problem.implicit.root_ift`; its primal runs through the entry, so differentiating
the rule again differentiates the solution by the rule. The reverse form is one `custom_vjp`:
its backward pass solves the transposed system for the multiplier of the cotangents on the
solution and its multipliers, and takes the vector-Jacobian product of the KKT residual in
`args` with it. It serves `jax.grad` and `jax.vjp` only, at one solve whatever the size of
`args`, and needs the problem's gradient to reverse-differentiate in `args`. The solver's
`adjoint` selects the form.

The problem's bounds carry a derivative too: a coordinate an active bound fixes moves with
it, so the solution's derivative in that bound is 1 there and its multiplier's is the rate of
the Lagrangian's gradient, and the objective's derivative in the bound is the multiplier. The
start point, the warm start and the problem's scales carry none, and a tangent on any of them
raises `ConfigurationError` naming the leaf, in both forms.

A run that did not end with `KKT_TOL` has no supported derivative. Under the forward form its
tangents are NaN, and its input tangents are zeroed first so a reverse pass gives it exact
zero cotangents. Under the reverse form its cotangent on `args` is NaN, or exactly zero when
the cotangents on its outputs are.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero, zero_from_primal
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree

from diff_soup._adjoints import AbstractAdjoint
from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.kkt import KKTSolve
from diff_soup.optimization.solvers._barrier import Bounds
from diff_soup.optimization.solvers._closures import Closures
from diff_soup.optimization.solvers._kkt_system import (
    active_set,
    residual_of,
    rows_of,
    system_at,
)
from diff_soup.optimization.solvers._protocol import SolveRecord
from diff_soup.problem.implicit.tangent_gates import poison, zero_where_not

__all__ = [
    "active_set",
    "implicit_run",
    "kkt_cotangent",
    "kkt_tangent",
    "rows_cotangent",
    "rows_tangent",
]


def _tangent_terms(
    closures: Closures,
    bounds: Bounds,
    x: Float[Array, "n"],
    record: SolveRecord,
    args: PyTree,
    dargs: PyTree,
    jacobian: Float[Array, "m n"] | None,
) -> tuple:
    """`(d_f, d_grad, d_eq, d_ineq, system, rows, d_rows, J_r)` along `dargs`, from one forward derivative.

    The exact form differentiates the objective, the Lagrangian gradient and the rows in `args`;
    `d_rows` and `J_r` are `None`. The Gauss-Newton form differentiates the residual rows and
    the constraint rows, and its Lagrangian gradient's derivative is `J_r^T d_rows`, with `J_r`
    the one given or else computed at `x`; `rows` is the primal of that derivative.
    """
    if not closures.gauss_newton_rule:
        _, (d_f, d_grad, d_eq, d_ineq), derivatives = jax.jvp(
            residual_of(closures, x, record), (args,), (dargs,), has_aux=True
        )
        system = system_at(closures, bounds, x, record, args, derivatives)
        return d_f, d_grad, d_eq, d_ineq, system, None, None, None
    (rows, _, _), (d_rows, d_eq, d_ineq), (jac_eq, jac_ineq) = jax.jvp(
        rows_of(closures, x), (args,), (dargs,), has_aux=True
    )
    if jacobian is None:
        jacobian = closures.residual_and_jacobian(x, args)[1]
    derivatives = (jacobian.T @ rows, jac_eq, jac_ineq)
    system = system_at(closures, bounds, x, record, args, derivatives, jacobian)
    return rows @ d_rows, jacobian.T @ d_rows, d_eq, d_ineq, system, rows, d_rows, jacobian


def _kkt_tangent(closures, bounds, kkt, x, record, args, dargs, jacobian, dbounds=None):
    """[kkt_tangent][], with the rows, their tangent and Jacobian of the Gauss-Newton form beside it."""
    n_eq = record.multipliers_equality.shape[0]
    d_f, d_grad, d_eq, d_ineq, system, rows, d_rows, jacobian = _tangent_terms(
        closures, bounds, x, record, args, dargs, jacobian
    )
    active, degenerate = system.active, system.active.degenerate
    d_lower, d_upper = _bound_tangents(dbounds, x)
    jac_eq, jac_ineq = system.jac_eq, system.jac_ineq
    # A coordinate an active bound fixes moves with that bound; the degenerate system's barrier
    # rows `z (x - l) = mu` and `z (u - x) = mu` move with theirs.
    dx_fixed = (1.0 - system.keep) * jnp.where(active.lower, d_lower, d_upper)
    shift = jnp.where(degenerate, system.sigma_lower * d_lower + system.sigma_upper * d_upper, 0.0)
    dx, dlam, solved = kkt.solve(
        system.primal,
        system.rows,
        0.0,
        system.dual,
        -(d_grad + system.hessian @ dx_fixed) * system.keep + shift,
        -jnp.concatenate(
            [d_eq + jac_eq @ dx_fixed, system.rows_i * (d_ineq + jac_ineq @ dx_fixed)]
        ),
    )
    dx = dx + dx_fixed
    dlam_e, dlam_i = dlam[:n_eq], dlam[n_eq:]

    stationarity = system.hessian @ dx + jac_eq.T @ dlam_e + jac_ineq.T @ dlam_i + d_grad
    barrier = degenerate & ~active.pinned
    dz_lower = jnp.where(
        barrier,
        -system.sigma_lower * (dx - d_lower),
        jnp.where(active.lower, stationarity, 0.0),
    )
    dz_upper = jnp.where(
        barrier,
        system.sigma_upper * (dx - d_upper),
        jnp.where(active.upper, -stationarity, 0.0),
    )
    ds = jnp.where(
        degenerate,
        -system.slack_ratio * dlam_i,
        jnp.where(active.inequality, 0.0, -(jac_ineq @ dx + d_ineq)),
    )
    tangent = eqx.tree_at(
        _differentiable,
        jax.tree.map(_zero, record),
        (dlam_e, dlam_i, ds, (dz_lower, dz_upper), d_f + system.gradient @ dx),
    )
    return dx, tangent, solved, (rows, d_rows, jacobian)


def kkt_tangent(
    closures: Closures,
    bounds: Bounds,
    kkt: KKTSolve,
    x: Float[Array, "n"],
    record: SolveRecord,
    args: PyTree,
    dargs: PyTree,
    jacobian: Float[Array, "m n"] | None = None,
    dbounds: tuple[Float[Array, "n"], Float[Array, "n"]] | None = None,
) -> tuple[Float[Array, "n"], SolveRecord, Bool[Array, ""]]:
    """The tangent of `(x*, record)` along `dargs` and `dbounds`, linear in both, and whether its system solved.

    One forward derivative of one evaluation gives the derivatives of the objective, of the
    Lagrangian gradient and of the rows in `args`, and its primal gives the Jacobians. A
    coordinate an active bound fixes moves with that bound, `dbounds = (d_lower, d_upper)`,
    and its bound multiplier's tangent is the rate of the Lagrangian's gradient there. The
    system goes through the solver's [KKTSolve][diff_soup.optimization.solvers.KKTSolve], with
    the inactive rows' dual diagonal 1, and does not solve when its inertia is not `(n, m, 0)`.
    Closures with the Gauss-Newton rule take its form, from `jacobian` when it is given.
    """
    dx, tangent, solved, _ = _kkt_tangent(
        closures, bounds, kkt, x, record, args, dargs, jacobian, dbounds
    )
    return dx, tangent, solved


def rows_tangent(
    closures: Closures,
    bounds: Bounds,
    kkt: KKTSolve,
    x: Float[Array, "n"],
    record: SolveRecord,
    args: PyTree,
    dargs: PyTree,
    jacobian: Float[Array, "m n"] | None = None,
) -> tuple[Float[Array, "m"], Float[Array, "m"], Bool[Array, ""]]:
    """The rows at the solution, their tangent `d_rows + J_r dx*`, and whether its system solved.

    The rows move with `args` directly and through the solution; under the Gauss-Newton rule
    the tangent is the projected Jacobian `(I - P) J_args` applied to `dargs`. Gauss-Newton
    form only.
    """
    dx, _, solved, (rows, d_rows, jacobian) = _kkt_tangent(
        closures, bounds, kkt, x, record, args, dargs, jacobian
    )
    return rows, d_rows + jacobian @ dx, solved


def _cotangent_terms(
    closures: Closures,
    bounds: Bounds,
    x: Float[Array, "n"],
    record: SolveRecord,
    args: PyTree,
    jacobian: Float[Array, "m n"] | None,
) -> tuple:
    """`(system, back, J_r)`: the system and the pullback of `(v, d_grad, d_eq, d_ineq, rows)`.

    `back(v_bar, d_grad_bar, d_eq_bar, d_ineq_bar, rows_bar)` is the cotangent on `args`; the
    transpose of [_tangent_terms][]. `rows_bar` is a cotangent on the residual rows themselves,
    read only by the Gauss-Newton form, and `J_r` is `None` in the exact form.
    """
    if not closures.gauss_newton_rule:
        _, pullback, derivatives = jax.vjp(residual_of(closures, x, record), args, has_aux=True)
        system = system_at(closures, bounds, x, record, args, derivatives)

        def back(v_bar, d_grad_bar, d_eq_bar, d_ineq_bar, rows_bar=None):
            return pullback((v_bar, d_grad_bar, d_eq_bar, d_ineq_bar))[0]

        return system, back, None
    (rows, _, _), pullback, (jac_eq, jac_ineq) = jax.vjp(rows_of(closures, x), args, has_aux=True)
    if jacobian is None:
        jacobian = closures.residual_and_jacobian(x, args)[1]
    derivatives = (jacobian.T @ rows, jac_eq, jac_ineq)
    system = system_at(closures, bounds, x, record, args, derivatives, jacobian)

    def back(v_bar, d_grad_bar, d_eq_bar, d_ineq_bar, rows_bar=None):
        direct = v_bar * rows + jacobian @ d_grad_bar
        if rows_bar is not None:
            direct = direct + rows_bar
        return pullback((direct, d_eq_bar, d_ineq_bar))[0]

    return system, back, jacobian


def _kkt_cotangent(closures, bounds, kkt, x, record, args, cotangents, jacobian, rows_bar):
    """[kkt_cotangent][], with a cotangent on the residual rows besides (Gauss-Newton form)."""
    n_eq = record.multipliers_equality.shape[0]
    x_bar, record_bar = cotangents
    lam_e_bar, lam_i_bar, s_bar, (zl_bar, zu_bar), v_bar = (
        _materialise(leaf, like)
        for leaf, like in zip(_differentiable(record_bar), _differentiable(record))
    )
    x_bar = _materialise(x_bar, x)
    system, back, jacobian = _cotangent_terms(closures, bounds, x, record, args, jacobian)
    if rows_bar is not None:
        x_bar = x_bar + jacobian.T @ rows_bar
    active, degenerate = system.active, system.active.degenerate
    jac_eq, jac_ineq = system.jac_eq, system.jac_ineq

    open_rows = jnp.where(degenerate | active.inequality, 0.0, s_bar)
    barrier = degenerate & ~active.pinned
    st_bar = jnp.where(
        barrier, 0.0, jnp.where(active.lower, zl_bar, 0.0) - jnp.where(active.upper, zu_bar, 0.0)
    )
    dx_bar = (
        x_bar
        + v_bar * system.gradient
        - jac_ineq.T @ open_rows
        + jnp.where(degenerate, system.sigma_upper * zu_bar - system.sigma_lower * zl_bar, 0.0)
        + system.hessian.T @ st_bar
    )
    dlam_bar = jnp.concatenate(
        [
            lam_e_bar + jac_eq @ st_bar,
            lam_i_bar + jac_ineq @ st_bar - jnp.where(degenerate, system.slack_ratio * s_bar, 0.0),
        ]
    )
    p, q, solved = kkt.solve(system.primal, system.rows, 0.0, system.dual, dx_bar, dlam_bar)
    d_grad_bar = st_bar - system.keep * p
    d_eq_bar = -q[:n_eq]
    d_ineq_bar = -open_rows - system.rows_i * q[n_eq:]
    # The transpose of the bounds' terms in [_kkt_tangent][]: a fixed coordinate's tangent
    # enters the solution directly and both right-hand sides, and the degenerate barrier rows.
    fixed_bar = (
        dx_bar
        - system.hessian.T @ (system.keep * p)
        - jac_eq.T @ q[:n_eq]
        - jac_ineq.T @ (system.rows_i * q[n_eq:])
    )
    fixed_bar = (1.0 - system.keep) * fixed_bar
    lower_bar = jnp.where(active.lower, fixed_bar, 0.0) + jnp.where(
        degenerate, system.sigma_lower * (p + zl_bar), 0.0
    )
    upper_bar = jnp.where(active.lower, 0.0, fixed_bar) + jnp.where(
        degenerate, system.sigma_upper * (p - zu_bar), 0.0
    )
    args_bar = back(v_bar, d_grad_bar, d_eq_bar, d_ineq_bar, rows_bar)
    return args_bar, (lower_bar, upper_bar), solved


def kkt_cotangent(
    closures: Closures,
    bounds: Bounds,
    kkt: KKTSolve,
    x: Float[Array, "n"],
    record: SolveRecord,
    args: PyTree,
    cotangents: tuple[Float[Array, "n"], SolveRecord],
    jacobian: Float[Array, "m n"] | None = None,
) -> tuple[PyTree, tuple[Float[Array, "n"], Float[Array, "n"]], Bool[Array, ""]]:
    """The cotangents on `args` and on the bounds of cotangents on `(x*, record)`, and whether its system solved.

    The transpose of [kkt_tangent][]: the cotangents on the point and the multipliers, carried
    back through the slack and bound-multiplier rows, are the right-hand side of the transposed
    KKT system, whose solution is the multiplier of the cotangent. The cotangent on `args` is
    the vector-Jacobian product of one evaluation's objective, Lagrangian gradient and rows in
    `args` with that multiplier and the cotangents the rows read directly; the cotangent on
    the bounds, `(lower_bar, upper_bar)`, collects what reaches the coordinates they fix.
    """
    args_bar, bounds_bar, solved = _kkt_cotangent(
        closures, bounds, kkt, x, record, args, cotangents, jacobian, None
    )
    return args_bar, bounds_bar, solved


def rows_cotangent(
    closures: Closures,
    bounds: Bounds,
    kkt: KKTSolve,
    x: Float[Array, "n"],
    record: SolveRecord,
    args: PyTree,
    rows_bar: Float[Array, "m"],
    jacobian: Float[Array, "m n"] | None = None,
) -> tuple[PyTree, Bool[Array, ""]]:
    """The cotangent on `args` of a cotangent on the residual rows at the solution, and whether it solved.

    The transpose of [rows_tangent][]: one transposed KKT solve and one vector-Jacobian product
    of the rows in `args`, whatever the size of `args`. Gauss-Newton form only.
    """
    zeros = jax.tree.map(_zero, record)
    args_bar, _, solved = _kkt_cotangent(
        closures, bounds, kkt, x, record, args, (None, zeros), jacobian, rows_bar
    )
    return args_bar, solved


def _differentiable(record: SolveRecord) -> tuple:
    """The record's fields that carry a derivative, in the order the tangent map writes them."""
    return (
        record.multipliers_equality,
        record.multipliers_inequality,
        record.slacks,
        record.bound_multipliers,
        record.value,
    )


def _materialise(cotangent: PyTree, like: PyTree) -> PyTree:
    """`cotangent` with zeros of `like`'s shape for a symbolic zero or a `float0`."""

    def leaf(c, x):
        if c is None or getattr(c, "dtype", None) == jax.dtypes.float0:
            return jnp.zeros_like(x)
        return c

    return jax.tree.map(leaf, cotangent, like, is_leaf=lambda c: c is None)


def _zero(leaf: Array) -> ArrayLike:
    """The zero tangent of a gradient-dead leaf (`float0` for an integer or bool)."""
    return zero_from_primal(leaf, symbolic_zeros=False)


def _bound_tangents(dbounds, x: Float[Array, "n"]) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
    """`(d_lower, d_upper)`, zeros for `None`."""
    if dbounds is None:
        return jnp.zeros_like(x), jnp.zeros_like(x)
    return dbounds


def _limits_of(prepare: Callable, static: PyTree, x: Float[Array, "n"]) -> Callable:
    """`dynamic -> (lower, upper)`: the bounds the rule reads, as a function of the problem's arrays."""

    def limits(dynamic):
        bounds = prepare(eqx.combine(dynamic, static), x)[1]
        return bounds.lower, bounds.upper

    return limits


def _carried(path: tuple) -> bool:
    """Whether a problem leaf at `path` is one the rule reads a derivative in, or never reads.

    The bounds carry their derivative through the rule; the problem's own `args` are not what
    a run reads (it reads the `args` it is given), so their tangent meets nothing.
    """
    return bool(path) and getattr(path[0], "name", None) in ("bounds", "args")


def _refuse(live: list[tuple[str, bool]]) -> None:
    """Raise `ConfigurationError` naming the first derivative-free leaf that carries a tangent."""
    for name, perturbed in live:
        if perturbed:
            raise ConfigurationError(
                f"A compiled solver's solution carries no derivative in {name}: the start point, "
                "the warm start and the problem's objective and constraint scales are "
                "derivative-free (CompiledSolver.run). Stop the tangent there with "
                "jax.lax.stop_gradient, or pass the value through `args`."
            )


def _named(dynamic: PyTree, x0: PyTree, warm: PyTree, flag: Callable) -> list[tuple[str, bool]]:
    """`(name, flag(leaf))` for every derivative-free leaf of the problem, `x0` and `warm`."""
    named = [
        (f"problem{jax.tree_util.keystr(path)}", flag(leaf))
        for path, leaf in jax.tree_util.tree_leaves_with_path(dynamic, is_leaf=_is_marked)
        if not _carried(path)
    ]
    named += [("x0", flag(leaf)) for leaf in jax.tree.leaves(x0, is_leaf=_is_marked)]
    named += [
        (f"warm_start{jax.tree_util.keystr(path)}", flag(leaf))
        for path, leaf in jax.tree_util.tree_leaves_with_path(warm, is_leaf=_is_marked)
    ]
    return named


def _is_marked(leaf: object) -> bool:
    """Whether `leaf` is a symbolic zero or a reverse rule's marked primal."""
    return isinstance(leaf, (SymbolicZero, CustomVJPPrimal))


def _has_tangent(tangent: object) -> bool:
    """Whether a forward rule's tangent leaf is neither a symbolic zero nor an integer's."""
    if isinstance(tangent, SymbolicZero):
        return False
    return getattr(tangent, "dtype", None) != jax.dtypes.float0


def _instantiate(tangents: PyTree, primals: PyTree) -> PyTree:
    """`tangents` with each symbolic zero made the zero of its primal."""

    def leaf(t, p):
        return _zero(jnp.asarray(p)) if isinstance(t, SymbolicZero) else t

    return jax.tree.map(leaf, tangents, primals, is_leaf=_is_marked)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def _entry(primal, prepare, kkt, static, dynamic, x0, args, warm, limits):
    """`primal(problem, x0, args, warm, limits) -> (x, record)` with the implicit derivative."""
    del prepare, kkt
    return primal(eqx.combine(dynamic, static), x0, args, warm, limits)


def _entry_jvp(primal, prepare, kkt, static, primals, tangents):
    """The KKT tangent at the solution, in `args` and the problem's bounds.

    A tangent on `x0`, `warm` or another array of the problem raises; `limits` carry none.
    """
    dynamic, x0, args, warm, limits = primals
    d_dynamic, dx0, dargs, dwarm, _ = tangents
    _refuse(_named(d_dynamic, dx0, dwarm, _has_tangent))
    out = _entry(primal, prepare, kkt, static, dynamic, x0, args, warm, limits)
    x, record = out
    closures, bounds = prepare(eqx.combine(dynamic, static), x)
    d_dynamic, dargs = _instantiate((d_dynamic, dargs), (dynamic, args))
    _, dbounds = jax.jvp(_limits_of(prepare, static, x), (dynamic,), (d_dynamic,))
    valid = record.status == TerminationStatus.KKT_TOL
    dargs, dbounds = zero_where_not(valid, (dargs, dbounds))
    dx, drecord, solved = kkt_tangent(
        closures, bounds, kkt, x, record, args, dargs, dbounds=dbounds
    )
    return out, poison((dx, drecord), valid & solved)


_entry.defjvp(_entry_jvp, symbolic_zeros=True)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _reverse_entry(primal, prepare, kkt, static, dynamic, x0, args, warm, limits):
    """`primal(problem, x0, args, warm, limits) -> (x, record)` with the reverse implicit derivative."""
    del prepare, kkt
    return primal(eqx.combine(dynamic, static), x0, args, warm, limits)


def _reverse_fwd(primal, prepare, kkt, static, dynamic, x0, args, warm, limits):
    """The run, keeping its inputs and solution for the backward pass.

    A perturbed `x0`, `warm` or problem array other than the bounds raises.
    """
    _refuse(_named(dynamic, x0, warm, lambda leaf: leaf.perturbed))
    dynamic, x0, args, warm, limits = jax.tree.map(
        lambda leaf: leaf.value, (dynamic, x0, args, warm, limits), is_leaf=_is_marked
    )
    out = _reverse_entry(primal, prepare, kkt, static, dynamic, x0, args, warm, limits)
    return out, (dynamic, x0, args, warm, limits, out)


def _reverse_bwd(primal, prepare, kkt, static, residuals, cotangents):
    """The transposed KKT solve at the solution, to `args` and the bounds; the rest get zeros.

    A run that did not end with `KKT_TOL`, or whose system did not solve, has no supported
    derivative: its cotangents are NaN, or exactly zero where its outputs' cotangents are, so
    that a discarded run contributes nothing.
    """
    del primal
    dynamic, x0, args, warm, limits, (x, record) = residuals
    cotangents = _instantiate(cotangents, (x, record))
    closures, bounds = prepare(eqx.combine(dynamic, static), x)
    args_bar, bounds_bar, solved = kkt_cotangent(closures, bounds, kkt, x, record, args, cotangents)
    _, pullback = jax.vjp(_limits_of(prepare, static, x), dynamic)
    (dynamic_bar,) = pullback(bounds_bar)
    valid = (record.status == TerminationStatus.KKT_TOL) & solved
    used = jnp.any(
        jnp.stack(
            [
                jnp.any(leaf != 0)
                for leaf in jax.tree.leaves(cotangents)
                if getattr(leaf, "dtype", None) != jax.dtypes.float0
            ]
        )
    )

    def gate(leaf):
        if getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        unsupported = jnp.where(used, jnp.full_like(leaf, jnp.nan), jnp.zeros_like(leaf))
        return jnp.where(valid, leaf, unsupported)

    dead = jax.tree.map(lambda leaf: jnp.asarray(_zero(leaf)), (x0, warm, limits))
    gated = jax.tree.map(gate, (dynamic_bar, args_bar))
    return gated[0], dead[0], gated[1], dead[1], dead[2]


_reverse_entry.defvjp(_reverse_fwd, _reverse_bwd, symbolic_zeros=True)


def implicit_run(
    primal: Callable,
    prepare: Callable,
    kkt: KKTSolve,
    problem: PyTree,
    x0: Float[ArrayLike, "n"],
    args: PyTree,
    warm: SolveRecord | None,
    adjoint: AbstractAdjoint,
    limits: tuple = (None, None),
) -> tuple[Float[Array, "n"], SolveRecord]:
    """Run `primal` with the implicit derivative of its solution in `args` and the bounds.

    Args:
        primal: `(problem, x0, args, warm, limits) -> (x, record)`, the solver's loop; never differentiated.
        prepare: `(problem, x) -> (closures, bounds)`, the raw closures and bounds of `problem`.
        kkt: The backend that solves the derivative's KKT system.
        problem: The numeric problem; its bounds carry a derivative, its scales and `x0` none
            (a tangent on them raises), and its own `args` are not read.
        x0: The start point, derivative-free: a tangent on it raises.
        args: What the derivative is taken in.
        warm: The warm-start record or `None`, derivative-free: a tangent on it raises.
        adjoint: [ForwardAdjoint][diff_soup.ForwardAdjoint] for the forward rule,
            [ImplicitAdjoint][diff_soup.nonlinear.ImplicitAdjoint] for the reverse one.
        limits: `(max_steps, tol)` of the run, runtime values or `None`, gradient-dead.

    Returns:
        `(x, record)`.
    """
    dynamic, static = eqx.partition(problem, eqx.is_array)
    entry = _ENTRIES[adjoint_mode(adjoint)]
    return entry(primal, prepare, kkt, static, dynamic, x0, args, warm, limits)


def adjoint_mode(adjoint: AbstractAdjoint) -> str:
    """`"jvp"` or `"vjp"`: the implicit derivative `adjoint` selects, or `ConfigurationError`."""
    mode = getattr(type(adjoint), "ift_mode", None)
    if not isinstance(adjoint, AbstractAdjoint) or mode not in _ENTRIES:
        raise ConfigurationError(
            "A compiled solver's adjoint must be ForwardAdjoint() (the forward rule) or "
            f"ImplicitAdjoint() (the reverse one); got {type(adjoint).__name__}."
        )
    return mode


_ENTRIES = {"jvp": _entry, "vjp": _reverse_entry}
