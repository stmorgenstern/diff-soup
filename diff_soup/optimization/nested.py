"""A nested least-squares fit: shared values over a batch of inner fits, posed as one problem.

Each of `count` members has own values `z_m` and rows `r_m(z_m, w)`. A member's own values
solve its inner least-squares fit at the shared values `w`; the shared values minimise the sum
of the members' costs. [NestedLeastSquares][diff_soup.optimization.NestedLeastSquares] poses
the outer fit as a [LeastSquaresProblem][diff_soup.optimization.LeastSquaresProblem] over `w`:
its rows are every member's rows at its own values after its inner steps at `w`, and its
Jacobian is the projected one, `(I - P) J_w`, which the Gauss-Newton form of the implicit rule
gives from the inner fit's `J_z` and `p` forward tangents of the rows in `w`. A Gauss-Newton
step on these rows is the joint fit's step with the own values eliminated, so one inner step
per outer step is the joint fit and inner fits run to convergence are variable projection;
[InnerSteps][diff_soup.optimization.InnerSteps] sets how far they run.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.custom_derivatives import zero_from_primal
from jaxtyping import Array, ArrayLike, Float, PyTree

from diff_soup._adjoints import AbstractAdjoint, ForwardAdjoint
from diff_soup._errors import ConfigurationError
from diff_soup.optimization import _derived
from diff_soup.optimization._binding import Derived
from diff_soup.optimization._least_squares import LeastSquaresProblem
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.lanes import _pad, map_lanes, per_lane, placement
from diff_soup.optimization.solvers import _hessian
from diff_soup.optimization.solvers._closures import closures_of, scaled_closures
from diff_soup.optimization.solvers._core import ConstrainedSolver
from diff_soup.optimization.solvers._implicit import adjoint_mode, rows_cotangent, rows_tangent
from diff_soup.optimization.solvers._protocol import SolveRecord, _int, _raw_gauss_newton

#: The statuses whose rows are poisoned with NaN: the member's fit failed or diverged.
_FAILED = (
    TerminationStatus.INFEASIBLE,
    TerminationStatus.SOLVER_FAILED,
    TerminationStatus.DIVERGED,
)


class InnerSteps(eqx.Module):
    """How far each member's inner fit runs at one outer evaluation.

    Each fit starts from the member's last own values and multipliers. At a trial of
    `NewtonTrustRegion` whose step has length `radius` in its scaled units (at most the trust
    radius), it runs until its scaled optimality error is at most `max(tol_min, kappa *
    radius**2)`, or until a step changes its cost by no more than its noise floor, with at
    least one step and at most `max_steps`. The square keeps a member's cost error, of the order
    of its tolerance on an active bound, below the decrease the step predicts, of the order of
    `radius**2`. Every other evaluation, and `NestedLeastSquares.solve`, runs to `tol_min`.

    Args:
        tol_min: The tightest inner tolerance, or `None` for the inner solver's `tol`.
        kappa: The inner tolerance per unit of squared outer step length.
        max_steps: The inner step budget, or `None` for the inner solver's `max_steps`.
        noise_floor: The change of a member's cost at or below which its fit stops, one for every member or one per member; `0` for none.
    """

    tol_min: float | None = eqx.field(static=True, default=None)
    kappa: float = eqx.field(static=True, default=0.1)
    max_steps: int | None = eqx.field(static=True, default=None)
    noise_floor: Float[ArrayLike, "..."] = 0.0


class _Spec(eqx.Module):
    """What the nested closures read besides `(w, args)`: static, hashable, array-free."""

    solver: ConstrainedSolver = eqx.field(static=True)
    inner: PyTree = eqx.field(static=True)
    tol_min: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    keep: bool = eqx.field(static=True)
    mesh: object = eqx.field(static=True)
    count: int = eqx.field(static=True)


def _zero(leaf: Array) -> Array:
    """The zero tangent of `leaf` (`float0` for an integer or bool)."""
    return jnp.asarray(zero_from_primal(leaf, symbolic_zeros=False))


def _fit(spec: _Spec, problem, x0, args, warm, tol, floor):
    """One member's inner fit: `(x, record, J_z)`, `J_z` kept from its last evaluation or `None`.

    The solver's own loop, run from `x0` (and `warm`'s multipliers) with the budget
    `spec.max_steps` and the tolerance `tol`, takes at least one step, and stops with
    `COST_TOL` once a step changes the cost by no more than `floor` (when `floor > 0`).
    """
    solver = spec.solver
    raw, closures, bounds = solver._prepare(problem, x0)
    if spec.keep:
        raw = raw._replace(keep_jacobian=True)
        closures = scaled_closures(raw)
    state = solver._init(problem, raw, closures, bounds, x0, args, warm, (spec.max_steps, tol))
    met = state.done & (state.status == TerminationStatus.KKT_TOL)
    state = dataclasses.replace(
        state,
        done=state.done & ~met,
        status=_int(jnp.where(met, TerminationStatus.MAX_STEPS, state.status)),
    )

    def cost(current):
        return current.evaluation.value / current.scales.objective

    def body(carry):
        current, last = carry
        new = solver._advance(closures, bounds, current, (args, current.scales))
        quiet = (floor > 0.0) & ~new.done & (jnp.abs(last - cost(new)) <= floor)
        new = dataclasses.replace(
            new,
            done=new.done | quiet,
            status=_int(jnp.where(quiet, TerminationStatus.COST_TOL, new.status)),
        )
        return new, cost(new)

    state, _ = lax.while_loop(lambda carry: ~carry[0].done, body, (state, cost(state)))
    jacobian = None
    if spec.keep:
        jacobian = state.evaluation.jacobian / jnp.sqrt(state.scales.objective)
    return state.x, solver._record(state, bounds), jacobian


def _usable(record: SolveRecord, *arrays: Array) -> Array:
    """Whether a member's fit neither failed nor diverged and its arrays are finite."""
    failed = jnp.isin(record.status, jnp.asarray([int(code) for code in _FAILED]))
    finite = [jnp.all(jnp.isfinite(a)) for a in arrays]
    return ~failed & functools.reduce(jnp.logical_and, finite, jnp.asarray(True))


def _poison(ok: Array, tree: PyTree) -> PyTree:
    """`tree` with NaN in every leaf where `ok` is false."""
    return jax.tree.map(lambda leaf: jnp.where(ok, leaf, jnp.nan), tree)


def _member_args(w: Array, member: Mapping) -> dict:
    """The inner problem's `args` of one member at the shared values `w`."""
    return {"shared": w, **member}


def _shared_tangent(member: Mapping, dw: Array) -> dict:
    """The tangent of a member's `args` along `dw` in the shared values alone."""
    return {"shared": dw, **jax.tree.map(_zero, member)}


def _linearized(spec, problem, w, member, x0, warm, tol, floor):
    """A member's fit at `w`, its rows and the projected Jacobian `(I - P) J_w`, `(m, p)`."""
    args = _member_args(w, member)
    x, record, kept = _fit(spec, problem, x0, args, warm, tol, floor)
    closures, bounds = _raw_gauss_newton(problem, x)

    def column(e):
        return rows_tangent(
            closures, bounds, spec.solver._kkt, x, record, args, _shared_tangent(member, e), kept
        )

    basis = jnp.eye(w.shape[0], dtype=w.dtype)
    rows, jacobian, solved = jax.vmap(column, out_axes=(None, 1, 0))(basis)
    ok = _usable(record, rows, jacobian) & jnp.all(solved)
    return x, record, _poison(ok, rows), _poison(ok, jacobian), ok


def _fitted_rows(spec, problem, w, member, x0, warm, tol, floor):
    """A member's fit at `w`, its rows (NaN when it failed), and the Jacobian it kept."""
    args = _member_args(w, member)
    x, record, kept = _fit(spec, problem, x0, args, warm, tol, floor)
    rows = _raw_gauss_newton(problem, x)[0].residual(x, args)
    return x, record, kept, _poison(_usable(record, rows), rows)


def _members(spec: _Spec, w: Array, args: dict, tol: Array, fn: Callable, *extra: Array) -> PyTree:
    """`fn(spec, problem, w, member, x0, warm, tol, floor, *extra)` over every member, through the layout.

    `w`, `tol` and `extra` reach every member as arguments of the mapped function, never as
    values it closes over (a closed-over tangent inside `shard_map` does not transpose).
    """
    problem = eqx.combine(args["inner"], spec.inner)

    def one(member, x0, warm, floor, w, tol, *extra):
        return fn(spec, problem, w, member, x0, warm, tol, floor, *extra)

    shared = tuple(_broadcast(v, spec.count) for v in (w, jnp.asarray(tol, w.dtype), *extra))
    return _over(spec, one, (args["members"], args["x"], args["record"], args["floor"], *shared))


def _broadcast(value: Array, count: int) -> Array:
    """`value` repeated for every member on a new leading axis."""
    return jnp.broadcast_to(value, (count, *jnp.shape(value)))


def _over(spec: _Spec, fn: Callable, batch: tuple) -> PyTree:
    """`fn` mapped over the members' axis of `batch`, padded to the mesh and cut back."""
    if spec.mesh is None:
        return map_lanes(fn, batch, None)
    devices = int(spec.mesh.shape["data"])
    padded = -(-spec.count // devices) * devices
    out = map_lanes(fn, _pad(batch, spec.count, padded), spec.mesh)
    return jax.tree.map(lambda leaf: leaf[: spec.count], out)


def _evaluate(spec: _Spec, w: Array, args: dict, radius: Array) -> tuple:
    """`((cost, J^T r, J^T J, failed), args_after, rows, J)` at `w`, the inner fits run to `max(tol_min, kappa * radius**2)`."""
    tol = jnp.maximum(spec.tol_min, spec.kappa * radius**2)
    x, record, rows, jacobian, ok = _members(spec, w, args, tol, _linearized)
    statistics = (
        0.5 * jnp.sum(rows * rows),
        jnp.einsum("cmp,cm->p", jacobian, rows),
        jnp.einsum("cmp,cmq->pq", jacobian, jacobian),
        jnp.sum(~ok),
    )
    return statistics, {**args, "x": x, "record": record}, rows, jacobian


def _gauss_newton(spec, w, args):
    """The outer Gauss-Newton statistics with the inner fits run to `tol_min`."""
    return _evaluate(spec, w, args, jnp.zeros((), w.dtype))[0]


def _carried_gauss_newton(spec, w, radius, args):
    """The statistics at a trial step of length `radius`, and the args its acceptance leaves."""
    return _evaluate(spec, w, args, radius)[:2]


def _value_and_gradient(spec, w, args):
    """The outer cost and its gradient `J^T r`."""
    statistics = _gauss_newton(spec, w, args)
    return statistics[0], statistics[1]


def _residual_and_jacobian(spec, w, args):
    """Every member's rows, stacked, and the projected Jacobian."""
    _, _, rows, jacobian = _evaluate(spec, w, args, jnp.zeros((), w.dtype))
    return rows.reshape(-1), jacobian.reshape(-1, w.shape[0])


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _member_forward(spec, dynamic, w, member, x0, warm, tol, floor):
    """One member's rows at its fit, differentiated in `w` by the forward Gauss-Newton rule."""
    problem = eqx.combine(dynamic, spec.inner)
    return _fitted_rows(spec, problem, w, member, x0, warm, tol, floor)[3]


@_member_forward.defjvp
def _member_forward_jvp(spec, primals, tangents):
    """One KKT solve per tangent; everything but `w` is gradient-dead."""
    dynamic, w, member, x0, warm, tol, floor = primals
    dw = tangents[1]
    problem = eqx.combine(dynamic, spec.inner)
    member_args = _member_args(w, member)
    x, record, kept = _fit(spec, problem, x0, member_args, warm, tol, floor)
    closures, bounds = _raw_gauss_newton(problem, x)
    rows, d_rows, solved = rows_tangent(
        closures,
        bounds,
        spec.solver._kkt,
        x,
        record,
        member_args,
        _shared_tangent(member, dw),
        kept,
    )
    ok = _usable(record, rows) & solved
    return _poison(ok, rows), _poison(ok, d_rows)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _member_reverse(spec, dynamic, w, member, x0, warm, tol, floor):
    """One member's rows at its fit, differentiated in `w` by the reverse Gauss-Newton rule."""
    problem = eqx.combine(dynamic, spec.inner)
    return _fitted_rows(spec, problem, w, member, x0, warm, tol, floor)[3]


def _member_reverse_fwd(spec, dynamic, w, member, x0, warm, tol, floor):
    """The fit, keeping its point, record and Jacobian for the backward pass."""
    problem = eqx.combine(dynamic, spec.inner)
    x, record, kept, rows = _fitted_rows(spec, problem, w, member, x0, warm, tol, floor)
    return rows, ((dynamic, w, member, x0, warm, tol, floor), x, record, kept)


def _member_reverse_bwd(spec, residuals, rows_bar):
    """One transposed KKT solve, whatever the number of shared values; NaN for a failed member."""
    inputs, x, record, kept = residuals
    dynamic, w, member = inputs[:3]
    problem = eqx.combine(dynamic, spec.inner)
    member_args = _member_args(w, member)
    closures, bounds = _raw_gauss_newton(problem, x)
    args_bar, solved = rows_cotangent(
        closures, bounds, spec.solver._kkt, x, record, member_args, rows_bar, kept
    )
    ok = _usable(record, args_bar["shared"]) & solved
    unsupported = jnp.where(jnp.any(rows_bar != 0.0), jnp.nan, 0.0)
    w_bar = jnp.where(ok, args_bar["shared"], unsupported)
    dead = jax.tree.map(_zero, inputs)
    return (dead[0], w_bar, *dead[2:])


_member_reverse.defvjp(_member_reverse_fwd, _member_reverse_bwd)
_RULES = {"jvp": _member_forward, "vjp": _member_reverse}


def _rows(spec, rule, w, args):
    """Every member's rows at its fit to `tol_min`, `(count, m)`, with the rule `rule` names in `w`."""
    member_rule = _RULES[rule]

    def one(spec, problem, w, member, x0, warm, tol, floor):
        dynamic = eqx.filter(problem, eqx.is_array)
        return member_rule(spec, dynamic, w, member, x0, warm, tol, floor)

    return _members(spec, w, args, jnp.asarray(spec.tol_min, w.dtype), one)


def _residual(spec, rule, w, args):
    """Every member's rows, stacked, with the implicit derivative in `w` that `rule` names."""
    return _rows(spec, rule, w, args).reshape(-1)


def _value(spec, rule, w, args):
    """Half the sum of every member's squared rows."""
    rows = _residual(spec, rule, w, args)
    return 0.5 * jnp.sum(rows * rows)


def _solved(spec, w, args):
    """Every member's own values and record at `w`, its fit run to `tol_min`."""
    x, record, _, _ = _members(spec, w, args, jnp.asarray(spec.tol_min, w.dtype), _fitted_rows)
    return x, record


class NestedLeastSquares(eqx.Module):
    """A least-squares fit of shared values over a batch of inner least-squares fits.

    Every member solves the same inner problem with its own `args` and own values. `problem`
    is the outer [LeastSquaresProblem][diff_soup.optimization.LeastSquaresProblem] over the
    shared values: its rows are every member's rows at its own values after its inner steps,
    its Jacobian the projected one `(I - P) J_w`, and its Gauss-Newton statistics are reduced
    member by member (`reduced_gauss_newton`), so any outer optimizer takes it. Its `args`
    hold the members' state: their own values and their solve records, which
    `NewtonTrustRegion` replaces after each accepted step and returns in
    `result.info["args"]`. A member whose fit failed or diverged gives NaN rows, which the
    trust region rejects as a trial; one that ran out of inner steps does not.

    Attributes:
        problem: The outer problem over the shared values.
    """

    problem: LeastSquaresProblem
    _spec: _Spec = eqx.field(static=True)

    @classmethod
    def pose(
        cls,
        problem: LeastSquaresProblem,
        solver: ConstrainedSolver,
        members: Mapping,
        x0: Float[ArrayLike, "count q"],
        w0: Float[ArrayLike, " p"],
        *,
        layout: object = None,
        steps: InnerSteps | None = None,
        adjoint: AbstractAdjoint | None = None,
        keep_jacobian: bool = True,
    ) -> NestedLeastSquares:
        """Pose the nested fit once.

        Args:
            problem: The inner least-squares problem; its closures take `args = {"shared": w, **member}`.
            solver: The compiled solver of the inner fits, such as `InteriorPoint(hessian="gauss-newton")`.
            members: Each member's `args` besides `"shared"`, a mapping batched on the leading axis.
            x0: The members' starting own values, `(count, q)`.
            w0: The starting shared values, `(p,)`.
            layout: A `Layout` whose mesh has a `"data"` axis places the members across devices; `None` for one device.
            steps: The [InnerSteps][diff_soup.optimization.InnerSteps], or `None` for the defaults.
            adjoint: `ForwardAdjoint()` (default) differentiates the rows in `w` by one KKT solve per member and tangent; `ImplicitAdjoint()` by one transposed solve per member for `jax.grad`.
            keep_jacobian: Whether the projection takes `J_z` from the inner fit's last evaluation, when its solver steps on Gauss-Newton; else it is formed again at the fit.

        Returns:
            The posed fit.
        """
        if not isinstance(problem, LeastSquaresProblem) or closures_of(problem).residual is None:
            raise ConfigurationError(
                "NestedLeastSquares needs a LeastSquaresProblem with residual rows as its inner problem."
            )
        if not isinstance(solver, ConstrainedSolver):
            raise ConfigurationError(
                "NestedLeastSquares runs the inner fits with a compiled constrained solver "
                f"(InteriorPoint, SQP or AugmentedLagrangian); got {type(solver).__name__}."
            )
        if not isinstance(members, Mapping) or "shared" in members:
            raise ConfigurationError(
                "NestedLeastSquares members must be a mapping of each member's args, without "
                "the key 'shared', which holds the shared values."
            )
        steps = InnerSteps() if steps is None else steps
        adjoint = ForwardAdjoint() if adjoint is None else adjoint
        rule = adjoint_mode(adjoint)
        x0 = jnp.asarray(x0)
        w0 = jnp.asarray(w0, dtype=x0.dtype)
        if x0.ndim != 2 or w0.ndim != 1:
            raise ConfigurationError(
                f"NestedLeastSquares x0 must be (count, q) and w0 (p,); got {x0.shape} and {w0.shape}."
            )
        count = int(x0.shape[0])
        sizes = {int(np.shape(leaf)[0]) for leaf in jax.tree.leaves(members)}
        if sizes - {count}:
            raise ConfigurationError(
                f"NestedLeastSquares members must be batched on {count} members; got {sorted(sizes)}."
            )
        if hasattr(solver, "_for_dtype"):
            solver = solver._for_dtype(x0.dtype)
        inner = problem.with_args(None)
        dynamic, static = eqx.partition(inner, eqx.is_array)
        keep = keep_jacobian and _hessian.uses_gauss_newton(solver.hessian, closures_of(problem))
        spec = _Spec(
            solver=solver,
            inner=static,
            tol_min=float(solver.tol if steps.tol_min is None else steps.tol_min),
            kappa=float(steps.kappa),
            max_steps=int(solver.max_steps if steps.max_steps is None else steps.max_steps),
            keep=bool(keep),
            mesh=placement(layout),
            count=count,
        )
        args = {
            "inner": dynamic,
            "members": members,
            "x": x0,
            "record": None,
            "floor": per_lane(jnp.asarray(steps.noise_floor, x0.dtype), count),
        }
        jit = _derived.jit
        outer = LeastSquaresProblem(
            x0=w0,
            structure=frozenset(
                {"value_gradient", "residual_jacobian", "gauss_newton", "carried_args"}
            ),
            args=args,
            ad="forward",
            reduced_gauss_newton=True,
            _value=jit(Derived(_value, spec, rule)),
            _value_and_gradient=jit(Derived(_value_and_gradient, spec)),
            _gauss_newton=jit(Derived(_gauss_newton, spec)),
            _residual=jit(Derived(_residual, spec, rule)),
            _residual_and_jacobian=jit(Derived(_residual_and_jacobian, spec)),
            _carried_gauss_newton=jit(Derived(_carried_gauss_newton, spec)),
        )
        return cls(outer, spec)

    def solve(
        self, w: Float[ArrayLike, " p"], args: dict | None = None
    ) -> tuple[Float[Array, "count q"], SolveRecord]:
        """Every member's own values and solve record at the shared values `w`, fitted to `tol_min`.

        Args:
            w: The shared values.
            args: The state to start the members from, such as `result.info["args"]` of a trust-region fit; `None` for `problem.args`.

        Returns:
            `(x, record)`, batched on the members.
        """
        args = self.problem.args if args is None else args
        return _solve(self._spec, jnp.asarray(w, dtype=self.problem.x0.dtype), args)


@functools.partial(jax.jit, static_argnums=0)
def _solve(spec, w, args):
    """`_solved` compiled once per posed fit."""
    return _solved(spec, w, args)


__all__ = ["InnerSteps", "NestedLeastSquares"]
