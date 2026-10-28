"""Solver-problem lowering shared by Recipe IR and explicit callback adapters.

The builders in this module consume explicit callbacks and host structure;
they do not depend on either authoring surface. Recipe lowering closes over
its static IR half, while other adapters supply callbacks to the same
constructions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.linear import SparseTopology
from diff_soup.problem._cache import BoundedContentCache
from diff_soup.problem.implicit._coo import coo_matvec
from diff_soup.problem.implicit.operator import MassOperator
from diff_soup.problem.implicit.problems import ImplicitProblem, MassProblem
from diff_soup.problem.implicit.residual import ResidualLinearization, ResidualSparsity
from diff_soup.problem.implicit.structure import DAEStructure, Dependencies

__all__ = [
    "augmented_rhs_sparsity",
    "solver_problem_from_ir",
    "default_theta_from_ir",
    "lower_mass_form",
]


class _IRTheta(eqx.Module):
    """Dynamic Recipe leaves in the historical ``theta.inner`` shape."""

    inner: object


def augmented_rhs_sparsity(topology, n_u: int, n_lam: int):
    """Extend one state pattern with conservative dense bordered-global couplings."""
    if topology is None:
        return "dense"
    n_u = int(n_u)
    n_lam = int(n_lam)
    n = n_u + n_lam
    rows = [np.asarray(topology.rows, dtype=np.int32)]
    cols = [np.asarray(topology.cols, dtype=np.int32)]
    if n_lam:
        state = np.arange(n_u, dtype=np.int32)
        lam = np.arange(n_u, n, dtype=np.int32)
        rows += [np.repeat(state, n_lam), np.repeat(lam, n_u), np.repeat(lam, n_lam)]
        cols += [np.tile(lam, n_u), np.tile(state, n_lam), np.tile(lam, n_lam)]
    return SparseTopology(rows=np.concatenate(rows), cols=np.concatenate(cols), n_dofs=n)


def _pinned_mask(pinned_rows, n: int) -> np.ndarray:
    """Boolean mask of the pinned rows (Dirichlet and symmetry rows of differential fields)."""
    mask = np.zeros((int(n),), dtype=bool)
    if pinned_rows is not None and len(pinned_rows):
        mask[np.asarray(pinned_rows, dtype=np.int64)] = True
    return mask


def _check_derivative_field(
    structure: DAEStructure,
    offsets: dict[str, tuple[int, int]],
    out_field: str,
    in_field: str,
    pinned: Optional[np.ndarray] = None,
) -> tuple[int, int]:
    """Check that ``in_field`` is differential, apart from its pinned rows.

    A boundary condition pins some rows of a differential field to algebraic
    rows. Those rows are exempt: the derivative-form lowering drops them, and
    their pinned residual stands.
    """
    if in_field not in offsets:
        raise ValueError(
            f"the derivative expression writing {out_field!r} references "
            f"{in_field!r}, which is not a declared field"
        )
    in_off, in_size = offsets[in_field]
    rows = structure.differential_vars[in_off : in_off + in_size]
    if pinned is not None:
        rows = rows | pinned[in_off : in_off + in_size]
    if not np.any(structure.differential_vars[in_off : in_off + in_size]) or not np.all(rows):
        raise ValueError(
            f"the equation writing {out_field!r} takes the time derivative "
            f"of algebraic field {in_field!r}"
        )
    return in_off, in_size


def _lower_implicit_forms(
    *,
    structure: DAEStructure,
    n_u: int,
    n_lam: int,
    forms: tuple,
    field_offsets: dict[str, tuple[int, int]],
    mass_values: Callable,
    rhs: Callable,
    rhs_sparsity,
    max_fy_reuse_steps: Optional[int],
    precision: Any = None,
    pinned_rows=None,
):
    """Lower nonlinear-in-slope rows to an exact fully implicit residual.

    Pinned rows (see ``_check_derivative_field``) keep their pinned residual,
    and their slopes are held at zero inside the derivative expressions.
    """
    from diff_soup.problem._spec.derivative_expr import slope_context

    n = int(n_u) + int(n_lam)
    pinned = _pinned_mask(pinned_rows, n)
    rows_list, cols_list = [], []
    form_rows: set[int] = set()
    for form in forms:
        out_field = form["out_field"]
        if out_field not in field_offsets:
            raise ValueError(
                f"the derivative-form equation writes {out_field!r}, which has no flat slot"
            )
        out_off, out_size = field_offsets[out_field]
        out_rows = np.arange(out_off, out_off + out_size, dtype=np.int64)
        out_rows = out_rows[~pinned[out_rows]]
        form_rows.update(out_rows.tolist())
        if form.get("nonlinear"):
            for in_field in form["ydot_fields"]:
                in_off, in_size = _check_derivative_field(
                    structure, field_offsets, out_field, in_field, pinned
                )
                in_cols = np.arange(in_off, in_off + in_size, dtype=np.int64)
                in_cols = in_cols[~pinned[in_cols]]
                rows_list.append(np.repeat(out_rows, in_cols.size))
                cols_list.append(np.tile(in_cols, out_rows.size))
        else:
            for in_field, term_rows, term_cols in form["terms"]:
                in_off, _ = _check_derivative_field(
                    structure, field_offsets, out_field, in_field, pinned
                )
                rows = np.asarray(term_rows, dtype=np.int64) + out_off
                cols = np.asarray(term_cols, dtype=np.int64) + in_off
                keep = ~(pinned[rows] | pinned[cols])
                rows_list.append(rows[keep])
                cols_list.append(cols[keep])

    base_idx = np.asarray(
        [i for i in structure.differential_indices if i not in form_rows], dtype=np.int64
    )
    dydot = SparseTopology(
        rows=np.concatenate([base_idx] + rows_list).astype(np.int32),
        cols=np.concatenate([base_idx] + cols_list).astype(np.int32),
        n_dofs=n,
    )
    form_mask = np.zeros((n,), dtype=bool)
    form_mask[sorted(form_rows)] = True
    field_slices = tuple((name, int(off), int(size)) for name, (off, size) in field_offsets.items())

    pinned_idx = np.flatnonzero(pinned)

    def residual(t, y, ydot, theta, control):
        free_ydot = ydot.at[pinned_idx].set(0.0) if pinned_idx.size else ydot
        slopes = {name: free_ydot[off : off + size] for name, off, size in field_slices}
        with slope_context(slopes):
            rhs_with_slopes = rhs(t, y, theta, control)
        plain_rhs = rhs(t, y, theta, control)
        base = mass_values(t, y, theta, control) * ydot - plain_rhs
        return jnp.where(jnp.asarray(form_mask), rhs_with_slopes, base)

    linearization = ResidualLinearization(
        sparsity=ResidualSparsity(dy=rhs_sparsity, dydot=dydot),
        max_fy_reuse_steps=max_fy_reuse_steps,
    )
    return ImplicitProblem(
        residual=residual,
        structure=structure,
        linearization=linearization,
        precision=precision,
    )


def _lower_linear_forms(
    *,
    structure: DAEStructure,
    n_u: int,
    n_lam: int,
    forms: tuple,
    field_offsets: dict[str, tuple[int, int]],
    mass_values: Callable,
    rhs: Callable,
    rhs_sparsity,
    max_fy_reuse_steps: Optional[int],
    precision: Any = None,
    pinned_rows=None,
):
    """Lower recorded linear derivative forms to one mass operator.

    Pinned rows (see ``_check_derivative_field``) keep their pinned residual,
    and their slopes are held at zero inside the derivative expressions.
    """
    from diff_soup.problem._spec.derivative_expr import slope_context

    n = int(n_u) + int(n_lam)
    pinned = _pinned_mask(pinned_rows, n)
    rows_list, cols_list = [], []
    state_rows_list, state_cols_list = [], []
    form_rows: set[int] = set()
    for form in forms:
        out_field, terms = form["out_field"], form["terms"]
        if out_field not in field_offsets:
            raise ValueError(
                f"the derivative-form equation writes {out_field!r}, which has no flat slot"
            )
        out_off, out_size = field_offsets[out_field]
        out_rows = np.arange(out_off, out_off + out_size, dtype=np.int64)
        out_rows = out_rows[~pinned[out_rows]]
        form_rows.update(out_rows.tolist())
        for in_field, term_rows, term_cols in terms:
            in_off, _ = _check_derivative_field(
                structure, field_offsets, out_field, in_field, pinned
            )
            rows = np.asarray(term_rows, dtype=np.int64) + out_off
            cols = np.asarray(term_cols, dtype=np.int64) + in_off
            keep = ~(pinned[rows] | pinned[cols])
            rows_list.append(rows[keep])
            cols_list.append(cols[keep])

        state_deps = form.get("state_deps", None)
        if state_deps == ():
            continue
        dep_cols = None
        if state_deps is not None:
            ranges = []
            for name in state_deps:
                if name not in field_offsets:
                    ranges = None
                    break
                dep_off, dep_size = field_offsets[name]
                ranges.append(np.arange(dep_off, dep_off + dep_size, dtype=np.int64))
            if ranges is not None:
                dep_cols = np.concatenate(ranges) if ranges else np.zeros(0, np.int64)
        if dep_cols is None:
            dep_cols = np.arange(n, dtype=np.int64)
        state_rows_list.append(np.repeat(out_rows, dep_cols.size))
        state_cols_list.append(np.tile(dep_cols, out_rows.size))

    base_idx = np.asarray(
        [i for i in structure.differential_indices if i not in form_rows], dtype=np.int32
    )
    form_mask = np.zeros((n,), dtype=bool)
    form_mask[sorted(form_rows)] = True
    coefficient_topology = SparseTopology(
        rows=np.concatenate([base_idx.astype(np.int64)] + rows_list).astype(np.int32),
        cols=np.concatenate([base_idx.astype(np.int64)] + cols_list).astype(np.int32),
        n_dofs=n,
    )
    field_slices = tuple((name, int(off), int(size)) for name, (off, size) in field_offsets.items())
    depends_on_state = bool(state_rows_list)
    state_action_sparsity = None
    if depends_on_state:
        state_action_sparsity = SparseTopology(
            rows=np.concatenate(state_rows_list).astype(np.int32),
            cols=np.concatenate(state_cols_list).astype(np.int32),
            n_dofs=n,
        )

    pinned_idx = np.flatnonzero(pinned)

    def apply(t, y, theta, control, ydot):
        base = jnp.zeros((n,), dtype=ydot.dtype)
        if base_idx.size:
            mass = mass_values(t, y, theta, control)
            base = base.at[base_idx].set(mass[base_idx] * ydot[base_idx])
        free_ydot = ydot.at[pinned_idx].set(0.0) if pinned_idx.size else ydot
        slopes = {name: free_ydot[off : off + size] for name, off, size in field_slices}
        with slope_context(slopes, include_residual=False):
            derivative_action = rhs(t, y, theta, control)
        return base + jnp.where(jnp.asarray(form_mask), derivative_action, 0.0)

    mass_operator = MassOperator(
        apply=apply,
        coefficient_sparsity=coefficient_topology,
        dependencies=Dependencies(
            time=True,
            state=depends_on_state,
            parameters=True,
            control=False,
        ),
        state_action_sparsity=state_action_sparsity,
    )
    lin = ResidualLinearization(
        sparsity=ResidualSparsity(dy=rhs_sparsity, dydot=coefficient_topology),
        max_fy_reuse_steps=max_fy_reuse_steps,
    )
    return MassProblem(
        rhs=rhs,
        mass=mass_operator,
        structure=structure,
        rhs_sparsity=rhs_sparsity,
        linearization=lin,
        **({} if precision is None else {"precision": precision}),
    )


def lower_mass_form(
    *,
    structure: DAEStructure,
    n_u: int,
    n_lam: int,
    dependencies: Dependencies,
    mass_values: Callable,
    rhs: Callable,
    rhs_sparsity,
    dmass_values: Optional[Callable] = None,
    forms: tuple = (),
    field_offsets: Optional[dict[str, tuple[int, int]]] = None,
    max_fy_reuse_steps: Optional[int] = None,
    precision: Any = None,
    pinned_rows=None,
):
    """Build the solver problem from explicit mass-form facts.

    ``precision`` is the problem's [Precision][diff_soup._precision.Precision],
    recorded on the returned solver problem (``None`` keeps the default).
    ``pinned_rows`` are the algebraic rows of differential fields (Dirichlet
    and symmetry rows), which the derivative forms leave pinned.
    """
    forms = tuple(forms or ())
    if forms:
        kwargs = dict(
            pinned_rows=pinned_rows,
            structure=structure,
            n_u=n_u,
            n_lam=n_lam,
            forms=forms,
            field_offsets={} if field_offsets is None else field_offsets,
            mass_values=mass_values,
            rhs=rhs,
            rhs_sparsity=rhs_sparsity,
            max_fy_reuse_steps=max_fy_reuse_steps,
            precision=precision,
        )
        if any(form.get("nonlinear") for form in forms):
            return _lower_implicit_forms(**kwargs)
        return _lower_linear_forms(**kwargs)

    n = int(n_u) + int(n_lam)
    diff_idx = np.asarray(structure.differential_indices, dtype=np.int32)
    diagonal = SparseTopology(rows=diff_idx, cols=diff_idx, n_dofs=n)

    def diagonal_values(t, y, theta, control):
        return mass_values(t, y, theta, control)[diff_idx]

    def apply(t, y, theta, control, ydot):
        return coo_matvec(diagonal_values(t, y, theta, control), diff_idx, diff_idx, n, ydot)

    state_action_sparsity = diagonal if dependencies.state else None
    state_action_values = None
    if dependencies.state:
        if dmass_values is None:
            raise ValueError("state-dependent mass lowering requires dmass_values")

        def state_action_values(t, y, theta, control, ydot):
            return (dmass_values(t, y, theta, control) * ydot)[diff_idx]

    mass_operator = MassOperator(
        apply=apply,
        coefficient_sparsity=diagonal,
        dependencies=dependencies,
        state_action_sparsity=state_action_sparsity,
        coefficient_values=diagonal_values,
        state_action_values=state_action_values,
        transpose_apply=apply,
    )
    lin = ResidualLinearization(
        sparsity=ResidualSparsity(dy=rhs_sparsity, dydot=diagonal),
        max_fy_reuse_steps=max_fy_reuse_steps,
    )
    return MassProblem(
        rhs=rhs,
        mass=mass_operator,
        structure=structure,
        rhs_sparsity=rhs_sparsity,
        linearization=lin,
        **({} if precision is None else {"precision": precision}),
    )


# Solver problems built from lowered Recipe IR, keyed on the problem's
# structure key. Problems equal in structure (value copies of one lowering)
# share their static partition, so they share one solver problem, whose
# callables take every value from ``theta``. Bounded like the lowering cache it
# follows.
_IR_SOLVER_PROBLEM_CACHE_CAPACITY = 128
_IR_SOLVER_PROBLEMS = BoundedContentCache(capacity=_IR_SOLVER_PROBLEM_CACHE_CAPACITY)


def default_theta_from_ir(problem):
    """Partition current inexact IR leaves for the explicit theta channel."""
    theta, _static = eqx.partition(problem, eqx.is_inexact_array)
    return _IRTheta(inner=theta)


def _static_equal(a, b) -> bool:
    """Whether two static partitions are equal leaf by leaf (``eqx.tree_equal``)."""
    return a is b or bool(eqx.tree_equal(a, b))


def solver_problem_from_ir(problem):
    """Return one solver problem built directly from Recipe IR.

    Shared by problems equal in structure: a problem whose structure key and
    static partition equal those of a problem already lowered here (the problem
    itself, or a value copy of the same lowering) gets that problem's solver
    problem, since its callables take every value from ``theta``.
    """
    getter = getattr(problem, "structure_key", None)
    structure_key = getter() if callable(getter) else None
    _theta, source_static = eqx.partition(problem, eqx.is_inexact_array)

    def build():
        return source_static, _build_solver_problem(problem, source_static)

    if structure_key is not None:
        shared_static, shared = _IR_SOLVER_PROBLEMS.get_or_build(
            ("structure", structure_key), None, build
        )
        if _static_equal(shared_static, source_static):
            return shared
    # No structure key, or a static partition other than the one this
    # structure key was built from (a static field rewritten in place): the
    # problem gets a solver problem of its own. The entry holds the problem, so
    # its id is not reused while the entry lives.
    return _IR_SOLVER_PROBLEMS.get_or_build(("identity", id(problem)), problem, build)[1]


def _build_solver_problem(problem, source_static):
    """Build the mass-form solver problem of one lowered Recipe problem."""
    with jax.ensure_compile_time_eval():
        n_u = int(problem.state_size())
        n_lam = int(problem.constraint_dim())
        n = n_u + n_lam
        mass_host = np.asarray(problem.mass_diagonal(), dtype=np.float64)
        # Per-global mass role: a differential global carries unit mass and a
        # differential-mask entry; algebraic globals keep zero mass. The layout
        # walk is the same global-row order the bordered vector uses.
        global_mass_host = np.zeros((n_lam,), dtype=np.float64)
        global_layout_fn = getattr(problem, "global_layout", None)
        if callable(global_layout_fn):
            for entry in global_layout_fn():
                if entry["mass"] == "differential":
                    off = int(entry["offset"])
                    global_mass_host[off : off + int(entry["size"])] = 1.0
        mask = np.concatenate([mass_host != 0.0, global_mass_host != 0.0])
        bc = problem.bc_dof_indices()
        bc_indices = () if bc is None else tuple(int(i) for i in np.asarray(bc))
        if bc_indices:
            mask[np.asarray(bc_indices, dtype=np.int32)] = False
        partitions = (("state", 0, n_u), ("global", n_u, n)) if n_lam else ()
        structure = DAEStructure(differential_vars=mask, partitions=partitions)
        topology = problem._topo
        rhs_sparsity = augmented_rhs_sparsity(topology, n_u, n_lam)
        forms = tuple(problem.derivative_forms() or ())
        field_offsets = {
            entry[0][1]: (int(entry[1]), int(entry[2]))
            for entry in tuple(problem._layout_entries or ())
        }
        # Rows a boundary condition pins inside a differential field: the
        # field has mass on its other rows, and these are algebraic.
        pinned_rows = []
        for off, size in field_offsets.values():
            rows = np.arange(off, off + size)
            if np.any(mass_host[rows] != 0.0):
                pinned_rows.extend(rows[~mask[rows]].tolist())
        mass_mode = problem.mass_mode()
        control_evaluator = getattr(problem, "_controls_fn", None)
        control_bindings = (
            None if control_evaluator is None else getattr(control_evaluator, "bindings", None)
        )
        has_bound_controls = bool(control_bindings is not None and control_bindings.bindings)
        dependencies = {
            "constant": Dependencies(time=False, state=False, parameters=False, control=False),
            "time": Dependencies(time=True, state=False, parameters=False, control=False),
            "state": Dependencies(time=True, state=True, parameters=False, control=False),
        }[mass_mode]

        def recombine(theta):
            dynamic = getattr(theta, "inner", theta)
            return eqx.combine(dynamic, source_static)

        _table_getter = getattr(problem, "operational_table", None)
        _has_operational = callable(_table_getter) and _table_getter() is not None

        def _control_payload(control, values_si):
            """The one payload every residual consumer reads this evaluation.

            An operational table picks its active row from the mode index the
            carried control state holds, so that state travels on — carrying
            the resolved values, when there are any. Every other problem passes
            the resolved values alone.
            """
            if not _has_operational:
                return values_si
            if control is None or values_si is None or isinstance(control, dict):
                return control
            return eqx.tree_at(
                lambda s: s.values, control, values_si, is_leaf=lambda leaf: leaf is None
            )

        def rhs(t, y, theta, control):
            concrete = recombine(theta)
            u, lam = y[:n_u], y[n_u:]
            # One resolution for every consumer: the dimensional values the model's
            # q.control / face-view reach reads and the closing rows are
            # written from are the same dict — one source of truth.  The
            # authored callables are read at dimensional time here and nowhere else.
            values_si = None
            if control_bindings is not None and not control_bindings.is_empty:
                values_si = control_bindings.bound_values(
                    concrete._controls, t, control, dtype=lam.dtype
                )
            f, g = concrete.physics_residual(u, lam, t, _control_payload(control, values_si))
            f = jnp.asarray(f)
            if bc_indices:
                bc_residual = f
                bc_mask = (
                    jnp.zeros((n_u,), dtype=jnp.bool_)
                    .at[jnp.asarray(bc_indices, dtype=jnp.int32)]
                    .set(True)
                )
                f = jnp.where(bc_mask, -jnp.asarray(bc_residual, dtype=f.dtype), f)
            g = jnp.asarray(g)
            if has_bound_controls:
                # The same resolved dimensional values, rescaled to the multiplier's own
                # internal units before they are written into the closing row.
                for binding in control_bindings.bindings:
                    lo = binding.lam_offset
                    hi = lo + binding.dim
                    value = binding.to_internal(values_si[binding.control_name])
                    # ``MassProblem.residual`` negates this RHS slice, yielding
                    # the closing row ``lambda - bound_value = 0``.
                    g = g.at[lo:hi].set(value.astype(lam.dtype) - lam[lo:hi])
            return jnp.concatenate([f, g])

        def mass_values(t, y, theta, control):
            del control
            concrete = recombine(theta)
            if mass_mode == "constant":
                state_mass = concrete.mass_diagonal()
            else:
                candidate = concrete.mass_diagonal_state(y[:n_u], y[n_u:], t)
                state_mass = concrete.mass_diagonal() if candidate is None else candidate
            # Structural mass data is commonly built from host geometry in
            # float64. The lowered residual must nevertheless follow the
            # explicitly supplied state dtype; otherwise ``float32`` states
            # silently promote the entire residual/Jacobian and defeat an
            # end-to-end reduced-precision policy.
            state_mass = jnp.asarray(state_mass, dtype=y.dtype)
            if bc_indices:
                state_mass = state_mass.at[jnp.asarray(bc_indices, dtype=jnp.int32)].set(0)
            return jnp.concatenate(
                [state_mass, jnp.asarray(global_mass_host, dtype=state_mass.dtype)]
            )

        dmass_values = None
        if dependencies.state:

            def dmass_values(t, y, theta, control):
                del t, control
                concrete = recombine(theta)
                values = concrete.dmass_du_diagonal(y[:n_u])
                if values is None:
                    raise ValueError("state-dependent Recipe mass returned no diagonal derivative")
                values = jnp.asarray(values, dtype=y.dtype)
                if bc_indices:
                    values = values.at[jnp.asarray(bc_indices, dtype=jnp.int32)].set(0)
                return jnp.concatenate([values, jnp.zeros((n_lam,), values.dtype)])

        lowered = lower_mass_form(
            structure=structure,
            n_u=n_u,
            n_lam=n_lam,
            dependencies=dependencies,
            mass_values=mass_values,
            rhs=rhs,
            rhs_sparsity=rhs_sparsity,
            dmass_values=dmass_values,
            forms=forms,
            field_offsets=field_offsets,
            max_fy_reuse_steps=getattr(problem, "max_fy_reuse_steps", None),
            precision=getattr(problem, "precision", None),
            pinned_rows=tuple(pinned_rows),
        )

    return lowered
