"""The Jacobian blocks of a full residual: its colorings, its fully declared field equations and its constraint borders."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.linear import SparseTopology
from diff_soup.problem._ir.constraint import (
    OperationalControlConstraint,
    ReduceResidualConstraint,
)
from diff_soup.problem._ir.equation import MultiEquation
from diff_soup.problem._ir.variable import GlobalVariable, Variable
from diff_soup.problem._lowering.full_residual import _build_full_residual_fn
from diff_soup.problem._lowering.provider_context import _Ctx
from diff_soup.problem._lowering.residual_fn import _lazy_coloring
from diff_soup.problem._spec.operating_modes import constant_current_slope

from .layout import _layout
from .operator_resolve import resolve_operator


def _is_plain_sum(reduce_fn) -> bool:
    """Return True iff ``reduce_fn`` is a plain element sum.

    Accepts ``jnp.sum`` / ``numpy.sum`` by identity and any wrapping callable
    named ``"sum"`` (e.g. ``lambda x: jnp.sum(x)``).  This ensures the same
    reduction always selects the same C-block formula regardless of how it was
    authored.
    """
    if reduce_fn is jnp.sum:
        return True
    if reduce_fn is np.sum:
        return True
    return getattr(reduce_fn, "__name__", "") == "sum"


class _AccessTrackingView:
    """View proxy that records which keys a residual closure reads.

    Used at lower-time to discover the true column support of a field
    equation's residual so the ``ReduceResidualConstraint`` C-block can decide
    whether the declared-topology scatter-sum shortcut is safe (every read is
    declared in ``inputs``) or whether it must fall back to dense jacrev.
    """

    def __init__(self, backing: dict):
        self._backing = backing
        self.accessed: set = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return self._backing[key]

    def __contains__(self, key):
        return key in self._backing


def _threaded_globals(eqset) -> tuple:
    """The bordered globals the residual places in every component's view.

    Returns ``(global, dim)`` pairs in global-row order: the residual reads a
    global closed by a row of one entry as a scalar and a wider one as a
    ``(dim,)`` slice of the bordered unknowns.
    """
    return tuple(
        (c.closes, int(c.dim))
        for c in eqset.constraints
        if isinstance(getattr(c, "closes", None), GlobalVariable)
    )


def _field_eq_is_fully_declared(comp, field_var, ops, threaded_globals=()) -> bool:
    """Return True iff every Variable read by ``field_var``'s equation is declared.

    Traces the residual closure that writes ``field_var`` against an
    access-tracking view, with ``ops`` the operator bank the residual hands the
    component's equations, and confirms every local ``Variable`` it reads
    appears in the equation's ``inputs``.  When the equation under-declares a
    column the declared-topology ``Avals`` would miss it, so the C-block
    scatter-sum shortcut is unsafe and the caller must use dense jacrev.

    The view holds the component's variables and every bordered global in
    ``threaded_globals`` (``(global, dim)`` pairs, as the residual threads
    them), so an equation that reads a global traces as it runs.  A global's
    column is ``B = df/dlam``, which the kernel differentiates in full, so a
    read of a global never makes the shortcut unsafe.

    A read the view does not hold (a ``KeyError`` from the view, or an
    ``AttributeError`` from the ``q`` accessor) returns ``False`` so the safe
    dense-jacrev path is taken; any other error the closure raises propagates.
    """
    target_eq = None
    for eq in comp.equations:
        out_vars = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
        if any(v is field_var for v in out_vars):
            target_eq = eq
            break
    if target_eq is None:
        return False
    declared = {inp for inp in target_eq.inputs if isinstance(inp, Variable)}
    try:
        backing = {v: jnp.zeros(v.shape, dtype=jnp.float64) for v in comp.variables}
        # Globals are read through the view but never enter the sparse df/du
        # topology: a global's value has the shape the residual gives it.
        for g, dim in threaded_globals:
            backing.setdefault(g, jnp.zeros(() if dim == 1 else (dim,), dtype=jnp.float64))
        for inp in target_eq.inputs:
            if isinstance(inp, GlobalVariable) and inp not in backing:
                backing[inp] = jnp.zeros((), dtype=jnp.float64)
        view = _AccessTrackingView(backing)
        ctx = _Ctx(ops)
        target_eq.rhs_fn(view, ctx, jnp.zeros(()))
    except (KeyError, AttributeError):
        return False
    read_vars = {k for k in view.accessed if isinstance(k, Variable)}
    return read_vars.issubset(declared)


class _ConstraintBorderProbe:
    """Trace-time work counters for the bordered-global derivative kernel.

    The compatibility type name predates differential globals. Each count is
    incremented when the border kernel (the row-restricted
    residual used to build ``C = dg/du`` / ``D = dg/dlam``) invokes an
    equation closure, a Family block, or a bordered residual DURING TRACING.
    Under an outer ``jax.jit`` the counts therefore reflect compilations, not
    executions — which is the meaningful measure: physics absent from the
    trace is absent from the compiled derivative program.
    """

    __slots__ = ("equation_evals", "family_evals", "constraint_evals")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Zero all counters (start a fresh measurement window)."""
        self.equation_evals: dict = {}
        self.family_evals: dict = {}
        self.constraint_evals: dict = {}

    def count_equation(self, name: str) -> None:
        self.equation_evals[name] = self.equation_evals.get(name, 0) + 1

    def count_family(self, name: str) -> None:
        self.family_evals[name] = self.family_evals.get(name, 0) + 1

    def count_constraint(self, name: str) -> None:
        self.constraint_evals[name] = self.constraint_evals.get(name, 0) + 1


class _ConstraintBorderPlan(NamedTuple):
    """Static evaluation plan of the bordered-global derivative kernel.

    Attributes:
        required_rows: ``(owner, field)`` residual slices the border kernel
            assembles (the residual-reducing fallbacks' declared fields).
        required_by: per-fallback-global-row ``(name, rows)`` pairs; an
            observable-only fallback (e.g. galvanostatic) requires no slice.
        fallback_constraints: compatibility-named global rows whose ``C``/``D`` come from
            dense differentiation of the border kernel.
        shortcut_constraints: algebraic constraints whose rows come from the
            scatter-sum-over-``Avals`` / slice-of-``B`` shortcut.  An
            operational table whose current is a field's residual sum is in
            both tuples: its ``current`` row comes from the shortcut, its
            other rows from the border kernel, which holds that current fixed.
        skipped_equations: equations the border kernel never evaluates.
        skipped_families: families the border kernel never evaluates.
        restricted: ``False`` when the border falls back to the full residual
            kernel (a fallback global row's residual slice could not be
            resolved to an owning component field).
    """

    required_rows: tuple
    required_by: tuple
    fallback_constraints: tuple
    shortcut_constraints: tuple
    skipped_equations: tuple
    skipped_families: tuple
    restricted: bool


def _build_full_jac_blocks_fn(
    eqset,
    residual_fn,
    topo: SparseTopology,
    total: int,
    *,
    compute_dtype=None,
    bcs=None,
):
    """Build the family/coupling KKT-Jacobian kernel ``(Avals, B, C, D, topo)``.

    ``Avals`` = sparse ``df/du`` via CPR-coloured ``SparseJacobian.evaluate``;
    ``B = df/dlam`` (``jacfwd`` over the small global axis). ``C = dg/du`` /
    ``D = dg/dlam`` are built per bordered-global row: a ``ReduceResidualConstraint`` with
    a plain-sum reduction over a fully-declared field uses a
    scatter-sum-from-``Avals`` + slice-and-sum-from-``B`` shortcut; every other
    row (including a ``DifferentialGlobalEquation``, ``GalvanostaticConstraint``,
    or any under-declared/non-sum reducer) takes the dense ``jacrev`` /
    ``jacfwd`` fallback.  An ``OperationalControlConstraint`` whose current is
    the residual sum of a fully-declared field, in a table without a ``power``
    mode, combines the two: the border kernel differentiates its row with that
    current held fixed (the ``voltage`` row's dependence), and the shortcut row
    is added when the operating mode is ``current``.

    The fallback differentiates the bordered-global kernel — the residual
    restricted to the fallback rows' declared residual slices and
    observables (built when ``compute_dtype`` and ``bcs`` are provided) — so
    ``C``/``D`` construction never differentiates unrelated component or
    Family physics while remaining numerically identical to differentiating
    the full residual's ``g``.  ``B = df/dlam`` stays a full physics
    derivative.  The returned kernel publishes its border introspection as
    ``constraint_border_plan``
    (a [_ConstraintBorderPlan][diff_soup.problem._lowering.full_jacobian._ConstraintBorderPlan])
    and ``constraint_border_probe``
    (a [_ConstraintBorderProbe][diff_soup.problem._lowering.full_jacobian._ConstraintBorderProbe]).
    """
    sparse_jac = _lazy_coloring(topo)
    n_c = int(sum(int(c.dim) for c in eqset.constraints))
    n_s = int(total)

    # ------------------------------------------------------------------
    # Host-time per-global-row Schur shortcut analysis. Each entry:
    #   - ("sum", row_lo, row_hi, lam_off, dim) for the scatter-sum path
    #   - ("fallback", lam_off, dim)            for the dense jacrev/jacfwd path
    #   - ("operational_sum", row_lo, row_hi, lam_off, dim, current_index, name) for
    #     an operational table whose current is a residual sum: the dense path
    #     with that current held fixed, plus the scatter-sum row whenever the
    #     operating mode index equals ``current_index`` (``None``: no such mode)
    # The constrained field is resolved to its owning component by Python ``is``
    # (host-side) so a structurally-equal Variable cannot misroute the slice.
    # ------------------------------------------------------------------
    layout = _layout(eqset)
    threaded_globals = _threaded_globals(eqset)

    def _residual_bank(comp):
        # The bank the residual hands the owner's equations as ``q.ops``: on
        # FEM the weak-form view, whose operators return the shapes the
        # equation's terms combine in.
        return resolve_operator(comp.domain, comp.operator_factory).for_residual()

    shortcut_meta: list = []
    shortcut_names: list = []
    fallback_names: list = []
    held_names: list = []
    required_by: list = []
    border_resolvable = True
    _lam_off = 0
    for c in eqset.constraints:
        dim_c = int(c.dim)
        if isinstance(c, ReduceResidualConstraint):
            field_var = c.field
            owner_comp = None
            owner_name = None
            for comp in eqset.components:
                if any(v is field_var for v in comp.variables):
                    owner_comp = comp
                    owner_name = comp.name
                    break
            if _is_plain_sum(c.reduce_fn):
                if owner_name is None:
                    raise ValueError(
                        f"constraint {c.name!r}: field {field_var.name!r} not found "
                        f"by identity in any component's variables"
                    )
                row_lo = layout.offset_of((owner_name, field_var.name))
                row_hi = row_lo + int(field_var.size)
                if _field_eq_is_fully_declared(
                    owner_comp, field_var, _residual_bank(owner_comp), threaded_globals
                ):
                    shortcut_meta.append(("sum", row_lo, row_hi, _lam_off, dim_c))
                    shortcut_names.append(c.name)
                else:
                    # Under-declared field equation: declared-topology Avals
                    # would drop the undeclared column.  Dense fallback over
                    # the field's residual slice keeps C == dg/du exact.
                    shortcut_meta.append(("fallback", _lam_off, dim_c))
                    fallback_names.append(c.name)
                    required_by.append((c.name, ((owner_name, field_var.name),)))
            else:
                shortcut_meta.append(("fallback", _lam_off, dim_c))
                fallback_names.append(c.name)
                if owner_name is None:
                    # The declared slice cannot be pinned to a component field;
                    # the border must evaluate the full residual to stay exact.
                    border_resolvable = False
                    required_by.append((c.name, ()))
                else:
                    required_by.append((c.name, ((owner_name, field_var.name),)))
        elif isinstance(c, OperationalControlConstraint) and c.field is not None:
            field_var = c.field
            owner_comp = next(
                (comp for comp in eqset.components if any(v is field_var for v in comp.variables)),
                None,
            )
            if owner_comp is None:
                raise ValueError(
                    f"operational control {c.name!r}: field {field_var.name!r} not found "
                    f"by identity in any component's variables"
                )
            row_lo = layout.offset_of((owner_comp.name, field_var.name))
            row_hi = row_lo + int(field_var.size)
            fallback_names.append(c.name)
            if constant_current_slope(c.modes) and _field_eq_is_fully_declared(
                owner_comp, field_var, _residual_bank(owner_comp), threaded_globals
            ):
                # The current enters its row linearly, so the row's derivative
                # is the border kernel's (current held fixed) plus, in the
                # ``current`` mode, the field's Jacobian rows summed.
                current_index = c.modes.index("current") if "current" in c.modes else None
                shortcut_meta.append(
                    ("operational_sum", row_lo, row_hi, _lam_off, dim_c, current_index, c.name)
                )
                shortcut_names.append(c.name)
                held_names.append(c.name)
                required_by.append((c.name, ()))
            else:
                # Power/resistance rows scale the current by voltage/target, and an
                # under-declared field equation would drop a column from
                # ``Avals``: differentiate the field's residual rows directly.
                shortcut_meta.append(("fallback", _lam_off, dim_c))
                required_by.append((c.name, ((owner_comp.name, field_var.name),)))
        else:
            shortcut_meta.append(("fallback", _lam_off, dim_c))
            fallback_names.append(c.name)
            required_by.append((c.name, ()))
        _lam_off += dim_c
    any_fallback = any(m[0] in ("fallback", "operational_sum") for m in shortcut_meta)

    # ------------------------------------------------------------------
    # Bordered-global kernel: the fallback C/D derivatives differentiate a
    # residual restricted to the fallback rows' declared slices, so
    # unrelated physics rows never enter the derivative trace.
    # ------------------------------------------------------------------
    required_rows = tuple(sorted({k for _name, keys in required_by for k in keys}))
    border_probe = _ConstraintBorderProbe()
    border_restricted = (
        any_fallback and border_resolvable and compute_dtype is not None and bcs is not None
    )
    if held_names and not border_restricted:
        # The border kernel is then the full residual, which holds no current
        # fixed: differentiating it gives these rows whole.
        shortcut_meta = [
            ("fallback", m[3], m[4]) if m[0] == "operational_sum" else m for m in shortcut_meta
        ]
        shortcut_names = [name for name in shortcut_names if name not in held_names]
        held_names = []
    if border_restricted:
        border_residual_fn = _build_full_residual_fn(
            eqset,
            layout,
            compute_dtype,
            bcs,
            restrict_rows_to=frozenset(required_rows),
            restrict_constraints_to=frozenset(fallback_names),
            hold_residual_currents=frozenset(held_names),
            border_probe=border_probe,
        )
    else:
        border_residual_fn = residual_fn
    if border_restricted:
        required_set = frozenset(required_rows)
        skipped_equations = []
        for comp in eqset.components:
            for eq in comp.equations:
                outs = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
                if not any((comp.name, v.name) in required_set for v in outs):
                    skipped_equations.append(getattr(eq, "name", None) or eq.var.name)
        skipped_families = [
            fam.name
            for fam in eqset.families
            if not any((fam.name, v.name) in required_set for v in fam.member_variables)
        ]
    else:
        skipped_equations = []
        skipped_families = []
    border_plan = _ConstraintBorderPlan(
        required_rows=required_rows,
        required_by=tuple(required_by),
        fallback_constraints=tuple(fallback_names),
        shortcut_constraints=tuple(shortcut_names),
        skipped_equations=tuple(skipped_equations),
        skipped_families=tuple(skipped_families),
        restricted=border_restricted,
    )
    # Device-side topology arrays so the scatter-sum path closes over them
    # without re-uploading host arrays per call (XLA folds them at trace time).
    topo_rows_dev = jnp.asarray(topo.rows, dtype=jnp.int32)
    topo_cols_dev = jnp.asarray(topo.cols, dtype=jnp.int32)

    def physics_jac_blocks_fn(
        equations,
        dirichlet_value_fns,
        flux_callables,
        robin_callables,
        traction_callables,
        constraints,
        couplings,
        u,
        lam,
        t,
        families=(),
        controls=(),
        control=None,
    ):
        def f_of_u(u_):
            f, _g = residual_fn(
                equations,
                dirichlet_value_fns,
                flux_callables,
                robin_callables,
                traction_callables,
                constraints,
                couplings,
                u_,
                lam,
                t,
                families,
                controls,
                control,
            )
            return f

        Avals = sparse_jac().evaluate(f_of_u, u)
        if n_c == 0:
            return (
                Avals,
                jnp.zeros((n_s, 0), dtype=u.dtype),
                jnp.zeros((0, n_s), dtype=u.dtype),
                jnp.zeros((0, 0), dtype=u.dtype),
                topo,
            )

        def f_of_lam(lam_):
            f, _g = residual_fn(
                equations,
                dirichlet_value_fns,
                flux_callables,
                robin_callables,
                traction_callables,
                constraints,
                couplings,
                u,
                lam_,
                t,
                families,
                controls,
                control,
            )
            return f

        # B = df/dlam via forward-mode (lam is constraint-dim, small); jacfwd
        # avoids the n_state-dim backward tape jacrev would build.
        B = jax.jacfwd(f_of_lam)(lam)

        # C = dg/du, D = dg/dlam.  Dense jacrev/jacfwd only when at least one
        # constraint takes the fallback; the plain-sum-over-fully-declared rows
        # take the scatter-sum-from-Avals + slice-and-sum-from-B shortcut.
        # The fallback differentiates the bordered-global kernel (the
        # residual restricted to the fallback rows' declared slices),
        # never the unrelated physics rows.
        full_C = None
        full_D = None
        if any_fallback:

            def g_of_u(u_):
                _f, g = border_residual_fn(
                    equations,
                    dirichlet_value_fns,
                    flux_callables,
                    robin_callables,
                    traction_callables,
                    constraints,
                    couplings,
                    u_,
                    lam,
                    t,
                    families,
                    controls,
                    control,
                )
                return g

            def g_of_lam(lam_):
                _f, g = border_residual_fn(
                    equations,
                    dirichlet_value_fns,
                    flux_callables,
                    robin_callables,
                    traction_callables,
                    constraints,
                    couplings,
                    u,
                    lam_,
                    t,
                    families,
                    controls,
                    control,
                )
                return g

            full_C = jax.jacrev(g_of_u)(u)
            full_D = jax.jacfwd(g_of_lam)(lam)

        def residual_sum_rows(row_lo, row_hi, dim_c):
            """The ``C`` and ``D`` rows of the sum of the residual rows ``[row_lo, row_hi)``."""
            # C row: scatter-sum Avals over the field's residual rows into
            # the u-columns: dg/du_j = sum_{i in [row_lo,row_hi)} A[i, j].
            row_mask = (topo_rows_dev >= row_lo) & (topo_rows_dev < row_hi)
            contrib = jnp.where(row_mask, Avals, jnp.zeros_like(Avals))
            C_row = jnp.zeros((dim_c, n_s), dtype=u.dtype)
            C_row = C_row.at[0, topo_cols_dev].add(contrib)
            # D row: sum B's field-row slice along axis 0.
            D_row = jnp.sum(B[row_lo:row_hi, :], axis=0, keepdims=True)
            return C_row, D_row

        c_rows: list = []
        d_rows: list = []
        for meta in shortcut_meta:
            if meta[0] == "sum":
                _, row_lo, row_hi, _lam_off_local, dim_c = meta
                C_row, D_row = residual_sum_rows(row_lo, row_hi, dim_c)
                c_rows.append(C_row)
                d_rows.append(D_row)
            elif meta[0] == "operational_sum":
                _, row_lo, row_hi, _lam_off_local, dim_c, current_index, table_name = meta
                C_row = full_C[_lam_off_local : _lam_off_local + dim_c]
                D_row = full_D[_lam_off_local : _lam_off_local + dim_c]
                if current_index is not None:
                    mode = getattr(control, "mode", None)
                    if mode is None:
                        raise ConfigurationError(
                            f"this problem declares an operational control table "
                            f"({table_name!r}); evaluate it with an operating control "
                            "state: ControlState.operating(mode=..., target=...)."
                        )
                    C_sum, D_sum = residual_sum_rows(row_lo, row_hi, dim_c)
                    on = jnp.asarray(mode, dtype=jnp.int32) == current_index
                    C_row = jnp.where(on, C_row + C_sum, C_row)
                    D_row = jnp.where(on, D_row + D_sum, D_row)
                c_rows.append(C_row)
                d_rows.append(D_row)
            else:  # fallback
                _, _lam_off_local, dim_c = meta
                c_rows.append(full_C[_lam_off_local : _lam_off_local + dim_c])
                d_rows.append(full_D[_lam_off_local : _lam_off_local + dim_c])
        C = jnp.concatenate(c_rows, axis=0)
        D = jnp.concatenate(d_rows, axis=0)
        return Avals, B, C, D, topo

    # Border introspection: the static evaluation plan and the trace-time work
    # counters of the bordered-global derivative kernel.
    physics_jac_blocks_fn.constraint_border_plan = border_plan
    physics_jac_blocks_fn.constraint_border_probe = border_probe
    return physics_jac_blocks_fn
