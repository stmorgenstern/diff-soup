"""The residual forms of a spec's equations: cell-weighted slopes and the derivative-return adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np


def _probe_coefficient_state_dependence(rhs: Callable, view, ctx, reference) -> Optional[tuple]:
    """Which fields a form's derivative coefficients read, by value probing.

    Re-evaluates the equation closure at a concrete synthetic state, then once
    more per field with only that field perturbed, and compares the normalized
    coefficient values. Runs host-side, once, at structure-capture time.

    Args:
        rhs: The equation's ``(view, ctx, t)`` adapter (the unwrapped one).
        view: The owner-keyed view of the capturing call — used only for its
            variable keys and array shapes/dtypes.
        ctx: The capturing call's lowering context (static geometry).
        reference: The
            [DerivativeIR][diff_soup.problem._ir.derivative_ir.DerivativeIR]
            captured from the real call —
            probe results must match its per-field patterns.

    Returns:
        A sorted tuple of field names whose perturbation moves any coefficient
        value; ``()`` for state-invariant coefficients; ``None`` when the
        probe cannot decide (the lowering then declares conservative
        dependence on the whole state).
    """
    import jax

    from diff_soup.problem._ir.derivative_ir import normalize_expression
    from diff_soup.problem._spec.derivative_expr import DerivativeExpr

    if not hasattr(view, "items"):
        return None
    try:
        # The capturing call may run inside an active trace (the build's
        # sparsity probes); the probe's inputs are concrete, so escape to
        # eager evaluation for the whole comparison.
        with jax.ensure_compile_time_eval():
            base_view = {}
            for var, arr in view.items():
                shape = tuple(getattr(arr, "shape", ()))
                dtype = getattr(arr, "dtype", None)
                if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
                    return None
                size = int(np.prod(shape)) if shape else 1
                vals = jnp.linspace(0.75, 1.25, size).astype(dtype).reshape(shape)
                base_view[var] = vals
            t_probe = jnp.asarray(0.37)

            def coefficient_values(probe_view):
                expr = rhs(probe_view, ctx, t_probe)
                if not isinstance(expr, DerivativeExpr):
                    return None
                ir = normalize_expression(expr, out_size=int(expr.size))
                if not hasattr(ir, "terms"):
                    return None
                return {
                    term.in_field: (term.pattern, np.asarray(term.values())) for term in ir.terms
                }

            base = coefficient_values(base_view)
            if base is None or set(base) != {t.in_field for t in reference.terms}:
                return None
            deps = []
            for var in base_view:
                probe = dict(base_view)
                probe[var] = base_view[var] * 1.003 + 0.0017
                got = coefficient_values(probe)
                if got is None or set(got) != set(base):
                    return None
                moved = False
                for field, (pattern, values) in got.items():
                    ref_pattern, ref_values = base[field]
                    if not (
                        np.array_equal(pattern[0], ref_pattern[0])
                        and np.array_equal(pattern[1], ref_pattern[1])
                    ):
                        return None
                    if values.shape != ref_values.shape or not np.array_equal(values, ref_values):
                        moved = True
                if moved:
                    deps.append(str(getattr(var, "name", var)))
        return tuple(sorted(deps))
    except Exception:
        return None


class _FormCell:
    """Identity-hashed holder for one equation's captured derivative structure.

    Lives on the static side of the adapter module, so it must be hashable;
    the default object identity hash is correct here — the cell is created
    per adapter and filled in place by the build-time capture pass.
    """

    __slots__ = ("data",)

    def __init__(self):
        self.data: dict = {}


def _weight_by(action, weight):
    """Multiply a derivative action by one weight per node, shared by a vector field's components."""
    action = jnp.asarray(action)
    weight = jnp.asarray(weight)
    if action.size != weight.size:
        return (action.reshape(weight.size, -1) * weight[:, None]).reshape(action.shape)
    return action * weight.reshape(action.shape)


def _lumped_weight(ops):
    """The lumped mass of a mesh bank, one weight per node."""
    from diff_soup.domains.operators.dispatch import _WeakTerm

    domain = getattr(ops, "domain", None)
    if domain is not None and hasattr(domain, "lumped_mass"):
        weight = domain.lumped_mass()
    else:
        weight = ops.lumped_mass()
    if isinstance(weight, _WeakTerm):
        weight = weight.value
    return jnp.asarray(weight)


def _lumped_derivative_action(ctx, action):
    """Weight a derivative action by the lumped mass on a mesh.

    On a mesh the rows of a residual are integrated against the test
    functions, and the explicit form weights ``d(field)/dt`` by the lumped
    mass (the problem's mass diagonal). The derivative action gets the same
    weight and is marked as already integrated. On a grid it is unchanged.
    """
    ops = getattr(ctx, "ops", None)
    if ops is None or getattr(ops, "discretization_type", None) != "fem":
        return action
    return ops.as_weak(_weight_by(action, _lumped_weight(ops)))


def _has_cell_scale(expr) -> bool:
    """Whether a per-cell coefficient scales any derivative in ``expr``."""
    from diff_soup.problem._spec.derivative_expr import (
        DerivativeScale,
        DerivativeSum,
        NonlinearDerivativeExpr,
    )

    if isinstance(expr, DerivativeScale) and expr.per_cell:
        return True
    if isinstance(expr, DerivativeSum):
        return any(_has_cell_scale(term) for term in expr.terms)
    if isinstance(expr, NonlinearDerivativeExpr):
        return any(_has_cell_scale(operand) for operand in expr.operands)
    inner = getattr(expr, "expr", None)
    return inner is not None and _has_cell_scale(inner)


def _split_cell_scale(term):
    """Split ``term`` into the expression without its per-cell scales and their product.

    A per-cell coefficient must scale a field's derivative directly, possibly
    through other scales; anywhere else it is refused.
    """
    from diff_soup.problem._spec.derivative_expr import DerivativeScale

    if isinstance(term, DerivativeScale):
        inner, cell = _split_cell_scale(term.expr)
        if term.per_cell:
            return inner, term.coeff if cell is None else cell * jnp.asarray(term.coeff)
        return (term if inner is term.expr else DerivativeScale(inner, term.coeff)), cell
    if _has_cell_scale(term):
        raise ValueError(
            "a per-cell coefficient must scale q.dt(field) directly, as in "
            "rho_c * q.dt(field), not a gathered, reduced, scattered, applied or "
            "nonlinear derivative expression"
        )
    return term, None


def _cell_terms(expr, ctx):
    """The terms of ``expr`` as ``(term without per-cell scales, coefficient per cell)``.

    Each per-cell coefficient is read against the mesh of the evaluating bank,
    which refuses another length than the number of cells; on a grid any
    length other than the field's is refused.
    """
    from diff_soup.domains.operators.dispatch import _read_fem_operand
    from diff_soup.problem._spec.derivative_expr import DerivativeSum, scale_size_error

    terms = expr.terms if isinstance(expr, DerivativeSum) else (expr,)
    ops = getattr(ctx, "ops", None)
    on_mesh = ops is not None and getattr(ops, "discretization_type", None) == "fem"
    out = []
    for term in terms:
        inner, cell = _split_cell_scale(term)
        if cell is not None:
            if not on_mesh:
                raise scale_size_error(int(np.shape(cell)[0]), int(inner.size))
            length = int(np.shape(cell)[0])
            cell, layout = _read_fem_operand(ops.domain, cell, "a per-cell coefficient of q.dt")
            if layout != "quad":
                raise scale_size_error(length, int(inner.size))
        out.append((inner, cell))
    return out


def _check_cell_scales(expr, ctx) -> None:
    """Refuse a per-cell coefficient of ``expr`` that its domain cannot read."""
    if _has_cell_scale(expr):
        _cell_terms(expr, ctx)


def _cell_weighted_parts(expr, slopes: dict, ctx, include: bool):
    """``(derivative action, residual part)`` of an expression with per-cell coefficients.

    A term scaled per cell by ``c`` is weighted by ``int c N_i``, the lumped
    mass with the coefficient inside the integral; the other terms take the
    lumped mass.
    """
    from diff_soup.domains.operators.dispatch import dispatch_integrate_scalar_at_quad
    from diff_soup.problem._ir.derivative_ir import evaluate_parts, sum_residual_terms
    from diff_soup.problem._spec.derivative_expr import DerivativeSum

    ops = ctx.ops
    size = int(expr.size)
    action = None
    for inner, cell in _cell_terms(expr, ctx):
        value, _ = evaluate_parts(inner, slopes, size, include_residual=False)
        if cell is None:
            weight = _lumped_weight(ops)
        else:
            weight = dispatch_integrate_scalar_at_quad(ops.domain, cell)
        weighted = _weight_by(value, weight)
        action = weighted if action is None else action + weighted
    residual = None
    if include and isinstance(expr, DerivativeSum):
        residual = sum_residual_terms(expr.residual_terms, size)
    return ops.as_weak(action), residual


def _slope_evaluation(expr, slopes: dict, ctx):
    """Evaluate a derivative expression inside a slope context.

    Returns the derivative action alone, or the full residual rows when the
    context includes the residual part. On a mesh the derivative action is
    weighted by the lumped mass (see ``_lumped_derivative_action``), and a term
    with a per-cell coefficient by the mass integral of that coefficient.
    """
    from diff_soup.problem._ir.derivative_ir import evaluate_parts
    from diff_soup.problem._spec.derivative_expr import slope_context_includes_residual

    include = slope_context_includes_residual()
    if _has_cell_scale(expr):
        action, residual = _cell_weighted_parts(expr, slopes, ctx, include)
    else:
        action, residual = evaluate_parts(expr, slopes, int(expr.size), include_residual=include)
        action = _lumped_derivative_action(ctx, action)
    if residual is None:
        return action
    return action + residual


class _ResidualFormAdapter(eqx.Module):
    """Adapt a residual-form derivative-expression return onto the rhs seam.

    A closure returning a derivative expression states the residual
    ``expression == 0``. Outside a slope context the adapter emits the
    NEGATED derivative-free part, so the ordinary rhs channel keeps the
    reading ``residual = derivative_part - rhs``. Inside a slope context
    (installed by the solver-problem lowering) the adapter evaluates the
    expression against the provided slopes — the full residual rows by
    default, or the derivative action alone when the context excludes the
    residual part.

    During the build-time capture pass (and only then, see
    [capture_forms][diff_soup.problem._ir.derivative_ir.capture_forms]) the
    adapter records the expression's static structure
    (``_derivative_form``): the written field, each contributing field's
    exact coefficient pattern, and which fields the coefficients read
    (``state_deps``, probed by value; ``None`` when undecidable).
    Expressions that are nonlinear in the state derivative are recorded as
    such; they lower through the general implicit form instead.

    An ``eqx.Module`` (not a plain closure) for the same reason as
    [_TracedQAdapter][diff_soup.problem._spec.q_adapter._TracedQAdapter]: the
    wrapped rhs sits in a non-static field, so
    a learnable-parameter module inside it keeps its array leaves visible to
    ``eqx.partition`` and gradient transformations.
    """

    inner: object
    equation_name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    cell: _FormCell = eqx.field(static=True)

    @property
    def _raw_rhs(self):
        """The wrapped rhs adapter — the content-identity of this equation."""
        return self.inner

    @property
    def _derivative_form(self) -> dict:
        """The captured structure cell (empty until the build-time capture pass)."""
        return self.cell.data

    def __call__(self, view, ctx, t):
        from diff_soup.problem._ir.derivative_ir import (
            ImplicitResidualIR,
            evaluate_expression,
            normalize_expression,
            record_form,
            record_plain,
            zero_slopes_for,
        )
        from diff_soup.problem._spec.derivative_expr import DerivativeExpr, current_slopes

        result = self.inner(view, ctx, t)
        form_cell = self.cell.data
        if not isinstance(result, DerivativeExpr):
            record_plain(form_cell)
            return result
        slopes = current_slopes()
        if slopes is not None:
            return _slope_evaluation(result, slopes, ctx)
        _check_cell_scales(result, ctx)
        ir = normalize_expression(result, out_size=int(result.size))
        record_form(
            form_cell,
            self.field_name,
            ir,
            state_deps=lambda: _probe_coefficient_state_dependence(self.inner, view, ctx, ir),
        )
        if isinstance(ir, ImplicitResidualIR):
            # No slope context: contribute the residual at zero slopes (the
            # derivative-free evaluation the rhs channel expects).
            return evaluate_expression(result, zero_slopes_for(result), int(result.size))
        return -ir.residual_part()


def _wrap_derivative_return(rhs: Callable, *, equation_name: str, field_name: str) -> Callable:
    """Wrap an equation rhs in the residual-form adapter (see the class)."""
    return _ResidualFormAdapter(
        inner=rhs,
        equation_name=equation_name,
        field_name=field_name,
        cell=_FormCell(),
    )


class _BundledReturnGuard(eqx.Module):
    """Reject derivative expressions inside a bundled multi-output return.

    An ``eqx.Module`` so the wrapped multi-output rhs keeps any learnable
    module leaves visible to ``eqx.partition`` (same rationale as
    [_ResidualFormAdapter][diff_soup.problem._spec.residual_forms._ResidualFormAdapter]).
    """

    inner: object
    equation_name: str = eqx.field(static=True)

    @property
    def _raw_rhs(self):
        """The wrapped rhs adapter — the content-identity of this equation."""
        return self.inner

    def __call__(self, view, ctx, t):
        from diff_soup.problem._spec.derivative_expr import DerivativeExpr

        result = self.inner(view, ctx, t)
        if isinstance(result, dict) and any(isinstance(v, DerivativeExpr) for v in result.values()):
            raise NotImplementedError(
                "derivative expressions are not supported inside a bundled "
                f"multi-output equation ({self.equation_name!r}); author "
                "residual-form rows as single-output equations."
            )
        return result


def _reject_derivative_returns_in_bundle(rhs: Callable, equation_name: str) -> Callable:
    """Wrap a bundled multi-output rhs in the derivative-return guard."""
    return _BundledReturnGuard(inner=rhs, equation_name=equation_name)
