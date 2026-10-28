"""Normalized derivative IR: structured linear terms or a general implicit residual.

[normalize_expression][diff_soup.problem._ir.derivative_ir.normalize_expression]
walks a derivative-expression tree (the value an
equation closure returns) and produces either

* [DerivativeIR][diff_soup.problem._ir.derivative_ir.DerivativeIR] — the
  expression is linear in the state derivative: one
  [DerivativeTermIR][diff_soup.problem._ir.derivative_ir.DerivativeTermIR]
  per contributing field, each carrying the
  exact COO pattern of its derivative coefficient block and the matching
  values, plus the derivative-free residual part; or
* [ImplicitResidualIR][diff_soup.problem._ir.derivative_ir.ImplicitResidualIR]
  — the expression is nonlinear in the state
  derivative: an evaluator that substitutes concrete slopes into the tree,
  with the referenced fields recorded for conservative sparsity.

Patterns are static host arrays; values may be traced. Normalization runs at
build/trace time only — nothing here executes inside a compiled step.
"""

from __future__ import annotations

import contextlib
import threading
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.problem._spec.derivative_expr import (
    DerivativeApply,
    DerivativeExpr,
    DerivativeGather,
    DerivativeReduce,
    DerivativeScale,
    DerivativeScatter,
    DerivativeSum,
    NonlinearDerivativeExpr,
    TimeDerivativeRef,
)

__all__ = [
    "DerivativeTermIR",
    "DerivativeIR",
    "ImplicitResidualIR",
    "normalize_expression",
    "evaluate_expression",
    "evaluate_parts",
    "sum_residual_terms",
    "zero_slopes_for",
    "capture_forms",
    "capturing_forms",
    "record_form",
    "record_plain",
    "stamp_capture_error",
]


def evaluate_expression(
    expr: DerivativeExpr, slopes: dict, out_size: int, *, include_residual: bool = True
):
    """Evaluate a derivative expression at concrete per-field slope values.

    With ``include_residual=True`` (the default) this yields the full residual
    rows — the derivative contribution applied to the slopes plus the
    derivative-free part. With ``include_residual=False`` it yields the
    derivative action alone: every derivative-free term is excluded, so the
    result is exactly the derivative-coefficient blocks applied to the slopes.

    Args:
        expr: The expression an equation closure returned.
        slopes: Mapping of field name to that field's slope values.
        out_size: The equation's row-block length.
        include_residual: Whether derivative-free terms contribute.

    Raises:
        KeyError-free ValueError naming a referenced field that has no slope
        entry.
    """
    try:
        return _evaluate(expr, slopes, out_size, include_residual)
    except KeyError as missing:
        raise ValueError(
            f"the derivative expression references field {missing.args[0]!r}, "
            "which has no slope entry; derivative expressions may only "
            "reference declared fields"
        ) from None


class _Coo(NamedTuple):
    """A COO block over ``(out_size, in_size)`` with static pattern, dynamic values."""

    rows: np.ndarray
    cols: np.ndarray
    values: object  # jnp array (possibly traced)
    out_size: int
    in_size: int


def _coalesce(block: _Coo) -> _Coo:
    """Sum duplicate (row, col) entries; keep the pattern static."""
    if block.rows.size == 0:
        return block
    keys = block.rows.astype(np.int64) * int(block.in_size) + block.cols.astype(np.int64)
    unique, inverse = np.unique(keys, return_inverse=True)
    if unique.size == block.rows.size:
        return block
    import jax.ops  # noqa: F401  (segment_sum lives on jax.ops in some versions)
    from jax.ops import segment_sum

    values = segment_sum(
        jnp.asarray(block.values), jnp.asarray(inverse), num_segments=int(unique.size)
    )
    rows = (unique // int(block.in_size)).astype(np.int64)
    cols = (unique % int(block.in_size)).astype(np.int64)
    return _Coo(rows, cols, values, block.out_size, block.in_size)


def _operator_coo(op, in_size: int) -> _Coo:
    """The COO block of a declared linear operator's coefficients."""
    topo = op.coefficient_sparsity
    rows = np.asarray(topo.rows, dtype=np.int64)
    cols = np.asarray(topo.cols, dtype=np.int64)
    deps = op.dependencies
    depends = any(bool(getattr(deps, name)) for name in ("time", "state", "parameters", "control"))
    if op.coefficient_values is not None and not depends:
        values = jnp.asarray(op.coefficient_values(None, None, None, None))
    elif not depends:
        # Probe the constant coefficients column by column (host-side, once).
        n = int(topo.n_dofs)
        dense_cols = []
        for j in range(n):
            basis = jnp.zeros((n,), dtype=jnp.float64).at[j].set(1.0)
            dense_cols.append(np.asarray(op.apply(None, None, None, None, basis)))
        dense = np.stack(dense_cols, axis=1)
        values = jnp.asarray(dense[rows, cols])
    else:
        raise NotImplementedError(
            "an applied MassOperator whose coefficients depend on time, "
            "state, parameters, or the control channel cannot be normalized "
            "to constant coefficient values; compute the coefficient values "
            "inside the equation and scale the derivative expression instead "
            "(for example, weights * q.dt(field))"
        )
    return _Coo(rows, cols, values, int(op.n), int(in_size))


def _linear_blocks(expr: DerivativeExpr) -> dict:
    """Map each contributing field to its coalesced coefficient block."""
    if isinstance(expr, TimeDerivativeRef):
        n = expr.size
        idx = np.arange(n, dtype=np.int64)
        return {expr.field_name: _Coo(idx, idx, jnp.ones((n,)), n, n)}
    if isinstance(expr, DerivativeScale):
        inner = _linear_blocks(expr.expr)
        out = {}
        for field, block in inner.items():
            coeff = expr.coeff
            if np.ndim(coeff) == 0 or (hasattr(coeff, "shape") and coeff.shape == ()):
                values = jnp.asarray(block.values) * coeff
            else:
                values = jnp.asarray(block.values) * jnp.asarray(coeff)[block.rows]
            out[field] = _Coo(block.rows, block.cols, values, block.out_size, block.in_size)
        return out
    if isinstance(expr, DerivativeApply):
        inner = _linear_blocks(expr.expr)
        op_block = _operator_coo(expr.op, in_size=expr.expr.size)
        out = {}
        for field, block in inner.items():
            out[field] = _compose(op_block, block)
        return out
    if isinstance(expr, DerivativeGather):
        inner = _linear_blocks(expr.expr)
        idx = np.asarray(expr.indices, dtype=np.int64)
        out = {}
        for field, block in inner.items():
            keep = np.isin(block.rows, idx)
            row_map = np.full(block.out_size, -1, dtype=np.int64)
            row_map[idx] = np.arange(idx.size)
            rows = row_map[block.rows[keep]]
            out[field] = _coalesce(
                _Coo(
                    rows,
                    block.cols[keep],
                    jnp.asarray(block.values)[np.where(keep)[0]],
                    int(idx.size),
                    block.in_size,
                )
            )
        return out
    if isinstance(expr, DerivativeReduce):
        inner = _linear_blocks(expr.expr)
        weights = jnp.asarray(expr.weights)
        out = {}
        for field, block in inner.items():
            values = jnp.asarray(block.values) * weights[block.rows]
            out[field] = _coalesce(
                _Coo(np.zeros_like(block.rows), block.cols, values, 1, block.in_size)
            )
        return out
    if isinstance(expr, DerivativeScatter):
        inner = _linear_blocks(expr.expr)
        idx = np.asarray(expr.indices, dtype=np.int64)
        out = {}
        for field, block in inner.items():
            out[field] = _Coo(
                idx[block.rows], block.cols, block.values, expr.out_size, block.in_size
            )
        return out
    if isinstance(expr, DerivativeSum):
        merged: dict = {}
        for term in expr.terms:
            for field, block in _linear_blocks(term).items():
                if field in merged:
                    prev = merged[field]
                    merged[field] = _coalesce(
                        _Coo(
                            np.concatenate([prev.rows, block.rows]),
                            np.concatenate([prev.cols, block.cols]),
                            jnp.concatenate([jnp.asarray(prev.values), jnp.asarray(block.values)]),
                            prev.out_size,
                            prev.in_size,
                        )
                    )
                else:
                    merged[field] = block
        return merged
    raise TypeError(f"unsupported derivative-expression node {type(expr).__name__}")


def _compose(outer: _Coo, inner: _Coo) -> _Coo:
    """COO product ``outer @ inner`` with a static pattern."""
    entries_r, entries_c, chunks = [], [], []
    inner_by_row: dict = {}
    for k in range(inner.rows.size):
        inner_by_row.setdefault(int(inner.rows[k]), []).append(k)
    inner_values = jnp.asarray(inner.values)
    outer_values = jnp.asarray(outer.values)
    for k in range(outer.rows.size):
        mid = int(outer.cols[k])
        for j in inner_by_row.get(mid, ()):
            entries_r.append(int(outer.rows[k]))
            entries_c.append(int(inner.cols[j]))
            chunks.append(outer_values[k] * inner_values[j])
    if not entries_r:
        return _Coo(
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            jnp.zeros((0,)),
            outer.out_size,
            inner.in_size,
        )
    block = _Coo(
        np.asarray(entries_r, np.int64),
        np.asarray(entries_c, np.int64),
        jnp.stack(chunks),
        outer.out_size,
        inner.in_size,
    )
    return _coalesce(block)


class DerivativeTermIR:
    """One field's derivative-coefficient block within a normalized equation."""

    __slots__ = ("in_field", "_block")

    def __init__(self, in_field: str, block: _Coo):
        self.in_field = in_field
        self._block = block

    @property
    def pattern(self) -> tuple:
        """Static ``(rows, cols)`` of the coefficient block."""
        return self._block.rows, self._block.cols

    @property
    def out_size(self) -> int:
        return self._block.out_size

    @property
    def in_size(self) -> int:
        return self._block.in_size

    def values(self):
        """Coefficient values over the term's pattern (possibly traced).

        See [pattern][diff_soup.problem._ir.derivative_ir.DerivativeTermIR.pattern].
        """
        return self._block.values

    def apply(self, ydot_field):
        """Apply the block to one field's slope values."""
        contrib = jnp.asarray(self.values()) * jnp.asarray(ydot_field)[self._block.cols]
        return jnp.zeros((self.out_size,), dtype=contrib.dtype).at[self._block.rows].add(contrib)


class DerivativeIR:
    """A residual row block that is linear in the state derivative."""

    __slots__ = ("terms", "_residual_terms", "out_size")

    def __init__(self, terms: tuple, residual_terms: tuple, out_size: int):
        self.terms = tuple(terms)
        self._residual_terms = tuple(residual_terms)
        self.out_size = int(out_size)

    def residual_part(self):
        """The derivative-free contribution of the residual rows.

        A weak-form term (an operator output on a mesh) keeps its tag, so the
        residual finalizer integrates only the still-nodal part.
        """
        total = sum_residual_terms(self._residual_terms, self.out_size)
        return jnp.zeros((self.out_size,)) if total is None else total


class ImplicitResidualIR:
    """A residual row block that is nonlinear in the state derivative."""

    __slots__ = ("_expr", "ydot_fields", "out_size")

    def __init__(self, expr: DerivativeExpr, out_size: int):
        self._expr = expr
        self.ydot_fields = tuple(sorted(_referenced_fields(expr)))
        self.out_size = int(out_size)

    def evaluate(self, slopes: dict):
        """Evaluate the residual rows at concrete per-field slope values."""
        return _evaluate(self._expr, slopes, self.out_size)


def zero_slopes_for(expr: DerivativeExpr) -> dict:
    """A per-field zero-slope mapping covering every leaf the expression reads."""
    sizes: dict = {}

    def _walk(node) -> None:
        if isinstance(node, TimeDerivativeRef):
            sizes[node.field_name] = node.size
            return
        inner = getattr(node, "expr", None)
        if isinstance(inner, DerivativeExpr):
            _walk(inner)
        for group_attr in ("terms", "operands"):
            for member in getattr(node, group_attr, ()) or ():
                if isinstance(member, DerivativeExpr):
                    _walk(member)

    _walk(expr)
    return {name: jnp.zeros((size,)) for name, size in sizes.items()}


def _reject_nested_residual_terms(expr: DerivativeExpr, *, top: bool) -> None:
    """Reject a derivative-free term buried inside a scaled or mapped sum.

    The linear normalization records derivative-coefficient blocks plus the
    top-level derivative-free residual terms; a derivative-free term inside a
    scaling, operator application, gather, reduce, or scatter has no place in
    that structure and would be silently dropped. Rejected loudly instead:
    apply the scaling or map to the derivative term alone and add the
    derivative-free part as a separate top-level term.
    """
    if isinstance(expr, DerivativeSum):
        if not top and expr.residual_terms:
            raise ValueError(
                "a derivative-free term inside a scaled or mapped derivative "
                "expression cannot be normalized to coefficient structure; "
                "apply the scaling or map to the derivative term alone and "
                "add the derivative-free part as a separate term"
            )
        for term in expr.terms:
            _reject_nested_residual_terms(term, top=False)
        return
    inner = getattr(expr, "expr", None)
    if isinstance(inner, DerivativeExpr):
        _reject_nested_residual_terms(inner, top=False)
    for member in getattr(expr, "operands", ()) or ():
        if isinstance(member, DerivativeExpr):
            _reject_nested_residual_terms(member, top=False)


def _referenced_fields(expr: DerivativeExpr) -> set:
    if isinstance(expr, TimeDerivativeRef):
        return {expr.field_name}
    fields: set = set()
    for attr in ("expr",):
        inner = getattr(expr, attr, None)
        if isinstance(inner, DerivativeExpr):
            fields |= _referenced_fields(inner)
    for group_attr in ("terms", "operands"):
        for inner in getattr(expr, group_attr, ()) or ():
            if isinstance(inner, DerivativeExpr):
                fields |= _referenced_fields(inner)
    return fields


def _is_row_value(value, out_size: int) -> bool:
    """Whether ``value`` is a plain number or one plain value per row.

    A tagged residual term (a mesh operator's weak-form output) is not, and
    neither is an array of another shape (values at a mesh's quadrature
    points).
    """
    if not isinstance(value, (np.ndarray, np.generic, int, float, complex, jax.Array)):
        return False
    shape = np.shape(value)
    return shape in ((), (1,), (int(out_size),))


def sum_residual_terms(terms, out_size: int):
    """Add derivative-free residual terms, keeping tagged terms intact.

    Plain numbers and one-value-per-row arrays are summed into one
    length-``out_size`` block. Any other term (a mesh operator's weak-form
    output, or values at the quadrature points) is combined with ``+`` and
    never converted with ``jnp.asarray``, which would drop its tag; the
    residual finalizer integrates it.

    Returns:
        The sum, or ``None`` when ``terms`` is empty.
    """
    terms = tuple(terms)
    if not terms:
        return None
    # Tagged terms lead, so an array of another shape is added to a tagged
    # term (which integrates it) rather than the other way round.
    tagged = sorted(
        (r for r in terms if not _is_row_value(r, out_size)),
        key=lambda r: isinstance(r, (np.ndarray, jax.Array)),
    )
    plain = [r for r in terms if _is_row_value(r, out_size)]
    total = None
    for r in tagged:
        total = r if total is None else total + r
    if plain:
        block = jnp.zeros((out_size,))
        for r in plain:
            block = block + jnp.asarray(r)
        total = block if total is None else total + block
    return total


def evaluate_parts(expr: DerivativeExpr, slopes: dict, out_size: int, *, include_residual=True):
    """Evaluate a derivative expression as ``(derivative part, residual part)``.

    The derivative part applies every derivative term to the slopes. The
    residual part sums the expression's top-level derivative-free terms; it is
    ``None`` when there are none or when ``include_residual`` is ``False``.
    Kept apart so the caller can weight the derivative part (the lumped mass on
    a mesh) before the two are added.

    Args:
        expr: The expression an equation closure returned.
        slopes: Mapping of field name to that field's slope values.
        out_size: The equation's row-block length.
        include_residual: Whether derivative-free terms contribute.
    """
    try:
        if isinstance(expr, DerivativeSum):
            action = None
            for term in expr.terms:
                value = _evaluate(term, slopes, out_size, include_residual)
                action = value if action is None else action + value
            residual = (
                sum_residual_terms(expr.residual_terms, out_size) if include_residual else None
            )
            return action, residual
        return _evaluate(expr, slopes, out_size, include_residual), None
    except KeyError as missing:
        raise ValueError(
            f"the derivative expression references field {missing.args[0]!r}, "
            "which has no slope entry; derivative expressions may only "
            "reference declared fields"
        ) from None


def _evaluate(expr, slopes: dict, out_size: int, include_residual: bool = True):
    if isinstance(expr, TimeDerivativeRef):
        return jnp.asarray(slopes[expr.field_name])
    if isinstance(expr, DerivativeScale):
        return jnp.asarray(expr.coeff) * _evaluate(expr.expr, slopes, out_size, include_residual)
    if isinstance(expr, DerivativeGather):
        return _evaluate(expr.expr, slopes, out_size, include_residual)[np.asarray(expr.indices)]
    if isinstance(expr, DerivativeReduce):
        inner = _evaluate(expr.expr, slopes, out_size, include_residual)
        return jnp.sum(jnp.asarray(expr.weights) * inner)[None]
    if isinstance(expr, DerivativeScatter):
        inner = _evaluate(expr.expr, slopes, out_size, include_residual)
        out = jnp.zeros((expr.out_size,), dtype=inner.dtype)
        return out.at[np.asarray(expr.indices)].set(inner)
    if isinstance(expr, DerivativeSum):
        total = None
        for term in expr.terms:
            value = _evaluate(term, slopes, out_size, include_residual)
            total = value if total is None else total + value
        if include_residual:
            residual = sum_residual_terms(expr.residual_terms, out_size)
            if residual is not None:
                total = residual if total is None else total + residual
        return total
    if isinstance(expr, NonlinearDerivativeExpr):
        return expr.combine(
            *[_evaluate(o, slopes, out_size, include_residual) for o in expr.operands]
        )
    if isinstance(expr, DerivativeApply):
        inner = _evaluate(expr.expr, slopes, out_size, include_residual)
        block = _operator_coo(expr.op, in_size=expr.expr.size)
        contrib = jnp.asarray(block.values) * inner[block.cols]
        return jnp.zeros((block.out_size,), dtype=contrib.dtype).at[block.rows].add(contrib)
    raise TypeError(f"unsupported derivative-expression node {type(expr).__name__}")


def normalize_expression(
    expr: DerivativeExpr, *, out_size: int
) -> Union[DerivativeIR, ImplicitResidualIR]:
    """Normalize an equation's derivative-expression return.

    Args:
        expr: The value the closure returned.
        out_size: The equation's row-block length (the written field's size).

    Returns:
        [DerivativeIR][diff_soup.problem._ir.derivative_ir.DerivativeIR] when the
        expression is linear in the state
        derivative,
        [ImplicitResidualIR][diff_soup.problem._ir.derivative_ir.ImplicitResidualIR]
        otherwise.

    Raises:
        ValueError: If the expression's output length disagrees with the
            equation's row block.
    """
    if int(expr.size) != int(out_size):
        raise ValueError(
            f"the derivative expression produces {expr.size} rows but the "
            f"equation's row block has size {out_size}"
        )
    if not expr.linear_in_ydot:
        return ImplicitResidualIR(expr, out_size)
    _reject_nested_residual_terms(expr, top=True)
    if isinstance(expr, DerivativeSum):
        blocks = _linear_blocks(expr)
        residual_terms = expr.residual_terms
    else:
        blocks = _linear_blocks(expr)
        residual_terms = ()
    terms = tuple(DerivativeTermIR(field, block) for field, block in sorted(blocks.items()))
    return DerivativeIR(terms=terms, residual_terms=residual_terms, out_size=out_size)


# ---------------------------------------------------------------------------
# Build-time capture of derivative forms
# ---------------------------------------------------------------------------
#
# A residual-form equation adapter (component or family member) records its
# derivative structure into its form cell only inside ``capture_forms``. Every
# other evaluation (stencil probes, traces, solves) leaves the cell alone, so
# the build-time capture pass is the one place a form is recorded.

_CAPTURE = threading.local()


def capturing_forms() -> bool:
    """Whether a build-time capture pass is evaluating the equations."""
    return bool(getattr(_CAPTURE, "active", False))


@contextlib.contextmanager
def _capture_context():
    previous = capturing_forms()
    _CAPTURE.active = True
    try:
        yield
    finally:
        _CAPTURE.active = previous


def _decided(cell) -> bool:
    return bool(cell.get("captured") or cell.get("captured_plain"))


def stamp_capture_error(cells, exc: BaseException) -> None:
    """Store why a capture pass failed in each cell it left undecided."""
    note = f"{type(exc).__name__}: {exc}"
    for cell in cells:
        if cell is not None and not _decided(cell):
            cell.setdefault("capture_error", note)


def capture_forms(cells, evaluate) -> None:
    """Run the build-time capture pass for the equations owning ``cells``.

    ``evaluate()`` evaluates those equations once, eagerly, at a generic
    state. Inside it each residual-form adapter records its structure and each
    plain equation marks itself plain. A failure is stored in every cell still
    undecided, so ``LoweredProblem.derivative_forms()`` raises with the reason
    instead of lowering the equation with the plain mass diagonal.

    Args:
        cells: The form cells the evaluation should decide.
        evaluate: A callable of no arguments that evaluates the equations.
    """
    pending = [cell for cell in cells if cell is not None and not _decided(cell)]
    if not pending:
        return
    try:
        with _capture_context(), jax.ensure_compile_time_eval():
            evaluate()
    except Exception as exc:  # noqa: BLE001 - stored, raised at the read.
        stamp_capture_error(pending, exc)


def record_form(cell, field_name: str, ir, *, state_deps, check_fields=None) -> None:
    """Record one equation's derivative structure, during a capture pass only.

    Args:
        cell: The equation's form cell.
        field_name: The field the equation writes.
        ir: The normalized expression (``normalize_expression``).
        state_deps: A callable of no arguments returning which fields the
            coefficients read (``None`` when undecidable); linear forms only.
        check_fields: An optional callable that rejects the referenced fields.
    """
    if not capturing_forms() or cell.get("captured"):
        return
    if isinstance(ir, ImplicitResidualIR):
        if check_fields is not None:
            check_fields(ir.ydot_fields)
        cell.update(out_field=field_name, terms=(), nonlinear=True, ydot_fields=ir.ydot_fields)
    else:
        if check_fields is not None:
            check_fields(term.in_field for term in ir.terms)
        cell.update(
            out_field=field_name,
            terms=tuple((term.in_field, term.pattern[0], term.pattern[1]) for term in ir.terms),
            nonlinear=False,
            state_deps=state_deps(),
        )
    cell["captured"] = True


def record_plain(cell) -> None:
    """Mark an equation as plain (no derivative structure), during a capture pass only."""
    if capturing_forms() and not cell.get("captured"):
        cell["captured_plain"] = True
