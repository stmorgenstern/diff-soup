"""Typed derivative-expression nodes for residual-form Recipe equations.

``q.dt(field)`` yields a
[TimeDerivativeRef][diff_soup.problem._spec.derivative_expr.TimeDerivativeRef]
leaf. A small linear algebra composes leaves into structured expressions —
addition/subtraction, scaling by a derivative-free value, application of a
declared linear operator, and gather/reduce/scatter with static index maps. An
equation closure that returns one of these expressions states a residual
(``expression == 0``); plain-array returns keep the right-hand-side reading.

Linearity in the state derivative is decided by construction: every node in
this module is linear in the derivative except an explicit product of two
derivative expressions, which is recorded as a
[NonlinearDerivativeExpr][diff_soup.problem._spec.derivative_expr.NonlinearDerivativeExpr]
and lowers to the general implicit problem form. Applying an arbitrary function
(``jnp.sin`` and friends) to a derivative expression is rejected at build time
— author such equations as a general implicit residual instead.

These objects are transient authoring-time values: they are produced while an
equation closure runs and are normalized away during lowering. They never
cross a JIT boundary themselves.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Optional

import numpy as np

__all__ = [
    "DerivativeExpr",
    "TimeDerivativeRef",
    "DerivativeScale",
    "DerivativeApply",
    "DerivativeGather",
    "DerivativeReduce",
    "DerivativeScatter",
    "DerivativeSum",
    "NonlinearDerivativeExpr",
    "slope_context",
    "current_slopes",
    "slope_context_includes_residual",
]

# Trace-time slope substitution: while a slope context is active, a
# residual-form equation adapter evaluates its derivative expression against
# the provided per-field slope values instead of emitting only the
# derivative-free part. The context is read at trace time and participates in
# whichever trace installs it.
_SLOPE_STATE = threading.local()


def current_slopes() -> Optional[dict]:
    """The active per-field slope mapping, or ``None`` outside a context."""
    return getattr(_SLOPE_STATE, "slopes", None)


def slope_context_includes_residual() -> bool:
    """Whether the active context asks for full residual rows.

    ``True`` (the default) yields the residual rows with slopes substituted —
    derivative terms plus the derivative-free part. ``False`` yields the
    derivative action alone: exactly the derivative-coefficient blocks applied
    to the slopes, with every derivative-free term excluded. The mass-form
    lowering uses the latter so a constant-coefficient operator materializes
    bit-identically at every state.
    """
    return getattr(_SLOPE_STATE, "include_residual", True)


@contextlib.contextmanager
def slope_context(slopes: dict, *, include_residual: bool = True):
    """Install per-field slope values for residual-form equation evaluation.

    Args:
        slopes: Mapping of field name to that field's slope values.
        include_residual: ``True`` for full residual rows; ``False`` for the
            derivative action alone (see
            [slope_context_includes_residual][diff_soup.problem._spec.derivative_expr.slope_context_includes_residual]).
    """
    previous = (
        getattr(_SLOPE_STATE, "slopes", None),
        getattr(_SLOPE_STATE, "include_residual", True),
    )
    _SLOPE_STATE.slopes = slopes
    _SLOPE_STATE.include_residual = include_residual
    try:
        yield
    finally:
        _SLOPE_STATE.slopes, _SLOPE_STATE.include_residual = previous


_IMPLICIT_GUIDANCE = (
    "a nonlinear function of a time-derivative expression cannot be lowered "
    "to the mass form; author the equation as a general implicit residual "
    "in the state derivative instead"
)


def _static_len(value) -> int:
    """The static leading length of an array-like, or 1 for a scalar."""
    shape = np.shape(value) if not hasattr(value, "shape") else value.shape
    return int(shape[0]) if len(shape) else 1


class DerivativeExpr:
    """Base of the derivative-expression algebra.

    Supports ``+``, ``-``, unary ``-``, scaling by a derivative-free value,
    ``*`` between two expressions (recorded as nonlinear), and the structural
    [gather][diff_soup.problem._spec.derivative_expr.DerivativeExpr.gather],
    [reduce][diff_soup.problem._spec.derivative_expr.DerivativeExpr.reduce], and
    [scatter][diff_soup.problem._spec.derivative_expr.DerivativeExpr.scatter]
    maps.
    """

    __slots__ = ()

    @property
    def size(self) -> int:
        """Output length of this expression."""
        raise NotImplementedError

    @property
    def linear_in_ydot(self) -> bool:
        """Whether the expression is linear in the state derivative."""
        raise NotImplementedError

    # -- algebra -----------------------------------------------------------
    def __add__(self, other):
        return _sum_of(self, other, negate_other=False)

    def __radd__(self, other):
        return _sum_of(self, other, negate_other=False)

    def __sub__(self, other):
        return _sum_of(self, other, negate_other=True)

    def __rsub__(self, other):
        return _sum_of(-self, other, negate_other=False)

    def __neg__(self):
        return DerivativeScale(self, -1.0)

    def __mul__(self, other):
        if isinstance(other, DerivativeExpr):
            return NonlinearDerivativeExpr(operands=(self, other), combine=lambda a, b: a * b)
        return DerivativeScale(self, other)

    def __rmul__(self, other):
        return DerivativeScale(self, other)

    def __truediv__(self, other):
        if isinstance(other, DerivativeExpr):
            return NonlinearDerivativeExpr(operands=(self, other), combine=lambda a, b: a / b)
        return DerivativeScale(self, 1.0 / other)

    # -- structural maps ---------------------------------------------------
    def gather(self, indices) -> "DerivativeGather":
        """Select entries of this expression at static ``indices``."""
        return DerivativeGather(self, indices)

    def reduce(self, weights) -> "DerivativeReduce":
        """The weighted sum of this expression's entries (a length-1 output)."""
        return DerivativeReduce(self, weights)

    def scatter(self, indices, out_size: int) -> "DerivativeScatter":
        """Place this expression's entries at ``indices`` of a length-``out_size`` row block."""
        return DerivativeScatter(self, indices, out_size)

    # -- explicit rejections ------------------------------------------------
    # NumPy ufuncs defer to the reflected operators above instead of trying to
    # convert the expression to an array; a direct conversion attempt (what
    # ``jnp.sin`` and ``np.asarray`` do) fails with guidance.
    __array_ufunc__ = None

    def __array__(self, dtype=None):
        raise TypeError(_IMPLICIT_GUIDANCE)

    def __iter__(self):
        raise TypeError(_IMPLICIT_GUIDANCE)

    def __bool__(self):
        raise TypeError(
            "a derivative expression has no truth value; compare or combine "
            "its lowered residual instead"
        )


def _sum_of(expr: DerivativeExpr, other, *, negate_other: bool) -> "DerivativeSum":
    """Combine ``expr`` with ``other`` (expression or derivative-free value)."""
    terms: list = []
    residual_terms: list = []

    def _absorb(value, negate: bool) -> None:
        if isinstance(value, DerivativeSum):
            for t in value.terms:
                terms.append(DerivativeScale(t, -1.0) if negate else t)
            for r in value.residual_terms:
                residual_terms.append(-r if negate else r)
        elif isinstance(value, DerivativeExpr):
            terms.append(DerivativeScale(value, -1.0) if negate else value)
        else:
            residual_terms.append(-value if negate else value)

    _absorb(expr, False)
    _absorb(other, negate_other)
    return DerivativeSum(terms=tuple(terms), residual_terms=tuple(residual_terms))


class TimeDerivativeRef(DerivativeExpr):
    """The time derivative of one field — the ``q.dt(field)`` leaf."""

    __slots__ = ("field_name", "var", "size_")

    def __init__(self, *, field_name: str, var: Optional[Any], size: int):
        self.field_name = str(field_name)
        self.var = var
        if int(size) <= 0:
            raise ValueError(f"q.dt({field_name!r}): field size must be positive, got {size}")
        self.size_ = int(size)

    @property
    def size(self) -> int:
        return self.size_

    @property
    def linear_in_ydot(self) -> bool:
        return True


class DerivativeScale(DerivativeExpr):
    """A derivative expression scaled by a derivative-free value."""

    __slots__ = ("expr", "coeff")

    def __init__(self, expr: DerivativeExpr, coeff):
        if isinstance(coeff, DerivativeExpr):
            raise TypeError(
                "a scaling coefficient must be derivative-free; multiply two "
                "derivative expressions directly to state a nonlinear residual"
            )
        coeff_len = _static_len(coeff)
        if coeff_len not in (1, expr.size):
            raise ValueError(
                f"scaling coefficient of length {coeff_len} does not match the "
                f"expression size {expr.size}"
            )
        self.expr = expr
        self.coeff = coeff

    @property
    def size(self) -> int:
        return self.expr.size

    @property
    def linear_in_ydot(self) -> bool:
        return self.expr.linear_in_ydot


class DerivativeApply(DerivativeExpr):
    """A declared linear operator applied to a derivative expression."""

    __slots__ = ("op", "expr", "out_size")

    def __init__(self, op, expr: DerivativeExpr, out_size: Optional[int] = None):
        self.op = op
        self.expr = expr
        inferred = out_size if out_size is not None else getattr(op, "n", None)
        if inferred is None:
            raise ValueError(
                "the applied operator declares no output size; pass out_size= "
                "or use an operator exposing its side length"
            )
        self.out_size = int(inferred)

    @property
    def size(self) -> int:
        return self.out_size

    @property
    def linear_in_ydot(self) -> bool:
        return self.expr.linear_in_ydot


class DerivativeGather(DerivativeExpr):
    """Static index selection of a derivative expression's entries."""

    __slots__ = ("expr", "indices")

    def __init__(self, expr: DerivativeExpr, indices):
        idx = tuple(int(i) for i in indices)
        bad = [i for i in idx if i < 0 or i >= expr.size]
        if bad:
            raise ValueError(
                f"gather index {bad[0]} is out of range for an expression of size {expr.size}"
            )
        self.expr = expr
        self.indices = idx

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def linear_in_ydot(self) -> bool:
        return self.expr.linear_in_ydot


class DerivativeReduce(DerivativeExpr):
    """The weighted sum of a derivative expression's entries."""

    __slots__ = ("expr", "weights")

    def __init__(self, expr: DerivativeExpr, weights):
        weights_len = _static_len(weights)
        if weights_len != expr.size:
            raise ValueError(
                f"reduction weights of length {weights_len} do not match the "
                f"expression size {expr.size}"
            )
        self.expr = expr
        self.weights = weights

    @property
    def size(self) -> int:
        return 1

    @property
    def linear_in_ydot(self) -> bool:
        return self.expr.linear_in_ydot


class DerivativeScatter(DerivativeExpr):
    """Placement of a derivative expression's entries into a wider row block."""

    __slots__ = ("expr", "indices", "out_size")

    def __init__(self, expr: DerivativeExpr, indices, out_size: int):
        idx = tuple(int(i) for i in indices)
        if len(idx) != expr.size:
            raise ValueError(
                f"scatter takes one index per entry: got {len(idx)} indices "
                f"for an expression of size {expr.size}"
            )
        out_n = int(out_size)
        bad = [i for i in idx if i < 0 or i >= out_n]
        if bad:
            raise ValueError(
                f"scatter index {bad[0]} is out of range for an output of size {out_n}"
            )
        self.expr = expr
        self.indices = idx
        self.out_size = out_n

    @property
    def size(self) -> int:
        return self.out_size

    @property
    def linear_in_ydot(self) -> bool:
        return self.expr.linear_in_ydot


class DerivativeSum(DerivativeExpr):
    """A sum of derivative terms plus derivative-free residual terms."""

    __slots__ = ("terms", "residual_terms")

    def __init__(self, *, terms: tuple, residual_terms: tuple = ()):
        if not terms:
            raise ValueError("a derivative sum needs at least one derivative term")
        sizes = {t.size for t in terms}
        if len(sizes) != 1:
            raise ValueError(
                f"derivative terms of mismatched size cannot be summed: {sorted(sizes)}"
            )
        (size,) = sizes
        for r in residual_terms:
            if np.ndim(r) >= 2:
                # Values at a mesh's quadrature points (a quad-native field
                # after arithmetic, ``(n_cells, n_q, ...)``): the residual
                # finalizer integrates them to one value per node.
                continue
            r_len = _static_len(r)
            if r_len not in (1, size):
                raise ValueError(
                    f"residual term of length {r_len} does not match the "
                    f"derivative-term size {size}"
                )
        self.terms = tuple(terms)
        self.residual_terms = tuple(residual_terms)

    @property
    def size(self) -> int:
        return self.terms[0].size

    @property
    def linear_in_ydot(self) -> bool:
        return all(t.linear_in_ydot for t in self.terms)


class NonlinearDerivativeExpr(DerivativeExpr):
    """A combination that is nonlinear in the state derivative.

    Produced by multiplying (or dividing) two derivative expressions. Such an
    equation lowers to the general implicit problem form rather than the mass
    form.
    """

    __slots__ = ("operands", "combine")

    def __init__(self, *, operands: tuple, combine):
        if not operands:
            raise ValueError("a nonlinear combination needs at least one operand")
        sizes = {o.size for o in operands if isinstance(o, DerivativeExpr)}
        if len(sizes) > 1:
            raise ValueError(f"operands of mismatched size cannot be combined: {sorted(sizes)}")
        self.operands = tuple(operands)
        self.combine = combine

    @property
    def size(self) -> int:
        for o in self.operands:
            if isinstance(o, DerivativeExpr):
                return o.size
        raise ValueError("a nonlinear combination needs a derivative operand")

    @property
    def linear_in_ydot(self) -> bool:
        return False
