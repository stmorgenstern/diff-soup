"""QuadField — a residual term tagged as living at FEM quadrature points.

A ``QuadField`` (``qf`` below) is built from a quad-shaped array with
``q.ops.wrap_quad_value(value)`` on an FEM domain.

Companion to ``dispatch._WeakTerm``: where ``_WeakTerm`` tracks nodal-vs-
integrated provenance, ``QuadField`` marks a quad-shaped value ``(n_cells,
n_q, *component)`` that must be integrated with the consistent-mass action
``int f N_i`` (``dispatch_integrate_scalar_at_quad``) — NOT the nodal mass
action — when it reaches the residual assembler.

Arithmetic is *sticky* (stays a ``QuadField`` under ``+``, ``-``, ``*``,
``/`` with scalars, same-batch arrays, and other ``QuadField`` s) so a
composed residual still carries the quad tag; multiplying by a tensor whose
component axes extend beyond the batch broadcasts the scalar over them
(``scale``).  Stickiness must survive even when a raw ``jax``/``numpy``
array is the *left* operand (``kappa * qf`` with a traced ``kappa``):
``QuadField`` is registered as a JAX **pytree** (value as the single leaf,
``(domain, batch_ndim)`` as static aux) and carries ``__array_priority__``,
so it is opaque to ``ArrayImpl.__mul__`` / ``ndarray.__mul__`` and Python
falls back to ``QuadField.__rmul__`` / ``__radd__`` — mirroring ``_WeakTerm``
(which is opaque for the same reason).  It deliberately does **not** define
``__jax_array__``: that protocol both (a) caused array-left operands to
unwrap to raw arrays (dropping the quad tag) and (b) is deprecated by JAX
("register your object as a pytree").  Apply a ``jnp`` ufunc with
[apply][diff_soup.domains.operators._quadfield.QuadField.apply]
(``qf.apply(jnp.log)`` — stays a ``QuadField``) or unwrap explicitly via
[value][diff_soup.domains.operators._quadfield.QuadField.value]
(``jnp.log(qf.value)``).

``QuadField (+|-) _WeakTerm`` integrates the ``QuadField`` to a nodal weak
vector first (consistent-mass action via ``integrate``), then defers to the
``_WeakTerm`` arithmetic so the two provenances combine consistently.

``batch_ndim`` is fixed at ``2`` (``cells, n_q``) for the cases the tensor-
native FEM path needs; higher-rank batches are out of scope for now.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import tree_util

from diff_soup.domains.operators._tensor import scale as _scale


class QuadField:
    """Quad-shaped residual term with sticky tensor-aware arithmetic."""

    __slots__ = ("value", "_domain", "batch_ndim")

    # Beat numpy's ``ndarray.__mul__`` / ``__add__`` so a numpy-array LEFT
    # operand defers to ``QuadField.__rmul__`` / ``__radd__`` instead of
    # broadcasting us into an object-dtype array (silent corruption).
    __array_priority__ = 100

    def __init__(self, value, domain, batch_ndim: int = 2) -> None:
        self.value = jnp.asarray(value)
        self._domain = domain
        self.batch_ndim = batch_ndim

    # -- ufunc interop: explicit, sticky -------------------------------
    def apply(self, fn):
        """Apply a ``jnp`` ufunc to the underlying value, staying a QuadField.

        ``qf.apply(jnp.log)`` returns a ``QuadField``; ``jnp.log(qf)`` is not
        supported because ``QuadField`` does not implement ``__jax_array__``.
        For a raw result, unwrap explicitly with ``jnp.log(qf.value)``.
        """
        return self._wrap(fn(self.value))

    # -- assembly ------------------------------------------------------
    def integrate(self):
        """Integrate a scalar quad field to a nodal weak vector (``int f N_i``)."""
        from diff_soup.domains.operators.dispatch import (
            dispatch_integrate_scalar_at_quad,
        )

        return dispatch_integrate_scalar_at_quad(self._domain, self.value)

    # -- helpers -------------------------------------------------------
    def _wrap(self, v):
        """Wrap an array ``v`` as a ``QuadField`` over the same domain and batch_ndim."""
        return QuadField(v, self._domain, self.batch_ndim)

    @staticmethod
    def _is_weakterm(other):
        """Return ``True`` iff ``other`` is a ``_WeakTerm`` integration result."""
        from diff_soup.domains.operators.dispatch import _WeakTerm

        return isinstance(other, _WeakTerm)

    def _is_component_tensor(self, o):
        """``o`` carries component axes beyond our batch (broadcast via scale).

        Two cases qualify, mirroring ``_tensor.scale``'s own dispatch:

        * a *batched* tensor field ``(n_cells, n_q, *component)`` — same batch
          prefix as the value, plus trailing component axes; and
        * a *constant* tensor (e.g. ``(d, d)`` or ``(d, d, d, d)``) whose
          leading axes do **not** match the value's batch.

        A same-batch *scalar* array ``(n_cells, n_q)`` is NOT a component
        tensor (it is the sticky elementwise case).
        """
        batch = self.value.shape[: self.batch_ndim]
        matches_batch = o.shape[: self.batch_ndim] == batch
        if matches_batch:
            return o.ndim > self.batch_ndim  # extra trailing component axes
        return o.ndim >= 1  # constant component tensor (batch prefix mismatch)

    def _binop(self, other, op, reflected: bool = False):
        """Execute a binary operation, handling ``_WeakTerm`` interop and tensor-scale dispatch."""
        # _WeakTerm interop: integrate self to a nodal weak vector, then defer
        # to the weak-term arithmetic so the two provenances combine cleanly.
        if self._is_weakterm(other):
            from diff_soup.domains.operators.dispatch import _WeakTerm

            integ = self.integrate()  # nodal, already integrated
            wt = _WeakTerm(integ, True, self._domain)
            return op(other, wt) if reflected else op(wt, other)

        if isinstance(other, QuadField):
            o = other.value
        else:
            o = jnp.asarray(other)
            # Multiplying by a component-tensor broadcasts the scalar field
            # over the trailing component axes (``c * sigma0``).
            if op is _MUL and self._is_component_tensor(o):
                return self._wrap(_scale(self.value, o))

        return self._wrap(op(o, self.value) if reflected else op(self.value, o))

    # -- arithmetic ----------------------------------------------------
    def __add__(self, o):
        return self._binop(o, _ADD)

    def __radd__(self, o):
        return self._binop(o, _ADD, reflected=True)

    def __sub__(self, o):
        return self._binop(o, _SUB)

    def __rsub__(self, o):
        return self._binop(o, _SUB, reflected=True)

    def __mul__(self, o):
        return self._binop(o, _MUL)

    def __rmul__(self, o):
        return self._binop(o, _MUL, reflected=True)

    def __truediv__(self, o):
        return self._binop(o, _DIV)

    def __rtruediv__(self, o):
        return self._binop(o, _DIV, reflected=True)

    def __neg__(self):
        return self._wrap(-self.value)


def _ADD(a, b):
    """Element-wise addition helper for ``_binop``."""
    return a + b


def _SUB(a, b):
    """Element-wise subtraction helper for ``_binop``."""
    return a - b


def _MUL(a, b):
    """Element-wise multiplication helper for ``_binop``."""
    return a * b


def _DIV(a, b):
    """Element-wise division helper for ``_binop``."""
    return a / b


# -- pytree registration -----------------------------------------------------
# Registering ``QuadField`` as a pytree (value = single leaf, domain +
# batch_ndim = static aux) is what makes it opaque to ``ArrayImpl.__mul__``:
# JAX returns ``NotImplemented`` for an unregistered-array operand, so Python
# falls back to ``QuadField.__rmul__`` and the quad tag survives an array-LEFT
# operand.  It is also the JAX-recommended replacement for the deprecated
# ``__jax_array__`` ufunc protocol.
def _qf_flatten(qf: QuadField):
    """Flatten ``QuadField`` pytree: value is the single leaf; domain + batch_ndim are static aux."""
    return (qf.value,), (qf._domain, qf.batch_ndim)


def _qf_unflatten(aux, children):
    """Reconstruct a ``QuadField`` from a flattened pytree."""
    domain, batch_ndim = aux
    obj = QuadField.__new__(QuadField)
    obj.value = children[0]
    obj._domain = domain
    obj.batch_ndim = batch_ndim
    return obj


tree_util.register_pytree_node(QuadField, _qf_flatten, _qf_unflatten)
