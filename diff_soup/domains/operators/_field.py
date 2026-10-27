"""Field — a quad-native FEM authoring wrapper that decays to a raw quad array.

The ergonomic default for FEM strong-form residual authoring.  A ``Field``
wraps a *nodal* DOF array and, the instant it touches any arithmetic or
``jnp`` ufunc, decays to a **raw quadrature-point array**
``(n_cells, n_q, *component)`` via ``ops.interp_to_quad(nodal)``.  This lets
a single inline strong-form closure written against ``q.c`` evaluate
quad-consistently on FEM — full quadrature accuracy for nonlinear pointwise
terms — while the residual assembler auto-integrates the leftover raw quad
array (``int f N_i``).

Why decay to a *raw array* and not a [QuadField][diff_soup.domains.operators._quadfield.QuadField]?
    A transcendental like ``jnp.log(x)`` only dispatches through a custom
    object if that object implements ``__jax_array__`` (JAX bypasses numpy's
    ``__array_ufunc__``).  ``QuadField`` deliberately has **no**
    ``__jax_array__`` (the protocol would drop the quad tag on array-left
    operands, and JAX deprecates it), so ``jnp.log(quadfield)`` would fail.  A raw
    quad array, by contrast, composes with every ``jnp`` op with zero friction
    — so ``Field`` decays to one under both ``jnp`` ufuncs (``__jax_array__``)
    and ``+ - * /`` (operator dunders).  ``__jax_array__`` is deprecated-but-
    functional in jax 0.6.2.

What survives the decay:
    * ``.nodal`` — the original DOF-space array (what operators consume).
    * pytree-ness — ``Field`` registers with ``jax.tree_util`` (leaf =
      ``nodal``, static aux = ``(ops,)``) so ``grad``/``jvp``/``vmap`` flow
      through the ``nodal`` leaf and the cached ``Operators`` is static.

``Field`` is constructed **only** on FEM domains, and only for fields the
author opted into with ``spec.field(quad=True)``.  On grids and for non-opted
fields ``q.c`` stays a plain cell array (FVM/FDM path unchanged).

Indexing / length are an ambiguous nodal-vs-quad coercion, so ``Field``
**raises** on ``f[idx]`` / ``len(f)`` and points the author at ``.nodal``
(DOF space) or the explicit quad value.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import tree_util

from diff_soup.domains.operators._tensor import scale as _scale


class Field:
    """Quad-native FEM field that decays to a raw quad array under math.

    Args:
        nodal: nodal DOF array (``(num_dofs,)`` scalar or ``(num_dofs, dim)``
            vector/tensor field).
        ops: the [Operators][diff_soup.domains.operators.base.Operators] instance
            bound to the field's FEM domain (provides ``interp_to_quad``).
    """

    __slots__ = ("nodal", "_ops")

    # Beat ``ndarray.__mul__`` / ``__add__`` (and ``ArrayImpl``'s) so an
    # array-LEFT operand defers to ``Field.__rmul__`` / ``__radd__`` rather
    # than broadcasting us into an object-dtype array.
    __array_priority__ = 100

    def __init__(self, nodal, ops) -> None:
        self.nodal = jnp.asarray(nodal)
        self._ops = ops

    # -- quad decay ----------------------------------------------------
    @property
    def _q(self):
        """Lazy nodal -> quad interpolation (``(n_cells, n_q, *component)``)."""
        return self._ops.interp_to_quad(self.nodal)

    def __jax_array__(self):
        """Decay to the raw quad array so ``jnp`` ufuncs dispatch through us."""
        return self._q

    @staticmethod
    def _o(o):
        """Coerce a ``Field`` operand to its raw quad array; pass others through."""
        return o._q if isinstance(o, Field) else o

    # -- tensor-aware multiply -----------------------------------------
    def _is_component_tensor(self, o):
        """``o`` carries component axes beyond the (scalar) quad batch.

        Mirrors [_is_component_tensor][diff_soup.domains.operators._field.QuadField._is_component_tensor]: a quad scalar
        ``(n_cells, n_q)`` times a constant component tensor (``(d, d)`` /
        ``(d, d, d, d)``) or a batched tensor field ``(n_cells, n_q, *comp)``
        must broadcast the scalar over the trailing component axes via
        [scale][diff_soup.domains.operators._tensor.scale] (so ``c * SIGMA0`` scales SIGMA0 per quad point),
        not elementwise.  A same-batch scalar array is NOT a component tensor.
        """
        q = self._q
        # Only the scalar-field case needs the scale rule; a vector/tensor
        # Field already broadcasts elementwise against its operand.
        if q.ndim != 2:
            return False
        batch = q.shape
        if o.ndim < 2:
            return o.ndim >= 1  # 1-D constant component vector
        if o.shape[:2] == batch:
            return o.ndim > 2  # same batch + extra trailing component axes
        return o.ndim >= 1  # constant component tensor (batch prefix mismatch)

    def _mul(self, o, reflected: bool = False):
        if isinstance(o, Field):
            return self._q * o._q
        other = jnp.asarray(o)
        if self._is_component_tensor(other):
            return _scale(self._q, other)
        return other * self._q if reflected else self._q * other

    # -- arithmetic (decay to raw quad arrays) -------------------------
    def __add__(self, o):
        return self._q + Field._o(o)

    def __radd__(self, o):
        return Field._o(o) + self._q

    def __sub__(self, o):
        return self._q - Field._o(o)

    def __rsub__(self, o):
        return Field._o(o) - self._q

    def __mul__(self, o):
        return self._mul(o)

    def __rmul__(self, o):
        return self._mul(o, reflected=True)

    def __truediv__(self, o):
        return self._q / Field._o(o)

    def __rtruediv__(self, o):
        return Field._o(o) / self._q

    def __neg__(self):
        return -self._q

    # -- indexing / coercion guard (decision 2) ------------------------
    def __getitem__(self, idx):
        raise TypeError(
            "Indexing a Field is ambiguous (DOF space vs. quadrature space). "
            "Use `.nodal` for the DOF-space array (boundary/observation ops), "
            "or index the quadrature value explicitly (e.g. `(c * 1.0)[idx]`)."
        )

    def __len__(self):
        raise TypeError(
            "len(Field) is ambiguous (DOF space vs. quadrature space). "
            "Use `len(field.nodal)` for the DOF count, or take the length of "
            "the explicit quadrature value (e.g. `len(field * 1.0)`)."
        )

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Field(nodal_shape={tuple(self.nodal.shape)})"


# -- pytree registration -----------------------------------------------------
# Leaf = nodal (so grad/jvp/vmap flow through DOF space); static aux = the
# cached Operators instance (never crosses the id(problem) jit cache key).
def _field_flatten(f: Field):
    return (f.nodal,), (f._ops,)


def _field_unflatten(aux, children):
    obj = Field.__new__(Field)
    obj.nodal = children[0]
    obj._ops = aux[0]
    return obj


tree_util.register_pytree_node(Field, _field_flatten, _field_unflatten)
