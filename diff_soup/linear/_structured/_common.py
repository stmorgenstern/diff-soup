"""Shared machinery of the pure-JAX structured backends.

A structured backend whose factor is a pytree of arrays (the block-tridiagonal
chain, the member-block Schur factor) implements two functions on one lane:

- ``_factor_tree(analysis, values, precision) -> (tree, status)``: the
  factor of the assembled operator ``A_bc`` as a pytree of arrays and an
  ``int32`` status;
- ``_apply_tree(analysis, values, tree, status, columns, transpose) -> (x,
  status)``: the solution of ``A_bc x = b`` (``A_bc^T x = b`` with
  ``transpose``) for right-hand-side columns ``(n, m)``.

[ArrayStructuredSolver][diff_soup.linear._structured._common.ArrayStructuredSolver]
turns them into the contract hooks: the handle is the flattened tree followed
by the status as an ``int32`` scalar leaf, a factor of either orientation
holds the factor of ``A_bc`` (its transposed solves come from the same
factor), a batch of lanes is one ``vmap`` of the lane functions, and a
refactor runs one conditional on whether any lane asks and then selects per
lane.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup.linear._api import LinearSolver
from diff_soup.linear._factor import Analysis, SolveStatus, effective_transpose, rung_dtype

__all__ = ["ArrayStructuredSolver", "index_constant", "lanes", "refuse_options"]

_SUCCESS = int(SolveStatus.SUCCESS)


def lanes(fn, batched: bool, *args):
    """``fn`` on unbatched arguments, or mapped over their shared leading axis."""
    return jax.vmap(fn)(*args) if batched else fn(*args)


def refuse_options(name: str, options: tuple) -> None:
    """Raise unless a backend that takes no options received none.

    Raises:
        ConfigurationError: If ``options`` is not empty.
    """
    if options:
        raise ConfigurationError(
            f"{name} takes no backend options and requires options=None; got "
            f"{[key for key, _ in options]}."
        )


@functools.lru_cache(maxsize=256)
def _tree_spec(backend: Any, analysis: Analysis, precision: str) -> tuple:
    """``(treedef, leaf avals)`` of a backend's factor tree for one lane (cached)."""
    lane = ShapedArray((analysis.nnz,), rung_dtype(precision))
    tree, _ = jax.eval_shape(lambda v: backend._factor_tree(analysis, v, precision), lane)
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return treedef, tuple(ShapedArray(tuple(leaf.shape), leaf.dtype) for leaf in leaves)


def _select(predicate: Array, new: Array, old: Array) -> Array:
    """``new`` where the per-lane ``predicate`` holds, else ``old`` (broadcast over the leaf)."""
    mask = jnp.reshape(predicate, predicate.shape + (1,) * (new.ndim - predicate.ndim))
    return jnp.where(mask, new, old)


class ArrayStructuredSolver(LinearSolver):
    """A pure-JAX structured backend whose factor is a pytree of arrays.

    Subclasses provide ``_analyze``, ``_factor_tree`` and ``_apply_tree``
    (see the module docstring). The factor of either orientation holds the
    factor of ``A_bc``; solves apply it or its transpose as the orientation
    and the ``transpose`` flag ask, so one factorization serves both
    orientations and ``transpose_is_free`` holds. Values are factored at
    double precision only.
    """

    factor_residency = "arrays"
    transpose_is_free = True
    vmap_factors = "native"
    supported_precisions = ("fp64",)
    is_exact = True
    accepts_operator = False

    # ---- lane functions (subclasses) -----------------------------------------
    def _factor_tree(self, analysis: Analysis, values: Array, precision: str) -> tuple:
        """The factor tree and ``int32`` status of one lane (subclass hook).

        Raises:
            NotImplementedError: On a subclass that does not define it.
        """
        raise NotImplementedError(f"{type(self).__name__} defines no _factor_tree.")

    def _apply_tree(
        self,
        analysis: Analysis,
        values: Array,
        tree: Any,
        status: Array,
        columns: Array,
        transpose: bool,
    ) -> tuple:
        """The solution columns and ``int32`` status of one lane (subclass hook).

        Raises:
            NotImplementedError: On a subclass that does not define it.
        """
        raise NotImplementedError(f"{type(self).__name__} defines no _apply_tree.")

    # ---- contract hooks --------------------------------------------------------
    def factor_avals(
        self, analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
    ) -> tuple:
        """The factor tree's leaves followed by the ``int32`` status leaf."""
        del values_aval, orientation, tol
        _, avals = _tree_spec(self, analysis, precision)
        return (*avals, ShapedArray((), jnp.int32))

    def empty_handle(
        self,
        analysis: Analysis,
        values_aval: Any,
        *,
        context_id: int,
        orientation: str,
        precision: str,
        tol: Any,
    ) -> tuple:
        """Zero leaves and the status leaf ``SolveStatus.EMPTY``; the context id is unused."""
        del context_id
        avals = self.factor_avals(
            analysis, values_aval, orientation=orientation, precision=precision, tol=tol
        )
        leaves = [jnp.zeros(aval.shape, aval.dtype) for aval in avals[:-1]]
        return (*leaves, jnp.asarray(int(SolveStatus.EMPTY), jnp.int32))

    def factor_fun(
        self,
        values: Array,
        token: Array,
        context_id: Array,
        *,
        order: Array,
        analysis: Analysis,
        orientation: str,
        precision: str,
        tol: Any,
    ) -> tuple:
        """Factor ``A_bc`` lane by lane; ``NONFINITE`` when a value is not finite.

        The slot token and the context id are unused and the order word is
        returned unchanged: the factor is its own arrays. Accepts one leading
        lane axis on ``values``.

        Returns:
            ``(*tree_leaves, status_leaf, status, order)``.
        """
        del token, context_id, orientation, tol
        dtype = rung_dtype(precision)

        def one(v):
            tree, status = self._factor_tree(analysis, v.astype(dtype), precision)
            status = jnp.where(jnp.all(jnp.isfinite(v)), status, int(SolveStatus.NONFINITE)).astype(
                jnp.int32
            )
            return (*jax.tree_util.tree_leaves(tree), status)

        outs = lanes(one, values.ndim == 2, values)
        return (*outs, outs[-1], order)

    def apply_fun(
        self,
        values: Array,
        *handle: Array,
        order: Array,
        b: Array,
        analysis: Analysis,
        orientation: str,
        transpose: bool,
        precision: str,
        tol: Any,
    ) -> tuple:
        """Solve with the held factor; ``nan`` wherever the status is not ``SUCCESS``.

        ``b`` is ``(n,)`` or ``(nrhs, n)`` per lane, already in the result
        dtype. The factor holds ``A_bc``, so a factor of orientation ``"AT"``
        applies the transpose of what a factor of ``"A"`` applies. The order
        word is returned unchanged.

        Returns:
            ``(x, status, order)``.
        """
        del tol
        treedef, _ = _tree_spec(self, analysis, precision)
        flip = effective_transpose(orientation, transpose)
        batched = values.ndim == 2

        def one(v, leaves, status, rhs):
            many = rhs.ndim == 2
            columns = rhs.T if many else rhs[:, None]
            tree = jax.tree_util.tree_unflatten(treedef, list(leaves))
            x, status = self._apply_tree(analysis, v.astype(rhs.dtype), tree, status, columns, flip)
            x = x.T if many else x[:, 0]
            x = jnp.where(status == _SUCCESS, x, jnp.nan).astype(rhs.dtype)
            return x, status.astype(jnp.int32)

        leaves, status = tuple(handle[:-1]), handle[-1]
        if batched and b.ndim == 1:
            b = jnp.broadcast_to(b, (values.shape[0],) + b.shape)
        return (*lanes(one, batched, values, leaves, status, b), order)

    def refactor_fun(
        self,
        values: Array,
        held_values: Array,
        predicate: Array,
        *handle: Array,
        order: Array,
        analysis: Analysis,
        orientation: str,
        precision: str,
        tol: Any,
        adopt: bool,
    ) -> tuple:
        """Refactor only when some lane asks, then keep each lane's own choice.

        One ``lax.cond`` on ``any(predicate)`` skips the factorization when
        no lane needs it; a per-lane select keeps the held leaves, values and
        status (``SolveStatus.EMPTY`` for an empty lane) where the predicate
        is false. ``adopt`` has nothing to select and the order word is
        returned unchanged: the factor is its own arrays.

        Returns:
            ``(kept_values, *handle, status, order)``.
        """
        del adopt
        predicate = jnp.asarray(predicate, dtype=jnp.bool_)
        kept_values = _select(predicate, values, held_values)
        token = jnp.zeros((), jnp.int32)
        context_id = jnp.asarray(-1, jnp.int32)

        def rebuild(_):
            outs = self.factor_fun(
                values,
                token,
                context_id,
                order=order,
                analysis=analysis,
                orientation=orientation,
                precision=precision,
                tol=tol,
            )
            return tuple(outs[:-2])

        def keep(_):
            return tuple(handle)

        new = jax.lax.cond(jnp.any(predicate), rebuild, keep, None)
        leaves = tuple(_select(predicate, n, o) for n, o in zip(new, handle))
        return (kept_values, *leaves, leaves[-1].astype(jnp.int32), order)


def index_constant(array: Any) -> np.ndarray:
    """A read-only contiguous ``int32`` host copy of an index array (a program constant)."""
    out = np.ascontiguousarray(np.asarray(array, dtype=np.int32))
    out.flags.writeable = False
    return out
