"""The mass operator: the linear map applied to ``ydot``.

A [MassOperator][diff_soup.problem.implicit.operator.MassOperator] describes ``D`` in the mass-form residual
``F = D(t, y, theta, control) @ ydot - f(t, y, theta, control)``. It may be
identity, diagonal, sparse, block sparse, matrix-free, or time-, state-,
parameter-, or control-dependent — one interface, one solver semantics.

Contract:

1. ``apply`` is exactly linear in its final ``ydot`` argument, even when the
   coefficients depend on the other inputs.
2. ``coefficient_sparsity`` contains every position where ``D`` can be
   nonzero, for any admissible input.
3. When the coefficients depend on the state, ``state_action_sparsity`` is
   required and contains every possible nonzero of ``d(D @ ydot)/dy``.
4. A ``False`` dependency flag is an assertion of invariance the
   implementation must honor; caches rely on it.
5. The optional value callbacks are exact accelerators. Omitting one selects
   automatic differentiation or operator application — it never weakens the
   mathematical derivative.

All differentiable operator data enters through ``theta`` or ``control``: a
values callback reads them explicitly, so parameter and control gradients
flow through the operator coefficients. Arrays passed to the factories are
model constants.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.linear import SparseTopology
from diff_soup.problem.implicit._coo import coo_matvec, coo_rmatvec
from diff_soup.problem.implicit.structure import Dependencies

__all__ = ["MassOperator"]

_CONSTANT = Dependencies(time=False, state=False, parameters=False, control=False)


def _diagonal_topology(n: int) -> SparseTopology:
    """The full-diagonal pattern on ``n`` unknowns."""
    idx = np.arange(n, dtype=np.int32)
    return SparseTopology(rows=idx, cols=idx, n_dofs=n)


def _validate_pattern(rows: np.ndarray, cols: np.ndarray, n: int, what: str) -> None:
    """Reject malformed or out-of-range COO index arrays."""
    if rows.shape != cols.shape or rows.ndim != 1:
        raise ValueError(
            f"{what}: rows and cols must be equal-length 1-D index arrays; "
            f"got shapes {rows.shape} and {cols.shape}."
        )
    if rows.size and (rows.min() < 0 or rows.max() >= n or cols.min() < 0 or cols.max() >= n):
        raise ValueError(
            f"{what}: indices must lie in [0, {n}); got row range "
            f"[{rows.min()}, {rows.max()}] and column range [{cols.min()}, {cols.max()}]."
        )


class MassOperator(eqx.Module):
    """Mass matrix `M(t, y)` applied to a time derivative, `M @ ydot`.

    Build one with a factory. Inside a residual-form equation, `op(q.dt("u"))`
    applies it to a field's derivative, and a
    [`MassProblem`][diff_soup.problem.MassProblem] takes one as its `mass`.

    Attributes:
        apply: Function `(t, y, theta, control, ydot)` returning `M @ ydot`.
        coefficient_sparsity: [`SparseTopology`][diff_soup.linear.SparseTopology] of the possibly nonzero entries of `M`.
        dependencies: [`Dependencies`][diff_soup.problem.Dependencies] naming the inputs the entries vary with.
        state_action_sparsity: Support of `d(M @ ydot)/dy`, required when the entries depend on the state.
        state_action_values: Values of `d(M @ ydot)/dy` on that support, or `None`.
        coefficient_values: Function `(t, y, theta, control)` returning the entries on `coefficient_sparsity`, or `None`.
        transpose_apply: Function `(t, y, theta, control, w)` returning `M.T @ w`, or `None`.
    """

    apply: Callable = eqx.field(static=True)
    coefficient_sparsity: SparseTopology = eqx.field(static=True)
    dependencies: Dependencies = eqx.field(static=True)
    state_action_sparsity: Optional[SparseTopology] = eqx.field(static=True, default=None)
    coefficient_values: Optional[Callable] = eqx.field(static=True, default=None)
    state_action_values: Optional[Callable] = eqx.field(static=True, default=None)
    transpose_apply: Optional[Callable] = eqx.field(static=True, default=None)

    def __post_init__(self):
        """Enforce the state-dependence contract at construction."""
        if self.dependencies.state and self.state_action_sparsity is None:
            raise ValueError(
                "a state-dependent mass operator (dependencies.state=True) must "
                "declare state_action_sparsity — the support of d(D @ ydot)/dy over "
                "admissible ydot. Declare the pattern, or set dependencies.state=False "
                "if the coefficients are state-invariant."
            )

    @property
    def n(self) -> int:
        """The operator's side length."""
        return int(self.coefficient_sparsity.n_dofs)

    def __call__(self, expr):
        """Apply the operator to a derivative expression (``op(q.dt(field))``).

        Returns a derivative-expression node so residual-form equations can
        state ``op(q.dt(field)) - rhs``. Only derivative expressions are
        accepted here; numeric application uses
        [apply][diff_soup.problem.implicit.operator.MassOperator.apply].
        """
        from diff_soup.problem._spec.derivative_expr import (
            DerivativeApply,
            DerivativeExpr,
        )

        if not isinstance(expr, DerivativeExpr):
            raise TypeError(
                "a MassOperator is called on a derivative expression "
                "(for example op(q.dt('u'))); for numeric application use "
                f"operator.apply(...). Got {type(expr).__name__}."
            )
        return DerivativeApply(self, expr)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @staticmethod
    def identity(n: int) -> "MassOperator":
        """Return the constant `n`-by-`n` identity.

        Args:
            n: Side length.
        """
        topo = _diagonal_topology(int(n))

        def apply(t, y, theta, control, ydot):
            del t, y, theta, control
            return jnp.asarray(ydot)

        def values(t, y, theta, control):
            del t, theta, control
            return jnp.ones((topo.n_dofs,), dtype=jnp.asarray(y).dtype)

        return MassOperator(
            apply=apply,
            coefficient_sparsity=topo,
            dependencies=_CONSTANT,
            coefficient_values=values,
            transpose_apply=apply,
        )

    @staticmethod
    def diagonal(
        values_or_callback,
        *,
        n: Optional[int] = None,
        dependencies: Optional[Dependencies] = None,
        state_action_sparsity: Optional[SparseTopology] = None,
        state_action_values: Optional[Callable] = None,
    ) -> "MassOperator":
        """Return a diagonal operator from a vector of entries or a callback.

        A callback needs `n` and `dependencies`. A vector is constant unless
        `dependencies` says otherwise.

        Args:
            values_or_callback: 1-D array of diagonal entries, or a callback `(t, y, theta, control)` returning them.
            n: Side length, required with a callback.
            dependencies: Inputs the entries vary with, required with a callback.
            state_action_sparsity: Support of `d(M @ ydot)/dy`, required when the entries depend on the state.
            state_action_values: Values of `d(M @ ydot)/dy` on that support, or `None`.
        """
        if callable(values_or_callback):
            if n is None:
                raise ValueError("diagonal(callback) requires n= (the side length).")
            if dependencies is None:
                raise ValueError(
                    "diagonal(callback) requires dependencies= — a values callback "
                    "must declare which solve-time inputs its values vary with."
                )
            diag_values = values_or_callback
            n = int(n)
        else:
            const = jnp.asarray(values_or_callback)
            if const.ndim != 1:
                raise ValueError(
                    f"diagonal expects a 1-D vector of diagonal entries; got shape {const.shape}."
                )
            if n is not None and int(n) != const.shape[0]:
                raise ValueError(
                    f"diagonal: n={n} disagrees with the {const.shape[0]}-entry vector."
                )
            n = int(const.shape[0])
            if dependencies is None:
                dependencies = _CONSTANT

            def diag_values(t, y, theta, control, _const=const):
                del t, y, theta, control
                return _const

        topo = _diagonal_topology(n)

        def apply(t, y, theta, control, ydot):
            return diag_values(t, y, theta, control) * jnp.asarray(ydot)

        return MassOperator(
            apply=apply,
            coefficient_sparsity=topo,
            dependencies=dependencies,
            state_action_sparsity=state_action_sparsity,
            coefficient_values=lambda t, y, theta, control: jnp.asarray(
                diag_values(t, y, theta, control)
            ),
            state_action_values=state_action_values,
            transpose_apply=apply,
        )

    @staticmethod
    def sparse(
        rows,
        cols,
        values_or_callback,
        *,
        n: int,
        dependencies: Optional[Dependencies] = None,
        state_action_sparsity: Optional[SparseTopology] = None,
        state_action_values: Optional[Callable] = None,
    ) -> "MassOperator":
        """Return an operator with entries at given `(row, col)` positions.

        Args:
            rows: Row index of each entry, in `[0, n)`.
            cols: Column index of each entry, in `[0, n)`, the same length as `rows`.
            values_or_callback: One value per entry, or a callback `(t, y, theta, control)` returning them.
            n: Side length.
            dependencies: Inputs the entries vary with, required with a callback.
            state_action_sparsity: Support of `d(M @ ydot)/dy`, required when the entries depend on the state.
            state_action_values: Values of `d(M @ ydot)/dy` on that support, or `None`.
        """
        n = int(n)
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        _validate_pattern(rows, cols, n, "MassOperator.sparse")
        topo = SparseTopology(rows=rows, cols=cols, n_dofs=n)

        if callable(values_or_callback):
            if dependencies is None:
                raise ValueError(
                    "sparse(callback) requires dependencies= — a values callback "
                    "must declare which solve-time inputs its values vary with."
                )
            entry_values = values_or_callback
        else:
            const = jnp.asarray(values_or_callback)
            if const.shape != rows.shape:
                raise ValueError(
                    f"sparse: expected {rows.shape[0]} values (one per support entry); "
                    f"got shape {const.shape}."
                )
            if dependencies is None:
                dependencies = _CONSTANT

            def entry_values(t, y, theta, control, _const=const):
                del t, y, theta, control
                return _const

        def apply(t, y, theta, control, ydot):
            return coo_matvec(entry_values(t, y, theta, control), rows, cols, n, ydot)

        def transpose_apply(t, y, theta, control, w):
            return coo_rmatvec(entry_values(t, y, theta, control), rows, cols, n, w)

        return MassOperator(
            apply=apply,
            coefficient_sparsity=topo,
            dependencies=dependencies,
            state_action_sparsity=state_action_sparsity,
            coefficient_values=lambda t, y, theta, control: jnp.asarray(
                entry_values(t, y, theta, control)
            ),
            state_action_values=state_action_values,
            transpose_apply=transpose_apply,
        )

    @staticmethod
    def block_sparse(
        blocks: Sequence = (),
        couplings: Sequence = (),
        *,
        n: int,
        dependencies: Optional[Dependencies] = None,
        state_action_sparsity: Optional[SparseTopology] = None,
        state_action_values: Optional[Callable] = None,
    ) -> "MassOperator":
        """Return a sparse operator assembled from dense sub-blocks and single-row couplings.

        Pass at least one block or coupling. Any callback entry needs `dependencies`.

        Args:
            blocks: Triples `(rows, cols, values)`, each a dense sub-block with values of shape `(len(rows), len(cols))` or a callback.
            couplings: Triples `(row, cols, weights)`, each one row reading `cols` with weights of shape `(len(cols),)` or a callback.
            n: Side length.
            dependencies: Inputs the entries vary with, required when any entry is a callback.
            state_action_sparsity: Support of `d(M @ ydot)/dy`, required when the entries depend on the state.
            state_action_values: Values of `d(M @ ydot)/dy` on that support, or `None`.
        """
        n = int(n)
        pattern_rows: list[np.ndarray] = []
        pattern_cols: list[np.ndarray] = []
        pieces: list = []  # (is_callable, value_or_fn, flat_size)
        any_callback = False

        for k, (b_rows, b_cols, b_vals) in enumerate(blocks):
            b_rows = np.asarray(b_rows, dtype=np.int32)
            b_cols = np.asarray(b_cols, dtype=np.int32)
            grid_rows = np.repeat(b_rows, b_cols.size)
            grid_cols = np.tile(b_cols, b_rows.size)
            pattern_rows.append(grid_rows)
            pattern_cols.append(grid_cols)
            shape = (b_rows.size, b_cols.size)
            if callable(b_vals):
                any_callback = True
                pieces.append((True, b_vals, shape))
            else:
                const = jnp.asarray(b_vals)
                if const.shape != shape:
                    raise ValueError(
                        f"block_sparse: block {k} declares a {shape} sub-block but its "
                        f"values have shape {const.shape}."
                    )
                pieces.append((False, const, shape))

        for k, (c_row, c_cols, c_weights) in enumerate(couplings):
            c_cols = np.asarray(c_cols, dtype=np.int32)
            pattern_rows.append(np.full(c_cols.shape, int(c_row), dtype=np.int32))
            pattern_cols.append(c_cols)
            shape = (c_cols.size,)
            if callable(c_weights):
                any_callback = True
                pieces.append((True, c_weights, shape))
            else:
                const = jnp.asarray(c_weights)
                if const.shape != shape:
                    raise ValueError(
                        f"block_sparse: coupling {k} reads {shape[0]} columns but its "
                        f"weights have shape {const.shape}."
                    )
                pieces.append((False, const, shape))

        if not pieces:
            raise ValueError("block_sparse needs at least one block or coupling.")
        rows = np.concatenate(pattern_rows)
        cols = np.concatenate(pattern_cols)
        if any_callback and dependencies is None:
            raise ValueError(
                "block_sparse with callback-valued entries requires dependencies= — "
                "a values callback must declare which solve-time inputs it varies with."
            )

        def entry_values(t, y, theta, control):
            parts = []
            for is_fn, val, _shape in pieces:
                piece = val(t, y, theta, control) if is_fn else val
                parts.append(jnp.reshape(jnp.asarray(piece), (-1,)))
            return jnp.concatenate(parts)

        return MassOperator.sparse(
            rows,
            cols,
            entry_values,
            n=n,
            dependencies=dependencies if dependencies is not None else _CONSTANT,
            state_action_sparsity=state_action_sparsity,
            state_action_values=state_action_values,
        )

    @staticmethod
    def from_apply(
        apply: Callable,
        *,
        coefficient_sparsity: SparseTopology,
        dependencies: Dependencies,
        state_action_sparsity: Optional[SparseTopology] = None,
        state_action_values: Optional[Callable] = None,
        transpose_apply: Optional[Callable] = None,
    ) -> "MassOperator":
        """Return a matrix-free operator from its action `apply(t, y, theta, control, ydot)`.

        Args:
            apply: Function `(t, y, theta, control, ydot)` returning `M @ ydot`.
            coefficient_sparsity: [`SparseTopology`][diff_soup.linear.SparseTopology] of the possibly nonzero entries of `M`.
            dependencies: Inputs the entries vary with.
            state_action_sparsity: Support of `d(M @ ydot)/dy`, required when the entries depend on the state.
            state_action_values: Values of `d(M @ ydot)/dy` on that support, or `None`.
            transpose_apply: Function `(t, y, theta, control, w)` returning `M.T @ w`, or `None`.
        """
        return MassOperator(
            apply=apply,
            coefficient_sparsity=coefficient_sparsity,
            dependencies=dependencies,
            state_action_sparsity=state_action_sparsity,
            coefficient_values=None,
            state_action_values=state_action_values,
            transpose_apply=transpose_apply,
        )

    # ------------------------------------------------------------------
    # Reference materialization (validation / dense fallbacks)
    # ------------------------------------------------------------------
    def dense(self, t, y, theta, control) -> jax.Array:
        """Return `M` as a dense `n`-by-`n` array at the given inputs.

        Args:
            t: Time.
            y: State vector.
            theta: Parameters.
            control: Control values.

        Returns:
            The dense matrix.
        """
        n = self.n
        eye = jnp.eye(n, dtype=jnp.asarray(y).dtype)
        cols = jax.vmap(lambda e: self.apply(t, y, theta, control, e))(eye)
        return jnp.transpose(cols)
