"""Dense CPU/GPU linear backend.

Materialises the sparse pattern as a full ``(n, n)`` matrix and holds its
pivoted LU as the factor of the [analyze][diff_soup.linear.analyze] /
[factor][diff_soup.linear.factor] / [solve][diff_soup.linear.solve]
contract. Intended for small systems (the automatic resolver picks it for at
most ten rows), unit tests, diagnostics, and environments without a sparse
backend.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup.linear._api import LinearSolver, SparseTopology
from diff_soup.linear._factor import (
    Analysis,
    SolveStatus,
    dirichlet_rows,
    rung_dtype,
    solve_dtype,
)


class Dense(LinearSolver):
    """A dense LU in JAX, on the CPU and on GPUs.

    It factors float64 and float32 matrices, real or complex. `linear="dense"` selects it, and
    `"auto"` picks it for systems of at most ten unknowns. Its factor needs no `SolveContext`.
    """

    backend_name = "dense"

    factor_residency = "arrays"
    transpose_is_free = True
    vmap_factors = "native"
    supported_precisions = ("fp64", "fp32")
    is_exact = True
    accepts_operator = False

    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> None:
        """Dense needs no symbolic work; it accepts no backend options.

        Raises:
            ConfigurationError: If ``options`` is not empty.
        """
        if options:
            raise ConfigurationError(
                f"Dense takes no backend options and requires options=None; got "
                f"{[key for key, _ in options]}."
            )
        return None

    def factor_avals(
        self, analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
    ) -> tuple:
        """The handle ``(lu, piv)``: ``(n, n)`` in the rung dtype and ``(n,)`` int32."""
        n = analysis.n
        return (
            jax.core.ShapedArray((n, n), rung_dtype(precision, values_aval.dtype)),
            jax.core.ShapedArray((n,), jnp.int32),
        )

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
        """The handle of an empty factor: a zero LU and pivots filled with ``-1``.

        A real pivot lies in ``[0, n)``, so a first pivot of ``-1`` marks the
        lane empty: its solves return ``nan`` with ``SolveStatus.EMPTY``. The
        context id is unused (the factor is its own arrays).

        Returns:
            ``(lu, piv)`` with the shapes and dtypes of ``factor_avals``.
        """
        del context_id
        lu, piv = self.factor_avals(
            analysis, values_aval, orientation=orientation, precision=precision, tol=tol
        )
        return jnp.zeros(lu.shape, lu.dtype), jnp.full(piv.shape, -1, piv.dtype)

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
        """Assemble ``M``, apply the Dirichlet rows and LU-factor it at the rung dtype.

        The status is decided from the values and the pivots, the same way
        on every platform (a GPU LU of an exactly singular matrix leaves a
        rounding remainder where LAPACK on the CPU leaves an exact zero, and
        may drop a ``nan`` of the input):

        - ``NONFINITE`` when a value or an entry of the LU is not finite;
        - else ``SINGULAR`` when a pivot is exactly zero or, for a
          double-precision LU, at most 16 unit roundoffs of the largest
          magnitude in both its row and its column of ``M``, which is the
          rounding remainder an exactly dependent row or column leaves;
        - else ``SUCCESS``.

        A single-precision LU takes only the zero and non-finite conditions,
        as on the cuDSS backend: at float32 the pivots of regular and of
        singular matrices overlap, and refinement against the double-precision
        residual decides whether such a factor is accurate enough. The
        verdict is written into the handle (a non-finite factor is filled
        with ``nan``, a singular pivot is set to zero), so every later solve
        and refactor reads the same status from the handle alone. Accepts one
        leading batch axis on ``values``. The factor is its own arrays, so the
        slot token and the context id are unused and the order word is
        returned unchanged.

        Returns:
            ``(lu, piv, status, order)``.
        """
        del token, context_id, tol

        def one(v):
            matrix = _assembled(analysis, v, rung_dtype(precision, v.dtype), orientation)
            lu, piv = jax.scipy.linalg.lu_factor(matrix)
            piv = piv.astype(jnp.int32)
            lu = _with_verdict(matrix, lu, piv, jnp.all(jnp.isfinite(v)))
            return lu, piv, _lu_status(lu)

        return (*_lanes(one, values.ndim == 2, values), order)

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
        """Solve with the held LU (``trans=1`` for ``transpose``); ``nan`` unless the LU is valid.

        The LU is already of the oriented operator ``M``, so ``transpose``
        alone selects ``lu_solve``'s ``trans``. The solve runs in the result
        dtype, the rung's dtype (complex of the same width when the LU or
        ``b`` is complex), and returns it; ``b`` arrives in that dtype. With
        a batched handle ``b`` carries the same leading axis; an unbatched
        ``(n,)`` ``b`` is broadcast. An empty lane (pivots ``-1``) returns
        ``nan`` with ``SolveStatus.EMPTY``. The order word is returned
        unchanged.

        Returns:
            ``(x, status, order)``.
        """
        del values, analysis, orientation, tol
        lu, piv = handle
        trans = int(bool(transpose))

        def one(lu, piv, rhs):
            status = _handle_status(lu, piv)
            # An empty lane's pivots are not a permutation; solve with the
            # identity one and discard the result below.
            piv = jnp.where(piv < 0, jnp.arange(piv.shape[-1], dtype=piv.dtype), piv)
            # The LU is complex exactly when the values are, so this is the
            # result dtype of the values and b at this rung.
            work = solve_dtype(precision, lu.dtype, rhs.dtype)
            many = rhs.ndim == 2
            columns = (rhs.T if many else rhs).astype(work)
            x = jax.scipy.linalg.lu_solve((lu.astype(work), piv), columns, trans=trans)
            x = x.T if many else x
            return jnp.where(status == int(SolveStatus.SUCCESS), x, jnp.nan), status

        if lu.ndim == 3 and b.ndim == 1:
            b = jnp.broadcast_to(b, (lu.shape[0],) + b.shape)
        return (*_lanes(one, lu.ndim == 3, lu, piv, b), order)

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
        no lane needs it; a per-lane select then keeps the held LU, the held
        values and the held status (``SolveStatus.EMPTY`` for an empty lane)
        on lanes whose predicate is false. Accepts one leading batch axis.
        The factors are arrays, so ``adopt`` has nothing to select and is
        accepted without effect, and the order word is returned unchanged.

        Returns:
            ``(kept_values, lu, piv, status, order)``.
        """
        del adopt
        lu, piv = handle
        predicate = jnp.asarray(predicate, dtype=jnp.bool_)
        kept_values = jnp.where(predicate[..., None], values, held_values)
        held_status = _lanes(_handle_status, lu.ndim == 3, lu, piv)
        token = jnp.zeros((), jnp.int32)
        context_id = jnp.asarray(-1, jnp.int32)

        def rebuild(_):
            new_lu, new_piv, new_status, _ = self.factor_fun(
                values,
                token,
                context_id,
                order=order,
                analysis=analysis,
                orientation=orientation,
                precision=precision,
                tol=tol,
            )
            return new_lu, new_piv, new_status

        def keep(_):
            return lu, piv, held_status

        new_lu, new_piv, new_status = jax.lax.cond(jnp.any(predicate), rebuild, keep, None)
        return (
            kept_values,
            jnp.where(predicate[..., None, None], new_lu, lu),
            jnp.where(predicate[..., None], new_piv, piv),
            jnp.where(predicate, new_status, held_status).astype(jnp.int32),
            order,
        )


def _assembled(analysis: Analysis, values: Array, dtype: Any, orientation: str) -> Array:
    """The dense oriented operator ``M`` with the topology's Dirichlet rows applied.

    Duplicate pattern entries add. Every entry in a Dirichlet row is
    multiplied by zero and ``1`` is added at the row's diagonal when the
    pattern holds it, exactly as the sparse backends assemble ``A_bc`` (so a
    non-finite value stays visible in the status even in a Dirichlet row).
    ``M`` is ``A_bc`` for orientation ``"A"`` and ``A_bc^T`` for ``"AT"``.
    """
    topology = analysis.topology
    n = analysis.n
    coef = values.astype(dtype)
    row_mask, diagonal = dirichlet_rows(topology)
    if row_mask is not None:
        coef = coef * np.where(row_mask, 0.0, 1.0).astype(dtype)
    matrix = jnp.zeros((n, n), dtype).at[topology.rows, topology.cols].add(coef)
    if diagonal.size:
        matrix = matrix.at[diagonal, diagonal].add(jnp.ones((), dtype))
    return matrix.T if orientation == "AT" else matrix


#: A double-precision pivot at most this many unit roundoffs of the largest
#: magnitude in its row and in its column of the factored matrix marks the
#: factor singular (the cuDSS backend uses the same margin on its pivots).
_SINGULAR_PIVOT_ROUNDOFFS = 16.0


def _with_verdict(matrix: Array, lu: Array, piv: Array, values_finite: Array) -> Array:
    """``lu`` with the factorization's verdict written in (see ``Dense.factor_fun``).

    Fills a factor of non-finite values with ``nan`` and sets every pivot
    the singularity rule flags to zero, so the status read back from the
    handle is the status of the factorization on every platform.
    """
    finite = values_finite & jnp.all(jnp.isfinite(lu))
    diagonal = jnp.diagonal(lu)
    singular = diagonal == 0
    if jnp.finfo(lu.dtype).bits >= 64:
        n = matrix.shape[-1]
        magnitude = jnp.abs(matrix)
        rows = jax.lax.linalg.lu_pivots_to_permutation(piv, n)
        row_scale = jnp.max(magnitude, axis=1)[rows]
        column_scale = jnp.max(magnitude, axis=0)
        roundoff = _SINGULAR_PIVOT_ROUNDOFFS * jnp.finfo(lu.dtype).eps / 2
        bound = roundoff * jnp.minimum(row_scale, column_scale)
        singular = singular | (jnp.abs(diagonal) <= bound)
    index = jnp.arange(lu.shape[-1])
    lu = lu.at[index, index].set(jnp.where(singular, jnp.zeros((), lu.dtype), diagonal))
    return jnp.where(finite, lu, jnp.full_like(lu, jnp.nan))


def _lu_status(lu: Array) -> Array:
    """``NONFINITE`` for a non-finite LU, ``SINGULAR`` for a zero pivot, else ``SUCCESS``."""
    singular = jnp.where(
        jnp.any(jnp.diagonal(lu, axis1=-2, axis2=-1) == 0),
        int(SolveStatus.SINGULAR),
        int(SolveStatus.SUCCESS),
    )
    return jnp.where(jnp.all(jnp.isfinite(lu)), singular, int(SolveStatus.NONFINITE)).astype(
        jnp.int32
    )


def _handle_status(lu: Array, piv: Array) -> Array:
    """``EMPTY`` for an empty handle (first pivot ``-1``), else the status of the LU."""
    empty = jnp.any(piv[..., :1] < 0)
    return jnp.where(empty, int(SolveStatus.EMPTY), _lu_status(lu)).astype(jnp.int32)


def _lanes(fn, batched: bool, *args):
    """``fn`` on unbatched arguments, or mapped over their shared leading axis."""
    return jax.vmap(fn)(*args) if batched else fn(*args)


# Public alias of ``Dense``.
DenseBackend = Dense

__all__ = ["Dense", "DenseBackend"]
