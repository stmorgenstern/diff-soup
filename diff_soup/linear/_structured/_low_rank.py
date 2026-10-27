"""Low-rank terms of a sparse operator and the Woodbury backend.

An operator with a low-rank term is

    A = A_b + U W^T,

where ``A_b`` is the sparse matrix of a base
[SparseTopology][diff_soup.linear.SparseTopology] and ``U``, ``W`` are
``(n, k)`` matrices with one column per coupling. ``U`` is numeric (its
entries are operator values, they change with every linearization); ``W`` is
structural (its weights are fixed when the term is declared; for the mean of
a group of unknowns they are ``1 / size`` on the group). A
[LowRankTerm][diff_soup.linear.LowRankTerm] records the supports of ``U`` and
``W`` and the weights of ``W``.

Values layout. The value vector of such an operator is the base values
followed by the entries of ``U`` in the term's order,

    values = concatenate([base values (nnz_b,), U entries (n_u,)]),

and its topology is [LowRankTerm.topology][diff_soup.linear.LowRankTerm.topology]
of the base: the base entries followed by one entry per entry of ``U``, at
the row of that entry and at the first row of its coupling's ``W`` support
(a position the term reaches). Backends that know the term (``Woodbury``,
``MemberBlock`` with a member term) read the trailing entries as ``U``; every
other backend would read them as plain matrix entries. The family's
``pattern_matvec`` asks these backends for ``(A_b + U W^T) x``, so the
derivative rules (``dx = solve(db - dA x)``, the matrix cotangent) and the
residual of ``solve_checked`` cover the low-rank term. Dirichlet rows are
applied to the whole operator: an entry of ``U`` in a Dirichlet row is
dropped with the row's base entries.

[Woodbury][diff_soup.linear.Woodbury] solves such an operator exactly with a
factor of ``A_b`` from any exact backend of the contract, by the
Sherman-Morrison-Woodbury identity. For orientation ``"A"`` the factor holds
the inner factor of ``B = A_b``, ``P = U`` and ``Q = W``; for ``"AT"`` the
inner factor of ``B = A_b^T``, ``P = W`` and ``Q = U``; either way ``M = B + P
Q^T`` and

    Y = B^-1 P,   C = I + Q^T Y   (k x k, pivoted LU),
    M^-1 b   = y - Y C^-1 Q^T y,         y = B^-1 b,
    M^-T b   = B^-T (b - Q C^-T Y^T b).

A factorization costs one inner factorization, one inner solve with ``k``
right-hand sides and a ``k x k`` LU; every solve, in either direction, costs
one inner solve (transposed for ``M^-T``) and ``O(n k)`` work.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jaxtyping import Array, ArrayLike

from diff_soup._errors import ConfigurationError
from diff_soup._jax_compat import mlir
from diff_soup._layout import Layout
from diff_soup.linear._api import LinearSolver, SparseTopology, analyze
from diff_soup.linear._factor import Analysis, SolveStatus, rung_dtype
from diff_soup.linear._primitive import (
    pattern_matvec,
    phased_factor_p,
    phased_invalidate_p,
    phased_refactor_p,
    phased_solve_p,
)
from diff_soup.linear._structured._common import _select, lanes
from diff_soup.linear._structured._small_lu import SINGULAR_PIVOT_ROUNDOFFS

__all__ = ["LowRankTerm", "Woodbury"]

_SUCCESS = int(SolveStatus.SUCCESS)


def _int_bytes(values: Any) -> tuple[bytes, np.ndarray]:
    """``(contiguous int32 bytes, array)`` of a host index array."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))
    return array.astype(np.int32).tobytes(), array


def _view(data: bytes, dtype: Any) -> np.ndarray:
    """A read-only host view of a bytes buffer."""
    return np.frombuffer(data, dtype=dtype)


class LowRankTerm(eqx.Module):
    """A low-rank term `U W^T` added to a sparse matrix, for the `Woodbury` solver.

    The entries of `U` take their values from the matrix values, after those of the sparse part. The
    entries of `W` have fixed weights.

    Args:
        n_dofs: The number of rows of the matrix.
        rank: The number of couplings, the columns of `U` and `W`.
        u_rows: The row of each entry of `U`.
        u_cols: The coupling of each entry of `U`.
        w_rows: The row of each entry of `W`.
        w_cols: The coupling of each entry of `W`.
        w_values: The weight of each entry of `W`.
    """

    n_dofs: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    _u_rows: bytes = eqx.field(static=True)
    _u_cols: bytes = eqx.field(static=True)
    _w_rows: bytes = eqx.field(static=True)
    _w_cols: bytes = eqx.field(static=True)
    _w_values: bytes = eqx.field(static=True)

    def __init__(
        self,
        *,
        n_dofs: int,
        rank: int,
        u_rows: Any,
        u_cols: Any,
        w_rows: Any,
        w_cols: Any,
        w_values: Any,
    ):
        """Validate and freeze the supports of ``U`` and ``W``."""
        n, k = int(n_dofs), int(rank)
        if n < 1 or k < 1:
            raise ConfigurationError(
                f"LowRankTerm needs n_dofs >= 1 and rank >= 1; got n_dofs={n}, rank={k}."
            )
        self.n_dofs, self.rank = n, k
        self._u_rows, u_r = _int_bytes(u_rows)
        self._u_cols, u_c = _int_bytes(u_cols)
        self._w_rows, w_r = _int_bytes(w_rows)
        self._w_cols, w_c = _int_bytes(w_cols)
        weights = np.ascontiguousarray(np.asarray(w_values, dtype=np.float64).reshape(-1))
        self._w_values = weights.tobytes()
        if u_r.shape != u_c.shape:
            raise ConfigurationError(
                "LowRankTerm needs one coupling per entry of U: u_rows and u_cols have "
                f"{u_r.size} and {u_c.size} entries."
            )
        if not (w_r.shape == w_c.shape == weights.shape):
            raise ConfigurationError(
                "LowRankTerm needs one coupling and one weight per entry of W: w_rows, w_cols and "
                f"w_values have {w_r.size}, {w_c.size} and {weights.size} entries."
            )
        for name, array, bound in (
            ("u_rows", u_r, n),
            ("w_rows", w_r, n),
            ("u_cols", u_c, k),
            ("w_cols", w_c, k),
        ):
            if array.size and (array.min() < 0 or array.max() >= bound):
                raise ConfigurationError(
                    f"LowRankTerm.{name} has an entry outside [0, {bound}); rows must address the "
                    "operator and columns a coupling."
                )
        for name, r, c in (("U", u_r, u_c), ("W", w_r, w_c)):
            keys = c * n + r
            if np.unique(keys).size != keys.size:
                raise ConfigurationError(
                    f"LowRankTerm lists an entry of {name} twice; every (row, coupling) pair of "
                    f"{name} must appear once."
                )
            missing = np.setdiff1d(np.arange(k), c)
            if missing.size:
                raise ConfigurationError(
                    f"LowRankTerm coupling {int(missing[0])} has no entry of {name}; every coupling "
                    "needs at least one entry of U and one of W."
                )
        if not np.all(np.isfinite(weights)):
            raise ConfigurationError("LowRankTerm needs finite weights w_values.")

    # ---- host views ------------------------------------------------------------
    @property
    def u_rows(self) -> np.ndarray:
        """Row of every entry of ``U`` (read-only host int32)."""
        return _view(self._u_rows, np.int32)

    @property
    def u_cols(self) -> np.ndarray:
        """Coupling of every entry of ``U`` (read-only host int32)."""
        return _view(self._u_cols, np.int32)

    @property
    def w_rows(self) -> np.ndarray:
        """Row of every entry of ``W`` (read-only host int32)."""
        return _view(self._w_rows, np.int32)

    @property
    def w_cols(self) -> np.ndarray:
        """Coupling of every entry of ``W`` (read-only host int32)."""
        return _view(self._w_cols, np.int32)

    @property
    def w_values(self) -> np.ndarray:
        """Weight of every entry of ``W`` (read-only host float64)."""
        return _view(self._w_values, np.float64)

    @property
    def n_u(self) -> int:
        """Number of entries of ``U``, the length of the values that follow the base values."""
        return int(self.u_rows.size)

    def __hash__(self) -> int:
        """Hash by content."""
        return hash(
            (
                self.n_dofs,
                self.rank,
                self._u_rows,
                self._u_cols,
                self._w_rows,
                self._w_cols,
                self._w_values,
            )
        )

    def __eq__(self, other: object) -> bool:
        """Equal when every support and weight is equal."""
        if not isinstance(other, LowRankTerm):
            return NotImplemented
        return (
            self.n_dofs == other.n_dofs
            and self.rank == other.rank
            and self._u_rows == other._u_rows
            and self._u_cols == other._u_cols
            and self._w_rows == other._w_rows
            and self._w_cols == other._w_cols
            and self._w_values == other._w_values
        )

    # ---- topology and values -----------------------------------------------------
    def _anchor_cols(self) -> np.ndarray:
        """The column each entry of ``U`` occupies in the operator's topology."""
        first = np.full((self.rank,), self.n_dofs, dtype=np.int64)
        np.minimum.at(first, self.w_cols.astype(np.int64), self.w_rows.astype(np.int64))
        return first[self.u_cols]

    def topology(self, base: SparseTopology) -> SparseTopology:
        """The pattern to analyze, `base` followed by the entries of `U`.

        Args:
            base: The `SparseTopology` of the sparse part.

        Returns:
            The `SparseTopology` to pass to `analyze` with a `Woodbury` backend.
        """
        if not isinstance(base, SparseTopology) or int(base.n_dofs) != self.n_dofs:
            raise ConfigurationError(
                f"LowRankTerm.topology needs the base SparseTopology of size {self.n_dofs}; got "
                f"{base!r}."
            )
        return SparseTopology(
            rows=np.concatenate([base.rows, self.u_rows]),
            cols=np.concatenate([base.cols, self._anchor_cols()]),
            n_dofs=self.n_dofs,
            bc_indices=base.bc_indices,
        )

    def base_topology(
        self, topology: SparseTopology, *, caller: str = "LowRankTerm"
    ) -> SparseTopology:
        """The base topology of a topology built by [topology][diff_soup.linear.LowRankTerm.topology].

        Args:
            topology: A topology of ``A_b + U W^T``.
            caller: The name used in the error message.

        Returns:
            The topology of ``A_b``.

        Raises:
            ConfigurationError: If ``topology`` has another size, or its last
                ``n_u`` entries are not this term's entries of ``U``.
        """
        n_u = self.n_u
        rows, cols = topology.rows, topology.cols
        if int(topology.n_dofs) != self.n_dofs or rows.size < n_u:
            raise ConfigurationError(
                f"{caller} received a topology of size {int(topology.n_dofs)} with {rows.size} "
                f"entries, but its low-rank term has size {self.n_dofs} and {n_u} entries of U; "
                "analyze LowRankTerm.topology(base) of the term the backend holds."
            )
        tail = rows.size - n_u
        if not (
            np.array_equal(rows[tail:], self.u_rows)
            and np.array_equal(cols[tail:], self._anchor_cols())
        ):
            raise ConfigurationError(
                f"{caller} received a topology whose last {n_u} entries are not the entries of U "
                "of its low-rank term; analyze LowRankTerm.topology(base) of the term the backend "
                "holds, with the values of U after the base values."
            )
        return SparseTopology(
            rows=rows[:tail], cols=cols[:tail], n_dofs=self.n_dofs, bc_indices=topology.bc_indices
        )

    def split(self, values: ArrayLike) -> tuple[Array, Array]:
        """``(base values, U values)`` of a value vector ``(..., nnz_b + n_u)``."""
        values = jnp.asarray(values)
        tail = values.shape[-1] - self.n_u
        return values[..., :tail], values[..., tail:]

    def join(self, base_values: ArrayLike, u_values: ArrayLike) -> Array:
        """The value vector ``concatenate([base values, U values])`` along the last axis."""
        return jnp.concatenate([jnp.asarray(base_values), jnp.asarray(u_values)], axis=-1)

    # ---- numeric actions (traceable; leading lane axes allowed) --------------------
    def w_transpose_apply(self, x: ArrayLike) -> Array:
        """``W^T x`` over the last axis: ``(..., n)`` to ``(..., k)``."""
        x = jnp.asarray(x)
        gathered = x[..., self.w_rows] * jnp.asarray(self.w_values, x.dtype)
        out = jnp.zeros(x.shape[:-1] + (self.rank,), gathered.dtype)
        return out.at[..., self.w_cols].add(gathered)

    def w_apply(self, s: ArrayLike) -> Array:
        """``W s`` over the last axis: ``(..., k)`` to ``(..., n)``."""
        s = jnp.asarray(s)
        contribution = s[..., self.w_cols] * jnp.asarray(self.w_values, s.dtype)
        out = jnp.zeros(s.shape[:-1] + (self.n_dofs,), contribution.dtype)
        return out.at[..., self.w_rows].add(contribution)

    def u_transpose_apply(self, u_values: ArrayLike, x: ArrayLike) -> Array:
        """``U^T x`` over the last axis; ``u_values`` ``(*lanes, n_u)``, ``x`` ``(*lanes, [m,] n)``."""
        u_values, x = jnp.asarray(u_values), jnp.asarray(x)
        if x.ndim == u_values.ndim + 1:
            u_values = u_values[..., None, :]
        gathered = x[..., self.u_rows] * u_values
        out = jnp.zeros(gathered.shape[:-1] + (self.rank,), gathered.dtype)
        return out.at[..., self.u_cols].add(gathered)

    def u_apply(self, u_values: ArrayLike, s: ArrayLike) -> Array:
        """``U s`` over the last axis; ``u_values`` ``(*lanes, n_u)``, ``s`` ``(*lanes, [m,] k)``."""
        u_values, s = jnp.asarray(u_values), jnp.asarray(s)
        if s.ndim == u_values.ndim + 1:
            u_values = u_values[..., None, :]
        contribution = s[..., self.u_cols] * u_values
        out = jnp.zeros(contribution.shape[:-1] + (self.n_dofs,), contribution.dtype)
        return out.at[..., self.u_rows].add(contribution)

    def apply(self, u_values: ArrayLike, x: ArrayLike, *, transpose: bool = False) -> Array:
        """``U W^T x`` (or ``W U^T x`` with ``transpose``) over the last axis of ``x``."""
        if transpose:
            return self.w_apply(self.u_transpose_apply(u_values, x))
        return self.u_apply(u_values, self.w_transpose_apply(x))

    def u_matrix(self, u_values: ArrayLike) -> Array:
        """Dense ``U`` of shape ``(n, k)`` from its entries."""
        u_values = jnp.asarray(u_values)
        out = jnp.zeros((self.n_dofs, self.rank), u_values.dtype)
        return out.at[self.u_rows, self.u_cols].add(u_values)

    def w_matrix(self, dtype: Any = jnp.float64) -> Array:
        """Dense structural ``W`` of shape ``(n, k)``."""
        out = jnp.zeros((self.n_dofs, self.rank), dtype)
        return out.at[self.w_rows, self.w_cols].add(jnp.asarray(self.w_values, dtype))

    def dense(self, u_values: ArrayLike) -> Array:
        """Dense ``U W^T`` of shape ``(n, n)`` (references and small problems)."""
        u = self.u_matrix(u_values)
        return u @ self.w_matrix(u.dtype).T

    @classmethod
    def group_means(
        cls, n_dofs: int, readers: Any, groups: Any, *, weights: Optional[Any] = None
    ) -> LowRankTerm:
        """The term of couplings that each feed the weighted mean of one group of unknowns.

        Coupling ``j`` reads the unknowns ``groups[j]`` with the weights
        ``weights[j]`` (``1 / len(groups[j])`` each by default, a plain mean)
        and feeds the rows ``readers[j]``: ``(U W^T x)_i = U[i, j] *
        mean_j(x)`` for a row ``i`` of ``readers[j]``.

        Args:
            n_dofs: Side length of the operator.
            readers: One integer array of rows per coupling.
            groups: One integer array of unknowns per coupling.
            weights: Optional weights per coupling, shaped like ``groups``.

        Returns:
            The term, whose entries of ``U`` follow ``readers`` coupling by
            coupling.

        Raises:
            ConfigurationError: If ``readers`` and ``groups`` (and ``weights``)
                list different numbers of couplings, or the term is invalid.
        """
        readers = [np.asarray(r, np.int64).reshape(-1) for r in readers]
        groups = [np.asarray(g, np.int64).reshape(-1) for g in groups]
        if len(readers) != len(groups) or (weights is not None and len(weights) != len(groups)):
            raise ConfigurationError(
                "LowRankTerm.group_means needs one array of reader rows, one group and (when "
                "given) one weight array per coupling; got "
                f"{len(readers)} readers and {len(groups)} groups."
            )
        if weights is None:
            weights = [np.full(g.shape, 1.0 / max(g.size, 1)) for g in groups]
        weights = [np.asarray(w, np.float64).reshape(-1) for w in weights]
        return cls(
            n_dofs=n_dofs,
            rank=len(groups),
            u_rows=np.concatenate(readers) if readers else np.zeros(0, np.int64),
            u_cols=np.concatenate([np.full(r.shape, j) for j, r in enumerate(readers)]),
            w_rows=np.concatenate(groups),
            w_cols=np.concatenate([np.full(g.shape, j) for j, g in enumerate(groups)]),
            w_values=np.concatenate(weights),
        )


def low_rank_matvec(
    term: LowRankTerm, keep: np.ndarray, u_values: ArrayLike, x: ArrayLike, *, transpose: bool
) -> Array:
    """``U_bc W^T x`` (or ``W U_bc^T x``), with ``U``'s entries in Dirichlet rows dropped.

    ``keep`` is the host mask (``0.0`` for an entry of ``U`` in a Dirichlet
    row, ``1.0`` otherwise). Linear in ``u_values`` and in ``x``.
    """
    u_values = jnp.asarray(u_values)
    return term.apply(u_values * jnp.asarray(keep, u_values.dtype), x, transpose=transpose)


# --------------------------------------------------------------------------- #
# Woodbury backend
# --------------------------------------------------------------------------- #
class _WoodburyPlan:
    """The inner analysis of ``A_b`` and the host masks of one Woodbury analysis."""

    __slots__ = ("inner", "nnz_base", "keep")

    def __init__(self, inner: Analysis, nnz_base: int, keep: np.ndarray):
        """Record the inner analysis, the base length and the Dirichlet mask of ``U``."""
        self.inner, self.nnz_base, self.keep = inner, nnz_base, keep


_ModuleMeta = type(eqx.Module)


class _WoodburyMeta(_ModuleMeta):
    """Builds every ``Woodbury`` as an instance of a subclass specialised to its inner backend.

    The contract reads a backend's capability flags on its class; a
    Woodbury backend reports its inner backend's flags, so ``Woodbury(inner,
    term)`` constructs an instance of the cached subclass for
    ``type(inner)``, whose flags are the inner class's.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Construct through the subclass specialised to the inner backend's class."""
        target = cls
        if cls is Woodbury:
            inner = kwargs["inner"] if "inner" in kwargs else (args[0] if args else None)
            if not isinstance(inner, LinearSolver):
                raise ConfigurationError(
                    "Woodbury needs inner= a configured diff_soup.linear.LinearSolver that "
                    f"factors the base operator, for example Dense() or CuDSS(); got "
                    f"{type(inner).__name__}."
                )
            target = _specialised(type(inner))
        return super(_WoodburyMeta, target).__call__(*args, **kwargs)


#: Capability flags a specialised Woodbury class copies from its inner backend's class.
_INHERITED_FLAGS = (
    "platforms",
    "factor_residency",
    "transpose_is_free",
    "vmap_factors",
    "supported_precisions",
    "supports_hybrid_memory",
    "supports_reordering",
)


class Woodbury(LinearSolver, metaclass=_WoodburyMeta):
    """A direct solver for a sparse matrix plus a low-rank term, `A + U W^T`.

    It factors `A` with `inner` and applies the Woodbury identity for the term. Analyze
    `term.topology(base)`, and pass the values of `A` followed by the values of `U`. It needs
    a `SolveContext` exactly when `inner` does.

    Args:
        inner: A direct backend for `A`, for example `Dense()` or `CuDSS()`.
        term: The `LowRankTerm` `U W^T`.

    !!! example

        ```python
        backend = Woodbury(Dense(), term)
        analysis = analyze(term.topology(base), backend=backend)
        f = factor(analysis, jnp.concatenate([a_values, u_values]))
        ```
    """

    backend_name = "woodbury"
    is_exact = True
    accepts_operator = False

    inner: LinearSolver = eqx.field(static=True)
    term: LowRankTerm = eqx.field(static=True)

    def __init__(self, inner: LinearSolver, term: LowRankTerm):
        """Hold the inner backend and the term."""
        cls = type(inner)
        if not cls.is_exact:
            raise ConfigurationError(
                f"Woodbury needs an exact inner backend (the identity applies the exact inverse "
                f"of A_b); {cls.__name__} declares is_exact=False."
            )
        if cls._analyze is LinearSolver._analyze:
            raise ConfigurationError(
                f"Woodbury needs an inner backend that implements the analyze/factor/solve "
                f"contract; {cls.__name__} defines no _analyze hook."
            )
        if not isinstance(term, LowRankTerm):
            raise ConfigurationError(
                f"Woodbury needs term= a diff_soup.linear.LowRankTerm; got {type(term).__name__}."
            )
        self.inner = inner
        self.term = term

    def __reduce__(self) -> tuple:
        """Pickle by content: unpickling rebuilds the specialised class through ``Woodbury``."""
        return (_rebuild_woodbury, (self.inner, self.term))

    # ---- analysis ----------------------------------------------------------------
    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> _WoodburyPlan:
        """Analyze the base topology with the inner backend (options and layout pass through).

        Raises:
            ConfigurationError: If the topology is not ``term.topology(base)``
                for this backend's term, or the inner backend refuses the base.
        """
        base = self.term.base_topology(topology, caller="Woodbury")
        inner = analyze(
            base, backend=self.inner, options=options, layout=layout, precision=precision
        )
        bc = topology.bc_indices
        keep = np.ones((self.term.n_u,), np.float64)
        if bc is not None and int(bc.size):
            keep[np.isin(self.term.u_rows, bc)] = 0.0
        keep.flags.writeable = False
        return _WoodburyPlan(inner, int(base.rows.size), keep)

    def _inner_avals(self, analysis: Analysis, dtype: Any, orientation: str, precision: str):
        """The abstract leaves of the inner handle."""
        inner = analysis.backend_analysis.inner
        return tuple(
            inner.backend.factor_avals(
                inner,
                ShapedArray((inner.nnz,), dtype),
                orientation=orientation,
                precision=precision,
                tol=None,
            )
        )

    def factor_avals(
        self, analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
    ) -> tuple:
        """The inner handle's leaves, then ``Y`` ``(k, n)``, the LU ``(k, k)``, pivots, status."""
        del tol
        k, n = self.term.rank, self.term.n_dofs
        dtype = rung_dtype(precision, values_aval.dtype)
        return (
            *self._inner_avals(analysis, values_aval.dtype, orientation, precision),
            ShapedArray((k, n), dtype),
            ShapedArray((k, k), dtype),
            ShapedArray((k,), jnp.int32),
            ShapedArray((), jnp.int32),
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
        """The inner backend's empty handle, zero Woodbury arrays and the status ``EMPTY``."""
        inner = analysis.backend_analysis.inner
        inner_handle = inner.backend.empty_handle(
            inner,
            ShapedArray((inner.nnz,), values_aval.dtype),
            context_id=context_id,
            orientation=orientation,
            precision=precision,
            tol=None,
        )
        own = self.factor_avals(
            analysis, values_aval, orientation=orientation, precision=precision, tol=tol
        )[len(inner_handle) :]
        zeros = [jnp.zeros(aval.shape, aval.dtype) for aval in own[:-1]]
        return (*inner_handle, *zeros, jnp.asarray(int(SolveStatus.EMPTY), jnp.int32))

    # ---- lane functions ----------------------------------------------------------
    def _p_rows(self, plan: _WoodburyPlan, u: Array, orientation: str, dtype: Any) -> Array:
        """``P^T`` as ``(k, n)`` rows: ``U_bc^T`` for ``"A"``, ``W^T`` for ``"AT"``."""
        term = self.term
        if orientation == "A":
            masked = u.astype(dtype) * jnp.asarray(plan.keep, dtype)
            return (
                jnp.zeros((term.rank, term.n_dofs), dtype).at[term.u_cols, term.u_rows].add(masked)
            )
        return term.w_matrix(dtype).T

    def _q_transpose(self, plan: _WoodburyPlan, u: Array, orientation: str, x: Array) -> Array:
        """``Q^T x`` over the last axis: ``W^T x`` for ``"A"``, ``U_bc^T x`` for ``"AT"``."""
        if orientation == "A":
            return self.term.w_transpose_apply(x)
        masked = u.astype(x.dtype) * jnp.asarray(plan.keep, x.dtype)
        return self.term.u_transpose_apply(masked, x)

    def _q_apply(self, plan: _WoodburyPlan, u: Array, orientation: str, s: Array) -> Array:
        """``Q s`` over the last axis: ``W s`` for ``"A"``, ``U_bc s`` for ``"AT"``."""
        if orientation == "A":
            return self.term.w_apply(s)
        masked = u.astype(s.dtype) * jnp.asarray(plan.keep, s.dtype)
        return self.term.u_apply(masked, s)

    def _inner_solve(
        self, analysis, base, inner_handle, order, rhs, *, orientation, transpose, precision
    ):
        """One solve with the inner factor (``(x, status, order)``)."""
        inner = analysis.backend_analysis.inner
        return phased_solve_p.bind(
            base,
            *inner_handle,
            order,
            rhs,
            n_handle=len(inner_handle),
            analysis=inner,
            orientation=orientation,
            transpose=transpose,
            precision=precision,
            tol=None,
            storage=None,
        )

    def _own_factor(
        self, analysis, values, inner_handle, order, *, orientation, precision
    ) -> tuple:
        """``(Y, lu, piv, status, order)`` of one lane from a held inner factor."""
        plan = analysis.backend_analysis
        base, u = self.term.split(values)
        dtype = rung_dtype(precision, values.dtype)
        p_rows = self._p_rows(plan, u, orientation, dtype)
        y_rows, _, order = self._inner_solve(
            analysis,
            base,
            inner_handle,
            order,
            p_rows,
            orientation=orientation,
            transpose=False,
            precision=precision,
        )
        capacitance = (
            jnp.eye(self.term.rank, dtype=dtype) + self._q_transpose(plan, u, orientation, y_rows).T
        )
        lu, piv = jax.scipy.linalg.lu_factor(capacitance)
        pivots = jnp.abs(jnp.diagonal(lu))
        scale = jnp.maximum(
            1.0, jnp.max(jnp.abs(capacitance - jnp.eye(self.term.rank, dtype=dtype)))
        )
        roundoff = SINGULAR_PIVOT_ROUNDOFFS * jnp.finfo(lu.dtype).eps / 2
        bad = (
            ~jnp.all(jnp.isfinite(lu))
            | ~jnp.all(jnp.isfinite(y_rows))
            | jnp.any(pivots <= roundoff * scale)
        )
        status = jnp.where(bad, int(SolveStatus.SINGULAR), _SUCCESS).astype(jnp.int32)
        return y_rows, lu, piv.astype(jnp.int32), status, order

    @staticmethod
    def _combine(inner_status: Array, own_status: Array) -> Array:
        """The inner status where it is not ``SUCCESS``, else the Woodbury status."""
        return jnp.where(inner_status == _SUCCESS, own_status, inner_status).astype(jnp.int32)

    # ---- contract hooks ----------------------------------------------------------
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
        """Factor ``A_b`` with the inner backend, then form ``Y`` and the capacitance LU.

        The inner factorization is bound on the base values with this
        factorization's slot token, context id and order word, so the inner
        backend stores it as it would a factor of its own; the inner solve
        that forms ``Y`` follows it, and its order word is returned.

        Returns:
            ``(*inner_handle, Y, lu, piv, own_status, status, order)``.
        """
        del tol
        inner = analysis.backend_analysis.inner

        def one(v, o):
            base, _ = self.term.split(v)
            *inner_handle, inner_status, o = phased_factor_p.bind(
                base,
                token,
                context_id,
                o,
                analysis=inner,
                orientation=orientation,
                precision=precision,
                tol=None,
            )
            *own, o = self._own_factor(
                analysis, v, inner_handle, o, orientation=orientation, precision=precision
            )
            return (*inner_handle, *own, self._combine(inner_status, own[-1]), o)

        return lanes(one, values.ndim == 2, values, order)

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
        """Solve with ``M`` (one inner solve) or ``M^T`` (one transposed inner solve).

        The inner solve takes the order word and its order word is returned.

        Returns:
            ``(x, status, order)``; ``nan`` wherever the status is not ``SUCCESS``.
        """
        del tol
        plan = analysis.backend_analysis
        n_inner = len(handle) - 4

        def one(v, leaves, o, rhs):
            inner_handle, (y_rows, lu, piv, own_status) = leaves[:n_inner], leaves[n_inner:]
            base, u = self.term.split(v)
            common = dict(orientation=orientation, precision=precision)
            if not transpose:
                y, inner_status, o = self._inner_solve(
                    analysis, base, inner_handle, o, rhs, transpose=False, **common
                )
                t = self._q_transpose(plan, u, orientation, y)
                z = jax.scipy.linalg.lu_solve((lu, piv), t.T if t.ndim == 2 else t)
                x = y - (z.T if z.ndim == 2 else z) @ y_rows
            else:
                s = jax.scipy.linalg.lu_solve((lu, piv), (rhs @ y_rows.T).T, trans=1)
                s = s.T if rhs.ndim == 2 else s
                x, inner_status, o = self._inner_solve(
                    analysis,
                    base,
                    inner_handle,
                    o,
                    rhs - self._q_apply(plan, u, orientation, s),
                    transpose=True,
                    **common,
                )
            status = self._combine(inner_status, own_status)
            return jnp.where(status == _SUCCESS, x, jnp.nan).astype(rhs.dtype), status, o

        batched = values.ndim == 2
        if batched and b.ndim == 1:
            b = jnp.broadcast_to(b, (values.shape[0],) + b.shape)
        return lanes(one, batched, values, tuple(handle), order, b)

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
        """Refactor the inner factor with the inner backend's rules, then the Woodbury arrays.

        The inner refactor receives the base values, the held base values,
        the predicate, ``adopt`` and the order word. The Woodbury arrays are
        rebuilt from the new inner factor under one ``lax.cond`` on whether
        any lane asks (their inner solve follows the inner refactor), and
        each lane keeps its held arrays where its predicate is false.

        Returns:
            ``(kept_values, *handle, status, order)``.
        """
        del tol
        inner = analysis.backend_analysis.inner
        n_inner = len(handle) - 4
        batched = values.ndim == 2
        predicate = jnp.asarray(predicate, jnp.bool_)

        def inner_refactor(v, h, p, inner_handle, o):
            return tuple(
                phased_refactor_p.bind(
                    self.term.split(v)[0],
                    self.term.split(h)[0],
                    p,
                    *inner_handle,
                    o,
                    n_handle=n_inner,
                    analysis=inner,
                    orientation=orientation,
                    precision=precision,
                    tol=None,
                    adopt=adopt,
                    storage=None,
                )
            )

        outs = lanes(
            inner_refactor,
            batched,
            values,
            held_values,
            predicate,
            tuple(handle[:n_inner]),
            order,
        )
        kept_base, new_inner, inner_status, order = outs[0], outs[1:-2], outs[-2], outs[-1]
        held_own = tuple(handle[n_inner:])

        def rebuild(_):
            def one(v, inner_handle, o):
                return self._own_factor(
                    analysis, v, inner_handle, o, orientation=orientation, precision=precision
                )

            return tuple(lanes(one, batched, values, tuple(new_inner), order))

        def keep(_):
            return (*held_own, order)

        *fresh, order = jax.lax.cond(jnp.any(predicate), rebuild, keep, None)
        own = tuple(_select(predicate, new, old) for new, old in zip(fresh, held_own))
        kept_u = _select(predicate, self.term.split(values)[1], self.term.split(held_values)[1])
        kept_values = self.term.join(kept_base, kept_u)
        status = self._combine(inner_status, own[-1])
        return (kept_values, *new_inner, *own, status, order)

    def label_empty_lanes(self, handle: tuple, lanes_: Any, size: int) -> tuple:
        """Relabel the inner handle's empty lanes with the inner backend's hook, if it has one."""
        label = getattr(self.inner, "label_empty_lanes", None)
        if label is None:
            return handle
        n_inner = len(handle) - 4
        return (*label(tuple(handle[:n_inner]), lanes_, size), *handle[n_inner:])

    def pattern_matvec(
        self, analysis: Analysis, dvalues: ArrayLike, x: ArrayLike, *, transpose: bool
    ) -> Array:
        """``(D_b + U_bc W^T) x`` (or the transpose) for entry values ``dvalues``.

        The family's derivative rules and residuals read the operator through
        this method: the base part through the inner analysis, the low-rank
        part through the term, every entry in a Dirichlet row dropped.
        """
        plan = analysis.backend_analysis
        base, u = self.term.split(dvalues)
        applied = pattern_matvec(plan.inner, base, x, transpose=transpose)
        low_rank = low_rank_matvec(self.term, plan.keep, u, x, transpose=transpose)
        return applied + low_rank.astype(applied.dtype)

    def _release(self, ctx: Any) -> None:
        """Release the inner backend's storage held for ``ctx``."""
        self.inner._release(ctx)

    def _resident_bytes(self, ctx: Any) -> int:
        """The inner backend's storage held for ``ctx``, counted once per storage owner."""
        mine = _storage_class(self)
        for registered in tuple(getattr(ctx, "_backends", ())):
            if _storage_class(registered) is mine:
                if registered is not self:
                    return 0
                break
        return int(self.inner._resident_bytes(ctx))


def _storage_class(backend: Any) -> type:
    """The class of the backend that owns a backend's native storage (through Woodbury)."""
    while isinstance(backend, Woodbury):
        backend = backend.inner
    return type(backend)


def _rebuild_woodbury(inner: LinearSolver, term: LowRankTerm) -> Woodbury:
    """Unpickle a Woodbury backend through its constructor."""
    return Woodbury(inner, term)


def _lower_invalidate(self, ctx: Any, *args: Any, **params: Any) -> Any:
    """Lower an invalidation of a Woodbury factor as the inner backend's invalidation."""
    analysis = params["analysis"]
    inner = analysis.backend_analysis.inner
    n_inner = int(params["n_handle"]) - 4

    def invalidate_inner(*operands):
        *handle, context_id, order = operands
        return list(
            phased_invalidate_p.bind(
                *handle[:n_inner],
                context_id,
                order,
                n_handle=n_inner,
                analysis=inner,
                orientation=params["orientation"],
                precision=params["precision"],
            )
        )

    return mlir.lower_fun(invalidate_inner, multiple_results=True)(ctx, *args)


@functools.lru_cache(maxsize=None)
def _specialised(inner_cls: type) -> type:
    """The subclass of ``Woodbury`` whose capability flags are ``inner_cls``'s."""
    namespace: dict[str, Any] = {
        name: getattr(inner_cls, name) for name in _INHERITED_FLAGS if hasattr(inner_cls, name)
    }
    namespace["__module__"] = Woodbury.__module__
    namespace["__qualname__"] = f"Woodbury[{inner_cls.__name__}]"
    namespace["__doc__"] = Woodbury.__doc__
    if getattr(inner_cls, "lower_invalidate", None) is not None:
        namespace["lower_invalidate"] = _lower_invalidate
    return _WoodburyMeta(f"Woodbury[{inner_cls.__name__}]", (Woodbury,), namespace)
