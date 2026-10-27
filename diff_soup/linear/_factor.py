"""Types of the analyze / factor / solve linear-solver contract.

A sparse linear system goes through three steps, each with its own cost and
its own place in a program:

1. [analyze][diff_soup.linear.analyze] is a host operation on the sparsity
   pattern. It returns an [Analysis][diff_soup.linear.Analysis], a static,
   hashable object memoised by content, so a pattern is analysed once per
   process however many factors, orientations and compiled programs use it.
2. [factor][diff_soup.linear.factor] is traced. It returns a
   [Factor][diff_soup.linear.Factor], an ``equinox.Module`` pytree whose
   dynamic leaves are the operator values, the backend's factor handle, a
   generation counter and an order word. A factor crosses ``jit``
   boundaries, rides ``lax.scan`` and ``lax.while_loop`` carries and batches
   under ``vmap`` like any other pytree.
3. [solve][diff_soup.linear.solve] is traced and linear in the right-hand
   side. It solves with the factored operator or its transpose and
   differentiates in both the right-hand side and the operator values: the
   tangent is ``solve(f, db - dA x)`` and the cotangent is a transposed solve
   on the same factor, so no derivative ever factors again. It returns the
   solution and the factor to use next, ``x, f = solve(f, b)``.

Operations on a factor are ordered by data, not by a JAX effect: every
operation takes the factor's order word and the factor it returns carries the
next one, so threading the factor through a program (``x, f = solve(f, b)``,
``f = refactor(f, values, predicate)``, ``f = invalidate(f)``) runs its
operations in program order. [after][diff_soup.linear.after] links the order
words of different factors that name the same storage.

Status is part of every solve. [solve_checked][diff_soup.linear.solve_checked]
returns a [SolveReport][diff_soup.linear.SolveReport] with a traced
[SolveStatus][diff_soup.linear.SolveStatus] code and the achieved residual; a
singular factorization never yields finite output silently. A factor of an
inexact backend (``Factor.exact`` is ``False``, for an iterative solver) stops
at a [Tolerance][diff_soup.linear.Tolerance] and returns a finite solution
even when it stopped short, so its report is the only signal of that.

Dirichlet rows (``SparseTopology.bc_indices``) are applied identically on
every backend: every entry in a Dirichlet row is replaced by zero and ``1`` is
added at the row's diagonal entry when the pattern contains it.
"""

from __future__ import annotations

import enum
import functools
from typing import Any, NamedTuple, Optional

import equinox as eqx
import numpy as np
from jaxtyping import Array

from diff_soup.linear._context import FactorContractError, SolveContext

#: The operator orientations a factor can represent.
ORIENTATIONS: tuple[str, ...] = ("A", "AT")
#: The precision rungs a factor can hold.
PRECISION_RUNGS: tuple[str, ...] = ("fp64", "fp32")
#: Where a backend keeps its factor storage.
FACTOR_RESIDENCIES: tuple[str, ...] = ("arrays", "device", "host")
#: How a backend serves a batch of factors under ``vmap``.
VMAP_FACTOR_MODES: tuple[str, ...] = ("native", "sequential", "unsupported")


class SolveStatus(enum.IntEnum):
    """The status code of a factorization or a solve, one int32 per matrix.

    Attributes:
        SUCCESS: The solve succeeded.
        SINGULAR: The factorization met a zero pivot, and its solves return `nan`.
        NONFINITE: The values or the factor hold a non-finite number, and solves return `nan`.
        MAX_ITERATIONS: An iterative solve reached its iteration cap.
        STAGNATION: An iterative solve stopped making progress.
        BREAKDOWN: An iterative solve broke down.
        PRECONDITIONER_FAILURE: The preconditioner failed to factor or to apply.
        EMPTY: The factor holds no factorization yet, and its solves return `nan`.
    """

    SUCCESS = 0
    SINGULAR = 1
    NONFINITE = 2
    MAX_ITERATIONS = 3
    STAGNATION = 4
    BREAKDOWN = 5
    PRECONDITIONER_FAILURE = 6
    EMPTY = 7


class Tolerance(NamedTuple):
    """The stopping rule of an iterative solve.

    A solve of `M x = b` stops when `||w * (b - M x)||_rms <= atol + rtol * ||w * b||_rms`,
    with `w` the solve's `weights`, or ones without them.

    Attributes:
        rtol: The relative tolerance, at least 0.
        atol: The absolute tolerance, at least 0.
        max_iter: The iteration cap, at least 0.
        restart: The iterations between GMRES restarts, at least 1, default `30`.
    """

    rtol: float
    atol: float
    max_iter: int
    restart: int = 30


def check_tolerance(tol: Any) -> Optional[Tolerance]:
    """Validate a tolerance argument and normalise its field types.

    Args:
        tol: ``None`` or a [Tolerance][diff_soup.linear.Tolerance].

    Returns:
        ``None``, or a ``Tolerance`` with ``float`` / ``float`` / ``int`` /
        ``int`` fields.

    Raises:
        FactorContractError: If ``tol`` is neither, a field is not a number,
            a tolerance or ``max_iter`` is negative, or ``restart`` is below
            ``1``.
    """
    if tol is None:
        return None
    if not isinstance(tol, Tolerance):
        raise FactorContractError(
            f"tol must be None or a diff_soup.linear.Tolerance(rtol, atol, max_iter, restart); "
            f"got {type(tol).__name__}."
        )
    try:
        rtol, atol = float(tol.rtol), float(tol.atol)
        max_iter, restart = int(tol.max_iter), int(tol.restart)
    except (TypeError, ValueError) as exc:
        raise FactorContractError(
            "Tolerance fields must be numbers (rtol and atol real, max_iter and restart "
            f"integers); got {tol}."
        ) from exc
    if rtol < 0.0 or atol < 0.0 or max_iter < 0:
        raise FactorContractError(
            "Tolerance fields must be non-negative (rtol >= 0, atol >= 0, max_iter >= 0); "
            f"got {tol}."
        )
    if restart < 1:
        raise FactorContractError(
            f"Tolerance.restart must be at least 1 (iterations between restarts); got {tol}."
        )
    return Tolerance(rtol, atol, max_iter, restart)


# --------------------------------------------------------------------------- #
# Orientation and precision helpers
# --------------------------------------------------------------------------- #
def check_orientation(orientation: Any) -> str:
    """Validate an orientation name.

    Args:
        orientation: ``"A"`` or ``"AT"``.

    Returns:
        The orientation.

    Raises:
        FactorContractError: For any other value.
    """
    if orientation not in ORIENTATIONS:
        raise FactorContractError(f"orientation must be 'A' or 'AT'; got {orientation!r}.")
    return str(orientation)


def flip_orientation(orientation: str) -> str:
    """The opposite orientation: ``"A"`` for ``"AT"`` and ``"AT"`` for ``"A"``."""
    return "AT" if check_orientation(orientation) == "A" else "A"


def effective_transpose(orientation: str, transpose: bool) -> bool:
    """Whether a solve applies the transpose of the assembled operator ``A_bc``.

    A factor of orientation ``"A"`` represents ``M = A_bc`` and one of
    orientation ``"AT"`` represents ``M = A_bc^T``; a solve with
    ``transpose=True`` applies ``M^T``. The assembled operator is therefore
    transposed exactly when one of the two flips holds.

    Args:
        orientation: The factor's orientation.
        transpose: The solve's ``transpose`` flag.

    Returns:
        ``(orientation == "AT") != transpose``.
    """
    return (check_orientation(orientation) == "AT") != bool(transpose)


def rung_dtype(precision: str, like: Any = None) -> np.dtype:
    """The floating dtype a factor holds at a precision rung.

    Args:
        precision: ``"fp64"`` or ``"fp32"``.
        like: Optional dtype of the operand; a complex ``like`` gives the
            complex dtype of the rung's width.

    Returns:
        ``float64`` / ``complex128`` for ``"fp64"``, ``float32`` /
        ``complex64`` for ``"fp32"``.

    Raises:
        FactorContractError: For an unknown rung.
    """
    complex_ = like is not None and np.issubdtype(np.dtype(like), np.complexfloating)
    if precision == "fp64":
        return np.dtype(np.complex128 if complex_ else np.float64)
    if precision == "fp32":
        return np.dtype(np.complex64 if complex_ else np.float32)
    raise FactorContractError(f"precision must be one of {PRECISION_RUNGS}; got {precision!r}.")


def solve_dtype(precision: str, values_dtype: Any, b_dtype: Any) -> np.dtype:
    """The dtype of a solve's result at a precision rung.

    The result has the rung's dtype whatever the width of the right-hand
    side: ``float64`` for ``"fp64"`` and ``float32`` for ``"fp32"``, or the
    complex counterpart of the same width when the values or the right-hand
    side are complex. A solve is linear in the right-hand side, so a complex
    right-hand side on a real operator has a complex solution.

    Args:
        precision: ``"fp64"`` or ``"fp32"``.
        values_dtype: The dtype of the operator values.
        b_dtype: The dtype of the right-hand side.

    Returns:
        ``float64`` / ``complex128`` for ``"fp64"``, ``float32`` /
        ``complex64`` for ``"fp32"``.

    Raises:
        FactorContractError: For an unknown rung.
    """
    complex_ = any(
        np.issubdtype(np.dtype(dtype), np.complexfloating) for dtype in (values_dtype, b_dtype)
    )
    return rung_dtype(precision, np.complex64 if complex_ else None)


@functools.lru_cache(maxsize=256)
def dirichlet_rows(topology: Any) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Host masks that apply a topology's Dirichlet rows to its value vector.

    The assembled operator ``A_bc`` has every entry whose row is a Dirichlet
    index replaced by zero and ``1`` added at that row's diagonal entry when
    the pattern contains it. Duplicate diagonal entries receive the ``1``
    once, so ``A_bc`` is the same on every backend whatever the duplicates.

    Args:
        topology: The [SparseTopology][diff_soup.linear.SparseTopology].

    Returns:
        ``(row_mask, diagonal)``: a read-only boolean array over the pattern
        entries, ``True`` where the entry lies in a Dirichlet row (``None``
        when the topology has no Dirichlet rows), and the read-only sorted
        int32 Dirichlet rows whose diagonal is in the pattern (empty when
        none).
    """
    bc = topology.bc_indices
    if bc is None or int(bc.size) == 0:
        empty = np.zeros((0,), dtype=np.int32)
        empty.flags.writeable = False
        return None, empty
    rows = np.asarray(topology.rows)
    cols = np.asarray(topology.cols)
    mask = np.isin(rows, bc)
    diagonal = np.unique(rows[mask & (rows == cols)]).astype(np.int32)
    mask.flags.writeable = False
    diagonal.flags.writeable = False
    return mask, diagonal


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
class Analysis:
    """The analysis of a sparsity pattern for one backend, as returned by `analyze`.

    Pass it to `factor` or `empty_factor`. It is immutable and hashable, and unpickling it
    analyzes the pattern again.

    Attributes:
        topology: The `SparseTopology`.
        backend: The backend object.
        options: The settings, as sorted `(key, value)` pairs.
        layout: The device placement.
        precision: `"fp64"` or `"fp32"`.
        symmetric: Whether the matrix was declared symmetric.
        spd: Whether the matrix was declared symmetric positive definite.
        n: The number of rows.
        nnz: The number of pattern entries.
    """

    __slots__ = (
        "topology",
        "backend",
        "options",
        "layout",
        "precision",
        "symmetric",
        "spd",
        "_backend_analysis",
        "_key",
        "_hash",
    )

    def __init__(
        self,
        topology: Any,
        backend: Any,
        options: tuple,
        layout: Any,
        precision: str,
        symmetric: bool,
        spd: bool,
        backend_analysis: Any = None,
    ):
        """Record the analysed content; use [analyze][diff_soup.linear.analyze] instead.

        Args:
            topology: The sparsity pattern.
            backend: The configured backend.
            options: Frozen backend options.
            layout: The layout.
            precision: The precision rung.
            symmetric: The symmetry declaration.
            spd: The positive-definiteness declaration.
            backend_analysis: The backend's private symbolic result.
        """
        key = (topology, backend, options, layout, precision, bool(symmetric), bool(spd))
        for name, value in zip(self.__slots__[:7], key):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_backend_analysis", backend_analysis)
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_hash", hash(("diff_soup.linear.Analysis",) + key))

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse mutation: an analysis is immutable."""
        raise AttributeError("Analysis is immutable; call diff_soup.linear.analyze again.")

    def __hash__(self) -> int:
        """Hash by content."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Equal when the analysed content is equal."""
        if not isinstance(other, Analysis):
            return NotImplemented
        return self is other or (self._hash == other._hash and self._key == other._key)

    def __reduce__(self) -> tuple:
        """Pickle by content; unpickling analyses again through the memo of ``analyze``.

        In the same process the round trip returns this very instance; in
        another process it runs the backend's symbolic analysis there once.
        """
        return (
            _analysis_from_content,
            (
                self.topology,
                self.backend,
                self.options,
                self.layout,
                self.precision,
                self.symmetric,
                self.spd,
            ),
        )

    def __copy__(self) -> Analysis:
        """An analysis is immutable, so a copy is the analysis itself."""
        return self

    def __deepcopy__(self, memo: dict) -> Analysis:
        """An analysis is immutable, so a deep copy is the analysis itself."""
        return self

    def __repr__(self) -> str:
        """Show the backend, size, rung and structure declaration."""
        return (
            f"Analysis(backend={self.backend_name!r}, n={self.n}, nnz={self.nnz}, "
            f"precision={self.precision!r}, symmetric={self.symmetric}, spd={self.spd})"
        )

    @property
    def n(self) -> int:
        """Side length of the operator (``topology.n_dofs``)."""
        return int(self.topology.n_dofs)

    @property
    def nnz(self) -> int:
        """Number of pattern entries, duplicates included (the values length)."""
        return int(self.topology.rows.shape[0])

    @property
    def backend_name(self) -> Optional[str]:
        """The backend's registered name, for example ``"dense"``."""
        return type(self.backend).backend_name

    @property
    def backend_analysis(self) -> Any:
        """The backend's private symbolic result (``None`` for Dense)."""
        return self._backend_analysis


def _analysis_from_content(
    topology: Any,
    backend: Any,
    options: Any,
    layout: Any,
    precision: Any,
    symmetric: Any,
    spd: Any,
) -> Analysis:
    """Rebuild an analysis from its content through the memo of ``analyze`` (unpickling)."""
    # Resolved on call: ``_api`` imports this module.
    from diff_soup.linear._api import analyze

    return analyze(
        topology,
        backend=backend,
        options=options,
        layout=layout,
        precision=precision,
        symmetric=symmetric,
        spd=spd,
    )


# --------------------------------------------------------------------------- #
# Factor and report
# --------------------------------------------------------------------------- #
class Factor(eqx.Module):
    """A factorization of a matrix, as returned by `factor`, `empty_factor` and `refactor`.

    It is a JAX pytree, so it passes through `jax.jit`, `jax.vmap` and loop carries. Pass on
    the factor each operation returns, which keeps the operations in order.

    Attributes:
        analysis: The `Analysis` the factor was built from.
        orientation: `"A"` for the matrix and `"AT"` for its transpose.
        precision: `"fp64"` or `"fp32"`.
        exact: Whether solves apply the exact inverse.
        values: The values the factor was built from.
        generation: `0` after `factor` and `-1` after `empty_factor`, plus one for every
            refactor that took new values.
        storage: The storage name given to `empty_factor`, or `None`.
    """

    analysis: Analysis = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    values: Array
    handle: tuple
    generation: Array
    order: Array
    storage: Optional[str] = eqx.field(static=True, default=None)

    @property
    def backend(self) -> Any:
        """The configured backend of the analysis."""
        return self.analysis.backend

    @property
    def transposed(self) -> bool:
        """Whether the factor represents ``A_bc^T``."""
        return self.orientation == "AT"


class SolveReport(eqx.Module):
    """The status and the residual of one `solve_checked` call.

    Its fields are arrays, so a report passes through `jax.jit` and `jax.vmap`.

    Attributes:
        status: The `SolveStatus` code, as an int32.
        residual: The largest `|b - M x|`, or the weighted root-mean-square residual when the
            solve had `weights`.
        iterations: The iteration count, `0` for a direct solve.
        applications: The count of operator or factor applications, `1` for a direct solve.
        ok: `True` where `status` is `SUCCESS`.
        converged: The same as `ok`.
    """

    status: Array
    residual: Array
    iterations: Array
    applications: Array

    @property
    def ok(self) -> Array:
        """Boolean array, ``True`` where the status is ``SUCCESS``."""
        return self.status == int(SolveStatus.SUCCESS)

    @property
    def converged(self) -> Array:
        """Boolean array, ``True`` where the solve met its contract (the same as ``ok``).

        An exact solve converges when its factorization succeeded; an
        inexact one when it met its tolerance.
        """
        return self.ok


__all__ = [
    "Analysis",
    "FACTOR_RESIDENCIES",
    "Factor",
    "FactorContractError",
    "ORIENTATIONS",
    "PRECISION_RUNGS",
    "SolveContext",
    "SolveReport",
    "SolveStatus",
    "Tolerance",
    "VMAP_FACTOR_MODES",
    "check_orientation",
    "check_tolerance",
    "dirichlet_rows",
    "effective_transpose",
    "flip_orientation",
    "rung_dtype",
    "solve_dtype",
]
