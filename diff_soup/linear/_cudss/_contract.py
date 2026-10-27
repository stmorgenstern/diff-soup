"""cuDSS on the analyze / factor / solve contract.

[analyze][diff_soup.linear.analyze], [factor][diff_soup.linear.factor],
[solve][diff_soup.linear.solve] and [refactor][diff_soup.linear.refactor] run
on cuDSS through this module and the contract handlers of the native
extension ``diff_soup._internal.cudss_ffi`` (``_native_solver``).

How the pieces fit:

- ``analyze`` builds, on the host, the compressed-row (CSR) pattern of the
  topology (entries sorted by row, then column, with duplicate coordinate
  entries merged into one CSR entry) and hands it to the extension, which
  keeps it under one native analysis id. Every
  [Analysis][diff_soup.linear.Analysis] of the same pattern and options
  shares that native analysis, whatever its precision rung or its
  ``symmetric``/``spd`` declaration, so the pattern is reordered once and
  stored on the device once; the native analysis is released when the last
  such ``Analysis`` is collected. The device copies of the pattern are
  made on first use in a [SolveContext][diff_soup.linear.SolveContext] and
  freed with the last context that uses them.
- The values of a factor or solve call arrive in the topology's coordinate
  order. Each lowered call first assembles them into CSR values with JAX
  operations in the compiled program ([assemble_csr][]): duplicates add,
  every entry of a Dirichlet row is multiplied by zero and ``1`` is added
  once at its diagonal. The native handlers only ever see CSR values.
- The factor handle is three ``int32`` scalars ``(context id, slot,
  generation)``. A slot holds one cuDSS factorization of ``A_bc``, for
  either orientation, and solves with ``A_bc^T`` (a transposed solve, or
  any solve of an ``"AT"`` factor) in cuDSS's transpose solve mode on that
  factorization; a native batch slot does the same for every lane. An
  analysis declared ``symmetric`` or ``spd`` solves both orientations in
  normal mode.
- A ``vmap`` over values runs as one native batch (``vmap_factors =
  "native"``): the factor call receives CSR values of shape ``(lanes,
  nse)`` and keeps all lanes in one batch slot. By default
  (``batch_layout="stacked"``) the lanes are factored and solved as one
  cuDSS system that holds them block-diagonally; with
  ``batch_layout="uniform"`` they run on cuDSS's uniform batch (one
  pattern, the lanes' values one after the other). Each handle leaf has
  shape ``(lanes,)`` and lane ``k`` of a batch is its position along that
  axis. A batched solve takes ``b`` of shape ``(lanes, n)`` or ``(lanes,
  nrhs, n)``. A batched refactor factors the lanes whose predicate holds
  from the values the call carries for them, and a lane that does not ask
  keeps its factor: the uniform batch factors only the asking lanes, the
  stacked system factors the other lanes again from the values they hold.
  An analysis with ``deterministic`` or ``hybrid_memory``, which cuDSS's
  uniform batch refuses, runs a native batch lane by lane in either layout,
  each lane on a factorization of its own, with the same results.
- An empty factor ([empty_factor][diff_soup.linear.empty_factor]) has the
  handle ``(context id, -1, -1)``: slot ``-1`` holds nothing. A solve on an
  empty lane returns ``nan`` with ``SolveStatus.EMPTY`` without calling
  cuDSS, an invalidation of it does nothing, and a refactor whose predicate
  holds factors it into the refactor call site's own slot (the generation
  of that lane goes from ``-1`` to the slot's first one).

Every native call takes the factor's order word as its last operand and
returns it as its last result, so the factorizations, solves,
refactorizations and invalidations of a factor that is passed from one
operation to the next run in that order. A handler finds its slot in one of
two ways:

- Static dispatch. When the compiled program states the identity, the
  lowering passes it as call attributes and the handler finds the slot on
  the host without reading any device operand. A factor call whose context
  id and lane token are constants of the program (as they are when a
  jitted function factors in a context it closes over) passes
  ``static_context`` and ``static_lane``. A solve, refactor or invalidation
  whose handle is the direct result of such a factor call, or of a refactor
  call whose own held handle resolves the same way, passes ``static_path``
  (the factor call's site, then each refactor call's site in order) with
  the factor's context and lane ([handle_path][]); such a refactor still
  reads its predicate. A static dispatch is counted as
  ``static_dispatches`` in [contract_stats][].
- Dynamic dispatch. Every other handle (one carried through a loop, passed
  into the program, computed, or produced under ``lax.map`` lanes or a
  traced context) is read back to the host before cuDSS is called: one
  device-to-host copy and one stream synchronisation per call, counted as
  ``host_reads``. The attributes are then ``-1`` and an empty path.

A solve does no other host traffic. A factorization waits for the analysis
phase it runs and for its verdict (the factor's pivots), and a refactorization
reads its predicate. A refactorization does not wait for the device: a solve
takes the factor's status from words the device holds (a zero or non-finite
pivot, or non-finite values, give a nan solution and SINGULAR or NONFINITE),
and the host reads the pivots for the full verdict, the ratio of the smallest
to the largest included, at the next call on the factor that reads the host
anyway (a refactor, a new factorization of it, [contract_stats][]), counted as
``deferred_status_reads``. Until then a refactorization whose only fault is
that ratio solves to its finite solution, and one that ends singular is
factored afresh by the call that settles it. cuDSS's own
solve phase waits on the stream inside the library, so a solve returns to the
host only once its result is ready, with or without a read of ours:
``host_reads == 0`` means that no read of ours synchronises, not that the
solve overlaps with work queued after it.

The FFI targets are registered with JAX the first time this route is used,
so importing the package with an extension that predates the contract
handlers stays harmless until the contract route is asked for.
"""

from __future__ import annotations

import hashlib
import threading
import weakref
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

from diff_soup._errors import ConfigurationError
from diff_soup._jax_compat import ir
from diff_soup._layout import Layout, linear_mode
from diff_soup.linear._factor import (
    Analysis,
    FactorContractError,
    SolveContext,
    dirichlet_rows,
)
from diff_soup.linear._primitive import (
    CALL_FENCE_ATTRIBUTE,
    empty_slot_handle,
    lower_ffi_call,
    lowered_site_key,
    storage_key,
)

__all__ = [
    "CSRPattern",
    "analyze_hook",
    "assemble_csr",
    "contract_analysis_info",
    "contract_stats",
    "csr_pattern",
    "cudss_contract_available",
    "declare_storage",
    "empty_handle",
    "factor_avals",
    "handle_path",
    "lower_apply",
    "lower_factor",
    "lower_invalidate",
    "lower_refactor",
    "release",
    "reset_contract_stats",
    "resident_bytes",
    "set_analysis_seeding",
    "static_int",
]


_ALG_NAME_TO_ID = {
    "default": 0,
    "alg0": 0,
    "alg1": 1,
    "alg2": 2,
    "alg3": 3,
    "alg4": 4,
    "alg5": 5,
}

# The public ``reorder`` option names map onto cuDSS's reordering algorithm
# ids (cudssReorderingAlg_t): 'metis' is the METIS-style nested dissection,
# 'amd' the approximate-minimum-degree ordering.
_REORDER_NAME_TO_ID = {
    "default": 0,
    "btf_colamd": 1,
    "colamd": 2,
    "amd": 3,
    "metis": 4,
    "nested_dissection": 4,
    "none": 5,
}


def _resolve_reordering_alg(opts: dict) -> int:
    """Resolve the reordering algorithm id from the analyze options.

    An explicit ``"reordering_alg"`` (integer or generic algorithm name) wins;
    otherwise the public ``"reorder"`` name (``'amd'`` / ``'metis'`` from
    [CuDSSOptions][diff_soup.linear.CuDSSOptions]) is translated to its cuDSS
    algorithm id. With neither set, ``-1`` leaves the library default in
    place. An unknown ``"reorder"`` name is rejected loudly rather than
    silently ignored.
    """
    explicit = opts.get("reordering_alg")
    if explicit is not None:
        return _parse_cudss_alg_id(explicit)
    reorder = opts.get("reorder")
    if reorder is None:
        return -1
    key = str(reorder).strip().lower()
    if key not in _REORDER_NAME_TO_ID:
        raise ValueError(
            f"Unsupported cuDSS reorder name {reorder!r}. Expected one of "
            f"{sorted(_REORDER_NAME_TO_ID)}."
        )
    return _REORDER_NAME_TO_ID[key]


def _parse_cudss_alg_id(value: object, *, default: int = -1) -> int:
    """Parse a cuDSS algorithm id from a name or integer option value."""
    if value is None:
        return int(default)
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _ALG_NAME_TO_ID:
            raise ValueError(
                f"Unsupported cuDSS algorithm name {value!r}. "
                f"Expected one of {sorted(_ALG_NAME_TO_ID)} or an integer."
            )
        return _ALG_NAME_TO_ID[key]
    return int(value)


#: Options [analyze][diff_soup.linear.analyze] accepts for cuDSS.
_OPTION_NAMES = (
    "reorder",
    "hybrid_memory",
    "hybrid_device_memory_limit_mb",
    "deterministic",
    "ir_nsteps",
    "symmetric_factorization",
    "zero_fill",
    "downcast_factors",
    "emulation_mantissa_bits",
    "batch_layout",
)

#: ``batch_layout`` values -> the native layout id.
_BATCH_LAYOUTS = {"stacked": 0, "uniform": 1}

#: ``downcast_factors`` values -> cuDSS's cudssDowncastFactorsMode_t.
_DOWNCAST_MODES = {False: 0, True: 1, "block_diagonal": 1, "strict_diagonal": 2}

#: Fill-reducing orderings the contract route accepts, by option value.
_REORDERINGS = ("amd", "metis")

#: Precision rung -> (FFI target suffix, dtype of the CSR values, ``b`` and ``x``).
_RUNGS: dict[str, tuple[str, Any]] = {
    "fp64": ("f64", np.dtype(np.float64)),
    "fp32": ("f32", np.dtype(np.float32)),
}

_TARGET_PREFIX = "diff_soup_cudss_contract_"
#: The factor, solve and refactor handlers take the operator values in the
#: coordinate order of the topology (``coo``) and assemble them on the device;
#: the refactor takes both value operands and returns the kept values first.
_FACTOR_KIND, _SOLVE_KIND, _REFACTOR_KIND = "factor_coo", "solve_coo", "refactor_coo"
_TARGET_NAMES = tuple(
    f"{_TARGET_PREFIX}{kind}_{suffix}"
    for kind in (_FACTOR_KIND, _SOLVE_KIND, _REFACTOR_KIND, "invalidate")
    for suffix in ("f64", "f32")
)
_FACTOR_TARGETS = frozenset(f"{_TARGET_PREFIX}{_FACTOR_KIND}_{s}" for s in ("f64", "f32"))
_REFACTOR_TARGETS = frozenset(f"{_TARGET_PREFIX}{_REFACTOR_KIND}_{s}" for s in ("f64", "f32"))
#: The operand positions of a refactor call's held handle: after the ordering
#: token, the values, the held values and the predicate.
_REFACTOR_HANDLE_OPERANDS = (4, 5, 6)

_REGISTRATION_LOCK = threading.Lock()
_REGISTERED = False


# --------------------------------------------------------------------------- #
# The native extension
# --------------------------------------------------------------------------- #
def _rebuild_command() -> str:
    """How to rebuild the cuDSS extension, as a clause after "build it"."""
    return (
        'as "Installing from source" in the installation guide describes, with the cuDSS 1.x '
        "wheel of the CUDA line and a CUDA compiler of that line in the build environment"
    )


def _native():
    """The loaded cuDSS extension, its contract FFI targets registered with JAX.

    The eight FFI targets of the contract handlers (factor, solve, refactor
    and invalidate, each at both rungs) are registered once per process, on
    the first call.

    Returns:
        The ``_native_solver`` extension module.

    Raises:
        ConfigurationError: If the extension cannot be imported, or was built
            without the contract handlers, with the command that rebuilds it.
    """
    global _REGISTERED
    try:
        from diff_soup.linear._cudss._ffi_loader import native_solver
    except ImportError as exc:
        raise ConfigurationError(
            "The cuDSS factor contract needs the native cuDSS extension, which could not be "
            f"imported ({exc}); build it {_rebuild_command()}."
        ) from exc
    if _REGISTERED:
        return native_solver
    with _REGISTRATION_LOCK:
        if _REGISTERED:
            return native_solver
        handlers_fn = getattr(native_solver, "contract_handlers", None)
        if handlers_fn is None:
            raise ConfigurationError(
                "The cuDSS factor contract needs the contract handlers of the native cuDSS "
                "extension, and the extension built in this checkout has none; rebuild it "
                f"{_rebuild_command()}."
            )
        handlers = dict(handlers_fn())
        missing = [name for name in _TARGET_NAMES if name not in handlers]
        if missing:
            raise ConfigurationError(
                f"The native cuDSS extension provides no handlers for {missing}, which the "
                f"cuDSS factor contract needs; rebuild it {_rebuild_command()}."
            )
        for name in _TARGET_NAMES:
            jax.ffi.register_ffi_target(name, handlers[name], platform="CUDA")
        _REGISTERED = True
    return native_solver


def cudss_contract_available() -> bool:
    """Whether cuDSS can run [factor][diff_soup.linear.factor] and its solves in this process.

    True when JAX sees a CUDA device and the native cuDSS extension loads
    with its contract handlers (which this call registers with JAX).

    Returns:
        ``True`` when the contract route can run, else ``False``.
    """
    try:
        if not jax.devices("cuda"):
            return False
    except Exception:  # noqa: BLE001 - no CUDA backend in this process
        return False
    try:
        _native()
    except ConfigurationError:
        return False
    return True


# --------------------------------------------------------------------------- #
# The host analysis record and the CSR patterns
# --------------------------------------------------------------------------- #
class CSRPattern(NamedTuple):
    """The compressed-row pattern of a topology and the map that assembles values into it.

    Built on the host by [csr_pattern][]. All arrays are read-only numpy.

    Attributes:
        n: The number of rows and columns.
        nnz: The number of coordinate entries of the topology.
        nse: The number of distinct positions, the length of the CSR pattern.
        offsets: ``int32[n + 1]`` row offsets of ``A``.
        columns: ``int32[nse]`` column indices of ``A``, ascending in each row.
        forward_index: ``int32[nnz]``, the CSR position of ``A`` of each
            coordinate entry; duplicate entries share a position.
        zeroed: ``int32`` coordinate entries that lie in a Dirichlet row.
        diagonal_forward: ``int32`` CSR positions of ``A`` of the Dirichlet
            rows' diagonals, each once.
        gather_offsets: ``int32[nse + 1]``; the coordinate entries that add
            into CSR position ``p`` of ``A`` are
            ``gather_entries[gather_offsets[p]:gather_offsets[p + 1]]``.
        gather_entries: ``int32[nnz]``, every coordinate entry once, grouped
            by its CSR position of ``A``; an entry in a Dirichlet row is
            stored as ``-(entry + 1)`` (its value is multiplied by zero).
        diagonal_add: ``uint8[nse]``, ``1`` at the CSR positions of ``A``
            that are a Dirichlet row's diagonal (which receives ``1``);
            empty when the topology has no Dirichlet rows.
    """

    n: int
    nnz: int
    nse: int
    offsets: np.ndarray
    columns: np.ndarray
    forward_index: np.ndarray
    zeroed: np.ndarray
    diagonal_forward: np.ndarray
    gather_offsets: np.ndarray
    gather_entries: np.ndarray
    diagonal_add: np.ndarray


def _frozen(array: np.ndarray) -> np.ndarray:
    """A contiguous read-only ``int32`` copy of ``array``."""
    out = np.ascontiguousarray(array, dtype=np.int32).copy()
    out.flags.writeable = False
    return out


def _gather_map(
    index: np.ndarray, nse: int, zeroed: np.ndarray, diagonal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The device assembly map of the CSR order: who adds into each position, and the ones.

    Args:
        index: ``int32[nnz]``, the CSR position of each coordinate entry.
        nse: The number of CSR positions.
        zeroed: The coordinate entries that lie in a Dirichlet row.
        diagonal: The CSR positions of the Dirichlet rows' diagonals.

    Returns:
        ``(gather_offsets, gather_entries, diagonal_add)`` as
        [CSRPattern][] documents them, read-only; ``diagonal_add`` is empty
        when ``diagonal`` is.
    """
    index = np.asarray(index, np.int64)
    entries = np.argsort(index, kind="stable").astype(np.int64)
    if zeroed.size:
        in_dirichlet_row = np.zeros(index.size, dtype=bool)
        in_dirichlet_row[np.asarray(zeroed, np.int64)] = True
        entries = np.where(in_dirichlet_row[entries], -(entries + 1), entries)
    offsets = np.zeros(nse + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(index, minlength=nse))
    add = np.zeros((nse if diagonal.size else 0,), dtype=np.uint8)
    if diagonal.size:
        add[np.asarray(diagonal, np.int64)] = 1
    add = np.ascontiguousarray(add)
    add.flags.writeable = False
    return _frozen(offsets), _frozen(entries), add


def _compressed(keys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compress row-major position keys ``row * n + col`` into CSR.

    Returns:
        ``(unique_keys, offsets, columns, index)``: the sorted distinct keys,
        the row offsets, the column of each distinct key and the CSR position
        of each input key.
    """
    unique, index = np.unique(keys, return_inverse=True)
    n64 = np.int64(n)
    rows = unique // n64
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(rows, minlength=n))
    return unique, offsets, unique % n64, np.asarray(index).reshape(-1)


def csr_pattern(topology: Any) -> CSRPattern:
    """Build the CSR pattern of ``A`` of a topology and its assembly map.

    Entries are sorted by row, then column; coordinate entries at one
    position share one CSR entry, so ``nse <= nnz``. Dirichlet rows are
    taken from ``topology.bc_indices`` as every backend applies them.

    Args:
        topology: The [SparseTopology][diff_soup.linear.SparseTopology].

    Returns:
        The [CSRPattern][].

    Raises:
        ConfigurationError: If an index lies outside ``[0, n)`` or the pattern
            exceeds the ``int32`` index range cuDSS takes.
    """
    n = int(topology.n_dofs)
    rows = np.asarray(topology.rows).astype(np.int64).reshape(-1)
    cols = np.asarray(topology.cols).astype(np.int64).reshape(-1)
    nnz = int(rows.size)
    if nnz and (min(rows.min(), cols.min()) < 0 or max(rows.max(), cols.max()) >= n):
        raise ConfigurationError(
            f"cuDSS needs every row and column index of the topology in [0, {n}); got indices "
            f"from {int(min(rows.min(), cols.min()))} to {int(max(rows.max(), cols.max()))}."
        )
    n64 = np.int64(n)
    keys, offsets, columns, forward_index = _compressed(rows * n64 + cols, n)
    nse = int(keys.size)
    limit = np.iinfo(np.int32).max
    if n >= limit or nse > limit:
        raise ConfigurationError(
            f"cuDSS indexes rows and entries with int32, so it needs n < {limit} and at most "
            f"{limit} distinct pattern positions; got n = {n} and {nse} positions."
        )
    mask, diagonal = dirichlet_rows(topology)
    zeroed = np.zeros((0,), np.int32) if mask is None else np.flatnonzero(np.asarray(mask))
    diagonal_keys = np.asarray(diagonal, np.int64) * (n64 + 1)
    diagonal_forward = np.searchsorted(keys, diagonal_keys)
    gather_offsets, gather_entries, diagonal_add = _gather_map(
        forward_index, nse, zeroed, diagonal_forward
    )
    return CSRPattern(
        n=n,
        nnz=nnz,
        nse=nse,
        offsets=_frozen(offsets),
        columns=_frozen(columns),
        forward_index=_frozen(forward_index),
        zeroed=_frozen(zeroed),
        diagonal_forward=_frozen(diagonal_forward),
        gather_offsets=gather_offsets,
        gather_entries=gather_entries,
        diagonal_add=diagonal_add,
    )


class _NativeAnalysis:
    """Owner of one native cuDSS analysis id; the native analysis is released with it.

    One instance exists per CSR pattern and set of analysis options in the
    process (see [analyze_hook][]); every host record of that pattern holds it.
    """

    __slots__ = ("native_id", "__weakref__")

    def __init__(self, native_id: int):
        """Own ``native_id``; ``contract_analysis_release`` runs when this object is collected."""
        self.native_id = int(native_id)
        weakref.finalize(self, _release_native_analysis, self.native_id)


class _CuDSSContractAnalysis:
    """Host record of one analysed pattern: the native analysis id, sizes and assembly map.

    The CSR pattern itself lives in the native analysis; the record keeps
    the map that assembles coordinate values into its CSR order (the fields
    of the same names on [CSRPattern][]). Nothing in it depends
    on the precision rung or the structure declaration, so every
    [Analysis][diff_soup.linear.Analysis] of one coordinate pattern and set
    of options holds the same record. The native analysis is released when
    the last record that holds it is collected.
    """

    __slots__ = (
        "_native",
        "native_id",
        "n",
        "nnz",
        "nse",
        "forward_index",
        "zeroed",
        "diagonal_forward",
        "reorder",
        "__weakref__",
    )

    def __init__(self, native: _NativeAnalysis, pattern: CSRPattern, reorder: str):
        """Record the assembly maps of ``pattern`` over the native analysis ``native``.

        Args:
            native: The owner of the native analysis of ``pattern``'s CSR structure.
            pattern: The pattern the native analysis was created from.
            reorder: The fill-reducing ordering's name (``"default"`` for
                cuDSS's own choice).
        """
        self._native = native
        self.native_id = native.native_id
        self.n = int(pattern.n)
        self.nnz = int(pattern.nnz)
        self.nse = int(pattern.nse)
        self.forward_index = pattern.forward_index
        self.zeroed = pattern.zeroed
        self.diagonal_forward = pattern.diagonal_forward
        self.reorder = reorder

    def __repr__(self) -> str:
        """Name the native id, the sizes and the ordering."""
        return (
            f"_CuDSSContractAnalysis(id={self.native_id}, n={self.n}, nnz={self.nnz}, "
            f"nse={self.nse}, reorder={self.reorder!r})"
        )


def _release_native_analysis(native_id: int) -> None:
    """Release a native analysis (its host record and any device copies of its patterns)."""
    try:
        _native().contract_analysis_release(native_id)
    except Exception:  # noqa: BLE001 - interpreter shutdown or a missing module
        pass


def assemble_csr(record: Any, values: Any) -> jax.Array:
    """Assemble coordinate-order values into the CSR values of ``A_bc``.

    Duplicate coordinate entries add into their shared CSR entry. Every
    entry of a Dirichlet row is multiplied by zero (so a non-finite value
    stays non-finite and the factorization reports it) and ``1`` is added
    once at the row's diagonal. The result is in the CSR order of ``A``.
    Plain JAX operations, the reference of the assembly the native handlers
    run on the device from the same map: it runs on the host and under
    ``jit`` and accepts leading lane axes.

    Args:
        record: The analysis record (``analysis.backend_analysis`` of a cuDSS
            [Analysis][diff_soup.linear.Analysis]) or a [CSRPattern][].
        values: Coordinate-order values of shape ``(*lanes, nnz)``.

    Returns:
        The CSR values, shape ``(*lanes, nse)``, in the dtype of ``values``.

    Raises:
        FactorContractError: If ``values`` does not end in the pattern length.
    """
    values = jnp.asarray(values)
    nnz, nse = int(record.nnz), int(record.nse)
    if values.ndim < 1 or values.shape[-1] != nnz:
        raise FactorContractError(
            f"assemble_csr needs values of shape (*lanes, {nnz}), one per coordinate entry of "
            f"the topology; got shape {list(values.shape)}."
        )
    index = record.forward_index
    diagonal = record.diagonal_forward
    coef = values
    if record.zeroed.size:
        coef = coef.at[..., record.zeroed].multiply(
            jnp.zeros((), values.dtype), indices_are_sorted=True, unique_indices=True
        )
    out = jnp.zeros(values.shape[:-1] + (nse,), values.dtype)
    out = out.at[..., index].add(coef, unique_indices=nse == nnz)
    if diagonal.size:
        out = out.at[..., diagonal].add(
            jnp.ones((), values.dtype), indices_are_sorted=True, unique_indices=True
        )
    return out


# --------------------------------------------------------------------------- #
# analyze: options, layout and the native analysis
# --------------------------------------------------------------------------- #
def _is_bool(value: Any) -> bool:
    """Whether ``value`` is a Python or numpy bool."""
    return isinstance(value, (bool, np.bool_))


def _is_int(value: Any) -> bool:
    """Whether ``value`` is an integer that is not a bool."""
    return isinstance(value, (int, np.integer)) and not _is_bool(value)


def _parse_options(options: tuple, layout: Layout) -> dict[str, Any]:
    """Validate the options [analyze][diff_soup.linear.analyze] passes to cuDSS.

    An option left out takes the layout's setting where the layout has one,
    as the cuDSS backend's other entry points do; an explicit option wins.
    For ``reorder`` that is a layout ``reorder`` other than the default
    ``"amd"``: the default layout leaves cuDSS's own ordering in place, while
    ``options={"reorder": "amd"}`` selects AMD. ``hybrid_memory`` and
    ``hybrid_device_memory_limit_mb`` follow the layout's settings.

    Args:
        options: The frozen ``(key, value)`` pairs.
        layout: The validated layout.

    Returns:
        ``reorder`` (the name, ``"default"`` for cuDSS's own ordering),
        ``reorder_alg`` (the cuDSS algorithm id, ``-1`` for its default),
        ``hybrid_mode`` (0 or 1), ``hybrid_device_memory_limit_mb`` (``-1``
        for no limit), ``deterministic`` (0 or 1), ``ir_nsteps`` (``-1``
        when left out: the factor's default), ``symmetric_factorization``
        (0 or 1), ``zero_fill`` (0 or 1, ``-1`` when left out),
        ``downcast_factors`` (cuDSS's downcast mode, 0 for none),
        ``emulation_mantissa_bits`` (0 for no emulation) and ``batch_layout``
        (0 for ``"stacked"``, the default, 1 for ``"uniform"``).

    Raises:
        ConfigurationError: For an unknown option or an invalid value.
    """
    opts = dict(options)
    unknown = sorted(set(opts) - set(_OPTION_NAMES))
    if unknown:
        raise ConfigurationError(
            f"cuDSS does not take the options {unknown}; it accepts {list(_OPTION_NAMES)}."
        )

    reorder = opts.get("reorder")
    if reorder is None and layout.reorder != "amd":
        reorder = layout.reorder
    if reorder is not None and reorder not in _REORDERINGS:
        raise ConfigurationError(
            f"cuDSS option reorder must be one of {list(_REORDERINGS)}; got {reorder!r}."
        )
    reorder_alg = _resolve_reordering_alg({} if reorder is None else {"reorder": reorder})

    hybrid = opts.get("hybrid_memory", layout.hybrid_memory)
    if not _is_bool(hybrid):
        raise ConfigurationError(
            f"cuDSS option hybrid_memory must be True or False; got {hybrid!r}."
        )
    limit = opts.get(
        "hybrid_device_memory_limit_mb",
        layout.hybrid_device_memory_limit_mb if bool(hybrid) else None,
    )
    if limit is not None:
        if not _is_int(limit) or int(limit) <= 0:
            raise ConfigurationError(
                "cuDSS option hybrid_device_memory_limit_mb must be None or a positive integer "
                f"(MB); got {limit!r}."
            )
        if not bool(hybrid):
            raise ConfigurationError(
                "cuDSS option hybrid_device_memory_limit_mb needs hybrid_memory=True; got "
                f"hybrid_device_memory_limit_mb={limit!r} with hybrid_memory=False."
            )

    deterministic = opts.get("deterministic", False)
    if not _is_bool(deterministic):
        raise ConfigurationError(
            f"cuDSS option deterministic must be True or False; got {deterministic!r}."
        )
    downcast = opts.get("downcast_factors")
    if downcast is not None and (
        not (_is_bool(downcast) or isinstance(downcast, str)) or downcast not in _DOWNCAST_MODES
    ):
        raise ConfigurationError(
            "cuDSS option downcast_factors must be None, False, True, 'block_diagonal' or "
            f"'strict_diagonal'; got {downcast!r}."
        )
    if downcast is not None and _DOWNCAST_MODES[downcast] and deterministic:
        raise ConfigurationError(
            "cuDSS option downcast_factors cannot be combined with deterministic=True: cuDSS "
            "does not downcast factors in deterministic mode. Drop one of the two."
        )
    # Left out (or None), ir_nsteps and zero_fill go to the native layer as -1, which picks
    # them per factor (one analysis serves both rungs): see RefinementSteps and ZeroFills in
    # contract_handlers.inc.
    ir_nsteps = opts.get("ir_nsteps")
    if ir_nsteps is None:
        ir_nsteps = -1
    elif not _is_int(ir_nsteps) or int(ir_nsteps) < 0:
        raise ConfigurationError(
            f"cuDSS option ir_nsteps must be None or a non-negative integer; got {ir_nsteps!r}."
        )
    if downcast is None:
        downcast_mode = 0
    else:
        downcast_mode = _DOWNCAST_MODES[downcast]
    if deterministic and int(ir_nsteps) > 0:
        raise ConfigurationError(
            "cuDSS deterministic mode runs without iterative refinement, so it needs "
            f"ir_nsteps=0 or ir_nsteps left out; got ir_nsteps={ir_nsteps!r} with "
            "deterministic=True."
        )
    symmetric_factorization = opts.get("symmetric_factorization", False)
    if not _is_bool(symmetric_factorization):
        raise ConfigurationError(
            "cuDSS option symmetric_factorization must be True or False; got "
            f"{symmetric_factorization!r}."
        )
    if symmetric_factorization and deterministic:
        # measured: cuDSS 1.0.0.35's deterministic symmetric factorization solves to non-finite
        # values on an RTX 4070 Ti, where the general LU of the same matrix is exact.
        raise ConfigurationError(
            "cuDSS option symmetric_factorization cannot be combined with deterministic=True: "
            "cuDSS's deterministic symmetric factorization does not solve accurately. Drop one "
            "of the two."
        )
    zero_fill = opts.get("zero_fill")
    if zero_fill is not None and not _is_bool(zero_fill):
        raise ConfigurationError(
            f"cuDSS option zero_fill must be None, True or False; got {zero_fill!r}."
        )
    bits = opts.get("emulation_mantissa_bits")
    if bits is not None and (not _is_int(bits) or not 1 <= int(bits) <= 64):
        raise ConfigurationError(
            "cuDSS option emulation_mantissa_bits must be None or an integer from 1 to 64; got "
            f"{bits!r}."
        )
    batch_layout = opts.get("batch_layout", "stacked")
    if not isinstance(batch_layout, str) or batch_layout not in _BATCH_LAYOUTS:
        raise ConfigurationError(
            f"cuDSS option batch_layout must be one of {list(_BATCH_LAYOUTS)}; got "
            f"{batch_layout!r}."
        )
    if downcast_mode and batch_layout == "uniform":
        # measured: cuDSS 1.0.0.35 returns CUDSS_STATUS_NOT_SUPPORTED for a factorization,
        # refactorization or solve under a partial CUDSS_DATA_UBATCH_MASK of a downcast uniform
        # batch, which a batched refactor and a batch with a failed lane both run.
        raise ConfigurationError(
            "cuDSS option downcast_factors needs the stacked batch_layout: cuDSS cannot "
            "factor or solve some lanes of a uniform batch whose factors are downcast, which "
            "batched refactors and batches with a failed lane do. Use batch_layout='stacked' "
            f"(the default), or leave downcast_factors out or False; got downcast_factors="
            f"{downcast!r} with batch_layout='uniform'."
        )
    return {
        "reorder": "default" if reorder is None else str(reorder),
        "reorder_alg": int(reorder_alg),
        "hybrid_mode": int(bool(hybrid)),
        "hybrid_device_memory_limit_mb": -1 if limit is None else int(limit),
        "deterministic": int(bool(deterministic)),
        "ir_nsteps": int(ir_nsteps),
        "symmetric_factorization": int(bool(symmetric_factorization)),
        "zero_fill": -1 if zero_fill is None else int(bool(zero_fill)),
        "downcast_factors": downcast_mode,
        "emulation_mantissa_bits": 0 if bits is None else int(bits),
        "batch_layout": _BATCH_LAYOUTS[batch_layout],
    }


#: Native analysis owners by (n, nse, CSR structure digest, options), and host records by
#: that key plus the coordinate maps. Weak values: an entry lives while some
#: [Analysis][diff_soup.linear.Analysis] holds its record.
_NATIVE_ANALYSES: "weakref.WeakValueDictionary[tuple, _NativeAnalysis]" = (
    weakref.WeakValueDictionary()
)
_RECORDS: "weakref.WeakValueDictionary[tuple, _CuDSSContractAnalysis]" = (
    weakref.WeakValueDictionary()
)
_MEMO_LOCK = threading.Lock()


def _digest(*arrays: np.ndarray) -> bytes:
    """A content digest of int32 arrays, sensitive to their lengths and order."""
    h = hashlib.blake2b(digest_size=32)
    for array in arrays:
        h.update(np.int64(array.size).tobytes())
        h.update(np.ascontiguousarray(array, dtype=np.int32).tobytes())
    return h.digest()


def analyze_hook(
    topology: Any, options: tuple, layout: Layout, precision: str
) -> _CuDSSContractAnalysis:
    """Build the CSR pattern of ``topology`` and the native cuDSS analysis record.

    The record is shared: every call with the same coordinate pattern and
    options returns the same record, which holds one native analysis (the
    CSR pattern and the map that assembles coordinate values into it,
    uploaded to the device on first use), whatever the precision rung or
    the ``symmetric``/``spd`` declaration of the
    [Analysis][diff_soup.linear.Analysis] that asks. The cuDSS symbolic
    analysis itself (the fill-reducing reordering) runs on the device on the
    first factorization of the pattern in the process; every later
    factorization of that pattern without matching (``spd`` slots and the
    tiles of native batches), at either rung, starts from that ordering.

    Args:
        topology: The sparsity pattern; its Dirichlet rows have their
            diagonal entry in the pattern.
        options: Frozen backend options: ``reorder`` (``"amd"`` or
            ``"metis"``), ``hybrid_memory``, ``hybrid_device_memory_limit_mb``,
            ``deterministic``, ``ir_nsteps``, ``symmetric_factorization``,
            ``zero_fill``, ``downcast_factors``, ``emulation_mantissa_bits``,
            ``batch_layout``.
        layout: The validated layout; one device per factorization.
        precision: The validated precision rung, ``"fp64"`` or ``"fp32"``.

    Returns:
        The host record of the native analysis.

    Raises:
        ConfigurationError: For an unknown or invalid option, a layout that
            spreads a factorization over several devices, or a missing native
            extension or one built without the contract handlers.
    """
    del precision  # the rung selects the FFI target at lowering time
    parsed = _parse_options(options, layout)
    mode = linear_mode(layout)
    if mode != "local":
        raise ConfigurationError(
            "cuDSS's analyze/factor/solve route factors on one device; the layout spreads a "
            f"factorization over a model axis ({mode!r}). Pass a layout without a 'model' mesh "
            "axis."
        )
    native = _native()
    pattern = csr_pattern(topology)
    options_key = tuple(sorted(parsed.items()))
    # The native analysis holds the CSR pattern and the assembly map, so one is kept per
    # coordinate pattern (not per CSR pattern) and set of options.
    native_key = (
        pattern.n,
        pattern.nse,
        pattern.nnz,
        _digest(
            pattern.offsets,
            pattern.columns,
            pattern.forward_index,
            pattern.zeroed,
            pattern.diagonal_forward,
        ),
        options_key,
    )
    record_key = native_key
    with _MEMO_LOCK:
        record = _RECORDS.get(record_key)
        if record is not None:
            return record
        holder = _NATIVE_ANALYSES.get(native_key)
        if holder is None:
            try:
                native_id = native.contract_analysis_create(
                    pattern.n,
                    pattern.offsets,
                    pattern.columns,
                    pattern.nnz,
                    pattern.gather_offsets,
                    pattern.gather_entries,
                    pattern.diagonal_add,
                    parsed["reorder_alg"],
                    parsed["hybrid_mode"],
                    parsed["hybrid_device_memory_limit_mb"],
                    parsed["deterministic"],
                    parsed["ir_nsteps"],
                    parsed["symmetric_factorization"],
                    parsed["zero_fill"],
                    parsed["downcast_factors"],
                    parsed["emulation_mantissa_bits"],
                    batch_layout=parsed["batch_layout"],
                )
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
            holder = _NativeAnalysis(native_id)
            _NATIVE_ANALYSES[native_key] = holder
        record = _CuDSSContractAnalysis(holder, pattern, parsed["reorder"])
        _RECORDS[record_key] = record
        return record


# --------------------------------------------------------------------------- #
# Static dispatch: what the lowered program states about a call's identity
# --------------------------------------------------------------------------- #
#: Operations whose single operand reaches their result with its integer value
#: unchanged: an integer conversion (checked against the result's range), a
#: broadcast and a reshape.
_VALUE_KEEPING_OPS = frozenset(
    {"stablehlo.convert", "stablehlo.broadcast_in_dim", "stablehlo.reshape"}
)

#: How many producing operations a static lookup follows before it gives up.
_MAX_CHAIN = 64


def _producer(value: Any) -> Optional[Any]:
    """The operation whose result ``value`` is; ``None`` for a block argument or on doubt."""
    try:
        owner = value.owner
        if owner is None or isinstance(owner, ir.Block):
            return None
        return owner.operation if hasattr(owner, "operation") else owner
    except Exception:  # noqa: BLE001 - introspection must never fail a lowering
        return None


def _integer_width(value: Any) -> Optional[int]:
    """The bit width of ``value``'s integer element type; ``None`` for any other type."""
    try:
        width = int(ir.IntegerType(ir.ShapedType(value.type).element_type).width)
    except Exception:  # noqa: BLE001 - not an integer tensor
        return None
    return width if width in (8, 16, 32, 64) else None


def _single_integer(attribute: Any) -> Optional[int]:
    """The one integer a dense constant holds (a scalar, or every element equal), else ``None``."""
    dense = ir.DenseIntElementsAttr(attribute)
    if dense.is_splat or len(dense) == 1:
        return int(dense[0])
    return None


def static_int(value: Any) -> Optional[int]:
    """The integer an operand of a lowered call holds when the program states it as a constant.

    The rule: follow the operand back to the operation that produced it,
    through integer conversions, broadcasts and reshapes (each keeps the
    value); when that operation is a ``stablehlo.constant`` holding one
    integer (a scalar, or an array whose elements are all equal), return it.
    A jaxpr literal, such as the NumPy ``int32`` scalar
    [factor][diff_soup.linear.factor] binds for a context id and the lane
    token, lowers to such a constant. Everything else is dynamic and gives
    ``None``: a block argument (an argument of the compiled function or of
    a loop or branch body, so also every ``lax.map`` lane and every value
    passed into ``jit``), any other producing operation, a constant that is
    not an integer or holds differing elements, and a value outside the
    range of an integer type it passes through. Never raises.

    Args:
        value: The MLIR value of an operand, as a lowering rule receives it.

    Returns:
        The integer, or ``None`` when the value is not a program constant.
    """
    try:
        widths = []
        for _ in range(_MAX_CHAIN):
            width = _integer_width(value)
            op = _producer(value)
            if width is None or op is None:
                return None
            widths.append(width)
            if op.name == "stablehlo.constant":
                number = _single_integer(op.attributes["value"])
                if number is None:
                    return None
                if any(not -(2 ** (w - 1)) <= number < 2 ** (w - 1) for w in widths):
                    return None
                return number
            if op.name not in _VALUE_KEEPING_OPS or len(op.operands) != 1:
                return None
            value = op.operands[0]
    except Exception:  # noqa: BLE001 - introspection must never fail a lowering
        return None
    return None


def _backend_config(op: Any) -> Any:
    """The attribute dictionary an FFI custom call carries its handler attributes in."""
    for name in ("backend_config", "mhlo.backend_config"):
        if name in op.attributes:
            return ir.DictAttr(op.attributes[name])
    return None


def _config_int(config: Any, key: str) -> Optional[int]:
    """An integer handler attribute of a custom call, ``None`` when it is absent or not an integer."""
    if config is None or key not in config:
        return None
    return int(ir.IntegerAttr(config[key]).value)


def _is_call_fence(op: Any) -> bool:
    """Whether ``op`` is an optimization barrier the contract lowering put around a native call."""
    try:
        return op.name == "stablehlo.optimization_barrier" and (
            CALL_FENCE_ATTRIBUTE in op.attributes
        )
    except Exception:  # noqa: BLE001 - introspection must never fail a lowering
        return False


def _unwrap(value: Any) -> Any:
    """``value``, or the value a chain of reshapes and call fences produced it from.

    A reshape keeps the lane order; a call fence (the optimization barrier
    around a native call, [lower_ffi_call][]) passes each operand
    to the result at the same position unchanged.
    """
    for _ in range(_MAX_CHAIN):
        op = _producer(value)
        if op is None:
            return value
        if op.name == "stablehlo.reshape" and len(op.operands) == 1:
            value = op.operands[0]
        elif _is_call_fence(op):
            value = op.operands[_result_number(value)]
        else:
            return value
    return value


def _result_number(value: Any) -> int:
    """The position of an operation result among its operation's results."""
    try:
        return int(value.result_number)
    except AttributeError:
        return int(ir.OpResult(value).result_number)


#: The result position of a call's first handle leaf, after the XLA token:
#: 2 for a refactor, which returns the kept values first, else 1.
_FIRST_HANDLE_RESULT: dict[str, int] = {name: 2 for name in _REFACTOR_TARGETS}


def _handle_call(leaves: tuple) -> Optional[Any]:
    """The one contract call whose consecutive handle results the leaves are, else ``None``.

    The handle results follow the XLA token, and on a refactor call
    also the kept values, which a refactor returns first.
    """
    call = None
    first = 1
    for offset, leaf in enumerate(leaves):
        leaf = _unwrap(leaf)
        op = _producer(leaf)
        if op is None or op.name != "stablehlo.custom_call":
            return None
        if call is None:
            call = op
            target = ir.StringAttr(op.attributes["call_target_name"]).value
            first = _FIRST_HANDLE_RESULT.get(target, 1)
        elif op != call:
            return None
        if _result_number(leaf) != first + offset:
            return None
    return call


def _resolve(leaves: tuple, analysis_id: Optional[int], depth: int) -> Optional[tuple]:
    """``(path, context, lane)`` of a handle by the rule of [handle_path][], else ``None``."""
    if depth > _MAX_CHAIN:
        return None
    call = _handle_call(leaves)
    if call is None:
        return None
    target = ir.StringAttr(call.attributes["call_target_name"]).value
    config = _backend_config(call)
    site = _config_int(config, "site")
    if site is None or site < 0:
        return None
    if analysis_id is not None and _config_int(config, "analysis_id") != analysis_id:
        return None
    if target in _FACTOR_TARGETS:
        context = _config_int(config, "static_context")
        lane = _config_int(config, "static_lane")
        if context is None or lane is None or context < 0 or lane < 0:
            return (site,), -1, -1
        return (site,), context, lane
    if target in _REFACTOR_TARGETS:
        operands = list(call.operands)
        if len(operands) <= max(_REFACTOR_HANDLE_OPERANDS):
            return None
        held = tuple(operands[i] for i in _REFACTOR_HANDLE_OPERANDS)
        inner = _resolve(held, analysis_id, depth + 1)
        if inner is None or inner[1] < 0:
            return (site,), -1, -1
        path, context, lane = inner
        return path + (site,), context, lane
    return None


def handle_path(
    context: Any, slot: Any, generation: Any, *, analysis_id: Optional[int] = None
) -> Optional[tuple[tuple[int, ...], int, int]]:
    """The static key of a held factor's handle, when the lowered program states it.

    The rule, for the three handle operands ``(context id, slot,
    generation)`` of a solve, refactor or invalidation:

    - they must be results 1, 2 and 3 (after the XLA token) of one
      custom call of a contract factor target, or results 2, 3 and 4 of one
      of a refactor target (which returns the kept values first), each directly or
      through reshapes, which keep the lane order (a nested ``vmap`` folds
      its lanes with reshapes), and through the optimization barriers the
      contract lowering puts around every native call, which are identities
      (any other optimization barrier makes the handle dynamic);
    - a factor call ends the walk: it contributes its ``site`` and its
      ``static_context`` and ``static_lane`` attributes; when both are set
      the handle's key is static;
    - a refactor call contributes its ``site`` after the path of its own
      held handle, which is resolved by the same rule;
    - with ``analysis_id`` given, every call on the way must carry that
      ``analysis_id``.

    When the walk reaches a factor call whose context id or lane token was
    not a program constant, or a refactor call whose held handle is
    dynamic, the key is unknown but the handle is still the result of one
    call of this program: the path is then that producing call's site
    alone with the context and lane ``-1``, and the handler takes the
    handle words that call wrote in the same execution of the program
    instead of reading them back from the device.

    Anything else is dynamic: a block argument (a handle carried through a
    loop, passed into ``jit`` or into a branch), a handle computed by any
    other operation (a slice, a broadcast, a select), or a call of another
    target. Never raises.

    Args:
        context: The MLIR value of the handle's context id leaf.
        slot: The MLIR value of the slot leaf.
        generation: The MLIR value of the generation leaf.
        analysis_id: The native analysis id the consuming call acts on.

    Returns:
        ``(path, context, lane)``: the factor call's site followed by each
        refactor call's site in order, and the factor call's context id and
        lane token; the producing call's site alone with ``-1, -1`` when
        only the producer is known; ``None`` when the handle is dynamic.
    """
    try:
        return _resolve((context, slot, generation), analysis_id, 0)
    except Exception:  # noqa: BLE001 - introspection must never fail a lowering
        return None


def _identity_attributes(context: Optional[int], lane: Optional[int]) -> dict[str, np.int64]:
    """``static_context`` and ``static_lane`` for a factor call; ``-1`` is dynamic."""
    if context is None or lane is None or context < 0 or lane < 0:
        context = lane = -1
    return {"static_context": np.int64(context), "static_lane": np.int64(lane)}


def _path_attributes(resolved: Optional[tuple]) -> dict[str, Any]:
    """``static_path``, ``static_context`` and ``static_lane`` for a call that consumes a handle.

    An empty path with ``-1`` for both means the handler reads the handle; a
    path with ``-1`` for both names the producing call whose result words
    the handler takes in the same execution.
    """
    if resolved is None:
        return {
            "static_path": np.zeros((0,), np.int64),
            "static_context": np.int64(-1),
            "static_lane": np.int64(-1),
        }
    path, context, lane = resolved
    return {
        "static_path": np.asarray(path, np.int64),
        "static_context": np.int64(context),
        "static_lane": np.int64(lane),
    }


def _storage_attributes(storage: Optional[str], context: Any) -> dict[str, np.int64]:
    """``storage`` and ``storage_context`` for a call on a factor that names shared storage.

    ``storage`` is the name's key ([storage_key][diff_soup.linear._primitive.storage_key],
    ``0`` for none) and ``storage_context`` the handle's context id when the
    program states it as a constant ([static_int][]), else ``-1``.
    """
    if storage is None:
        return {}
    known = static_int(context)
    return {
        "storage": storage_key(storage),
        "storage_context": np.int64(-1 if known is None or known < 0 else known),
    }


def _call_shapes(ctx: Any) -> tuple:
    """The shapes and dtypes of a call's operands, which tell calls of equal programs apart."""
    return tuple((tuple(aval.shape), str(aval.dtype)) for aval in ctx.avals_in)


def declare_storage(analysis: Analysis, context_id: int, storage: str) -> None:
    """Record that a factor of the context names the shared storage ``storage`` on ``analysis``.

    Called by [empty_factor][diff_soup.linear.empty_factor]. A call on such a
    factor whose context id is not a program constant (a loop carry) then
    finds the storage without reading the handle, as long as only one open
    context has declared that name on the analysis; with several it reads
    the handle's context id.
    """
    record = _record(analysis)
    _native().contract_storage_declare(
        int(record.native_id), int(context_id), int(storage_key(storage))
    )


# --------------------------------------------------------------------------- #
# factor / solve / refactor / invalidate lowerings
# --------------------------------------------------------------------------- #
def _record(analysis: Analysis) -> _CuDSSContractAnalysis:
    """The cuDSS host record behind ``analysis``."""
    record = analysis.backend_analysis
    if not isinstance(record, _CuDSSContractAnalysis):
        raise FactorContractError(
            "The cuDSS contract hooks need an analysis built by "
            f"diff_soup.linear.analyze(..., backend=CuDSS()); got a record {record!r}."
        )
    return record


def _check_real(dtype: Any, precision: str, what: str = "values") -> None:
    """Refuse values or right-hand sides that are not float32 or float64.

    The native handlers assemble float32 and float64 values at either rung
    (casting to the rung's dtype); complex and half-precision arrays are not
    taken.

    Raises:
        FactorContractError: For any other dtype.
    """
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise FactorContractError(
            f"cuDSS takes float32 or float64 {what} (the {precision} rung casts them to its "
            f"dtype); got {np.dtype(dtype).name}. Cast the {what} before the call."
        )


def _rung(precision: str) -> tuple[str, np.dtype]:
    """The FFI target suffix and the value dtype of a precision rung."""
    try:
        return _RUNGS[precision]
    except KeyError:
        raise FactorContractError(
            f"cuDSS factors at the precision rungs {sorted(_RUNGS)}; got {precision!r}."
        ) from None


def factor_avals(
    analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
) -> tuple:
    """The handle: three ``int32`` scalars, the context id, the slot and its generation.

    Raises:
        FactorContractError: If the values are not real.
    """
    del analysis, orientation, tol
    _check_real(values_aval.dtype, precision)
    scalar = ShapedArray((), jnp.int32)
    return (scalar, scalar, scalar)


def empty_handle(
    analysis: Analysis,
    values_aval: Any,
    *,
    context_id: int,
    orientation: str,
    precision: str,
    tol: Any,
) -> tuple:
    """The handle of an empty factor: ``(context id, -1, -1)``, slot ``-1`` holding nothing.

    Raises:
        FactorContractError: If the values are not real.
    """
    del analysis, orientation, tol
    _check_real(values_aval.dtype, precision)
    return empty_slot_handle(context_id, leaves=3)


def lower_factor(
    ctx: Any,
    values: Any,
    token: Any,
    context_id: Any,
    order: Any,
    *,
    analysis: Analysis,
    orientation: str,
    precision: str,
    tol: Any,
) -> list:
    """Lower a factorization to one FFI call ordered by the order word.

    The call names its native site by a key computed from the lowered call
    ([lowered_site_key][diff_soup.linear._primitive.lowered_site_key]); the
    handler creates the site on the first call with that key, so two
    lowerings of one program share their sites and their program text. The
    values go to the handler in the coordinate order of the topology; it assembles
    them on the device into the CSR values of ``A_bc`` at the rung's
    precision, straight into the slot's value copy, and factors them, for
    either orientation (an ``"AT"`` factor solves in cuDSS's transpose solve
    mode); a batch slot of orientation ``"AT"`` assembles and factors
    ``A_bc^T`` instead, since cuDSS has no transpose mode for batches. At run time the handler
    keeps one slot per ``(context id, slot token)`` at that site; values of
    shape ``(lanes, nnz)`` make that slot a batch slot of ``lanes`` factors.
    When the context id and the slot token are constants of the program
    ([static_int][]), the call carries them as ``static_context`` and
    ``static_lane`` and the handler finds the slot without reading them
    back; otherwise both are ``-1``. No assembly runs in the program: the
    lowered program holds no index constant of the pattern.

    Returns:
        The MLIR values of ``(context id, slot, generation, status, order)``.
    """
    del tol
    record = _record(analysis)
    suffix, _ = _rung(precision)
    _native()
    transposed = orientation == "AT"
    _check_real(ctx.avals_in[0].dtype, precision)
    site = lowered_site_key(
        ctx, "cudss factor", record.native_id, orientation, precision, _call_shapes(ctx)
    )
    results, order = lower_ffi_call(
        ctx,
        f"{_TARGET_PREFIX}{_FACTOR_KIND}_{suffix}",
        order,
        [values, token, context_id],
        list(ctx.avals_in[:3]),
        ctx.avals_out[:-1],
        site=site,
        transposed=bool(transposed),
        symmetric=bool(analysis.symmetric),
        spd=bool(analysis.spd),
        analysis_id=np.int64(record.native_id),
        **_identity_attributes(static_int(context_id), static_int(token)),
    )
    return [*results, order]


def lower_apply(
    ctx: Any,
    values: Any,
    held_context: Any,
    slot: Any,
    generation: Any,
    order: Any,
    b: Any,
    *,
    n_handle: int,
    analysis: Analysis,
    orientation: str,
    transpose: bool,
    precision: str,
    tol: Any,
    storage: Optional[str] = None,
) -> list:
    """Lower a solve to one FFI call ordered by the order word.

    Both orientations solve with the slot's own factorization: a single
    slot factors ``A_bc`` and solves with ``A_bc^T`` (``transpose=True`` on
    an ``"A"`` factor, or ``transpose=False`` on an ``"AT"`` one) in cuDSS's
    transpose solve mode, a native batch slot for every lane at once; an
    analysis declared ``symmetric`` or ``spd`` solves both in normal mode.
    The values travel in the coordinate order of the topology; the handler
    assembles them on the device only when it must factor them: a slot
    whose generation differs from the
    handle's is refactored from them first. ``b`` of shape ``(nrhs, n)`` is
    solved as one block; the
    solution is ``nan`` unless the factorization succeeded. A native batch
    passes values ``(lanes, nnz)``, handle leaves ``(lanes,)`` and ``b`` of
    shape ``(lanes, n)`` or ``(lanes, nrhs, n)``, lane ``k`` solving with
    lane ``k`` of the batch slot. When the handle resolves by the rule of
    [handle_path][], the call carries ``static_path``, ``static_context``
    and ``static_lane`` and the handler finds the slot without reading the
    handle back; otherwise they are an empty path and ``-1``.

    A solve with ``transpose=False`` of a factor that names shared storage
    (``storage``) carries the storage's key and, when the program states it,
    the handle's context id: when a refactor of that storage ran earlier in
    the same execution, the handler uses the storage as it stands without
    reading the handle; otherwise it follows the rule above. A transposed
    solve of such a factor follows the rule above.

    Returns:
        The MLIR values of ``(x, status, order)``.

    Raises:
        FactorContractError: If ``b`` is complex.
    """
    del n_handle, orientation, tol
    _check_real(ctx.avals_in[0].dtype, precision)
    _check_real(ctx.avals_in[-1].dtype, precision, "right-hand sides")
    record = _record(analysis)
    suffix, _ = _rung(precision)
    _native()
    resolved = handle_path(held_context, slot, generation, analysis_id=record.native_id)
    avals = list(ctx.avals_in)
    results, order = lower_ffi_call(
        ctx,
        f"{_TARGET_PREFIX}{_SOLVE_KIND}_{suffix}",
        order,
        [values, held_context, slot, generation, b],
        [*avals[:4], avals[5]],
        ctx.avals_out[:-1],
        transpose=bool(transpose),
        analysis_id=np.int64(record.native_id),
        **_path_attributes(resolved),
        **_storage_attributes(None if transpose else storage, held_context),
    )
    return [*results, order]


def lower_refactor(
    ctx: Any,
    values: Any,
    held_values: Any,
    predicate: Any,
    held_context: Any,
    held_slot: Any,
    held_generation: Any,
    order: Any,
    *,
    n_handle: int,
    analysis: Analysis,
    orientation: str,
    precision: str,
    tol: Any,
    adopt: bool,
    storage: Optional[str] = None,
) -> list:
    """Lower a refactorization where ``predicate`` holds to one ordered FFI call.

    The call names its native site by a key computed from the lowered call,
    as [lower_factor][] does. A held slot that is the site's
    own is refactored in place (a loop that carries a factor keeps one
    slot), and so is any held slot when ``adopt`` is set (the site then
    holds no slot of its own and the older factor that named the slot is
    stale: its next solve rebuilds the slot from its own values); otherwise
    another site's held slot gets the site's slot for it in the handle's
    context, so other holders of that factor keep it, and so does an empty
    handle (slot ``-1``). A false predicate returns the held handle with
    that slot's status (``SolveStatus.EMPTY`` for an empty lane). A native
    batch passes values ``(lanes, nnz)`` and a predicate and handle leaves
    ``(lanes,)``; when any lane's predicate holds, every lane of the batch
    is refactored, a lane that asks from the values the call carries for it
    and a lane that does not from the values its slot already holds. When
    the held handle resolves by the rule of [handle_path][], the call
    carries ``static_path``, ``static_context`` and ``static_lane`` and the
    handler finds the held slot without reading the handle back; otherwise
    they are an empty path and ``-1``. The predicate is always read back to
    the host.

    The call receives ``values`` and ``held_values`` in the coordinate order
    of the topology and the predicate before the handle; the handler
    assembles the values of the lanes that ask on the device, straight into
    the slot's value copy, and writes the kept values (``values`` in a lane
    that asks, ``held_values`` elsewhere) into the buffer of the ``values``
    operand, which it returns as its first result.

    A factor that names shared storage (``storage``) is refactored in that
    storage whichever site holds it: the call carries the storage's key and,
    when the program states it, the handle's context id, and the handler
    assembles the values each lane keeps into the storage while comparing
    them with the values it holds, reads one word per lane (the predicate
    merged with the comparison) and factors the lanes whose values changed;
    a lane that asks for the very values the storage holds keeps its factor,
    which is the one a factorization would make.

    Returns:
        The MLIR values of ``(kept values, context id, slot, generation, status, order)``.
    """
    del n_handle, tol
    record = _record(analysis)
    suffix, _ = _rung(precision)
    _native()
    transposed = orientation == "AT"
    _check_real(ctx.avals_in[0].dtype, precision)
    resolved = handle_path(held_context, held_slot, held_generation, analysis_id=record.native_id)
    site = lowered_site_key(
        ctx,
        "cudss refactor",
        record.native_id,
        orientation,
        precision,
        _call_shapes(ctx),
        bool(adopt),
        storage,
    )
    results, order = lower_ffi_call(
        ctx,
        f"{_TARGET_PREFIX}{_REFACTOR_KIND}_{suffix}",
        order,
        [values, held_values, predicate, held_context, held_slot, held_generation],
        list(ctx.avals_in[:-1]),
        ctx.avals_out[:-1],
        operand_output_aliases={0: 0},
        site=site,
        transposed=bool(transposed),
        symmetric=bool(analysis.symmetric),
        spd=bool(analysis.spd),
        adopt=bool(adopt),
        analysis_id=np.int64(record.native_id),
        **_path_attributes(resolved),
        **_storage_attributes(storage, held_context),
    )
    return [*results, order]


def lower_invalidate(
    ctx: Any,
    held_context: Any,
    slot: Any,
    generation: Any,
    context_id: Any,
    order: Any,
    *,
    n_handle: int,
    analysis: Analysis,
    orientation: str,
    precision: str,
    storage: Optional[str] = None,
) -> list:
    """Lower an invalidation to one FFI call ordered by the order word.

    At run time the slot the handle names (every lane of a batch slot for a
    batched handle) no longer stands for any factor: the next solve on it
    refactors it from the values that solve carries. The context id the
    handle carries is the one used; the separate ``context_id`` operand is
    not passed on. When the handle resolves by the rule of [handle_path][],
    the call carries ``static_path``, ``static_context`` and
    ``static_lane``; otherwise an empty path and ``-1``. A factor that names
    shared storage (``storage``) invalidates that storage, found from its
    key as a solve finds it.

    Returns:
        The MLIR value of the order word, the call's only buffer result.
    """
    del context_id, n_handle, orientation
    record = _record(analysis)
    suffix, _ = _rung(precision)
    _native()
    resolved = handle_path(held_context, slot, generation, analysis_id=record.native_id)
    _, order = lower_ffi_call(
        ctx,
        f"{_TARGET_PREFIX}invalidate_{suffix}",
        order,
        [held_context, slot, generation],
        list(ctx.avals_in[:3]),
        [],
        analysis_id=np.int64(record.native_id),
        **_path_attributes(resolved),
        **_storage_attributes(storage, held_context),
    )
    return [order]


# --------------------------------------------------------------------------- #
# Lifetimes and diagnostics
# --------------------------------------------------------------------------- #
def release(ctx: SolveContext) -> None:
    """Free every slot held for ``ctx`` (batch slots included) and record its id as released.

    Also drops the context's reference to each analysis it used; an
    analysis no context references frees its device patterns.
    """
    _native().contract_context_release(int(ctx.context_id))


def resident_bytes(ctx: SolveContext) -> int:
    """Device bytes of the slots held for ``ctx``: CSR value copies and cuDSS factor storage."""
    return int(_native().contract_resident_bytes(int(ctx.context_id)))


def contract_stats() -> dict:
    """Process-wide counters of the cuDSS contract handlers.

    Reading the counters first settles the verdict of every factorization
    that no call has read on the host yet (one stream synchronisation each),
    so ``singular`` and ``nonfinite`` include them.

    Returns:
        A dict with ``cold_analyses`` and ``seeded_analyses`` (cuDSS analysis
        phases run with their own reordering, or, on a symmetric positive
        definite side or a batch tile, with the ordering and partition tree
        of the pattern's first such analysis, repeated for each lane on a
        stacked tile), ``factorizations`` (numeric
        phases of single slots), ``refactorizations``, ``rebuilds``
        (factorizations a solve started because its slot held another
        generation; numeric phases of batch tiles for a batch), ``solves``
        (solves of single slots), ``transposed_solves`` (solves in cuDSS's
        transpose mode, of single slots, batch tiles and batch lanes: a
        transposed solve of an ``"A"`` factor, or any solve of an ``"AT"``
        factor), ``nonfinite``, ``singular`` (verdicts settled on the host),
        ``deferred_status_reads`` (refactorization verdicts the host read
        after the refactorization, each in one stream synchronisation, shared
        with the read of the call that settles it when it has one),
        ``released_context_errors``,
        ``host_reads`` (calls that read their handle or lane token back to
        the host), ``static_dispatches`` (calls that found their slot from
        their attributes), ``predicate_reads``, ``invalidations``,
        ``batched_factorizations`` and ``batched_solves`` (one per batch
        tile), ``lanes_factored`` (the lanes those numeric phases factored:
        on a uniform batch a batched refactor factors only the lanes that
        ask, a stacked tile factors all its lanes),
        ``lanes_factored_unasked`` (of those, lanes of a stacked tile that
        the request did not ask for, factored again from the values they
        hold), ``status_solves`` (solves of uniform-batch tiles that settle
        their lanes' statuses, and of factors of the symmetric indefinite
        factorization that settle theirs; a stacked tile reads its lanes'
        statuses from its factor diagonal), ``batch_analyses`` (analyses of batch tiles, also counted
        in ``cold_analyses`` or ``seeded_analyses``), ``lane_rescues`` and
        ``lane_rescue_solves`` (lanes of a batch factored and solved again
        on their own, with matching, because the batch factor did not serve
        them), ``lanes_factored_alone`` and ``lane_solves_alone`` (lanes of
        a batch factored and solved on a factorization of their own because
        the analysis runs in deterministic or hybrid memory mode, which
        cuDSS's uniform batch refuses),
        ``zero_fills`` (solution buffers zeroed before a solve; none under
        ``zero_fill=False``), ``pinned_blocks_allocated`` and
        ``pinned_blocks_reused`` (pinned host blocks that hybrid-memory
        factors took from the process's pool: newly pinned, and handed out
        again after another factor returned them), ``downcast_configurations``
        (single factors and batch tiles set up to store a float64 factor
        downcast to float32, see ``downcast_factors``), ``assemblies`` and ``assembled_lanes`` (device
        assemblies of coordinate values into CSR values, and the lanes they
        covered), ``value_checks`` (assemblies that also compared the values
        with the copy they replaced) and ``rebuilds_avoided`` (rebuilds a
        value check found unnecessary because the factor's storage already
        held those values; batch tiles for a batch), ``storage_value_changes``
        (lanes a refactor of shared storage factored because the values it
        keeps differed from the storage's without its predicate asking, as at
        the entry of a recomputed segment), ``storage_asks_unchanged`` (lanes
        whose predicate asked a refactor of shared storage with the very values
        the storage held, which keep their factor and generation), and the gauges
        ``live_slots``, ``live_batch_slots``, ``live_sites``, ``live_analyses`` and ``live_device_analyses``, as
        the native extension reports them.

    Raises:
        ConfigurationError: If the extension or its contract handlers are missing.
    """
    return dict(_native().contract_stats())


def reset_contract_stats() -> None:
    """Zero the counters of [contract_stats][]; the ``live_*`` gauges are kept.

    Every factorization verdict not yet read on the host is settled first, so
    the counters after the reset start from the calls that follow it.

    Raises:
        ConfigurationError: If the extension or its contract handlers are missing.
    """
    _native().contract_reset_stats()


def contract_analysis_info(analysis: Any) -> dict:
    """The native state of one cuDSS analysis.

    Args:
        analysis: A cuDSS [Analysis][diff_soup.linear.Analysis] or its record.

    Returns:
        A dict with ``device_resident``, ``contexts`` (contexts holding the
        analysis), ``cold_analyses``, ``seeded_analyses``, ``seed_refusals``,
        ``batch_seeded`` (whether the last batch tile's analysis started
        from the pattern's ordering; ``None`` before the first batch),
        ``batch_analyses``, ``tree_seeded_analyses`` (seeded analyses that
        also reused the first analysis's partition tree), ``lu_nnz_forward``
        and ``pivot_ratio_forward`` (the last factor of a single slot),
        ``lu_nnz_batch`` (the fill per lane of the last batch tile),
        ``analysis_seconds_cold``, ``analysis_seconds_seeded_last``,
        ``analysis_seconds_batch_last``, ``nd_tree_bytes``,
        ``pattern_bytes`` and ``batch_layout`` (``"stacked"`` or
        ``"uniform"``), as the native extension reports them.

    Raises:
        ConfigurationError: If the extension or its contract handlers are missing.
        FactorContractError: If ``analysis`` is not a cuDSS analysis.
    """
    record = _record(analysis) if isinstance(analysis, Analysis) else analysis
    if not isinstance(record, _CuDSSContractAnalysis):
        raise FactorContractError(
            "contract_analysis_info needs a cuDSS Analysis built by diff_soup.linear.analyze "
            f"or its backend_analysis record; got {type(analysis).__name__}."
        )
    return dict(_native().contract_analysis_info(record.native_id))


def set_analysis_seeding(enabled: bool) -> bool:
    """Choose whether later analyses without matching reuse their pattern's first ordering.

    Symmetric positive definite slots and the tiles of native batches run
    without matching. With seeding on (the default), the first such analysis
    of a pattern in the process computes the reordering and every later one
    of that pattern starts from its permutation and partition tree, which
    gives the same fill and the same factorization and solve speed in about
    a tenth of the analysis time; off, each runs its own. General (and symmetric indefinite) slots always run their own
    analysis, because cuDSS's matching makes their ordering depend on the
    values: an ordering kept from other values can be far from fill-reducing
    for them. Applies to the whole process from the next analysis phase on.

    Args:
        enabled: Seed analyses without matching from the pattern's first ordering.

    Returns:
        The previous setting.

    Raises:
        ConfigurationError: If the extension or its contract handlers are missing.
    """
    native = _native()
    previous = bool(native.contract_seeding())
    native.contract_set_seeding(bool(enabled))
    return previous
