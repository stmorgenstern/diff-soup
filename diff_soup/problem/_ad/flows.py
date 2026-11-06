"""Tangent flows and the index tables that move them, all in NumPy.

A flow is a boolean CSR matrix with one row per element of an array, row-major, and
one column per seed: entry ``(i, j)`` says the tangent seeded at ``j`` can reach
element ``i``. An operation that only moves elements, such as a slice, a gather or a
pad, is run on a table of row numbers, where -1 marks an element no tangent reaches.
"""

from __future__ import annotations

import jax
import numpy as np
from scipy import sparse


def select(flow, table):
    """The rows of ``flow`` listed in ``table``; a negative entry is an empty row."""
    table = np.asarray(table).ravel()
    valid = table >= 0
    lengths = np.zeros(table.size, dtype=np.int64)
    lengths[valid] = np.diff(flow.indptr)[table[valid]]
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    within = np.arange(indptr[-1]) - np.repeat(indptr[:-1], lengths)
    take = within + np.repeat(np.where(valid, flow.indptr[np.maximum(table, 0)], 0), lengths)
    indices = flow.indices[take]
    data = np.ones(indices.size, dtype=bool)
    return sparse.csr_matrix((data, indices, indptr), shape=(table.size, flow.shape[1]))


def union(first, second):
    """Row-wise union of two flows, either of which may be ``None``."""
    if first is None or first is second:
        return second
    return first if second is None else (first + second).tocsr()


def gathering(rows: int, table, columns: int):
    """Boolean ``rows`` by ``columns`` matrix sending column ``k`` to row ``table[k]``."""
    table = np.asarray(table).ravel()
    keep = table >= 0
    data = np.ones(int(keep.sum()), dtype=bool)
    positions = (table[keep], np.arange(table.size)[keep])
    return sparse.csr_matrix((data, positions), shape=(rows, columns))


def anywhere(flows, sizes, seeds: int):
    """Flows of ``sizes`` elements, each reached by every column reaching any of ``flows``."""
    columns = np.unique(np.concatenate([flow.indices for flow in flows]))
    outputs = []
    for rows in sizes:
        indptr = np.arange(rows + 1, dtype=np.int64) * columns.size
        data = np.ones(rows * columns.size, dtype=bool)
        outputs.append(sparse.csr_matrix((data, np.tile(columns, rows), indptr), (rows, seeds)))
    return outputs


def broadcast_flow(flow, shape, out_shape):
    """``flow`` of an array of ``shape``, broadcast to ``out_shape`` as a lax operation does."""
    if tuple(shape) == tuple(out_shape):
        return flow
    return select(flow, np.broadcast_to(np.arange(flow.shape[0]).reshape(shape), out_shape))


def stack(flows, seeds: int):
    """The flows one after another, one block of rows each; ``None`` is an empty block."""
    return (
        sparse.vstack(flows, format="csr") if flows else sparse.csr_matrix((0, seeds), dtype=bool)
    )


def gather_table(table, indices, dimension_numbers, slice_sizes, mode, fill: int = -1):
    """``lax.gather`` of an index table, with ``fill`` for a window out of bounds.

    Follows the semantics of XLA's gather (``jax.lax.gather``): batching dimensions,
    collapsed and offset dimensions, ``clip`` and ``fill`` modes.
    """
    numbers = dimension_numbers
    shape = np.asarray(table.shape, dtype=np.int64)
    rank = table.ndim
    indices = np.asarray(indices, dtype=np.int64)
    batch_shape = indices.shape[:-1]
    count = int(np.prod(batch_shape, dtype=np.int64))
    starts_given = indices.reshape(count, indices.shape[-1])
    starts = np.zeros((count, rank), dtype=np.int64)
    for position, axis in enumerate(numbers.start_index_map):
        starts[:, axis] = starts_given[:, position]
    if numbers.operand_batching_dims:
        coordinates = np.indices(batch_shape).reshape(len(batch_shape), -1)
        for axis, index_axis in zip(
            numbers.operand_batching_dims, numbers.start_indices_batching_dims
        ):
            starts[:, axis] = coordinates[index_axis]
    sizes = np.asarray(slice_sizes, dtype=np.int64)
    upper = shape - sizes
    valid = np.ones(count, dtype=bool)
    if mode == jax.lax.GatherScatterMode.FILL_OR_DROP:
        valid = np.all((starts >= 0) & (starts <= upper), axis=1)
    starts = np.clip(starts, 0, upper)
    window = np.indices(sizes).reshape(rank, -1).T
    full = starts[:, None, :] + window[None, :, :]
    flat = np.ravel_multi_index(tuple(full.reshape(-1, rank).T), tuple(shape)) if rank else 0
    values = np.asarray(table).ravel()[np.reshape(flat, (count, window.shape[0]))].copy()
    values[~valid] = fill
    dropped = set(numbers.collapsed_slice_dims) | set(numbers.operand_batching_dims)
    kept = [axis for axis in range(rank) if axis not in dropped]
    values = values.reshape(*batch_shape, *(int(sizes[axis]) for axis in kept))
    out_rank = len(batch_shape) + len(kept)
    offsets = list(numbers.offset_dims)
    batch_axes = iter(range(len(batch_shape)))
    source = [
        len(batch_shape) + offsets.index(axis) if axis in offsets else next(batch_axes)
        for axis in range(out_rank)
    ]
    return np.transpose(values, source)


def scatter_targets(shape, indices, dimension_numbers, update_shape, mode):
    """The flat operand element each update element of a scatter writes, or -1 if dropped."""
    numbers = dimension_numbers
    collapsed = (*numbers.inserted_window_dims, *numbers.operand_batching_dims)
    window = iter(update_shape[axis] for axis in numbers.update_window_dims)
    sizes = tuple(1 if axis in collapsed else next(window) for axis in range(len(shape)))
    gather_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=numbers.update_window_dims,
        collapsed_slice_dims=numbers.inserted_window_dims,
        start_index_map=numbers.scatter_dims_to_operand_dims,
        operand_batching_dims=numbers.operand_batching_dims,
        start_indices_batching_dims=numbers.scatter_indices_batching_dims,
    )
    elements = np.arange(int(np.prod(shape, dtype=np.int64))).reshape(shape)
    return gather_table(elements, indices, gather_numbers, sizes, mode).ravel()


def _clamped(starts, shape, sizes):
    """Start indices clamped as ``dynamic_slice`` clamps them."""
    return [
        int(np.clip(int(np.asarray(start)), 0, int(dim) - int(size)))
        for start, dim, size in zip(starts, shape, sizes)
    ]


def dynamic_slice_table(table, starts, slice_sizes):
    """``lax.dynamic_slice`` of an index table."""
    starts = _clamped(starts, table.shape, slice_sizes)
    return table[tuple(slice(start, start + size) for start, size in zip(starts, slice_sizes))]


def dynamic_update_slice_table(table, update, starts):
    """``lax.dynamic_update_slice`` of an index table."""
    starts = _clamped(starts, table.shape, update.shape)
    out = np.array(table, copy=True)
    out[tuple(slice(start, start + size) for start, size in zip(starts, update.shape))] = update
    return out


def pad_table(table, value, padding_config):
    """``lax.pad`` of an index table, with negative padding cropping."""
    grown = [
        max(low, 0) + max(high, 0) + n + max(n - 1, 0) * interior
        for (low, high, interior), n in zip(padding_config, table.shape)
    ]
    out = np.full(grown, int(np.asarray(value)), dtype=np.int64)
    out[
        tuple(
            slice(max(low, 0), max(low, 0) + (n - 1) * (interior + 1) + 1, interior + 1)
            for (low, _, interior), n in zip(padding_config, table.shape)
        )
    ] = table
    crop = tuple(
        slice(-min(low, 0), size + min(high, 0))
        for (low, high, _), size in zip(padding_config, grown)
    )
    return out[crop]


def split_tables(table, sizes, axis):
    """``lax.split`` of an index table."""
    return np.split(table, np.cumsum(sizes)[:-1], axis=axis)
