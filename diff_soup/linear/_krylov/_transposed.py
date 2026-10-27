"""The pattern of the transposed assembled operator, as a plain topology.

A factor of orientation ``"AT"`` represents ``A_bc^T``, the transpose of the
assembled operator with its Dirichlet rows applied (every entry of a Dirichlet
row replaced by zero and ``1`` on its diagonal). An iterative backend that
preconditions such a factor with a primary factor of ``A_bc^T`` (one whose
solves are not transposed) analyses ``A_bc^T`` as a matrix of its own:
[transposed_topology][diff_soup.linear._krylov.transposed_topology] gives its
pattern, without a Dirichlet declaration, and the gather that maps the values
of ``A`` to the values of ``A_bc^T`` on it.
"""

from __future__ import annotations

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup.linear._api import SparseTopology

__all__ = ["transposed_topology", "transposed_values"]


@functools.lru_cache(maxsize=64)
def transposed_topology(topology: SparseTopology) -> tuple[SparseTopology, np.ndarray]:
    """The pattern of ``A_bc^T`` as a topology without Dirichlet rows, and its value gather.

    The entries of every row of ``topology`` that is not a Dirichlet row are
    transposed (entry ``(i, j)`` becomes ``(j, i)``, in the original order),
    and one unit diagonal entry ``(r, r)`` follows for every Dirichlet row
    ``r`` (sorted, each once). The values of ``A_bc^T`` on that pattern are

        values_T = concatenate([values[..., index], ones(n_bc)])

    (see [transposed_values][diff_soup.linear._krylov.transposed_values]),
    so the operator the new pattern and values describe is exactly ``A_bc^T``:
    a Dirichlet row of ``A_bc`` holds only its unit diagonal, which becomes a
    unit diagonal of the transpose, and every other entry is transposed.

    Args:
        topology: The [SparseTopology][diff_soup.linear.SparseTopology] of ``A``.

    Returns:
        ``(topology_T, index)``: the plain topology of ``A_bc^T`` and the
        read-only ``int32`` positions of the kept entries of ``A``.
    """
    rows = np.asarray(topology.rows, dtype=np.int32)
    cols = np.asarray(topology.cols, dtype=np.int32)
    bc = topology.bc_indices
    if bc is None or int(np.asarray(bc).size) == 0:
        index = np.arange(rows.size, dtype=np.int32)
        dirichlet = np.zeros((0,), dtype=np.int32)
    else:
        index = np.flatnonzero(~np.isin(rows, np.asarray(bc))).astype(np.int32)
        dirichlet = np.unique(np.asarray(bc, dtype=np.int32))
    transposed = SparseTopology(
        rows=np.concatenate([cols[index], dirichlet]),
        cols=np.concatenate([rows[index], dirichlet]),
        n_dofs=int(topology.n_dofs),
    )
    index = np.array(index, copy=True)
    index.flags.writeable = False
    return transposed, index


def transposed_values(topology: SparseTopology, values: Any) -> Array:
    """The values of ``A_bc^T`` on [transposed_topology][diff_soup.linear._krylov.transposed_topology].

    Args:
        topology: The topology of ``A``.
        values: The values of ``A``, ``(*lanes, nnz)``.

    Returns:
        ``(*lanes, nnz_T)``: the kept values gathered by the index, then one
        ``1`` per Dirichlet row, in the dtype of ``values``.
    """
    transposed, index = transposed_topology(topology)
    values = jnp.asarray(values)
    gathered = values[..., index]
    extra = int(transposed.rows.size) - int(index.size)
    if extra == 0:
        return gathered
    ones = jnp.ones(values.shape[:-1] + (extra,), values.dtype)
    return jnp.concatenate([gathered, ones], axis=-1)
