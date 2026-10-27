"""domains/base.py — Abstract base class for all diff_soup geometric domains.

A ``Domain`` captures the geometric and topological information of a spatial
discretization: node coordinates, connectivity, and spatial dimension.

All concrete domains are ``eqx.Module`` pytrees so they can be carried through
JAX transforms (jit, grad, vmap) and stored as fields of other modules.

Concrete subclasses
-------------------
* [Mesh][diff_soup.domains.mesh.Mesh] — unstructured simplex/quad mesh (FEM)
* [Grid1D][diff_soup.domains.grid.Grid1D] — 1D structured grid (FVM/FDM)
* [GridND][diff_soup.domains.grid.GridND] — N-D Cartesian tensor-product grid (FVM/FDM)
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import numpy as np


class Domain(eqx.Module):
    """The base class of every grid and mesh that `spec.domain` registers.

    Subclass it to add a domain type. The operator bank chooses how to evaluate each operator from `discretization_type`.
    """

    # ------------------------------------------------------------------
    # Geometry queries — must be implemented by every concrete subclass
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dim(self) -> int:
        """The number of spatial dimensions."""
        ...

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """The number of nodes, which on a grid is the number of cells."""
        ...

    @property
    @abstractmethod
    def discretization_type(self) -> str:
        """The discretization, `"fvm"`, `"fdm"` or `"fem"` for the built-in domains."""
        ...

    @abstractmethod
    def node_coordinates(self) -> jax.Array:
        """Return the coordinates of every node.

        Returns:
            An array of shape `(num_nodes, dim)`.
        """
        ...

    def sparsity_pattern(self, n_vars: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """COO sparsity pattern ``(rows, cols)`` for ``n_vars`` DOFs per node.

        Returns int32 arrays.  Must be implemented by concrete subclasses
        that support sparse Jacobian assembly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement sparsity_pattern")
