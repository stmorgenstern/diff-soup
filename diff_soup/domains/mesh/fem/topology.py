"""domains/fem/topology.py — Finite element space as eqx.Module.

The [FiniteElementSpace][] holds pre-computed shape functions, gradients,
quadrature weights, and Dirichlet BC data for a given mesh and element type.

Use the [make_fe_space][] factory to construct instances.  Direct
construction (passing all fields) is also supported for advanced use.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.domains.mesh import Mesh
from diff_soup.domains.mesh.core import metric_exponent
from diff_soup.domains.mesh.fem.basis import (
    get_face_shape_vals_and_grads,
    get_shape_vals_and_grads,
)
from diff_soup.domains.mesh.fem.dirichlet_bc import (
    DirichletBC,
    build_dirichlet_data,
)

Array = jax.Array


class FiniteElementSpace(eqx.Module):
    """Finite element space (immutable, pytree-compatible).

    All topology/geometry metadata is static; precomputed arrays are
    JAX array leaves (non-differentiable but carried through transforms).

    ``cells`` vs ``dof_cells``
    --------------------------
    Every FEM dispatch function uses two connectivity arrays:

    * ``cells`` (via ``shape_grads``, ``jxw``, ``points[cells]``): the
      **physical** element connectivity for Jacobian / geometry computation.
    * ``dof_cells``: the **DOF** connectivity for solution gather
      (``u[dof_cells]``) and residual scatter
      (``scatter_elem_residual(..., dof_cells, num_dofs, ...)``).

    For non-periodic meshes ``dof_cells == cells`` and
    ``num_dofs == num_total_nodes``.  For periodic meshes, tenant nodes
    (on the high periodic face) are remapped to landlord DOF indices
    (on the low face) by [_build_periodic_dof_map][], so assembly
    automatically merges contributions from opposite boundaries.

    **Any new FEM operator must follow this pattern**: use ``cells`` for
    geometry and ``dof_cells`` for solution gather / residual scatter.
    """

    # Static metadata — required.  Use [make_fe_space][] to construct
    # instances; direct construction must populate every field (no defaults)
    # so that shape grads / quadrature data cannot silently be ``None``.
    # ``mesh`` is a regular pytree leaf (contains numpy arrays); scalars/
    # strings below are truly static.
    mesh: Mesh
    vec: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    ele_type: str = eqx.field(static=True)
    gauss_order: int | None = eqx.field(static=True)
    num_cells: int = eqx.field(static=True)
    num_total_nodes: int = eqx.field(static=True)
    num_total_dofs: int = eqx.field(static=True)
    num_quads: int = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)

    # Pre-computed reference element data
    shape_vals: Array
    shape_grads_ref: Array
    quad_weights: Array
    face_shape_vals: Array
    face_shape_grads_ref: Array
    face_quad_weights: Array
    face_normals: Array
    face_inds: Array

    # Pre-computed physical element data
    shape_grads: Array
    # The integration measure at each quadrature point, ``(n_cells, n_q)``:
    # the quadrature weight times the Jacobian determinant, times the
    # coordinate metric ``r**p`` on a curvilinear mesh.  Every volume integral
    # reads this one array, so the metric reaches all of them at once.
    jxw: Array

    # JAX versions of mesh arrays (for JIT)
    points: Array
    cells: Array  # Physical connectivity (for Jacobian / shape_grads)

    # DOF connectivity for periodic meshes.  For non-periodic meshes these
    # are identical to ``cells`` / ``num_total_nodes``.  For periodic meshes
    # tenant (periodic-face) nodes are remapped to landlord (opposite-face)
    # DOF indices so the assembly scatter merges them automatically.
    dof_cells: Array  # DOF connectivity (for solution gather / residual scatter)
    num_dofs: int = eqx.field(static=True)  # Reduced DOF count after periodic identification

    # Dirichlet BC data.  Tuples of per-BC arrays (empty tuple means "no
    # BCs").  Not ``static=True`` because the tuple contains JAX array leaves
    # that trip Equinox's static-array warning — but the tuple structure is
    # never mutated and is safe to carry as a regular pytree leaf set.
    node_inds_list: tuple = ()
    vec_inds_list: tuple = ()
    vals_list: tuple = ()

    def __post_init__(self):
        """Validate that ``make_fe_space`` produced the precomputed arrays."""
        # Fail fast if someone constructed this directly without the required
        # precomputed arrays (e.g. via ``FiniteElementSpace(mesh, vec=1)``).
        if self.shape_grads is None or self.shape_vals is None:
            raise ValueError(
                "FiniteElementSpace constructed without precomputed arrays. "
                "Use diff_soup.domains.mesh.fem.make_fe_space(mesh, vec, ...) to "
                "build instances instead of calling the constructor directly."
            )
        if int(self.num_total_nodes) == 0:
            raise ValueError(
                "FiniteElementSpace has num_total_nodes=0. "
                "Use make_fe_space() to construct the space."
            )

    # Essential boundary conditions are represented outside this immutable
    # FE topology.


def _build_periodic_dof_map(
    mesh: Mesh,
    periodic_axes: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    """Build the periodic DOF remapping (landlord-tenant identification).

    For each periodic axis, nodes on the high face ("tenants") are
    identified with corresponding nodes on the low face ("landlords").
    The mapping is applied to the cell connectivity so that the FEM
    assembly scatter automatically merges periodic DOFs.

    Args:
        mesh: A structured rectangle/box mesh.
        periodic_axes: Axes on which to apply periodic identification.

    Returns:
        old_to_new:
            int32 array of shape ``(num_total_nodes,)`` mapping original node
            indices to compacted DOF indices.
        num_dofs:
            Number of unique DOFs after periodic identification.

    Raises:
        ValueError: If the mesh is one-dimensional.  Matching a high-face node
            to its low-face partner compares their coordinates on every *other*
            axis; a one-dimensional mesh has no other axis, so every node would
            match every node.  Impose periodicity on a 1D domain through the
            finite-volume or finite-difference path instead.
    """
    pts = np.asarray(mesh.points)
    n_nodes = pts.shape[0]
    dim = pts.shape[1]
    tol = 1e-10

    if dim == 1:
        raise ValueError(
            "periodic node identification needs at least two spatial axes: the "
            "high-face node is matched to the low-face node that agrees on the "
            "remaining axes, and a 1D mesh has none. Use a periodic Grid1D "
            "(finite-volume or finite-difference) for a periodic 1D domain."
        )

    # Start with identity mapping
    canonical = np.arange(n_nodes, dtype=np.int32)

    for axis in periodic_axes:
        # Find low-face and high-face nodes
        coord = pts[:, axis]
        lo, hi = float(coord.min()), float(coord.max())
        low_mask = np.abs(coord - lo) < tol
        high_mask = np.abs(coord - hi) < tol

        low_ids = np.where(low_mask)[0]
        high_ids = np.where(high_mask)[0]

        # Match high-face nodes to low-face nodes by their coordinates
        # on the other axes (ignoring the periodic axis)
        other_axes = [a for a in range(dim) if a != axis]
        for h_id in high_ids:
            h_coords = pts[h_id, other_axes]
            dists = np.linalg.norm(pts[low_ids][:, other_axes] - h_coords, axis=1)
            match_idx = np.argmin(dists)
            if dists[match_idx] > tol:
                continue  # No match (shouldn't happen on structured meshes)
            # Tenant (high face) → landlord (low face)
            canonical[h_id] = canonical[low_ids[match_idx]]

    # Resolve chains (e.g., corner nodes mapped through two axes)
    for _ in range(len(periodic_axes)):
        canonical = canonical[canonical]

    # Compact: assign contiguous DOF indices to unique canonical nodes
    unique_canonical = np.unique(canonical)
    num_dofs = len(unique_canonical)
    old_to_new = np.empty(n_nodes, dtype=np.int32)
    for new_idx, old_idx in enumerate(unique_canonical):
        old_to_new[old_idx] = new_idx
    # Map all nodes through canonical first, then to new indices
    dof_map = np.array([old_to_new[canonical[i]] for i in range(n_nodes)], dtype=np.int32)

    return dof_map, num_dofs


def make_fe_space(
    mesh: Mesh,
    vec: int = 1,
    gauss_order: int | None = None,
    dirichlet_bcs: Sequence[DirichletBC] | None = None,
    periodic_axes: tuple[int, ...] = (),
) -> FiniteElementSpace:
    """Factory: build a FiniteElementSpace from a Mesh.

    Args:
        mesh: The [Mesh][diff_soup.domains.mesh.Mesh] to build on.
        vec: Number of DOFs per node.
        gauss_order: Override Gauss quadrature order (default: auto from element type).
        dirichlet_bcs: Sequence of
            [DirichletBC][diff_soup.domains.mesh.fem.dirichlet_bc.DirichletBC] objects.
        periodic_axes: Tuple of spatial axes (0, 1, 2) with periodic DOF identification.
            Nodes on the high face of each axis are identified with
            corresponding nodes on the low face.  The mesh geometry is
            unchanged (all elements keep correct Jacobians); only the DOF
            numbering is modified so the assembly scatter merges periodic
            DOFs automatically.
    """
    ele_type = mesh.ele_type
    dim = mesh.dim
    points = jnp.asarray(mesh.points)
    cells = jnp.asarray(mesh.cells, dtype=jnp.int32)
    num_cells = int(cells.shape[0])
    num_total_nodes = int(points.shape[0])
    num_total_dofs = num_total_nodes * vec

    metric_p = metric_exponent(getattr(mesh, "coord_system", None))
    shape_vals, shape_grads_ref, quad_weights = get_shape_vals_and_grads(
        ele_type, gauss_order, metric_degree=metric_p
    )
    (
        face_shape_vals,
        face_shape_grads_ref,
        face_quad_weights,
        face_normals,
        face_inds,
    ) = get_face_shape_vals_and_grads(ele_type, gauss_order)

    shape_vals = jnp.asarray(shape_vals)
    shape_grads_ref = jnp.asarray(shape_grads_ref)
    quad_weights = jnp.asarray(quad_weights)
    face_shape_vals = jnp.asarray(face_shape_vals)
    face_shape_grads_ref = jnp.asarray(face_shape_grads_ref)
    face_quad_weights = jnp.asarray(face_quad_weights)
    face_normals = jnp.asarray(face_normals)
    face_inds = jnp.asarray(face_inds, dtype=jnp.int32)

    num_quads = int(shape_vals.shape[0])
    num_nodes = int(shape_vals.shape[1])

    # Compute physical gradients and Jacobian-weighted quadrature
    # Uses original cells (physical connectivity) — correct Jacobians
    physical = points[cells]  # (c, n, d)
    jac = jnp.einsum("cnd,qnk->cqdk", physical, shape_grads_ref)
    jac_inv = jnp.linalg.inv(jac)
    jac_det = jnp.linalg.det(jac)
    shape_grads = jnp.einsum("qnk,cqkd->cqnd", shape_grads_ref, jac_inv)
    # The integration measure, not merely |J| w: on a curvilinear radial mesh
    # the volume element is r**p dr, and folding that factor in here is what
    # gives every volume integral in the library — stiffness, consistent and
    # lumped mass, sources, projections, reductions — the coordinate metric
    # with no kernel of its own to change.
    jxw = jac_det * quad_weights[None, :]
    if metric_p:
        quad_points = jnp.einsum("qn,cnd->cqd", shape_vals, physical)
        jxw = jxw * quad_points[..., 0] ** metric_p

    # Build DOF connectivity (periodic identification if requested)
    if periodic_axes:
        dof_map, num_dofs = _build_periodic_dof_map(mesh, periodic_axes)
        dof_map_jax = jnp.asarray(dof_map)
        dof_cells = dof_map_jax[cells]  # Remap node indices to DOF indices
    else:
        dof_cells = cells
        num_dofs = num_total_nodes

    # Build Dirichlet BC data
    node_inds_list, vec_inds_list, vals_list = build_dirichlet_data(points, vec, dirichlet_bcs)

    return FiniteElementSpace(
        mesh=mesh,
        vec=vec,
        dim=dim,
        ele_type=ele_type,
        gauss_order=gauss_order,
        num_cells=num_cells,
        num_total_nodes=num_total_nodes,
        num_total_dofs=num_total_dofs,
        num_quads=num_quads,
        num_nodes=num_nodes,
        shape_vals=shape_vals,
        shape_grads_ref=shape_grads_ref,
        quad_weights=quad_weights,
        face_shape_vals=face_shape_vals,
        face_shape_grads_ref=face_shape_grads_ref,
        face_quad_weights=face_quad_weights,
        face_normals=face_normals,
        face_inds=face_inds,
        shape_grads=shape_grads,
        jxw=jxw,
        points=points,
        cells=cells,
        dof_cells=dof_cells,
        num_dofs=num_dofs,
        node_inds_list=node_inds_list,
        vec_inds_list=vec_inds_list,
        vals_list=vals_list,
    )


_SCALAR_SPACES: dict = {}


def scalar_fe_space(mesh: Mesh) -> FiniteElementSpace:
    """The single-field finite-element space of *mesh*, built once per mesh object.

    Every operator call of a lowered problem reuses the same space, whose
    arrays are concrete. Inside a traced program they are closed over once,
    so the program's constant count does not grow with the number of operator
    calls and no shape gradient or quadrature weight is recomputed in the
    trace. A mesh whose node coordinates are being traced (a mesh passed as
    an argument through ``jax.jit``) gets a space built from those tracers.
    """
    points = getattr(mesh, "points", None)
    if isinstance(points, jax.core.Tracer):
        return make_fe_space(mesh, vec=1, periodic_axes=getattr(mesh, "periodic_axes", ()))
    key = id(mesh)
    entry = _SCALAR_SPACES.get(key)
    if entry is not None:
        holder, space = entry
        if holder() is mesh:
            return space
    space = make_fe_space(mesh, vec=1, periodic_axes=getattr(mesh, "periodic_axes", ()))
    try:
        _SCALAR_SPACES[key] = (
            weakref.ref(mesh, lambda _ref, key=key: _SCALAR_SPACES.pop(key, None)),
            space,
        )
    except TypeError:
        # A domain that cannot be weakly referenced is not cached.
        pass
    return space
