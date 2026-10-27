"""Boundary-aware ops: ``boundary_integral``, ``boundary_trace``, ``surface_flux``.

FEM-only.  Uses [diff_soup.domains.mesh.fem._boundary.boundary_face_assembly][]
to resolve geometry at runtime.
"""

from __future__ import annotations

import jax.numpy as jnp


def boundary_integral(field, *, tag, mesh):
    """Surface integral of a scalar bulk field over a tagged boundary."""
    from diff_soup.domains.mesh.fem._boundary import boundary_face_assembly

    face_indices, face_areas, _normals = boundary_face_assembly(mesh, tag)
    field_at_faces = jnp.mean(field[face_indices], axis=1)
    return jnp.sum(field_at_faces * face_areas)


def boundary_trace(field, *, tag, mesh):
    """Restrict a bulk field to unique boundary nodes for a tag."""
    from diff_soup.domains.mesh.fem._boundary import boundary_face_assembly

    face_indices, _areas, _normals = boundary_face_assembly(mesh, tag)
    return field[jnp.unique(face_indices.reshape(-1))]


def surface_flux(field, *, tag, kinetics, mesh, ctx=None, t=0.0):
    """Apply ``kinetics(c_boundary, ctx, t)`` to the boundary trace of ``field``."""
    return kinetics(boundary_trace(field, tag=tag, mesh=mesh), ctx, t)


__all__ = ["boundary_integral", "boundary_trace", "surface_flux"]
