"""Library-owned integration measures for host-side reductions.

These let an application integrate a host array (a data trace, a boundary rate)
over the domain or a boundary *without re-deriving the mesh's integration
weights*.  They are the same weights the residual assembly uses, so a
constraint target built from them stays consistent with the solver:

* [domain_measure][] carries the lumped nodal volume ``∫ N_i dV`` (the
  [lumped_mass][diff_soup.domains.operators.Operators.lumped_mass] diagonal), optionally
  restricted to a chosen node set.  ``measure.total`` is the measured volume and
  ``measure.integrate(values)`` is ``Σ_i weight_i · values_i`` — a volumetric
  reduction such as the mass rate injected by a set of boundary nodes.
* [surface_measure][] carries a tagged boundary's facet length/area, so
  ``measure.integrate(values)`` is the surface integral ``∫_Γ values dS``.

Both ``integrate`` methods are jit- and gradient-safe in ``values``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diff_soup.domains.boundary_set import BoundarySet

__all__ = ["domain_measure", "surface_measure"]


def _n_nodes(mesh) -> int:
    return int(np.asarray(mesh.points).shape[0])


def _tag_nodes(mesh, name: str) -> set:
    from diff_soup.domains.mesh.fem._boundary import boundary_face_assembly

    fi, _fl, _n = boundary_face_assembly(mesh, name)
    return {int(i) for i in np.asarray(fi).reshape(-1)}


def _resolve_nodes(mesh, on: Any) -> np.ndarray:
    """Resolve ``on`` to an integer node-index array.

    An explicit integer array preserves its order (so it can be contracted
    against data given in that same node order); a tag name or
    [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] resolves
    to that boundary's nodes, sorted and unique.
    """
    if isinstance(on, str):
        return np.asarray(sorted(_tag_nodes(mesh, on)), dtype=np.int64)
    if isinstance(on, BoundarySet):
        node_set = on.reduce(
            on_tags=lambda names: set().union(*[_tag_nodes(mesh, n) for n in names]),
            on_nodes=lambda idx: {int(i) for i in idx},
            on_union=lambda a, b: a | b,
            on_difference=lambda a, b: a - b,
        )
        return np.asarray(sorted(node_set), dtype=np.int64)
    arr = np.asarray(on)
    if arr.dtype.kind not in "iu":
        raise TypeError(
            "domain_measure(on=...): pass node indices (int array), a tag name, "
            "or a ds.tags(...) selector."
        )
    return arr.astype(np.int64).reshape((-1,))


def _dedup_facets(fi: np.ndarray, fl: Any, nm: Any):
    fi = np.asarray(fi)
    seen: set = set()
    keep: list = []
    for i, row in enumerate(fi):
        key = tuple(sorted(int(v) for v in row))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    idx = np.asarray(keep, dtype=np.int64)
    return fi[idx], jnp.asarray(fl)[idx], jnp.asarray(nm)[idx]


def _resolve_facets(mesh, on: Any):
    from diff_soup.domains.mesh.fem._boundary import boundary_face_assembly

    if isinstance(on, str):
        names: tuple[str, ...] = (on,)
    elif isinstance(on, BoundarySet):
        if on.op == "tags":
            names = on.names
        elif on.op == "union":
            fil, fll, nml = _resolve_facets(mesh, on.left)
            fir, flr, nmr = _resolve_facets(mesh, on.right)
            return _dedup_facets(
                np.concatenate([np.asarray(fil), np.asarray(fir)], axis=0),
                jnp.concatenate([jnp.asarray(fll), jnp.asarray(flr)]),
                jnp.concatenate([jnp.asarray(nml), jnp.asarray(nmr)]),
            )
        else:
            raise ValueError(
                "surface_measure supports tag names and their unions; for a set "
                "difference, build the facet selection explicitly."
            )
    else:
        raise TypeError("surface_measure(on=...): pass a tag name or a ds.tags(...) selector.")
    parts = [boundary_face_assembly(mesh, n) for n in names]
    return _dedup_facets(
        np.concatenate([np.asarray(p[0]) for p in parts], axis=0),
        jnp.concatenate([jnp.asarray(p[1]) for p in parts]),
        jnp.concatenate([jnp.asarray(p[2]) for p in parts]),
    )


class _DomainMeasure(eqx.Module):
    """Lumped nodal-volume measure over the domain (or a chosen node set)."""

    _w: Any
    _nodes: Any
    _n_nodes: int = eqx.field(static=True)

    def __init__(self, weights, nodes, n_nodes) -> None:
        self._w = jnp.asarray(weights)
        self._nodes = None if nodes is None else np.asarray(nodes, dtype=np.int64)
        self._n_nodes = int(n_nodes)

    @property
    def total(self):
        """The measured volume ``Σ_i weight_i`` (``|Ω|`` for the whole domain).

        A scalar that composes inside a traced ``target_fn``; convert with
        ``float(...)`` for a host value.
        """
        return jnp.sum(self._w)

    @property
    def weights(self):
        """The per-node integration weights, in the measure's node order."""
        return self._w

    @property
    def nodes(self):
        """The selected node indices, or ``None`` for the whole domain."""
        return self._nodes

    def integrate(self, values):
        """``Σ_i weight_i · values_i``.

        ``values`` has one entry per selected node (in this measure's node
        order), or one entry per mesh node (gathered to the selection).  Any
        other length raises.
        """
        v = jnp.asarray(values).reshape((-1,))
        k = int(self._w.shape[0])
        if v.shape[0] == k:
            vv = v
        elif self._nodes is not None and v.shape[0] == self._n_nodes:
            vv = v[jnp.asarray(self._nodes)]
        else:
            expected = [f"selected nodes ({k})"]
            if self._nodes is not None:
                expected.append(f"all mesh nodes ({self._n_nodes})")
            raise ValueError(
                f"domain_measure.integrate: values has length {v.shape[0]}, "
                f"expected {' or '.join(expected)}."
            )
        return jnp.sum(self._w * vv)


class _SurfaceMeasure(eqx.Module):
    """Facet length/area measure over a tagged boundary."""

    _fi: Any
    _fl: Any
    _normals: Any
    _n_nodes: int = eqx.field(static=True)

    def __init__(self, face_indices, face_measure, normals, n_nodes) -> None:
        self._fi = np.asarray(face_indices)
        self._fl = jnp.asarray(face_measure)
        self._normals = jnp.asarray(normals)
        self._n_nodes = int(n_nodes)

    @property
    def total(self):
        """The boundary length (2D) or area (3D), ``Σ_facets face_measure``.

        A scalar that composes inside a traced ``target_fn``; convert with
        ``float(...)`` for a host value.
        """
        return jnp.sum(self._fl)

    @property
    def face_measure(self):
        """The per-facet length/area weights."""
        return self._fl

    @property
    def normals(self):
        """The per-facet outward unit normals, ``(n_facets, dim)``."""
        return self._normals

    def integrate(self, values):
        """The surface integral ``∫_Γ values dS``.

        ``values`` is either a full nodal field (length ``n_nodes``; face-meaned
        onto each facet) or one value per facet.  Any other length raises.
        """
        flat = jnp.asarray(values).reshape((-1,))
        n_facets = int(self._fi.shape[0])
        if flat.shape[0] == n_facets:
            per_facet = flat
        elif flat.shape[0] == self._n_nodes:
            per_facet = jnp.mean(flat[jnp.asarray(self._fi)], axis=1)
        else:
            raise ValueError(
                f"surface_measure.integrate: values has length {flat.shape[0]}, "
                f"expected per-facet ({n_facets}) or per-node ({self._n_nodes})."
            )
        return jnp.sum(per_facet * self._fl)


def domain_measure(mesh, on: Any = None) -> _DomainMeasure:
    """A lumped nodal-volume measure over ``mesh`` (or a chosen node set).

    Args:
        mesh: The discretized domain.
        on: ``None`` for the whole domain; an integer node-index array (order
            preserved, so it can be contracted against data in that node order);
            or a tag name / ``ds.tags(...)`` selector for that boundary's nodes.

    Returns:
        A measure with ``.total``, ``.weights``, ``.nodes`` and
        ``.integrate(values)`` (``Σ_i weight_i · values_i``).
    """
    from diff_soup.domains.operators import Operators

    lumped = jnp.asarray(Operators(mesh).lumped_mass())
    n_nodes = int(lumped.shape[0])
    if on is None:
        return _DomainMeasure(lumped, None, n_nodes)
    nodes = _resolve_nodes(mesh, on)
    return _DomainMeasure(lumped[jnp.asarray(nodes)], nodes, n_nodes)


def surface_measure(mesh, on: Any) -> _SurfaceMeasure:
    """A facet length/area measure over a tagged boundary of ``mesh``.

    Args:
        mesh: The discretized domain.
        on: A tag name or a ``ds.tags(...)`` selector (tag names and their
            unions) identifying the boundary facets.

    Returns:
        A measure with ``.total``, ``.face_measure``, ``.normals`` and
        ``.integrate(values)`` (the surface integral ``∫_Γ values dS``).
    """
    fi, fl, normals = _resolve_facets(mesh, on)
    return _SurfaceMeasure(fi, fl, normals, _n_nodes(mesh))
