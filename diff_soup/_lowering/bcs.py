"""Boundary condition resolution for the lowering provider.

Owns:

- ``_ScaledDirichletValueFn``: traced wrapper for time- + state-scaled
  Dirichlet values.
- ``_ResolvedDirichlet`` / ``_ResolvedFlux`` / ``_ResolvedRobin`` /
  ``_ResolvedTraction`` / ``_ResolvedSymmetry`` / ``_ResolvedBCs``: dataclasses
  describing each BC's resolved DOF indices, value/flux closures, etc.
- ``_resolve_tag``, ``_resolve_grid_local_indices``, ``_grid_face_of``,
  ``_grid_face_factor``: geometry helpers for translating ``BoundarySpec`` into
  DOF indices and per-face weights.
- ``_wrap_dirichlet_value_fn``: rewraps a value_fn for non-identity Scales.
- ``_bc_dofs``: top-level entry that assembles ``_ResolvedBCs``.
- ``_extract_flux_callables`` / ``_extract_robin_callables`` /
  ``_extract_traction_callables``: traced-cache refresh helpers.

Dirichlet, Flux, and Robin conditions resolve on both FEM (``TaggedMesh``) and
FVM/FDM (``Grid1D`` / ``GridND``) domains.  Traction and Symmetry are FEM-only:
both are stated on boundary facets and need the facet normals and measures a
tagged mesh carries.

This module lives under ``diff_soup/_lowering`` and may import both
``diff_soup.domains`` (L1) and ``diff_soup.problem._ir`` (L2 IR nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import LoweringError
from diff_soup.domains.grid.core import Grid1D, GridND
from diff_soup.domains.mesh.core import TaggedMesh
from diff_soup.domains.mesh.fem._boundary import (
    boundary_face_assembly_for_mask,
    boundary_facet_mask_for_ids,
    boundary_facet_mask_for_nodes,
)
from diff_soup.problem._ir.bc import (
    DirichletBC,
    FluxBC,
    PeriodicBC,
    RobinBC,
    SymmetryBC,
    TractionBC,
    _BoundaryFn,
    constant_coefficient_of,
)
from diff_soup.problem._ir.content_hash import (
    _named_values_from_defaults,
    declared_callable_layers,
    declared_named_values,
)
from diff_soup.problem._ir.equation_set import EquationSet
from diff_soup.problem._ir.variable import Variable
from diff_soup.problem._spec.q_accessor import _BoundaryQ

from .couplings import _read_through_alias_owner, _resolve_coupling_endpoints
from .layout import _BlockLayout
from .operator_resolve import resolve_operator


def _boundary_context(equations):
    """Read declared namespaces from the current evaluation's equation leaves."""
    for equation in equations:
        for _, layer in declared_callable_layers(getattr(equation, "rhs_fn", None)):
            if type(layer).__name__ == "_TracedQAdapter":
                params, functions = layer.params, layer.functions
            else:
                params, functions = _named_values_from_defaults(layer)
            if params is not None:
                return dict(
                    params=params.as_dict(),
                    functions=functions.as_dict(),
                    pointwise_functions=getattr(functions, "pointwise_names", ()),
                )
    return {}


def _has_boundary_fn(fn):
    """Whether a time/scaling wrapper encloses a boundary accessor callback."""
    while fn is not None:
        if isinstance(fn, _BoundaryFn):
            return True
        fn = getattr(fn, "inner", None)
    return False


def _eval_boundary_value(fn, q, t):
    """Evaluate a boundary value with its dimensional clock and value scaling."""
    if isinstance(fn, _ScaledDirichletValueFn):
        value = _eval_boundary_value(fn.inner, q, t * fn.time_ref)
        return (jnp.asarray(value) - fn.offset) / fn.scale
    if isinstance(fn, _SITimeFn):
        return _eval_boundary_value(fn.inner, q, t * fn.time_ref)
    return fn(q, t) if isinstance(fn, _BoundaryFn) else fn(t)


# ---------------------------------------------------------------------------
# Boundary face view (the ``face`` handed to a field-dependent flux ``fn``)
# ---------------------------------------------------------------------------


_FaceView = _BoundaryQ


def _facet_local_flux_value(fn, backing, global_names, fi, normals, centroids, t, **context):
    """Evaluate a facet-local flux closure once per facet.

    Each call sees a one-facet view — ``face.<field>`` is that facet's own
    face-mean, shape ``(1,)`` — so a read of another facet is impossible by
    construction and the facet vertex-to-vertex Jacobian stencil is exact.
    The per-facet value must be a single number; a closure producing more is
    refused by the reshape below at trace time.

    Returns the ``(n_facets,)`` flux-density vector the scatter consumes.
    """
    import jax  # noqa: PLC0415

    def one_facet(fi_row, n_row, c_row):
        face = _FaceView(
            backing,
            global_names,
            fi_row[None, :],
            n_row[None, :],
            c_row[None, :],
            t=t,
            **context,
        )
        return jnp.asarray(fn(face, t)).reshape(())

    return jax.vmap(one_facet)(
        jnp.asarray(fi, dtype=jnp.int32),
        jnp.asarray(normals),
        jnp.asarray(centroids),
    )


def _face_centroids(domain, face_indices) -> np.ndarray:
    """Per-facet centroid coordinates ``(n_facets, dim)`` for a tagged FEM mesh."""
    points = np.asarray(domain.points, dtype=np.float64)
    fi = np.asarray(face_indices, dtype=np.int64)
    if fi.size == 0:
        return np.zeros((0, points.shape[1]), dtype=np.float64)
    return points[fi].mean(axis=1)


def _grid_face_geometry(domain, indices, *, bc_desc: str, face=None):
    """Boundary-face geometry of grid cells on one face, for a ``face`` view.

    Returns the ``(face_indices, normals, centroids)`` triple the face view
    reads, each shaped so one boundary cell stands in for one facet:
    ``(n_cells, 1)`` cell indices, ``(n_cells, dim)`` outward unit normals and
    ``(n_cells, dim)`` cell centres. With one index per row the face view's
    facet mean over ``value[face_indices]`` is the boundary cell's own value.

    On a one-dimensional grid each end cell exposes a single outer face: the low
    end's normal points along ``-x``, the high end's along ``+x``. On a grid of
    two or three dimensions the cells must lie on one face, named by ``face`` or
    inferred by [_grid_face_of][diff_soup._lowering.bcs._grid_face_of], and every
    cell takes that face's normal.

    Raises:
        NotImplementedError: on a domain that is not a structured grid.
        ValueError: when a selected cell exposes no outward boundary face, or the
            cells do not lie on exactly one face.
    """
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if isinstance(domain, GridND):
        axis, side = _grid_face_of(domain, idx, face=face, bc_desc=bc_desc)
        normals = np.zeros((idx.size, domain.dim), dtype=np.float64)
        normals[:, axis] = -1.0 if side == "low" else 1.0
        centres = np.asarray(domain.node_coordinates(), dtype=np.float64)[idx]
        return idx.reshape(-1, 1).astype(np.int32), normals, centres
    if not isinstance(domain, Grid1D):
        raise NotImplementedError(
            f"{bc_desc}: a field-dependent flux fn(face, t) reads the boundary's "
            "outward normal and position, which a structured grid or a tagged "
            f"FEM mesh supplies and a {type(domain).__name__} domain does not. State "
            "the density with value= / value_fn=, or put the field on a grid or a "
            "tagged FEM mesh."
        )
    n = int(domain.num_nodes)
    if face is not None:
        axis, side = _grid_face_of(domain, idx, face=face, bc_desc=bc_desc)
        normals = np.full((idx.size, 1), -1.0 if side == "low" else 1.0)
        centres = np.asarray(domain.nodes, dtype=np.float64)[idx].reshape(-1, 1)
        return idx.reshape(-1, 1).astype(np.int32), normals, centres
    interior = idx[(idx != 0) & (idx != n - 1)]
    if interior.size:
        raise ValueError(
            f"{bc_desc}: cell {int(interior[0])} lies inside the {n}-cell line, so "
            "it exposes no outward boundary face for a fn(face, t) flux to cross. "
            f"Select cell 0 or cell {n - 1}."
        )
    centres = np.asarray(domain.nodes, dtype=np.float64)[idx].reshape(-1, 1)
    normals = np.where(idx == 0, -1.0, 1.0).astype(np.float64).reshape(-1, 1)
    return idx.reshape(-1, 1).astype(np.int32), normals, centres


def _face_flux_accessed_names(
    face_flux, comp, eqset, *, bc_var, fi, normals, centroids, bc_desc: str
) -> set:
    """Probe a ``face``-style flux and return the names it reads, failing loud.

    Mirrors [_flux_fn_accessed_names][diff_soup._lowering.bcs._flux_fn_accessed_names]
    but builds a [_FaceView][diff_soup._lowering.bcs._FaceView] over
    full-shape zero dummies for the permitted names and calls ``fn(face, t)``.
    Reading a name outside the permitted set raises a directing
    [LoweringError][diff_soup._errors.LoweringError].
    """
    dummies = _flux_permitted_dummies(comp, eqset, bc_var)
    global_names = {g.name for g in eqset.globals} | set(_direct_control_names(eqset))

    params, functions = declared_named_values(eqset)
    face = _BoundaryQ(
        dummies,
        global_names,
        fi,
        normals,
        centroids,
        ops=_ProbeCtx(comp).ops,
        params=params.as_dict() if params is not None else {},
        functions=functions.as_dict() if functions is not None else {},
        pointwise_functions=getattr(functions, "pointwise_names", ()),
        controls={name: jnp.zeros(()) for name in _direct_control_names(eqset)},
    )
    try:
        face_flux(face, jnp.zeros(()))
    except (KeyError, AttributeError) as e:
        bad = e.args[0] if e.args else "?"
        raise LoweringError(
            f"{bc_desc}: boundary fn read q.{bad}, which is not a field of "
            f"component {comp.name!r}, a variable coupled into it, a global "
            "variable, or a declared control. Couple it in with "
            "spec.couple(...), or declare it with spec.global_var(...) or "
            "spec.control.direct(...)."
        ) from None
    except LoweringError:
        raise
    except Exception as e:  # noqa: BLE001 — fail closed, naming the closure.
        raise LoweringError(
            f"{bc_desc}: flux fn raised {type(e).__name__} when evaluated at "
            "lower time against zero inputs; a field-dependent flux must be "
            "callable as fn(face, t), reading only its component's fields, "
            f"coupled-in variables, globals, and face.geom ({e})."
        ) from e
    return set(face.accessed) | {f"nodal:{name}" for name in face.nodal_accessed}


# ---------------------------------------------------------------------------
# Boundary flux_fn access probe
# ---------------------------------------------------------------------------


class _RestrictedView:
    """View proxy exposing only the names a boundary ``flux_fn`` may read.

    Populated at lower-time with the flux's permitted names — its own BC field,
    variables coupled into the owning component, and globals — each a full-shape
    zero dummy.  Reading any other key raises ``KeyError``, which the probe turns
    into a directing [LoweringError][diff_soup._errors.LoweringError].  Every
    successful read is recorded in ``accessed`` so the sparsity builder can
    declare the matching Jacobian dependency.
    """

    def __init__(self, backing: dict):
        self._backing = backing
        self.accessed: set = set()

    def __getitem__(self, key):
        value = self._backing[key]  # raises KeyError for non-permitted keys
        self.accessed.add(key)
        return value

    def __contains__(self, key):
        return key in self._backing

    def items(self):
        # A dict-wide iterator can reach every backing value, so record them all
        # as accessed: a flux that reduces or indexes through items() depends on
        # the columns of each, and the sparsity builder must declare them.
        self.accessed.update(self._backing.keys())
        return self._backing.items()


class _ProbeCtx:
    """``ctx`` handed to a ``flux_fn`` during the lower-time access probe.

    Exposes the owning component's operator bank as ``ctx.ops``, resolved lazily
    so a pointwise flux that never touches ``ctx.ops`` pays nothing.  These are
    the same plain operators the residual passes the closure at run time.
    """

    __slots__ = ("_comp", "_ops")

    def __init__(self, comp) -> None:
        self._comp = comp
        self._ops = None

    @property
    def ops(self):
        """The owning component's operator bank (resolved on first access)."""
        if self._ops is None:
            self._ops = resolve_operator(self._comp.domain, self._comp.operator_factory)
        return self._ops


def _flux_permitted_dummies(comp, eqset, bc_var) -> dict:
    """Full-shape zero dummies for every name a flux on ``comp`` may read.

    Permitted = every field of the owning component (the BC variable and its
    siblings), read-through aliases consumed by the owner, and globals. A
    surface reaction is routinely a function of several fields evaluated at the
    same boundary node — a concentration and a chemical potential, say — so a
    flux reads its component's state, and anything outside that component must
    still be brought in explicitly with a coupling or declared as a global. Each
    permitted name maps to a zero array of the value's full (not
    boundary-restricted) shape so a closure that indexes or reduces the whole
    field traces cleanly.
    """
    local_by_name = {v.name: v for v in comp.variables}
    dummies: dict = {
        name: jnp.zeros(v.shape, dtype=jnp.float32) for name, v in local_by_name.items()
    }
    dummies[bc_var.name] = jnp.zeros(bc_var.shape, dtype=jnp.float32)
    endpoints = _resolve_coupling_endpoints(eqset)
    for owner_name, target_name in endpoints:
        if owner_name != comp.name or target_name not in local_by_name:
            continue
        target = local_by_name[target_name]
        dummies[target_name] = jnp.zeros(target.shape, dtype=jnp.float32)
    for owner_name, alias_var in _read_through_alias_owner(eqset).values():
        if owner_name == comp.name:
            dummies[alias_var.name] = jnp.zeros(alias_var.shape, dtype=jnp.float32)
    for g in eqset.globals:
        dummies[g.name] = jnp.zeros(np.shape(g.init), dtype=jnp.float32)
    for name in _direct_control_names(eqset):
        dummies[name] = jnp.zeros((), dtype=jnp.float32)
    return dummies


def _direct_control_names(eqset) -> tuple:
    """The declared direct-control names a flux closure may read by name."""
    return tuple(
        str(ctl.name)
        for ctl in (getattr(eqset, "controls", ()) or ())
        if getattr(ctl, "binds_to", None) is None
    )


def _flux_fn_accessed_names(flux_fn, comp, eqset, *, bc_var, bc_desc: str) -> set:
    """Probe ``flux_fn`` and return the names it reads, failing closed on misuse.

    Runs ``flux_fn(view, ctx, t)`` once against zero dummies for the permitted
    names (see [_flux_permitted_dummies][diff_soup._lowering.bcs._flux_permitted_dummies]).
    Reading a name outside that set raises
    [LoweringError][diff_soup._errors.LoweringError], directing the author to
    couple it in or declare a global; any other exception the closure raises
    against zero inputs is also re-raised as a ``LoweringError`` naming the
    closure, so a broken flux fails at lower time rather than silently.  Returns
    the set of permitted names actually read — used to declare the flux's
    Jacobian rows / coupling cross-blocks.
    """
    view = _RestrictedView(_flux_permitted_dummies(comp, eqset, bc_var))
    try:
        flux_fn(view, _ProbeCtx(comp), jnp.zeros(()))
    except KeyError as e:
        bad = e.args[0] if e.args else "?"
        raise LoweringError(
            f"{bc_desc}: flux_fn read {bad!r}, which is not a field of component "
            f"{comp.name!r}, a variable coupled into it, or a global variable. "
            "Couple it in with spec.couple(...) or declare it with "
            "spec.global_var(...)."
        ) from None
    except LoweringError:
        raise
    except Exception as e:  # noqa: BLE001 — fail closed, naming the closure.
        raise LoweringError(
            f"{bc_desc}: flux_fn raised {type(e).__name__} when evaluated at "
            "lower time against zero inputs; a flux closure must be callable as "
            "flux_fn(view, ctx, t), reading only its component's fields, "
            f"coupled-in variables, and globals ({e})."
        ) from e
    return set(view.accessed)


class _ScaledDirichletValueFn(eqx.Module):
    """dimensional-author value_fn(t_SI) wrapped for non-dim residual calls.

    The internal solver feeds non-dim time ``t_internal``; this wrapper
    multiplies by ``time_ref`` to recover dimensional time before calling the
    user's ``inner`` value function, then rescales the dimensional return value to
    internal storage units via ``(value_SI - offset) / scale``.

    Identity defaults (``scale=1.0``, ``offset=0.0``, ``time_ref=1.0``)
    reproduce the un-scaled behaviour bit-for-bit.

    The wrapper is itself an ``eqx.Module`` so ``eqx.partition`` recurses
    into ``inner`` and surfaces any traced parameters carried by an
    ``eqx.Module`` value-fn for AD — load-bearing for inversion-time BC
    parameter tracing through the dimensional-author boundary.
    """

    inner: Any
    scale: float = eqx.field(static=True)
    offset: float = eqx.field(static=True)
    time_ref: float = eqx.field(static=True)

    def __call__(self, t_internal):
        t_si = jnp.asarray(self.time_ref) * jnp.asarray(t_internal)
        v_si = jnp.asarray(self.inner(t_si))
        return (v_si - self.offset) / self.scale


class _SITimeFn(eqx.Module):
    """Present a dimensional-authored callback to the internal-time residual kernel.

    Boundary and constraint callbacks — flux densities, Robin coefficients,
    traction vectors — are authored against the experiment's own clock, like
    controls and Dirichlet values.  The kernel evaluates at internal time, so
    this shim multiplies the final positional argument (always the time) by
    ``time_ref`` before calling the author's function.  The identity default
    reproduces the un-scaled behaviour bit-for-bit.

    An ``eqx.Module`` for the same reason as
    [_ScaledDirichletValueFn][diff_soup._lowering.bcs._ScaledDirichletValueFn]:
    ``eqx.partition`` must recurse into ``inner`` so traced parameters carried
    by a module-valued callback reach AD.
    """

    inner: Any
    time_ref: float = eqx.field(static=True)

    def __call__(self, *args):
        t_si = jnp.asarray(self.time_ref) * jnp.asarray(args[-1])
        return self.inner(*args[:-1], t_si)


def _si_time(fn, time_ref: float):
    """Wrap ``fn`` so its trailing time argument arrives in dimensional units."""
    return _SITimeFn(inner=fn, time_ref=float(time_ref))


@dataclass(frozen=True)
class _ResolvedDirichlet:
    var: Variable
    dof_indices: np.ndarray
    value_fn: object
    callback_geometry: object = None


@dataclass(frozen=True)
class _ResolvedFlux:
    """Static geometry + runtime callable for one FluxBC.

    On FEM domains the kernel scatters ``flux_value * face_length /
    verts_per_face`` to each face vertex (linear-Lagrange surface mass).  On
    FVM/FDM grids the kernel adds ``flux_value * face_area / cell_volume`` at
    each boundary cell.

    ``comp_name`` is the owning component, so the residual hands the flux closure
    that component's post-transfer view (coupled-in values, aliases, globals)
    rather than a name-collision-prone global view.
    """

    comp_name: str
    var_name: str
    var_offset: int
    var_size: int
    var_shape: tuple
    residual_rows: object
    permitted_names: tuple[str, ...]
    accessed_names: tuple[str, ...]
    is_fem: bool
    # FEM:
    face_indices: object  # (n_faces, verts_per_face) int32
    face_lengths: object  # (n_faces,) float
    face_normals: object  # (n_faces, dim) float
    # FVM/FDM:
    node_indices: object  # (n_cells,) int32
    face_factor: object  # (n_cells,) float = face_area / cell_volume
    # Per-facet centroids ``(n_faces, dim)`` for the ``face.geom.x`` of a
    # field-dependent flux; ``None`` unless the flux is one.
    face_centroids: object = None
    # Stable handle declared by ``flux(..., name=)``, in any of its forms
    # (``value=``, ``value_fn=``, ``fn=``); ``None`` for an unnamed flux.  A
    # ``flux_total`` closure links to a flux by this name.
    bc_name: object = None
    # True when the flux is a field-dependent ``fn(face, t)`` closure rather than
    # a constant or time-dependent density, which lower to ``(view, ctx, t)``.
    is_face_flux: bool = False
    # Declared locality of a field-dependent flux.  ``False`` (the default):
    # the closure is evaluated once per facet, cross-facet reads are impossible
    # by construction, and the facet vertex-to-vertex stencil is its exact
    # Jacobian pattern.  ``True``: the closure sees the whole boundary and the
    # sparsity builder declares the conservative boundary-wide pattern.
    cross_facet: bool = False


@dataclass(frozen=True)
class _ResolvedRobin:
    """Static geometry + runtime callables for one RobinBC.

    Carries the boundary measure the prescribed outward derivative
    ``(value - alpha*u)/beta`` is weighted by: facet lengths spread over each
    facet's vertices on FEM, the finite-volume ``face_area / cell_volume`` at
    each boundary cell on FVM/FDM.
    """

    var_offset: int
    var_size: int
    is_fem: bool
    face_indices: object
    face_lengths: object
    face_normals: object
    node_indices: object
    face_factor: object
    callback_geometry: object = None
    # The owning component, so the residual can rescale ``face_factor`` when
    # that component's grid is scaled by a declared parameter.
    comp_name: object = None


@dataclass(frozen=True)
class _ResolvedTraction:
    """Static geometry + runtime callable for one TractionBC.

    Always FEM.  Distributes ``value * face_length / verts_per_face`` to
    each face vertex per component.  Component count is inferred from
    ``var_shape[-1]``.
    """

    var_offset: int
    var_size: int
    var_shape: tuple
    face_indices: object
    face_lengths: object
    face_normals: object
    callback_geometry: object = None


@dataclass(frozen=True)
class _ResolvedSymmetry:
    """Static geometry for one symmetry (slip-wall) boundary condition.

    Always FEM.  Each tagged node contributes one algebraic row imposing the
    normal projection ``n·u = 0`` on the vector field: ``rows[i]`` is the
    residual row the condition replaces — the DOF of the component that
    dominates that node's unit normal — and the row reads

        ``sum_k coefficients[i, k] * u[cols[i, k]] + constants[i]``

    with ``coefficients[i]`` the node's unit normal and ``cols[i]`` the node's
    component DOFs.  ``constants[i]`` carries the offset term that expresses the
    dimensional condition in internal storage units and is zero for an un-offset field.
    The node's remaining component rows keep their assembled physics, which is
    what leaves the tangential motion free.

    ``rows`` and ``cols`` are global (the variable's block offset is already
    folded in); ``nodes`` are the mesh node indices, kept for diagnostics.
    """

    var_name: str
    nodes: np.ndarray  # (n_nodes,) int64 — mesh node indices
    rows: np.ndarray  # (n_nodes,) int32 — global residual rows
    cols: np.ndarray  # (n_nodes, dim) int32 — global DOF columns
    coefficients: np.ndarray  # (n_nodes, dim) float64 — unit normal entries
    constants: np.ndarray  # (n_nodes,) float64


@dataclass(frozen=True)
class _ResolvedBCs:
    dirichlet: tuple[_ResolvedDirichlet, ...]
    flux: tuple[_ResolvedFlux, ...] = ()
    robin: tuple[_ResolvedRobin, ...] = ()
    traction: tuple[_ResolvedTraction, ...] = ()
    symmetry: tuple[_ResolvedSymmetry, ...] = ()


def _resolve_tag(domain, tag: str) -> np.ndarray:
    for attr in ("node_tags", "tags", "boundary_tags"):
        mapping = getattr(domain, attr, None)
        if isinstance(mapping, dict) and tag in mapping:
            return np.asarray(mapping[tag], dtype=np.int32)
    method = getattr(domain, "nodes_for_tag", None)
    if callable(method):
        return np.asarray(method(tag), dtype=np.int32)
    # TaggedMesh: the unique nodes of the boundary facets (edges in 2D, faces
    # in 3D and 1D) that carry the name.
    facets_method = getattr(domain, "facets_for_name", None)
    if callable(facets_method):
        try:
            facets = facets_method(tag)
        except KeyError:
            facets = None
        if facets is not None and facets.size > 0:
            return np.unique(facets.reshape(-1)).astype(np.int32)
    raise ValueError(f"cannot resolve boundary tag {tag!r}")


def _ids_for_name(domain, name: str) -> list:
    """Return all boundary-tag ids for one facet-tag name on a tagged mesh.

    Args:
        domain: A [TaggedMesh][diff_soup.domains.mesh.core.TaggedMesh]
            exposing ``_tag_ids_for_name``, ``tag_names``, and ``tag_groups``.
        name: The facet-tag name to look up.

    Returns:
        A list of integer tag ids whose name equals ``name`` (one or more).  A
        group name returns the ids of its group.

    Raises:
        ValueError: If ``name`` is neither a tag name nor a group name on
            ``domain``; the message names the missing tag and lists the
            available names.
    """
    try:
        return list(domain._tag_ids_for_name(name))
    except KeyError:
        available = sorted(
            set((domain.tag_names or {}).values()) | set(getattr(domain, "tag_groups", None) or {})
        )
        raise ValueError(
            f"boundary tag {name!r} is not defined on this mesh; available tag names: {available}"
        ) from None


def _resolve_selector(domain, selector, *, want, bc_desc=None):
    """Resolve a [BoundarySet][diff_soup.domains.boundary_set.BoundarySet].

    A [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] is a
    domain-free expression tree (tag-name leaves,
    raw-node-index leaves, unions, and differences).  This evaluates that tree
    against a concrete ``domain`` and returns one of two views:

    * ``want == "nodes"`` (a node view): a unique, sorted ``int32`` array of node
      indices.  Tag leaves expand to the union of every node on the named
      facets (via [_resolve_tag][diff_soup._lowering.bcs._resolve_tag]), node
      leaves contribute their indices
      directly, unions take the set union, and differences take the set
      difference (left minus right).
    * ``want == "facets"`` (a facet view): the
      ``(face_indices, face_lengths, face_normals)`` triple from
      [boundary_face_assembly_for_ids][diff_soup.domains.mesh.fem._boundary.
      boundary_face_assembly_for_ids] for the resolved set of boundary
      facets ([_resolve_facet_mask][diff_soup._lowering.bcs._resolve_facet_mask]):
      a tag leaf selects its facets, a node leaf the outer facets whose vertices
      all lie in the node set.

    Args:
        domain: The domain to resolve against.  A facet view requires a tagged
            FEM mesh ([TaggedMesh][diff_soup.domains.mesh.core.TaggedMesh]);
            a node view works on any domain whose tags
            [_resolve_tag][diff_soup._lowering.bcs._resolve_tag] understands.
        selector: The
            [BoundarySet][diff_soup.domains.boundary_set.BoundarySet]
            to resolve.
        want: Either ``"nodes"`` (node view) or ``"facets"`` (facet view).
        bc_desc: Optional human-readable description of the boundary target,
            woven into error messages to point back at the offending recipe.

    Returns:
        For ``want == "nodes"``: a sorted ``int32`` `numpy.ndarray` of
        unique node indices.  For ``want == "facets"``: the
        ``(face_indices, face_lengths, face_normals)`` triple.

    Raises:
        TypeError: If a facet view is requested on a domain that is not a
            tagged FEM mesh; the message names the actual domain type.
        ValueError: If a facet view resolves to no boundary facets (e.g. a
            node set holding no whole facet), or if ``want`` is neither
            ``"nodes"`` nor ``"facets"``.
    """
    if want == "nodes":
        node_set = selector.reduce(
            on_tags=lambda names: set().union(
                *[set(int(i) for i in _resolve_tag(domain, n)) for n in names]
            ),
            on_nodes=lambda idx: set(int(i) for i in idx),
            on_union=lambda a, b: a | b,
            on_difference=lambda a, b: a - b,
        )
        return np.array(sorted(node_set), dtype=np.int32)

    if want == "facets":
        if not isinstance(domain, TaggedMesh):
            raise TypeError(
                "facet-valued boundary targets (flux / Robin / traction / boundary "
                "integrals) require a tagged FEM mesh; "
                f"got domain of type {type(domain).__name__}, which has no boundary "
                "facets to integrate over."
            )
        return boundary_face_assembly_for_mask(
            domain, _resolve_facet_mask(domain, selector, bc_desc=bc_desc)
        )

    raise ValueError(f"_resolve_selector: want must be 'nodes' or 'facets', got {want!r}")


def _resolve_facet_mask(domain, selector, *, bc_desc=None) -> np.ndarray:
    """Boolean mask over ``domain.boundary_facets()`` of the facets a selector names.

    A tag leaf selects the facets carrying that name. A node leaf selects the
    outer boundary facets whose vertices all lie in the node set; an interface
    facet is never selected by nodes. Unions and differences combine the masks.

    Raises:
        ValueError: when the selection names no boundary facet.
    """
    mask = selector.reduce(
        on_tags=lambda names: boundary_facet_mask_for_ids(
            domain, [i for n in names for i in _ids_for_name(domain, n)]
        ),
        on_nodes=lambda idx: boundary_facet_mask_for_nodes(domain, idx),
        on_union=lambda a, b: a | b,
        on_difference=lambda a, b: a & ~b,
    )
    if not mask.any():
        target = f" for {bc_desc}" if bc_desc else ""
        raise ValueError(
            f"boundary target{target} resolved to no boundary facets. A node set "
            "selects the boundary facets whose vertices all lie in it; select a "
            'tagged boundary with tag="electrode" or on=ds.tags("electrode") instead.'
        )
    return mask


def _resolve_grid_local_indices(domain, where) -> np.ndarray:
    """Return local node/cell indices for a Grid1D/GridND boundary spec.

    Resolves the spec's
    [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] selector
    against the grid as a node
    view: tag-name leaves expand through the grid's tag mapping, raw-node leaves
    contribute their indices directly, and unions / differences take the set
    union / difference.  Grids carry no facet geometry, so the node view is the
    correct input to the grid flux-factor path.
    """
    indices = _resolve_selector(domain, where.selector, want="nodes")
    n = int(domain.num_nodes)
    indices = np.where(indices < 0, indices + n, indices)
    if np.any((indices < 0) | (indices >= n)):
        raise ValueError(
            f"boundary node indices must lie between {-n} and {n - 1} for this {n}-node domain"
        )
    return np.unique(indices).astype(np.int32)


_SIDES = ("low", "high")


def _grid_shape(domain) -> tuple[int, ...]:
    """Cell counts per axis of a ``Grid1D`` or ``GridND``."""
    if isinstance(domain, GridND):
        return tuple(int(n) for n in domain.shape)
    return (int(domain.num_nodes),)


def _grid_face_cells(domain, axis: int, side: str) -> np.ndarray:
    """Flat, sorted indices of the cells on one face of a grid, ``(axis, side)``."""
    shape = _grid_shape(domain)
    _check_face(shape, (axis, side), bc_desc="grid face")
    flat = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    index = [slice(None)] * len(shape)
    index[axis] = 0 if side == "low" else shape[axis] - 1
    return np.sort(flat[tuple(index)].reshape(-1)).astype(np.int32)


def _face_label(face) -> str:
    return f"axis={face[0]}, side={face[1]!r}"


def _check_face(shape, face, *, bc_desc: str) -> None:
    """Refuse an ``(axis, side)`` pair that names no face of a grid of ``shape``."""
    axis, side = face
    if side not in _SIDES:
        raise ValueError(f"{bc_desc}: side must be 'low' or 'high', got {side!r}.")
    if not 0 <= int(axis) < len(shape):
        raise ValueError(
            f"{bc_desc}: axis={axis} names no axis of this {len(shape)}-dimensional grid "
            f"(axes 0 to {len(shape) - 1})."
        )


def _grid_face_of(domain, indices, *, face=None, bc_desc: str) -> tuple[int, str]:
    """The one grid face the selected cells lie on, as ``(axis, side)``.

    A face ``(d, "low")`` holds the cells whose index along axis ``d`` is the
    first, ``(d, "high")`` the cells whose index is the last. With ``face``
    given, every cell must lie on it. Without it, the face is inferred: the
    cells must all lie on exactly one face.

    Raises:
        ValueError: when a cell lies off the named face, when the cells lie on
            more than one face at once (only edge or corner cells, or an axis
            with one cell), or when they lie on no single face (interior cells,
            or cells from several faces).
    """
    shape = _grid_shape(domain)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    multi = np.unravel_index(idx, shape)

    def on(f):
        d, side = f
        end = 0 if side == "low" else shape[d] - 1
        return multi[d] == end

    if face is not None:
        face = (int(face[0]), str(face[1]))
        _check_face(shape, face, bc_desc=bc_desc)
        off = idx[~on(face)]
        if off.size:
            raise ValueError(
                f"{bc_desc}: cell {int(off[0])} does not lie on the grid face "
                f"{_face_label(face)}. Select only cells of that face, or drop axis= "
                "and side=."
            )
        return face
    faces = [(d, side) for d in range(len(shape)) for side in _SIDES]
    candidates = [f for f in faces if bool(np.all(on(f)))]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = "; ".join(_face_label(f) for f in candidates)
        raise ValueError(
            f"{bc_desc}: the selected cells lie on {len(candidates)} grid faces at once "
            f"({names}), so the face whose area weights them is ambiguous. Select "
            "every cell of the face you mean, or name it with axis= and side=."
        )
    on_any = np.zeros(idx.shape, dtype=bool)
    for f in faces:
        on_any |= on(f)
    if not bool(np.all(on_any)):
        bad = int(idx[~on_any][0])
        raise ValueError(
            f"{bc_desc}: cell {bad} lies inside the grid, on no boundary face. The "
            "selection must be cells of one boundary face; name it with axis= and side=."
        )
    first_faces = [f for f in faces if bool(on(f)[0])]
    bad = next(
        int(idx[k]) for k in range(idx.size) if not any(bool(on(f)[k]) for f in first_faces)
    )
    raise ValueError(
        f"{bc_desc}: cells {int(idx[0])} and {bad} lie on different grid faces. The "
        "selection must be the cells of one boundary face; declare one condition per "
        "face, each with axis= and side=."
    )


def _grid_face_factor(domain, indices: np.ndarray, *, face=None, bc_desc=None) -> np.ndarray:
    """Return ``face_area / cell_volume`` at boundary cells of a Grid1D/GridND.

    This is the finite-volume weight that turns a flux density through a cell's
    outer face into the value added to that cell's residual row. Each cell is
    weighted by one face: the face named by ``face`` (``(axis, side)``) or, on a
    grid of two or three dimensions, the one face the cells lie on
    ([_grid_face_of][diff_soup._lowering.bcs._grid_face_of]). That face's area
    at a cell is the axis' end face area times the other axes' cell volumes, so
    the factor is the end face area over the axis' cell volume at the cell.

    A one-dimensional grid with no ``face`` keeps its end-cell reading: cell 0
    takes the low face, the last cell the high face.

    Raises:
        ValueError: when the cells do not lie on exactly one face.
        TypeError: on a domain that is not a structured grid.
    """
    bc_desc = bc_desc or "boundary condition"
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if isinstance(domain, Grid1D) and face is None:
        n = int(domain.num_nodes)
        face_areas = np.asarray(domain.face_areas, dtype=np.float64)
        cell_volumes = np.asarray(domain.cell_volumes, dtype=np.float64)
        out = np.empty(indices.shape, dtype=np.float64)
        for k, i in enumerate(indices.tolist()):
            i = int(i)
            # Outer face is face_areas[i+1] for the right boundary
            # cell (i==n-1), face_areas[0] for the left cell (i==0);
            # for interior cells (rare in BC context) we pick the
            # neighbouring outer face arbitrarily.
            if i == n - 1:
                area = float(face_areas[-1])
            elif i == 0:
                area = float(face_areas[0])
            else:
                area = float(face_areas[i])
            out[k] = area / float(cell_volumes[i])
        return out
    if isinstance(domain, (Grid1D, GridND)):
        axis, side = _grid_face_of(domain, indices, face=face, bc_desc=bc_desc)
        ax = domain if isinstance(domain, Grid1D) else domain.axes[axis]
        position = np.unravel_index(indices, _grid_shape(domain))[axis]
        end_area = float(np.asarray(ax.face_areas, dtype=np.float64)[0 if side == "low" else -1])
        return end_area / np.asarray(ax.cell_volumes, dtype=np.float64)[position]
    raise TypeError(
        "grid boundary metrics are defined for Grid1D and GridND domains; got "
        f"domain type {type(domain).__name__}"
    )


@dataclass(frozen=True)
class _BoundaryMeasure:
    """Resolved boundary integration weights for a boundary-integral constraint.

    ``tag_nodes`` / ``nodal_measure`` are the per-node lumped surface measure
    (the default integrand length). ``n_facets`` / ``face_areas`` surface the
    per-facet measure so a per-facet integrand integrates directly without a
    per-node round-trip. ``n_nodes`` is the owning field's full node count, the
    only length for which a full-field gather ``flat[tag_nodes]`` is in range.
    On grids ``n_facets`` / ``face_areas`` are ``None`` (no tagged-facet table).
    """

    tag_nodes: object
    nodal_measure: object
    n_facets: object
    face_areas: object
    n_nodes: int


def _boundary_integral_measure(domain, selector, *, constraint_desc: str) -> _BoundaryMeasure:
    """Resolve a boundary selector to nodal + per-facet integration weights.

    FEM facet measures are distributed equally to their linear-Lagrange
    vertices and accumulated at shared nodes. Grids carry no facet table, so a
    selected cell's measure is the area of the boundary face(s) it exposes —
    its ``face_area / cell_volume`` factor times its volume — which makes a
    grid boundary integral a face-area-weighted sum over the selected cells,
    each weighted by the one face the selection lies on.
    """
    if isinstance(domain, TaggedMesh):
        face_indices, face_measure, _normals = _resolve_selector(
            domain,
            selector,
            want="facets",
            bc_desc=constraint_desc,
        )
        faces = np.asarray(face_indices, dtype=np.int64)
        measure = np.asarray(face_measure, dtype=np.float64)
        nodes, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        nodal = np.zeros(nodes.shape, dtype=np.float64)
        per_vertex = np.repeat(measure / faces.shape[1], faces.shape[1])
        np.add.at(nodal, inverse, per_vertex)
        return _BoundaryMeasure(
            tag_nodes=nodes.astype(np.int32),
            nodal_measure=nodal,
            n_facets=int(faces.shape[0]),
            face_areas=measure,
            n_nodes=int(np.asarray(domain.points).shape[0]),
        )
    if isinstance(domain, (Grid1D, GridND)):
        nodes = _resolve_selector(domain, selector, want="nodes")
        n_nodes = int(getattr(domain, "num_nodes", nodes.shape[0]))
        factors = _grid_face_factor(domain, nodes, bc_desc=constraint_desc)
        volumes = np.asarray(domain.cell_volumes, dtype=np.float64)[nodes]
        return _BoundaryMeasure(
            tag_nodes=nodes.astype(np.int32),
            nodal_measure=factors * volumes,
            n_facets=None,
            face_areas=None,
            n_nodes=n_nodes,
        )
    raise TypeError(
        f"{constraint_desc}: boundary integrals are not supported on "
        f"domain type {type(domain).__name__}"
    )


def _wrap_dirichlet_value_fn(value_fn, scales, comp_name: str, var_name: str):
    """Wrap a DirichletBC value_fn for dimensional time + internal-units value.

    The wrapper is an
    [_ScaledDirichletValueFn][diff_soup._lowering.bcs._ScaledDirichletValueFn]
    ``eqx.Module`` so traced ``eqx.Module`` value functions surface their
    array leaves to AD via `eqx.partition`.  When the per-variable scale is
    ``1.0`` and
    offset is ``0.0`` and ``time_ref == 1.0``, the wrapped value is
    bit-identical to the raw ``value_fn`` evaluation.
    """
    s = float(scales.state_ref.get((comp_name, var_name), 1.0))
    o = float(scales.state_offset.get((comp_name, var_name), 0.0))
    t_ref = float(scales.time_ref)
    return _ScaledDirichletValueFn(inner=value_fn, scale=s, offset=o, time_ref=t_ref)


def _robin_face_stencil_pairs(
    eqset: EquationSet, layout: _BlockLayout
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary-face vertex-vertex Jacobian pairs induced by every RobinBC.

    A Robin condition prescribes the outward derivative
    ``(value - alpha*u)/beta``, whose ``alpha*u`` part reads the boundary trace.
    On FEM that trace is ``mean(u[face_verts])``, scattered back to each face
    vertex (see ``provider._build_full_residual_fn``), so for every tagged
    boundary face the residual at vertex ``a`` depends on *every* other vertex
    ``b`` sharing that face — an honest off-diagonal ``dR_a/du_b``.
    The reads-derived ``_sparsity`` walk never inspects RobinBC geometry, so
    when a field's equation has no volumetric self-stencil these face
    off-diagonals are dropped, mis-coloring the sparse assembly and corrupting
    Newton.

    Returns global ``(rows, cols)`` (offset into the flat state by the
    variable's block offset) for the full vertex-vertex cross product of each
    Robin boundary face.  Only FEM (``TaggedMesh``) domains contribute; on
    FVM/FDM grids the Robin term is a pure diagonal already covered by the
    unconditional diagonal in ``_sparsity``.  Empty when no FEM RobinBC.
    """
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for comp in eqset.components:
        domain = comp.domain
        if not isinstance(domain, TaggedMesh):
            continue
        for bc in comp.bcs:
            if not isinstance(bc, RobinBC):
                continue
            off = layout.offset_of((comp.name, bc.var.name))
            fi, _fl, _fn = _resolve_selector(
                domain,
                bc.where.selector,
                want="facets",
                bc_desc=f"RobinBC on {bc.var.name!r}",
            )
            fi = np.asarray(fi, dtype=np.int64)  # (n_faces, verts_per_face)
            if fi.size == 0:
                continue
            n_verts = fi.shape[1]
            # Full vertex-vertex cross product within each face.
            face_rows = np.repeat(fi, n_verts, axis=1).reshape(-1)
            face_cols = np.tile(fi, (1, n_verts)).reshape(-1)
            rows_list.append(face_rows + off)
            cols_list.append(face_cols + off)
    if not rows_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return (
        np.concatenate(rows_list).astype(np.int64),
        np.concatenate(cols_list).astype(np.int64),
    )


def _check_robin_beta(bc: RobinBC, var_name: str) -> None:
    """Reject a Robin condition whose ``beta`` is a constant zero.

    ``beta`` divides the prescribed outward derivative
    ``(value - alpha*u)/beta``, so a zero ``beta`` has no Robin meaning: it
    leaves ``alpha*u = value``, a Dirichlet condition.  A ``beta`` supplied as a
    callable is only known when it is evaluated and passes unchecked.
    """
    beta = constant_coefficient_of(bc.beta_fn)
    if beta is None or not bool(np.any(np.asarray(beta) == 0.0)):
        return
    raise LoweringError(
        f"spec.bc.robin on {var_name!r}: beta is 0, but alpha*u + beta*n.grad(u) = value "
        "prescribes the boundary derivative (value - alpha*u)/beta and needs a nonzero "
        "beta. With beta = 0 the condition reads alpha*u = value; declare that with "
        "spec.bc.dirichlet(...) instead, or pass a nonzero beta."
    )


def _symmetry_nodal_normals(
    face_indices: np.ndarray,
    face_measures: np.ndarray,
    face_normals: np.ndarray,
    *,
    bc_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure-weighted outward unit normal at each node of a tagged facet set.

    Every facet contributes its outward unit normal, weighted by the facet
    measure, to each of its vertices; the weighted mean is then renormalised to
    unit length. A zero-measure radial point uses the unweighted facet normal.
    On a flat tagged boundary every facet carries the same normal and the result
    reproduces it exactly; on a curved one it is the usual area-weighted nodal
    normal.

    Args:
        face_indices: ``(n_facets, verts_per_facet)`` facet connectivity.
        face_measures: ``(n_facets,)`` facet length (2D) or area (3D).
        face_normals: ``(n_facets, dim)`` outward unit facet normals.
        bc_desc: Description of the boundary condition, for error messages.

    Returns:
        ``(nodes, normals)``: the sorted unique node indices and their
        ``(n_nodes, dim)`` outward unit normals.

    Raises:
        ValueError: the facets meeting at a node have opposing normals whose
            weighted mean vanishes, so no symmetry plane passes through it.
    """
    verts_per_face = int(face_indices.shape[1])
    nodes, inverse = np.unique(face_indices.reshape(-1), return_inverse=True)
    weights = np.repeat(np.asarray(face_measures, dtype=np.float64), verts_per_face)
    per_vertex = np.repeat(face_normals, verts_per_face, axis=0) * weights[:, None]
    slot = inverse.reshape(-1)
    summed = np.zeros((nodes.shape[0], face_normals.shape[1]), dtype=np.float64)
    np.add.at(summed, slot, per_vertex)
    total = np.zeros((nodes.shape[0],), dtype=np.float64)
    np.add.at(total, slot, weights)
    averaged = np.empty_like(summed)
    positive_measure = total > 0.0
    averaged[positive_measure] = summed[positive_measure] / total[positive_measure, None]
    if np.any(~positive_measure):
        unweighted = np.zeros_like(summed)
        np.add.at(unweighted, slot, np.repeat(face_normals, verts_per_face, axis=0))
        counts = np.zeros((nodes.shape[0],), dtype=np.int64)
        np.add.at(counts, slot, 1)
        averaged[~positive_measure] = (
            unweighted[~positive_measure] / counts[~positive_measure, None]
        )
    lengths = np.linalg.norm(averaged, axis=1)
    folded = np.flatnonzero(lengths <= 1e-8)
    if folded.size:
        raise ValueError(
            f"{bc_desc}: the tagged facets meeting at node {int(nodes[folded[0]])} have "
            "opposing outward normals whose mean is zero, so no symmetry plane passes "
            "through that node. Tag one side of the fold at a time, or pin the node "
            "with spec.bc.dirichlet."
        )
    return nodes.astype(np.int64), averaged / lengths[:, None]


def _symmetry_row_column_pairs(bcs: _ResolvedBCs) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian pairs induced by every symmetry (slip-wall) row.

    A symmetry row reads ``n·u`` across all components of its node, so it
    depends on the node's OTHER component DOFs as well as its own.  The
    reads-derived sparsity walk never inspects boundary-condition geometry, so
    those cross-component entries have to be declared here or the coloured
    assembly drops them and Newton sees a wrong Jacobian.

    Returns global ``(rows, cols)``; empty when no symmetry condition is
    declared.
    """
    if not bcs.symmetry:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rows = np.concatenate(
        [np.repeat(s.rows.astype(np.int64), s.cols.shape[1]) for s in bcs.symmetry]
    )
    cols = np.concatenate([s.cols.reshape(-1).astype(np.int64) for s in bcs.symmetry])
    return rows, cols


def _dirichlet_dof_indices(nodes: np.ndarray, component, var) -> np.ndarray:
    """Expand the nodes a Dirichlet condition names into the rows it pins.

    A field's block is node-major: node ``k`` owns the ``d`` consecutive slots
    starting at ``k*d``, where ``d`` is the number of values the field carries
    per node (``1`` for a scalar field).  ``component=k`` selects slot ``k`` of
    each named node; ``component=None`` selects all of them, which is the only
    reading available on a scalar field.

    Args:
        nodes: Sorted node indices the location selector resolved to.
        component: The per-node value to pin, or ``None`` for all of them.
        var: The [Variable][diff_soup.problem._ir.variable.Variable] being
            pinned; its ``shape`` supplies the node count and the per-node
            width.

    Returns:
        The field-local flat row indices, ascending.

    Raises:
        ValueError: If a node index lies outside the field's node count, or if
            ``component`` names a value the field's nodes do not carry.
    """
    n_nodes = int(var.shape[0])
    width = int(var.size) // n_nodes if n_nodes else 0
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.size and (int(nodes.min()) < 0 or int(nodes.max()) >= n_nodes):
        bad = int(nodes.max()) if int(nodes.max()) >= n_nodes else int(nodes.min())
        raise ValueError(
            f"Dirichlet condition on {var.name!r}: node index {bad} is outside the "
            f"{n_nodes} nodes the field lives on (0 to {n_nodes - 1}). The location "
            "names nodes; to pin one value of a node, pass component=."
        )
    if component is None:
        if width == 1:
            return nodes
        return (nodes[:, None] * width + np.arange(width, dtype=np.int64)[None, :]).reshape(-1)
    component = int(component)
    if width == 1:
        raise ValueError(
            f"Dirichlet condition on {var.name!r}: component={component} selects one of "
            f"the values a node carries, but {var.name!r} carries a single value per "
            "node. Drop component=, or declare the field with components=."
        )
    if component < 0 or component >= width:
        raise ValueError(
            f"Dirichlet condition on {var.name!r}: component={component} is outside the "
            f"{width} values each node of {var.name!r} carries (0 to {width - 1})."
        )
    return nodes * width + component


def _value_callback_geometry(bc, comp, eqset):
    """Resolve geometry and probe the names a value callback reads."""
    if not _has_boundary_fn(bc.value_fn):
        return None
    domain = comp.domain
    if getattr(domain, "discretization_type", None) == "fem":
        fi, _, normals = _resolve_selector(domain, bc.where.selector, want="facets")
        centroids = _face_centroids(domain, fi)
    else:
        nodes = _resolve_grid_local_indices(domain, bc.where)
        fi, normals, centroids = _grid_face_geometry(
            domain,
            nodes,
            bc_desc=f"{type(bc).__name__} on {bc.var.name!r}",
            face=getattr(bc.where, "face", None),
        )
    accessed = _face_flux_accessed_names(
        bc.value_fn,
        comp,
        eqset,
        bc_var=bc.var,
        fi=fi,
        normals=normals,
        centroids=centroids,
        bc_desc=f"{type(bc).__name__} on {bc.var.name!r}",
    )
    return (
        comp.name,
        np.asarray(fi),
        np.asarray(normals),
        np.asarray(centroids),
        tuple(sorted(accessed)),
    )


def _bc_dofs(eqset: EquationSet, layout: _BlockLayout) -> _ResolvedBCs:
    """Resolve all BC nodes to host-side geometry + runtime callables.

    Handles DirichletBC, FluxBC, RobinBC, TractionBC, and SymmetryBC.  A
    SymmetryBC resolves to one normal-projection row per tagged node (see
    [_ResolvedSymmetry][diff_soup._lowering.bcs._ResolvedSymmetry]) rather than
    to a Dirichlet pin, because ``n·u = 0`` couples the node's components on any
    boundary that is not axis aligned.
    """
    dirichlets: list[_ResolvedDirichlet] = []
    fluxes: list[_ResolvedFlux] = []
    robins: list[_ResolvedRobin] = []
    tractions: list[_ResolvedTraction] = []
    symmetries: list[_ResolvedSymmetry] = []

    for comp in eqset.components:
        domain = comp.domain
        # Original (array-bearing) flux callables for the locality check.
        # flux_callables is populated in Component.__init__.
        _comp_flux_callables = list(getattr(comp, "flux_callables", ()))
        _flux_idx = 0
        _comp_dirichlet_value_fns = list(getattr(comp, "dirichlet_value_fns", ()))
        _dirichlet_idx = 0
        for bc in comp.bcs:
            if isinstance(bc, PeriodicBC):
                continue  # applied at the operator (periodic ops), scatters no DOF row
            var = bc.var
            off = layout.offset_of((comp.name, var.name))

            if isinstance(bc, DirichletBC):
                if _dirichlet_idx < len(_comp_dirichlet_value_fns):
                    value_fn = _comp_dirichlet_value_fns[_dirichlet_idx]
                else:
                    value_fn = bc.value_fn
                _dirichlet_idx += 1
                callback_geometry = _value_callback_geometry(bc, comp, eqset)
                nodes = _resolve_selector(domain, bc.where.selector, want="nodes")
                local = _dirichlet_dof_indices(nodes, bc.component, var)
                # Pre-wrap user value_fn so it sees dimensional time and returns
                # internal-units value at residual time.  Identity Scales
                # defaults make this bit-identical to bare bc.value_fn.
                dirichlets.append(
                    _ResolvedDirichlet(
                        var=var,
                        dof_indices=(local + off).astype(np.int32),
                        value_fn=_wrap_dirichlet_value_fn(
                            value_fn, eqset.scales, comp.name, var.name
                        ),
                        callback_geometry=callback_geometry,
                    )
                )

            elif isinstance(bc, FluxBC):
                # Get the original (array-bearing) callable for the locality check.
                if _flux_idx < len(_comp_flux_callables):
                    _orig_flux_fn = _comp_flux_callables[_flux_idx]
                else:
                    _orig_flux_fn = bc.flux_fn  # fallback (old Component without field)
                _flux_idx += 1
                permitted_names = tuple(_flux_permitted_dummies(comp, eqset, var).keys())
                is_face_flux = isinstance(_orig_flux_fn, _BoundaryFn)
                # The handle rides on the BC node for every flux form; a
                # ``_BoundaryFn`` also mirrors it, which is what a FluxBC built
                # directly from a face flux (rather than through the spec)
                # carries.
                bc_name = getattr(bc, "bc_name", None)
                if bc_name is None and is_face_flux:
                    bc_name = _orig_flux_fn.bc_name
                if isinstance(domain, TaggedMesh):
                    fi, fl, fn = _resolve_selector(
                        domain,
                        bc.where.selector,
                        want="facets",
                        bc_desc=f"FluxBC on {var.name!r}",
                    )
                    centroids = _face_centroids(domain, fi)
                    if is_face_flux:
                        accessed_names = tuple(
                            sorted(
                                _face_flux_accessed_names(
                                    _orig_flux_fn,
                                    comp,
                                    eqset,
                                    bc_var=var,
                                    fi=fi,
                                    normals=fn,
                                    centroids=centroids,
                                    bc_desc=f"FluxBC on {var.name!r}",
                                )
                            )
                        )
                    else:
                        accessed_names = tuple(
                            sorted(
                                _flux_fn_accessed_names(
                                    _orig_flux_fn,
                                    comp,
                                    eqset,
                                    bc_var=var,
                                    bc_desc=f"FluxBC on {var.name!r}",
                                )
                            )
                        )
                    residual_rows = np.unique(np.asarray(fi, dtype=np.int64).reshape(-1)) + off
                    fluxes.append(
                        _ResolvedFlux(
                            comp_name=comp.name,
                            var_name=var.name,
                            var_offset=off,
                            var_size=var.size,
                            var_shape=tuple(var.shape),
                            residual_rows=residual_rows.astype(np.int64),
                            permitted_names=permitted_names,
                            accessed_names=accessed_names,
                            is_fem=True,
                            face_indices=np.asarray(fi, dtype=np.int32),
                            face_lengths=np.asarray(fl, dtype=np.float64),
                            face_normals=np.asarray(fn, dtype=np.float64),
                            node_indices=None,
                            face_factor=None,
                            face_centroids=np.asarray(centroids, dtype=np.float64),
                            bc_name=bc_name,
                            is_face_flux=is_face_flux,
                            cross_facet=bool(getattr(bc, "cross_facet", False)),
                        )
                    )
                elif isinstance(domain, (Grid1D, GridND)):
                    if bc_name is not None:
                        raise ValueError(
                            f"spec.bc.flux(..., name={bc_name!r}) on component "
                            f"{comp.name!r}: a flux name is the handle "
                            "spec.constraint.flux_total(from_bc=...) integrates over, "
                            "and that integral needs boundary facets, which a grid "
                            "domain does not have. Drop name=, or put the field on a "
                            "tagged FEM mesh."
                        )
                    local = _resolve_grid_local_indices(domain, bc.where)
                    if np.any(local < 0) or np.any(local >= var.size):
                        raise ValueError("FluxBC node index out of range")
                    face = getattr(bc.where, "face", None)
                    factor = _grid_face_factor(
                        domain, local, face=face, bc_desc=f"flux condition on {var.name!r}"
                    )
                    grid_fi = grid_normals = grid_centroids = None
                    if is_face_flux:
                        grid_fi, grid_normals, grid_centroids = _grid_face_geometry(
                            domain, local, bc_desc=f"FluxBC on {var.name!r}", face=face
                        )
                        accessed_names = tuple(
                            sorted(
                                _face_flux_accessed_names(
                                    _orig_flux_fn,
                                    comp,
                                    eqset,
                                    bc_var=var,
                                    fi=grid_fi,
                                    normals=grid_normals,
                                    centroids=grid_centroids,
                                    bc_desc=f"FluxBC on {var.name!r}",
                                )
                            )
                        )
                    else:
                        accessed_names = tuple(
                            sorted(
                                _flux_fn_accessed_names(
                                    _orig_flux_fn,
                                    comp,
                                    eqset,
                                    bc_var=var,
                                    bc_desc=f"FluxBC on {var.name!r}",
                                )
                            )
                        )
                    fluxes.append(
                        _ResolvedFlux(
                            comp_name=comp.name,
                            var_name=var.name,
                            var_offset=off,
                            var_size=var.size,
                            var_shape=tuple(var.shape),
                            residual_rows=(local.astype(np.int64) + off),
                            permitted_names=permitted_names,
                            accessed_names=accessed_names,
                            is_fem=False,
                            face_indices=grid_fi,
                            face_lengths=None,
                            face_normals=grid_normals,
                            node_indices=local.astype(np.int32),
                            face_factor=factor,
                            face_centroids=grid_centroids,
                            is_face_flux=is_face_flux,
                            cross_facet=bool(getattr(bc, "cross_facet", False)),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"FluxBC: domain type {type(domain).__name__} not supported"
                    )

            elif isinstance(bc, RobinBC):
                _check_robin_beta(bc, var.name)
                callback_geometry = _value_callback_geometry(bc, comp, eqset)
                if isinstance(domain, TaggedMesh):
                    fi, fl, fn = _resolve_selector(
                        domain,
                        bc.where.selector,
                        want="facets",
                        bc_desc=f"RobinBC on {var.name!r}",
                    )
                    robins.append(
                        _ResolvedRobin(
                            callback_geometry=callback_geometry,
                            comp_name=comp.name,
                            var_offset=off,
                            var_size=var.size,
                            is_fem=True,
                            face_indices=np.asarray(fi, dtype=np.int32),
                            face_lengths=np.asarray(fl, dtype=np.float64),
                            face_normals=np.asarray(fn, dtype=np.float64),
                            node_indices=None,
                            face_factor=None,
                        )
                    )
                elif isinstance(domain, (Grid1D, GridND)):
                    local = _resolve_grid_local_indices(domain, bc.where)
                    factor = _grid_face_factor(
                        domain,
                        local,
                        face=getattr(bc.where, "face", None),
                        bc_desc=f"Robin condition on {var.name!r}",
                    )
                    robins.append(
                        _ResolvedRobin(
                            callback_geometry=callback_geometry,
                            comp_name=comp.name,
                            var_offset=off,
                            var_size=var.size,
                            is_fem=False,
                            face_indices=None,
                            face_lengths=None,
                            face_normals=None,
                            node_indices=local.astype(np.int32),
                            face_factor=factor,
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"RobinBC: domain type {type(domain).__name__} not supported"
                    )

            elif isinstance(bc, TractionBC):
                if not isinstance(domain, TaggedMesh):
                    raise NotImplementedError(
                        "TractionBC currently requires a TaggedMesh FEM domain"
                    )
                fi, fl, fn = _resolve_selector(
                    domain,
                    bc.where.selector,
                    want="facets",
                    bc_desc=f"TractionBC on {var.name!r}",
                )
                tractions.append(
                    _ResolvedTraction(
                        callback_geometry=_value_callback_geometry(bc, comp, eqset),
                        var_offset=off,
                        var_size=var.size,
                        var_shape=tuple(var.shape),
                        face_indices=np.asarray(fi, dtype=np.int32),
                        face_lengths=np.asarray(fl, dtype=np.float64),
                        face_normals=np.asarray(fn, dtype=np.float64),
                    )
                )

            elif isinstance(bc, SymmetryBC):
                bc_desc = f"symmetry boundary condition on {var.name!r}"
                if not isinstance(domain, TaggedMesh):
                    raise NotImplementedError(
                        f"{bc_desc}: the normal projection n·u = 0 is stated on tagged "
                        "boundary facets and is supported on FEM meshes (TaggedMesh) "
                        f"only; {var.name!r} lives on a {type(domain).__name__} domain. "
                        "Move the field to a tagged mesh, or pin the wall-normal "
                        "component of the boundary cells with spec.bc.dirichlet."
                    )
                if len(var.shape) != 2:
                    detail = (
                        "a field with one value per node has no direction to project"
                        if len(var.shape) < 2
                        else "the extra axes leave the projection undefined"
                    )
                    raise ValueError(
                        f"{bc_desc}: n·u = 0 projects the field onto the boundary "
                        "normal, so the field must carry one component per spatial "
                        f"dimension, shape (num_nodes, dim). {var.name!r} has shape "
                        f"{tuple(var.shape)}: {detail}. Declare the field with "
                        "components=<spatial dim>, or state the condition directly "
                        "with spec.bc.dirichlet or spec.bc.flux."
                    )
                fi, fl, fn = _resolve_selector(
                    domain,
                    bc.where.selector,
                    want="facets",
                    bc_desc=bc_desc,
                )
                fi_np = np.asarray(fi, dtype=np.int64)
                fl_np = np.asarray(fl, dtype=np.float64)
                fn_np = np.asarray(fn, dtype=np.float64)
                dim = int(var.shape[-1])
                if dim != int(fn_np.shape[1]):
                    raise ValueError(
                        f"{bc_desc}: n·u = 0 needs one field component per spatial "
                        f"dimension, but {var.name!r} has {dim} components while the "
                        f"boundary normal has {int(fn_np.shape[1])}. Declare the field "
                        f"with components={int(fn_np.shape[1])}."
                    )
                nodes, normals = _symmetry_nodal_normals(fi_np, fl_np, fn_np, bc_desc=bc_desc)
                if nodes.size and int(nodes[-1]) * dim + dim > int(var.size):
                    raise ValueError(
                        f"{bc_desc}: the tagged facets reach node {int(nodes[-1])}, past "
                        f"the {int(var.size) // dim} nodes {var.name!r} carries. Declare "
                        "the field on the same domain the boundary tag belongs to."
                    )
                cols = (nodes[:, None] * dim + np.arange(dim)[None, :] + off).astype(np.int32)
                # The row the condition replaces is the DOF of the component
                # that dominates the node's normal: on an axis-aligned facet
                # that is the wall-normal component, and the other rows keep
                # their physics, leaving the tangential traction free.
                dominant = np.argmax(np.abs(normals), axis=1)
                rows = cols[np.arange(nodes.shape[0]), dominant].astype(np.int32)
                # The condition is authored on the dimensional field.  With
                # ``u_SI = offset + scale * u_internal`` the row assembled in
                # internal units is ``n·u_internal + (offset/scale) * sum_k n_k``;
                # the constant vanishes for an un-offset field.
                state_scale = float(eqset.scales.state_ref.get((comp.name, var.name), 1.0))
                state_offset = float(eqset.scales.state_offset.get((comp.name, var.name), 0.0))
                symmetries.append(
                    _ResolvedSymmetry(
                        var_name=var.name,
                        nodes=nodes,
                        rows=rows,
                        cols=cols,
                        coefficients=normals,
                        constants=(state_offset / state_scale) * normals.sum(axis=1),
                    )
                )

    if dirichlets:
        all_dofs = np.concatenate([x.dof_indices for x in dirichlets])
        if len(np.unique(all_dofs)) != len(all_dofs):
            raise ValueError("duplicate DirichletBC DOF")
    _check_symmetry_rows_disjoint(symmetries)
    return _ResolvedBCs(
        dirichlet=tuple(dirichlets),
        flux=tuple(fluxes),
        robin=tuple(robins),
        traction=tuple(tractions),
        symmetry=tuple(symmetries),
    )


def _check_symmetry_rows_disjoint(symmetries: list[_ResolvedSymmetry]) -> None:
    """Reject two symmetry conditions that claim one residual row.

    Each tagged node gives up exactly one row — the DOF of its dominant normal
    component — so two symmetry planes meeting at a node are representable only
    while their dominant components differ (the usual quarter-model corner,
    where one plane claims the x row and the other the y row).  When both claim
    the same row, one condition would silently overwrite the other.
    """
    if not symmetries:
        return
    rows = np.concatenate([s.rows for s in symmetries])
    unique_rows, counts = np.unique(rows, return_counts=True)
    if unique_rows.size == rows.size:
        return
    clash = int(unique_rows[int(np.argmax(counts))])
    owners = [s for s in symmetries if bool(np.any(s.rows == clash))]
    node = int(owners[0].nodes[int(np.flatnonzero(owners[0].rows == clash)[0])])
    names = sorted({s.var_name for s in owners})
    raise ValueError(
        f"two symmetry boundary conditions on {' and '.join(repr(n) for n in names)} "
        f"impose n·u = 0 through the same residual row at node {node}: their normals "
        "share a dominant component, so the second condition would replace the first. "
        "Declare the two facet sets in one spec.bc.symmetry call when they share a "
        "plane, or pin the node's components with spec.bc.dirichlet when they do not."
    )


def _extract_flux_callables(eqset: EquationSet) -> tuple:
    """Extract per-FluxBC callables in the same order ``_bc_dofs`` walks them.

    Reads from ``comp.flux_callables`` (the non-static runtime copy with JAX
    array leaves) rather than ``comp.bcs[i].flux_fn``, which carries a stripped
    (array-free) version for static fingerprinting.  Every callable is wrapped
    so its time argument arrives in dimensional units — the clock every callback shares.
    """
    time_ref = _callback_time_ref(eqset)
    out = []
    for comp in eqset.components:
        # Prefer the non-static flux_callables field which holds the
        # originals; fall back to comp.bcs for old Component objects without the field.
        fc = getattr(comp, "flux_callables", None)
        if fc is not None:
            out.extend(fc)
        else:
            for bc in comp.bcs:
                if isinstance(bc, FluxBC):
                    out.append(bc.flux_fn)
    return tuple(_si_time(fn, time_ref) for fn in out)


def _extract_robin_callables(eqset: EquationSet) -> tuple:
    """Extract per-RobinBC ``(alpha_fn, beta_fn, value_fn)`` triples.

    Each callable is wrapped so its time argument arrives in dimensional units.
    """
    time_ref = _callback_time_ref(eqset)
    out = []
    for comp in eqset.components:
        for bc in comp.bcs:
            if isinstance(bc, RobinBC):
                out.append(
                    (
                        _si_time(bc.alpha_fn, time_ref),
                        _si_time(bc.beta_fn, time_ref),
                        _si_time(bc.value_fn, time_ref),
                    )
                )
    return tuple(out)


def _extract_traction_callables(eqset: EquationSet) -> tuple:
    """Extract per-TractionBC value_fn callables.

    Each callable is wrapped so its time argument arrives in dimensional units.
    """
    time_ref = _callback_time_ref(eqset)
    out = []
    for comp in eqset.components:
        for bc in comp.bcs:
            if isinstance(bc, TractionBC):
                out.append(_si_time(bc.value_fn, time_ref))
    return tuple(out)


def _callback_time_ref(eqset: EquationSet) -> float:
    """The dimensional seconds one internal time unit stands for (``1.0`` unscaled)."""
    scales = getattr(eqset, "scales", None)
    return float(getattr(scales, "time_ref", 1.0)) if scales is not None else 1.0
