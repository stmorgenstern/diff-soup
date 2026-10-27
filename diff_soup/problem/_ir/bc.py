"""Boundary condition IR nodes.

Frozen, pytree-registered ``eqx.Module`` nodes representing boundary conditions
in the EquationSet IR.  Each node carries the target [Variable][] and a
[BoundarySpec][] selecting where the condition applies.  Lowering
dispatches by ``isinstance`` on the [BoundaryCondition][] marker base.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diff_soup.domains.boundary_set import BoundarySet, from_nodes, tags

from .variable import Variable


class BoundarySpec(eqx.Module):
    """Boundary selector: *where* a boundary condition applies.

    The source of truth is [selector][], a [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] expression
    tree (build with ``tags(...)``, combine with ``|`` / ``-``).  Downstream
    consumers resolve a selection against a concrete mesh via
    ``selector.reduce(...)``.

    Construct with exactly one of ``selector=``, ``tag=``, or ``node_indices=``:

    * ``tag="left"`` is sugar for ``selector=tags("left")``.
    * ``node_indices=...`` (an integer array-like or a boolean mask) is sugar for
      ``selector=from_nodes(...)``.
    * ``selector=`` takes any [BoundarySet][diff_soup.domains.boundary_set.BoundarySet], including a composed one.

    For the two simple construction paths the matching convenience attribute is
    also populated so existing readers keep working: a selector that is a single
    ``tags`` leaf of one name exposes that name as [tag][], and a single
    ``nodes`` leaf exposes those indices as [node_indices][] (a tuple of
    Python ints).  For any composed selector both convenience attributes are
    ``None`` — read [selector][] instead.

    ``node_indices`` is stored as a tuple of Python ints rather than a NumPy
    array.  Equinox's static-field check treats both NumPy and JAX arrays as
    "array leaves" and warns on construction; the tuple keeps the field
    hashable, leaf-free, and free of that warning, while
    ``np.asarray(spec.node_indices, dtype=np.int32)`` recovers the array form on
    demand.  [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] is itself frozen and hashable, so the static
    selector field keeps [BoundarySpec][] hashable and frozen-style.
    """

    selector: Optional[BoundarySet] = eqx.field(static=True, default=None)
    tag: Optional[str] = eqx.field(static=True, default=None)
    node_indices: Optional[tuple[int, ...]] = eqx.field(static=True, default=None)

    def __init__(self, *, selector=None, node_indices=None, tag: str | None = None):
        given = [selector is not None, node_indices is not None, tag is not None]
        if sum(given) != 1:
            raise ValueError("BoundarySpec requires exactly one of selector, tag, or node_indices")
        if tag is not None:
            self.selector = tags(tag)
            self.tag = tag
            self.node_indices = None
        elif node_indices is not None:
            arr0 = np.asarray(node_indices)
            if arr0.dtype == bool:
                # A boolean mask selects the True positions; casting it
                # straight to int would coerce True/False to the literal
                # indices {0, 1} and silently pin the wrong rows.
                node_indices = np.nonzero(arr0.reshape(-1))[0]
            arr = np.asarray(node_indices, dtype=np.int32).reshape(-1)
            idx = tuple(int(x) for x in arr)
            self.selector = from_nodes(idx)
            self.tag = None
            self.node_indices = idx
        else:
            self.selector = selector
            # Expose the convenience attributes only for a single simple leaf;
            # a composed selector leaves both None (read .selector instead).
            self.tag = (
                selector.names[0] if selector.op == "tags" and len(selector.names) == 1 else None
            )
            self.node_indices = (
                tuple(int(x) for x in selector.nodes) if selector.op == "nodes" else None
            )


class BoundaryCondition(eqx.Module):
    """Marker base for boundary condition nodes."""

    var: Variable
    where: BoundarySpec


def _constant_fn(value):
    """Wrap a constant boundary datum in a callable of time.

    Every boundary datum callable is a function of time alone —
    ``DirichletBC.value_fn(t)``, ``TractionBC.value_fn(t)``, and
    ``RobinBC.{alpha,beta,value}_fn(t)`` — so a constant becomes a callable that
    ignores the time and returns the constant.

    The default ``_value`` is the LAST positional default so the JSON codec can
    extract it via ``fn.__defaults__[-1]``.
    """
    arr = jnp.asarray(value)

    def value_fn(t, _value=arr):  # noqa: ARG001
        return _value

    return value_fn


def _time_flux_fn(value_fn):
    """Adapt a boundary flux density ``value_fn(t)`` to the internal flux form.

    ``FluxBC.flux_fn`` is called as ``(view, ctx, t)`` so that a flux may depend
    on the boundary state; a flux declared as a function of time alone ignores
    the first two arguments.

    The user's callable is the LAST positional default so the JSON codec can
    recover it via ``fn.__defaults__[-1]``.
    """

    def flux_fn(view, ctx, t, _value_fn=value_fn):  # noqa: ARG001
        return _value_fn(t)

    return flux_fn


def constant_coefficient_of(fn):
    """The constant carried by a boundary coefficient, or ``None`` if it varies.

    A coefficient handed to a boundary condition as a plain number is wrapped by
    [_constant_fn][diff_soup.problem._ir.bc._constant_fn], which parks the value
    in the closure's last positional default.  Recovering it lets a caller
    validate a constant coefficient before any evaluation happens.  Returns
    ``None`` for a genuine callable, whose value is only known when it is called.

    A runtime type-check wrapper (the import hook used under test) replaces the
    closure's ``__defaults__``, so unwrap first; ``inspect.unwrap`` is a no-op
    when nothing wrapped it.
    """
    raw = inspect.unwrap(fn)
    if (
        getattr(raw, "__qualname__", None) == "_constant_fn.<locals>.value_fn"
        and getattr(raw, "__module__", None) == __name__
        and raw.__defaults__
    ):
        return raw.__defaults__[-1]
    return None


class DirichletBC(BoundaryCondition):
    """Dirichlet row replacement for a variable.

    ``where`` names *nodes*.  On a field carrying several values per node the
    state is node-major — node ``k`` owns the flat slots ``k*d`` through
    ``k*d + d - 1`` — and ``component`` says which of those slots the condition
    pins: an integer in ``range(d)`` pins that one, and ``None`` pins all of
    them (the full clamp).  ``component`` is static and never enters the
    residual numerics.
    """

    value_fn: Callable
    component: Optional[int] = eqx.field(static=True, default=None)

    def __init__(
        self,
        *,
        var: Variable,
        where: BoundarySpec,
        value=None,
        value_fn=None,
        component: Optional[int] = None,
    ):
        if value_fn is not None and value is not None:
            raise ValueError("provide only one of value or value_fn")
        if value_fn is None:
            value_fn = _constant_fn(0.0 if value is None else value)
        self.var = var
        self.where = where
        self.value_fn = value_fn
        self.component = None if component is None else int(component)


class FluxBC(BoundaryCondition):
    """Flux boundary contribution.

    ``bc_name`` is the optional stable handle a
    [FluxTotalConstraint][diff_soup.problem._ir.constraint.FluxTotalConstraint]
    links to.  It belongs to the boundary condition, not to the callable that
    states its density, so it is carried the same way whether ``flux_fn`` is a
    constant, a function of time, or a field-dependent
    [_BoundaryFn][diff_soup.problem._ir.bc._BoundaryFn].  It is static and never
    enters the residual numerics.
    """

    flux_fn: Callable
    bc_name: Optional[str] = eqx.field(static=True, default=None)
    cross_facet: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        *,
        var: Variable,
        where: BoundarySpec,
        flux_fn: Callable,
        bc_name: Optional[str] = None,
        cross_facet: bool = False,
    ):
        self.var = var
        self.where = where
        self.flux_fn = flux_fn
        self.bc_name = bc_name
        self.cross_facet = bool(cross_facet)


class _BoundaryFn(eqx.Module):
    """A boundary value authored as ``fn(q, t)`` for any condition kind.

    The evaluator supplies the owning component's traced parameters, functions,
    controls, and facet geometry through ``q`` at each residual evaluation.
    """

    fn: Callable
    bc_name: Optional[str] = eqx.field(static=True, default=None)

    def __call__(self, q, t):
        q._t = t
        return self.fn(q, t)


class RobinBC(BoundaryCondition):
    """Robin BC: ``alpha * u + beta * n·grad(u) = value`` (or ``value_fn(t)``).

    ``n`` is the outward normal, so the condition prescribes the outward normal
    derivative ``(value - alpha * u) / beta`` at the boundary.  Lowering
    assembles that derivative as a boundary flux density positive *into* the
    domain, matching the sign convention of [FluxBC][], and weights it with the
    same boundary measure (the finite-volume ``face_area / cell_volume`` on a
    grid, the facet length spread over the facet's vertices on a mesh).

    ``beta`` must be nonzero — ``beta == 0`` states a Dirichlet condition, and
    lowering rejects it.  ``alpha``, ``beta``, and ``value`` may each be a plain
    number or a callable of time alone.
    """

    alpha_fn: Callable
    beta_fn: Callable
    value_fn: Callable

    def __init__(
        self,
        *,
        var: Variable,
        where: BoundarySpec,
        alpha,
        beta,
        value=None,
        value_fn=None,
    ):
        if value_fn is not None and value is not None:
            raise ValueError("RobinBC: provide only one of value or value_fn")
        if value_fn is None:
            value_fn = _constant_fn(0.0 if value is None else value)
        if not callable(alpha):
            alpha_fn = _constant_fn(alpha)
        else:
            alpha_fn = alpha
        if not callable(beta):
            beta_fn = _constant_fn(beta)
        else:
            beta_fn = beta
        self.var = var
        self.where = where
        self.alpha_fn = alpha_fn
        self.beta_fn = beta_fn
        self.value_fn = value_fn


class TractionBC(BoundaryCondition):
    """Traction BC: ``sigma · n = t`` for a vector displacement field.

    Operates on the boundary trace of a vector field; ``value`` is the traction
    vector (or ``value_fn(t)`` returning one).
    """

    value_fn: Callable

    def __init__(
        self,
        *,
        var: Variable,
        where: BoundarySpec,
        value=None,
        value_fn=None,
    ):
        if value_fn is not None and value is not None:
            raise ValueError("TractionBC: provide only one of value or value_fn")
        if value_fn is None:
            value_fn = _constant_fn(jnp.zeros(()) if value is None else value)
        self.var = var
        self.where = where
        self.value_fn = value_fn


class SymmetryBC(BoundaryCondition):
    """Symmetry BC: ``n·u = 0`` for a vector field on a boundary facet set.

    Material slides along a symmetry plane but never through it, so the
    condition is the scalar normal projection ``n·u = 0`` at each tagged node,
    with ``n`` the measure-weighted unit normal of the facets meeting there.
    Used for half-cell mirrors, quarter models, slip walls.

    Lowering replaces one residual row per tagged node — the DOF of the
    component that dominates that node's normal — with the projection, and
    leaves the node's remaining rows on their assembled physics.  No traction
    term is assembled: whatever the field equation contributes to the
    tangential rows at the boundary stands, which for a weak-form momentum or
    elasticity equation is the natural (traction-free) condition.

    ``var`` must be a vector field with one component per spatial dimension —
    a scalar has no direction to project and is refused — and the domain must be
    a tagged FEM mesh, the only discretization carrying the facet normals the
    projection needs.
    """

    def __init__(self, *, var: Variable, where: BoundarySpec):
        self.var = var
        self.where = where


class PeriodicBC(BoundaryCondition):
    """Periodic wrap-around for a field on a cartesian grid.

    Declares that ``var`` is periodic on the listed grid ``axes`` — the value at
    one end of each axis wraps to the other end — and the Jacobian sparsity gains
    the wrap entries. There is no boundary location to select, so
    [where][diff_soup.problem._ir.bc.BoundaryCondition.where]
    carries an empty node selector.

    The wrap reaches the residual through the operator bank: ``laplacian``,
    ``div_flux`` and ``advect_upwind`` take a boundary-condition list, so they
    use a wrap-around stencil instead of the natural zero-flux boundary. The
    operators built from a one-sided edge difference — ``gradient`` / ``grad``,
    ``divergence`` / ``div``, ``face_flux`` — have no such argument and raise
    under a periodic declaration rather than return their non-periodic value.
    """

    axes: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, *, var: Variable, axes: tuple[int, ...]):
        self.var = var
        self.where = BoundarySpec(node_indices=())
        self.axes = tuple(int(a) for a in axes)
