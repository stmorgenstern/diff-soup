"""Unified spatial operator dispatch.

``Operators(domain)`` provides a common API for PDE operators that routes
to the correct backend (FVM/FDM/FEM) based on ``domain.discretization_type``.

Common operators (work on all domain types):
    ``laplacian``, ``div_flux``, ``gradient``, ``divergence``,
    ``mass``, ``lumped_mass``

Quad-accurate integrated forms (explicit advanced surface):
    ``integrate``

Grid-specific operators (FVM/FDM only):
    ``face_flux``

FEM-specific operators (Mesh only):
    ``strain``, ``interp_to_quad``

All functions are pure, JIT-compatible, and AD-compatible.
"""

from __future__ import annotations

import functools
import inspect
from abc import abstractmethod
from typing import Any, ClassVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from diff_soup.domains.base import Domain
from diff_soup.domains.grid.advection.bcs import PeriodicBC
from diff_soup.domains.mesh.fem.assembly import scatter_elem_residual
from diff_soup.domains.operators._boundary_ops import boundary_integral as _bi
from diff_soup.domains.operators._boundary_ops import boundary_trace as _bt
from diff_soup.domains.operators._boundary_ops import surface_flux as _sf
from diff_soup.domains.operators._fem_scalar import cell_indicator as _cell_indicator
from diff_soup.domains.operators._fem_scalar import grad as _scalar_grad
from diff_soup.domains.operators._fem_scalar import integrate_grad
from diff_soup.domains.operators._quadfield import QuadField
from diff_soup.domains.operators._tensor import det as _det
from diff_soup.domains.operators._tensor import double_dot as _dd
from diff_soup.domains.operators._tensor import inv as _inv
from diff_soup.domains.operators._tensor import outer as _outer
from diff_soup.domains.operators._tensor import scale as _scale
from diff_soup.domains.operators._tensor import skew as _skew
from diff_soup.domains.operators._tensor import sym as _sym
from diff_soup.domains.operators._tensor import tensor_contract as _tc
from diff_soup.domains.operators._tensor import trace as _trace
from diff_soup.domains.operators._tensor import transpose as _trsp

from . import dispatch as _d
from ._bruggeman import tortuosity_corrected as _tortuosity_corrected
from ._field import Field as _Field
from ._fluid import strain_rate as _strain_rate
from ._fluid import viscous_stress_newtonian as _viscous_stress_newtonian
from ._solid import cauchy_stress_from_S_F as _cauchy_stress_from_S_F
from ._solid import deformation_gradient as _deformation_gradient
from ._solid import linear_elastic_stress as _linear_elastic_stress
from ._solid import strain as _strain

# Real union type alias enforced by the jaxtyping + beartype hook at runtime.
# Methods where a Field is legitimately acceptable (FEM-applicable entry
# points that funnel through ``_as_dofs``) annotate their primary field
# argument as ``_NodealOrField`` carrying the Float shape spec.  The beartype
# hook rejects wrong types (str, wrong-ndim arrays) at the signature boundary.
# Methods where Field can never legitimately arrive keep the original pure
# ``Float[Array, "..."]`` annotations (face_flux, lumped_mass).
_NodealOrField = Union[Float[Array, "..."], _Field]  # type: ignore[misc]


#: Capability members of the operator-bank contract: every one exists on
#: ``AbstractOperator`` as a raising stub, and ``provides`` reports the subset
#: a bank overrides.  ``_WeakFormView`` routes exactly these names through its
#: composed bank (everything else forwards via ``__getattr__``).
_CAPABILITY_MEMBERS = frozenset(
    {
        "laplacian",
        "div_flux",
        "gradient",
        "divergence",
        "curl",
        "mass",
        "lumped_mass",
        "advect_upwind",
        "div_advect",
        "wrap_quad_value",
        "quad_gradient",
        "integrate_shape",
        "integrate_shape_gradient",
    }
)

#: Members defined concretely on the ABC that a subclass may still override
#: and the residual view must therefore route to its composed bank rather
#: than answer from the ABC default.
_ADAPTER_MEMBERS = frozenset({"unwrap_field", "wrap_quad_field"})


class AbstractOperator(eqx.Module):
    """The base class of an operator bank, which an equation closure reaches as `q.ops`.

    Subclass it for a new discretization, implementing `discretization_type` and the operators it offers, and register it with `spec.domain(name, domain, operators=...)`. Operators it leaves out raise `NotImplementedError`. A bank used under `spec.bc.periodic` must accept `bcs=` on `laplacian`, `div_flux`, `advect_upwind` and `div_advect`, and one used under `spec.bc.neumann` on `laplacian` and `div_flux`.

    Attributes:
        domain: The domain the bank is bound to.
        weak_form: The names of operators whose result on a mesh is already integrated against the test functions, merged across subclasses.
    """

    domain: Any
    weak_form: ClassVar[frozenset] = frozenset()

    @property
    @abstractmethod
    def discretization_type(self) -> str:
        """A string tag for the discretization strategy.

        Subclasses must implement this property.

        Returns:
            One of ``'fvm'``, ``'fdm'``, ``'fem'`` for the built-in backends,
            or a custom string for novel discretizations.
        """

    def laplacian(
        self,
        u: Any,
        *,
        bcs: "Any | None" = None,
        t: "Any | None" = None,
    ) -> Any:
        """The Laplacian ∇²u.

        The base class raises `NotImplementedError`.

        Args:
            u: The field values.
            bcs: Grid boundary conditions to apply to the result, or `None`.
            t: The time for time-dependent boundary conditions, or `None`.

        Returns:
            The Laplacian term.
        """
        del u, bcs, t
        raise self._capability_error("laplacian", "laplacian assembly")

    @property
    def provides(self) -> frozenset:
        """The names of the standard operators this bank implements, such as `laplacian` or `mass`."""
        out = set()
        for name in _CAPABILITY_MEMBERS:
            if getattr(type(self), name, None) is not getattr(AbstractOperator, name, None):
                out.add(name)
        return frozenset(out)

    @classmethod
    def _weak_form_union(cls) -> frozenset:
        """``weak_form`` unioned over the MRO.

        A subclass declares only its own already-integrated operators; the
        union keeps every inherited declaration, so forgetting the parent set
        cannot silently untag an inherited operator.
        """
        out: frozenset = frozenset()
        for klass in cls.__mro__:
            declared = klass.__dict__.get("weak_form")
            if isinstance(declared, frozenset):
                out = out | declared
        return out

    # ------------------------------------------------------------------
    # Capability-error helper
    # ------------------------------------------------------------------

    def _capability_error(self, method: str, capability: str) -> NotImplementedError:
        """Build the standard capability error for an unsupported optional member.

        The message names the concrete bank, the method, and the capability it
        was needed for. Subclasses that support the capability override the
        method; banks that do not let the default raise this error.
        """
        return NotImplementedError(
            f"{type(self).__name__} does not implement {method}; required for {capability}"
        )

    # ------------------------------------------------------------------
    # Optional strong-form hooks (safe defaults; subclasses override)
    # ------------------------------------------------------------------

    def unwrap_field(self, value):
        """Return the nodal/DOF storage backing a wrapped field.

        Default is the identity: a bank with no separate wrapped-field
        representation returns ``value`` unchanged. FEM banks override this to
        unwrap a quad-native field view to its nodal DOF array.
        """
        return value

    def wrap_quad_field(self, value):
        """Create the quad-native ``q.<field>`` view of a nodal/DOF array.

        Non-FEM banks have no quadrature, so this is the identity. FEM banks
        must override it to build the quad-native field view; the base raises a
        capability error there rather than silently passing the array through.
        """
        if self.discretization_type == "fem":
            raise self._capability_error("wrap_quad_field", "FEM quad-native field views")
        return value

    def wrap_quad_value(self, value):
        """Create an explicit, sticky quadrature value (``q.ops.wrap_quad_value``).

        This is a quadrature capability rather than a safe adapter. Banks that
        support explicit quadrature values must override it; the base raises a
        capability error rather than silently treating a nodal value as a
        quadrature value.
        """
        raise self._capability_error("wrap_quad_value", "explicit quadrature values")

    def quad_gradient(self, value):
        r"""Gradient evaluated at quadrature points.

        Scalar input returns ``(n_cells, n_q, dim)`` and vector input returns
        ``(n_cells, n_q, n_components, dim)``, with ``H[..., i, j] = du_i/dx_j``.
        Banks without quadrature do not support this; subclasses that do must
        override it.
        """
        raise self._capability_error("quad_gradient", "quadrature-point gradients")

    def integrate_shape(self, value_q):
        r"""Assemble :math:`\int \text{value\_q}\, N_i`.

        Returns an already-integrated residual term. Banks without shape-function
        assembly do not support this; subclasses that do must override it.
        """
        raise self._capability_error("integrate_shape", "shape-function integration")

    def integrate_shape_gradient(self, flux_q):
        r"""Assemble the integration-by-parts form :math:`\int \text{flux\_q} \cdot \nabla N_i`.

        Includes the backend's supported scalar-flux and stress representations
        and returns an already-integrated residual term. Banks without
        shape-gradient assembly do not support this; subclasses that do must
        override it.
        """
        raise self._capability_error("integrate_shape_gradient", "shape-gradient integration")

    def as_weak(self, value):
        """Mark `value` as an already-integrated weak-form term.

        On a mesh, pass a hand-assembled residual through it so the residual is not integrated a second time. On a grid it returns `value` unchanged.

        Args:
            value: One value per node, already integrated against the test functions.

        Returns:
            The marked term on a mesh, `value` on a grid.
        """
        if isinstance(value, _d._WeakTerm):
            return value
        if self.discretization_type == "fem":
            return _d._WeakTerm(value, integrated=True, domain=self.domain)
        return value

    def for_residual(self):
        """Return the bank as equation closures see it, as `q.ops`.

        On a mesh the view marks the result of every operator in `weak_form` as already integrated. On a grid it is the bank itself.
        """
        if self.discretization_type != "fem":
            return self
        return _WeakFormView(domain=self.domain, inner=self)

    # ------------------------------------------------------------------
    # Optional common operators (capability error by default)
    # ------------------------------------------------------------------

    def div_flux(self, *args, **kwargs):
        """The diffusion term ∇·(D∇u), called as `div_flux(D, u, *, bcs=None, t=None)`.

        The base class raises `NotImplementedError`.
        """
        raise self._capability_error("div_flux", "divergence-of-flux assembly")

    def gradient(self, *args, **kwargs):
        """The gradient ∇u, called as `gradient(u)`.

        The base class raises `NotImplementedError`.
        """
        raise self._capability_error("gradient", "gradient assembly")

    def divergence(self, *args, **kwargs):
        """The divergence ∇·v, called as `divergence(v)`.

        The base class raises `NotImplementedError`.
        """
        raise self._capability_error("divergence", "divergence assembly")

    def curl(self, *args, **kwargs):
        r"""Curl :math:`\nabla \times \mathbf{u}` of a vector field.

        Banks that support it must override this method; the base raises a
        capability error.
        """
        raise self._capability_error("curl", "curl assembly")

    def mass(self, *args, **kwargs):
        """The mass-matrix action on a field, called as `mass(u)`.

        The base class raises `NotImplementedError`.
        """
        raise self._capability_error("mass", "mass-matrix action")

    def lumped_mass(self, *args, **kwargs):
        """The integration weights of the domain, called as `lumped_mass()`.

        The base class raises `NotImplementedError`.
        """
        raise self._capability_error("lumped_mass", "lumped-mass integration weights")

    def advect_upwind(self, *args, **kwargs):
        """Upwind advection of a field, called as `advect_upwind(u, velocity, *, bcs=None, t=None)`.

        The base class raises `NotImplementedError`.
        """
        raise self._capability_error("advect_upwind", "upwind advection assembly")

    def div_advect(self, *args, **kwargs):
        """Conservative advection of a field, called as `div_advect(u, velocity, *, scheme="tvd", limiter="van_leer", bcs=None, t=None)`.

        The base class raises `NotImplementedError`.
        """
        raise self._capability_error("div_advect", "conservative advection assembly")


class Operators(AbstractOperator):
    """The built-in operator bank, reached in an equation closure as `q.ops`.

    On a grid each operator returns pointwise values, and on a mesh its weak form, integrated against each basis function. Subclass it to add an operator, listing in `weak_form` any that returns an integrated term, and register the class with `spec.domain(name, domain, operators=...)`.

    Args:
        domain: The [Domain][diff_soup.domains.Domain] to act on.
    """

    domain: Domain

    #: The built-in operators whose FEM return value is already integrated
    #: against the test functions.  Subclasses union in their own additions
    #: automatically; see ``AbstractOperator.weak_form``.
    weak_form: ClassVar[frozenset] = frozenset(
        {
            "laplacian",
            "div_flux",
            "gradient",
            "grad",
            "divergence",
            "div",
            "curl",
            "mass",
            "integrate",
        }
    )

    def __check_init__(self):
        """Validate that ``domain`` is a recognized Domain subtype."""
        if not isinstance(self.domain, Domain):
            raise TypeError(
                f"Operators: expected a Domain instance "
                f"(Mesh / Grid1D / GridND / ...), got {type(self.domain).__name__}"
            )

    @property
    def discretization_type(self) -> str:
        """The discretization of the bound domain, `"fvm"`, `"fdm"` or `"fem"`."""
        return self.domain.discretization_type

    # ------------------------------------------------------------------
    # Field wrap/unwrap (FEM strong-form authoring ergonomics)
    # ------------------------------------------------------------------

    def unwrap_field(self, value):
        """Return the nodal values of a field view such as `q.u`, or `value` unchanged otherwise.

        The built-in operators apply it to their inputs, so they accept either form.

        Args:
            value: A field view or an array.

        Returns:
            The nodal values.
        """
        return self._as_dofs(value)

    @staticmethod
    def _as_dofs(u):
        """Unwrap a `Field` to its nodal DOF array; pass others through.

        A quad-opted ``q.<name>`` returns a
        [Field][diff_soup.domains.operators._field.Field] that decays to a raw
        quad array under arithmetic / ``jnp`` ufuncs, but differential and
        assembly operators consume the *nodal* DOF vector.  Every such operator
        funnels its primary field argument through this helper so the
        grid-style strong-form closure (``q.ops.laplacian(q.c)``) runs on FEM
        with no quad/nodal bookkeeping.
        """
        return u.nodal if isinstance(u, _Field) else u

    def wrap_quad_field(self, val):
        """Wrap nodal values in the field view that `q.<name>` returns on a mesh.

        Arithmetic on the view samples the field at the quadrature points, and `.nodal` gives the nodal values. Indexing the view raises `TypeError`, so index `.nodal` instead. On a grid it returns `val` unchanged.

        Args:
            val: The nodal values.

        Returns:
            The view on a mesh, `val` on a grid.
        """
        if getattr(self.domain, "discretization_type", None) != "fem":
            return val
        return _Field(val, self)

    # ------------------------------------------------------------------
    # Common operators (all domain types)
    # ------------------------------------------------------------------

    def laplacian(
        self,
        u: _NodealOrField,
        *,
        bcs: "Sequence | None" = None,
        t: "Array | None" = None,
    ) -> Float[Array, "..."]:
        """The Laplacian ∇²u, pointwise on a grid and the weak form -∫ ∇u·∇N_i on a mesh.

        On a curvilinear grid it includes the metric.

        Args:
            u: A scalar field.
            bcs: Flux and periodic conditions applied on a grid, which problems declare with `spec.bc`.
            t: The time passed to a time-dependent flux condition, zero if `None`.

        Returns:
            One value per node.
        """
        u = self._as_dofs(u)
        result = _d.dispatch_laplacian(self.domain, u)
        if bcs:
            result = _d.dispatch_apply_bcs(self.domain, result, u, jnp.array(1.0), bcs, t=t)
        return result

    def div_flux(
        self,
        D: Any,
        u: _NodealOrField,
        *,
        bcs: "Sequence | None" = None,
        t: "Array | None" = None,
    ) -> Float[Array, "..."]:
        """The diffusion term ∇·(D∇u), pointwise on a grid and the weak form -∫ D∇u·∇N_i on a mesh.

        On a grid a tuple `(Dx, Dy[, Dz])` gives each axis its own diffusivity, a diagonal tensor. A full tensor is refused.

        Args:
            D: The diffusivity, a scalar or one value per node, on a mesh also one per quadrature point, on a grid also one entry per axis.
            u: A scalar field.
            bcs: Flux and periodic conditions applied on a grid, which problems declare with `spec.bc`.
            t: The time passed to a time-dependent flux condition, zero if `None`.

        Returns:
            One value per node.
        """
        D = self._as_dofs(D)
        u = self._as_dofs(u)
        result = _d.dispatch_div_flux(self.domain, D, u)
        if bcs:
            result = _d.dispatch_apply_bcs(self.domain, result, u, D, bcs, t=t)
        return result

    def advect_upwind(
        self,
        field: _NodealOrField,
        velocity: Any,
        *,
        bcs: "Sequence | None" = None,
        t: "Array | None" = None,
    ) -> Float[Array, "..."]:
        """First-order upwind advection v·∇u, on a grid.

        The sign of the velocity at each cell picks the upwind side. The result is the operator itself, so a transport equation writes `-q.ops.advect_upwind(q.u, v)`. Boundaries that are not periodic are zero-gradient outflow.

        Args:
            field: The advected field.
            velocity: One constant speed for every axis, or a sequence with one entry per axis, each a constant or one value per cell.
            bcs: Periodic conditions only, which problems declare with `spec.bc.periodic`.
            t: Ignored.

        Returns:
            The advection term, shaped like `field`.
        """
        del t
        u = self._as_dofs(field)
        return _d.dispatch_advect_upwind(self.domain, u, velocity, bcs)

    def div_advect(
        self,
        field: _NodealOrField,
        velocity: Any,
        *,
        scheme: str = "tvd",
        limiter: str = "van_leer",
        bcs: "Sequence | None" = None,
        t: "Array | None" = None,
    ) -> Float[Array, "..."]:
        """Advection in conservative form ∇·(v u), on a grid.

        Each interior face carries the face velocity times a face value taken from the side the flow comes from. Under `"tvd"` the face value is the upstream value plus a limited share of the change to the downstream one: second order where `u` is smooth, and no new extrema at any cell Péclet number. Under `"weno3"` it is the third-order WENO reconstruction of Jiang and Shu. Under `"upwind"` it is the upstream value. Under `"central"` the face carries the mean of `v u` in its two cells instead. A face next to a boundary cell lacks the second upstream cell, and `"tvd"` and `"weno3"` drop to the upstream value there. A boundary face carries only an outgoing flux, so a transport equation writes `-q.ops.div_advect(q.c, v)` and feeds an inlet with `spec.bc.inflow`.

        Args:
            field: The advected field.
            velocity: One constant speed for every axis, or a sequence with one entry per axis, each a constant or one value per cell.
            scheme: The face flux, `"tvd"` (default, flux-limited second order), `"weno3"` (third order), `"central"` (second order, unlimited) or `"upwind"` (first order).
            limiter: The `"tvd"` limiter, `"van_leer"` (default), `"koren"` or `"minmod"`.
            bcs: Periodic conditions only, which problems declare with `spec.bc.periodic`.
            t: Ignored.

        Returns:
            The advection term, shaped like `field`.
        """
        del t
        u = self._as_dofs(field)
        return _d.dispatch_div_advect(self.domain, u, velocity, bcs, scheme, limiter)

    def gradient(self, u: _NodealOrField) -> Float[Array, "..."]:
        """The gradient ∇u, pointwise on a one-dimensional grid.

        On a mesh, a scalar field gives the weak form ∫ ∇u N_i, shape `(n, dim)`, and a vector field its Jacobian at the nodes, shape `(n, components, dim)`. It has no periodic form, so it raises under `spec.bc.periodic`.

        Args:
            u: A scalar field, or on a mesh a vector field.

        Returns:
            The gradient at every node.
        """
        u = self._as_dofs(u)
        return _d.dispatch_gradient(self.domain, u)

    def divergence(
        self,
        vec: _NodealOrField,
    ) -> Float[Array, "n"]:
        """The divergence ∇·v, pointwise on a grid and the weak form ∫ (∇·v) N_i on a mesh.

        On a mesh it also takes a nodal tensor, row by row, or a symmetric tensor at the quadrature points, assembled as ∫ σ : ∇N_i. It has no periodic form, so it raises under `spec.bc.periodic`.

        Args:
            vec: A vector field of shape `(n, dim)`, or on a mesh a tensor of shape `(n, dim, dim)` or `(n_cells, n_q, dim, dim)`.

        Returns:
            One value per node for a vector field, one vector per node for a tensor.
        """
        vec = self._as_dofs(vec)
        return _d.dispatch_divergence(self.domain, vec)

    def curl(self, u: _NodealOrField) -> Float[Array, "..."]:
        """The curl ∇×u of a vector field with one component per axis, in two or three dimensions.

        In two dimensions it is the scalar ∂u_y/∂x - ∂u_x/∂y, in three the curl vector. It is pointwise on a grid and the weak form ∫ (∇×u) N_i on a mesh. It has no periodic form, so it raises under `spec.bc.periodic`.

        Args:
            u: A vector field of shape `(n, dim)`.

        Returns:
            One value per node in two dimensions, one vector per node in three.
        """
        u = self._as_dofs(u)
        return _d.dispatch_curl(self.domain, u)

    def mass(self, u: _NodealOrField) -> Float[Array, "n"]:
        """The mass-matrix action, the identity on a grid and the consistent mass ∫ u N_i on a mesh.

        Use it in a custom residual that applies the mass weighting explicitly.

        Args:
            u: Nodal values, or on a mesh values at the quadrature points.

        Returns:
            One value per node.
        """
        u = self._as_dofs(u)
        return _d.dispatch_mass(self.domain, u)

    def lumped_mass(self) -> Float[Array, "n"]:
        """The integration weights, the cell volumes on a grid and the lumped nodal masses on a mesh.

        `jnp.sum(q.ops.lumped_mass() * f)` integrates a field over the domain.

        Returns:
            One weight per node.
        """
        return _d.dispatch_lumped_mass(self.domain)

    # ------------------------------------------------------------------
    # Grid-specific operators (FVM/FDM)
    # ------------------------------------------------------------------

    def face_flux(
        self,
        u: Float[Array, "..."],
        D: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """The diffusive flux -D ∂u/∂x at the faces of a one-dimensional grid.

        The two end faces carry zero flux. It has no periodic form, so it raises under `spec.bc.periodic`.

        Args:
            u: One value per cell.
            D: The diffusivity, a scalar, one value per cell, or one per face.

        Returns:
            One flux per face, `n + 1` values.
        """
        return _d.dispatch_face_flux(self.domain, u, D)

    # ------------------------------------------------------------------
    # Continuum-mechanics and transport verbs (q.ops)
    # ------------------------------------------------------------------

    def strain(self, u: _NodealOrField) -> Float[Array, "n dim dim"]:
        """The small strain sym(∇u) of a displacement field, at the nodes of a mesh.

        Args:
            u: A displacement field, shape `(n, dim)`.

        Returns:
            The strain at each node, shape `(n, dim, dim)`.
        """
        _d._require_fem(self.domain, "strain")
        u = self._as_dofs(u)
        return _strain(u, mesh=self.domain)

    def linear_elastic_stress(
        self,
        u: _NodealOrField,
        *,
        E,
        nu,
        eigenstrain=None,
    ) -> Float[Array, "n dim dim"]:
        """The isotropic linear-elastic stress λ tr(ε) I + 2μ ε, at the nodes of a mesh.

        The Lamé constants λ and μ follow from `E` and `nu`, and the strain ε is sym(∇u) less any eigenstrain.

        Args:
            u: A displacement field, shape `(n, dim)`.
            E: Young's modulus.
            nu: Poisson's ratio.
            eigenstrain: A nodal strain subtracted before the stress law, shape `(n, dim, dim)`.

        Returns:
            The stress at each node, shape `(n, dim, dim)`.
        """
        _d._require_fem(self.domain, "linear_elastic_stress")
        u = self._as_dofs(u)
        eigenstrain = self._as_dofs(eigenstrain)  # nodal (n, d, d); None passes through
        return _linear_elastic_stress(u, E=E, nu=nu, mesh=self.domain, eigenstrain=eigenstrain)

    def deformation_gradient(self, gradu):
        """The deformation gradient F = I + ∇u.

        Args:
            gradu: The displacement gradient, shape `(..., dim, dim)`.

        Returns:
            `F`, of the same shape.
        """
        return _deformation_gradient(gradu)

    def cauchy_stress_from_S_F(self, S, F):
        """The Cauchy stress F S Fᵀ / det(F) from the second Piola-Kirchhoff stress `S`.

        Args:
            S: The second Piola-Kirchhoff stress, shape `(..., dim, dim)`.
            F: The deformation gradient, of the same shape.

        Returns:
            The Cauchy stress, of the same shape.
        """
        return _cauchy_stress_from_S_F(S, F)

    # Fluid
    def strain_rate(self, u):
        """The strain rate sym(∇v) of a velocity field, at the nodes of a mesh.

        Args:
            u: A velocity field, shape `(n, dim)`.

        Returns:
            The strain rate at each node, shape `(n, dim, dim)`.
        """
        _d._require_fem(self.domain, "strain_rate")
        u = self._as_dofs(u)
        return _strain_rate(u, mesh=self.domain)

    def viscous_stress_newtonian(self, u, *, mu):
        """The Newtonian viscous stress 2μ sym(∇v) of a velocity field, at the nodes of a mesh.

        Args:
            u: A velocity field, shape `(n, dim)`.
            mu: The dynamic viscosity.

        Returns:
            The stress at each node, shape `(n, dim, dim)`.
        """
        _d._require_fem(self.domain, "viscous_stress_newtonian")
        u = self._as_dofs(u)
        return _viscous_stress_newtonian(u, mu=mu, mesh=self.domain)

    # Battery
    def tortuosity_corrected(self, D, eps, beta: float = 1.5):
        """The Bruggeman-corrected diffusivity D ε^β of a porous medium.

        Args:
            D: The bulk diffusivity.
            eps: The porosity ε.
            beta: The Bruggeman exponent β.

        Returns:
            The effective diffusivity, broadcast over `D` and `eps`.
        """
        return _tortuosity_corrected(D, eps, beta=beta)

    def cell_indicator(self, name):
        """One per cell of a named region of a tagged mesh and zero elsewhere, shape `(n_cells,)`.

        Args:
            name: A region name in the mesh's `cell_tag_names`.
        """
        return _cell_indicator(self.domain, name)

    def boundary_integral(self, field, *, tag):
        """The integral of a nodal field over a tagged boundary of a mesh.

        Each facet contributes the consistent share of its node values times its measure.

        Args:
            field: A nodal field.
            tag: The boundary, by name or by integer tag.

        Returns:
            The integral, a scalar.
        """
        field = self._as_dofs(field)  # boundary ops index nodal DOFs
        return _bi(field, tag=tag, mesh=self.domain)

    def boundary_trace(self, field, *, tag):
        """The values of a nodal field at the nodes of a tagged boundary of a mesh.

        Args:
            field: A nodal field.
            tag: The boundary, by name or by integer tag.

        Returns:
            The field at those nodes, in ascending node order.
        """
        field = self._as_dofs(field)
        return _bt(field, tag=tag, mesh=self.domain)

    def surface_flux(self, field, *, tag, kinetics, ctx=None, t=0.0):
        """A boundary flux from a kinetics law, evaluated on a field's values at a tagged boundary of a mesh.

        Args:
            field: A nodal field.
            tag: The boundary, by name or by integer tag.
            kinetics: A function `kinetics(trace, ctx, t)`, called with the boundary values from `boundary_trace`.
            ctx: Passed to `kinetics`.
            t: Passed to `kinetics`.

        Returns:
            The result of `kinetics`.
        """
        field = self._as_dofs(field)
        return _sf(field, tag=tag, kinetics=kinetics, mesh=self.domain, ctx=ctx, t=t)

    # Pointwise tensor primitives
    def trace(self, T):
        """The trace of a tensor over its last two axes.

        Args:
            T: A tensor, shape `(..., dim, dim)`.

        Returns:
            The trace, shape `(...)`.
        """
        return _trace(T)

    def det(self, T):
        """The determinant of a tensor over its last two axes.

        Args:
            T: A tensor, shape `(..., dim, dim)`.

        Returns:
            The determinant, shape `(...)`.
        """
        return _det(T)

    def inv(self, T):
        """The inverse of a tensor over its last two axes.

        Args:
            T: A tensor, shape `(..., dim, dim)`.

        Returns:
            The inverse, of the same shape.
        """
        return _inv(T)

    def transpose(self, T):
        """The transpose of a tensor over its last two axes.

        Args:
            T: A tensor, shape `(..., a, b)`.

        Returns:
            The transpose, shape `(..., b, a)`.
        """
        return _trsp(T)

    def sym(self, T):
        """The symmetric part (T + Tᵀ) / 2 of a tensor.

        Args:
            T: A tensor, shape `(..., dim, dim)`.

        Returns:
            The symmetric part, of the same shape.
        """
        return _sym(T)

    def skew(self, T):
        """The antisymmetric part (T - Tᵀ) / 2 of a tensor.

        Args:
            T: A tensor, shape `(..., dim, dim)`.

        Returns:
            The antisymmetric part, of the same shape.
        """
        return _skew(T)

    def outer(self, u, v):
        """The outer product of two vectors, batched over leading axes.

        Args:
            u: A vector, shape `(..., dim)`.
            v: A vector, shape `(..., dim)`.

        Returns:
            The tensor `u_i v_j`, shape `(..., dim, dim)`.
        """
        return _outer(u, v)

    def tensor_contract(self, A, B, axes: int = 1):
        """The contraction of the last `axes` axes of `A` with the first `axes` axes of `B`, as `jnp.tensordot`.

        Args:
            A: The first tensor.
            B: The second tensor.
            axes: The number of axes to contract.

        Returns:
            The contracted tensor.
        """
        return _tc(A, B, axes)

    def double_dot(self, A, B):
        """The double contraction A : B over the last two axes.

        A rank-four `A`, with trailing shape `(dim, dim, dim, dim)`, gives the tensor C : ε, as in a stress law. Otherwise the result is one scalar per point.

        Args:
            A: A tensor, shape `(..., dim, dim)`, or a stiffness, shape `(..., dim, dim, dim, dim)`.
            B: A tensor, shape `(..., dim, dim)`.

        Returns:
            A scalar per point, or a tensor per point for a rank-four `A`.
        """
        return _dd(A, B)

    def scale(self, scalar_field, tensor):
        """The product of a scalar field and a tensor, broadcast over the tensor's components.

        Args:
            scalar_field: One scalar per point.
            tensor: A constant tensor, or one whose leading axes match `scalar_field`.

        Returns:
            The scaled tensor at every point.
        """
        return _scale(scalar_field, tensor)

    def grad_quad(
        self,
        u_vec: _NodealOrField,
    ) -> Float[Array, "n_cells n_q dim dim"]:
        """The gradient of a vector field at the quadrature points of a mesh.

        Args:
            u_vec: A nodal vector field, shape `(n, components)`.

        Returns:
            The gradient `∂u_i/∂x_j`, shape `(n_cells, n_q, components, dim)`.
        """
        u_vec = self._as_dofs(u_vec)
        return _d.dispatch_grad_quad(self.domain, u_vec)

    # Convenience aliases used by closure idioms
    def grad(self, u):
        """An alias of `gradient`."""
        u = self._as_dofs(u)
        return _d.dispatch_gradient(self.domain, u)

    def div(self, vec_or_tensor):
        """An alias of `divergence`."""
        vec_or_tensor = self._as_dofs(vec_or_tensor)
        return _d.dispatch_divergence(self.domain, vec_or_tensor)

    def interp_to_quad(
        self,
        u: _NodealOrField,
    ) -> Float[Array, "n_cells n_q"]:
        """Sample a nodal field at the quadrature points of a mesh.

        Args:
            u: A nodal field, scalar or with trailing component axes.

        Returns:
            The values at the quadrature points, shape `(n_cells, n_q, ...)`.
        """
        u = self._as_dofs(u)
        return _d.dispatch_interp_to_quad(self.domain, u)

    def integrate(self, expr: _NodealOrField) -> Float[Array, "n"]:
        """The weak integral ∫ f N_i of an expression on a mesh, the identity on a grid.

        Args:
            expr: Nodal values, or values at the quadrature points of shape `(n_cells, n_q, ...)`.

        Returns:
            One value per node.
        """
        expr = self._as_dofs(expr)
        return _d.dispatch_integrate(self.domain, expr)

    # ------------------------------------------------------------------
    # Strong-form authoring hooks (FEM-only; the Recipe verbs route here)
    # ------------------------------------------------------------------
    #
    # ``q.grad`` / ``q.flux`` / ``q.source`` and ``q.ops.wrap_quad_value`` are
    # thin routers over these four hooks: the authoring layer never touches element
    # connectivity, shape arrays, quadrature weights, or scatter routines —
    # this bank owns those details.  On a grid (FVM/FDM) every hook raises the
    # capability error and points at the strong-form operators instead.

    def quad_gradient(self, value):
        """The gradient of a nodal field at the quadrature points of a mesh, the value `q.grad` returns.

        Args:
            value: A nodal scalar or vector field.

        Returns:
            The gradient, shape `(n_cells, n_q, dim)` for a scalar field and `(n_cells, n_q, components, dim)` for a vector field.
        """
        if self.discretization_type != "fem":
            raise self._capability_error("quad_gradient", "quadrature-point gradients")
        nodal = self._as_dofs(value)
        if jnp.ndim(nodal) == 1:
            fe = _d._get_fe_space(self.domain)
            cell_u = nodal[fe.dof_cells]  # (n_cells, n_nodes_per_cell)
            return jax.vmap(_scalar_grad)(fe.shape_grads, cell_u)  # (n_cells, n_q, dim)
        # Vector / tensor field: the displacement-gradient tensor per quad point.
        return self.grad_quad(nodal)

    def integrate_shape(self, value_q):
        """Assemble ∫ f N_i from values at the quadrature points of a mesh, the term `q.source` returns.

        Args:
            value_q: The values at the quadrature points, shape `(n_cells, n_q, ...)`.

        Returns:
            One value per node, marked as an integrated weak-form term.
        """
        if self.discretization_type != "fem":
            raise self._capability_error("integrate_shape", "shape-function integration")
        arr = jnp.asarray(self._as_dofs(value_q))
        # Assemble the RAW (untagged) nodal integral directly through the
        # dispatch layer, then tag it once with ``as_weak``.
        raw = _d.dispatch_integrate_scalar_at_quad(self.domain, arr)
        return self.as_weak(raw)

    def integrate_shape_gradient(self, flux_q):
        """Assemble ∫ F·∇N_i from a flux at the quadrature points of a mesh, the term `q.flux` returns.

        A flux vector gives one value per node. A symmetric stress in Voigt order gives one vector per node.

        Args:
            flux_q: A flux of shape `(n_cells, n_q, dim)`, or a Voigt stress of shape `(n_cells, n_q, 3)` in 2-D and `(n_cells, n_q, 6)` in 3-D.

        Returns:
            The integrated term per node, marked as a weak-form term.
        """
        if self.discretization_type != "fem":
            raise self._capability_error("integrate_shape_gradient", "shape-gradient integration")
        arr = jnp.asarray(self._as_dofs(flux_q))
        fe = _d._get_fe_space(self.domain)
        dim = fe.dim
        if arr.ndim != 3:
            raise ValueError(
                "integrate_shape_gradient expects a quad-point flux of shape "
                "(n_cells, n_q, dim) for a scalar field or "
                "(n_cells, n_q, n_voigt) for a Voigt stress tensor; got shape "
                f"{tuple(arr.shape)}."
            )
        last = arr.shape[-1]
        n_voigt = (1, 3, 6)[dim - 1]
        # ``dim == n_voigt`` only in 1D (both 1): the scalar flux-divergence path
        # is correct there; the Voigt branch is reached only for a genuine stress
        # tensor in 2D/3D.
        if last == dim:
            # Assemble the RAW (untagged) nodal residual directly; tagging is
            # applied once below via ``as_weak``.
            f_elem = jax.vmap(integrate_grad)(fe.shape_grads, fe.jxw, arr)
            raw = scatter_elem_residual(f_elem, fe.dof_cells, fe.num_dofs, n_vars=1)
            return self.as_weak(raw)
        if last == n_voigt:
            # Voigt stress: assemble through the same dispatch path ``div``
            # uses for a quad-point tensor, so the value matches it exactly,
            # then tag once.  Calling ``self.div`` would double-tag on the
            # provenance-tagging ``q.ops`` view.
            raw = _d.dispatch_stress_divergence(self.domain, arr)
            return self.as_weak(raw)
        raise ValueError(
            f"integrate_shape_gradient: trailing axis {last} of {tuple(arr.shape)} "
            f"matches neither dim={dim} (scalar flux vector) nor "
            f"n_voigt={n_voigt} (Voigt stress tensor)."
        )

    def wrap_quad_value(self, value):
        """Wrap values at the quadrature points of a mesh so arithmetic on them stays there.

        Multiplying by a constant tensor broadcasts over its components. Apply a nonlinearity with `.apply(fn)`, which keeps the wrapper. A residual integrates the result as ∫ f N_i.

        Args:
            value: The values at the quadrature points, shape `(n_cells, n_q, ...)`.

        Returns:
            The wrapped values.
        """
        if self.discretization_type != "fem":
            raise self._capability_error("wrap_quad_value", "FEM quadrature values")

        return QuadField(value, self.domain)


class _WeakFormView(AbstractOperator):
    """``q.ops`` view that COMPOSES an FEM bank and tags its weak-form outputs.

    Built by [AbstractOperator.for_residual][diff_soup.domains.operators.base.AbstractOperator.for_residual]
    around whatever bank the component resolved to — built-in or
    recipe-supplied — so subclass overrides always survive.  Operators named
    in the composed bank's ``weak_form`` MRO union come back wrapped by
    ``as_weak`` (idempotent, so an operator that assembled with
    ``integrate_shape`` is not tagged twice); every other member forwards
    unchanged.  A residual closure can then write bare strong-form math and
    the leftover nodal part is integrated exactly once by
    ``finalize_residual``, while every already-integrated term passes through
    untouched.

    The bare ``Operators`` bank is deliberately left untouched so direct
    ``Operators(domain).laplacian(...)`` call sites keep returning plain
    arrays; tagging is scoped entirely to ``q.ops``.
    """

    inner: AbstractOperator

    @property
    def discretization_type(self) -> str:
        """The composed bank's discretization tag."""
        return self.inner.discretization_type

    @property
    def provides(self) -> frozenset:
        """The composed bank's capability set."""
        return self.inner.provides

    def for_residual(self):
        """This view is already the residual view."""
        return self

    def as_weak(self, value):
        """Tag through the composed bank (idempotent)."""
        return self.inner.as_weak(value)

    def _route(self, name: str):
        """The composed bank's ``name``, tagged when it is a weak-form member."""
        inner = self.inner
        member = getattr(inner, name)
        if name in type(inner)._weak_form_union() and callable(member):

            @functools.wraps(member)
            def tagged(*args, **kwargs):
                return inner.as_weak(member(*args, **kwargs))

            return tagged
        return member

    def __getattr__(self, name):
        """Forward any member this view does not define to the composed bank."""
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(name) from None
        return self._route(name)


def _make_weak_route(name: str):
    """Build the explicit ``_WeakFormView`` route for an ABC-defined member.

    Members defined on ``AbstractOperator`` (capability stubs and adapters)
    resolve by normal attribute lookup, so ``__getattr__`` never fires for
    them; each gets an explicit route so the composed bank's implementation —
    not the ABC default — answers.
    """

    def route(self, *args, **kwargs):
        return self._route(name)(*args, **kwargs)

    route.__name__ = name
    route.__qualname__ = f"_WeakFormView.{name}"
    route.__doc__ = f"Route ``{name}`` to the composed bank, tagging weak-form output."
    return route


for _name in sorted(_CAPABILITY_MEMBERS | _ADAPTER_MEMBERS):
    setattr(_WeakFormView, _name, _make_weak_route(_name))


def _accepts_bcs(method) -> bool:
    """Whether ``method`` takes a ``bcs`` argument the periodic wrap can ride on.

    True when the callable declares a ``bcs`` parameter that can be passed by
    keyword, or absorbs one through ``**kwargs``. A callable whose signature
    cannot be read (a builtin, a C extension) is assumed to accept it, so an
    unreadable signature never blocks a bank that would in fact work.
    """
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return True
    for p in params.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == "bcs" and p.kind is not inspect.Parameter.POSITIONAL_ONLY:
            return True
    return False


class _PeriodicOperatorView(AbstractOperator):
    """``q.ops`` view that carries a periodic wrap onto the bank it composes.

    Selected as the residual operator view for a component whose fields are
    declared periodic on a cartesian grid. It composes the bank the component
    resolved to — built-in or recipe-supplied via ``operators=`` — so the
    declaration reaches a custom bank exactly as it reaches the built-in one.

    ``laplacian``, ``div_flux``, ``advect_upwind`` and ``div_advect`` take a
    boundary-condition list, so the wrap rides in as one
    [PeriodicBC][diff_soup.domains.grid.advection.bcs.PeriodicBC] per declared
    axis and a spec body such as ``dc/dt = q.ops.laplacian(q.c)`` is periodic
    without naming any boundary condition.

    ``gradient`` / ``grad``, ``divergence`` / ``div`` and ``face_flux`` build
    their edge values from a one-sided difference and take no boundary-condition
    list, so the wrap has no channel into them; under a periodic declaration
    they raise rather than return the non-periodic value. Every other member —
    pointwise algebra, integration weights, field wrap/unwrap, FEM assembly — is
    forwarded to the composed bank unchanged.
    """

    inner: AbstractOperator
    periodic_axes: tuple[int, ...] = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Bank identity: ``domain`` is the inherited declared field, set to the
    # composed bank's own domain (identity preserved) at construction.
    # ------------------------------------------------------------------

    @property
    def discretization_type(self) -> str:
        """The composed bank's discretization tag."""
        return self.inner.discretization_type

    def for_residual(self):
        """This view is already the residual view; re-entry must keep the wrap."""
        return self

    def __getattr__(self, name):
        """Forward any member this view does not define to the composed bank."""
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)

    # ------------------------------------------------------------------
    # Operators the wrap reaches through their ``bcs`` argument.
    # ------------------------------------------------------------------

    def _with_periodic(self, bcs):
        """Merge one periodic tag per declared axis into a boundary-condition list."""
        extra = [PeriodicBC(axis=a) for a in self.periodic_axes]
        if not extra:
            return bcs
        return list(bcs) + extra if bcs else extra

    def _wrappable(self, name: str):
        """The composed bank's ``name``, or an error if it cannot take the wrap."""
        method = getattr(self.inner, name)
        if not _accepts_bcs(method):
            raise NotImplementedError(
                f"{type(self.inner).__name__}.{name} takes no bcs argument, so the "
                f"periodic wrap on grid axes {list(self.periodic_axes)} cannot be "
                f"passed to it. An operator bank used under a periodic declaration "
                f"must accept bcs= on laplacian, div_flux and advect_upwind."
            )
        return method

    def laplacian(self, u, *, bcs=None, t=None):
        """Laplacian with the declared periodic wrap merged into ``bcs``."""
        return self._wrappable("laplacian")(u, bcs=self._with_periodic(bcs), t=t)

    def div_flux(self, D, u, *, bcs=None, t=None):
        """Divergence-of-flux with the declared periodic wrap merged into ``bcs``."""
        return self._wrappable("div_flux")(D, u, bcs=self._with_periodic(bcs), t=t)

    def advect_upwind(self, field, velocity, *, bcs=None, t=None):
        """Upwind advection with the declared periodic wrap merged into ``bcs``."""
        return self._wrappable("advect_upwind")(field, velocity, bcs=self._with_periodic(bcs), t=t)

    def div_advect(self, field, velocity, *, scheme="tvd", limiter="van_leer", bcs=None, t=None):
        """Conservative advection with the declared periodic wrap merged into ``bcs``."""
        return self._wrappable("div_advect")(
            field, velocity, scheme=scheme, limiter=limiter, bcs=self._with_periodic(bcs), t=t
        )

    # ------------------------------------------------------------------
    # Operators with no periodic stencil: refused, never answered.
    # ------------------------------------------------------------------

    def _no_periodic_stencil(self, name: str) -> NotImplementedError:
        """Build the refusal for an operator the periodic wrap cannot reach."""
        return NotImplementedError(
            f"{name} has no periodic form: the field it reads is declared periodic on "
            f"grid axes {list(self.periodic_axes)}, but {name} differences one-sidedly "
            f"at the ends of each axis instead of wrapping across them, so its result "
            f"would be the non-periodic one. The operators that carry the periodic "
            f"wrap are laplacian, div_flux, advect_upwind and div_advect."
        )

    def gradient(self, *args, **kwargs):
        """Refused: the grid gradient has no periodic stencil."""
        raise self._no_periodic_stencil("gradient")

    def grad(self, *args, **kwargs):
        """Refused: alias of `gradient`, which has no periodic stencil."""
        raise self._no_periodic_stencil("grad")

    def divergence(self, *args, **kwargs):
        """Refused: the grid divergence has no periodic stencil."""
        raise self._no_periodic_stencil("divergence")

    def div(self, *args, **kwargs):
        """Refused: alias of `divergence`, which has no periodic stencil."""
        raise self._no_periodic_stencil("div")

    def face_flux(self, *args, **kwargs):
        """Refused: the raw face fluxes have no periodic wrap-around face."""
        raise self._no_periodic_stencil("face_flux")

    def curl(self, *args, **kwargs):
        """Refused: the grid curl differences each axis with no periodic stencil."""
        raise self._no_periodic_stencil("curl")

    # ------------------------------------------------------------------
    # Non-differential members of the bank contract: straight delegation.
    # ------------------------------------------------------------------

    def mass(self, *args, **kwargs):
        """Mass-matrix action, from the composed bank."""
        return self.inner.mass(*args, **kwargs)

    def lumped_mass(self, *args, **kwargs):
        """Lumped-mass integration weights, from the composed bank."""
        return self.inner.lumped_mass(*args, **kwargs)

    def unwrap_field(self, value):
        """Unwrap a wrapped field to its DOF storage, via the composed bank."""
        return self.inner.unwrap_field(value)

    def wrap_quad_field(self, value):
        """Wrap DOFs as a quad-native field, via the composed bank."""
        return self.inner.wrap_quad_field(value)

    def wrap_quad_value(self, value):
        """Build the sticky quadrature value, via the composed bank."""
        return self.inner.wrap_quad_value(value)

    def quad_gradient(self, value):
        """Quadrature-point gradient, from the composed bank."""
        return self.inner.quad_gradient(value)

    def integrate_shape(self, value_q):
        """Shape-function integration, from the composed bank."""
        return self.inner.integrate_shape(value_q)

    def integrate_shape_gradient(self, flux_q):
        """Shape-gradient integration, from the composed bank."""
        return self.inner.integrate_shape_gradient(flux_q)

    def as_weak(self, value):
        """Already-integrated weak-form provenance tag, from the composed bank."""
        return self.inner.as_weak(value)


class _FieldConditionView:
    """``q.ops`` view that applies conditions stated on particular fields.

    ``spec.bc.neumann`` fixes the outward normal derivative of one field. The
    condition reaches the residual through ``laplacian`` and ``div_flux``: when
    either is applied to that field's own values (``q.<name>``, recognised by
    identity), the view imposes the field's evaluated conditions with the
    operator's own diffusivity. On a grid a
    [_NormalDerivativeBC][diff_soup.domains.operators.flux_bc._NormalDerivativeBC]
    joins ``bcs``. On a mesh a
    [_MeshNormalDerivative][diff_soup.domains.operators.flux_bc._MeshNormalDerivative]
    adds the natural boundary term ``int D s N_i dS``. An operator applied to
    anything else, or any other member, is the composed bank's own.
    """

    __slots__ = ("_inner", "_targets")

    def __init__(self, inner, targets) -> None:
        self._inner = inner
        self._targets = tuple(targets)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_inner"), name)

    def _conditions(self, u) -> list:
        values = getattr(u, "nodal", u) if isinstance(u, _Field) else u
        for array, conditions in self._targets:
            if values is array or u is array:
                return list(conditions)
        return []

    def _call(self, name: str, u, args, kwargs):
        from diff_soup.domains.mesh.fem._boundary import boundary_load
        from diff_soup.domains.operators.flux_bc import _MeshNormalDerivative

        method = getattr(self._inner, name)
        extra = self._conditions(u)
        on_mesh = [c for c in extra if isinstance(c, _MeshNormalDerivative)]
        on_grid = [c for c in extra if not isinstance(c, _MeshNormalDerivative)]
        if on_grid:
            if not _accepts_bcs(method):
                raise NotImplementedError(
                    f"{type(self._inner).__name__}.{name} takes no bcs argument, so the "
                    "spec.bc.neumann condition on its field cannot be passed to it. An "
                    "operator bank used under spec.bc.neumann must accept bcs= on "
                    "laplacian and div_flux."
                )
            bcs = kwargs.get("bcs")
            kwargs = {**kwargs, "bcs": list(bcs) + on_grid if bcs else on_grid}
        result = method(*args, **kwargs)
        D = args[0] if name == "div_flux" else 1.0
        D = getattr(D, "nodal", D) if isinstance(D, _Field) else D
        if on_mesh and jnp.ndim(D) == 2:
            # A quadrature-point diffusivity on a mesh with as many cells as nodes
            # reads as nodal in the boundary load; div_flux reads it as quad.
            _, layout = _d._read_fem_operand(self.domain, D, "div_flux", ambiguous="quad")
            _, load_layout = _d._read_fem_operand(self.domain, D, "div_flux", ambiguous="nodal")
            if layout != load_layout:
                raise NotImplementedError(
                    "spec.bc.neumann on a mesh with as many cells as nodes cannot yet read a "
                    "diffusivity given at the quadrature points; pass it per node or per cell, "
                    "or state the flux density with spec.bc.flux."
                )
        for condition in on_mesh:
            mask = np.zeros(int(np.asarray(self.domain.boundary_facets()).shape[0]), dtype=bool)
            mask[list(condition.facets)] = True
            load = boundary_load(self.domain, mask, condition.value, D, op_name="spec.bc.neumann")
            result = result + self._inner.as_weak(load)
        return result

    def laplacian(self, u, **kwargs):
        """Laplacian with the field's normal-derivative conditions imposed."""
        return self._call("laplacian", u, (u,), kwargs)

    def div_flux(self, D, u, **kwargs):
        """Divergence-of-flux with the field's normal-derivative conditions imposed."""
        return self._call("div_flux", u, (D, u), kwargs)
