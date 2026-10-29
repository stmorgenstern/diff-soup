# 754 LOC -- Operators dispatch fans out across 9 element types x volume + face quadrature; splitting harms the at-a-glance lookup table; splitting deferred.
r"""Operator dispatch routing to FVM/FDM/FEM backends.

Each dispatch function checks ``domain.discretization_type`` and calls the
corresponding pure function from ``fvm_ops``, ``fdm_ops``, or performs FEM
gather-vmap-scatter assembly.

Convention
---------
**FEM** operators return the **weak form** of the PDE operator, including
the integration-by-parts sign flip.  For example,
``dispatch_laplacian`` returns ``-K @ u`` where ``K`` is the assembled
stiffness matrix.  This means ``ops.laplacian(c)`` represents
:math:`\\int \\nabla^2 c\\, N_i\\, d\\Omega = -\\int \\nabla c \\cdot \\nabla N_i\\, d\\Omega`.
The ``mass_diagonal`` for FEM problems should be the lumped mass so that
the DAE ``M\\, du/dt = f_{\\text{weak}}`` is correct.

**FVM / FDM** operators return the **strong form** (pointwise) of the PDE
operator (e.g. :math:`\\nabla^2 c`), with ``mass_diagonal = 1``.

The FEM stress-divergence assembly (``dispatch_stress_divergence``, reached
through ``div`` of a quad-point tensor and ``integrate_shape_gradient`` of a
Voigt stress) returns the raw assembled weak-form integral without a sign flip.

FEM connectivity convention
---------------------------
Two arrays on [FiniteElementSpace][diff_soup.domains.mesh.fem.topology.FiniteElementSpace]:

* ``fe.cells`` — **physical** node connectivity for geometry / Jacobian
  (``fe.shape_grads``, ``fe.jxw``, ``fe.points[fe.cells]``).
* ``fe.dof_cells`` — **DOF** connectivity for solution gather
  (``u[fe.dof_cells]``) and residual scatter (``fe.num_dofs``).

For non-periodic meshes the two are identical and ``num_dofs == num_total_nodes``.
For periodic meshes ``dof_cells`` remaps tenant nodes onto landlord DOF indices
so the assembly scatter merges contributions across the periodic boundary.
"""

from __future__ import annotations

import contextlib
import contextvars

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup.domains.base import Domain
from diff_soup.domains.grid import Grid1D, GridND
from diff_soup.domains.grid.advection.bcs import PeriodicBC
from diff_soup.domains.grid.advection.collocated import (
    LIMITERS,
    SCHEMES,
    advect_upwind_1d,
    advect_upwind_nd,
    div_advect_1d,
    div_advect_nd,
    normalize_velocity,
)
from diff_soup.domains.grid.fdm._kernels import (
    fdm_apply_normal_derivative,
    fdm_apply_periodic_bc_1d,
    fdm_apply_periodic_bc_nd,
    fdm_div_flux_1d,
    fdm_div_flux_2d,
    fdm_div_flux_3d,
    fdm_div_flux_axes,
    fdm_gradient_1d,
    fdm_laplacian_1d,
    fdm_laplacian_2d,
    fdm_laplacian_3d,
)
from diff_soup.domains.grid.fvm._kernels import (
    fvm_apply_normal_derivative,
    fvm_apply_periodic_bc_1d,
    fvm_apply_periodic_bc_nd,
    fvm_div_flux_1d,
    fvm_div_flux_2d,
    fvm_div_flux_3d,
    fvm_div_flux_axes,
    fvm_face_flux_1d,
    fvm_gradient_1d,
    fvm_laplacian_1d,
    fvm_laplacian_2d,
    fvm_laplacian_3d,
)
from diff_soup.domains.mesh.fem.assembly import assemble_lumped_mass, scatter_elem_residual
from diff_soup.domains.mesh.fem.topology import scalar_fe_space
from diff_soup.domains.operators._fem_scalar import (
    diffusion_elem,
    mass_elem,
    reaction_elem,
    stiffness_action_elem,
)
from diff_soup.domains.operators._fem_scalar import grad as fem_grad
from diff_soup.domains.operators._fem_tensor import (
    fem_div_tensor,
    grad_vec,
    integrate_stress,
    tensor_to_voigt_2d,
    tensor_to_voigt_3d,
)
from diff_soup.domains.operators._field import Field
from diff_soup.domains.operators._quadfield import QuadField
from diff_soup.domains.operators.flux_bc import FluxBC, _NormalDerivativeBC
from diff_soup.domains.staggered_grid import StaggeredGrid

# =========================================================================
# Implicit weak-form mass — provenance tagging (FEM path only)
# =========================================================================
#
# On FEM domains, operator outputs (laplacian, div_flux, divergence) are
# already *integrated* weak-form vectors
# (``int (...) N_i dOmega``), whereas a raw ``q.<field>`` read and plain
# pointwise arithmetic on them (e.g. ``mu - f'(c)``) are *nodal*.  Summing a
# nodal term with an integrated term in a single weak residual is only
# consistent once the nodal part has itself been integrated against the
# test functions (the consistent-mass action).  ``_WeakTerm`` carries this
# ``integrated`` provenance through arithmetic so the FEM residual
# assembler can integrate any still-nodal contribution exactly once,
# letting the user write the bare math (``mu - f'(c) + kappa*lap(c)``)
# instead of the explicit ``ops.mass(mu - f'(c)) + kappa*lap(c)``.
#
# On grids the tag is a pass-through: ``dispatch_*`` already returns plain
# strong-form arrays and ``dispatch_mass`` is ``return u``, so nothing here
# wraps grid outputs and ``finalize_residual`` is a literal identity.
#
# The wrapper lives *entirely inside one residual trace*: FEM operator
# methods on [Operators][diff_soup.domains.operators.base.Operators] produce
# it, RHS closures combine it arithmetically, and
# ``finalize_residual`` unwraps it to a plain array before the residual
# vector is assembled.  It therefore never crosses an autodiff / vmap
# boundary (the Jacobian closures differentiate the already-finalized
# plain-array residual), which is what keeps this change non-invasive.


class _WeakTerm:
    """A residual term tagged with its FEM weak-form provenance.

    ``value`` is the underlying nodal-length JAX array; ``integrated`` is
    ``True`` when ``value`` is already an integrated weak-form vector
    (an operator output) and ``False`` when it is a pointwise/nodal
    quantity that still needs the consistent-mass action before it can be
    summed into a weak residual.

    Arithmetic propagates the tag: ``nodal (+|-) nodal -> nodal``,
    ``integrated (+|-) integrated -> integrated``; mixing the two
    integrates the nodal side first (FEM consistent mass) and yields an
    integrated term.  Scalar / array scaling and negation preserve the
    tag.  An integrated term scales row by row, so a
    [Field][diff_soup.domains.operators._field.Field] factor enters through
    its nodal values and a factor at the quadrature points (or one per
    cell) is refused; a term not yet integrated multiplies pointwise.  The
    class is deliberately minimal: it is only ever produced and consumed
    within a single FEM residual trace.
    """

    __slots__ = ("value", "integrated", "_domain")

    def __init__(self, value: Array, integrated: bool, domain: Domain) -> None:
        self.value = jnp.asarray(value)
        self.integrated = integrated
        self._domain = domain

    # -- helpers -------------------------------------------------------
    def _as_integrated(self) -> Array:
        """Return ``value`` lifted to an integrated weak-form vector."""
        if self.integrated:
            return self.value
        return _integrate_nodal_or_quad(self._domain, self.value)

    @staticmethod
    def _unwrap(other):
        if isinstance(other, _WeakTerm):
            return other.value, other.integrated
        if isinstance(other, Field):
            # A quad-native field read sums in at its quadrature values.
            return other._q, None
        return other, None  # plain array/scalar: provenance unknown -> nodal

    def _scale_by(self, other, op_name: str):
        """Return the per-row factor for ``self * other`` or ``self / other``.

        An integrated term is a vector of test-function integrals, one per
        node, so it can only be scaled row by row: by a scalar, a nodal array,
        or the nodal values of a [Field][diff_soup.domains.operators._field.Field]
        (the same as scaling by ``q.ops.unwrap_field(field)``). A value at the
        quadrature points, or one per cell, would have to be multiplied in
        before the integral was taken, so it is refused.

        A term that is not yet integrated multiplies pointwise: a ``Field``
        enters at its quadrature values and a nodal value is moved to the
        quadrature points first when the other side lives there.
        """
        if isinstance(other, _WeakTerm):
            raise TypeError(
                f"cannot {op_name} two weak-form terms: each is already integrated "
                "against the test functions. Multiply the integrands instead, for "
                "example inside div_flux(D, u), or integrate the product with "
                "q.ops.integrate_shape(...)."
            )
        if not self.integrated:
            if isinstance(other, Field):
                return other._q
            return other
        if isinstance(other, Field):
            return other.nodal
        arr = jnp.asarray(other)
        if self._domain.discretization_type != "fem" or arr.ndim == 0:
            return arr
        # Any other factor (a component vector, say) broadcasts as before.
        _, layout = _read_fem_operand(
            self._domain, arr, op_name, strict=False, ambiguous="nodal"
        )
        if layout == "quad":
            raise TypeError(
                f"cannot {op_name} an integrated weak-form term (such as "
                "q.ops.laplacian(u)) by a value at the quadrature points or one per "
                "cell: the factor would have to enter before the integral. Put it "
                "inside the operator (div_flux(D, u) takes D at the quadrature "
                "points), or scale by a nodal array or a field read such as q.T."
            )
        return arr

    def _combine(self, other, op):
        o_val, o_integrated = self._unwrap(other)
        if o_integrated is None:
            o_integrated = False  # a bare array/scalar is a nodal quantity
        if self.integrated == o_integrated:
            return _WeakTerm(op(self.value, o_val), self.integrated, self._domain)
        # Mixed: integrate the still-nodal-or-quad operand, then combine as
        # integrated.  A bare array operand may be a strong-form *nodal* source
        # (leading axis ``num_dofs`` -> nodal consistent-mass action) OR a raw
        # quad-point value (leading axes ``(n_cells, n_q[, ...])`` -> quad action
        # ``int f N_i``).  ``q.ops.laplacian(c) + f_prime(c)_at_quad`` is the
        # the mixed case: a ``_WeakTerm`` plus a fully-decayed quad array.
        left = self._as_integrated()
        right = o_val if o_integrated else _integrate_nodal_or_quad(self._domain, o_val)
        return _WeakTerm(op(left, right), True, self._domain)

    # -- arithmetic ----------------------------------------------------
    def __add__(self, other):
        return self._combine(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._combine(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._combine(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._combine(other, lambda a, b: b - a)

    def _pointwise(self, factor):
        """Return ``value`` on the layout of ``factor``: quadrature if it lives there."""
        value = self.value
        if (
            not self.integrated
            and self._domain.discretization_type == "fem"
            and jnp.ndim(factor) >= 2
            and jnp.ndim(value) == 1
        ):
            value = dispatch_interp_to_quad(self._domain, value)
        return value

    def __mul__(self, other):
        # Scaling by a scalar / nodal coefficient preserves provenance.
        factor = self._scale_by(other, "multiply")
        return _WeakTerm(self._pointwise(factor) * factor, self.integrated, self._domain)

    def __rmul__(self, other):
        factor = self._scale_by(other, "multiply")
        return _WeakTerm(factor * self._pointwise(factor), self.integrated, self._domain)

    def __truediv__(self, other):
        factor = self._scale_by(other, "divide")
        return _WeakTerm(self._pointwise(factor) / factor, self.integrated, self._domain)

    def __neg__(self):
        return _WeakTerm(-self.value, self.integrated, self._domain)


def _integrate_nodal_or_quad(
    domain: Domain, value: ArrayLike, op_name: str = "integrate", ambiguous: str = "raise"
) -> Array:
    """Lift a still-nodal-or-quad bare operand to an integrated weak-form vector.

    Used in the ``_WeakTerm`` mixed-provenance branch (and ``_as_integrated``)
    to integrate a bare array operand against the test functions.  The action is
    chosen by shape so a fully-decayed quad-point value (e.g. ``f_prime(c)``
    evaluated at quadrature, leading axes ``(n_cells, n_q[, ...])``) is
    integrated with the quad action ``int f N_i``
    ([dispatch_integrate_scalar_at_quad][]), while a strong-form nodal source
    (leading axis ``num_dofs``) gets the nodal consistent-mass action
    ([_fem_mass_action][]).  This mirrors the same shape discrimination
    [finalize_residual][] applies to a standalone bare array.  The shape is
    read by [_read_fem_operand][]: a constant integrates as a nodal constant
    and a per-cell value as a value at the quadrature points.
    """
    arr, layout = _read_fem_operand(domain, value, op_name, ambiguous=ambiguous)
    if layout == "quad":
        return _fem_quad_action(domain, arr)
    if layout == "scalar":
        arr = jnp.broadcast_to(arr, (_get_fe_space(domain).num_dofs,))
    return _fem_mass_action(domain, arr, op_name)


# The GENERAL per-equation residual finalizer (not FEM-specific despite the
# history): the IDENTITY on FVM / grid domains, performing FEM consistent-mass
# integration only when the domain is actually FEM.
def finalize_residual(domain: Domain, term) -> Array:
    """Unwrap a residual term, integrating only a leftover-nodal weak term.

    Called once per equation at residual assembly.  Behaviour:

    * A [_WeakTerm][] carrying ``integrated=False`` (a closure whose
      result is still nodal after combining a weak operator with pointwise
      math) is integrated once via the FEM consistent-mass action.
    * A [_WeakTerm][] carrying ``integrated=True`` passes through as a
      plain array.
    * A **plain array on FEM is a strong-form nodal source and is integrated
      once** via the consistent-mass action — so bare strong-form math (a
      standalone reaction source ``j0(c)*sinh(eta)``, an algebraic identity,
      a pointwise residual) "just works" and matches the ``M du/dt`` LHS.
      Closures that hand-assemble the weak form themselves (e.g. via
      ``scatter_elem_residual``) already return an integrated quantity and
      must tag it ``integrated=True`` — through a ``q.ops`` weak operator
      (``integrate_shape``, ``integrate_shape_gradient``, ...) or the explicit
      [Operators.as_weak][] escape hatch — so it is NOT re-integrated.

    A [QuadField][] (or a bare quad-shaped ``(n_cells, n_q[, ...])``
    array on FEM) is a value living at quadrature points and is integrated
    via the *quad* consistent-mass action ``int f N_i``
    (``dispatch_integrate_scalar_at_quad``) — distinct from the nodal
    ``_fem_mass_action`` used for a strong-form *nodal* source.

    On grid domains this is always the identity (grids carry
    ``mass_diagonal == 1``; both ``_WeakTerm`` values and plain arrays are
    strong form and pass through unchanged).
    """
    if isinstance(term, QuadField):
        # A quad-point value: integrate via ``int f N_i`` on FEM; on a grid
        # there are no quad points, so the raw value passes through.
        if domain.discretization_type != "fem":
            return term.value
        return term.integrate()
    if isinstance(term, _WeakTerm):
        value = term.value
        if domain.discretization_type != "fem":
            return value
        return value if term.integrated else _integrate_nodal_or_quad(domain, value, "residual")
    # Plain array on FEM: distinguish a quad-shaped value (leading axes
    # ``(n_cells, n_q[, ...])`` — integrated via the quad action ``int f N_i``)
    # from a strong-form *nodal* source (leading axis ``num_dofs``, scalar
    # ``(n,)`` or vector/tensor ``(n, *comp)`` — integrated via the nodal
    # consistent-mass action so it matches the ``M du/dt`` LHS).  ``ndim`` alone
    # is ambiguous (a nodal vector field is also 2-D), so dispatch on the leading
    # axis sizes against the FE space.  Identity on grids (lumped mass == 1).
    if domain.discretization_type == "fem":
        return _integrate_nodal_or_quad(domain, term, "residual")
    return term


def _read_fem_operand(
    domain: Domain,
    value: ArrayLike,
    op_name: str,
    *,
    strict: bool = True,
    ambiguous: str = "raise",
) -> tuple[Array, str]:
    """Read an FEM operand by its shape, the one rule every FEM operator uses.

    A 0-d value is a constant (layout ``"scalar"``). A one-dimensional array of
    length ``num_dofs`` is nodal (``"nodal"``). One of length ``n_cells`` holds
    one value per cell and is broadcast to that cell's quadrature points
    (``"quad"``). An array of two or more dimensions is ``"quad"`` when its
    leading axes are ``(n_cells, n_q)`` and ``"nodal"`` when its leading axis is
    ``num_dofs``.

    A one-dimensional array that fits both lengths (a fully periodic ``QUAD4``
    or ``HEX8`` mesh has as many cells as DOFs) is read as nodal, since a field
    read such as ``q.T`` reaches the operators as its nodal array; give a
    per-cell value there with the shape ``(n_cells, n_q)``.

    Args:
        domain: FEM mesh domain.
        value: The operand.
        op_name: The operator, for the message.
        strict: Raise on any other shape. If `False`, return layout ``"other"``.
        ambiguous: What an array of two or more dimensions that fits both
            readings is: ``"raise"``, ``"quad"`` or ``"nodal"``.

    Returns:
        The operand as an array, with a per-cell value broadcast to
        ``(n_cells, n_q)``, and its layout.

    Raises:
        ValueError: If the shape fits no reading and `strict` is set, or fits
            both and `ambiguous` is ``"raise"``.
    """
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return arr, "scalar"
    fe = _get_fe_space(domain)
    n_dofs = fe.num_dofs
    n_cells = fe.shape_grads.shape[0]
    n_q = fe.shape_vals.shape[0]
    nodal = arr.shape[0] in (n_dofs, fe.num_total_nodes)
    if arr.ndim == 1:
        if nodal:
            return arr, "nodal"
        if arr.shape[0] == n_cells:
            return jnp.broadcast_to(arr[:, None], (n_cells, n_q)), "quad"
        if not strict:
            return arr, "other"
        raise ValueError(
            f"{op_name}: a one-dimensional operand on this mesh holds one value per "
            f"node (length num_dofs={n_dofs}) or one per cell (length "
            f"n_cells={n_cells}); got length {arr.shape[0]}."
        )
    quad = arr.shape[0] == n_cells and arr.shape[1] == n_q
    if quad and nodal:
        if ambiguous == "raise":
            raise ValueError(
                f"{op_name}: shape {tuple(arr.shape)} reads both as a quadrature value "
                f"(n_cells={n_cells}, n_q={n_q}) and as a nodal field "
                f"(num_dofs={n_dofs}) on this mesh, so the integration rule is "
                f"ambiguous. Integrate a quadrature value with "
                f"q.ops.integrate_shape(...) or tag an already-integrated "
                f"weak form with q.ops.as_weak(...)."
            )
        return arr, ambiguous
    if quad:
        return arr, "quad"
    if nodal:
        return arr, "nodal"
    if not strict:
        return arr, "other"
    hint = (
        " If this is a hand-assembled weak form, tag it with q.ops.as_weak(...)."
        if op_name == "residual"
        else ""
    )
    raise ValueError(
        f"{op_name}: shape {tuple(arr.shape)} is neither nodal (leading axis "
        f"num_dofs={n_dofs}) nor a value at the quadrature points (leading axes "
        f"(n_cells, n_q) = ({n_cells}, {n_q})).{hint}"
    )


def _require_nodal(domain: Domain, u: ArrayLike, op_name: str) -> Array:
    """Return `u` if it holds one value per node, else raise.

    A gather ``u[fe.dof_cells]`` clamps out-of-range indices, so a field of the
    wrong length would otherwise give a wrong answer without an error.

    Raises:
        ValueError: If `u` is not nodal.
    """
    arr, layout = _read_fem_operand(domain, u, op_name, strict=False, ambiguous="nodal")
    if layout != "nodal":
        raise ValueError(
            f"{op_name}: the field must hold one value per node (leading axis "
            f"num_dofs={_get_fe_space(domain).num_dofs}); got shape {tuple(jnp.shape(u))}."
        )
    return arr


def dispatch_integrate(domain: Domain, expr: Array) -> Array:
    """Explicit weak-form integration escape hatch.

    FEM: a nodal operand takes the consistent-mass action
    ``int expr N_i dOmega`` (same as the auto-integration applied to a nodal
    residual term); a decayed quad-point operand (leading axes
    ``(n_cells, n_q[, ...])``, e.g. pointwise math on a quad-native ``Field``)
    takes the quad action ``int f N_i`` — the same shape dispatch as
    [finalize_residual][].  Grids: identity.
    """
    if isinstance(expr, _WeakTerm):
        expr = expr.value
    if domain.discretization_type == "fem":
        return _integrate_nodal_or_quad(domain, expr)
    return expr


# =========================================================================
# Common operators
# =========================================================================


def _reject_staggered(domain: Domain, op_name: str) -> None:
    """Refuse a collocated-grid operator on a staggered (MAC) grid.

    A [StaggeredGrid][diff_soup.domains.staggered_grid.StaggeredGrid] reports
    ``discretization_type == "fvm"`` because it is a finite-volume layout, but
    its face-centered variables need MAC stencils that this bank does not
    carry.  Without this guard the finite-volume router reaches a cell-grid
    kernel that has no staggered branch, and the resulting message reads as an
    unfinished kernel rather than a domain that was never the right argument.

    Raises:
        NotImplementedError: If ``domain`` is a
            [StaggeredGrid][diff_soup.domains.staggered_grid.StaggeredGrid].
    """
    if isinstance(domain, StaggeredGrid):
        raise NotImplementedError(
            f"{op_name} is not defined on a StaggeredGrid: this bank has no MAC "
            "stencils for face-centered values. Call it on the collocated "
            "cell-center grid (`domain.cell_grid`), or supply an operator bank "
            "that implements the staggered stencils via "
            "spec.domain(..., operators=...)."
        )


def dispatch_laplacian(domain: Domain, u: Array) -> Array:
    """Route a Laplacian evaluation to the FVM, FDM, or FEM backend.

    FVM/FDM return the strong-form pointwise Laplacian; FEM returns the
    weak-form assembled residual (with the integration-by-parts sign flip).

    Args:
        domain: Spatial domain selecting the backend via ``discretization_type``.
        u: Nodal / cell-centered field of length ``domain.num_nodes``.

    Returns:
        Backend-appropriate Laplacian array with the same layout as ``u``.

    Raises:
        NotImplementedError: If ``domain`` is a staggered (MAC) grid.
        ValueError: If ``domain.discretization_type`` is not recognized.
    """
    _reject_staggered(domain, "laplacian")
    dt = domain.discretization_type
    if dt == "fvm":
        return _fvm_laplacian(domain, u)
    elif dt == "fdm":
        return _fdm_laplacian(domain, u)
    elif dt == "fem":
        return _fem_laplacian(domain, u)
    raise ValueError(f"Unknown discretization type: {dt!r}")


def dispatch_div_flux(domain: Domain, D: "ArrayLike | tuple | list", u: Array) -> Array:
    """Route ``div(D grad u)`` to the FVM, FDM, or FEM backend.

    Args:
        domain: Spatial domain selecting the backend.
        D: Diffusivity, a scalar or a nodal field; on a grid also a per-axis
            tuple ``(Dx, Dy[, Dz])``, each entry a scalar or a nodal field.
        u: Nodal / cell-centered solution field.

    Returns:
        Divergence of the diffusive flux; strong form for FVM/FDM and weak
        form (assembled residual) for FEM.

    Raises:
        NotImplementedError: If ``domain`` is a staggered (MAC) grid.
        ValueError: If ``domain.discretization_type`` is not recognized.
    """
    _reject_staggered(domain, "div_flux")
    dt = domain.discretization_type
    if isinstance(D, (tuple, list)):
        return _per_axis_div_flux(domain, D, u, dt)
    # Normalize a scalar (constant) diffusivity to a 0-d array so the strictly
    # typed backend kernels (``Float[Array, ...]``) accept it; ``jnp.ndim``-based
    # scalar detection downstream treats a 0-d array exactly like a Python float.
    D = jnp.asarray(D)
    if dt in ("fvm", "fdm"):
        _check_grid_coefficient(domain, D)
    if dt == "fvm":
        return _fvm_div_flux(domain, D, u)
    elif dt == "fdm":
        return _fdm_div_flux(domain, D, u)
    elif dt == "fem":
        return _fem_div_flux(domain, D, u)
    raise ValueError(f"Unknown discretization type: {dt!r}")


def _tensor_refusal(shape) -> ValueError:
    return ValueError(
        f"div_flux: a diffusivity of shape {tuple(shape)} is not a scalar, one value "
        "per cell or a per-axis tuple. A grid takes a diagonal dispersion tensor as a "
        "tuple (Dx, Dy[, Dz]), each entry a scalar or one value per cell; a full "
        "tensor with off-diagonal terms is not supported."
    )


def _check_grid_coefficient(domain: Domain, D: Array) -> None:
    """Refuse a grid diffusivity that is neither a scalar nor one value per cell."""
    if D.ndim == 0:
        return
    shape = tuple(getattr(domain, "shape", (domain.num_nodes,)))
    n = int(domain.num_nodes)
    if D.shape in (shape, (n,)):
        return
    if isinstance(domain, Grid1D) and D.shape == (n + 1,):
        return  # face values
    raise _tensor_refusal(D.shape)


def _per_axis_coefficients(domain: Domain, D) -> tuple:
    """Check a per-axis diffusivity tuple and shape each entry for the grid kernels."""
    dim = 1 if isinstance(domain, Grid1D) else domain.dim
    if len(D) != dim:
        raise ValueError(
            f"div_flux: a per-axis diffusivity gives one entry per grid axis, {dim} on "
            f"this {dim}-dimensional grid; got {len(D)}."
        )
    out = []
    for entry in D:
        entry = jnp.asarray(entry)
        _check_grid_coefficient(domain, entry)
        out.append(entry if isinstance(domain, Grid1D) else _maybe_reshape_coefficient(domain, entry))
    return tuple(out)


def _per_axis_div_flux(domain: Domain, D, u: Array, dt: str) -> Array:
    """``sum_d d/dx_d (D_d du/dx_d)`` on a grid, one diffusivity per axis."""
    if dt not in ("fvm", "fdm"):
        raise ValueError(
            "div_flux: a per-axis diffusivity tuple is a structured-grid form; on a mesh "
            "pass a scalar, one value per node or one value per quadrature point."
        )
    D_axes = _per_axis_coefficients(domain, D)
    if isinstance(domain, Grid1D):
        return fvm_div_flux_1d(domain, u, D_axes[0]) if dt == "fvm" else fdm_div_flux_1d(
            domain, u, D_axes[0]
        )
    u_grid = _reshape_to_grid(domain, u)
    kernel = fvm_div_flux_axes if dt == "fvm" else fdm_div_flux_axes
    out = kernel(domain, u_grid, D_axes)
    return out.reshape(-1) if u.ndim == 1 else out


def _axis_coefficient(domain: Domain, D, axis: int):
    """The diffusivity along ``axis``: the tuple entry, or the shared value, grid-shaped."""
    if isinstance(D, (tuple, list)):
        return _per_axis_coefficients(domain, D)[axis]
    D = jnp.asarray(D)
    return D if isinstance(domain, Grid1D) else _maybe_reshape_coefficient(domain, D)


def dispatch_gradient(domain: Domain, u: Array) -> Array:
    """Route a gradient evaluation to the FVM, FDM, or FEM backend.

    FVM/FDM return the pointwise gradient; FEM returns the weak-form
    projected gradient assembled at nodes.

    Rank dispatch:
        * Scalar input ``(n,)`` -> vector ``(n, dim)`` — both FEM (weak,
          then lumped-mass projected at nodes) and grid (pointwise) backends.
        * Vector input ``(n, n_components)`` -> Jacobian
          ``(n, n_components, dim)`` with ``J[node, i, j] = du_i/dx_j`` —
          component axis first, spatial axis second, the same index order as
          the quad-point tensor from [dispatch_grad_quad][]. The two differ
          only in normalization: this one is the weak gradient divided by the
          lumped mass (a pointwise value at nodes), that one the exact
          derivative at each quadrature point. **FEM-only** (no grid branch).

    Args:
        domain: Spatial domain selecting the backend.
        u: Nodal / cell-centered scalar field, or vector field for FEM.

    Returns:
        Gradient with one extra trailing component axis (``..., dim``).

    Raises:
        ValueError: If ``domain.discretization_type`` is not recognized.
        NotImplementedError: ``grad`` of a vector input on a non-FEM domain,
            or any gradient on a staggered (MAC) grid.
    """
    _reject_staggered(domain, "gradient")
    dt = domain.discretization_type
    if u.ndim == 1:
        if dt == "fvm":
            return _fvm_gradient(domain, u)
        elif dt == "fdm":
            return _fdm_gradient(domain, u)
        elif dt == "fem":
            return _fem_gradient(domain, u)
        raise ValueError(f"Unknown discretization type: {dt!r}")
    if u.ndim == 2:
        # Vector input: per-component grad assembled into a Jacobian.
        # Returns the *projected pointwise* gradient (lumped-mass projection
        # of the weak gradient) so the result composes pointwise with eigen-
        # strains and stress laws.  Scalar grad stays weak-form (used for
        # diffusion flux); vector grad is pointwise.
        if dt != "fem":
            raise NotImplementedError(
                f"grad(vector) is an FEM-only operator; "
                f"{type(domain).__name__} is a grid (strong-form) domain. "
                "Grids support only scalar grad ``(n,) -> (n, dim)``."
            )

        fe = _get_fe_space(domain)
        m_lumped = jnp.asarray(assemble_lumped_mass(fe))
        m_lumped = jnp.where(m_lumped > 0, m_lumped, 1.0)  # safe divide
        rows = []
        for i in range(u.shape[-1]):
            weak = _fem_gradient(domain, u[..., i])  # (n_nodes, dim) weak grad
            rows.append(weak / m_lumped[:, None])  # pointwise grad at nodes
        # Stack so the result has shape (n_nodes, n_components, dim), i.e.
        # J[node, i, j] = d u_i / d x_j — one Jacobian ROW per component.
        return jnp.stack(rows, axis=-2)
    raise ValueError(f"gradient: expected (n,) or (n,d), got shape {u.shape}")


# =========================================================================
# Grid-specific dispatch
# =========================================================================


def dispatch_face_flux(domain: Domain, u: Array, D: ArrayLike) -> Array:
    """Compute face-centered diffusive flux ``-D du/dx`` (Grid1D only).

    Args:
        domain: Must be a [Grid1D][].
        u: Cell-centered solution values.
        D: Diffusivity (scalar or nodal).

    Returns:
        Face-centered flux array with length ``num_nodes + 1``.

    Raises:
        NotImplementedError: If ``domain`` is not a [Grid1D][].
    """
    if not isinstance(domain, Grid1D):
        raise NotImplementedError(
            f"{type(domain).__name__} does not support face_flux (requires Grid1D)"
        )
    return fvm_face_flux_1d(domain, u, jnp.asarray(D))


def dispatch_advect_upwind(domain: Domain, u: Array, velocity, bcs: "Sequence | None" = None):
    """Route a first-order upwind advection ``v . grad u`` to the grid kernels.

    Collocated (cell/node-centered) upwind advection is discretization-agnostic
    on a cartesian grid — the non-conservative form reads only nodal values and
    the collocation-point spacing — so ``"fvm"`` and ``"fdm"`` grids share the
    same kernel ([diff_soup.domains.grid.advection.collocated][]).

    Args:
        domain: A cartesian [Grid1D][] or [GridND][].
        u: Field values (flat ``(num_nodes,)`` or grid-shaped).
        velocity: A constant speed, or a per-axis sequence whose entries are
            constants or cell-centred arrays (a velocity field).
        bcs: Optional boundary tags.  Each
            [PeriodicBC][diff_soup.domains.grid.advection.bcs.PeriodicBC] marks its
            axis periodic; every other axis takes the zero-gradient (outflow)
            default.  Any non-periodic tag is rejected.

    Returns:
        The upwind ``v . grad u`` at the collocation points, matching ``u``'s
        layout.

    Raises:
        NotImplementedError: on a non-grid (e.g. FEM) or staggered domain.
        ValueError: on a velocity of the wrong shape or an unsupported BC.
    """
    _reject_staggered(domain, "advect_upwind")

    dt = domain.discretization_type
    if dt not in ("fvm", "fdm"):
        raise NotImplementedError(
            "advect_upwind is a structured-grid operator for cartesian FVM/FDM grids; "
            f"{type(domain).__name__} uses {dt!r} discretization. Author advection on a "
            "cartesian Grid1D or GridND."
        )

    periodic_axes: set[int] = set()
    if bcs:
        for bc in bcs:
            if isinstance(bc, PeriodicBC):
                periodic_axes.add(int(bc.axis))
            else:
                raise ValueError(
                    "advect_upwind honors periodic boundaries (declare the field periodic "
                    "with spec.bc.periodic) and applies a zero-gradient (outflow) condition "
                    f"at every non-periodic boundary; it does not accept a {type(bc).__name__} "
                    "on the advected field. Remove that boundary condition or make the field "
                    "periodic."
                )

    if isinstance(domain, Grid1D):
        (v0,) = normalize_velocity(velocity, 1, (domain.n,))
        return advect_upwind_1d(domain, u, v0, periodic=(0 in periodic_axes))
    if isinstance(domain, GridND):
        vpa = normalize_velocity(velocity, domain.dim, tuple(domain.shape))
        u_grid = _reshape_to_grid(domain, u)
        out = advect_upwind_nd(domain, u_grid, vpa, tuple(sorted(periodic_axes)))
        return out.reshape(-1) if u.ndim == 1 else out
    raise NotImplementedError(f"advect_upwind is not implemented for {type(domain).__name__}.")


# The wide grid operators called inside ``recorded_stencil_reach``, as
# ``(cells per axis, call)``; ``None`` outside it.
_STENCIL_REACH: contextvars.ContextVar = contextvars.ContextVar("stencil_reach", default=None)


@contextlib.contextmanager
def recorded_stencil_reach():
    """Collect ``(reach, call)`` for each grid operator called inside that reads past one cell.

    A kernel that is not stencil-probed (a family member kernel) reads this to
    check its declared reach.
    """
    calls: list = []
    token = _STENCIL_REACH.set(calls)
    try:
        yield calls
    finally:
        _STENCIL_REACH.reset(token)


def dispatch_div_advect(
    domain: Domain,
    u: Array,
    velocity,
    bcs: "Sequence | None" = None,
    scheme: str = "tvd",
    limiter: str = "van_leer",
):
    """Route the conservative advection ``div(v u)`` to the grid kernels.

    Each interior face carries the TVD, WENO3, central or upwind flux; a
    boundary face carries only an outgoing flux, so an inflow enters through
    ``spec.bc.inflow``. ``"fvm"`` and ``"fdm"`` grids share the kernel, which
    weights each face by ``face_area / cell_volume``.

    Args:
        domain: A cartesian [Grid1D][] or [GridND][].
        u: Field values (flat ``(num_nodes,)`` or grid-shaped).
        velocity: A constant, or a per-axis sequence of constants or cell arrays.
        bcs: Optional [PeriodicBC][diff_soup.domains.grid.advection.bcs.PeriodicBC]
            tags; any other condition is rejected.
        scheme: ``"tvd"``, ``"weno3"``, ``"central"`` or ``"upwind"``.
        limiter: ``"van_leer"``, ``"koren"`` or ``"minmod"``, used by ``"tvd"``.

    Returns:
        ``div(v u)`` at the cells, matching ``u``'s layout.
    """
    if scheme not in SCHEMES:
        raise ValueError(
            f'div_advect: scheme must be "tvd", "weno3", "central" or "upwind"; got {scheme!r}.'
        )
    if limiter not in LIMITERS:
        raise ValueError(
            f'div_advect: limiter must be "van_leer", "koren" or "minmod"; got {limiter!r}.'
        )
    _reject_staggered(domain, "div_advect")
    dt = domain.discretization_type
    if dt not in ("fvm", "fdm"):
        raise NotImplementedError(
            "div_advect is a structured-grid operator for cartesian FVM/FDM grids; "
            f"{type(domain).__name__} uses {dt!r} discretization."
        )
    periodic_axes: set[int] = set()
    for bc in bcs or ():
        if not isinstance(bc, PeriodicBC):
            raise ValueError(
                "div_advect takes periodic conditions only (spec.bc.periodic). A face the "
                "flow leaves through carries the outgoing flux on its own; feed a face the "
                f"flow enters through with spec.bc.inflow, not a {type(bc).__name__}."
            )
        periodic_axes.add(int(bc.axis))
    calls = _STENCIL_REACH.get()
    if calls is not None and scheme in ("tvd", "weno3"):
        calls.append((2, f'div_advect(scheme="{scheme}")'))
    if isinstance(domain, Grid1D):
        (v0,) = normalize_velocity(velocity, 1, (domain.n,), op_name="div_advect")
        return div_advect_1d(
            domain, u, v0, periodic=(0 in periodic_axes), scheme=scheme, limiter=limiter
        )
    if isinstance(domain, GridND):
        vpa = normalize_velocity(velocity, domain.dim, tuple(domain.shape), op_name="div_advect")
        u_grid = _reshape_to_grid(domain, u)
        out = div_advect_nd(domain, u_grid, vpa, tuple(sorted(periodic_axes)), scheme, limiter)
        return out.reshape(-1) if u.ndim == 1 else out
    raise NotImplementedError(f"div_advect is not implemented for {type(domain).__name__}.")


# =========================================================================
# FVM backend
# =========================================================================


def _reshape_to_grid(domain: "GridND", u: Array) -> Array:
    """Reshape a flat ``(nx*ny[*nz],)`` state view to the grid shape ``(nx, ny[, nz])``.

    Spec-authored equations and the solver carry state as a flat 1D vector
    (via the IR ``_BlockLayout``), but the 2D/3D FVM/FDM backend kernels
    take natural ``(nx, ny[, nz])``-shaped arrays so their vmap /
    dimension-splitting logic reads cleanly.  This helper bridges the two
    conventions at the dispatch boundary, keeping both sides unchanged.

    If ``u`` is already grid-shaped (e.g. a caller reaching straight into
    the low-level kernel via ``Operators``), it is returned unchanged.
    """
    if u.ndim == len(domain.shape) and u.shape == domain.shape:
        return u
    if u.ndim == 1 and u.shape[0] == domain.num_nodes:
        return u.reshape(domain.shape)
    raise ValueError(
        f"state shape {u.shape} incompatible with domain shape {domain.shape} "
        f"(num_nodes={domain.num_nodes})"
    )


def _maybe_reshape_coefficient(domain: "GridND", D: ArrayLike) -> Array:
    """Reshape a flat nodal coefficient to grid shape; leave scalars alone.

    Face-valued coefficients and tuples are passed through unchanged —
    the backend will raise if it doesn't know what to do with them.
    """
    D = jnp.asarray(D)
    if D.ndim == 0:
        return D
    if D.ndim == 1 and D.shape[0] == domain.num_nodes:
        return D.reshape(domain.shape)
    return D


def _fvm_laplacian(domain: Domain, u: Array) -> Array:
    """FVM Laplacian backend: dispatches to 1D/2D/3D kernels by domain type."""
    if isinstance(domain, Grid1D):
        return fvm_laplacian_1d(domain, u)
    elif isinstance(domain, GridND):
        u_grid = _reshape_to_grid(domain, u)
        d = domain.dim
        if d == 2:
            out = fvm_laplacian_2d(domain, u_grid)
        elif d == 3:
            out = fvm_laplacian_3d(domain, u_grid)
        else:
            raise NotImplementedError(f"FVM laplacian not implemented for GridND dim={d}")
        # Flatten back only if the caller gave us a flat input.
        return out.reshape(-1) if u.ndim == 1 else out
    raise NotImplementedError(
        f"FVM laplacian not implemented for {type(domain).__name__} dim={domain.dim}"
    )


def _fvm_div_flux(domain: Domain, D: ArrayLike, u: Array) -> Array:
    """FVM ``div(D grad u)`` backend: 1D/2D/3D kernel dispatch."""
    if isinstance(domain, Grid1D):
        return fvm_div_flux_1d(domain, u, D)
    elif isinstance(domain, GridND):
        u_grid = _reshape_to_grid(domain, u)
        D_g = _maybe_reshape_coefficient(domain, D)
        d = domain.dim
        if d == 2:
            out = fvm_div_flux_2d(domain, u_grid, D_g)
        elif d == 3:
            out = fvm_div_flux_3d(domain, u_grid, D_g)
        else:
            raise NotImplementedError(f"FVM div_flux not implemented for GridND dim={d}")
        return out.reshape(-1) if u.ndim == 1 else out
    raise NotImplementedError(
        f"FVM div_flux not implemented for {type(domain).__name__} dim={domain.dim}"
    )


def _fvm_gradient(domain: Domain, u: Array) -> Array:
    """FVM gradient backend (currently 1D only)."""
    if isinstance(domain, Grid1D):
        return fvm_gradient_1d(domain, u)
    raise NotImplementedError(f"FVM gradient not implemented for {type(domain).__name__}")


# =========================================================================
# FDM backend
# =========================================================================


def _fdm_laplacian(domain: Domain, u: Array) -> Array:
    """FDM Laplacian backend: dispatches to 1D/2D/3D kernels by domain type."""
    if isinstance(domain, Grid1D):
        return fdm_laplacian_1d(domain, u)
    elif isinstance(domain, GridND):
        u_grid = _reshape_to_grid(domain, u)
        d = domain.dim
        if d == 2:
            out = fdm_laplacian_2d(domain, u_grid)
        elif d == 3:
            out = fdm_laplacian_3d(domain, u_grid)
        else:
            raise NotImplementedError(f"FDM laplacian not implemented for GridND dim={d}")
        return out.reshape(-1) if u.ndim == 1 else out
    raise NotImplementedError(
        f"FDM laplacian not implemented for {type(domain).__name__} dim={domain.dim}"
    )


def _fdm_div_flux(domain: Domain, D: ArrayLike, u: Array) -> Array:
    """FDM ``div(D grad u)`` backend: 1D/2D/3D kernel dispatch."""
    if isinstance(domain, Grid1D):
        return fdm_div_flux_1d(domain, u, D)
    elif isinstance(domain, GridND):
        u_grid = _reshape_to_grid(domain, u)
        D_g = _maybe_reshape_coefficient(domain, D)
        d = domain.dim
        if d == 2:
            out = fdm_div_flux_2d(domain, u_grid, D_g)
        elif d == 3:
            out = fdm_div_flux_3d(domain, u_grid, D_g)
        else:
            raise NotImplementedError(f"FDM div_flux not implemented for GridND dim={d}")
        return out.reshape(-1) if u.ndim == 1 else out
    raise NotImplementedError(
        f"FDM div_flux not implemented for {type(domain).__name__} dim={domain.dim}"
    )


def _fdm_gradient(domain: Domain, u: Array) -> Array:
    """FDM gradient backend (currently 1D only)."""
    if isinstance(domain, Grid1D):
        return fdm_gradient_1d(domain, u)
    raise NotImplementedError(f"FDM gradient not implemented for {type(domain).__name__}")


# =========================================================================
# FEM backend (gather → vmap → scatter)
#
# FEM operators use two connectivity arrays from FiniteElementSpace:
#   fe.cells      — physical connectivity for Jacobian / geometry
#   fe.dof_cells  — DOF connectivity for solution gather & residual scatter
# For non-periodic meshes dof_cells == cells.  For periodic meshes
# tenant nodes are remapped to landlord DOFs (see topology.py).
# All FEM operators below follow this pattern.
# =========================================================================


def _get_fe_space(domain):
    """The single-field finite-element space of a Mesh domain, built once per mesh."""
    if hasattr(domain, "_fe_space") and domain._fe_space is not None:
        return domain._fe_space

    return scalar_fe_space(domain)


def _fem_laplacian(domain: Domain, u: Array) -> Array:
    """FEM Laplacian backend: assemble ``-K u`` (weak form, IBP sign flipped)."""
    fe = _get_fe_space(domain)
    u = _require_nodal(domain, u, "laplacian")
    cell_u = u[fe.dof_cells]  # (n_cells, n_nodes_per_cell)
    # stiffness_action_elem returns K_e @ u_e = int grad(u) . grad(N_i) dOmega
    # (positive stiffness action; the residual of -nabla^2 u).  The weak form
    # of nabla^2 u is the negative of the stiffness action (IBP sign flip):
    #   int nabla^2(u) N_i dOmega = -int grad(u) . grad(N_i) dOmega
    # We negate to return the faithful weak nabla^2 u (IBP sign included).
    f_elem = jax.vmap(stiffness_action_elem)(fe.shape_grads, fe.jxw, cell_u)
    raw = scatter_elem_residual(f_elem, fe.dof_cells, fe.num_dofs, n_vars=1)
    return -raw


def _fem_div_flux(domain: Domain, D: ArrayLike, u: Array) -> Array:
    """FEM ``div(D grad u)`` backend: assemble negative diffusion stiffness."""
    fe = _get_fe_space(domain)
    u = _require_nodal(domain, u, "div_flux")
    cell_u = u[fe.dof_cells]  # (n_cells, n_nodes_per_cell)

    n_cells = fe.dof_cells.shape[0]
    n_q = fe.shape_vals.shape[0]
    # D may be scalar, nodal, per-cell, or quad-sampled (strong-form authoring:
    # pointwise math on a quad-native ``Field`` decays to a raw
    # ``(n_cells, n_q)`` array).  A per-cell D broadcasts to the quad points.
    D, layout = _read_fem_operand(domain, D, "div_flux", ambiguous="quad")
    if layout == "scalar":
        f_elem = jax.vmap(lambda grads, jxw, u_e: diffusion_elem(grads, jxw, D, u_e))(
            fe.shape_grads, fe.jxw, cell_u
        )
    elif D.ndim == 1 or D.shape == (n_cells, n_q):
        # Nodal D is interpolated to the quad points; quad-sampled D is used
        # directly (full quadrature accuracy for nonlinear mobilities).
        D_q = dispatch_interp_to_quad(domain, D) if layout == "nodal" else D
        f_elem = jax.vmap(diffusion_elem)(fe.shape_grads, fe.jxw, D_q, cell_u)
    else:
        raise ValueError(
            "div_flux(D, u) on FEM: D must be a scalar, a nodal array "
            f"({fe.num_dofs},), a per-cell array ({n_cells},), or a quad-sampled "
            f"array ({n_cells}, {n_q}); got shape {jnp.shape(D)}."
        )

    # diffusion_elem returns int D grad(u) . grad(N_i) dOmega (positive
    # stiffness action).  The weak form of div(D grad u) is the negative
    # of the diffusion stiffness action (IBP sign):
    #   int div(D grad u) N_i dOmega = -int D grad(u) . grad(N_i) dOmega
    # We negate to return the weak form of div(D grad u).
    raw = scatter_elem_residual(f_elem, fe.dof_cells, fe.num_dofs, n_vars=1)
    return -raw


def _fem_gradient(domain: Domain, u: Array) -> Array:
    """Assembled weak gradient for FEM: int grad(u)_d N_i dOmega per component."""
    fe = _get_fe_space(domain)
    cell_u = u[fe.dof_cells]  # (n_cells, n_nodes_per_cell)

    # Gradient at quad points: (n_cells, n_q, dim)
    grad_q = jax.vmap(fem_grad)(fe.shape_grads, cell_u)

    # Project each component to nodes via scatter (weak form, no mass division)
    dim = fe.dim
    grad_nodal = []
    for d in range(dim):
        # Integrate: sum_q grad_q[d] * N_i * jxw
        f_elem = jnp.einsum("cq,qn,cq->cn", grad_q[:, :, d], fe.shape_vals, fe.jxw)
        g_d = (
            jnp.zeros(fe.num_dofs, dtype=u.dtype)
            .at[fe.dof_cells.reshape(-1)]
            .add(f_elem.reshape(-1))
        )
        grad_nodal.append(g_d)

    return jnp.stack(grad_nodal, axis=-1)  # (num_dofs, dim)


# =========================================================================
# mass / lumped_mass / divergence
# =========================================================================


def _grid_cell_volumes(domain: Domain) -> Array:
    """Return per-node/cell volumes for a Grid1D or GridND (flattened)."""
    if isinstance(domain, (Grid1D, GridND)):
        return domain.cell_volumes
    raise NotImplementedError(f"cell volumes not implemented for {type(domain).__name__}")


def dispatch_mass(domain: Domain, u: Array) -> Array:
    """Apply test-function weighting to a pointwise quantity.

    FEM: a nodal operand takes the consistent mass matrix action
    ``int u N_j N_i dOmega``, converting a pointwise field into a weak-form
    residual consistent with the other FEM operators (laplacian, div_flux)
    that return weak form; a decayed quad-point operand (leading
    axes ``(n_cells, n_q[, ...])``, e.g. pointwise math on a quad-native
    ``Field``) takes the quad action ``int f N_i`` — the same shape dispatch
    as [finalize_residual][].

    FVM/FDM: identity (returns ``u`` unchanged).  Grid operators already
    return strong-form values with ``mass_diagonal = 1``, so no additional
    weighting is needed.

    For raw integration weights, use [dispatch_lumped_mass][] instead.
    """
    dt = domain.discretization_type
    if dt in ("fvm", "fdm"):
        return u
    if dt == "fem":
        return _integrate_nodal_or_quad(domain, u, "mass")
    raise ValueError(f"Unknown discretization type: {dt!r}")


def _fem_mass_action(domain: Domain, u: ArrayLike, op_name: str = "mass") -> Array:
    """Consistent mass matrix action for FEM: int N_i N_j u_j dOmega.

    Handles a scalar nodal field ``(n_nodes,)`` and a vector/tensor nodal
    field ``(n_nodes, *comp)`` — the latter by applying the scalar
    consistent-mass action independently to each trailing component (the
    mass matrix is block-diagonal across components).  This lets a residual
    closure return bare strong-form vector math (e.g. a momentum source) and
    have it integrated on FEM exactly like a scalar source.
    """
    fe = _get_fe_space(domain)
    u = _require_nodal(domain, u, op_name)

    def _scalar_action(col):
        cell_u = col[fe.dof_cells]
        f_elem = jax.vmap(mass_elem, in_axes=(None, 0, 0))(fe.shape_vals, fe.jxw, cell_u)
        return scatter_elem_residual(f_elem, fe.dof_cells, fe.num_dofs, n_vars=1)

    if u.ndim <= 1:
        return _scalar_action(u)
    # Vector / tensor nodal field: vmap the scalar action over the flattened
    # component axis, then restore the original shape.
    orig_shape = u.shape
    flat = u.reshape(orig_shape[0], -1)
    out = jax.vmap(_scalar_action, in_axes=1, out_axes=1)(flat)
    return out.reshape(orig_shape)


def dispatch_lumped_mass(domain: Domain) -> Array:
    """Diagonal of the lumped mass matrix.

    Raises:
        NotImplementedError: If ``domain`` is a staggered (MAC) grid, whose
            face-centered values have no cell-volume weight.
        ValueError: If ``domain.discretization_type`` is not recognized.
    """
    _reject_staggered(domain, "lumped_mass")
    dt = domain.discretization_type
    if dt in ("fvm", "fdm"):
        return _grid_cell_volumes(domain)
    if dt == "fem":
        return _fem_lumped_mass(domain)
    raise ValueError(f"Unknown discretization type: {dt!r}")


def _fem_lumped_mass(domain: Domain) -> Array:
    """FEM lumped mass: row-sum of consistent mass matrix.

    row_i = sum_j int N_i N_j dOmega = int N_i (sum_j N_j) dOmega = int N_i dOmega.
    This is immutable geometry metadata, so assemble it once in deterministic
    host cell/quadrature order. A device scatter-add can vary by a few ulps
    across GPU processes; when the weights are captured by a constraint
    closure those ulps change the problem's structural content hash and defeat
    persistent compilation/precompute caches.
    """
    fe = _get_fe_space(domain)
    return jnp.asarray(assemble_lumped_mass(fe))


def dispatch_divergence(domain: Domain, vec_or_tensor: Array) -> Array:
    """Divergence of a vector or tensor field at nodes/cells.

    Rank dispatch:
        * Vector input ``(n, dim)`` -> scalar ``(n,)`` — both FEM (weak
          divergence) and grid (per-axis sum of 1D gradients) backends.
        * Tensor input ``(n, dim, dim)`` -> vector ``(n, dim)`` giving the
          row-wise divergence of σ.  **FEM-only** (no grid branch).
        * Quad-point tensor ``(n_cells, n_q, dim, dim)`` -> nodal force
          ``(num_dofs, dim)`` — the weak stress-divergence ``∫ σ : ∇N``
          assembled directly from a per-quad-point stress tensor (no Voigt
          round-trip at the call site).  **FEM-only**.

    Raises:
        NotImplementedError: If ``domain`` is a staggered (MAC) grid, or a
            tensor input arrives on a grid domain.
        ValueError: On an unrecognized discretization type or input rank.
    """
    _reject_staggered(domain, "divergence")

    # A sticky ``QuadField`` (e.g. ``double_dot(C4, eps) - qf * SIGMA0`` with
    # ``qf = q.ops.wrap_quad_value(c_q)``)
    # carries the quad-point stress tensor; unwrap to its raw value so the
    # rank dispatch below routes it through the quad-tensor branch.
    if isinstance(vec_or_tensor, QuadField):
        vec_or_tensor = vec_or_tensor.value
    dt = domain.discretization_type
    if vec_or_tensor.ndim == 2:
        if dt == "fem":
            return _fem_divergence(domain, vec_or_tensor)
        if dt in ("fvm", "fdm"):
            return _grid_divergence(domain, vec_or_tensor)
        raise ValueError(f"Unknown discretization type: {dt!r}")
    if vec_or_tensor.ndim == 3:
        if dt != "fem":
            raise NotImplementedError(
                f"div(tensor) is an FEM-only operator; "
                f"{type(domain).__name__} is a grid (strong-form) domain. "
                "Grids support only vector divergence ``(n, dim) -> (n,)``."
            )
        return fem_div_tensor(domain, vec_or_tensor)
    if vec_or_tensor.ndim == 4:
        # Quad-point tensor field (n_cells, n_q, dim, dim): assemble
        # ∫ σ : ∇N  by routing through the existing Voigt stress-divergence.
        if dt != "fem":
            raise NotImplementedError("div(quad tensor) is an FEM-only operator.")

        dim = vec_or_tensor.shape[-1]
        to_voigt = tensor_to_voigt_2d if dim == 2 else tensor_to_voigt_3d
        sigma_v = to_voigt(vec_or_tensor)  # (n_cells, n_q, voigt)
        return dispatch_stress_divergence(domain, sigma_v)
    raise ValueError(
        f"divergence: expected (n,d), (n,d,d) or (n_cells,n_q,d,d), got shape {vec_or_tensor.shape}"
    )


def _fem_divergence(domain: Domain, vec: Array) -> Array:
    """Weak divergence of a nodal vector field: int div(v) N_i dOmega."""
    fe = _get_fe_space(domain)
    # cell_v: (n_cells, n_nodes_per_cell, dim)
    cell_v = vec[fe.dof_cells]
    # div_q[c, q] = sum_n sum_d grads[c, q, n, d] * v[c, n, d]
    div_q = jnp.einsum("cqnd,cnd->cq", fe.shape_grads, cell_v)
    # Integrate N_i * div_q (weak form, no mass division)
    f_elem = jnp.einsum("cq,qn,cq->cn", div_q, fe.shape_vals, fe.jxw)
    g = jnp.zeros(fe.num_dofs, dtype=vec.dtype).at[fe.dof_cells.reshape(-1)].add(f_elem.reshape(-1))
    return g


def _grid_axis_derivative(domain: "GridND", comp: Array, axis: int) -> Array:
    """Partial derivative ``d comp / dx_axis`` on a structured grid.

    ``comp`` is one flat ``(num_nodes,)`` scalar component.  The grid is a
    tensor product, so the derivative along one axis is that axis' own 1D
    gradient kernel applied to every line of cells parallel to it: move the
    axis to the front, flatten the remaining axes into a batch, and vmap the
    1D kernel over that batch.

    Returns:
        The flat ``(num_nodes,)`` derivative, in the same row-major order.
    """
    grad1d = fvm_gradient_1d if domain.discretization_type == "fvm" else fdm_gradient_1d
    ax = domain.axes[axis]
    moved = jnp.moveaxis(comp.reshape(domain.shape), axis, 0)  # (n_axis, ...rest)
    flat = moved.reshape(moved.shape[0], -1)  # (n_axis, n_rest)
    grad_flat = jax.vmap(lambda col: grad1d(ax, col), in_axes=1, out_axes=1)(flat)
    return jnp.moveaxis(grad_flat.reshape(moved.shape), 0, axis).reshape(-1)


def _grid_divergence(domain: Domain, vec: Array) -> Array:
    """Divergence on structured grids via per-axis gradients of each component."""
    if isinstance(domain, Grid1D):
        grad1d = fvm_gradient_1d if domain.discretization_type == "fvm" else fdm_gradient_1d
        # vec expected shape (n, 1) or (n,)
        v0 = vec[..., 0] if vec.ndim > 1 else vec
        return grad1d(domain, v0)
    if isinstance(domain, GridND):
        out = jnp.zeros(domain.num_nodes, dtype=vec.dtype)
        for axis in range(domain.dim):
            out = out + _grid_axis_derivative(domain, vec[:, axis], axis)
        return out
    raise NotImplementedError(f"divergence not implemented for {type(domain).__name__}")


def dispatch_curl(domain: Domain, vec: Array) -> Array:
    r"""Curl :math:`\nabla \times \mathbf{u}` of a vector field.

    The field must carry one component per spatial axis, and the domain must be
    two- or three-dimensional: a curl crosses the gradient with the coordinate
    axes, so a missing component or a missing axis has no derivative to take.

    Rank of the result follows the dimension, as the cross product does:

    * 2D — the scalar :math:`\partial u_y/\partial x - \partial u_x/\partial y`,
      shape ``(n,)``; it is the out-of-plane component the plane cannot hold.
    * 3D — the curl vector, shape ``(n, 3)``.

    Convention per backend, as for the other differential operators: FVM/FDM
    grids return the **pointwise strong form** at cells, and FEM returns the
    **integrated weak form** :math:`\int (\nabla \times \mathbf{u})\,N_i
    \,d\Omega` (assembled, not mass-divided) — per component in 3D.

    Args:
        domain: Spatial domain selecting the backend.
        vec: Vector field at nodes/cells, shape ``(n, 2)`` or ``(n, 3)``.

    Returns:
        The scalar curl ``(n,)`` in 2D or the curl vector ``(n, 3)`` in 3D.

    Raises:
        ValueError: If ``vec`` is not a ``(n, dim)`` field of the domain's own
            dimension, or if ``domain.discretization_type`` is not recognized.
        NotImplementedError: On a one-dimensional domain, which has no curl, or
            on a staggered grid, which stores its components on faces.
    """
    _reject_staggered(domain, "curl")
    dim = domain.dim
    if dim not in (2, 3):
        raise NotImplementedError(
            f"curl is defined on 2-D and 3-D domains; {type(domain).__name__} is {dim}-dimensional."
        )
    if vec.ndim != 2 or vec.shape[-1] != dim:
        raise ValueError(
            f"curl: expected a vector field of shape (n, {dim}): one component "
            f"per spatial axis of this {dim}-D domain; got shape {tuple(vec.shape)}."
        )
    dt = domain.discretization_type
    if dt == "fem":
        return _fem_curl(domain, vec)
    if dt in ("fvm", "fdm"):
        return _grid_curl(domain, vec)
    raise ValueError(f"Unknown discretization type: {dt!r}")


def _fem_curl(domain: Domain, vec: Array) -> Array:
    """Weak curl of a nodal vector field: ``int (curl u) N_i dOmega``."""
    # H[cell, q, i, j] = du_i/dx_j at the quadrature points.
    H = dispatch_grad_quad(domain, vec)
    if vec.shape[-1] == 2:
        curl_q = H[..., 1, 0] - H[..., 0, 1]
    else:
        curl_q = jnp.stack(
            [
                H[..., 2, 1] - H[..., 1, 2],
                H[..., 0, 2] - H[..., 2, 0],
                H[..., 1, 0] - H[..., 0, 1],
            ],
            axis=-1,
        )
    # Component-wise shape-function lift; the consistent-mass action is
    # block-diagonal across components, so the 3-D case assembles in one call.
    return dispatch_integrate_scalar_at_quad(domain, curl_q)


def _grid_curl(domain: Domain, vec: Array) -> Array:
    """Pointwise curl on a structured grid, from per-axis component gradients."""
    if not isinstance(domain, GridND):
        raise NotImplementedError(f"curl not implemented for {type(domain).__name__}")

    def d(comp: int, axis: int) -> Array:
        return _grid_axis_derivative(domain, vec[:, comp], axis)

    if domain.dim == 2:
        return d(1, 0) - d(0, 1)
    return jnp.stack(
        [d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)],
        axis=-1,
    )


# =========================================================================
# FEM-only operators (guarded by _require_fem)
#
# Every dispatcher below exists ONLY on the FEM (weak-form) backend — there
# is no grid (strong-form) counterpart, so each one calls ``_require_fem``
# first and raises a "<op> is an FEM-only operator" NotImplementedError on a
# grid domain:
#   stress_divergence (continuum mechanics),
#   interp_to_quad, integrate_scalar_at_quad (quadrature plumbing).
# The rank-2/3 branches of
# ``dispatch_gradient`` / ``dispatch_divergence`` are FEM-only for the same
# reason.
# =========================================================================


def _require_fem(domain: Domain, op_name: str) -> None:
    """Guard an FEM-only operator: raise if ``domain`` is not an FEM mesh.

    The message names the operator as **FEM-only up front** so the diagnostic
    reads as "this op only exists on FEM" rather than trailing the requirement
    at the end of a generic "not supported" sentence.
    """
    if domain.discretization_type != "fem":
        raise NotImplementedError(
            f"{op_name} is an FEM-only operator; "
            f"{type(domain).__name__} is a grid (strong-form) domain. "
            "Use a Mesh domain for continuum-mechanics / weak-form-assembly operators."
        )


def dispatch_grad_quad(domain: Domain, u_vec: Array) -> Array:
    """Quad-point displacement-gradient TENSOR (FEM-only, full tensor form).

    Unlike ``dispatch_gradient`` (which projects a vector grad to nodes) this
    returns the per-quad-point tensor used by the tensor-native composition
    path: ``H[c, q, i, j] = du_i/dx_j``.

    Args:
        domain: FEM mesh domain.
        u_vec: Nodal displacement array ``(num_dofs, dim)``.

    Returns:
        Array ``(n_cells, n_q, dim, dim)``.
    """
    _require_fem(domain, "grad_quad")

    fe = _get_fe_space(domain)
    cell_u = u_vec[fe.dof_cells]  # (n_cells, n_nodes_per_cell, dim)
    return jax.vmap(grad_vec)(fe.shape_grads, cell_u)  # (n_cells, n_q, dim, dim)


def dispatch_stress_divergence(domain: Domain, sigma_q: Array) -> Array:
    """Assemble weak stress-divergence residual ``int sigma : grad N dOmega``.

    FEM-only: the assembled weak residual has no grid (strong-form) analogue.

    Args:
        domain: FEM mesh domain.
        sigma_q: Quad-point stress tensor, shape ``(n_cells, n_q, voigt)``.

    Returns:
        Nodal force residual of shape ``(num_dofs, dim)``.
    """
    _require_fem(domain, "div of a quad-point stress tensor")

    fe = _get_fe_space(domain)
    dim = fe.dim
    # integrate_stress -> (n_cells, n_nodes_per_cell, dim)
    f_elem = jax.vmap(integrate_stress)(fe.shape_grads, fe.jxw, sigma_q)
    # scatter expects (n_cells, n_local) flat; we need interleaved dof order
    # scatter_elem_residual with n_vars=dim expects f_elem shape (n_cells, n_nodes_per_cell * n_vars)
    # and builds global dofs via n_vars * cell + i. Reshape accordingly.
    n_cells, n_nodes_per_cell, _ = f_elem.shape
    f_flat = f_elem.reshape(n_cells, n_nodes_per_cell * dim)
    flat = scatter_elem_residual(f_flat, fe.dof_cells, fe.num_dofs, n_vars=dim)
    return flat.reshape(fe.num_dofs, dim)


def dispatch_interp_to_quad(domain: Domain, u: Array) -> Array:
    """Interpolate a nodal field to FEM quadrature points.

    FEM-only: there are no quadrature points on a grid domain.

    Args:
        domain: FEM mesh domain.
        u: Nodal scalar field of shape ``(num_dofs,)`` or vector/tensor field
            of shape ``(num_dofs, *component_dims)``.

    Returns:
        Array of shape ``(n_cells, n_q)`` for scalar fields or
        ``(n_cells, n_q, *component_dims)`` for vector/tensor fields.
    """
    _require_fem(domain, "interp_to_quad")
    fe = _get_fe_space(domain)
    u = _require_nodal(domain, u, "interp_to_quad")
    # cell_u: (n_cells, n_nodes_per_cell) for scalar,
    #         (n_cells, n_nodes_per_cell, *component_dims) for vector/tensor.
    cell_u = u[fe.dof_cells]
    if cell_u.ndim == 2:
        # Scalar field: standard interpolation.
        return jnp.einsum("qn,cn->cq", fe.shape_vals, cell_u)
    # Vector/tensor field: interpolate each component axis independently.
    # cell_u: (n_cells, n_nodes_per_cell, *comp)
    # Result: (n_cells, n_q, *comp) via shape_vals (n_q, n_nodes_per_cell).
    return jnp.einsum("qn,cn...->cq...", fe.shape_vals, cell_u)


def dispatch_integrate_scalar_at_quad(domain: Domain, f_q: ArrayLike) -> Array:
    """Integrate a quad-point field against shape functions: ``int f_q N_i dOmega``.

    FEM-only: consumes quadrature-point values, which only exist on FEM.

    Handles scalar ``(n_cells, n_q)`` and vector/tensor ``(n_cells, n_q, *comp)``
    quad arrays.  Vector/tensor inputs are integrated component-wise (the
    consistent-mass action is block-diagonal across components), returning
    a nodal array of shape ``(num_dofs, *comp)``.

    Args:
        domain: FEM mesh domain.
        f_q: Quad-point field, shape ``(n_cells, n_q)`` or
            ``(n_cells, n_q, *component_dims)``.

    Returns:
        Nodal residual of shape ``(num_dofs,)`` for scalar inputs or
        ``(num_dofs, *component_dims)`` for vector/tensor inputs.
    """
    _require_fem(domain, "integrate_scalar_at_quad")
    # A constant, a per-cell value or a nodal value is read by its shape.
    return _integrate_nodal_or_quad(domain, f_q, "integrate_shape", ambiguous="quad")


def _fem_quad_action(domain: Domain, f_q: Array) -> Array:
    """Integrate a value at the quadrature points: ``int f_q N_i dOmega``."""
    fe = _get_fe_space(domain)

    def _scalar_integrate(f_scalar_q):
        # f_scalar_q: (n_cells, n_q)
        f_elem = jax.vmap(reaction_elem, in_axes=(None, 0, 0))(fe.shape_vals, fe.jxw, f_scalar_q)
        return scatter_elem_residual(f_elem, fe.dof_cells, fe.num_dofs, n_vars=1)

    if f_q.ndim <= 2:
        return _scalar_integrate(f_q)
    # Vector / tensor quad field: integrate each component independently.
    # Reshape to (n_cells, n_q, n_comp_flat), vmap over the last axis, then
    # restore the original component shape.
    orig_comp = f_q.shape[2:]
    flat = f_q.reshape(f_q.shape[0], f_q.shape[1], -1)  # (n_cells, n_q, n_comp)
    # vmap over comp axis: each call gets (n_cells, n_q), returns (num_dofs,).
    # out_axes=1 inserts the batch dim at position 1 → (num_dofs, n_comp).
    out = jax.vmap(_scalar_integrate, in_axes=2, out_axes=1)(flat)  # (num_dofs, n_comp)
    return out.reshape(fe.num_dofs, *orig_comp)


# =========================================================================
# FluxBC / PeriodicBC dispatch (thin router to backend files)
# =========================================================================


def dispatch_apply_bcs(
    domain: Domain,
    result: Array,
    u: Array,
    D: "ArrayLike | tuple | list",
    bcs: "Sequence",
    t: "Array | None" = None,
) -> Array:
    """Apply FluxBC / PeriodicBC corrections to an operator result.

    Called by ``Operators.div_flux`` and ``Operators.laplacian`` when a
    ``bcs=`` argument is provided.  Routes each BC to the correct backend
    based on ``domain.discretization_type`` and the BC type.

    Args:
        domain: Spatial domain.
        result: Already-evaluated operator result (zero-flux boundaries).
        u: Solution field (for solution-dependent flux BCs).
        D: Diffusivity (scalar or nodal; passed to backends for flux scaling).
        bcs: Sequence of [FluxBC][diff_soup.domains.operators.flux_bc.FluxBC]
            and/or [PeriodicBC][diff_soup.domains.grid.advection.bcs.PeriodicBC]
            objects.
        t: Current time.  Passed to ``flux_fn(u_boundary, t)``.
            ``None`` evaluates the conditions at ``t = 0.0``, which is exact for
            time-independent ``flux_fn``.
    """
    if t is None:
        t = jnp.array(0.0)

    dt = domain.discretization_type
    for bc in bcs:
        if isinstance(bc, FluxBC):
            result = _apply_flux_bc(domain, result, u, D, bc, dt, t)
        elif isinstance(bc, _NormalDerivativeBC):
            result = _apply_normal_derivative(domain, result, u, D, bc, dt)
        elif isinstance(bc, PeriodicBC):
            result = _apply_periodic_bc(domain, result, u, D, bc, dt)
    return result


def _apply_normal_derivative(domain, result, u, D, bc, dt):
    """Impose an evaluated outward normal derivative on one grid face."""
    if not isinstance(domain, (Grid1D, GridND)):
        raise ConfigurationError(
            f"a normal-derivative condition reached the operator bank on a "
            f"{type(domain).__name__}; it is imposed on structured grids only. On a "
            "mesh, state the flux density with spec.bc.flux."
        )
    flat_in = isinstance(domain, GridND) and result.ndim == 1
    if flat_in:
        result, u = result.reshape(domain.shape), u.reshape(domain.shape)
    D_axis = _axis_coefficient(domain, D, bc.axis)
    if dt == "fvm":
        value = bc.value if bc.mask is None else bc.value * bc.mask
        result = fvm_apply_normal_derivative(domain, result, D_axis, bc.axis, bc.side, value)
    elif dt == "fdm":
        result = fdm_apply_normal_derivative(
            domain, result, u, D_axis, bc.axis, bc.side, bc.value, bc.mask
        )
    else:
        raise ValueError(f"Unknown discretization type: {dt!r}")
    return result.reshape(-1) if flat_in else result


def _apply_flux_bc(
    domain: Domain,
    result: Array,
    u: Array,
    D: "ArrayLike | tuple | list",
    bc,
    dt: str,
    t: Array,
) -> Array:
    """Route FluxBC to the correct backend.

    ``flux_fn`` returns the outward normal derivative at each boundary cell; the
    condition is imposed as a ghost value (finite volume) or a ghost node
    (finite difference) by [_apply_normal_derivative][].

    Raises:
        ValueError: If a grid FluxBC names no ``side``, or the grid's
            discretization type is not recognized.
        ConfigurationError: If ``domain`` is a mesh; a flux through a tagged
            mesh boundary is declared with ``spec.bc.flux(var, tag=...)``.
    """
    if isinstance(domain, (Grid1D, GridND)):
        if dt not in ("fvm", "fdm"):
            raise ValueError(f"Unknown discretization type: {dt!r}")
        side = bc.side
        if side is None:
            raise _flux_bc_needs_side(domain, bc)
        axis = bc.axis if bc.axis is not None else 0
        flat_in = isinstance(domain, GridND) and result.ndim == 1
        u_grid = u.reshape(domain.shape) if flat_in else u
        end = 0 if side == "low" else u_grid.shape[axis] - 1
        u_b = jnp.take(u_grid, end, axis=axis)
        du_dn = jax.vmap(lambda ub: bc.flux_fn(ub, t))(u_b.reshape(-1)).reshape(u_b.shape)
        condition = _NormalDerivativeBC(axis=axis, side=side, value=du_dn)
        return _apply_normal_derivative(domain, result, u, D, condition, dt)

    _require_fem(domain, "FluxBC")
    raise _flux_bc_on_mesh(domain, bc)


def _flux_bc_needs_side(domain: Domain, bc) -> ValueError:
    """Build the error for a structured-grid FluxBC that names no side."""
    return ValueError(
        f"FluxBC on a {type(domain).__name__} must name the boundary it acts on: "
        f"pass side='low' or side='high' (with axis= on a GridND). "
        f"This condition carries tag={bc.tag!r}, axis={bc.axis!r}, side=None, "
        "which identifies no grid face."
    )


def _flux_bc_on_mesh(domain: Domain, bc) -> ConfigurationError:
    """Build the error for a FluxBC handed to the operator bank on a mesh."""
    return ConfigurationError(
        f"FluxBC(tag={bc.tag!r}) was passed to the operator bank on a "
        f"{type(domain).__name__}; the bank applies a FluxBC only on a Grid1D or "
        "GridND (by side= and axis=). A flux through a tagged mesh boundary is a "
        "boundary integral that the problem's boundary declaration assembles: "
        "declare it with spec.bc.flux(var, tag=...) and drop the FluxBC from bcs=."
    )


def _apply_periodic_bc(
    domain: Domain,
    result: Array,
    u: Array,
    D: "ArrayLike | tuple | list",
    bc,
    dt: str,
) -> Array:
    """Route PeriodicBC to the correct backend.

    Raises:
        NotImplementedError: If ``domain`` is not a cartesian grid, or the
            grid's discretization has no periodic wrap kernel.
    """
    if isinstance(domain, Grid1D):
        D_1 = _axis_coefficient(domain, D, 0)
        if dt == "fvm":
            return fvm_apply_periodic_bc_1d(domain, result, u, D_1)
        elif dt == "fdm":
            return fdm_apply_periodic_bc_1d(domain, result, u, D_1)
        raise _no_periodic_wrap(domain, dt)

    if isinstance(domain, GridND):
        axes = _periodic_axes_from_bc(bc, domain)
        flat_in = result.ndim == 1
        shape = domain.shape
        if flat_in:
            result = result.reshape(shape)
            u_grid = u.reshape(shape)
        else:
            u_grid = u
        if dt not in ("fvm", "fdm"):
            raise _no_periodic_wrap(domain, dt)
        wrap = fvm_apply_periodic_bc_nd if dt == "fvm" else fdm_apply_periodic_bc_nd
        for axis in axes:
            result = wrap(domain, result, u_grid, _axis_coefficient(domain, D, axis), (axis,))
        return result.reshape(-1) if flat_in else result

    raise _no_periodic_wrap(domain, dt)


def _no_periodic_wrap(domain: Domain, dt: str) -> NotImplementedError:
    """Build the error for a PeriodicBC on a domain with no wrap kernel."""
    return NotImplementedError(
        f"PeriodicBC has no wrap kernel for {type(domain).__name__} "
        f"(discretization {dt!r}); periodic wrapping is defined on cartesian "
        "Grid1D / GridND domains built with method='fvm' or method='fdm'. "
        "On a mesh, tie the opposing boundaries with a constraint instead."
    )


def _periodic_axes_from_bc(bc, domain: "GridND") -> tuple[int, ...]:
    """Resolve which grid axes a PeriodicBC wraps.

    The advection-layer ``PeriodicBC`` tag carries a single ``axis``; a
    whole-grid periodic wrap is expressed as one such tag per periodic
    axis, so this normally returns a single-axis tuple. If a caller passes
    a BC with no ``axis`` attribute, every domain axis is wrapped.
    """
    axis = getattr(bc, "axis", None)
    if axis is not None:
        return (int(axis),)
    return tuple(range(len(domain.axes)))
