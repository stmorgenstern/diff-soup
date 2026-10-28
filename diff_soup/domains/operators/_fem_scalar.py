"""Scalar field operators for FEM element-level assembly.

Provides quadrature-point operations (interpolation, gradient), integration
operations (test-function weighted), and composite element operators
(diffusion, Laplacian, reaction, mass) for scalar fields.

All functions are pure, JIT-compatible, AD-compatible, and vmap-compatible.
They operate on a single element; vmap over cells externally.

Private module — users should access these symbols via
[diff_soup.domains.operators.fem][] or the high-level [diff_soup.Operators][]
dispatch class rather than importing from here directly.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    # Quadrature-point operations
    "interp",
    "grad",
    # Integration operations
    "integrate_scalar",
    "integrate_grad",
    # Composite element operators
    "diffusion_elem",
    "stiffness_action_elem",
    "reaction_elem",
    "mass_elem",
    "div_flux_elem",
    # Region coefficients
    "cell_indicator",
]


# ---------------------------------------------------------------------------
# Quadrature-point operations (building blocks)
# ---------------------------------------------------------------------------


def interp(
    shape_vals: Float[Array, "n_q n_nodes"],
    u_nodes: Float[Array, "n_nodes"],
) -> Float[Array, "n_q"]:
    """Interpolate nodal scalar field to quadrature points."""
    return shape_vals @ u_nodes


def grad(
    grads_e: Float[Array, "n_q n_nodes dim"],
    u_nodes: Float[Array, "n_nodes"],
) -> Float[Array, "n_q dim"]:
    """Gradient of scalar field at quadrature points."""
    return jnp.einsum("qnd,n->qd", grads_e, u_nodes)


# ---------------------------------------------------------------------------
# Integration operations (test-function weighted)
# ---------------------------------------------------------------------------


def integrate_scalar(
    shape_vals: Float[Array, "n_q n_nodes"],
    jxw: Float[Array, "n_q"],
    f_q: Float[Array, "n_q"],
) -> Float[Array, "n_nodes"]:
    r"""Element residual :math:`\int f \, N_i \, d\Omega`.

    Integrates a scalar field ``f_q`` at quadrature points against shape
    functions, weighted by ``jxw``.
    """
    return jnp.einsum("q,qn,q->n", f_q, shape_vals, jxw)


def integrate_grad(
    grads_e: Float[Array, "n_q n_nodes dim"],
    jxw: Float[Array, "n_q"],
    flux_q: Float[Array, "n_q dim"],
) -> Float[Array, "n_nodes"]:
    r"""Element residual :math:`\int \mathbf{flux} \cdot \nabla N_i \, d\Omega`.

    Integrates a vector flux at quadrature points against shape function
    gradients (weak gradient operator).
    """
    # flux_q dot grad_N at each (q, node), then integrate over q
    return jnp.einsum("qd,qnd,q->n", flux_q, grads_e, jxw)


# ---------------------------------------------------------------------------
# Composite element operators
# ---------------------------------------------------------------------------


def diffusion_elem(
    grads_e: Float[Array, "n_q n_nodes dim"],
    jxw: Float[Array, "n_q"],
    D: Float[Array, "..."],
    u_nodes: Float[Array, "n_nodes"],
) -> Float[Array, "n_nodes"]:
    r"""Element residual for :math:`-\nabla \cdot (D \nabla u)`.

    Weak form: :math:`\int D \, \nabla u \cdot \nabla N_i \, d\Omega`.

    Args:
        grads_e: Shape function gradients, ``(n_q, n_nodes, dim)``.
        jxw: Integration measure at each quadrature point, ``(n_q,)`` — the
            quadrature weight times the Jacobian determinant, times the
            coordinate metric on a curvilinear mesh.
        D: Diffusivity -- scalar (constant) or ``(n_q,)`` (spatially varying).
        u_nodes: Nodal solution values, ``(n_nodes,)``.

    Returns:
        Element residual contribution, ``(n_nodes,)``.
    """
    grad_u = grad(grads_e, u_nodes)  # (n_q, dim)
    flux_q = jnp.expand_dims(D, axis=-1) * grad_u if jnp.ndim(D) > 0 else D * grad_u
    return integrate_grad(grads_e, jxw, flux_q)


def stiffness_action_elem(
    grads_e: Float[Array, "n_q n_nodes dim"],
    jxw: Float[Array, "n_q"],
    u_nodes: Float[Array, "n_nodes"],
) -> Float[Array, "n_nodes"]:
    r"""Element stiffness action :math:`(K@u)_e = \int \nabla u \cdot \nabla N_i\,d\Omega` (the residual of :math:`-\nabla^2 u`).

    Returns ``+K@u`` (positive semi-definite): the weak form of
    :math:`-\nabla^2 u`.  When building a gradient-penalty term in the
    :math:`\mu`-residual of Cahn--Hilliard / Allen--Cahn, the correct sign is
    ``-kappa * stiffness_action_elem`` (equivalently ``+kappa * q.ops.laplacian``
    which already carries the IBP sign flip).

    Equivalent to ``diffusion_elem`` with ``D = 1``.
    """
    grad_u = grad(grads_e, u_nodes)  # (n_q, dim)
    return integrate_grad(grads_e, jxw, grad_u)


def reaction_elem(
    shape_vals: Float[Array, "n_q n_nodes"],
    jxw: Float[Array, "n_q"],
    r_q: Float[Array, "n_q"],
) -> Float[Array, "n_nodes"]:
    r"""Element residual for reaction source :math:`r(x)`.

    Weak form: :math:`\int r \, N_i \, d\Omega`.

    Args:
        shape_vals: Shape function values, ``(n_q, n_nodes)``.
        jxw: Integration measure at each quadrature point, ``(n_q,)`` — the
            quadrature weight times the Jacobian determinant, times the
            coordinate metric on a curvilinear mesh.
        r_q: Reaction source values at quadrature points, ``(n_q,)``.

    Returns:
        Element residual contribution, ``(n_nodes,)``.
    """
    return integrate_scalar(shape_vals, jxw, r_q)


def mass_elem(
    shape_vals: Float[Array, "n_q n_nodes"],
    jxw: Float[Array, "n_q"],
    u_nodes: Float[Array, "n_nodes"],
) -> Float[Array, "n_nodes"]:
    r"""Element mass contribution :math:`\int u \, N_i \, d\Omega`.

    Args:
        shape_vals: Shape function values, ``(n_q, n_nodes)``.
        jxw: Integration measure at each quadrature point, ``(n_q,)`` — the
            quadrature weight times the Jacobian determinant, times the
            coordinate metric on a curvilinear mesh.
        u_nodes: Nodal solution values, ``(n_nodes,)``.

    Returns:
        Element residual contribution, ``(n_nodes,)``.
    """
    u_q = interp(shape_vals, u_nodes)  # (n_q,)
    return integrate_scalar(shape_vals, jxw, u_q)


def div_flux_elem(
    grads_e: Float[Array, "n_q n_nodes dim"],
    jxw: Float[Array, "n_q"],
    D_q: Float[Array, "..."],
    u_nodes: Float[Array, "n_nodes"],
) -> Float[Array, "n_nodes"]:
    r"""Element residual for :math:`-\nabla \cdot (D(x) \nabla u)` with general diffusivity.

    Weak form: :math:`\int D \, \nabla u \cdot \nabla N_i \, d\Omega` (scalar D)
    or :math:`\int (D \nabla u) \cdot \nabla N_i \, d\Omega` (tensor D).

    Args:
        grads_e: Shape function gradients, ``(n_q, n_nodes, dim)``.
        jxw: Integration measure at each quadrature point, ``(n_q,)`` — the
            quadrature weight times the Jacobian determinant, times the
            coordinate metric on a curvilinear mesh.
        D_q: Diffusivity at quadrature points. Either ``(n_q,)`` for isotropic
            spatially varying, or ``(n_q, dim, dim)`` for anisotropic tensor.
        u_nodes: Nodal solution values, ``(n_nodes,)``.

    Returns:
        Element residual contribution, ``(n_nodes,)``.
    """
    grad_u = grad(grads_e, u_nodes)  # (n_q, dim)

    if jnp.ndim(D_q) == 1:
        # Scalar D: D_q (n_q,), grad_u (n_q, dim) -> flux (n_q, dim)
        flux_q = D_q[:, None] * grad_u
    elif jnp.ndim(D_q) == 3:
        # Tensor D: D_q (n_q, dim, dim), grad_u (n_q, dim) -> flux (n_q, dim)
        flux_q = jnp.einsum("qij,qj->qi", D_q, grad_u)
    else:
        raise ValueError(f"D_q must have 1 or 3 dimensions, got ndim={jnp.ndim(D_q)}")

    return integrate_grad(grads_e, jxw, flux_q)


# ---------------------------------------------------------------------------
# Region coefficients
# ---------------------------------------------------------------------------


def cell_indicator(domain, name: str) -> Float[Array, " n_cells"]:
    """One per cell of a named region of a tagged mesh and zero elsewhere.

    A weighted sum such as ``0.1 * ind_a + 1.0 * ind_b`` is a per-cell
    coefficient for ``div_flux``, ``mass`` and ``integrate``.

    Raises:
        TypeError: If ``domain`` carries no cell regions.
        KeyError: If the mesh names no region ``name``.
    """
    method = getattr(domain, "cell_indicator", None)
    if method is None or getattr(domain, "cell_tags", None) is None:
        raise TypeError(
            f"cell_indicator({name!r}): a {type(domain).__name__} domain carries no cell "
            "regions. Build a TaggedMesh with cell_tags and cell_tag_names, for example "
            "with polygons_to_tagged_mesh or read_mesh of a gmsh file with physical surfaces."
        )
    return method(name)
