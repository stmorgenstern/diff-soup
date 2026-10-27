"""Tensor and vector field operators for continuum mechanics (L2 Discrete Physics).

Element-level operations for FEM assembly of momentum balance and related
equations.  All functions are pure, JIT-compatible, AD-compatible, and
vmap-compatible.

Voigt convention (standard, no sqrt(2) factors):
    2D: [sigma_xx, sigma_yy, sigma_xy]            (3 components)
    3D: [sigma_xx, sigma_yy, sigma_zz,            (6 components)
         sigma_yz, sigma_xz, sigma_xy]
Engineering shear strain: gamma_xy = 2*eps_xy.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from diff_soup.utils import safe_sqrt

__all__ = [
    # quadrature-point operations
    "grad_vec",
    "sym_grad",
    "strain_voigt",
    "double_contract",
    # tensor <-> Voigt conversion
    "tensor_to_voigt_2d",
    "voigt_to_tensor_2d",
    "tensor_to_voigt_3d",
    "voigt_to_tensor_3d",
    "strain_to_voigt",
    # integration
    "integrate_stress",
    "integrate_body_force",
    # stiffness utilities
    "rotate_stiffness_2d",
    "plane_stress_stiffness",
    "plane_strain_stiffness",
    "isotropic_3d_stiffness",
    # 3D tensor helper (tests + docs)
    "voigt_6x6_to_4th_order",
    # derived quantities
    "von_mises_2d",
    "hydrostatic",
    "deviatoric_voigt",
    # full-tensor divergence
    "fem_div_tensor",
    # not exported (not in active use): orthorhombic_stiffness_3d,
    # bond_rotation_matrix_3d, fourth_order_to_voigt_6x6,
    # plane_strain_reduction, plane_stress_reduction, reduce_eigenstrain_to_2d
]


def fem_div_tensor(domain, sigma):
    """Per-row weak divergence of a (n_nodes, dim, dim) tensor field.

    Builds on the existing scalar ``_fem_divergence`` (rank-1 path) by
    treating each row of ``sigma`` as a vector field and assembling the
    weak divergence of that row.  Returns ``(n_nodes, dim)`` — the row-wise
    divergence of σ.

    Used by ``q.ops.div(sigma)`` for solid mechanics and by
    ``linear_elastic_stress`` momentum residuals.

    Args:
        domain: FEM mesh (or any domain that ``_fem_divergence`` accepts).
        sigma: ``(n_nodes, dim, dim)`` tensor field.

    Returns:
        ``(n_nodes, dim)`` row-wise weak divergence.
    """
    from diff_soup.domains.operators.dispatch import _fem_divergence

    rows = []
    for i in range(sigma.shape[-2]):
        rows.append(_fem_divergence(domain, sigma[..., i, :]))
    return jnp.stack(rows, axis=-1)


# ---------------------------------------------------------------------------
# 1. Quadrature-point operations
# ---------------------------------------------------------------------------


def grad_vec(
    grads_e: Float[Array, "nq nodes dim"],
    u_nodes: Float[Array, "nodes dim"],
) -> Float[Array, "nq dim dim"]:
    """Displacement gradient tensor at quadrature points.

    Returns H[q, i, j] = du_i/dx_j = sum_n grads_e[q,n,j] * u_nodes[n,i].
    """
    # grads_e: (nq, nodes, dim), u_nodes: (nodes, dim)
    # H[q, i, j] = sum_n u_nodes[n, i] * grads_e[q, n, j]
    return jnp.einsum("qnj,ni->qij", grads_e, u_nodes)


def sym_grad(
    grads_e: Float[Array, "nq nodes dim"],
    u_nodes: Float[Array, "nodes dim"],
) -> Float[Array, "nq dim dim"]:
    """Symmetric gradient (infinitesimal strain) at quadrature points.

    eps = (grad_u + grad_u^T) / 2.
    """
    H = grad_vec(grads_e, u_nodes)
    return 0.5 * (H + jnp.swapaxes(H, -2, -1))


def strain_voigt(
    grads_e: Float[Array, "nq nodes dim"],
    u_nodes: Float[Array, "nodes dim"],
) -> Float[Array, "nq nv"]:
    """Strain in Voigt notation at quadrature points.

    2D -> (nq, 3): [eps_xx, eps_yy, gamma_xy]
    3D -> (nq, 6): [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]

    Engineering shear strains: gamma_ij = 2 * eps_ij.
    """
    eps = sym_grad(grads_e, u_nodes)
    dim = grads_e.shape[-1]
    return strain_to_voigt(eps, dim)


def double_contract(
    C: Float[Array, "nv nv"],
    strain_v: Float[Array, "*batch nv"],
) -> Float[Array, "*batch nv"]:
    """Stress from constitutive law: sigma = C : eps (in Voigt notation).

    C: (nv, nv) stiffness matrix, strain_v: (..., nv) -> (..., nv).
    """
    return jnp.einsum("ij,...j->...i", C, strain_v)


# ---------------------------------------------------------------------------
# 2. Tensor <-> Voigt conversion
# ---------------------------------------------------------------------------


def tensor_to_voigt_2d(
    T: Float[Array, "*batch 2 2"],
) -> Float[Array, "*batch 3"]:
    """Symmetric 2D tensor -> Voigt vector [xx, yy, xy]."""
    return jnp.stack([T[..., 0, 0], T[..., 1, 1], T[..., 0, 1]], axis=-1)


def voigt_to_tensor_2d(
    v: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 2 2"]:
    """Voigt vector [xx, yy, xy] -> symmetric 2D tensor."""
    xx = v[..., 0]
    yy = v[..., 1]
    xy = v[..., 2]
    row0 = jnp.stack([xx, xy], axis=-1)
    row1 = jnp.stack([xy, yy], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def tensor_to_voigt_3d(
    T: Float[Array, "*batch 3 3"],
) -> Float[Array, "*batch 6"]:
    """Symmetric 3D tensor -> Voigt vector [xx, yy, zz, yz, xz, xy]."""
    return jnp.stack(
        [
            T[..., 0, 0],
            T[..., 1, 1],
            T[..., 2, 2],
            T[..., 1, 2],
            T[..., 0, 2],
            T[..., 0, 1],
        ],
        axis=-1,
    )


def voigt_to_tensor_3d(
    v: Float[Array, "*batch 6"],
) -> Float[Array, "*batch 3 3"]:
    """Voigt vector [xx, yy, zz, yz, xz, xy] -> symmetric 3D tensor."""
    xx, yy, zz = v[..., 0], v[..., 1], v[..., 2]
    yz, xz, xy = v[..., 3], v[..., 4], v[..., 5]
    row0 = jnp.stack([xx, xy, xz], axis=-1)
    row1 = jnp.stack([xy, yy, yz], axis=-1)
    row2 = jnp.stack([xz, yz, zz], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def strain_to_voigt(
    eps_tensor: Float[Array, "*batch dim dim"],
    dim: int,
) -> Float[Array, "*batch nv"]:
    """Strain tensor -> engineering Voigt (gamma = 2*eps for shear components)."""
    if dim == 2:
        return jnp.stack(
            [
                eps_tensor[..., 0, 0],
                eps_tensor[..., 1, 1],
                2.0 * eps_tensor[..., 0, 1],  # gamma_xy = 2 * eps_xy
            ],
            axis=-1,
        )
    elif dim == 3:
        return jnp.stack(
            [
                eps_tensor[..., 0, 0],
                eps_tensor[..., 1, 1],
                eps_tensor[..., 2, 2],
                2.0 * eps_tensor[..., 1, 2],  # gamma_yz
                2.0 * eps_tensor[..., 0, 2],  # gamma_xz
                2.0 * eps_tensor[..., 0, 1],  # gamma_xy
            ],
            axis=-1,
        )
    elif dim == 1:
        raise ValueError(
            "strain_to_voigt has no 1D convention: engineering Voigt notation "
            "orders the shear components, and a line element has none. Solve "
            "mechanics on a 2D or 3D mesh."
        )
    else:
        raise ValueError(f"strain_to_voigt: dim must be 2 or 3, got {dim}")


# ---------------------------------------------------------------------------
# 3. Integration operations
# ---------------------------------------------------------------------------


def integrate_stress(
    grads_e: Float[Array, "nq nodes dim"],
    jxw: Float[Array, "nq"],
    stress_v: Float[Array, "nq nv"],
) -> Float[Array, "nodes dim"]:
    """Integrate stress divergence: f_n_i = int sigma_ij dN_n/dx_j dOmega.

    This is the standard weak-form momentum residual.  For 2D with
    stress_v = [sigma_xx, sigma_yy, sigma_xy]:

        f[n, 0] = int (sigma_xx * dN_n/dx + sigma_xy * dN_n/dy) dOmega
        f[n, 1] = int (sigma_xy * dN_n/dx + sigma_yy * dN_n/dy) dOmega

    Sign convention: positive sigma_xx with positive dN/dx gives positive
    force (tensile stress pulls outward).
    """
    dim = grads_e.shape[-1]

    # Reconstruct full symmetric stress tensor from Voigt
    if dim == 2:
        stress_tensor = voigt_to_tensor_2d(stress_v)  # (nq, 2, 2)
    else:
        stress_tensor = voigt_to_tensor_3d(stress_v)  # (nq, 3, 3)

    # f[n, i] = sum_q sum_j sigma[q, i, j] * grads_e[q, n, j] * jxw[q]
    return jnp.einsum("qij,qnj,q->ni", stress_tensor, grads_e, jxw)


def integrate_body_force(
    shape_vals: Float[Array, "nq nodes"],
    jxw: Float[Array, "nq"],
    b_q: Float[Array, "nq dim"],
) -> Float[Array, "nodes dim"]:
    """Integrate body force: f_n_i = int b_i * N_n dOmega.

    b_q: body force at quadrature points, (nq, dim).
    Returns (nodes, dim) element body force contribution.
    """
    # f[n, i] = sum_q b_q[q, i] * shape_vals[q, n] * jxw[q]
    return jnp.einsum("qi,qn,q->ni", b_q, shape_vals, jxw)


# ---------------------------------------------------------------------------
# 4. Stiffness tensor utilities
# ---------------------------------------------------------------------------


def rotate_stiffness_2d(
    C: Float[Array, "3 3"],
    theta: Float[Array, ""],
) -> Float[Array, "3 3"]:
    """Rotate 2D Voigt stiffness matrix by angle theta (radians).

    Applies the Bond matrix transformation C' = T C T^T, where T is the
    stress transformation matrix for rotation by theta.

    For Voigt ordering [xx, yy, xy], the Bond matrix is::

        T = [[ c^2,  s^2,  2sc ],
             [ s^2,  c^2, -2sc ],
             [ -sc,   sc,  c^2 - s^2 ]]

    where c = cos(theta), s = sin(theta).
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    c2 = c * c
    s2 = s * s
    sc = s * c

    T = jnp.array(
        [
            [c2, s2, 2.0 * sc],
            [s2, c2, -2.0 * sc],
            [-sc, sc, c2 - s2],
        ]
    )
    return T @ C @ T.T


def plane_stress_stiffness(
    E: Float[Array, ""],
    nu: Float[Array, ""],
) -> Float[Array, "3 3"]:
    """Isotropic plane stress stiffness matrix (3x3 Voigt).

    C = E / (1 - nu^2) * [[1,  nu, 0          ],
                           [nu, 1,  0          ],
                           [0,  0,  (1-nu)/2   ]]
    """
    fac = E / (1.0 - nu * nu)
    return fac * jnp.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ]
    )


def plane_strain_stiffness(
    E: Float[Array, ""],
    nu: Float[Array, ""],
) -> Float[Array, "3 3"]:
    """Isotropic plane strain stiffness matrix (3x3 Voigt).

    C = E / ((1+nu)(1-2nu)) * [[1-nu,  nu,   0            ],
                                [nu,    1-nu, 0            ],
                                [0,     0,    (1-2nu)/2    ]]
    """
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return fac * jnp.array(
        [
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ]
    )


def isotropic_3d_stiffness(
    E: Float[Array, ""],
    nu: Float[Array, ""],
) -> Float[Array, "6 6"]:
    """Full 3D isotropic stiffness matrix (6x6 Voigt).

    Uses Lame parameters: lambda = E*nu / ((1+nu)(1-2nu)), mu = E / (2(1+nu)).

    C = [[lam+2mu, lam,     lam,     0,  0,  0 ],
         [lam,     lam+2mu, lam,     0,  0,  0 ],
         [lam,     lam,     lam+2mu, 0,  0,  0 ],
         [0,       0,       0,       mu, 0,  0 ],
         [0,       0,       0,       0,  mu, 0 ],
         [0,       0,       0,       0,  0,  mu]]
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    d = lam + 2.0 * mu
    z = jnp.zeros_like(E)
    return jnp.array(
        [
            [d, lam, lam, z, z, z],
            [lam, d, lam, z, z, z],
            [lam, lam, d, z, z, z],
            [z, z, z, mu, z, z],
            [z, z, z, z, mu, z],
            [z, z, z, z, z, mu],
        ]
    )


# ---------------------------------------------------------------------------
# 4b. 3D material utilities
# ---------------------------------------------------------------------------
# orthorhombic_stiffness_3d, bond_rotation_matrix_3d,
# fourth_order_to_voigt_6x6, plane_strain_reduction, plane_stress_reduction,
# reduce_eigenstrain_to_2d — not exported (no active callers).
# voigt_6x6_to_4th_order is kept: has callers in tests + docs.


def voigt_6x6_to_4th_order(C_voigt: np.ndarray) -> np.ndarray:
    """Convert 6x6 Voigt stiffness to full (3,3,3,3) 4th-order tensor.

    Accounts for the factor-of-2 Voigt shear convention.
    """
    voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    voigt_fac = [1, 1, 1, 2, 2, 2]
    C4 = np.zeros((3, 3, 3, 3))
    for vi in range(6):
        for vj in range(6):
            i, j = voigt_map[vi]
            k, m = voigt_map[vj]
            val = C_voigt[vi, vj] / (voigt_fac[vi] * voigt_fac[vj])
            for ii, jj, kk, ll in [
                (i, j, k, m),
                (j, i, k, m),
                (i, j, m, k),
                (j, i, m, k),
            ]:
                C4[ii, jj, kk, ll] = val
    return C4


# ---------------------------------------------------------------------------
# 5. Derived quantities
# ---------------------------------------------------------------------------


def von_mises_2d(
    stress_v: Float[Array, "*batch 3"],
) -> Float[Array, "*batch"]:
    """Von Mises equivalent stress from 2D plane stress Voigt [sxx, syy, sxy].

    sigma_vm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2).
    """
    sxx = stress_v[..., 0]
    syy = stress_v[..., 1]
    sxy = stress_v[..., 2]
    disc = sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy
    return safe_sqrt(disc)  # Double-where square root keeps the gradient finite at zero.


def hydrostatic(
    stress_v: Float[Array, "*batch nv"],
    dim: int,
) -> Float[Array, "*batch"]:
    """Hydrostatic (mean) stress: p = tr(sigma) / dim.

    2D: (sxx + syy) / 2
    3D: (sxx + syy + szz) / 3
    """
    if dim == 2:
        return (stress_v[..., 0] + stress_v[..., 1]) / 2.0
    elif dim == 3:
        return (stress_v[..., 0] + stress_v[..., 1] + stress_v[..., 2]) / 3.0
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def deviatoric_voigt(
    stress_v: Float[Array, "*batch nv"],
    dim: int,
) -> Float[Array, "*batch nv"]:
    """Deviatoric stress in Voigt notation: s = sigma - p * I.

    Only the normal components are modified; shear components are unchanged.
    """
    p = hydrostatic(stress_v, dim)
    if dim == 2:
        return jnp.stack(
            [
                stress_v[..., 0] - p,
                stress_v[..., 1] - p,
                stress_v[..., 2],
            ],
            axis=-1,
        )
    elif dim == 3:
        return jnp.stack(
            [
                stress_v[..., 0] - p,
                stress_v[..., 1] - p,
                stress_v[..., 2] - p,
                stress_v[..., 3],
                stress_v[..., 4],
                stress_v[..., 5],
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
