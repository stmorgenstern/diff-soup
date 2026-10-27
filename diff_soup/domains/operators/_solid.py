"""Solid-mechanics kernels behind the operator bank's solid verbs.

Combines the [diff_soup.domains.operators._tensor][] primitives with
``dispatch_gradient`` (vector input -> Jacobian) to give
[Operators.strain][diff_soup.domains.operators.base.Operators.strain],
``linear_elastic_stress``, ``deformation_gradient`` and
``cauchy_stress_from_S_F``.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from diff_soup.domains.operators._tensor import det, sym, trace
from diff_soup.domains.operators.dispatch import dispatch_gradient


def strain(u: Float[Array, "n d"], *, mesh: Any) -> Float[Array, "n d d"]:
    """Small-strain tensor ``eps = sym(grad u)``.

    Args:
        u: Vector field of shape ``(n_nodes, dim)``.
        mesh: FEM mesh (must support FEM gradient via dispatch).
    """
    return sym(dispatch_gradient(mesh, u))


def linear_elastic_stress(
    u: Float[Array, "n d"],
    *,
    E: float,
    nu: float,
    mesh: Any,
    eigenstrain: Optional[Float[Array, "n d d"]] = None,
) -> Float[Array, "n d d"]:
    """Cauchy stress under isotropic linear elasticity.

    ``sigma = lambda * tr(eps - eps*) * I + 2 * mu * (eps - eps*)``

    Args:
        u: Displacement vector field ``(n_nodes, dim)``.
        E: Young's modulus.
        nu: Poisson's ratio.
        mesh: FEM mesh.
        eigenstrain: Optional ``(n_nodes, dim, dim)`` eigenstrain to subtract
            from the geometric strain before computing stress.
    """
    eps = strain(u, mesh=mesh)
    if eigenstrain is not None:
        eps = eps - eigenstrain
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    d = eps.shape[-1]
    tr_eps = trace(eps)
    eye_d = jnp.eye(d)
    return lam * tr_eps[..., None, None] * eye_d[None, :, :] + 2.0 * mu * eps


def deformation_gradient(gradu: Float[Array, "... d d"]) -> Float[Array, "... d d"]:
    """Deformation gradient ``F = I + grad u``."""
    return gradu + jnp.eye(gradu.shape[-1])


def cauchy_stress_from_S_F(
    S: Float[Array, "... d d"],
    F: Float[Array, "... d d"],
) -> Float[Array, "... d d"]:
    """Cauchy stress pulled back from second Piola-Kirchhoff: ``sigma = (1/J) F S F^T``."""
    J = det(F)
    return jnp.einsum("...ij,...jk,...lk->...il", F, S, F) / J[..., None, None]


__all__ = ["strain", "linear_elastic_stress", "deformation_gradient", "cauchy_stress_from_S_F"]
