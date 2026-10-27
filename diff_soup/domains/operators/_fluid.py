"""Fluid-mechanics kernels behind the operator bank's fluid verbs."""

from __future__ import annotations

from jaxtyping import Array, Float

from diff_soup.domains.operators._solid import strain


def strain_rate(u: Float[Array, "n d"], *, mesh) -> Float[Array, "n d d"]:
    """Symmetric velocity gradient ``eps_dot = sym(grad u)``."""
    return strain(u, mesh=mesh)


def viscous_stress_newtonian_pointwise(
    eps_dot: Float[Array, "... d d"],
    *,
    mu: float,
) -> Float[Array, "... d d"]:
    """Newtonian viscous stress at the field level: ``tau = 2 mu eps_dot``."""
    return 2.0 * mu * eps_dot


def viscous_stress_newtonian(
    u: Float[Array, "n d"],
    *,
    mu: float,
    mesh,
) -> Float[Array, "n d d"]:
    """Newtonian viscous stress for a velocity field on a mesh."""
    return viscous_stress_newtonian_pointwise(strain_rate(u, mesh=mesh), mu=mu)


__all__ = [
    "strain_rate",
    "viscous_stress_newtonian",
    "viscous_stress_newtonian_pointwise",
]
