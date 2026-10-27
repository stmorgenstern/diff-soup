"""Curated public re-exports of low-level FEM element operators.

This module is the **escape hatch** for users who need direct access to FEM
element-level primitives (quadrature-point ops, integration ops, constitutive
building blocks). For most users, the high-level [Operators][]
dispatch class is the recommended entry point — it handles assembly and works
identically on FEM meshes, FVM grids, and FDM grids.

Use this module only when you need to write custom element kernels or
escape-hatch assembly loops not covered by [Operators][].  The module ships
element primitives, not per-physics equation kernels: compose your own kernels
from the scalar / tensor primitives below.  In a recipe, ``q.grad``, ``q.flux``
and ``q.source`` assemble the same integrals without touching element arrays.

Example:
    >>> from diff_soup.domains.operators.fem import stiffness_action_elem, integrate_grad
    >>> f_elem = jax.vmap(stiffness_action_elem)(shape_grads, jxw, cell_u)
"""

from __future__ import annotations

from ._fem_scalar import (
    diffusion_elem,
    div_flux_elem,
    grad,
    integrate_grad,
    integrate_scalar,
    interp,
    mass_elem,
    reaction_elem,
    stiffness_action_elem,
)
from ._fem_tensor import (
    deviatoric_voigt,
    double_contract,
    grad_vec,
    hydrostatic,
    integrate_body_force,
    integrate_stress,
    isotropic_3d_stiffness,
    plane_strain_stiffness,
    plane_stress_stiffness,
    rotate_stiffness_2d,
    strain_to_voigt,
    strain_voigt,
    sym_grad,
    tensor_to_voigt_2d,
    tensor_to_voigt_3d,
    voigt_to_tensor_2d,
    voigt_to_tensor_3d,
    von_mises_2d,
)

__all__ = [
    # _fem_scalar
    "interp",
    "grad",
    "integrate_scalar",
    "integrate_grad",
    "diffusion_elem",
    "stiffness_action_elem",
    "reaction_elem",
    "mass_elem",
    "div_flux_elem",
    # _fem_tensor
    "grad_vec",
    "sym_grad",
    "strain_voigt",
    "double_contract",
    "tensor_to_voigt_2d",
    "voigt_to_tensor_2d",
    "tensor_to_voigt_3d",
    "voigt_to_tensor_3d",
    "strain_to_voigt",
    "integrate_stress",
    "integrate_body_force",
    "rotate_stiffness_2d",
    "plane_stress_stiffness",
    "plane_strain_stiffness",
    "isotropic_3d_stiffness",
    "von_mises_2d",
    "hydrostatic",
    "deviatoric_voigt",
]
