# Operators

Inside an equation closure, `q.ops` is the operator bank bound to the
field's domain. The [operators overview](../../overview/modeling/operators.md)
explains the conventions.

The methods, grouped by kind:

- Differential operators: `laplacian`, `div_flux`, `gradient` (alias
  `grad`), `divergence` (alias `div`), `curl`, `advect_upwind`,
  `div_advect`, `face_flux`.
- Mass and integration weights: `mass`, `lumped_mass`.
- Solid and fluid mechanics: `strain`, `linear_elastic_stress`,
  `deformation_gradient`, `cauchy_stress_from_S_F`, `strain_rate`,
  `viscous_stress_newtonian`.
- Porous media: `tortuosity_corrected`.
- Mesh boundaries: `boundary_integral`, `boundary_trace`, `surface_flux`.
- Pointwise tensor algebra: `trace`, `det`, `inv`, `transpose`, `sym`,
  `skew`, `outer`, `tensor_contract`, `double_dot`, `scale`.
- Finite element quadrature: `grad_quad`, `interp_to_quad`, `integrate`,
  `quad_gradient`, `integrate_shape`, `integrate_shape_gradient`,
  `wrap_quad_value`, `wrap_quad_field`, `unwrap_field`.

::: diff_soup.domains.Operators

---

??? abstract "Abstract operator bank"

    The details matter only when writing a bank for a new discretization.

    ::: diff_soup.domains.AbstractOperator
        options:
          members:
            - provides
            - laplacian
            - div_flux
            - gradient
            - divergence
            - mass
            - lumped_mass
            - advect_upwind
            - div_advect
            - as_weak
            - for_residual
