# Changelog

## 1.0.1 (unreleased)

Fixes and additions from the first round of tutorials written against 1.0.0. Entry ids refer to the
issue ledger of that effort.

### Behaviour changes

- `spec.observe.spatial_mean` and `spec.observe.boundary_mean` now average `fn`'s nodal values
  (volume-weighted over the domain; face-area or facet-measure weighted over `tag`) and return a
  scalar. A `fn` that returns a scalar is refused at build; one that already averaged per node is
  averaged again. `boundary_mean(tag=)` must name a tag the domain resolves. 1.0.0 recorded the label
  only. (A-6)
- `spec.bc.neumann` imposes the normal derivative `du/dn = s` on FVM, FDM and FEM; it was an alias of
  `spec.bc.flux` (`D du/dn = s`). `name=` and `cross_facet=` stay on `flux`. (R-14)
- On 2D and 3D grids, flux, Robin and Neumann conditions and boundary integrals weight each selected
  cell by one face. A node selection that spans several faces, or holds only edge or corner cells,
  raises instead of over-counting. (R-29)
- The operator-level `FluxBC` had the wrong sign on the low side of an FVM grid and crashed with a
  per-cell `D` on a 2D grid; both are fixed, and `FluxBC` works on 2D/3D FDM grids. (R-35)
- `NewtonTrustRegion` and `LevenbergMarquardt` end `STALLED` (`converged=False`) when a step or
  cost test passes on a step the trust radius or damping held short while the model still predicts a
  decrease. `result.info["stall_reason"]` says why (`radius_collapsed`, `evaluation_noise` or
  `invalid_trials`). A cost-tolerance stop now needs a gain ratio above 0.25. (R-25, C-5)
- Fitted window starts (`Multishoot` with `Windows.uniform` or `Windows.at`) solve each window once
  per evaluation, saving its frames, instead of once per observed frame. With 110 frames in 5 windows
  the integrated time drops from 1265 to 110, the residual is 14x and the Gauss-Newton Jacobian 12.6x
  faster, with much less memory. Fitted values are unchanged up to the solver tolerance. (R-21)
- `optsolve` on a problem with fitted window starts defaults to
  `NewtonTrustRegion(hessian="gauss-newton")`. `hessian_tile` defaults to `"auto"`: the true Hessian
  is formed whole up to `dense_cutoff` parameters and 32, 16 or 8 directions at a time above, as the
  device allows; a Hessian that cannot fit the device stops the fit before its first step with a
  `ConfigurationError` naming the alternatives. (C-1, R-23)
- `optsolve(checkpoint=)` resumes `LevenbergMarquardt` and `NewtonTrustRegion` fits from their saved
  steps and refuses a directory written by a fit of other parameters. (R-20)
- A fit seeds `with_start`, `posterior_target`, `information(at=)`, `identifiability`, `profile`,
  `predict` and `OptResult.load(problem=)` when its parameter layout matches, so a rebuilt recipe or
  another solver tolerance no longer refuses it; a mapping `{handle: value}` is accepted as a start.
  (WE1-1, R-18)
- `verify_gradient` returns a verdict (`passed`, `failed`, `inconclusive`); `sample(kernel="nuts")`
  audits the gradient at a jittered chain start and refuses only a failed audit. (WE1-2, R-19)
- The identifiability check skips learnable-function (network) columns, on joint (multi-experiment)
  problems too; a fit of network weights alone records a note instead of warning and builds no
  diagnostics Jacobian; a fit with scalars beside a network differentiates only the scalar columns.
  `theta_singular_values`, `jacobian_cond` and `param_group_cond` cover the non-network columns.
  (10, U-2, R-26)
- `jax.grad` through a solve with the default forward-only adjoint raises a `ValueError` naming
  `CheckpointAdjoint` instead of JAX's while-loop message. (C-4)
- `ds.default_theta(problem, values)` validates its keys, values and shapes, takes a whole module for
  a function, and refuses a handle that owns no array; two problems equal in content no longer share
  one set of overrides; fixed parameters declared from one array, and functions declared from one
  module, are copied. (7, R-1, R-2, R-3, R-4, R-5)
- `pdesolve(theta=)` refuses a theta that is not the problem's parameter pytree, naming
  `ds.default_theta`. (R-5)
- Meshes: element volumes use `|det J|`, so a clockwise polygon, mask or hand-built mesh gives
  positive measures and the right sign on flux and Robin terms; a tangled element raises; HRZ lumping
  raises instead of skipping elements. (2)
- FEM operators read a one-dimensional coefficient by its length: `num_dofs` is nodal, `n_cells` is
  one value per cell; any other length raises. (5)
- The mixed `Precision.fp32_factor()` policy solves its float32 factors unrefined, as documented;
  it was running a hidden cuDSS refinement step on every corrector solve. (P-1)
- `stats.linear_solve_mode` reports `"fp32_factor"` for the mixed policy (float32 factors under
  float64 compute) and keeps `"fp32"` for end-to-end float32; 1.0.0 reported `"fp32"` for both. (P13)

### Additions

- `ds.compile(problem, solver, t_span, y0, ...)` compiles a `pdesolve` call without running it; the
  returned `CompiledSolve` runs on new `t_span`, `y0`, `theta` and `lambda0` of the same shapes and
  dtypes and reports `compile_seconds`, `cost_analysis()` and `memory_analysis()`. Save times must be
  `SaveAt(ts=..., ordered=True)`. (A-7)
- `theta[handle]` reads a parameter's value and `theta.replace(handle, value)` (or a mapping) writes it;
  parameters are located by key path, so a theta that went through `jax.tree.map`, a device transfer
  or a tracer still takes `ds.default_theta(problem, {handle: value})`. (R-1)
- `ds.parameter_gradient(grads, handle)` reads `dloss/dhandle` from `grads = jax.grad(loss)(theta)`,
  summing the copies of a parameter that several equations read. (P14)
- `est.values(x)[handle]` reads parameters from any vector in the fit coordinate; `ParameterSchema` is
  exported from `ds.inference`. (WE1-4)
- `recipe.spec(**inputs)` returns the `ProblemSpec` a recipe authors, so `spec.assemble_state({})`
  renders the declared `init=`. (8, R-8)
- `grid.coordinates` (per-axis cell-centre arrays) and `grid.meshgrid()` (arrays of shape `grid.shape`
  in field storage order) on both grid classes. (D-2)
- `c * q.dt(field)` on a mesh accepts one value of `c` per cell. (U-1)
- Grid transport: velocity fields in `q.ops.advect_upwind`, the conservative `q.ops.div_advect`,
  a per-axis `(Dx, Dy, Dz)` in `q.ops.div_flux`, `axis=`/`side=` on `spec.bc.dirichlet`, `flux`,
  `robin` and `neumann`, and `spec.bc.inflow(axis=, side=, velocity=, value=)` (Danckwerts with
  `div_advect`). `fn=` callbacks work on one face of a 2D/3D grid. (WE2-5, R-29)
- `spec.bc.periodic(var, axes=(0,))`: periodic on some axes, other conditions on the rest. (A-1)
- `q.ops.cell_indicator(name)`; mesh conditions by `node_indices` select the facets whose vertices
  all lie in the set. (4, R-32)
- The residual form `q.dt(...)` works on meshes and keeps the boundary conditions of the field it
  writes, on grids too; a Dirichlet condition on that field is allowed. (1, R-9, R-10)
- `TaggedMesh(cell_tags=, cell_tag_names=)`, `TaggedMesh.cell_indicator(name)`,
  `polygons_to_tagged_mesh({name: polygon}, ...)` with a conforming interface, and `read_mesh`
  region groups. (4, R-15)
- Flux and Robin conditions on QUAD4, TRI6, QUAD8, HEX8 and TET4 meshes. (R-17)
- `Field` on a mesh composes with weak terms and nodal arrays and supports `**`, `abs` and
  comparisons; a scalar joins a residual on a mesh. (R-12, R-13)
- `ds.inference.evidence(est, fit)` and `laplace(est, fit).log_evidence`. (WE1-3)
- `Measurement(observe="name")` names a `spec.observe` declaration. (C-2)
- `information`, `laplace`, `identifiability` and `predict` on a multiple-shooting fit. (R-22)
- `ProblemSpec` and the recipe handles print short reprs. (A-5)
- `meshio` joins the `[mesh]` extra. (R-16)

### Docs

- Precision modes: the mixed modes owe their accuracy to the float64 residual; `fp32_ir` is the
  recommended mixed mode and `fp32_factor` its bare variant; `fp64` for tight tolerances on
  ill-conditioned DAEs; measured speeds (P-1). `step_tol` and `cost_tol` default to `1e-8` in
  float64 (R-24). `subproblem="auto"`, `linear_solver="auto"` and `hessian_tile` as the code
  behaves (C-1, R-28). `converged` means a local stopping test passed (C-5).

### Planned for the next release

- Explicit parameters: `theta` as `{"p": {name: array}, "fn": {name: module}}`, a public view over the
  1.0.1 key-path table, with a design document first (private channel for the non-parameter arrays,
  names for family members and replicates, migration of saved fits, semver).
- A tiled Gauss-Newton Jacobian for multiple shooting and Levenberg-Marquardt; a refinement target for
  `Precision.fp32_ir`; boundary facets on TET10, HEX20 and HEX27; the WIRE coordinate network.
