# Changelog

## 1.1.0 (unreleased)

New [tutorials](https://stmorgenstern.github.io/diff-soup/tutorials/) and [worked examples](https://stmorgenstern.github.io/diff-soup/examples/), and the fixes and features found and added while writing them.

### Breaking changes

- `spec.observe.{spatial_mean,boundary_mean}` now average. `fn` returns one value per node, weighted by cell volume or facet measure. Previously the name was only a label. A scalar-valued `fn` is now refused.

  ```python
  # Before (1.0.0): fn computed the mean
  spec.observe.spatial_mean(name="T_avg", fn=lambda q: jnp.mean(q.T))
  # After: fn returns nodal values
  spec.observe.spatial_mean(name="T_avg", fn=lambda q: q.T)
  ```

- `spec.bc.neumann` now imposes `du/dn = s` on FVM, FDM and FEM. Previously it was an alias of `spec.bc.flux`, which imposes `D du/dn = s`. `name=` and `cross_facet=` are now on `flux` only.

  ```python
  # Before (1.0.0): neumann was an alias of flux
  spec.bc.neumann("u", on="left", value=s)      # imposed D du/dn = s
  # After: say which one you mean
  spec.bc.flux("u", on="left", value=s)         # D du/dn = s
  spec.bc.neumann("u", on="left", value=s)      # du/dn = s
  ```

- Grid boundary conditions and boundary integrals in 2D and 3D now weight each selected cell by one face. Previously edge and corner cells were counted more than once. A selection spanning several faces now raises. Pick one with `axis=` and `side=`.
- Fixed the sign of flux and Robin terms on clockwise meshes, which includes every `mask_to_mesh` mesh. Element volumes now use `|det J|`. A tangled element now raises.
- FEM operators now read a 1D coefficient by its length: `num_dofs` values are nodal and `n_cells` values are per cell. Previously a per-cell array was read as nodal.
- The residual form `q.dt(...)` now works on meshes (previously its coefficients were dropped), and keeps the boundary conditions of the field it writes on grids and meshes alike.
- `NewtonTrustRegion` and `LevenbergMarquardt` now end `STALLED` (`converged=False`) when the trust radius or damping holds a step short of a stopping test. Previously these fits reported converged. `result.info["stall_reason"]` says why.
- `NewtonTrustRegion(hessian=)` now defaults to `"auto"`, which picks Gauss-Newton on any problem with residual rows, including every `InferenceProblem`. `optsolve` and `constrained_optsolve` follow. Previously the default was the true Hessian. Pass `hessian="true"` to get it back.
- `NewtonTrustRegion(hessian_tile=)` now defaults to `"auto"`, and a true Hessian or Gauss-Newton Jacobian too large for the device stops the fit before its first step with a `ConfigurationError`. Previously the default was `"full"`.
- `sample(kernel="nuts")` now adapts all chains together (`adapt="pooled"`, `warmup=200`), and each chain keeps its adapted state. Previously one chain adapted for 500 steps and the others started from a perturbed fit. `chains=1, adapt="window", warmup=500` reproduces the old draws.
- Identifiability checks now skip network columns, including on multi-experiment problems, so a network-only fit no longer warns. `theta_singular_values`, `jacobian_cond` and `param_group_cond` now cover the non-network columns only.
- `stats.linear_solve_mode` now reports `"fp32_factor"` for `Precision.fp32_factor()`. Previously it reported `"fp32"`, which now means end-to-end float32 only.

### Features

- Added `ds.compile`, which compiles a `pdesolve` call without running it. The result runs on new `t_span`, `y0` and `theta` of the same shapes and dtypes, and reports `compile_seconds`, `cost_analysis()` and `memory_analysis()`. See [Running a simulation](https://stmorgenstern.github.io/diff-soup/overview/simulation/running-a-simulation/).

  ```python
  compiled = ds.compile(problem, solver, (0.0, 0.1), y0)
  sol = compiled((0.0, 0.2), y0)
  ```

- `theta[handle]` reads a parameter, and `theta.replace(handle, value)` writes every copy of it. A theta that went through `jax.tree.map`, a device transfer or a tracer still works with `ds.default_theta`. See [Learnable parameters](https://stmorgenstern.github.io/diff-soup/overview/modeling/parameters/).
- Added `ds.parameter_gradient(grads, handle)`, which reads `dloss/dparam` from `jax.grad(loss)(theta)` and sums the copies of a parameter that several equations read.
- Added `est.values(x)[handle]`, which reads parameters from a vector in the fit coordinate. `ParameterSchema` is exported from `ds.inference`.
- Added `recipe.spec(**inputs)`, which returns the `ProblemSpec` a recipe authors, and `grid.{coordinates,meshgrid()}` on both grid classes.
- Grid transport: `q.ops.advect_upwind` takes velocity fields, `q.ops.div_advect` is the conservative form with central face fluxes by default (second order, no numerical dispersion) or `scheme="upwind"`, and `q.ops.div_flux` takes a per-axis `(Dx, Dy, Dz)`. `spec.bc.{dirichlet,flux,robin,neumann}` take `axis=` and `side=`, and the new `spec.bc.inflow` gives the Danckwerts condition with `div_advect`.
- `spec.bc.periodic(var, axes=(0,))` makes a field periodic on some axes and allows other conditions on the rest.
- Mesh regions: `TaggedMesh(cell_tags=, cell_tag_names=)`, `q.ops.cell_indicator(name)`, `polygons_to_tagged_mesh({name: polygon}, ...)` with conforming interfaces, and `read_mesh` physical groups as cell and boundary tags.
- Mesh boundary conditions: flux, Robin and Neumann conditions, `surface_measure` and `q.ops.boundary_integral` on QUAD4, TRI6, QUAD8, HEX8, TET4, TET10, HEX20 and HEX27. A condition by `node_indices` selects the facets whose vertices all lie in the set.
- On a mesh, `Field` composes with weak terms and nodal arrays and supports `**`, `abs` and comparisons. `c * q.dt(field)` takes one value of `c` per cell.
- Added `ds.inference.approx.WIRE`, a coordinate network with the Gabor-wavelet activation of Saragadam et al. (2023) and the interface of `SIREN`. See [Function approximation](https://stmorgenstern.github.io/diff-soup/overview/inference/function-approximation/).
- The Gauss-Newton Jacobian is now tiled: `NewtonTrustRegion(hessian_tile=)` applies to it, and `LevenbergMarquardt(jacobian_tile=)` is new. Previously an integer `hessian_tile` was ignored under Gauss-Newton. See [Solvers](https://stmorgenstern.github.io/diff-soup/overview/inference/solvers/).
- NUTS options: `sample` takes `adapt=`, `mass=` (`"diag"` or `"dense"`), `step_size=` and `max_tree_depth=`, and `draws.info["adaptation"]` records what ran. See [MCMC](https://stmorgenstern.github.io/diff-soup/overview/inference/mcmc/).
- Added `ds.inference.evidence(est, fit)` and `laplace(est, fit).log_evidence`, the Laplace evidence of a MAP fit. Its stationarity check scales with the solver tolerance and precision of the problem, it accepts float32 posteriors, and a refusal names the solver tolerance. `information`, `laplace`, `identifiability` and `predict` now take a multiple-shooting fit.
- `Measurement(observe="name")` reads a `spec.observe` declaration.
- A fit now seeds `with_start`, `posterior_target`, `information(at=)`, `identifiability`, `profile`, `predict` and `OptResult.load(problem=)` whenever its parameter layout matches. A rebuilt recipe or another solver tolerance no longer refuses it, and a start can be a mapping `{handle: value}`.
- `verify_gradient` now returns a `verdict`: `"passed"`, `"failed"` or `"inconclusive"`.
- `ds.default_theta` takes a whole module for a function parameter.

### Bugfixes

- `ds.default_theta` overrides no longer leak between two problems equal in content. Fixed parameters declared from one array, and functions declared from one module, are now copied.
- `optsolve(checkpoint=)` now resumes `LevenbergMarquardt` and `NewtonTrustRegion` fits from their saved steps. It refuses a directory written by a fit of other parameters.
- `sample(kernel="nuts")` now audits the gradient at a jittered chain start instead of the MAP, where the audit meant nothing. It refuses only a failed audit.
- `posterior_target(gradient="reverse")`, `log_posterior` and `LogDensity.from_callable` now scale their reverse/forward check with the solver `rtol` and run it one percent away from the start, so a float32 fit passes at its mode. `jax.jit` and `eqx.filter_jit` now compile a `LogDensity`.
- Fixed `ExperimentSet.log_posterior`, which raised on every call.
- Meshes: a scalar source term in the residual form now builds, a first use under `jax.jit` no longer leaves a tracer in the cache, and `read_mesh` no longer reports TET10 meshes as tangled.
- Fixed the operator-level `FluxBC`: the wrong sign on the low side of an FVM grid, and a crash with a per-cell `D` on a 2D grid. It now also works on 2D and 3D FDM grids.
- Importing JAX before diff-soup no longer turns off constant hoisting, which made closures over large arrays compile to large programs.

### Performance

- Fitted window starts now solve each window once per evaluation instead of once per observed frame. With 110 frames in 5 windows the integrated time drops from 1265 to 110. Fitted values are unchanged up to the solver tolerance.
- With fitted window starts, the Gauss-Newton Jacobian now pushes each direction through all windows at once. On an 803-variable FitzHugh-Nagumo fit one Jacobian takes 7.2 s instead of 31.0 s on CPU, 0.78 s instead of 3.30 s on an RTX 4070 Ti, and 0.33 GiB of compiled memory instead of 1.26 GiB.
- `Precision.fp32_factor()` now solves its float32 factors unrefined, as documented. On Gray-Scott 256x256 at `rtol=1e-6` on an RTX 4070 Ti, a solve drops from 5.10 s to 3.06 s with the same steps and errors.
- Fit diagnostics now differentiate only the non-network columns, which removes most of their cost on network fits.

### Other

- Better errors: `jax.grad` through a solve with the default forward-only adjoint raises a `ValueError` naming `CheckpointAdjoint`. `pdesolve(theta=)` refuses a theta that is not the problem's parameter pytree, and `ds.default_theta` checks keys, values and shapes.
- An end-to-end `Precision.fp32()` solve that cannot initialize, as on some stiff DAEs where float32 rounding alone exceeds the tolerance, now says so and names `Precision.fp32_ir()` and `Precision.fp64()`.
- Warnings now point at your own line. A float32 solve with `jax_enable_x64` off no longer prints JAX's truncation warnings from inside diff-soup.
- Clearer output: results print as `diff_soup.RESULT<...>`, `ProblemSpec` and the recipe handles print short reprs, and the event-stop and window-rank messages use plain words.
- diff-soup now requires `blackjax>=1.7`, and `meshio` joins the `[mesh]` extra.
- Docs: [Precision](https://stmorgenstern.github.io/diff-soup/overview/simulation/precision/) explains the mixed modes, recommends `fp32_ir`, gives the float32 initialization floor and says that end-to-end float32 fits on a GPU are not bitwise reproducible from run to run. [Boundary conditions](https://stmorgenstern.github.io/diff-soup/overview/modeling/boundary-conditions/) states the order of a per-cell value on a grid face. The docs now state the float64 `step_tol` and `cost_tol` defaults (`1e-8`), what `converged` means, and when to run small problems on the CPU.

**Full Changelog**: https://github.com/stmorgenstern/diff-soup/compare/v1.0.0...v1.1.0
