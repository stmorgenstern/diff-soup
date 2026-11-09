# Changelog

## 1.2.0 (unreleased)

Complex linear solves on every exact backend, a public schedule solve, and fixes to JIDA and consistent initialization found on large battery models.

### Breaking changes

- Removed `diff_soup.problem.implicit.get_or_build_solver_view`. Use `ds.solver_view`, which returns the same cached view.
- Removed `ds.inference.constrained_optsolve` and `ds.inference.InequalityConstraints`. Declare the constraints as `ds.optimization.Constraints` on an `InferenceProblem` and solve it with `ds.optimization.InteriorPoint` or `ds.optimization.AugmentedLagrangian`.
- JIDA with algebraic rows out of the error test, its stock setting, now takes a second Newton iteration when a step's first correction exceeds weighted norm 1. Previously a stale convergence rate could accept it, leaving an algebraic state far off its constraint and its gradients wrong. Results on DAEs change at the level of the solver tolerance.
- Consistent initialization now accepts a final residual within `ic_atol + ic_rtol * S`, with `S = ‖|F_y| |y| + |F_ydot| |ydot|‖` the size of the residual's terms at the final iterate, plus the rounding bound of its longest row, after Newton has tried the absolute target. Previously a start at the rounding floor of large states failed with `INCONSISTENT_IC`. `ToleranceSpec(ic_rtol=0.0)` keeps the absolute gate, as does `ic_atol=` without `ic_rtol=`, and so does an engine built with `ic_tol=` and no `ic_rtol=`, which previously took the step's `rtol`.
- While a crossing is located, `Event` conditions now receive the solve's `theta` as `args`, as they do when the event is detected, so a condition can read and differentiate a parameter. Previously they received `None` there, and a condition that tests `args is None` now takes the other branch.
- `spec.build()` now declares only the Jacobian entries a tangent can reach, unless an equation uses `q.dt` (`exact_sparsity=None`, the new default). Pass `exact_sparsity=False` for the declared blocks with a dense border per global.
- `InteriorPoint`, `SQP` and `AugmentedLagrangian` with `hessian="gauss-newton"` on a least-squares problem now differentiate their solution by the Gauss-Newton form of the implicit rule, first order throughout: `J^T J` in place of the Lagrangian Hessian, the mixed term from the residual rows' derivative in `args`, no constraint curvature. It equals the exact rule at a zero-residual solution. Pass `hessian="exact"` or `"auto"` for the exact rule.
- A `SolveContext` now belongs to the thread that first factors in it: `factor` and `empty_factor` in it from another thread raise `FactorContractError`. Create one context per thread.

### Features

- `ds.linear.{KLU,SuperLU,CuDSS}` now factor and solve complex matrices (`complex128`, and `complex64` on `CuDSS`) on the real symbolic analysis, solve a complex right-hand side on a real factor, and differentiate both ways.
- Added `ds.solve_schedule` and `ds.schedule_arrays`, which run a `Schedule` against a model's operating table, with `max_steps` per segment and first and second derivatives with respect to segment durations. See [Events and protocols](https://stmorgenstern.github.io/diff-soup/reference/simulation/events/).
- Added a `"resistance"` row to `spec.control.operational` and `Segment(resistance=)`, which hold `voltage = target * current`.
- Added `spec.build(exact_sparsity=True)`, which declares only the Jacobian entries a tangent can reach.
- Added `ds.solver_view(problem)`, the cached view the solvers take of a problem (`ds.SolverView`), and `ds.jacobian_nnz(problem)`, the entry count of its iteration matrix. See [Problem interface](https://stmorgenstern.github.io/diff-soup/reference/modeling/problem-interface/).
- Under `jax.default_device`, `linear="auto"` and consistent initialization choose their backend by that device's platform, and native factors are kept per device, so one process can run lanes on a GPU and on CPUs. `checkpointed_map(device_groups=)` accepts groups of another platform than the default backend's.
- `KLU.native_stats` reports `live_lu_entries`.
- `SuperLU.native_stats` reports `singular`, the factor requests that ended singular, as `KLU.native_stats` does.
- Added `ds.jacobian_pattern(fn, y, *args, within=None)`, the entries of a Jacobian that can be nonzero, read from the function's tangent map without evaluating it. Rules cover loops, branches, gathers, scatters at known indices, cumulative sums, convolutions, linear solves, `jax.checkpoint`, and `jax.custom_vjp` functions, read through their `bwd`; an operation without one falls back to every entry with a warning, bounded by `within`. See [Problem interface](https://stmorgenstern.github.io/diff-soup/reference/modeling/problem-interface/).
- `ds.topology_report` adds `pullback_count`, and its `color_count` is now the solver's own: the colors of the rows the solver colors, with the rows it holds out recovered by pullback. Previously it colored the whole pattern.
- `TerminationStatus` members are now also their integer codes, so a compiled solver's `SolveRecord.status` compares with them under `jit` and `TerminationStatus(code)` reads a code back. Their string values are unchanged.
- Added `hessian_composition=` to `OptimizationProblem.from_value` and `LeastSquaresProblem.from_residual`. Under `"forward-over-forward"` the gradient, the Hessians and the Lagrangian Hessians `with_constraints` adds are all forward mode, so `InteriorPoint`, `SQP`, `AugmentedLagrangian` and their implicit derivative take an objective whose inner solve supports forward mode only, such as a DAE solve under `ForwardAdjoint`. The default, `"forward-over-reverse"`, is unchanged.
- Added `CheckpointAdjoint(reverse_over_reverse=True)`, under which a gradient through a solve can itself be differentiated in reverse mode, as in `jax.grad` of `jax.grad`, and the solve also takes `jax.jvp`. Forward mode over the gradient then costs about twice as much, so the default, `False`, keeps the previous rule.
- Added `ds.optimization.NestedLeastSquares` and `ds.optimization.InnerSteps`: shared values fitted over a batch of inner least-squares fits as one `LeastSquaresProblem`, whose Jacobian is the projected one from the inner fits' Gauss-Newton system, its members run warm to a tolerance tied to the outer step, in `lanes` over a `Layout`. `NewtonTrustRegion` carries the members' state from step to step (the `"carried_args"` structure) and returns it in `result.info["args"]`.
- `run(max_steps=, tol=)` of the compiled solvers and `ds.optimization.lanes(max_steps=, tol=, warm_start=)` take the budget and the tolerance as runtime values, one per lane or one for all, so a new value no longer compiles the loop again.
- Added `adjoint=` to `InteriorPoint`, `SQP` and `AugmentedLagrangian`. `ds.nonlinear.ImplicitAdjoint()` differentiates the solution in `args` by one reverse rule, a transposed KKT solve, so `jax.grad` costs one solve however large `args` is and runs through closures that differentiate in reverse mode only. The default, `ds.ForwardAdjoint()`, keeps the forward rule.
- `InteriorPoint`, `SQP` and `AugmentedLagrangian` run in float32 and return float32 records. A tolerance left unset follows the iterate's dtype: `max(default, 100 * eps)`, which is the float64 default unchanged and `1.2e-5` in float32.

### Bugfixes

- Fixed consistent initialization of a family problem whose step backend is a `MemberBlock` over the full state, which raised a `ConfigurationError`. Initialization now solves its smaller system on the backend `linear="auto"` chooses for its size.
- Fixed `SaveAt(output="projected")` rows that came back `NaN` at a tight `atol`. Projected samples now pass the same gate as the solve's own initialization.
- Fixed JIDA solves that stalled at a tight `atol` after a short step, where Newton's corrections are pure rounding. A correction within twice the rounding floor on a fresh matrix is now accepted as the root.
- Fixed the derivative of consistent initialization with respect to the start time.
- Fixed `SolveContext.resident_bytes` on `KLU`, which read uninitialized memory for any factor with a singleton block.
- Fixed a MAP fit of a joint problem, which read every shared parameter's prior from one handle. Each occurrence now takes its prior from the declarations of the experiments behind it, and a MAP refuses an occurrence whose schema records none.
- A `KLU`, `SuperLU` or `CuDSS` factor lowered into a program over several devices of one process now raises `FactorContractError`. Previously XLA failed with an internal error, or, in a `shard_map` body, the devices shared one native factor.
- Complex values on an analysis declared `spd=True`, and on a `CuDSS` analysis with `downcast_factors` or `emulation_mantissa_bits`, now raise `FactorContractError`. Previously cuDSS factored them as a real symmetric positive definite matrix, or ignored the option.
- Fixed `SuperLU` on more than one thread, which returned a wrong solution for a few factors in a hundred with a success status. Its factors on any number of threads now equal one thread's, bit for bit; the build patches SuperLU_MT 4.0.2 (`diff_soup/_internal/superlu_ffi/patches/`).
- Fixed `CuDSS` solves that could apply another factor and report success: two executions of one compiled program at once on one `SolveContext`, or two calls of one jitted function inside one program, which share a factor slot. A solve now always applies its own handle's factor.
- A released `Analysis` of `KLU`, `SuperLU` or `CuDSS` now frees its native call sites and, once its factors' contexts are released, its native analysis. Previously the sites kept every analysis for the life of the process.
- Fixed a `SuperLU` solve with real values of a shared storage that a refactor with complex values filled earlier in the same program, which returned `nan` with a success status. It now factors the real values first, as `KLU` does.

### Performance

- A vmapped solve keeps fewer copies of its saved output (two state rows per save time, from five), and `ds.topology_report` no longer builds an n-by-n adjacency for a problem with a full row. An LFP half cell with 206,518 states and 1,001 saved times now completes on a 12 GB GPU.
- JIDA does its whole-array work only on the path an attempt takes, with identical results.
- Outside `vmap`, the branches of an implicit step (JIDA's history update, the Jacobian refreshes, the retries) compile to one conditional and are traced once. Under `vmap` a branch still runs only when some lane takes it, with one compiled copy of each branch.
- Consistent initialization's derivative skips the start-time term when the start time is constant, and its relative gate evaluates the residual's scale only where the absolute target was missed, on the algebraic rows alone on the reduced route.
- `spec.build(exact_sparsity=True)` reads its pattern with one compiled program instead of one per operation, 0.85 s instead of 3.6 s at 18 families, and neither build colors the declared pattern unless `physics_jacobian_blocks` is called. Loops, cumulative sums, convolutions and linear solves in a residual keep their pattern instead of filling it.
- Member rows that read every site of their member are recovered by shared pullbacks instead of one color per site: an LFP half cell with 50 volumes of 10 particles needs 8 colors instead of 108.

### Other

- Three undocumented module paths moved: `diff_soup.timestepping.consistent_ic` and `diff_soup.problem.consistent_init` are now `diff_soup.problem.implicit.consistent_init`, and `diff_soup.timestepping.vsvo.common.weights` is now `diff_soup.utils.weights`.
- `ds.optimization.inference.approx.FluxNet` raises `AttributeError` like any other missing name. It raised a `TypeError` naming `eqx.nn.MLP` since its removal.
- `CuDSS`'s `contract_stats` no longer reports `perturbed_pivot_factorizations`, which was always 0: every cuDSS factor runs with pivot perturbation off.
- The cuDSS extension's default CUDA architectures add `87-real` (Jetson Orin), and the installation page says how to build the KLU extension on ARM (`KLU_VENDOR_DIR`).
- A diff-soup environment variable set to an empty string now counts as unset and takes its default; the variables and the changes `import diff_soup` makes are listed on the installation page. Previously an empty `DIFFSOUP_PROCESSES` or `DIFFSOUP_NODES` raised `ValueError`.
- `diff_soup.inference` is now `diff_soup.optimization.inference`, and `optsolve`, `OptResult`, the optimization problems and the solvers are also at the top of `ds.optimization`. `ds.inference` and every `diff_soup.inference` import path still work for this release, return the same objects and warn once with a `DeprecationWarning`; they will be removed in 1.3.

**Full Changelog**: https://github.com/stmorgenstern/diff-soup/compare/v1.1.0...v1.2.0

## 1.1.1 (unreleased)

A bugfix release: each build of a recipe template is now fitted under its own priors.

### Bugfixes

- Fixed MAP fits of a `@ds.recipe` template, which took every prior from the template's first build in the process, so a later build with another prior was fitted under the first one's without a word. Each fit now takes each prior from its own experiments' builds (per owner for a parameter that varies by owner), and a `GaussianCoefficientPrior` centers on its own build's coefficients. Results change only where one process built a template with different priors.
- Builds that differ only in a prior now return two problems. Previously they returned one problem, whose declarations followed the latest build.
- A MAP fit now refuses a parameter shared by experiments whose builds declare different priors. Previously it used the first build's prior.

**Full Changelog**: https://github.com/stmorgenstern/diff-soup/compare/v1.1.0...v1.1.1

## 1.1.0 (2026-10-29)

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
- Grid transport: `q.ops.advect_upwind` takes velocity fields, `q.ops.div_advect` is the conservative form, with flux-limited face values by default (`scheme="tvd"`, `limiter="van_leer"`, `"koren"` or `"minmod"`: second order where smooth, no new extrema at any cell Péclet number), third-order `"weno3"`, `"central"` or first-order `"upwind"`, and `q.ops.div_flux` takes a per-axis `(Dx, Dy, Dz)`. `spec.bc.{dirichlet,flux,robin,neumann}` take `axis=` and `side=`, and the new `spec.bc.inflow` gives the Danckwerts condition with `div_advect`.
- `spec.bc.periodic(var, axes=(0,))` makes a field periodic on some axes and allows other conditions on the rest.
- Mesh regions: `TaggedMesh(cell_tags=, cell_tag_names=)`, `q.ops.cell_indicator(name)`, `polygons_to_tagged_mesh({name: polygon}, ...)` with conforming interfaces and per-region edge tags (`segment_tags=`, `segment_names=`), and `read_mesh` physical groups as cell and boundary tags.
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
