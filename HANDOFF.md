# NF1 hand-off: the nested least-squares problem and the Gauss-Newton form of the implicit rule

Clone `/drive2/research2/diff_soup_wave2/nf-nested`, branch `opt/nf-nested` cut from `batjax/stack-1.1` 627d147.
Brief: `/drive2/research2/RemainingThesisWork/batjax/docs/briefs/nested/nf1-ds-nested.md`; design page
`/drive2/research2/RemainingThesisWork/batjax/docs/2026-11-08-nested-fit-design.md` (N1-N3).
Native modules copied from `rf-store/rf-linear/final` (klu, superlu, cudss cu12) and `review-final` (cudss cu13,
`klu_ffi/lib`); untracked, excluded in `.git/info/exclude`. Throwaway checks live in `.ports/` (excluded).

Run everything as
`slot.sh env PYTHONPATH=/drive2/research2/diff_soup_wave2/nf-nested JAX_PLATFORMS=cpu /drive2/research2/diff_soup_umbrella/envs/ds111/bin/python <script>`
with `slot.sh` = `/drive2/research2/RemainingThesisWork/batjax-wt/slot.sh`.

## Milestones

| # | what | commit | check |
|---|---|---|---|
| 1 | Gauss-Newton form of the implicit rule (N1) | 925c9ff | `.ports/m1_rule.py` |
| 2 | runtime step budget and KKT tolerance | d1a257e | `.ports/m2_limits.py` |
| 3 | `NestedLeastSquares`, `InnerSteps`, the trust region's carried args | 84299ee, dcf251a | `.ports/m3_toy.py` |
| 4 | the synthetic 39x201 cost table (recorded) | see below | `.ports/m4_synth.py <route>` |
| 5 | tests, reference page, changelog, ruff; custom rules moved inside the member map | see below | the suite files below |

### 1. The Gauss-Newton form (N1)

- `Closures` (private) gains `residual`, `residual_and_jacobian` (the least-squares rows with the regularization's
  rows appended), `gauss_newton_rule` and `keep_jacobian`. `closures_of` fills the two closures from any problem
  with the `"residual_jacobian"` structure.
- The static switch: `ConstrainedSolver.run` hands the implicit derivative `_raw_gauss_newton` instead of `_raw`
  when `hessian == "gauss-newton"`; it sets `gauss_newton_rule` when the problem has residual rows. `"auto"` (GN plus
  BFGS, which learns the rest of the curvature), `"exact"` and `"bfgs"` keep the exact form, as does a problem
  without residual rows (a reduced-statistics lowering). No new public class.
- `system_at(..., jacobian=None)`: with `J_r` the primal block is `J_r^T J_r`; bound, barrier and degenerate handling
  unchanged. `kkt_tangent`/`kkt_cotangent` in the GN form: one forward (reverse) derivative of
  `rows_of = args -> (r, c_E, c_I)`; `d grad_x L = J_r^T d_r`, `d f = r . d_r`; constraint curvature (in `x` and the
  mixed term) dropped; the cotangent's `r` cotangent is `v_bar r + J_r d_grad_bar`. `J_r` given (kept) or
  recomputed at `x`.
- New private maps for the nested problem: `rows_tangent` (`d_r + J_r dx*`, the projected Jacobian applied to a
  tangent) and `rows_cotangent` (its transpose: one transposed KKT solve and one vjp of the rows in `args`).
- `Evaluation.jacobian` (default `None`): closures with `keep_jacobian` take the GN statistics from `J_r` and keep
  it (scaled by `sqrt(objective scale)` in the scaled problem). Off everywhere unless the nested problem asks.

Gate (rule equality, 1e-10 relative): zero-residual toy, 3 own values, 2 args, nonlinear rows zero at the solution
plus two linear rows that hold `z0` on its lower bound (multiplier 0.3) and `z1` on an inequality whose limit reads
`args` (multiplier 0.46), nondegenerate; `InteriorPoint(tol=1e-12)` with `hessian="gauss-newton"` against
`hessian="exact"`, `jacfwd` of the point, multipliers, slacks and value, and `jax.grad` of a scalar under
`ImplicitAdjoint`, jitted and in `ds.optimization.lanes` (three lanes):
jit tangent 8.1e-13, jit cotangent 5.5e-13, lanes tangent 8.4e-13, lanes cotangent 4.6e-13; exit 0.
At a nonzero residual the forms differ (5.0e-2 relative, `.ports/m1_live.py`), so the switch is live.

Affected suite files at 925c9ff (`test_compiled_solvers`, `test_optimization_solvers`, `test_optimization_lanes`,
`test_compile_cache`, `test_hierarchy`, `test_control`): 84 passed, 2 skipped (cuDSS owed), exit 0.

### 2. A runtime step budget and tolerance

- `CoreState` gains `budget` and `tol`, runtime leaves of the loop; `terminate` reads `state.budget`, and the three
  solvers' `_converged` read `state.tol` (the augmented Lagrangian's `feasibility_tol` and the barrier floor stay
  static). `Start` carries `limits`; `_init(..., limits=(max_steps, tol))` defaults each `None` from the static field
  (so a dtype-resolved `tol`, `opt/rf-tol`, feeds it unchanged). The QPs of SQP start from their own static fields.
- `run(..., max_steps=None, warm_start=None, tol=None)`: both are runtime values (arrays) handed through
  `implicit_run(..., limits)` to `_run_primal`; the implicit entries take `limits` as a gradient-dead argument. A
  traced or batched budget or tolerance needs no new compile. Previously `max_steps` replaced the static field.
- `lanes(..., max_steps=, tol=, warm_start=)`: one per member or one for all, runtime lanes of the program; the
  mapping is `map_lanes(fn, batch, mesh)` (shared with the nested problem), `placement(layout)`, `per_lane`.

Gate (`.ports/m2_limits.py`, exit 0): `jit(run)` with the budget and tolerance as arguments, budgets 2, 5, 200:
one trace, cache size 1, steps 2, 5, 9 (status MAX_STEPS, MAX_STEPS, KKT_TOL at 6.8e-14). At `tol=1e-3` the run stops
after 4 steps with KKT error 3.4e-4 (status KKT_TOL), against 9 steps and 6.8e-14 at 1e-12; `|x_loose - x_tight|`
4.9e-5. `lanes` with per-member budgets `[2, 5, 200]` and tolerances `[1e-12, 1e-12, 1e-3]` in one program:
steps `[2, 5, 4]`, the third member's KKT error 1.7e-4.
Affected suite files (the six above plus `test_implicit_layer`): 91 passed, 2 skipped, exit 0.

Merge note for batch B (`opt/rf-tol` 9bd0d2b): it rewrites `ConstrainedSolver.run`'s first line
(`solver = self._for_dtype(...)`), as this branch rewrites `run`'s body. The resolution keeps both:
`solver = self._for_dtype(iterate_dtype(x0))`, then this branch's `limits = ...` and
`return implicit_run(solver._run_primal, self._rule, self._kkt, problem, x0, args, warm_start, self.adjoint, limits)`
(no `dataclasses.replace` for `max_steps`: it is a runtime limit now). `_init` defaults `tol` from `self.tol`, which
is the resolved solver there.

### 3. `NestedLeastSquares` (N3)

Public: `ds.optimization.NestedLeastSquares.pose(problem, solver, members, x0, w0, *, layout=None, steps=None,
adjoint=None, keep_jacobian=True)`, `nested.problem`, `nested.solve(w, args=None) -> (x, record)`, and
`ds.optimization.InnerSteps(tol_min=None, kappa=0.1, max_steps=None, noise_floor=0.0)` (`diff_soup/optimization/nested.py`).
Names settled: the inner problem is the first positional `problem` and the adaptive rule is `steps=` (the design
page's sketch used `inner` for both). `members` is a mapping without `"shared"`; member m's inner `args` are
`{"shared": w, **member_m}`.

- `nested.problem` is a `LeastSquaresProblem` over `w`, `reduced_gauss_newton=True`, structure `value_gradient`,
  `residual_jacobian`, `gauss_newton` and the new `carried_args`. Its `args` hold the member state:
  `{"inner": the inner problem's arrays, "members", "x": (count, q) own values, "record": the last SolveRecords
  (None before the first evaluation, so the first fit is a cold start: one extra compile), "floor": per-member noise
  floors}`.
- Per member and evaluation (vmap, or shard_map over a `"data"` mesh through `lanes.map_lanes`): the solver's own
  loop (`_prepare`, `_init` with runtime `(max_steps, tol)`, `_advance`, `_record`) warm from the state, at least one
  step, a `COST_TOL` stop when a step changes the cost by at most the noise floor; then `rows_tangent` over the `p`
  basis directions of `"shared"`: one GN KKT factor, `p` solves, `p` forward tangents of the rows; rows and
  projected Jacobian `(m, p)`; statistics `sum J~^T r`, `sum J~^T J~`, cost. `J_z` comes from the last evaluation of
  the inner loop when `keep_jacobian` and the solver steps on GN (closures `keep_jacobian`, `Evaluation.jacobian`),
  else `residual_and_jacobian` again at the fit.
- NaN rows (and a `failed` count) only for a member whose status is INFEASIBLE, SOLVER_FAILED or DIVERGED, whose rows
  or projection are not finite, or whose GN system did not solve. MAX_STEPS, COST_TOL and RESTORATION_FAILED keep
  their rows.
- The hook, settled: a structure key `"carried_args"` with the closure
  `carried_gauss_newton(x, radius, args) -> ((cost, g, H, failed), args_after)`. `NewtonTrustRegion` (lanes GN path)
  evaluates its first point with `radius_init` and every trial with the trial step's scaled length, replaces its
  carried args with `args_after` when the trial is accepted (and at the start), and returns them in
  `result.info["args"]`. Nothing lives in Python state; other optimizers use the plain closures, which run the
  members to `tol_min` from the posed state (variable projection).
- Inner tolerance: `max(tol_min, kappa * radius**2)`. The design page's `kappa * radius` was tried first and lost on
  the toy: an active-bound member solved by the interior point to a loose tolerance carries a barrier cost error of
  the order of that tolerance (`mu`), larger than the step's predicted decrease (order `radius**2`) once steps are
  short, so trials were rejected with rho = -20 and the fit crawled (154 outer steps, `|w - w_joint|` 6e-8 against
  12 steps and 2.8e-9 with the square). Recorded for the coordinator: a deviation in form, not in the ruling (N2's
  "tied to the outer trust radius").
- `residual(w, args)` (and `value`) run the members to `tol_min` with a custom derivative: `ForwardAdjoint` a
  `custom_jvp` (one KKT solve per member and tangent, `jax.grad` transposes it), `ImplicitAdjoint` a `custom_vjp`
  (`rows_cotangent`: one transposed solve and one vjp of the rows per member, whatever `p`). `args` is gradient-dead.

Gates (`.ports/m3_toy.py`, exit 0 after the `radius**2` change): three members, two own values (`z1 >= 0`, active for
the third member, multiplier 2.38), two shared values, 25 rows each; `InteriorPoint(hessian="gauss-newton",
tol=1e-12)` inside, `NewtonTrustRegion(hessian="gauss-newton", grad_tol=1e-14, cost_tol=1e-15, step_tol=1e-15)`
outside, then `nested.solve(fit.params, fit.info["args"])`; joint fit: one interior point solve of all 8 values at
tol 1e-13 (KKT 2.1e-14).
- nested equals joint: `|w - w_joint|` 2.79e-9, `|z - z_joint|` 9.9e-10, 12 outer steps, all accepted (gate 1e-8).
- outer gradient at w = (0.9, 1.9) against central differences (h = 1e-5): forward (`jax.grad` through the
  custom_jvp) 9.0e-10, reverse (custom_vjp) 9.0e-10, the statistics' `J^T r` 9.0e-10 relative (gate 1e-6);
  `jacfwd(residual)` equals the statistics' projected Jacobian exactly.
- recorded, not gated: one inner step per outer step (`kappa=1e6`, the joint-step regime) reaches `|w - w_joint|`
  3.9e-8 in 139 outer steps (88 accepted): the interior point's barrier on the active bound perturbs the cost at
  one step per evaluation.
Affected suite files (compiled solvers, optimization solvers, lanes, fits, structural fitting, optimizer stops,
lowered study, architecture, hierarchy): 157 passed, exit 0.

### 4. The synthetic 39 x 201 problem (recorded, not gated)

`.ports/m4_synth.py <route> [max_outer]`, one route per process on the slot queue (`SLOT_MEM=10G`/`12G`); rows in
`.ports/m4_table.jsonl`. 39 members, 201 own values each (a map on a grid of 201 points: a sine background plus a
Gaussian ridge whose place and height vary by member, read by linear interpolation at 2000 observation points),
8 shared values in every row through `w0 + z s + w1 tanh(w2 s + w3 z) + w4 s^2 + w7 sin(w5 u + w6 z)`, whitened by
0.01, plus 199 second-difference rows: 2199 rows per member, no DAE. Start: own values 0, shared values off by
5-20 %. Outer `NewtonTrustRegion(hessian="gauss-newton")`, inner `InteriorPoint(hessian="gauss-newton")` (tol 1e-8).
Model passes per member per evaluation are counted from the code (q = 201, p = 8, k = inner steps): an inner fit
evaluates its start and every step's end with `1 + q` passes, and the projection adds `1 + p`.

| route | outer steps (accepted) | evaluations | inner steps per member per evaluation, mean (max) | model passes per member per evaluation | wall incl. compile | peak RSS | cost at the end |
|---|---|---|---|---|---|---|---|
| nested, `J_z` kept (default `InnerSteps`) | 12 (12) | 13 | 1.23 (2) | (k+1)(1+q) + 1 + p, about 460 | 6.6 s | 1.74 GB | 35222.703023 |
| nested, `J_z` recomputed (`keep_jacobian=False`) | 12 (12) | 13 | 1.23 (2) | that plus 1 + q, about 660 | 6.1 s | 1.27 GB | 35222.703023 |
| joint: one inner step per outer step (`InnerSteps(max_steps=1)`) | 12 (12) | 13 | 1.00 (1) | 2(1+q) + 1 + p = 413 | 6.3 s | 1.74 GB | 35222.703023 |
| exact-Hessian route: full inner fits from the start each evaluation (`InteriorPoint()`, `"auto"`), `jacfwd` through the exact rule, as `population_fit` did | 12 (12) | 13 | 6.33 (10) | (k+1)(1+q) + q forward-over-reverse Hessian passes + p(1+q) second-order mixed passes, about 3700 | 13.8 s | 2.47 GB | 35222.703023 |

All four end at the same cost (to 1e-11 relative) and the same shared values to 3e-8 (COST_TOL). The whole table
runs in about 33 s of wall plus process starts, well under the 15-minute budget. Headline: the nested problem costs
one joint attempt per outer step (1.2 inner steps per member), takes as many outer steps as the exact route, and
costs about 8 times fewer model passes per outer step. The kept `J_z` saves `1 + q` passes per member and
evaluation, and costs 0.47 GB of resident memory here. On this cheap model the wall clock is dominated by compiling,
so the walls do not show the per-step ratio.
Not run: a dense joint fit (85,761 x 7,847 Jacobian, 5.4 GB). The joint row is the design's equivalence: the
nested problem at one inner step. A first exact-route attempt that took the exact Hessian forward over forward
(`hessian=False` inner problem) was killed at the 10 GB cap (exit 137).

### 5. Tests, reference page, changelog

- `tests/contracts/test_compiled_solvers.py`: `test_the_gauss_newton_form_equals_the_exact_form_at_a_zero_residual_solution`
  (jit and lanes, tangents and cotangents, bound 1e-10, measured 8.4e-13) and
  `test_a_runtime_budget_and_tolerance_run_without_a_new_compile` (one trace across budgets 2, 5, 200; tol 1e-3
  stops sooner, within it and above 1e-12).
- `tests/public_api/test_nested_fit.py` (new verb group, its own manifest row): `test_a_nested_fit_finds_the_joint_fits_shared_values`
  (bound 1e-8, measured 2.8e-9 after 20 outer steps with the cost and step tests off) and
  `test_the_outer_gradient_matches_central_differences[forward|reverse]` (bound 1e-6, measured 9.0e-10).
- Ceilings set to the suite's actual size: files 51 -> 52, functions 461 -> 465, collected 785 -> 790, lines
  16,773 -> 16,970; `public_api` 21/202/323/6,569 -> 22/204/326/6,673; `contracts` 22/199/388/6,662 ->
  22/201/390/6,755. `tests/suite_policy.py` passes.
- `docs/reference/optimization/solvers.md`: the Gauss-Newton form under "Derivatives", the runtime budget and tolerance
  under "Batches of problems", a "Nested fits" section (`NestedLeastSquares` with `pose` and `solve`, `InnerSteps`).
  `mkdocs build --strict` (rel-docs env) clean.
- `CHANGELOG.md`: one breaking-change line (the Gauss-Newton form under `hessian="gauss-newton"`), two feature lines
  (`NestedLeastSquares`/`InnerSteps` with the trust region's carried args; runtime `max_steps`/`tol` in `run` and
  `lanes`).
- The custom derivative of the rows now sits per member inside the member map (as `lanes` holds `run`'s), and `w`,
  the tolerance and tangents reach the mapped function as arguments.
- `ruff check` and `ruff format --check` clean on `diff_soup` and `tests`.
- Suite files run on the slot queue (CPU): `test_nested_fit.py` 3 passed; `test_compiled_solvers.py`,
  `test_optimization_solvers.py`, `test_optimization_lanes.py`, `test_surface.py` with it: 79 passed and the nested
  file's first version failed once on a cost stop (fixed above, then 3 passed); earlier in this branch the trust-region
  files (`test_fits`, `test_structural_fitting`, `test_optimizer_stops`, `test_lowered_study`, `test_architecture`,
  `test_hierarchy`): 157 passed.

## Not verified

- **Reverse mode over a `"data"` mesh crashes XLA's CPU compiler** (segfault in `backend_compile_and_load`) for
  `jax.grad` of `nested.problem.value` under a `Layout`, in both rules, on two forced host devices. On the same mesh
  the statistics (`gauss_newton`) and `jacfwd(residual)` run and equal one device's; on one device every route works.
  Repro: `XLA_FLAGS=--xla_force_host_platform_device_count=2 ... .ports/m3_layout.py mesh` (`none` passes). The trust
  region's path (statistics only) is unaffected. Not chased further.
- The full CPU suite (`--run-integration --run-validation`) was not run from this clone; only the affected files.
- No GPU run (cuDSS rows owed as before); the nested problem never ran on a GPU.
- The noise-floor stop (`InnerSteps.noise_floor > 0`) and the per-member floor have no test or gate.
- The cost table's walls include compiling; per-step walls on a cheap model would need a longer run.
- `opt/rf-tol` merge (batch B): textual conflicts expected in `ConstrainedSolver.run` (resolution above), in
  `tests/suite_manifest.json` and `tests/README.md` ceilings (resolution: add both increments, rf-tol's +2 collected
  and +8 lines in `contracts` and the whole suite), checked by a scratch merge below if listed.
