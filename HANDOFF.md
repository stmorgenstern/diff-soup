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
| 1 | Gauss-Newton form of the implicit rule (N1) | see below | `.ports/m1_rule.py` |

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
