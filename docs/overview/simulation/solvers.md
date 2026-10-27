# ODE/DAE solvers

This page provides an overview of the time-stepping engines behind
`ds.pdesolve`: the native solvers, their options and presets, which
options are static under jit compilation, and the diffrax wrapper.

## Recommended default

`ds.JIDA()` at default settings is the recommended solver. It handles
every problem form the Recipe API produces, from pure ODEs to fully
implicit index-1 DAEs. For a non-stiff problem whose rows are all
differential, `DiffraxStepper` opens the explicit Diffrax collection
([the diffrax wrapper](#the-diffrax-wrapper)).

## The native solvers

Three native engines implement the variable-step, variable-order
family of implicit multistep methods in pure JAX, GPU-resident end to
end.

- `JIDA` (recommended) solves the fully implicit form
  \(F(u, du/dt, t) = 0\) directly, the method of the SUNDIALS IDA
  lineage (Hindmarsh et al., 2005). It handles ODEs and fully implicit
  index-1 DAEs.
- `JBDF` implements the classic backward differentiation formula (BDF)
  family with a fixed-leading-coefficient corrector, the coefficient
  family of IDA (Hindmarsh et al., 2005). It handles ODEs and mass-form
  DAEs \(M(t)\, du/dt = f\) with a constant or time-dependent mass. A
  state-dependent mass is rejected.
- `JNDF` implements the Klopfenstein-Shampine numerical differentiation
  formula (NDF) family, the method of QNDF and MATLAB's `ode15s`
  (Shampine and Reichelt, 1997). The NDFs modify the BDFs with smaller
  error constants at the same orders. It handles ODEs and mass-form
  DAEs \(M(u, t)\, du/dt = f\), including a state-dependent mass.

## Constructor options

The three constructors share their options. Each option states its
default.

- `rtol=1e-6` and `atol=1e-6`, the relative and absolute integration
  tolerances.
- `linear="auto"` names the linear backend, `"auto"`, `"cudss"`,
  `"klu"`, or `"dense"` ([linear solvers](linear.md)).
- `newton_tol_factor=1.0` scales the Newton corrector's acceptance
  threshold. The step error test keeps the unscaled tolerances.
- `ic_max_newton_iters=10` caps the consistent-initialization Newton
  on `JNDF` and `JBDF`
  ([consistent initialization](consistent-init.md)). `JIDA` runs its
  own initialization modes and refuses a non-default value.
- `precision=` and `jacobian_values_dtype=` are left unset: the solve
  inherits the problem's precision policy ([precision](precision.md)).
  A value given here is used only when the problem declares none or
  the same one.
- `adjoint=None` carries a default differentiation strategy on the
  solver, overridden per call
  ([sensitivities and autodiff](sensitivities.md)).

`JIDA` adds initialization and error-test options.

- `consistent_ic=True` projects the initial state before the first
  step and is a no-op for a pure ODE
  ([consistent initialization](consistent-init.md)).
- `suppress_alg=True` keeps algebraic rows out of the local error
  test. Differential globals stay in the error norm, and `False`
  includes every algebraic row.
- `id=None` overrides the differential-row mask used by the error
  test. The default derives the mask from the problem's mass
  structure.
- `ic_mode=` selects the consistent-initialization route
  ([consistent initialization](consistent-init.md)). Any value outside
  the four below is refused at construction.
    - `"ya_ydp"`, the default, solves the algebraic values and the
      differential slopes with a relinearizing Newton projection, on
      the reduced algebraic block when the mass matrix is diagonal and
      on the joint system otherwise.
    - `"ya_ydp_euler"` solves the same unknowns with a fixed-Jacobian
      Euler-Newton kernel, factoring once and iterating against the
      frozen factor.
    - `"y_fixed_ydot"` solves the full state at a zero leading
      coefficient while holding the derivative at the engine's slope
      seed, zeros on the initial entry and the post-event slope on a
      restart.
    - `"assume_consistent"` skips the projection.

!!! warning "The Euler routes gate convergence on a weighted RMS norm"
    The Euler-Newton kernel accepts when the weighted root-mean-square
    norm of its Newton step falls below its gate, while the default
    route gates on the residual 2-norm. The RMS normalization divides
    by the square root of the row count, so on a large algebraic block
    the Euler gate is looser by roughly that factor.

## Presets

`<solver>.stock()` and `<solver>.flash()` assemble the two execution
presets on each solver class. Both accept the constructor options as
keyword overrides.

- `.stock()`, the default profile of every bare constructor:
  step-size control matching the engine's reference implementation,
  with Jacobian reuse across steps.
- `.flash()`: a proportional-integral step-size controller with a
  fresh Jacobian each corrector iteration.

## Solve options

`solve()` takes the per-run options `pdesolve` forwards
([running a simulation](running-a-simulation.md)). The engine keywords
and their defaults:

- `dt0=0.0`, the initial step size. `0` estimates it.
- `hmin=0.0` and `hmax=0.0`, the step-size bounds. `0` leaves each
  bound to the engine.
- `max_order=5`, the maximum integration order.
- `max_steps=10_000`, the step budget, which also sizes the fixed
  output and trace buffers ([solutions and output](solutions.md)).
- `linear=` overrides the configured backend for this call.
- `trace=False`. `True` attaches a per-step diagnostic payload to the
  returned solution.

The remaining call options (`saveat=`, `theta=`, `event=`,
`control_state0=`, `consistent_ic=`, `lambda0=`, `adjoint=`) are
covered in [running a simulation](running-a-simulation.md).

## Compilation

Each solve is jit compiled and cached on the solver configuration, the
problem object's identity, and the argument shapes. The three native
engines share one contract.

- Dynamic arguments reuse the compiled program: the time span, the
  initial state, the parameter values (`theta=`), the control values
  (`control_state0=`), `dt0=`, and the save-time values for a fixed
  number of save times.
- Static options compile a fresh program: every constructor option
  (tolerances, backend, precision, profile, the solver-level adjoint),
  `max_steps=`, `max_order=`, `hmin=`, `hmax=`, `trace=`, the event
  declaration, `consistent_ic=`, and the output plan's shape (the
  number of save times, dense output on or off).
- Rebuilding the problem also compiles fresh. Keep the problem object
  stable across calls and vary values through `theta=`.

## The diffrax wrapper

`DiffraxStepper` provides a wrapper to the ODE solvers in the
[Diffrax](https://docs.kidger.site/diffrax/) package (install the
optional dependency with `pip install diffrax`). Diffrax provides the
explicit Runge-Kutta solvers, among them `Euler`, `Heun`, `Bosh3`,
`Tsit5`, `Dopri5`, and `Dopri8`
([the Diffrax solver collection](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/)).
`DiffraxStepper()` defaults to the `Tsit5` solver with `rtol=1e-6`,
`atol=1e-6`, and `max_steps=10_000`, and `inner=` takes any Diffrax
solver instance.

The eligibility contract, checked before integration:

- Every differential field and differential global carries constant
  identity mass. Non-unit, time-dependent, and state-dependent mass are
  refused.
- Dirichlet rows are eligible, integrated through the boundary value's
  time derivative (below).
- Algebraic fields, algebraic globals, and derived-control closing rows
  are refused.
- Events are terminate-only, and a continuous event must use
  `direction="any"`, because Diffrax does not represent directional
  crossings.

The per-call options narrow with the backend. `tolerances=`,
`control_state0=`, `lambda0=`, `trace=True`, `linear=`, and the native
step-bound options (`hmin`, `hmax`, `max_order`, `max_events`) are
refused, `saveat=` accepts `steps=True` in its boolean form only
([solutions and output](solutions.md)), and dense output and the
`"projected"` sampling policy are refused.

A Dirichlet pin \(u_b(t) = c_b(t)\) is an algebraic row in the lowered
system ([boundary conditions](../modeling/boundary-conditions.md)), so
the backend reduces it to the explicit equation \(du_b/dt = dc_b/dt\) and
seeds the pinned value consistently at `t0`. `consistent_ic=False`
asserts that the caller already supplied the intended start.

!!! warning "Boundary drift on long horizons"
    The differentiated constraint is exact from a consistent start in
    exact arithmetic, but integration error accumulates in the pinned
    rows, so loose tolerances can drift the boundary values over a long
    horizon. The native engines enforce the pin algebraically at every
    step.

The shared adjoints map as follows:

| diff_soup strategy | Diffrax strategy |
| --- | --- |
| `None` or `CheckpointAdjoint` | `RecursiveCheckpointAdjoint` |
| `ForwardAdjoint` | `ForwardMode` |
| `BacksolveAdjoint` | `BacksolveAdjoint` |
| `ImplicitAdjoint` | `ImplicitAdjoint` |

A `CheckpointAdjoint(checkpoints=k)` carries its count across.
`BacksolveAdjoint` and `ImplicitAdjoint` execute only on this backend,
and the native engines refuse both.

For example, to solve the nonlinear heat equation from the
[Diffrax examples](https://docs.kidger.site/diffrax/examples/nonlinear_heat_pde/),

$$
\frac{\partial u}{\partial t} = (1 - u)\,\nabla^2 u, \qquad x \in (-1, 1),
$$

$$
u(-1, t) = u(1, t) = 1, \qquad u(x, 0) = x^2,
$$

the recipe declares the nonlinear equation and the wrapper drops into
the same `pdesolve` call:

```python
import numpy as np
import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d

n = 50
spec = ds.problem("nonlinear_heat")
spec.domain("rod", cartesian_grid_1d(n, -1.0, 1.0, method="fvm"))
spec.field("u", mass="differential", scale=1.0, init=lambda x: x**2)

@spec.equation(writes="u", reads=("u",))
def nonlinear_heat(q):
    return (1.0 - q.u) * q.ops.laplacian(q.u)

spec.bc.dirichlet("u", node_indices=np.array([0, n - 1]), value=1.0)

problem = spec.build()
y0 = spec.assemble_state({})

sol = ds.pdesolve(problem, ds.DiffraxStepper(), (0.0, 1.0), y0)
```

The pinned end values hold at exactly 1 through the differentiated
constraint, and the solution matches a native `ds.JIDA()` solve of the
same problem to the requested tolerances.

## References

- A. C. Hindmarsh, P. N. Brown, K. E. Grant, S. L. Lee, R. Serban,
  D. E. Shumaker, and C. S. Woodward. SUNDIALS: Suite of nonlinear and
  differential/algebraic equation solvers. ACM Transactions on
  Mathematical Software, 31(3):363–396, 2005.
- L. F. Shampine and M. W. Reichelt. The MATLAB ODE Suite. SIAM
  Journal on Scientific Computing, 18(1):1–22, 1997.
