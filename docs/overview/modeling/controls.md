# Controls

This page provides an overview of driving a problem with controls
declared through `spec.control`. A control turns an authored forcing
protocol into a value the solve reads at runtime.

## Declaring a control

Two verbs declare controls:

```python
spec.control.direct(name="hold", fn=lambda t: 4.1)
spec.control.derived(name="drive", fn=lambda t: 0.5 * t, binds_to="Q")
```

- `direct` declares a direct time-dependent input variable. Its
  resolved value is read by name, as `q.control.<name>` in equation,
  constraint, and boundary-condition closures. Observations do not receive controls.
- `derived` declares a controlled quantity the solver assembles, such
  as a total current. `binds_to=` provides the
  [global](fields.md#globals) variable that carries it, and exactly
  one [constraint](constraints.md) must close the bound global. At
  build time the control replaces that constraint's closing row with
  \(\text{global} = \text{fn}(t)\). Each global takes exactly one
  driving control.
- `potentiostatic` and `galvanostatic` are wrappers for the common
  electrochemical controls, a set voltage (`direct`) and a set
  current (`derived`).
- A control is authored in dimensional time. `fn` receives the
  dimensional time and returns the dimensional value, and the solver
  side non-dimensionalizes through `spec.time_scale` and the driven
  variable's `scale=` ([the two clocks](fields.md#the-two-clocks)).

## Overriding at solve time

Passing `control_state0=` to `pdesolve` replaces the authored
protocols for the whole solve:

```python
sol = pdesolve(problem, solver, t_span, y0, control_state0={"drive": 0.7})
```

- The mapping names every declared control. A derived control's
  value is a scalar or one value per entry of the driven global, in
  the same dimensional units the control callable returns. A direct
  control's payload is shape-checked against the authored value's
  width.
- A supplied mapping replaces the authored `fn` entirely, resolved
  once before the solve runs. This is how one compiled problem runs
  many drive protocols.

## Operating protocols

Some models switch what is controlled while the solve runs, a
battery moving from constant current to constant voltage.
`spec.control.operational` declares the full table of closing rows
once, current, voltage, and power rows closing one algebraic current
unknown. Which row is active is not authored. It is runtime data,
supplied to the solve as a `ControlState`:

```python
control = ds.ControlState.operating(mode=0, target=0.25)
sol = pdesolve(problem, solver, t_span, y0, control_state0=control)
```

- `mode` selects the active row and `target` is the value the row
  drives to. Both are runtime data, so switching them reuses the one
  compiled solve with no retracing.
- Switching mid-solve is a solver-side event. The event machinery,
  the `ControlState` epoch, and the bookkeeping a switch leaves in
  the solution are covered with [events](../simulation/events.md).

A protocol that runs several modes one after another is declared as a
`ds.Schedule` of `ds.Segment`s, each holding one row of the table at a
target for a duration:

```python
protocol = ds.Schedule([
    ds.Segment(current=-0.68, duration=500.0),
    ds.Segment(voltage=4.1, duration=1500.0),
])
```

- A segment names its mode by keyword, `current=`, `voltage=`, or
  `power=`, and the value is that row's target. A tuple of numbers gives
  a target that follows a polynomial in the time since the segment
  started, constant term first.
- Each segment starts where the one before it ended, with the
  differential unknowns carried over. A segment that changes the mode
  re-initializes the algebraic unknowns.
- `carry_step=None` continues the integrator's step size and history
  into a segment that keeps the mode, and starts afresh when the mode
  changes. `True` always continues, and `False` always starts afresh.
- An experiment follows a protocol through `ds.optimization.inference.Experiment(...,
  schedule=protocol)` ([fitting
  parameters](../inference/fitting.md#declare-the-experiment)).

## Control signals

An optimal control problem
([`ds.optimization.control.ControlProblem`](../../reference/optimization/control/problem.md))
designs the targets of a schedule. `SignalControls` takes them from the
time signals of [function approximation](../inference/function-approximation.md#time-varying-signals),
one block after another:

```python
from diff_soup.optimization.inference.approx import BSpline, PiecewiseConstant

current = BSpline(jnp.linspace(0.0, 1200.0, 10), jnp.zeros(12))
controls = ds.optimization.control.SignalControls([(current, "current", -1.0)])
```

- A block is `(signal, mode, scale)`. It holds the row `mode` at
  `scale * signal(t)` over the signal's knots, starting where the block
  before it ends.
- Each segment of the signal becomes one schedule segment whose target is
  the segment's polynomial, so the schedule integrates the signal
  exactly. The integrator continues across the knots of a block.
- The decision vector is the coefficients of every block in order, the
  control points of a `BSpline`. A `BSpline` stays inside the range of its
  control points, so bounds on them bound the whole current.
- Two blocks give a mixed protocol, a current block then a voltage
  block.

`free_durations=True` adds each block's duration to the end of the
decision vector, so a switch time or a total time becomes a decision
variable:

```python
hold = lambda: PiecewiseConstant(jnp.array([0.0, 1.0]), jnp.zeros(1))
cccv = ds.optimization.control.SignalControls(
    [(hold(), "current", -1.0), (hold(), "voltage", 1.0)], free_durations=True
)
problem = ds.optimization.control.ControlProblem(
    problem=model, controls=cccv, cost=lambda sol, u, args: u[-2] + u[-1],
    save_fractions=(0.5, 1.0), ...
)
```

- A block of duration `D` stretches its signal's knots to span `D`.
  Bound each duration above zero.
- The sample times then move with the durations, so the problem takes
  `save_fractions`, positions within every segment, in place of
  `save_times`. A fraction of `1` is the segment's end.
- A minimum-time problem is one solve with the durations in its cost.
  The receding horizon steps a fixed grid and refuses free durations.

