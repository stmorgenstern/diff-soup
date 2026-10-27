# Events

This page provides an overview of solve-time events. Events are
supplied to `pdesolve(..., event=...)` as `ds.Event` objects, outside
the Recipe API. An event requires a condition, which determines when
it will activate, and an action, which defines what happens after it
activates. A termination event (`affect=None`, the default)
stops the solve at the event time, and an affect event rewrites the
state and continues.

## Declaring an event

`ds.Event(cond_fn, affect=None, root_finder=None, direction="any",
state_policy=None)`:

- `cond_fn(t, u, args, *, control=None)` returning a boolean declares
  a discrete event, checked at each accepted-step boundary. Returning
  a real scalar declares a continuous event, whose exact crossing
  time is located on the dense interpolant by `root_finder`.
- The `control` argument is the resolved control mapping, the same
  values the model reads. A problem with an operating table receives
  its [`ControlState`](../modeling/controls.md) instead.
- `direction=` fires on an `"up"` crossing (negative to positive),
  `"down"`, or `"any"` (the default).

## Termination events

A termination event stops the solve at the event time, and the stop
is a successful outcome: `sol.ok` is `True` and `sol.result` reads
`EVENT_STOP`. On the heat-equation solve from
[running a simulation](running-a-simulation.md), stopping when the
center temperature falls through 0.15:

```python
from diff_soup.timestepping import Bisection

cool = ds.Event(
    cond_fn=lambda t, u, args, control=None: u[n // 2] - 0.15,
    root_finder=Bisection(),
    direction="down",
)
sol = ds.pdesolve(problem, ds.JIDA(), (0.0, 0.2), y0, event=cool)
sol.terminal.t, sol.terminal.y   # the located crossing
```

A discrete condition stops at the first accepted step past the
crossing, a continuous condition at the located crossing itself.

## Locating the crossing

A continuous condition needs a root finder.
`Bisection(rtol=..., atol=...)`, from `diff_soup.timestepping`,
locates the crossing by fixed-trip-count interval halving on the
dense interpolant, branchless and therefore safe under `jit` and
`vmap`. The iteration count is sized from `rtol` and the bracket
width.

## Affect events

An affect event rewrites the state and the control mapping at the
crossing through `affect=lambda t, u, args, control:
(u_new, control_new)`, and the solve continues. On the heat solve, a
reheat pulse at each crossing:

```python
def reheat(t, u, args, control):
    return u.at[1:-1].add(0.1), control

pulse = ds.Event(
    cond_fn=lambda t, u, args, control=None: u[n // 2] - 0.15,
    root_finder=Bisection(),
    direction="down",
    affect=reheat,
)
sol = ds.pdesolve(problem, ds.JIDA(), (0.0, 0.2), y0, event=pulse)
sol.event_log.count   # one entry per fire, the solve ran to t1
```

- The engines restart at the switch: order one, fresh Jacobian and
  factorization caches.
- On a problem with declared controls, the restart advances the
  control epoch by one, invalidating every quantity cached against
  the pre-switch controls ([controls](../modeling/controls.md)).
- On a problem with no declared controls and no `control_state0=`, the
  affect's control value is folded into the parameter pytree. There is no
  epoch-bearing `ControlState` to advance, so `Solution.control_epochs`
  and the event-log epochs stay zero. The restart still resets the
  linearization and factor caches, and `sol.stats.event_solves` counts it.
- On a problem with an operating table, the affect receives the
  carried `ControlState` and returns its update, for example
  `control.with_operating(mode=..., target=...)`.

## Projected event states

Continuous conditions require mid-step candidates to be evaluated. Under
the default `"fast"` policy, local and global differential and algebraic
states are sampled mid-step through interpolation. For algebraic
variables, interpolation may violate the
[constraint manifold](consistent-init.md#the-constraint-manifold), the
tradeoff for not re-solving the constraints. To ensure strict
satisfaction of the algebraic constraints, use
`Event(state_policy=ds.DenseOutputPolicy(mode="projected"))` to project
each mid-step candidate onto the algebraic constraints before the
condition reads it.

## What the solution records

- A terminating event writes its exact sample into the reserved final
  save slot, so the stop state is recorded even when it falls between
  requested save times.
- `Solution.control_epochs` holds the epoch owning each saved sample.
  Invalid slots are `-1`.
- `sol.event_log` keeps a ledger of all triggered events, including the
  times, triggering event indices, kinds of events, crossing directions,
  pre- and post-affect states with their corresponding epochs, a count,
  and an overflow flag. Native JNDF, JBDF, and JIDA solves reserve 16
  entries by default, and `max_events=` on `ds.pdesolve` sets the
  capacity, with fires past capacity setting only the overflow flag.
- `sol.terminal` records the exact terminal state of a terminating
  event and the epoch that owns it.
