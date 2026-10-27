# Consistent initialization

This page provides an overview of how a solve determines consistent
initial conditions for DAEs.

For DAE systems of the form

$$
M(u, t) \, \frac{du}{dt} = f(u, t\,;\, \text{params}, \text{controls}),
$$

the initial state cannot be chosen freely. The algebraic variables
carry no time derivative, so at the initial time they must already
satisfy their algebraic equations and constraints. Writing the system
as a residual, consistent initialization pins the differential values
\(u_d(t_0)\) at their assembled values and solves

$$
F\big(u(t_0), \tfrac{du}{dt}(t_0), t_0\big) = 0
$$

for the algebraic values \(u_a(t_0)\) and the differential slopes
\(du_d/dt(t_0)\), with \(du_a/dt(t_0) = 0\).

In the common case of a diagonal mass matrix the problem splits: the
algebraic values are solved on their own reduced block, and the
differential slopes then follow by dividing each right-hand side by
its mass entry. Otherwise the algebraic values and the slopes are
solved jointly, with the differential values pinned. The route is
selected automatically, and both run a Newton iteration through the
same [linear backends](linear.md) as the time stepper.

## A DAE model problem

The heat-equation recipe from
[running a simulation](running-a-simulation.md) extends to a DAE with
an algebraic field `w` holding the heating rate \(\nabla^2 u\):

```python
spec.field("w", mass="algebraic", scale=1.0)

@spec.equation(writes="w", reads=("w", "u"))
def rate_closure(q):
    return q.w - q.ops.laplacian(q.u)
```

An algebraic field declares no initial condition of its own, so
[`assemble_state`](../modeling/fields.md#initial-conditions) fills its
entries with a guess, and the solve corrects them before the first
step:

```python
problem = spec.build()
y0 = spec.assemble_state({})   # w's entries are a guess

sol = ds.pdesolve(problem, ds.JIDA(), (0.0, 0.1), y0,
                  saveat=ds.SaveAt(t0=True))
sol.ys[0]                      # w rows already solved to lap(u0)
```

The pass runs at the start of every solve. `consistent_ic=` on
`pdesolve` overrides the solver's default, and disabling it on a
problem with algebraic rows is refused.

## The initialization modes

`JIDA.ic_mode` selects the initialization route. The four modes and
their tradeoffs:

- `"ya_ydp"`, the default, solves the algebraic values and the
  differential slopes with a relinearizing Newton projection. When the
  mass matrix is diagonal, it solves on the reduced algebraic block and
  otherwise solves on the full joint system. It fixes the differential
  values and employs a tight residual 2-norm gate. This is the most
  robust initialization method and should be used unless computational
  cost becomes a concern.
- `"ya_ydp_euler"` solves the same unknowns with a fixed-Jacobian
  Euler-Newton kernel. The consistency system is posed as one small
  implicit Euler step, and the iteration matrix is assembled and
  factored once at the seed and reused by every Newton iteration. At
  the tradeoff of slower convergence and less strict gates, this method
  will be computationally cheaper due to a fixed iteration matrix.
- `"y_fixed_ydot"` holds the derivative fixed and solves for every
  state value, differential values included. The derivative is held at
  the engine's slope seed, which is zero at the start of a solve and
  the post-event slope on an event restart. With the zero seed this
  solves \(F(y, 0) = 0\), relaxing the whole state onto a steady
  state. Use it when the assembled state is itself a guess.
- `"assume_consistent"` skips the projection and is accepted only on a
  problem with no algebraic rows.

Any other value is refused at construction.

## The constraint manifold

The algebraic equations and constraints induce a constraint manifold,
the set of states that satisfy them exactly. Naive state variable
interpolation violates the constraint manifold. To strictly satisfy
the constraints during interpolation, the differential states are held
fixed and the algebraic entries are re-solved by the exact same
procedure as consistent initialization.

`ds.project_algebraic` runs this projection on any state at any time.
Differential entries of `state` are kept bit-for-bit, and the
algebraic entries are re-solved by Newton starting from `seed`:

```python
proj = ds.project_algebraic(problem, t=0.0, state=y0, seed=y0)
proj.state                       # differential kept, algebraic re-solved
proj.status, proj.residual_norm  # traced diagnostics
```

`ds.AlgebraicProjectionOptions` carries the tolerances, the iteration
cap, the linear route, and the differentiation mode. Inside a solve
the same projection backs the on-manifold sampling policies:
`DenseOutputPolicy(mode="projected")` projects every dense sample
([solutions and output](solutions.md)), and a projected event
condition projects each root candidate ([events](events.md)).

## Diagnosing a failed initialization

An in-solve initialization failure is reported through `sol.result` as
`INCONSISTENT_IC` ([solutions and output](solutions.md)).
`check_consistent_init`, from `diff_soup.problem.implicit`, runs the
initialization eagerly on the host and raises with a diagnosis instead.
A singular system is reported with the implicated unknowns and equation
rows, and a non-converged Newton with its residual norm against the
tolerance. A converged run returns the projected state, the slopes, and
the iteration count. `theta=None` resolves the problem's declared
parameter values.
