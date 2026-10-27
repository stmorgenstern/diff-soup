# Fields and globals

This page provides an overview of declaring a problem's unknowns. A
field is a scalar- or vector-valued variable of interest that is
defined on a domain. A global is a state variable with no domain,
typically a scalar that closes an integral constraint.

## The variable types

By default the built problem takes the mass-matrix form

$$
M(u, t) \, \frac{du}{dt} = f(u, t\,;\, \text{params}, \text{controls}),
$$

where the state vector \(u\) concatenates the declared variables and
\(M\) is the mass matrix. A differential variable has nonzero entries
on its rows of \(M\), so its equation determines its time derivative.
An algebraic variable has zero rows, so its equation is a constraint
that must hold at every instant, and any algebraic variable makes the
system a DAE rather than an ODE. Crossing the two roles with where the
variable lives gives four types:

- Local differential: a field with a time derivative, declared
  `spec.field(name, mass="differential")`.
- Local algebraic: a field without an explicit time derivative,
  declared `spec.field(name, mass="algebraic")`.
- Global differential: a global with its own time derivative, declared
  `spec.global_var(name, mass="differential")` and written by exactly
  one equation ([globals](#globals)).
- Global algebraic: a global with no equation of its own, closed by a
  [constraint](constraints.md) or driven by a [control](controls.md),
  declared `spec.global_var(name)`.

The roles are declared and checked at build time. The engines use them
to partition the state, and the error control can include or exclude
the algebraic rows (see
[ODE/DAE solvers](../simulation/solvers.md)).

## Declaring a field

`spec.field(name, mass=...)` declares one field. The running example
from [the Recipe API](recipe-api.md) declares the temperature on the
rod:

```python
spec.field("u", mass="differential", scale=1.0, init=lambda x: x * (1.0 - x))
```

The options:

- `mass` (required): `"differential"` or `"algebraic"`, as defined
  above.
- `components=d`: declares a vector field with `d` values per node.
  `shape` sets the array shape directly instead. Pass at most one of
  the two.
- `domain`: the name of the registered domain the field lives on.
  Required when more than one domain is registered.
- `init`: the field's default initial condition, discussed below.
- `scale`, `offset`, `residual_scale`: scaling options, discussed
  below.
- `quad`: on finite-element domains, whether `q.<name>` in equation
  closures is a quadrature-native field object
  ([operators](operators.md#fem-nodal-values-and-the-weak-form))
  rather than a plain nodal array. Defaults to true on meshes and has
  no effect on grids.

## Initial conditions

The declared `init` is only a default. We employ a flexible system for
redeclaring the initial condition, reusing the compiled problem and
allowing easy differentiation through the initial condition:

```python
y0 = spec.assemble_state({})                              # the declared init
y0 = spec.assemble_state({"u": values})                   # an explicit array
y0 = spec.assemble_state({"u": lambda x: x * (1.0 - x)})  # a callable of the node coordinates
```

- An explicit entry overrides the declared `init`, and `default=`
  fills any field left out.
- A `field(init=...)` callable takes one coordinate array per spatial
  dimension (`init=lambda x, y: ...` in two dimensions). A callable
  passed to `assemble_state` instead takes the full node-coordinate
  array of shape `(num_nodes, dim)`.
- For a differential field the assembled value is its starting value.
  For an algebraic field it is only a guess, made consistent
  automatically when the solve starts.

## Scaling

- `scale` and `offset` non-dimensionalize the field's state as
  \(u_{\text{dimensional}} = \text{offset} + \text{scale} \cdot
  u_{\text{internal}}\), so the solver works with internal values near
  one.
- Omitting `scale` defaults it to 1.0 and warns at build time. An
  explicit `scale=1.0` marks the field as intentionally dimensionless
  and does not warn.
- `residual_scale` divides the field's residual rows to improve the
  Jacobian condition number.
- `spec.scales(state={...})` sets state scales centrally by field name
  and overrides the inline declarations. `spec.time_scale(tau)` sets
  the time scale.
- A scale acts at the boundary of the problem: the state is stored as
  \((u_{\text{dimensional}} - \text{offset}) / \text{scale}\), closures read
  the internal state values, and observations read the state
  rewrapped to dimensional units.

### The two clocks

`spec.time_scale(tau)` integrates the model on internal time
\(t_{\text{int}} = t_{\text{dim}} / \tau\). The residual's own
closures keep internal time, and every authored callback runs on the
dimensional clock, the experiment's own, so the same callback states
the same physics whether or not the model is nondimensionalized:

| Internal time | Dimensional time |
|---|---|
| equation closures | control `fn` |
| constraint integrand closures | Dirichlet, flux, Robin, and traction value callbacks |
| | boundary condition `fn(q, t)` |
| | constraint `target_fn` |
| | family member fluxes (`members.bc.flux`) |

## Globals

`spec.global_var(name, initial=..., mass="algebraic", scale=None)`
declares a global, a scalar or one-dimensional unknown with no
domain. A differential global has its own time derivative, written
by exactly one equation that returns it:

```python
spec.global_var("z", initial=1.0, mass="differential")

@spec.equation(writes="z", reads=("z",))
def relax(q):
    return q.p.a - q.z              # dz/dt
```

- A differential global carries unit mass and takes part in error
  control, interpolation, and consistent initialization as an
  ordinary unknown.

An algebraic global (the default) has no equation of its own. It is
closed by exactly one [constraint](constraints.md) or driven by a
[control](controls.md):

```python
spec.global_var("load", initial=0.0, scale=1.0)
```

- A global is read in equation closures as `q.<name>` after being
  listed in `reads=`.
- Globals are not part of the state assembled by `assemble_state`. The
  solver appends them to the state during the solve, seeded from
  `initial`, and their solved values are read from
  `Solution.lambdas`. Partitioning the globals this way lets the
  solver apply a Schur complement to the global block, avoiding dense
  rows in the sparse Jacobian (a global's constraint row touches
  every unknown it integrates over).
- `scale` non-dimensionalizes the global like a field.

## Consistent initialization

For DAEs, the algebraic variables cannot start from arbitrary values.
At the start of a solve we solve the reduced system for consistent
initial conditions, in which the algebraic variables satisfy their
algebraic equations and constraints at the initial time. The
assembled algebraic values only seed that solve. The pass and its
options are covered in
[consistent initialization](../simulation/consistent-init.md).
