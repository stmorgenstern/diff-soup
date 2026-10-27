# Equations

This page provides an overview of writing equations with the Recipe
API. The spatial operators the equations draw on are covered in
[operators](operators.md).

## The equation interface

`@spec.equation(writes=..., reads=...)` registers a closure as the
equation for one field. The running example from
[the Recipe API](recipe-api.md):

```python
@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.laplacian(q.u)             # du/dt = lap(u)
```

- `writes` names the field the equation determines.
- `reads` names every field and [global](fields.md#globals) the
  closure depends on.
- `name` labels the equation and defaults to the function's name.

## The access object

Inside the closure, `q` is the access object:

- `q.<field>` is the field's current state. On finite-element domains
  it is a quadrature-native field object, covered with
  [operators](operators.md#fem-nodal-values-and-the-weak-form).
- `q.<global>` is a global's value.
- `q.p.<name>` is a declared
  [parameter](parameters.md) and `q.fn.<name>` a declared function,
  called explicitly as `q.fn.<name>(q.t)`.
- `q.t` is the time and `q.x` the node coordinates.
- `q.ops` is the domain's [operator bank](operators.md).
- `q.dt(field)` is the field's time derivative, discussed below.
- `q.<alias>` reads a value coupled in from another component, discussed
  below.

## Pointwise interpretation

An equation closure is ordinary JAX code. An expression that does not
call an operator is interpreted pointwise, evaluated independently at
every node or cell, so arbitrary nonlinear functions of the state
apply directly. Spatial coupling between nodes enters through the
[operator bank](operators.md):

```python
@spec.equation(writes="u", reads=("u",))
def reaction_diffusion(q):
    return q.ops.laplacian(q.u) + q.u - q.u**3   # du/dt = lap(u) + u - u^3
```

## The two forms

An equation closure has two readings, distinguished by what it
returns. Returning a plain array states the mass-matrix form

$$
M(u, t) \, \frac{du}{dt} = f(u, t\,;\, \text{params}, \text{controls}),
$$

with the return value read as the right-hand side
\(d(\text{field})/dt = \text{value}\). Returning an expression that
contains `q.dt(field)` states the fully implicit form

$$
F\!\left(u, \frac{du}{dt}, t\,;\, \text{params}, \text{controls}\right) = 0,
$$

with the expression read as \(\text{expression} = 0\). The running
example in implicit form:

```python
@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.dt("u") - q.ops.laplacian(q.u)  # du/dt - lap(u) = 0
```

Both forms build the same problem, and they correspond to the two
problem types the engines accept
([the problem interface](problem-interface.md)). The implicit form
exists for what the mass-matrix form cannot state: mass terms that are
non-diagonal or nonlinear in the derivative.

## The derivative algebra

`q.dt(field)` composes through a small linear algebra:

- addition and subtraction of derivative terms,
- scaling by any derivative-free value, including values computed from
  the state, the time, or a declared parameter,
- application of a declared
  [`ds.MassOperator`](../../reference/modeling/recipe.md#mass-operators)
  for non-diagonal mass blocks,
- structural `gather` / `reduce` / `scatter` maps with static indices,
  discussed below.

A banded mass block applies a declared operator to the writing
field's derivative:

```python
op = ds.MassOperator.sparse(rows, cols, values, n=n)

@spec.equation(writes="u", reads=("u",))
def balance(q):
    return op(q.dt("u")) - flux(q)              # M @ du/dt = flux
```

An expression linear in the derivative lowers to the mass form with
the exact sparse coefficient structure of the expression. A
nonlinear dependence on the derivative lowers to the general implicit
residual form.

Field roles and derivative use must agree. A field declared
`mass="differential"` is expected under `q.dt` somewhere, and taking
`q.dt` of a field declared `mass="algebraic"` is a build error naming
the field and both roles.

## Structural maps

Fields are plain arrays inside a closure, so selecting and summing
their entries uses ordinary array operations (`q.u[indices]`,
`weights @ q.u`). A derivative expression is symbolic rather than an
array, so it carries its own three structural maps, with static
indices:

- `q.dt(field).gather(indices)` selects entries of the derivative.
- `q.dt(field).reduce(weights)` forms the weighted sum of its
  entries, a single row.
- `q.dt(field).scatter(indices, out_size)` places its entries into a
  larger row block.

A mean-rate closure row combines another field's derivative:

```python
@spec.equation(writes="rate", reads=("rate",))
def rate_row(q):
    return q.rate - q.dt("c").reduce(weights)   # rate = weights . dc/dt
```

## Multi-output equations

`@spec.equations(writes=(...), reads=(...))` (plural) registers one
closure that writes several fields at once, returning a dict keyed by
field name. The closure runs once per residual call, for kernels that
produce residuals for several coupled fields in a single pass:

```python
@spec.equations(writes=("c", "mu"), reads=("c", "mu", "phi"), name="kernel")
def _rhs(q):
    return {
        "c": -(q.c - 0.5) + q.mu + q.phi,
        "mu": (q.mu - q.c) + 1e-3 * q.ops.laplacian(q.c),
    }
```

## Populations

Some models contain many copies of the same sub-model: the same
domain, fields, equations, and boundary conditions repeated `count`
times, differing only in parameter values.
`spec.family(name, count=..., domain=..., params=...)` declares such
a population once, as a member template:

```python
members = spec.family(
    "particles",
    count=2,
    domain=cartesian_grid_1d(2, 0.0, 1.0, method="fvm"),
    params={"rate": [1.0, 2.0]},
)
members.field_decl("c", mass="differential")

@members.equation(writes="c", reads=("c",))
def member_decay(q, m):
    return -m.rate * q.c
```

- The returned handle carries the member template:
  `members.field_decl(...)` declares a member field,
  `@members.equation(...)` a member equation, and `members.bc.flux(...)`
  a member boundary condition.
- A member equation closure takes two arguments, the access object `q`
  and the member view `m`, with `m.<name>` the member's entry of a
  per-member parameter array and `m.member_index` its index.
- `params` holds the per-member parameter arrays (each of shape
  `(count,)`, or a shared scalar). They are the family's
  differentiable and vmappable leaves, so a fit can learn them.
- `stencil_radius=r` declares how far a member kernel's Jacobian
  reaches on the member grid, in cells per axis. The default reach
  is one cell, which the built-in operators satisfy. Declare `2` for
  a composed `laplacian(laplacian(c))`. Over-declaring is safe.
  Under-declaring is not detected and gives a structurally wrong
  sparse Jacobian.
- `members.field_decl(name, mass="algebraic", shape=(1,))` declares a
  per-member scalar, one algebraic unknown per member closed by its
  own member equation (a surface rate, an overpotential). It reads
  the member's profile fields, `members.field(name).surface()`
  projects its only value, and its Jacobian blocks against the
  profile fields are exact. A differential per-member scalar is
  refused.
- The population lowers to one batched block: the member kernel is
  traced once and evaluated over all members in a single `vmap`, so
  `count` members cost one kernel.
- `domain` accepts one shared member domain, or a sequence of `count`
  domains declaring a ragged population. `vmap` batching needs uniform
  member shapes, while numerical stability often ties the
  discretization to the member's size. We resolve this by bucketing:
  members sharing a domain form one uniform size bucket, and each
  bucket lowers to its own batched group.
- Member initial conditions are supplied at `assemble_state` time
  under the key `"<family>.<field>"`, and `scale=` /
  `residual_scale=` map member fields to their scales.
- Members connect to the rest of the problem through the couplings
  discussed below.

A member boundary condition is a flux,
`members.bc.flux(var, on=..., value= | value_fn= | fn=...)`, with the
same three forms a component takes
([boundary conditions](boundary-conditions.md#time-dependent-and-nonlinear-conditions)).
Inside `fn=lambda q, t: ...`, `q.<field>` is the member field at the
selected boundary cell, `q.m.<name>` the member's entry of a
per-member parameter, such as a per-particle reaction rate, and
`q.nodal.<field>` the member's full field. The flux scatters onto the
selected boundary cells, member selectors name node indices directly,
and a condition the member kernel cannot assemble is refused at build.

Members of one family can also exchange with each other pairwise,
through a graph abstraction. The edges are declared once, and one
callback states the physics of an edge:

```python
edges = np.stack((np.arange(count - 1), np.arange(1, count)))  # a chain: 0-1, 1-2, ...
members.graph("chain", edges=edges, directed=False, weights=1.0)

@members.graph_exchange(graph="chain", writes="c", reads=("c",))
def exchange(source, target, edge):
    return edge.weight * (source.c - target.c)   # flow into the target
```

- `edges` is a `(2, E)` array holding each edge's source and target
  member index.
- The callback receives the two endpoint members and the edge, with
  `source.<field>` and `target.<field>` the fields named in `reads`
  and `edge.weight` the declared weight. It returns the flow into
  the target, and lowering scatters it with opposite signs onto the
  two endpoint rows, so every exchange conserves by construction.
- `reads` declares the endpoint fields the exchange sees, which
  fixes the exact cross-member Jacobian blocks.

## Coupling components

Each registered domain, and each family, forms a component with its
own fields and equations. Where fields in different components couple, we express the access
patterns through `spec.couple`. Its keyword `onto` names the receiving
component, and `at` gives the node indices where the transferred values
live.

The running example for this section is solute exchange between oil
and water. A solute diffuses along a water channel, and a population
of oil droplets suspended in it exchanges solute with the water at
each droplet's location.

The water channel is a one-dimensional finite-volume grid holding
the solute concentration `c_w`. We keep the handle `spec.field`
returns, because a coupling names its source by handle:

```python
water = cartesian_grid_1d(64, 0.0, 1.0, method="fvm")
spec = ds.problem("oil_water")
spec.domain("water", water)
c_w = spec.field("c_w", mass="differential", scale=1.0)
```

The droplets are a [population](#populations): each is its own small
one-dimensional domain holding an internal concentration `c_o`, with
a per-droplet exchange rate `k`:

```python
droplets = spec.family(
    "droplets",
    count=2,
    domain=cartesian_grid_1d(8, 0.0, 1e-3, method="fvm"),
    params={"k": [1.0, 2.0]},
)
droplets.field_decl("c_o", mass="differential")
```

Each droplet sits at a physical location in the channel, and `at`
converts those locations into node indices, one per droplet, by
finding the channel cell nearest each location:

```python
x_cells = water.node_coordinates()[:, 0]                     # cell-center coordinates
xs = np.array([0.2, 0.8])                                    # droplet locations
at = np.array([np.argmin(np.abs(x_cells - x)) for x in xs])  # nearest cell to each droplet
```

### Gathering onto a family

Each droplet needs the water concentration at its own location.
`spec.couple` hands member `m` the source field's value at node
`at[m]`, and the member closure lists the source in `reads` and
reads it by name:

```python
spec.couple(c_w, onto="droplets", at=at)

@droplets.equation(writes="c_o", reads=("c_o", "c_w"))
def droplet_uptake(q, m):
    return q.ops.laplacian(q.c_o) + m.k * (q.c_w - q.c_o)
```

### Scattering onto a domain

The water sees the droplets in return: a scatter sends one value per
member into the receiving field's rows at the same nodes.

```python
droplets.couple(droplets.field("c_o").surface(), onto="water", at=at, reduce="sum")
```

The source is a member-field projection:

- `.surface()`: the member's value at its outer surface, one per
  member.
- `.boundary(tag)`: the member's trace at a tagged boundary.
- `.at(idx)`: the member's value at a chosen node index.

`reduce=` merges members that land on the same receiving node:

- `"sum"` (the default) adds them.
- `"mean"` averages them.
- `"none"` declares that every member has its own node.

### Read-through aliases

When the oil phase is a second continuous layer over the same
channel grid rather than a population, the transfer between the two
domains is a read. `spec.couple` with `as_=` injects the source values under a new
name, read as `q.<alias>` in the receiving closures, while the unknown
stays owned by the source. Each oil node reads the
water concentration directly below it:

```python
spec.domain("oil", cartesian_grid_1d(64, 0.0, 1.0, method="fvm"))
c_o = spec.field("c_o", domain="oil", mass="differential", scale=1.0)
spec.couple(c_w, onto="oil", at=np.arange(64), as_="c_w_below")

@spec.equation(writes="c_o", reads=("c_o", "c_w_below"))
def oil_phase(q):
    return q.ops.laplacian(q.c_o) + K * (q.c_w_below - q.c_o)
```

## Observations

`spec.observe` registers named outputs computed from the state.
Observations ride along the built problem as a callable for
post-processing and fitting, outside the residual:

- `spec.observe.scalar(name=..., fn=...)` is a single scalar,
- `spec.observe.field(name=..., fn=...)` a spatially resolved output,
- `spec.observe.spatial_mean(name=..., fn=...)` a domain average,
- `spec.observe.boundary_mean(name=..., fn=..., tag=...)` a mean over
  a tagged boundary.

Every verb also takes `domain=`, the domain the observation is evaluated
on. By default this is the domain whose fields `fn` reads, or the first
declared domain if `fn` reads no fields.

Each `fn` is a closure over the same access object `q`:

```python
spec.observe.scalar(name="mean_u", fn=lambda q: jnp.mean(q.u))
```
