# Constraints

This page provides an overview of declaring constraints with
`spec.constraint`. A constraint appends one equation to the system,
stating what a quantity computed from the state must equal, and names
the global that closes it.

## The constraint pair

For the built system \(M(u, t)\,du/dt = f\), a constraint appends the
algebraic row

$$
c(u, t) = \text{target}(t),
$$

where \(c\) is the constrained quantity and \(\text{target}\) the
target protocol. The row needs an unknown to determine, and that
unknown is a declared [global](fields.md#globals):

- `closes=` names the global. The global has no equation of its own.
  The constraint is its defining equation, and the pair keeps the
  system square: one new row, one new unknown.
- `name=` labels the constraint, and `target_fn=lambda t: ...` is the
  target protocol, receiving dimensional time
  ([the two clocks](fields.md#the-two-clocks)). Every argument is
  keyword-only.
- Constraint rows are algebraic rows of \(M\), holding at every
  instant. At the start of a solve the globals are determined by
  [consistent initialization](../simulation/consistent-init.md), and
  the global block carries the Schur treatment described with
  [globals](fields.md#globals).
- `integrand_scale=` divides the target, reconciling a target
  protocol in dimensional units with a non-dimensional constrained
  quantity.
  Every constraint kind takes it inline, and
  `spec.scales(integrand={name: value})` sets the same scale
  centrally, keyed by `name=`.

## The constraint kinds

The constrained quantity is computed by a closure over the same
access object `q` as [equations](equations.md#the-access-object):

- `spec.constraint.reduce_residual(name=..., field=..., reduce_fn=...,
  target_fn=..., closes=...)` constrains a reduction of the field's
  rows of \(f\). In the mass-matrix form those rows are the field's
  rates, so the constraint closes a global against a total rate.
  Discussed below.
- `spec.constraint.boundary_integral(name=..., field=..., on=...,
  integrand_fn=..., target_fn=..., closes=...)` constrains the
  boundary integral \(\oint c \, d\Gamma = \text{target}(t)\) over a
  tagged boundary of a triangle or tetrahedral mesh, or over
  selected boundary cells of a grid, weighted by the face-area
  measure, with `integrand_fn=lambda q: ...` the integrand. The
  closure may return a scalar, one value per boundary node, one
  value per facet, or a full nodal field, each weighted by the
  matching boundary measure. On a grid, do not select the ends of an
  axis declared periodic: the two ends of a wrapped axis are not
  boundaries, and the selector machinery resolves whatever cells it
  names.
- `spec.constraint.flux_total(from_bc=..., closes=..., target_fn=...)`
  constrains the total flux through a named
  [flux boundary condition](boundary-conditions.md#time-dependent-and-nonlinear-conditions).
  Discussed below.
- `spec.constraint.domain_integral(name=..., field=...,
  integrand_fn=..., target_fn=..., closes=...)` constrains the domain
  integral \(\int c \, dV = \text{target}(t)\), the integrand
  weighted by the domain measure. An integrand whose length does not
  match the domain is refused.

## Closing against a total rate

The running example's rod gains an unknown uniform source, a global
the solver adjusts so the rod heats at a prescribed total rate. The
source strength is read by the equation like any global:

```python
grid = cartesian_grid_1d(n, 0.0, 1.0, method="fvm")
spec = ds.problem("heated_rod")
spec.domain("rod", grid)
spec.field("u", mass="differential", scale=1.0)
spec.global_var("Q", initial=0.0)

@spec.equation(writes="u", reads=("u", "Q"))
def heat(q):
    return q.ops.laplacian(q.u) + q.Q
```

The constraint closes `Q` against the total rate, the volume-weighted
sum of the field's rate rows:

```python
vols = jnp.asarray(grid.cell_volumes)

spec.constraint.reduce_residual(
    name="total_rate",
    field="u",
    reduce_fn=lambda f: jnp.sum(f * vols),
    target_fn=lambda t: 1.0,
    closes="Q",
)
```

- `reduce_fn` receives the field's assembled rows of \(f\), here the
  per-cell rates, and returns the reduced scalar.
- A plain `jnp.sum` reducer lowers to a sparse accumulation row. A
  general reducer differentiates the field's rows densely.

## Closing against a boundary flux

The bowl of soup's radiative rim
([boundary conditions](boundary-conditions.md#time-dependent-and-nonlinear-conditions))
leaves the loss coefficient free: declared as a global, it becomes
the degree of freedom the solver adjusts so the total heat loss
through the rim follows a protocol:

```python
spec.global_var("eps", initial=0.05)

spec.bc.flux("u", tag="rim", fn=lambda q, t: -q.eps * q.u**4, name="rim_loss")

spec.constraint.flux_total(
    from_bc="rim_loss",
    closes="eps",
    target_fn=lambda t: -0.5,
)
```

- `from_bc` names the flux condition by its `name=` handle. The
  constraint integrates the same flux closure over the same facets
  the condition scatters to, so the constrained total and the
  boundary term agree by construction.
- `from_bc` accepts a list of names, summing the totals.
- `q.<global>` reads the closed global inside the flux closure.

## Electrochemical protocols

Battery and electrochemistry protocols prescribe total currents and
voltages. We provide constraint and control verbs named for these
protocols:

- `spec.constraint.galvanostatic(name=..., current_fn=...,
  target_fn=..., closes=...)` drives the scalar computed by
  `current_fn=lambda q: ...` to the target protocol,
  \(c(u, t) = \text{current\_fn}(q)\). The closure computes the
  quantity itself, so any already-assembled scalar can be driven
  this way. The name reflects its most common use, a prescribed
  total current.
- `spec.constraint.galvanostatic(name=..., boundary=..., on=...,
  target_fn=..., closes=...)` forwards to the boundary-integral
  constraint, integrating the closure over the tagged boundary.

A constrained global is usually driven in time by a derived control
(`spec.control.derived`, or its electrochemical alias
`spec.control.galvanostatic`), which replaces the global's closing
row with the authored protocol, covered with
[controls](controls.md).
