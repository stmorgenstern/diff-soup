# Learnable parameters

This page provides an overview of declaring parameters and named
functions with `spec.param` and `spec.function`. Any coefficient in a
recipe can become a declared parameter, and a declared parameter is a
coefficient the [inference layer](../inference/index.md)
can fit.

## Declaring a parameter

Making the heat problem's diffusivity learnable:

```python
D = spec.param("D", value=0.1, learnable=True, bounds=(0.05, 1.5))

@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.p.D * q.ops.laplacian(q.u)
```

The options:

- `value` (required): the nominal value used in a plain solve, a
  scalar or an array.
- `learnable=True` marks the parameter as eligible for fitting. The
  fit then selects which learnable parameters become decision
  variables.
- `bounds=(lo, hi)`: bounds for the fit, applied when the parameter
  is a fitted decision variable. A parameter declared without
  `bounds` is unbounded, and either side may be `None` for a
  one-sided bound.
- `prior=`: a declared prior, consumed by MAP estimation.

`spec.param` returns a handle for later access. The fit layer
selects parameters by handle, and the handle indexes the fitted
values in a result.

## Where parameters are read

- `q.p.<name>` reads a declared parameter inside equation closures,
  constraint integrand closures, and observation closures.
- A boundary-condition closure of the form `fn=lambda q, t: ...`
  reads `q.p.<name>` and `q.fn.<name>`, the fitted values during a fit
  ([boundary conditions](boundary-conditions.md#time-dependent-and-nonlinear-conditions)).
  A constraint-target or control closure receives the time alone, and
  a coefficient used there is captured by the Python closure itself.
- A [population](equations.md#populations)'s member closures read
  their per-member parameters through the member view as `m.<name>`.

## Named functions

`spec.function(name, fn, learnable=False, prior=None,
pointwise=False)` declares a named callable for coefficients that
are functions rather than scalars. `q.fn.<name>` is the declared
callable itself, called explicitly with the arguments it takes.
`jnp` operations broadcast over nodal arrays naturally, while an
equinox module requires `jax.vmap` to broadcast, so `pointwise=True`
applies the function at every node and keeps the closure natural.

```python
spec.function("drive", lambda t: jnp.sin(t))

@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.p.D * q.ops.laplacian(q.u) + q.fn.drive(q.t)
```


A learnable function is an `eqx.Module` whose array leaves are the
fitted coefficients, such as a
`diff_soup.inference.approx.LegendreExpansion` coefficient series. A
neural network is declared the same way, here as a learned reaction
rate:

```python
import equinox as eqx

r_nn = eqx.nn.MLP(in_size="scalar", out_size="scalar", width_size=16,
                  depth=2, activation=jax.nn.gelu, key=jax.random.PRNGKey(0))
spec.function("reaction", r_nn, learnable=True, pointwise=True)

@spec.equation(writes="u", reads=("u",))
def reaction_diffusion(q):
    return q.ops.laplacian(q.u) + q.fn.reaction(q.u)
```


## The size of a domain

`spec.geometry(name, ...)` declares a parameter that sets the size of a
one-dimensional grid. The grid is declared on the unit interval, and at
every evaluation its coordinates are multiplied by the parameter, so a
particle radius can be fitted like any other parameter:

```python
spec.domain("particle", spherical_grid_1d(32, 1.0))
R = spec.geometry("R", value=1.0, learnable=True, bounds=(0.1, 10.0))
```

- Closures see physical coordinates in `q.x`, and every operator acts on
  the scaled grid. A closure that needs the size itself reads `q.p.R`.
- `init=` and `spec.assemble_state` evaluate on the unit grid.
- Models that differ only in the size share one compiled program, and an
  experiment sets the size through `inputs={"R": r}`.

## Fitting declared parameters

A fit selects declared learnable parameters by their handles and
respects the declared bounds, and MAP estimation reads the declared
priors. `ds.inference.InferenceProblem(parameters=)` takes the handles
to fit, or `ds.ParameterSet(shared=, varying=)` for parameters tied
across experiments. The estimation objectives, parameter transforms,
and optimizers are covered with [inference](../inference/index.md).

## Recipe templates

`@ds.recipe` turns a function that returns a `ProblemSpec` into a
template, built once per distinct set of arguments and carrying one
handle per declared parameter across every build:

```python
@ds.recipe
def particle(R=100.0):
    spec = ds.problem("particle")
    spec.domain("sphere", spherical_grid_1d(24, R, method="fvm"))
    spec.field("c", mass="differential", scale=1.0, init=lambda r: 0.2 + 0.0 * r)
    spec.param("D", value=3e3, learnable=True, bounds=(1e2, 1e6))

    @spec.equation(writes="c", reads=("c",))
    def diffusion(q):
        return q.p.D * q.ops.laplacian(q.c)

    return spec


problem = particle.build(R=200.0)
D = particle.handle("D")
```

- The function's arguments describe the geometry or the protocol of one
  build. `particle.build(R=200.0)` builds the recipe for those values
  and returns the same problem on every later call with the same values.
- `particle.handle("D")` is the handle of a declared parameter or
  function, accepted by every build of the template and by an inference
  problem fitting several of them together
  ([hierarchical fits](../inference/hierarchical.md)). A declaration
  keeps its size and its `learnable=` flag across builds.
- A name no build declares is refused.
- `problem.assemble_state` on a built problem ignores `init=`. Pass the
  initial values to it, as
  [initial conditions](fields.md#initial-conditions) describes.
