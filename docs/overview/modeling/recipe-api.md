# The Recipe API

This page provides an overview of building a problem using diff-soup's
Recipe API. Each step links to the page covering its options in detail.

## Build a problem

The running example is the heat equation on the unit interval with
Dirichlet boundary conditions and a quadratic initial condition:

$$
\frac{\partial u}{\partial t} = \nabla^2 u, \qquad x \in (0, 1),
$$

$$
u(0, t) = u(1, t) = 0, \qquad u(x, 0) = x(1 - x).
$$

Discretizing this equation in space using 64 finite-volume cells turns
this system into 64 coupled ODEs. The full recipe, which turns on JAX's
64-bit mode first because the default solvers and the cuDSS backend assume
it (see [enable float64](../../installation/index.md#enable-float64)):

```python
import numpy as np
import diff_soup as ds
import jax
from diff_soup.domains import cartesian_grid_1d

jax.config.update("jax_enable_x64", True)

n = 64
spec = ds.problem("heat")                                         # open the builder
spec.domain("rod", cartesian_grid_1d(n, 0.0, 1.0, method="fvm"))  # 64 finite-volume cells on [0, 1]
spec.field("u", mass="differential", scale=1.0, init=lambda x: x * (1.0 - x))

@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.laplacian(q.u)                                   # du/dt = lap(u)

spec.bc.dirichlet("u", node_indices=np.array([0, n - 1]), value=0.0)

problem = spec.build()                                            # lower the recipe
y0 = spec.assemble_state({})                                      # assemble the initial state
```

### Open the problem

`ds.problem` opens a mutable builder for the problem, with the
problem's name as its argument.

```python
spec = ds.problem("heat")
```

### Specify the domain

We first specify the domain, the spatial discretization the unknowns
live on, registered under a name. The factories and their options are
covered in [domains and discretizations](domains.md).

```python
n = 64
spec.domain("rod", cartesian_grid_1d(n, 0.0, 1.0, method="fvm"))
```

### Define the fields

We then define the relevant variables, called fields. A field
declaration provides the name, the mass type (`"differential"` for
variables with time derivatives and `"algebraic"` for variables without
explicit time derivatives), the scale (the variable's typical physical
magnitude, used internally as `state_SI = scale * state_internal`), and
the variable's initial condition. An algebraic variable's initial
condition is only a guess and is made consistent automatically at the
start of the solve. See [fields and globals](fields.md).

```python
spec.field("u", mass="differential", scale=1.0, init=lambda x: x * (1.0 - x))
```

### Write the equation

Next we write one equation per field, as a plain Python function of the
access object `q`: `q.u` is the field's current state and
`q.ops.laplacian` is the discretized Laplacian for the field's domain.

By default, the equation expects an equation of the form
\(M(u, t)\,du/dt = f(u, t\,;\, \text{params}, \text{controls})\), but you
can also specify a fully implicit equation of the form
\(F(u, du/dt, t\,;\, \text{params}, \text{controls}) = 0\) by
including a `q.dt(field)` term in the equation. Both forms and the
access object are covered in [equations](equations.md), and the
operator bank in [operators](operators.md).

```python
@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.laplacian(q.u)             # du/dt = lap(u)
```

### Set the boundary conditions

The boundary condition pins \(u = 0\) at the first and last cells, the
two ends of the rod. On a 1-D grid, boundary locations are selected by
node index. The condition kinds, mesh tag selection, and nonlinear
conditions are covered in
[boundary conditions](boundary-conditions.md).

```python
spec.bc.dirichlet("u", node_indices=np.array([0, n - 1]), value=0.0)
```

### Build the problem

Finally, `build` lowers the recipe into the solve-ready problem, and
`assemble_state` renders the declared initial conditions into the flat
state vector a solve takes as its argument.

```python
problem = spec.build()
y0 = spec.assemble_state({})
```

A [recipe template](parameters.md#recipe-templates) such as `particle`
returns the built problem from `build(**inputs)`, and a built problem's `assemble_state`
ignores `init=`. Its `spec(**inputs)` returns the spec instead, which
renders the declared initial conditions:

```python
y0 = particle.spec(R=200.0).assemble_state({})
```
