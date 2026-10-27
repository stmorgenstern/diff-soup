# The problem interface

This page provides an overview of the problem, the seam between
model authoring and the engines. A built recipe lowers to a problem,
and the same problem can be authored by hand.

## The problem types

`spec.build()` lowers a recipe into a `LoweredProblem` carrying the
problem the solver consumes, in one of the two forms
[equations](equations.md#the-two-forms) lower to:

- `MassProblem` states the mass-matrix form
  \(M(u, t)\,\frac{du}{dt} = f(u, t\,;\, \text{params}, \text{controls})\).
  A recipe whose equations are linear in the derivative lowers here.
- `ImplicitProblem` states the fully implicit form
  \(F\!\left(u, \frac{du}{dt}, t\,;\, \text{params}, \text{controls}\right) = 0\),
  for systems nonlinear in the derivative.

The problems carry the metadata for variable types and the Jacobian
sparsity pattern.

`problem.observables` lists the declared observations, and
`problem.observe(u, t, names=...)` evaluates a chosen subset of them.

The engines accept a `MassProblem` or `ImplicitProblem` directly, or
any object with a callable `solver_problem()` hook returning one.
A built recipe implements the hook, so it drops straight into
`pdesolve`, and the hook is the extension point for authoring
outside the Recipe API.
The result of the hook is cached on the model's content together with
its sparsity, colorings, and linearization plan, so equal problems,
rebuilt ones included, reuse compiled programs and cached factorization
structure.

## The heat equation, three ways

The running example is the heat equation on a periodic interval,

$$
\frac{\partial u}{\partial t} = \nabla^2 u, \qquad x \in [0, 1),
\qquad u(x + 1) = u(x).
$$

Discretized on \(n\) finite-volume cells of width \(dx\), each cell
couples to its two neighbors, wrapping around at the ends. The three
declarations below build the same discrete system.

### A recipe

```python
n = 64
spec = ds.problem("heat")
spec.domain("ring", cartesian_grid_1d(n, 0.0, 1.0, method="fvm"))
spec.field("u", mass="differential", scale=1.0)

@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.laplacian(q.u)

spec.bc.periodic("u")
problem = spec.build()
```

The lowering derives the stencil sparsity from the domain and lowers
the recipe to a `MassProblem`.

### A hand-authored MassProblem

The same system authored directly states the wraparound stencil and
declares its sparsity pattern, one (row, column) pair per nonzero of
\(\partial f / \partial u\):

```python
from diff_soup.linear import SparseTopology

dx = 1.0 / n

def rhs(t, u, theta, control):
    return (jnp.roll(u, 1) - 2.0 * u + jnp.roll(u, -1)) / dx**2

idx = np.arange(n)
rows = np.concatenate([idx, idx, idx])
cols = np.concatenate([(idx - 1) % n, idx, (idx + 1) % n])

heat_mass = ds.MassProblem(
    rhs=rhs,
    mass=ds.MassOperator.identity(n),
    structure=ds.DAEStructure(differential_vars=np.ones(n, dtype=bool)),
    rhs_sparsity=SparseTopology(rows=rows, cols=cols, n_dofs=n),
)

sol = ds.pdesolve(heat_mass, ds.JNDF(), (0.0, 0.05), u0)
```

- `rhs(t, u, theta, control)` returns one value per unknown. The
  assembled residual is \(M\,du/dt - \text{rhs}\), so an algebraic
  equation \(g = 0\) enters `rhs` as \(-g\).
- `mass` is the mass-matrix operator, a
  `ds.MassOperator` applying \(M\) to \(du/dt\), built by the
  factories `identity(n)`, `diagonal`,
  `sparse(rows, cols, values, n=...)`, `block_sparse`, and
  `from_apply`. The operator declares coefficients on the
  differential rows. A state- or time-dependent \(M(u, t)\) declares its
  dependencies and the sparsity of
  \(\partial(M\,du/dt)/\partial u\).
- `structure` is `ds.DAEStructure(differential_vars=...)`, one
  boolean per unknown. A `("global", lo, hi)` partition marks
  appended rows for the [global-block](fields.md#globals) treatment.
- `rhs_sparsity` holds the (row, column) pairs of
  \(\partial f/\partial u\). Passing `"dense"` (the default)
  differentiates the right-hand side densely instead.
- `theta` is whatever the callables accept, passed to `pdesolve` as
  `theta=`.

### A hand-authored ImplicitProblem

The implicit form states the residual directly and declares both
Jacobian supports, \(\partial F/\partial u\) and
\(\partial F/\partial (du/dt)\):

```python
heat_implicit = ds.ImplicitProblem(
    residual=lambda t, u, dudt, theta, control: dudt - rhs(t, u, theta, control),
    structure=ds.DAEStructure(differential_vars=np.ones(n, dtype=bool)),
    linearization=ds.ResidualLinearization(
        sparsity=ds.ResidualSparsity(
            dy=SparseTopology(rows=rows, cols=cols, n_dofs=n),
            dydot=SparseTopology(rows=idx, cols=idx, n_dofs=n),
        )
    ),
)
```

- `residual(t, u, dudt, theta, control)` states \(F = 0\), here
  \(du/dt - \text{stencil} = 0\). The form exists for what the
  mass-matrix form cannot state: systems nonlinear in the
  derivative.
- `linearization` is required. `ResidualSparsity(dy=..., dydot=...)`
  declares the supports of \(\partial F/\partial u\) and
  \(\partial F/\partial (du/dt)\), each a `SparseTopology` or
  `"dense"`. Here `dydot` is the diagonal.
- The fully implicit engine `ds.JIDA` solves it. The mass-form
  engines reject it before tracing.

## Scales and non-dimensionalization

The solver works with internal values near one, related to the dimensional
state by the declared scales:

$$
u_{\text{dimensional}} = \text{offset} + \text{scale} \cdot u_{\text{internal}},
\qquad
t_{\text{dimensional}} = \tau \cdot t_{\text{internal}}.
$$

- Scales are declared inline
  ([`spec.field(scale=, offset=, residual_scale=)`](fields.md#scaling),
  `spec.global_var(scale=)`) or centrally:
  `spec.scales(state={...}, state_offset={...}, time=...)` sets them
  by variable name and overrides the inline declarations, warning on
  a conflict. `spec.time_scale(tau)` sets the time scale, and
  `spec.equation_scale(field, value)` sets a field's residual
  scaling centrally.
- Conversion happens at the boundaries. `assemble_state` renders the
  dimensional state, and the solve converts it on entry. Inside equation
  closures every read is internal: `q.<field>` and `q.t` are the
  internal values, and a closure that needs a dimensional quantity rescales
  explicitly. [Observations](equations.md#observations) convert back
  to dimensional.
- Solution arrays are internal. `sol.ys` returns the internal state,
  and observations are the dimensional route out.
- `spec.time_scale` divides the differential rows of the mass
  operator, so the solver steps in internal time.

## Consistent initialization

Consistent initialization for DAEs is automatic during a solve and
is covered in
[consistent initialization](../simulation/consistent-init.md). Code
that needs an explicit projected state uses
`diff_soup.problem.implicit.consistent_initial_state` with the
problem, the time, the complete augmented state, and the
parameters, where `theta=None` resolves the problem's declared
values. The result carries the projected state, the derivative,
a convergence flag, the iteration count, and the residual norm, and
an unconverged result must not be consumed.

The [problem interface reference](../../reference/modeling/problem-interface.md)
documents the problem types themselves;
[Simulation](../simulation/index.md) describes what the engines do
with them.
