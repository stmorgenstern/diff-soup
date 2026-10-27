# The factor contract

These functions factor and solve a sparse matrix in code of your own, with any backend
([linear solvers](../../overview/simulation/linear.md)).

```python
import jax.numpy as jnp
import numpy as np
from diff_soup.linear import Dense, SparseTopology, analyze, factor, solve

rows, cols = np.nonzero(np.ones((3, 3)))
analysis = analyze(SparseTopology(rows=rows, cols=cols, n_dofs=3), backend=Dense())
f = factor(analysis, jnp.asarray([4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 3.0]))
x, f = solve(f, jnp.asarray([1.0, 2.0, -1.0]))
x_t, f = solve(f, jnp.asarray([1.0, 2.0, -1.0]), transpose=True)
```

---

### Functions

Each function on a factor returns the factor to use next.

::: diff_soup.linear.analyze

::: diff_soup.linear.factor

::: diff_soup.linear.empty_factor

::: diff_soup.linear.refactor

::: diff_soup.linear.solve

::: diff_soup.linear.solve_checked

::: diff_soup.linear.invalidate

::: diff_soup.linear.refine

::: diff_soup.linear.after

---

### Types

::: diff_soup.linear.SparseTopology

::: diff_soup.linear.Analysis
    options:
      members: false

::: diff_soup.linear.Factor
    options:
      members: false

::: diff_soup.linear.SolveContext
    options:
      members:
        - release
        - resident_bytes

::: diff_soup.linear.SolveReport
    options:
      members: false

::: diff_soup.linear.SolveStatus
    options:
      members: false

::: diff_soup.linear.Tolerance
    options:
      members: false

::: diff_soup.linear.RefineReport
    options:
      members: false

::: diff_soup.linear.FactorContractError
    options:
      members: false

---

### Matrix-free operators

The iterative solves work on a matrix given by its action.

::: diff_soup.linear.LinearOperator

::: diff_soup.linear.operator_from_values

::: diff_soup.linear.operator_from_jvp

::: diff_soup.linear.gmres
