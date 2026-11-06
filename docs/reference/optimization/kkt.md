# KKT solves

Each interior point and SQP step solves one KKT system, and the backend is what solves it. Pass one as `kkt=` on `InteriorPoint` or `SQP` ([constrained solvers](solvers.md)).

!!! tip "Recommended default"
    Leave `kkt=DenseKKT()`, the only backend so far.

---

### Backends

::: diff_soup.optimization.solvers.DenseKKT
    options:
      members: false

---

??? abstract "Abstract base class"

    The details matter only when writing a KKT backend of your own.

    ::: diff_soup.optimization.solvers.KKTSolve
        options:
          members:
            - factor
            - solve_factored
            - solve
