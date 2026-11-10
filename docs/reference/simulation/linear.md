# Linear solvers

Every Newton iteration solves one linear system, and the backend is what solves it. Pass
one as `linear=` on a solver: a name for the defaults, or the object with settings
([linear solvers](../../overview/simulation/linear.md)).

!!! tip "Recommended defaults"
    Leave `linear="auto"`, which picks KLU on a CPU host, cuDSS on a GPU, and `Dense` at
    or below ten unknowns. Pass a backend object only to change a setting.

---

### Direct backends

Each one is a name on `linear=` and a class here.

::: diff_soup.linear.Dense
    options:
      members: false

::: diff_soup.linear.KLU
    options:
      members: false

!!! warning "SuperLU_MT on more than one thread"
    SuperLU_MT's threaded factorization can return a wrong solution and
    report `SolveStatus.SUCCESS`, an upstream defect. A source build uses
    every core the process may run on unless `OMP_NUM_THREADS` caps it.
    Set `OMP_NUM_THREADS=1`, or use `KLU`.

::: diff_soup.linear.SuperLU
    options:
      members: false

::: diff_soup.linear._cudss._backend.CuDSS
    options:
      show_root_full_path: false
      members: false

::: diff_soup.linear.CuDSSOptions
    options:
      members: false

---

### Composed backends

These take arguments, so they are passed as objects.

::: diff_soup.linear.Krylov
    options:
      members: false

::: diff_soup.linear.BlockTridiagonal
    options:
      members: false

::: diff_soup.linear.MemberBlock
    options:
      members: false

::: diff_soup.linear.MemberBlockLayout
    options:
      members: false

::: diff_soup.linear.Woodbury
    options:
      members: false

::: diff_soup.linear.LowRankTerm
    options:
      members:
        - topology

---

??? abstract "Abstract base class"

    The details matter only when writing a backend of your own.

    ::: diff_soup.linear.LinearSolver
        options:
          members:
            - backend_name
            - platforms
            - factor_residency
            - transpose_is_free
            - vmap_factors
            - supported_precisions
            - is_exact
            - accepts_operator
