# Consistent initialization

A DAE solve starts from a state that satisfies its algebraic equations.
See [consistent
initialization](../../overview/simulation/consistent-init.md).

::: diff_soup.problem.implicit.consistent_init.consistent_initial_state

::: diff_soup.problem.implicit.consistent_init.check_consistent_init

::: diff_soup.problem.implicit.consistent_init.ConsistentInitResult
    options:
      members: false

---

### Algebraic projection

`project_algebraic` solves the algebraic entries of a state assembled or
edited outside a solve.

::: diff_soup.project_algebraic

::: diff_soup.AlgebraicProjectionOptions
    options:
      members: false

::: diff_soup.AlgebraicProjectionResult
    options:
      members: false

::: diff_soup.AlgebraicProjectionStatus
    options:
      members: false

::: diff_soup.AlgebraicProjectionError
    options:
      members: false
