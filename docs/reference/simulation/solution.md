# Solution

Every solver returns a `Solution`, with its counters in a `SolverStats` and
its outcome as a `RESULT`. See [solutions and
output](../../overview/simulation/solutions.md).

!!! warning
    No solver raises on a non-convergent solve, so read `sol.ok` or
    `sol.result` before trusting the arrays.

::: diff_soup.Solution
    options:
      members:
        - ts
        - ys
        - lambdas
        - valid_mask
        - ok
        - result
        - stats
        - tolerances
        - event_log
        - control_state
        - interpolation
        - outputs
        - evaluate
        - trimmed

---

::: diff_soup.SolverStats
    options:
      members: false

::: diff_soup.RESULT
    options:
      members: false
