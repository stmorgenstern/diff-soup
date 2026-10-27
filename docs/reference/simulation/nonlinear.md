# Nonlinear solvers

`Newton` and `BoxConstrainedActiveSet` are standalone solvers for a bordered
system `R(u, lam) = 0`, `g(u, lam) = 0`. The engines of `ds.pdesolve`
run their own corrector, described in [nonlinear
solvers](../../overview/simulation/nonlinear.md).

!!! warning
    `BoxConstrainedActiveSet` is experimental.

::: diff_soup.Newton
    options:
      members: false

::: diff_soup.BoxConstrainedActiveSet
    options:
      members:
        - config
        - bounds

::: diff_soup.nonlinear.BoxBounds

---

??? abstract "diff_soup.nonlinear.NonlinearSolver"

    The details matter only when writing a corrector of your own.

    ::: diff_soup.nonlinear.NonlinearSolver
        options:
          members:
            - config
            - linear
            - adjoint
            - init
            - step
            - terminate
            - solve

    ::: diff_soup.nonlinear.NewtonInfo
        options:
          members: false
