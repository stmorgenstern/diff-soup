# Adjoints

An adjoint selects how derivatives flow through a solve, passed as `adjoint=`
on `ds.pdesolve`. See
[sensitivities and autodiff](../../overview/simulation/sensitivities.md).

### Recommended

`ForwardAdjoint` is the default, and `CheckpointAdjoint` supports `jax.grad`
and `jax.vjp`.

::: diff_soup.ForwardAdjoint
    options:
      members: false

::: diff_soup.CheckpointAdjoint
    options:
      members: false

---

### Nonlinear solves

::: diff_soup.nonlinear.ImplicitAdjoint
    options:
      members: false

---

??? abstract "diff_soup.nonlinear.AbstractAdjoint"

    The details matter only when writing a strategy of your own.

    ::: diff_soup.nonlinear.AbstractAdjoint
        options:
          members:
            - ift_mode
            - compositions
            - apply
