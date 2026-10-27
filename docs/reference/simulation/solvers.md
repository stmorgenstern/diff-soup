# Solvers

`JNDF`, `JBDF`, and `JIDA` are diff-soup's variable-step, variable-order
implicit engines, and `DiffraxStepper` wraps an explicit Diffrax solver
for identity-mass problems. See [ODE/DAE
solvers](../../overview/simulation/solvers.md) for how to choose.

!!! tip "Recommended defaults"
    Construct engines through the `stock` preset, the profile of every bare
    constructor, as in `ds.JNDF.stock(rtol=1e-6, atol=1e-6, linear="auto")`.
    The `flash` preset replaces Jacobian reuse across steps with a fresh
    Jacobian each corrector iteration. Leave `profile` to the preset.

---

### Implicit engines

::: diff_soup.JNDF
    options:
      members:
        - adjoint
        - stock
        - flash
        - solve
      inherited_members:
        - stock
        - flash
        - solve

::: diff_soup.JBDF
    options:
      members:
        - adjoint
        - stock
        - flash
        - solve
      inherited_members:
        - stock
        - flash
        - solve

::: diff_soup.JIDA
    options:
      members:
        - consistent_ic
        - ic_mode
        - suppress_alg
        - id
        - adjoint
        - stock
        - flash
        - solve
      inherited_members:
        - stock
        - flash
        - solve

??? abstract "Settings shared by JNDF, JBDF, and JIDA"

    The three engines take the same keyword settings.

    ::: diff_soup.timestepping.vsvo.classes._VSVOSolver
        options:
          show_root_heading: false
          show_root_toc_entry: false
          members:
            - profile

    ::: diff_soup.timestepping.vsvo.classes.ExecutionProfile
        options:
          show_root_full_path: false
          members: false

---

### Explicit Diffrax solvers

::: diff_soup.timestepping.diffrax_wrapper.DiffraxStepper
    options:
      show_root_full_path: false
      members:
        - inner
        - rtol
        - atol
        - dt0
        - max_steps
        - adjoint
        - solve

---

### Tolerances

`ToleranceSpec` is the per-solve tolerance request of `JNDF`, `JBDF`, and
`JIDA`, and `ResolvedTolerances` is the record a solve returns on
`Solution.tolerances`.

::: diff_soup.ToleranceSpec
    options:
      members: false

::: diff_soup.ResolvedTolerances
    options:
      members: false
