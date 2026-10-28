# pdesolve

`pdesolve` integrates a problem with a solver over a time span and
returns its [Solution](solution.md), and `compile` compiles that call
without running it. See [running a
simulation](../../overview/simulation/running-a-simulation.md).

!!! tip "Recommended defaults"
    Most calls pass a problem, a solver, the span, and the initial state,
    plus `saveat=` when the final state alone is not enough. Leave
    `adjoint=None` unless you need `jax.grad` through the solve. Leave
    `lambda0`, `consistent_ic`, and `control_state0` unset unless a DAE's
    initialization or an operating protocol demands them.

::: diff_soup.pdesolve.pdesolve

::: diff_soup.compile

::: diff_soup._compile.CompiledSolve
    options:
      show_root_full_path: false
      members:
        - __call__
        - cost_analysis
        - memory_analysis
