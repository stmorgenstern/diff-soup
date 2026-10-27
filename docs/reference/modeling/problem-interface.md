# Problem interface

`spec.build()` turns a recipe into a `LoweredProblem`, which `ds.pdesolve`
and the inference verbs take. The [problem interface
overview](../../overview/modeling/problem-interface.md) builds one problem
three ways.

---

### Building

`to_equation_set()` freezes a recipe into an `EquationSet` and `lower`
compiles it.

::: diff_soup.problem.lower

::: diff_soup.problem.EquationSet
    options:
      members: false

::: diff_soup.problem.LoweredProblem
    options:
      members:
        - observables
        - observe
        - controls
        - default_theta
        - assemble_state
        - initial_u
        - state_size
        - global_layout
        - layout
        - precision
        - solver_problem

---

### Problem types

::: diff_soup.problem.MassProblem
    options:
      members: false

::: diff_soup.problem.ImplicitProblem
    options:
      members: false

::: diff_soup.problem.DAEStructure
    options:
      members: false

::: diff_soup.problem.ResidualSparsity

::: diff_soup.problem.ResidualLinearization
    options:
      members: false

::: diff_soup.problem.Dependencies

---

### Structure diagnostics

::: diff_soup.problem.topology_report

::: diff_soup.problem.TopologyReport
    options:
      members: false

::: diff_soup.problem.RowOwner
    options:
      members: false

---

??? abstract "Families and couplings"

    These matter only when you build an `EquationSet` by hand or write a
    custom transfer.

    ::: diff_soup.problem.Family
        options:
          members: false

    ::: diff_soup.problem.FamilyGraph
        options:
          members: false

    ::: diff_soup.problem.FamilyGraphExchange
        options:
          members: false

    ::: diff_soup.problem.TransferOp
        options:
          members:
            - apply
            - sparsity

    ::: diff_soup.problem.NodalExtractOp
        options:
          members: false

    ::: diff_soup.problem.NodalInjectOp
        options:
          members: false

    ::: diff_soup.problem.VarRef
        options:
          members: false

    ::: diff_soup.problem.Scales
        options:
          members: false
