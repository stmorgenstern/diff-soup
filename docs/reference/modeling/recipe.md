# Recipe API

`ds.problem(name)` opens a `ProblemSpec`, which collects a recipe's
declarations until `build()`. The [Recipe API
overview](../../overview/modeling/recipe-api.md) shows the pattern with
examples.

!!! tip "Recommended defaults"
    Most recipes need only `domain`, `field`, `equation`, `bc` and `build`.
    Set each field's `scale` to its typical magnitude and leave `offset` at
    its default. `build` runs the same static validation as `spec.validate()`.

::: diff_soup.problem.problem

::: diff_soup.problem.ProblemSpec
    options:
      members:
        - domain
        - field
        - global_var
        - param
        - geometry
        - function
        - family
        - equation
        - equations
        - couple
        - time_scale
        - scales
        - equation_scale
        - build
        - to_equation_set
        - assemble_state
        - validate

---

### Boundary conditions

`spec.bc` holds one method per [boundary
condition](../../overview/modeling/boundary-conditions.md).

::: diff_soup.problem._spec.bcs_namespace._BCNamespace
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

::: diff_soup.problem.tags

::: diff_soup.problem.BoundarySet
    options:
      members:
        - reduce
        - from_nodes

---

### Observations

`spec.observe` declares the quantities a solution reports and a fit
compares with data.

::: diff_soup.problem._spec.observe_namespace._ObserveNamespace
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

---

### Controls

`spec.control` turns a forcing
[protocol](../../overview/modeling/controls.md) into a value the solve reads
as it runs.

::: diff_soup.problem._spec.control_namespace._ControlNamespace
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

---

### Families

`spec.family(...)` declares a population of identical members and returns
the object that declares their fields, equations and boundary conditions.

::: diff_soup.problem._spec.family_builder._FamilyHandle
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

---

### Parameters

`spec.param` declares a [parameter](../../overview/modeling/parameters.md),
read in closures as `q.p.<name>`.

::: diff_soup.problem.ParameterSet
    options:
      members: false

::: diff_soup.problem.default_theta

::: diff_soup.problem.implicit.solver_lowering._IRTheta
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      filters: []
      members:
        - __getitem__
        - replace

::: diff_soup.inference.learnables

---

### Mass operators

`MassOperator` declares a non-diagonal mass block inside a residual-form
equation, as in `op(q.dt("u")) - flux(q)`.

::: diff_soup.problem.MassOperator
    options:
      members:
        - identity
        - diagonal
        - sparse
        - block_sparse
        - dense
        - from_apply
