# Constraints

A constraint closes a global unknown by driving a quantity computed from
the state to a target. The [constraints overview](../../overview/modeling/constraints.md)
shows each form with an example.

::: diff_soup.problem._spec.constraints_namespace._ConstraintNamespace
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

---

??? abstract "Constraint records"

    These matter only when you build an `EquationSet` by hand.

    ::: diff_soup.problem.Constraint
        options:
          members: false

    ::: diff_soup.problem.GalvanostaticConstraint
        options:
          members: false

    ::: diff_soup.problem.DomainIntegralConstraint
        options:
          members: false

    ::: diff_soup.problem.ReduceResidualConstraint
        options:
          members: false

    ::: diff_soup.problem.BoundaryIntegralConstraint
        options:
          members: false

    ::: diff_soup.problem.OperationalControlConstraint
        options:
          members: false
