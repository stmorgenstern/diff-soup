# Modeling

diff-soup is built to solve time-dependent PDEs by discretizating the equations in space on a given domain. We provide a flexible front-end for defining coupled, nonlinear PDEs to describe complex physical systems.
In this section, we provide an overview of diff-soup's modular modeling system:

- [The Recipe API](recipe-api.md) is the authoring api
- The core modules and recipe API primitives:
  [domains and discretizations](domains.md),
  [fields and globals](fields.md),
  [equations](equations.md),
  [operators](operators.md),
  [boundary conditions](boundary-conditions.md),
  [constraints](constraints.md),
  [learnable parameters](parameters.md), and
  [controls](controls.md).
- [The problem interface](problem-interface.md) contains the final lowered system that the solvers consume and the alternative, custom methods for defining problems through physics agnostic approaches.
