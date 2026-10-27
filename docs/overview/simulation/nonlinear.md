# Nonlinear solvers

This page provides an overview of the nonlinear corrector inside the
time-stepping engines.

Each trial step's multistep formula turns the governing equations into
a nonlinear algebraic system for the new state, and Newton's method is
the corrector. Each iteration solves the linearized system built from
the iteration matrix, the weighted combination of the mass matrix and
the state Jacobian that the step size scales. The state block solves
through the configured [linear backend](linear.md), and the bordered
global block closes through a small dense Schur complement. The
corrector runs a few bounded Newton iterations per step attempt,
accepting on a weighted norm of the update informed by the observed
convergence rate and declaring divergence when the rate approaches
one.

## Reusing the linearization

Forming the Jacobian and factorizing the iteration matrix are the
dominant costs of the whole integration, so the engines freeze both as
long as they can. A setup gate re-forms them on the first step, after
a convergence failure, or when the step-size scaling drifts out of a
window around the value frozen into the factors. In between, the
cached factors serve every Newton iterate, with the update rescaled to
compensate the drifted coefficient. The `flash` preset re-forms them
every iteration instead ([ODE/DAE solvers](solvers.md)).
`Solution.stats` counts the economy directly: `newton_iters`,
`jac_refreshes`, `factorizations`, and `linear_solves`.

## Failure

A corrector failure restores the step history, shrinks the step, and
retries with a fresh setup. A failure that persists terminates the
solve as `NONLINEAR_DIVERGED` ([solutions and output](solutions.md)).

## Box constraints

A box-constrained corrector, enforcing state bounds through an
active-set Newton iteration, is experimental and planned for a future
version.
