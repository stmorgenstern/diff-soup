"""Sparse Jacobian assembly via graph coloring for assembled problems.

Provides efficient sparse Jacobian evaluation for FEM/FDM/FVM problems
using Curtis-Powell-Reid (CPR) coloring to minimise the number of JVPs
needed to recover all nonzero entries.

Scope and boundary
------------------

This subpackage computes the **primal** sparse Jacobian values that the
Newton linear solve consumes. It does *not* define the autodiff rules
that propagate tangents and cotangents *through* a nonlinear or
time-stepping solver. Those custom VJP / JVP rules live alongside the
solvers themselves ([solver][],
[box_newton][], and the
[vsvo][] engines) and implement the implicit
function theorem boundary: rather than differentiating through the inner
Newton / NDF iterations (which would accumulate huge intermediate graphs
and numerical noise from the iteration transient), the custom rules
solve one additional *linearised* system at the converged primal to
obtain the tangent (forward mode) or cotangent (reverse mode). This
subpackage supplies the same sparse Jacobian structure used by both the
primal solves and those implicit-rule linearisations, but the rules
themselves are authored where the solver state is owned.

Usage::

    from diff_soup.problem._ad import SparseJacobian

    # From FEM topology
    sjac = SparseJacobian.from_coo(rows, cols, n_dofs)

    # Evaluate sparse Jacobian of residual w.r.t. state
    values = sjac.evaluate(residual_fn, u)

    # Can also use with jacfwd for parameter Jacobians
    J_sparse = sjac.evaluate(lambda u: residual(u, params), u)
"""

from .sparse_jac import SparseJacobian

__all__ = [
    "SparseJacobian",
]
