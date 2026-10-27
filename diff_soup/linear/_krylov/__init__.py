"""Iterative solves: restarted GMRES and the Krylov backend of the linear-solver contract.

* [Krylov][diff_soup.linear.Krylov] is the inexact backend: GMRES on the
  current operator, right-preconditioned by a held factor of an exact backend
  (cuDSS, SuperLU_MT, KLU, Dense), with the convergence report of
  [solve_checked][diff_soup.linear.solve_checked].
* [gmres][diff_soup.linear.gmres] runs the same iteration on a
  [LinearOperator][diff_soup.linear.LinearOperator] with any preconditioner.
* [transposed_topology][diff_soup.linear._krylov.transposed_topology] is the
  pattern of the transposed assembled operator, on which a factor of
  orientation ``"AT"`` holds its preconditioner.
* [exact_backend][diff_soup.linear._krylov.exact_backend] and
  [exact_analysis][diff_soup.linear._krylov.exact_analysis] give the exact
  backend behind a Krylov backend, for the consumers that need exact inverses.
* [record_reports][diff_soup.linear._krylov.record_reports] is a diagnostic
  that collects every Krylov solve's status, iterations and residual on the
  host.
"""

from diff_soup.linear._krylov._backend import (
    Krylov,
    exact_analysis,
    exact_backend,
    record_reports,
    report_outcome,
)
from diff_soup.linear._krylov._gmres import gmres, gmres_core
from diff_soup.linear._krylov._transposed import transposed_topology, transposed_values

__all__ = [
    "Krylov",
    "exact_analysis",
    "exact_backend",
    "gmres",
    "gmres_core",
    "record_reports",
    "report_outcome",
    "transposed_topology",
    "transposed_values",
]
