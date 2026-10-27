"""Row-degree, coloring, and ownership diagnostics of the iteration matrix.

The report answers the structural questions that drive sparse-solver cost:
how wide are the widest rows, how many colors does sparse-Jacobian recovery
need, and which variable's equations create the wide rows. Row ownership is
resolved from the problem's layout bookkeeping when present (recipe-built
problems carry it); a bare solver problem reports index-only owners.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class RowOwner(NamedTuple):
    """One row of a Jacobian pattern, with its nonzero count and owning variable.

    Attributes:
        row: Row index.
        degree: Number of nonzeros in the row.
        variable: Name of the field or global the row belongs to, or `None`.
    """

    row: int
    degree: int
    variable: Optional[str]


class TopologyReport(NamedTuple):
    """Summary of a problem's Jacobian sparsity pattern, returned by [`topology_report`][diff_soup.problem.topology_report].

    Attributes:
        n: Number of unknowns.
        nnz: Number of structural nonzeros.
        max_row_degree: Largest number of nonzeros in one row.
        mean_row_degree: Mean number of nonzeros per row.
        color_count: Colors needed to compute the Jacobian by automatic differentiation, or `None` for a dense pattern.
        widest_rows: [`RowOwner`][diff_soup.problem.RowOwner] records of the rows with the most nonzeros, widest first.
        per_variable_nnz: Nonzeros per field or global name, empty when the rows carry no names.
    """

    n: int
    nnz: int
    max_row_degree: int
    mean_row_degree: float
    color_count: Optional[int]
    widest_rows: tuple
    per_variable_nnz: dict


def _row_names(sources, n: int) -> list:
    """Per-row owning-variable names from layout bookkeeping, or ``None``s.

    Walks the candidate ``sources`` (a problem and any wrappers around it) and
    uses the first one that carries per-DOF layout entries.
    """
    names: list = [None] * n
    for owner_source in sources:
        if owner_source is None:
            continue
        entries = getattr(owner_source, "_layout_entries", ()) or ()
        if not entries:
            continue
        state_end = 0
        # Entry shape: ((component, variable), offset, size, field_shape).
        for entry in entries:
            (_, variable_name), offset, size = entry[0], int(entry[1]), int(entry[2])
            for k in range(offset, min(offset + size, n)):
                names[k] = str(variable_name)
            state_end = max(state_end, offset + size)
        # Global rows follow the state block: (name, offset, size) entries.
        for global_entry in getattr(owner_source, "_global_layout_entries", ()) or ():
            gname, offset, size = global_entry[0], int(global_entry[1]), int(global_entry[2])
            for k in range(state_end + offset, min(state_end + offset + size, n)):
                names[k] = str(gname)
        break
    return names


def topology_report(problem, *, top_k: int = 5) -> TopologyReport:
    """Summarize the Jacobian sparsity pattern a problem presents to the solvers.

    A problem without a sparse pattern reports a dense one.

    Args:
        problem: A [`LoweredProblem`][diff_soup.problem.LoweredProblem], [`MassProblem`][diff_soup.problem.MassProblem], or [`ImplicitProblem`][diff_soup.problem.ImplicitProblem].
        top_k: Number of widest rows to report.

    Returns:
        A [`TopologyReport`][diff_soup.problem.TopologyReport].
    """
    from diff_soup.problem._ad.sparse_jac import SparseJacobian
    from diff_soup.problem.implicit.solver_problem import get_or_build_solver_view
    from diff_soup.timestepping.vsvo.common.contract import coerce_problem

    solver_contract = coerce_problem(problem)
    view = get_or_build_solver_view(solver_contract)
    owner_sources = (problem, solver_contract, view.source)
    topo = view.full_topology
    if topo is None:
        n = int(view.problem.n)
        degrees = np.full((n,), n, dtype=np.int64)
        rows = None
        nnz = n * n
        color_count: Optional[int] = None
    else:
        n = int(topo.n_dofs)
        rows = np.asarray(topo.rows)
        cols = np.asarray(topo.cols)
        degrees = np.bincount(rows, minlength=n).astype(np.int64)
        nnz = int(rows.shape[0])
        color_count = int(
            SparseJacobian.from_coo(rows.astype(np.int32), cols.astype(np.int32), n).n_colors
        )

    names = _row_names(owner_sources, n)
    order = np.argsort(-degrees, kind="stable")[: max(int(top_k), 0)]
    widest = tuple(RowOwner(int(r), int(degrees[int(r)]), names[int(r)]) for r in order)

    per_variable_nnz: dict = {}
    if rows is not None and any(name is not None for name in names):
        name_array = np.array([name if name is not None else "" for name in names])
        uniq, counts = np.unique(name_array[rows], return_counts=True)
        per_variable_nnz = {str(u): int(c) for u, c in zip(uniq, counts) if u}

    return TopologyReport(
        n=n,
        nnz=nnz,
        max_row_degree=int(degrees.max()) if n else 0,
        mean_row_degree=float(degrees.mean()) if n else 0.0,
        color_count=color_count,
        widest_rows=widest,
        per_variable_nnz=per_variable_nnz,
    )
