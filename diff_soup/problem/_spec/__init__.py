# Recipe API spec builder + lower() seam.
"""diff_soup Recipe API spec builder + ``lower()`` seam.

The Recipe API is the high-level authoring surface for PDEs and DAEs.
Users write a *recipe* using ``ds.problem(...)`` chaining; the builder
lowers it to the EquationSet IR::

    spec = ds.problem("heat").domain("omega", mesh)
    c    = spec.field("c", mass="differential")

    @spec.equation(writes="c", reads=("c",))
    def heat(q):
        return -q.p.D * q.ops.laplacian(q.c)

    spec.bc.dirichlet("c", node_indices=..., value=0.0)
    eqset = spec.to_equation_set()

The spec records here (FieldSpec, EquationSpec, BCSpec, ...) are mutable
host-side records consumed by ``spec.to_equation_set()`` to emit a frozen
``EquationSet``. ``lower()`` transforms a frozen ``EquationSet`` into a
solve-ready ``LoweredProblem``.

This subpackage was promoted from a single-file stub into a package so
each module stays focused and readable.
"""

from __future__ import annotations

from diff_soup.problem._ir.lower import lower

from .problem_spec import ProblemSpec, problem

__all__ = ["ProblemSpec", "problem", "lower"]


# ---------------------------------------------------------------------------
# lower() — the EquationSet → LoweredProblem transform. Re-exported here so
# ``problem.__init__`` / ``diff_soup.__init__`` keep the same ``lower`` binding.
# The residual / Jacobian / sparsity kernel construction is injected via
# ``_ir.lower.set_lowering_ops`` (the dependency-injection seam for the
# discretization provider).
# ---------------------------------------------------------------------------
