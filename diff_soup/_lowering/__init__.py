"""Lowering-analysis provider package.

``diff_soup/_lowering`` bridges the problem IR (``diff_soup.problem._ir``) and
the domain operator bank (``diff_soup.domains.operators``).
``diff_soup.problem._ir.lower`` calls [analyze][diff_soup._lowering.provider.analyze]
to build the residual, Jacobian, sparsity and mass kernels of an equation set,
importing it on first call. See [diff_soup._lowering.provider][] for the
implementation.
"""

from __future__ import annotations

from .provider import analyze

__all__ = ["analyze"]
