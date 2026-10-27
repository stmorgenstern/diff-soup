"""Lowering-analysis provider package.

``diff_soup/_lowering`` is the package that bridges the high-level problem IR
(``diff_soup.problem._ir``) with the L1 domain operator bank
(``diff_soup.domains.operators``).  Importing this package registers the
same-domain analysis provider with ``lower()`` via
``diff_soup.problem._ir.lower.set_lowering_ops`` — the single module-level seam
that lets ``lower()`` build residual / Jacobian / sparsity / mass kernels from
the domain operator bank.

The top-level ``diff_soup`` package imports this module at startup so the seam
is always wired.  See [diff_soup._lowering.provider][] for the provider
implementation.
"""

from __future__ import annotations

from diff_soup.problem._ir.lower import set_lowering_ops

from .provider import analyze

set_lowering_ops(analyze)

__all__ = ["analyze"]
