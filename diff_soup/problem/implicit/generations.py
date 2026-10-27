"""Generation counters and the exact-root certificate (shared re-export).

The contract itself lives in [`diff_soup.nonlinear.generations`][diff_soup.nonlinear.generations], the
shared solve-support layer, so solver engines and the solver problem
layer consume one definition. This module re-exports it for solver-problem
callers.
"""

from __future__ import annotations

from diff_soup.nonlinear.generations import (
    ExactRootCertificate,
    Generations,
    assert_certified,
    certificate_authorizes,
    certify_exact_root,
    initial_generations,
)

__all__ = [
    "ExactRootCertificate",
    "Generations",
    "assert_certified",
    "certificate_authorizes",
    "certify_exact_root",
    "initial_generations",
]
