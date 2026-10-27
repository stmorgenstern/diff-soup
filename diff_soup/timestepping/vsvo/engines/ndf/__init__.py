"""The numerical-differentiation-formula (NDF) engine.

Implements the variable-order (1-5), quasi-constant-step-size NDF method of
Shampine & Reichelt (1997, SIAM J. Sci. Comput. 18(1)) with the adaptive
step/order controller of Alberdi Celaya, Anza Aguirrezabala &
Chatzipantelidis (2014). The implementation follows the QNDF solver of
OrdinaryDiffEq.jl and its simplified-Newton layer (MIT license; see
``THIRD_PARTY_NOTICES.md``); the dense output uses the textbook Newton
backward-difference interpolant in the product form SciPy's BDF dense
output uses (BSD-3).

The engine plugs into the shared VSVO driver through the ``Engine``
protocol:
[ndf_engine][diff_soup.timestepping.vsvo.engines.ndf.step.ndf_engine] builds
the step kernel from a solver execution
profile, tolerances, and a linear-backend name.
"""

from __future__ import annotations

from .step import NDFEngine, NDFEngineCarry, ndf_engine

__all__ = ["NDFEngine", "NDFEngineCarry", "ndf_engine"]
