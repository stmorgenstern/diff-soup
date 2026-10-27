"""The differential-algebraic (IDA) native time-stepping engine.

Implements a fully-implicit, variable-order, variable-step
backward-differentiation corrector for index-1 differential-algebraic
systems (the BDF family of DAE integrators; see Brenan, Campbell &
Petzold, Numerical Solution of Initial-Value Problems in DAEs, 1996).
The kernel is assembled from this package's modules; the public
surface is the [IDAEngine][diff_soup.timestepping.vsvo.engines.ida.step.IDAEngine]
step kernel and the [ida_engine][diff_soup.timestepping.vsvo.engines.ida.step.ida_engine]
factory the solver façade builds it through.
"""

from __future__ import annotations

from .step import IDAEngine, IDAEngineCarry, ida_engine

__all__ = ["IDAEngine", "IDAEngineCarry", "ida_engine"]
