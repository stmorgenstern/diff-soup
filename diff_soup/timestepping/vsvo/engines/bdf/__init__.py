"""The backward-differentiation-formula (BDF) native time-stepping engine.

Implements a variable-order, variable-step backward-differentiation-formula
(BDF) corrector for the mass-matrix ODE/DAE form, expressed against a
Nordsieck history. The kernel is
assembled from this package's modules; [bdf_engine][diff_soup.timestepping.vsvo.engines.bdf.step.bdf_engine] builds the public
step-kernel [BDFEngine][diff_soup.timestepping.vsvo.engines.bdf.step.BDFEngine] the shared driver runs.
"""

from __future__ import annotations

from .step import BDFEngine, BDFEngineCarry, bdf_engine

__all__ = ["BDFEngine", "BDFEngineCarry", "bdf_engine"]
