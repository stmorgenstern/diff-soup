"""Native time-stepping engine entry points for the scalar solver suite.

Each function builds one of the three native engine kernels — the
numerical-differentiation-formula (NDF) engine, the
backward-differentiation-formula (BDF) engine, and the implicit
differential-algebraic (DAE) engine — from a solver's execution profile,
tolerances, and linear-backend choice, ready for the shared driver to run.

The solver entry points import this package lazily, which can happen inside
an active ``jax.jit`` trace (a user's first-ever solve wrapped in ``jit``).
Module-level code in this package must therefore execute NO jax operations —
not even ``jnp.asarray`` of a constant, whose device_put is staged as a
tracer under an active trace and leaks into the module global. Keep
module-level tables in host numpy and convert at traced use sites.
"""

from __future__ import annotations

from .bdf import bdf_engine
from .ida import ida_engine
from .ndf import ndf_engine

__all__ = ["ndf_engine", "bdf_engine", "ida_engine"]
