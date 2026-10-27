"""Consistent-initialization public errors.

Initialization itself belongs to the solver problem and is shared
by every engine through ``SolverView.consistent_initial_state``.
"""

from __future__ import annotations


class InconsistentInitializationError(ValueError):
    """Raised when a DAE cannot satisfy its initialization system."""
