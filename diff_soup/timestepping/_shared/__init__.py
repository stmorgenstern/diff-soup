"""Private helpers shared by the variable-step engines and consistent-init.

This package is private (underscore prefix). Public surface
(``domains/``, ``physics/``, ``optimization/``, ``problem/``) does not
import from here; the engines and the consistent-initialization machinery
import down into it.

Subpackages:
  - ``robustness/`` — algebraic-derivative recovery for the DAE consistent-IC
    projection (``derivative_ic``).
  - ``controllers/`` — the H211b step-size filter (``h211b``).
"""
