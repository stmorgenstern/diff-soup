"""Private helpers shared by the variable-step engines.

This package is private (underscore prefix). Public surface
(``domains/``, ``physics/``, ``optimization/``, ``problem/``) does not
import from here; the engines import down into it.

Subpackages:
  - ``controllers/`` — the H211b step-size filter (``h211b``).
"""
