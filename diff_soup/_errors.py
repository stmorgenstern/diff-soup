"""Error taxonomy: two error regimes.

Host-time failures (bad composition, unsupported capability, malformed spec,
missing optional dep) raise these typed exceptions EAGERLY at the boundary
(``__check_init__``, role materialization, ``analyze``, ``lower()``) — never
deep in a step loop. Trace-time failures use traced status codes instead
(``RESULT`` at the linear level, ``Solution.status`` at the integrator level).

Principles: error identity is always type (host) or code (traced) — never
substring dispatch. Messages state what was violated and what would satisfy it.
Flat hierarchy, no deep trees. Root-private leaf.
"""

from __future__ import annotations


class DiffSoupError(Exception):
    """Base class of the errors specific to diff-soup."""


class ConfigurationError(DiffSoupError):
    """Raised when an argument, an option, or a combination of them is invalid."""


class LoweringError(DiffSoupError):
    """Raised when `spec.build()` or `lower` cannot turn its input into a problem.

    A typical cause is a boundary function that reads a field its component
    neither owns nor couples in. The message names the declaration and the fix.
    """


class BackendError(DiffSoupError):
    """Raised when a batched evaluation of a study fails while it runs, for example when device memory runs out."""


class OptionalDependencyError(DiffSoupError, ImportError):
    """Raised when a feature needs an optional package that is not installed.

    It subclasses `ImportError`. The message names the install command.
    """
