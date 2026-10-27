"""Variable roles and dependency declarations for implicit problems.

[DAEStructure][diff_soup.problem.implicit.structure.DAEStructure] states which components of the unknown vector are
differential variables — an explicit semantic declaration, never inferred
from a mass diagonal, a rank sample, or empty Jacobian rows. Cross-row
derivative couplings are legal: an equation may read the derivative of any
differential variable, while a variable declared algebraic must never gain a
``ydot`` column anywhere in the residual.

[Dependencies][diff_soup.problem.implicit.structure.Dependencies] states which solve-time inputs a callback's values vary
with. A ``False`` flag is an assertion of invariance the implementation must
honor — caches use it to decide validity, so a false declaration produces
silently stale values, not a fallback.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np

__all__ = ["Dependencies", "DAEStructure"]


class Dependencies(eqx.Module):
    """Which solve-time inputs a set of coefficients varies with.

    Args:
        time: Whether the coefficients depend on `t`.
        state: Whether they depend on the state `y`.
        parameters: Whether they depend on `theta`.
        control: Whether they depend on the control values.
    """

    time: bool = eqx.field(static=True)
    state: bool = eqx.field(static=True)
    parameters: bool = eqx.field(static=True)
    control: bool = eqx.field(static=True)

    def __init__(self, *, time: bool, state: bool, parameters: bool, control: bool):
        """Record the four dependency flags (each one is required)."""
        self.time = bool(time)
        self.state = bool(state)
        self.parameters = bool(parameters)
        self.control = bool(control)

    @property
    def constant(self) -> bool:
        """Whether the coefficients depend on none of the four inputs."""
        return not (self.time or self.state or self.parameters or self.control)


class DAEStructure(eqx.Module):
    """Which unknowns of a problem are differential and which are algebraic, plus named index ranges.

    Properties `differential_indices`, `algebraic_indices` and `n_differential`
    read the mask, and `partition(name)` returns a range by name.

    Args:
        differential_vars: Nonempty 1-D boolean mask, true for each differential unknown.
        partitions: Triples `(name, lo, hi)`, each the index range `[lo, hi)` within the unknowns.
    """

    n: int = eqx.field(static=True)
    _mask_bytes: bytes = eqx.field(static=True)
    partitions: tuple = eqx.field(static=True, default=())

    def __init__(self, *, differential_vars, partitions: tuple = ()):
        """Validate and freeze the variable-role mask and named partitions."""
        mask = np.asarray(differential_vars)
        if mask.ndim != 1:
            raise ValueError(
                f"differential_vars must be a 1-D mask with one entry per unknown; "
                f"got shape {mask.shape}."
            )
        if mask.size == 0:
            raise ValueError(
                "differential_vars is empty; a problem must have at least one unknown."
            )
        mask = mask.astype(bool)
        self.n = int(mask.size)
        self._mask_bytes = mask.tobytes()
        parts = []
        for part in partitions:
            name, lo, hi = part
            lo = int(lo)
            hi = int(hi)
            if not (0 <= lo <= hi <= self.n):
                raise ValueError(
                    f"partition {name!r} spans [{lo}, {hi}) but the problem has "
                    f"{self.n} unknowns; partitions must lie within [0, {self.n}]."
                )
            parts.append((str(name), lo, hi))
        self.partitions = tuple(parts)

    @property
    def differential_vars(self) -> np.ndarray:
        """The read-only host boolean mask, shape ``(n,)``."""
        mask = np.frombuffer(self._mask_bytes, dtype=bool)
        mask.flags.writeable = False
        return mask

    @property
    def differential_indices(self) -> np.ndarray:
        """Indices of differential variables (read-only host int64)."""
        idx = np.flatnonzero(self.differential_vars)
        idx.flags.writeable = False
        return idx

    @property
    def algebraic_indices(self) -> np.ndarray:
        """Indices of algebraic variables (read-only host int64)."""
        idx = np.flatnonzero(~self.differential_vars)
        idx.flags.writeable = False
        return idx

    @property
    def n_differential(self) -> int:
        """Number of differential variables."""
        return int(self.differential_vars.sum())

    def partition(self, name: str) -> tuple[int, int]:
        """Return the ``(lo, hi)`` index range of the named partition.

        Raises:
            KeyError: If no partition with this name was declared.
        """
        for part_name, lo, hi in self.partitions:
            if part_name == name:
                return (lo, hi)
        raise KeyError(
            f"no partition named {name!r}; declared partitions: "
            f"{tuple(p[0] for p in self.partitions)}."
        )
