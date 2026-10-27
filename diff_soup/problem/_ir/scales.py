"""Scales — bookkeeping for non-dim ↔ dimensional translation at the LoweredProblem boundary.

The IR is born non-dimensionalised — equation closures, mass diagonal, sparsity,
and BC residuals all live in non-dim coordinates by the time ``lower()`` sees the
EquationSet.  [Scales][diff_soup.problem._ir.scales.Scales] carries dimensional reference
values used at the
[LoweredProblem][diff_soup.problem._ir.lowered_problem.LoweredProblem] boundary to
translate user input (``initial_u``, ``assemble_state``) and observation output
back into dimensional.

Convention: ``state_SI = state_offset + state_ref · state_internal`` (per
``(component, variable)`` pair).  ``t_SI = time_ref · t_internal``.
``integrand_ref[constraint_name]`` is the dimensional value of one unit of the non-dim
integrand.

[Scales][diff_soup.problem._ir.scales.Scales] is not read at residual time — it
lives only at the boundary.  Therefore ``Component.name`` does not enter the
population grouping key.

Coordinates are never rescaled: grids carry dimensional coordinates and the stencils
differentiate against them, so there is no spatial reference to translate at the
boundary and none is carried here.

The frozen dataclass blocks field reassignment; ``__eq__`` / ``__hash__``
normalize the mappings on the fly.  Validation in ``__post_init__`` reads the
fields without mutation.  The public surface (``.state_ref.items()``,
``.get(…)``, iteration) is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field


def _check_finite(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")


def _check_finite_nonzero(value: float, label: str) -> None:
    _check_finite(value, label)
    if value == 0.0:
        raise ValueError(f"{label} must be non-zero, got {value!r}")


@dataclass(frozen=True)
class Scales:
    """Scales of an equation set, relating internal values to dimensional ones.

    A dimensional value is `offset + scale * internal`. `spec.build()` fills one
    from the inline and central scales of a recipe.

    Attributes:
        state_ref: State scale per `(component, field)` key, with component `""` for a global, finite and nonzero.
        state_offset: State offset per `(component, field)` key, finite.
        time_ref: Time scale, finite and nonzero.
        integrand_ref: Target divisor per constraint name, finite and nonzero.
    """

    state_ref: Mapping[tuple[str, str], float] = field(default_factory=dict)
    state_offset: Mapping[tuple[str, str], float] = field(default_factory=dict)
    time_ref: float = 1.0
    integrand_ref: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate by reading the fields — no mutation.  The frozen dataclass
        # already blocks field reassignment.
        for key, value in self.state_ref.items():
            _check_finite_nonzero(float(value), f"state_ref[{key!r}]")
        for key, value in self.state_offset.items():
            _check_finite(float(value), f"state_offset[{key!r}]")
        _check_finite_nonzero(float(self.time_ref), "time_ref")
        for key, value in self.integrand_ref.items():
            _check_finite_nonzero(float(value), f"integrand_ref[{key!r}]")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scales):
            return NotImplemented
        return (
            dict(self.state_ref) == dict(other.state_ref)
            and dict(self.state_offset) == dict(other.state_offset)
            and float(self.time_ref) == float(other.time_ref)
            and dict(self.integrand_ref) == dict(other.integrand_ref)
        )

    def __hash__(self) -> int:
        return hash(
            (
                tuple(sorted(self.state_ref.items())),
                tuple(sorted(self.state_offset.items())),
                float(self.time_ref),
                tuple(sorted(self.integrand_ref.items())),
            )
        )


def identity_scales() -> Scales:
    """Identity Scales: every reference 1.0, every offset 0.0, no entries."""
    return Scales()
