# ``spec.observe.*`` authoring namespace.
"""``spec.observe.*`` namespace.

Each method appends an
[ObservationSpec][diff_soup.problem._spec.specs.ObservationSpec] to the parent
ProblemSpec. Observations are not part of the residual; they are exposed by the
lowered problem as a callable for post-processing or fitting.

Every verb takes an optional ``domain=``: the name of the domain whose grid
(``q.x``) and operator bank (``q.ops``) the observation evaluates with, and
whose read-through aliases (``couple(..., as_=...)``) it may read. Without
``domain=`` the build traces the observation once and uses the domain whose
fields it reads; an observation that reads only globals, parameters or
functions uses the first declared domain that carries fields. An observation
that reads the fields of two or more domains must name its domain with
``domain=``; the build refuses it otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from .specs import ObservationSpec


class _ObserveNamespace:
    """The observation declarations of a recipe, reached as `spec.observe`."""

    __slots__ = ("_spec",)

    def __init__(self, spec) -> None:
        self._spec = spec

    def scalar(self, *, name: str, fn: Callable, domain: Optional[str] = None) -> None:
        """Declare a scalar observation, computed from the state by `fn(q)`.

        Pass `domain` when `fn` reads the fields of more than one component.

        Args:
            name: Observation name.
            fn: Closure `lambda q: ...` over the fields and globals, returning one array.
            domain: Domain the observation evaluates on, or `None` for the domain whose fields `fn` reads.
        """
        self._spec._observations.append(
            ObservationSpec(name=name, kind="scalar", fn=fn, domain=domain)
        )

    def field(self, *, name: str, fn: Callable, domain: Optional[str] = None) -> None:
        """Declare a spatially resolved observation, computed from the state by `fn(q)`.

        Pass `domain` when `fn` reads the fields of more than one component.

        Args:
            name: Observation name.
            fn: Closure `lambda q: ...` over the fields and globals, returning one array.
            domain: Domain the observation evaluates on, or `None` for the domain whose fields `fn` reads.
        """
        self._spec._observations.append(
            ObservationSpec(name=name, kind="field", fn=fn, domain=domain)
        )

    def boundary_mean(
        self, *, name: str, fn: Callable, tag: str, domain: Optional[str] = None
    ) -> None:
        """Declare the mean of a nodal quantity over a tagged boundary.

        `fn(q)` returns one value per node, and the observation averages the values
        on `tag` weighted by the boundary measure: face areas on a grid, facet
        measures on a mesh. Pass `domain` when `fn` reads the fields of more than
        one component.

        Args:
            name: Observation name.
            fn: Closure `lambda q: ...` over the fields and globals, returning one value per node.
            tag: Name of the boundary tag the mean is over.
            domain: Domain the observation evaluates on, or `None` for the domain whose fields `fn` reads.
        """
        self._spec._observations.append(
            ObservationSpec(
                name=name, kind="boundary_mean", fn=fn, where={"tag": tag}, domain=domain
            )
        )

    def spatial_mean(self, *, name: str, fn: Callable, domain: Optional[str] = None) -> None:
        """Declare the mean of a nodal quantity over a domain.

        `fn(q)` returns one value per node, and the observation averages them
        weighted by the cell volumes on a grid or the lumped mass on a mesh. Pass
        `domain` when `fn` reads the fields of more than one component.

        Args:
            name: Observation name.
            fn: Closure `lambda q: ...` over the fields and globals, returning one value per node.
            domain: Domain the observation evaluates on, or `None` for the domain whose fields `fn` reads.
        """
        self._spec._observations.append(
            ObservationSpec(name=name, kind="spatial_mean", fn=fn, domain=domain)
        )
