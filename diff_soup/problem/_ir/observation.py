"""Observation IR nodes — first-class outputs computed from state.

Observations are not part of the residual; the lowered problem exposes them as a
callable invoked by post-processing or fitting layers.

The marker [Observation][diff_soup.problem._ir.observation.Observation] base carries no fields — each concrete subclass
lists its own.  Lowering dispatches by ``isinstance`` on the marker base.

Every concrete observation records, at build, what it reads and what it
returns:

* ``owner`` — the component whose grid and operator bank the observation
  evaluates with;
* ``inputs`` — the sorted read set, one tuple per name read:
  ``("field", component, field)``, ``("alias", component, alias)``,
  ``("global", name)``, ``("param", name)`` or ``("function", name)``;
* ``out_shape`` / ``out_dtype`` — the shape and dtype of the value, as traced
  at build against zero state of the state dtype (``float64`` with 64-bit
  JAX enabled, ``float32`` otherwise).

[Observable][diff_soup.problem._ir.observation.Observable] is the public,
read-only summary of one observation that
``LoweredProblem.observables`` returns.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx


class Observation(eqx.Module):
    """Marker base; lowering dispatches by ``isinstance``."""


class ScalarObservation(Observation):
    """Scalar reduction of state; e.g. ``q.c.mean()``."""

    name: str = eqx.field(static=True)
    fn: Callable
    owner: str = eqx.field(static=True, default="")
    inputs: tuple[tuple[str, ...], ...] = eqx.field(static=True, default=())
    out_shape: tuple[int, ...] = eqx.field(static=True, default=())
    out_dtype: str = eqx.field(static=True, default="")


class FieldObservation(Observation):
    """Full-field output (same shape as the underlying Variable)."""

    name: str = eqx.field(static=True)
    fn: Callable
    owner: str = eqx.field(static=True, default="")
    inputs: tuple[tuple[str, ...], ...] = eqx.field(static=True, default=())
    out_shape: tuple[int, ...] = eqx.field(static=True, default=())
    out_dtype: str = eqx.field(static=True, default="")


class BoundaryMeanObservation(Observation):
    """An observation labelled as the mean over a tagged boundary.

    `fn` computes the value. The label and `tag` only record what it is.
    """

    name: str = eqx.field(static=True)
    fn: Callable
    tag: str = eqx.field(static=True, default="")
    owner: str = eqx.field(static=True, default="")
    inputs: tuple[tuple[str, ...], ...] = eqx.field(static=True, default=())
    out_shape: tuple[int, ...] = eqx.field(static=True, default=())
    out_dtype: str = eqx.field(static=True, default="")


class SpatialMeanObservation(Observation):
    """An observation labelled as the mean over a domain.

    `fn` computes the value. The label only records what it is.
    """

    name: str = eqx.field(static=True)
    fn: Callable
    owner: str = eqx.field(static=True, default="")
    inputs: tuple[tuple[str, ...], ...] = eqx.field(static=True, default=())
    out_shape: tuple[int, ...] = eqx.field(static=True, default=())
    out_dtype: str = eqx.field(static=True, default="")


_KIND_BY_TYPE: dict[type, str] = {
    ScalarObservation: "scalar",
    FieldObservation: "field",
    BoundaryMeanObservation: "boundary_mean",
    SpatialMeanObservation: "spatial_mean",
}


class Observable(eqx.Module):
    """Read-only summary of one declared observation.

    Attributes:
        name: The declared observation name (the key in ``observe``'s result).
        kind: ``"scalar"``, ``"field"``, ``"boundary_mean"`` or
            ``"spatial_mean"`` — the ``spec.observe`` verb that declared it.
        shape: The shape of the observation's value.
        dtype: The dtype name of the observation's value (for example
            ``"float64"``). It follows the state dtype: ``float64`` with 64-bit
            JAX enabled, ``float32`` otherwise.
        owner: The component (domain) whose grid and operator bank the
            observation evaluates with.
        reads: The sorted read set, one tuple per name read:
            ``("field", component, field)``, ``("alias", component, alias)``,
            ``("global", name)``, ``("param", name)`` or
            ``("function", name)``.
    """

    name: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    owner: str = eqx.field(static=True)
    reads: tuple[tuple[str, ...], ...] = eqx.field(static=True)

    @classmethod
    def of(cls, obs: Observation) -> Observable:
        """The summary of the observation IR node ``obs``."""
        return cls(
            name=obs.name,
            kind=_KIND_BY_TYPE.get(type(obs), type(obs).__name__),
            shape=tuple(getattr(obs, "out_shape", ())),
            dtype=str(getattr(obs, "out_dtype", "")),
            owner=str(getattr(obs, "owner", "")),
            reads=tuple(getattr(obs, "inputs", ())),
        )
