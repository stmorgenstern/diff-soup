"""series.py — observed data bound to its observable.

A [Series][] carries the observation times, the observed values *in
observable space*, the [ObservationOperator][diff_soup.inference.data.base.ObservationOperator]
that maps solver state to that space, optional ``1/σ`` weights, and an optional
time-frame mask for windowed-in-time observation.

Three convenience observable operators are provided:

* [FullState][] — observe the full state (identity); the default observable.
* [Component][] — observe one field of a multi-field state (by name or index).
* [FnObservable][] — wrap a bare ``fn(u) -> obs`` callable as an observable.

The observed ``values`` (and any ``weight``) are cast once at construction.
By default the dtype follows JAX's active x64 setting for backward
compatibility. Precision-controlled applications should pass ``dtype=`` from
their :class:`~diff_soup.Precision` so a process with x64 enabled can still run
a genuinely fp32 optimization.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from diff_soup.inference.data.base import ObservationOperator


def _compute_dtype() -> Any:
    """Return the active compute dtype (fp64 when x64 is enabled, else fp32).

    Derived from the active default float (``jnp.zeros(()).dtype``) — no hard fp64
    cast anywhere in the fit layer.  The dtype follows JAX's active x64 configuration
    so an fp32 stream stays fp32 and is never silently promoted.
    """
    return jnp.zeros(()).dtype


# ---------------------------------------------------------------------------
# Observable operators (observable-space maps; ObservationOperator subclasses)
# ---------------------------------------------------------------------------


class FullState(ObservationOperator):
    """Identity observable: observe the full state ``u`` (the default observation operator)."""

    @property
    def is_full_state(self) -> bool:
        """Whether this operator reports the solver state unchanged (it does)."""
        return True

    def predict(self, u: Array, t: Any, theta: Any) -> Array:
        """Return ``u`` unchanged, ignoring ``t`` and ``theta``."""
        del t, theta
        return u


def observes_full_state(observable: Any) -> bool:
    """Whether an observation operator reports the solver state unchanged.

    [FullState][] is the plain spelling, but the same identity readout also
    arrives wrapped: a measurement declared with no ``observe=`` carries the
    default readout through the channel adapters. An operator that reports the
    state unchanged declares ``is_full_state``; every other operator (a
    component, a projection, a learned readout) does not.

    Args:
        observable: The stream's observation operator.

    Returns:
        ``True`` when the operator's prediction is the state itself.
    """
    return bool(getattr(observable, "is_full_state", False))


class Component(ObservationOperator):
    """Observe one component of the nodal state by indexing its last axis.

    The operator evaluates ``state[..., field]``. On the flat nodal state a
    solver returns, an integer therefore selects one entry and a ``slice``
    selects a contiguous block (for example the nodes of one field in a
    stacked state); a name is resolved through ``field_index`` to one of those.

    Attributes:
        field: An integer index, a ``slice``, or a name resolved against
            ``field_index``.
        field_index: Optional ``{name: index_or_slice}`` map for name
            resolution (static).
    """

    field: Union[str, int, slice] = eqx.field(static=True)
    field_index: Optional[dict] = eqx.field(static=True, default=None)

    def predict(self, u: Array, t: Any, theta: Any) -> Array:
        """Slice the requested component out of the state, ignoring ``t`` and ``theta``."""
        del t, theta
        f = self.field
        if isinstance(f, str):
            if self.field_index is None or f not in self.field_index:
                raise KeyError(
                    f"Component(field={f!r}): no field_index mapping provided; "
                    "pass field_index={name: idx} or an integer/slice index."
                )
            f = self.field_index[f]
        return u[..., f]


class FnObservable(ObservationOperator):
    """Wrap a bare ``fn(u) -> obs`` callable as an observable.

    Attributes:
        fn: The observation map ``u -> obs`` (must be JAX-traceable / differentiable).
    """

    fn: Callable = eqx.field(static=True)

    def predict(self, u: Array, t: Any, theta: Any) -> Array:
        """Apply ``self.fn`` to the state, ignoring ``t`` and ``theta``."""
        del t, theta
        return self.fn(u)


def as_observable(obs: Any) -> ObservationOperator:
    """Coerce a user observable spec into an [ObservationOperator][].

    Accepts an [ObservationOperator][] (returned unchanged), a bare callable
    (wrapped in [FnObservable][]), or ``None`` (→ [FullState][]).

    Args:
        obs: An operator, a ``fn(u) -> obs`` callable, or ``None``.

    Returns:
        An [ObservationOperator][diff_soup.inference.data.base.ObservationOperator].
    """
    if obs is None:
        return FullState()
    if isinstance(obs, ObservationOperator):
        return obs
    if callable(obs):
        return FnObservable(fn=obs)
    raise TypeError(
        f"observable must be an ObservationOperator, a callable, or None; got {type(obs).__name__}"
    )


# ---------------------------------------------------------------------------
# Series — data bound to its observable
# ---------------------------------------------------------------------------


class Series(eqx.Module):
    """A named observed time series bound to its observable. Frozen.

    A ``Series`` is one measurement stream of a single latent trajectory.  A
    problem may carry several streams of the same trajectory; each is a
    ``Series`` with a unique ``name``.  The name labels the stream in
    diagnostics and selects an observed-state stream as a multiple-shooting
    source (``Windows.from_data(source=name)``).

    Attributes:
        name: Unique stream name within a problem (e.g. ``"concentration"``).
        ts: Observation times, shape ``(n_frames,)``.
        values: Observed values in observable space, shape ``(n_frames, ...)``.
        observable: The
            [ObservationOperator][diff_soup.inference.data.base.ObservationOperator]
            mapping state → observable
            space (default [FullState][diff_soup.inference.fitting.series.FullState]).
        weight: ``1/σ`` rows broadcast to ``values`` (or ``None``).
        likelihood: Optional normalized likelihood with an exact
            ``least_squares_rows(observed, predicted, context=...)`` map. This
            is how a fixed correlated Gaussian whitens a complete time-series
            event. It is mutually exclusive with ``weight`` so observations
            cannot be weighted twice.
        likelihood_context: Optional metadata forwarded to ``likelihood``.
        frame_mask: Optional ``(n_frames,)`` bool mask selecting observed frames
            (windowed-in-time observation).
    """

    name: str = eqx.field(static=True)
    ts: Float[Array, " n_frames"]
    values: Float[Array, "n_frames ..."]
    observable: ObservationOperator = eqx.field(default_factory=FullState)
    weight: Optional[Array] = None
    likelihood: Any = None
    likelihood_context: Any = None
    frame_mask: Optional[Bool[Array, " n_frames"]] = None

    def __init__(
        self,
        name: str,
        ts: Any,
        values: Any,
        observable: Any = None,
        weight: Any = None,
        likelihood: Any = None,
        likelihood_context: Any = None,
        frame_mask: Any = None,
        dtype: Any = None,
    ):
        """Build a frozen [Series][diff_soup.inference.fitting.series.Series].

        Casts data to ``compute_dtype``.

        Args:
            name: Unique stream name within a problem.
            ts: Observation times (cast to ``compute_dtype``).
            values: Observed values in observable space (cast to ``compute_dtype``).
            observable: An
                [ObservationOperator][diff_soup.inference.data.base.ObservationOperator],
                a bare ``fn(u) -> obs``
                callable, or ``None`` (→ [FullState][diff_soup.inference.fitting.series.FullState]).
            weight: ``1/σ`` rows or ``None`` (cast to ``compute_dtype`` if present).
            likelihood: Optional exact least-squares likelihood. Its
                ``least_squares_rows`` method acts on the complete retained
                observation event, so it can represent temporal covariance.
            likelihood_context: Optional metadata passed to the likelihood.
            frame_mask: Optional ``(n_frames,)`` bool mask.
            dtype: Floating dtype for times, observations, weights, and
                floating likelihood leaves. ``None`` preserves the historical
                active-JAX-default behavior.
        """
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"Series: 'name' must be a non-empty string (the stream name); got {name!r}. "
                "Construct as Series(name, ts, values, observable=...)."
            )
        dt = _compute_dtype() if dtype is None else jnp.dtype(dtype)
        if not jnp.issubdtype(dt, jnp.floating):
            raise TypeError(f"Series dtype must be floating; got {dtype!r}.")
        ts_arr = jnp.asarray(ts, dtype=dt)
        if ts_arr.ndim != 1:
            raise ValueError(f"Series.ts must be 1-D (n_frames,); got shape {ts_arr.shape}.")
        if ts_arr.shape[0] < 2:
            raise ValueError(f"Series.ts must have at least 2 frames; got {ts_arr.shape[0]}.")
        self.name = name
        self.ts = ts_arr
        self.values = jnp.asarray(values, dtype=dt)
        self.observable = as_observable(observable)
        if weight is not None and likelihood is not None:
            raise ValueError(
                "Series accepts either pointwise weight or likelihood whitening, "
                "not both; supplying both would count the noise model twice."
            )
        if likelihood is not None and not callable(getattr(likelihood, "least_squares_rows", None)):
            raise TypeError(
                "Series likelihood must expose a callable "
                "least_squares_rows(observed, predicted, context=...) method."
            )
        self.weight = None if weight is None else jnp.asarray(weight, dtype=dt)
        if likelihood is not None and dtype is not None:
            likelihood = jax.tree_util.tree_map(
                lambda leaf: (
                    jnp.asarray(leaf, dtype=dt)
                    if hasattr(leaf, "dtype")
                    and jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
                    else leaf
                ),
                likelihood,
            )
        self.likelihood = likelihood
        self.likelihood_context = likelihood_context
        self.frame_mask = None if frame_mask is None else jnp.asarray(frame_mask, dtype=bool)

    @property
    def n_frames(self) -> int:
        """Number of observation frames."""
        return int(self.ts.shape[0])

    def observe(self, states: Array, ts: Optional[Array] = None, theta: Any = None) -> Array:
        """Map a ``(n_frames, ...)`` stack of states to observable space.

        Args:
            states: Per-frame solver states, leading axis = frames.
            ts: Optional per-frame observation times, shape ``(n_frames,)``.
                Always passed to the operator's ``predict``; time-independent
                operators ignore it, so passing the frame times is always safe.
            theta: Optional parameter pytree (broadcast over frames, not
                mapped). Always passed to the operator's ``predict`` — this is
                how a fit's residual evaluation hands the current trial
                parameters to a parameter-dependent observable.
                Parameter-blind operators ignore it, so passing ``theta`` is
                always safe.

        Returns:
            Observable-space values, leading axis = frames.
        """
        op = self.observable
        if ts is None:
            return jax.vmap(lambda u: op.predict(u, None, theta))(states)
        return jax.vmap(lambda u, t: op.predict(u, t, theta))(states, jnp.asarray(ts))
