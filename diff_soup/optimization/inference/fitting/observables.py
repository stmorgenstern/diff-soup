"""observables.py — measurement channels for identifiability analysis.

An [Observable][] is one measurement channel of a solve: a differentiable
map ``fn(u, *, t=None, theta=None) -> values`` from the solver state (and
optionally the frame time and the analyzed parameters) to the quantities an
instrument reports, bundled with the channel's noise model. The same object
drives Fisher-information assembly ([diff_soup.inference.design.fisher][])
and, through its ``1/σ`` weights, least-squares fitting.

[DiagonalNoise][] is the independent-Gaussian noise model: one ``σ`` per
channel entry (scalar, per-channel, or per-frame-per-channel).

Distribution-space channels (soft histograms over a population field) reuse the
shipped kernels: [SoftHistogram][diff_soup.inference.data.distribution.SoftHistogram]
is re-exported here so an operando/distribution channel is written with the same
imports as a scalar channel.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from diff_soup.inference.data.base import ObservationOperator
from diff_soup.inference.data.distribution import CDFBins, Moments, SoftHistogram

__all__ = [
    "Observable",
    "DiagonalNoise",
    "SoftHistogram",
    "CDFBins",
    "Moments",
    "as_series",
]


class DiagonalNoise(eqx.Module):
    """Independent Gaussian noise with a diagonal covariance. Frozen.

    Attributes:
        sigma: The standard deviation(s). A scalar applies one ``σ`` to every
            entry; a ``(n_channel,)`` vector gives one ``σ`` per channel entry
            (broadcast over frames); a ``(n_frames, n_channel)`` array gives one
            ``σ`` per frame and entry.
    """

    sigma: Array

    def __init__(self, sigma: Any):
        """Validate and store the standard deviation(s).

        Args:
            sigma: Scalar or array of positive standard deviations.

        Raises:
            ValueError: If any entry of ``sigma`` is not strictly positive.
        """
        arr = jnp.asarray(sigma)
        if not bool(jnp.all(arr > 0.0)):
            raise ValueError(
                f"DiagonalNoise: every sigma must be strictly positive; got {sigma!r}."
            )
        self.sigma = arr

    def weights(self) -> Array:
        """Return the least-squares row weights ``1/σ``."""
        return 1.0 / self.sigma

    def precisions(self, shape: Optional[tuple] = None) -> Array:
        """Return the precisions ``1/σ²``, optionally broadcast to ``shape``.

        Args:
            shape: Target shape (typically ``(n_frames, n_channel)``); ``None``
                returns the precisions in ``sigma``'s own shape.

        Returns:
            The precision array.
        """
        prec = 1.0 / (self.sigma * self.sigma)
        if shape is None:
            return prec
        return jnp.broadcast_to(prec, shape)


class Observable(eqx.Module):
    """A measurement channel: a differentiable state readout plus its noise.

    The readout contract is ``fn(u, *, t=None, theta=None) -> values``: ``u`` is
    one solver state frame, ``t`` the frame time (``None`` when the caller has
    no time grid), and ``theta`` the analyzed-parameter pytree for channels that
    read a parameter directly (an internal-resistance drop, a scale factor).
    ``fn`` must accept the ``t`` and ``theta`` keywords even if it ignores them,
    and must be JAX-traceable.

    Attributes:
        name: The channel name used in reports and diagnostics.
        fn: The readout ``fn(u, *, t=None, theta=None) -> values``.
        noise: The channel's noise model, or ``None`` for unit weights.
        wrt_theta: Whether ``fn`` reads ``theta``. When ``True``, Fisher
            assembly adds the direct ``∂fn/∂θ`` term to the chain rule, and a
            least-squares fit through
            [as_series][diff_soup.inference.fitting.observables.as_series]
            passes the current
            trial theta to the readout at every residual evaluation.
    """

    name: str = eqx.field(static=True)
    fn: Callable = eqx.field(static=True)
    noise: Optional[DiagonalNoise] = None
    wrt_theta: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        name: str,
        fn: Callable,
        noise: Optional[DiagonalNoise] = None,
        wrt_theta: bool = False,
    ):
        """Build a frozen [Observable][].

        Args:
            name: Non-empty channel name.
            fn: The readout ``fn(u, *, t=None, theta=None) -> values``.
            noise: Optional [DiagonalNoise][].
            wrt_theta: Whether ``fn`` reads ``theta`` directly.

        Raises:
            TypeError: If ``name`` is not a non-empty string or ``fn`` is not
                callable.
        """
        if not isinstance(name, str) or not name:
            raise TypeError(f"Observable: 'name' must be a non-empty string; got {name!r}.")
        if not callable(fn):
            raise TypeError(
                f"Observable: 'fn' must be callable with signature "
                f"fn(u, *, t=None, theta=None); got {type(fn).__name__}."
            )
        self.name = name
        self.fn = fn
        self.noise = noise
        self.wrt_theta = wrt_theta

    def __call__(self, u: Array, *, t: Any = None, theta: Any = None) -> Array:
        """Evaluate the readout on one state frame.

        Args:
            u: The solver state frame.
            t: Optional frame time.
            theta: Optional analyzed-parameter pytree.

        Returns:
            The channel values.
        """
        return self.fn(u, t=t, theta=theta)


class _ObservableOperator(ObservationOperator):
    """Adapt an [Observable][] readout to the fit observation protocol.

    Carries the channel's ``fn`` and an optional fixed parameter pytree; the
    fit seam supplies each frame's time through the ``t`` keyword and the
    current trial parameters through the ``theta`` keyword. Which theta the
    readout sees:

    * an explicitly bound ``theta`` is a frozen calibration constant and
      always wins;
    * otherwise a channel declared ``wrt_theta=True`` (it reads fitted
      parameters — a gain, an offset) receives the caller's trial theta, so
      such a channel converges through the ordinary fit path;
    * otherwise the readout gets ``theta=None``, exactly as if no fit were
      in progress.
    """

    fn: Callable = eqx.field(static=True)
    theta: Any = None
    wrt_theta: bool = eqx.field(static=True, default=False)

    @property
    def is_full_state(self) -> bool:
        """Whether the readout reports the solver state unchanged.

        True for the default readout of a measurement declared with no
        ``observe=``, which lets such a measurement seed observed-state
        windows.
        """
        return bool(getattr(self.fn, "is_full_state", False))

    def _theta_for_call(self, theta: Any) -> Any:
        """Select the theta the readout sees (bound > trial-if-declared > None)."""
        if self.theta is not None:
            return self.theta
        return theta if self.wrt_theta else None

    def predict(self, u: Array, t: Any, theta: Any) -> Array:
        """Evaluate the readout at one frame, threading time and parameters."""
        return self.fn(u, t=t, theta=self._theta_for_call(theta))


def as_series(
    observable: Observable,
    ts: Any,
    values: Any,
    *,
    theta: Any = None,
    name: Optional[str] = None,
) -> Any:
    """Bind observed data to a channel as a fit-ready data stream.

    The one-channel-two-uses bridge: the same [Observable][] that drives
    Fisher-information assembly also produces the ``Series`` a least-squares
    fit consumes — the channel's readout becomes the observation operator
    (frame times threaded through ``t``, trial parameters through ``theta``)
    and its noise ``σ`` becomes the ``1/σ`` residual weights.

    Args:
        observable: The measurement channel.
        ts: Observation times, shape ``(n_frames,)``.
        values: Observed values in channel space, shape ``(n_frames, ...)``.
        theta: Optional fixed parameter pytree for the readout. The values
            are frozen into the stream and take precedence over the fit's
            trial theta. For a channel that reads fitted parameters, declare
            the [Observable][] with ``wrt_theta=True`` and leave this
            ``None``: during a fit the readout then receives the current
            trial theta each residual evaluation.
        name: Stream name (default: the channel's name).

    Returns:
        A [Series][diff_soup.inference.fitting.series.Series].
    """
    from diff_soup.inference.fitting.series import Series

    weight = None if observable.noise is None else observable.noise.weights()
    return Series(
        name or observable.name,
        ts,
        values,
        observable=_ObservableOperator(
            fn=observable.fn, theta=theta, wrt_theta=observable.wrt_theta
        ),
        weight=weight,
    )
