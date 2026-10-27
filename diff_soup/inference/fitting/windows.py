"""windows.py — multishoot window specs + continuity config.

[Windows][diff_soup.inference.fitting.windows.Windows] is the multiple-shooting window specification.  Its *constructor
choice* selects the multishoot mode ("Mode rule — no string flag"):

* [from_data][diff_soup.inference.fitting.windows.Windows.from_data] — **observed-IC windows**.  Each window starts
  from the observed full state at its frame; there are NO continuity rows (segments
  are independent observed-IC one-frame predictions, scoped to full-state data via
  the eager ``__check_init__`` on the problem).
* [at][diff_soup.inference.fitting.windows.Windows.at] / [uniform][diff_soup.inference.fitting.windows.Windows.uniform] — **decision-variable windows**.
  The window-start states become decision variables ``s̃_w`` and the LM stack gains
  continuity rows.

[Continuity][diff_soup.inference.fitting.windows.Continuity] configures the continuity-row weight for decision-variable
windows.  ``weight="auto"`` is the reference-scaled WRMS, matched to the
RMS magnitude of the observation rows' declared `Series.weight`: the literal
``1/(atol+rtol|s̃|)`` formula blows up conditioning at production tolerances, so
the per-row weight is made a dimensionless ratio instead.

[StateBasis][diff_soup.inference.fitting.windows.StateBasis] chooses the coordinate the
decision-variable window starts
live in.  ``StateBasis.full()`` (the default) makes every differential degree of
freedom a decision variable; ``StateBasis.from_modes(B)`` represents a window
start as an anchor plus ``B a`` in a low-dimensional orthonormal subspace, which
is what keeps decision-variable windows tractable on a large state.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Mode tags (static; selected by the constructor, never a user string flag).
_MODE_FROM_DATA = "from_data"
_MODE_FROM_STATES = "from_states"
_MODE_AT = "at"
_MODE_UNIFORM = "uniform"


@dataclasses.dataclass(frozen=True)
class WindowInitialization:
    """Exact, reusable initialization for decision-variable windows.

    Supplying this object to
    [InferenceProblem][diff_soup.inference.fitting.inference_problem.InferenceProblem] makes
    construction pure: no
    manifold forward sweep and no separate endpoint-validation solve are
    executed. Applications can content-address and persist it alongside a
    checkpoint so reduced coordinates always refer to the same anchors.

    Attributes:
        anchors: Differential start-state anchor for every window,
            ``(n_windows, n_diff)``.
        initial_coordinates: Optional initial decision coordinates,
            ``(n_decision_windows, n_state)``. ``None`` means zero
            coefficients for a reduced basis or the corresponding anchors in
            a full-state coordinate.
        reached_tend: Whether the initialization builder's endpoint validation
            reached the final horizon.
        provenance: Arbitrary immutable-by-convention metadata (content key,
            source fingerprint, mesh key, and so on). It is never traced.
    """

    anchors: Any
    initial_coordinates: Any = None
    reached_tend: bool = True
    provenance: Any = None


class RestartStep(eqx.Module):
    """The integrator's first-step policy at a window restart.

    Declared on ``Multishoot(first_step=...)`` under the public name
    ``diff_soup.inference.FirstStep``: ``FirstStep.auto()`` uses the stock
    integrator estimate, ``FirstStep.fixed(dt)`` a fixed first step,
    ``FirstStep.fraction(f)`` a clipped fraction of the segment duration, and
    ``FirstStep.hinted(...)`` per-segment hints. A policy resolves to the
    solver's dynamic ``dt0`` operand (``0`` means the stock estimate); the
    resolved value is gradient-dead, clipped to the segment duration, and
    does not alter executable shape.
    """

    mode: str = eqx.field(static=True)
    value: float = eqx.field(static=True, default=0.0)
    min_step: float = eqx.field(static=True, default=0.0)
    max_step: float = eqx.field(static=True, default=0.0)
    hints: Any = None

    @classmethod
    def auto(cls) -> "RestartStep":
        """Use the stock solver's initial-step estimate (``dt0=0``)."""
        return cls(mode="auto")

    @classmethod
    def fixed(cls, value: float) -> "RestartStep":
        """Use one magnitude on every segment, clipped to its duration."""
        if float(value) <= 0.0:
            raise ValueError("RestartStep.fixed value must be positive.")
        return cls(mode="fixed", value=float(value))

    @classmethod
    def fraction(
        cls,
        value: float,
        *,
        min_step: float = 0.0,
        max_step: float = 0.0,
    ) -> "RestartStep":
        """Use a clipped fraction of each segment duration."""
        if not 0.0 < float(value) <= 1.0:
            raise ValueError("RestartStep.fraction value must be in (0, 1].")
        if float(min_step) < 0.0 or float(max_step) < 0.0:
            raise ValueError("RestartStep bounds must be nonnegative.")
        if max_step and min_step > max_step:
            raise ValueError("RestartStep min_step cannot exceed max_step.")
        return cls(
            mode="fraction",
            value=float(value),
            min_step=float(min_step),
            max_step=float(max_step),
        )

    @classmethod
    def hinted(
        cls,
        hints: Any,
        *,
        max_step: float = 0.0,
    ) -> "RestartStep":
        """Use per-segment positive hints; invalid hints fall back to auto."""
        array = jnp.asarray(hints)
        if array.ndim != 1:
            raise ValueError(f"RestartStep hints must be one-dimensional; got {array.shape}.")
        if array.size == 0:
            raise ValueError("RestartStep hints must contain at least one value.")
        if float(max_step) < 0.0:
            raise ValueError("RestartStep max_step must be nonnegative.")
        return cls(mode="hinted", max_step=float(max_step), hints=array)

    def resolve(self, t0: Any, t1: Any, segment_index: Any = 0) -> jax.Array:
        """Return the signed, gradient-dead ``dt0`` for one segment."""
        t0 = jnp.asarray(t0)
        t1 = jnp.asarray(t1, dtype=t0.dtype)
        duration = t1 - t0
        magnitude = jnp.abs(duration)
        if self.mode == "auto":
            candidate = jnp.zeros((), dtype=t0.dtype)
        elif self.mode == "fixed":
            candidate = jnp.asarray(self.value, dtype=t0.dtype)
        elif self.mode == "fraction":
            candidate = jnp.asarray(self.value, dtype=t0.dtype) * magnitude
            if self.min_step > 0.0:
                candidate = jnp.maximum(candidate, jnp.asarray(self.min_step, t0.dtype))
        elif self.mode == "hinted":
            hints = jnp.asarray(self.hints, dtype=t0.dtype)
            index = jnp.asarray(segment_index, dtype=jnp.int32)
            in_range = (index >= 0) & (index < hints.shape[0])
            safe_index = jnp.clip(index, 0, max(hints.shape[0] - 1, 0))
            hinted = hints[safe_index]
            valid = in_range & jnp.isfinite(hinted) & (hinted > 0.0)
            candidate = jnp.where(valid, hinted, jnp.zeros((), dtype=t0.dtype))
        else:  # pragma: no cover - constructor spelling is internal
            raise ValueError(f"unknown RestartStep mode {self.mode!r}.")
        if self.max_step > 0.0:
            candidate = jnp.minimum(candidate, jnp.asarray(self.max_step, t0.dtype))
        candidate = jnp.minimum(candidate, magnitude)
        signed = jnp.where(duration < 0.0, -candidate, candidate)
        return jax.lax.stop_gradient(signed)


class Windows(eqx.Module):
    """The windows of a [Multishoot][diff_soup.inference.Multishoot] trajectory, built by one of its constructors.

    `from_data` and `from_states` start every window from a known state and fit only the
    parameters. `at` and `uniform` fit the window starts beside the parameters.
    """

    mode: str = eqx.field(static=True)
    boundaries: Optional[tuple] = eqx.field(static=True, default=None)
    n_windows: Optional[int] = eqx.field(static=True, default=None)
    source: Optional[str] = eqx.field(static=True, default=None)
    seed_times: Optional[Any] = None
    seed_states: Optional[Any] = None

    @classmethod
    def from_data(cls, source: Optional[str] = None) -> "Windows":
        """Start every window from a measured full state, one window per consecutive pair of frames.

        Args:
            source: The name of the measurement that seeds the windows, needed when the experiment has two.
        """
        return cls(mode=_MODE_FROM_DATA, source=source)

    @classmethod
    def from_states(cls, times: Any, states: Any) -> "Windows":
        """Start every window from a given full state, one window per consecutive pair of frames.

        The frame times must equal those of the experiment's one measurement.

        Args:
            times: The strictly increasing frame times, at least two.
            states: The full state at each frame time, shape `(n_frames, n_state)`.
        """
        times_np = np.asarray(times, dtype=float)
        states_np = np.asarray(states)
        if times_np.ndim != 1 or times_np.shape[0] < 2 or not np.all(np.diff(times_np) > 0):
            raise ValueError(
                "Windows.from_states needs at least two strictly increasing frame "
                f"times; got shape {times_np.shape}."
            )
        if states_np.ndim != 2 or states_np.shape[0] != times_np.shape[0]:
            raise ValueError(
                "Windows.from_states needs one full state per frame time, shape "
                f"(n_frames, n_state) with n_frames={times_np.shape[0]}; got {states_np.shape}."
            )
        return cls(
            mode=_MODE_FROM_STATES,
            seed_times=jnp.asarray(times_np),
            seed_states=jnp.asarray(states_np),
        )

    @classmethod
    def at(cls, ts: Any) -> "Windows":
        """Place window boundaries at the times `ts` and fit the window starts.

        Args:
            ts: At least two boundary times, reaching from the start of the time span to the last frame.
        """
        bounds = tuple(float(t) for t in np.asarray(ts).reshape((-1,)).tolist())
        if len(bounds) < 2:
            raise ValueError("Windows.at(ts): need at least 2 boundary times.")
        return cls(mode=_MODE_AT, boundaries=bounds)

    @classmethod
    def uniform(cls, n: int) -> "Windows":
        """Split the span from the start time to the last frame into `n` equal windows and fit their starts.

        Args:
            n: The number of windows, at least one.
        """
        if int(n) < 1:
            raise ValueError("Windows.uniform(n): n must be >= 1.")
        return cls(mode=_MODE_UNIFORM, n_windows=int(n))

    @property
    def is_observed_ic(self) -> bool:
        """``True`` for the ``from_data`` and ``from_states`` (observed-IC, no-continuity) modes."""
        return self.mode in (_MODE_FROM_DATA, _MODE_FROM_STATES)

    @property
    def is_decision_variable(self) -> bool:
        """``True`` for the decision-variable (continuity-row) modes."""
        return self.mode in (_MODE_AT, _MODE_UNIFORM)


class StateBasis(eqx.Module):
    """The coordinates in which multiple shooting fits each window start.

    `StateBasis.full()` fits every differential entry. `StateBasis.from_modes(modes)` fits
    only coefficients `c`, with each window start its initial value plus `modes @ c`.
    """

    modes: Optional[np.ndarray] = None

    @classmethod
    def full(cls) -> "StateBasis":
        """Fit every differential entry of each window start."""
        return cls(modes=None)

    @classmethod
    def from_modes(cls, modes: Any) -> "StateBasis":
        """Fit each window start in the span of the columns of `modes`.

        Args:
            modes: A column-orthonormal `(n_diff, n_modes)` array with one row per differential entry of the state.

        Raises:
            ValueError: If `modes` is not two-dimensional, finite and column-orthonormal.
        """
        raw = np.asarray(modes)
        dtype = raw.dtype if np.issubdtype(raw.dtype, np.floating) else np.dtype(float)
        array = np.ascontiguousarray(raw, dtype=dtype)
        if array.ndim != 2:
            raise ValueError(
                f"StateBasis.from_modes: modes must be a two-dimensional "
                f"(n_diff, n_modes) array; got shape {array.shape}."
            )
        n_rows, n_modes = array.shape
        if n_modes > n_rows:
            raise ValueError(
                f"StateBasis.from_modes: a basis of {n_rows} differential rows holds "
                f"at most {n_rows} orthonormal columns; got {n_modes}."
            )
        # Before the Gram test, because a NaN deviation compares false against any
        # tolerance and would let a non-finite basis through as orthonormal.
        n_bad = int(np.count_nonzero(~np.isfinite(array)))
        if n_bad:
            raise ValueError(
                f"StateBasis.from_modes: modes must be finite; {n_bad} of {array.size} "
                "entries are NaN or infinite. Supply a basis whose entries are all finite."
            )
        gram = array.T @ array
        error = float(np.max(np.abs(gram - np.eye(n_modes, dtype=array.dtype)))) if n_modes else 0.0
        tolerance = max(
            1e-8,
            32.0 * float(np.finfo(array.dtype).eps) * max(1, n_modes),
        )
        if error > tolerance:
            raise ValueError(
                f"StateBasis.from_modes: modes must be column-orthonormal "
                f"(modesᵀ modes == I); the largest deviation is {error:.3e} "
                f"(dtype-aware tolerance {tolerance:.3e}). "
                "Orthonormalize the columns first, for example with a QR "
                "factorization. Orthonormality is what makes the projected "
                "continuity rows carry the in-subspace norm."
            )
        return cls(modes=array)

    @property
    def is_full(self) -> bool:
        """``True`` for the full (unreduced) coordinate."""
        return self.modes is None

    @property
    def n_modes(self) -> int:
        """Column count of a reduced basis; ``0`` for the full coordinate."""
        return 0 if self.modes is None else int(self.modes.shape[1])

    def width(self, n_diff: int) -> int:
        """Decision-block width per window for a state with ``n_diff`` differential rows."""
        return int(n_diff) if self.modes is None else int(self.modes.shape[1])

    def check_state_size(self, n_diff: int) -> None:
        """Validate the basis against the problem's differential dimension.

        Raises:
            ValueError: If a reduced basis has the wrong number of rows.
        """
        if self.modes is None:
            return
        if int(self.modes.shape[0]) != int(n_diff):
            raise ValueError(
                f"StateBasis modes have {self.modes.shape[0]} rows but the problem has "
                f"{n_diff} differential degrees of freedom. The basis spans window-start "
                "states, so it needs one row per differential degree of freedom."
            )


class Continuity(eqx.Module):
    """The weight of the continuity term that joins fitted multiple-shooting windows.

    Attributes:
        weight: `"auto"` scales the term from the solver tolerances and the observation weights, and a positive float `W` weights every row by `sqrt(W)`.
    """

    weight: Union[str, float] = "auto"

    def __check_init__(self):
        """Validate ``weight`` is ``"auto"`` or a positive float."""
        w = self.weight
        if isinstance(w, str):
            if w != "auto":
                raise ValueError(f"Continuity.weight string must be 'auto'; got {w!r}.")
        elif float(w) <= 0.0:
            raise ValueError(f"Continuity.weight float must be positive; got {w!r}.")
