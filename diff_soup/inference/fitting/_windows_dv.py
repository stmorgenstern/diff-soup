# LOC > 500 justification: the decision-variable multishoot mechanics (layout,
# continuity auto-weight resolution, the segment plan, consistent-projection
# propagation, the residual stack, manifold init, and the window diagnostics)
# are one cohesive unit (DecisionWindowBuild) that keeps inference_problem.py thin;
# splitting it would scatter the residual-stack pieces that must stay co-located.
"""_windows_dv.py — decision-variable multishoot machinery.

A decision-variable [Multishoot][diff_soup.inference.fitting.inference_problem.Multishoot] optimizes
``x = [θ̃ | s̃₁ … s̃_{W−1}]`` where ``s̃_w`` are the *differential* components
of the window-start state (window 0 starts at the fixed IC ``y0``).  The
residual stack is:

    r = [ per-window obs rows:   w_f ⊙ (h(u_w(t_f)) − d_f)            ]
     ++ [ continuity rows:       √W_c ⊙ (s̃_w − Φ_diff(s_{w−1}, θ))   ]
     ++ [ state-prior rows:      √w_p ⊙ (s̃ − s̃_forward-sweep)  (opt.) ]

Tikhonov θ rows are appended by the LM engine via the problem regularizer.

Key design decisions:

* **One batched propagation set.**  Because every window start is a decision
  variable, the windows are mutually independent:
  [build_segment_plan][diff_soup.inference.fitting._windows_dv.build_segment_plan]
  enumerates the distinct ``(window, target time)`` propagations once, host-side,
  and
  [DecisionWindowBuild.residual][diff_soup.inference.fitting._windows_dv.DecisionWindowBuild.residual]
  runs them all through a single
  ``jax.vmap``.  The traced program is then independent of the window count, and
  a target reached by both an observation frame and the next window's continuity
  row is solved once instead of twice.  Rows are gathered from one pooled state
  array, so no per-row branching survives into the trace.

* **Continuity auto-weight.**  The naive ``√W_c/(atol+rtol|s̃|)`` weight blows
  up conditioning at production tolerances. The ``"auto"`` mode adds a
  reference scale and matches the observation rows' RMS weight:
  ``w_i = w_obs · √W_c · s_ref/(atol+rtol|s̃_i|)`` with
  ``s_ref = atol + rtol·RMS(|s̃|)`` so the per-row weight is a dimensionless
  ratio ~``√W_c`` and does NOT dominate the observation block.  ``W_c``
  defaults to 10.  A plain ``Continuity(weight=W_c)`` uses the un-referenced
  ``√W_c`` factor.

* **Consistent-projection window starts.**  For a DAE the algebraic
  components are reconstructed by ``pdesolve(..., consistent_ic=True)`` at
  the window start.  Active only when the problem has algebraic rows;
  opt out with ``project_windows=False``.

* **Manifold init.**  One forward sweep at θ₀ seeds every ``s̃_w``; it also
  serves as the single-shoot diagnostic baseline.

* **All-or-nothing failure stamping.** Any unsuccessful or non-finite
  propagation stamps the whole stack with the sentinel, so a step that broke
  the integration cannot be accepted on the strength of rows that survived it.

* **Reduced window coordinate.**  A declared
  [StateBasis][diff_soup.inference.fitting.windows.StateBasis] represents a window start
  as ``anchor + B a`` for an orthonormal ``B``, so each window contributes ``B``'s
  column count to the decision vector instead of the full differential dimension.
  The anchors are the manifold-init sweep states, so zero coefficients reproduce
  the unreduced starting point exactly.  Continuity rows become the Galerkin
  projection ``Bᵀ`` of the weighted mismatch, and the part the subspace cannot
  represent is reported rather than dropped.

* **Dense Gauss-Newton ceiling.**  The Jacobian is computed by ``jacfwd``
  over the full ``x``.  The block-bidiagonal continuity structure is not
  exploited, and it cannot usefully be: the window transition operator of a
  parabolic problem is dense even though its spatial Jacobian is sparse, so
  condensing or a banded normal-equation solve would gain nothing.  A smaller
  decision vector is the lever, which is what ``StateBasis`` provides.  A warning
  is issued beyond ~2 000 decision variables.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._logging import log as logger
from diff_soup.inference.fitting._forward_seam import _solution_failed
from diff_soup.inference.fitting._lane_map import folds_enclosing_batches, lane_map
from diff_soup.inference.fitting.dependencies import (
    ResidualDependencyBlock,
    ResidualDependencyBucket,
)

# Dense-GN decision-variable ceiling: beyond this, the dense normal-equation
# solve is the cost wall; condensed sparse Gauss-Newton is unavailable.
_DENSE_GN_CEIL = 2000


def _trial_steps(sol: Any) -> jax.Array:
    """The integrator's trial-step count of one solution, zero when it reports none."""
    steps = getattr(getattr(sol, "stats", None), "trial_steps", None)
    if steps is None:
        return jnp.zeros((), dtype=jnp.int32)
    return jnp.asarray(steps, dtype=jnp.int32)


class WindowLayout:
    """Host-side layout of the decision vector ``x = [θ̃ | s̃₀? | s̃₁ … s̃_{W−1}]``.

    With a fixed initial state, window 0 starts at the fixed IC ``y0`` (NOT a
    decision variable) and the state blocks are windows ``1 … W−1``.  With a learned
    initial state, window 0's differential start ``s̃₀`` is *also* a decision block,
    so the state blocks are windows ``0 … W−1``.  ``first_state_window`` is the index
    of the first window carrying a decision-state block (``1`` fixed, ``0`` learned).

    A block is ``n_diff`` wide in the full coordinate and ``n_state`` wide when a
    reduced [StateBasis][diff_soup.inference.fitting.windows.StateBasis] is declared, in
    which case the block holds subspace coefficients rather than state values.

    Attributes:
        n_theta: Number of learnable θ scalars.
        n_diff: Number of differential DOFs per window-start state.
        n_state: Decision-block width per window (``n_diff`` unless reduced).
        n_windows: Total window count ``W``.
        boundaries: The ``(W+1,)`` window boundary times.
        learn_initial: Whether window 0's differential start is a decision block.
        first_state_window: Index of the first decision-state window (``0`` or ``1``).
    """

    def __init__(
        self,
        n_theta: int,
        diff_mask: np.ndarray,
        boundaries: np.ndarray,
        learn_initial: bool = False,
        n_state: Optional[int] = None,
    ):
        """Build the layout from θ count, the differential mask, and boundary times."""
        self.n_theta = int(n_theta)
        self.diff_mask = np.asarray(diff_mask, dtype=bool)
        self.n_diff = int(self.diff_mask.sum())
        self.n_state = self.n_diff if n_state is None else int(n_state)
        self.boundaries = np.asarray(boundaries, dtype=float)
        self.n_windows = int(len(self.boundaries) - 1)
        self.learn_initial = bool(learn_initial)
        # Decision states span windows first_state_window .. W-1: window 0 is a
        # decision block iff its initial state is learned, else the fixed IC.
        self.first_state_window = 0 if self.learn_initial else 1
        self.n_decision_windows = self.n_windows - self.first_state_window
        self.n_decision = self.n_theta + self.n_decision_windows * self.n_state

    def theta_of(self, x: jax.Array) -> jax.Array:
        """Slice the θ̃ block out of x."""
        return x[: self.n_theta]

    def state_block(self, x: jax.Array, w: int) -> jax.Array:
        """Slice the decision block for window ``w`` (``w ≥ first_state_window``).

        The block holds differential state values in the full coordinate and
        subspace coefficients under a reduced basis.
        """
        i = self.n_theta + (w - self.first_state_window) * self.n_state
        return x[i : i + self.n_state]

    def state_blocks(self, x: jax.Array) -> jax.Array:
        """All decision blocks as ``(n_decision_windows, n_state)``."""
        return jnp.reshape(jnp.asarray(x)[self.n_theta :], (self.n_decision_windows, self.n_state))

    def scatter_diff(self, s_diff: jax.Array, y0: jax.Array) -> jax.Array:
        """Scatter a differential block into a full state (algebraic seeded from ``y0``).

        The algebraic components are seeded at ``y0``'s algebraic values so the
        consistent-projection (DAE) has a near-manifold start; for a pure ODE every
        row is differential and the full state is exactly ``s_diff``.
        """
        mask = jnp.asarray(self.diff_mask)
        full = jnp.asarray(y0)
        return jnp.where(mask, _embed_diff(s_diff, self.diff_mask, full.dtype), full)


def _embed_diff(s_diff: jax.Array, diff_mask: np.ndarray, dtype: Any) -> jax.Array:
    """Embed a packed differential vector into full-state positions (zeros elsewhere)."""
    idx = jnp.asarray(np.where(diff_mask)[0])
    full = jnp.zeros((diff_mask.shape[0],), dtype=dtype)
    return full.at[idx].set(jnp.asarray(s_diff, dtype=dtype))


def observation_row_scale(series: Any) -> float:
    """RMS whitening magnitude of one scored stream's observation rows.

    Pointwise-weighted rows carry ``1/sigma`` explicitly, so their scale is
    the root-mean-square of the declared weights. Likelihood-scored rows carry
    the family's own whitening instead; their reference scale is the family's
    expected-information whitening evaluated where its expected observation
    reproduces the data. A stream with neither reports one.

    Args:
        series: The scored [Series][diff_soup.inference.fitting.series.Series].

    Returns:
        A positive finite scale for the stream's observation rows.
    """
    if series.weight is not None:
        weights = np.asarray(series.weight, dtype=float)
        return float(np.sqrt(np.mean(weights**2)))
    likelihood = getattr(series, "likelihood", None)
    if likelihood is None:
        return 1.0
    observed = jnp.asarray(series.values)
    try:
        information = likelihood.expected_information(
            likelihood._saturated_prediction(observed),
            context=getattr(series, "likelihood_context", None),
        )
        gain = np.asarray(information.whiten(jnp.ones_like(observed)), dtype=float)
    except (NotImplementedError, AttributeError, TypeError):
        return 1.0
    gain = gain[np.isfinite(gain) & (gain > 0.0)]
    if gain.size == 0:
        return 1.0
    return float(np.sqrt(np.mean(gain**2)))


def resolve_continuity_weight(
    weight: Any,
    s_diff: jax.Array,
    atol: float,
    rtol: float,
    *,
    observation_weight_scale: float = 1.0,
) -> jax.Array:
    """Resolve a [Continuity][diff_soup.inference.fitting.windows.Continuity] weight spec to per-row ``√W_c`` factors.

    * A plain float ``W_c`` → the un-referenced ``√W_c`` row factor.
    * ``"auto"`` → the reference-scaled WRMS factor:
      ``w_obs · √W_c · s_ref/(atol + rtol·|s̃|)`` with
      ``s_ref = atol + rtol·RMS(|s̃|)`` and ``W_c = 10``. This keeps the WRMS
      shape while matching the scale of weighted observation rows.

    Args:
        weight: ``"auto"`` or a positive float ``W_c``.
        s_diff: The window-start differential decision block (for the reference scale).
        atol: The solver's absolute tolerance.
        rtol: The solver's relative tolerance.
        observation_weight_scale: RMS magnitude of the observation-row
            weights. Used only by ``"auto"``; defaults to one.

    Returns:
        A ``(n_diff,)`` (or scalar) array of per-row ``√W_c`` factors at ``s_diff``'s
        dtype.
    """
    dtype = jnp.asarray(s_diff).dtype
    if isinstance(weight, str):  # "auto"
        w_c = jnp.asarray(10.0, dtype=dtype)
        abs_s = jnp.abs(jnp.asarray(s_diff))
        s_typ = jnp.sqrt(jnp.mean(abs_s**2))
        s_ref = jnp.asarray(atol, dtype=dtype) + jnp.asarray(rtol, dtype=dtype) * s_typ
        wrms = s_ref / (jnp.asarray(atol, dtype=dtype) + jnp.asarray(rtol, dtype=dtype) * abs_s)
        obs_scale = jnp.asarray(observation_weight_scale, dtype=dtype)
        return obs_scale * jnp.sqrt(w_c) * wrms
    scalar = jnp.sqrt(jnp.asarray(float(weight), dtype=dtype))
    return jnp.full_like(jnp.asarray(s_diff), scalar)


def boundaries_from_windows(windows: Any, ts: np.ndarray, t0: float) -> np.ndarray:
    """Resolve a [Windows][diff_soup.inference.fitting.windows.Windows] spec to the ``(W+1,)`` boundary times.

    Args:
        windows: A decision-variable [Windows][diff_soup.inference.fitting.windows.Windows] (``uniform`` / ``at``).
        ts: The data frame times (the horizon end is ``ts[-1]``).
        t0: The IC time (the horizon start; the state ``y0`` is at ``t0``).

    Returns:
        Monotone boundary times spanning ``[t0, ts[-1]]``.

    Raises:
        ValueError: If an ``at`` spec does not span the data horizon.
    """
    t0 = float(t0)
    t_end = float(ts[-1])
    if windows.mode == "uniform":
        n = int(windows.n_windows)
        return np.linspace(t0, t_end, n + 1)
    # explicit "at" boundaries
    bnd = np.asarray(windows.boundaries, dtype=float)
    if bnd[0] > t0 + 1e-12 or bnd[-1] < t_end - 1e-12:
        raise ValueError(
            f"Windows.at boundaries {tuple(bnd)} must span the data horizon [{t0}, {t_end}]."
        )
    return bnd


def _filter_kept_frames(frames: np.ndarray, kept: Optional[set]) -> np.ndarray:
    """Drop frame indices not observed per a ``frame_mask`` (all kept when ``kept`` is None).

    A windowed-in-time observation (``Series.frame_mask``) marks which frames carry
    a measurement; unobserved frames must not contribute residual rows even though the
    trajectory is still integrated across the full horizon.
    """
    if kept is None:
        return frames
    return np.asarray([int(i) for i in frames if int(i) in kept], dtype=int)


def frames_in_window(ts: np.ndarray, t0: float, t1: float, *, first: bool) -> np.ndarray:
    """Indices of data frames falling in ``(t0, t1]`` (or ``[t0, t1]`` for window 0).

    The first window includes its left endpoint (the IC frame is at ``t0``); later
    windows take the half-open ``(t0, t1]`` so each interior frame belongs to exactly
    one window.
    """
    ts = np.asarray(ts)
    if first:
        return np.where((ts >= t0 - 1e-12) & (ts <= t1 + 1e-12))[0]
    return np.where((ts > t0 + 1e-12) & (ts <= t1 + 1e-12))[0]


class SegmentPlan:
    """Host-side plan mapping one batch of window propagations onto residual rows.

    Every residual row a decision-variable window stack needs is a function of
    either a window-start state or the endpoint of one propagation from a window
    start.  Enumerating the *distinct* propagations once, up front, has two
    consequences that matter:

    * a target time reached by both an observation frame and the next window's
      continuity row is solved **once**, not twice — with boundaries placed on
      the data frames, which is the usual configuration, that halves the solves;
    * the propagations are independent (every window start is a decision
      variable), so they run as a single ``jax.vmap`` instead of a Python loop.
      The traced program is then independent of the window count, and a sparse
      GPU backend factorizes the batch in one dispatch rather than serially.

    Rows are assembled by gathering from one pooled state array,
    ``concatenate([segment_endpoints, window_starts])``, so the indices below are
    offsets into that pool and no per-row branching survives into the trace.

    Attributes:
        seg_window: Source window of each propagation, shape ``(n_segments,)``.
        seg_t0: Start time of each propagation.
        seg_t1: Target time of each propagation.
        obs_pool: Pool index supplying each observation row group's state.
        obs_frames: Data-frame index scored by each observation row group.
        cont_pool: Pool index supplying each continuity row's propagated state.
        cont_windows: Window whose decision block each continuity row constrains.
        n_segments: Number of distinct propagations.
    """

    def __init__(self, seg_window, seg_t0, seg_t1, obs_pool, obs_frames, cont_pool, cont_windows):
        """Store the plan as concrete host arrays."""
        self.seg_window = np.asarray(seg_window, dtype=np.int64)
        self.seg_t0 = np.asarray(seg_t0, dtype=float)
        self.seg_t1 = np.asarray(seg_t1, dtype=float)
        self.obs_pool = np.asarray(obs_pool, dtype=np.int64)
        self.obs_frames = np.asarray(obs_frames, dtype=np.int64)
        self.cont_pool = np.asarray(cont_pool, dtype=np.int64)
        self.cont_windows = np.asarray(cont_windows, dtype=np.int64)

    @property
    def n_segments(self) -> int:
        """Number of distinct propagations in the batch."""
        return int(self.seg_window.shape[0])


def build_segment_plan(win_frames: list, ts: np.ndarray, boundaries: np.ndarray) -> SegmentPlan:
    """Enumerate the distinct window propagations and index the rows onto them.

    A frame coinciding with its window's start time needs no propagation — its
    state *is* the window start — so it reads the window-start section of the
    pool directly.  Every other observation frame, and every continuity row,
    reads a propagation endpoint; identical ``(window, target time)`` requests
    collapse to one entry.

    Args:
        win_frames: Per-window arrays of observed data-frame indices.
        ts: Data frame times.
        boundaries: Window boundary times, length ``n_windows + 1``.

    Returns:
        The [SegmentPlan][diff_soup.inference.fitting._windows_dv.SegmentPlan] for this
        window configuration.
    """
    ts = np.asarray(ts, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    n_windows = len(win_frames)

    seg_window: list[int] = []
    seg_t0: list[float] = []
    seg_t1: list[float] = []

    def segment_index(window: int, t_target: float) -> int:
        """Return the index of ``(window, t_target)``, adding it if it is new."""
        for existing, (w, t1) in enumerate(zip(seg_window, seg_t1)):
            if w == window and abs(t1 - t_target) <= 1e-12 * max(1.0, abs(t_target)):
                return existing
        seg_window.append(int(window))
        seg_t0.append(float(boundaries[window]))
        seg_t1.append(float(t_target))
        return len(seg_window) - 1

    # Continuity requests come first so the aligned configuration — boundaries on
    # the frames — lays the batch out in window order.
    cont_pool_segments = [segment_index(w - 1, float(boundaries[w])) for w in range(1, n_windows)]

    obs_segments: list[tuple[int, bool, int]] = []
    obs_frames: list[int] = []
    for window in range(n_windows):
        t_start = float(boundaries[window])
        for frame in win_frames[window]:
            t_frame = float(ts[int(frame)])
            obs_frames.append(int(frame))
            if abs(t_frame - t_start) <= 1e-12:
                obs_segments.append((window, True, window))
            else:
                obs_segments.append((window, False, segment_index(window, t_frame)))

    # Pool layout: propagation endpoints first, then the window-start states.
    n_segments = len(seg_window)
    obs_pool = [
        (n_segments + index) if from_start else index for _w, from_start, index in obs_segments
    ]
    cont_pool = list(cont_pool_segments)
    cont_windows = list(range(1, n_windows))
    return SegmentPlan(seg_window, seg_t0, seg_t1, obs_pool, obs_frames, cont_pool, cont_windows)


def warn_if_over_ceiling(n_decision: int) -> Optional[str]:
    """Warn when the decision vector exceeds the dense-GN ceiling."""
    if n_decision > _DENSE_GN_CEIL:
        msg = (
            f"Multishoot (decision-variable windows): the decision vector has "
            f"{n_decision} variables (> {_DENSE_GN_CEIL}); the dense Gauss-Newton "
            "normal-equation solve is the cost wall. Declare a StateBasis to "
            "represent the window starts in a subspace, reduce the window count, or "
            "narrow the learnable set."
        )
        warnings.warn(msg, stacklevel=3)
        logger.warning(msg)
        return msg
    return None


class DecisionWindowBuild:
    """Built closures + host metadata for a decision-variable [Multishoot][diff_soup.inference.fitting.inference_problem.Multishoot].

    Encapsulates the residual stack so ``inference_problem.py`` stays thin: the
    per-window propagation through ``pdesolve`` (theta threaded for AD,
    consistent-projection at window starts for DAEs), the obs / continuity /
    state-prior rows, the manifold-init x0, and the window diagnostics.
    """

    def __init__(
        self,
        *,
        seam: Any,
        solver: Any,
        series: Any,
        layout: WindowLayout,
        y0: jax.Array,
        continuity: Any,
        state_prior_weight: float,
        theta_only_steps: int,
        project_windows: bool,
        has_algebraic: bool,
        atol: float,
        rtol: float,
        failed_sentinel: float,
        states: Any = None,
        initialization: Any = None,
        restart_step: Any = None,
    ):
        """Build the decision-variable window closures (eager, outside any trace)."""
        from diff_soup import SaveAt
        from diff_soup.inference.fitting.windows import StateBasis

        self.seam = seam
        self.solver = solver
        self.series = series
        self._likelihood = getattr(series, "likelihood", None)
        self._likelihood_context = getattr(series, "likelihood_context", None)
        self._observation_weight_scale = observation_row_scale(series)
        self.layout = layout
        self.states = states if states is not None else StateBasis.full()
        self._modes = None if self.states.is_full else jnp.asarray(self.states.modes)
        self.y0 = jnp.asarray(y0)
        self.continuity = continuity
        self.state_prior_weight = float(state_prior_weight)
        self.theta_only_steps = int(theta_only_steps)
        # Project window starts exactly when algebraic rows exist and projection
        # has not been disabled.
        self.project = bool(has_algebraic and project_windows)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self._sentinel = float(failed_sentinel)
        self._saveat_t1 = SaveAt(t1=True)
        self.restart_step = restart_step
        self._initialization_provenance = (
            None if initialization is None else initialization.provenance
        )
        self._baseline_reached_tend = (
            None if initialization is None else bool(initialization.reached_tend)
        )

        bnd = layout.boundaries
        ts = np.asarray(series.ts)
        # Frame index sets per window (frames belong to exactly one window).  A
        # frame_mask (windowed-in-time observation) drops unobserved frames so they
        # add no residual rows; the trajectory is still solved over the full horizon.
        if series.frame_mask is None:
            kept_frames: Optional[set] = None
        else:
            observed = np.where(np.asarray(series.frame_mask).reshape((-1,)).astype(bool))[0]
            kept_frames = {int(i) for i in observed}
        self._win_frames = [
            _filter_kept_frames(
                frames_in_window(ts, float(bnd[w]), float(bnd[w + 1]), first=(w == 0)),
                kept_frames,
            )
            for w in range(layout.n_windows)
        ]
        self._plan = build_segment_plan(self._win_frames, ts, bnd)
        self._diff_idx = jnp.asarray(np.where(layout.diff_mask)[0])
        # The forward-sweep states: the manifold init, the optional state prior's
        # reference, and — under a reduced basis — the per-window anchors that zero
        # coefficients reproduce.
        if initialization is None:
            self._sweep_states = self._forward_sweep_states(jnp.asarray(self.seam.x0))
            self._anchor_stack = jnp.asarray(
                np.stack([self._sweep_states[w] for w in range(layout.n_windows)]),
                dtype=self.y0.dtype,
            )
            self._x0 = self._manifold_init()
        else:
            anchors = np.asarray(initialization.anchors)
            expected = (layout.n_windows, layout.n_diff)
            if anchors.shape != expected:
                raise ValueError(
                    f"WindowInitialization anchors have shape {anchors.shape}; expected {expected}."
                )
            if not np.all(np.isfinite(anchors)):
                raise ValueError("WindowInitialization anchors must be finite.")
            self._anchor_stack = jnp.asarray(anchors, dtype=self.y0.dtype)
            self._sweep_states = {
                window: anchors[window].copy() for window in range(layout.n_windows)
            }
            coordinates = initialization.initial_coordinates
            if coordinates is None:
                if self._modes is None:
                    coordinates = anchors[layout.first_state_window :]
                else:
                    coordinates = np.zeros(
                        (layout.n_decision_windows, layout.n_state),
                        dtype=anchors.dtype,
                    )
            coordinates = np.asarray(coordinates)
            expected_coordinates = (layout.n_decision_windows, layout.n_state)
            if coordinates.shape != expected_coordinates:
                raise ValueError(
                    "WindowInitialization initial_coordinates have shape "
                    f"{coordinates.shape}; expected {expected_coordinates}."
                )
            if not np.all(np.isfinite(coordinates)):
                raise ValueError("WindowInitialization initial_coordinates must be finite.")
            self._x0 = np.concatenate((np.asarray(self.seam.x0), coordinates.reshape((-1,))))

    # --- propagation --------------------------------------------------------

    def _theta_pytree(self, x: jax.Array) -> Any:
        """Reconstruct the full θ pytree from the θ̃ block of x."""
        return self.seam.theta_of(self.layout.theta_of(x))

    def _full_start(self, s_diff: jax.Array) -> jax.Array:
        """Reconstruct a full window-start state from the differential block."""
        return self.layout.scatter_diff(s_diff, self.y0)

    def _propagate(
        self,
        theta_p: Any,
        y_start: jax.Array,
        t0: float,
        t1: float,
        segment_index: Any = 0,
    ) -> Any:
        """Solve one window forward from ``y_start``.

        Returns the full
        [Solution][diff_soup.timestepping.vsvo.solution.Solution].
        """
        return self._propagate_traced(theta_p, y_start, t0, t1, segment_index)

    def _propagate_traced(
        self,
        theta_p: Any,
        y_start: jax.Array,
        t0: Any,
        t1: Any,
        segment_index: Any = 0,
    ) -> Any:
        """Solve one segment forward; ``t0``/``t1`` may be traced (mapped) scalars.

        The save specification is ``t1``-only and built once outside any trace, so
        the interval endpoints are the only per-segment values and the batch maps
        over them directly.
        """
        from diff_soup import pdesolve

        kwargs = dict(
            saveat=self._saveat_t1,
            theta=theta_p,
            adjoint=getattr(self.solver, "adjoint", None),
        )
        if self.project:
            # Consistent-projection window start: solve the algebraic block at t0 so
            # the reconstructed start lies on the DAE manifold.
            kwargs["consistent_ic"] = True
        if self.restart_step is not None:
            kwargs["dt0"] = self.restart_step.resolve(t0, t1, segment_index)
        return pdesolve(self.seam.problem, self.solver, (t0, t1), y_start, **kwargs)

    def _window_start_state(self, x: jax.Array, w: int) -> jax.Array:
        """Full start state for window ``w``.

        Window 0 is the fixed IC ``y0`` unless its initial state is learned, in which
        case (like every interior window) it is reconstructed from its decision block
        — lifted through the state basis first when one is declared.
        """
        if w == 0 and not self.layout.learn_initial:
            return self.y0
        index = w - self.layout.first_state_window
        s_diff = self._lift_blocks(self.layout.state_blocks(x))[index]
        return self._full_start(s_diff)

    def states_at_frames(self, x: jax.Array, w: int) -> list:
        """Predicted full states at every observed frame in window ``w``."""
        theta_p = self._theta_pytree(x)
        bnd = self.layout.boundaries
        t0 = float(bnd[w])
        y_start = self._window_start_state(x, w)
        out = []
        for fi in self._win_frames[w]:
            tf = float(np.asarray(self.series.ts)[fi])
            if abs(tf - t0) <= 1e-12:
                # The frame coincides with the window start (window-0 IC frame).
                out.append((fi, y_start))
                continue
            sol = self._propagate(theta_p, y_start, t0, tf)
            out.append((fi, sol.ys[-1]))
        return out

    # --- residual stack -----------------------------------------------------

    def residual(self, x: jax.Array) -> jax.Array:
        """The full residual stack at x (obs ++ continuity ++ state-prior)."""
        return self.residual_kernel(
            self._theta_pytree(x),
            self.layout.state_blocks(x),
            self.y0,
            self._anchor_stack,
            jnp.asarray(self.series.values),
            None if self.series.weight is None else jnp.asarray(self.series.weight),
        )

    @staticmethod
    def _state_pool(ends: jax.Array, starts: jax.Array) -> jax.Array:
        """Pool physical endpoints and possibly global-augmented solve starts."""
        if ends.shape[0] == 0:
            return starts
        if ends.shape[1:] != starts.shape[1:]:
            # Endpoint requests and window starts are two different pools, so
            # their leading counts are not required to match.  Only any batch
            # axes between that pool axis and the state axis must agree.
            if (
                ends.ndim != starts.ndim
                or ends.shape[1:-1] != starts.shape[1:-1]
                or starts.shape[-1] < ends.shape[-1]
            ):
                raise ValueError(
                    "window endpoints and starts have incompatible state shapes: "
                    f"{ends.shape} versus {starts.shape}"
                )
            # A DAE solve start may append global/Lagrange multiplier values
            # that pdesolve consumes but does not return in Solution. They are
            # not physical observation or continuity coordinates.
            starts = starts[..., : ends.shape[-1]]
        return jnp.concatenate([ends, starts])

    def residual_kernel(
        self,
        theta_p: Any,
        state_block: jax.Array,
        y0: jax.Array,
        anchors: jax.Array,
        targets: jax.Array,
        weight: Optional[jax.Array],
        *,
        parallel_segments: bool = True,
        with_steps: bool = False,
    ) -> Any:
        """The residual stack as a pure function of one trajectory's dynamic data.

        This is the single implementation of the window residual algebra. The leaf
        problem calls it with its own arrays through
        [residual][diff_soup.inference.fitting._windows_dv.DecisionWindowBuild.residual]; a
        homogeneous joint maps the same function over its child axis, passing each
        child's initial state, anchors, targets, and weights. Keeping one algebra
        with two callers is what stops the joint and leaf paths from drifting.

        Every propagation the stack needs runs through one mapped operation over
        the plan's distinct ``(window, target time)`` requests, so the traced
        program does not grow with the window count and a target reached by both
        an observation frame and a continuity row is solved once. Leaf
        problems use `jax.vmap`; a homogeneous joint passes
        ``parallel_segments=False`` because it already vmaps children and sparse
        direct factor banks support one matrix-batch axis.

        Any unsuccessful or non-finite propagation stamps the whole stack with
        the sentinel, so a step that broke the integration cannot be accepted
        on the strength of rows that happened to survive it.

        Args:
            theta_p: The reconstructed parameter pytree.
            state_block: Decision blocks ``(n_decision_windows, n_state)``.
            y0: The trajectory's initial full state.
            anchors: Per-window anchor differential states ``(n_windows, n_diff)``.
            targets: Observed values in observable space.
            weight: Optional ``1/σ`` rows, or ``None``.
            parallel_segments: Use a segment ``vmap`` when true. When false,
                execute segments through ``lax.map`` so an enclosing child vmap
                remains the only matrix-batch axis.
            with_steps: Also return the integrator's trial-step count summed
                over the trajectory's segments, so a sharded joint can report
                this child's real step count.

        Returns:
            The flat residual stack for this trajectory, or ``(stack, steps)``
            when ``with_steps`` is set.
        """
        starts = self._start_stack_from(state_block, y0, anchors)
        ends, any_failed, steps = self._propagate_segments(
            theta_p, starts, parallel=parallel_segments
        )
        pool = self._state_pool(ends, starts)
        rows = [
            self._obs_rows_from(theta_p, pool, targets, weight),
            self._continuity_rows_from(state_block, anchors, pool),
        ]
        if self.state_prior_weight > 0.0:
            rows.append(self._state_prior_rows_from(state_block, anchors))
        stack = jnp.concatenate(rows)
        stack = jnp.where(any_failed, jnp.full_like(stack, self._sentinel), stack)
        return (stack, steps) if with_steps else stack

    def _lift_state_block(
        self,
        state_block: jax.Array,
        anchors: jax.Array,
        window: int,
    ) -> jax.Array:
        """Lift one decision-state block without touching sibling windows."""
        layout = self.layout
        block = jnp.asarray(state_block)[window - layout.first_state_window]
        if self._modes is None:
            return block
        return jnp.asarray(anchors)[window] + block @ self._modes.T

    def _window_start_from(
        self,
        state_block: jax.Array,
        y0: jax.Array,
        anchors: jax.Array,
        window: int,
    ) -> jax.Array:
        """Reconstruct exactly one window start from its declared coordinate."""
        if window < self.layout.first_state_window:
            return jnp.asarray(y0)
        return self._full_start(self._lift_state_block(state_block, anchors, window))

    def window_residual_kernel(
        self,
        theta_p: Any,
        state_block: jax.Array,
        y0: jax.Array,
        anchors: jax.Array,
        targets: jax.Array,
        weight: Optional[jax.Array],
        window: int,
    ) -> jax.Array:
        """Evaluate one exact window-local residual block.

        The block contains that window's observation rows, its outgoing
        continuity rows, and the prior on its own decision-state coordinate.
        Consequently it observes at most the parameter prefix, the current
        state block, and the adjacent next state block.

        A declared likelihood scores this window's frames on their own, which
        reproduces the complete stack's rows for any likelihood that
        factorizes over frames.
        """
        window = int(window)
        layout = self.layout
        plan = self._plan
        if not 0 <= window < layout.n_windows:
            raise IndexError(f"window must be in [0, {layout.n_windows}); got {window}")

        segment_indices = np.where(plan.seg_window == window)[0]
        segment_lookup = {
            int(segment): position for position, segment in enumerate(segment_indices.tolist())
        }
        observation_positions = []
        for position, pool_index in enumerate(plan.obs_pool.tolist()):
            if pool_index < plan.n_segments:
                source_window = int(plan.seg_window[pool_index])
            else:
                source_window = int(pool_index - plan.n_segments)
            if source_window == window:
                observation_positions.append(position)
        continuity_positions = [
            position
            for position, target_window in enumerate(plan.cont_windows.tolist())
            if int(target_window) == window + 1
        ]

        start = self._window_start_from(state_block, y0, anchors, window)
        if segment_indices.size:
            actual_indices = jnp.asarray(segment_indices, dtype=jnp.int32)
            t0 = jnp.asarray(plan.seg_t0[segment_indices])
            t1 = jnp.asarray(plan.seg_t1[segment_indices])

            def propagate(segment_index, start_time, end_time):
                solution = self._propagate_traced(
                    theta_p,
                    start,
                    start_time,
                    end_time,
                    segment_index,
                )
                return solution.ys[-1], _solution_failed(solution)

            endpoints, failures = jax.vmap(propagate)(actual_indices, t0, t1)
            any_failed = jnp.any(failures)
        else:
            endpoints = jnp.zeros((0,) + start.shape, dtype=start.dtype)
            any_failed = jnp.asarray(False)

        rows = []
        if observation_positions:
            predicted = []
            frames = []
            for position in observation_positions:
                pool_index = int(plan.obs_pool[position])
                frames.append(int(plan.obs_frames[position]))
                if pool_index < plan.n_segments:
                    predicted.append(endpoints[segment_lookup[pool_index]])
                else:
                    predicted.append(start)
            frame_array = jnp.asarray(frames, dtype=jnp.int32)
            predicted_array = jnp.stack(predicted)
            times = jnp.asarray(np.asarray(self.series.ts)[frames])
            observed = self.series.observe(
                predicted_array,
                ts=times,
                theta=theta_p,
            )
            likelihood, context = self._likelihood_view(frames)
            if likelihood is not None:
                observation_rows = likelihood.least_squares_rows(
                    jnp.asarray(targets)[frame_array],
                    observed,
                    context=context,
                )
            else:
                observation_rows = observed - jnp.asarray(targets)[frame_array]
                if weight is not None:
                    weight_array = jnp.asarray(weight)
                    frame_aligned = (
                        weight_array.ndim >= 1
                        and weight_array.shape[0] == np.asarray(self.series.ts).shape[0]
                    )
                    observation_rows = (
                        weight_array[frame_array] if frame_aligned else weight_array
                    ) * observation_rows
            rows.append(jnp.reshape(observation_rows, (-1,)))

        if continuity_positions:
            if len(continuity_positions) != 1:
                raise ValueError(
                    f"window {window} has {len(continuity_positions)} continuity targets"
                )
            position = continuity_positions[0]
            pool_index = int(plan.cont_pool[position])
            predicted = endpoints[segment_lookup[pool_index]][self._diff_idx]
            target_window = int(plan.cont_windows[position])
            target = self._lift_state_block(
                state_block,
                anchors,
                target_window,
            )
            sqrt_wc = resolve_continuity_weight(
                self.continuity.weight,
                target,
                self.atol,
                self.rtol,
                observation_weight_scale=self._observation_weight_scale,
            )
            mismatch = sqrt_wc * (target - predicted)
            if self._modes is not None:
                mismatch = mismatch @ self._modes
            rows.append(jnp.reshape(mismatch, (-1,)))

        if self.state_prior_weight > 0.0 and window >= layout.first_state_window:
            block = jnp.asarray(state_block)[window - layout.first_state_window]
            sqrt_weight = jnp.sqrt(jnp.asarray(self.state_prior_weight, dtype=block.dtype))
            if self._modes is None:
                prior = sqrt_weight * (block - jnp.asarray(anchors)[window])
            else:
                prior = sqrt_weight * block
            rows.append(jnp.reshape(prior, (-1,)))

        stack = (
            jnp.concatenate(rows) if rows else jnp.zeros((0,), dtype=jnp.asarray(state_block).dtype)
        )
        return jnp.where(
            any_failed,
            jnp.full_like(stack, self._sentinel),
            stack,
        )

    def window_residual(self, x: jax.Array, window: int) -> jax.Array:
        """Evaluate one window-local block from the complete child point."""
        return self.window_residual_kernel(
            self._theta_pytree(x),
            self.layout.state_blocks(x),
            self.y0,
            self._anchor_stack,
            jnp.asarray(self.series.values),
            None if self.series.weight is None else jnp.asarray(self.series.weight),
            window,
        )

    def residual_dependency_blocks(self) -> tuple[ResidualDependencyBlock, ...]:
        """Declare the exact parameter/current/adjacent dependency per window."""
        layout = self.layout
        plan = self._plan
        n_observed_values = int(np.prod(np.asarray(self.series.values).shape[1:]))
        blocks = []
        for window in range(layout.n_windows):
            columns = list(range(layout.n_theta))
            elimination_groups = []
            if window >= layout.first_state_window:
                start = layout.n_theta + (window - layout.first_state_window) * layout.n_state
                current_group = tuple(range(start, start + layout.n_state))
                columns.extend(current_group)
                elimination_groups.append(current_group)
            has_continuity = bool(np.any(plan.cont_windows == window + 1))
            if has_continuity:
                target = layout.n_theta + (window + 1 - layout.first_state_window) * layout.n_state
                next_group = tuple(range(target, target + layout.n_state))
                columns.extend(next_group)
                elimination_groups.append(next_group)

            observation_count = 0
            for pool_index in plan.obs_pool.tolist():
                if pool_index < plan.n_segments:
                    source_window = int(plan.seg_window[pool_index])
                else:
                    source_window = int(pool_index - plan.n_segments)
                observation_count += int(source_window == window)
            row_count = observation_count * max(n_observed_values, 1)
            if has_continuity:
                row_count += layout.n_state
            if self.state_prior_weight > 0.0 and window >= layout.first_state_window:
                row_count += layout.n_state

            def residual(point, *, _window=window):
                return self.window_residual(point, _window)

            blocks.append(
                ResidualDependencyBlock(
                    name=f"window-{window}",
                    columns=tuple(columns),
                    row_count=row_count,
                    residual_fn=residual,
                    elimination_groups=tuple(elimination_groups),
                    metadata=(("window_index", window),),
                )
            )
        return tuple(blocks)

    def residual_dependency_buckets(self) -> tuple[ResidualDependencyBucket, ...]:
        """Bucket aligned one-segment windows behind shared traced bodies.

        More general window plans retain the exact per-block declarations and
        use the safe unbucketed path. The aligned case covers the common design
        with boundaries on observation frames: one propagation/observation per
        window and one outgoing continuity row except at the final window.

        One shared traced body serves every lane, so it cannot carry a
        per-lane likelihood view; a likelihood-scored stream therefore keeps
        the per-block path.
        """
        layout = self.layout
        plan = self._plan
        if self._likelihood is not None:
            return ()
        blocks = self.residual_dependency_blocks()
        segments_by_window = [
            np.where(plan.seg_window == window)[0] for window in range(layout.n_windows)
        ]
        observations_by_window: list[list[int]] = [[] for _ in range(layout.n_windows)]
        for position, pool_index in enumerate(plan.obs_pool.tolist()):
            if pool_index >= plan.n_segments:
                return ()
            source = int(plan.seg_window[pool_index])
            observations_by_window[source].append(position)
        continuity_by_source: list[list[int]] = [[] for _ in range(layout.n_windows)]
        for position, target_window in enumerate(plan.cont_windows.tolist()):
            continuity_by_source[int(target_window) - 1].append(position)
        for window in range(layout.n_windows):
            segments = segments_by_window[window]
            observations = observations_by_window[window]
            continuity = continuity_by_source[window]
            if len(segments) != 1 or len(observations) != 1:
                return ()
            segment = int(segments[0])
            if int(plan.obs_pool[observations[0]]) != segment:
                return ()
            expected_continuity = 0 if window == layout.n_windows - 1 else 1
            if len(continuity) != expected_continuity:
                return ()
            if continuity and int(plan.cont_pool[continuity[0]]) != segment:
                return ()

        grouped: dict[tuple[bool, bool, bool, int], list[int]] = {}
        for window, block in enumerate(blocks):
            has_current = window >= layout.first_state_window
            has_next = window < layout.n_windows - 1
            has_prior = bool(self.state_prior_weight > 0.0 and has_current)
            key = (has_current, has_next, has_prior, block.row_count)
            grouped.setdefault(key, []).append(window)

        targets_all = jnp.asarray(self.series.values)
        times_all = np.asarray(self.series.ts)
        weight_all = None if self.series.weight is None else jnp.asarray(self.series.weight)
        frame_aligned_weight = bool(
            weight_all is not None
            and weight_all.ndim >= 1
            and weight_all.shape[0] == times_all.shape[0]
        )
        buckets = []
        for bucket_index, (
            (has_current, has_next, has_prior, row_count),
            windows,
        ) in enumerate(grouped.items()):
            segment_indices = np.asarray(
                [int(segments_by_window[window][0]) for window in windows],
                dtype=np.int32,
            )
            observation_positions = [observations_by_window[window][0] for window in windows]
            frames = np.asarray(
                [int(plan.obs_frames[position]) for position in observation_positions],
                dtype=np.int32,
            )
            start_times = jnp.asarray(plan.seg_t0[segment_indices])
            end_times = jnp.asarray(plan.seg_t1[segment_indices])
            observation_times = jnp.asarray(times_all[frames])
            targets = targets_all[jnp.asarray(frames)]
            lane_weights = weight_all[jnp.asarray(frames)] if frame_aligned_weight else weight_all
            current_anchors = (
                jnp.asarray(self._anchor_stack)[jnp.asarray(windows)] if has_current else None
            )
            next_anchors = (
                jnp.asarray(self._anchor_stack)[jnp.asarray([window + 1 for window in windows])]
                if has_next
                else None
            )
            n_theta = layout.n_theta
            n_state = layout.n_state
            modes = self._modes
            y0 = self.y0
            sentinel = self._sentinel

            def residual(
                active,
                lane,
                *,
                _has_current=has_current,
                _has_next=has_next,
                _has_prior=has_prior,
                _segment_indices=jnp.asarray(segment_indices),
                _start_times=start_times,
                _end_times=end_times,
                _observation_times=observation_times,
                _targets=targets,
                _lane_weights=lane_weights,
                _frame_aligned_weight=frame_aligned_weight,
                _current_anchors=current_anchors,
                _next_anchors=next_anchors,
                _modes=modes,
                _y0=y0,
                _sentinel=sentinel,
            ):
                active = jnp.asarray(active)
                theta_p = self.seam.theta_of(active[:n_theta])
                offset = n_theta
                current = None
                if _has_current:
                    current = active[offset : offset + n_state]
                    offset += n_state
                    current_diff = (
                        current if _modes is None else _current_anchors[lane] + current @ _modes.T
                    )
                    start = layout.scatter_diff(current_diff, _y0)
                else:
                    start = jnp.asarray(_y0)
                next_state = None
                if _has_next:
                    next_state = active[offset : offset + n_state]

                solution = self._propagate_traced(
                    theta_p,
                    start,
                    _start_times[lane],
                    _end_times[lane],
                    _segment_indices[lane],
                )
                endpoint = solution.ys[-1]
                observed = self.series.observe(
                    endpoint[None],
                    ts=jnp.reshape(_observation_times[lane], (1,)),
                    theta=theta_p,
                )[0]
                observation_rows = observed - _targets[lane]
                if _lane_weights is not None:
                    observation_rows = (
                        _lane_weights[lane] if _frame_aligned_weight else _lane_weights
                    ) * observation_rows
                rows = [jnp.reshape(observation_rows, (-1,))]

                if _has_next:
                    target_diff = (
                        next_state
                        if _modes is None
                        else _next_anchors[lane] + next_state @ _modes.T
                    )
                    predicted_diff = endpoint[self._diff_idx]
                    sqrt_wc = resolve_continuity_weight(
                        self.continuity.weight,
                        target_diff,
                        self.atol,
                        self.rtol,
                        observation_weight_scale=self._observation_weight_scale,
                    )
                    mismatch = sqrt_wc * (target_diff - predicted_diff)
                    if _modes is not None:
                        mismatch = mismatch @ _modes
                    rows.append(jnp.reshape(mismatch, (-1,)))
                if _has_prior:
                    sqrt_weight = jnp.sqrt(
                        jnp.asarray(self.state_prior_weight, dtype=current.dtype)
                    )
                    prior = (
                        sqrt_weight * current
                        if _modes is not None
                        else sqrt_weight * (current - _current_anchors[lane])
                    )
                    rows.append(jnp.reshape(prior, (-1,)))
                stack = jnp.concatenate(rows)
                failed = _solution_failed(solution)
                return jnp.where(
                    failed,
                    jnp.full_like(stack, _sentinel),
                    stack,
                )

            buckets.append(
                ResidualDependencyBucket(
                    name=f"aligned-window-{bucket_index}",
                    columns=tuple(blocks[window].columns for window in windows),
                    row_count=row_count,
                    residual_fn=residual,
                    metadata=(("windows", tuple(windows)),),
                )
            )
        if sum(bucket.block_count for bucket in buckets) != len(blocks):
            return ()
        return tuple(buckets)

    def rows_per_trajectory(self) -> int:
        """Residual row count one trajectory contributes (host-side, static)."""
        plan = self._plan
        n_obs_values = int(np.prod(np.asarray(self.series.values).shape[1:]))
        n_obs = int(plan.obs_pool.shape[0]) * max(n_obs_values, 1)
        n_cont = int(plan.cont_pool.shape[0]) * self.layout.n_state
        n_prior = (
            self.layout.n_decision_windows * self.layout.n_state
            if self.state_prior_weight > 0.0
            else 0
        )
        return n_obs + n_cont + n_prior

    def anchors(self) -> jax.Array:
        """Per-window anchor differential states, shape ``(n_windows, n_diff)``.

        The anchors are the manifold-init forward-sweep states.  Under a reduced
        basis a window start is ``anchor + B a``, so zero coefficients reproduce the
        unreduced starting point exactly and declaring a basis never moves the
        initial iterate.
        """
        return self._anchor_stack

    def _lift_blocks(self, blocks: jax.Array, anchors: Optional[jax.Array] = None) -> jax.Array:
        """Map decision blocks to differential window-start states.

        In the full coordinate the blocks *are* the differential states.  Under a
        reduced basis they are subspace coefficients and the state is
        ``anchor + B a``.
        """
        if self._modes is None:
            return blocks
        stack = self._anchor_stack if anchors is None else anchors
        return stack[self.layout.first_state_window :] + blocks @ self._modes.T

    def _start_stack_from(
        self, state_block: jax.Array, y0: jax.Array, anchors: jax.Array
    ) -> jax.Array:
        """Full window-start states for every window, shape ``(n_windows, state)``.

        Window 0 is the fixed initial state unless it is learned; the remaining
        window starts scatter their differential values over the initial state's
        algebraic values, which puts a consistent-projection start within reach of
        the algebraic solve.
        """
        layout = self.layout
        y0 = jnp.asarray(y0)
        stack = jnp.broadcast_to(y0, (layout.n_windows,) + y0.shape)
        if layout.n_decision_windows == 0:
            return stack
        s_diff = self._lift_blocks(state_block, anchors)
        rows = jnp.asarray(np.arange(layout.first_state_window, layout.n_windows))
        return stack.at[rows[:, None], self._diff_idx[None, :]].set(s_diff)

    def _start_stack(self, x: jax.Array) -> jax.Array:
        """Window-start states at the decision vector ``x``."""
        return self._start_stack_from(self.layout.state_blocks(x), self.y0, self._anchor_stack)

    def _propagate_segments(
        self, theta_p: Any, starts: jax.Array, *, parallel: bool = True
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Propagate every planned window segment in one batch.

        Returns:
            ``(endpoints, any_failed, trial_steps)`` with endpoints shape
            ``(n_segments, state)``, a scalar boolean set when any segment hit
            the solver's step limit, and the integrator's trial-step count
            summed over the segments (zero when the solver reports none).
        """
        plan = self._plan
        starts = jnp.asarray(starts)
        if plan.n_segments == 0:
            return (
                jnp.zeros((0,) + starts.shape[1:], dtype=starts.dtype),
                jnp.asarray(False),
                jnp.zeros((), dtype=jnp.int32),
            )
        seg_starts = starts[jnp.asarray(plan.seg_window)]

        def one(theta, segment_index, t0, t1, y_start):
            sol = self._propagate_traced(theta, y_start, t0, t1, segment_index)
            return sol.ys[-1], _solution_failed(sol), _trial_steps(sol)

        segment_data = (
            jnp.arange(plan.n_segments, dtype=jnp.int32),
            jnp.asarray(plan.seg_t0),
            jnp.asarray(plan.seg_t1),
            seg_starts,
        )
        if parallel:
            # On a forward-only seam an enclosing vmap over parameter points folds
            # into the segment axis, keeping the solves on one matrix-batch axis.
            ends, failed, steps = lane_map(
                one,
                (None, 0, 0, 0, 0),
                theta_p,
                *segment_data,
                fold=folds_enclosing_batches(self.solver),
            )
        else:
            ends, failed, steps = jax.lax.map(lambda seg: one(theta_p, *seg), segment_data)
        return ends, jnp.any(failed), jnp.sum(steps)

    def _segment_endpoints(self, x: jax.Array, starts: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Segment endpoints at the decision vector ``x``."""
        ends, failed, _steps = self._propagate_segments(self._theta_pytree(x), starts)
        return ends, failed

    def _likelihood_view(self, frames: Any) -> tuple:
        """Declared likelihood and context restricted to the given data frames.

        Returns ``(None, None)`` when the stream is weighted rather than
        likelihood-scored.
        """
        if self._likelihood is None:
            return None, None
        return self._likelihood.for_frames(
            jnp.asarray(np.asarray(frames, dtype=np.int64)),
            full_prediction_shape=tuple(np.asarray(self.series.values).shape),
            context=self._likelihood_context,
        )

    def _obs_rows_from(
        self,
        theta_p: Any,
        pool: jax.Array,
        targets: jax.Array,
        weight: Optional[jax.Array],
    ) -> jax.Array:
        """Observation-misfit rows ``w_f ⊙ (h(u(t_f)) − d_f)`` over all windows.

        A stream declaring a likelihood is scored by that likelihood's own
        least-squares rows over the complete retained observation event, so
        the windows minimize the same objective a single shoot does.
        """
        plan = self._plan
        pool = jnp.asarray(pool)
        if plan.obs_pool.shape[0] == 0:
            return jnp.zeros((0,), dtype=pool.dtype)
        ts = np.asarray(self.series.ts)
        frames = jnp.asarray(plan.obs_frames)
        predicted = pool[jnp.asarray(plan.obs_pool)]
        obs = self.series.observe(predicted, ts=jnp.asarray(ts[plan.obs_frames]), theta=theta_p)
        observed = jnp.asarray(targets)[frames]
        likelihood, context = self._likelihood_view(plan.obs_frames)
        if likelihood is not None:
            rows = likelihood.least_squares_rows(observed, obs, context=context)
            return jnp.reshape(rows, (-1,))
        r = obs - observed
        # Each row group compares against a single frame, so it takes that frame's
        # ``1/σ`` row.  A frame-aligned weight (leading axis == n_frames) is indexed
        # at the frame; a scalar or per-state weight broadcasts identically.
        if weight is not None:
            w_arr = jnp.asarray(weight)
            frame_aligned = w_arr.ndim >= 1 and w_arr.shape[0] == ts.shape[0]
            r = (w_arr[frames] if frame_aligned else w_arr) * r
        return jnp.reshape(r, (-1,))

    def _weighted_mismatch_from(
        self, state_block: jax.Array, anchors: jax.Array, pool: jax.Array
    ) -> jax.Array:
        """Weighted continuity mismatch per window, shape ``(n_continuity, n_diff)``.

        This is ``√W_c ⊙ (s̃_w − Φ_diff(s_{w−1}, θ))`` in state units, before any
        projection.  The full coordinate uses it directly as the continuity rows;
        a reduced basis projects it and reports the discarded part.
        """
        plan = self._plan
        layout = self.layout
        s_all = self._lift_blocks(state_block, anchors)
        rows = jnp.asarray(plan.cont_windows - layout.first_state_window)
        s_diff = s_all[rows]
        pred_diff = jnp.asarray(pool)[jnp.asarray(plan.cont_pool)][:, self._diff_idx]
        sqrt_wc = jax.vmap(
            lambda s: resolve_continuity_weight(
                self.continuity.weight,
                s,
                self.atol,
                self.rtol,
                observation_weight_scale=self._observation_weight_scale,
            )
        )(s_diff)
        return sqrt_wc * (s_diff - pred_diff)

    def weighted_continuity_mismatch(self, x: jax.Array) -> jax.Array:
        """Public weighted mismatch at ``x``, propagating the windows once."""
        starts = self._start_stack(x)
        ends, _failed = self._segment_endpoints(x, starts)
        pool = self._state_pool(ends, starts)
        return self._weighted_mismatch_from(self.layout.state_blocks(x), self._anchor_stack, pool)

    def _continuity_rows_from(
        self, state_block: jax.Array, anchors: jax.Array, pool: jax.Array
    ) -> jax.Array:
        """Continuity rows: the weighted mismatch, projected onto the state basis.

        In the full coordinate the rows are the weighted mismatch itself.  Under a
        reduced basis they are its Galerkin projection ``Bᵀ`` — ``n_modes`` rows per
        window instead of ``n_diff``.  Because ``B`` is orthonormal the projected
        rows carry exactly the norm of the mismatch's in-subspace part, and what the
        subspace cannot represent is reported by
        [window_diagnostics][diff_soup.inference.fitting._windows_dv.DecisionWindowBuild.window_diagnostics]
        rather than silently dropped.
        """
        plan = self._plan
        if plan.cont_pool.shape[0] == 0:
            return jnp.zeros((0,), dtype=jnp.asarray(pool).dtype)
        weighted = self._weighted_mismatch_from(state_block, anchors, pool)
        if self._modes is None:
            return jnp.reshape(weighted, (-1,))
        return jnp.reshape(weighted @ self._modes, (-1,))

    def _continuity_rows_at(self, x: jax.Array) -> jax.Array:
        """Continuity rows at ``x``, propagating the windows once."""
        starts = self._start_stack(x)
        ends, _failed = self._segment_endpoints(x, starts)
        pool = self._state_pool(ends, starts)
        return self._continuity_rows_from(self.layout.state_blocks(x), self._anchor_stack, pool)

    def _state_prior_rows_from(self, state_block: jax.Array, anchors: jax.Array) -> jax.Array:
        """State-prior rows ``√w_p ⊙ (s̃ − s̃_forward-sweep)`` (default off).

        Under a reduced basis the sweep states *are* the anchors, so the deviation
        is ``B a`` and the rows reduce to ``√w_p ⊙ a`` — the same penalty, in
        ``n_modes`` rows instead of ``n_diff``, because ``B`` is orthonormal.
        """
        layout = self.layout
        blocks = jnp.asarray(state_block)
        if layout.n_decision_windows == 0:
            return jnp.zeros((0,), dtype=blocks.dtype)
        sqrt_wp = jnp.sqrt(jnp.asarray(self.state_prior_weight, dtype=blocks.dtype))
        if self._modes is not None:
            return jnp.reshape(sqrt_wp * blocks, (-1,))
        reference = jnp.asarray(anchors)[layout.first_state_window :]
        return jnp.reshape(sqrt_wp * (blocks - reference), (-1,))

    def _state_prior_rows(self, x: jax.Array) -> jax.Array:
        """State-prior rows at the decision vector ``x``."""
        return self._state_prior_rows_from(self.layout.state_blocks(x), self._anchor_stack)

    # --- manifold init ------------------------------------------------------

    def _forward_sweep_states(self, theta_flat: jax.Array) -> dict:
        """Forward sweep at θ: differential window-start states (host numpy)."""
        theta_p = self.seam.theta_of(theta_flat)
        bnd = self.layout.boundaries
        idx = np.where(self.layout.diff_mask)[0]
        states = {0: np.asarray(self.y0)[idx]}
        y = self.y0
        for w in range(1, self.layout.n_windows):
            sol = self._propagate(
                theta_p,
                y,
                float(bnd[w - 1]),
                float(bnd[w]),
                w - 1,
            )
            y = sol.ys[-1]
            if not bool(sol.ok) or not bool(np.all(np.isfinite(np.asarray(y)))):
                policy = (
                    "the stock estimate" if self.restart_step is None else repr(self.restart_step)
                )
                raise ValueError(
                    "The window-initialization sweep failed in window "
                    f"{w - 1} ({float(bnd[w - 1])!r} to {float(bnd[w])!r}): the forward solve "
                    + ("produced non-finite states" if bool(sol.ok) else "did not complete")
                    + f" with the first-step policy {policy}. A fit needs every window's "
                    "start on the manifold; change the first-step policy "
                    "(Multishoot(first_step=...)), the window boundaries, or the solver "
                    "tolerances."
                )
            states[w] = np.asarray(y)[idx]
        return states

    def _manifold_init(self) -> np.ndarray:
        """Pack ``x0 = [θ₀ | sweep s̃ over the decision windows]`` (model-manifold init).

        The decision windows are ``first_state_window … W−1``; the forward sweep at
        ``θ₀`` (seeded from ``y0``, which is the learned-state guess when the initial
        state is learned) supplies each window-start differential block.
        """
        layout = self.layout
        theta0 = np.asarray(self.seam.x0)
        if layout.n_decision_windows == 0:
            return theta0
        if self._modes is not None:
            # The anchors already carry the sweep, so zero coefficients reproduce it.
            # The coefficient block takes θ₀'s dtype: an untyped zero block would be
            # float64 and would silently widen a float32 decision vector.
            return np.concatenate(
                [
                    theta0,
                    np.zeros(layout.n_decision_windows * layout.n_state, dtype=theta0.dtype),
                ]
            )
        sweep = self._sweep_states
        blocks = [theta0]
        for w in range(layout.first_state_window, layout.n_windows):
            blocks.append(np.asarray(sweep[w]))
        return np.concatenate(blocks)

    @property
    def x0(self) -> np.ndarray:
        """The manifold-init decision vector."""
        return self._x0

    def initialization(self):
        """Return the exact anchors/coordinates needed to rebuild without solves."""
        from diff_soup.inference.fitting.windows import WindowInitialization

        coordinates = np.asarray(self._x0)[self.layout.n_theta :].reshape(
            (self.layout.n_decision_windows, self.layout.n_state)
        )
        return WindowInitialization(
            anchors=np.asarray(self._anchor_stack),
            initial_coordinates=coordinates,
            reached_tend=bool(
                True if self._baseline_reached_tend is None else self._baseline_reached_tend
            ),
            provenance=self._initialization_provenance,
        )

    # --- diagnostics (read by diagnostics.compute_diagnostics) --------------

    def window_states_of(self, x_flat: np.ndarray) -> np.ndarray:
        """Final differential decision states stacked ``(n_blocks, n_diff)`` (warm-start).

        The blocks are the decision windows ``first_state_window … W−1`` — the learned
        ``s̃₀`` is included when the initial state is learned.
        """
        x = jnp.asarray(x_flat)
        if self.layout.n_decision_windows == 0:
            return np.zeros((0, self.layout.n_diff))
        return np.asarray(self._lift_blocks(self.layout.state_blocks(x)))

    def window_diagnostics(self, x_flat: np.ndarray):
        """Per-window obs-block rank + continuity-defect WRMS + INFO notes.

        The window-state observation block ``∂(obs rows)/∂s̃_w`` is rank-deficient for
        distribution observables (permutation invariance + marginal-only observation).
        This makes that under-determination visible as an INFO note; it is expected
        for population observables and is never raised as a warning-as-error.
        """
        x = jnp.asarray(x_flat)
        notes: list[str] = []
        ranks: list[int] = []
        idx = np.where(self.layout.diff_mask)[0]
        fsw = self.layout.first_state_window

        for w in range(fsw, self.layout.n_windows):
            if len(self._win_frames[w]) == 0:
                # A frame_mask can leave a window with no observed frame; its
                # observation block is empty (rank 0), so skip the jacfwd probe.
                ranks.append(0)
                continue

            n_state = self.layout.n_state

            def obs_w(sw, _w=w, _n=n_state):
                start = self.layout.n_theta + (_w - fsw) * _n
                xw = x.at[start : start + _n].set(sw)
                rows = [
                    self.series.observable.predict(yp, None, None)
                    for _fi, yp in self.states_at_frames(xw, _w)
                ]
                return jnp.concatenate([jnp.reshape(r, (-1,)) for r in rows])

            s_w = self.layout.state_block(x, w)
            jac = np.asarray(jax.jacfwd(obs_w)(s_w))
            if jac.size == 0:
                ranks.append(0)
                continue
            sv = np.linalg.svd(jac, compute_uv=False)
            tol = max(jac.shape) * (sv[0] if sv.size else 0.0) * np.finfo(float).eps
            rank = int(np.sum(sv > tol))
            ranks.append(rank)
            if rank < n_state:
                notes.append(
                    f"optsolve: window {w} observation block is rank-deficient "
                    f"({rank}/{n_state}) — the window state is "
                    "under-determined by the observable alone (expected for "
                    "distribution / marginal observables). The continuity rows + "
                    "manifold init make it observable through the dynamics."
                )

        # Continuity-defect at the solution.  The design's WRMS in literal
        # 1/(atol+rtol|s̃|) units is not directly gate-assertable.  The
        # meaningful, gate-assertable quantity is the raw physical defect RMS
        # ||s̃ − Φ_diff||, so this field reports the raw RMS defect.
        theta_p = self._theta_pytree(x)
        bnd = self.layout.boundaries
        lifted = np.asarray(self._lift_blocks(self.layout.state_blocks(x)))
        defects = []
        for w in range(1, self.layout.n_windows):
            s_diff = lifted[w - fsw]
            y_prev = self._window_start_state(x, w - 1)
            sol = self._propagate(theta_p, y_prev, float(bnd[w - 1]), float(bnd[w]))
            pred_diff = np.asarray(sol.ys[-1])[idx]
            d = s_diff - pred_diff
            defects.append(float(np.sqrt(np.mean(d**2))))

        # The share of the weighted continuity mismatch the state basis cannot
        # represent.  The full coordinate represents everything, so it is identically
        # zero; a reduced basis reports how much of the mismatch its projection
        # dropped, which is the number that says whether the subspace is wide enough.
        #
        # Computed as ``||v - B (Bᵀ v)||`` rather than the algebraically equal
        # ``sqrt(||v||² - ||Bᵀ v||²)``: when the basis is nearly complete those two
        # squared norms almost cancel, and the subtraction would leave a spurious
        # floor around the square root of machine epsilon.  This diagnostic exists to
        # be compared against a measurement noise level, so it must not manufacture
        # a floor of its own.
        weighted = np.asarray(self.weighted_continuity_mismatch(x))
        if self._modes is None:
            unrepresented = np.zeros((weighted.shape[0],))
        else:
            modes = np.asarray(self.states.modes)
            residual = weighted - (weighted @ modes) @ modes.T
            unrepresented = np.linalg.norm(residual, axis=1)
            scale = np.linalg.norm(weighted, axis=1)
            for position, (dropped, magnitude) in enumerate(zip(unrepresented, scale)):
                if magnitude > 0.0 and dropped > 0.05 * magnitude:
                    notes.append(
                        f"optsolve: window {int(self._plan.cont_windows[position])} continuity "
                        f"mismatch is {100.0 * dropped / magnitude:.1f}% outside the "
                        "declared state subspace, so the projected continuity rows cannot "
                        "correct it. Widen the StateBasis if this share matters at the "
                        "measurement noise level."
                    )

        for n in notes:
            logger.info(n)
        return (
            tuple(ranks) if ranks else None,
            np.asarray(defects) if defects else None,
            np.asarray(unrepresented) if defects else None,
            tuple(notes),
        )
