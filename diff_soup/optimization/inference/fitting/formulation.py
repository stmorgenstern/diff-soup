"""How an inference problem is formulated: its objective, its trajectory and its initial state."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.fitting.windows import (
    Continuity,
    RestartStep,
    StateBasis,
    Windows,
)

# ---------------------------------------------------------------------------
# Objective objects
# ---------------------------------------------------------------------------


class Objective(eqx.Module):
    """Base class of the objective an [InferenceProblem][diff_soup.optimization.inference.fitting.inference_problem.InferenceProblem] takes through ``objective=``.

    An objective says how the misfit of the observations becomes the rows of
    the problem's residual.
    """


class LeastSquares(Objective):
    """The least-squares objective, half the squared norm of the residual."""


# ---------------------------------------------------------------------------
# Trajectory strategy objects
# ---------------------------------------------------------------------------


class Trajectory(eqx.Module):
    """Base class for trajectory strategies (how the latent trajectory is solved)."""


class SingleShoot(Trajectory):
    """Single shooting, one solve over the experiment's whole time span, the default trajectory."""


class Multishoot(Trajectory):
    """Multiple shooting, one solve per window of the time span, each from its own start state.

    Pass it as `trajectory=` of an `Experiment`, or of an `InferenceProblem` to apply it to
    every experiment.

    Args:
        windows: The [Windows][diff_soup.optimization.inference.Windows] that split the time span.
        continuity: The weight of the continuity term between fitted windows, `Continuity()` when `None`.
        states: The [StateBasis][diff_soup.optimization.inference.StateBasis] of the fitted window starts, or `None` to fit every differential entry.
        warm_start_steps: The number of optimizer steps on the parameters alone, with the window starts held, before the full fit.
        first_step: A `FirstStep` policy for the first time step of each window's solve, or `None` to leave it to the solver.
    """

    windows: Windows
    continuity: Continuity = eqx.field(default_factory=Continuity)
    states: StateBasis = eqx.field(default_factory=StateBasis.full)
    warm_start_steps: int = eqx.field(static=True, default=0)
    first_step: Optional[RestartStep] = None

    def __init__(
        self,
        windows: Windows,
        continuity: Optional[Continuity] = None,
        *,
        states: Optional[StateBasis] = None,
        warm_start_steps: int = 0,
        first_step: Optional[RestartStep] = None,
    ):
        """Store the window spec, continuity config, window-state coordinate, and step policies."""
        self.windows = windows
        self.continuity = continuity if continuity is not None else Continuity()
        if states is not None and not isinstance(states, StateBasis):
            raise TypeError(
                f"Multishoot 'states' must be a StateBasis; got {type(states).__name__}."
            )
        self.states = states if states is not None else StateBasis.full()
        if isinstance(warm_start_steps, bool) or int(warm_start_steps) < 0:
            raise ValueError(
                "Multishoot 'warm_start_steps' must be a nonnegative integer; "
                f"got {warm_start_steps!r}."
            )
        self.warm_start_steps = int(warm_start_steps)
        if first_step is not None and not isinstance(first_step, RestartStep):
            raise TypeError(
                "Multishoot 'first_step' must be a FirstStep policy (FirstStep.auto(), "
                f".fixed(dt), .fraction(f), .hinted(...)) or None; got {type(first_step).__name__}."
            )
        self.first_step = first_step


class InitialState(eqx.Module):
    """Base class for the trajectory initial-state policy."""


class FixedInitialState(InitialState):
    """A known initial state, the form a bare array passed as `initial_state=` takes.

    Attributes:
        y0: The state at the start of the experiment's time span.
        time: The start time, which the experiment sets to the start of its span.
    """

    y0: ArrayLike | Sequence | None = None
    time: float = 0.0


class LearnedInitialState(InitialState):
    """An initial state fitted together with the parameters, starting from `guess`.

    Pass it as `initial_state=` of an `Experiment`. The fitted state follows the parameters
    in `result.params`, and `Windows.from_data` refuses it.

    Attributes:
        guess: The starting value of the full state, a 1-D array.
        time: The start time, which the experiment sets to the start of its span.
    """

    guess: ArrayLike | Sequence
    time: float = 0.0


class ComputedInitialState(InitialState):
    """An initial state computed from the experiment's parameters at every evaluation.

    `fn(theta)` receives the model's parameter values, read as `theta[handle]` with the
    objects `spec.param` returned, and returns the state at the start of the span, laid
    out as a fixed initial state is (for a recipe, as `spec.assemble_state` lays it out).
    A start that reads a fitted parameter moves with it, and every derivative of the fit
    carries that motion. The algebraic unknowns of the returned state are guesses, made
    consistent by the solve as they are for a fixed state.

    `fn` is called inside the compiled solve: it is traced, differentiated in forward
    and reverse mode and batched with `jax.vmap`, so it uses `jax.numpy`, reads no value
    on the host, and returns an array of one shape whatever the parameters. It is also
    called once when the experiment is lowered, at its starting parameters, to fix that
    shape.

    Pass it as `initial_state=` of an
    [Experiment][diff_soup.optimization.inference.Experiment] solved in one shoot.
    Under `Study(reads="declared")` the parameters `fn` reads are read by that
    experiment: `theta` holds only the parameters its model declares. A study records
    `fn` in its fingerprint by its code and the values it closes over, so those values
    are arrays, parameter handles, plain containers or functions of them, not the
    recipe's `spec` itself.

    ```python
    s = spec.param("s", value=1.0, learnable=True)
    rest = spec.assemble_state({"u": 0.0})
    unit = spec.assemble_state({"u": 1.0}) - rest
    start = inf.ComputedInitialState(lambda theta: rest + unit * theta[s])
    experiment = inf.Experiment(model, solver, (0.0, 2.0), start, measurements)
    ```

    Attributes:
        fn: The map from the model's parameter values to the initial state.
        time: The start time, which the experiment sets to the start of its span.
    """

    fn: Callable = eqx.field(static=True)
    time: float = 0.0

    def __check_init__(self):
        """Refuse a start that is not a function."""
        if not callable(self.fn):
            raise TypeError(
                "ComputedInitialState takes a function of the parameter values, "
                f"fn(theta) -> state; got {type(self.fn).__name__}."
            )


def _computed_start(seam: Any, initial_state: ComputedInitialState, x: Any) -> Any:
    """The state ``initial_state`` computes at the flat fitted-parameter vector ``x`` of ``seam``."""
    return jnp.asarray(initial_state.fn(seam.theta_of(x)))


def _computed_start_template(seam: Any, initial_state: ComputedInitialState) -> np.ndarray:
    """The state ``initial_state`` computes at the seam's starting parameters, on the host.

    It fixes the shape of the state, and stands in for the start wherever a number is
    needed outside a solve: a size, a digest, the filler of a sample slot the solve did
    not reach.
    """
    return np.asarray(_computed_start(seam, initial_state, jnp.asarray(seam.x0)))


def _initial_state_array(
    initial_state: InitialState | ArrayLike | Sequence,
) -> ArrayLike | Sequence | None:
    """Return the state array a declaration carries.

    A policy carries it as ``y0`` (fixed) or ``guess`` (learned); a bare array
    is the state itself. The array is what a prediction or a provenance digest
    reads, whichever policy holds it. A computed start carries none: ``None``.
    """
    if isinstance(initial_state, ComputedInitialState):
        return None
    if isinstance(initial_state, LearnedInitialState):
        return initial_state.guess
    if isinstance(initial_state, FixedInitialState):
        return initial_state.y0
    return initial_state


def _initial_state_policy(
    initial_state: InitialState | ArrayLike | Sequence, *, time: float
) -> InitialState:
    """Return the trajectory initial-state policy of a declaration, timed at ``time``.

    A bare array becomes ``FixedInitialState``; a policy keeps its kind and is
    re-timed, because the state a declaration carries is the state at the start
    of that declaration's own span.
    """
    if isinstance(initial_state, InitialState):
        return dataclasses.replace(initial_state, time=float(time))
    return FixedInitialState(initial_state, time=float(time))


def _check_initial_state_time(
    initial_state: InitialState | ArrayLike | Sequence, data: tuple
) -> None:
    """Reject a FixedInitialState time that contradicts the frame it labels.

    A [FixedInitialState][diff_soup.optimization.inference.fitting.formulation.FixedInitialState]
    with no ``y0`` takes its state from the first stream's first observed frame, so
    the only time that state can carry is that frame's own time.

    Raises:
        ValueError: If ``y0`` is ``None`` and ``time`` names a different instant.
    """
    if not isinstance(initial_state, FixedInitialState) or initial_state.y0 is not None:
        return
    first = float(np.asarray(data[0].ts).reshape((-1,))[0])
    stated = float(initial_state.time)
    if abs(stated - first) > 1e-12 * max(1.0, abs(first)):
        raise ValueError(
            f"FixedInitialState(time={stated}) with no y0 takes its state from the "
            f"first frame of Series {data[0].name!r}, which is observed at t={first}. "
            f"State time={first} to match the frame, pass FixedInitialState(y0, "
            f"time={stated}) to state a different instant with an explicit y0, or omit "
            "initial_state to take both the state and the time from the first frame."
        )


@dataclasses.dataclass(frozen=True, init=False)
class _FrozenStateMultishoot:
    """Theta-only reduced view of a decision-variable InferenceProblem.

    The window states are frozen at the manifold-init values; the decision vector
    is only θ̃.  Drives the LM engine through the same residual stack with the
    state blocks pinned.  Submodule-internal: the freeze-then-relax warm-start.
    """

    _build: Any
    regularization: Any
    _x0: np.ndarray

    def __init__(self, build: Any, regularization: Any):
        """Pin the window states at init; expose θ̃ as the only decision variable."""
        _set = object.__setattr__
        _set(self, "_build", build)
        _set(self, "regularization", regularization)
        _set(self, "_x0", np.asarray(build.x0)[: build.layout.n_theta])

    def _full_x(self, theta_flat: jnp.ndarray) -> jnp.ndarray:
        """Reassemble the full x with frozen (init) window states."""
        tail = jnp.asarray(self._build.x0)[self._build.layout.n_theta :]
        return jnp.concatenate([jnp.asarray(theta_flat), tail])

    @property
    def x0(self) -> np.ndarray:
        """Theta-only init vector."""
        return self._x0

    @property
    def n_segments(self) -> int:
        """Single residual block."""
        return 1

    @property
    def segment_fns(self) -> tuple:
        """The full residual stack, but over the theta-only reduced vector."""
        return (lambda th: self._build.residual(self._full_x(th)),)

    @property
    def unflatten_fn(self):
        """Inverse ravel of θ̃ (for reconstruct; the LM reads this)."""
        return self._build.seam.unflatten

    @property
    def prob_static(self):
        """Frozen theta partition."""
        return self._build.seam.theta_frozen
