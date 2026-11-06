"""State estimators for the receding horizon: one hook, three implementations."""

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx


class PlantEstimate(eqx.Module):
    """The state of `FullState`: the plant state the next horizon starts from.

    Attributes:
        plant: The plant state estimate.
    """

    plant: Any


class Estimator(eqx.Module):
    """The estimator's slot in the loop: from its state, the last control and a measurement to its next state."""

    @abc.abstractmethod
    def init(self, args: Any) -> Any:
        """The estimator's state before the first interval, from the control problem's `args`."""
        raise NotImplementedError

    @abc.abstractmethod
    def estimate(self, state: Any, u_prev: Any, measurement: Any, args: Any) -> Any:
        """The next estimator state; `state.plant` is the plant state estimate the horizon starts from."""
        raise NotImplementedError


class FullState(Estimator):
    """Full-state feedback: the measurement is the plant state, and the estimate is the measurement."""

    def init(self, args: Any) -> Any:
        """The initial state of the control problem."""
        return PlantEstimate(plant=args[0])

    def estimate(self, state: Any, u_prev: Any, measurement: Any, args: Any) -> Any:
        """Return the measured plant state."""
        return PlantEstimate(plant=measurement)


class ExtendedKalmanFilter(Estimator):
    """An EKF whose one-interval transition and measurement maps are linearized by forward-mode AD through the schedule solve.

    Args:
        problem: The model the controller uses.
        measured: `measured(state, theta) -> measurement` as a JAX function.
        process_covariance: `Q`, `(n_states, n_states)` or a diagonal.
        measurement_covariance: `R`, `(n_measured, n_measured)` or a diagonal.
    """

    problem: Any = eqx.field(static=True)
    measured: Any = eqx.field(static=True)
    process_covariance: Any = None
    measurement_covariance: Any = None

    def init(self, args: Any) -> Any:
        """The initial estimate and its covariance."""
        raise NotImplementedError

    def estimate(self, state: Any, u_prev: Any, measurement: Any, args: Any) -> Any:
        """Predict through one interval, then correct with the measurement."""
        raise NotImplementedError


class MovingHorizonEstimator(Estimator):
    """Moving-horizon estimation: an `InferenceProblem` over the last `window` measurements, solved by the same solver.

    The state at the window start is the parameter, the problem's constraints apply, and each
    window warm-starts from the previous solution.

    Args:
        problem: The model the controller uses.
        measured: `measured(state, theta) -> measurement` as a JAX function.
        window: The number of intervals in the window.
        solver: The [CompiledSolver][diff_soup.optimization.CompiledSolver] of the window fit.
        arrival_covariance: The prior on the window's start state, or `None`.
    """

    problem: Any = eqx.field(static=True)
    measured: Any = eqx.field(static=True)
    window: int = eqx.field(static=True, default=5)
    solver: Any = eqx.field(static=True, default=None)
    arrival_covariance: Any = None

    def init(self, args: Any) -> Any:
        """An empty window around the initial estimate."""
        raise NotImplementedError

    def estimate(self, state: Any, u_prev: Any, measurement: Any, args: Any) -> Any:
        """Push the measurement into the window and refit the window's start state."""
        raise NotImplementedError
