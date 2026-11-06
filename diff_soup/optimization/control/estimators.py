"""State estimators for the receding horizon: one hook, three implementations."""

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from diff_soup._schedule import solve_schedule


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


class KalmanEstimate(eqx.Module):
    """The state of `ExtendedKalmanFilter`: the plant state estimate and its covariance.

    Attributes:
        plant: The plant state estimate.
        covariance: Its covariance, `(n_states, n_states)`.
    """

    plant: Any
    covariance: Any


class ExtendedKalmanFilter(Estimator):
    """An EKF whose one-interval transition and measurement maps are linearized by forward-mode AD through the schedule solve.

    Args:
        problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem] whose model, solver and interval the plant follows.
        measured: `measured(state, theta) -> measurement` as a JAX function.
        process_covariance: `Q`, `(n_states, n_states)` or a diagonal.
        measurement_covariance: `R`, `(n_measured, n_measured)` or a diagonal.
        initial_covariance: The covariance of the initial state, `Q` when `None`.
    """

    # The control problem holds the model's arrays, so it is a leaf, not static.
    problem: Any = None
    measured: Any = eqx.field(static=True, default=None)
    process_covariance: Any = None
    measurement_covariance: Any = None
    initial_covariance: Any = None

    def init(self, args: Any) -> Any:
        """The initial estimate and its covariance."""
        plant = jnp.asarray(args[0])
        covariance = (
            self.process_covariance if self.initial_covariance is None else self.initial_covariance
        )
        return KalmanEstimate(plant=plant, covariance=_matrix(covariance, plant.shape[0]))

    def estimate(self, state: Any, u_prev: Any, measurement: Any, args: Any) -> Any:
        """Predict through one interval, then correct with the measurement."""
        theta = args[1]
        n = state.plant.shape[0]
        transition, predicted = jax.jacfwd(
            lambda x: (one_interval(self.problem, x, u_prev, theta),) * 2, has_aux=True
        )(state.plant)
        covariance = transition @ state.covariance @ transition.T
        covariance = covariance + _matrix(self.process_covariance, n)
        sensitivity, expected = jax.jacfwd(
            lambda x: (jnp.atleast_1d(self.measured(x, theta)),) * 2, has_aux=True
        )(predicted)
        noise = _matrix(self.measurement_covariance, expected.shape[0])
        innovation = sensitivity @ covariance @ sensitivity.T + noise
        factor = jsl.cho_factor(innovation, lower=True)
        gain = jsl.cho_solve(factor, sensitivity @ covariance).T
        plant = predicted + gain @ (jnp.atleast_1d(measurement) - expected)
        # The Joseph form keeps the covariance symmetric and positive semidefinite.
        keep = jnp.eye(n, dtype=covariance.dtype) - gain @ sensitivity
        covariance = keep @ covariance @ keep.T + gain @ noise @ gain.T
        return KalmanEstimate(plant=plant, covariance=0.5 * (covariance + covariance.T))


def one_interval(control_problem: Any, state: Any, u: Any, theta: Any) -> Any:
    """The state after one interval of `control_problem`'s controls held at `u`, from `state`.

    Args:
        control_problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem] whose model, solver and interval apply.
        state: The state at the interval's start: the field unknowns, then the bordered globals.
        u: The interval's target.
        theta: The model parameters.

    Returns:
        The state at the interval's end, in the same layout.
    """
    controls = control_problem.controls
    model = control_problem.problem
    interval = type(controls)(mode=controls.mode, intervals=1, duration=controls.duration)
    solution, _ = solve_schedule(
        model,
        control_problem.solver,
        interval.segments(jnp.atleast_1d(u), tuple(model.operational_modes())),
        state,
        t0=0.0,
        t_end=controls.duration,
        times=jnp.asarray([controls.duration]),
        theta=theta,
    )
    terminal = solution.terminal
    if terminal.lam is None:
        return terminal.y
    return jnp.concatenate([terminal.y, terminal.lam])


def _matrix(covariance: Any, n: int) -> Any:
    """`covariance` as an `(n, n)` matrix: a diagonal vector or a scalar is spread on the diagonal."""
    covariance = jnp.asarray(covariance)
    if covariance.ndim == 2:
        return covariance
    return jnp.diag(jnp.broadcast_to(covariance, (n,)))


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
