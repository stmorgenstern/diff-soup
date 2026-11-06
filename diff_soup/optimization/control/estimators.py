"""State estimators for the receding horizon: one hook, three implementations."""

from __future__ import annotations

import abc
import dataclasses
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
    duration = controls.interval
    interval = dataclasses.replace(controls, intervals=1, duration=duration)
    solution, _ = solve_schedule(
        model,
        control_problem.solver,
        interval.segments(jnp.atleast_1d(u), tuple(model.operational_modes())),
        state,
        t0=0.0,
        t_end=duration,
        times=jnp.asarray([duration]),
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


class MovingHorizonEstimate(eqx.Module):
    """The state of `MovingHorizonEstimator`: the estimate, the window's prior, and the window.

    Attributes:
        plant: The state estimate at the window's end, which the next horizon starts from.
        start: The prior of the state at the window's start, the arrival cost's centre.
        correction: The last fit's correction of `start`, along the estimator's basis.
        controls: The applied controls of the window, oldest first, `(window,)`.
        measurements: The measurement after each of them, `(window, n_measured)`.
        count: How many entries of the window are filled, at most `window`.
    """

    plant: Any
    start: Any
    correction: Any
    controls: Any
    measurements: Any
    count: Any


class MovingHorizonEstimator(Estimator):
    """Moving-horizon estimation: a least-squares fit of the window's start state to its measurements.

    After each interval the window takes the applied control and the measurement, and the
    start state `start + basis @ z` is fitted: the residual is every measurement in the window
    against the model run from that state through the window's controls, whitened by the
    measurement covariance, then the arrival cost `z` whitened by its covariance. The fit is a
    `LeastSquaresProblem` whose `args` hold the window, solved by `solver` from the last
    correction, so it runs inside the loop's `lax.scan`. Once the window is full it slides:
    the next start's prior is the fitted start advanced one interval.

    Args:
        problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem] whose model, solver and interval the plant follows.
        measured: `measured(state, theta) -> measurement` as a JAX function.
        window: The number of intervals in the window.
        solver: The `CompiledSolver` of the window fit, as the horizon's.
        arrival_covariance: The covariance of `z`, `(k, k)`, a diagonal or a scalar.
        measurement_covariance: The covariance of one measurement, `(n_measured, n_measured)`, a diagonal or a scalar.
        basis: The directions `(n_states, k)` the start state is corrected along; every state when `None`.
    """

    # The control problem holds the model's arrays, so it is a leaf, not static.
    problem: Any = None
    measured: Any = eqx.field(static=True, default=None)
    window: int = eqx.field(static=True, default=5)
    solver: Any = eqx.field(static=True, default=None)
    arrival_covariance: Any = None
    measurement_covariance: Any = None
    basis: Any = None

    def init(self, args: Any) -> Any:
        """An empty window whose start prior is the control problem's initial state."""
        plant = jnp.asarray(args[0])
        shape = jax.eval_shape(lambda y: jnp.atleast_1d(self.measured(y, args[1])), plant)
        return MovingHorizonEstimate(
            plant=plant,
            start=plant,
            correction=jnp.zeros((self._basis(plant).shape[1],), plant.dtype),
            controls=jnp.zeros((self.window,), plant.dtype),
            measurements=jnp.zeros((self.window, shape.shape[0]), plant.dtype),
            count=jnp.zeros((), jnp.int32),
        )

    def estimate(self, state: Any, u_prev: Any, measurement: Any, args: Any) -> Any:
        """Push the interval into the window, refit the start state, and run it to the window's end."""
        from diff_soup.optimization._abc import LeastSquaresProblem

        theta = args[1]
        controls = jnp.concatenate([state.controls[1:], jnp.reshape(u_prev, (1,))])
        measured = jnp.atleast_1d(measurement)[None, :].astype(state.measurements.dtype)
        measurements = jnp.concatenate([state.measurements[1:], measured])
        count = jnp.minimum(state.count + 1, self.window)
        basis = self._basis(state.plant)
        k = basis.shape[1]
        noise = jnp.linalg.cholesky(_matrix(self.measurement_covariance, measured.shape[1]))
        arrival = jnp.linalg.cholesky(_matrix(self.arrival_covariance, k))
        data = (state.start, controls, measurements, count, theta)

        def residual(z, data):
            return self._residual(z, data, basis, noise, arrival)[0]

        def residual_and_jacobian(z, data):
            jacobian, value = jax.jacfwd(lambda x: (residual(x, data),) * 2, has_aux=True)(z)
            return value, jacobian

        fit = LeastSquaresProblem(
            x0=jnp.zeros((k,), state.plant.dtype),
            structure=frozenset({"value_gradient", "residual_jacobian", "gauss_newton"}),
            args=data,
            ad="forward",
            _residual=residual,
            _residual_and_jacobian=residual_and_jacobian,
        )
        solver = self.solver
        if solver is None:
            from diff_soup.optimization.solvers.sqp import SQP

            solver = SQP()
        z, _ = solver.run(fit, state.correction, data)
        _, (first, end) = self._residual(z, data, basis, noise, arrival)
        # A full window slides at the next interval: its start's prior is the fitted start
        # advanced one interval, about which the next correction starts from zero.
        full = count == self.window
        return MovingHorizonEstimate(
            plant=end,
            start=jnp.where(full, first, state.start),
            correction=jnp.where(full, jnp.zeros_like(z), z),
            controls=controls,
            measurements=measurements,
            count=count,
        )

    def _basis(self, plant: Any) -> Any:
        """The correction directions, the identity when none are given."""
        if self.basis is None:
            return jnp.eye(plant.shape[0], dtype=plant.dtype)
        return jnp.asarray(self.basis, plant.dtype)

    def _residual(self, z, data, basis, noise, arrival):
        """The whitened window residual, the state after the window's first filled interval, and its end."""
        start, controls, measurements, count, theta = data
        offset = self.window - count

        def interval(carry, k):
            x, first = carry
            on = k >= offset
            x = jnp.where(on, one_interval(self.problem, x, controls[k], theta), x)
            first = jnp.where(k == offset, x, first)
            error = jnp.atleast_1d(self.measured(x, theta)) - measurements[k]
            row = jax.scipy.linalg.solve_triangular(noise, error, lower=True)
            return (x, first), jnp.where(on, row, 0.0)

        x0 = start + basis @ z
        (end, first), rows = jax.lax.scan(interval, (x0, x0), jnp.arange(self.window))
        prior = jax.scipy.linalg.solve_triangular(arrival, z, lower=True)
        return jnp.concatenate([rows.ravel(), prior]), (first, end)
