"""State estimators for the receding horizon: one hook, three implementations."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, ArrayLike, Float, Int, PyTree

from diff_soup._errors import ConfigurationError
from diff_soup._schedule import solve_schedule
from diff_soup.optimization._abc import LeastSquaresProblem
from diff_soup.optimization._binding import Derived, Held
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.solvers.sqp import SQP

if TYPE_CHECKING:
    from diff_soup.optimization.control.problem import ControlProblem, PiecewiseConstant
    from diff_soup.optimization.solvers import CompiledSolver
    from diff_soup.timestepping.vsvo.classes import JBDF, JIDA, JNDF

#: The `args` of a control problem: the initial state, `theta` and the limits.
ControlArgs = tuple[PyTree, PyTree, PyTree]


class PlantEstimate(eqx.Module):
    """The state of `FullState`: the plant state the next horizon starts from.

    Attributes:
        plant: The plant state estimate.
    """

    plant: Float[ArrayLike, " n_states"]


class Estimator(eqx.Module):
    """The estimator's slot in the loop: from its state, the last control and a measurement to its next state."""

    @abc.abstractmethod
    def init(self, args: ControlArgs) -> eqx.Module:
        """The estimator's state before the first interval, from the control problem's `args`."""
        raise NotImplementedError

    @abc.abstractmethod
    def estimate(
        self, state: eqx.Module, u_prev: ArrayLike, measurement: ArrayLike, args: ControlArgs
    ) -> eqx.Module:
        """The next estimator state; `state.plant` is the plant state estimate the horizon starts from."""
        raise NotImplementedError


class FullState(Estimator):
    """Full-state feedback: the measurement is the plant state, and the estimate is the measurement."""

    def init(self, args: ControlArgs) -> PlantEstimate:
        """The initial state of the control problem."""
        return PlantEstimate(plant=args[0])

    def estimate(
        self, state: PlantEstimate, u_prev: ArrayLike, measurement: ArrayLike, args: ControlArgs
    ) -> PlantEstimate:
        """Return the measured plant state."""
        return PlantEstimate(plant=measurement)


class KalmanEstimate(eqx.Module):
    """The state of `ExtendedKalmanFilter`: the plant state estimate and its covariance.

    Attributes:
        plant: The plant state estimate.
        covariance: Its covariance, `(n_states, n_states)`.
    """

    plant: Float[Array, " n_states"]
    covariance: Float[Array, "n_states n_states"]


def _require(owner: str, module: eqx.Module, names: tuple[str, ...]) -> None:
    """Refuse `module` when any of the fields `names` is `None`."""
    missing = [name for name in names if getattr(module, name) is None]
    if missing:
        raise ConfigurationError(f"{owner} needs {', '.join(missing)}.")


class ExtendedKalmanFilter(Estimator):
    """An EKF whose one-interval transition and measurement maps are linearized by forward-mode AD through the schedule solve.

    A failed interval solve makes the estimate NaN, so the next horizon is a rejected trial.

    Args:
        problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem] whose model, solver and interval the plant follows.
        measured: `measured(state, theta) -> measurement` as a JAX function.
        process_covariance: `Q`, `(n_states, n_states)` or a diagonal.
        measurement_covariance: `R`, `(n_measured, n_measured)` or a diagonal.
        initial_covariance: The covariance of the initial state, `Q` when `None`.
    """

    problem: ControlProblem | None = None
    measured: Callable | None = eqx.field(static=True, default=None)
    process_covariance: ArrayLike | None = None
    measurement_covariance: ArrayLike | None = None
    initial_covariance: ArrayLike | None = None

    def __check_init__(self) -> None:
        """Refuse a filter without its problem, measurement map or covariances."""
        _require(
            "ExtendedKalmanFilter",
            self,
            ("problem", "measured", "process_covariance", "measurement_covariance"),
        )

    def init(self, args: ControlArgs) -> KalmanEstimate:
        """The initial estimate and its covariance."""
        plant = jnp.asarray(args[0])
        covariance = (
            self.process_covariance if self.initial_covariance is None else self.initial_covariance
        )
        return KalmanEstimate(plant=plant, covariance=_matrix(covariance, plant.shape[0]))

    def estimate(
        self, state: KalmanEstimate, u_prev: ArrayLike, measurement: ArrayLike, args: ControlArgs
    ) -> KalmanEstimate:
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


def one_interval(
    control_problem: ControlProblem,
    state: Float[ArrayLike, " n_states"],
    u: ArrayLike,
    theta: PyTree,
) -> Float[Array, " n_states"]:
    """The state after one interval of `control_problem`'s controls held at `u`, from `state`.

    Args:
        control_problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem] whose model, solver and interval apply.
        state: The state at the interval's start: the field unknowns, then the bordered globals.
        u: The interval's target.
        theta: The model parameters.

    Returns:
        The state at the interval's end, in the same layout; NaN when the interval's solve fails.
    """
    return _advance(
        control_problem.problem,
        control_problem.solver,
        control_problem.controls.interval_controls(),
        state,
        u,
        theta,
    )


def _advance(
    model: Held,
    solver: JIDA | JBDF | JNDF,
    controls: PiecewiseConstant,
    state: Float[ArrayLike, " n_states"],
    u: ArrayLike,
    theta: PyTree,
) -> Float[Array, " n_states"]:
    """One interval of `controls` held at `u` from `state`; NaN when the solve fails."""
    duration = controls.horizon
    solution, _ = solve_schedule(
        model.value,
        solver,
        controls.segments(jnp.atleast_1d(u), tuple(model.value.operational_modes())),
        state,
        t0=0.0,
        t_end=duration,
        times=jnp.asarray([duration]),
        theta=theta,
    )
    terminal = solution.terminal
    end = terminal.y if terminal.lam is None else jnp.concatenate([terminal.y, terminal.lam])
    return jnp.where(solution.ok, end, jnp.nan)


def _matrix(covariance: ArrayLike, n: int) -> Float[Array, "n n"]:
    """`covariance` as an `(n, n)` matrix: a diagonal vector or a scalar is spread on the diagonal."""
    covariance = jnp.asarray(covariance)
    if covariance.ndim == 2:
        return covariance
    return jnp.diag(jnp.broadcast_to(covariance, (n,)))


class MovingHorizonEstimate(eqx.Module):
    """The state of `MovingHorizonEstimator`: the estimate, the window's prior, and the window.

    Attributes:
        plant: The state estimate at the window's end, which the next horizon starts from; NaN after a failed fit.
        start: The prior of the state at the window's start, the arrival cost's centre.
        correction: The last fit's correction of `start`, along the estimator's basis.
        controls: The applied controls of the window, oldest first, `(window,)`.
        measurements: The measurement after each of them, `(window, n_measured)`.
        count: How many entries of the window are filled, at most `window`.
        status: The last window fit's [TerminationStatus][diff_soup.optimization.TerminationStatus] code.
    """

    plant: Float[Array, " n_states"]
    start: Float[Array, " n_states"]
    correction: Float[Array, " n_basis"]
    controls: Float[Array, " window"]
    measurements: Float[Array, "window n_measured"]
    count: Int[Array, ""]
    status: Int[Array, ""]


class _Window(NamedTuple):
    """The `args` of a window fit: the prior, the window's data and the whitening factors."""

    start: Float[Array, " n_states"]
    controls: Float[Array, " window"]
    measurements: Float[Array, "window n_measured"]
    count: Int[Array, ""]
    theta: PyTree
    basis: Float[Array, "n_states n_basis"]
    noise: Float[Array, "n_measured n_measured"]
    arrival: Float[Array, "n_basis n_basis"]


class _Configuration(NamedTuple):
    """What a window fit's residual reads besides `(z, args)`; hashable, free of arrays."""

    model: Held
    solver: JIDA | JBDF | JNDF
    controls: PiecewiseConstant
    measured: Callable
    window: int


class MovingHorizonEstimator(Estimator):
    """Moving-horizon estimation: a least-squares fit of the window's start state to its measurements.

    After each interval the window takes the applied control and the measurement, and the
    start state `start + basis @ z` is fitted: the residual is every measurement in the window
    against the model run from that state through the window's controls, whitened by the
    measurement covariance, then the arrival cost `z` whitened by its covariance. The fit is
    one `LeastSquaresProblem` whose `args` hold the window, solved by `solver` from the last
    correction inside the loop's `lax.scan`. Once the window is full it slides: the next
    start's prior is the fitted start advanced one interval. A fit that fails (a status from
    `INFEASIBLE` on) makes the estimate NaN; its status is carried in the state.

    Args:
        problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem] whose model, solver and interval the plant follows.
        measured: `measured(state, theta) -> measurement` as a JAX function.
        window: The number of intervals in the window.
        solver: The `CompiledSolver` of the window fit; `SQP()` when `None`.
        arrival_covariance: The covariance of `z`, `(k, k)`, a diagonal or a scalar.
        measurement_covariance: The covariance of one measurement, `(n_measured, n_measured)`, a diagonal or a scalar.
        basis: The directions `(n_states, k)` the start state is corrected along; every state when `None`.
    """

    problem: ControlProblem | None = None
    measured: Callable | None = eqx.field(static=True, default=None)
    window: int | None = eqx.field(static=True, default=None)
    solver: CompiledSolver | None = eqx.field(static=True, default=None)
    arrival_covariance: ArrayLike | None = None
    measurement_covariance: ArrayLike | None = None
    basis: Float[ArrayLike, "n_states n_basis"] | None = None

    def __check_init__(self) -> None:
        """Refuse an estimator without its problem, measurement map, window or covariances."""
        _require(
            "MovingHorizonEstimator",
            self,
            ("problem", "measured", "window", "arrival_covariance", "measurement_covariance"),
        )

    def init(self, args: ControlArgs) -> MovingHorizonEstimate:
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
            status=jnp.zeros((), jnp.int32),
        )

    def estimate(
        self,
        state: MovingHorizonEstimate,
        u_prev: ArrayLike,
        measurement: ArrayLike,
        args: ControlArgs,
    ) -> MovingHorizonEstimate:
        """Push the interval into the window, refit the start state, and run it to the window's end."""
        controls = jnp.concatenate([state.controls[1:], jnp.reshape(u_prev, (1,))])
        measured = jnp.atleast_1d(measurement)[None, :].astype(state.measurements.dtype)
        measurements = jnp.concatenate([state.measurements[1:], measured])
        count = jnp.minimum(state.count + 1, self.window)
        basis = self._basis(state.plant)
        data = _Window(
            start=state.start,
            controls=controls,
            measurements=measurements,
            count=count,
            theta=args[1],
            basis=basis,
            noise=jnp.linalg.cholesky(_matrix(self.measurement_covariance, measured.shape[1])),
            arrival=jnp.linalg.cholesky(_matrix(self.arrival_covariance, basis.shape[1])),
        )
        configuration = _Configuration(
            model=self.problem.problem,
            solver=self.problem.solver,
            controls=self.problem.controls.interval_controls(),
            measured=self.measured,
            window=self.window,
        )
        fit = LeastSquaresProblem.from_residual(
            jnp.zeros_like(state.correction),
            Derived(_window_residual, configuration),
            args=data,
            hessian=False,
        )
        solver = SQP() if self.solver is None else self.solver
        z, record = solver.run(fit, state.correction, data)
        _, (first, end) = _window(configuration, z, data)
        failed = record.status >= int(TerminationStatus.INFEASIBLE)
        # A full window slides at the next interval: its start's prior is the fitted start
        # advanced one interval, about which the next correction starts from zero.
        full = count == self.window
        return MovingHorizonEstimate(
            plant=jnp.where(failed, jnp.nan, end),
            start=jnp.where(full, first, state.start),
            correction=jnp.where(full, jnp.zeros_like(z), z),
            controls=controls,
            measurements=measurements,
            count=count,
            status=jnp.asarray(record.status, jnp.int32),
        )

    def _basis(self, plant: Float[Array, " n_states"]) -> Float[Array, "n_states n_basis"]:
        """The correction directions, the identity when none are given."""
        if self.basis is None:
            return jnp.eye(plant.shape[0], dtype=plant.dtype)
        return jnp.asarray(self.basis, plant.dtype)


def _window_residual(
    configuration: _Configuration, z: Float[Array, " n_basis"], data: _Window
) -> Float[Array, " n_rows"]:
    """The whitened window residual of the correction `z`."""
    return _window(configuration, z, data)[0]


def _window(
    configuration: _Configuration, z: Float[Array, " n_basis"], data: _Window
) -> tuple[Float[Array, " n_rows"], tuple[Float[Array, " n_states"], Float[Array, " n_states"]]]:
    """The whitened window residual, the state after the window's first filled interval, and its end.

    The slots before the window fills are skipped behind a conditional, so the model is never
    run on them.
    """
    offset = configuration.window - data.count

    def advance(x, k):
        return _advance(
            configuration.model,
            configuration.solver,
            configuration.controls,
            x,
            data.controls[k],
            data.theta,
        )

    def interval(carry, k):
        x, first = carry
        on = k >= offset
        x = jax.lax.cond(on, advance, lambda x, k: x, x, k)
        first = jnp.where(k == offset, x, first)
        error = jnp.atleast_1d(configuration.measured(x, data.theta)) - data.measurements[k]
        row = jsl.solve_triangular(data.noise, error, lower=True)
        return (x, first), jnp.where(on, row, 0.0)

    x0 = data.start + data.basis @ z
    (end, first), rows = jax.lax.scan(interval, (x0, x0), jnp.arange(configuration.window))
    prior = jsl.solve_triangular(data.arrival, z, lower=True)
    return jnp.concatenate([rows.ravel(), prior]), (first, end)
