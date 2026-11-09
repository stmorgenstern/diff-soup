"""A control problem: cost, path and terminal functionals of a model's solution over a horizon."""

from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, PyTree

from diff_soup._errors import ConfigurationError
from diff_soup._schedule import Schedule, ScheduleArrays, Segment, schedule_arrays
from diff_soup.optimization._binding import Held, hold
from diff_soup.optimization._derived import as_bounds
from diff_soup.optimization._numeric import OptimizationProblem
from diff_soup.optimization.constraints import Constraints
from diff_soup.optimization.control._horizon import Configuration, Horizon
from diff_soup.optimization.problem import AbstractOptimizationProblem
from diff_soup.problem._spec.operating_modes import operating_mode_names

if TYPE_CHECKING:
    from diff_soup._adjoints import AbstractAdjoint
    from diff_soup._layout import Layout
    from diff_soup.optimization._abc import OptSolver
    from diff_soup.timestepping.vsvo.classes import JBDF, JIDA, JNDF

#: How far above the horizon a save time may lie and still be its end: Higham's bound on
#: the rounding error of a sum of `n` positive terms added in order, `(n - 1) * u * sum`
#: with `u` the unit roundoff (Higham, Accuracy and Stability of Numerical Algorithms,
#: 2nd ed., eq. 4.4), taken as `n * eps * horizon`.
_END_ROUNDING = np.finfo(np.float64).eps


def _lengths(duration: ArrayLike | Sequence[float]) -> float | tuple:
    """A duration as Python floats, a scalar or a tuple, so it is static and hashable."""
    if np.ndim(duration) == 0:
        return float(duration)
    return tuple(float(d) for d in np.asarray(duration, dtype=np.float64).ravel())


class AbstractControls(eqx.Module):
    """The parameterization of a horizon's operational targets by a decision vector.

    A [ControlProblem][diff_soup.optimization.control.ControlProblem] reads `size` and lowers
    each decision vector through `segments` onto the schedule it solves, which runs to
    `horizon`, or to the end of its traced durations when `free_durations`.
    [receding_horizon][diff_soup.optimization.control.receding_horizon] and the estimators step
    the controls by `interval`, through `durations` and `interval_controls()`.
    """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """The length of the decision vector."""

    @property
    @abc.abstractmethod
    def free_durations(self) -> bool:
        """Whether the segment durations are entries of the decision vector, traced in `segments`."""

    @property
    @abc.abstractmethod
    def durations(self) -> tuple:
        """The length of each segment, static."""

    @property
    @abc.abstractmethod
    def horizon(self) -> float:
        """The end of the horizon: the durations added in order, as the schedule adds them."""

    @property
    @abc.abstractmethod
    def interval(self) -> float:
        """The one length of every interval, which the receding horizon and the estimators step by."""

    @abc.abstractmethod
    def interval_controls(self) -> AbstractControls:
        """The parameterization of one interval, which the estimators advance the model by."""

    @abc.abstractmethod
    def segments(self, u: ArrayLike, modes: tuple[str, ...]) -> ScheduleArrays:
        """The schedule arrays of the horizon for the decision vector `u`, traced.

        Args:
            u: The decision vector, `(size,)`.
            modes: The model's mode names, `problem.operational_modes()`.
        """


class PiecewiseConstant(AbstractControls):
    """Controls that hold one target of an operational row per interval of a fixed grid.

    Interval `k` holds the target `scale * u[k]`.

    Args:
        mode: The row of `spec.control.operational` the targets drive: `"current"`, `"voltage"`, `"power"` or `"resistance"`.
        intervals: The number of intervals, the length of the decision vector.
        duration: The length of each interval in the model's time unit, or a tuple of one length per interval.
        carry_step: As `ds.Schedule(carry_step=)`: `None` continues the integrator's step into the next interval, `False` starts each afresh.
        scale: The factor from the decision vector to the model's target, such as a C-rate's current with its sign.
    """

    mode: str = eqx.field(static=True)
    intervals: int = eqx.field(static=True)
    duration: float | tuple = eqx.field(static=True, converter=_lengths)
    carry_step: bool | None = eqx.field(static=True, default=None)
    scale: float = eqx.field(static=True, default=1.0, converter=float)

    @property
    def size(self) -> int:
        """The length of the decision vector, one target per interval."""
        return self.intervals

    @property
    def free_durations(self) -> bool:
        """`False`: the grid's durations are static."""
        return False

    def __check_init__(self) -> None:
        """Refuse a grid whose lengths do not match `intervals` or are not positive."""
        durations = np.asarray(self.durations, dtype=np.float64)
        if durations.shape != (self.intervals,) or not np.all(durations > 0):
            raise ConfigurationError(
                f"PiecewiseConstant needs {self.intervals} positive durations; got "
                f"{self.duration!r}."
            )

    @property
    def durations(self) -> tuple:
        """The length of each interval."""
        if isinstance(self.duration, tuple):
            return tuple(float(d) for d in self.duration)
        return (float(self.duration),) * self.intervals

    @property
    def horizon(self) -> float:
        """The end of the horizon: the durations added in order, as the schedule adds them."""
        end = 0.0
        for duration in self.durations:
            end += duration
        return end

    @property
    def interval(self) -> float:
        """The one length of every interval, which the receding horizon and the estimators step by."""
        durations = self.durations
        if any(d != durations[0] for d in durations):
            raise ConfigurationError(
                "This needs intervals of one length; the PiecewiseConstant has "
                f"durations {durations}."
            )
        return durations[0]

    def interval_controls(self) -> PiecewiseConstant:
        """The parameterization of one interval of this grid, keeping its mode, scale and step carry."""
        return dataclasses.replace(self, intervals=1, duration=self.interval)

    def segments(self, u: ArrayLike, modes: tuple[str, ...]) -> ScheduleArrays:
        """The schedule arrays of the horizon for the decision vector `u`, traced, never retraced for a new `u`.

        Args:
            u: The decision vector, `(intervals,)`.
            modes: The model's mode names, `problem.operational_modes()`.

        Returns:
            The [ScheduleArrays][diff_soup.ScheduleArrays] whose `target` leaf is `scale * u`.
        """
        template = schedule_arrays(
            Schedule(
                [Segment(**{self.mode: 0.0}, duration=d) for d in self.durations],
                carry_step=self.carry_step,
            ),
            modes,
        )
        u = jnp.asarray(u)
        if u.shape != (self.intervals,):
            raise ConfigurationError(
                f"PiecewiseConstant over {self.intervals} intervals takes a decision vector of "
                f"shape ({self.intervals},); got {u.shape}."
            )
        return dataclasses.replace(template, target=self.scale * u)


def _dotted(path: tuple) -> str:
    """A pytree key path as the dotted name `learnable_paths()` uses."""
    parts = []
    for key in path:
        if isinstance(key, jax.tree_util.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jax.tree_util.SequenceKey):
            parts.append(str(key.idx))
        else:
            parts.append(str(getattr(key, "key", key)))
    return ".".join(parts)


@dataclasses.dataclass(frozen=True, repr=False)
class _Block:
    """One block of `SignalControls`: its signal without the coefficients, static and compared by value.

    Attributes:
        structure: The signal's pytree structure, its static fields included.
        constants: Per leaf, `None` for a coefficient leaf, else `(values, shape, dtype)`.
        coefficients: Per coefficient leaf, in `learnable_paths()` order, `(leaf index, shape)`.
        mode: The operational row the block drives.
        scale: The factor from the signal to the row's target.
        lengths: The length of each of the signal's segments, from its knots.
        span: The signal's knots' span, `knots[-1] - knots[0]`.
        degree: The degree of the signal's polynomials.
        kind: The signal's class name, for the repr.
    """

    structure: Any
    constants: tuple
    coefficients: tuple
    mode: str
    scale: float
    lengths: tuple[float, ...]
    span: float
    degree: int
    kind: str = dataclasses.field(compare=False)

    @classmethod
    def of(cls, index: int, block: Any) -> _Block:
        """Read block `index`, a `(signal, mode, scale)` triple, refusing what cannot lower exactly."""
        if not isinstance(block, tuple) or len(block) != 3:
            raise ConfigurationError(
                f"SignalControls block {index} must be a (signal, mode, scale) tuple; got {block!r}."
            )
        signal, mode, scale = block
        kind = type(signal).__name__
        if mode not in operating_mode_names():
            raise ConfigurationError(
                f"SignalControls block {index} drives the mode {mode!r}; the operational modes "
                f"are {operating_mode_names()!r}."
            )
        if not callable(getattr(signal, "polynomials", None)):
            raise ConfigurationError(
                f"SignalControls block {index} holds a {kind}, which has no polynomials(); a block "
                "lowers a piecewise polynomial signal exactly: PiecewiseConstant, PiecewiseLinear, "
                "CubicHermite or BSpline from diff_soup.optimization.inference.approx."
            )
        knots = getattr(signal, "knots", None)
        if isinstance(knots, jax.core.Tracer) or knots is None:
            raise ConfigurationError(
                f"SignalControls block {index}: the {kind}'s knots are static; give concrete knots."
            )
        knots = np.asarray(knots, dtype=np.float64)
        lengths = np.diff(knots)
        if knots.ndim != 1 or lengths.size < 1 or not np.all(lengths > 0):
            raise ConfigurationError(
                f"SignalControls block {index}: the {kind}'s knots must ascend strictly; got {knots}."
            )
        shape = jax.eval_shape(signal.polynomials).shape
        if len(shape) != 2 or shape[0] != lengths.size:
            raise ConfigurationError(
                f"SignalControls block {index}: a block's {kind} has one component, polynomials of "
                f"shape ({lengths.size}, degree + 1); got {shape}."
            )
        paths = tuple(str(path) for path in signal.learnable_paths())
        flat, structure = jax.tree_util.tree_flatten_with_path(signal)
        names = [_dotted(path) for path, _ in flat]
        missing = [path for path in paths if path not in names]
        if not paths or missing:
            raise ConfigurationError(
                f"SignalControls block {index}: the {kind}'s learnable_paths() {paths} name no "
                f"leaves {missing or ''} among {names}."
            )
        coefficients = tuple(
            (names.index(path), np.shape(flat[names.index(path)][1])) for path in paths
        )
        taken = {leaf for leaf, _ in coefficients}
        constants = []
        for leaf, (_, value) in enumerate(flat):
            if leaf in taken:
                constants.append(None)
                continue
            value = np.asarray(value)
            constants.append((tuple(value.ravel().tolist()), value.shape, value.dtype.str))
        return cls(
            structure=structure,
            constants=tuple(constants),
            coefficients=coefficients,
            mode=mode,
            scale=float(scale),
            lengths=tuple(float(length) for length in lengths),
            span=float(knots[-1] - knots[0]),
            degree=shape[1] - 1,
            kind=kind,
        )

    @property
    def width(self) -> int:
        """The block's entries of the decision vector, its coefficient leaves' sizes added."""
        return sum(math.prod(shape) for _, shape in self.coefficients)

    def signal(self, values: Float[Array, " width"]) -> eqx.Module:
        """The block's signal with its coefficient leaves taken in order from `values`."""
        leaves = [
            None
            if constant is None
            else jnp.asarray(np.asarray(constant[0], dtype=constant[2]).reshape(constant[1]))
            for constant in self.constants
        ]
        start = 0
        for leaf, shape in self.coefficients:
            size = math.prod(shape)
            leaves[leaf] = jnp.reshape(values[start : start + size], shape)
            start += size
        return jax.tree_util.tree_unflatten(self.structure, leaves)

    def __repr__(self) -> str:
        """The signal's class and segment count, the mode and the scale."""
        return f"({self.kind} over {len(self.lengths)} segments, {self.mode!r}, {self.scale!r})"


def _no_grid() -> ConfigurationError:
    """The refusal of a grid of moves from `SignalControls`."""
    return ConfigurationError(
        "SignalControls hold signals, not a grid of moves of one interval each; the receding "
        "horizon and the estimators step PiecewiseConstant controls."
    )


class SignalControls(AbstractControls):
    """Controls that hold time signals of operational rows, one block after another.

    Block `(signal, mode, scale)` holds the row `mode` at `scale * signal(t)` over the signal's
    knots, shifted to start where the block before it ends. The signal is a piecewise basis of
    `diff_soup.optimization.inference.approx`, `PiecewiseConstant`, `PiecewiseLinear`,
    `CubicHermite` or `BSpline`, and each of its segments lowers exactly to one schedule
    segment, whose target is the segment's polynomial from `signal.polynomials()`. The decision
    vector holds each block's coefficient leaves, its `learnable_paths()`, one block after
    another; the signals' own coefficient values are not read.

    One block of a `BSpline` current is a smooth current profile; a current block then a
    voltage block is a charge at constant current, then at constant voltage. With
    `free_durations` the switch from one block to the next is a decision variable, and a
    minimum-time problem is one solve with the durations in its cost.

    Args:
        blocks: The blocks in time order, each `(signal, mode, scale)`: a signal of one component, the operational row it drives (`"current"`, `"voltage"`, `"power"` or `"resistance"`), and the factor from the signal to the row's target.
        carry_step: As `ds.Schedule(carry_step=)`: `None` continues the integrator across the knots of a block, where the target is continuous, and starts it afresh where the mode changes; `False` starts every segment afresh.
        free_durations: Whether each block's duration is an entry of the decision vector, one per block after every coefficient. A block of duration `D` runs its signal with the knots stretched from their span to `D`. Keep each duration positive through `bounds`.
    """

    blocks: tuple[_Block, ...] = eqx.field(static=True)
    carry_step: bool | None = eqx.field(static=True, default=None)
    free_durations: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        blocks: Sequence[tuple],
        carry_step: bool | None = None,
        free_durations: bool = False,
    ):
        """Read the blocks; see the class docstring."""
        if isinstance(blocks, tuple) and len(blocks) == 3 and isinstance(blocks[1], str):
            raise ConfigurationError(
                "SignalControls takes a sequence of (signal, mode, scale) blocks; wrap one block "
                "as [(signal, mode, scale)]."
            )
        self.blocks = tuple(_Block.of(index, block) for index, block in enumerate(blocks))
        if not self.blocks:
            raise ConfigurationError("SignalControls needs at least one block.")
        if carry_step is not None and not isinstance(carry_step, (bool, np.bool_)):
            raise ConfigurationError(
                f"SignalControls carry_step must be None, True or False; got {carry_step!r}."
            )
        self.carry_step = None if carry_step is None else bool(carry_step)
        if not isinstance(free_durations, (bool, np.bool_)):
            raise ConfigurationError(
                f"SignalControls free_durations must be True or False; got {free_durations!r}."
            )
        self.free_durations = bool(free_durations)

    @property
    def size(self) -> int:
        """The length of the decision vector: every block's coefficient leaves, then its duration if free."""
        coefficients = sum(block.width for block in self.blocks)
        return coefficients + (len(self.blocks) if self.free_durations else 0)

    @property
    def durations(self) -> tuple:
        """The length of each segment: the spacing of each block's knots, block after block.

        With free durations these are the lengths at which every block lasts its knots'
        span; the decision vector's durations stretch each block from there.
        """
        return tuple(length for block in self.blocks for length in block.lengths)

    @property
    def horizon(self) -> float:
        """The end of the horizon: the durations added in order, as the schedule adds them.

        With free durations, the end at which every block lasts its knots' span.
        """
        end = 0.0
        for duration in self.durations:
            end += duration
        return end

    @property
    def interval(self) -> float:
        """Refused: signals are not a grid of moves for the receding horizon to step."""
        raise _no_grid()

    def interval_controls(self) -> AbstractControls:
        """Refused, as `interval`."""
        raise _no_grid()

    def segments(self, u: ArrayLike, modes: tuple[str, ...]) -> ScheduleArrays:
        """The schedule arrays of the horizon for the decision vector `u`, traced, never retraced for a new `u`.

        Args:
            u: The decision vector, `(size,)`.
            modes: The model's mode names, `problem.operational_modes()`.

        Returns:
            The [ScheduleArrays][diff_soup.ScheduleArrays], one segment per segment of each
            block's signal, whose `target` and `terms` leaves are `scale` times the signal's
            polynomials. With free durations the `duration` leaf is traced too, and a block of
            duration `D` scales the coefficient of `tau**j` by `(span / D)**j`, `span` its
            knots' span.
        """
        degree = max(block.degree for block in self.blocks)
        template = schedule_arrays(
            Schedule(
                [
                    Segment(**{block.mode: 0.0}, duration=length)
                    for block in self.blocks
                    for length in block.lengths
                ],
                carry_step=self.carry_step,
            ),
            modes,
            degree=degree,
        )
        u = jnp.asarray(u)
        if u.shape != (self.size,):
            raise ConfigurationError(
                f"SignalControls of {self.size} coefficients take a decision vector of shape "
                f"({self.size},); got {u.shape}."
            )
        rows, lengths, start = [], [], 0
        coefficients = self.size - (len(self.blocks) if self.free_durations else 0)
        for index, block in enumerate(self.blocks):
            polynomials = block.signal(u[start : start + block.width]).polynomials()
            start += block.width
            polynomials = jnp.pad(polynomials, ((0, 0), (0, degree - block.degree)))
            if self.free_durations:
                duration = u[coefficients + index]
                # The block's time runs span / duration times slower than its signal's.
                stretch = block.span / duration
                powers = [jnp.ones_like(stretch)]
                for _ in range(degree):
                    powers.append(powers[-1] * stretch)
                polynomials = polynomials * jnp.stack(powers)
                shares = np.asarray(block.lengths, dtype=np.float64) / block.span
                lengths.append(duration * jnp.asarray(shares, dtype=duration.dtype))
            rows.append(block.scale * polynomials)
        rows = jnp.concatenate(rows)
        arrays = dataclasses.replace(
            template, target=rows[:, 0], terms=rows[:, 1:] if degree else None
        )
        if self.free_durations:
            arrays = dataclasses.replace(arrays, duration=jnp.concatenate(lengths))
        return arrays


def _static_times(times: Any) -> tuple[float, ...] | None:
    """Save times as Python floats, static and hashable; a traced value is refused."""
    if times is None:
        return None
    if isinstance(times, jax.core.Tracer):
        raise ConfigurationError(
            "ControlProblem save_times are static: give concrete times, not traced ones."
        )
    return tuple(float(t) for t in np.asarray(times, dtype=np.float64).ravel())


def _static_fractions(fractions: Any) -> tuple[float, ...] | None:
    """Save fractions as Python floats, static and hashable; a traced value is refused."""
    if fractions is None:
        return None
    if isinstance(fractions, jax.core.Tracer):
        raise ConfigurationError(
            "ControlProblem save_fractions are static: give concrete fractions, not traced ones."
        )
    return tuple(float(f) for f in np.asarray(fractions, dtype=np.float64).ravel())


def _static_start(start: Any) -> tuple[float, ...] | None:
    """The first guess as Python floats, static; a traced value is refused."""
    if start is None:
        return None
    if isinstance(start, jax.core.Tracer):
        raise ConfigurationError(
            "ControlProblem start is static: give a concrete first guess, not a traced one."
        )
    return tuple(float(value) for value in np.asarray(start, dtype=np.float64).ravel())


class ControlProblem(AbstractOptimizationProblem, eqx.Module):
    """Minimize a cost over the controls of a model under path and terminal constraints.

    The horizon runs through `ds.solve_schedule` with the segment targets as traced arrays of
    `u`. The path rows' Jacobian is forward mode through the schedule tangents; the cost
    gradient is reverse mode under `CheckpointAdjoint` when it pays. `args` carries the initial
    state, `theta` and the limits, so a batch of cells or scenarios is one `vmap`. The model,
    the save times and the first guess are static: the model is held by identity, so two
    problems over one model, one configuration and different `args` share a compile.

    Args:
        problem: A built diff-soup problem whose operational table has the row `controls` drives.
        controls: The parameterization of the decision vector, [AbstractControls][diff_soup.optimization.control.AbstractControls]: [PiecewiseConstant][diff_soup.optimization.control.PiecewiseConstant] or [SignalControls][diff_soup.optimization.control.SignalControls].
        cost: `cost(sol, u, args) -> scalar`, with `sol` the `Solution` over the horizon at `save_times`.
        path: `path(sol, u, args) -> rows` at or below 0 at every saved time, or `None`.
        terminal: `terminal(sol, u, args) -> rows` at 0 at the end of the horizon, or `None`.
        n_path: The number of rows `path` returns per saved time.
        n_terminal: The number of rows `terminal` returns.
        bounds: `(lower, upper)` on `u`, broadcast, or `None`.
        constraints: [Constraints][diff_soup.optimization.Constraints] on `u` itself, functions of `(u, args)`, or `None`. Their scales are padded with ones over the terminal and path rows, and their `jacobian` mode differentiates their own rows.
        save_times: The times the solution is sampled at for the functionals, after the horizon's start at 0. A last time at the horizon's end, within the rounding of its sum, is the schedule's own end.
        save_fractions: In place of `save_times`, the positions within every segment the solution is sampled at, increasing within `(0, 1]`, `1` the segment's end: the times follow the durations in the trace. Controls with free durations need them.
        initial_state: The state the horizon starts from; an `args` leaf.
        theta: The model parameters; an `args` leaf.
        limits: What the functionals read as limits, such as a voltage ceiling; an `args` leaf.
        start: The first guess of `u`; the middle of the bounds, or zeros, when `None`.
        solver: The time-stepping solver and its tolerances.
        adjoint: The differentiation strategy of the horizon solve, `None` for forward mode.
        layout: The placement of a batch of horizons.
    """

    problem: Held = eqx.field(static=True, converter=hold)
    constraints: Constraints | None = None
    bounds: tuple[ArrayLike | None, ArrayLike | None] | None = None
    controls: AbstractControls | None = eqx.field(static=True, default=None)
    cost: Callable | None = eqx.field(static=True, default=None)
    path: Callable | None = eqx.field(static=True, default=None)
    terminal: Callable | None = eqx.field(static=True, default=None)
    n_path: int = eqx.field(static=True, default=0)
    n_terminal: int = eqx.field(static=True, default=0)
    save_times: tuple[float, ...] | None = eqx.field(
        static=True, default=None, converter=_static_times
    )
    save_fractions: tuple[float, ...] | None = eqx.field(
        static=True, default=None, converter=_static_fractions
    )
    initial_state: Float[ArrayLike, "*batch n_states"] | None = None
    theta: PyTree = None
    limits: PyTree = None
    start: tuple[float, ...] | None = eqx.field(static=True, default=None, converter=_static_start)
    solver: JIDA | JBDF | JNDF | None = eqx.field(static=True, default=None)
    adjoint: AbstractAdjoint | None = eqx.field(static=True, default=None)
    layout: Layout | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        """Take a last save time within the rounding of the horizon's sum as its end."""
        times = _static_times(self.save_times)
        if (
            times
            and isinstance(self.controls, AbstractControls)
            and not self.controls.free_durations
        ):
            end = self.controls.horizon
            rounding = len(self.controls.durations) * _END_ROUNDING * end
            if end < times[-1] <= end + rounding:
                self.save_times = (*times[:-1], end)

    @property
    def model(self) -> object:
        """The built diff-soup problem, held by identity."""
        return self.problem.value

    def numeric_problem(self, optimizer: OptSolver) -> OptimizationProblem:
        """Lower to closures over `u` and `args`; see [AbstractOptimizationProblem.numeric_problem][diff_soup.optimization.AbstractOptimizationProblem.numeric_problem]."""
        horizon = self._closures()
        n = self.controls.size
        start = _middle(self.bounds, n) if self.start is None else jnp.asarray(self.start)
        x0 = jnp.reshape(jnp.asarray(start, dtype=jnp.float64), (n,))
        structure = {
            "value_gradient",
            "hessian",
            "constraint_jacobian",
            "lagrangian_hessian",
            "constraint_hessian",
        }
        n_equality, n_inequality = horizon.n_equality, horizon.n_inequality
        if n_equality or n_inequality:
            structure.add("constraints")
        return OptimizationProblem(
            x0=x0,
            structure=frozenset(structure),
            bounds=as_bounds(self.bounds, x0),
            args=self.args(),
            n_equality=n_equality,
            n_inequality=n_inequality,
            constraint_scales=self._row_scales(),
            ad="forward" if self.adjoint is None else "reverse",
            _value=horizon.closure("value"),
            _value_and_gradient=horizon.closure("value_and_gradient"),
            _value_gradient_hessian=horizon.closure("value_gradient_hessian"),
            _constraints=horizon.closure("constraints"),
            _constraints_and_jacobians=horizon.closure("constraints_and_jacobians"),
            _lagrangian_hessian=horizon.closure("lagrangian_hessian"),
            _constraint_hessian=horizon.closure("constraint_hessian"),
            _evaluate=horizon.closure("evaluate"),
            _evaluate_with_derivatives=horizon.closure("evaluate_with_derivatives"),
        )

    def args(self) -> tuple[PyTree, PyTree, PyTree]:
        """`(initial_state, theta, limits)`, the leaves a batch varies."""
        return (self.initial_state, self.theta, self.limits)

    def _closures(self) -> Horizon:
        """The horizon's closures, equal for every control problem of this configuration.

        Problems over the same model with the same functionals, controls, constraints on `u`,
        save times, solver and adjoint have equal closures, so a solver compiled for one
        serves the next without retracing.
        """
        constraints = self.constraints
        if constraints is not None:
            # The scales are arrays the solver reads from the numeric problem, not the closures.
            constraints = dataclasses.replace(constraints, scales=None)
        return Horizon(
            Configuration(
                model=self.problem,
                controls=self.controls,
                cost=self.cost,
                path=self.path,
                terminal=self.terminal,
                n_path=self.n_path,
                n_terminal=self.n_terminal,
                constraints=constraints,
                save_times=self.save_times,
                save_fractions=self.save_fractions,
                solver=self.solver,
                adjoint=self.adjoint,
            )
        )

    def _row_scales(self) -> tuple[Float[Array, " n_eq"], Float[Array, " n_ineq"]] | None:
        """The scales of the `Constraints` on `u`, after ones for the terminal and path rows."""
        scales = None if self.constraints is None else self.constraints.row_scales()
        if scales is None:
            return None
        path = self.n_path * self._closures().n_samples
        equality, inequality = scales
        return (
            jnp.concatenate([jnp.ones((self.n_terminal,), equality.dtype), equality]),
            jnp.concatenate([jnp.ones((path,), inequality.dtype), inequality]),
        )

    def __check_init__(self) -> None:
        """Refuse a control problem missing what its lowering needs, or with save times off its horizon."""
        missing = [
            name
            for name in ("controls", "cost", "save_times", "initial_state", "solver")
            if getattr(self, name) is None
        ]
        if "save_times" in missing:
            at = missing.index("save_times")
            if self.save_fractions is None:
                missing[at] = "save_times or save_fractions"
            else:
                del missing[at]
        if self.model is None:
            missing.insert(0, "problem")
        if missing:
            raise ConfigurationError(f"ControlProblem needs {', '.join(missing)}.")
        if not isinstance(self.controls, AbstractControls):
            raise ConfigurationError(
                "ControlProblem controls must be AbstractControls, such as PiecewiseConstant; "
                f"got {type(self.controls).__name__}."
            )
        if (self.path is None) != (self.n_path == 0) or (self.terminal is None) != (
            self.n_terminal == 0
        ):
            raise ConfigurationError(
                "ControlProblem: give path with n_path and terminal with n_terminal, or "
                "neither of each pair."
            )
        if self.save_times is not None and self.save_fractions is not None:
            raise ConfigurationError(
                "ControlProblem samples at save_times or at save_fractions of every segment; "
                "give one of them."
            )
        if self.save_fractions is not None:
            fractions = np.asarray(self.save_fractions, dtype=np.float64)
            if (
                fractions.size == 0
                or np.any(np.diff(fractions) <= 0)
                or not (0 < fractions[0] and fractions[-1] <= 1)
            ):
                raise ConfigurationError(
                    "ControlProblem save_fractions must increase strictly within (0, 1]."
                )
            return
        if self.controls.free_durations:
            raise ConfigurationError(
                "ControlProblem: the controls' durations are decision variables, so a sample "
                "time moves with them; give save_fractions, the positions within every segment, "
                "in place of save_times."
            )
        times = np.asarray(self.save_times, dtype=np.float64)
        horizon = self.controls.horizon
        if times.size == 0 or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > horizon:
            raise ConfigurationError(
                f"ControlProblem save_times must increase strictly within [0, {horizon}]."
            )


def _middle(bounds: tuple[ArrayLike | None, ArrayLike | None] | None, n: int) -> Float[Array, " n"]:
    """The middle of finite bounds, the finite bound where one side is open, zero where both are."""
    if bounds is None:
        return jnp.zeros((n,))
    lower, upper = (jnp.broadcast_to(jnp.asarray(side, dtype=jnp.float64), (n,)) for side in bounds)
    finite_lower, finite_upper = jnp.isfinite(lower), jnp.isfinite(upper)
    middle = jnp.where(finite_lower & finite_upper, 0.5 * (lower + upper), 0.0)
    middle = jnp.where(finite_lower & ~finite_upper, lower, middle)
    return jnp.where(~finite_lower & finite_upper, upper, middle)
