"""A schedule's segments, the schedule, and the arrays it lowers to."""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Iterator
from typing import Any, Optional, Union

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup.problem._spec.operating_modes import operating_mode_names

_ARRAY_FIELDS = ("mode", "target", "duration", "active", "stop_kind", "stop_level", "carry_step")


_HostArray = Union[np.ndarray, Array]


def _as_real(value: Any, what: str) -> float:
    """``value`` as a finite Python float, or an error that names ``what``.

    Args:
        value: A real number: a Python or NumPy number, or an array of one element.
        what: The quantity, as the error message names it.

    Raises:
        TypeError: If ``value`` is traced, a boolean, not a real number, or an array
            of more than one element.
        ValueError: If ``value`` is not finite.
    """
    if isinstance(value, jax.core.Tracer):
        raise TypeError(
            f"{what} is a traced value, and a schedule holds plain numbers; build the "
            "schedule from concrete numbers, outside jax.jit, jax.vmap and differentiation."
        )
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} is the boolean {value!r}; pass a real number.")
    array = None if isinstance(value, (str, bytes)) else np.asarray(value)
    if array is None or not (
        np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(
            f"{what} must be a real number; got {value!r} of type {type(value).__name__}."
        )
    if array.size != 1:
        raise TypeError(
            f"{what} must be one number; got an array of {array.size} elements with shape "
            f"{array.shape}."
        )
    number = float(array.reshape(()))
    if not math.isfinite(number):
        raise ValueError(f"{what} must be a finite number; got {number!r}.")
    return number


def _as_target(value: Any, what: str) -> Union[float, tuple]:
    """A segment's target: one finite float, or a tuple of two or more of them.

    A sequence (or one-dimensional array) gives the coefficients of a
    polynomial in the time since the segment's start, constant term first; a
    sequence of one coefficient is that constant.

    Raises:
        TypeError: If ``value`` is not a real number or a sequence of them.
        ValueError: If a coefficient is not finite, or the sequence is empty.
    """
    if isinstance(value, (list, tuple)) or (
        isinstance(value, (np.ndarray, jax.Array))
        and not isinstance(value, jax.core.Tracer)
        and np.ndim(value) == 1
    ):
        items = tuple(value)
        if not items:
            raise ValueError(
                f"{what} is an empty sequence; give a number, or the coefficients of a "
                "polynomial in the time since the segment's start, constant term first."
            )
        coefficients = tuple(
            _as_real(item, f"coefficient {index} of {what}") for index, item in enumerate(items)
        )
        return coefficients[0] if len(coefficients) == 1 else coefficients
    return _as_real(value, what)


def _as_capacity(capacity: Any) -> int:
    """``capacity`` as a Python int, or a ``TypeError`` that says what it must be."""
    if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, numbers.Integral):
        raise TypeError(f"capacity must be a whole number of segments; got {capacity!r}.")
    return int(capacity)


def _mode_names(modes: Any) -> Optional[tuple]:
    """``modes`` as a tuple of strings, or ``None`` when it is not a sequence of them."""
    try:
        names = tuple(modes)
    except TypeError:
        return None
    return names if all(isinstance(name, str) for name in names) else None


@dataclasses.dataclass(frozen=True, init=False, repr=False)
class Segment:
    """One step of a `Schedule`, holding one row of the operating table at a target for a duration.

    Pass exactly one of `current`, `voltage`, `power`, or `resistance`. A tuple target is a polynomial in the time since the segment started, constant term first. All values are concrete numbers, built outside `jax.jit`.

    Args:
        current: Target of the current row.
        voltage: Target of the voltage row.
        power: Target of the power row.
        resistance: Target of the resistance row, ``voltage - target * current``.
        duration: Positive length of the segment.
    """

    mode: str
    target: Union[float, tuple]
    duration: float

    def __init__(
        self,
        *,
        current: Any = None,
        voltage: Any = None,
        power: Any = None,
        resistance: Any = None,
        duration: Any,
    ) -> None:
        """Build a segment from one current, voltage, power or resistance target.

        Args:
            current: The current to hold: a number, or the coefficients of a
                polynomial in the time since the segment's start, constant
                term first.
            voltage: The voltage to hold, in the same forms.
            power: The power to hold, in the same forms.
            resistance: The resistance to hold, in the same forms.
            duration: How long to hold it; a positive finite number.

        Raises:
            TypeError: If none or more than one of ``current``, ``voltage``,
                ``power``, ``resistance`` is given, or a value is not a real number or a
                sequence of them (a boolean, a multi-dimensional array, a
                traced value).
            ValueError: If a value is not finite, a sequence of coefficients
                is empty, or the duration is not positive.
        """
        given = tuple(
            (name, value)
            for name, value in (
                ("current", current),
                ("voltage", voltage),
                ("power", power),
                ("resistance", resistance),
            )
            if value is not None
        )
        if len(given) != 1:
            received = "none of them" if not given else " and ".join(f"{n}=" for n, _ in given)
            raise TypeError(
                "a Segment holds exactly one mode: pass one of current=, voltage=, power= or resistance= "
                f"with its target; got {received}."
            )
        mode, target = given[0]
        target = _as_target(target, f"the {mode} target of a Segment")
        duration = _as_real(duration, "the duration of a Segment")
        if duration <= 0.0:
            raise ValueError(f"the duration of a Segment must be positive; got {duration!r}.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "duration", duration)

    @property
    def coefficients(self) -> tuple:
        """The target's coefficients in the time since the segment's start, constant term first."""
        return self.target if isinstance(self.target, tuple) else (self.target,)

    @property
    def degree(self) -> int:
        """The degree of the target's polynomial; ``0`` for a constant target."""
        return len(self.coefficients) - 1

    def __repr__(self) -> str:
        """The constructor call that builds this segment."""
        return f"Segment({self.mode}={self.target!r}, duration={self.duration!r})"


@dataclasses.dataclass(frozen=True, init=False)
class Schedule:
    """An operating protocol, a sequence of segments run one after another.

    Each segment starts where the one before it ended. Pass the schedule as `schedule=` on `ds.optimization.inference.Experiment`.

    Args:
        segments: Sequence of at least one `Segment`.
        carry_step: Whether each segment keeps the step size and history of the one before, with `None` keeping them only when the mode is unchanged.
    """

    segments: tuple[Segment, ...]
    carry_step: Optional[bool] = None

    def __init__(self, segments: Any, *, carry_step: Optional[bool] = None) -> None:
        """Build a schedule from a sequence of segments.

        Args:
            segments: A non-empty sequence of ``Segment`` entries, in the order
                they run.
            carry_step: Which later segments continue the integrator's step
                size, order and history: ``None`` (the default) across a
                switch that keeps the mode, ``True`` across every switch,
                ``False`` across none. ``False`` resolves the derivative with
                respect to one segment's target at its switch, which a
                continued segment gives to first order in the step there.

        Raises:
            TypeError: If ``segments`` is not a sequence of ``Segment`` entries,
                or ``carry_step`` is not ``None``, ``True`` or ``False``.
            ValueError: If ``segments`` is empty.
        """
        if carry_step is not None and not isinstance(carry_step, (bool, np.bool_)):
            raise TypeError(
                f"carry_step must be None, True or False; got {carry_step!r} of type "
                f"{type(carry_step).__name__}."
            )
        if isinstance(segments, Segment):
            raise TypeError(
                "Schedule takes a sequence of segments; wrap a single segment as "
                "Schedule([segment])."
            )
        try:
            items = tuple(segments)
        except TypeError:
            raise TypeError(
                f"Schedule takes a sequence of Segment entries; got {type(segments).__name__}."
            ) from None
        if not items:
            raise ValueError("a Schedule needs at least one segment; got an empty sequence.")
        for index, item in enumerate(items):
            if not isinstance(item, Segment):
                raise TypeError(
                    f"entry {index} of the schedule is a {type(item).__name__}; every entry "
                    "must be a Segment."
                )
        object.__setattr__(self, "segments", items)
        object.__setattr__(self, "carry_step", None if carry_step is None else bool(carry_step))

    def __len__(self) -> int:
        """The number of segments."""
        return len(self.segments)

    def __iter__(self) -> Iterator[Segment]:
        """The segments, in order."""
        return iter(self.segments)

    @property
    def duration(self) -> float:
        """The total duration: the durations of the segments added in order."""
        total = 0.0
        for segment in self.segments:
            total += segment.duration
        return total

    @property
    def degree(self) -> int:
        """The largest target degree of the segments; ``0`` when every target is constant."""
        return max(segment.degree for segment in self.segments)

    def carries(self) -> tuple:
        """Per segment, whether it continues the integrator where the previous one ended.

        The first segment never does; a later one does by ``carry_step``.
        """
        flags = [False]
        for before, segment in zip(self.segments[:-1], self.segments[1:]):
            if self.carry_step is None:
                flags.append(segment.mode == before.mode)
            else:
                flags.append(bool(self.carry_step))
        return tuple(flags)


class ScheduleArrays(eqx.Module):
    """A schedule lowered against one model's mode names: arrays of one length.

    Built by [schedule_arrays][diff_soup._schedule.schedule_arrays] on the host,
    with NumPy only. The length is the capacity: entries past the schedule's own
    segments are inactive and do not run. Every leaf enters a compiled solve as
    an argument, so arrays of one capacity and one target degree share one
    program.

    Attributes:
        mode: ``int32``; the index of each segment's mode in the model's
            mode-name tuple.
        target: Each segment's target at its start: the constant term of its
            polynomial.
        duration: Each segment's duration.
        active: ``bool``; whether the entry is a segment of the schedule.
        stop_kind: ``int32``; how each segment ends. ``0``, the only value
            supported, ends it when its duration has elapsed.
        stop_level: The level of a stop condition; unused while every
            ``stop_kind`` is ``0``.
        carry_step: ``bool``; whether each segment continues the integrator
            where the segment before it ended (``Schedule.carries()``);
            ``False`` for the first and for inactive entries.
        terms: ``(capacity, degree)``; row ``k`` holds the coefficients of
            ``tau, ..., tau**degree`` of segment ``k``'s target, ``tau`` the
            time since its start, zero past the segment's own degree. ``None``
            when every target is constant, so a schedule of constant targets
            has no such leaf.
    """

    mode: _HostArray
    target: _HostArray
    duration: _HostArray
    active: _HostArray
    stop_kind: _HostArray
    stop_level: _HostArray
    carry_step: _HostArray
    terms: Optional[_HostArray] = None

    @property
    def degree(self) -> int:
        """The target degree the arrays hold: the columns of ``terms``, ``0`` without it."""
        return 0 if self.terms is None else int(np.shape(self.terms)[-1])

    def padded(
        self, capacity: numbers.Integral, *, degree: Optional[numbers.Integral] = None
    ) -> ScheduleArrays:
        """These arrays with inactive entries appended up to ``capacity``.

        An inactive entry has mode ``0``, target ``0``, duration ``0``, carries
        no step and does not run. With ``degree``, the target terms gain zero
        columns up to that degree (a constant schedule gains a ``terms`` leaf
        of zeros). Pad on the host, before the arrays enter a traced function.

        Args:
            capacity: The length of the result; at least the current length.
            degree: The target degree of the result; at least the current one.
                ``None`` keeps the current degree.

        Raises:
            TypeError: If ``capacity`` or ``degree`` is not a whole number, or a
                leaf is traced.
            ValueError: If ``capacity`` is smaller than the current length, or
                ``degree`` smaller than the current degree.
        """
        capacity = _as_capacity(capacity)
        degree = self.degree if degree is None else _as_degree(degree)
        leaves = [getattr(self, name) for name in _ARRAY_FIELDS] + [self.terms]
        if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            raise TypeError(
                "ScheduleArrays.padded runs on the host; pad the arrays before jax.jit or "
                "jax.vmap traces them."
            )
        count = int(np.shape(self.mode)[0])
        if capacity < count:
            raise ValueError(
                f"capacity {capacity} is smaller than the {count} entries of these arrays; "
                f"pass a capacity of at least {count}."
            )
        if degree < self.degree:
            raise ValueError(
                f"degree {degree} is smaller than the target degree {self.degree} of these "
                f"arrays; pass a degree of at least {self.degree}."
            )
        if capacity == count and degree == self.degree:
            return self

        def extend(leaf):
            leaf = np.asarray(leaf)
            return np.concatenate([leaf, np.zeros((capacity - count,), dtype=leaf.dtype)])

        terms = None
        if degree:
            dtype = np.asarray(self.target).dtype
            terms = np.zeros((capacity, degree), dtype=dtype)
            if self.terms is not None:
                terms[:count, : self.degree] = np.asarray(self.terms)
        values = {name: extend(getattr(self, name)) for name in _ARRAY_FIELDS}
        return ScheduleArrays(**values, terms=terms)


def _as_degree(degree: Any) -> int:
    """``degree`` as a non-negative Python int, or an error that says what it must be."""
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, numbers.Integral):
        raise TypeError(f"degree must be a whole number; got {degree!r}.")
    if int(degree) < 0:
        raise ValueError(f"degree must be at least 0; got {int(degree)}.")
    return int(degree)


def schedule_arrays(
    schedule: Schedule,
    modes: Any,
    *,
    capacity: Optional[numbers.Integral] = None,
    dtype: Any = np.float64,
    degree: Optional[numbers.Integral] = None,
) -> ScheduleArrays:
    """Lower ``schedule`` against a model's mode names to arrays of one length.

    Args:
        schedule: The schedule.
        modes: The model's mode-name tuple, ``problem.operational_modes()``.
        capacity: The length of the arrays; ``None`` is the number of segments.
            Entries past the schedule are inactive.
        dtype: The floating dtype of the targets, durations and stop levels.
        degree: The target degree of the arrays; ``None`` is the schedule's own
            (``Schedule.degree``). A degree of ``0`` gives no ``terms`` leaf.

    Returns:
        The [ScheduleArrays][diff_soup._schedule.ScheduleArrays], NumPy leaves.

    Raises:
        TypeError: If ``schedule`` is not a ``Schedule``, ``modes`` is not a
            sequence of mode names, ``capacity`` or ``degree`` is not a whole
            number, or ``dtype`` is not a floating dtype.
        ValueError: If ``capacity`` is smaller than the number of segments, or
            ``degree`` smaller than the schedule's target degree.
        ConfigurationError: If a segment's mode is not one of ``modes``.
    """
    if not isinstance(schedule, Schedule):
        raise TypeError(f"schedule must be a Schedule; got {type(schedule).__name__}.")
    names = None if isinstance(modes, str) else _mode_names(modes)
    if names is None:
        raise TypeError(
            f"modes must be the model's tuple of mode names, such as {operating_mode_names()!r}; "
            f"got {modes!r}."
        )
    modes = names
    count = len(schedule)
    capacity = count if capacity is None else _as_capacity(capacity)
    if capacity < count:
        raise ValueError(
            f"capacity {capacity} is smaller than the {count} segments of the schedule; "
            f"pass a capacity of at least {count}."
        )
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"dtype must be a floating dtype; got {dtype}.")
    indices = []
    for index, segment in enumerate(schedule):
        if segment.mode not in modes:
            raise ConfigurationError(
                f"segment {index} holds the mode {segment.mode!r}, which the model's "
                f"operational table does not declare; the model's modes are {modes!r}. Use "
                f"one of those modes, or declare {segment.mode!r} in "
                "spec.control.operational(modes=...)."
            )
        indices.append(modes.index(segment.mode))
    own = schedule.degree
    degree = own if degree is None else _as_degree(degree)
    if degree < own:
        raise ValueError(
            f"degree {degree} is smaller than the schedule's target degree {own}; pass a "
            f"degree of at least {own}."
        )
    terms = None
    if degree:
        terms = np.zeros((count, degree), dtype=dtype)
        for index, segment in enumerate(schedule):
            higher = segment.coefficients[1:]
            terms[index, : len(higher)] = higher
    arrays = ScheduleArrays(
        mode=np.asarray(indices, dtype=np.int32),
        target=np.asarray([segment.coefficients[0] for segment in schedule], dtype=dtype),
        duration=np.asarray([segment.duration for segment in schedule], dtype=dtype),
        active=np.ones((count,), dtype=bool),
        stop_kind=np.zeros((count,), dtype=np.int32),
        stop_level=np.zeros((count,), dtype=dtype),
        carry_step=np.asarray(schedule.carries(), dtype=bool),
        terms=terms,
    )
    return arrays.padded(capacity)


class SegmentEnds(eqx.Module):
    """How each segment of a scheduled solve ended; one entry per segment.

    Attributes:
        t_end: The time each segment that ran reached; ``NaN`` for a segment
            that did not run.
        ran: Whether each segment ran.
        result_code: The solver's termination code of each segment that ran;
            ``-1`` for a segment that did not run.
        first_step: The size of each segment's first trial step, its solve's
            ``SolverStats.first_step``; ``NaN`` for a segment that did not run,
            and for every segment when the solver reports no step statistics.
        last_step: The step each segment's solve would have taken next when it
            stopped, its ``SolverStats.last_step``: the step a later segment
            that continues starts from. ``NaN`` as for ``first_step``.
        continued: Whether each segment was handed the integrator's history
            from the segment before it (its ``carry_step`` entry was set and
            that segment reached its end); ``False`` for a segment that started
            afresh or did not run. Across a switch that changes what the row
            demands, a segment handed the history starts at order one from the
            step it carries.
    """

    t_end: Array
    ran: Array
    result_code: Array
    first_step: Array
    last_step: Array
    continued: Array
