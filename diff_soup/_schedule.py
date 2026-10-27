"""Protocols of operating segments, and the solve that runs them in order.

A model that declares an operational control table
(``spec.control.operational(...)``) closes one algebraic unknown with one of the
rows ``current``, ``voltage`` or ``power``, chosen at run time by a mode index and
a target. A ``Schedule`` lists ``Segment`` entries, each of which holds one of
those rows at a target for a duration: a charge at constant current, then a
hold at constant voltage, and so on. A target is a constant, or a polynomial in
the time since the segment started.

This module holds the two public types, their lowering to arrays of one static
length against one model's mode names (``schedule_arrays``), and the solve that
runs the segments one after another from the carried state
(``solve_schedule``). Every number of a lowered schedule enters a compiled
program as an argument, so schedules of one length and one target degree share
one program.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Iterator
from typing import Any, NamedTuple, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup._solution import RESULT
from diff_soup.pdesolve import pdesolve
from diff_soup.problem.implicit import get_or_build_solver_view
from diff_soup.timestepping import ControlState, SaveAt
from diff_soup.timestepping.event_records import TerminalRecord
from diff_soup.timestepping.vsvo.solution import Solution, SolveHistory

__all__ = [
    "Schedule",
    "ScheduleArrays",
    "Segment",
    "SegmentEnds",
    "schedule_arrays",
    "solve_schedule",
]

_MODES = ("current", "voltage", "power")
_ARRAY_FIELDS = ("mode", "target", "duration", "active", "stop_kind", "stop_level", "carry_step")
# A segment that does not run still executes one solve, because under ``vmap``
# both sides of a choice execute. That solve spans this fraction of the whole
# solve, starting where the last segment that ran ended, and its outputs are
# discarded.
_IDLE_FRACTION = 1e-6

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

    Pass exactly one of `current`, `voltage`, or `power`. A tuple target is a polynomial in the time since the segment started, constant term first. All values are concrete numbers, built outside `jax.jit`.

    Args:
        current: Target of the current row.
        voltage: Target of the voltage row.
        power: Target of the power row.
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
        duration: Any,
    ) -> None:
        """Build a segment from exactly one of ``current``, ``voltage``, ``power``.

        Args:
            current: The current to hold: a number, or the coefficients of a
                polynomial in the time since the segment's start, constant
                term first.
            voltage: The voltage to hold, in the same forms.
            power: The power to hold, in the same forms.
            duration: How long to hold it; a positive finite number.

        Raises:
            TypeError: If none or more than one of ``current``, ``voltage`` and
                ``power`` is given, or a value is not a real number or a
                sequence of them (a boolean, a multi-dimensional array, a
                traced value).
            ValueError: If a value is not finite, a sequence of coefficients
                is empty, or the duration is not positive.
        """
        given = tuple(
            (name, value)
            for name, value in (("current", current), ("voltage", voltage), ("power", power))
            if value is not None
        )
        if len(given) != 1:
            received = "none of them" if not given else " and ".join(f"{n}=" for n, _ in given)
            raise TypeError(
                "a Segment holds exactly one mode: pass one of current=, voltage= or power= "
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

    Each segment starts where the one before it ended. Pass the schedule as `schedule=` on `ds.inference.Experiment`.

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
            f"modes must be the model's tuple of mode names, such as {_MODES!r}; got {modes!r}."
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


class _Samples(NamedTuple):
    """The merged per-sample buffers of a scheduled solve."""

    ts: Any
    ys: Any
    lambdas: Any
    valid: Any
    epochs: Any
    event_indices: Any
    event_kinds: Any
    event_directions: Any
    result_codes: Any


class _Carry(NamedTuple):
    """What one segment hands to the next."""

    y: Any
    end: Any
    last_mode: Any
    last_target: Any
    last_end: Any
    failed: Any
    result: Any
    samples: _Samples
    step: Any
    last_terms: Any
    last_start: Any
    history: Any = None


def _may_carry(schedule: ScheduleArrays) -> bool:
    """Whether a segment of ``schedule`` may continue where the one before it ended.

    ``False`` only when the ``carry_step`` leaf is concrete and every entry is
    ``False``; a traced leaf may hold anything.
    """
    flags = schedule.carry_step
    if isinstance(flags, jax.core.Tracer):
        return True
    return bool(np.any(np.asarray(flags)))


def _segment_count(schedule: ScheduleArrays) -> int:
    """The capacity of ``schedule``; refuses leaves of different or empty lengths."""
    shapes = {name: tuple(np.shape(getattr(schedule, name))) for name in _ARRAY_FIELDS}
    first = shapes["mode"]
    if len(first) != 1 or first[0] < 1 or any(shape != first for shape in shapes.values()):
        raise ValueError(
            f"the schedule arrays {_ARRAY_FIELDS} must be one-dimensional and share one "
            f"nonzero length; got shapes {shapes}. Build them with schedule_arrays."
        )
    if schedule.terms is not None:
        terms = tuple(np.shape(schedule.terms))
        if len(terms) != 2 or terms[0] != first[0] or terms[1] < 1:
            raise ValueError(
                f"the target terms of a schedule have shape (capacity, degree) with capacity "
                f"{first[0]} and degree at least 1; got shape {terms}. Build them with "
                "schedule_arrays."
            )
    return first[0]


def _refuse_stop_conditions(schedule: ScheduleArrays) -> None:
    """Refuse a concrete nonzero ``stop_kind``: segments end on their duration only."""
    kinds = schedule.stop_kind
    if isinstance(kinds, jax.core.Tracer):
        return
    nonzero = np.flatnonzero(np.asarray(kinds).reshape(-1) != 0)
    if nonzero.size:
        raise ConfigurationError(
            f"stop_kind is nonzero at segment {int(nonzero[0])}; a segment stops when its "
            "duration has elapsed, and no other stop condition is supported. Set every "
            "stop_kind entry to 0."
        )


def _refuse_unknown_modes(schedule: ScheduleArrays, modes: tuple) -> None:
    """Refuse a concrete active mode index outside the model's mode tuple."""
    if isinstance(schedule.mode, jax.core.Tracer) or isinstance(schedule.active, jax.core.Tracer):
        return
    codes = np.asarray(schedule.mode).reshape(-1)
    active = np.asarray(schedule.active, dtype=bool).reshape(-1)
    outside = np.flatnonzero(active & ((codes < 0) | (codes >= len(modes))))
    if outside.size:
        index = int(outside[0])
        raise ConfigurationError(
            f"segment {index} selects mode index {int(codes[index])}, and the model's "
            f"operational table has the modes {modes!r}, indices 0 to {len(modes) - 1}; "
            "lower the schedule against this model with "
            "schedule_arrays(schedule, problem.operational_modes())."
        )


def _refuse_empty_span(t0: Any, t_end: Any) -> None:
    """Refuse concrete ends of the solve with ``t_end`` not after ``t0``."""
    if isinstance(t0, jax.core.Tracer) or isinstance(t_end, jax.core.Tracer):
        return
    start, stop = np.asarray(t0), np.asarray(t_end)
    if start.shape == () and stop.shape == () and not float(stop) > float(start):
        raise ValueError(
            f"t_end ({float(stop)!r}) must be after t0 ({float(start)!r}): a schedule is "
            "solved over the times after t0 up to t_end."
        )


def _full_state(view: Any, y0: Any) -> Any:
    """``y0`` as the full state, field unknowns then bordered globals.

    A state of the field unknowns alone takes the globals ``pdesolve`` would
    give it: the bordered part of the solver view's own translation of that
    state, which holds the values the model declares.
    """
    n_u, n_lam = int(view.n_u), int(view.n_lam)
    full = n_u + n_lam
    if y0.ndim != 1 or int(y0.shape[0]) not in (n_u, full):
        if n_lam:
            accepted = (
                f"of length {full} (the field unknowns, then the model's {n_lam} bordered "
                f"globals) or of length {n_u} (the field unknowns alone; the globals then "
                "start from the values the model declares)"
            )
        else:
            accepted = f"of length {full}"
        raise ValueError(
            f"the initial state has shape {tuple(y0.shape)}; a schedule starts from a "
            f"one-dimensional state {accepted}."
        )
    if int(y0.shape[0]) == full:
        return y0
    translated = jnp.asarray(view.to_internal_state(y0))
    if translated.shape != (full,):
        raise ValueError(
            f"this model does not give its {n_lam} bordered globals a start of their own; "
            f"pass the initial state of length {full}, the field unknowns then the globals."
        )
    return jnp.concatenate([y0, translated[n_u:].astype(y0.dtype)])


def _field_units(view: Any) -> Optional[tuple]:
    """``(scale, offset)`` that map the solver's field unknowns back to the form of ``y0``.

    ``pdesolve`` hands ``y0`` to the solver view's ``to_internal_state``: a lowered
    problem stores its field unknowns as ``(value - offset) / scale``
    (``LoweredProblem.initial_u``) and takes the bordered globals of a full state as
    they stand; any other problem takes ``y0`` as it stands. The solver's states, the
    terminal one included, stay in the stored units, so the inverse is
    ``offset + scale * stored``. Returns ``None`` when the translation is the identity.

    Raises:
        ConfigurationError: If the problem translates its initial state with a
            conversion that has no inverse here.
    """
    source = getattr(view, "source", None)
    if source is None:
        return None
    if not view.direct_source:
        if callable(getattr(source, "to_internal_state", None)):
            raise ConfigurationError(
                f"a schedule hands each segment the state the previous one ended at, and "
                f"{type(source).__name__} translates the initial state with its own "
                "to_internal_state, which the schedule cannot invert; solve the segments one "
                "by one with pdesolve instead."
            )
        return None
    if not (
        callable(getattr(source, "initial_u", None))
        and callable(getattr(source, "initial_lambda", None))
    ):
        return None
    arrays = getattr(source, "_scale_arrays", None)
    if not callable(arrays):
        raise ConfigurationError(
            f"a schedule hands each segment the state the previous one ended at, and "
            f"{type(source).__name__} stores its field unknowns through initial_u without "
            "the field scales and offsets the schedule needs to map them back; solve the "
            "segments one by one with pdesolve instead."
        )
    # The arrays are constants of the model; evaluate them now even when the
    # schedule is being traced, so the identity can be recognized.
    with jax.ensure_compile_time_eval():
        scale, offset = arrays()
    if jnp.shape(scale) != (int(view.n_u),) or jnp.shape(offset) != (int(view.n_u),):
        raise ConfigurationError(
            f"the field scales of {type(source).__name__} cover {jnp.shape(scale)} and "
            f"{jnp.shape(offset)} entries, not the {int(view.n_u)} field unknowns; the "
            "schedule cannot map the carried state back."
        )
    traced = isinstance(scale, jax.core.Tracer) or isinstance(offset, jax.core.Tracer)
    if not traced and np.all(np.asarray(scale) == 1.0) and np.all(np.asarray(offset) == 0.0):
        return None
    return jnp.asarray(scale), jnp.asarray(offset)


def _configured_step(solver: Any) -> float:
    """The first step ``solver`` is configured with (its ``dt0`` attribute), ``0`` without one.

    Raises:
        TypeError: If the attribute is not a real number.
    """
    value = getattr(solver, "dt0", None)
    return 0.0 if value is None else _as_real(value, "the dt0 of the solver")


def _next_step(solution: Solution, dtype: Any) -> Any:
    """The step ``solution``'s solve would have taken next, ``0`` when its solver reports none."""
    step = getattr(solution.stats, "last_step", None)
    if step is None:
        return jnp.zeros((), dtype=dtype)
    return jax.lax.stop_gradient(jnp.asarray(step).astype(dtype))


def _history_template(solve: Any) -> Optional[SolveHistory]:
    """A history that does not continue, shaped as the one ``solve()`` returns.

    ``solve`` is the first segment's solve asked for its history. It is
    traced for the shapes only, never compiled or run. ``None`` when its
    solution carries no history.
    """
    shapes = jax.eval_shape(lambda: solve().history)
    if shapes is None:
        return None
    zeros = jax.tree_util.tree_map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes)
    return SolveHistory(continues=jnp.zeros((), dtype=bool), state=zeros.state)


def _merge(merged: Any, part: Any, take: Any, count: int) -> Any:
    """``merged`` with the rows where ``take`` holds replaced by ``part``'s."""
    if merged is None or part is None:
        return merged
    part = jnp.asarray(part)[:count]
    mask = jnp.reshape(take, take.shape + (1,) * (part.ndim - 1))
    return jnp.where(mask, part.astype(merged.dtype), merged)


def solve_schedule(
    problem: Any,
    solver: Any,
    schedule: ScheduleArrays,
    y0: Any,
    *,
    t0: Any,
    t_end: Any,
    times: Any,
    theta: Any = None,
    adjoint: Any = None,
    dt0: Any = None,
) -> tuple[Solution, SegmentEnds]:
    """Solve the segments of a schedule one after another and merge their samples.

    Segment ``k`` runs over ``(a_k, b_k]``, where ``a_0 = t0``, each later
    segment starts where the one before it ends, and ``b_k`` is its end cut at
    ``t_end``. A segment runs when it is active, starts before ``t_end`` and no
    earlier segment failed. It is one ``pdesolve`` from the carried state, with
    ``ControlState.operating(mode=..., target=...)`` as its control state, the
    solver's default consistent initialization, and ``times`` as ordered save
    times. The next segment starts from the state the solve ended at,
    ``Solution.terminal``, handed over in the form ``pdesolve`` takes: the solver
    keeps a field declared with a ``scale`` or ``offset`` as
    ``(value - offset) / scale``, so the terminal field unknowns are mapped back
    with ``offset + scale * value``; the bordered globals are taken as they
    stand, as ``pdesolve`` takes them.

    Sample ``i`` belongs to the running segment with ``a_k < times[i] <= b_k``;
    the first segment also owns ``times[i] == t0``. The merged solution takes
    each sample from its segment, in the solver's units as ``pdesolve`` saves
    it; a sample without one, or not valid there, is not valid (``NaN``, epoch
    ``-1``). ``control_epochs`` holds each sample's
    segment index; ``result`` is the result of the segment that failed, success
    otherwise; the counters of ``stats`` add up the segments that ran,
    ``t_reached``, ``terminal`` and ``last_step`` come from the last one,
    ``ic_alg_residual_norm`` is the largest, and ``first_step``, where the
    solver reports it, is the first step of the first segment that ran.

    A segment whose ``carry_step`` entry is set continues the integrator where
    the last segment that ran ended: its solve is given that segment's
    ``Solution.history`` (``history=``), from which it takes the step size,
    the order and the history of past steps. Every other segment starts
    afresh from ``dt0``. The history's shape comes from one trace of the
    first segment's solve, which is not compiled or run. When the solver keeps
    no history, a segment whose entry is set starts instead from the step the
    last segment would have taken next (``SolverStats.last_step``, held out of
    differentiation), and a solve that reports no step (``last_step`` of
    ``0``) leaves nothing to carry. When no segment can continue (a concrete
    ``carry_step`` leaf of ``False`` only), each segment's solve is called as
    without the option.

    A segment with target terms holds
    ``target + terms[0] tau + terms[1] tau**2 + ...`` with ``tau`` the time
    since its start, through ``ControlState.operating(target_terms=...,
    target_origin=<its start>)``; its consistent initialization reads the
    target at ``tau = 0``.

    With more than one segment the segments run inside ``jax.lax.scan``, so the
    program holds one solve whatever the capacity. A segment that does not run
    still executes a short solve whose outputs are discarded and whose work is
    not counted. Inside the scan the solve cannot see a differentiation applied
    around it: reverse mode then differentiates through JAX's transposition of
    the forward rule. Every number may be traced; the function is valid under
    ``jax.jit``, ``jax.vmap``, ``jax.jacfwd``, and ``jax.grad`` with a reverse
    adjoint such as ``CheckpointAdjoint``.

    Args:
        problem: What ``pdesolve`` accepts; its model must declare an
            operational control table.
        solver: The solver, as for ``pdesolve``.
        schedule: The [ScheduleArrays][diff_soup._schedule.ScheduleArrays],
            lowered against this model's mode names.
        y0: The initial state: the field unknowns then the bordered globals, or
            the field unknowns alone, the globals then starting from the values
            the model declares, as ``pdesolve`` treats them.
        t0: The time of the initial state.
        t_end: The last sample time; no segment runs past it.
        times: The sample times, strictly increasing. Entries after ``t_end``
            pad and are not valid.
        theta: The parameters, as for ``pdesolve``.
        adjoint: The differentiation strategy, as for ``pdesolve``.
        dt0: The first step a segment's solve starts from when it carries no
            step, passed to its ``pdesolve`` as ``dt0=``; ``0`` lets the
            integrator estimate it, and it may be traced. ``None`` passes no
            first step, so such a segment starts as the solver does by default;
            when a segment may carry a step, ``None`` passes the first step
            the solver is configured with (its ``dt0`` attribute), or ``0``
            when it names none.

    Returns:
        ``(solution, ends)``: the merged ``Solution`` and the
        [SegmentEnds][diff_soup._schedule.SegmentEnds].

    Raises:
        TypeError: If ``schedule`` is not a ``ScheduleArrays``.
        ValueError: If the arrays differ in length, the initial state has
            neither accepted length, or a concrete ``t_end`` is not after ``t0``.
        ConfigurationError: If the model declares no operational control table,
            a concrete mode index is outside the model's modes, a concrete
            ``stop_kind`` entry is nonzero, or the problem translates its initial
            state in a way the carried state cannot be mapped back through.
    """
    if not isinstance(schedule, ScheduleArrays):
        raise TypeError(
            f"schedule must be a ScheduleArrays; got {type(schedule).__name__}. Lower a "
            "Schedule with schedule_arrays(schedule, problem.operational_modes())."
        )
    count = _segment_count(schedule)
    _refuse_stop_conditions(schedule)
    view = get_or_build_solver_view(problem)
    table = getattr(view, "operational_table", None)
    if table is None:
        raise ConfigurationError(
            "a schedule selects rows of the model's operational control table, and this "
            "model declares none; declare one with spec.control.operational(...)."
        )
    _refuse_unknown_modes(schedule, tuple(table.modes))
    _refuse_empty_span(t0, t_end)
    units = _field_units(view)

    y0 = jnp.asarray(y0)
    # Wide enough to hold the solve's own precision exactly: the carried state
    # and the merged samples round-trip through it without loss.
    dtype = jnp.promote_types(y0.dtype, jax.dtypes.canonicalize_dtype(jnp.float64))
    start_state = _full_state(view, y0.astype(dtype))
    t0 = jnp.asarray(t0, dtype=dtype)
    t_end = jnp.asarray(t_end, dtype=dtype)
    times = jnp.asarray(times, dtype=dtype)
    samples_count = int(times.shape[0])
    n_u, n_lam = int(view.n_u), int(view.n_lam)
    idle = _IDLE_FRACTION * (t_end - t0)
    saveat = SaveAt(ts=times, ordered=True)
    carrying = _may_carry(schedule)
    if carrying:
        base_step = jnp.asarray(_configured_step(solver) if dt0 is None else dt0, dtype=dtype)
    with_terms = schedule.terms is not None
    segments = (
        jnp.arange(count, dtype=jnp.int32),
        jnp.asarray(schedule.mode, dtype=jnp.int32),
        jnp.asarray(schedule.target, dtype=dtype),
        jnp.asarray(schedule.duration, dtype=dtype),
        jnp.asarray(schedule.active, dtype=bool),
        jnp.asarray(schedule.carry_step, dtype=bool) if carrying else None,
        jnp.asarray(schedule.terms, dtype=dtype) if with_terms else None,
    )

    def segment_solve(carry, segment, **extra):
        """Segment ``segment``'s solve from ``carry``, and where it runs."""
        index, mode, target, duration, active, _, terms = segment
        start = carry.end
        end = start + jnp.where(active, duration, jnp.zeros_like(duration))
        stop = jnp.minimum(end, t_end)
        run = active & (start < t_end) & jnp.logical_not(carry.failed)
        varying = {}
        if with_terms:
            # A segment that does not run keeps the last terms and their origin.
            varying = {
                "target_terms": jnp.where(run, terms, carry.last_terms),
                "target_origin": jnp.where(run, start, carry.last_start),
            }
        solution = pdesolve(
            problem,
            solver,
            (jnp.where(run, start, carry.last_end), jnp.where(run, stop, carry.last_end + idle)),
            carry.y,
            saveat=saveat,
            theta=theta,
            adjoint=adjoint,
            control_state0=ControlState.operating(
                mode=jnp.where(run, mode, carry.last_mode),
                target=jnp.where(run, target, carry.last_target),
                **varying,
            ),
            **extra,
        )
        return solution, start, end, stop, run

    def step(carry, segment):
        index, mode, target, _, _, carries, terms = segment
        if with_history:
            # A segment that continues takes the integrator's state from the
            # history the last segment that ran ended with.
            continues = carries & carry.history.continues
            extra = {
                "dt0": base_step,
                "history": SolveHistory(continues=continues, state=carry.history.state),
            }
        elif carrying:
            # A segment that carries a step starts from the last one's next step.
            continues = jnp.zeros((), dtype=bool)
            carried = carries & (carry.step != 0)
            extra = {"dt0": jnp.where(carried, carry.step, base_step)}
        else:
            continues = jnp.zeros((), dtype=bool)
            extra = {} if dt0 is None else {"dt0": dt0}
        solution, start, end, stop, run = segment_solve(carry, segment, **extra)
        solved = jnp.asarray(solution.ok, dtype=bool)
        failed_here = run & jnp.logical_not(solved)
        terminal = solution.terminal
        fields = terminal.y.astype(dtype)
        if units is not None:
            fields = units[1].astype(dtype) + units[0].astype(dtype) * fields
        reached = jnp.concatenate([fields, terminal.lam.astype(dtype)])
        owned = ((start < times) & (times <= stop)) | ((index == 0) & (times == t0))
        valid_there = jnp.asarray(solution.valid_mask, dtype=bool)[:samples_count]
        take = run & owned & valid_there
        merged = carry.samples
        samples = _Samples(
            ts=_merge(merged.ts, solution.ts, take, samples_count),
            ys=_merge(merged.ys, solution.ys, take, samples_count),
            lambdas=_merge(merged.lambdas, solution.lambdas, take, samples_count),
            valid=merged.valid | take,
            epochs=jnp.where(take, index, merged.epochs),
            event_indices=_merge(merged.event_indices, solution.event_indices, take, samples_count),
            event_kinds=_merge(merged.event_kinds, solution.event_kinds, take, samples_count),
            event_directions=_merge(
                merged.event_directions, solution.event_directions, take, samples_count
            ),
            result_codes=_merge(merged.result_codes, solution.result_codes, take, samples_count),
        )
        following = _Carry(
            y=jnp.where(run & solved, reached, carry.y),
            end=end,
            last_mode=jnp.where(run, mode, carry.last_mode),
            last_target=jnp.where(run, target, carry.last_target),
            last_end=jnp.where(run, stop, carry.last_end),
            failed=carry.failed | failed_here,
            result=RESULT.where(failed_here, solution.result, carry.result),
            samples=samples,
            step=(
                jnp.where(run, _next_step(solution, dtype), carry.step)
                if carrying and not with_history
                else None
            ),
            last_terms=jnp.where(run, terms, carry.last_terms) if with_terms else None,
            last_start=jnp.where(run, start, carry.last_start) if with_terms else None,
            history=(
                jax.tree_util.tree_map(
                    lambda new, old: jnp.where(run, new, old), solution.history, carry.history
                )
                if with_history
                else None
            ),
        )
        # The work of a segment that did not run is not counted.
        stats = jax.tree_util.tree_map(
            lambda leaf: jnp.where(run, leaf, jnp.zeros_like(leaf)), solution.stats
        )
        return following, (run, stats, terminal, run & continues)

    def codes():
        return jnp.full((samples_count,), -1, dtype=jnp.int32)

    initial = _Carry(
        y=start_state,
        end=t0,
        last_mode=segments[1][0],
        last_target=segments[2][0],
        last_end=t0,
        failed=jnp.zeros((), dtype=bool),
        result=RESULT.SUCCESS,
        step=jnp.zeros((), dtype=dtype) if carrying else None,
        last_terms=segments[6][0] if with_terms else None,
        last_start=t0 if with_terms else None,
        samples=_Samples(
            ts=jnp.full((samples_count,), jnp.nan, dtype=dtype),
            ys=jnp.full((samples_count, n_u), jnp.nan, dtype=dtype),
            lambdas=None if n_lam == 0 else jnp.full((samples_count, n_lam), jnp.nan, dtype=dtype),
            valid=jnp.zeros((samples_count,), dtype=bool),
            epochs=codes(),
            event_indices=codes(),
            event_kinds=codes(),
            event_directions=codes(),
            result_codes=codes(),
        ),
    )
    # A history to continue needs a later segment that may continue and a
    # solver that keeps one; its shape is the first segment's.
    first_segment = tuple(None if values is None else values[0] for values in segments)
    template = None
    if carrying and count > 1:
        template = _history_template(
            lambda: segment_solve(initial, first_segment, dt0=base_step, history=True)[0]
        )
    with_history = template is not None
    if with_history:
        initial = initial._replace(history=template, step=None)
    if count == 1:
        final, per_segment = step(initial, first_segment)
        per_segment = jax.tree_util.tree_map(lambda leaf: jnp.expand_dims(leaf, 0), per_segment)
    else:
        final, per_segment = jax.lax.scan(step, initial, segments)
    ran, stats, terminals, continued = per_segment

    last = jnp.max(jnp.where(ran, jnp.arange(count, dtype=jnp.int32), -1))
    any_ran = last >= 0
    at = jnp.maximum(last, 0)
    picked = jax.tree_util.tree_map(lambda leaf: leaf[at], terminals)
    unset = TerminalRecord.empty(
        int(picked.y.shape[0]), int(picked.lam.shape[0]), dtype=picked.t.dtype
    )
    terminal = jax.tree_util.tree_map(
        lambda kept, empty: jnp.where(any_ran, kept, empty), picked, unset
    )
    totals = jax.tree_util.tree_map(lambda leaf: jnp.sum(leaf, axis=0, dtype=leaf.dtype), stats)
    totals = eqx.tree_at(
        lambda record: (record.t_reached, record.ic_alg_residual_norm, record.status),
        totals,
        (
            jnp.where(any_ran, stats.t_reached[at], t0.astype(stats.t_reached.dtype)),
            jnp.max(stats.ic_alg_residual_norm, axis=0),
            final.result,
        ),
    )
    if hasattr(totals, "first_step"):
        # A first step is a size, not a count: the first segment that ran gives it.
        first = jnp.argmax(ran)
        totals = eqx.tree_at(lambda record: record.first_step, totals, stats.first_step[first])
    if hasattr(totals, "last_step"):
        # So is the next step: the last segment that ran gives it.
        totals = eqx.tree_at(
            lambda record: record.last_step,
            totals,
            jnp.where(any_ran, stats.last_step[at], jnp.zeros_like(stats.last_step[at])),
        )

    merged = final.samples
    valid = merged.valid
    out_dtype = picked.y.dtype

    def states(values):
        if values is None:
            return None
        mask = jnp.reshape(valid, valid.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, values, jnp.nan).astype(out_dtype)

    def labels(values):
        return jnp.where(valid, values, -1).astype(jnp.int32)

    solution = Solution(
        ts=states(merged.ts),
        ys=states(merged.ys),
        lambdas=states(merged.lambdas),
        valid_mask=valid,
        valid_count=jnp.sum(valid, dtype=jnp.int32),
        control_epochs=labels(merged.epochs),
        event_indices=labels(merged.event_indices),
        event_kinds=labels(merged.event_kinds),
        event_directions=labels(merged.event_directions),
        result_codes=labels(merged.result_codes),
        stats=totals,
        result=final.result,
        terminal=terminal,
    )

    def per_segment(name):
        values = getattr(stats, name, None)
        dtype = terminals.t.dtype
        if values is None:
            return jnp.full(ran.shape, jnp.nan, dtype=dtype)
        return jnp.where(ran, jnp.asarray(values, dtype=dtype), jnp.nan)

    ends = SegmentEnds(
        t_end=jnp.where(ran, terminals.t, jnp.nan),
        ran=ran,
        result_code=jnp.where(ran, terminals.result_code, -1).astype(jnp.int32),
        first_step=per_segment("first_step"),
        last_step=per_segment("last_step"),
        continued=continued,
    )
    return solution, ends
