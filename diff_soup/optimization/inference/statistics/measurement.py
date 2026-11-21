"""User-facing measurements and experiments for ``diff_soup.optimization.inference``."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from diff_soup.optimization.inference.data.deferred import DeferredValues
from diff_soup.optimization.inference.fitting.formulation import Trajectory
from diff_soup.optimization.inference.fitting.inference_problem import InferenceProblem
from diff_soup.optimization.inference.fitting.observables import Observable
from diff_soup.optimization.inference.fitting.parameter_groups import ParameterGroups
from diff_soup.optimization.inference.statistics.experiment import Channel, _CompiledExperiment
from diff_soup.optimization.inference.statistics.likelihoods import Gaussian, Likelihood


@dataclasses.dataclass(frozen=True)
class _ObserveAdapter:
    """Call a user ``observe`` with the keywords its signature accepts.

    Two adapters over the same function compare equal, so experiments that
    share an ``observe`` keep the joint's homogeneous batched kernel.
    """

    observe: Callable | None
    accepts_t: bool
    accepts_theta: bool

    @property
    def is_full_state(self) -> bool:
        """Whether this readout reports the solver state unchanged.

        True when the measurement declared no ``observe=``: the default readout
        is the state itself, so such a measurement can seed observed-state
        windows.
        """
        return self.observe is None

    def __call__(self, state, *, t=None, theta=None):
        """Evaluate the readout on one state."""
        if self.observe is None:
            return state
        if self.accepts_t and self.accepts_theta:
            return self.observe(state, t=t, theta=theta)
        if self.accepts_t:
            return self.observe(state, t=t)
        if self.accepts_theta:
            return self.observe(state, theta=theta)
        return self.observe(state)


@dataclasses.dataclass(frozen=True)
class _GlobalsObservation:
    """Call a user ``observe`` that reads the model's global unknowns, ``observe(state, globals)``.

    A function whose signature names a ``globals`` parameter receives, beside the
    saved fields, the global unknowns of the model at the same time (the
    bordered vector: a cell's terminal current or potential held by an
    operational control), in the order of ``model.global_layout()`` and in the
    solver's units, as the saved fields are. A solve over the span (the lanes of
    a lowered study, or an experiment fitted in one shoot) hands such an
    observation the saved fields and the saved global unknowns one after the
    other, ``fields`` entries then the globals; this readout splits them. Two readouts over the same function and the same
    field count compare equal, so experiments that share one keep one batched
    program.

    Attributes:
        observe: The user's function.
        accepts_t: Whether it takes ``t=``.
        accepts_theta: Whether it takes ``theta=``.
        fields: The number of field unknowns of the model, set by the experiment
            from its model; ``None`` before.
    """

    observe: Callable
    accepts_t: bool
    accepts_theta: bool
    fields: int | None = None

    @property
    def reads_globals(self) -> bool:
        """Always true: the lanes pass this readout the saved global unknowns."""
        return True

    @property
    def is_full_state(self) -> bool:
        """False: the readout is not the solver state."""
        return False

    def bound(self, model: Any, name: str) -> _GlobalsObservation:
        """This readout with the field count of ``model``.

        Raises:
            ValueError: If ``model`` has no global unknowns.
        """
        if int(model.constraint_dim()) == 0:
            raise ValueError(
                f"{name!r} observe= reads globals=, and the model has no global unknowns; "
                "observe a field, or declare the global with spec.global_var(...)."
            )
        return dataclasses.replace(self, fields=int(model.state_size()))

    def __call__(self, state, *, t=None, theta=None):
        """Evaluate the readout on the saved fields followed by the saved global unknowns."""
        state = jnp.asarray(state)
        if self.fields is None or state.shape[-1] <= self.fields:
            raise ValueError(
                "this observation reads globals=, and it was called with the fields alone; "
                "the global unknowns at a sample time are supplied by a solve over the span "
                "(the lanes of ds.lower(study), or one shoot), not by windows of a trajectory."
            )
        keywords = {"globals": state[..., self.fields :]}
        if self.accepts_t:
            keywords["t"] = t
        if self.accepts_theta:
            keywords["theta"] = theta
        return self.observe(state[..., : self.fields], **keywords)


def _bound_observe(observe: Any, model: Any, name: str) -> Any:
    """``observe`` resolved against ``model``: a declared observation's name, or a readout of globals."""
    if isinstance(observe, str):
        return _NamedObservation.of(model, observe, name)
    if isinstance(observe, _GlobalsObservation):
        return observe.bound(model, name)
    return observe


@dataclasses.dataclass(frozen=True, eq=False)
class _NamedObservation:
    """Read one declared observation (``spec.observe.*``) of a model, in dimensional units.

    Two readouts of the same name on models of one structure compare equal, so
    experiments that observe the same declaration keep the joint's batched kernel.
    During a fit the model is rebuilt from the trial theta, so an observation that
    reads a parameter sees the trial value.
    """

    name: str
    model: Any = dataclasses.field(metadata={"fingerprint": False})
    static: PyTree = dataclasses.field(metadata={"fingerprint": False})
    key: Any = dataclasses.field(metadata={"fingerprint": False})

    @classmethod
    def of(cls, model: Any, name: str, measurement: str) -> _NamedObservation:
        """Resolve ``name`` against the observations ``model`` declares."""
        declared = tuple(item.name for item in getattr(model, "observables", ()))
        if name not in declared:
            listed = ", ".join(repr(item) for item in declared) or "none"
            raise ValueError(
                f"Measurement {measurement!r} observe={name!r} names no declared observation "
                f"of the model; it declares {listed}."
            )
        key_of = getattr(model, "structure_key", None)
        key = key_of() if callable(key_of) else None
        static = eqx.partition(model, eqx.is_inexact_array)[1]
        return cls(name, model, static, ("id", id(model)) if key is None else key)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _NamedObservation)
            and self.name == other.name
            and self.key == other.key
        )

    def __hash__(self) -> int:
        return hash((self.name, self.key))

    @property
    def reads_globals(self) -> bool:
        """Whether the declared observation reads a global unknown of the model.

        The lanes then pass it the saved global unknowns after the fields.
        """
        record = next(
            item for item in getattr(self.model, "observables", ()) if item.name == self.name
        )
        return any(read[0] == "global" for read in getattr(record, "reads", ()))

    def __call__(self, state, *, t=None, theta=None):
        """Evaluate the observation on one solver state at time ``t``."""
        model = self.model
        if theta is not None:
            model = eqx.combine(getattr(theta, "inner", theta), self.static)
        state = jnp.asarray(state)
        size = int(model.state_size())
        extra = int(model.constraint_dim())
        lam = None
        if extra and state.shape[-1] == size + extra:
            state, lam = state[..., :size], state[..., size:]
        time = jnp.zeros((), dtype=state.dtype) if t is None else t
        return model.observe(state, time, lam, names=(self.name,))[self.name]


def _observe_adapter(observe: Callable | str | None) -> Callable | str:
    """Accept the friendly ``observe(state)`` and advanced keyword signatures."""
    if observe is None:
        return _ObserveAdapter(None, False, False)
    if isinstance(observe, (_ObserveAdapter, _NamedObservation, _GlobalsObservation)):
        return observe
    if isinstance(observe, str):
        # A declared observation's name, resolved against the model by Experiment.
        if not observe:
            raise ValueError("Measurement observe names a declared observation; got ''.")
        return observe
    if not callable(observe):
        raise TypeError(
            "Measurement observe must be callable, a declared observation's name, or None; "
            f"got {type(observe).__name__}."
        )
    try:
        parameters = inspect.signature(observe).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    names = {parameter.name for parameter in parameters}
    has_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    accepts_t = has_kwargs or "t" in names
    accepts_theta = has_kwargs or "theta" in names
    if "globals" in names:
        # Only a parameter named globals asks for them; **kwargs does not.
        return _GlobalsObservation(observe, accepts_t, accepts_theta)
    if accepts_t and accepts_theta:
        # A fully compatible callable keeps its identity: the joint selects its
        # batched kernel by static observable equality.
        return observe
    return _ObserveAdapter(observe, accepts_t, accepts_theta)


@dataclasses.dataclass(frozen=True)
class Measurement:
    """One stream of observed values, compared with an observed quantity of the solution at the given times.

    A measurement takes `weight=` or `likelihood=`, not both. Without a likelihood the fit is weighted least squares, and the statistics functions that need a noise model refuse it.

    Args:
        name: The label of the measurement in results and diagnostics, unique within an experiment.
        times: The observation times, a one-dimensional array.
        values: The observed values with one row per time, or a function of no arguments returning them, called once when first needed.
        weight: A positive weight on the squared residual, a scalar or an array that broadcasts against `values`.
        likelihood: The noise model, such as [Gaussian][diff_soup.optimization.inference.Gaussian].
        observe: The observed quantity: a function `observe(state)`, optionally taking `t=` and `theta=`, the name of a `spec.observe` declaration, or `None` for the state. A function that also names a `globals` parameter, `observe(state, globals)`, receives the model's global unknowns at the same time, in the order of `model.global_layout()` and in the solver's units, as `state`; a declaration that reads a global receives them too. See [Output][diff_soup.optimization.inference.Output] for the times they are read at.
        mask: One boolean per time, `True` where the time carries a measurement.
        shape: The shape of `values`, checked against them.
        dtype: The dtype of `values`, checked against them.
    """

    name: str
    times: Any
    values: Any
    _: dataclasses.KW_ONLY
    weight: Any = None
    likelihood: Likelihood | None = None
    observe: Callable | str | None = None
    mask: Any = None
    shape: tuple | None = None
    dtype: Any = None
    _deferred: DeferredValues | None = dataclasses.field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self):
        """Normalize arrays and validate the ordinary response-data contract."""
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("Measurement name must be a non-empty string.")
        times = jnp.asarray(self.times)
        if times.ndim != 1:
            raise ValueError(f"Measurement {self.name!r} times must be one-dimensional.")
        deferred = None
        if callable(self.values):
            deferred = DeferredValues(
                self.values,
                name=self.name,
                n_frames=int(times.shape[0]),
                shape=self.shape,
                dtype=self.dtype,
            )
            values = self.values
        else:
            values = jnp.asarray(self.values)
            if values.ndim < 1 or values.shape[0] != times.shape[0]:
                raise ValueError(
                    f"Measurement {self.name!r} values need {times.shape[0]} leading frames; "
                    f"got shape {values.shape}."
                )
            if self.shape is not None and tuple(self.shape) != tuple(values.shape):
                raise ValueError(
                    f"Measurement {self.name!r} declares shape {tuple(self.shape)} but its "
                    f"values have shape {tuple(values.shape)}."
                )
            if self.dtype is not None and np.dtype(self.dtype) != values.dtype:
                raise ValueError(
                    f"Measurement {self.name!r} declares dtype {np.dtype(self.dtype)} but its "
                    f"values have dtype {values.dtype}."
                )
        if self.likelihood is not None and self.weight is not None:
            raise TypeError(
                f"Measurement {self.name!r} accepts weight= or likelihood=, not both: "
                "weight= scales a least-squares residual and likelihood= declares a "
                "noise model that sets the residual scale itself."
            )
        if self.likelihood is not None and not isinstance(self.likelihood, Likelihood):
            raise TypeError(
                f"Measurement likelihood must be a diff_soup likelihood; got "
                f"{type(self.likelihood).__name__}."
            )
        if isinstance(self.likelihood, Gaussian) and not bool(
            np.all(np.asarray(self.likelihood.scale) > 0.0)
        ):
            raise ValueError("Gaussian measurement scale must be strictly positive.")
        weight = self.weight
        if weight is not None:
            weight = jnp.asarray(weight)
            host = np.asarray(weight)
            if not bool(np.all(np.isfinite(host)) and np.all(host > 0.0)):
                raise ValueError(
                    f"Measurement {self.name!r} weight must be finite and strictly "
                    "positive; it multiplies the squared residual of each entry."
                )
            if deferred is None:
                try:
                    np.broadcast_shapes(host.shape, tuple(values.shape))
                except ValueError as exc:
                    raise ValueError(
                        f"Measurement {self.name!r} weight has shape {host.shape}, which "
                        f"does not broadcast against its values of shape "
                        f"{tuple(values.shape)}."
                    ) from exc
        mask = self.mask
        if mask is not None:
            mask = jnp.asarray(mask)
            if mask.shape != times.shape or mask.dtype != jnp.bool_:
                raise ValueError(
                    f"Measurement {self.name!r} mask must be boolean with shape {times.shape}."
                )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "observe", _observe_adapter(self.observe))
        if deferred is None:
            object.__setattr__(self, "shape", tuple(values.shape))
            object.__setattr__(self, "dtype", np.dtype(values.dtype))
        else:
            object.__setattr__(self, "shape", None if self.shape is None else tuple(self.shape))
            object.__setattr__(self, "dtype", None if self.dtype is None else np.dtype(self.dtype))
        object.__setattr__(self, "_deferred", deferred)

    @property
    def deferred(self) -> bool:
        """Whether ``values`` is a callable that loads on the owning process."""
        return self._deferred is not None

    def _data(self) -> Array | DeferredValues:
        """The value passed to the compiled experiment: an array or its deferred loader."""
        return self.values if self._deferred is None else self._deferred

    def values_shape(self) -> tuple:
        """Shape of the observed values: the array's, or the declared or probed shape."""
        if self._deferred is None:
            return tuple(self.values.shape)
        return tuple(self._deferred.metadata()[0])

    def _residual_scaling(self) -> Likelihood:
        """The scaling this measurement applies to its residual rows.

        A declared likelihood whitens its own rows. Without one the rows are
        ``sqrt(weight) * (values - prediction)``, which is the scaling of a
        Gaussian of standard deviation ``1 / sqrt(weight)``; an omitted weight
        is one, so the rows are the plain residual.
        """
        if self.likelihood is not None:
            return self.likelihood
        if self.weight is None:
            return Gaussian(sigma=1.0)
        return Gaussian(scale=1.0 / jnp.sqrt(self.weight))

    def _channel(self) -> Channel:
        """Lower to the existing prediction/least-squares channel primitive."""
        return Channel(
            name=self.name,
            saveat=self.times,
            observable=Observable(
                self.name,
                self.observe,
                wrt_theta=True,
            ),
            likelihood=self._residual_scaling(),
            role="response",
            mask=self.mask,
        )


@dataclasses.dataclass(frozen=True)
class Output:
    """A series an experiment produces at given times, for other experiments to be driven by.

    An output is read from the solution as a measurement is, and carries no data and no likelihood. An experiment declares it in `outputs=`; another experiment reads it through a [Link][diff_soup.optimization.inference.Link].

    An output or a measurement may read the global unknowns of the model, such as a cell's terminal current or potential held by `spec.control.operational`: its function names a `globals` parameter, or it names a declaration that reads a global. In a solve over the span (the lanes of `ds.lower(study)`, or an experiment fitted in one shoot) it then receives the global unknowns the solver saved at each sample time, from the same dense output of the accepted step as the fields, with their derivative in every mode. On a lane that follows a schedule, a sample at the time a segment ends belongs to that segment: it reads the value the segment reached there, before the next segment's operating condition applies.

    Args:
        name: The label of the output, unique within an experiment.
        times: The times of the series, a one-dimensional increasing array inside the experiment's `t_span`.
        observe: The produced quantity: a function `observe(state)`, optionally taking `t=` and `theta=`, or `observe(state, globals)`; the name of a `spec.observe` declaration; or `None` for the state.
    """

    name: str
    times: Any
    observe: Callable | str | None = None

    def __post_init__(self):
        """Normalize the times and the readout."""
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("Output name must be a non-empty string.")
        times = np.asarray(self.times, dtype=np.float64)
        if times.ndim != 1 or times.shape[0] < 1:
            raise ValueError(
                f"Output {self.name!r} times must be a one-dimensional array of at least one time."
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError(f"Output {self.name!r} times must increase strictly.")
        object.__setattr__(self, "times", jnp.asarray(times))
        object.__setattr__(self, "observe", _observe_adapter(self.observe))


@dataclasses.dataclass(frozen=True)
class Link:
    """A fixed signal of an experiment's model, given its values by another experiment's output.

    The driven experiment declares it in `links=`. At every point of a fit the source experiment is solved first, and the values of its output become the values of the signal `into`; the derivative of the driven experiment's residual carries the source's parameters through them. The signal is a function the driven model declares with `spec.function(into, signal, learnable=False)` whose `values` array has one entry per time of the output. Its knots stay as declared, so the caller maps the output's times onto the driven experiment's clock.

    A study with links is fitted on its lanes, `ds.lower(study)` with [TrustRegion][diff_soup.optimization.inference.TrustRegion]. The parameters that move the source's output are shared parameters of the study, declared by every experiment: a model that does not read one declares it inert. A source declares no links itself.

    Args:
        source: The id of the experiment that produces the output.
        output: The name of the source's [Output][diff_soup.optimization.inference.Output].
        into: The name of the driven model's fixed signal function that takes the output's values.
    """

    source: Hashable
    output: str
    into: str

    def __post_init__(self):
        """Check the declaration on its own; the study checks it against its experiments."""
        try:
            hash(self.source)
        except TypeError as exc:
            raise TypeError("Link source must be a hashable experiment id.") from exc
        for name in ("output", "into"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"Link {name} must be a non-empty string; got {value!r}.")


def _signal_values(declaration: Any) -> Any:
    """The ``values`` array of a declared signal function, or ``None`` when it holds none."""
    values = getattr(getattr(declaration, "fn", None), "values", None)
    if values is None or not hasattr(values, "shape") or not hasattr(values, "dtype"):
        return None
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        return None
    return values


def _check_links(experiments: tuple, experiment_ids: tuple) -> None:
    """Refuse links that do not fit a study's experiments.

    A link names another experiment of the study, an output that experiment
    declares, and a fixed signal of the driven model whose values have one
    entry per time of the output. A source declares no links, so links have
    one level and no cycle.
    """
    by_id = dict(zip(experiment_ids, experiments))
    for experiment, experiment_id in zip(experiments, experiment_ids):
        for link in experiment.links:
            named = f"experiment {experiment_id!r} links its signal {link.into!r}"
            if link.source == experiment_id:
                raise ValueError(
                    f"{named} to its own output; a link reads the output of another experiment "
                    "of the study."
                )
            source = by_id.get(link.source)
            if source is None:
                raise ValueError(
                    f"{named} to experiment {link.source!r}, which is not in the study; its "
                    f"experiments are {list(experiment_ids)}."
                )
            outputs = {output.name: output for output in source.outputs}
            output = outputs.get(link.output)
            if output is None:
                raise ValueError(
                    f"{named} to the output {link.output!r} of experiment {link.source!r}, which "
                    f"declares the outputs {sorted(outputs)}."
                )
            if source.links:
                raise ValueError(
                    f"experiment {link.source!r} drives experiment {experiment_id!r} and is "
                    "driven by a link itself; links have one level: a source declares no links."
                )
            values = _signal_values(experiment._declarations[link.into])
            count = int(np.shape(output.times)[0])
            if values.ndim < 1 or int(values.shape[0]) != count:
                raise ValueError(
                    f"{named} to the output {link.output!r} of experiment {link.source!r}, which "
                    f"has {count} times, and the signal's values have the shape "
                    f"{tuple(values.shape)}; give the signal one value per time of the output."
                )


def _check_scale_input(name: str, value: ArrayLike | Sequence) -> None:
    """Refuse a grid scale (``spec.geometry``) given a size that is not finite and positive."""
    import jax

    try:
        size = np.asarray(value, dtype=float)
    except jax.errors.TracerArrayConversionError:
        return  # a traced value has no size to check on the host
    if size.shape != () or not np.isfinite(size) or not size > 0.0:
        raise ValueError(
            f"Experiment inputs {name!r}={value!r}: {name!r} is the size of a grid "
            "declared with spec.geometry, which multiplies the grid's coordinates, so "
            "it must be one finite positive number."
        )


def _check_schedule(schedule: Any, model: Any, t_span: tuple, trajectory: Any) -> None:
    """Refuse a schedule the model cannot follow over ``t_span`` in one shoot."""
    from diff_soup._errors import ConfigurationError
    from diff_soup._schedule import Schedule, schedule_arrays
    from diff_soup.optimization.inference.fitting.formulation import SingleShoot

    if not isinstance(schedule, Schedule):
        raise TypeError(
            "Experiment schedule must be a ds.Schedule of ds.Segment entries, or None; got "
            f"{type(schedule).__name__}."
        )
    table_of = getattr(model, "operational_table", None)
    table = table_of() if callable(table_of) else None
    if table is None:
        raise ConfigurationError(
            "Experiment schedule selects rows of the model's operational control table, and "
            "the model declares none; declare one with spec.control.operational(...), or "
            "leave schedule=None."
        )
    schedule_arrays(schedule, tuple(table.modes))
    t0, t1 = t_span
    covered = t0 + schedule.duration
    if t1 - covered > 1e-12 * max(abs(t0), abs(t1), t1 - t0):
        raise ValueError(
            f"Experiment schedule lasts {schedule.duration!r}, which ends before the span "
            f"t_span={tuple(t_span)!r} of length {t1 - t0!r}; give the schedule segments "
            "whose durations add up to at least the length of the span."
        )
    if trajectory is not None and not isinstance(trajectory, SingleShoot):
        raise ValueError(
            f"Experiment schedule is solved in a single shoot over t_span, and trajectory="
            f"{type(trajectory).__name__}(...) splits the span into windows; leave "
            "trajectory=None or pass SingleShoot() with a schedule."
        )


def _checked_members(members: Any) -> dict:
    """Validate an experiment's members of the levels of a hierarchy; return a new dict."""
    if not isinstance(members, Mapping):
        raise TypeError(
            "Experiment parameters= maps each level of the study's hierarchy to this "
            "experiment's member of it, for example {'cell': 'c1', 'particle': 'p1'}; got "
            f"{type(members).__name__}."
        )
    for key, value in members.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"Experiment parameters= takes level names as non-empty strings; got {key!r}."
            )
        if key == "experiment":
            raise ValueError(
                "Experiment parameters= names a member of 'experiment'; every experiment is its "
                "own member of that level, under its id, so leave 'experiment' out."
            )
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError(
                f"Experiment parameters= gives level {key!r} the member {value!r}; a member is "
                "a string or an integer label. A fixed value of a parameter goes in inputs=."
            )
    return dict(members)


def _checked_links(experiment: Any, model: Any, declarations: dict, t_span: tuple) -> tuple:
    """An experiment's outputs and links, checked against its model; ``(outputs, links)``.

    An output lies inside ``t_span`` and a declared observation's name is
    resolved against ``model``. A link names a fixed signal function the model
    declares, holding a ``values`` array, once. An experiment with outputs or
    links is solved in one shoot.
    """
    from diff_soup.optimization.inference.fitting.formulation import SingleShoot

    outputs, links = experiment.outputs, experiment.links
    if isinstance(outputs, Output) or isinstance(links, Link):
        raise TypeError("Experiment outputs= and links= are tuples; write (item,) for one.")
    outputs, links = tuple(outputs), tuple(links)
    if not all(isinstance(item, Output) for item in outputs):
        raise TypeError("Experiment outputs must be a tuple of Output objects.")
    if not all(isinstance(item, Link) for item in links):
        raise TypeError("Experiment links must be a tuple of Link objects.")
    names = [output.name for output in outputs]
    if len(set(names)) != len(names):
        raise ValueError(f"Experiment output names must be unique; got {names}.")
    targets = [link.into for link in links]
    if len(set(targets)) != len(targets):
        raise ValueError(f"Experiment links each fill one signal once; got the signals {targets}.")
    t0, t1 = t_span
    for output in outputs:
        times = np.asarray(output.times)
        if np.any(times < t0) or np.any(times > t1):
            raise ValueError(f"Output {output.name!r} contains times outside t_span {t_span}.")
    functions = sorted(name for name, handle in declarations.items() if handle.kind == "function")
    for link in links:
        declaration = declarations.get(link.into)
        if declaration is None or declaration.kind != "function":
            raise ValueError(
                f"Link into={link.into!r} names no function of the model; it declares the "
                f"functions {functions}."
            )
        if declaration.learnable:
            raise ValueError(
                f"Link into={link.into!r} names a learnable function; a link fills a fixed "
                "signal, declared with spec.function(..., learnable=False)."
            )
        if _signal_values(declaration) is None:
            raise ValueError(
                f"Link into={link.into!r} names a function that holds no floating-point values "
                "array; a link fills the values of a signal such as approx.PiecewiseLinear."
            )
    trajectory = experiment.trajectory
    if (outputs or links) and trajectory is not None and not isinstance(trajectory, SingleShoot):
        raise ValueError(
            f"Experiment outputs= and links= are read from one solve over t_span, and "
            f"trajectory={type(trajectory).__name__}(...) splits the span into windows; leave "
            "trajectory=None or pass SingleShoot()."
        )
    outputs = tuple(
        dataclasses.replace(output, observe=_bound_observe(output.observe, model, output.name))
        if isinstance(output.observe, (str, _GlobalsObservation))
        else output
        for output in outputs
    )
    return outputs, links


def _check_computed_start(initial_state: Any, trajectory: Any) -> None:
    """Refuse a start computed from the parameters where it cannot act.

    A [ComputedInitialState][diff_soup.optimization.inference.ComputedInitialState] starts
    one solve over the span. Windows of a multiple-shooting trajectory start from their
    own states, measured or fitted, so the computed start would not reach them.

    Raises:
        ValueError: If the start is computed and ``trajectory`` is a ``Multishoot``.
    """
    from diff_soup.optimization.inference.fitting.formulation import (
        ComputedInitialState,
        Multishoot,
    )

    if isinstance(initial_state, ComputedInitialState) and isinstance(trajectory, Multishoot):
        raise ValueError(
            "Experiment initial_state=ComputedInitialState(...) starts one solve over "
            "t_span, and trajectory=Multishoot(...) starts each window from its own state; "
            "leave trajectory=None or pass SingleShoot()."
        )


@dataclasses.dataclass(frozen=True)
class Experiment:
    """One solve of a model, compared against its measurements.

    Pass one or several experiments to an [InferenceProblem][diff_soup.optimization.inference.InferenceProblem] or a [Study][diff_soup.optimization.inference.Study]. Every measurement time lies inside `t_span`.

    Args:
        model: The built problem, or a Recipe template built with the template arguments in `inputs`.
        solver: The time-stepping solver used for every solve of the fit.
        t_span: The interval `(t0, t1)` of the solve, with `t0 < t1`.
        initial_state: The state the solve starts from at `t0`: an array, a [FixedInitialState][diff_soup.optimization.inference.FixedInitialState], a [LearnedInitialState][diff_soup.optimization.inference.LearnedInitialState] fitted with the parameters, or a [ComputedInitialState][diff_soup.optimization.inference.ComputedInitialState] computed from them at every evaluation.
        measurements: A non-empty tuple of [Measurement][diff_soup.optimization.inference.Measurement] with distinct names.
        inputs: Template arguments and fixed parameter values for this experiment alone, by name.
        id: A hashable name for the experiment, or `None` to build one from `inputs` or, without inputs, to number the experiment.
        trajectory: [SingleShoot][diff_soup.optimization.inference.SingleShoot] or [Multishoot][diff_soup.optimization.inference.Multishoot], or `None` for one solve over `t_span`.
        schedule: An operating protocol the solve follows, a `ds.Schedule`.
        parameters: A mapping from each level of the study's [Hierarchy][diff_soup.optimization.inference.Hierarchy] to this experiment's member of it.
        outputs: A tuple of [Output][diff_soup.optimization.inference.Output] series, with distinct names, that other experiments of a study are driven by.
        links: A tuple of [Link][diff_soup.optimization.inference.Link], each giving a fixed signal of the model the values of another experiment's output.
    """

    model: Any
    solver: Any
    t_span: tuple[float, float]
    initial_state: PyTree
    measurements: tuple[Measurement, ...]
    inputs: Mapping[str, object] = dataclasses.field(default_factory=dict)
    id: Hashable | None = None
    trajectory: Trajectory | None = None
    schedule: Any = None
    # A mapping from level name to member, or None; annotated Any so that a
    # wrong value reaches the refusal that explains the field.
    parameters: Any = None
    outputs: tuple = ()
    links: tuple = ()
    _delegate: _CompiledExperiment = dataclasses.field(init=False, repr=False, compare=False)
    _fixed_inputs: dict[str, object] = dataclasses.field(init=False, repr=False, compare=False)
    _declarations: dict[str, Any] = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate the declaration and prepare the existing prediction lowering."""
        measurements = tuple(self.measurements)
        if not measurements or not all(isinstance(item, Measurement) for item in measurements):
            raise TypeError(
                "Experiment measurements must be a non-empty tuple of Measurement objects."
            )
        names = [measurement.name for measurement in measurements]
        if len(set(names)) != len(names):
            raise ValueError(f"Experiment measurement names must be unique; got {names}.")
        t0, t1 = (float(self.t_span[0]), float(self.t_span[1]))
        if not t0 < t1:
            raise ValueError("Experiment t_span must satisfy t0 < t1.")
        for measurement in measurements:
            times = np.asarray(measurement.times)
            if np.any(times < t0) or np.any(times > t1):
                raise ValueError(
                    f"Measurement {measurement.name!r} contains times outside t_span {self.t_span}."
                )
        if self.trajectory is not None and not isinstance(self.trajectory, Trajectory):
            raise TypeError(
                "Experiment trajectory must be a diff_soup.optimization.inference Trajectory "
                "(SingleShoot or Multishoot), or None for a single shoot over t_span; "
                f"got {type(self.trajectory).__name__}."
            )
        _check_computed_start(self.initial_state, self.trajectory)
        if not isinstance(self.inputs, Mapping):
            raise TypeError("Experiment inputs must be a mapping of named conditioning inputs.")
        if self.parameters is not None:
            object.__setattr__(self, "parameters", _checked_members(self.parameters))
        from diff_soup.problem._registry import parameter_handles
        from diff_soup.problem._spec.recipe import RecipeTemplate

        model = self.model
        argument_names = model.argument_names if isinstance(model, RecipeTemplate) else ()
        author_inputs = {
            name: value for name, value in self.inputs.items() if name in argument_names
        }
        if isinstance(model, RecipeTemplate):
            model = model.build(**author_inputs)
        declarations = {
            handle.name: handle
            for handles in parameter_handles(model).values()
            for handle in handles
            if type(handle).__name__ != "_TemplateHandle"
        }
        declarations.update(
            {handle.name: handle for handle in getattr(model, "_recipe_declarations", ())}
        )
        fixed = {name: value for name, value in self.inputs.items() if name not in argument_names}
        parameter_names = {
            name for name, handle in declarations.items() if handle.kind in ("scalar", "array")
        }
        unknown = sorted(set(fixed) - parameter_names)
        if unknown:
            raise ValueError(
                f"Experiment inputs {unknown} are unknown; template arguments: {list(argument_names)}; declared parameters: {sorted(parameter_names)}"
            )
        scale_names = tuple(getattr(model, "_geometry_scales", ()))
        for name in sorted(set(fixed) & set(scale_names)):
            _check_scale_input(name, fixed[name])
        if fixed:
            from diff_soup.problem import default_theta
            from diff_soup.problem._ir.lower import with_theta

            theta = default_theta(
                model, {declarations[name]: value for name, value in fixed.items()}
            )
            # The model holding these fixed values: its own object, the same one
            # for equal fixed inputs, carrying the model's bounds, declarations
            # and handles; the recipe's build is left as it was.
            model = with_theta(model, theta.inner)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "_fixed_inputs", fixed)
        object.__setattr__(self, "_declarations", declarations)
        if self.id is None and self.inputs:
            object.__setattr__(
                self,
                "id",
                ", ".join(f"{name}={value!r}" for name, value in sorted(self.inputs.items())),
            )
        if self.id is not None:
            try:
                hash(self.id)
            except TypeError as exc:
                raise TypeError("Experiment id must be hashable.") from exc
        if self.schedule is not None:
            _check_schedule(self.schedule, model, (t0, t1), self.trajectory)

        measurements = tuple(
            dataclasses.replace(
                measurement,
                observe=_bound_observe(measurement.observe, model, measurement.name),
            )
            if isinstance(measurement.observe, (str, _GlobalsObservation))
            else measurement
            for measurement in measurements
        )
        outputs, links = _checked_links(self, model, declarations, (t0, t1))
        channels = tuple(measurement._channel() for measurement in measurements)
        delegate = _CompiledExperiment(
            problem=self.model,
            solver=self.solver,
            t_span=(t0, t1),
            initial_state=self.initial_state,
            channels=channels,
            controls={},
            data={measurement.name: measurement._data() for measurement in measurements},
            schedule=self.schedule,
        )
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "t_span", (t0, t1))
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "_delegate", delegate)

    @property
    def experiment_id(self) -> Hashable | None:
        """Stable Study identifier, with a deterministic single-object fallback."""
        return self.id

    def as_inference_problem(
        self,
        *,
        parameters: ParameterGroups | None = None,
        objective: Any = None,
        trajectory: Trajectory | None = None,
        regularizer: Any = None,
        bounds: Any = None,
        t0: float | None = None,
        load_data: bool = True,
        reuse: InferenceProblem | None = None,
        state_prior: Any = None,
    ) -> InferenceProblem:
        """Lower this declaration through its private optimization delegate.

        Args:
            parameters: Optional ``ParameterGroups`` choosing the learnable
                theta; defaults to the problem's declared learnables.
            objective: Optional objective, defaulting to least squares.
            trajectory: Overrides the declared ``trajectory`` for this lowering.
                Omit it to use the declaration's own, which is the form a Study
                lowers with.
            regularizer: Optional extra least-squares rows over the parameters.
            bounds: Optional named box bounds over the parameter groups.
            t0: Optional integration start, defaulting to ``t_span[0]``.
            load_data: Whether deferred measurement values are loaded here.
                ``False`` lowers over zero placeholders of the declared shape,
                which a Study uses on the processes that do not own this
                experiment under a multi-process mesh.
            reuse: A problem this declaration lowered earlier over
                placeholders; its lowering is kept and the residuals are
                rebuilt over the values loaded now.
            state_prior: Optional Gaussian prior on the latent window states
                of a decision-variable multiple-shooting trajectory (the
                objective's ``MAP(state_prior=...)``).

        Returns:
            An ordinary :class:`~diff_soup.optimization.inference.InferenceProblem`.
        """
        return self._delegate.as_inference_problem(
            parameters=parameters,
            objective=objective,
            trajectory=self.trajectory if trajectory is None else trajectory,
            regularizer=regularizer,
            bounds=bounds,
            t0=t0,
            load_data=load_data,
            reuse=reuse,
            state_prior=state_prior,
        )

    def predictor(self, *, parameters: ParameterGroups | None = None) -> Callable:
        """Return the shared, JAX-transformable prediction closure."""
        return self._delegate.predictor(parameters=parameters)

    @property
    def channels(self) -> tuple:
        """Measurement channels consumed by diagnostics and prediction verbs."""
        return self._delegate.channels

    @property
    def data(self) -> dict:
        """Observed response data keyed by measurement name.

        Deferred values are loaded here, once, on the calling process.
        """
        return {
            name: values.materialize() if isinstance(values, DeferredValues) else values
            for name, values in self._delegate.data.items()
        }


def measurements_without_likelihood(source: Any) -> tuple[str, ...]:
    """Name the measurements of a declaration that carry no noise model.

    Args:
        source: An estimation problem, a
            [Study][diff_soup.optimization.inference.Study], or a single
            [Experiment][diff_soup.optimization.inference.Experiment].

    Returns:
        The names, in declaration order, of every measurement built without
        ``likelihood=``.
    """
    study = getattr(source, "study", source)
    experiments = getattr(study, "experiments", (study,))
    return tuple(
        measurement.name
        for experiment in experiments
        for measurement in getattr(experiment, "measurements", ())
        if measurement.likelihood is None
    )


def require_likelihoods(source: Any, verb: str) -> None:
    """Refuse a statistical verb on measurements declared without a noise model.

    Args:
        source: An estimation problem, Study, or Experiment.
        verb: The name of the verb, quoted in the error message.

    Raises:
        ValueError: If any of the declaration's measurements has no
            ``likelihood=``.
    """
    names = measurements_without_likelihood(source)
    if not names:
        return
    listed = ", ".join(repr(name) for name in dict.fromkeys(names))
    raise ValueError(
        f"{verb} needs a noise model on every measurement; none is declared for "
        f"{listed}. Pass likelihood= on those measurements, for example "
        "Measurement(..., likelihood=Gaussian(sigma=...))."
    )


__all__ = ["Measurement", "Experiment", "Output", "Link"]
