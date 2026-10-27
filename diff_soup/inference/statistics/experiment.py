"""Private compiled experiment machinery for [diff_soup.inference][].

The public declaration is [diff_soup.inference.Experiment][]. Its
private `_CompiledExperiment` delegate is a thin immutable lowering:
one problem, one
solver, one trajectory window, and a set of measurement `Channel`s,
each carrying its observable, its likelihood, and its provenance role. The
declaration never optimizes anything — it lowers:

* `Experiment.predictor` returns the one pure prediction closure
  (a single shared solve over the union of every channel's times), usable at
  the design stage with no data attached;
* `Experiment.with_data` attaches observed values immutably;
* `Experiment.as_inference_problem` lowers the data-bearing declaration to
  the structural [InferenceProblem][diff_soup.inference.fitting.inference_problem.InferenceProblem],
  which ``optsolve`` solves — the experiment never grows its own
  optimizer.
* `ExperimentSet` groups declarations under stable IDs and explicit
  owner metadata. It lowers through the existing
  [JointInferenceProblem][diff_soup.inference.fitting.joint.JointInferenceProblem] engine and adds
  information/log likelihoods only when cross-experiment independence was
  declared (or an explicit joint-likelihood protocol was supplied).

Provenance is explicit: every channel declares a ``role``
(``initial_condition``, ``control``, ``boundary``, ``response``, or
``reference``) and an optional stable ``source_id``. Only response channels
are ever scored. A measured input that also appears as a response (the same
``source_id`` on both a conditioning and a response channel) is an error
unless deliberately acknowledged, and every acknowledgment is recorded.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup.inference.data.base import ObservationOperator
from diff_soup.inference.data.deferred import DeferredValues
from diff_soup.inference.fitting.observables import Observable, _ObservableOperator
from diff_soup.inference.fitting.series import Series
from diff_soup.inference.statistics.likelihoods import (
    CorrelatedGaussian,
    Gaussian,
    Likelihood,
    LogNormal,
    MultiplicativeGaussian,
    Poisson,
    StudentT,
    _MultiplicativeDeviance,
)

__all__ = ["Channel", "ExperimentSet", "Predictor", "CHANNEL_ROLES"]


#: The provenance roles a channel may declare. Only ``response`` channels are
#: scored by a fit; the others are conditioning inputs or comparison data.
CHANNEL_ROLES = ("initial_condition", "control", "boundary", "response", "reference")


class Channel(eqx.Module):
    """One measurement channel of an experiment. Frozen.

    Attributes:
        name: Unique channel name within the experiment.
        saveat: Observation times, shape ``(n_frames,)``.
        observable: The `Observable` mapping solver state (and
            optionally time/theta) to channel space.
        likelihood: The channel's normalized observation model.
        kernel: Optional per-frame instrument map, composed after the
            observable and before the likelihood. A dense matrix acts on the
            flattened observable value; an object with a JAX-traceable
            ``matvec`` receives the flattened value; a differentiable callable
            receives its native shape.
        context: Optional metadata passed to likelihood methods (for example a
            fixed histogram total).
        role: Provenance role, one of `CHANNEL_ROLES`. Only
            ``"response"`` channels contribute residual rows to a fit.
        source_id: Optional stable identifier of the underlying measured
            record, used to detect the same datum entering the analysis
            through two channels.
        mask: Optional ``(n_frames,)`` boolean frame mask; masked-out frames
            are excluded from prediction rows and residuals (row selection,
            never NaN arithmetic).
    """

    name: str = eqx.field(static=True)
    saveat: Array
    observable: Observable
    likelihood: Likelihood
    role: str = eqx.field(static=True, default="response")
    source_id: Optional[str] = eqx.field(static=True, default=None)
    mask: Optional[Array] = None
    kernel: Any = None
    context: Any = None

    def __post_init__(self):
        """Validate the declared role and observation-kernel protocol."""
        if self.role not in CHANNEL_ROLES:
            raise ValueError(
                f"Channel {self.name!r}: role must be one of {CHANNEL_ROLES}; got {self.role!r}."
            )
        saveat = jnp.asarray(self.saveat)
        if saveat.ndim != 1:
            raise ValueError(
                f"Channel {self.name!r}: saveat must have shape (n_frames,); got {saveat.shape}."
            )
        object.__setattr__(self, "saveat", saveat)
        if self.mask is not None:
            mask = jnp.asarray(self.mask)
            if mask.shape != saveat.shape:
                raise ValueError(
                    f"Channel {self.name!r}: mask must have shape {saveat.shape} "
                    f"to match saveat; got {mask.shape}."
                )
            if mask.dtype != jnp.bool_:
                raise ValueError(
                    f"Channel {self.name!r}: mask must be boolean; got dtype {mask.dtype}."
                )
            object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "kernel", _validate_kernel(self.name, self.kernel))


def _validate_kernel(channel: str, kernel: Any) -> Any:
    """Validate a kernel and normalize dense matrices to JAX arrays."""
    if kernel is None:
        return None
    if hasattr(kernel, "matvec"):
        if callable(getattr(kernel, "matvec")):
            return kernel
        raise TypeError(
            f"Channel {channel!r} kernel matvec attribute must be callable; "
            f"got {type(getattr(kernel, 'matvec')).__name__}."
        )
    if callable(kernel):
        return kernel
    try:
        matrix = jnp.asarray(kernel)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Channel {channel!r} kernel must be a dense 2-D matrix, an object "
            "with matvec, a differentiable callable, or None; got "
            f"{type(kernel).__name__}."
        ) from exc
    if matrix.ndim != 2:
        raise TypeError(
            f"Channel {channel!r} kernel must be a dense 2-D matrix, an object "
            f"with matvec, a differentiable callable, or None; got an array "
            f"with {matrix.ndim} dimensions."
        )
    return matrix


def _channel_likelihood_view(channel: Channel, predicted: Array) -> tuple:
    """Return likelihood/context aligned with a channel's kept predictions."""
    mask = getattr(channel, "mask", None)
    context = getattr(channel, "context", None)
    if mask is None:
        return channel.likelihood, context
    keep = np.flatnonzero(np.asarray(mask, dtype=bool))
    predicted = jnp.asarray(predicted)
    if predicted.ndim < 1 or predicted.shape[0] != keep.size:
        raise ValueError(
            f"Channel {channel.name!r}: post-mask prediction must have "
            f"{keep.size} leading frames; got shape {predicted.shape}."
        )
    full_shape = (int(channel.saveat.shape[0]),) + tuple(predicted.shape[1:])
    return channel.likelihood.for_frames(
        jnp.asarray(keep),
        full_prediction_shape=full_shape,
        context=context,
    )


def _apply_kernel(kernel: Any, value: Array) -> Array:
    """Apply one channel kernel to one native-shape observable frame."""
    if kernel is None:
        return value
    if callable(getattr(kernel, "matvec", None)):
        return jnp.asarray(kernel.matvec(jnp.ravel(value)))
    if callable(kernel):
        return jnp.asarray(kernel(value))
    return jnp.asarray(kernel) @ jnp.ravel(value)


class _ChannelOperator(ObservationOperator):
    """One shared observable→kernel operator for prediction and fitting."""

    observable: _ObservableOperator
    kernel: Any

    @property
    def is_full_state(self) -> bool:
        """Whether the channel reports the solver state unchanged.

        True for the default readout with no instrument kernel, which is what
        lets such a measurement seed observed-state windows.
        """
        return self.kernel is None and self.observable.is_full_state

    def predict(self, u: Array, t: Any, theta: Any) -> Array:
        """Thread time/theta through the observable, then apply the kernel."""
        return _apply_kernel(self.kernel, self.observable.predict(u, t, theta))


def _channel_operator(channel: Channel) -> _ChannelOperator:
    """Build the per-frame observation operator of a channel."""
    return _ChannelOperator(
        observable=_ObservableOperator(
            fn=channel.observable.fn, wrt_theta=channel.observable.wrt_theta
        ),
        kernel=channel.kernel,
    )


class _PredictionStream:
    """Values-free stream protocol adapter for one channel.

    Exposes exactly what the shared prediction factory needs — times, frame
    mask, and an ``observe`` that dispatches through the same operator a data
    fit uses — with ``values``/``weight`` absent, so design-stage prediction
    runs the identical code path.
    """

    def __init__(self, channel: Channel):
        self.name = channel.name
        self.ts = channel.saveat
        self.values = None
        self.weight = None
        self.frame_mask = channel.mask
        # Placeholder-values Series purely for its observe() dispatch (theta
        # and frame-time gating); observe never reads the values.
        self._series = Series(
            channel.name,
            channel.saveat,
            jnp.zeros((jnp.asarray(channel.saveat).shape[0],)),
            observable=_channel_operator(channel),
        )

    def observe(self, states, ts=None, theta=None):
        return self._series.observe(states, ts, theta)


@dataclasses.dataclass(frozen=True)
class Predictor:
    """The pure prediction closure of an experiment, with its coordinates.

    Calling the predictor with a flat learnable-theta vector returns
    ``(predictions, failed)``: a dict mapping channel names to their
    kept-frame observable-space values, and the solver's failure flag.

    Attributes:
        x0: The initial flat learnable-theta vector.
        theta_names: Ordered names of the flat vector's scalars.
        channel_names: The channels predicted, in declaration order.
    """

    x0: Any
    theta_names: tuple
    channel_names: tuple
    _predict: Callable

    def __call__(self, x_flat: Array) -> tuple:
        """Predict every channel at ``x_flat``; returns ``(dict, failed)``."""
        preds, failed = self._predict(x_flat)
        return dict(zip(self.channel_names, preds)), failed


def _trajectory_keywords(trajectory: Any, state_prior: Any) -> dict:
    """The leaf keywords a public trajectory declaration and state prior lower to."""
    from diff_soup.inference.fitting.inference_problem import Multishoot

    keywords: dict = {}
    if isinstance(trajectory, Multishoot):
        keywords["theta_only_steps"] = int(trajectory.warm_start_steps)
        keywords["restart_step"] = trajectory.first_step
    if state_prior is not None:
        keywords["state_prior_weight"] = 1.0 / float(np.asarray(state_prior.scale).reshape(()))
    return keywords


@dataclasses.dataclass(frozen=True)
class _CompiledExperiment:
    """Private predictor/problem delegate for the public UQ experiment.

    Attributes:
        problem: The Recipe API build or solver problem used by every
            design.
        solver: The solver configuration used for every lowering.
        t_span: The trajectory window ``(t0, t1)``.
        initial_state: The differential initial state at ``t_span[0]``: the
            state array, or an initial-state policy carrying it —
            ``FixedInitialState(y0)`` (the default reading of a bare array) or
            ``LearnedInitialState(guess)``, which fits the state along with the
            parameters. A policy's own ``time`` is replaced by the start of
            this span, which is the instant the state stands for.
        channels: The measurement `Channel` declarations.
        controls: Conditioning inputs by name (measured drives, boundary
            traces); recorded for provenance, not scored.
        data: Observed values by channel name (attach via `with_data`).
        allow_overlap: ``source_id`` values whose response/conditioning
            double use is deliberate and acknowledged.
        provenance_notes: Plain-words records of acknowledged overlaps.
        schedule: The operating protocol (a ``ds.Schedule``) the solve follows
            from ``t_span[0]``, or ``None`` for a solve without one.
    """

    problem: Any
    solver: Any
    t_span: tuple
    initial_state: Any
    channels: tuple
    controls: dict = dataclasses.field(default_factory=dict)
    data: dict = dataclasses.field(default_factory=dict)
    allow_overlap: tuple = ()
    provenance_notes: tuple = ()
    schedule: Any = None
    _predictor_cache: dict = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        """Validate channel uniqueness, data binding, and source overlap."""
        from diff_soup.inference.fitting.inference_problem import (
            InitialState,
            _initial_state_array,
        )

        if (
            isinstance(self.initial_state, InitialState)
            and _initial_state_array(self.initial_state) is None
        ):
            raise ValueError(
                f"Experiment initial_state={type(self.initial_state).__name__}() carries "
                "no state. Give the state at the start of t_span: the array itself, "
                "FixedInitialState(y0), or LearnedInitialState(guess) to fit it."
            )
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError(f"Experiment channels must have unique names; got {names}.")
        unknown = sorted(set(self.data) - set(names))
        if unknown:
            raise ValueError(
                f"Experiment data names {unknown} match no declared channel; "
                f"declared channels are {sorted(names)}."
            )
        for name, values in self.data.items():
            channel = self.channel(name)
            n_frames = int(jnp.asarray(channel.saveat).shape[0])
            if isinstance(values, DeferredValues):
                # A deferred loader validates its leading frames when it is
                # declared or loaded; it is not invoked here.
                if values.n_frames != n_frames:
                    raise ValueError(
                        f"Experiment data for channel {name!r} has "
                        f"{values.n_frames} frames but the channel "
                        f"declares {n_frames} observation times."
                    )
                continue
            if jnp.asarray(values).shape[0] != n_frames:
                raise ValueError(
                    f"Experiment data for channel {name!r} has "
                    f"{jnp.asarray(values).shape[0]} frames but the channel "
                    f"declares {n_frames} observation times."
                )

        notes = list(self.provenance_notes)
        response_sources = {
            c.source_id for c in self.channels if c.role == "response" and c.source_id
        }
        for c in self.channels:
            if c.role in ("response", "reference") or c.source_id is None:
                continue
            if c.source_id in response_sources:
                if c.source_id in self.allow_overlap:
                    note = (
                        f"source {c.source_id!r} deliberately enters both as "
                        f"conditioning ({c.role}: {c.name!r}) and as a response; "
                        "acknowledged via allow_overlap"
                    )
                    if note not in notes:
                        notes.append(note)
                else:
                    raise ValueError(
                        f"source {c.source_id!r} enters both as conditioning "
                        f"({c.role}: {c.name!r}) and as a response channel. If "
                        "this double use is deliberate, acknowledge it with "
                        f"allow_overlap=({c.source_id!r},); otherwise give the "
                        "channels distinct sources."
                    )
        object.__setattr__(self, "provenance_notes", tuple(notes))

    # -- declaration helpers ----------------------------------------------

    def channel(self, name: str) -> Channel:
        """Return the channel declared under ``name``."""
        for c in self.channels:
            if c.name == name:
                return c
        raise ValueError(
            f"Experiment has no channel {name!r}; declared channels are "
            f"{sorted(c.name for c in self.channels)}."
        )

    def with_data(self, *, allow_overlap: tuple = (), **channel_data) -> "_CompiledExperiment":
        """Return a new experiment with observed values attached.

        Args:
            allow_overlap: Additional acknowledged ``source_id`` overlaps.
            **channel_data: Observed values keyed by channel name; each must
                have one leading frame per declared observation time.

        Returns:
            A new `Experiment`; the declaration itself is immutable.
        """
        data = dict(self.data)
        for name, values in channel_data.items():
            data[name] = jnp.asarray(values)
        updated = dataclasses.replace(
            self,
            data=data,
            allow_overlap=tuple(self.allow_overlap) + tuple(allow_overlap),
        )
        # Data and overlap acknowledgements do not alter the forward model.
        # Reuse prepared prediction closures across the immutable data attach.
        object.__setattr__(updated, "_predictor_cache", self._predictor_cache)
        return updated

    # -- lowering -----------------------------------------------------------

    def observed(self, name: str) -> Any:
        """Observed values of one channel, loading a deferred measurement once."""
        values = self.data[name]
        return values.materialize() if isinstance(values, DeferredValues) else values

    def _response_series(self, *, load_data: bool = True) -> tuple:
        """Build the scored Series streams from data-bearing response channels."""
        series = []
        for c in self.channels:
            if c.role != "response" or c.name not in self.data:
                continue
            values = self.data[c.name]
            placeholder = False
            if isinstance(values, DeferredValues):
                placeholder = not load_data
                values = values.materialize() if load_data else values.placeholder()
            supported = (
                Gaussian,
                CorrelatedGaussian,
                StudentT,
                Poisson,
                LogNormal,
                MultiplicativeGaussian,
            )
            if not isinstance(c.likelihood, supported):
                raise ValueError(
                    "InferenceProblem accepts Gaussian, CorrelatedGaussian, StudentT, Poisson, "
                    "LogNormal, and MultiplicativeGaussian likelihoods; "
                    f"measurement {c.name!r} uses {type(c.likelihood).__name__}."
                )
            likelihood = c.likelihood
            if isinstance(likelihood, MultiplicativeGaussian):
                if not placeholder:
                    _reject_zero_retained(c.name, values, c.mask)
                likelihood = _MultiplicativeDeviance(scale=likelihood.scale)

            series.append(
                Series(
                    c.name,
                    c.saveat,
                    values,
                    observable=_channel_operator(c),
                    # Keep the diagonal affine lowering while giving Measurement
                    # the observed-minus-predicted residual convention.
                    weight=(
                        jnp.broadcast_to(-1.0 / likelihood.scale, values.shape)
                        if isinstance(likelihood, Gaussian)
                        else None
                    ),
                    likelihood=None if isinstance(likelihood, Gaussian) else likelihood,
                    likelihood_context=None if isinstance(likelihood, Gaussian) else c.context,
                    frame_mask=c.mask,
                )
            )
        return tuple(series)

    def as_inference_problem(
        self,
        *,
        parameters: Any = None,
        objective: Any = None,
        trajectory: Any = None,
        regularizer: Any = None,
        bounds: Any = None,
        t0: Optional[float] = None,
        load_data: bool = True,
        reuse: Any = None,
        state_prior: Any = None,
    ) -> Any:
        """Lower the data-bearing declaration to a standard ``InferenceProblem``.

        Only response channels with attached data become residual streams;
        conditioning channels never contribute scored rows. The result is an
        ordinary [InferenceProblem][diff_soup.inference.InferenceProblem] — fit it with
        ``diff_soup.inference.optsolve``.

        Args:
            parameters: Optional ``ParameterGroups`` choosing the learnable
                theta; defaults to the problem's declared learnables.
            objective: Optional objective, defaulting to least squares.
            trajectory: Optional trajectory strategy, defaulting to single-shoot.
            regularizer: Optional extra least-squares rows over the parameters.
            bounds: Optional named box bounds over the parameter groups.
            t0: Initial-state time, defaulting to ``t_span[0]``.
            load_data: Whether deferred measurement values are loaded on this
                process; ``False`` lowers over zero placeholders of their shape.
            reuse: An ``InferenceProblem`` this experiment lowered earlier (over
                placeholders); its lowering is kept and only the residuals are
                rebuilt over the values loaded now.
            state_prior: Optional Gaussian prior on the latent window states of
                a decision-variable multiple-shooting trajectory; its scale is
                the standard deviation of a window start's distance from its
                initialization.

        Raises:
            ValueError: If no response channel has data (attach observations
                with `with_data` first), or a scored channel's
                likelihood has no exact least-squares representation.
        """
        from diff_soup.inference.fitting.inference_problem import (
            InferenceProblem,
            LeastSquares,
            SingleShoot,
            _initial_state_policy,
        )

        series = self._response_series(load_data=load_data)
        if reuse is not None and series:
            return reuse.with_data(series)
        if not series:
            raise ValueError(
                "Experiment.as_inference_problem needs observed response data; attach "
                "it with with_data(<channel_name>=values) first. For design-stage "
                "analysis use predictor() or information()."
            )
        trajectory = trajectory if trajectory is not None else SingleShoot()
        return InferenceProblem(
            self.problem,
            data=series,
            solver=self.solver,
            parameters=parameters,
            objective=objective if objective is not None else LeastSquares(),
            trajectory=trajectory,
            initial_state=_initial_state_policy(
                self.initial_state,
                time=float(self.t_span[0]) if t0 is None else t0,
            ),
            regularizer=regularizer,
            bounds=bounds,
            schedule=self.schedule,
            **_trajectory_keywords(trajectory, state_prior),
        )

    def predictor(self, *, parameters: Any = None) -> Predictor:
        """Build the one pure prediction closure over every channel.

        Works with or without attached data: predictions are a function of
        the flat learnable-theta vector alone, produced by a single shared
        solve over the union of every channel's observation times.

        Args:
            parameters: Optional [ParameterGroups][diff_soup.inference.fitting.inference_problem.ParameterGroups]
                choosing the learnable theta; defaults to the problem's declared
                learnables, matching ``InferenceProblem``'s default.

        Returns:
            A `Predictor` carrying ``x0`` and the closure.
        """
        from diff_soup.inference.fitting._forward_seam import (
            build_forward_seam,
            resolve_forward_adjoint,
            union_prediction_factory,
        )
        from diff_soup.inference.fitting.inference_problem import (
            _initial_state_array,
            _lowered_schedule,
            _resolve_parameters,
        )

        theta_spec, groups = _resolve_parameters(parameters, self.problem)
        # Resolved group records contain names and frozensets of selected leaf
        # identities, so equivalent declarations have a stable hashable key.
        cached = self._predictor_cache.get(groups)
        if cached is not None:
            return cached

        seam = build_forward_seam(self.problem, theta_spec)
        solver = resolve_forward_adjoint(self.solver)
        streams = tuple(_PredictionStream(c) for c in self.channels)
        predict, _, _, _ = union_prediction_factory(
            seam,
            streams=streams,
            y0=jnp.asarray(_initial_state_array(self.initial_state)),
            solver=solver,
            t0=float(self.t_span[0]),
            schedule=_lowered_schedule(self.schedule, self.problem),
        )
        predictor = Predictor(
            x0=seam.x0,
            theta_names=tuple(seam.theta_names),
            channel_names=tuple(c.name for c in self.channels),
            _predict=predict,
        )
        self._predictor_cache[groups] = predictor
        return predictor


# ---------------------------------------------------------------------------
# Multi-experiment declaration and lowering
# ---------------------------------------------------------------------------


def _validated_experiment_ids(experiments: tuple, experiment_ids: Any) -> tuple:
    """Validate stable, unique IDs, deriving them only from declarations."""
    if experiment_ids is None:
        missing = tuple(i for i, exp in enumerate(experiments) if not hasattr(exp, "experiment_id"))
        if missing:
            raise ValueError(
                "ExperimentSet needs stable experiment IDs. Pass experiment_ids= "
                "explicitly, or give every declaration an experiment_id attribute; "
                f"experiments at indices {missing} provide neither."
            )
        experiment_ids = tuple(exp.experiment_id for exp in experiments)
    else:
        experiment_ids = tuple(experiment_ids)
    if len(experiment_ids) != len(experiments):
        raise ValueError(
            f"ExperimentSet received {len(experiment_ids)} experiment IDs for "
            f"{len(experiments)} experiments; provide exactly one stable ID per experiment."
        )
    for experiment_id in experiment_ids:
        if experiment_id is None or (isinstance(experiment_id, str) and not experiment_id):
            raise ValueError(
                "ExperimentSet experiment IDs must be non-empty stable identifiers; "
                f"got {experiment_id!r}."
            )
        try:
            hash(experiment_id)
        except TypeError as exc:
            raise TypeError(
                "ExperimentSet experiment IDs must be hashable stable identifiers; "
                f"got {experiment_id!r}."
            ) from exc
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError(
            "ExperimentSet experiment IDs must be unique; got "
            f"{experiment_ids}. Repeated observations need distinct stable IDs."
        )
    return experiment_ids


def _normalized_ownership(ownership: Any, n_experiments: int) -> Any:
    """Normalize JointInferenceProblem's map vocabulary without resolving groups."""
    if ownership is None:
        return {}
    if callable(ownership):
        return ownership
    if not isinstance(ownership, dict):
        raise TypeError(
            "ExperimentSet 'ownership' must be a mapping, a callable "
            f"owner(group, experiment), or None; got {type(ownership).__name__}."
        )

    normalized = {}
    expected = set(range(n_experiments))
    for group, declared in ownership.items():
        if not isinstance(group, str) or not group:
            raise TypeError(
                f"ExperimentSet ownership group names must be non-empty strings; got {group!r}."
            )
        if isinstance(declared, dict):
            keys = set(declared)
            if keys != expected:
                raise ValueError(
                    f"ExperimentSet ownership[{group!r}] must name every experiment "
                    f"index {sorted(expected)} exactly once; got {sorted(keys)}."
                )
            labels = tuple(declared[i] for i in range(n_experiments))
        else:
            try:
                labels = tuple(declared)
            except TypeError as exc:
                raise TypeError(
                    f"ExperimentSet ownership[{group!r}] must be an experiment-index "
                    "mapping or a sequence of owner IDs."
                ) from exc
            if len(labels) != n_experiments:
                raise ValueError(
                    f"ExperimentSet ownership[{group!r}] has {len(labels)} owner IDs "
                    f"for {n_experiments} experiments."
                )
        for label in labels:
            try:
                hash(label)
            except TypeError as exc:
                raise TypeError(
                    f"ExperimentSet owner IDs must be hashable; group {group!r} contains {label!r}."
                ) from exc
        normalized[group] = {i: label for i, label in enumerate(labels)}
    return normalized


def _sum_information_field(results: tuple, field: str) -> Any:
    """Add one optional InformationResult field without an integer seed."""
    values = tuple(
        getattr(result, field) for result in results if getattr(result, field) is not None
    )
    if not values:
        return None
    total = jnp.zeros_like(values[0])
    for value in values:
        total = total + value
    return total


def _reject_zero_retained(name: str, values: Any, mask: Any) -> None:
    """Refuse zero retained observations under the multiplicative Gaussian likelihood."""
    values = np.asarray(values)
    retained = (
        np.ones(values.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )
    frame_axes = tuple(range(1, values.ndim))
    zero = (
        retained & np.any(values == 0, axis=frame_axes) if frame_axes else retained & (values == 0)
    )
    count = int(np.count_nonzero(zero))
    if count:
        first = int(np.flatnonzero(zero)[0])
        raise ValueError(
            f"MultiplicativeGaussian requires nonzero retained observations; measurement "
            f"{name!r} has {count} zero-valued retained frame(s), the first at index {first}. "
            "Mask those frames out of the measurement, or use Gaussian for it."
        )


def _kept_response_data(experiment: Any, channels: tuple) -> dict:
    """Return response data with the same frame selection as prediction."""
    kept = {}
    for channel in channels:
        values = experiment.observed(channel.name)
        if channel.mask is not None:
            indices = np.flatnonzero(np.asarray(channel.mask, dtype=bool))
            values = jnp.asarray(values)[jnp.asarray(indices)]
        kept[channel.name] = values
    return kept


class ExperimentSet(eqx.Module):
    """A frozen set of experiments with explicit ownership and independence.

    ``ExperimentSet`` is a declaration/lowering layer, not a second fitting
    engine. `as_joint` lowers every child through
    `Experiment.as_inference_problem` and hands the result to the established
    [JointInferenceProblem][diff_soup.inference.fitting.joint.JointInferenceProblem], preserving the exact
    device-layout object. Statistical aggregation is additive only when
    ``independent=True`` was explicitly declared. Correlated experiments must
    provide a ``joint_likelihood`` object exposing the corresponding
    ``information(experiment_set, ...)`` and/or
    ``log_posterior(experiment_set, ...)`` protocol; absence is an error,
    never an implicit independence assumption.

    Attributes:
        experiments: Non-empty tuple of experiment declarations.
        ownership: The same mapping/callable vocabulary accepted by
            [JointInferenceProblem][diff_soup.inference.fitting.joint.JointInferenceProblem].
        independent: Whether likelihoods are declared independent across
            experiments.
        experiment_ids: One unique, hashable, stable ID per experiment.
        joint_likelihood: Optional explicit correlated-likelihood protocol.
    """

    experiments: tuple
    ownership: Any
    independent: bool = eqx.field(static=True)
    experiment_ids: tuple
    joint_likelihood: Any

    def __init__(
        self,
        experiments: Any,
        *,
        ownership: Any = None,
        independent: bool,
        experiment_ids: Any = None,
        joint_likelihood: Any = None,
    ):
        """Validate and freeze a multi-experiment declaration."""
        experiments = tuple(experiments)
        if not experiments:
            raise ValueError("ExperimentSet requires at least one experiment.")
        if not isinstance(independent, (bool, np.bool_)):
            raise TypeError(
                f"ExperimentSet 'independent' must be a boolean declaration; got {independent!r}."
            )
        if bool(independent) and joint_likelihood is not None:
            raise ValueError(
                "ExperimentSet cannot declare independent=True and also provide "
                "a joint likelihood; choose the independence factorization or set "
                "independent=False for the correlated model."
            )
        self.experiments = experiments
        self.ownership = _normalized_ownership(ownership, len(experiments))
        self.independent = bool(independent)
        self.experiment_ids = _validated_experiment_ids(experiments, experiment_ids)
        self.joint_likelihood = joint_likelihood

    def subset(self, indices: Any) -> "ExperimentSet":
        """Select experiments in order, preserving IDs and remapping child keys.

        Owner IDs are semantic labels and stay unchanged; only the integer
        experiment indices are compacted to ``0..len(indices)-1``. A
        correlated joint likelihood must implement ``subset(indices)`` so its
        covariance/dependence structure cannot silently become misaligned.
        """
        try:
            indices = tuple(int(i) for i in indices)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "ExperimentSet.subset indices must be an iterable of integers."
            ) from exc
        if not indices:
            raise ValueError("ExperimentSet.subset requires at least one experiment index.")
        if len(set(indices)) != len(indices):
            raise ValueError(f"ExperimentSet.subset indices must be unique; got {indices}.")
        invalid = tuple(i for i in indices if i < 0 or i >= len(self.experiments))
        if invalid:
            raise IndexError(
                f"ExperimentSet.subset indices {invalid} are outside "
                f"0..{len(self.experiments) - 1}."
            )

        if callable(self.ownership):
            source_ownership = self.ownership

            def ownership(group, child):
                return source_ownership(group, indices[child])

        else:
            ownership = {
                group: {new: mapping[old] for new, old in enumerate(indices)}
                for group, mapping in self.ownership.items()
            }

        joint_likelihood = self.joint_likelihood
        if joint_likelihood is not None:
            subset_joint = getattr(joint_likelihood, "subset", None)
            if not callable(subset_joint):
                raise ValueError(
                    "ExperimentSet.subset on correlated experiments requires the "
                    "explicit joint likelihood to implement subset(indices), so its "
                    "correlation structure stays aligned."
                )
            joint_likelihood = subset_joint(indices)

        return ExperimentSet(
            tuple(self.experiments[i] for i in indices),
            ownership=ownership,
            independent=self.independent,
            experiment_ids=tuple(self.experiment_ids[i] for i in indices),
            joint_likelihood=joint_likelihood,
        )

    def as_joint(self, *, parameters: Any = None, layout: Any = None) -> Any:
        """Lower to the existing ``JointInferenceProblem`` fit engine.

        ``parameters.shared`` supplies the globally tied group names; owner
        declarations flow through unchanged, and ``layout`` is forwarded by
        identity so existing vmap/shard-map/multi-GPU behavior composes with
        this declaration layer.
        """
        from diff_soup.inference.fitting.joint import JointInferenceProblem

        children = tuple(
            experiment.as_inference_problem(parameters=parameters)
            for experiment in self.experiments
        )
        shared = () if parameters is None else tuple(parameters.shared)
        return JointInferenceProblem(
            children,
            shared=shared,
            ownership=self.ownership,
            layout=layout,
        )

    def information(
        self,
        *,
        at: Any = None,
        kind: str = "expected",
        prior: Any = None,
        parameters: Any = None,
    ) -> Any:
        """Add information from independent experiments and one prior.

        Every child is evaluated in the same named physical coordinate. Its
        data information is computed without a prior, then the child matrices
        and whitened sensitivity rows are stacked/summed. A supplied prior
        precision is added once at set level, never once per experiment.
        """
        if not self.independent:
            if self.joint_likelihood is None:
                raise ValueError(
                    "ExperimentSet.information cannot add correlated experiments: "
                    "declare independent=True or provide an explicit joint likelihood."
                )
            joint_information = getattr(self.joint_likelihood, "information", None)
            if not callable(joint_information):
                raise ValueError(
                    "The explicit joint likelihood must implement "
                    "information(experiment_set, ...) for correlated information."
                )
            return joint_information(
                self,
                at=at,
                kind=kind,
                prior=prior,
                parameters=parameters,
            )

        from diff_soup.inference.statistics.information import (
            _resolve_at,
            assemble_information,
            information,
        )

        predictors = tuple(exp.predictor(parameters=parameters) for exp in self.experiments)
        theta_names = tuple(predictors[0].theta_names)
        for experiment_id, predictor in zip(self.experiment_ids[1:], predictors[1:]):
            if tuple(predictor.theta_names) != theta_names:
                raise ValueError(
                    "ExperimentSet.information can add experiments only in one "
                    "shared physical parameter coordinate; experiment "
                    f"{experiment_id!r} has names {tuple(predictor.theta_names)} "
                    f"but the first has {theta_names}. Lower owner/local models "
                    "with as_joint() and use the block-statistics path."
                )
        x = _resolve_at(predictors[0], at)
        child_results = tuple(
            information(
                experiment,
                at=x,
                kind=kind,
                prior=None,
                parameters=parameters,
            )
            for experiment in self.experiments
        )
        whitened = jnp.concatenate(
            tuple(result.whitened_jacobian for result in child_results), axis=0
        )

        prior_precision = None
        if prior is not None:
            if getattr(prior, "coordinate", "x") != "x":
                raise ValueError(
                    "ExperimentSet.information requires a prior declared on the "
                    f"physical coordinate 'x'; got {prior.coordinate!r}."
                )
            prior_size = getattr(prior, "size", None)
            if prior_size is not None and int(prior_size) != len(theta_names):
                raise ValueError(
                    f"ExperimentSet.information: the prior covers {int(prior_size)} "
                    f"scalars but the experiments use {len(theta_names)} "
                    f"parameters ({theta_names})."
                )
            try:
                prior_precision = prior.precision()
            except (ValueError, NotImplementedError):
                prior_precision = prior.precision(at=x)

        provenance = {
            "at": tuple(float(v) for v in np.asarray(x)),
            "theta_names": theta_names,
            "experiment_ids": self.experiment_ids,
            "independence": "declared independent",
            "children": tuple(result.provenance for result in child_results),
        }
        return assemble_information(
            whitened_jacobian=whitened,
            expected=_sum_information_field(child_results, "expected"),
            observed=_sum_information_field(child_results, "observed"),
            ggn=_sum_information_field(child_results, "ggn"),
            prior_precision=prior_precision,
            coordinate="x",
            provenance=provenance,
        )

    def log_posterior(
        self,
        *,
        prior: Any,
        parameterization: Any = None,
        gradient: Optional[str] = "forward",
        parameters: Any = None,
    ) -> Any:
        """Build the normalized product posterior of independent experiments.

        Each child's normalized response likelihood contributes once and the
        proper physical-coordinate prior contributes once. With a
        parameterization, its pullback and exact Jacobian correction are also
        applied only once to the full shared parameter vector.
        """
        if not self.independent:
            if self.joint_likelihood is None:
                raise ValueError(
                    "ExperimentSet.log_posterior cannot multiply correlated "
                    "experiment likelihoods: declare independent=True or provide "
                    "an explicit joint likelihood."
                )
            joint_target = getattr(self.joint_likelihood, "log_posterior", None)
            if not callable(joint_target):
                raise ValueError(
                    "The explicit joint likelihood must implement "
                    "log_posterior(experiment_set, ...) for correlated inference."
                )
            return joint_target(
                self,
                prior=prior,
                parameterization=parameterization,
                gradient=gradient,
                parameters=parameters,
            )
        if prior is None:
            raise ValueError(
                "ExperimentSet.log_posterior needs one proper prior; without a "
                "normalized prior the posterior is not a probability model."
            )
        if getattr(prior, "coordinate", "x") != "x":
            raise ValueError(
                "ExperimentSet.log_posterior requires a prior declared on the "
                f"physical coordinate 'x'; got {prior.coordinate!r}."
            )

        from diff_soup.inference.statistics.sample import LogDensity

        predictors = tuple(exp.predictor(parameters=parameters) for exp in self.experiments)
        theta_names = tuple(predictors[0].theta_names)
        records = []
        for experiment_id, experiment, predictor in zip(
            self.experiment_ids, self.experiments, predictors
        ):
            if tuple(predictor.theta_names) != theta_names:
                raise ValueError(
                    "ExperimentSet.log_posterior needs one shared physical "
                    f"coordinate; experiment {experiment_id!r} has names "
                    f"{tuple(predictor.theta_names)} but the first has {theta_names}."
                )
            response = tuple(
                channel for channel in experiment.channels if channel.role == "response"
            )
            scored = tuple(channel for channel in response if channel.name in experiment.data)
            if not scored:
                raise ValueError(
                    "ExperimentSet.log_posterior needs observed response data on "
                    f"every included experiment; {experiment_id!r} has none."
                )
            records.append((predictor, scored, _kept_response_data(experiment, scored)))

        n_parameters = len(theta_names)
        prior_size = getattr(prior, "size", None)
        if prior_size is not None and int(prior_size) != n_parameters:
            raise ValueError(
                f"ExperimentSet.log_posterior: the prior covers {int(prior_size)} "
                f"scalars but the experiments use {n_parameters} parameters "
                f"({theta_names})."
            )

        def log_likelihood(x_flat: Array) -> Array:
            total = jnp.zeros((), dtype=jnp.asarray(x_flat).dtype)
            for predictor, scored, kept in records:
                predictions, _ = predictor(x_flat)
                for channel in scored:
                    likelihood, context = _channel_likelihood_view(
                        channel, predictions[channel.name]
                    )
                    total = total + likelihood.log_prob(
                        kept[channel.name],
                        predictions[channel.name],
                        context=context,
                    )
            return total

        x0 = jnp.asarray(predictors[0].x0)
        to_physical = None
        if parameterization is None:
            coordinate = "x"
            names = theta_names
            init = x0

            def target_fn(z: Array) -> Array:
                return log_likelihood(z) + prior.log_prob(z)

        else:
            if parameterization.n_parameters != n_parameters:
                raise ValueError(
                    "ExperimentSet.log_posterior: the parameterization covers "
                    f"{parameterization.n_parameters} scalars but the experiments "
                    f"use {n_parameters} parameters ({theta_names})."
                )
            coordinate = "z"
            names = tuple(parameterization.names())
            init = parameterization.inverse(x0)
            to_physical = parameterization.transform

            def target_fn(z: Array) -> Array:
                x = parameterization.transform(z)
                return (
                    log_likelihood(x)
                    + prior.log_prob(x)
                    + jnp.sum(parameterization.log_abs_det_jacobian(z))
                )

        target = LogDensity.from_callable(
            target_fn,
            init=init,
            names=names,
            coordinate=coordinate,
            gradient=gradient,
        )
        if to_physical is not None:
            target = dataclasses.replace(target, _to_physical=to_physical)
        return target
