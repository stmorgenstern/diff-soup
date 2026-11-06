"""The identity of a lane: what decides which lanes share one compiled program.

The lanes of one bucket run through one compiled program. That program closes
over the solver problem, the solver, the parameter map and the observation
operators of one representative experiment, and takes every number of a lane
as an argument. The lanes of two experiments can therefore share the program
exactly when everything the program closes over is the same for both; their
numbers may differ. The lanes of the windows of one experiment differ in
their numbers only.

The identity of a lane is a digest of those components, each named by the word
the plan uses for it:

| component | covers |
|---|---|
| lane kind | how the experiment is fitted: in one shoot, in windows, or with a learned initial state |
| precision | the model's dtype policy and the dtype of its parameters |
| layout | the model's ``Layout`` |
| model content | the model's content key: equations, domains, boundary conditions, scales and the values their closures capture, with layout and precision folded in |
| model structure | the solver problem's kind, unknown counts, differential mask, partitions and sparsity patterns |
| parameter structure | leaf paths, shapes and dtypes of the parameters |
| control rows | the direct and derived controls, and the degree of a schedule's targets when one varies in time |
| solver | type, options, tolerances, linear backend, adjoint |
| fitted parameters | names and count of the fitted parameters |
| fixed-parameter shapes | leaf paths, shapes and dtypes of the parameters that are not fitted |
| streams | per stream the lanes score, its name, the shape of one frame, its observation operator with the values it holds, and the family of its likelihood object: type, static fields, and leaf paths, dtypes and shapes per frame |

Parameter values, initial states, spans, sample times, targets, weights, masks,
the numbers of a likelihood and frame counts enter the program as arguments and
are part of no component.
A bucket adds three sizes to the identity of its lanes, its frame class, its
derivative width and its segment class;
[COMPONENTS][diff_soup.inference.engine._identity.COMPONENTS] lists all
fourteen words in the order the plan reports them.

Each component is encoded with the recorder of the study fingerprint and hashed
with SHA-256. A value the recorder cannot encode is compared by object: the
identity keeps the object alive and encodes its address, so two lanes holding
the same object agree and two equal but distinct objects differ. Such a
component can split a bucket but never merges two; the identity names it in
``by_object``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import jax
import numpy as np

from diff_soup.inference.engine._layout import _window_count
from diff_soup.inference.statistics._fingerprint import (
    UnstableFingerprintError,
    _canonical_json,
    _qualified_type,
    _Recorder,
)
from diff_soup.inference.statistics.likelihoods import CorrelatedGaussian

LANE_COMPONENTS: tuple[str, ...] = (
    "lane kind",
    "precision",
    "layout",
    "model content",
    "model structure",
    "parameter structure",
    "control rows",
    "solver",
    "fitted parameters",
    "fixed-parameter shapes",
    "streams",
)
"""The components of a lane's identity, in the order the plan reports them."""

COMPONENTS: tuple[str, ...] = LANE_COMPONENTS + ("frame class", "derivative width", "segment class")
"""Every component of a bucket's identity: the lane's, then the bucket's sizes."""

_LANE_KINDS = {
    "single": "one shoot from a fixed initial state",
    "observed_ic": "windows that start from observed states",
    "window": "windows whose starting states are fitted",
    "latent": "one shoot from a fitted initial state",
}


@dataclasses.dataclass(frozen=True)
class LaneIdentity:
    """The identity of one lane, or of a bucket. Equal identities share one program.

    A lane's identity has the components of
    [LANE_COMPONENTS][diff_soup.inference.engine._identity.LANE_COMPONENTS];
    a bucket's identity ([bucket_identity][diff_soup.inference.engine._identity.bucket_identity])
    adds its three sizes. Equality compares ``digest`` only. ``str()`` gives
    its first twelve hexadecimal characters, the form the plan prints.

    Attributes:
        digest: SHA-256, in hexadecimal, of the component digests in the order
            of [COMPONENTS][diff_soup.inference.engine._identity.COMPONENTS].
        components: One ``(component, digest)`` pair per component, in that
            order.
        by_object: The components that hold a value the recorder cannot
            encode and that are compared by object.
        held: The objects compared by object, kept alive so that no other
            object can take their addresses while this identity exists.
    """

    digest: str
    components: tuple = dataclasses.field(compare=False)
    by_object: tuple = dataclasses.field(default=(), compare=False)
    held: tuple = dataclasses.field(default=(), compare=False, repr=False)

    def __str__(self) -> str:
        return self.digest[:12]

    def differs(self, other: LaneIdentity) -> tuple[str, ...]:
        """The components in which ``other`` differs from this identity, in the plan's order."""
        mine, theirs = dict(self.components), dict(other.components)
        return tuple(name for name in COMPONENTS if mine.get(name) != theirs.get(name))


class _IdentityRecorder(_Recorder):
    """The fingerprint recorder, comparing by object whatever it cannot encode."""

    def __init__(self) -> None:
        super().__init__()
        self.component = ""
        self.by_object: list[str] = []
        self.held: list[Any] = []

    def record(self, value: Any, *, path: str) -> Any:
        """Encode ``value``; an object without a stable encoding is encoded by its address."""
        try:
            return super().record(value, path=path)
        except UnstableFingerprintError:
            return self.by_address(value)

    def by_address(self, value: Any) -> dict:
        """Encode ``value`` by object and keep it alive."""
        if self.component not in self.by_object:
            self.by_object.append(self.component)
        self.held.append(value)
        return {"kind": "object", "type": _qualified_type(value), "address": id(value)}


def _frame_shape(series: Any) -> tuple:
    return tuple(int(size) for size in np.shape(series.values)[1:])


def _leaf_shapes(tree: Any) -> tuple:
    """Leaf paths, shapes and dtypes of ``tree``, read without touching a device."""
    leaves = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        leaves.append(
            (
                jax.tree_util.keystr(path),
                None if shape is None else tuple(int(size) for size in shape),
                None if dtype is None else np.dtype(dtype).str,
            )
        )
    return tuple(leaves)


def _likelihood_leaves(series: Any) -> tuple:
    """Each leaf of a stream's likelihood object as ``(path, leaf, span)``, read on the host.

    ``span`` is ``"none"`` for a leaf that holds the same numbers for every
    frame (a pointwise leaf that broadcasts to one frame's shape, or the
    factor of a ``CorrelatedGaussian`` whose events lie within one frame);
    ``"frame"`` for a pointwise leaf with one entry per frame along its leading
    axis; ``"event"`` for the factor of a ``CorrelatedGaussian`` whose one
    event holds every frame of a stream of one value per frame; and
    ``"spans"`` for one whose events cross frames without holding all of them.
    """
    likelihood = series.likelihood
    shape = tuple(int(size) for size in np.shape(series.values))
    frame = shape[1:]
    leaves = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(likelihood):
        size = tuple(int(entry) for entry in np.shape(leaf))
        if isinstance(likelihood, CorrelatedGaussian):
            if int(np.prod(frame, dtype=np.int64)) % size[0] == 0:
                span = "none"
            elif not frame and size == (shape[0], shape[0]):
                span = "event"
            else:
                span = "spans"
        else:
            try:
                span = "none" if np.broadcast_shapes(size, frame) == frame else "frame"
            except ValueError:
                span = "frame"
        leaves.append((jax.tree_util.keystr(path), leaf, span))
    return tuple(leaves)


def _family(series: Any) -> tuple:
    """A stream's likelihood family as plain values: its type, static fields and leaves.

    Each leaf is recorded by its path, dtype, span and shape for one frame, so
    the lanes of one family share a program whatever their numbers and frame
    counts.
    """
    likelihood = series.likelihood
    fields = dataclasses.fields(likelihood) if dataclasses.is_dataclass(likelihood) else ()
    static = tuple(
        (field.name, getattr(likelihood, field.name))
        for field in fields
        if field.metadata.get("static", False)
    )
    frame = tuple(int(size) for size in np.shape(series.values)[1:])
    leaves = []
    for path, leaf, span in _likelihood_leaves(series):
        dtype = leaf.dtype if hasattr(leaf, "dtype") else np.asarray(leaf).dtype
        size = {"frame": frame, "event": ()}.get(span, tuple(int(n) for n in np.shape(leaf)))
        leaves.append((path, np.dtype(dtype).str, span, size))
    return (_qualified_type(likelihood), static, tuple(leaves))


def _window_streams(child: Any) -> tuple:
    """``(seed, scored)`` of an experiment fitted in observed-start windows.

    ``seed`` is the stream whose measured states start the windows, ``scored``
    the stream they are scored on: the seed stream itself when it is the one
    scored.
    """
    seed, scored = child._resolve_observed_ic_streams(child.data, child.trajectory.windows)
    return seed, seed if scored is None else scored


def _lane_streams(child: Any) -> tuple:
    """The streams the lanes of an experiment score, in the order of their numbers.

    Every stream of an experiment solved in one shoot; the scored stream of an
    experiment fitted in observed-start windows.
    """
    if not _window_count(child):
        return tuple(child.data)
    return (_window_streams(child)[1],)


def _components(child: Any, recorder: _IdentityRecorder) -> tuple:
    """The record of every component of ``child``, in the order of ``LANE_COMPONENTS``."""
    from diff_soup.inference.fitting.joint import _child_mode

    seam = child._seam
    view = seam.problem
    signature = view.structural_signature
    schedule = getattr(child, "_schedule_arrays", None)
    # A schedule whose targets vary in time adds a leaf of their terms to the
    # program; constant targets add none, and leave the component as it was.
    degree = 0 if schedule is None else int(schedule.degree)
    controls = signature.control_bindings
    values = {
        "lane kind": _child_mode(child),
        "precision": (view.precision, np.dtype(np.asarray(seam.x0).dtype).str),
        "layout": view.layout,
        "model content": signature.source_key,
        "model structure": (
            signature.problem_kind,
            signature.plan_mode,
            signature.n,
            signature.n_u,
            signature.n_lam,
            signature.id_mask,
            signature.partitions,
            signature.fy_topology,
            signature.fydot_topology,
            signature.full_topology,
        ),
        "parameter structure": signature.theta_structure,
        "control rows": controls if not degree else (controls, ("target degree", degree)),
        "solver": child.solver,
        "fitted parameters": tuple(str(name) for name in child.theta_names),
        "fixed-parameter shapes": _leaf_shapes(seam.theta_frozen),
        "streams": tuple(
            (str(series.name), _frame_shape(series), series.observable)
            + (() if getattr(series, "likelihood", None) is None else (_family(series),))
            for series in _lane_streams(child)
        ),
    }
    # A model without a declared content key is known by the object that built it.
    source_key = signature.source_key
    by_object = isinstance(source_key, tuple) and bool(source_key) and source_key[0] == "identity"
    records = []
    for name in LANE_COMPONENTS:
        recorder.component = name
        if name == "model content" and by_object:
            owner = view.source if view.source is not None else view.problem
            records.append((name, recorder.by_address(owner)))
        else:
            records.append((name, recorder.record(values[name], path=name)))
    return tuple(records)


def _sha256(record: Any) -> str:
    return hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


def lane_identity(child: Any) -> LaneIdentity:
    """The identity of the lanes of one experiment of a joint problem.

    Every lane of the experiment has it: the lanes of its windows differ only
    in their numbers.

    Args:
        child: One experiment's fitted problem, an entry of ``joint.problems``.

    Returns:
        The [LaneIdentity][diff_soup.inference.engine._identity.LaneIdentity].
    """
    recorder = _IdentityRecorder()
    components = tuple((name, _sha256(record)) for name, record in _components(child, recorder))
    return LaneIdentity(
        digest=_sha256([list(pair) for pair in components]),
        components=components,
        by_object=tuple(recorder.by_object),
        held=tuple(recorder.held),
    )


def bucket_identity(lane: LaneIdentity, *, frames: Any, width: int, segments: Any) -> LaneIdentity:
    """The identity of a bucket: the identity of its lanes and the sizes of its program.

    Args:
        lane: The identity every lane of the bucket has.
        frames: The frame class: union sample slots, then frame slots per
            stream.
        width: The derivative width.
        segments: The segment class, or ``None`` when the lanes follow no
            schedule.

    Returns:
        A [LaneIdentity][diff_soup.inference.engine._identity.LaneIdentity]
        whose components are those of ``lane`` followed by "frame class",
        "derivative width" and "segment class".
    """
    union, streams = frames
    sizes = (
        ("frame class", (int(union), tuple(int(count) for count in streams))),
        ("derivative width", int(width)),
        ("segment class", None if segments is None else int(segments)),
    )
    recorder = _IdentityRecorder()
    components = tuple(lane.components) + tuple(
        (name, _sha256(recorder.record(value, path=name))) for name, value in sizes
    )
    return LaneIdentity(
        digest=_sha256([list(pair) for pair in components]),
        components=components,
        by_object=lane.by_object,
        held=lane.held,
    )


def lane_refusal(child: Any) -> str | None:
    """Why the batched lanes cannot take an experiment, or ``None`` when they can.

    A lane solves one span in one shoot from a fixed state: an experiment's
    whole span from its initial state, following its schedule when it has
    one, or one window of an experiment fitted in windows that start from
    observed states, from the measured state at the window's start. It forms
    each scored stream's rows from its weights or from its likelihood object,
    padding and masking whole frames. Refused are: an experiment of another
    lane kind (a fitted initial state, or windows whose starting states are
    fitted); one whose model declares an operational control table and that
    follows no schedule, since its solve needs an operating mode and target;
    one whose schedule stops a segment on anything but its duration (a
    nonzero ``stop_kind``); and a scored stream under a ``CorrelatedGaussian``
    whose events couple frames, unless one event holds every frame and none
    is masked.

    Args:
        child: One experiment's fitted problem, an entry of ``joint.problems``.

    Returns:
        The reason, a sentence that names what the lanes cannot take, or
        ``None``.
    """
    from diff_soup.inference.fitting.joint import _child_mode

    mode = _child_mode(child)
    if mode not in ("single", "observed_ic"):
        return (
            f"its lane kind is {_LANE_KINDS.get(mode, mode)}, and the batched lanes take only "
            f"the lane kinds {_LANE_KINDS['single']} and {_LANE_KINDS['observed_ic']}; fit it "
            "from a fixed initial state, in one shoot or in windows that start from observed "
            "states"
        )
    schedule = getattr(child, "_schedule_arrays", None)
    if schedule is None:
        # The solver view the experiment is solved on, where a scheduled solve
        # looks for the table.
        table = getattr(child._seam.problem, "operational_table", None)
        if table is not None:
            return (
                f"its model declares an operational control table ({table.name!r}) and it "
                "follows no schedule, so its solve has no operating mode and target; give the "
                "experiment a schedule of segments (Experiment(schedule=ds.Schedule(...)))"
            )
    else:
        stop_kind = np.asarray(schedule.stop_kind).reshape(-1)
        nonzero = np.flatnonzero(stop_kind != 0)
        if nonzero.size:
            index = int(nonzero[0])
            return (
                f"its schedule has stop_kind {int(stop_kind[index])} at segment {index}, and the "
                "batched lanes stop a segment only when its duration has elapsed; set every "
                "stop_kind entry to 0"
            )
    for series in _lane_streams(child):
        if getattr(series, "likelihood", None) is None:
            continue
        spans = {span for _path, _leaf, span in _likelihood_leaves(series)}
        mask = getattr(series, "frame_mask", None)
        masked = mask is not None and not np.all(np.asarray(mask))
        if "spans" in spans or ("event" in spans and masked):
            return (
                f"its stream {series.name!r} declares a CorrelatedGaussian whose events couple "
                "frames, and a lane pads and masks whole frames; declare a covariance over the "
                "values of one frame, or over every frame of a stream of one value per frame "
                "that masks none of them"
            )
    return None


__all__ = [
    "COMPONENTS",
    "LANE_COMPONENTS",
    "LaneIdentity",
    "bucket_identity",
    "lane_identity",
    "lane_refusal",
]
