"""The plan of a study's lanes: its buckets, their sizes, and the lanes that get none.

Before anything is compiled, the lanes of a study are sorted into buckets: one
lane per experiment solved in one shoot, one per window of an experiment
fitted in windows that start from observed states. The lanes of one bucket
have equal identities
([lane_identity][diff_soup.inference.engine._identity.lane_identity]) and one
derivative width, fall in one frame class
([frame_classes][diff_soup.inference.engine._capacity.frame_classes]), and
either all follow a schedule or none does; a bucket runs through one compiled
program per mode. Lanes whose schedules have different numbers of segments
share a bucket: its segment class is the largest segment count among its
lanes, and every lane's schedule is padded to it with inactive segments, which
do not run. The plan records:

- per bucket, its identity, model, lanes, owners, frame slots, derivative
  width, segment class, padding and, per mode, how its units are cut into
  calls of the program;
- the experiments the batched lanes cannot take, each with its reason;
- per recipe whose experiments land in more than one bucket, every component
  that differs between those buckets and the template inputs that separate
  them. A component such as "model content" tells an author that a value the
  recipe captures differs between its experiments; declaring it a parameter
  lets them share a bucket.

Building a plan reads the fitted problems of the study and compiles and solves
nothing. The library prints nothing itself; ``str(plan)`` is a table.
"""

from __future__ import annotations

import dataclasses
import itertools
import numbers
import types
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import numpy as np

from diff_soup.inference.engine._capacity import (
    WASTE_BOUND,
    Frames,
    frame_classes,
    lane_frames,
)
from diff_soup.inference.engine._identity import (
    COMPONENTS,
    LaneIdentity,
    _IdentityRecorder,
    _sha256,
    bucket_identity,
    lane_identity,
    lane_refusal,
)
from diff_soup.inference.engine._kernel import MODES
from diff_soup.inference.engine._layout import (
    _experiment_problems,
    _lane_ids,
    lane_layout,
    lane_problems,
)

WIDTH_CAP: int = 1024
"""The widest chunk chosen when none is given: the widest batch that was measured."""


class Chunk(NamedTuple):
    """How the units of a bucket are cut into calls of its program in one mode.

    Attributes:
        width: ``W``, the units one call evaluates.
        chunks: ``k = ceil(n / W)``, the calls for the bucket's ``n`` units.
        padded: ``k W - n``, the copies of a unit that fill the last call;
            their results are dropped.
    """

    width: int
    chunks: int
    padded: int


class Refusal(NamedTuple):
    """An experiment the batched lanes cannot take.

    Attributes:
        lane: The experiment's id in the study.
        reason: What the lanes cannot take, and what would make them take it.
    """

    lane: Any
    reason: str


@dataclasses.dataclass(frozen=True, eq=False)
class Bucket:
    """Lanes that run through one compiled program per mode.

    Attributes:
        index: The bucket's position in the plan.
        identity: The first twelve hexadecimal characters of the bucket's
            identity.
        model: The name of the recipe or model of its lanes.
        lanes: The ids of its lanes, in the study's lane order (see
            [LaneLayout][diff_soup.inference.engine._layout.LaneLayout]): an
            experiment's id for its one lane, a window's own id for each lane
            of an experiment fitted in observed-start windows.
        owners: The owner of each lane, in the order of ``lanes``; ``None``
            for every lane of a study without an owner tier.
        frames: The frame class: union sample slots, then frame slots per
            stream.
        width: The derivative width.
        segments: The segment class: the number of segment entries every
            lane's schedule is padded to, the largest segment count among the
            bucket's lanes; ``None`` when its lanes follow no schedule.
        padding: The fraction of the bucket's frame slots, over its lanes and
            streams, that are padding.
        chunk: Per mode the program runs in, its
            [Chunk][diff_soup.inference.engine._plan.Chunk] for one member of
            a population; in a plan of
            [population_plan][diff_soup.inference.engine._plan.population_plan],
            for every member of the population.
        children: The index of each lane in the study's lane layout, in the
            order of ``lanes``; the experiment index when every experiment has
            one lane.
        full_identity: The bucket's whole
            [identity][diff_soup.inference.engine._identity.LaneIdentity].
    """

    index: int
    identity: str
    model: str
    lanes: tuple
    owners: tuple
    frames: Frames
    width: int
    segments: Any
    padding: float
    chunk: Mapping
    children: tuple
    full_identity: LaneIdentity = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True, eq=False)
class Split:
    """A recipe whose experiments land in more than one bucket.

    Attributes:
        recipe: The name of the recipe or model.
        buckets: The indices of the buckets that hold its experiments.
        differs: Every component that differs between those buckets, in the
            order of [COMPONENTS][diff_soup.inference.engine._identity.COMPONENTS].
        inputs: The template inputs that separate the buckets: each has one
            value within every bucket and not the same value in all of them.
    """

    recipe: str
    buckets: tuple
    differs: tuple
    inputs: tuple


@dataclasses.dataclass(frozen=True, eq=False)
class Plan:
    """The buckets of a study's lanes, built before anything is compiled.

    Attributes:
        buckets: The [Bucket][diff_soup.inference.engine._plan.Bucket]
            records, by the index of their first lane.
        refused: One [Refusal][diff_soup.inference.engine._plan.Refusal] per
            experiment that gets no bucket, by experiment index.
        splits: One [Split][diff_soup.inference.engine._plan.Split] per recipe
            whose experiments land in more than one bucket.
        device: The device the lanes run on; on a mesh, its first device,
            which holds the tables and level systems the lane results are
            reduced into.
        waste_bound: The waste bound of the frame classes.
        mesh: The mesh whose data axis the lanes are placed over, inside one
            process: every call of a program is split over the devices of
            that axis, and every chunk width is a multiple of their number.
            ``None`` runs every call on ``device``.
    """

    buckets: tuple
    refused: tuple
    splits: tuple
    device: Any
    waste_bound: float
    mesh: Any = None

    @property
    def devices(self) -> int:
        """The number of devices one call of a program is split over: 1 without a mesh."""
        return 1 if self.mesh is None else int(self.mesh.shape["data"])

    def __str__(self) -> str:
        header = ("bucket", "identity", "model", "lanes", "frames", "width", *MODES, "padding")
        rows = [header] + [_row(bucket) for bucket in self.buckets]
        sizes = [max(len(row[column]) for row in rows) for column in range(len(header))]
        lines = [
            "  ".join(cell.ljust(size) for cell, size in zip(row, sizes)).rstrip() for row in rows
        ]
        lines.append("")
        if self.mesh is not None:
            lines.append(f"placement: every call split over the {self.devices} devices of the mesh")
        lines.append(f"refused lanes: {len(self.refused) or 'none'}")
        lines += [f"  {entry.lane}: {entry.reason}" for entry in self.refused]
        lines.append(f"splits: {len(self.splits) or 'none'}")
        for split in self.splits:
            inputs = ", ".join(split.inputs) or "none"
            buckets = ", ".join(str(index) for index in split.buckets)
            lines.append(
                f"  {split.recipe}, buckets {buckets}: differ in {', '.join(split.differs)}; "
                f"template inputs {inputs}"
            )
        return "\n".join(lines)


def _row(bucket: Bucket) -> tuple:
    """One table row describing ``bucket``: its lanes and the chunking of each mode."""
    lanes = [str(lane) for lane in bucket.lanes]
    shown = ", ".join(lanes if len(lanes) <= 4 else lanes[:3] + ["...", lanes[-1]])
    cells = []
    for mode in MODES:
        entry = bucket.chunk.get(mode)
        if entry is None:
            cells.append("-")
        elif entry.padded:
            cells.append(f"W {entry.width} k {entry.chunks} +{entry.padded}")
        else:
            cells.append(f"W {entry.width} k {entry.chunks}")
    return (
        str(bucket.index),
        bucket.identity,
        bucket.model,
        f"{len(lanes)}: {shown}",
        f"{bucket.frames.union} | {', '.join(str(count) for count in bucket.frames.streams)}",
        str(bucket.width),
        *cells,
        f"{bucket.padding:.2f}",
    )


def _model_name(model: Any) -> str:
    """A display name for ``model``: its ``__name__`` or ``name``, else its type's name."""
    for attribute in ("__name__", "name"):
        name = getattr(model, attribute, None)
        if isinstance(name, str) and name:
            return name
    return type(model).__name__


def _recipes(experiment: Any) -> tuple:
    """The recipes an experiment's model was built from, as ``(key, name)`` pairs.

    An experiment holds the model its recipe built from its inputs, not the
    recipe. A recipe build is registered with the recipe's handles under the
    recipe's name, so the recipe is found by that name. A model built without
    a recipe is its own recipe.
    """
    from diff_soup.problem._registry import parameter_handles
    from diff_soup.problem._spec.recipe import _TemplateHandle

    model = experiment.model
    names = sorted(
        str(namespace)
        for namespace, handles in parameter_handles(model).items()
        if any(isinstance(handle, _TemplateHandle) for handle in handles)
    )
    if names:
        return tuple((("recipe", name), name) for name in names)
    return ((("model", id(model)), _model_name(model)),)


def _chunk_width(chunk: Any) -> int | None:
    """``chunk`` validated as a positive integer width, or ``None`` for the default."""
    if chunk is None:
        return None
    if isinstance(chunk, bool) or not isinstance(chunk, numbers.Integral) or chunk < 1:
        raise ValueError(
            "chunk= is the number of units one call of a program evaluates, a positive integer, "
            f"or None to choose it; got {chunk!r}."
        )
    return int(chunk)


def n_padded(n: int, n_devices: int) -> int:
    """Smallest multiple of ``n_devices`` that is at least ``n``.

    A batch split over the devices of a mesh's data axis is padded to this
    size, so every device takes the same number of rows.
    """
    return ((n + n_devices - 1) // n_devices) * n_devices


def _chunks(units: int, modes: tuple, forced: int | None, devices: int = 1) -> Mapping:
    """The chunking of ``units`` per mode: the width (``forced`` or the next power of two, padded to ``devices``), the count and the padding."""
    width = forced if forced is not None else min(1 << (units - 1).bit_length(), WIDTH_CAP)
    width = n_padded(width, devices)
    count = -(-units // width)
    entry = Chunk(width=width, chunks=count, padded=count * width - units)
    return types.MappingProxyType({mode: entry for mode in modes})


def _modes(child: Any) -> tuple:
    """The modes the program of ``child`` runs in.

    A solver with a reverse adjoint has no ``blocks`` mode. The ``hessian`` mode
    needs an experiment whose adjoints differentiate its gradient forward over
    reverse.
    """
    adjoint = getattr(child.solver, "adjoint", None)
    reverse = not bool(getattr(adjoint, "is_forward_only", True))
    second = "forward-over-reverse" in getattr(child, "supported_compositions", frozenset())
    return tuple(
        mode
        for mode in MODES
        if not (reverse and mode == "blocks") and (second or mode != "hessian")
    )


def _segment_count(child: Any) -> int | None:
    """The number of segment entries of an experiment's schedule; ``None`` without one.

    Reads the schedule's arrays on the host, as the lanes carry them.
    """
    schedule = getattr(child, "_schedule_arrays", None)
    return None if schedule is None else int(np.shape(schedule.mode)[0])


def _padding(slots: Frames, lanes: list) -> float:
    """The fraction of the frame slots the lanes leave unused."""
    total = len(lanes) * sum(slots.streams)
    if not total:
        return 0.0
    used = sum(sum(lane.streams) for lane in lanes)
    return float(total - used) / float(total)


def _separating_inputs(groups: list, experiments: tuple) -> tuple:
    """The inputs with one value within every group of experiments, not one value in all.

    ``groups`` holds, per bucket of a split, the experiment indices of the
    recipe's experiments in it.
    """
    recorder = _IdentityRecorder()
    missing = {"kind": "missing"}

    def value(child: int, name: str) -> str:
        inputs = experiments[child].inputs
        return _sha256(recorder.record(inputs[name], path=name) if name in inputs else missing)

    names = sorted(
        {str(name) for group in groups for child in group for name in experiments[child].inputs}
    )
    separating = []
    for name in names:
        per_group = [{value(child, name) for child in group} for group in groups]
        if all(len(values) == 1 for values in per_group) and len(set().union(*per_group)) > 1:
            separating.append(name)
    return tuple(separating)


def _splits(buckets: tuple, experiments: tuple) -> tuple:
    """The recipe splits of the buckets' children, one entry per distinct recipe key."""
    recipes: dict[tuple, tuple] = {}
    for bucket in buckets:
        for child in bucket.children:
            for key, name in _recipes(experiments[child]):
                _, members = recipes.setdefault(key, (name, {}))
                members.setdefault(bucket.index, []).append(child)
    splits = []
    for name, members in recipes.values():
        if len(members) < 2:
            continue
        indices = tuple(sorted(members))
        differing = set()
        for one, two in itertools.combinations(indices, 2):
            differing.update(buckets[one].full_identity.differs(buckets[two].full_identity))
        splits.append(
            Split(
                recipe=name,
                buckets=indices,
                differs=tuple(component for component in COMPONENTS if component in differing),
                inputs=_separating_inputs([members[index] for index in indices], experiments),
            )
        )
    return tuple(splits)


def _linear_backend(child: Any, platform: str) -> Any:
    """The linear backend an experiment's solver resolves to on ``platform``.

    The solver's ``linear`` setting resolves as its engine resolves it: a
    backend instance as it is, a name by the experiment's state size and the
    platform. ``None`` for a solver without a linear backend, and for a name
    this host cannot serve, whose solve reports why.
    """
    from diff_soup.linear import LinearSolver
    from diff_soup.linear._resolver import BackendResolver
    from diff_soup.timestepping.vsvo.common.backends import resolve_backend
    from diff_soup.timestepping.vsvo.common.contract import coerce_problem

    linear = getattr(getattr(child, "solver", None), "linear", None)
    if linear is None or isinstance(linear, LinearSolver):
        return linear
    try:
        size = int(coerce_problem(child._seam.problem).n_u)
        return resolve_backend(
            BackendResolver(backend=linear).choose(block_size=size, platform=platform)
        )
    except (RuntimeError, ValueError):
        return None


def refuse_shared_storage(problems: tuple, mesh: Any) -> None:
    """Refuse native factor storage that the devices of a mesh's data axis would share.

    The experiments placed over the devices of a data axis run in one
    execution, and a backend whose factors are not arrays (``KLU``,
    ``SuperLU``, ``CuDSS``, or ``Woodbury`` or ``Krylov`` over one of them)
    keeps them in this process's storage, which does not tell the devices
    apart. A data axis with one device in this process, and backends whose
    factors are arrays, are accepted. The refusal names the backend as
    ``diff_soup.linear`` exports it (``KLU``, ``Woodbury over KLU``). On a
    mesh that also has a ``model`` axis of several devices, each entry of the
    data axis is a model group whose factorization only ``CuDSS`` serves
    (``Dense`` is refused there), so the remedy the refusal names is one
    model group per process: a launch of one process per entry of the data
    axis.

    Args:
        problems: The fitted problem of every placed experiment.
        mesh: The mesh whose ``data`` axis places them.

    Raises:
        ConfigurationError: If this process holds two or more entries of the
            data axis (devices, or model groups on a mesh with a ``model``
            axis) and an experiment's solver resolves to a backend whose
            factors are not arrays.
    """
    from diff_soup._errors import ConfigurationError
    from diff_soup.linear._api import public_backend_name

    if "data" not in mesh.axis_names:
        return
    rows = np.moveaxis(np.asarray(mesh.devices), mesh.axis_names.index("data"), 0)
    rows = rows.reshape(rows.shape[0], -1)
    here = [row for row in rows if any(d.process_index == jax.process_index() for d in row)]
    if len(here) < 2:
        return
    platform = next(d.platform for d in here[0] if d.process_index == jax.process_index())
    model = int(mesh.shape.get("model", 1))
    for child in problems:
        backend = _linear_backend(child, platform)
        residency = getattr(backend, "factor_residency", "arrays")
        if residency == "arrays":
            continue
        name = public_backend_name(backend)
        where = "" if len(here) == len(rows) else " on this process"
        if model > 1:
            raise ConfigurationError(
                f"{name} keeps its factors in {residency} storage that the {len(here)} model "
                f"groups along the layout's data axis{where} would share, and the layout's "
                f"model axis of {model} devices factors with CuDSS only (Dense is refused on "
                "it); place one model group per process (a launch of one process per entry of "
                "the data axis), or use a layout whose data axis has size 1."
            )
        raise ConfigurationError(
            f"{name} keeps its factors in {residency} storage that the {len(here)} devices of "
            f"the layout's data axis{where} would share; use Dense, whose factors are arrays, "
            f"or a layout whose data axis has one device{' on each process' if where else ''}."
        )


def _placement_mesh(joint: Any, device: Any) -> Any:
    """The mesh whose data axis places the lanes inside this process; ``None`` for one device.

    The placement is the [Layout][diff_soup.Layout] the fitted problem was
    built with: a mesh whose ``data`` axis has more than one device places the
    lanes over those devices. A layout whose ``processes`` differs from the
    number of processes of the run, a mesh placement in a run of several
    processes, and a mesh holding another process's devices are refused. A
    placement over several devices takes backends whose factors are arrays
    (see [refuse_shared_storage][diff_soup.inference.engine._plan.refuse_shared_storage]).
    """
    from diff_soup._errors import ConfigurationError
    from diff_soup.inference.engine._transport import world

    layout = getattr(joint, "layout", None)
    size = world().size
    expected = getattr(layout, "processes", None)
    if expected is not None and expected != size:
        raise ConfigurationError(
            f"the layout expects a run of {expected} process(es), and this run has {size}; "
            f"launch {expected} process(es), or build the layout with processes=None to accept "
            "any run."
        )
    mesh = getattr(layout, "mesh", None)
    if mesh is None or int(mesh.shape.get("data", 1)) < 2:
        return None
    if size > 1:
        raise ConfigurationError(
            f"the layout places the lanes over the {int(mesh.shape['data'])} devices of its "
            f"mesh's data axis, and this run places them over {size} processes; a study takes "
            "one data placement at a time: lower it with a mesh whose data axis has one device, "
            "or run one process."
        )
    foreign = sorted(
        {d.process_index for d in mesh.devices.flat if d.process_index != jax.process_index()}
    )
    if foreign:
        raise ConfigurationError(
            f"the layout's mesh holds devices of the processes {foreign}, and the lanes of a "
            "study are placed over the devices of one process; lower it with a mesh of this "
            "process's devices (jax.local_devices()), or without a mesh."
        )
    if device is not None and device not in set(mesh.devices.flat):
        raise ValueError(
            f"device={device!r} is not a device of the layout's mesh, which places the lanes; "
            "give one of its devices, or None for its first."
        )
    refuse_shared_storage(_experiment_problems(joint), mesh)
    return mesh


def build_plan(study: Any, joint: Any, *, chunk: Any = None, device: Any = None) -> Plan:
    """The plan of a study's lanes, built without a compile or a solve.

    Lanes of equal identity and derivative width that all follow a schedule,
    or all follow none, are cut into frame classes; each class is a bucket.
    The segment class of a bucket whose lanes follow schedules is the largest
    segment count among them.

    The lanes are placed by the [Layout][diff_soup.Layout] ``joint`` was
    built with: a mesh whose ``data`` axis has ``n > 1`` devices splits every
    call of every program over those devices, and every chunk width is
    rounded up to a multiple of ``n``; without one, every call runs on
    ``device``.

    Args:
        study: The [Study][diff_soup.inference.Study]; its experiment ids name
            the lanes: the one lane of an experiment solved in one shoot has
            the experiment's id, the windows of an experiment fitted in
            observed-start windows have ids of their own.
        joint: The fitted problem built from ``study``: its joint problem,
            whose experiment ``i`` is the study's experiment ``i``, or the
            fitted problem of its one experiment.
        chunk: The units one call of every program evaluates, taken exactly
            (on a mesh, rounded up to a multiple of its data axis); ``None``
            chooses the smallest power of two that holds a bucket's units, at
            most 1,024.
        device: The JAX device the lanes run on; ``None`` takes this
            process's first device, ``jax.local_devices()[0]``, or on a mesh
            its first device.

    Returns:
        The [Plan][diff_soup.inference.engine._plan.Plan].

    Raises:
        ValueError: If ``chunk`` is not a positive integer, ``joint`` holds
            another number of experiments than ``study``, or ``device`` is
            not a device of the placing mesh.
        ConfigurationError: If the layout expects another number of
            processes than the run has, places the lanes over the devices
            of a mesh in a run of several processes, its mesh holds
            another process's devices, or it places them over several
            devices and an experiment's solver keeps its factors in native
            storage.
    """
    forced = _chunk_width(chunk)
    ids = tuple(study.experiment_ids)
    mesh = _placement_mesh(joint, device)
    if device is None:
        device = jax.local_devices()[0] if mesh is None else mesh.devices.flat[0]
    devices = 1 if mesh is None else int(mesh.shape["data"])
    every = _experiment_problems(joint)
    if len(every) != len(ids):
        raise ValueError(
            f"the fitted problem holds {len(every)} experiments and the study {len(ids)}; "
            "build the plan from the fitted problem of this study."
        )
    layout = lane_layout(joint)
    problems = lane_problems(joint)
    names, _ = _lane_ids(ids, every)
    declared = tuple(study.experiments)
    experiments = tuple(declared[index] for index in layout.experiments)
    widths = {index: len(columns) for index, columns in enumerate(layout.columns)}
    owners = {
        index: layout.labels[node] if layout.labels else None
        for index, node in enumerate(layout.nodes)
    }
    lanes_of: dict = {}
    for index, experiment in enumerate(layout.experiments):
        lanes_of.setdefault(experiment, []).append(index)

    refused, groups, identities, counts = [], {}, {}, {}
    for experiment, child in enumerate(every):
        reason = lane_refusal(child)
        if reason is not None:
            refused.append(Refusal(ids[experiment], reason))
            continue
        identity = lane_identity(child)
        for index in lanes_of[experiment]:
            counts[index] = _segment_count(child)
            # A lane that follows a schedule runs another program than one that follows none.
            key = (identity.digest, widths[index], counts[index] is not None)
            identities.setdefault(key, identity)
            groups.setdefault(key, []).append(index)

    drafts = []
    for key, members in groups.items():
        own = [lane_frames(problems[index]) for index in members]
        for slots, positions in frame_classes(own):
            children = tuple(members[position] for position in positions)
            drafts.append((children, slots, key[1], identities[key], [own[p] for p in positions]))
    drafts.sort(key=lambda draft: draft[0][0])

    buckets = []
    for index, (children, slots, width, identity, own) in enumerate(drafts):
        scheduled = [counts[child] for child in children if counts[child] is not None]
        segments = max(scheduled) if scheduled else None
        full = bucket_identity(identity, frames=slots, width=width, segments=segments)
        buckets.append(
            Bucket(
                index=index,
                identity=str(full),
                model=_recipes(experiments[children[0]])[0][1],
                lanes=tuple(names[child] for child in children),
                owners=tuple(owners[child] for child in children),
                frames=slots,
                width=int(width),
                segments=segments,
                padding=_padding(slots, own),
                chunk=_chunks(len(children), _modes(problems[children[0]]), forced, devices),
                children=children,
                full_identity=full,
            )
        )
    buckets = tuple(buckets)
    return Plan(
        buckets, tuple(refused), _splits(buckets, experiments), device, WASTE_BOUND, mesh=mesh
    )


def population_plan(plan: Plan, members: int, *, chunk: Any = None) -> Plan:
    """A study's plan cut for requests of ``members`` points at once.

    A request of ``P`` points runs ``P`` units per lane of a bucket, so a
    bucket of ``L`` lanes has ``P L`` units to cut into calls of its program.
    The returned plan has the buckets of ``plan`` (the same lanes, owners,
    frames, identities and modes, in the same order), the same refused lanes
    and splits; each bucket's [Chunk][diff_soup.inference.engine._plan.Chunk]
    counts the ``P L`` units: the width the rule of
    [build_plan][diff_soup.inference.engine._plan.build_plan] chooses for
    them, or ``chunk``, then the calls and the copies that fill the last one.

    Args:
        plan: The plan of the study, for one point per request.
        members: ``P``, the points of one request, a positive integer.
        chunk: The units one call of every program evaluates, taken exactly
            (on a mesh, rounded up to a multiple of its data axis); ``None``
            chooses, per bucket, the smallest power of two that holds
            its ``P L`` units, at most 1,024.

    Returns:
        The [Plan][diff_soup.inference.engine._plan.Plan] of the population.

    Raises:
        ValueError: If ``members`` is not a positive integer, or ``chunk`` is
            not a positive integer or ``None``.
    """
    members = _member_count(members)
    forced = _chunk_width(chunk)
    buckets = tuple(
        dataclasses.replace(
            bucket,
            chunk=_chunks(members * len(bucket.lanes), tuple(bucket.chunk), forced, plan.devices),
        )
        for bucket in plan.buckets
    )
    return dataclasses.replace(plan, buckets=buckets)


def _member_count(members: Any) -> int:
    """``members`` as an int, refusing anything but a positive integer."""
    if isinstance(members, bool) or not isinstance(members, numbers.Integral) or members < 1:
        raise ValueError(
            "members= is the number of points one request of the population evaluates, a "
            f"positive integer; got {members!r}."
        )
    return int(members)


__all__ = [
    "WIDTH_CAP",
    "Bucket",
    "Chunk",
    "Plan",
    "Refusal",
    "Split",
    "build_plan",
    "n_padded",
    "population_plan",
    "refuse_shared_storage",
]
