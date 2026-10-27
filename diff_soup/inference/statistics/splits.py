"""Leakage-safe grouped cross-validation splits for experiment campaigns.

The splitting unit in a hierarchical campaign is an owner, not an individual
response row.  This module therefore keeps the statistical partition and its
provenance together: every [CrossValidationSplit][diff_soup.inference.statistics.splits.CrossValidationSplit] contains immutable
training/held-out [ExperimentSet][diff_soup.inference.statistics.experiment.ExperimentSet]
views, a scalar-row manifest for the held-out likelihood events, and a
separate manifest of the inputs on which those predictions are conditioned.

Response manifests preserve channel event boundaries.  A masked image channel,
for example, is one likelihood event with several scalar row identifiers; it
is never represented as unrelated scalar pseudo-observations.  This matters
for correlated and count likelihoods and gives downstream model comparison a
precise, auditable no-leakage contract.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "ConditioningRecord",
    "CrossValidationSplit",
    "HeldOutResponseGroup",
    "grouped_kfold",
    "leave_one_entity_out",
    "one_standard_error",
    "within_entity_holdout",
]


RowId = tuple[str, str, str | None, int, tuple[int, ...]]


@dataclasses.dataclass(frozen=True)
class HeldOutResponseGroup:
    """One data-bearing response channel kept as one likelihood event.

    Attributes:
        experiment_id: Stable ID of the experiment owning the response.
        channel_name: Declared response-channel name.
        source_id: Stable source identifier, when one was declared.
        kept_frame_indices: Indices into the original, pre-mask channel.
        kept_times: Observation times corresponding to the kept frames.
        observed_shape: Shape of the complete kept likelihood event.
        row_ids: Stable identifiers for every scalar in the event.  The final
            component is the scalar's trailing index within a frame.
    """

    experiment_id: str
    channel_name: str
    source_id: str | None
    kept_frame_indices: tuple[int, ...]
    kept_times: tuple[float, ...]
    observed_shape: tuple[int, ...]
    row_ids: tuple[RowId, ...]


@dataclasses.dataclass(frozen=True)
class ConditioningRecord:
    """Provenance for one input available while predicting a held-out child.

    ``value_digest`` is a content digest rather than the value itself.  It
    makes equality of conditioning manifests meaningful without putting large
    arrays in split metadata.
    """

    experiment_id: str
    kind: str
    name: str
    role: str
    source_id: str | None = None
    channel_name: str | None = None
    observed_shape: tuple[int, ...] | None = None
    value_digest: str | None = None


@dataclasses.dataclass(frozen=True)
class CrossValidationSplit:
    """One deterministic, leakage-audited experiment-set partition.

    Attributes:
        cohort: Stable identifier for the original experiment cohort.
        split_id: Stable identifier for this particular partition.
        level: Partition level, currently ``"owner"`` or ``"child"``.
        entity: Held-out owner for entity/child splits; a tuple of owners for
            a grouped fold.
        train_indices: Original cohort indices included in training.
        held_out_indices: Original cohort indices included in evaluation.
        train: Experiment-set subset containing only ``train_indices``.
        held_out: Experiment-set subset containing only
            ``held_out_indices``.  Its response data remains attached for
            evaluation, but it is never part of ``train``.
        held_out_response_manifest: Event-aware manifest of response rows.
        conditioning_manifest: Inputs available to held-out predictions.
        train_response_row_ids: Scalar response identifiers used in training,
            retained so downstream scoring can assert disjointness.
    """

    cohort: str
    split_id: str
    level: str
    entity: Any
    train_indices: tuple[int, ...]
    held_out_indices: tuple[int, ...]
    train: Any
    held_out: Any
    held_out_response_manifest: tuple[HeldOutResponseGroup, ...]
    conditioning_manifest: tuple[ConditioningRecord, ...]
    train_response_row_ids: tuple[RowId, ...]


def _as_host_array(value: Any) -> np.ndarray:
    """Return a concrete host array or raise a provenance-specific error."""
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Cross-validation manifests require concrete array-valued data; "
            "traced values cannot define stable row identities."
        ) from exc


def _value_digest(value: Any) -> tuple[tuple[int, ...], str] | tuple[None, None]:
    """Return ``(shape, digest)`` for a conditioning value."""
    if value is None:
        return None, None
    array = _as_host_array(value)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(repr(tuple(int(i) for i in contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return tuple(int(i) for i in contiguous.shape), digest.hexdigest()


def _experiment_ids(experiment_set: Any) -> tuple[str, ...]:
    """Validate and return stable experiment IDs from an experiment set."""
    experiments = tuple(getattr(experiment_set, "experiments", ()))
    if not experiments:
        raise ValueError("Cross-validation requires at least one experiment.")
    try:
        ids = tuple(str(value) for value in experiment_set.experiment_ids)
    except AttributeError as exc:
        raise TypeError(
            "Cross-validation requires an ExperimentSet with stable experiment_ids."
        ) from exc
    if len(ids) != len(experiments):
        raise ValueError(
            "ExperimentSet experiment_ids length must match its experiments; "
            f"got {len(ids)} IDs for {len(experiments)} experiments."
        )
    if len(set(ids)) != len(ids):
        raise ValueError("ExperimentSet experiment IDs must be unique.")
    return ids


def _cohort_id(experiment_set: Any, experiment_ids: tuple[str, ...]) -> str:
    """Return an explicit cohort label or a stable digest-derived fallback."""
    declared = getattr(experiment_set, "cohort", None)
    if declared is not None:
        return str(declared)
    digest = hashlib.sha256()
    for experiment_id in experiment_ids:
        encoded = experiment_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"experiment-set:{digest.hexdigest()[:16]}"


def _mapping_values(mapping: Any, n_experiments: int, *, group: str) -> tuple[Any, ...]:
    """Normalize one ownership declaration to an aligned label tuple."""
    if isinstance(mapping, Mapping):
        expected = set(range(n_experiments))
        keys = set(mapping)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"ownership group {group!r} must label every experiment index; "
                f"missing={missing}, extra={extra}."
            )
        return tuple(mapping[i] for i in range(n_experiments))
    if isinstance(mapping, (str, bytes)):
        raise TypeError(
            f"ownership group {group!r} must be an index mapping or sequence, not text."
        )
    try:
        labels = tuple(mapping)
    except TypeError as exc:
        raise TypeError(f"ownership group {group!r} must be an index mapping or sequence.") from exc
    if len(labels) != n_experiments:
        raise ValueError(
            f"ownership group {group!r} has {len(labels)} labels for {n_experiments} experiments."
        )
    return labels


def _owner_labels(experiment_set: Any, *, by: str = "owner") -> tuple[Any, ...]:
    """Resolve the requested ownership partition to one label per experiment."""
    n_experiments = len(_experiment_ids(experiment_set))
    ownership = getattr(experiment_set, "ownership", None)
    if callable(ownership):
        labels = tuple(ownership(by, index) for index in range(n_experiments))
        if any(label is None for label in labels):
            suffix = (
                " Pass by='<owned parameter group>' when the callable does not "
                "define the common 'owner' label."
                if by == "owner"
                else ""
            )
            raise ValueError(
                f"ownership callable did not define a complete {by!r} partition.{suffix}"
            )
        return labels
    if not isinstance(ownership, Mapping) or not ownership:
        raise ValueError(
            "Grouped cross-validation requires a non-empty ExperimentSet ownership mapping."
        )

    if by != "owner":
        if by not in ownership:
            raise ValueError(
                f"unknown ownership group {by!r}; available groups are "
                f"{tuple(str(group) for group in ownership)}."
            )
        return _mapping_values(ownership[by], n_experiments, group=str(by))

    if "owner" in ownership:
        return _mapping_values(ownership["owner"], n_experiments, group="owner")

    declared = tuple(ownership.items())
    reference_name, reference_mapping = declared[0]
    reference = _mapping_values(reference_mapping, n_experiments, group=str(reference_name))
    for group, mapping in declared[1:]:
        labels = _mapping_values(mapping, n_experiments, group=str(group))
        if labels != reference:
            raise ValueError(
                "by='owner' is ambiguous because ownership groups induce "
                "different child partitions; select an explicit ownership "
                f"group from {tuple(str(name) for name in ownership)}."
            )
    return reference


def _unique_in_order(values: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return equality-unique labels in first-occurrence order."""
    unique: list[Any] = []
    for value in values:
        if not any(value == previous for previous in unique):
            unique.append(value)
    return tuple(unique)


def _indices_for_label(labels: tuple[Any, ...], entity: Any) -> tuple[int, ...]:
    """Return original indices belonging to ``entity``."""
    return tuple(i for i, label in enumerate(labels) if label == entity)


def _response_groups(
    experiment_set: Any, indices: tuple[int, ...]
) -> tuple[HeldOutResponseGroup, ...]:
    """Build event-aware response manifests for original experiment indices."""
    experiment_ids = _experiment_ids(experiment_set)
    experiments = tuple(experiment_set.experiments)
    groups: list[HeldOutResponseGroup] = []
    for index in indices:
        experiment = experiments[index]
        experiment_id = experiment_ids[index]
        data = getattr(experiment, "data", {})
        for channel in tuple(getattr(experiment, "channels", ())):
            if getattr(channel, "role", "response") != "response":
                continue
            channel_name = str(channel.name)
            if channel_name not in data:
                continue

            observed = _as_host_array(data[channel_name])
            times = _as_host_array(channel.saveat)
            if observed.ndim < 1:
                raise ValueError(
                    f"response {experiment_id!r}/{channel_name!r} must have a "
                    "leading frame dimension."
                )
            if observed.shape[0] != times.shape[0]:
                raise ValueError(
                    f"response {experiment_id!r}/{channel_name!r} has "
                    f"{observed.shape[0]} frames but declares {times.shape[0]} times."
                )

            mask = getattr(channel, "mask", None)
            if mask is None:
                kept = np.arange(times.shape[0], dtype=int)
            else:
                concrete_mask = _as_host_array(mask)
                if concrete_mask.shape != times.shape or concrete_mask.dtype != np.bool_:
                    raise ValueError(
                        f"response mask for {experiment_id!r}/{channel_name!r} "
                        f"must be boolean with shape {times.shape}; got "
                        f"shape {concrete_mask.shape}, dtype {concrete_mask.dtype}."
                    )
                kept = np.flatnonzero(concrete_mask)

            kept_observed = observed[kept]
            trailing_shape = tuple(int(size) for size in observed.shape[1:])
            trailing_indices = tuple(np.ndindex(trailing_shape))
            source_id = getattr(channel, "source_id", None)
            row_ids = tuple(
                (
                    experiment_id,
                    channel_name,
                    source_id,
                    int(frame),
                    tuple(int(i) for i in trailing),
                )
                for frame in kept
                for trailing in trailing_indices
            )
            groups.append(
                HeldOutResponseGroup(
                    experiment_id=experiment_id,
                    channel_name=channel_name,
                    source_id=source_id,
                    kept_frame_indices=tuple(int(i) for i in kept),
                    kept_times=tuple(float(times[i]) for i in kept),
                    observed_shape=tuple(int(size) for size in kept_observed.shape),
                    row_ids=row_ids,
                )
            )
    return tuple(groups)


def _conditioning_records(
    experiment_set: Any, indices: tuple[int, ...]
) -> tuple[ConditioningRecord, ...]:
    """Build deterministic held-out conditioning provenance."""
    from diff_soup.inference.fitting.inference_problem import _initial_state_array

    experiment_ids = _experiment_ids(experiment_set)
    experiments = tuple(experiment_set.experiments)
    records: list[ConditioningRecord] = []
    for index in indices:
        experiment = experiments[index]
        experiment_id = experiment_ids[index]
        data = getattr(experiment, "data", {})
        for channel in tuple(getattr(experiment, "channels", ())):
            role = str(getattr(channel, "role", "response"))
            if role == "response":
                continue
            name = str(channel.name)
            shape, digest = _value_digest(data.get(name))
            records.append(
                ConditioningRecord(
                    experiment_id=experiment_id,
                    kind="channel",
                    name=name,
                    role=role,
                    source_id=getattr(channel, "source_id", None),
                    channel_name=name,
                    observed_shape=shape,
                    value_digest=digest,
                )
            )

        controls = getattr(experiment, "controls", {})
        if not isinstance(controls, Mapping):
            raise TypeError(
                f"Experiment {experiment_id!r} controls must be a mapping for "
                "conditioning provenance."
            )
        for name in sorted(controls, key=str):
            shape, digest = _value_digest(controls[name])
            records.append(
                ConditioningRecord(
                    experiment_id=experiment_id,
                    kind="control",
                    name=str(name),
                    role="control",
                    observed_shape=shape,
                    value_digest=digest,
                )
            )

        shape, digest = _value_digest(
            _initial_state_array(getattr(experiment, "initial_state", None))
        )
        records.append(
            ConditioningRecord(
                experiment_id=experiment_id,
                kind="initial_state",
                name="initial_state",
                role="initial_condition",
                observed_shape=shape,
                value_digest=digest,
            )
        )
    return tuple(records)


def _response_row_ids(experiment_set: Any, indices: tuple[int, ...]) -> tuple[RowId, ...]:
    """Flatten response-group row IDs while retaining their declared order."""
    return tuple(
        row for group in _response_groups(experiment_set, indices) for row in group.row_ids
    )


def _subset(experiment_set: Any, indices: tuple[int, ...]) -> Any:
    """Take an immutable subset through the ExperimentSet public seam."""
    subset = getattr(experiment_set, "subset", None)
    if not callable(subset):
        raise TypeError(
            "Cross-validation requires ExperimentSet.subset(indices) so child "
            "ownership can be remapped without changing stable owner IDs."
        )
    return subset(indices)


def _make_split(
    experiment_set: Any,
    *,
    train_indices: tuple[int, ...],
    held_out_indices: tuple[int, ...],
    split_id: str,
    level: str,
    entity: Any,
) -> CrossValidationSplit:
    """Construct one split and assert its row-level leakage invariant."""
    experiment_ids = _experiment_ids(experiment_set)
    all_indices = set(range(len(experiment_ids)))
    train_set = set(train_indices)
    held_set = set(held_out_indices)
    if not held_set:
        raise ValueError("A cross-validation split must hold out at least one experiment.")
    if train_set & held_set:
        raise ValueError("Training and held-out experiment indices must be disjoint.")
    if train_set | held_set != all_indices:
        raise ValueError("Training and held-out indices must partition the experiment cohort.")

    held_groups = _response_groups(experiment_set, held_out_indices)
    train_rows = _response_row_ids(experiment_set, train_indices)
    held_rows = tuple(row for group in held_groups for row in group.row_ids)
    overlap = set(train_rows) & set(held_rows)
    if overlap:
        raise ValueError(
            "Response-row leakage detected between training and held-out manifests; "
            f"overlapping row IDs include {tuple(sorted(overlap, key=repr))[:3]}."
        )

    return CrossValidationSplit(
        cohort=_cohort_id(experiment_set, experiment_ids),
        split_id=split_id,
        level=level,
        entity=entity,
        train_indices=train_indices,
        held_out_indices=held_out_indices,
        train=_subset(experiment_set, train_indices),
        held_out=_subset(experiment_set, held_out_indices),
        held_out_response_manifest=held_groups,
        conditioning_manifest=_conditioning_records(experiment_set, held_out_indices),
        train_response_row_ids=train_rows,
    )


def grouped_kfold(experiment_set: Any, k: int, *, by: str = "owner") -> tuple:
    """Partition an experiment set into deterministic owner-grouped folds.

    No ownership entity is ever split between training and evaluation.  Owner
    groups are assigned greedily by descending child count to keep fold sizes
    balanced; ties retain declaration order, so repeated calls are identical.

    Args:
        experiment_set: An ExperimentSet with stable IDs and ownership.
        k: Number of folds, between two and the number of ownership entities.
        by: ``"owner"`` for the common aligned owner partition, or a named
            ownership group when several distinct partitions are declared.

    Returns:
        A tuple of [CrossValidationSplit][diff_soup.inference.statistics.splits.CrossValidationSplit] objects.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer; got {type(k).__name__}.")
    labels = _owner_labels(experiment_set, by=by)
    entities = _unique_in_order(labels)
    if k < 2 or k > len(entities):
        raise ValueError(
            f"k must be between 2 and the number of ownership entities ({len(entities)}); got {k}."
        )

    members = tuple(_indices_for_label(labels, entity) for entity in entities)
    assignment_order = sorted(range(len(entities)), key=lambda i: (-len(members[i]), i))
    fold_entity_indices: list[list[int]] = [[] for _ in range(int(k))]
    fold_sizes = [0 for _ in range(int(k))]
    for entity_index in assignment_order:
        fold_index = min(range(int(k)), key=lambda i: (fold_sizes[i], i))
        fold_entity_indices[fold_index].append(entity_index)
        fold_sizes[fold_index] += len(members[entity_index])

    all_indices = tuple(range(len(labels)))
    splits = []
    for fold_index, entity_indices in enumerate(fold_entity_indices):
        entity_indices.sort()
        held_entities = tuple(entities[i] for i in entity_indices)
        held_set = {index for i in entity_indices for index in members[i]}
        held = tuple(i for i in all_indices if i in held_set)
        train = tuple(i for i in all_indices if i not in held_set)
        splits.append(
            _make_split(
                experiment_set,
                train_indices=train,
                held_out_indices=held,
                split_id=f"grouped-{by}:fold-{fold_index + 1}-of-{k}",
                level="owner",
                entity=held_entities,
            )
        )
    return tuple(splits)


def leave_one_entity_out(experiment_set: Any, *, by: str = "owner") -> tuple:
    """Return one owner-level fold per entity, in first-occurrence order.

    ``by`` has the same semantics as [grouped_kfold][diff_soup.inference.statistics.splits.grouped_kfold]; naming an owned
    parameter group is required when a callable declaration does not expose a
    common ``"owner"`` pseudo-group.
    """
    labels = _owner_labels(experiment_set, by=by)
    entities = _unique_in_order(labels)
    if len(entities) < 2:
        raise ValueError("Leave-one-entity-out requires at least two ownership entities.")
    all_indices = tuple(range(len(labels)))
    splits = []
    for entity in entities:
        held = _indices_for_label(labels, entity)
        held_set = set(held)
        train = tuple(i for i in all_indices if i not in held_set)
        splits.append(
            _make_split(
                experiment_set,
                train_indices=train,
                held_out_indices=held,
                split_id=f"leave-one-owner-out:{entity!s}",
                level="owner",
                entity=entity,
            )
        )
    return tuple(splits)


def within_entity_holdout(experiment_set: Any, entity: Any, *, by: str = "owner") -> tuple:
    """Hold out each child of a multi-child owner in turn.

    The held-out child remains intact in ``split.held_out`` so its conditioning
    inputs and response likelihood event are available for evaluation.  It is
    absent from ``split.train``; a sibling of the same owner therefore remains
    the source of owner-level information during every fold. ``by`` selects
    the ownership partition exactly as in [grouped_kfold][diff_soup.inference.statistics.splits.grouped_kfold].
    """
    labels = _owner_labels(experiment_set, by=by)
    entities = _unique_in_order(labels)
    if not any(entity == declared for declared in entities):
        raise ValueError(
            f"unknown ownership entity {entity!r}; available entities are {entities!r}."
        )
    children = _indices_for_label(labels, entity)
    if len(children) < 2:
        raise ValueError(
            f"within-entity holdout requires at least two children for {entity!r}; "
            f"found {len(children)}."
        )
    all_indices = tuple(range(len(labels)))
    splits = []
    for ordinal, held_index in enumerate(children):
        held = (held_index,)
        train = tuple(i for i in all_indices if i != held_index)
        splits.append(
            _make_split(
                experiment_set,
                train_indices=train,
                held_out_indices=held,
                split_id=f"within-owner:{entity!s}:child-{ordinal + 1}-of-{len(children)}",
                level="child",
                entity=entity,
            )
        )
    return tuple(splits)


def _curve_value(curve: Any, name: str, *, aliases: tuple[str, ...] = ()) -> Any:
    """Read one CV-curve field from a mapping or record-like object."""
    names = (name,) + aliases
    if isinstance(curve, Mapping):
        for candidate in names:
            if candidate in curve:
                return curve[candidate]
    else:
        for candidate in names:
            if hasattr(curve, candidate):
                return getattr(curve, candidate)
    choices = ", ".join(repr(value) for value in names)
    raise ValueError(f"cross-validation curve is missing required field {choices}.")


def one_standard_error(cv_curve: Any) -> Any:
    """Choose the simplest candidate within one SE of the empirical best.

    The threshold uses the standard error *at the best candidate*.  For a
    minimizing metric a candidate is eligible when its mean is no greater
    than ``best_mean + best_SE``; the inequality is reversed for a maximizing
    metric.  Among eligible candidates, minimum declared complexity wins and
    original curve order breaks ties.

    Args:
        cv_curve: Mapping or record with ``candidates``, ``mean``,
            ``standard_error`` (``se`` is accepted as an alias), ``complexity``,
            and ``direction`` (``"minimize"`` or ``"maximize"``).

    Returns:
        The selected entry from ``candidates``.
    """
    candidates = tuple(_curve_value(cv_curve, "candidates"))
    mean = np.asarray(_curve_value(cv_curve, "mean"), dtype=float)
    standard_error = np.asarray(
        _curve_value(cv_curve, "standard_error", aliases=("se",)), dtype=float
    )
    try:
        complexity = np.asarray(_curve_value(cv_curve, "complexity"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cross-validation complexity values must be finite numeric scalars."
        ) from exc
    direction = str(_curve_value(cv_curve, "direction")).lower()

    n_candidates = len(candidates)
    if n_candidates == 0:
        raise ValueError("cross-validation curve must contain at least one candidate.")
    for name, values in (
        ("mean", mean),
        ("standard_error", standard_error),
        ("complexity", complexity),
    ):
        if values.ndim != 1 or values.shape[0] != n_candidates:
            raise ValueError(
                f"cross-validation curve {name} length/shape must match "
                f"{n_candidates} candidates; got shape {values.shape}."
            )
    if not np.all(np.isfinite(mean)):
        raise ValueError("cross-validation mean values must all be finite.")
    if not np.all(np.isfinite(standard_error)) or np.any(standard_error < 0.0):
        raise ValueError("cross-validation standard errors must be finite and non-negative.")
    if not np.all(np.isfinite(complexity)):
        raise ValueError("cross-validation complexity values must all be finite.")
    if direction not in ("minimize", "maximize"):
        raise ValueError("cross-validation direction must be 'minimize' or 'maximize'.")

    best_index = int(np.argmin(mean) if direction == "minimize" else np.argmax(mean))
    if direction == "minimize":
        eligible = mean <= mean[best_index] + standard_error[best_index]
    else:
        eligible = mean >= mean[best_index] - standard_error[best_index]
    eligible_indices = np.flatnonzero(eligible)
    try:
        selected = min(eligible_indices, key=lambda i: (complexity[int(i)], int(i)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cross-validation complexity values must be finite, scalar, and orderable."
        ) from exc
    return candidates[int(selected)]
