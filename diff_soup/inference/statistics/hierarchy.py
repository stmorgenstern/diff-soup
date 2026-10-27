"""Nested levels of parameters: the study, its intermediate levels, and its experiments.

A [Hierarchy][diff_soup.inference.Hierarchy] declares which learnable Recipe
parameters take one value for the whole study, one value per member of an
intermediate level (a cell, a particle), or one value per experiment. The
levels nest: every member of an inner level sits under one member of the level
above it. Each experiment names its member of every intermediate level in
``Experiment(parameters=...)``; the tree of members is derived from those
names, never written out.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from diff_soup.problem.parameters import _validate_handles, parameter_decoder

EXPERIMENT = "experiment"


def _quoted(names: Any) -> str:
    """``names`` as a comma-separated list of their representations."""
    return ", ".join(repr(name) for name in names)


def _levels_text(levels: tuple) -> str:
    """The intermediate levels of a hierarchy, for a message."""
    return _quoted(levels) if levels else "none (its experiments sit directly under the study)"


@dataclasses.dataclass(frozen=True, init=False)
class Hierarchy:
    """Declare which parameters of a study are shared and which take one value per member of a level.

    Pass it as `parameters=` of a [`Study`][diff_soup.inference.Study]. Each handle appears once,
    in `shared` or under one level of `per`, and experiments that name the same member share its
    values.

    Args:
        levels: The level names between the study and its experiments, outermost first, or
            `None` to take them from the keys of `per`.
        shared: The handles with one value for the whole study.
        per: A mapping from level name to the handles with one value per member of that level,
            where the key `"experiment"` means each experiment.
    """

    levels: tuple[str, ...]
    shared: tuple
    per: Mapping[str, tuple]

    def __init__(self, *, levels: Any = None, shared: Any = (), per: Any = None):
        """Validate the levels and the placement of every handle, then freeze.

        Args:
            levels: The intermediate level names, outermost first, as a tuple
                of non-empty strings, or ``None`` for the keys of ``per``
                other than ``"experiment"``, in the order given.
            shared: Recipe handles with one value for the whole study.
            per: ``{level name: handles}``; the keys are levels of
                ``levels`` or ``"experiment"``.

        Raises:
            TypeError: If ``levels`` is not a tuple of non-empty strings,
                ``per`` is not a mapping from level names, or an entry is not
                an iterable of Recipe handles.
            ValueError: If ``levels`` lists ``"experiment"`` or a name twice,
                ``per`` names a level that is not declared, no parameter is
                placed, a parameter is placed twice or is not learnable, the
                handles come from several Recipes, or two handles share a name.
        """
        if per is None:
            per = {}
        if not isinstance(per, Mapping):
            raise TypeError(
                "Hierarchy per= maps a level name to the parameters with one value per member "
                f"of that level; got {type(per).__name__}."
            )
        for key in per:
            if not isinstance(key, str) or not key:
                raise TypeError(
                    "Hierarchy per= maps a level name to the parameters with one value per "
                    f"member of that level; got the key {key!r}."
                )
        if levels is None:
            levels = tuple(key for key in per if key != EXPERIMENT)
        elif not isinstance(levels, tuple) or not all(
            isinstance(name, str) and name for name in levels
        ):
            raise TypeError(
                "Hierarchy levels= names the intermediate levels, outermost first, as a tuple "
                f"of non-empty strings; got {levels!r}."
            )
        if EXPERIMENT in levels:
            raise ValueError(
                "Hierarchy levels= lists 'experiment', the innermost level that every "
                "hierarchy has; list only the levels between the study and its experiments."
            )
        repeated = sorted({name for name in levels if levels.count(name) > 1})
        if repeated:
            raise ValueError(
                f"Hierarchy levels= names {repeated} more than once; name each level once."
            )
        for key in per:
            if key != EXPERIMENT and key not in levels:
                if levels:
                    known = f"its levels are {_quoted(levels)} and 'experiment'"
                else:
                    known = "its only level is 'experiment'"
                raise ValueError(
                    f"Hierarchy per= places parameters at {key!r}, which is not a level of this "
                    f"hierarchy; {known}. Add {key!r} to levels=, or place those parameters at "
                    "one of these levels."
                )

        shared_handles = _validate_handles(shared, where="shared", declaration="Hierarchy")
        placed: dict[str, tuple] = {}
        for key in levels + (EXPERIMENT,):
            if key in per:
                placed[key] = _validate_handles(
                    per[key], where=f"per[{key!r}]", declaration="Hierarchy"
                )
        all_handles = shared_handles + tuple(
            handle for handles in placed.values() for handle in handles
        )
        if not all_handles:
            raise ValueError("Hierarchy requires at least one parameter in shared= or per=.")
        identities = [handle.identity for handle in all_handles]
        twice = sorted(
            {handle.name for handle in all_handles if identities.count(handle.identity) > 1}
        )
        if twice:
            raise ValueError(
                f"Hierarchy places {twice} more than once; place each parameter in shared= or "
                "at exactly one level of per=."
            )
        namespaces = {handle.identity[0] for handle in all_handles}
        if len(namespaces) > 1:
            raise ValueError(
                "Hierarchy handles must come from one Recipe declaration; "
                "mixing same-named handles from independent specs is ambiguous."
            )
        names = [handle.name for handle in all_handles]
        repeated_names = sorted({name for name in names if names.count(name) > 1})
        if repeated_names:
            raise ValueError(
                f"Hierarchy handle names must be unique; got duplicate name(s) "
                f"{repeated_names}. Parameter names label optimizer groups and "
                "cannot be ambiguous across scalar and function declarations."
            )

        object.__setattr__(self, "levels", tuple(levels))
        object.__setattr__(self, "shared", shared_handles)
        object.__setattr__(self, "per", MappingProxyType(placed))

    @property
    def depth(self) -> int:
        """The number of levels below the study, ``"experiment"`` included."""
        return len(self.levels) + 1

    @property
    def handles(self) -> tuple:
        """Every handle: the shared ones, then each level's in order, ``"experiment"`` last."""
        return self.shared + tuple(handle for handles in self.per.values() for handle in handles)

    @property
    def varying(self) -> Mapping[str, tuple]:
        """Read-only mapping from each level that carries parameters to its handles, in order."""
        return MappingProxyType({key: handles for key, handles in self.per.items() if handles})

    def level_for(self, handle: Any) -> str | None:
        """The level ``handle`` takes one value per member of, or ``None`` if shared.

        Raises:
            KeyError: If the hierarchy does not place ``handle``.
        """
        if any(candidate == handle for candidate in self.shared):
            return None
        for key, handles in self.per.items():
            if any(candidate == handle for candidate in handles):
                return key
        raise KeyError(f"Hierarchy does not contain handle {getattr(handle, 'name', handle)!r}.")

    def cluster_for(self, handle: Any) -> str | None:
        """The same as [level_for][diff_soup.inference.Hierarchy.level_for]."""
        return self.level_for(handle)


@dataclasses.dataclass(frozen=True)
class _Tree:
    """The tree of members a hierarchy and its experiments derive. Frozen.

    Levels are numbered 1 .. D below the study (level 0), ``"experiment"`` at
    D. ``names``, ``members`` and the columns of ``nodes`` hold levels 1 .. D
    at positions 0 .. D-1; ``parents`` and ``widths`` are indexed by the level
    number itself.

    Attributes:
        names: The D level names, ``"experiment"`` last.
        ids: The experiment ids, in study order.
        members: Per level, its member labels in first-appearance order over
            the experiments; at ``"experiment"``, the experiment ids.
        nodes: ``(E, D)`` int32, each experiment's node index at every level.
        parents: ``(None, None, parents_2, ..., parents_D)``, ``parents_l``
            the ``(N_l,)`` int32 node indices at level ``l - 1``.
        widths: ``(g, v_1, ..., v_D)``, the decoded sizes of the shared
            handles and of each level's handles.
    """

    names: tuple
    ids: tuple
    members: tuple
    nodes: np.ndarray
    parents: tuple
    widths: tuple

    @property
    def depth(self) -> int:
        """The number of levels below the study, ``"experiment"`` included."""
        return len(self.names)

    @property
    def counts(self) -> tuple:
        """The number of nodes of each level 1 .. D."""
        return tuple(len(labels) for labels in self.members)


def _tree(hierarchy: Hierarchy, experiments: Any) -> _Tree:
    """Derive the tree of members from the experiments' ``parameters=``.

    Node indices follow first appearance over the experiments in study order,
    which is also the order of the members' blocks in the fitted vector.

    Raises:
        ValueError: If an experiment names a level the hierarchy does not have
            or misses one, two members of a level print alike, or a member sits
            under two members of the level above.
    """
    experiments = tuple(experiments)
    ids = tuple(
        experiment.id if experiment.id is not None else f"experiment-{index}"
        for index, experiment in enumerate(experiments)
    )
    levels = tuple(hierarchy.levels)
    rows = []
    for experiment, experiment_id in zip(experiments, ids):
        named = dict(getattr(experiment, "parameters", None) or {})
        unknown = [key for key in named if key not in levels]
        if unknown:
            message = (
                f"experiment {experiment_id!r} names level(s) {_quoted(unknown)} in "
                "parameters=, which the study's hierarchy does not have; its levels are "
                f"{_levels_text(levels)}."
            )
            declared = [key for key in unknown if key in (experiment._declarations or {})]
            if declared:
                message += (
                    f" {declared[0]!r} is a declared parameter of the experiment's model: a "
                    "fixed value of it goes in inputs=."
                )
            raise ValueError(message)
        missing = [level for level in levels if level not in named]
        if missing:
            raise ValueError(
                f"experiment {experiment_id!r} names no member of level(s) {_quoted(missing)}; "
                f"the study's hierarchy has the levels {_levels_text(levels)}, and every "
                "experiment names one member of each in parameters=."
            )
        rows.append(tuple(named[level] for level in levels))

    members: list[tuple] = []
    columns: list[tuple] = []
    for position, level in enumerate(levels):
        index_of: dict = {}
        printed: dict = {}
        for row in rows:
            label = row[position]
            if label in index_of:
                continue
            text = str(label)
            if text in printed:
                raise ValueError(
                    f"level {level!r} has the members {printed[text]!r} and {label!r}, which "
                    "print alike; members name fitted values such as 'k@p1', so give the "
                    "members of one level distinct labels."
                )
            printed[text] = label
            index_of[label] = len(index_of)
        members.append(tuple(index_of))
        columns.append(tuple(index_of[row[position]] for row in rows))
    if not levels:
        # The experiments are the members of the first tied level: their ids
        # name fitted values too.
        printed = {}
        for experiment_id in ids:
            text = str(experiment_id)
            if text in printed:
                raise ValueError(
                    f"level {EXPERIMENT!r} has the members {printed[text]!r} and "
                    f"{experiment_id!r}, which print alike; members name fitted values such as "
                    "'k@p1', so give the members of one level distinct labels."
                )
            printed[text] = experiment_id
    members.append(ids)
    columns.append(tuple(range(len(ids))))
    names = levels + (EXPERIMENT,)
    depth = len(names)
    nodes = np.asarray(columns, dtype=np.int32).T.reshape(len(ids), depth)

    parents: list = [None, None]
    for level in range(2, depth + 1):
        parent_of = np.full(len(members[level - 1]), -1, dtype=np.int32)
        first: dict = {}
        for index in range(len(ids)):
            member = int(nodes[index, level - 1])
            parent = int(nodes[index, level - 2])
            if parent_of[member] < 0:
                parent_of[member] = parent
                first[member] = index
            elif parent_of[member] != parent:
                above = names[level - 2]
                earlier = first[member]
                raise ValueError(
                    f"member {members[level - 1][member]!r} of level {names[level - 1]!r} sits "
                    f"under two members of level {above!r}: experiment {ids[earlier]!r} names "
                    f"{members[level - 2][int(parent_of[member])]!r} and experiment "
                    f"{ids[index]!r} names {members[level - 2][parent]!r}. A member has one "
                    f"parent, so name the same {above!r} member in every experiment of "
                    f"{members[level - 1][member]!r}; levels= lists the levels outermost first."
                )
        parents.append(parent_of)

    def width(handles: tuple) -> int:
        return int(sum(parameter_decoder(handle)[0] for handle in handles))

    widths = (width(hierarchy.shared),) + tuple(
        width(hierarchy.per.get(name, ())) for name in names
    )
    return _Tree(
        names=names,
        ids=ids,
        members=tuple(members),
        nodes=nodes,
        parents=tuple(parents),
        widths=widths,
    )


__all__ = ["Hierarchy"]
