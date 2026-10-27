"""Study aggregation and handle-to-optimizer schema compilation."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax
import numpy as np

from diff_soup._layout import Layout
from diff_soup.inference.fitting.bounds import Bounds
from diff_soup.inference.fitting.inference_problem import ParameterGroups
from diff_soup.inference.fitting.joint import (
    JointInferenceProblem,
    _sharding_requested,
    plan_child_data_owners,
)
from diff_soup.inference.statistics._fingerprint import study_content_fingerprint
from diff_soup.inference.statistics.hierarchy import EXPERIMENT, Hierarchy, _tree
from diff_soup.inference.statistics.measurement import Experiment
from diff_soup.problem._registry import parameter_handles
from diff_soup.problem.parameters import (
    ParameterEntry,
    ParameterOccurrence,
    ParameterSchema,
    ParameterSet,
    parameter_decoder,
)


def _unique_in_order(values: Any) -> tuple:
    """Return unique hashable values without sorting away semantic order."""
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _is_unique_partition(labels: tuple) -> bool:
    """Whether every experiment has a distinct label in this cluster."""
    return len(_unique_in_order(labels)) == len(labels)


def _validate_handle_model(handle: Any, model: Any, *, experiment_id: Any, declarations=None):
    """Resolve one compatible declared name on the experiment's own model."""
    if declarations is None:
        declarations = {
            candidate.name: candidate
            for values in parameter_handles(model).values()
            for candidate in values
        }
        declarations.update(
            {candidate.name: candidate for candidate in getattr(model, "_recipe_declarations", ())}
        )
    candidate = declarations.get(handle.name)
    if candidate is None:
        raise ValueError(
            f"parameter handle {handle.name!r} is absent from experiment {experiment_id!r}; declared names: {sorted(declarations)}"
        )
    size, _ = parameter_decoder(handle)
    own_size, _ = parameter_decoder(candidate)
    if size != own_size or handle.learnable != candidate.learnable:
        raise ValueError(
            f"parameter handle {handle.name!r} has decoded size {size} and learnable={handle.learnable}, but experiment {experiment_id!r} declares size {own_size} and learnable={candidate.learnable}; both must agree"
        )
    return candidate


def _experiment_rows(experiment: Experiment) -> int:
    """The number of observed values an experiment contributes: its residual rows."""
    return sum(
        int(np.prod(measurement.values_shape(), dtype=np.int64))
        for measurement in experiment.measurements
    )


def balance_owners(experiments: Any, lanes: int, *, weight: Callable | None = None) -> Any:
    """Reorder a study's experiments so that equal consecutive blocks of owners carry balanced loads.

    The owners are the members of the study's outermost level or cluster, or else the
    experiments themselves. Each owner's experiments stay together, and the owners are assigned
    heaviest first, each to the lightest block with room.

    Args:
        experiments: A [`Study`][diff_soup.inference.Study] or a sequence of
            [`Experiment`][diff_soup.inference.Experiment].
        lanes: The number of consecutive blocks, filled in order with `ceil(n_owners / lanes)`
            owners each.
        weight: A function giving the load of one experiment, or `None` to count its measured
            values.

    Returns:
        The reordered study, or a tuple of the reordered experiments.
    """
    lanes = int(lanes)
    if lanes < 1:
        raise ValueError("balance_owners needs at least one lane.")
    weight = _experiment_rows if weight is None else weight
    if isinstance(experiments, Study):
        study = experiments
        items = study.experiments
        tiers = study._tied_clusters()
        labels = tuple(study.clusters[tiers[0]]) if tiers else tuple(range(len(items)))
    else:
        study = None
        items = tuple(experiments)
        if not all(isinstance(item, Experiment) for item in items):
            raise TypeError("balance_owners takes a Study or a sequence of Experiments.")
        labels = tuple(range(len(items)))
    owners = _unique_in_order(labels)
    members = {owner: [i for i, label in enumerate(labels) if label == owner] for owner in owners}
    loads = {owner: float(sum(weight(items[i]) for i in members[owner])) for owner in owners}
    # The placement rule fills the lanes in order with blocks of ``capacity``
    # owners, so lane k holds exactly ``quota[k]`` owners; the choice is which.
    capacity = -(-len(owners) // lanes)
    quota = [min(capacity, max(0, len(owners) - lane * capacity)) for lane in range(lanes)]
    lane_load = [0.0] * lanes
    lane_owners: list[list] = [[] for _ in range(lanes)]
    for owner in sorted(owners, key=lambda owner: (-loads[owner], owners.index(owner))):
        lane = min(
            (lane for lane in range(lanes) if len(lane_owners[lane]) < quota[lane]),
            key=lambda lane: (lane_load[lane], lane),
        )
        lane_owners[lane].append(owner)
        lane_load[lane] += loads[owner]
    order = [index for lane in lane_owners for owner in lane for index in members[owner]]
    reordered = tuple(items[index] for index in order)
    if study is None:
        return reordered
    if isinstance(study.parameters, Hierarchy):
        # The experiments carry their members; the clusters derive from them.
        return Study(reordered, parameters=study.parameters, independent=study.independent)
    return Study(
        reordered,
        parameters=study.parameters,
        clusters={
            name: tuple(study.clusters[name][index] for index in order) for name in study.clusters
        },
        independent=study.independent,
    )


@dataclasses.dataclass(frozen=True, init=False)
class Study:
    """A group of experiments with the parameters they share or vary, lowered with `ds.lower` for a joint fit.

    With a [Hierarchy][diff_soup.inference.Hierarchy], each experiment names its members in its own `parameters=`. With a `ds.ParameterSet`, `clusters=` gives the owner of each experiment for each cluster.

    Args:
        experiments: One [Experiment][diff_soup.inference.Experiment] or a non-empty iterable of them, with distinct ids.
        parameters: A `ds.ParameterSet` or a [Hierarchy][diff_soup.inference.Hierarchy].
        clusters: A mapping from cluster name to one owner label per experiment, for a `ds.ParameterSet` only.
        independent: Whether the experiments are independent, which must be `True` for several experiments under a `ds.ParameterSet`.
    """

    experiments: tuple[Experiment, ...]
    parameters: ParameterSet | Hierarchy
    clusters: Mapping[str, tuple]
    independent: bool
    fingerprint: str

    def __init__(
        self,
        experiments: Experiment | Any,
        *,
        parameters: ParameterSet | Hierarchy,
        clusters: Mapping[str, Any] | None = None,
        independent: bool | None = None,
    ):
        """Validate statistical independence and cluster alignment once.

        Raises:
            TypeError: If ``parameters`` is neither a ``ParameterSet`` nor a
                ``Hierarchy``, or a declaration has the wrong type.
            ValueError: If the experiments, clusters, and parameters do not
                fit together: with a hierarchy, ``clusters=`` is given, an
                experiment names a level the hierarchy does not have or misses
                one, a member sits under two members of the level above, two
                members of a level print alike, or a learnable parameter is
                placed nowhere; with a ``ParameterSet``, an experiment names
                members in ``parameters=``.
        """
        if isinstance(experiments, Experiment):
            experiments = (experiments,)
        else:
            experiments = tuple(experiments)
        if not experiments or not all(isinstance(item, Experiment) for item in experiments):
            raise TypeError("Study requires one Experiment or a non-empty iterable of them.")
        if not isinstance(parameters, (ParameterSet, Hierarchy)):
            raise TypeError(
                "Study parameters must be a diff_soup.ParameterSet or a "
                f"diff_soup.inference.Hierarchy; got {type(parameters).__name__}."
            )
        hierarchy = parameters if isinstance(parameters, Hierarchy) else None
        if (
            hierarchy is not None
            and clusters is not None
            and (not isinstance(clusters, Mapping) or len(clusters) > 0)
        ):
            raise ValueError(
                "Study clusters= labels the owners of a ds.ParameterSet declaration, and this "
                "study is declared with inf.Hierarchy, whose members each experiment names in "
                "parameters=; leave clusters= out."
            )
        if independent is None:
            if len(experiments) > 1 and hierarchy is None:
                raise ValueError(
                    "InferenceProblem with several experiments requires independent experiments or clusters="
                )
            independent = True
        if not isinstance(independent, bool):
            raise TypeError("Study independent must be a boolean declaration.")
        if not independent:
            raise NotImplementedError(
                "InferenceProblem with several experiments requires independent experiments or clusters="
            )

        if clusters is None:
            clusters = {}
        if not isinstance(clusters, Mapping):
            raise TypeError("Study clusters must map names to one owner ID per experiment.")
        resolved_clusters: dict[str, tuple] = {}
        for name, labels in clusters.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"Study cluster names must be non-empty strings; got {name!r}.")
            labels = tuple(labels)
            if len(labels) != len(experiments):
                raise ValueError(
                    f"Study cluster {name!r} has {len(labels)} owners for "
                    f"{len(experiments)} experiments."
                )
            for label in labels:
                try:
                    hash(label)
                except TypeError as exc:
                    raise TypeError(
                        f"Study cluster owner IDs must be hashable; got {label!r}."
                    ) from exc
            resolved_clusters[name] = labels

        missing_clusters = sorted(set(parameters.varying) - set(resolved_clusters))
        if missing_clusters and hierarchy is None:
            raise ValueError(
                f"ParameterSet varying ownership names unknown Study cluster(s) {missing_clusters}."
            )

        experiment_ids = tuple(
            experiment.id if experiment.id is not None else f"experiment-{index}"
            for index, experiment in enumerate(experiments)
        )
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError(f"Study experiment IDs must be unique; got {experiment_ids}.")
        if hierarchy is None:
            for experiment, experiment_id in zip(experiments, experiment_ids):
                if experiment.parameters:
                    raise ValueError(
                        f"experiment {experiment_id!r} names members in parameters=, which a "
                        "study declared with ds.ParameterSet does not read; declare the study "
                        "with inf.Hierarchy, or leave parameters= out of the experiment."
                    )
        declaration = type(parameters).__name__
        for experiment, experiment_id in zip(experiments, experiment_ids):
            conflicts = set(experiment._fixed_inputs) & {
                handle.name for handle in parameters.handles
            }
            if conflicts:
                raise ValueError(
                    f"experiment {experiment_id!r} fixes inputs {sorted(conflicts)} that are also in the {declaration}; an input must be fixed or fitted, not both"
                )
            for handle in parameters.handles:
                _validate_handle_model(
                    handle,
                    experiment.model,
                    experiment_id=experiment_id,
                    declarations=experiment._declarations,
                )

        recorded_clusters = resolved_clusters
        if hierarchy is not None:
            tree = _tree(hierarchy, experiments)
            placed = {handle.name for handle in hierarchy.handles}
            for experiment, experiment_id in zip(experiments, experiment_ids):
                for name, declared in experiment._declarations.items():
                    if (
                        bool(getattr(declared, "learnable", False))
                        and name not in experiment._fixed_inputs
                        and name not in placed
                    ):
                        raise ValueError(
                            f"parameter {name!r} of experiment {experiment_id!r} is learnable "
                            "and the study's hierarchy places it nowhere; place it in shared= "
                            "or at one level of per=, fix it in inputs=, or declare it with "
                            "learnable=False."
                        )
            # The clusters of a hierarchy study are derived from the members
            # the experiments name; the fingerprint records them there.
            resolved_clusters = {
                level: tuple(tree.members[position][int(node)] for node in tree.nodes[:, position])
                for position, level in enumerate(hierarchy.levels)
            }
            resolved_clusters[EXPERIMENT] = experiment_ids
            recorded_clusters = {}

        digest = study_content_fingerprint(
            experiments=experiments,
            experiment_ids=experiment_ids,
            parameters=parameters,
            clusters=recorded_clusters,
            independent=bool(independent),
        )
        object.__setattr__(self, "experiments", experiments)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "clusters", MappingProxyType(resolved_clusters))
        object.__setattr__(self, "independent", bool(independent))
        object.__setattr__(self, "fingerprint", digest)
        self._joint_bounds()

    @property
    def experiment_ids(self) -> tuple:
        """Stable IDs in declaration order."""
        return tuple(
            experiment.id if experiment.id is not None else f"experiment-{index}"
            for index, experiment in enumerate(self.experiments)
        )

    def _parameter_groups(self, model: Any) -> ParameterGroups:
        """Lower Recipe parameter handles into the existing named grouping machinery."""
        shared = {
            handle.name: _validate_handle_model(handle, model, experiment_id="parameter groups")
            for handle in self.parameters.shared
        }
        local = {
            handle.name: _validate_handle_model(handle, model, experiment_id="parameter groups")
            for handles in self.parameters.varying.values()
            for handle in handles
        }
        return ParameterGroups(local=local, shared=shared)

    def _ownership_clusters(self) -> tuple[str, ...]:
        """Select the varying clusters represented by the intermediate owner tier.

        ``JointInferenceProblem`` has one explicit repeated-owner tier plus one local
        block per experiment.  A genuinely repeated partition (for example a
        physical cell with several pulses) therefore takes precedence, while
        unique pulse/run parameters remain child-local beneath it.  If no
        repeated partition is declared, the first varying cluster becomes the
        owner tier so owner-aware algorithms such as block LM and cluster
        bootstrap retain one owner per experiment. With a hierarchy, the owner
        tier is its outermost tied level.
        """
        if isinstance(self.parameters, Hierarchy):
            return self._tied_clusters()[:1]
        varying = tuple(self.parameters.varying)
        repeated = tuple(
            cluster
            for cluster in varying
            if not _is_unique_partition(tuple(self.clusters[cluster]))
        )
        return repeated if repeated else varying[:1]

    def _tied_clusters(self) -> tuple[str, ...]:
        """The clusters whose members hold their own blocks of the fitted vector, outermost first.

        With a hierarchy: its intermediate levels, or ``("experiment",)`` when
        it has none, so that every experiment is then an owner. With a
        ``ParameterSet``: the owner tier.
        """
        if isinstance(self.parameters, Hierarchy):
            return tuple(self.parameters.levels) or (EXPERIMENT,)
        return self._ownership_clusters()

    def _joint_bounds(self) -> Bounds | None:
        """Use each experiment's declaration and reject incompatible tied bounds."""
        result = {}
        seen = {}
        ownership_clusters = set(self._tied_clusters())
        for index, experiment in enumerate(self.experiments):
            for handle in self.parameters.handles:
                own = experiment._declarations[handle.name]
                bounds = getattr(own, "bounds", None)
                cluster = self.parameters.cluster_for(handle)
                if cluster is None:
                    group = handle.name
                else:
                    owner = (
                        self.clusters[cluster][index] if cluster in ownership_clusters else index
                    )
                    group = f"{handle.name}@{owner}"
                if group in seen:
                    previous_id, previous_bounds = seen[group]
                    if (previous_bounds or (None, None)) != (bounds or (None, None)):
                        raise ValueError(
                            f"tied parameter group {group!r} has disagreeing bounds: experiment {previous_id!r} declares {previous_bounds}, experiment {self.experiment_ids[index]!r} declares {bounds}; bounds within a shared group or owner must agree"
                        )
                seen[group] = self.experiment_ids[index], bounds
                if bounds is not None:
                    result[group] = bounds
        return Bounds(spec=result) if result else None

    def compile_inference_problem(
        self, *, layout: Layout | None = None, state_prior: Any = None
    ) -> Any:
        """Lower this Study to one existing leaf or owner-aware joint problem.

        Args:
            state_prior: Optional Gaussian prior on the latent window states of
                every decision-variable multiple-shooting experiment (the
                objective's ``MAP(state_prior=...)``).
            layout: Optional device placement for the child batch. A
                :class:`~diff_soup.Layout` carrying a ``mesh`` with a
                ``data`` axis fans the experiments across the mesh, each device
                running the same compiled kernel on its own slice. Placement is
                an execution choice rather than part of the Study's statistical
                content, which is why it is declared here and stays out of the
                content fingerprint.

        Returns:
            The single child's :class:`~diff_soup.inference.InferenceProblem`, or a
            :class:`~diff_soup.inference.JointInferenceProblem` over all of them.

        Raises:
            ValueError: If a placement is given for a single-experiment Study,
                or a device mesh for a hierarchy study with more than one
                intermediate level or without a parameter at its owner level.
        """
        if layout is not None and len(self.experiments) == 1:
            raise ValueError(
                "Study layout fans the experiments of a Study across a device mesh, "
                "but a single-experiment Study lowers to one leaf problem with "
                "no child batch to place. Omit layout, or declare the experiments "
                "that are meant to share the mesh."
            )
        if len(self.experiments) == 1:
            return self.experiments[0].as_inference_problem(
                state_prior=state_prior,
                parameters=self._parameter_groups(self.experiments[0].model),
            )
        if isinstance(self.parameters, Hierarchy):
            return self._compile_hierarchy(layout, state_prior)

        # JointInferenceProblem already has three exact tiers: global shared,
        # repeated owner blocks, and one local block per child. A repeated
        # physical-cell partition takes the owner tier and unique pulse/run
        # labels lower beneath it. With no repeated partition, the first unique
        # varying cluster becomes the owner tier so owner-aware solvers still
        # see one explicit owner per experiment. Multiple repeated tiers remain
        # subject to JointInferenceProblem's identical-label partition constraint.
        ownership_clusters = set(self._ownership_clusters())
        ownership = {
            handle.name: {
                index: self.clusters[cluster][index] for index in range(len(self.experiments))
            }
            for cluster, handles in self.parameters.varying.items()
            if cluster in ownership_clusters
            for handle in handles
        }
        shared = tuple(handle.name for handle in self.parameters.shared)
        children, holders = self._lower_children(shared, ownership, layout, state_prior)
        return JointInferenceProblem(
            children,
            shared=shared,
            ownership=ownership,
            bounds=self._joint_bounds(),
            layout=layout,
            data_holders=holders,
        )

    def _compile_hierarchy(self, layout: Layout | None, state_prior: Any) -> Any:
        """The joint problem of a hierarchy study: one block per member, level by level."""
        hierarchy = self.parameters
        tied = self._tied_clusters()
        if layout is not None and _sharding_requested(layout):
            if len(hierarchy.levels) > 1:
                raise ValueError(
                    "Study layout fans the experiments across a device mesh by the owners of "
                    f"one level, and this study's hierarchy has {len(hierarchy.levels)} "
                    f"intermediate levels ({', '.join(repr(n) for n in hierarchy.levels)}); "
                    "omit layout for this study."
                )
            if not hierarchy.per.get(tied[0]):
                raise ValueError(
                    "Study layout fans the experiments across a device mesh by the owners of "
                    "one level, and this study's hierarchy places no parameter at its level "
                    f"{tied[0]!r}; omit layout for this study."
                )
        n_experiments = len(self.experiments)
        levels = tuple(
            (
                tuple(self.clusters[name]),
                tuple(handle.name for handle in hierarchy.per.get(name, ())),
            )
            for name in tied
        )
        # The owner tier as the placement plan reads it: its groups and the
        # member of each experiment.
        ownership = {
            handle.name: {index: self.clusters[tied[0]][index] for index in range(n_experiments)}
            for handle in hierarchy.per.get(tied[0], ())
        }
        shared = tuple(handle.name for handle in hierarchy.shared)
        children, holders = self._lower_children(shared, ownership, layout, state_prior)
        return JointInferenceProblem(
            children,
            shared=shared,
            levels=levels,
            level_names=tied,
            bounds=self._joint_bounds(),
            layout=layout,
            data_holders=holders,
        )

    def build(
        self,
        *,
        objective: Any = None,
        layout: Layout | None = None,
        regularization: Any = None,
        trial_predicate: Any = None,
        trajectory: Any = None,
        chunk: int | None = None,
        device: Any = None,
        evaluation_noise: float | None = None,
        retry_kinds: Any = ("nonlinear_diverged", "step_collapse"),
        retry_first_step_factors: Any = (0.1, 1e-3),
        retry_tolerance_factors: Any = (),
        measured: Any = None,
    ) -> Any:
        """Lower this study for the batched evaluation of its experiments.

        ``ds.lower(study, ...)`` is this call. Nothing is compiled or solved
        beyond what the ``InferenceProblem`` constructor does; call
        ``prepare()`` on the result to compile its programs.

        Args:
            objective: ``MLE()`` or ``MAP(...)``; ``None`` fits by maximum
                likelihood, ``MLE()``.
            layout: Optional device placement of the experiment batch for the
                routes of ``InferenceProblem``; the lanes run on ``device``.
            regularization: Optional regularizer on the fitted vector.
            trial_predicate: Optional JAX-traceable ``x -> bool`` over the
                fitted vector.
            trajectory: Optional trajectory strategy replacing every
                experiment's own.
            chunk: The units one call of every compiled program evaluates,
                taken exactly as given; ``None`` takes, per bucket, the
                smallest power of two that holds its units, at most 1,024.
            device: The JAX device the lanes run on (their programs, inputs,
                tables and level systems); ``None`` takes the first device,
                ``jax.devices()[0]``.
            evaluation_noise: The largest relative noise with which the lanes
                evaluate the cost (the noise of the cost's evaluation, not the
                data's), a positive number, when the user knows it; ``None``
                takes the largest ``rtol`` of the experiments' solvers.
                ``TrustRegion``
                reports it beside the noise it measures from its own trials
                (``LoweredStudy.evaluation_noise``).
            retry_kinds: The failure kinds after which a lane is solved again,
                within the same evaluation, from a smaller first step: names
                of ``KINDS`` other than ``"ok"``, each once; ``()`` solves no
                lane again (``LoweredStudy.retry_kinds``).
            retry_first_step_factors: The rungs of that retry, positive
                factors in the order they are tried: rung ``r`` starts the
                failed lane's solve from ``retry_first_step_factors[r - 1]``
                times the first step its own solve took, through the same
                compiled program. The results say which rung produced each
                lane's result (``LoweredStudy.retry_first_step_factors``).
            retry_tolerance_factors: Must be ``()``: a tolerance is part of a
                compiled program, so the lanes retry from a smaller first step
                only.
            measured: In a run of several processes, balance the lanes over
                the processes by the seconds a previous run measured, from its
                checkpoint: the checkpoint directory of a fit or a bootstrap
                (or one of its ``attempt-`` or ``population-`` files), a
                [Ledger][diff_soup.inference.engine.Ledger], or a mapping of
                lane id to the seconds one evaluation of the lane took. A lane
                without a measurement is balanced by its modelled cost.
                ``None`` balances every lane by its modelled cost. Ignored in
                a run of one process, so the same script runs everywhere.

        Returns:
            The [LoweredStudy][diff_soup.inference.LoweredStudy].

        Raises:
            TypeError: If ``device`` is not a JAX device or ``None``, or
                ``measured`` is none of the forms above.
            ValueError: If ``evaluation_noise`` is not a positive finite number
                or ``None``.
            ConfigurationError: If ``retry_kinds`` names a kind that is not a
                failure kind or names one twice, a factor of
                ``retry_first_step_factors`` is not a positive finite number,
                or ``retry_tolerance_factors`` is not empty.
        """
        from diff_soup.inference import LoweredStudy
        from diff_soup.inference.statistics.estimation import MLE

        return LoweredStudy(
            self,
            objective=MLE() if objective is None else objective,
            layout=layout,
            regularization=regularization,
            trial_predicate=trial_predicate,
            trajectory=trajectory,
            chunk=chunk,
            device=device,
            evaluation_noise=evaluation_noise,
            retry_kinds=retry_kinds,
            retry_first_step_factors=retry_first_step_factors,
            retry_tolerance_factors=retry_tolerance_factors,
            measured=measured,
        )

    def _ds_lower(self, **options: Any) -> Any:
        """``ds.lower(study, **options)``: the same as [build][diff_soup.inference.Study.build]."""
        return self.build(**options)

    def _lower_children(
        self, shared: tuple, ownership: dict, layout: Layout | None, state_prior: Any = None
    ) -> tuple:
        """Lower every experiment, loading deferred data only where this process owns it.

        With one process, or no experiment mesh, each experiment loads its
        deferred measurement values once here. Under a multi-process mesh the
        experiments are lowered once, over placeholders of the declared or
        probed value shapes; the joint's placement rules decide which
        experiments each process evaluates, and those get their values loaded
        into the same lowering (only the residual closures are rebuilt); the
        others keep the placeholders, which no kernel on this process reads.

        Returns ``(children, holders)`` with one frozenset of holding process
        indices per experiment.
        """

        def lower(experiment, load_data, reuse=None):
            return experiment.as_inference_problem(
                parameters=self._parameter_groups(experiment.model),
                load_data=load_data,
                reuse=reuse,
                state_prior=state_prior,
            )

        everyone = frozenset(range(jax.process_count()))
        deferred = tuple(
            any(measurement.deferred for measurement in experiment.measurements)
            for experiment in self.experiments
        )
        if not any(deferred) or layout is None or jax.process_count() == 1:
            children = tuple(lower(experiment, True) for experiment in self.experiments)
            if jax.process_index() == 0:
                for child in children:
                    if hasattr(child, "persist_window_initialization"):
                        child.persist_window_initialization()
            return children, (everyone,) * len(children)
        for experiment, is_deferred in zip(self.experiments, deferred):
            if is_deferred:
                for measurement in experiment.measurements:
                    if measurement.deferred:
                        measurement._deferred.metadata()
        children = tuple(
            lower(experiment, not is_deferred)
            for experiment, is_deferred in zip(self.experiments, deferred)
        )
        plans = tuple(
            plan_child_data_owners(children, shared, ownership, layout, process_index=process)
            for process in sorted(everyone)
        )
        holders = tuple(
            frozenset(process for process in everyone if plans[process][index])
            for index in range(len(children))
        )
        mine = jax.process_index()
        children = tuple(
            lower(experiment, True, reuse=child) if is_deferred and mine in held else child
            for experiment, child, is_deferred, held in zip(
                self.experiments, children, deferred, holders
            )
        )
        # The process that holds an experiment's data owns it: it alone stores
        # the experiment's window initialization for later constructions.
        for child, held in zip(children, holders):
            if mine == min(held) and hasattr(child, "persist_window_initialization"):
                child.persist_window_initialization()
        return children, holders

    def parameter_schema(self, problem: Any) -> ParameterSchema:
        """Compile handle occurrences against an already-lowered flat layout."""
        group_columns = {
            name: tuple(int(column) for column in columns)
            for name, columns in zip(
                problem.param_group_names,
                problem.param_group_cols,
            )
        }
        entries = []
        for handle in self.parameters.shared:
            size, decode = parameter_decoder(handle)
            columns = group_columns.get(handle.name)
            if columns is None:
                raise ValueError(
                    f"compiled problem has no shared group for handle {handle.name!r}."
                )
            if len(columns) != size:
                raise ValueError(
                    f"handle {handle.name!r} exposes {size} values but optimizer group "
                    f"owns {len(columns)} columns."
                )
            entries.append(
                ParameterEntry(
                    handle=handle,
                    occurrences=(ParameterOccurrence(owner=None, columns=columns),),
                    size=size,
                    decode=decode,
                )
            )

        for cluster, handles in self.parameters.varying.items():
            labels = tuple(self.clusters[cluster])
            owners = _unique_in_order(labels)
            child_local = cluster not in set(self._tied_clusters())
            for handle in handles:
                size, decode = parameter_decoder(handle)
                occurrences = []
                for owner in owners:
                    if len(self.experiments) == 1:
                        group_name = handle.name
                    elif child_local:
                        child_index = labels.index(owner)
                        group_name = f"{handle.name}@{child_index}"
                    else:
                        group_name = f"{handle.name}@{owner}"
                    columns = group_columns.get(group_name)
                    if columns is None:
                        raise ValueError(
                            f"compiled problem has no owner group {group_name!r} for "
                            f"handle {handle.name!r}."
                        )
                    if len(columns) != size:
                        raise ValueError(
                            f"owner group {group_name!r} has {len(columns)} columns; expected {size}."
                        )
                    occurrences.append(ParameterOccurrence(owner=owner, columns=columns))
                entries.append(
                    ParameterEntry(
                        handle=handle,
                        occurrences=tuple(occurrences),
                        size=size,
                        decode=decode,
                    )
                )
        return ParameterSchema(entries, fingerprint=self.fingerprint)


__all__ = ["Study"]
