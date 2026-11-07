# LOC > 500 justification: the joint problem ties named shared parameter
# groups across leaf problems and owns BOTH the geometry-axis batched
# residual (homogeneous children vmapped through one kernel) and the
# static-concatenation fallback (heterogeneous children), plus the joint
# decision-vector layout, construction-time validation, and per-child result
# reconstruction.  These share the joint-layout index machinery and belong
# together; the leaf InferenceProblem lives in inference_problem.py.
"""joint.py — fitting several leaf problems with tied shared parameters.

A [JointInferenceProblem][diff_soup.optimization.inference.fitting.joint.JointInferenceProblem] composes several leaf
[InferenceProblem][diff_soup.optimization.inference.fitting.inference_problem.InferenceProblem] children and ties
explicitly named **shared** parameter groups across them while each child keeps
its **local** parameter groups, geometry, solver, data, trajectory, and
trajectory states.

The joint decision vector is ``[shared | local_0, local_1, ... | states_0,
states_1, ...]``: each shared group is represented once; local groups and
trajectory-state blocks stay per child.  The seam threads each child's parameter
pytree explicitly into ``pdesolve``, so a shared scalar in the joint vector fans
out to every child and its gradient sums back automatically.

Two residual paths, picked at construction:

* **homogeneous geometry-axis batching** — when every child shares problem
  structure, solver configuration, parameter-group structure, trajectory
  strategy, and data-stream schema (only numeric leaves differ), the per-child
  residual runs through one ``jax.vmap`` kernel, so the traced program (and its
  jaxpr equation count) is independent of the child count;
* **static concatenation** — heterogeneous children (or latent-window children,
  whose learned initial state has no fixed decision-block offset) each evaluate
  their own residual closure and the rows are concatenated; each distinct
  structure compiles once.

Both expose the duck-typed surface the solver engines read, so
``optsolve(joint, optimizer=...)`` drives a joint fit exactly like a leaf fit.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup.optimization.inference.fitting._forward_seam import (
    _FAILED_SENTINEL,
)
from diff_soup.optimization.inference.fitting._joint_grouped import (
    _build_grouped_lsq,
    _build_static_residual,
    _FrozenTailRegularizer,
    _JointRegularizer,
)
from diff_soup.optimization.inference.fitting._joint_groups import (
    _child_group_cols,
    _normalise_levels,
    _normalise_ownership,
    _validate_and_shared_init,
)
from diff_soup.optimization.inference.fitting._joint_layout import (
    JointLayout,
    _assemble_x0,
    _build_layout,
    _joint_group_cols,
    _joint_theta_names,
    _reconstruct_child_flat,
)
from diff_soup.optimization.inference.fitting._joint_residual import (
    _build_homogeneous_residual,
    _StepBalance,
)
from diff_soup.optimization.inference.fitting._joint_structure import (
    _child_mode,
    _is_homogeneous,
    _sharding_requested,
    _validate_layout,
    child_lanes,
)
from diff_soup.optimization.inference.fitting._residual_lowering import (
    _loss_and_grad_from_residual,
    _loss_fn_from_residual,
    _loss_grad_hessian_from_gradient,
    _stacked_residual_fn,
    build_stacked_residual_fns,
)
from diff_soup.optimization.inference.fitting.bounds import Bounds
from diff_soup.optimization.inference.fitting.inference_problem import InferenceProblem
from diff_soup.optimization.inference.fitting.result import ChildFit

# ---------------------------------------------------------------------------
# The joint problem
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, init=False)
class JointInferenceProblem:
    """Several leaf problems fit together with tied shared parameter groups.

    Frozen, eager-build.  Composes a tuple of leaf
    [InferenceProblem][diff_soup.optimization.inference.fitting.inference_problem.InferenceProblem] children and ties the
    named ``shared`` groups across them; each child keeps its local groups,
    geometry, solver, data, trajectory strategy, and trajectory states.

    Construct as::

        joint = JointInferenceProblem(
            problems=(particle_a, particle_b, particle_c),
            shared=("j0",),
        )
        result = optsolve(joint, optimizer=LevenbergMarquardt(...))

    The joint decision vector is ``[global | owner blocks | child locals | states]``.
    ``ownership={"field": {0: "particle-a", 1: "particle-a", ...}}`` ties the
    named group only among children with the same owner ID. With ``ownership=None``
    the owner section is empty and the original shared/local layout is unchanged.
    ``levels=`` ties groups at nested levels instead: one ``(member per child,
    group names)`` pair per level, outermost first, each member of an inner level
    under one member of the level above; the vector then holds the member blocks
    level by level, ``[global | level-1 blocks | level-2 blocks | ... | child
    locals | states]``, and the members of the first level are the owners. When
    every child is structurally homogeneous the per-child residual runs through one
    ``jax.vmap`` kernel (the traced program is independent of the child count);
    heterogeneous children concatenate their own residual closures.

    **Multi-GPU child-batch sharding.**  Pass a [Layout][diff_soup.Layout] with a
    mesh with a ``data`` axis to fan the children across a device mesh::

        from diff_soup import Layout
        from jax.sharding import AxisType, Mesh

        mesh = Mesh(jax.devices()[:2], ("data",), axis_types=(AxisType.Auto,))
        joint = JointInferenceProblem(children, shared=("j0",),
                                 layout=Layout(mesh=mesh))
        result = optsolve(joint, LevenbergMarquardt())   # or BFGS()

    Each device owns a slice of the children and runs the same compiled kernel on its
    slice; a child count that is not a multiple of the device count is padded with
    masked children that contribute nothing.  The optimizer reduces only the objective,
    gradient, and Gauss-Newton matrix across the mesh — the global residual and Jacobian
    are never assembled on one device, so per-device memory scales with the local child
    count.  The single-device path (``layout=None``, or a one-device mesh) is the
    numerical baseline and follows the same code.  Each child's own linear solve stays
    local on its device, with a backend whose factors are arrays (``Dense``,
    ``BlockTridiagonal``, ``MemberBlock``, or ``Woodbury`` / ``Krylov`` over one of
    these); a backend that keeps its factors in native storage (``KLU``, ``SuperLU``,
    ``CuDSS``, alone or inside ``Woodbury`` / ``Krylov``) is refused when this process
    holds several devices of the data axis.  Levenberg-Marquardt and BFGS / gradient
    descent are the supported sharded solvers.

    The ``model`` axis distributes each child's sparse factorization over the devices
    of one model group, with ``CuDSS``, the one backend that factors across devices
    (``Dense`` is refused there). CuDSS keeps its factors in native storage, so a mesh
    with both axes holds one model group per process: launch one process per entry of
    the data axis; a process holding several model groups is refused. Structurally
    heterogeneous children use submeshes.
    Children that learn their initial state raise a
    [ConfigurationError][diff_soup._errors.ConfigurationError]. Decision-variable
    window children with a fixed initial state *are* supported: their per-window
    decision blocks have one fixed width, so the joint reconstructs them with a
    batched slice and the windows of every child run in one mapped propagation.

    Attributes:
        problems: The leaf children, in input order.
        shared: The shared group names tied across children.
        ownership: Owner-tied group declarations, normalized to a mapping from
            group name to every child index and its owner ID (with ``levels=``,
            to every child index and its member of the group's level).
        regularization: A regularizer over the joint vector (concatenates the
            children's own regularizer rows).
        bounds: Optional named box bounds over the JOINT parameter groups
            (shared groups by their name, each child's local groups by the
            joint group naming). Solvers project the replicated joint vector
            in their outer loop, so bounds compose with child-batch sharding
            without touching the sharded kernel.
    """

    problems: tuple
    shared: tuple
    ownership: Any
    data_holders: tuple
    regularization: Any
    bounds: Optional[Bounds]
    _layout: JointLayout
    _placement: Optional[Layout]
    _x0: np.ndarray
    _segment_fns: tuple
    _batched_r: Optional[Callable]
    _batched_rj: Optional[Callable]
    _stacked_r: Optional[Callable]
    _stacked_rj: Optional[Callable]
    _residual_builder: Optional[Callable]
    _residual_seams: dict
    _loss_fn: Optional[Callable]
    _loss_grad_fns: dict
    _loss_grad_hessian_fns: dict
    _evaluate_lsq: Optional[Callable]
    _evaluate_cost: Optional[Callable]
    _owner_residual_fns: Optional[tuple]
    _owner_reducer_fns: Optional[tuple]
    _owner_reducer_inputs: Optional[tuple]
    _owner_narrow_kernel_cache: Optional[dict]
    _owner_point_plan_cache: Optional[tuple]
    _restricted_initialization: bool
    _trial_predicate: Optional[Callable]
    _n_seg: int
    _rows_per_child: tuple
    _homogeneous: bool
    _theta_names: tuple
    _param_group_names: tuple
    _param_group_cols: tuple
    _baseline_reached_tend: Optional[bool]
    _level_names: tuple

    def __init__(
        self,
        problems: tuple,
        *,
        shared: tuple = (),
        shared_init: Optional[dict] = None,
        ownership: Any = None,
        layout: Optional[Layout] = None,
        bounds: Optional[Bounds] = None,
        residual_rows: Optional[tuple] = None,
        data_holders: Optional[tuple] = None,
        levels: Any = None,
        level_names: Optional[tuple] = None,
    ):
        """Validate shared declarations, lay out the joint vector, build the residual.

        Args:
            problems: The leaf children, in input order.
            shared: Names of the parameter groups tied across children.
            shared_init: Optional initial value per shared group, overriding the
                children's own initial values when they disagree.
            ownership: Either ``{group_name: {child_index: owner_id}}``, a
                sequence-valued equivalent, a callable ``owner(group, child)``,
                or ``None``. Groups tied at this level must induce the same child
                partition so each owner remains an independent local block.
            layout: Optional device placement.  ``None`` (default) fits on one
                device.  A [Layout][diff_soup.Layout] with a ``mesh`` and
                ``data`` axis fans the children out across the mesh
                (child-batch sharding); without a ``data`` axis it records
                placement but keeps single-device execution.
            bounds: Optional [Bounds][diff_soup.optimization.inference.fitting.bounds.Bounds]
                keyed by the joint parameter-group names.
            residual_rows: Optional precomputed per-child residual row counts,
                in child order.  When provided (for example from a persisted
                structure record) the heterogeneous construction validates the
                first child against an abstract evaluation and skips deriving
                the remaining counts; when omitted, counts come from
                `jax.eval_shape` -- construction never executes a child
                residual either way.  Ignored for homogeneous joints, whose
                row counts come from the shared batched structure.
            data_holders: Optional tuple with one frozenset of process indices
                per child naming the processes that loaded that child's
                observed data (see ``plan_child_data_owners``); defaults to
                every process for every child.
            levels: Optional nested levels of tied groups, in place of
                ``ownership``: one ``(member labels, group names)`` pair per
                level, outermost first, with one member label per child. Each
                named group takes one value per member of its level, and every
                member of an inner level sits under one member of the level
                above. More than one level cannot be placed on a data mesh.
            level_names: Optional names of the levels, one per entry of
                ``levels``, used in messages.

        Raises:
            ConfigurationError: If both ``ownership`` and ``levels`` are given,
                if more than one level is given with a data mesh, if the
                declarations do not fit the children, or if this process
                holds several devices of the data axis and a child's solver
                keeps its factors in native storage.
        """
        placement = layout  # the device-placement Layout; `layout` below is the vector layout
        children = tuple(problems)
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError(
                f"JointInferenceProblem 'bounds' must be a Bounds instance; got {type(bounds).__name__}."
            )
        if len(children) < 1:
            raise ValueError("JointInferenceProblem requires at least one child problem.")
        if not all(isinstance(c, InferenceProblem) for c in children):
            raise TypeError(
                "JointInferenceProblem 'problems' must all be InferenceProblem objects."
            )
        shared = tuple(shared)
        shared_init = dict(shared_init) if shared_init is not None else {}
        if len(set(shared)) != len(shared):
            raise ValueError("JointInferenceProblem 'shared' group names must be unique.")

        if levels is not None and ownership is not None:
            raise ConfigurationError(
                "JointInferenceProblem takes the owner-tied groups either in ownership= or "
                "in levels=; pass one of them."
            )

        shared_vals = _validate_and_shared_init(children, shared, shared_init)
        if levels is None:
            owned, owner_ids, child_owners, owner_vals, ownership_map = _normalise_ownership(
                children, shared, ownership
            )
            layout = _build_layout(children, shared, owned, owner_ids, child_owners)
            level_vals = (owner_vals,) if owned else ()
        else:
            level_groups, level_ids, child_levels, level_vals, ownership_map = _normalise_levels(
                children, shared, levels
            )
            if (
                len(level_ids) > 1
                and isinstance(placement, Layout)
                and _sharding_requested(placement)
            ):
                raise ConfigurationError(
                    "JointInferenceProblem places children on a device mesh by the owners of "
                    f"one level, and levels= declares {len(level_ids)}; build it without a "
                    "data mesh."
                )
            layout = _build_layout(children, shared, levels=(level_groups, level_ids, child_levels))
        x0 = _assemble_x0(children, layout, shared_vals, level_vals)
        if level_names is None:
            level_names = tuple(f"level {n}" for n in range(1, len(layout.level_ids) + 1))
        else:
            named = () if isinstance(level_names, str) else tuple(level_names)
            if len(named) != len(layout.level_ids):
                raise ConfigurationError(
                    "JointInferenceProblem level_names= names each of the "
                    f"{len(layout.level_ids)} levels once; got {level_names!r}."
                )
            level_names = named

        homogeneous = _is_homogeneous(children)
        _validate_layout(placement, children)
        if homogeneous:
            mode = _child_mode(children[0])
            (
                batched_r,
                batched_rj,
                evaluate_lsq,
                evaluate_cost,
                rows,
                _child_fn,
                _child_inputs,
            ) = _build_homogeneous_residual(children, layout, mode, placement)
            segment_fns = (batched_r,)
            rows_per_child = tuple(rows for _ in children)
            n_seg = len(children)
        else:
            residual, rows_per_child = _build_static_residual(
                children, layout, x0, rows_per_child=residual_rows
            )
            batched_r, batched_rj = None, None
            # A heterogeneous joint placed on a mesh shards as a sum over its homogeneous
            # structural groups; off the sharded path it keeps the static residual.
            if _sharding_requested(placement):
                evaluate_lsq, evaluate_cost = _build_grouped_lsq(
                    children, layout, shared, ownership_map, placement
                )
            else:
                evaluate_lsq, evaluate_cost = None, None
            segment_fns = (residual,)
            n_seg = len(children)

        theta_names = _joint_theta_names(children, layout)
        group_names, group_cols = _joint_group_cols(children, layout)
        _set = object.__setattr__
        _set(self, "problems", children)
        _set(self, "shared", shared)
        _set(self, "ownership", ownership_map)
        everyone = frozenset(range(jax.process_count()))
        _set(
            self,
            "data_holders",
            (everyone,) * len(children) if data_holders is None else tuple(data_holders),
        )
        _set(self, "bounds", bounds)
        _set(self, "regularization", _JointRegularizer(children, layout))
        _set(self, "_layout", layout)
        _set(self, "_placement", placement)
        _set(self, "_x0", np.asarray(x0))
        _set(self, "_segment_fns", segment_fns)
        _set(self, "_batched_r", batched_r)
        _set(self, "_batched_rj", batched_rj)
        _set(self, "_stacked_r", None)
        _set(self, "_stacked_rj", None)
        _set(self, "_residual_builder", None)
        _set(self, "_residual_seams", {})
        _set(self, "_loss_fn", None)
        _set(self, "_loss_grad_fns", {})
        _set(self, "_loss_grad_hessian_fns", {})
        _set(self, "_evaluate_lsq", evaluate_lsq)
        _set(self, "_evaluate_cost", evaluate_cost)
        _set(self, "_owner_residual_fns", None)
        _set(self, "_owner_reducer_fns", None)
        _set(self, "_owner_reducer_inputs", None)
        # Created here so warm-started and regularized views (shallow copies)
        # share one cache of owner kernels and their executables.
        _set(self, "_owner_narrow_kernel_cache", {})
        _set(self, "_owner_point_plan_cache", None)
        _set(self, "_restricted_initialization", False)
        _set(self, "_trial_predicate", None)
        _set(self, "_n_seg", n_seg)
        _set(self, "_rows_per_child", tuple(rows_per_child))
        _set(self, "_homogeneous", homogeneous)
        _set(self, "_theta_names", theta_names)
        _set(self, "_param_group_names", group_names)
        _set(self, "_param_group_cols", group_cols)
        _set(self, "_baseline_reached_tend", None)
        _set(self, "_level_names", tuple(str(name) for name in level_names))

    # --- LM-engine duck-typed surface ---
    @property
    def x0(self) -> np.ndarray:
        """Joint initial vector ``[global | owners | child locals | states]``."""
        return self._x0

    def with_start(self, start: ArrayLike | Sequence) -> "JointInferenceProblem":
        """Return an immutable warm-started view with unchanged residual kernels.

        Joint construction compiles the experiment structure and its residual
        kernels once.  Statistical front ends frequently need to reuse that
        structure with a fitted or user-supplied physical-coordinate start;
        rebuilding every child just to replace ``x0`` is both expensive and
        error-prone.  The decision-vector layout is therefore validated here
        while the compiled closures are shared by a shallow immutable copy.
        """
        start = np.asarray(start, dtype=np.asarray(self._x0).dtype)
        if start.shape != np.asarray(self._x0).shape:
            raise ValueError(
                f"JointInferenceProblem start has shape {start.shape}; expected "
                f"{np.asarray(self._x0).shape}."
            )
        if not np.all(np.isfinite(start)):
            raise ValueError("JointInferenceProblem start must contain only finite values.")
        warmed = copy.copy(self)
        object.__setattr__(warmed, "_x0", start.copy())
        return warmed

    def with_regularization(self, regularization: Any) -> "JointInferenceProblem":
        """Return a view carrying structured extra least-squares rows.

        The regularization supplies ``residual(x)`` plus ``block_rows(x)``.
        ``block_rows`` yields ``(columns, residual, jacobian)`` triples that
        declare which global or owner-local columns each extra row touches,
        with the Jacobian taken only with respect to ``columns``.

        This method deliberately refuses to replace existing child
        regularization. Composition requires an explicit ownership declaration
        for both sources and must not silently discard either one.
        """
        if not callable(getattr(regularization, "residual", None)) or not callable(
            getattr(regularization, "block_rows", None)
        ):
            raise TypeError(
                "JointInferenceProblem structured regularization requires residual(x) "
                "and block_rows(x) methods."
            )
        existing = self.regularization.residual(jnp.asarray(self._x0))
        if int(existing.shape[0]) != 0:
            raise ValueError(
                "JointInferenceProblem cannot replace existing child regularization with "
                "structured rows; compose both with explicit ownership first."
            )
        warmed = copy.copy(self)
        object.__setattr__(warmed, "regularization", regularization)
        return warmed

    def with_trial_predicate(self, predicate: Any) -> "JointInferenceProblem":
        """Return a view with a cheap JAX-compatible joint trial validator."""
        if not callable(predicate):
            raise TypeError("trial predicate must be callable.")
        out = copy.copy(self)
        object.__setattr__(out, "_trial_predicate", predicate)
        return out

    def trial_is_valid(self, x: ArrayLike):
        """Validate a joint point before any residual/PDE trial is launched."""
        point = jnp.asarray(x)
        valid = jnp.asarray(True)
        if self._trial_predicate is not None:
            valid = valid & jnp.asarray(self._trial_predicate(point), dtype=bool)
        for child_index, child in enumerate(self.problems):
            checker = getattr(child, "trial_is_valid", None)
            if checker is not None:
                valid = valid & jnp.asarray(
                    checker(_reconstruct_child_flat(point, child_index, self._layout)),
                    dtype=bool,
                )
        return valid

    @property
    def n_params(self) -> int:
        """Total scalar parameters in the joint vector."""
        return len(self._x0)

    @property
    def n_segments(self) -> int:
        """Residual-block count (one per child)."""
        return self._n_seg

    @property
    def residual_rows_per_child(self) -> tuple:
        """Host-static residual row count contributed by each child."""
        return self._rows_per_child

    @property
    def n_residuals(self) -> int:
        """Total host-static data-residual rows across all children."""
        return int(sum(self._rows_per_child))

    @property
    def segment_fns(self) -> tuple:
        """The forward-resolved residual closure used by Gauss-Newton solvers."""
        return self._residual_seam("forward")[0]

    def _residual_seam(self, ad: str) -> tuple:
        """Return cached residual and reduced evaluators for one AD mode."""
        cached = self._residual_seams.get(ad)
        if cached is not None:
            return cached
        if self._residual_builder is not None:
            residual = self._residual_builder(ad)
            stacked = _stacked_residual_fn((residual,))
            cached = ((residual,), None, None, stacked, None, None)
            self._residual_seams[ad] = cached
            return cached
        children = tuple(child._view_for_ad(ad) for child in self.problems)
        if self._homogeneous:
            mode = _child_mode(children[0])
            (
                batched,
                batched_rj,
                evaluate_lsq,
                evaluate_cost,
                _rows,
                _child_fn,
                _child_inputs,
            ) = _build_homogeneous_residual(children, self._layout, mode, self._placement)
            segments = (batched,)
        else:
            residual, _rows = _build_static_residual(
                children,
                self._layout,
                self._x0,
                rows_per_child=self._rows_per_child,
            )
            batched, batched_rj = None, None
            if _sharding_requested(self._placement):
                evaluate_lsq, evaluate_cost = _build_grouped_lsq(
                    children,
                    self._layout,
                    self.shared,
                    self.ownership,
                    self._placement,
                )
            else:
                evaluate_lsq, evaluate_cost = None, None
            segments = (residual,)
        stacked = _stacked_residual_fn(segments, batched)
        cached = (segments, batched, batched_rj, stacked, evaluate_lsq, evaluate_cost)
        self._residual_seams[ad] = cached
        return cached

    @property
    def stacked_r_fn(self) -> Callable:
        """The jitted concatenated-segment residual ``x -> r``, built once per joint.

        Heterogeneous joints solve through this pair; caching it means repeated
        solves reuse the compiled kernels instead of retracing every child.
        """
        if self._stacked_r is None:
            _segments, _batched, rj_fn, r_fn, _lsq, _cost = self._residual_seam("forward")
            if rj_fn is None:
                _r, rj_fn = build_stacked_residual_fns((_segments[0],))
            _set = object.__setattr__
            _set(self, "_stacked_r", r_fn)
            _set(self, "_stacked_rj", rj_fn)
        return self._stacked_r

    @property
    def stacked_rj_fn(self) -> Callable:
        """Jitted companion of [stacked_r_fn][diff_soup.optimization.inference.fitting.joint.JointInferenceProblem.stacked_r_fn].

        Returns the ``x -> (r, J)`` pair; built once.
        """
        if self._stacked_rj is None:
            self.stacked_r_fn  # noqa: B018 — builds and caches both callables
        return self._stacked_rj

    @property
    def batched_r_fn(self) -> Optional[Callable]:
        """The geometry-axis ``vmap`` residual (homogeneous children), or ``None``."""
        if self._batched_r is None:
            return None
        return self._residual_seam("forward")[1]

    @property
    def batched_rj_fn(self) -> Optional[Callable]:
        """The geometry-axis ``vmap`` ``x -> (r, J)`` (homogeneous children), or ``None``."""
        if self._batched_r is None:
            return None
        return self._residual_seam("forward")[2]

    @property
    def loss_fn(self) -> Callable:
        """The cached joint data-misfit loss."""
        if self._loss_fn is None:
            residual = self._residual_seam("reverse")[3]
            object.__setattr__(self, "_loss_fn", _loss_fn_from_residual(residual))
        return self._loss_fn

    def loss_and_grad_fn(self, ad: str = "reverse") -> Callable:
        """Return the cached joint data-misfit value and gradient for one AD mode.

        Raises:
            ConfigurationError: If ``ad`` is not ``"forward"`` or ``"reverse"``,
                or no child supports it.
        """
        if ad not in ("forward", "reverse"):
            raise ConfigurationError(
                f"optsolve differentiation mode must be 'forward' or 'reverse'; got {ad!r}."
            )
        cached = self._loss_grad_fns.get(ad)
        if cached is not None:
            return cached
        supported = self.supported_compositions
        if ad not in supported:
            raise ConfigurationError(
                f"JointInferenceProblem does not support differentiation mode {ad!r}; "
                f"all children must support it, and their common modes are {sorted(supported)}."
            )
        residual = self._residual_seam(ad)[3]
        cached = _loss_and_grad_from_residual(residual, ad)
        self._loss_grad_fns[ad] = cached
        return cached

    @property
    def supported_compositions(self) -> frozenset[str]:
        """Differentiation modes supported by every child problem."""
        common = set(self.problems[0].supported_compositions)
        for child in self.problems[1:]:
            common.intersection_update(child.supported_compositions)
        return frozenset(common)

    def resolved_hessian_composition(self, composition: str = "auto") -> str:
        """Resolve and validate a joint true-Hessian composition.

        ``"auto"`` resolves to ``"forward-over-reverse"`` when every child
        supports it, else to ``"forward-over-forward"``, on one device and on
        a device mesh alike.

        Raises:
            ConfigurationError: If the composition is unknown or not supported
                by every child.
        """
        supported = self.supported_compositions
        if composition == "auto":
            if "forward-over-reverse" in supported:
                return "forward-over-reverse"
            composition = "forward-over-forward"
        allowed = ("forward-over-reverse", "forward-over-forward")
        if composition not in allowed:
            raise ConfigurationError(
                "loss_grad_hessian_fn composition must be 'auto', "
                "'forward-over-reverse', or 'forward-over-forward'; "
                f"got {composition!r}."
            )
        if composition not in supported:
            raise ConfigurationError(
                f"Hessian composition {composition!r} is not supported by every "
                f"JointInferenceProblem child; common modes are {sorted(supported)}."
            )
        return composition

    def loss_grad_hessian_fn(self, composition: str = "auto") -> Callable:
        """Return cached joint data-misfit value, gradient, and true Hessian."""
        resolved = self.resolved_hessian_composition(composition)
        cached = self._loss_grad_hessian_fns.get(resolved)
        if cached is None:
            ad = "reverse" if resolved == "forward-over-reverse" else "forward"
            cached = _loss_grad_hessian_from_gradient(self.loss_and_grad_fn(ad))
            self._loss_grad_hessian_fns[resolved] = cached
        return cached

    @property
    def evaluate_lsq_fn(self) -> Optional[Callable]:
        """Reduced ``x -> (cost, J^T r, J^T J, n_active_rows, any_failed)`` (or ``None``).

        Available on the homogeneous path.  Forms the Gauss-Newton statistics by summing
        each child's local contributions across the device mesh, so a sharded joint is
        solved without assembling the global residual or Jacobian on one device.  On one
        device it reduces locally and is numerically identical to contracting the batched
        Jacobian.
        """
        return self._residual_seam("forward")[4]

    @property
    def step_balance(self) -> Optional[_StepBalance]:
        """The step planner state behind the homogeneous reduction, or ``None``.

        Exposes the last trial-step counts per child (``last_steps``), the current
        slot order (``order``), and how many reassignments were applied
        (``reassignments``); see ``Layout.balance_steps``.
        """
        return getattr(self.evaluate_lsq_fn, "step_balance", None)

    @property
    def evaluate_cost_fn(self) -> Optional[Callable]:
        """Reduced ``x -> (cost, n_active_rows, any_failed)`` for trial steps (or ``None``).

        Available on the homogeneous path; reduces the scalar cost across the mesh
        without gathering the residual.
        """
        return self._residual_seam("forward")[5]

    @property
    def unflatten_fn(self):
        """No single fitted problem for a joint vector (per-child reconstruction)."""
        return None

    @property
    def prob_static(self):
        """No single static for a joint vector (per-child reconstruction)."""
        return None

    @property
    def theta_names(self) -> tuple:
        """Per-scalar names of the joint θ-block (shared then per-child locals)."""
        return self._theta_names

    @property
    def function_columns(self) -> tuple:
        """The columns of the joint theta block that hold learnable-function (network) weights."""
        layout = self._layout
        columns: set = set()
        for index, child in enumerate(self.problems):
            child_columns = getattr(child, "function_columns", ())
            if not child_columns:
                continue
            # Child theta column -> joint column, as _reconstruct_child_flat scatters them.
            to_joint = {
                col: layout.shared_slice[0] + k for k, col in enumerate(layout.shared_cols[index])
            }
            to_joint.update(zip(layout.owner_cols[index], layout.child_tied_columns[index]))
            start = layout.local_slices[index][0]
            to_joint.update((col, start + k) for k, col in enumerate(layout.local_cols[index]))
            columns.update(to_joint[col] for col in child_columns if col in to_joint)
        return tuple(sorted(columns))

    @property
    def is_homogeneous(self) -> bool:
        """Whether the geometry-axis vmap kernel is engaged (vs static concatenation)."""
        return self._homogeneous

    @property
    def is_decision_variable(self) -> bool:
        """Whether at least one child carries decision-window state blocks."""
        return any(child.is_decision_variable for child in self.problems)

    @property
    def theta_only_steps(self) -> int:
        """Requested joint freeze-then-relax budget (maximum child request)."""
        return max(
            (int(getattr(child, "theta_only_steps", 0) or 0) for child in self.problems),
            default=0,
        )

    def theta_only_problem(self) -> "JointInferenceProblem":
        """Return a joint view with every trajectory-state coordinate frozen.

        The view remains a
        [JointInferenceProblem][diff_soup.optimization.inference.fitting.joint.JointInferenceProblem], so ownership-aware
        Block-LM still applies. Owner residuals receive only global/owner/local
        parameter coordinates and splice the fixed state tail back in before
        evaluating the original kernels.
        """
        if not self.is_decision_variable:
            raise ValueError("theta_only_problem needs decision-variable children.")
        n_active = int(self._layout.n_theta_joint)
        full_start = jnp.asarray(self._x0)
        # Build the original cached owner functions before copying.
        original_owner_fns = self.owner_residual_fns
        original_owner_indices = tuple(
            self._owner_parameter_indices_at(i) for i in range(len(self.owner_ids))
        )

        out = copy.copy(self)
        frozen_state_slices = tuple((n_active, n_active) for _ in self.problems)
        reduced_layout = dataclasses.replace(
            self._layout,
            state_slices=frozen_state_slices,
            n_joint=n_active,
        )
        reduced_owner_fns = []
        for owner_index, (owner_fn, old_indices) in enumerate(
            zip(original_owner_fns, original_owner_indices)
        ):
            new_indices = tuple(range(reduced_layout.n_shared)) + tuple(
                index
                for index in self._owner_parameter_indices_at(owner_index)[self._layout.n_shared :]
                if index < n_active
            )
            old_point = full_start[jnp.asarray(old_indices, dtype=jnp.int32)]
            old_position = {column: pos for pos, column in enumerate(old_indices)}
            target_positions = jnp.asarray(
                [old_position[column] for column in new_indices], dtype=jnp.int32
            )

            def reduced_owner(
                q,
                *,
                _owner_fn=owner_fn,
                _old_point=old_point,
                _positions=target_positions,
            ):
                return _owner_fn(_old_point.at[_positions].set(q))

            reduced_owner_fns.append(jax.jit(reduced_owner))

        object.__setattr__(out, "_layout", reduced_layout)
        object.__setattr__(out, "_x0", np.asarray(self._x0)[:n_active].copy())
        object.__setattr__(
            out,
            "regularization",
            _FrozenTailRegularizer(self.regularization, np.asarray(self._x0), n_active),
        )
        object.__setattr__(
            out,
            "_segment_fns",
            (lambda x: self.segment_fns[0](jnp.concatenate((x, full_start[n_active:]))),),
        )

        def reduced_residual(ad):
            full_residual = self._residual_seam(ad)[3]
            return lambda x: full_residual(jnp.concatenate((x, full_start[n_active:])))

        object.__setattr__(out, "_residual_builder", reduced_residual)
        object.__setattr__(out, "_batched_r", None)
        object.__setattr__(out, "_batched_rj", None)
        object.__setattr__(out, "_stacked_r", None)
        object.__setattr__(out, "_stacked_rj", None)
        object.__setattr__(out, "_residual_seams", {})
        object.__setattr__(out, "_loss_fn", None)
        object.__setattr__(out, "_loss_grad_fns", {})
        object.__setattr__(out, "_loss_grad_hessian_fns", {})
        object.__setattr__(out, "_evaluate_lsq", None)
        object.__setattr__(out, "_evaluate_cost", None)
        # The reduced wrappers are owner-specific compact closures. They retain
        # the full problem's owner-device placement while frozen state
        # coordinates disappear from every executable input.
        object.__setattr__(out, "_owner_residual_fns", tuple(reduced_owner_fns))
        object.__setattr__(out, "_owner_reducer_fns", tuple(None for _ in reduced_owner_fns))
        object.__setattr__(out, "_owner_reducer_inputs", tuple(None for _ in reduced_owner_fns))
        object.__setattr__(out, "_owner_narrow_kernel_cache", None)
        object.__setattr__(out, "_owner_point_plan_cache", None)
        object.__setattr__(out, "_restricted_initialization", True)
        if self._trial_predicate is not None:
            original_predicate = self._trial_predicate

            def reduced_predicate(x):
                return original_predicate(jnp.concatenate((jnp.asarray(x), full_start[n_active:])))

            object.__setattr__(out, "_trial_predicate", reduced_predicate)

        # Result reconstruction needs the frozen coordinates: splice the fixed
        # state tail back and delegate to the full problem, exactly as the
        # reduced residual and trial-predicate wrappers do. Without this, children
        # receive a stateless vector their window layouts reject.
        full_start_host = np.asarray(self._x0)

        def reduced_child_results(x_flat: np.ndarray) -> tuple:
            return self.child_results(
                np.concatenate((np.asarray(x_flat), full_start_host[n_active:]))
            )

        def reduced_shared_values(x_flat: np.ndarray) -> dict:
            return self.shared_values(
                np.concatenate((np.asarray(x_flat), full_start_host[n_active:]))
            )

        object.__setattr__(out, "child_results", reduced_child_results)
        object.__setattr__(out, "shared_values", reduced_shared_values)
        return out

    def with_warm_theta(self, x_full: np.ndarray) -> "JointInferenceProblem":
        """Warm only the joint parameter prefix, retaining state coordinates."""
        n_theta = int(self._layout.n_theta_joint)
        values = np.asarray(x_full)
        if values.size < n_theta:
            raise ValueError(
                f"joint theta warm start needs at least {n_theta} values; got {values.size}."
            )
        start = np.asarray(self._x0).copy()
        start[:n_theta] = values[:n_theta]
        return self.with_start(start)

    @property
    def baseline_reached_tend(self) -> bool:
        """Whether every child's forward solve at ``x0`` reaches ``t_end``.

        Each child's solve runs the first time this is read, not at
        construction.
        """
        if self._baseline_reached_tend is None:
            reached = all(bool(c.baseline_reached_tend) for c in self.problems)
            object.__setattr__(self, "_baseline_reached_tend", reached)
        return self._baseline_reached_tend

    # --- device placement ---
    @property
    def layout(self) -> Optional[Layout]:
        """The device-placement [Layout][diff_soup.Layout], or ``None`` (one device)."""
        return self._placement

    @property
    def ownership_layout(self) -> JointLayout:
        """Host-static slices for global, owner, child-local, and state blocks."""
        return self._layout

    @property
    def ownership_groups(self) -> tuple:
        """Names of parameter groups represented once per owner."""
        return self._layout.ownership_groups

    @property
    def owner_ids(self) -> tuple:
        """Stable owner labels in joint-vector order."""
        return self._layout.owner_ids

    @property
    def child_owner_indices(self) -> tuple:
        """Per child, the integer position of its owner in owner_ids.

        See [owner_ids][diff_soup.optimization.inference.fitting.joint.JointInferenceProblem.owner_ids].
        """
        return self._layout.child_owner_indices

    def child_flat(self, x_flat: ArrayLike, child_index: int) -> jax.Array:
        """Reconstruct one child's complete flat vector from a joint vector."""
        if not 0 <= int(child_index) < len(self.problems):
            raise IndexError(
                f"child_index must be in [0, {len(self.problems)}); got {child_index}."
            )
        return _reconstruct_child_flat(jnp.asarray(x_flat), int(child_index), self._layout)

    def owner_parameter_indices(self, owner: Hashable) -> tuple:
        """Indices of ``[global | owner | its child locals/states]``.

        ``owner`` is an ID from
        [owner_ids][diff_soup.optimization.inference.fitting.joint.JointInferenceProblem.owner_ids];
        integer IDs are treated as
        IDs, never ambiguously as positional indices. The returned order is the
        coordinate expected by
        [owner_residual_fns]
        [diff_soup.optimization.inference.fitting.joint.JointInferenceProblem.owner_residual_fns].
        """
        try:
            owner_index = self.owner_ids.index(owner)
        except ValueError as exc:
            raise KeyError(
                f"unknown owner ID {owner!r}; known owners are {self.owner_ids}."
            ) from exc
        return self._owner_parameter_indices_at(owner_index)

    def _owner_parameter_indices_at(self, owner_index: int) -> tuple:
        """Narrow parameter indices for one positional owner index.

        The owner's columns are its own block and the blocks of the members
        below it, level by level, then its children's locals and states.
        """
        indices: list[int] = list(range(self._layout.n_shared))
        indices.extend(self._layout.owner_columns[owner_index])
        for child_index, child_owner in enumerate(self.child_owner_indices):
            if child_owner != owner_index:
                continue
            lo, hi = self._layout.local_slices[child_index]
            indices.extend(range(lo, hi))
        for child_index, child_owner in enumerate(self.child_owner_indices):
            if child_owner != owner_index:
                continue
            lo, hi = self._layout.state_slices[child_index]
            indices.extend(range(lo, hi))
        return tuple(indices)

    def _require_one_level(self, route: str) -> None:
        """Refuse ``route``, which sums one owner tier, when groups are tied at nested levels.

        Raises:
            ConfigurationError: If the joint ties groups at more than one level.
        """
        n_levels = len(self._layout.level_ids)
        if n_levels <= 1:
            return
        names = ", ".join(repr(name) for name in self._level_names)
        raise ConfigurationError(
            f"{route} sums each owner's experiments into one block over one intermediate "
            f"level, and this problem's hierarchy has {n_levels} ({names}); evaluate such a "
            "study through ds.lower(study), with LoweredStudy.evaluate and "
            "engine.reduce_levels."
        )

    @property
    def owner_residual_fns(self) -> tuple:
        """Cached residuals over each owner's narrow parameter coordinate.

        Function ``i`` accepts exactly the columns returned by
        ``_owner_parameter_indices_at(i)`` (or by
        ``owner_parameter_indices(owner_ids[i])``). Homogeneous children of one owner run
        through the same geometry-axis ``vmap`` kernel as the full joint;
        heterogeneous owner groups use static concatenation. No function sees
        coordinates belonging to another owner.

        Raises:
            ConfigurationError: If the joint ties groups at more than one level.
        """
        self._require_one_level("JointInferenceProblem.owner_residual_fns")
        if self._owner_residual_fns is None:
            built = tuple(
                self._owner_residual_kernel_at(owner_index)
                for owner_index in range(len(self.owner_ids))
            )
            object.__setattr__(self, "_owner_residual_fns", tuple(item[0] for item in built))
            object.__setattr__(self, "_owner_reducer_fns", tuple(item[1] for item in built))
            object.__setattr__(self, "_owner_reducer_inputs", tuple(item[2] for item in built))
        return self._owner_residual_fns

    def owner_residual_fn_at(self, owner_index: int, ad: Optional[str] = None) -> Callable:
        """Return one cached narrow residual for the requested AD mode."""
        owner_index = self._checked_owner_index(owner_index)
        if ad is None and self._owner_residual_fns is not None:
            return self._owner_residual_fns[owner_index]
        return self._owner_residual_kernel_at(owner_index, ad=ad)[0]

    def _owner_residual_kernel_at(self, owner_index: int, ad: Optional[str] = None) -> tuple:
        self._require_one_level("JointInferenceProblem.owner_residual_fns")
        owner_index = self._checked_owner_index(owner_index)
        cache = self._owner_kernel_cache()
        key = ("residual_owner", ad, owner_index)
        built = cache.get(key)
        if built is not None:
            return built
        member_indices = tuple(
            child_index
            for child_index, child_owner in enumerate(self.child_owner_indices)
            if child_owner == owner_index
        )
        children = tuple(self.problems[index] for index in member_indices)
        owner_shared = self.shared + self.ownership_groups
        owner_layout = _build_layout(children, owner_shared)
        if _is_homogeneous(children):
            selected_children = (
                children if ad is None else tuple(child._view_for_ad(ad) for child in children)
            )
            (
                residual,
                _rj,
                _lsq,
                _cost,
                _rows,
                child_fn,
                child_inputs,
            ) = _build_homogeneous_residual(
                selected_children,
                owner_layout,
                _child_mode(selected_children[0]),
                None,
            )
        else:
            indices = self._owner_parameter_indices_at(owner_index)
            owner_x0 = np.asarray(self.x0)[list(indices)]
            residual, _rows = _build_static_residual(
                children,
                owner_layout,
                owner_x0,
                ad=ad,
            )
            residual = jax.jit(residual)
            child_fn = None
            child_inputs = None
        built = (residual, child_fn, child_inputs)
        cache[key] = built
        return built

    @property
    def owner_reducer_batches(self) -> tuple:
        """Per-owner child kernels and numeric child batches for block reducers."""
        self.owner_residual_fns  # build the three aligned caches
        return tuple(zip(self._owner_reducer_fns, self._owner_reducer_inputs))

    def _owner_kernel_cache(self) -> dict:
        cache = self._owner_narrow_kernel_cache
        if cache is None:
            cache = {}
            object.__setattr__(self, "_owner_narrow_kernel_cache", cache)
        return cache

    def _checked_owner_index(self, owner_index: int) -> int:
        owner_index = int(owner_index)
        if owner_index < 0 or owner_index >= len(self.owner_ids):
            raise IndexError(
                f"owner_index must be in [0, {len(self.owner_ids)}); got {owner_index}."
            )
        return owner_index

    @property
    def is_restricted_initialization(self) -> bool:
        """Whether this is the compact frozen-state initialization view."""
        return self._restricted_initialization

    @property
    def owner_point_plans(self) -> tuple:
        """Compact gather/scatter plans in deterministic owner order."""
        cache = self._owner_point_plan_cache
        if cache is None:
            from diff_soup.optimization.inference.hierarchical.owner_points import (
                build_owner_point_plans,
            )

            cache = build_owner_point_plans(self)
            object.__setattr__(self, "_owner_point_plan_cache", cache)
        return cache

    @property
    def is_sharded(self) -> bool:
        """Whether the placement requests child-batch sharding (a data mesh axis)."""
        return _sharding_requested(self._placement)

    @property
    def placement(self) -> tuple[tuple[int, int], ...]:
        """Per child, ``(lane, process_index)``: the experiment lane that evaluates it.

        See [child_lanes][diff_soup.optimization.inference.fitting._joint_structure.child_lanes] for the rule.
        """
        return child_lanes(self.problems, self.shared, self.ownership, self._placement)

    # --- diagnostics-by-name surface ---
    @property
    def stream_names(self) -> tuple:
        """Joint fits report cost per child block, not per named stream."""
        return ()

    @property
    def stream_residual_slices(self) -> tuple:
        """Rows are organized per child, not per stream."""
        return ()

    @property
    def param_group_names(self) -> tuple:
        """Named parameter groups over the joint θ-block (shared once, locals per child)."""
        return self._param_group_names

    @property
    def param_group_cols(self) -> tuple:
        """Per-group joint θ-block columns (parallel to ``param_group_names``)."""
        return self._param_group_cols

    @property
    def parameter_schema(self):
        """Recipe handles mapped to shared, owner-tied, or child-local values.

        Shared handles return one value. Owner-tied handles return a mapping
        keyed by the label of the member of their level (the owner label when
        one level is tied), and local handles use child indices. Handles
        from any compatible child's Recipe resolve by declared name and size.
        If any child lacks Recipe handle metadata, this property is None.
        """
        import hashlib

        from diff_soup.problem.parameters import (
            ParameterEntry,
            ParameterOccurrence,
            ParameterSchema,
        )

        schemas = tuple(getattr(child, "parameter_schema", None) for child in self.problems)
        if any(schema is None for schema in schemas):
            return None
        entries = {}
        occurrences = {}
        layout = self._layout
        for index, schema in enumerate(schemas):
            shared = dict(zip(layout.shared_cols[index], range(*layout.shared_slice)))
            local = dict(zip(layout.local_cols[index], range(*layout.local_slices[index])))
            owned = dict(zip(layout.owner_cols[index], layout.child_tied_columns[index]))
            # The level and member label of each tied child column, in the
            # order of ``owner_cols`` (level by level, groups in declared order).
            member_of = {}
            child = self.problems[index]
            columns_in_order = iter(layout.owner_cols[index])
            for level, member in enumerate(layout.child_levels[index]):
                for group in layout.level_groups[level]:
                    for _ in _child_group_cols(child, group) or ():
                        member_of[next(columns_in_order)] = (
                            level,
                            layout.level_ids[level][member],
                        )
            mapping = {**shared, **owned, **local}
            for entry in schema.entries:
                name = entry.handle.name
                previous = entries.setdefault(name, entry)
                if previous.size != entry.size:
                    raise ValueError(
                        f"joint handle {name!r} has incompatible decoded sizes "
                        f"{previous.size} and {entry.size}; child declarations must agree"
                    )
                for occurrence in entry.occurrences:
                    columns = occurrence.columns
                    if set(columns) <= shared.keys():
                        label = None
                    elif set(columns) <= owned.keys() and (
                        len({member_of[c] for c in columns}) == 1
                    ):
                        label = member_of[columns[0]][1]
                    elif set(columns) <= local.keys():
                        label = index
                    else:
                        raise ValueError(
                            f"joint handle {name!r} spans ownership levels; select its "
                            "complete value in one shared, owner, or local group"
                        )
                    value = ParameterOccurrence(
                        owner=label,
                        columns=tuple(mapping[c] for c in columns),
                        declarations=occurrence.declarations,
                    )
                    values = occurrences.setdefault(name, [])
                    if value not in values:
                        values.append(value)
                    else:
                        # The children sharing an occurrence each declare it: a MAP checks
                        # that their priors agree.
                        at = values.index(value)
                        values[at] = dataclasses.replace(
                            values[at],
                            declarations=values[at].declarations + value.declarations,
                        )
        record = (
            tuple(schema.fingerprint for schema in schemas),
            self.param_group_names,
            self.param_group_cols,
            tuple((name, tuple(values)) for name, values in occurrences.items()),
        )
        fingerprint = "structural:" + hashlib.sha256(repr(record).encode()).hexdigest()
        return ParameterSchema(
            [
                ParameterEntry(
                    handle=entry.handle,
                    occurrences=tuple(occurrences[name]),
                    size=entry.size,
                    decode=entry.decode,
                )
                for name, entry in entries.items()
            ],
            fingerprint=fingerprint,
        )

    def count_failed(self, x_flat: np.ndarray) -> tuple[int, tuple[int, ...]]:
        """How many children hit the failure sentinel (re-evaluated host-side)."""
        fn = self._batched_r if self._batched_r is not None else self._segment_fns[0]
        r = fn(jnp.asarray(x_flat))
        if isinstance(r, jax.Array) and not r.is_fully_addressable:
            from diff_soup._jax_compat import multihost_utils

            r = multihost_utils.process_allgather(r, tiled=True)
        r = np.asarray(r)
        idx: list[int] = []
        start = 0
        for i, n in enumerate(self._rows_per_child):
            block = r[start : start + n]
            if block.size and np.any(np.abs(block) >= _FAILED_SENTINEL):
                idx.append(i)
            start += n
        return len(idx), tuple(idx)

    # --- joint result reconstruction ---
    def child_results(self, x_flat: np.ndarray) -> tuple:
        """Reconstruct each child's fitted problem + local/state values from ``x_flat``.

        Returns a tuple of
        [ChildFit][diff_soup.optimization.inference.fitting.result.ChildFit], one per
        child in input order.
        """
        x = jnp.asarray(x_flat)
        out = []
        for i, child in enumerate(self.problems):
            xc = _reconstruct_child_flat(x, i, self._layout)
            fitted = None
            if child.unflatten_fn is not None and child.prob_static is not None:
                import equinox as eqx

                fitted = eqx.combine(child.unflatten_fn(xc), child.prob_static)
            local = {}
            for gname in child.parameters.local:
                cols = _child_group_cols(child, gname)
                if cols:
                    local[gname] = np.asarray(xc)[list(cols)]
            window_states = child.window_states_of(np.asarray(xc))
            out.append(ChildFit(opt_params=fitted, local=local, window_states=window_states))
        return tuple(out)

    def shared_values(self, x_flat: np.ndarray) -> dict:
        """Extract the fitted value of each shared group from the joint vector."""
        x = np.asarray(x_flat)
        vals = {}
        start = 0
        rep = self.problems[0]
        for gname in self.shared:
            cols = _child_group_cols(rep, gname)
            n = len(cols) if cols else 0
            vals[gname] = x[start : start + n]
            start += n
        return vals

    def owner_values(self, x_flat: np.ndarray) -> dict:
        """Extract each owner-tied group's fitted value, keyed by owner ID.

        Raises:
            ConfigurationError: If the joint ties groups at more than one level.
        """
        self._require_one_level("JointInferenceProblem.owner_values")
        x = np.asarray(x_flat)
        values: dict = {}
        for owner_index, owner_id in enumerate(self._layout.owner_ids):
            groups: dict = {}
            for group, (lo, hi) in zip(
                self._layout.ownership_groups,
                self._layout.owner_group_slices[owner_index],
            ):
                groups[group] = x[lo:hi]
            values[owner_id] = groups
        return values
