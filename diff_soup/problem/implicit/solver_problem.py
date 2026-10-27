"""The engine-facing view of one solver problem.

Time-stepping engines consume a single object per solve: the residual
``F(t, y, ydot, theta, control)`` and the exact iteration-matrix
blocks of ``A(cj) = F_y + cj * F_ydot`` — the state block (dense, or values
over the view's ``state_topology``) plus the bordered-global coupling blocks
for the Schur reduction — all read from one shared linearization plan with
``cj`` applied exactly once. An algebraic global's row is the negated
constraint that closes it; a differential global's row is the ordinary
``d(global)/dt - rhs``. Either way the coupling blocks already match the
engines' Schur convention, so no engine re-scales or re-signs them.

An authored problem may retain its source object behind the view for parameter
defaults and state-layout conversion, but every numerical operation is read
from the solver problem and its shared linearization plan.

Views are memoized on the content of the problem they present, so a repeated
solve with the same problem, or with a rebuilt problem equal to it, reuses one
view and one compiled program.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Any, Hashable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._precision import Precision
from diff_soup.linear import SparseTopology
from diff_soup.problem._cache import BoundedContentCache
from diff_soup.problem._ir.content_hash import _graph_callable_config_id
from diff_soup.problem.implicit.consistent_init import (
    ConsistentInitPlan,
    consistent_initial_state,
    reduced_route_eligible,
    resolve_backend_route,
)
from diff_soup.problem.implicit.linearize import build_linearization_plan
from diff_soup.problem.implicit.problems import ImplicitProblem, MassProblem
from diff_soup.problem.implicit.root_ift import (
    implicit_step_root,
    iteration_matrix_offering,
)
from diff_soup.problem.implicit.signature import StructureSignature, declared_structure_key
from diff_soup.problem.implicit.theta_codec import ThetaCodec, pytree_structure_signature

__all__ = [
    "SolverView",
    "StructureSignature",
    "ThetaCodec",
    "get_or_build_solver_view",
]


def _view_precision(problem: Any, source: Any) -> Precision:
    """The one precision a solver problem and its authored source declare.

    The solver problem's policy is authoritative. An authored source that
    carries its own ``precision`` must agree with it, unless one of the two
    is the default policy.
    """
    model = getattr(problem, "precision", None)
    model = model if isinstance(model, Precision) else Precision()
    declared = getattr(source, "precision", None) if source is not None else None
    if not isinstance(declared, Precision) or declared == model:
        return model
    if model == Precision():
        return declared
    if declared == Precision():
        return model
    raise ConfigurationError(
        f"{type(source).__name__} declares precision {declared!r}, but the solver "
        f"problem its solver_problem() returns declares {model!r}; build the solver "
        "problem with the source's precision."
    )


class SolverView:
    """One solver problem, presented the way the engines consume it.

    Attributes:
        problem: The solver problem (mass form or fully implicit).
        source: The authored problem that supplied ``solver_problem()``, or
            ``None`` when the solver problem was supplied directly.
        plan: The shared exact linearization plan.
        structure: The problem's variable-role structure.
        n: Total unknowns.
        n_u: Local-state block length (everything before the bordered globals).
        n_lam: Bordered-global block length, including differential and
            algebraic globals.
        id_mask: Differential-row mask over the augmented state.
        precision: The problem's [Precision][diff_soup._precision.Precision].
        state_topology: The state block's sparsity pattern (the union
            pattern's state sub-block), or ``None`` on the dense path.

    Two views are equal when they present one model: sources with one
    declared structure key that share one solver problem (value copies of one
    lowering), problems equal in content, or one source object. A compiled
    solve takes the view as a static argument and every value through
    ``theta``, so equal views share one compiled program, and a view rebuilt
    for a model compares equal to the view it replaces.
    """

    def _identity(self) -> tuple:
        """The model key, with the solver problem for a declared structure key."""
        key = self.structural_signature.source_key
        if key[0] == "declared":
            # Value copies of one lowering share one solver problem; a copy
            # whose static fields differ gets its own, so it stays distinct.
            return (key, id(self.problem))
        return (key,)

    def __hash__(self) -> int:
        """Hash of the model key."""
        return hash(self._identity())

    def __eq__(self, other) -> bool:
        """Equal when both present one model."""
        if self is other:
            return True
        if not isinstance(other, SolverView):
            return NotImplemented
        return self._identity() == other._identity()

    def __init__(
        self,
        problem,
        source: Optional[Any] = None,
        *,
        direct_source: bool = False,
        source_key: Optional[Hashable] = None,
    ):
        """Build the gather maps that split plan values into engine blocks.

        ``source_key`` is the model key of the structure signature; ``None``
        uses the source's declared structure key, or the source's identity.
        """
        self.source = source
        self.direct_source = bool(direct_source)
        self.problem = problem
        control_evaluator = getattr(source, "_controls_fn", None)
        self.control_bindings = (
            None if control_evaluator is None else getattr(control_evaluator, "bindings", None)
        )
        self._control_source_static = None
        if self.control_bindings is not None and not self.control_bindings.is_empty:
            _dynamic, self._control_source_static = eqx.partition(source, eqx.is_inexact_array)
        table_getter = getattr(source, "operational_table", None)
        self.operational_table = table_getter() if callable(table_getter) else None
        self._operational_lam_offset = None
        if self.operational_table is not None:
            closes_name = self.operational_table.closes.name
            for entry in source.global_layout():
                if entry["name"] == closes_name:
                    self._operational_lam_offset = int(entry["offset"])
                    break
        self.plan = build_linearization_plan(problem)
        self.structure = problem.structure
        n = int(self.structure.n)
        self.n = n

        partitions = {name: (lo, hi) for name, lo, hi in self.structure.partitions}
        if "global" in partitions:
            lo, hi = partitions["global"]
            self.n_u, self.n_lam = int(lo), int(hi - lo)
        elif self.direct_source:
            self.n_u = int(source.state_size())
            self.n_lam = int(source.constraint_dim())
        elif source is not None:
            source_n_lam = int(getattr(source, "n_lam", 0) or 0)
            self.n_u = int(getattr(source, "n_u", n - source_n_lam))
            self.n_lam = source_n_lam
        else:
            self.n_u, self.n_lam = n, 0
        if self.n_u + self.n_lam != n:
            raise ValueError(
                f"the source declares n_u={self.n_u} and n_lam={self.n_lam}, "
                f"but its solver problem has {n} unknowns."
            )

        self.id_mask = self.structure.differential_vars.astype(np.int32)
        self.precision = _view_precision(problem, source)

        self.state_topology: Optional[SparseTopology] = None
        if self.plan.mode == "sparse":
            rows = np.asarray(self.plan.union_topology.rows)
            cols = np.asarray(self.plan.union_topology.cols)
            n_u = self.n_u
            in_state = (rows < n_u) & (cols < n_u)
            self._state_take = np.flatnonzero(in_state)
            self.state_topology = SparseTopology(
                rows=rows[self._state_take], cols=cols[self._state_take], n_dofs=n_u
            )
            b_mask = (rows < n_u) & (cols >= n_u)
            c_mask = (rows >= n_u) & (cols < n_u)
            d_mask = (rows >= n_u) & (cols >= n_u)
            self._b_take = np.flatnonzero(b_mask)
            self._b_rows, self._b_cols = rows[b_mask], cols[b_mask] - n_u
            self._c_take = np.flatnonzero(c_mask)
            self._c_rows, self._c_cols = rows[c_mask] - n_u, cols[c_mask]
            self._d_take = np.flatnonzero(d_mask)
            self._d_rows, self._d_cols = rows[d_mask] - n_u, cols[d_mask] - n_u

        theta_getter = getattr(source, "default_theta", None)
        theta_template = theta_getter() if callable(theta_getter) else None
        self._theta_codecs: dict[str, ThetaCodec] = {}
        self.structural_signature = StructureSignature.build(
            source=source,
            problem=problem,
            plan=self.plan,
            n=self.n,
            n_u=self.n_u,
            n_lam=self.n_lam,
            id_mask=self.id_mask,
            partitions=tuple(self.structure.partitions),
            theta_structure=pytree_structure_signature(theta_template),
            control_bindings=self.control_bindings,
            source_key=source_key,
        )

    # ------------------------------------------------------------------
    # The engine surface
    # ------------------------------------------------------------------
    @property
    def full_topology(self) -> Optional[SparseTopology]:
        """The row-major full-system COO pattern of ``A(cj)``, or ``None``.

        Sparse [iteration_matrix_values][] entries use this exact ordering.
        Dense problems return ``None`` and expose a full ``(n, n)`` matrix.
        """
        return self.plan.union_topology if self.plan.mode == "sparse" else None

    @property
    def mass_dependence(self) -> str:
        """Classify derivative dependence from the declared mass operator."""
        if isinstance(self.problem, MassProblem):
            deps = self.problem.mass.dependencies
            if deps.state:
                return "state_strong"
            if deps.time:
                return "time"
            return "constant"
        # A fully implicit residual gets the most demanding classification.
        return "state_strong"

    def residual(self, t, y, ydot, theta, control=None):
        """The residual ``F(t, y, ydot, theta, control)``."""
        return self.problem.residual(t, y, ydot, theta, control)

    @property
    def has_recipe_controls(self) -> bool:
        """Whether this Recipe view declares controls that ride the override payload."""
        return bool(self.control_bindings is not None and not self.control_bindings.is_empty)

    @property
    def has_operational_control(self) -> bool:
        """Whether the source problem declares an operational control table."""
        return self.operational_table is not None

    def normalize_control(self, control, *, dtype=None):
        """Validate and shape the solve's control input.

        An explicit Recipe control dimensional mapping keeps its existing
        normalization.  For a problem with an operational control table the
        input must be an operating
        [ControlState][diff_soup.timestepping.control_state.ControlState];
        the returned state carries the mode resolved to its ``int32`` table
        index (names never enter traced data), the target cast to ``dtype``
        (the solve dtype), and a materialized current-seed leaf (``nan``
        marks "no explicit seed") so the state's pytree structure is stable
        across in-solve control updates. A target's time terms and origin, when
        it has them, are cast to the target's dtype.
        """
        from diff_soup._errors import ConfigurationError
        from diff_soup.timestepping.control_state import ControlState

        table = self.operational_table
        operating = isinstance(control, ControlState) and control.is_operating
        if table is None:
            if operating:
                raise ConfigurationError(
                    "this problem declares no operational control table, but "
                    "the control state carries an operating point; author the "
                    "table with spec.control.operational(...) or pass plain "
                    "control values."
                )
            if control is None or not self.has_recipe_controls:
                return control
            return self.control_bindings.normalize_explicit(control)
        modes = tuple(table.modes)
        if not operating:
            raise ConfigurationError(
                "this problem's operational control table needs an operating "
                "point; pass control_state0=ControlState.operating(mode=..., "
                f"target=...) with a mode from {modes!r}."
            )
        if control.mode_name is not None:
            if control.mode_name not in modes:
                raise ConfigurationError(
                    f"unknown operational mode {control.mode_name!r}; this "
                    f"problem's modes are {modes!r}."
                )
            mode = jnp.asarray(modes.index(control.mode_name), dtype=jnp.int32)
        else:
            mode = jnp.asarray(control.mode, dtype=jnp.int32)
            if not isinstance(control.mode, jax.core.Tracer):
                idx = int(mode)
                if not 0 <= idx < len(modes):
                    raise ConfigurationError(
                        f"operational mode index {idx} is out of range; this "
                        f"problem declares {len(modes)} modes {modes!r}."
                    )
        target = (
            jnp.asarray(control.target) if dtype is None else jnp.asarray(control.target, dtype)
        )
        seed_dtype = target.dtype
        if control.current_seed is None:
            seed = jnp.full((), jnp.nan, dtype=seed_dtype)
        else:
            seed = jnp.asarray(control.current_seed, dtype=seed_dtype)
        values = control.values
        if values is not None and self.has_recipe_controls:
            values = self.control_bindings.normalize_explicit(values)
        terms = getattr(control, "target_terms", None)
        origin = getattr(control, "target_origin", None)
        if terms is not None:
            terms = jnp.asarray(terms, dtype=target.dtype)
            origin = jnp.asarray(origin, dtype=target.dtype)
        return ControlState(
            values=values,
            epoch=jnp.asarray(control.epoch, dtype=jnp.int32),
            mode=mode,
            target=target,
            current_seed=seed,
            mode_name=None,
            target_terms=terms,
            target_origin=origin,
        )

    @staticmethod
    def control_epoch(control):
        """Return the explicit control generation, or epoch zero.

        Consumers use this accessor instead of introspecting a
        [ControlState][] or inventing a second control wrapper. Plain
        mappings and authored-callable ``None`` controls are generation zero.
        """
        epoch = getattr(control, "epoch", None)
        if epoch is None:
            return jnp.zeros((), dtype=jnp.int32)
        return jnp.asarray(epoch, dtype=jnp.int32)

    def control_values(self, t, theta, control):
        """Resolve the dimensional Recipe values visible to the model and event callbacks.

        ``t`` is internal time; each authored control callable is handed the dimensional
        time it corresponds to, and its dimensional return value is passed back as-is.
        Direct and derived-quantity controls both appear.  The residual
        rescales the derived ones to the driven multipliers' own units when it
        writes the closing rows.
        """
        if not self.has_recipe_controls:
            return control
        dynamic = getattr(theta, "inner", theta)
        concrete = eqx.combine(dynamic, self._control_source_static)
        values = self.control_bindings.bound_values(concrete._controls, t, control)
        from diff_soup.timestepping.control_state import ControlState

        if isinstance(control, ControlState) and control.is_operating:
            return eqx.tree_at(
                lambda state: state.values,
                control,
                values,
                is_leaf=lambda leaf: leaf is None,
            )
        return values

    def operational_seed_multipliers(self, t0, u0, lam0, theta, control):
        """Bordered-global seed with the operational algebraic-current slot resolved.

        An explicit current seed always wins. Otherwise, when the active mode
        is ``"power"``, the terminal voltage at the start state is probed
        through the table's own voltage row; when its magnitude exceeds the
        table's ``seed_voltage_threshold`` the current slot defaults to
        ``target / V``. A nonzero power target at near-zero voltage with no
        explicit seed fails here — before any Newton iteration — with a typed
        error; under tracing (where no value is concrete) the guarded select
        keeps the prior slot instead. Non-power modes write only an explicit
        seed.
        """
        from diff_soup._errors import ConfigurationError

        table = self.operational_table
        if table is None or control is None or getattr(control, "target", None) is None:
            return lam0
        off = self._operational_lam_offset
        modes = tuple(table.modes)
        thr = float(table.seed_voltage_threshold)
        lam0 = jnp.asarray(lam0)
        u0 = jnp.asarray(u0)
        target = jnp.asarray(control.target)
        if getattr(control, "target_terms", None) is not None:
            # A target that varies in time seeds with its value at the start.
            target = jnp.asarray(control.target_at(t0))
        seed = None if control.current_seed is None else jnp.asarray(control.current_seed)
        mode = jnp.asarray(control.mode, dtype=jnp.int32)
        prior = lam0[off]
        # Probe V(y0) through the table's own voltage row: a negative mode is
        # the internal voltage-probe selector (users cannot construct one),
        # with a zero target so the row value IS the terminal voltage.
        from diff_soup.timestepping.control_state import ControlState

        probe = ControlState(
            values=control.values,
            epoch=control.epoch,
            mode=jnp.asarray(-1, dtype=jnp.int32),
            target=jnp.zeros_like(target),
            current_seed=None,
            mode_name=None,
        )
        y = jnp.concatenate([u0, lam0.astype(u0.dtype)])
        rhs = getattr(self.problem, "rhs", None)
        if callable(rhs):
            v0 = rhs(t0, y, theta, probe)[self.n_u + off]
        else:
            # This operational-control slot is an algebraic global whose
            # residual row is the negated constraint; recover the row value
            # with a sign flip.
            v0 = -self.problem.residual(t0, y, jnp.zeros_like(y), theta, probe)[self.n_u + off]
        traced = any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in (v0, target, mode, prior) + (() if seed is None else (seed,))
        )
        if not traced:
            has_seed = seed is not None and not bool(jnp.isnan(seed))
            if has_seed:
                slot = seed
            elif modes[int(mode)] == "power":
                if abs(float(v0)) > thr:
                    slot = target / v0
                elif float(target) != 0.0:
                    raise ConfigurationError(
                        f"operational control {table.name!r}: power mode with "
                        f"target {float(target)!r} starts at terminal voltage "
                        f"{float(v0)!r} (magnitude at or below the seed "
                        f"threshold {thr!r}), so the current unknown has no "
                        "usable default start; supply a current seed "
                        "(ControlState.operating(current_seed=...)) or a "
                        "physically valid nonzero voltage seed."
                    )
                else:
                    slot = prior
            else:
                slot = prior
        else:
            seed_arr = jnp.full((), jnp.nan, dtype=target.dtype) if seed is None else seed
            has_seed = jnp.logical_not(jnp.isnan(seed_arr))
            seed_val = jnp.where(has_seed, jnp.nan_to_num(seed_arr), jnp.zeros_like(seed_arr))
            power_mask = jnp.asarray(np.array([m == "power" for m in modes]))
            is_power = jnp.take(power_mask, mode)
            ok_v = jnp.abs(v0) > thr
            v_safe = jnp.where(ok_v, v0, jnp.ones_like(v0))
            power_default = jnp.where(ok_v, target / v_safe, prior)
            slot = jnp.where(has_seed, seed_val, jnp.where(is_power, power_default, prior))
        return lam0.at[off].set(jnp.asarray(slot, dtype=lam0.dtype))

    def default_theta(self):
        """The default parameter pytree (``None`` when the problem declares none)."""
        if self.source is not None:
            getter = getattr(self.source, "default_theta", None)
            return getter() if callable(getter) else None
        return None

    def theta_codec(self, *, flat_dtype=None) -> ThetaCodec:
        """Return a cached native-buffer codec for the public theta pytree.

        ``flat_dtype`` selects the native buffer dtype. Leaf dtypes and shapes
        remain those of [default_theta][]; unraveling casts each segment
        back to its authored dtype so primal and derivative pytrees retain the
        public contract.
        """
        dtype_key = "<promoted>" if flat_dtype is None else np.dtype(flat_dtype).str
        codec = self._theta_codecs.get(dtype_key)
        if codec is None:
            codec = ThetaCodec.from_template(
                self.default_theta(),
                flat_dtype=flat_dtype,
            )
            self._theta_codecs[dtype_key] = codec
        return codec

    def to_internal_state(self, y0):
        """Map a source-authored initial state to the solver problem's internal units."""
        if self.source is None:
            return jnp.asarray(y0)
        if not self.direct_source:
            convert = getattr(self.source, "to_internal_state", None)
            if callable(convert):
                return convert(y0)
            return jnp.asarray(y0)
        initial_u = getattr(self.source, "initial_u", None)
        initial_lambda = getattr(self.source, "initial_lambda", None)
        if not callable(initial_u) or not callable(initial_lambda):
            return jnp.asarray(y0)
        n_total = self.n_u + self.n_lam
        lam = None
        u = y0
        if isinstance(y0, (jax.Array, np.ndarray)) and jnp.ndim(y0) == 1:
            if self.n_lam and int(y0.shape[0]) == n_total:
                u, lam = y0[: self.n_u], y0[self.n_u :]
        u_internal = jnp.asarray(initial_u(u)).reshape((-1,))
        if lam is None:
            lam_internal = jnp.asarray(initial_lambda()).reshape((-1,))
        else:
            # An explicitly augmented state already carries the bordered globals
            # in the source's internal representation.
            lam_internal = jnp.asarray(lam).reshape((-1,))
        return jnp.concatenate([u_internal, lam_internal.astype(u_internal.dtype)])

    @property
    def layout(self):
        """The device-placement [Layout][diff_soup.Layout] declared at lowering.

        A lowered problem records its ``Layout`` as the single source of
        truth for device placement; the engines read it through this view to
        place every linear analysis of a solve (the corrector's state block
        and the sensitivity pattern alike). ``None`` when the authored source
        declares no layout (a single local device).
        """
        return getattr(self.source, "layout", None)

    def layout_entries(self) -> tuple:
        """Static flat-state layout: ``(field_entries, global_entries)``.

        Forwards to the authored problem behind this view; a problem authored
        directly against the solver problem records no field layout and
        returns empty tuples.
        """
        getter = getattr(self.source, "layout_entries", None)
        if callable(getter):
            return getter()
        return (
            tuple(getattr(self.source, "_layout_entries", ()) or ()),
            tuple(getattr(self.source, "_global_layout_entries", ()) or ()),
        )

    def bind_step_root(
        self,
        backend,
        analysis,
        primal_fn,
        t,
        cj,
        q,
        theta,
        control,
        guess,
        primal_args,
        *,
        reverse_analysis=None,
        certificate_extractor=None,
    ):
        """Bind an accepted step root to its implicit-function derivative.

        ``primal_fn(t, cj, q, theta, control, guess, primal_args) -> (y_star,
        ok, aux)`` is the engine's converged corrector run; the returned root
        carries the exact root derivative (the sensitivity matrix rebuilt from
        this problem's plan at the root) through
        [implicit_step_root][diff_soup.problem.implicit.root_ift.implicit_step_root],
        and the bookkeeping and ``primal_args`` stay gradient-dead.
        ``analysis`` is the engine's analysis of ``full_topology`` for the
        forward sensitivity solves: a contract
        [Analysis][diff_soup.linear.Analysis] or the state its backend's
        ``analyze`` returns (``None`` on the dense path).
        ``reverse_analysis`` is the analysis the transposed solves of a
        reverse sweep factor ``A^T`` on (its precision is the reverse
        sensitivity policy); ``None`` uses ``analysis``. Traced solver state
        must ride ``primal_args``, never the ``primal_fn`` closure.

        ``certificate_extractor`` is an optional static
        ``(aux, t, cj, theta, control, primal_args) -> IterationMatrixCertificate``
        that the derivative rule calls on the ``aux`` the primal already
        returns, so the engine sources its already-assembled iteration-matrix
        values without a second corrector run. When the offering's attestations
        hold, the sensitivity solve consumes those values instead of rebuilding
        them from the plan; the offering is gradient-dead and never changes the
        primal or the derivative value — a failed attestation always falls back
        to the rebuild.
        """
        sens_backend = None if analysis is None else backend
        return implicit_step_root(
            self.plan,
            sens_backend,
            analysis,
            primal_fn,
            reverse_analysis,
            t,
            cj,
            q,
            theta,
            control,
            guess,
            primal_args,
            certificate_extractor,
        )

    def consistent_initial_state(
        self,
        t0,
        y0,
        theta,
        control=None,
        *,
        ydot0=None,
        tol,
        max_iters=30,
        backend="auto",
        route="auto",
        layout=None,
        precision=None,
        reverse_precision=None,
    ):
        """Solve the consistent initialization on the solver problem.

        Differential entries of ``y0`` are pinned; algebraic entries and the
        optional slope seed are guesses. ``route`` selects the solve by
        mathematical capability: ``"auto"`` (the default) runs the reduced
        algebraic-block elimination exactly when the problem has a diagonal
        invertible mass and the full joint solve otherwise. ``backend``
        selects the linear route for the factored system (``"auto"`` is
        dense for tiny systems, sparse above the small-size bar). ``layout``
        and ``precision`` are the device-placement and
        dtype policies the factored system is analyzed with — pass the
        policies the rest of the solve runs under so this projection
        factorizes where and at the precision the caller asked for; ``None``
        leaves the defaults. ``reverse_precision`` is the dtype policy of the
        transposed solve a reverse sweep runs at the root (``None`` uses
        ``precision``). One static solve plan per resolved
        ``(tol, max_iters, linear route, solve route, layout, precision,
        reverse_precision)`` tuple is cached on the view so repeat solves reuse one
        compiled program. Both routes are resolved BEFORE keying, so every
        spelling that lands on the same solve shares one plan — and one
        symbolic analysis — no matter which call site asked first.

        Args:
            t0: Initial time.
            y0: Initial state; differential rows are pinned and algebraic rows
                seed the projection.
            theta: Problem parameters. ``None`` uses [default_theta][].
            control: Control values read by the residual, or ``None``.
            ydot0: Optional initial slope guess.
            tol: Absolute residual 2-norm gate for convergence.
            max_iters: Newton iteration budget.
            backend: Linear-backend selection for the initialization system.
            route: ``"auto"``, ``"joint"``, or ``"reduced"`` solve route.
            layout: Optional device-placement policy for the linear analysis.
            precision: Optional dtype policy for the linear analysis.
            reverse_precision: Optional dtype policy for the transposed
                solve of a reverse sweep.
        """
        theta = self.default_theta() if theta is None else theta
        route_name, _ = self.initialization_backend(backend)
        if route == "auto":
            ic_route = "reduced" if reduced_route_eligible(self.problem) else "joint"
        else:
            ic_route = route
        key = (
            float(tol),
            int(max_iters),
            route_name,
            ic_route,
            layout,
            precision,
            reverse_precision,
        )
        cache = self.__dict__.setdefault("_init_plans", {})
        init_plan = cache.get(key)
        if init_plan is None:
            init_plan = ConsistentInitPlan(
                self.problem,
                self.plan,
                tol=tol,
                max_iters=max_iters,
                backend=backend,
                route=ic_route,
                layout=layout,
                precision=precision,
                reverse_precision=reverse_precision,
            )
            cache[key] = init_plan
        return consistent_initial_state(
            self.problem,
            t0,
            y0,
            theta,
            control,
            ydot0=ydot0,
            init_plan=init_plan,
        )

    def initialization_backend(self, backend="auto"):
        """Resolve ``backend`` to the linear route the initialization solve uses.

        Returns ``(name, instance)`` exactly as
        [consistent_initial_state][diff_soup.problem.implicit.solver_problem.SolverView.consistent_initial_state]
        resolves its ``backend`` argument for this problem: ``"auto"`` stays
        dense for a dense-mode linearization plan and otherwise picks the
        platform's route from the system size; a concrete name or a
        [LinearSolver][diff_soup.linear.LinearSolver] instance is honored. The
        instance is ``None`` on the dense route. Callers that batch
        initialization solves (for example projected output sampling) read the
        resolved backend's capabilities from the returned instance.
        """
        return resolve_backend_route(backend, plan_mode=self.plan.mode, n_dofs=self.n)

    def startup_slope(
        self,
        t,
        y,
        theta,
        control=None,
        *,
        ydot0=None,
        tol,
        backend="auto",
        layout=None,
        precision=None,
        reverse_precision=None,
    ):
        """Return the flow slope at ``(t, y)``.

        The capability-routed initialization plan derives a mass-form slope
        from its right-hand side and mass operator, and derives a fully
        implicit slope from the joint residual root. Failed roots are
        NaN-poisoned before an engine can consume them. ``layout``,
        ``precision`` and ``reverse_precision`` reach the plan's linear
        analyses exactly as they do on
        [consistent_initial_state][diff_soup.problem.implicit.solver_problem.SolverView.consistent_initial_state].
        """
        result = self.consistent_initial_state(
            t,
            y,
            theta,
            control,
            ydot0=ydot0,
            tol=tol,
            backend=backend,
            layout=layout,
            precision=precision,
            reverse_precision=reverse_precision,
        )
        ydot = jnp.asarray(result.ydot)
        return jnp.where(
            jnp.asarray(result.converged),
            ydot,
            jnp.full_like(ydot, jnp.nan),
        )

    def _split_engine_blocks(self, combined):
        """Split full-system ``A(cj)`` (dense or union values) into engine blocks."""
        n_u, n_lam = self.n_u, self.n_lam
        if self.plan.mode == "dense":
            a = combined
            return a[:n_u, :n_u], a[:n_u, n_u:], a[n_u:, :n_u], a[n_u:, n_u:]
        values = jnp.asarray(combined)
        a_state = values[self._state_take]
        if n_lam == 0:
            return (
                a_state,
                jnp.zeros((n_u, 0), dtype=values.dtype),
                jnp.zeros((0, n_u), dtype=values.dtype),
                jnp.zeros((0, 0), dtype=values.dtype),
            )
        b = jnp.zeros((n_u, n_lam), values.dtype)
        b = b.at[self._b_rows, self._b_cols].add(values[self._b_take])
        c = jnp.zeros((n_lam, n_u), values.dtype)
        c = c.at[self._c_rows, self._c_cols].add(values[self._c_take])
        d = jnp.zeros((n_lam, n_lam), values.dtype)
        d = d.at[self._d_rows, self._d_cols].add(values[self._d_take])
        return a_state, b, c, d

    @staticmethod
    def _round_values(values, values_dtype, dtype):
        """Round values through a reduced precision and back to the state dtype."""
        if values_dtype is None:
            return values
        return jnp.asarray(values).astype(values_dtype).astype(dtype)

    def iteration_matrix_values(
        self,
        t,
        y,
        ydot,
        theta,
        cj,
        values_dtype=None,
        control=None,
    ):
        """Return the complete ``A(cj) = F_y + cj F_ydot``.

        Sparse plans return values in the row-major COO ordering of
        [full_topology][]. Dense plans return the full ``(n, n)`` matrix.
        ``cj`` is applied exactly once by the shared linearization plan.
        """
        dtype = jnp.asarray(y).dtype
        if self.plan.mode == "dense":
            combined = self.plan.combined_dense(t, y, ydot, theta, control, cj)
        else:
            combined = self.plan.combined_values(t, y, ydot, theta, control, cj)
        return self._round_values(combined, values_dtype, dtype)

    def state_jacobian_blocks(self, t, y, ydot, theta, cj, values_dtype=None, control=None):
        """The exact ``A(cj)`` split into ``(A_state, B, C, D)`` engine blocks.

        ``A_state`` is dense ``(n_u, n_u)`` or values over ``state_topology``;
        ``B``, ``C``, ``D`` are the dense bordered-global couplings in the Schur
        convention. ``values_dtype`` rounds the block values through the given
        precision and casts back to the state dtype (the reduced-precision
        iteration-matrix path); ``None`` keeps the exact values. ``control``
        is the control-value pytree the residual reads (``None`` for a
        problem without a control channel).
        """
        combined = self.iteration_matrix_values(
            t,
            y,
            ydot,
            theta,
            cj,
            values_dtype=values_dtype,
            control=control,
        )
        return self._split_engine_blocks(combined)

    def state_partials(self, t, y, ydot, theta, values_dtype=None, control=None):
        """Split partial values ``(F_y, F_ydot)`` for later recomposition.

        The pair feeds [recomposed_state_blocks][]: a multistep engine
        saves it at a setup, then recomposes with a new leading coefficient
        (optionally re-evaluating ``F_ydot`` alone through
        [fydot_partial][]) without re-evaluating ``F_y``. ``control`` is
        the control-value pytree the residual reads.
        """
        dtype = jnp.asarray(y).dtype
        fy = self.plan.fy_values(t, y, ydot, theta, control)
        fydot = self.plan.fydot_values(t, y, ydot, theta, control)
        return (
            self._round_values(fy, values_dtype, dtype),
            self._round_values(fydot, values_dtype, dtype),
        )

    def fydot_partial(self, t, y, ydot, theta, control=None):
        """Fresh ``F_ydot`` values alone — the cheap re-evaluation for a rescale."""
        return self.plan.fydot_values(t, y, ydot, theta, control)

    def zero_state_partials(self, dtype):
        """Zero-valued split partials (a reuse-carry seed; never consumed)."""
        if self.plan.mode == "dense":
            zero = jnp.zeros((self.n, self.n), dtype=dtype)
            return zero, zero
        return (
            jnp.zeros((self.plan.fy_topology.rows.shape[0],), dtype=dtype),
            jnp.zeros((self.plan.fydot_topology.rows.shape[0],), dtype=dtype),
        )

    def recomposed_state_blocks(self, fy, fydot, cj, values_dtype=None):
        """Engine blocks of ``A(cj)`` recomposed from split partials — no re-evaluation.

        ``values_dtype`` rounds the combined ``A(cj)`` values through the given
        precision and casts back before splitting, matching
        [state_jacobian_blocks][] when the split partials are passed at full
        precision; ``None`` keeps the recomposed values exactly.
        """
        values = self.plan.recompose(fy, fydot, cj)
        if values_dtype is not None:
            values = self._round_values(values, values_dtype, jnp.asarray(fy).dtype)
        return self._split_engine_blocks(values)

    def iteration_matrix_offering(self, fy, fydot, cj, at_root):
        """Build the step-root iteration-matrix offering from saved partials.

        Recomposes ``A(cj)`` over this problem's union support from the
        corrector's saved split partials and wraps it in the gradient-dead
        offering the step-root boundary consumes when ``at_root`` holds (see
        [iteration_matrix_offering][diff_soup.problem.implicit.root_ift.iteration_matrix_offering]).
        Engines reach the offering builder through this view so they never
        import the boundary module directly.
        """
        return iteration_matrix_offering(self.plan, fy, fydot, cj, at_root)


# The solver-view cache, keyed on the content of the problem a view presents.
# A stable view identity is what lets the solver's compile cache reuse one
# program across repeat solves of one model. Past the bound the least recently
# used view is rebuilt on its next use, and compares equal to the one it
# replaces.
_SOLVER_VIEW_CACHE_CAPACITY = 32
_SOLVER_VIEW_CACHE = BoundedContentCache(capacity=_SOLVER_VIEW_CACHE_CAPACITY)

# The model key of each recently presented problem object, least recently used
# first, so repeated lookups with one object digest its content or values once.
# An entry holds its object, so the id is not reused while the entry lives.
_MODEL_KEY_FRONT: OrderedDict[int, tuple[Any, tuple[Hashable, Hashable, Any]]] = OrderedDict()
_MODEL_KEY_FRONT_LOCK = Lock()


def _value_digest(obj) -> str:
    """Digest of every array and scalar leaf of ``obj`` (dtype, shape, bytes).

    Raises:
        TypeError: If a leaf is traced, so it has no concrete value.
    """
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(obj):
        if isinstance(leaf, jax.core.Tracer):
            raise TypeError("a traced leaf has no concrete value to key on")
        if isinstance(leaf, (np.ndarray, np.generic, jax.Array)):
            arr = np.asarray(leaf)
            digest.update(f"array:{arr.dtype.str}:{arr.shape}:".encode())
            digest.update(arr.tobytes())
        elif leaf is None or isinstance(leaf, (bool, int, float, complex, str)):
            digest.update(f"{type(leaf).__name__}:{leaf!r};".encode())
        else:
            # Callables and other static leaves are covered by the structure
            # key and checked against the cached source's static partition.
            digest.update(b"static;")
    return digest.hexdigest()


def _content_digest(obj) -> str:
    """Digest of a value-sensitive fingerprint of ``obj``.

    Arrays enter by their bytes and callables by their code, defaults,
    closure cells and the module globals they read, so two problems differing
    in a captured value, or in the value a global their functions read holds
    when each is digested, differ here.

    Raises:
        TypeError: If ``obj`` holds a value the fingerprint cannot describe
            (including a traced value).
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        text = "|".join(
            [f"{cls.__module__}.{cls.__qualname__}"]
            + [
                f"{field.name}={_graph_callable_config_id(getattr(obj, field.name), force_static=True)}"
                for field in sorted(dataclasses.fields(obj), key=lambda item: item.name)
            ]
        )
    else:
        text = _graph_callable_config_id(obj, force_static=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _static_equal(a, b) -> bool:
    """Whether two objects have equal static (non-inexact-array) partitions."""
    if a is b:
        return True
    _, static_a = eqx.partition(a, eqx.is_inexact_array)
    _, static_b = eqx.partition(b, eqx.is_inexact_array)
    return bool(eqx.tree_equal(static_a, static_b))


def _model_key(problem) -> tuple[Hashable, Hashable, Any]:
    """The ``(cache key, structure-signature source key, handle)`` of ``problem``.

    A source declaring a structure key is keyed on that key and a digest of its
    values (the view serves the source's values: its default parameters and
    its initial-state conversion). Any other problem is keyed on a fingerprint
    of its content, or, when its content cannot be fingerprinted, on its
    identity; the handle then holds the object, so its id is not reused while
    the entry lives.
    """
    try:
        structure_key = declared_structure_key(problem)
        if structure_key is not None:
            source_key = ("declared", structure_key)
            return source_key + (_value_digest(problem),), source_key, None
        source_key = ("content", _content_digest(problem))
        return source_key, source_key, None
    except (TypeError, ValueError, RuntimeError):
        source_key = ("identity", id(problem))
        return source_key, source_key, problem


def _front_model_key(problem) -> tuple[Hashable, Hashable, Any]:
    """The [_model_key][diff_soup.problem.implicit.solver_problem._model_key] of ``problem``.

    Served from the identity front when the entry stored under ``id(problem)``
    holds this very object; otherwise computed and stored, evicting the least
    recently used entry past the solver-view cache's capacity.
    """
    slot = id(problem)
    with _MODEL_KEY_FRONT_LOCK:
        entry = _MODEL_KEY_FRONT.get(slot)
        if entry is not None and entry[0] is problem:
            _MODEL_KEY_FRONT.move_to_end(slot)
            return entry[1]
    model_key = _model_key(problem)
    with _MODEL_KEY_FRONT_LOCK:
        _MODEL_KEY_FRONT[slot] = (problem, model_key)
        _MODEL_KEY_FRONT.move_to_end(slot)
        while len(_MODEL_KEY_FRONT) > _SOLVER_VIEW_CACHE_CAPACITY:
            _MODEL_KEY_FRONT.popitem(last=False)
    return model_key


def _build_solver_view(problem, source_key: Hashable) -> SolverView:
    """Build the engine-facing view of ``problem``."""
    # The view's structural constants (gather maps, patterns, colorings) must
    # be concrete host arrays even when the first solve happens inside a
    # user's jit trace.
    with jax.ensure_compile_time_eval():
        if isinstance(problem, (MassProblem, ImplicitProblem)):
            return SolverView(problem, source=None, source_key=source_key)
        hook = getattr(problem, "solver_problem", None)
        if callable(hook):
            model = hook()
            if not isinstance(model, (MassProblem, ImplicitProblem)):
                raise TypeError(
                    f"solver_problem() returned "
                    f"{type(model).__name__}; it must return a "
                    "MassProblem or an ImplicitProblem."
                )
        elif callable(getattr(problem, "canonical_problem", None)):
            raise TypeError(
                f"{type(problem).__name__} provides canonical_problem(); "
                "the hook the solver reads is named solver_problem(). "
                "Rename the method to solver_problem()."
            )
        else:
            raise TypeError(
                f"{type(problem).__name__} supplies no solver problem. "
                "Author a MassProblem or ImplicitProblem, or provide "
                "solver_problem() returning one of those two types."
            )
        direct_source = callable(getattr(problem, "state_size", None)) and callable(
            getattr(problem, "constraint_dim", None)
        )
        return SolverView(
            model,
            source=problem,
            direct_source=direct_source,
            source_key=source_key,
        )


def get_or_build_solver_view(problem) -> SolverView:
    """Return the cached engine-facing view of ``problem``, building it once.

    Accepts a solver problem ([MassProblem][] / [ImplicitProblem][]),
    a problem with a callable ``solver_problem()`` hook, or an already-built
    view (returned unchanged). Other objects are rejected before any JAX
    callback is traced. A problem equal in content to one already presented
    (a rebuilt problem, or a lowered problem with the same structure and
    values) is served the same view. The key of a recently presented object
    is kept with the object, so repeated calls with one object compute its
    content or value digest once.

    Raises:
        TypeError: If the object supplies no solver problem, or if
            ``solver_problem()`` returns anything other than a
            ``MassProblem`` or ``ImplicitProblem``.
    """
    if isinstance(problem, SolverView):
        return problem
    cache_key, source_key, handle = _front_model_key(problem)
    view = _SOLVER_VIEW_CACHE.get_or_build(
        cache_key, handle, lambda: _build_solver_view(problem, source_key)
    )
    if source_key[0] != "declared" or view.source is problem or _static_equal(view.source, problem):
        return view
    # A source whose static fields differ from the cached source's under one
    # structure key presents its own solver problem.
    source_key = ("identity", id(problem))
    return _SOLVER_VIEW_CACHE.get_or_build(
        source_key, problem, lambda: _build_solver_view(problem, source_key)
    )
