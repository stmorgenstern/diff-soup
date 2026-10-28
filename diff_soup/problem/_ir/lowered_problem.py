"""Concrete Recipe IR produced by [EquationSet][diff_soup.problem._ir.EquationSet] lowering.

The solver-facing problem object that [diff_soup.problem._ir.lower][] returns.  Carries field
groups for sizes, structure, IR nodes, kernels, layout bookkeeping, scaling,
and the ``layout`` / ``precision`` pillars.

Design invariants:

* **No ``object.__setattr__`` on fields, ever.** ``lower()`` returns a
  [LoweredProblem][diff_soup.problem._ir.lowered_problem.LoweredProblem] that
  holds its own declared values and never changes afterwards; id-stability
  comes from the bounded caches ([diff_soup.problem._registry][]) returning
  the same object for an equal structure and equal values.  Mutating a field
  of a live frozen object would silently invalidate ``filter_jit`` caches.
* **Static fields hold only hashable host data.** The IR-node tuples
  (``_equations``, ``_constraints``, ...) are runtime fields so ``eqx.partition``
  surfaces any traced ``eqx.Module`` closure leaves for AD.  The kernels
  (``_physics_residual_fn``, ...) are static and capture nothing traced — they
  take the runtime IR-node tuples as arguments.

The BC-row identity-override hook (``bc_dof_indices`` / ``bc_residual``) is
load-bearing for stiff multi-DOF DAEs: without it the consistent-init Newton
sees BC rows scaled by ``-hinvGak`` and the condition-number / row-scaling shift
destabilises the cuDSS solver on Allen-Cahn-mechanics-class problems.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._layout import Layout
from diff_soup._precision import Precision
from diff_soup.linear import SparseTopology

from .observation import Observable
from .scales import Scales, identity_scales


def _expand_family_form(cell: dict, *, count: int, member_state: int) -> dict:
    """Expand a member-local derivative form across a family population.

    The captured pattern addresses one member's field block; member ``k``'s
    rows and columns shift by ``k * member_state`` within the family field's
    flat block (a form-recording family has exactly one member field, so the
    block is contiguous and ``member_state`` equals the field size).  A
    nonlinear form carries no pattern — it passes through with its referenced
    fields.
    """
    out = dict(cell)
    if not cell.get("nonlinear"):
        expanded = []
        for in_field, rows, cols in cell.get("terms", ()):
            rows = np.asarray(rows, dtype=np.int64)
            cols = np.asarray(cols, dtype=np.int64)
            shifts = np.arange(count, dtype=np.int64) * int(member_state)
            expanded.append(
                (
                    in_field,
                    (rows[None, :] + shifts[:, None]).reshape(-1),
                    (cols[None, :] + shifts[:, None]).reshape(-1),
                )
            )
        out["terms"] = tuple(expanded)
    return out


def _decided_form(eq, label: str):
    """The captured form of ``eq``, ``None`` for a plain equation, or raise.

    The one guard for component and family member equations alike: a form
    cell the build-time capture pass left undecided raises with the stored
    reason.
    """
    cell = getattr(eq.rhs_fn, "_derivative_form", None)
    if cell is None or cell.get("captured_plain"):
        return None
    if cell.get("captured"):
        return cell
    reason = cell.get("capture_error", "the build never evaluated it")
    raise ValueError(
        f"{label} {getattr(eq, 'name', '?')!r} could not be evaluated at a "
        f"generic positive state during the build ({reason}). The build "
        "evaluates every equation there once to record any time-derivative "
        "structure (q.dt) before the solve; make the equation evaluable there."
    )


class LoweredProblem(eqx.Module):
    """Solve-ready problem compiled from a recipe or an equation set.

    `spec.build()` and [`lower`][diff_soup.problem.lower] return one, and
    [`pdesolve`][diff_soup.pdesolve.pdesolve] and the inference verbs take it.
    It is a pytree whose floating-point leaves are the declared values.

    Attributes:
        layout: [`Layout`][diff_soup.Layout] the problem was built with.
        precision: [`Precision`][diff_soup.Precision] the problem was built with.
        quad_fields: Names of the fields that `q` reads at quadrature points on a finite-element domain.
    """

    _state_size: int = eqx.field(static=True)
    _constraint_dim: int = eqx.field(static=True)
    _mass_diagonal: jax.Array
    _topo: SparseTopology  # all-static content-bytes record (no array leaves)
    _equations: tuple
    _dirichlet_value_fns: tuple
    _constraints: tuple
    _observations: tuple
    _controls: tuple
    _couplings: tuple
    _physics_residual_fn: Callable = eqx.field(static=True)
    _physics_jac_blocks_fn: Callable = eqx.field(static=True)
    _initial_lambda: jax.Array
    # Per-Dirichlet-block flat DOF indices (hashable int tuples) + the matching
    # value_fn(t) callables. Solver-problem lowering reads ``bc_dof_indices()`` at
    # construction time and replaces those rows with their algebraic
    # Dirichlet residuals — load-bearing for stiff multi-DOF DAEs.
    _bc_dof_indices_static: tuple = eqx.field(static=True, default=())
    _observe_fn: Optional[Callable] = eqx.field(static=True, default=None)
    _controls_fn: Optional[Callable] = eqx.field(static=True, default=None)
    # Boundary kernel callables — runtime tuples so traced eqx.Module flux/value
    # closures surface through eqx.partition.
    _flux_bc_callables: tuple = ()
    _robin_bc_callables: tuple = ()
    _traction_bc_callables: tuple = ()
    # Scales bookkeeping for dimensional <-> non-dim translation at the boundary methods.
    # Static — never read at residual time, only at user-facing boundary calls.
    _scales: Scales = eqx.field(static=True, default_factory=identity_scales)
    # Per-DOF (component, variable) layout entries used by the boundary rewrap to
    # broadcast scalar refs across each variable's flat slot.
    _layout_entries: tuple = eqx.field(static=True, default=())
    # Per-global (name, offset, size) layout entries for initial_lambda rewrap.
    _global_layout_entries: tuple = eqx.field(static=True, default=())
    # dimensional initial lambda before scaling — kept so initial_lambda can rescale to
    # internal each call. Identity Scales => bit-identical to _initial_lambda.
    _initial_lambda_si: jax.Array = eqx.field(default=None)
    # Flat indices into u for algebraic DOFs (mass diagonal == 0). Resolved
    # host-side so the consistent-IC "pure ODE?" structural guard stays
    # JIT-traceable.
    _alg_indices: tuple = eqx.field(static=True, default=())
    # Family populations (() on the slim single-domain path). NON-static so each
    # family's ``member_params`` leaves reach AD; the spine is folded structurally.
    # Consumed by ``assemble_state`` / ``family_view`` / the family residual blocks.
    _families: tuple = ()
    # Single source of truth for Layout / Precision; included in the cache key;
    # threaded to the analysis provider.  Default values reproduce prior jaxprs
    # bit-identically.
    layout: Layout = eqx.field(static=True, default_factory=Layout)
    precision: Precision = eqx.field(static=True, default_factory=Precision)
    # Names of fields opted into the quad-native Field path.  Exposed for
    # introspection (e.g. test_field_quad_flag).  The equation adapters carry
    # the same set internally; this attribute makes it visible on the built
    # problem without ``object.__setattr__``.
    quad_fields: "frozenset[str]" = eqx.field(static=True, default_factory=frozenset)
    # Value-independent IR/Layout/Precision/Scales content key. The solver
    # view uses it to prove structural equivalence across optimization/OED
    # problem instances without comparing numeric leaves or closure identity.
    _structure_key: Any = eqx.field(static=True, default=None)

    def structure_key(self):
        """Hashable compile-relevant identity used by the solver view."""
        return self._structure_key

    # ------------------------------------------------------------------
    # Solver-facing KKT surface — kernels capture nothing traced;
    # the runtime IR-node tuples are passed as arguments so AD reaches them.
    # ------------------------------------------------------------------

    def physics_residual(
        self, u: jax.Array, lam: jax.Array, t: jax.Array, control: Any = None
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the physics residual ``(f, g)`` block-wise.

        Args:
            u: Internal-units state vector, shape ``(state_size,)``.
            lam: Internal-units bordered-global values, shape ``(constraint_dim,)``.
            t: Internal-units time scalar.
            control: Optional runtime control payload — a
                ``{control_name: dimensional value}`` mapping, or a
                [ControlState][diff_soup.timestepping.control_state.ControlState]
                carrying that mapping — overriding declared controls for this
                evaluation; a control absent from the mapping evaluates its
                authored ``fn`` at dimensional time. A problem that declares an
                operational control table requires an operating
                ``ControlState`` here, because its active row is chosen by the
                state's mode index.

        Returns:
            ``(f, g)`` — local-state residual ``(state_size,)`` and
            bordered-global residual ``(constraint_dim,)``.
        """
        return self._physics_residual_fn(
            self._equations,
            self._dirichlet_value_fns,
            self._flux_bc_callables,
            self._robin_bc_callables,
            self._traction_bc_callables,
            self._constraints,
            self._couplings,
            u,
            lam,
            t,
            self._families,
            self._controls,
            control,
        )

    def operational_table(self):
        """The declared operational control table node, or ``None``.

        Returns the
        [OperationalControlConstraint][diff_soup.problem._ir.constraint.OperationalControlConstraint]
        authored with ``spec.control.operational(...)`` when the problem
        declares one.
        """
        from diff_soup.problem._ir.constraint import OperationalControlConstraint

        for c in self._constraints:
            if isinstance(c, OperationalControlConstraint):
                return c
        return None

    def operational_modes(self) -> tuple:
        """The static ordered mode-name tuple of the operational control table.

        Use it to map mode names to the integer indices carried in
        [ControlState][diff_soup.timestepping.control_state.ControlState] — e.g.
        ``problem.operational_modes().index("voltage")`` inside an event
        affect.

        Raises:
            ConfigurationError: when the problem declares no operational
                control table.
        """
        table = self.operational_table()
        if table is None:
            from diff_soup._errors import ConfigurationError

            raise ConfigurationError(
                "this problem declares no operational control table; author "
                "one with spec.control.operational(...)."
            )
        return tuple(table.modes)

    def derivative_forms(self) -> tuple:
        """Static structure of the residual-form derivative equations.

        One dict per equation authored with a derivative expression, with
        keys ``out_field``, ``terms`` (each ``(in_field, rows, cols)`` — the
        exact local coefficient pattern; empty for a nonlinear residual),
        ``nonlinear``, and ``ydot_fields`` (nonlinear only). Empty when no
        equation uses the residual form. The structure is recorded by the
        build-time capture pass, which evaluates every equation once. An
        equation that pass could not evaluate raises with the stored reason:
        dropping its form would lower it with the plain mass diagonal, which
        loses scaled and cross-field derivative terms.

        Family member equations contribute too: a member form is captured in
        MEMBER-LOCAL coordinates and expanded here across the population
        (member ``k``'s rows/cols shift by ``k * member_state_size``), so the
        surfaced pattern covers the family field's whole flat block.

        Raises:
            ValueError: when a form-referenced field name is carried by more
                than one owner (forms resolve fields by bare name), or when an
                equation or a family member equation could not be evaluated
                for capture.
        """
        forms = []
        for eq in self._equations:
            cell = _decided_form(eq, "equation")
            if cell is not None:
                forms.append(dict(cell))
        for fam in self._families:
            member_state = int(fam.member_state_size)
            count = int(fam.count)
            for eq in fam.member_equations:
                cell = _decided_form(eq, f"family {fam.name!r} member equation")
                if cell is not None:
                    forms.append(_expand_family_form(cell, count=count, member_state=member_state))
        if forms:
            self._check_form_field_names_unambiguous(forms)
        return tuple(forms)

    def _check_form_field_names_unambiguous(self, forms) -> None:
        """Reject derivative forms whose field names are carried by two owners.

        The derivative-form lowering resolves fields by bare name against the
        flat layout; a name owned by two blocks (a component field shadowing a
        family member field, or the same member name in two families) would
        silently bind to one of them.
        """
        referenced: set = set()
        for form in forms:
            referenced.add(form["out_field"])
            referenced.update(in_field for in_field, _r, _c in form.get("terms", ()))
            referenced.update(form.get("ydot_fields", ()))
        owners_by_name: dict = {}
        for (owner, var_name), _off, _size, _shape in self._layout_entries:
            owners_by_name.setdefault(var_name, []).append(owner)
        for name in sorted(referenced):
            owners = owners_by_name.get(name, [])
            if len(owners) > 1:
                raise ValueError(
                    f"field {name!r} is referenced by a derivative-form "
                    f"equation but is carried by multiple owners "
                    f"{sorted(owners)!r}; derivative forms resolve fields by "
                    "bare name — give the fields distinct names."
                )

    def physics_jacobian_blocks(
        self, u: jax.Array, lam: jax.Array, t: jax.Array, control: Any = None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, SparseTopology]:
        """Evaluate the KKT Jacobian as ``(Avals, B, C, D, topo)``.

        Args:
            u: Internal-units state vector, shape ``(state_size,)``.
            lam: Internal-units bordered-global values, shape ``(constraint_dim,)``.
            t: Internal-units time scalar.
            control: Optional ``{control_name: dimensional value}`` mapping (or a
                ``ControlState``) overriding declared controls, exactly as
                [physics_residual][diff_soup.problem._ir.lowered_problem.LoweredProblem.physics_residual]
                takes it — the Jacobian is evaluated at the same control
                values as the residual it linearizes.

        Returns:
            ``(Avals, B, C, D, topo)`` — see [physics_jacobian_blocks]
        [diff_soup.problem._ir.lowered_problem.LoweredProblem.physics_jacobian_blocks]
            shapes
            (``Avals`` = nonzero ``df/du`` values; ``B`` = ``df/dlam``;
            ``C`` = ``dg/du``; ``D`` = ``dg/dlam``; ``topo`` = SparseTopology).
        """
        return self._physics_jac_blocks_fn(
            self._equations,
            self._dirichlet_value_fns,
            self._flux_bc_callables,
            self._robin_bc_callables,
            self._traction_bc_callables,
            self._constraints,
            self._couplings,
            u,
            lam,
            t,
            self._families,
            self._controls,
            control,
        )

    def mass_diagonal(self) -> jax.Array:
        """Diagonal of the mass matrix (zero rows = algebraic / BC-pinned)."""
        return self._mass_diagonal

    def mass_mode(self) -> str:
        """Recipe field masses are structural constants after lowering."""
        return "constant"

    def mass_diagonal_state(self, u, lam, t):
        """No runtime mass variation is present on lowered Recipe IR."""
        del u, lam, t
        return None

    def dmass_du_diagonal(self, u):
        """The structural Recipe mass has zero state derivative."""
        del u
        return None

    def state_size(self) -> int:
        """Return the length of the state vector, which excludes the global unknowns."""
        return self._state_size

    def constraint_dim(self) -> int:
        """Full bordered-global dimension (``0`` with no global unknowns).

        The method name is retained for solver-protocol compatibility; the
        counted ``g`` rows may be differential or algebraic.
        """
        return self._constraint_dim

    def family_view(self, name: str):
        """Return a frozen, pure-jnp accessor over a Family's member slab.

        ``FamilyView`` exposes ``.extract(u) -> (count, member_state_size)``,
        ``.field_slice(f)``, and ``.member_weights`` so a population observable
        can gather the slab without reaching into lowering internals.
        Pure-jnp / differentiable.

        Args:
            name: The family name (``Family.name``).

        Returns:
            A [FamilyView][diff_soup._lowering.family_view.FamilyView] for
            ``name``.

        Raises:
            KeyError: when no family named ``name`` exists on this problem.
        """
        from diff_soup._lowering.family_view import _build_family_view

        for fam in self._families:
            if fam.name == name:
                if not fam.member_variables:
                    base = int(self._state_size)
                else:
                    base = next(
                        off
                        for (owner, var), off, _size, _shape in self._layout_entries
                        if owner == name and var == fam.member_variables[0].name
                    )
                return _build_family_view(fam, base)
        raise KeyError(
            f"no family named {name!r}; families are {sorted(f.name for f in self._families)!r}"
        )

    # ------------------------------------------------------------------
    # Non-dim boundary rewrap (Scales) — invisible to L3.
    # ------------------------------------------------------------------

    def _scale_arrays(self) -> tuple[jax.Array, jax.Array]:
        """Per-DOF (scale, offset) arrays from ``_layout_entries`` (identity default)."""
        n = int(self._state_size)
        scale = np.ones(n, dtype=np.float64)
        offset = np.zeros(n, dtype=np.float64)
        for (comp_name, var_name), off, size, _shape in self._layout_entries:
            s = float(self._scales.state_ref.get((comp_name, var_name), 1.0))
            o = float(self._scales.state_offset.get((comp_name, var_name), 0.0))
            scale[off : off + size] = s
            offset[off : off + size] = o
        return jnp.asarray(scale), jnp.asarray(offset)

    def _lambda_scale_arrays(self) -> tuple[jax.Array, jax.Array]:
        """Per-global (scale, offset) arrays for lambda rewrap (globals keyed ``("", name)``)."""
        n = int(self._constraint_dim)
        scale = np.ones((n,), dtype=np.float64)
        offset = np.zeros((n,), dtype=np.float64)
        for global_name, off, size in self._global_layout_entries:
            s = float(self._scales.state_ref.get(("", global_name), 1.0))
            o = float(self._scales.state_offset.get(("", global_name), 0.0))
            scale[off : off + size] = s
            offset[off : off + size] = o
        return jnp.asarray(scale), jnp.asarray(offset)

    def initial_u(self, y0: jax.Array) -> jax.Array:
        """Convert a dimensional state to internal units, `(y0 - offset) / scale` per field.

        Args:
            y0: Full state vector in dimensional units.

        Returns:
            The state vector in internal units.
        """
        y0 = jnp.asarray(y0)
        scale, offset = self._scale_arrays()
        return (y0 - offset.astype(y0.dtype)) / scale.astype(y0.dtype)

    def _scale_lambda_to_internal(self, lam_si: jax.Array) -> jax.Array:
        scale, offset = self._lambda_scale_arrays()
        lam_si = jnp.asarray(lam_si)
        if lam_si.size == 0:
            return lam_si
        return (lam_si - offset.astype(lam_si.dtype)) / scale.astype(lam_si.dtype)

    def _scale_lambda_to_si(self, lam_internal: jax.Array) -> jax.Array:
        scale, offset = self._lambda_scale_arrays()
        lam_internal = jnp.asarray(lam_internal)
        if lam_internal.size == 0:
            return lam_internal
        return offset.astype(lam_internal.dtype) + scale.astype(lam_internal.dtype) * lam_internal

    def initial_lambda(self) -> jax.Array:
        """Internal-units initial lambda (dimensional ``GlobalVariable.init`` rescaled)."""
        if self._initial_lambda_si is None or self._initial_lambda_si.size == 0:
            return self._initial_lambda
        return self._scale_lambda_to_internal(self._initial_lambda_si)

    def global_layout(self) -> tuple[dict, ...]:
        """Describe where each global unknown sits in the global block.

        Returns:
            One dict per global with its `name`, `offset`, `size`, and `mass`, `"differential"` or `"algebraic"`.
        """
        roles = {c.closes.name: c.closes.mass for c in self._constraints}
        return tuple(
            {"name": name, "offset": int(off), "size": int(size), "mass": roles[name]}
            for name, off, size in self._global_layout_entries
        )

    def field_slice(self, field: str, *, component: Optional[str] = None) -> slice:
        """Return a declared field's slice in the flat state vector.

        Args:
            field: The declared field name.
            component: Optional component/domain name. It is required when
                several components declare the same field name.

        Returns:
            The contiguous state-vector slice occupied by the field.

        Raises:
            KeyError: If no matching field exists, or if ``field`` is
                ambiguous without ``component``.
        """
        matches = tuple(
            (owner, int(offset), int(size))
            for (owner, name), offset, size, _shape in self._layout_entries
            if name == field and (component is None or owner == component)
        )
        if not matches:
            qualifier = "" if component is None else f" on component {component!r}"
            available = sorted({name for (_owner, name), *_rest in self._layout_entries})
            raise KeyError(
                f"no field named {field!r}{qualifier}; available fields are {available!r}"
            )
        if len(matches) > 1:
            owners = sorted(owner for owner, _offset, _size in matches)
            raise KeyError(
                f"field {field!r} occurs on components {owners!r}; pass component=... "
                "to select one."
            )
        _owner, offset, size = matches[0]
        return slice(offset, offset + size)

    @property
    def observables(self) -> tuple[Observable, ...]:
        """One record per declared observation, with its `name`, `kind`, `shape`, `dtype`, `owner` and `reads`."""
        return tuple(Observable.of(obs) for obs in self._observations)

    def observe(
        self,
        u: jax.Array,
        t: jax.Array,
        lam_internal: Optional[jax.Array] = None,
        *,
        names: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """Evaluate the declared observations at an internal state and time, in dimensional units.

        Args:
            u: State vector in internal units.
            t: Internal time.
            lam_internal: Global unknowns in internal units, or `None`.
            names: Observation names to evaluate, or `None` for all, where an undeclared name raises `ConfigurationError`.

        Returns:
            Mapping from observation name to its value, empty when nothing is declared.
        """
        observations = self._observations
        if names is not None:
            if isinstance(names, str):
                names = (names,)
            by_name = {obs.name: obs for obs in self._observations}
            unknown = [n for n in names if n not in by_name]
            if unknown:
                from diff_soup._errors import ConfigurationError

                raise ConfigurationError(
                    f"observe(names=...) names {unknown[0]!r}, which is not a declared "
                    f"observation; names= takes declared observation names: "
                    f"{list(by_name)!r}"
                )
            observations = tuple(by_name[n] for n in dict.fromkeys(names))
        if self._observe_fn is None or not observations:
            return {}
        scale, offset = self._scale_arrays()
        u_si = offset.astype(u.dtype) + scale.astype(u.dtype) * u
        t_si = jnp.asarray(self._scales.time_ref) * jnp.asarray(t)
        lam_si = None if lam_internal is None else self._scale_lambda_to_si(lam_internal)
        return self._observe_fn(
            self._equations,
            self._constraints,
            observations,
            u_si,
            t_si,
            lam_si,
        )

    def controls(self, t: jax.Array) -> dict[str, Any]:
        """Evaluate every declared control at an internal time.

        Args:
            t: Internal time, multiplied by the time scale before each control function is called.

        Returns:
            Mapping from control name to its dimensional value, empty when no control is declared.
        """
        if self._controls_fn is None:
            return {}
        return self._controls_fn(self._controls, t)

    def bc_dof_indices(self) -> Optional[jax.Array]:
        """Concatenated Dirichlet DOF indices, or ``None`` (the identity-override hook).

        Solver-problem lowering materialises this as a hashable ``tuple[int, ...]``
        and replaces those rows with algebraic Dirichlet residuals.
        """
        if not self._bc_dof_indices_static:
            return None
        parts = [np.asarray(idx, dtype=np.int32) for idx in self._bc_dof_indices_static]
        # Return a jax.Array (the boundary annotation / beartype contract); the
        # Solver-problem lowering does ``np.asarray`` on it. Host-concrete int32 —
        # no trace dependence.
        return jnp.asarray(np.concatenate(parts))

    def bc_residual(self, u: jax.Array, t: jax.Array) -> Optional[jax.Array]:
        """Full-size BC residual ``u[bc] - target(t)`` at pinned DOFs (zero elsewhere)."""
        if not self._bc_dof_indices_static:
            return None
        from .bc import _BoundaryFn

        def boundary_callback(fn):
            while fn is not None:
                if isinstance(fn, _BoundaryFn):
                    return True
                fn = getattr(fn, "inner", None)
            return False

        if any(boundary_callback(fn) for fn in self._dirichlet_value_fns):
            residual, _ = self.physics_residual(
                u, jnp.zeros((self.constraint_dim(),), dtype=u.dtype), t
            )
            mask = jnp.zeros_like(u)
            for idx in self._bc_dof_indices_static:
                mask = mask.at[jnp.asarray(idx)].set(1)
            return residual * mask
        res = jnp.zeros_like(u)
        for idx, value_fn in zip(self._bc_dof_indices_static, self._dirichlet_value_fns):
            idx_dev = jnp.asarray(np.asarray(idx, dtype=np.int32))
            val = jnp.asarray(value_fn(t), dtype=u.dtype)
            val_b = jnp.broadcast_to(val, idx_dev.shape)
            res = res.at[idx_dev].set(u[idx_dev] - val_b)
        return res

    def assemble_state(self, values: dict[str, Any], *, default: float = 0.0) -> jax.Array:
        """Assemble the flat initial state from per-field values.

        Fields missing from `values` take `default`, not their declared `init`, and a
        callable value raises `TypeError`. `ProblemSpec.assemble_state` evaluates
        declared initial conditions and coordinate callables.

        Args:
            values: Mapping from field name to a scalar or an array of the field's shape, with family fields keyed `"<family>.<field>"`.
            default: Value of every field left out.

        Returns:
            The state vector, component fields in layout order, then family members.
        """
        # Family-keyed layout entries are handled by the family branch (their
        # ``comp`` field is the Family.name); the component loop skips them.
        family_names = {fam.name for fam in self._families}
        pieces = []
        for (_comp, var), _offset, size, shape in sorted(self._layout_entries, key=lambda e: e[1]):
            if _comp in family_names:
                continue  # materialised below via the family branch.
            v = values.get(var, default)
            if callable(v):
                raise TypeError(
                    f"LoweredProblem.assemble_state: field {var!r} received a "
                    "callable; a lowered problem has no node coordinates — pass a "
                    "concrete array, or use ProblemSpec.assemble_state for "
                    "coordinate closures"
                )
            arr = jnp.broadcast_to(jnp.asarray(v), shape) if jnp.ndim(v) == 0 else jnp.asarray(v)
            flat = jnp.reshape(arr, (-1,))
            if flat.size != size:
                raise ValueError(
                    f"field {var!r}: value materialised to size {flat.size}, "
                    f"but the field's flat size is {size} (shape {shape})"
                )
            pieces.append(flat)
        # Family populations follow all components in ``families`` order;
        # their ``(coords, m)`` closures are evaluable here
        # because a Family carries its own member ``domain`` + ``member_params``.
        if self._families:
            from diff_soup._lowering.family_assemble import assemble_family_pieces

            pieces.extend(assemble_family_pieces(self._families, values, default))
        return jnp.concatenate(pieces) if pieces else jnp.zeros(0)

    def default_theta(self):
        """Return the declared parameter values, the pytree a solve takes as `theta`."""
        from diff_soup.problem.implicit.solver_lowering import default_theta_from_ir

        return default_theta_from_ir(self)

    def solver_problem(self):
        """Return the [`MassProblem`][diff_soup.problem.MassProblem] or [`ImplicitProblem`][diff_soup.problem.ImplicitProblem] the solvers integrate."""
        from diff_soup.problem.implicit.solver_lowering import solver_problem_from_ir

        return solver_problem_from_ir(self)
