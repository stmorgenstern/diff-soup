"""The per-particle backend: member elimination and a shared Schur complement.

[MemberSchur][diff_soup.linear.MemberSchur] factors an operator laid out by a
[MemberSchurLayout][diff_soup.linear.MemberSchurLayout]: each member (a
block-tridiagonal chain of grid points with a few scalars of its own) is
eliminated in JAX, all members at once, and the Schur complement left on the
shared unknowns is factored by another exact backend of the contract, the way
[Woodbury][diff_soup.linear.Woodbury] factors its base operator.

The elimination is the lead's core (``_member_elimination.py``): one lane,
right-hand-side columns ``(n, m)``,

    plan = member_schur_plan(layout, topology, elimination)
    tree, shared_values, status = eliminate(plan, values)
    carry, rhs_shared = reduce_rhs(plan, values, tree, columns, transpose)
    x = back_substitute(plan, values, tree, carry, x_shared, transpose)

The handle of one lane is the shared backend's handle leaves, the shared
values the shared factor was made from, the flattened elimination tree and
an ``int32`` status. The factor of either orientation holds the elimination
of ``A_bc``; the member part applies the transpose that
``effective_transpose(orientation, transpose)`` names, and the shared backend
receives the orientation and the flag unchanged, so its factor of ``S`` (for
``"A"``) or ``S^T`` (for ``"AT"``) applies the same transpose.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup._jax_compat import mlir
from diff_soup._layout import Layout
from diff_soup.linear._api import LinearSolver, SparseTopology, analyze
from diff_soup.linear._factor import Analysis, SolveStatus, effective_transpose, rung_dtype
from diff_soup.linear._phased_ad import pattern_matvec
from diff_soup.linear._primitive import (
    phased_factor_p,
    phased_invalidate_p,
    phased_refactor_p,
    phased_solve_p,
)
from diff_soup.linear._structured._common import _select, lanes
from diff_soup.linear._structured._low_rank import Woodbury
from diff_soup.linear._structured._member_block import _ROUNDING_FLOOR, _PlainView
from diff_soup.linear._structured._member_elimination import (
    MemberSchurLayout,
    back_substitute,
    eliminate,
    member_schur_plan,
    reduce_rhs,
)

__all__ = ["MemberSchur"]

_SUCCESS = int(SolveStatus.SUCCESS)
_ELIMINATIONS = ("thomas", "cyclic", "native")


class _MemberSchurPlan:
    """The core's plan, the shared analysis (``None`` without shared unknowns) and leaf counts."""

    __slots__ = ("core", "shared", "treedef", "n_tree", "diagonal", "avals", "__weakref__")

    def __init__(self, core: Any, shared: Optional[Analysis], treedef: Any, n_tree: int, diagonal):
        """Record the plan, the shared analysis and the elimination tree's structure."""
        self.core, self.shared, self.treedef, self.n_tree = core, shared, treedef, n_tree
        self.diagonal, self.avals = diagonal, {}

    @property
    def n_own(self) -> int:
        """Handle leaves after the shared handle: shared values, tree leaves, marker, status."""
        return self.n_tree + 3


_ModuleMeta = type(eqx.Module)


class _MemberSchurMeta(_ModuleMeta):
    """Builds every ``MemberSchur`` as an instance of a subclass specialised to its shared backend.

    The contract reads a backend's capability flags on its class; a
    MemberSchur backend reports its shared backend's flags, as ``Woodbury``
    reports its inner backend's.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Construct through the subclass specialised to the shared backend's class."""
        target = cls
        if cls is MemberSchur:
            shared = kwargs.get("shared", args[1] if len(args) > 1 else None)
            if shared is None:
                from diff_soup.linear import Dense

                shared = Dense()
                kwargs["shared"] = shared
                args = args[:1]
            if not isinstance(shared, LinearSolver):
                raise ConfigurationError(
                    "MemberSchur needs shared= a configured diff_soup.linear.LinearSolver for the "
                    f"shared Schur complement, for example Dense() or KLU(); got "
                    f"{type(shared).__name__}."
                )
            target = _specialised(type(shared))
        return super(_MemberSchurMeta, target).__call__(*args, **kwargs)


#: Capability flags a specialised MemberSchur class copies from its shared backend's class.
_INHERITED_FLAGS = (
    "platforms",
    "factor_residency",
    "transpose_is_free",
    "vmap_factors",
    "supported_precisions",
    "supports_hybrid_memory",
    "supports_reordering",
)


class MemberSchur(LinearSolver, metaclass=_MemberSchurMeta):
    """A direct solver for many members that couple only through shared unknowns.

    Each member is a block-tridiagonal chain of grid points with scalars of its own. The
    members are eliminated in JAX, all at once, and the Schur complement on the shared
    unknowns is factored by `shared`. A member scalar that shares an entry with another
    member is shared.

    Args:
        layout: The `MemberSchurLayout` of the members.
        shared: An exact backend for the shared Schur complement; `None` means `Dense()`.
        elimination: `"thomas"` (a sweep along the grid), `"cyclic"` (block cyclic
            reduction) or `"native"` (the threaded processor kernel).
        residual_tolerance: The relative residual each row of a solve must reach when
            `refinement_sweeps > 0`.
        refinement_sweeps: The most correction sweeps a solve runs; `0` skips the
            residual check.

    !!! example

        ```python
        layout = MemberSchurLayout(members, scalars=scalars, n_dofs=n)
        backend = MemberSchur(layout, shared=KLU())
        f = factor(analyze(topology, backend=backend), values)
        ```
    """

    backend_name = "member_schur"
    is_exact = True
    accepts_operator = False

    layout: MemberSchurLayout = eqx.field(static=True)
    shared: LinearSolver = eqx.field(static=True)
    elimination: str = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    refinement_sweeps: int = eqx.field(static=True)

    def __init__(
        self,
        layout: MemberSchurLayout,
        shared: Optional[LinearSolver] = None,
        *,
        elimination: str = "thomas",
        residual_tolerance: float = 1.0e-10,
        refinement_sweeps: int = 0,
    ):
        """Bind the backend to a layout and a shared backend."""
        if not isinstance(layout, MemberSchurLayout):
            raise ConfigurationError(
                f"MemberSchur needs a diff_soup.linear.MemberSchurLayout; got {type(layout).__name__}."
            )
        cls = type(shared)
        if not cls.is_exact:
            raise ConfigurationError(
                f"MemberSchur needs an exact shared backend; {cls.__name__} declares is_exact=False."
            )
        if cls._analyze is LinearSolver._analyze:
            raise ConfigurationError(
                f"MemberSchur needs a shared backend that implements the analyze/factor/solve "
                f"contract; {cls.__name__} defines no _analyze hook."
            )
        if elimination not in _ELIMINATIONS:
            raise ConfigurationError(
                f"MemberSchur needs elimination= one of {_ELIMINATIONS}; got {elimination!r}."
            )
        if not float(residual_tolerance) > 0.0:
            raise ConfigurationError(
                f"MemberSchur needs residual_tolerance > 0; got {residual_tolerance!r}."
            )
        if int(refinement_sweeps) < 0:
            raise ConfigurationError(
                f"MemberSchur needs refinement_sweeps >= 0; got {refinement_sweeps!r}."
            )
        self.layout = layout
        self.shared = shared
        self.elimination = elimination
        self.residual_tolerance = float(residual_tolerance)
        self.refinement_sweeps = int(refinement_sweeps)

    def __reduce__(self) -> tuple:
        """Pickle by content: unpickling rebuilds the specialised class through ``MemberSchur``."""
        return (
            _rebuild_member_schur,
            (
                self.layout,
                self.shared,
                self.elimination,
                self.residual_tolerance,
                self.refinement_sweeps,
            ),
        )

    # ---- analysis ----------------------------------------------------------------
    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> _MemberSchurPlan:
        """Plan the elimination, then analyze the shared pattern with the shared backend.

        Options, layout and precision pass through to the shared backend.

        Raises:
            ConfigurationError: If the topology does not match the layout, or
                the shared backend refuses the shared pattern.
        """
        from diff_soup.linear._factor import dirichlet_rows

        core = member_schur_plan(self.layout, topology, self.elimination)
        shared = None
        if core.n_shared:
            shared = analyze(
                core.shared_topology,
                backend=self.shared,
                options=options,
                layout=layout,
                precision=precision,
            )
        lane = ShapedArray((int(np.asarray(topology.rows).size),), rung_dtype(precision))
        tree = jax.eval_shape(lambda v: eliminate(core, v)[0], lane)
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        return _MemberSchurPlan(core, shared, treedef, len(leaves), dirichlet_rows(topology)[1])

    def _tree_avals(self, analysis: Analysis, dtype: Any) -> tuple:
        """The abstract leaves of one lane's shared values and elimination tree."""
        return _own_avals(self, analysis, np.dtype(dtype))

    def _shared_avals(self, analysis: Analysis, dtype: Any, orientation: str, precision: str):
        """The abstract leaves of the shared handle (none without shared unknowns)."""
        shared = analysis.backend_analysis.shared
        if shared is None:
            return ()
        return tuple(
            shared.backend.factor_avals(
                shared,
                ShapedArray((shared.nnz,), dtype),
                orientation=orientation,
                precision=precision,
                tol=None,
            )
        )

    def factor_avals(
        self, analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
    ) -> tuple:
        """The shared handle's leaves, the shared values, the tree's leaves, a marker, the status.

        The marker is an ``int32`` array whose length is the shared handle's
        leaf count, so ``label_empty_lanes`` finds the shared handle without
        the analysis.
        """
        del tol
        dtype = rung_dtype(precision, values_aval.dtype)
        shared = self._shared_avals(analysis, dtype, orientation, precision)
        return (
            *shared,
            *self._tree_avals(analysis, dtype),
            ShapedArray((len(shared),), jnp.int32),
            ShapedArray((), jnp.int32),
        )

    def empty_handle(
        self,
        analysis: Analysis,
        values_aval: Any,
        *,
        context_id: int,
        orientation: str,
        precision: str,
        tol: Any,
    ) -> tuple:
        """The shared backend's empty handle, zero arrays and the status ``EMPTY``."""
        shared = analysis.backend_analysis.shared
        dtype = rung_dtype(precision, values_aval.dtype)
        shared_handle = ()
        if shared is not None:
            shared_handle = tuple(
                shared.backend.empty_handle(
                    shared,
                    ShapedArray((shared.nnz,), dtype),
                    context_id=context_id,
                    orientation=orientation,
                    precision=precision,
                    tol=None,
                )
            )
        zeros = [jnp.zeros(a.shape, a.dtype) for a in self._tree_avals(analysis, dtype)]
        return (
            *shared_handle,
            *zeros,
            _marker(len(shared_handle)),
            jnp.asarray(int(SolveStatus.EMPTY), jnp.int32),
        )

    # ---- lane functions ----------------------------------------------------------
    @staticmethod
    def _combine(shared_status: Array, own_status: Array) -> Array:
        """The member status where it is not ``SUCCESS``, else the shared backend's status."""
        return jnp.where(own_status == _SUCCESS, shared_status, own_status).astype(jnp.int32)

    def _eliminate(self, plan: _MemberSchurPlan, values: Array, precision: str) -> tuple:
        """``(shared_values, tree_leaves, status)`` of one lane; ``NONFINITE`` for a bad value."""
        dtype = rung_dtype(precision, values.dtype)
        tree, shared_values, status = eliminate(plan.core, values.astype(dtype))
        status = jnp.where(
            jnp.all(jnp.isfinite(values)), status, int(SolveStatus.NONFINITE)
        ).astype(jnp.int32)
        return shared_values.astype(dtype), tuple(jax.tree_util.tree_leaves(tree)), status

    def _shared_solve(
        self, plan, shared_values, shared_handle, order, rhs, *, orientation, transpose, precision
    ):
        """One solve with the shared factor (``(x, status, order)``), rows ``(m, n_shared)``."""
        return phased_solve_p.bind(
            shared_values,
            *shared_handle,
            order,
            rhs,
            n_handle=len(shared_handle),
            analysis=plan.shared,
            orientation=orientation,
            transpose=transpose,
            precision=precision,
            tol=None,
            storage=None,
        )

    def _structured(
        self, plan, values, leaves, columns, order, *, orientation, transpose, precision
    ):
        """``(x, shared_status, order)``: the member reduction, the shared solve, the back sweep."""
        n_inner = len(leaves) - plan.n_own
        shared_handle, shared_values = leaves[:n_inner], leaves[n_inner]
        tree = jax.tree_util.tree_unflatten(plan.treedef, list(leaves[n_inner + 1 : -2]))
        flip = effective_transpose(orientation, transpose)
        carry, rhs_shared = reduce_rhs(plan.core, values, tree, columns, flip)
        if plan.shared is None:
            return back_substitute(plan.core, values, tree, carry, rhs_shared, flip), None, order
        x_shared, shared_status, order = self._shared_solve(
            plan,
            shared_values,
            shared_handle,
            order,
            rhs_shared.T.astype(columns.dtype),
            orientation=orientation,
            transpose=transpose,
            precision=precision,
        )
        x = back_substitute(plan.core, values, tree, carry, x_shared.T, flip)
        return x, shared_status, order

    def _operator(self, analysis: Analysis, values: Array, x: Array, transpose: bool) -> Array:
        """``A_bc x`` (or ``A_bc^T x``) for columns ``x`` ``(n, m)``."""
        diagonal = analysis.backend_analysis.diagonal
        out = pattern_matvec(_PlainView(analysis.topology), values, x.T, transpose=transpose).T
        if diagonal.size:
            out = out.at[diagonal].add(x[diagonal])
        return out

    def _check(self, analysis, values, columns, x, transpose, row_abs):
        """``(residual, accepted)`` of ``MemberBlock``'s backward-error check for columns ``x``."""
        residual = self._operator(analysis, values, x, transpose) - columns
        abs_x = jnp.abs(x)
        componentwise = self._operator(analysis, jnp.abs(values), abs_x, transpose) + jnp.abs(
            columns
        )
        row_normwise = componentwise + row_abs[:, None] * jnp.max(abs_x, axis=0, keepdims=True)
        floor = _ROUNDING_FLOOR * analysis.n * jnp.finfo(x.dtype).eps
        scale = jnp.where(componentwise > floor * row_normwise, componentwise, row_normwise)
        error = jnp.abs(residual)
        rows_ok = ((scale == 0) & (error == 0)) | (error <= self.residual_tolerance * scale)
        accepted = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(residual)) & jnp.all(rows_ok)
        return residual, accepted

    # ---- contract hooks ----------------------------------------------------------
    def factor_fun(
        self,
        values: Array,
        token: Array,
        context_id: Array,
        *,
        order: Array,
        analysis: Analysis,
        orientation: str,
        precision: str,
        tol: Any,
    ) -> tuple:
        """Eliminate the members, then factor the shared Schur values with the shared backend.

        The shared factorization is bound with this factorization's slot
        token, context id and order word, so the shared backend stores it as
        a factor of its own; its order word is returned.

        Returns:
            ``(*shared_handle, shared_values, *tree_leaves, marker, own_status, status, order)``.
        """
        del tol
        plan = analysis.backend_analysis

        def one(v, o):
            shared_values, leaves, own = self._eliminate(plan, v, precision)
            if plan.shared is None:
                return (shared_values, *leaves, _marker(0), own, own, o)
            *shared_handle, shared_status, o = phased_factor_p.bind(
                shared_values,
                token,
                context_id,
                o,
                analysis=plan.shared,
                orientation=orientation,
                precision=precision,
                tol=None,
            )
            status = self._combine(shared_status, own)
            marker = _marker(len(shared_handle))
            return (*shared_handle, shared_values, *leaves, marker, own, status, o)

        return lanes(one, values.ndim == 2, values, order)

    def apply_fun(
        self,
        values: Array,
        *handle: Array,
        order: Array,
        b: Array,
        analysis: Analysis,
        orientation: str,
        transpose: bool,
        precision: str,
        tol: Any,
    ) -> tuple:
        """Solve with the held factor: member reduction, one shared solve, back substitution.

        With ``refinement_sweeps > 0`` the solve is checked row by row and
        corrected until it passes or the sweeps run out (then ``SINGULAR``).

        Returns:
            ``(x, status, order)``; ``nan`` wherever the status is not ``SUCCESS``.
        """
        del tol
        plan = analysis.backend_analysis
        flip = effective_transpose(orientation, transpose)
        common = dict(orientation=orientation, transpose=transpose, precision=precision)

        def one(v, leaves, o, rhs):
            many = rhs.ndim == 2
            columns = rhs.T if many else rhs[:, None]
            v = v.astype(rhs.dtype)
            own = leaves[-1]
            x, shared_status, o = self._structured(plan, v, leaves, columns, o, **common)
            status = own if shared_status is None else self._combine(shared_status, own)
            if self.refinement_sweeps:
                ones = jnp.ones((analysis.n, 1), jnp.abs(columns).dtype)
                row_abs = self._operator(analysis, jnp.abs(v), ones, flip)[:, 0]
                residual, accepted = self._check(analysis, v, columns, x, flip, row_abs)
                good = status == _SUCCESS

                def rejected(carry):
                    sweep, _, _, ok, _ = carry
                    return good & ~ok & (sweep < self.refinement_sweeps)

                def refine(carry):
                    sweep, xk, rk, _, ok_o = carry
                    dx, _, ok_o = self._structured(plan, v, leaves, rk, ok_o, **common)
                    xk = xk - dx
                    rk, ok = self._check(analysis, v, columns, xk, flip, row_abs)
                    return sweep + 1, xk, rk, ok, ok_o

                _, x, _, accepted, o = jax.lax.while_loop(
                    rejected, refine, (jnp.zeros((), jnp.int32), x, residual, accepted, o)
                )
                status = jnp.where(good & ~accepted, int(SolveStatus.SINGULAR), status)
            x = x.T if many else x[:, 0]
            x = jnp.where(status == _SUCCESS, x, jnp.nan).astype(rhs.dtype)
            return x, status.astype(jnp.int32), o

        batched = values.ndim == 2
        if batched and b.ndim == 1:
            b = jnp.broadcast_to(b, (values.shape[0],) + b.shape)
        return lanes(one, batched, values, tuple(handle), order, b)

    def refactor_fun(
        self,
        values: Array,
        held_values: Array,
        predicate: Array,
        *handle: Array,
        order: Array,
        analysis: Analysis,
        orientation: str,
        precision: str,
        tol: Any,
        adopt: bool,
    ) -> tuple:
        """Eliminate again where a lane asks, then refactor the shared factor by its own rules.

        One ``lax.cond`` on whether any lane asks runs the elimination; each
        lane keeps its held shared values and tree where its predicate is
        false. The shared refactor then receives the new and held shared
        values, the predicate, ``adopt`` and the order word (it needs the new
        Schur values, so it follows the elimination).

        Returns:
            ``(kept_values, *handle, status, order)``.
        """
        del tol
        plan = analysis.backend_analysis
        n_inner = len(handle) - plan.n_own
        batched = values.ndim == 2
        predicate = jnp.asarray(predicate, jnp.bool_)
        held_shared, held_own = tuple(handle[:n_inner]), tuple(handle[n_inner:])

        def rebuild(_):
            def one(v):
                shared_values, leaves, own = self._eliminate(plan, v, precision)
                return (shared_values, *leaves, _marker(n_inner), own)

            return tuple(lanes(one, batched, values))

        def keep(_):
            return held_own

        fresh = jax.lax.cond(jnp.any(predicate), rebuild, keep, None)
        own = tuple(_select(predicate, new, old) for new, old in zip(fresh, held_own))
        kept_values = _select(predicate, values, held_values)
        if plan.shared is None:
            return (kept_values, *own, own[-1].astype(jnp.int32), order)

        def shared_refactor(new, held, p, shared_handle, o):
            return tuple(
                phased_refactor_p.bind(
                    new,
                    held,
                    p,
                    *shared_handle,
                    o,
                    n_handle=n_inner,
                    analysis=plan.shared,
                    orientation=orientation,
                    precision=precision,
                    tol=None,
                    adopt=adopt,
                    storage=None,
                )
            )

        outs = lanes(shared_refactor, batched, own[0], held_own[0], predicate, held_shared, order)
        kept_shared, new_shared, shared_status, order = outs[0], outs[1:-2], outs[-2], outs[-1]
        own = (kept_shared, *own[1:])
        status = self._combine(shared_status, own[-1])
        return (kept_values, *new_shared, *own, status, order)

    def label_empty_lanes(self, handle: tuple, lanes_: Any, size: int) -> tuple:
        """Relabel the shared handle's empty lanes with the shared backend's hook, if it has one.

        The shared handle leads the handle; its length is the marker's, the
        leaf before the status.
        """
        label = getattr(self.shared, "label_empty_lanes", None)
        n_inner = int(handle[-2].shape[-1])
        if label is None or n_inner == 0:
            return handle
        return (*label(tuple(handle[:n_inner]), lanes_, size), *handle[n_inner:])

    def _release(self, ctx: Any) -> None:
        """Release the shared backend's storage held for ``ctx``."""
        self.shared._release(ctx)

    def _resident_bytes(self, ctx: Any) -> int:
        """The shared backend's storage held for ``ctx``, counted once per storage owner."""
        mine = _storage_class(self)
        for registered in tuple(getattr(ctx, "_backends", ())):
            if _storage_class(registered) is mine:
                if registered is not self:
                    return 0
                break
        return int(self.shared._resident_bytes(ctx))


def _marker(n_inner: int) -> Array:
    """The handle leaf whose length is the shared handle's leaf count."""
    return jnp.zeros((n_inner,), jnp.int32)


def _own_avals(backend: MemberSchur, analysis: Analysis, dtype: np.dtype) -> tuple:
    """Shared values and tree leaves of one lane as abstract arrays (traced once per dtype)."""
    plan = analysis.backend_analysis
    if dtype not in plan.avals:
        lane = ShapedArray((analysis.nnz,), dtype)
        precision = "fp64" if np.finfo(dtype).bits >= 64 else "fp32"
        shared_values, leaves, _ = jax.eval_shape(
            lambda v: backend._eliminate(plan, v, precision), lane
        )
        plan.avals[dtype] = tuple(
            ShapedArray(tuple(a.shape), a.dtype) for a in (shared_values, *leaves)
        )
    return plan.avals[dtype]


def _storage_class(backend: Any) -> type:
    """The class of the backend that owns a backend's native storage (through the wrappers)."""
    while isinstance(backend, (MemberSchur, Woodbury)):
        backend = backend.shared if isinstance(backend, MemberSchur) else backend.inner
    return type(backend)


def _rebuild_member_schur(layout, shared, elimination, residual_tolerance, refinement_sweeps):
    """Unpickle a MemberSchur backend through its constructor."""
    return MemberSchur(
        layout,
        shared,
        elimination=elimination,
        residual_tolerance=residual_tolerance,
        refinement_sweeps=refinement_sweeps,
    )


def _lower_invalidate(self, ctx: Any, *args: Any, **params: Any) -> Any:
    """Lower an invalidation of a MemberSchur factor as the shared backend's invalidation."""
    analysis = params["analysis"]
    plan = analysis.backend_analysis
    n_inner = int(params["n_handle"]) - plan.n_own

    def invalidate_shared(*operands):
        *handle, context_id, order = operands
        if plan.shared is None:
            return [order]
        return list(
            phased_invalidate_p.bind(
                *handle[:n_inner],
                context_id,
                order,
                n_handle=n_inner,
                analysis=plan.shared,
                orientation=params["orientation"],
                precision=params["precision"],
            )
        )

    return mlir.lower_fun(invalidate_shared, multiple_results=True)(ctx, *args)


@functools.lru_cache(maxsize=64)
def _specialised(shared_cls: type) -> type:
    """The subclass of ``MemberSchur`` whose capability flags are ``shared_cls``'s."""
    namespace: dict[str, Any] = {
        name: getattr(shared_cls, name) for name in _INHERITED_FLAGS if hasattr(shared_cls, name)
    }
    namespace["__module__"] = MemberSchur.__module__
    namespace["__qualname__"] = f"MemberSchur[{shared_cls.__name__}]"
    namespace["__doc__"] = MemberSchur.__doc__
    if getattr(shared_cls, "lower_invalidate", None) is not None:
        namespace["lower_invalidate"] = _lower_invalidate
    return _MemberSchurMeta(f"MemberSchur[{shared_cls.__name__}]", (MemberSchur,), namespace)
