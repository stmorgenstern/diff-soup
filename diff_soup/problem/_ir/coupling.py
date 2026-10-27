"""Coupling IR node + TransferOp wrappers.

Cross-component data transfer.  A [Coupling][] carries source + target
``Variable`` handles plus a [TransferOp][] that maps source DOF values into
the target's slot at residual time.  Lowering resolves these to global-index
injections between per-component ``var_view`` dicts.

The IR encodes the source/target plumbing so tree-rewrites and the content-hash
machinery compose cleanly through coupled problems.

The ``FamilyScatterOp.reduce`` field uses a frozen [Reduce][] union
([Sum][] | [Mean][]) with an explicit ``injective: bool`` flag,
replacing the legacy 3-value ``reduce`` string.  The former ``'none'`` mode —
an injectivity-assertion marker that behaved like ``'sum'`` at residual time —
is now expressed as ``Sum()`` with ``injective=True``.  The codec round-trips a
[Reduce][] back to a string for wire stability (``Sum()`` → ``'sum'``,
``Mean()`` → ``'mean'``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .variable import Variable


class Reduce(eqx.Module):
    """Frozen marker base for a [FamilyScatterOp][] reduction.

    Concrete subclasses are [Sum][] (scatter-ADD, no divisor) and
    [Mean][] (scatter-ADD then divide each target row by its host-known
    per-segment member count).  The union is closed — adding a member requires
    updating all dispatch sites.  The codec encodes a [Reduce][] back to a
    string for wire stability.
    """

    @property
    def codec_name(self) -> str:
        """Wire-stable string for the codec (``'sum'`` / ``'mean'``)."""
        raise NotImplementedError(f"{type(self).__name__} must override codec_name")


class Sum(Reduce):
    """Scatter-ADD reduction: accumulate per-member contributions, no divisor."""

    @property
    def codec_name(self) -> str:
        """Wire name for content-hash and codec serialisation."""
        return "sum"


class Mean(Reduce):
    """Scatter-ADD then divide each target row by its per-segment member count.

    The divisor is host-known from ``at_indices`` (computed statically with
    `numpy.bincount`), never a data-dependent count.
    """

    @property
    def codec_name(self) -> str:
        """Wire name for content-hash and codec serialisation."""
        return "mean"


def _coerce_reduce(reduce: "Reduce | str") -> Reduce:
    """Accept a [Reduce][] instance or its ``'sum'``/``'mean'`` wire name.

    Raw-IR / codec callers may still pass the wire string; the facade passes a
    typed [Reduce][]. Express injective sum behavior as ``Sum()`` with
    ``injective=True``.
    """
    if isinstance(reduce, Reduce):
        return reduce
    if reduce == "sum":
        return Sum()
    if reduce == "mean":
        return Mean()
    raise ValueError(
        "FamilyScatterOp.reduce must be a Reduce instance (Sum()/Mean()) or "
        f"the string 'sum'/'mean'; got {reduce!r}. For injective sum behavior, "
        "use Sum() with injective=True."
    )


class TransferOp(eqx.Module):
    """Base class of the operators that move a field's values into another component.

    Pass one as `spec.couple(..., via=op)`. A subclass implements `apply` and
    `sparsity`, and overrides `alias_size(n_source=..., n_target=...)` when the
    coupled value is not the size of the target component.
    """

    def alias_size(self, *, n_source: int, n_target: int) -> int:
        """Length of the read-through alias this op produces on the consumer.

        Defaults to the consuming component's node count (inject-style ops
        deliver a full-size array with zeros elsewhere). An op producing a
        compact output (a gather or trace) overrides this with its actual
        output length.
        """
        del n_source
        return int(n_target)

    def apply(self, source: Array, target_size: int) -> Array:
        """Map a source vector to a target vector.

        Args:
            source: Values of the coupled field.
            target_size: Length of the returned vector.

        Returns:
            The transferred values, of length `target_size`.
        """
        raise NotImplementedError(f"{type(self).__name__} must override apply(source, target_size)")

    def sparsity(self, source_size: int, target_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the nonzero pattern of the Jacobian `d(target)/d(source)`.

        A dense transfer lists every pair of target row and source column.

        Args:
            source_size: Length of the source vector.
            target_size: Length of the target vector.

        Returns:
            Integer arrays `(target_rows, source_cols)` of the nonzeros.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override sparsity(source_size, target_size) to "
            "return the (target_rows, source_cols) nonzeros of d(target)/d(source); a "
            "dense transfer returns the explicit product over target_size and source_size"
        )


class NodalInjectOp(TransferOp):
    """Transfer that writes source entry `k` into target row `target_indices[k]` and zero elsewhere.

    Args:
        target_indices: Target rows, one per source entry.
    """

    target_indices: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, target_indices):
        if isinstance(target_indices, np.ndarray):
            ints = tuple(int(x) for x in target_indices.flat)
        else:
            ints = tuple(int(x) for x in target_indices)
        self.target_indices = ints

    def apply(self, source: Array, target_size: int) -> Array:
        """Scatter ``source`` into a ``target_size`` zero array at ``target_indices``."""
        out = jnp.zeros(target_size, dtype=source.dtype)
        idx = jnp.asarray(self.target_indices, dtype=jnp.int32)
        return out.at[idx].set(source)

    def sparsity(self, source_size: int, target_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(target_rows, source_cols)`` for the inject's sparse Jacobian."""
        # ``target[target_indices[k]] = source[k]`` for k in [0, source_size);
        # other target rows are identically zero.
        del target_size
        rows = np.asarray(self.target_indices, dtype=np.int64)
        cols = np.arange(source_size, dtype=np.int64)
        if rows.shape != cols.shape:
            raise ValueError(
                "NodalInjectOp.sparsity: target_indices length "
                f"({rows.size}) must equal source_size ({source_size})"
            )
        return rows, cols


class SphericalSurfaceTraceOp(TransferOp):
    """Extract the surface DOF (last node) of a spherical particle.

    The output is shape ``(1,)`` regardless of source size.
    """

    def alias_size(self, *, n_source: int, n_target: int) -> int:
        """A surface trace always produces a single value."""
        del n_source, n_target
        return 1

    def apply(self, source: Array, target_size: int) -> Array:
        """Extract the last element of ``source`` (the radial surface DOF)."""
        del target_size
        return source[-1:]

    def sparsity(self, source_size: int, target_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse pattern: target[0] depends only on source[-1]."""
        # ``target[0] = source[-1]``; target_size is always 1 by construction.
        del target_size
        rows = np.zeros(1, dtype=np.int64)
        cols = np.array([source_size - 1], dtype=np.int64)
        return rows, cols


class RadialSurfaceInjectOp(TransferOp):
    """Inject a 2D particle's outermost-radial surface trace into a 1D axial target.

    Maps a 2D ``(axial, radial)`` particle variable's surface (outermost-radial)
    trace into a 1D axial target at the specified axial indices.

    Source must be a flat ``(n_axial * n_radial,)`` view of a Variable with
    shape ``(n_axial, n_radial)``.  The op reshapes, slices ``[:, -1]`` to pull
    out the surface concentration per axial node, then scatters into target at
    ``target_indices``.

    Used by P2D fixtures (full radial diffusion in particles) to feed the
    cell-side electrolyte / Phi_e closures the surface concentration without
    introducing a redundant ``c_s_surf`` algebraic variable.
    """

    target_indices: Tuple[int, ...] = eqx.field(static=True)
    n_radial: int = eqx.field(static=True)

    def __init__(self, target_indices, n_radial):
        if isinstance(target_indices, np.ndarray):
            ints = tuple(int(x) for x in target_indices.flat)
        else:
            ints = tuple(int(x) for x in target_indices)
        self.target_indices = ints
        self.n_radial = int(n_radial)

    def apply(self, source: Array, target_size: int) -> Array:
        """Extract the outermost-radial surface trace and scatter into target at target_indices."""
        source_2d = source.reshape(-1, self.n_radial)
        surf = source_2d[:, -1]
        out = jnp.zeros((target_size,), dtype=source.dtype)
        idx = jnp.asarray(self.target_indices, dtype=jnp.int32)
        return out.at[idx].set(surf)

    def sparsity(self, source_size: int, target_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse pattern: each axial target node depends on its outermost radial source node."""
        del target_size
        n_axial = source_size // self.n_radial
        if n_axial != len(self.target_indices):
            raise ValueError(
                "RadialSurfaceInjectOp.sparsity: source has "
                f"{n_axial} axial slices but target_indices has "
                f"{len(self.target_indices)} entries"
            )
        rows = np.asarray(self.target_indices, dtype=np.int64)
        cols = np.array(
            [(i + 1) * self.n_radial - 1 for i in range(n_axial)],
            dtype=np.int64,
        )
        return rows, cols


class NodalExtractOp(TransferOp):
    """Transfer that reads entry `i` of the coupled value from source row `source_indices[i]`.

    Args:
        source_indices: Source rows, one per entry of the coupled value.
    """

    source_indices: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, source_indices):
        if isinstance(source_indices, np.ndarray):
            ints = tuple(int(x) for x in source_indices.flat)
        else:
            ints = tuple(int(x) for x in source_indices)
        self.source_indices = ints

    def alias_size(self, *, n_source: int, n_target: int) -> int:
        """A compact gather's alias has one entry per selected source index."""
        del n_source, n_target
        return len(self.source_indices)

    def apply(self, source: Array, target_size: int) -> Array:
        """Gather ``source`` at ``source_indices``, producing a compact ``target_size`` output."""
        del target_size
        idx = jnp.asarray(self.source_indices, dtype=jnp.int32)
        return source[idx]

    def sparsity(self, source_size: int, target_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse pattern: target[i] depends only on source[source_indices[i]]."""
        # ``target[i] = source[source_indices[i]]`` for i in [0, target_size).
        del source_size
        rows = np.arange(target_size, dtype=np.int64)
        cols = np.asarray(self.source_indices, dtype=np.int64)
        if rows.shape != cols.shape:
            raise ValueError(
                "NodalExtractOp.sparsity: source_indices length "
                f"({cols.size}) must equal target_size ({target_size})"
            )
        return rows, cols


class FamilyScatterOp(TransferOp):
    """Many:1 scatter-ADD of per-member surface traces into a shared target.

    Each Family member contributes one DOF (the member-local surface trace at
    ``member_surface_dof``) which is *accumulated* — ``.at[].add``, not
    ``.at[].set`` — into the target row given by ``at_indices``.  Members that
    share an ``at_indices`` value (co-located particles) therefore sum into the
    same target row, the physical behaviour for e.g. a population of particles
    feeding a common cell-side flux.

    For a [Mean][] reduce each target row is divided by its per-segment
    member count.  That divisor is host-known from ``at_indices`` (computed
    statically with `numpy.bincount`), never a data-dependent count.

    ``reduce`` is a frozen [Reduce][] ([Sum][] | [Mean][]). When
    ``injective=True`` the op behaves at residual time exactly like ``Sum()``
    (scatter-ADD, no divisor), but lowering-time static validate
    (``_validate_families``) rejects it when any two members share an
    ``at_indices`` target.  The facade desugars its ``reduce='none'`` form to
    ``Sum()`` + ``injective=True`` after running that injectivity check; a
    raw-IR op may carry ``injective=True`` so the lowering-time guard fires for
    direct-IR declarations.

    Static fields:
        at_indices: Target row index per member; ``len == count``.
        member_surface_dof: DOF carrying the surface trace, counted from the
            start of the projected field inside a member (the lowering adds the
            field's block offset, which already places that field within the
            member layout).  For a field with ``n`` nodes this is ``n - 1``,
            whether or not the family carries other member fields.
        n_member_dof: DOFs per member (member stride into the flat source).
        reduce: A [Reduce][] ([Sum][] accumulate / [Mean][]
            accumulate-then-divide).
        injective: When ``True``, lowering validates that no two members share
            an ``at_indices`` target; residual behavior is identical to
            [Sum][].
    """

    at_indices: Tuple[int, ...] = eqx.field(static=True)
    member_surface_dof: int = eqx.field(static=True)
    n_member_dof: int = eqx.field(static=True)
    reduce: Reduce = eqx.field(static=True)
    injective: bool = eqx.field(static=True)

    def __init__(
        self,
        at_indices,
        member_surface_dof,
        n_member_dof,
        reduce: "Reduce | str" = Sum(),
        injective: bool = False,
    ):
        if isinstance(at_indices, np.ndarray):
            ints = tuple(int(x) for x in at_indices.flat)
        else:
            ints = tuple(int(x) for x in at_indices)
        self.at_indices = ints
        self.member_surface_dof = int(member_surface_dof)
        self.n_member_dof = int(n_member_dof)
        self.reduce = _coerce_reduce(reduce)
        self.injective = bool(injective)

    def apply(self, source: Array, target_size: int) -> Array:
        """Scatter per-member surface values into a ``target_size`` array, then reduce."""
        # ``source`` is a flat per-member vector of surface traces (one value
        # per member, in member order).  Scatter-ADD into the target.
        idx = jnp.asarray(self.at_indices, dtype=jnp.int32)
        out = jnp.zeros((target_size,), dtype=source.dtype)
        out = out.at[idx].add(source)
        if isinstance(self.reduce, Mean):
            # Per-row member count is host-known from at_indices (NOT a
            # data-dependent reduction): bincount over the target rows.
            counts = np.bincount(
                np.asarray(self.at_indices, dtype=np.int64),
                minlength=target_size,
            ).astype(np.float64)
            divisor = jnp.asarray(np.where(counts == 0.0, 1.0, counts), dtype=out.dtype)
            out = out / divisor
        return out

    def sparsity(self, source_size: int, target_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse pattern: one non-zero per member (surface DOF -> target row)."""
        # One non-zero per member: target row at_indices[m] depends on the
        # member's surface DOF at column ``m * n_member_dof + surf`` relative
        # to the projected field's block offset (field-relative, like ``apply``'s
        # caller-side gather).
        del source_size, target_size
        n_member = len(self.at_indices)
        rows = np.asarray(self.at_indices, dtype=np.int64)
        cols = np.array(
            [m * self.n_member_dof + self.member_surface_dof for m in range(n_member)],
            dtype=np.int64,
        )
        return rows, cols


class Coupling(eqx.Module):
    """Cross-component data transfer.

    During lowering, the framework wires
    ``source.value -> op.apply(source.value, target.size) -> target slot`` so the
    target component's residual closure can read it as if local.

    ``read_through`` (default ``False``) makes this a NO-DOF *read-through*
    coupling: instead of overwriting a state-DOF slot owned by the target
    component, the transferred value ``op.apply(source, target.size)`` is
    injected into the consuming closure's
    [MergedView][diff_soup.problem._ir._merged_view.MergedView]
    under the ``target`` Variable's alias — so the
    closure reads it as ``view[target]`` (i.e. ``q.<alias>``) by name, with
    **no state DOF, no residual row, no IC**.  The ``target`` Variable still
    carries the alias's ``(name, shape, mass)`` and owning-component identity
    (resolved from the consuming equation's ``inputs``), but it is NEVER added to
    any ``Component.variables`` — layout allocates no DOF for it.  This
    generalises the existing family-gather read-through (component->family) to
    component->component (``NodalExtractOp``) and family->component-surface (the
    family-slab surface trace mapped via ``at_indices``).  Default ``False``
    leaves every existing coupling unchanged.

    ``target_owner`` (default ``None``) names the component that consumes a
    read-through alias.  Lowering otherwise infers the owner by scanning every
    equation's ``inputs`` for the alias, which finds nothing when the alias is
    read ONLY by a boundary-condition callback (a flux closure) and by no
    equation.  Setting ``target_owner`` makes the owning component explicit so
    such an alias still resolves; the equation scan stays the fallback when it
    is ``None``.
    """

    name: str = eqx.field(static=True)
    source: Variable
    target: Variable
    op: TransferOp
    read_through: bool = eqx.field(static=True, default=False)
    target_owner: Optional[str] = eqx.field(static=True, default=None)


__all__ = [
    "Coupling",
    "TransferOp",
    "NodalInjectOp",
    "NodalExtractOp",
    "FamilyScatterOp",
    "RadialSurfaceInjectOp",
    "SphericalSurfaceTraceOp",
    "Reduce",
    "Sum",
    "Mean",
]
