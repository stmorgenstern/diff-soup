"""BoxBounds: Python-primitive-only static config for box-constrained Newton.

The box-constrained corrector splits its bounds into a static side and a
traced side:

- [BoxBounds][diff_soup.nonlinear.box_bounds.BoxBounds] lives in ``cfg`` /
  role objects. It is a frozen dataclass restricted to Python primitives
  (``float`` / ``int`` / ``tuple`` / ``bool``) so it remains hashable for
  ``custom_vjp`` ``nondiff_argnum`` compatibility — passing a `jax.Array`
  here silently breaks frozen-dataclass hashing under JAX transforms.

- [_ResolvedBounds][diff_soup.nonlinear.box_bounds._ResolvedBounds] is the
  trace-time materialization. It carries the
  per-DOF arrays (``constrained_mask``, ``lower_per_dof``, ``upper_per_dof``)
  and rides through ``ctx`` rather than ``cfg`` so the static path stays static
  and the data path stays traced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import jax
import jax.numpy as jnp

_FloatLike = Union[float, tuple[float, ...]]


@dataclass(frozen=True)
class BoxBounds:
    """Lower and upper bounds on selected entries of the state.

    Select the entries with exactly one of `mask` or `indices`. Bounds are Python floats or tuples with one value per selected entry, never JAX arrays.

    Args:
        lower: Lower bound, a float or one value per selected entry.
        upper: Upper bound, strictly above `lower` at every selected entry.
        mask: One boolean per state entry, `True` where bounded.
        indices: Indices of the bounded state entries.
    """

    lower: _FloatLike = 0.0
    upper: _FloatLike = 1.0
    mask: Optional[tuple[bool, ...]] = None
    indices: Optional[tuple[int, ...]] = None

    def __post_init__(self):
        """Reject ``jax.Array`` fields (they break frozen-dataclass hashing)."""
        # jax.Array inside a frozen dataclass under custom_vjp.nondiff_argnum
        # silently breaks hashing — guard at construction so callers who
        # mean "per-DOF bounds" route them through ctx instead.
        assert not isinstance(self.lower, jax.Array), (
            "BoxBounds.lower must be a Python float or tuple of floats; "
            "pass per-DOF arrays via ctx, not cfg"
        )
        assert not isinstance(self.upper, jax.Array), (
            "BoxBounds.upper must be a Python float or tuple of floats"
        )
        assert not isinstance(self.mask, jax.Array), "BoxBounds.mask must be a tuple[bool, ...]"

    def resolve(self, *, n_state: int, dtype) -> "_ResolvedBounds":
        """Expand the bounds to per-entry arrays for a state of `n_state` entries.

        Args:
            n_state: Length of the state.
            dtype: Dtype of the bound arrays.

        Returns:
            The selection mask and full-length lower and upper bound arrays, zero outside the selection.

        Raises:
            ValueError: If the selection or the bounds do not fit `n_state`, or `lower` is not below `upper`.
        """
        if self.mask is not None and self.indices is not None:
            raise ValueError("Specify exactly one of mask= or indices=, not both.")
        if self.mask is None and self.indices is None:
            raise ValueError("Specify exactly one of mask= or indices=.")

        if self.mask is not None:
            if len(self.mask) != n_state:
                raise ValueError(f"mask length {len(self.mask)} != n_state {n_state}")
            constrained_mask = jnp.asarray(list(self.mask), dtype=jnp.bool_)
            indices = tuple(i for i, m in enumerate(self.mask) if m)
        else:
            for i in self.indices:
                if not (0 <= int(i) < n_state):
                    raise ValueError(f"index {i} out of range [0, {n_state})")
            idx_arr = jnp.asarray(self.indices, dtype=jnp.int32)
            constrained_mask = jnp.zeros(n_state, dtype=jnp.bool_).at[idx_arr].set(True)
            indices = tuple(int(i) for i in self.indices)

        n_c = len(indices)

        # Python-time bound validation — runs even inside JIT trace context
        # because the values are Python primitives.
        def _per_index_value(val: _FloatLike, name: str, k: int) -> float:
            if isinstance(val, tuple):
                if len(val) != n_c:
                    raise ValueError(
                        f"{name} tuple length {len(val)} != number of constrained DOFs {n_c}"
                    )
                return float(val[k])
            return float(val)

        for k in range(n_c):
            lo_k = _per_index_value(self.lower, "lower", k)
            hi_k = _per_index_value(self.upper, "upper", k)
            if not (lo_k < hi_k):
                raise ValueError(
                    "lower must be strictly < upper at every constrained DOF; "
                    f"got lower={lo_k}, upper={hi_k} at constrained slot {k}"
                )

        def _broadcast(val: _FloatLike, name: str) -> jax.Array:
            del name
            if isinstance(val, tuple):
                arr = jnp.asarray(list(val), dtype=dtype)
            else:
                arr = jnp.full((n_c,), float(val), dtype=dtype)
            full = jnp.zeros(n_state, dtype=dtype)
            full = full.at[jnp.asarray(indices, dtype=jnp.int32)].set(arr)
            return full

        lower_per_dof = _broadcast(self.lower, "lower")
        upper_per_dof = _broadcast(self.upper, "upper")

        return _ResolvedBounds(
            constrained_mask=constrained_mask,
            lower_per_dof=lower_per_dof,
            upper_per_dof=upper_per_dof,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ResolvedBounds:
    """Trace-time form of [BoxBounds][diff_soup.nonlinear.box_bounds.BoxBounds].

    Lives in ``ctx`` (per-step traced state), not in ``cfg``. Registered as a
    JAX pytree so it flows through ``while_loop`` / ``scan`` carries and through
    ``custom_vjp`` boundaries cleanly.

    Attributes:
        constrained_mask: Per-DOF bool mask, shape ``(n_state,)``.
        lower_per_dof: Per-DOF lower bound, shape ``(n_state,)``.
        upper_per_dof: Per-DOF upper bound, shape ``(n_state,)``.
    """

    constrained_mask: jax.Array
    lower_per_dof: jax.Array
    upper_per_dof: jax.Array

    def tree_flatten(self):
        """Flatten to ``((mask, lower, upper), None)`` for the pytree registry."""
        return (
            (self.constrained_mask, self.lower_per_dof, self.upper_per_dof),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from flattened children (inverse of tree_flatten).

        See [tree_flatten][diff_soup.nonlinear.box_bounds._ResolvedBounds.tree_flatten].
        """
        del aux
        return cls(*children)


def resolve_box_arrays(ctx, cfg, n_u: int, dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single source of truth for ``(lb, ub, mask)`` at solver call time.

    Two paths:

    * ``ctx.bounds`` is a
      [_ResolvedBounds][diff_soup.nonlinear.box_bounds._ResolvedBounds]:
      ``lb / ub`` come from
      ``lower_per_dof / upper_per_dof``, ``mask`` from ``constrained_mask``.
    * ``ctx.bounds is None``: ``lb / ub`` broadcast from
      ``cfg.lower_scalar / cfg.upper_scalar``, ``mask`` reconstructed from
      ``cfg.constrained_indices`` (mirrors the uniform-bound fast path; ``mask``
      is all-False when ``constrained_indices`` is empty, preserving
      unconstrained-call semantics).

    Args:
        ctx: Step context. Reads ``ctx.bounds`` (a
            [_ResolvedBounds][diff_soup.nonlinear.box_bounds._ResolvedBounds]
            or ``None``).
        cfg: Box-corrector config. Reads ``cfg.lower_scalar``,
            ``cfg.upper_scalar``, ``cfg.constrained_indices``.
        n_u: Total number of state DOFs.
        dtype: Float dtype for the returned bound arrays.

    Returns:
        ``(lb, ub, mask)`` with ``lb.shape == ub.shape == mask.shape == (n_u,)``;
        ``mask.dtype == bool``. Unconstrained slots carry ``±inf`` bounds so an
        unconditional ``jnp.clip`` is a no-op there.
    """
    bounds = getattr(ctx, "bounds", None)
    if bounds is None:
        lb_raw = jnp.full((n_u,), float(cfg.lower_scalar), dtype=dtype)
        ub_raw = jnp.full((n_u,), float(cfg.upper_scalar), dtype=dtype)
        mask = jnp.zeros((n_u,), dtype=jnp.bool_)
        if len(cfg.constrained_indices) > 0:
            idx = jnp.asarray(cfg.constrained_indices, dtype=jnp.int32)
            mask = mask.at[idx].set(True)
    else:
        lb_raw = jnp.asarray(bounds.lower_per_dof, dtype=dtype)
        ub_raw = jnp.asarray(bounds.upper_per_dof, dtype=dtype)
        mask = jnp.asarray(bounds.constrained_mask, dtype=jnp.bool_)

    # ``lb`` / ``ub`` must be a no-op at unconstrained slots so the unified
    # primal's [diff_soup.nonlinear._helpers.projected_armijo_scan][] (which
    # clips unconditionally) does
    # not squash free DOFs. Pad with ``±inf`` outside the mask.
    inf = jnp.asarray(jnp.inf, dtype=dtype)
    lb = jnp.where(mask, lb_raw, -inf)
    ub = jnp.where(mask, ub_raw, inf)
    return lb, ub, mask
