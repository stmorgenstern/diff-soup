"""Static Dirichlet boundary conditions for finite-element spaces.

[DirichletBC][diff_soup.domains.mesh.fem.dirichlet_bc.DirichletBC] holds a
location predicate and a value function;
[build_dirichlet_data][diff_soup.domains.mesh.fem.dirichlet_bc.build_dirichlet_data]
evaluates them once at build time into the per-BC node, component and value
arrays the finite-element space stores.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

Array = jax.Array


@final
class DirichletBC(eqx.Module):
    """Static Dirichlet boundary condition.

    ``location_fn`` and ``value_fn`` are stored as ``static`` fields so the
    BC can be composed into other ``eqx.Module`` pytrees (the callables are
    hashed into the JIT cache key rather than traced as leaves).
    """

    location_fn: Callable[..., Array] = eqx.field(static=True)
    component: int = eqx.field(static=True)
    value_fn: Callable[[Array], Array] = eqx.field(static=True)

    def __check_init__(self):
        """Validate callability of ``location_fn`` / ``value_fn`` and component sign."""
        if not callable(self.location_fn):
            raise TypeError(
                f"DirichletBC: expected callable location_fn, got {type(self.location_fn).__name__}"
            )
        if not callable(self.value_fn):
            raise TypeError(
                f"DirichletBC: expected callable value_fn, got {type(self.value_fn).__name__}"
            )
        if not isinstance(self.component, int) or self.component < 0:
            raise ValueError(
                f"DirichletBC: expected component to be a non-negative int, got {self.component!r}"
            )


def build_dirichlet_data(
    points: Array,
    vec: int,
    bcs: Sequence[DirichletBC] | None,
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]:
    """Return (node_inds, vec_inds, vals) as tuples of per-BC arrays.

    Tuples (not lists) are returned so the result is immutable and safe to
    store in ``eqx.field(static=True)`` slots without sharing state across
    instances.
    """
    if not bcs:
        return (), (), ()

    # Dirichlet index arrays are *structural*: their length is a function of
    # the mesh/BC topology, not of the numerical state. ``jnp.argwhere`` below
    # requires a concrete mask, so this routine cannot run inside a traced
    # region (``jax.jit``/``vmap``/``grad``). Fail loudly with an actionable
    # message rather than the cryptic concretization error from ``argwhere``.
    if isinstance(points, jax.core.Tracer):
        raise TypeError(
            "build_dirichlet_data was called inside a traced region "
            "(jax.jit / vmap / grad). Dirichlet BC topology is structural "
            "and must be built with concrete inputs.\n\n"
            "Build the FiniteElementSpace / Problem once, outside jit; the "
            "resulting BC indices are cached on the problem pytree and "
            "ride through the solver as constants. Only the numerical "
            "state (``y0``, ``theta``, learnable BC knot values) should "
            "cross the jit boundary."
        )

    node_inds_list: list[Array] = []
    vec_inds_list: list[Array] = []
    vals_list: list[Array] = []
    ids = jnp.arange(points.shape[0], dtype=jnp.int32)

    for bc in bcs:
        num_args = bc.location_fn.__code__.co_argcount
        if num_args == 1:
            mask = jax.vmap(lambda p: bc.location_fn(p))(points)
        elif num_args == 2:
            mask = jax.vmap(lambda p, i: bc.location_fn(p, i))(points, ids)
        else:
            raise ValueError("DirichletBC.location_fn must take 1 or 2 arguments")

        node_inds = jnp.argwhere(mask).reshape(-1).astype(jnp.int32)
        vec_inds = jnp.full_like(node_inds, bc.component)
        vals = jax.vmap(bc.value_fn)(points[node_inds]).reshape(-1)
        node_inds_list.append(node_inds)
        vec_inds_list.append(vec_inds)
        vals_list.append(vals)

    return tuple(node_inds_list), tuple(vec_inds_list), tuple(vals_list)
