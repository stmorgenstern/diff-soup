"""The views and adapters through which an equation's functions read the state q."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from diff_soup.problem._ir.content_hash import _closure_fingerprint

from .q_accessor import _QView


def _make_q_view(view, ctx, t, var_map, params, functions, quad_fields=()):
    controls = getattr(ctx, "controls", None)
    domain = getattr(ctx.ops, "domain", None)
    coords_fn = getattr(domain, "node_coordinates", None) if domain is not None else None
    # Grids store their coordinates in double precision; a single-precision
    # problem hands its equations coordinates in its own (time) dtype.
    time_dtype = getattr(t, "dtype", None)
    narrow = time_dtype is not None and jnp.promote_types(jnp.float64, time_dtype) != time_dtype
    in_field_dtype = getattr(domain, "in_field_dtype", None)
    if narrow and in_field_dtype is not None:
        coords_fn = in_field_dtype(jnp.zeros((), time_dtype)).node_coordinates
    coords = coords_fn() if callable(coords_fn) else None
    if narrow and coords is not None and jnp.promote_types(coords.dtype, time_dtype) != time_dtype:
        coords = jnp.asarray(coords).astype(time_dtype)
    params_dict = params.as_dict() if hasattr(params, "as_dict") else params
    functions_dict = functions.as_dict() if hasattr(functions, "as_dict") else functions
    # The pointwise marks ride the functions namespace itself, so every closure
    # that reads ``q.fn`` — equation, constraint integrand, observation — sees
    # the same per-node application.
    pointwise_functions = getattr(functions, "pointwise_names", ())
    return _QView(
        view=view,
        var_map=var_map,
        ops=ctx.ops,
        params=params_dict,
        functions=functions_dict,
        pointwise_functions=pointwise_functions,
        t=t,
        coords=coords,
        quad_fields=quad_fields,
        controls=controls,
    )


class _TracedQAdapter(eqx.Module):
    """Wrap a user closure ``f(q)`` so its ``eqx.Module`` leaves survive partition.

    When the user's closure is an ``eqx.Module``, wrapping it in a plain
    Python function would bury its parameter arrays in a closure cell —
    those leaves would never reach
    ``eqx.partition(problem, eqx.is_inexact_array)[0]`` and ``jax.grad``
    would silently return zero.  Routing through this adapter (itself an
    ``eqx.Module``) keeps the ``inner`` field non-static so its leaves
    surface on the AD boundary.
    """

    inner: object
    var_map: object
    params: object
    functions: object
    learnable_params: tuple[str, ...] = eqx.field(static=True, default=())
    learnable_functions: tuple[str, ...] = eqx.field(static=True, default=())
    # Sorted tuple of quad-opted field names (a tuple, not a frozenset, so the
    # content-hash static-leaf walk fingerprints it deterministically).
    quad_fields: tuple[str, ...] = eqx.field(static=True, default=())

    def __call__(self, view, ctx, t):
        q = _make_q_view(view, ctx, t, self.var_map, self.params, self.functions, self.quad_fields)
        return self.inner(q)


def _make_q_adapter(
    spec_fn: Callable,
    var_map: dict[str, Any],
    params: Any,
    functions: Any,
    *,
    learnable_params: Iterable[str] = (),
    learnable_functions: Iterable[str] = (),
    quad_fields: Iterable[str] = (),
) -> Callable:
    """Wrap a spec-level closure ``f(q)`` for IR-level callsite ``f(view, ctx, t)``.

    ``ctx.ops`` carries an [Operators][] whose ``.domain`` exposes
    ``node_coordinates()`` for ``q.x``.  ``eqx.Module`` callables route
    through [_TracedQAdapter][] to keep their leaves visible under
    AD partition. Declared parameters and declared functions also use a
    traced adapter, so their values are leaves of the lowered problem: fixed
    per-experiment inputs can be replaced without rebuilding the residual, and
    no declared value is baked into a compiled program.
    A plain callable with no learnable declarations still obeys the strict
    static-capture contract, even when that adapter carries fixed parameters.

    ``quad_fields`` is the set of field names declared ``spec.field(quad=True)``;
    it rides into the per-call [_QView][].  Stored as a SORTED TUPLE (the
    closure default / static eqx field must be content-hashable — a frozenset is
    not).  Defaults to empty (no field wrapped — today's behaviour).
    """
    learnable_params = tuple(learnable_params)
    learnable_functions = tuple(learnable_functions)
    quad_fields = tuple(sorted(quad_fields))
    needs_traced_adapter = (
        isinstance(spec_fn, eqx.Module)
        or bool(learnable_params)
        or bool(learnable_functions)
        or bool(getattr(params, "names", ()))
        or bool(getattr(functions, "names", ()))
    )
    if needs_traced_adapter:
        if not isinstance(spec_fn, eqx.Module) and not (learnable_params or learnable_functions):
            _closure_fingerprint(spec_fn)
        return _TracedQAdapter(
            inner=spec_fn,
            var_map=var_map,
            params=params,
            functions=functions,
            learnable_params=learnable_params,
            learnable_functions=learnable_functions,
            quad_fields=quad_fields,
        )

    def adapter(
        view,
        ctx,
        t,
        _spec_fn=spec_fn,
        _var_map=var_map,
        _params=params,
        _functions=functions,
        _quad_fields=quad_fields,
    ):
        q = _make_q_view(view, ctx, t, _var_map, _params, _functions, _quad_fields)
        return _spec_fn(q)

    return adapter
