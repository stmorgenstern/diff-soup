"""The context the provider's residual and observation builders evaluate a spec in, and the checks they share."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from diff_soup._errors import ConfigurationError
from diff_soup.problem._ir.equation import MultiEquation


def _check_multi_equation_return(eq: MultiEquation, r) -> None:
    """Reject a bundled-equation return that does not cover its written fields.

    A [MultiEquation][diff_soup.problem._ir.equation.MultiEquation] closure must return a dict with one residual per
    written field name; anything else fails here with the equation named,
    rather than as a downstream key or shape error.
    """
    written = tuple(v.name for v in eq.vars)
    if not isinstance(r, dict):
        raise TypeError(
            f"equation {eq.name!r} writes the fields {written!r} and must "
            f"return a dict keyed by their names; got {type(r).__name__}."
        )
    missing = [name for name in written if name not in r]
    if missing:
        raise ValueError(
            f"equation {eq.name!r} returned no residual for {missing!r}; its "
            f"dict must provide one entry per written field {written!r}."
        )


class _Ctx:
    """Minimal residual/observation context carrying a component's ops.

    The per-equation closures read ``ctx.ops`` for the operator bank and
    ``ctx.controls`` for the evaluation's resolved dimensional control values (``None``
    on surfaces controls do not reach).  A plain attribute holder —
    ``object.__setattr__`` is not overridden.
    """

    def __init__(self, ops: Any, controls: Any = None) -> None:
        self.ops = ops
        self.controls = controls


def _geometry_scale_value(boundary_context, comp) -> jax.Array:
    """The current value of the declared parameter that scales ``comp``'s grid.

    ``boundary_context`` is the evaluation's declared namespaces
    ([_boundary_context][diff_soup.problem._lowering.bcs._boundary_context]), read from
    the runtime equation leaves, so the value is the one this evaluation carries
    (a fixed input, a fitted iterate, or a traced lane value).
    """
    params = (boundary_context or {}).get("params") or {}
    name = comp.geometry_scale
    if name not in params:
        raise ConfigurationError(
            f"component {comp.name!r} is scaled by the parameter {name!r}, but the "
            f"evaluation carries no declared parameter of that name. Declare the "
            f"scale with spec.geometry({name!r}) so its value reaches the kernels."
        )
    return jnp.asarray(params[name])


def _periodic_axes_of_component(comp) -> tuple[int, ...]:
    """Union of periodic axes declared on ``comp`` (empty if none)."""
    from diff_soup.problem._ir.bc import PeriodicBC

    axes: set[int] = set()
    for bc in comp.bcs:
        if isinstance(bc, PeriodicBC):
            axes.update(int(a) for a in bc.axes)
    return tuple(sorted(axes))


def _assembly_dtype(state_dtype, compute_dtype):
    """The residual-assembly dtype for one evaluation: follow a lowered state.

    The boundary cast raises closure outputs to the assembly dtype so a
    closure computed at reduced precision cannot corrupt the residual noise
    floor. When the CALLER deliberately feeds a reduced-precision state — the
    Jacobian-values seam evaluates the residual at fp32 inside its
    differentiation seeds — raising outputs back to fp64 would silently undo
    that choice, so the assembly dtype follows the state downward. It never
    exceeds the configured compute dtype, and with the default fp64 state it
    IS the compute dtype (the emitted program is unchanged).
    """
    sd = jnp.dtype(state_dtype)
    cd = jnp.dtype(compute_dtype)
    return sd if sd.itemsize < cd.itemsize else cd


def _resolve_direct_controls(names, controls, time_ref, t, payload, *, dtype=None):
    """Resolve each direct control's dimensional value for one residual evaluation.

    ``names`` is the static tuple of declared direct-control names.  Each value
    is the override payload's entry when one rides the evaluation, otherwise
    the authored ``fn`` at dimensional time — the same resolution order the solve-level
    control channel applies, so every surface reads one value.

    ``dtype`` is the dtype the evaluation's field values carry.  A control is
    handed to the closures beside those values — a name in the flux face view,
    an entry of ``ctx.controls`` — and multiplies them there, so an authored
    callable returning double precision would widen the whole closure on a
    single-precision state.  Each resolved value is therefore rounded to the
    field dtype exactly once, here; a value that already carries it (the
    solve-level channel casts its payload before it arrives) is passed through
    untouched, so no value is rounded twice.  ``None`` keeps whatever dtype the
    payload or the authored callable produced.
    """
    if not names:
        return {}
    if payload is not None:
        values = getattr(payload, "values", None)
        if hasattr(payload, "epoch") and not callable(values):
            payload = values
    by_name = {str(ctl.name): ctl.fn for ctl in controls}
    t_si = t if time_ref == 1.0 else jnp.asarray(time_ref) * jnp.asarray(t)
    want = None if dtype is None else jnp.dtype(dtype)

    def in_field_dtype(value):
        arr = jnp.asarray(value)
        return arr if want is None or arr.dtype == want else arr.astype(want)

    out = {}
    for name in names:
        if payload is not None and name in payload:
            out[name] = in_field_dtype(payload[name])
            continue
        fn = by_name.get(name)
        if fn is None:
            raise ValueError(
                f"the runtime controls tuple has no entry named {name!r}; it "
                f"must carry every declared control (available: "
                f"{sorted(by_name)!r})."
            )
        out[name] = in_field_dtype(fn(t_si))
    return out


def _face_signature_flux(
    flux_fn,
    is_face_flux,
    owner_view,
    permitted_names,
    plain_ops,
    controls=None,
    boundary_context=None,
):
    """Present one resolved flux callable to a flux-total closure as ``fn(face, t)``.

    A flux authored as ``fn=lambda face, t: ...`` already has that signature.
    The constant and time-dependent forms lower to a ``(view, ctx, t)`` callable
    reading nodal values instead of a face view, so they are wrapped to ignore
    the face view and evaluate against their own component's post-transfer
    values.  Either way the flux-total integral and the natural boundary
    condition evaluate the one density the recipe declared.
    """
    if is_face_flux:

        def boundary_flux(q, t):
            from diff_soup.problem._spec.q_accessor import _BoundaryQ

            query = _BoundaryQ(
                owner_view,
                q._globals,
                q._fi,
                q.geom.normal,
                q.geom.x,
                ops=plain_ops,
                controls=controls,
                t=t,
                **(boundary_context or {}),
            )
            return flux_fn(query, t)

        return boundary_flux
    bc_view = {name: owner_view[name] for name in permitted_names}
    ctx = _Ctx(plain_ops, controls)

    def nodal_flux_as_face_flux(face, t):  # noqa: ARG001 — a nodal flux reads no face
        return flux_fn(bc_view, ctx, t)

    return nodal_flux_as_face_flux
