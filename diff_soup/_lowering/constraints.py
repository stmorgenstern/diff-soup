"""Constraint residual evaluation.

``_eval_constraint_residual`` is the per-constraint dispatcher that each
constraint kind (Galvanostatic / DomainIntegral / ReduceResidual / DifferentialGlobalEquation /
BoundaryIntegral / OperationalControlConstraint) flows through.  Called from inside the residual kernel in
``provider.py``; kept separate because the dispatch logic is dense.

This module lives under ``diff_soup/_lowering`` and may import
``diff_soup.problem._ir`` IR nodes.  A volumetric integral always carries the
domain measure its lowering resolved (grid cell volumes, or the lumped nodal
volumes ``∫ N_i dV`` of a mesh), so ``domain_measure`` is required.  A boundary
integral carries a nodal surface measure only when the constraint names a
boundary; with no boundary to measure (``boundary_measure=None``) it reduces to
a bare element sum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diff_soup.problem._ir.constraint import (
    BoundaryIntegralConstraint,
    DifferentialGlobalEquation,
    DomainIntegralConstraint,
    FluxTotalConstraint,
    GalvanostaticConstraint,
    OperationalControlConstraint,
    ReduceResidualConstraint,
)

from .bcs import _FaceView


def _eval_constraint_residual(
    constraint,
    var_view,
    ctx,
    lam_slice,
    t,
    *,
    f=None,
    field_slice=None,
    boundary_measure=None,
    domain_measure=None,
    flux_total_link=None,
    control=None,
    time_ref: float = 1.0,
    hold_residual_current: bool = False,
) -> jax.Array:
    """Compute one constraint's residual.  Dispatches by type.

    ``f`` is the assembled physics residual; ``field_slice`` is a host-side
    ``(offset, size)`` tuple pre-resolved by the residual builder from
    ``constraint.field`` against the owning component's layout slot.
    ``ReduceResidualConstraint`` and an ``OperationalControlConstraint`` whose
    current is a field's residual sum consume these to read the field's slice
    without re-running the field-equation kernel.  ``hold_residual_current``
    holds that residual-sum current constant under differentiation: the
    bordered-global derivative kernel sets it for a table whose current row
    derivative is assembled from the Jacobian blocks instead.

    ``boundary_measure`` / ``domain_measure`` are host-resolved measures.
    ``DomainIntegralConstraint`` requires ``domain_measure``; a boundary
    integral that names no boundary gets ``boundary_measure=None`` and sums its
    integrand unweighted.

    ``control`` is the operating state an ``OperationalControlConstraint`` row
    reads its mode and target from; a target with time terms is evaluated at
    ``t``, the solve's own time. ``time_ref`` is the time scale that turns
    internal time into the dimensional seconds every ``target_fn`` callback expects.
    """
    del lam_slice  # globals reach var_view via the merged-view global slot.
    if isinstance(constraint, DifferentialGlobalEquation):
        # The row supplies the RHS of ``d(global)/dt = rhs``; the global's
        # unit mass turns the canonical residual into ``ydot - rhs`` here.
        rhs = jnp.asarray(constraint.rhs_fn(var_view, ctx, t))
        return jnp.broadcast_to(rhs.reshape((-1,)), (constraint.dim,))
    if isinstance(constraint, OperationalControlConstraint):
        mode = getattr(control, "mode", None)
        target = getattr(control, "target", None)
        if mode is None or target is None:
            from diff_soup._errors import ConfigurationError

            raise ConfigurationError(
                f"this problem declares an operational control table "
                f"({constraint.name!r}); evaluate it with an operating control "
                "state: ControlState.operating(mode=..., target=...)."
            )
        if constraint.field is not None:
            if f is None or field_slice is None:
                raise RuntimeError(
                    f"operational control {constraint.name!r} takes its current from "
                    "the residual rows of a field and needs the assembled residual "
                    "and the field's slice; call sites must pass f= and field_slice="
                )
            off, size = field_slice
            i_obs = jnp.sum(f[off : off + size]).reshape(())
            if hold_residual_current:
                i_obs = jax.lax.stop_gradient(i_obs)
        else:
            i_obs = jnp.asarray(constraint.current_fn(var_view, ctx, t)).reshape(())
        v_obs = jnp.asarray(constraint.voltage_fn(var_view, ctx, t)).reshape(())
        forms = {"current": i_obs, "voltage": v_obs, "power": v_obs * i_obs}
        rows = jnp.stack([forms[m] for m in constraint.modes])
        mode_idx = jnp.asarray(mode, dtype=jnp.int32)
        selected = jnp.take(rows, mode_idx)
        tgt = jnp.asarray(target)
        if getattr(control, "target_terms", None) is not None:
            # A target that varies in time, at the residual's own time.
            tgt = control.target_at(t)
        # A negative mode is the initialization path's internal terminal-
        # voltage probe; user-facing states reject negative indices eagerly.
        value = jnp.where(mode_idx < 0, v_obs - tgt, selected - tgt)
        return jnp.broadcast_to(value.reshape((-1,)), (constraint.dim,))
    # Residual is ``nondim_quantity - target_SI / integrand_scale``.
    # Default ``integrand_scale=1.0`` reproduces the un-scaled residual.
    # ``target_fn`` is a callback and runs on the dimensional clock, like every
    # boundary callback; the integrand/current closures keep internal time,
    # the clock every equation closure reads.
    t_target = jnp.asarray(time_ref) * jnp.asarray(t)
    scale = float(getattr(constraint, "integrand_scale", 1.0))
    if isinstance(constraint, GalvanostaticConstraint):
        cur = jnp.asarray(constraint.current_fn(var_view, ctx, t))
        tgt = jnp.asarray(constraint.target_fn(t_target)) / scale
        return jnp.broadcast_to((cur - tgt).reshape((-1,)), (constraint.dim,))
    if isinstance(constraint, DomainIntegralConstraint):
        integrand = jnp.asarray(constraint.integrand_fn(var_view, ctx, t))
        tgt = jnp.asarray(constraint.target_fn(t_target)) / scale
        if domain_measure is None:
            raise RuntimeError(
                "DomainIntegralConstraint requires the host-resolved domain "
                "measure; call sites must pass domain_measure="
            )
        measure = jnp.asarray(domain_measure).reshape((-1,))
        n_measure = int(measure.shape[0])
        if integrand.ndim == 0:
            # A uniform integrand integrates to itself times the domain volume.
            weighted = integrand * jnp.sum(measure)
        else:
            flat = integrand.reshape((-1,))
            m = int(flat.shape[0])
            if m != n_measure:
                raise ValueError(
                    f"domain integral {constraint.name!r}: integrand has length {m}, "
                    f"but the domain measure has {n_measure} entries. Return one "
                    "value per node of the domain (each weighted by its cell volume "
                    "or lumped nodal volume) or a single scalar (weighted by the "
                    "total volume of the domain)."
                )
            weighted = jnp.sum(flat * measure)
        return jnp.broadcast_to((weighted - tgt).reshape((-1,)), (constraint.dim,))
    if isinstance(constraint, ReduceResidualConstraint):
        if f is None or field_slice is None:
            raise RuntimeError(
                "ReduceResidualConstraint requires the assembled residual "
                "and pre-resolved field slice; call sites must pass f= and "
                "field_slice="
            )
        off, size = field_slice
        integrand = f[off : off + size]
        reduced = jnp.asarray(constraint.reduce_fn(integrand))
        tgt = jnp.asarray(constraint.target_fn(t_target)) / scale
        return jnp.broadcast_to((reduced - tgt).reshape((-1,)), (constraint.dim,))
    if isinstance(constraint, BoundaryIntegralConstraint):
        integrand = jnp.asarray(constraint.integrand_fn(var_view, ctx, t))
        tgt = jnp.asarray(constraint.target_fn(t_target)) / scale
        if boundary_measure is None:
            if integrand.ndim == 0:
                return jnp.broadcast_to((integrand - tgt).reshape((-1,)), (constraint.dim,))
            return jnp.broadcast_to((jnp.sum(integrand) - tgt).reshape((-1,)), (constraint.dim,))
        nodal_measure = jnp.asarray(boundary_measure.nodal_measure)
        n_tag_nodes = int(nodal_measure.shape[0])
        if integrand.ndim == 0:
            weighted = integrand * jnp.sum(nodal_measure)
            return jnp.broadcast_to((weighted - tgt).reshape((-1,)), (constraint.dim,))
        flat = integrand.reshape((-1,))
        m = int(flat.shape[0])
        # Dispatch on the integrand length, fail-loud on a mismatch.  n_tag_nodes
        # is checked first so every currently-correct per-node integrand keeps
        # its existing weighting even if another length coincides with it.
        if m == n_tag_nodes:
            weighted = jnp.sum(flat * nodal_measure)
        elif boundary_measure.n_facets is not None and m == int(boundary_measure.n_facets):
            weighted = jnp.sum(flat * jnp.asarray(boundary_measure.face_areas))
        elif m == int(boundary_measure.n_nodes):
            gathered = flat[jnp.asarray(boundary_measure.tag_nodes, dtype=jnp.int32)]
            weighted = jnp.sum(gathered * nodal_measure)
        else:
            expected = [f"n_tag_nodes={n_tag_nodes}"]
            if boundary_measure.n_facets is not None:
                expected.append(f"n_facets={int(boundary_measure.n_facets)}")
            expected.append(f"n_nodes={int(boundary_measure.n_nodes)}")
            raise ValueError(
                f"boundary integral {constraint.name!r}: integrand has length {m}, "
                f"which matches none of the boundary's expected lengths "
                f"({', '.join(expected)}). Return one value per boundary node "
                "(per-node lumped measure), one value per facet (per-facet "
                "measure), or one value per mesh node (gathered to the boundary)."
            )
        return jnp.broadcast_to((weighted - tgt).reshape((-1,)), (constraint.dim,))
    if isinstance(constraint, FluxTotalConstraint):
        if flux_total_link is None:
            raise RuntimeError(
                "FluxTotalConstraint requires its resolved per-boundary flux link; "
                "call sites must pass flux_total_link="
            )
        total = jnp.asarray(0.0)
        for (
            flux_fn,
            fi,
            face_measure,
            normals,
            centroids,
            nodal_by_name,
            global_names,
            facet_local,
        ) in flux_total_link:
            if facet_local:
                from .bcs import _facet_local_flux_value  # noqa: PLC0415

                flux_val = _facet_local_flux_value(
                    flux_fn, nodal_by_name, global_names, fi, normals, centroids, t
                )
            else:
                face = _FaceView(nodal_by_name, global_names, fi, normals, centroids)
                flux_val = jnp.asarray(flux_fn(face, t))
            per_facet = jnp.broadcast_to(flux_val.reshape((-1,)), (int(face_measure.shape[0]),))
            total = total + jnp.sum(per_facet * face_measure)
        tgt = jnp.asarray(constraint.target_fn(t_target)) / scale
        return jnp.broadcast_to((total - tgt).reshape((-1,)), (constraint.dim,))
    raise TypeError(f"unknown constraint type {type(constraint).__name__}")
