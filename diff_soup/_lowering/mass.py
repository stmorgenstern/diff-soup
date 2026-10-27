"""Assembled diagonal mass for the lowered problem (component + Family blocks).

The ``mass`` kind on each [Variable][diff_soup.problem._ir.variable.Variable]
selects how the NDF integrator treats that field:

* ``mass="differential"`` → ``M du/dt = f`` (time-integrated row).  Its
  mass-diagonal slice is the (lumped / consistent) mass weight, which must be
  strictly positive — a differential field with an all-zero mass slice is
  degenerate (really algebraic, ``0 = f``) and is rejected here.
* ``mass="algebraic"`` → ``0 = f`` (per-point constraint).  Its mass-diagonal
  slice is exactly zero.

Mass-weight convention is discretisation-specific:

* FEM — the residual is the weak-form integral ``∫ φ_i f`` (not
  volume-normalised), so the row carries the consistent / lumped mass from
  ``lumped_mass()``.
* FVM (``Grid1D`` / staggered / spherical) — the kernels already divide the
  net face flux by the cell volume (``I du/dt = f``), so the mass diagonal must
  be ``ones``, NOT the cell / shell volumes.  Weighting an already-volume-
  normalised FVM RHS by cell volumes double-counts the metric and produces a
  resolution-dependent time clock.  For a family's unit-ball member grid
  (``spherical_grid_1d``) the member mass is ``ones``, not shell volumes.

The assembled differential mass is divided by ``eqset.scales.time_ref``
(internal-time convention ``t_phys = time_ref * t_int``): a differential row
``M du/dt_phys = f`` becomes ``(M / time_ref) du/dt_int = f`` in internal
time.  Algebraic rows have ``M == 0`` so this only scales the differential rows;
``time_ref`` defaults to 1.0 (identity) so non-time-scaled models are
unaffected.

A residual-row scale ``S`` (``field(residual_scale=S)`` /
``spec.equation_scale(field, S)``) divides the differential mass slice for that
field as well as its residual rows, on component and family blocks alike:
multiplying the row ``M du/dt = f`` through by ``1/S`` gives
``(M / S) du/dt = f / S``, so the solution is unchanged and only the row's
Jacobian entries shrink.  Dividing the residual alone would instead integrate
``M du/dt = f / S`` — a different trajectory.  Algebraic rows have ``M == 0``
and stay exactly zero under the scaling.

This module lives under ``diff_soup/_lowering``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diff_soup.problem._ir.equation_set import EquationSet

from .layout import _BlockLayout
from .operator_resolve import resolve_operator


def _component_mass_weight(domain, operator_factory=None) -> jax.Array:
    """Per-node mass weight for one domain (FEM lumped / FVM ones).

    ``operator_factory`` is the owning component/family operator factory (the
    built-in bank when ``None``); the resolved bank supplies FEM lumped mass when
    the domain does not expose ``lumped_mass`` itself.
    """
    is_fem = getattr(domain, "discretization_type", None) == "fem"
    if is_fem:
        if hasattr(domain, "lumped_mass"):
            return jnp.asarray(domain.lumped_mass())
        return resolve_operator(domain, operator_factory).lumped_mass()
    n_nodes = getattr(domain, "num_nodes", None)
    if n_nodes is None:
        # Fall back to the first member field's size at the caller's discretion;
        # a domain without ``num_nodes`` and without FEM lumped mass is unusual.
        raise ValueError(
            "mass diagonal: non-FEM domain lacks ``num_nodes`` for an ones-weighted FVM mass"
        )
    return jnp.ones(int(n_nodes))


def _mass_diagonal(eqset: EquationSet, layout: _BlockLayout) -> jax.Array:
    """Assemble the diagonal mass per component / variable + per family member.

    The ``layout`` argument is accepted for signature parity with the full-path
    builder (the component / family slab order is pinned by ``_layout``); the
    assembly order here matches it (components first, then families in
    ``EquationSet.families`` order) so the diagonal lines up with the state
    vector slot-for-slot.

    A differential field's slice is divided by its residual-row scale when one
    is set — ``EquationSet.residual_scale[(component, variable)]`` for a
    component field, ``Family.residual_scale[variable]`` for a member field —
    matching the division already applied to that row's residual.

    Raises:
        ValueError: when a ``differential`` variable's assembled mass-diagonal
            slice is all zero (the loud-error replacement for a silently-wrong
            mass kind).
    """
    del layout  # slab order is structural (components then families).
    # ``(component_name, var_name) -> factor``: the same keys the residual rows
    # are divided by, so both sides of a scaled row move together.
    comp_res_scale = {k: float(v) for k, v in dict(getattr(eqset, "residual_scale", {})).items()}
    parts = []
    for comp in eqset.components:
        weight = _component_mass_weight(comp.domain, comp.operator_factory)
        for var in comp.variables:
            if var.mass == "differential":
                if var.size == comp.domain.num_nodes:
                    slice_ = weight
                else:
                    slice_ = jnp.broadcast_to(weight[..., None], var.shape).reshape(-1)
                scale = comp_res_scale.get((comp.name, var.name))
                if scale is not None:
                    slice_ = slice_ / scale
                if bool(jnp.all(slice_ == 0)):
                    raise ValueError(
                        f"variable {var.name!r} in component {comp.name!r} is "
                        f"declared mass='differential' but its assembled mass "
                        f"diagonal is all zero — this is degenerate (really "
                        f"algebraic). Declare it mass='algebraic' if you mean "
                        f"0=f, or supply a domain with positive (lumped/"
                        f"consistent) mass."
                    )
                parts.append(slice_)
            else:
                parts.append(jnp.zeros((var.size,), dtype=weight.dtype))

    # Family blocks live STRICTLY AFTER all components, in ``EquationSet.
    # families`` order, matching ``_layout``.  The member slab is laid out
    # MEMBER-MAJOR with each member's fields blocked in declaration order
    # (``[c(n_r), mu(n_r)]`` per member, ``var_local_off``-banded — see
    # ``layout._layout`` / ``_FamilyResidualBlock``).  The mass diagonal MUST be
    # assembled in that same order: build ONE member's mass slab (each field's
    # weight if differential, zeros if algebraic) and tile it ``count`` times.
    # A split Cahn-Hilliard member carries a differential ``c`` AND an algebraic
    # ``mu``, so a family CAN have algebraic member rows — they interleave with
    # the differential rows per member; appending a separate field-major
    # ``[all c | all mu]`` block would misalign the mass with the member-major
    # state for any family with more than one member field.  The member mass
    # weight is discretisation-specific (FVM member grid → ``ones``, NOT shell
    # volumes); each field is divided by the member residual_scale (a scalar)
    # if set.
    for fam in getattr(eqset, "families", ()):
        member_dom = fam.domain
        member_is_fem = getattr(member_dom, "discretization_type", None) == "fem"
        if member_is_fem:
            if hasattr(member_dom, "lumped_mass"):
                w_member = jnp.asarray(member_dom.lumped_mass())
            else:
                w_member = resolve_operator(member_dom, fam.operator_factory).lumped_mass()
        else:
            n_r = getattr(member_dom, "num_nodes", None)
            if n_r is None:
                n_r = fam.member_variables[0].size
            w_member = jnp.ones(int(n_r))
        res_scale = fam.residual_scale or {}
        member_mass_parts = []
        for var in fam.member_variables:
            if var.mass == "differential":
                if var.size == w_member.shape[0]:
                    member_slice = w_member
                elif int(var.size) == 1:
                    raise ValueError(
                        f"member variable {var.name!r} in family {fam.name!r} is a "
                        "per-member scalar declared mass='differential', but a "
                        "member domain supplies one mass weight per node, not a "
                        "scalar mass row. Declare the scalar mass='algebraic' "
                        "(0 = f, closed by its member equation) or make it a "
                        "profile field on the member domain."
                    )
                else:
                    member_slice = jnp.broadcast_to(w_member[..., None], var.shape).reshape(-1)
                scale = res_scale.get(var.name)
                if scale is not None:
                    member_slice = member_slice / scale
                if bool(jnp.all(member_slice == 0)):
                    raise ValueError(
                        f"member variable {var.name!r} in family {fam.name!r} "
                        f"is declared mass='differential' but its assembled "
                        f"mass diagonal is all zero — this is degenerate "
                        f"(really algebraic). Declare it mass='algebraic' if "
                        f"you mean 0=f, or supply a member domain with positive "
                        f"(lumped/consistent) mass."
                    )
                member_mass_parts.append(member_slice)
            else:
                member_mass_parts.append(jnp.zeros((var.size,), dtype=w_member.dtype))
        # One member's blocked mass slab, tiled MEMBER-MAJOR across the population.
        member_mass = jnp.concatenate(member_mass_parts)
        parts.append(jnp.tile(member_mass, fam.count))

    scales = getattr(eqset, "scales", None)
    time_ref = float(getattr(scales, "time_ref", 1.0)) if scales is not None else 1.0
    return jnp.concatenate(parts) / time_ref
