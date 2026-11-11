"""The residual of a spec without a family layer, and its Jacobian blocks."""

from __future__ import annotations

import functools

import jax.numpy as jnp

from diff_soup.domains.operators.dispatch import finalize_residual
from diff_soup.linear import SparseTopology
from diff_soup.problem._ad.sparse_jac import SparseJacobian
from diff_soup.problem._ir.equation import MultiEquation
from diff_soup.problem._ir.equation_set import EquationSet
from diff_soup.problem._ir.lower import _apply_boundary_cast
from diff_soup.problem._lowering.provider_context import (
    _assembly_dtype,
    _check_multi_equation_return,
    _Ctx,
    _geometry_scale_value,
)

from .operator_resolve import resolve_operator


def _build_residual_fn(eqset: EquationSet, offset_of: dict, total: int, compute_dtype):
    """Build the static same-domain residual kernel ``(f, g)``.

    Takes the IR-node ``equations`` tuple as a runtime arg so ``eqx.Module``
    closure leaves reach AD (the hybrid-cache contract).  Per equation:
    evaluates ``rhs_fn(var_view, ctx, t)`` with ``ctx.ops`` the implicit-mass
    the weak-form residual view, runs ``finalize_residual`` on leftover-nodal terms,
    applies the residual-row scaling verb (``residual_scale``; identity default),
    casts up to ``compute_dtype``, and scatters into ``f``.  ``g`` (constraint
    residual) is empty on the slim path.
    """
    comp = eqset.components[0]
    unit_dom = comp.domain
    unit_tag_ops = resolve_operator(unit_dom, comp.operator_factory).for_residual()
    scaled = getattr(comp, "geometry_scale", None) is not None
    per_vars = tuple(comp.variables)
    residual_scale = {k: float(v) for k, v in dict(getattr(eqset, "residual_scale", {})).items()}

    def physics_residual_fn(
        equations,
        dirichlet_value_fns,  # noqa: ARG001 — slim path: no BCs.
        flux_callables,  # noqa: ARG001
        robin_callables,  # noqa: ARG001
        traction_callables,  # noqa: ARG001
        constraints,  # noqa: ARG001 — slim path: no constraints.
        couplings,  # noqa: ARG001 — slim path: no couplings.
        u,
        lam,  # noqa: ARG001 — slim path: no globals.
        t,
        families=(),  # noqa: ARG001 — slim path: no families.
        controls=(),  # noqa: ARG001 — slim path: controls force the full path.
        control=None,  # noqa: ARG001 — slim path: no control-driven rows.
    ):
        view = {}
        o = 0
        for v in per_vars:
            view[v.name] = u[o : o + v.size].reshape(v.shape)
            o += v.size
        var_view = {v: view[v.name] for v in per_vars}
        dom, tag_ops = unit_dom, unit_tag_ops
        if scaled:
            # The grid is scaled by a declared parameter: evaluate on the grid
            # at the parameter's current value.
            from .bcs import _boundary_context

            length = _geometry_scale_value(_boundary_context(equations), comp)
            dom = unit_dom.scaled(length)
            tag_ops = resolve_operator(dom, comp.operator_factory).for_residual()
        ctx = _Ctx(tag_ops)
        f = jnp.zeros((total,), dtype=u.dtype)
        asm_dtype = _assembly_dtype(u.dtype, compute_dtype)
        for eq in equations:
            r = eq.rhs_fn(var_view, ctx, t)
            if isinstance(eq, MultiEquation):
                _check_multi_equation_return(eq, r)
                for v in eq.vars:
                    arr = jnp.asarray(
                        _apply_boundary_cast(finalize_residual(dom, r[v.name]), asm_dtype)
                    ).reshape(-1)
                    if arr.size != v.size:
                        raise ValueError(
                            f"equation residual for variable {v.name!r} has size "
                            f"{int(arr.size)} but the field has {int(v.size)} "
                            f"DOFs; the rhs_fn must return one value per DOF "
                            f"(shape {v.shape})."
                        )
                    _rs = residual_scale.get((comp.name, v.name))
                    if _rs is not None:
                        arr = arr / _rs
                    ofs = offset_of[(comp.name, v.name)]
                    f = f.at[ofs : ofs + v.size].add(arr)
            else:
                arr = jnp.asarray(
                    _apply_boundary_cast(finalize_residual(dom, r), asm_dtype)
                ).reshape(-1)
                if arr.size != eq.var.size:
                    raise ValueError(
                        f"equation residual for variable {eq.var.name!r} has size "
                        f"{int(arr.size)} but the field has {int(eq.var.size)} "
                        f"DOFs; the rhs_fn must return one value per DOF "
                        f"(shape {eq.var.shape})."
                    )
                _rs = residual_scale.get((comp.name, eq.var.name))
                if _rs is not None:
                    arr = arr / _rs
                ofs = offset_of[(comp.name, eq.var.name)]
                f = f.at[ofs : ofs + eq.var.size].add(arr)
        return f, jnp.zeros((0,), dtype=u.dtype)

    return physics_residual_fn


def _lazy_coloring(topo: SparseTopology):
    """The coloring of ``topo``, built at the kernel's first call.

    The solvers color the plan's pattern instead, so a build colors this one only
    when ``physics_jacobian_blocks`` is called.
    """
    return functools.cache(lambda: SparseJacobian.from_topology(topo))


def _build_jac_blocks_fn(residual_fn, topo: SparseTopology, total: int):
    """Build the static same-domain KKT-Jacobian kernel ``(Avals, B, C, D, topo)``.

    ``Avals`` = sparse ``df/du`` via CPR-coloured ``SparseJacobian.evaluate``
    (AD of ``residual_fn``).  On the slim path there are no constraints, so
    ``B``/``C``/``D`` are empty.
    """
    sparse_jac = _lazy_coloring(topo)

    def physics_jac_blocks_fn(
        equations,
        dirichlet_value_fns,
        flux_callables,
        robin_callables,
        traction_callables,
        constraints,
        couplings,
        u,
        lam,
        t,
        families=(),
        controls=(),
        control=None,
    ):
        def f_of_u(u_):
            f, _g = residual_fn(
                equations,
                dirichlet_value_fns,
                flux_callables,
                robin_callables,
                traction_callables,
                constraints,
                couplings,
                u_,
                lam,
                t,
                families,
                controls,
                control,
            )
            return f

        Avals = sparse_jac().evaluate(f_of_u, u)
        return (
            Avals,
            jnp.zeros((total, 0), dtype=u.dtype),
            jnp.zeros((0, total), dtype=u.dtype),
            jnp.zeros((0, 0), dtype=u.dtype),
            topo,
        )

    return physics_jac_blocks_fn
