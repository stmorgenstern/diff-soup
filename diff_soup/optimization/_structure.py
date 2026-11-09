"""The structure a numeric problem names: its keys, the closures each key needs, the refusals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from diff_soup._errors import ConfigurationError

if TYPE_CHECKING:
    from diff_soup.optimization._numeric import OptimizationProblem


#: The structure an [OptimizationProblem][diff_soup.optimization.OptimizationProblem]
#: can provide, each key a set of closures it then holds, every one taking ``args`` last:
#:
#: - ``"value_gradient"``: ``value(x, args)`` and ``value_and_gradient(x, args)``.
#: - ``"hessian"``: ``value_gradient_hessian(x, args)`` and ``hessian_vector(x, v, args)``.
#: - ``"gauss_newton"``: ``gauss_newton(x, args) -> (cost, gradient, matrix, failed)``.
#: - ``"residual_jacobian"``: ``residual(x, args)`` and ``residual_and_jacobian(x, args)``
#:   (a [LeastSquaresProblem][diff_soup.optimization.LeastSquaresProblem]).
#: - ``"constraints"``: ``constraints(x, args) -> (equality, inequality)`` returns at least
#:   one row; an equality row holds at 0, an inequality row at or below 0.
#: - ``"constraint_jacobian"``: ``constraints_and_jacobians(x, args) -> (eq, ineq, J_eq,
#:   J_ineq)``, the dense Jacobians.
#: - ``"lagrangian_hessian"``: ``lagrangian_hessian(x, lam_eq, lam_ineq, args)`` and
#:   ``lagrangian_hessian_vector(x, lam_eq, lam_ineq, v, args)``, the exact second
#:   derivatives of ``f + lam_eq . c_E + lam_ineq . c_I``.
#: - ``"constraint_hessian"``: ``constraint_hessian(x, lam_eq, lam_ineq, args)``, the
#:   second derivatives of ``lam_eq . c_E + lam_ineq . c_I`` alone; zeros without rows.
#: - ``"batched_lanes"``: the objective is evaluated by the lanes of a lowered
#:   study; ``problem.lanes`` is that study and ``problem.joint_view()`` its
#:   joint problem with the level structure.
STRUCTURE_KEYS = frozenset(
    {
        "value_gradient",
        "hessian",
        "gauss_newton",
        "residual_jacobian",
        "constraints",
        "constraint_jacobian",
        "lagrangian_hessian",
        "constraint_hessian",
        "batched_lanes",
    }
)

#: The structure key each closure belongs to.
_KEY_OF = {
    "value": "value_gradient",
    "value_and_gradient": "value_gradient",
    "value_gradient_hessian": "hessian",
    "hessian_vector": "hessian",
    "gauss_newton": "gauss_newton",
    "residual": "residual_jacobian",
    "residual_and_jacobian": "residual_jacobian",
    "is_valid": None,
    "constraints": "constraints",
    "constraints_and_jacobians": "constraint_jacobian",
    "lagrangian_hessian": "lagrangian_hessian",
    "lagrangian_hessian_vector": "lagrangian_hessian",
    "constraint_hessian": "constraint_hessian",
    "evaluate": "value_gradient",
    "evaluate_with_derivatives": "constraint_jacobian",
}

#: The closures every problem provides, with empty rows when it has no constraints.
_ALWAYS = frozenset({"is_valid", "constraints", "constraints_and_jacobians"})

#: The closures `closure(name)` derives on read when the problem was not given them.
_ON_READ = frozenset({"is_valid", "evaluate", "evaluate_with_derivatives"})

#: The stored fields each structure key needs.
_NEEDS = {
    "value_gradient": ("_value", "_value_and_gradient"),
    "hessian": ("_value_gradient_hessian", "_hessian_vector"),
    "gauss_newton": ("_gauss_newton",),
    "residual_jacobian": ("_residual", "_residual_and_jacobian"),
    "constraints": ("_constraints",),
    "constraint_jacobian": ("_constraints_and_jacobians",),
    "lagrangian_hessian": ("_lagrangian_hessian", "_lagrangian_hessian_vector"),
    "constraint_hessian": ("_constraint_hessian",),
    "batched_lanes": ("_joint_view",),
}


def _missing(problem: OptimizationProblem, key: str, method: str) -> ConfigurationError:
    """The error for `method` on a problem whose structure lacks `key`."""
    provided = ", ".join(sorted(problem.structure)) or "nothing"
    return ConfigurationError(
        f"{type(problem).__name__}.{method} needs the {key!r} structure, and this problem "
        f"provides {provided}; build the problem with that structure or use a solver whose "
        "requires the problem meets."
    )


def _check_scales(scales: Any, rows: tuple[int, int]) -> None:
    """Refuse constraint scales that are not one per row, `(n_equality,)` and `(n_inequality,)`."""
    try:
        equality, inequality = scales
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "OptimizationProblem constraint_scales must be an (equality, inequality) pair."
        ) from exc
    shapes = (jnp.shape(equality), jnp.shape(inequality))
    if shapes != ((rows[0],), (rows[1],)):
        raise ConfigurationError(
            f"OptimizationProblem constraint_scales must be one scale per row, shapes "
            f"(({rows[0]},), ({rows[1]},)); got {shapes}."
        )


__all__ = ["STRUCTURE_KEYS"]
