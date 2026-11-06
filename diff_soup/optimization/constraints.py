"""Constraints on a problem's decision variables, and their lowering to closures.

A numeric problem with constraints holds, beside `"constraints"`
(`constraints(x, args) -> (equality, inequality)`, an equality row satisfied at 0, an
inequality row at or below 0), the structure keys `"constraint_jacobian"` (the rows and
their dense Jacobians, forward or reverse mode by shape), `"lagrangian_hessian"` (the exact
second derivatives of `f + lam_eq . c_E + lam_ineq . c_I`) and `"constraint_hessian"` (those
of the constraint terms alone).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from diff_soup._errors import ConfigurationError
from diff_soup.optimization._binding import Derived, Lowered

_MODES = ("auto", "forward", "reverse")


class _Spec(NamedTuple):
    """The static part of a `Constraints`, which its lowered closures are derived from."""

    equality: Callable | None
    inequality: Callable | None
    n_equality: int
    n_inequality: int
    jacobian: str


class Constraints(eqx.Module):
    """Equality and inequality constraints on a problem's decision variables.

    Each function takes the decision variables in the problem's own terms and `args`. On an
    `InferenceProblem` that is the fitted values by parameter, `values[handle]`; on an
    `OptimizationProblem` it is the flat vector.

    !!! example

        ```python
        constraints = ds.optimization.Constraints(
            inequality=lambda values, args: values[rate] - args["rate_max"], n_inequality=1
        )
        problem = ds.optimization.inference.InferenceProblem(
            experiment, parameters=rate, constraints=constraints, args={"rate_max": 0.3}
        )
        ```

    Args:
        equality: `equality(params, args)` returning rows satisfied at 0, or `None`.
        inequality: `inequality(params, args)` returning rows satisfied at or below 0, or `None`.
        n_equality: The number of rows `equality` returns.
        n_inequality: The number of rows `inequality` returns.
        jacobian: `"auto"` takes forward mode when rows exceed columns or the problem's `hessian_composition` is `"forward-over-forward"`, and reverse mode otherwise; `"forward"` and `"reverse"` force one.
        scales: One positive scale per row, `(equality_scales, inequality_scales)` of shapes `(n_equality,)` and `(n_inequality,)`, either `None` for ones, or `None` for the solver's own scaling.
    """

    equality: Callable | None = eqx.field(static=True, default=None)
    inequality: Callable | None = eqx.field(static=True, default=None)
    n_equality: int = eqx.field(static=True, default=0)
    n_inequality: int = eqx.field(static=True, default=0)
    jacobian: str = eqx.field(static=True, default="auto")
    scales: tuple[Float[Array, " n_eq"], Float[Array, " n_ineq"]] | None = None

    def __post_init__(self) -> None:
        """Give each side of `scales` one scale per row: `None` is ones."""
        if self.scales is None:
            return
        try:
            equality, inequality = self.scales
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "Constraints scales must be an (equality_scales, inequality_scales) pair."
            ) from exc
        self.scales = (
            _side_scales(equality, self.n_equality, "equality"),
            _side_scales(inequality, self.n_inequality, "inequality"),
        )

    def __check_init__(self) -> None:
        """Refuse an unknown Jacobian mode, a negative row count, or a function without rows."""
        if self.jacobian not in _MODES:
            raise ConfigurationError(
                f"Constraints jacobian must be one of {_MODES}; got {self.jacobian!r}."
            )
        for name, function, count in (
            ("equality", self.equality, self.n_equality),
            ("inequality", self.inequality, self.n_inequality),
        ):
            if not isinstance(count, int) or count < 0:
                raise ConfigurationError(f"Constraints n_{name} must be a count; got {count!r}.")
            if function is None and count:
                raise ConfigurationError(
                    f"Constraints declares n_{name}={count} and no {name} function."
                )

    def row_scales(self) -> tuple[Any, Any] | None:
        """The scales, one per row, or `None` for the solver's own scaling."""
        return self.scales

    def lower(
        self,
        objective: Callable | None,
        unflatten: Callable,
        *,
        composition: str = "forward-over-reverse",
    ) -> Lowered:
        """Lower to closures on the flat vector, with Jacobians and Lagrangian Hessians by AD.

        Lowering the same `Constraints` again, with the same `objective` and `unflatten`, gives
        equal closures, so a compiled solver reuses its compile.

        Args:
            objective: `objective(x, args)` on the flat vector, whose Hessian joins the Lagrangian's, or `None` for the Hessian of the constraint terms alone.
            unflatten: The map from the flat vector to the parameter view the functions take.
            composition: `"forward-over-reverse"` takes the Lagrangian Hessians in forward mode over a reverse-mode gradient; `"forward-over-forward"` over a forward-mode one, with the Jacobians forward mode under `jacobian="auto"`.

        Returns:
            The closures `constraints`, `constraints_and_jacobians`, `lagrangian_hessian` and `lagrangian_hessian_vector` by name, each taking `args` last.
        """
        forward = composition == "forward-over-forward"
        jacobian = "forward" if forward and self.jacobian == "auto" else self.jacobian
        spec = _Spec(self.equality, self.inequality, self.n_equality, self.n_inequality, jacobian)
        gradient = jax.jacfwd if forward else jax.grad
        return {
            "constraints": Derived(_rows, spec, unflatten),
            "constraints_and_jacobians": Derived(_rows_and_jacobians, spec, unflatten),
            "lagrangian_hessian": Derived(
                _lagrangian_hessian, spec, unflatten, objective, gradient
            ),
            "lagrangian_hessian_vector": Derived(
                _lagrangian_hessian_vector, spec, unflatten, objective, gradient
            ),
        }


def _side_scales(scales: Any, count: int, name: str) -> Any:
    """One side's scales as a `(count,)` array; ones for `None`."""
    dtype = jnp.result_type(float)
    if scales is None:
        return jnp.ones((count,), dtype)
    if jnp.shape(scales) != (count,):
        raise ConfigurationError(
            f"Constraints {name} scales must hold one scale per row, shape ({count},); got "
            f"shape {jnp.shape(scales)}."
        )
    return jnp.asarray(scales, dtype)


def _side(function, count, params, args, dtype, name):
    """One side's rows, 1-D and of its declared length; empty without a function."""
    if function is None:
        return jnp.zeros((0,), dtype)
    values = jnp.reshape(jnp.asarray(function(params, args), dtype), (-1,))
    if values.shape[0] != count:
        raise ConfigurationError(
            f"Constraints {name} returned {values.shape[0]} rows; n_{name} declares {count}."
        )
    return values


def _rows(spec: _Spec, unflatten: Callable, x: Any, args: Any) -> tuple[Any, Any]:
    """`(equality, inequality)` on the flat vector."""
    params = unflatten(x)
    dtype = jnp.result_type(x)
    return (
        _side(spec.equality, spec.n_equality, params, args, dtype, "equality"),
        _side(spec.inequality, spec.n_inequality, params, args, dtype, "inequality"),
    )


def _rows_and_jacobians(spec: _Spec, unflatten: Callable, x: Any, args: Any) -> tuple:
    """`(equality, inequality, J_eq, J_ineq)`, the Jacobians in the mode `spec.jacobian` picks."""
    rows = spec.n_equality + spec.n_inequality
    forward = spec.jacobian == "forward" or (spec.jacobian == "auto" and rows > jnp.shape(x)[-1])
    differentiate = jax.jacfwd if forward else jax.jacrev

    def twice(y):
        values = _rows(spec, unflatten, y, args)
        return values, values

    jacobians, values = differentiate(twice, has_aux=True)(x)
    return (*values, *jacobians)


def _lagrangian(spec, unflatten, objective, x, lam_eq, lam_ineq, args):
    """`f + lam_eq . c_E + lam_ineq . c_I`, or the constraint terms alone without `objective`."""
    equality, inequality = _rows(spec, unflatten, x, args)
    weighted = lam_eq @ equality + lam_ineq @ inequality
    return weighted if objective is None else objective(x, args) + weighted


def _lagrangian_hessian(spec, unflatten, objective, gradient, x, lam_eq, lam_ineq, args):
    """The Lagrangian's Hessian in `x`, forward over `gradient` (`jax.grad` or `jax.jacfwd`)."""

    def lagrangian(y):
        return _lagrangian(spec, unflatten, objective, y, lam_eq, lam_ineq, args)

    return jax.jacfwd(gradient(lagrangian))(x)


def _lagrangian_hessian_vector(spec, unflatten, objective, gradient, x, lam_eq, lam_ineq, v, args):
    """The Lagrangian's Hessian in `x` times `v`, forward over `gradient`."""

    def lagrangian(y):
        return _lagrangian(spec, unflatten, objective, y, lam_eq, lam_ineq, args)

    return jax.jvp(gradient(lagrangian), (x,), (v,))[1]


__all__ = ["Constraints"]
