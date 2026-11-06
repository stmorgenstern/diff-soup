"""The numeric problem a solver reads: closures of one flat vector and `args`."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike, Shaped

from diff_soup._errors import ConfigurationError
from diff_soup.optimization import _derived
from diff_soup.optimization._binding import Bound, Derived, hold, ignoring_args
from diff_soup.optimization._penalty import FOLDED, penalized
from diff_soup.optimization._structure import (
    _ALWAYS,
    _KEY_OF,
    _NEEDS,
    _ON_READ,
    STRUCTURE_KEYS,
    _check_scales,
    _missing,
)

if TYPE_CHECKING:
    from diff_soup.optimization.constraints import Constraints


def _is_none(leaf: Any) -> bool:
    """Whether a pytree node is `None`, which `eqx.tree_at` must see as a leaf to replace."""
    return leaf is None


class OptimizationProblem(eqx.Module):
    """The objective an optimizer minimizes, with its starting point, bounds and derivatives.

    `ds.optimization.optsolve` builds one from the problem it is given and passes it to
    `OptSolver.solve`. `structure` names the functions it provides, and asking for any
    other raises `ConfigurationError`.

    Every closure takes `args` last: `value(x, args)`, `constraints(x, args)`. A solver that
    compiles its loop takes the raw closure from `closure(name)` and passes `args` itself,
    so problems that differ only in `args` share its compile. The properties return the
    closures bound to the problem's own `args`, so `problem.value(x)` works.

    A `regularization` is a penalty on the objective. `closure(name)` folds it into the
    objective's closures, as a compiled solver minimizes it; the properties leave it out,
    and a host-loop solver adds it itself.

    Attributes:
        x0: The starting point, a 1-D array.
        structure: The names of the functions this problem provides, such as `"value_gradient"`, `"hessian"` or `"gauss_newton"`.
        bounds: The `(lower, upper)` arrays aligned with `x0`, or `None` when unbounded.
        args: The pytree every closure takes last: limits, initial states, outer parameters. `None` for a problem whose closures read nothing else.
        n_equality: The number of equality rows `constraints` returns.
        n_inequality: The number of inequality rows `constraints` returns.
        constraint_scales: `(equality_scales, inequality_scales)`, one positive scale per row, `(n_equality,)` and `(n_inequality,)`, or `None` for the solver's own scaling.
        objective_scale: A positive scale of the objective, or `None` for the solver's own scaling.
    """

    x0: Shaped[ArrayLike, " n"]
    structure: frozenset[str] = eqx.field(static=True)
    bounds: tuple[ArrayLike, ArrayLike] | None = None
    args: Any = None
    n_equality: int = eqx.field(static=True, default=0)
    n_inequality: int = eqx.field(static=True, default=0)
    constraint_scales: tuple[ArrayLike, ArrayLike] | None = None
    objective_scale: Any = None
    # The fitted model's reconstruction and the regularization can hold arrays; they are
    # held by identity, so the problem's static data stays hashable and array-free.
    _unflatten: Any = eqx.field(static=True, default=None, converter=hold)
    _static: Any = eqx.field(static=True, default=None, converter=hold)
    _regularization: Any = eqx.field(static=True, default=None, converter=hold)
    lanes: Any = eqx.field(static=True, default=None)
    ad: str | None = eqx.field(static=True, default=None)
    hessian_composition: str | None = eqx.field(static=True, default=None)
    n_segments: int = eqx.field(static=True, default=1)
    reduced_gauss_newton: bool = eqx.field(static=True, default=False)
    _value: Callable | None = eqx.field(static=True, default=None)
    _value_and_gradient: Callable | None = eqx.field(static=True, default=None)
    _value_gradient_hessian: Callable | None = eqx.field(static=True, default=None)
    _hessian_vector: Callable | None = eqx.field(static=True, default=None)
    _gauss_newton: Callable | None = eqx.field(static=True, default=None)
    _is_valid: Callable | None = eqx.field(static=True, default=None)
    _constraints: Callable | None = eqx.field(static=True, default=None)
    _constraints_and_jacobians: Callable | None = eqx.field(static=True, default=None)
    _lagrangian_hessian: Callable | None = eqx.field(static=True, default=None)
    _lagrangian_hessian_vector: Callable | None = eqx.field(static=True, default=None)
    _constraint_hessian: Callable | None = eqx.field(static=True, default=None)
    _evaluate: Callable | None = eqx.field(static=True, default=None)
    _evaluate_with_derivatives: Callable | None = eqx.field(static=True, default=None)
    _joint_view: Callable | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        """Store what follows from what is given; a problem without rows gets empty ones.

        A value from a value and gradient, a Hessian-vector product from a Hessian, the
        constraint rows from the rows and Jacobians, and a Lagrangian Hessian-vector product
        from the Lagrangian Hessian.
        """
        if self._value is None and self._value_and_gradient is not None:
            self._value = Derived(_derived.first, self._value_and_gradient)
        if self._hessian_vector is None and self._value_gradient_hessian is not None:
            self._hessian_vector = Derived(_derived.hessian_times, self._value_gradient_hessian)
        if self._constraints is None and self._constraints_and_jacobians is not None:
            self._constraints = Derived(_derived.rows, self._constraints_and_jacobians)
        if self.n_equality == 0 and self.n_inequality == 0:
            # No rows: the constraint closures are empty, and the Lagrangian is the objective.
            if self._constraints is None:
                self._constraints = _derived.no_constraints
            if self._constraints_and_jacobians is None:
                self._constraints_and_jacobians = _derived.no_constraint_jacobians
            if self._lagrangian_hessian is None and self._value_gradient_hessian is not None:
                self._lagrangian_hessian = Derived(
                    _derived.objective_hessian, self._value_gradient_hessian
                )
            if self._constraint_hessian is None:
                self._constraint_hessian = _derived.no_constraint_hessian
            implied = {"constraint_jacobian", "constraint_hessian"}
            if self._lagrangian_hessian is not None:
                implied.add("lagrangian_hessian")
            self.structure = frozenset(self.structure) | implied
        if self._lagrangian_hessian_vector is None and self._lagrangian_hessian is not None:
            self._lagrangian_hessian_vector = Derived(
                _derived.lagrangian_times, self._lagrangian_hessian
            )

    def __check_init__(self) -> None:
        """Refuse a structure key outside `STRUCTURE_KEYS`, or a key without its closures."""
        unknown = set(self.structure) - STRUCTURE_KEYS
        if unknown:
            raise ConfigurationError(
                f"OptimizationProblem structure names {sorted(unknown)}, which are not "
                f"structure keys; the keys are {sorted(STRUCTURE_KEYS)}."
            )
        for key in self.structure:
            absent = [name for name in _NEEDS[key] if getattr(self, name, None) is None]
            if absent:
                raise ConfigurationError(
                    f"OptimizationProblem structure names {key!r}, which needs the "
                    f"closures {absent}; pass them or leave {key!r} out of structure."
                )
        rows = (self.n_equality, self.n_inequality)
        if any(not isinstance(count, int) or count < 0 for count in rows):
            raise ConfigurationError(
                f"OptimizationProblem n_equality and n_inequality must be counts; got {rows}."
            )
        if self._constraints is None:
            raise ConfigurationError(
                f"OptimizationProblem declares {rows[0]} equality and {rows[1]} inequality "
                "rows and no closure returning them; pass constraints or "
                "constraints_and_jacobians."
            )
        if self.constraint_scales is not None:
            _check_scales(self.constraint_scales, rows)

    def closure(self, name: str) -> Callable:
        """The raw closure `name`, which takes `args` last, for a solver that passes `args` itself.

        It supplies what the problem was not given (the trial predicate, the evaluations of
        the objective with the rows) and folds the `regularization`'s penalty into the
        objective's closures: the value, the gradient, the Gauss-Newton statistics and the
        Hessians. The properties read the same closures without the penalty.

        Args:
            name: A property of this problem, such as `"value_and_gradient"` or `"constraints_and_jacobians"`.

        Returns:
            The closure, unbound.

        Raises:
            ConfigurationError: If the problem's structure does not provide `name`.
        """
        return penalized(name, self._unpenalized(name), self._regularization)

    def _unpenalized(self, name: str) -> Callable:
        """The closure `name` without the penalty, derived on read where it is not stored."""
        if name not in _KEY_OF:
            raise ConfigurationError(
                f"OptimizationProblem has no closure {name!r}; the closures are {sorted(_KEY_OF)}."
            )
        key = _KEY_OF[name]
        stored = getattr(self, "_" + name, None)
        if stored is None and name in _ON_READ:
            stored = self._derive(name)
        if stored is None or (key not in self.structure and name not in _ALWAYS):
            raise _missing(self, key, name)
        return stored

    def _derive(self, name: str) -> Callable | None:
        """The closure `name` of `_ON_READ` from the stored ones, or `None` without them."""
        if name == "is_valid":
            return _derived.valid_everywhere
        if name == "evaluate" and self._value is not None:
            return Derived(_derived.evaluate, self._value, self._constraints)
        first_order = (self._value_and_gradient, self._constraints_and_jacobians)
        if name == "evaluate_with_derivatives" and None not in first_order:
            return Derived(_derived.evaluate_with_derivatives, *first_order)
        return None

    def _bound(self, name: str) -> Callable:
        """The closure `name` bound to this problem's `args`, without the penalty."""
        return Bound(self._unpenalized(name), self.args)

    def with_args(self, args: Any) -> OptimizationProblem:
        """This problem with `args` replaced, as a lane of a batch holds it."""
        return eqx.tree_at(lambda problem: problem.args, self, args, is_leaf=_is_none)

    def with_constraints(
        self, constraints: Constraints | None, *, unflatten: Callable | None = None
    ) -> OptimizationProblem:
        """This problem under `constraints`, lowered to closures with AD Jacobians and Lagrangian Hessians.

        Args:
            constraints: The [Constraints][diff_soup.optimization.Constraints], or `None` for this problem unchanged.
            unflatten: The map from the flat vector to the view the constraint functions take; the flat vector itself when `None`.

        Returns:
            A problem whose structure adds `"constraints"`, `"constraint_jacobian"`, `"lagrangian_hessian"` and `"constraint_hessian"`.
        """
        if constraints is None:
            return self
        if self.n_equality or self.n_inequality:
            raise ConfigurationError(
                "OptimizationProblem.with_constraints: this problem has constraints already."
            )
        if "value_gradient" not in self.structure:
            raise _missing(self, "value_gradient", "with_constraints")
        # The constraint rows are differentiated twice alone; the objective's Hessian is its own
        # exact one, or forward over reverse of its value.
        lowered = constraints.lower(None, _derived.identity if unflatten is None else unflatten)
        if "hessian" in self.structure:
            objective_hessian = self._value_gradient_hessian
            objective_product = self._hessian_vector
        else:
            objective_hessian = Derived(_derived.value_gradient_hessian, self._value)
            objective_product = Derived(_derived.hessian_vector, self._value)
        keys = {"constraint_jacobian", "lagrangian_hessian", "constraint_hessian"}
        if constraints.n_equality or constraints.n_inequality:
            keys.add("constraints")
        return dataclasses.replace(
            self,
            structure=frozenset(self.structure) | keys,
            n_equality=constraints.n_equality,
            n_inequality=constraints.n_inequality,
            constraint_scales=constraints.row_scales(),
            _constraints=lowered["constraints"],
            _constraints_and_jacobians=lowered["constraints_and_jacobians"],
            _lagrangian_hessian=Derived(
                _derived.plus_objective_hessian, lowered["lagrangian_hessian"], objective_hessian
            ),
            _lagrangian_hessian_vector=Derived(
                _derived.plus_objective_hessian_vector,
                lowered["lagrangian_hessian_vector"],
                objective_product,
            ),
            _constraint_hessian=lowered["lagrangian_hessian"],
            # A given evaluation of the objective with the rows does not cover the new rows.
            _evaluate=None,
            _evaluate_with_derivatives=None,
        )

    def _penalty_folded(self) -> OptimizationProblem:
        """This problem with the regularization's penalty in its closures and no regularization.

        Every closure, read through `closure(name)` or a property, is then the regularized
        objective's; this problem itself when it has no regularization.
        """
        if self.regularization is None:
            return self
        folded = {
            "_" + name: self.closure(name)
            for name in FOLDED
            if getattr(self, "_" + name) is not None
        }
        return dataclasses.replace(self, _regularization=None, **folded)

    @property
    def n(self) -> int:
        """The number of decision variables, static."""
        return int(jnp.shape(self.x0)[-1])

    @property
    def value(self) -> Callable:
        """The objective, a function `value(x)` returning a scalar."""
        return self._bound("value")

    @property
    def value_and_gradient(self) -> Callable:
        """A function `value_and_gradient(x)` returning the objective and its gradient."""
        return self._bound("value_and_gradient")

    @property
    def value_gradient_hessian(self) -> Callable:
        """A function `value_gradient_hessian(x)` returning the objective, its gradient and its Hessian (`"hessian"`)."""
        return self._bound("value_gradient_hessian")

    @property
    def hessian_vector(self) -> Callable:
        """A function `hessian_vector(x, v)` returning the objective's Hessian at `x` times `v` (`"hessian"`)."""
        return self._bound("hessian_vector")

    @property
    def gauss_newton(self) -> Callable:
        """A function `gauss_newton(x)` returning the objective, its gradient, the Gauss-Newton matrix and a failure count (`"gauss_newton"`)."""
        return self._bound("gauss_newton")

    @property
    def is_valid(self) -> Callable:
        """A function `is_valid(x)` returning whether `x` is a trial point the objective may be evaluated at; true everywhere without a predicate."""
        return self._bound("is_valid")

    @property
    def constraints(self) -> Callable:
        """A function `constraints(x)` returning the equality and inequality rows, empty without constraints."""
        return self._bound("constraints")

    @property
    def constraints_and_jacobians(self) -> Callable:
        """A function `constraints_and_jacobians(x)` returning the rows and their dense Jacobians, `(0, n)` without constraints."""
        return self._bound("constraints_and_jacobians")

    @property
    def evaluate(self) -> Callable:
        """A function `evaluate(x)` returning the objective and the rows from one evaluation."""
        return self._bound("evaluate")

    @property
    def evaluate_with_derivatives(self) -> Callable:
        """A function `evaluate_with_derivatives(x)` returning the objective, the rows and their first derivatives from one evaluation."""
        return self._bound("evaluate_with_derivatives")

    @property
    def lagrangian_hessian(self) -> Callable:
        """A function `lagrangian_hessian(x, lam_eq, lam_ineq)` returning the Hessian of `f + lam_eq . c_E + lam_ineq . c_I` (`"lagrangian_hessian"`)."""
        return self._bound("lagrangian_hessian")

    @property
    def lagrangian_hessian_vector(self) -> Callable:
        """A function `lagrangian_hessian_vector(x, lam_eq, lam_ineq, v)` returning the Lagrangian Hessian times `v` (`"lagrangian_hessian"`)."""
        return self._bound("lagrangian_hessian_vector")

    def joint_view(self) -> Any:
        """The joint problem of the lowered study, with its level structure (`"batched_lanes"`)."""
        if "batched_lanes" not in self.structure:
            raise _missing(self, "batched_lanes", "joint_view")
        return self._joint_view()

    @property
    def unflatten(self) -> Callable | None:
        """A function of `x` returning the dynamic part of the fitted model's parameter pytree, or `None`."""
        return self._unflatten.value

    @property
    def static(self) -> Any:
        """The frozen part of the fitted model, or `None`."""
        return self._static.value

    @property
    def regularization(self) -> Any:
        """The regularization whose rows or penalty the solver adds, or `None`."""
        return self._regularization.value

    def fitted(self, x: ArrayLike) -> Any:
        """The fitted model at `x`, or `None` without `unflatten` and `static`."""
        if self.unflatten is None or self.static is None:
            return None
        return eqx.combine(self.unflatten(x), self.static)

    @classmethod
    def from_value(
        cls,
        x0: Any,
        value: Callable,
        *,
        regularization: Any = None,
        bounds: Any = None,
        is_valid: Callable | None = None,
        unflatten: Callable | None = None,
        static: Any = None,
        hessian: bool = False,
    ) -> OptimizationProblem:
        """Pose a problem from a scalar objective of a 1-D array.

        The gradient comes from reverse-mode autodiff, and every function is compiled with
        `jax.jit`.

        Args:
            x0: The starting point, a 1-D array.
            value: A JAX-traceable function `value(x)` returning the scalar objective.
            regularization: A penalty the solver adds to the objective, such as `Tikhonov`, or `None`.
            bounds: A `(lower, upper)` pair aligned with `x0`, where `None` on either side leaves it unbounded.
            is_valid: A function `is_valid(x)` returning whether a trial point may be evaluated.
            unflatten: A function rebuilding the fitted model's arrays from `x`, used with `static`.
            static: The static part of the fitted model, combined with `unflatten(x)`.
            hessian: If `True`, also provide the exact Hessian by autodiff.

        Returns:
            An `OptimizationProblem` with the `"value_gradient"` structure, and `"hessian"` when requested.
        """
        x0 = _derived.as_vector(x0)
        value = ignoring_args(value)
        jit = _derived.jit
        structure = {"value_gradient", "hessian"} if hessian else {"value_gradient"}
        return cls(
            x0=x0,
            structure=frozenset(structure),
            _unflatten=unflatten,
            _static=static,
            bounds=_derived.as_bounds(bounds, x0),
            _regularization=regularization,
            ad="reverse",
            hessian_composition="forward-over-reverse" if hessian else None,
            _value=jit(value),
            _value_and_gradient=jit(Derived(_derived.value_and_gradient, value)),
            _value_gradient_hessian=(
                jit(Derived(_derived.value_gradient_hessian, value)) if hessian else None
            ),
            _hessian_vector=jit(Derived(_derived.hessian_vector, value)) if hessian else None,
            _is_valid=jit(ignoring_args(is_valid)),
        )


__all__ = ["STRUCTURE_KEYS", "OptimizationProblem"]
