"""A numeric problem whose objective is half the squared norm of a residual."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx

from diff_soup._errors import ConfigurationError
from diff_soup.optimization import _derived
from diff_soup.optimization._binding import Bound, Derived, ignoring_args
from diff_soup.optimization._numeric import OptimizationProblem


class LeastSquaresProblem(OptimizationProblem):
    """An optimization problem whose objective is half the squared norm of a residual vector.

    `ds.optimization.optsolve` builds one from an `InferenceProblem`, and `from_residual` poses
    one from any JAX-traceable residual function. `ds.optimization.LevenbergMarquardt` needs
    this form.
    """

    _residual: Callable | None = eqx.field(static=True, default=None)
    _residual_and_jacobian: Callable | None = eqx.field(static=True, default=None)
    _segment_fns: tuple[Callable, ...] = eqx.field(static=True, default=())
    n_residuals: int | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        """Store the value, `J^T r` and `J^T J` from the residual where none is given."""
        rj = self._residual_and_jacobian
        if rj is not None:
            if self._value_and_gradient is None:
                self._value_and_gradient = Derived(_derived.gradient, rj)
            if self._gauss_newton is None:
                self._gauss_newton = Derived(_derived.gauss_newton, rj)
        if self._value is None and self._residual is not None:
            self._value = Derived(_derived.half_squared, self._residual)
        if not self._segment_fns and self._residual is not None:
            self._segment_fns = (self._residual,)
        super().__post_init__()

    def __check_init__(self) -> None:
        """Refuse a least-squares problem without its residual structure."""
        needed = {"value_gradient", "residual_jacobian", "gauss_newton"}
        if not needed <= set(self.structure):
            raise ConfigurationError(
                f"LeastSquaresProblem structure must hold {sorted(needed)}; got "
                f"{sorted(self.structure)}."
            )

    @property
    def residual(self) -> Callable:
        """The residual, a function `residual(x)` returning a 1-D array."""
        return self._bound("residual")

    @property
    def residual_and_jacobian(self) -> Callable:
        """A function `residual_and_jacobian(x)` returning the residual and its Jacobian."""
        return self._bound("residual_and_jacobian")

    @property
    def segment_fns(self) -> tuple[Callable, ...]:
        """The residual functions of `x` whose outputs concatenate to `residual(x)`."""
        return tuple(Bound(fn, self.args) for fn in self._segment_fns)

    @classmethod
    def from_residual(
        cls,
        x0: Any,
        residual: Callable,
        *,
        args: Any = None,
        jacobian: Callable | None = None,
        regularization: Any = None,
        bounds: Any = None,
        is_valid: Callable | None = None,
        unflatten: Callable | None = None,
        static: Any = None,
        segment_fns: tuple[Callable, ...] = (),
        hessian: bool = True,
        hessian_composition: str = "forward-over-reverse",
    ) -> LeastSquaresProblem:
        """Pose a least-squares problem from a residual function of a 1-D array.

        Posed again from the same functions, the problem has equal closures, so a compiled
        solver reuses its compile.

        Args:
            x0: The starting point, a 1-D array.
            residual: A JAX-traceable function `residual(x)` returning the residual vector, or `residual(x, args)` when `args` is given.
            args: The pytree `residual`, `jacobian`, `is_valid` and `segment_fns` take last, held as the problem's `args` leaf; `None` when they take `x` alone.
            jacobian: A function `jacobian(x)` (or `jacobian(x, args)`) returning the Jacobian of `residual`, or `None` to compute it with `jax.jacfwd`.
            regularization: A penalty the solver adds to the objective, such as `Tikhonov`, or `None`.
            bounds: A `(lower, upper)` pair aligned with `x0`, where `None` on either side leaves it unbounded.
            is_valid: A function `is_valid(x)` returning whether a trial point may be evaluated.
            unflatten: A function rebuilding the fitted model's arrays from `x`, used with `static`.
            static: The static part of the fitted model, combined with `unflatten(x)`.
            segment_fns: Residual functions whose outputs concatenate to `residual(x)`, with `()` meaning `(residual,)`.
            hessian: If `True`, also provide the exact Hessian of the objective by autodiff.
            hessian_composition: `"forward-over-reverse"` takes the Hessians in forward mode over a reverse-mode gradient. `"forward-over-forward"` takes them over a forward-mode gradient, at the cost of `n` forward passes per gradient: choose it when `residual` runs an inner solve that supports forward mode only, such as a DAE solve under a forward-only adjoint.

        Returns:
            A `LeastSquaresProblem`.

        !!! example

            ```python
            t = jnp.linspace(0.0, 2.0, 20)
            y = 3.0 * jnp.exp(-1.3 * t)
            problem = ds.optimization.LeastSquaresProblem.from_residual(
                jnp.array([1.0, 1.0]), lambda x: x[0] * jnp.exp(-x[1] * t) - y)
            result = ds.optimization.optsolve(problem, ds.optimization.LevenbergMarquardt())
            ```
        """
        x0 = _derived.as_vector(x0)
        jit = _derived.jit
        # Functions of `x` alone take and drop the trailing `args` every closure is given.
        closure = _derived.identity if args is not None else ignoring_args
        residual = closure(residual)
        residual_and_jacobian = Derived(_derived.residual_and_jacobian, residual, closure(jacobian))
        value = Derived(_derived.half_squared, residual)
        _, hessian_rule, product_rule = _derived.objective_rules(hessian_composition)
        forward = hessian_composition == "forward-over-forward"
        structure = {"value_gradient", "residual_jacobian", "gauss_newton"}
        if hessian:
            structure.add("hessian")
        return cls(
            x0=x0,
            structure=frozenset(structure),
            args=args,
            _unflatten=unflatten,
            _static=static,
            bounds=_derived.as_bounds(bounds, x0),
            _regularization=regularization,
            ad="forward",
            # Recorded for `with_constraints` even without a Hessian, unless it is the default.
            hessian_composition=hessian_composition if hessian or forward else None,
            _value=jit(value),
            _value_and_gradient=jit(Derived(_derived.gradient, residual_and_jacobian)),
            _value_gradient_hessian=jit(Derived(hessian_rule, value)) if hessian else None,
            _hessian_vector=jit(Derived(product_rule, value)) if hessian else None,
            _gauss_newton=jit(Derived(_derived.gauss_newton, residual_and_jacobian)),
            _is_valid=jit(closure(is_valid)),
            _residual=jit(residual),
            _residual_and_jacobian=jit(residual_and_jacobian),
            _segment_fns=tuple(jit(closure(fn)) for fn in segment_fns),
        )


__all__ = ["LeastSquaresProblem"]
