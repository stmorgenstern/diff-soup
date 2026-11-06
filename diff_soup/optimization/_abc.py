"""Abstract base classes for ``diff_soup.inference``.

This module defines the interfaces shared by the optimization solvers, the
problems they solve, and the function-approximation bases:

- [OptimizationProblem][diff_soup.inference._abc.OptimizationProblem]: the
  numerical problem a solver reads, a pytree whose ``structure`` names what
  it provides (a value and gradient, a Hessian, Gauss-Newton statistics, a
  residual and Jacobian, constraints, the lanes of a lowered study).
- [LeastSquaresProblem][diff_soup.inference._abc.LeastSquaresProblem]: an
  optimization problem whose objective is half the squared norm of a
  residual; ``LeastSquaresProblem.from_residual`` builds one from a plain
  residual function.
- [OptSolver][diff_soup.inference._abc.OptSolver]: a solver, which names in
  ``requires`` the structure it needs.

Import only this module when you need type annotations: it does not import
any solver, problem, or fitting machinery, so it is safe to use anywhere in
the package without creating import cycles.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError

if TYPE_CHECKING:
    from diff_soup.inference.fitting.result import OptResult


#: The structure an [OptimizationProblem][diff_soup.inference._abc.OptimizationProblem]
#: can provide, each key a set of methods it then answers:
#:
#: - ``"value_gradient"``: ``value(x)`` and ``value_and_gradient(x)``.
#: - ``"hessian"``: ``value_gradient_hessian(x)`` and ``hessian_vector(x, v)``.
#: - ``"gauss_newton"``: ``gauss_newton(x) -> (cost, gradient, matrix, failed)``.
#: - ``"residual_jacobian"``: ``residual(x)`` and ``residual_and_jacobian(x)``
#:   (a [LeastSquaresProblem][diff_soup.inference._abc.LeastSquaresProblem]).
#: - ``"constraints"``: ``constraints(x)`` returns at least one equality or
#:   inequality row.
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
        "batched_lanes",
    }
)

#: The entry of a solver's ``requires`` that names a capability of the solver
#: rather than a structure of the problem: its ``solve`` takes ``state=``, the
#: solver state a step record or a saved result carries, and continues from it.
RESUME = "resume"


def _missing(problem: Any, key: str, method: str) -> ConfigurationError:
    """The error for ``method`` on a problem whose structure lacks ``key``."""
    provided = ", ".join(sorted(problem.structure)) or "nothing"
    return ConfigurationError(
        f"{type(problem).__name__}.{method} needs the {key!r} structure, and this problem "
        f"provides {provided}; build the problem with that structure or use a solver whose "
        "requires the problem meets."
    )


class _Held:
    """A value held by identity: hashable, and not a pytree whatever it holds."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        """Hold ``value`` by identity, outside the pytree."""
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


def _hold(value: Any) -> _Held:
    """Wrap ``value`` in a ``_Held`` unless it is one already."""
    return value if isinstance(value, _Held) else _Held(value)


class OptimizationProblem(eqx.Module):
    """The objective an optimizer minimizes, with its starting point, bounds and derivatives.

    `ds.inference.optsolve` builds one from the problem it is given and passes it to
    `OptSolver.solve`. `structure` names the functions it provides, and asking for any
    other raises `ConfigurationError`.

    Attributes:
        x0: The starting point, a 1-D array.
        structure: The names of the functions this problem provides, such as `"value_gradient"`, `"hessian"` or `"gauss_newton"`.
        bounds: The `(lower, upper)` arrays aligned with `x0`, or `None` when unbounded.
    """

    x0: Any
    structure: frozenset = eqx.field(static=True)
    bounds: Any = None
    # The fitted model's reconstruction and the regularization can hold arrays; they are
    # held by identity, so the problem's static data stays hashable and array-free.
    _unflatten: Any = eqx.field(static=True, default=None, converter=_hold)
    _static: Any = eqx.field(static=True, default=None, converter=_hold)
    _regularization: Any = eqx.field(static=True, default=None, converter=_hold)
    lanes: Any = eqx.field(static=True, default=None)
    ad: Any = eqx.field(static=True, default=None)
    hessian_composition: Any = eqx.field(static=True, default=None)
    n_segments: int = eqx.field(static=True, default=1)
    reduced_gauss_newton: bool = eqx.field(static=True, default=False)
    _value: Any = eqx.field(static=True, default=None)
    _value_and_gradient: Any = eqx.field(static=True, default=None)
    _value_gradient_hessian: Any = eqx.field(static=True, default=None)
    _hessian_vector: Any = eqx.field(static=True, default=None)
    _gauss_newton: Any = eqx.field(static=True, default=None)
    _is_valid: Any = eqx.field(static=True, default=None)
    _constraints: Any = eqx.field(static=True, default=None)
    _joint_view: Any = eqx.field(static=True, default=None)

    def __post_init__(self):
        """Derive a value from a value and gradient, a Hessian-vector product from a Hessian."""
        if self._value is None and self._value_and_gradient is not None:
            self._value = _first_of(self._value_and_gradient)
        if self._hessian_vector is None and self._value_gradient_hessian is not None:
            self._hessian_vector = _hessian_times(self._value_gradient_hessian)

    def __check_init__(self):
        """Refuse a structure key outside ``STRUCTURE_KEYS``, or a key without its closures."""
        unknown = set(self.structure) - STRUCTURE_KEYS
        if unknown:
            raise ConfigurationError(
                f"OptimizationProblem structure names {sorted(unknown)}, which are not "
                f"structure keys; the keys are {sorted(STRUCTURE_KEYS)}."
            )
        for key, closures in _CLOSURES.items():
            if key in self.structure:
                absent = [name for name in closures if getattr(self, name) is None]
                if absent:
                    raise ConfigurationError(
                        f"OptimizationProblem structure names {key!r}, which needs the "
                        f"closures {absent}; pass them or leave {key!r} out of structure."
                    )

    def _closure(self, key: str, name: str, method: str) -> Callable:
        """The stored closure ``name``, or the error naming ``method`` when ``key`` is absent."""
        if key not in self.structure:
            raise _missing(self, key, method)
        return getattr(self, name)

    # The evaluation methods are properties returning the stored closures, so that
    # ``jax.jit(problem.value_and_gradient)`` compiles a hashable function.

    @property
    def value(self) -> Callable:
        """The objective, a function `value(x)` returning a scalar."""
        return self._closure("value_gradient", "_value", "value")

    @property
    def value_and_gradient(self) -> Callable:
        """A function `value_and_gradient(x)` returning the objective and its gradient."""
        return self._closure("value_gradient", "_value_and_gradient", "value_and_gradient")

    @property
    def value_gradient_hessian(self) -> Callable:
        """``x -> (value, gradient, Hessian)`` of the objective (``"hessian"``)."""
        return self._closure("hessian", "_value_gradient_hessian", "value_gradient_hessian")

    @property
    def hessian_vector(self) -> Callable:
        """``(x, v) -> `` the Hessian of the objective at ``x`` times ``v`` (``"hessian"``)."""
        return self._closure("hessian", "_hessian_vector", "hessian_vector")

    @property
    def gauss_newton(self) -> Callable:
        """A function `gauss_newton(x)` returning the objective, its gradient, the Gauss-Newton matrix and a failure count.

        Available when `structure` holds `"gauss_newton"`.
        """
        return self._closure("gauss_newton", "_gauss_newton", "gauss_newton")

    @property
    def is_valid(self) -> Callable:
        """``x -> `` whether ``x`` is a trial point the objective may be evaluated at.

        A scalar bool array; every point is valid for a problem without a
        trial predicate.
        """
        return _always_valid if self._is_valid is None else self._is_valid

    @property
    def constraints(self) -> Callable:
        """``x -> (equality, inequality)`` constraint rows at ``x``, both 1-D.

        An equality row is satisfied at 0, an inequality row at or below 0.
        Both are empty for a problem without ``"constraints"``.
        """
        return _no_constraints if self._constraints is None else self._constraints

    def joint_view(self) -> Any:
        """The joint problem of the lowered study, with its level structure (``"batched_lanes"``)."""
        return self._closure("batched_lanes", "_joint_view", "joint_view")()

    @property
    def unflatten(self) -> Any:
        """``x -> `` the dynamic part of the fitted model's parameter pytree, or ``None``."""
        return self._unflatten.value

    @property
    def static(self) -> Any:
        """The frozen part of the fitted model, or ``None``."""
        return self._static.value

    @property
    def regularization(self) -> Any:
        """``None``, or the regularization whose rows or penalty the solver adds."""
        return self._regularization.value

    def fitted(self, x: Any) -> Any:
        """The fitted model at ``x``, or ``None`` without ``unflatten`` and ``static``."""
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
        x0 = _as_vector(x0)
        structure = {"value_gradient"}
        value_gradient_hessian = hessian_vector = None
        if hessian:
            structure.add("hessian")

            @jax.jit
            def value_gradient_hessian(x):
                cost, gradient = jax.value_and_grad(value)(x)
                return cost, gradient, jax.hessian(value)(x)

            @jax.jit
            def hessian_vector(x, v):
                return jax.jvp(jax.grad(value), (x,), (v,))[1]

        return cls(
            x0=x0,
            structure=frozenset(structure),
            _unflatten=unflatten,
            _static=static,
            bounds=_as_bounds(bounds, x0),
            _regularization=regularization,
            ad="reverse",
            hessian_composition="forward-over-reverse" if hessian else None,
            _value=jax.jit(value),
            _value_and_gradient=jax.jit(jax.value_and_grad(value)),
            _value_gradient_hessian=value_gradient_hessian,
            _hessian_vector=hessian_vector,
            _is_valid=None if is_valid is None else jax.jit(is_valid),
        )


#: The closures each structure key needs.
_CLOSURES = {
    "value_gradient": ("_value", "_value_and_gradient"),
    "hessian": ("_value_gradient_hessian", "_hessian_vector"),
    "gauss_newton": ("_gauss_newton",),
    "batched_lanes": ("_joint_view",),
}


def _always_valid(x: Any) -> Any:
    """The validity predicate of a problem without one: every point is valid."""
    return jnp.asarray(True)


def _no_constraints(x: Any) -> tuple:
    """The constraint function of a problem without constraints: two empty vectors."""
    empty = jnp.zeros((0,), dtype=jnp.result_type(x))
    return empty, empty


def _first_of(value_and_gradient: Callable) -> Callable:
    """The value function derived from a value-and-gradient closure."""

    def value(x):
        return value_and_gradient(x)[0]

    return value


def _hessian_times(value_gradient_hessian: Callable) -> Callable:
    """The Hessian-vector product derived from a value-gradient-Hessian closure."""

    def hessian_vector(x, v):
        return value_gradient_hessian(x)[2] @ v

    return hessian_vector


def _as_vector(x0: Any) -> Any:
    """``x0`` as a 1-D array; a ``ConfigurationError`` for any other shape."""
    if not isinstance(x0, (jax.Array, np.ndarray)):
        x0 = np.asarray(x0)
    if x0.ndim != 1:
        raise ConfigurationError(
            f"an optimization problem's x0 must be a 1-D array; got shape {tuple(x0.shape)}."
        )
    return x0


def _as_bounds(bounds: Any, x0: Any) -> Any:
    """``bounds`` as the ``(lb, ub)`` pair broadcast to ``x0``'s shape and dtype, or ``None``."""
    if bounds is None:
        return None
    try:
        lower, upper = bounds
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "bounds must be None or a (lb, ub) pair of arrays aligned with x0."
        ) from exc
    dtype = jnp.result_type(jnp.asarray(x0))
    shape = np.shape(x0)
    lower = jnp.broadcast_to(jnp.asarray(-jnp.inf if lower is None else lower, dtype), shape)
    upper = jnp.broadcast_to(jnp.asarray(jnp.inf if upper is None else upper, dtype), shape)
    return lower, upper


def _half_squared(residual: Callable) -> Callable:
    """The least-squares value ``0.5 * sum(r**2)`` derived from a residual closure."""

    def value(x):
        r = residual(x)
        return 0.5 * jnp.sum(r * r)

    return value


def _gradient_of(residual_and_jacobian: Callable) -> Callable:
    """The value and gradient ``J^T r`` derived from a residual-and-Jacobian closure."""

    def value_and_gradient(x):
        r, jacobian = residual_and_jacobian(x)
        return 0.5 * jnp.sum(r * r), jacobian.T @ r

    return value_and_gradient


def _gauss_newton_of(residual_and_jacobian: Callable) -> Callable:
    """The Gauss-Newton statistics ``(value, J^T r, J^T J, 0)`` derived from a residual-and-Jacobian closure."""

    def gauss_newton(x):
        r, jacobian = residual_and_jacobian(x)
        return 0.5 * jnp.sum(r * r), jacobian.T @ r, jacobian.T @ jacobian, 0

    return gauss_newton


class LeastSquaresProblem(OptimizationProblem):
    """An optimization problem whose objective is half the squared norm of a residual vector.

    `ds.inference.optsolve` builds one from an `InferenceProblem`, and `from_residual` poses
    one from any JAX-traceable residual function. `ds.inference.LevenbergMarquardt` needs
    this form.
    """

    _residual: Any = eqx.field(static=True, default=None)
    _residual_and_jacobian: Any = eqx.field(static=True, default=None)
    segment_fns: tuple = eqx.field(static=True, default=())
    n_residuals: Any = eqx.field(static=True, default=None)

    def __post_init__(self):
        """Derive the value, ``J^T r`` and ``J^T J`` from the residual where none is given."""
        rj = self._residual_and_jacobian
        if rj is not None:
            if self._value_and_gradient is None:
                self._value_and_gradient = _gradient_of(rj)
            if self._gauss_newton is None:
                self._gauss_newton = _gauss_newton_of(rj)
        if self._value is None and self._residual is not None:
            self._value = _half_squared(self._residual)
        if not self.segment_fns and self._residual is not None:
            self.segment_fns = (self._residual,)
        super().__post_init__()

    def __check_init__(self):
        """Refuse a least-squares problem without its residual structure."""
        needed = {"value_gradient", "residual_jacobian", "gauss_newton"}
        if not needed <= set(self.structure):
            raise ConfigurationError(
                f"LeastSquaresProblem structure must hold {sorted(needed)}; got "
                f"{sorted(self.structure)}."
            )
        if self._residual is None or self._residual_and_jacobian is None:
            raise ConfigurationError(
                "LeastSquaresProblem needs a residual and a residual-and-Jacobian closure."
            )

    @property
    def residual(self) -> Callable:
        """The residual, a function `residual(x)` returning a 1-D array."""
        return self._residual

    @property
    def residual_and_jacobian(self) -> Callable:
        """A function `residual_and_jacobian(x)` returning the residual and its Jacobian."""
        return self._residual_and_jacobian

    @classmethod
    def from_residual(
        cls,
        x0: Any,
        residual: Callable,
        *,
        jacobian: Callable | None = None,
        regularization: Any = None,
        bounds: Any = None,
        is_valid: Callable | None = None,
        unflatten: Callable | None = None,
        static: Any = None,
        segment_fns: tuple = (),
        hessian: bool = True,
    ) -> LeastSquaresProblem:
        """Pose a least-squares problem from a residual function of a 1-D array.

        Args:
            x0: The starting point, a 1-D array.
            residual: A JAX-traceable function `residual(x)` returning the residual vector.
            jacobian: A function `jacobian(x)` returning the Jacobian of `residual`, or `None` to compute it with `jax.jacfwd`.
            regularization: A penalty the solver adds to the objective, such as `Tikhonov`, or `None`.
            bounds: A `(lower, upper)` pair aligned with `x0`, where `None` on either side leaves it unbounded.
            is_valid: A function `is_valid(x)` returning whether a trial point may be evaluated.
            unflatten: A function rebuilding the fitted model's arrays from `x`, used with `static`.
            static: The static part of the fitted model, combined with `unflatten(x)`.
            segment_fns: Residual functions whose outputs concatenate to `residual(x)`, with `()` meaning `(residual,)`.
            hessian: If `True`, also provide the exact Hessian of the objective by autodiff.

        Returns:
            A `LeastSquaresProblem`.

        !!! example

            ```python
            t = jnp.linspace(0.0, 2.0, 20)
            y = 3.0 * jnp.exp(-1.3 * t)
            problem = ds.inference.LeastSquaresProblem.from_residual(
                jnp.array([1.0, 1.0]), lambda x: x[0] * jnp.exp(-x[1] * t) - y)
            result = ds.inference.optsolve(problem, ds.inference.LevenbergMarquardt())
            ```
        """
        x0 = _as_vector(x0)
        residual_fn = jax.jit(residual)
        if jacobian is None:
            residual_and_jacobian = jax.jit(lambda x: (residual(x), jax.jacfwd(residual)(x)))
        else:
            residual_and_jacobian = jax.jit(lambda x: (residual(x), jacobian(x)))

        objective = _half_squared(residual)
        value_and_gradient = jax.jit(_gradient_of(residual_and_jacobian))
        gauss_newton = jax.jit(_gauss_newton_of(residual_and_jacobian))

        structure = {"value_gradient", "residual_jacobian", "gauss_newton"}
        value_gradient_hessian = hessian_vector = None
        if hessian:
            structure.add("hessian")

            @jax.jit
            def value_gradient_hessian(x):
                value, gradient = jax.value_and_grad(objective)(x)
                return value, gradient, jax.hessian(objective)(x)

            @jax.jit
            def hessian_vector(x, v):
                return jax.jvp(jax.grad(objective), (x,), (v,))[1]

        return cls(
            x0=x0,
            structure=frozenset(structure),
            _unflatten=unflatten,
            _static=static,
            bounds=_as_bounds(bounds, x0),
            _regularization=regularization,
            ad="forward",
            hessian_composition="forward-over-reverse" if hessian else None,
            _value=jax.jit(objective),
            _value_and_gradient=value_and_gradient,
            _value_gradient_hessian=value_gradient_hessian,
            _hessian_vector=hessian_vector,
            _gauss_newton=gauss_newton,
            _is_valid=None if is_valid is None else jax.jit(is_valid),
            _residual=residual_fn,
            _residual_and_jacobian=residual_and_jacobian,
            segment_fns=tuple(segment_fns) or (residual_fn,),
        )


class OptSolver(eqx.Module):
    """Base class for an optimizer that `ds.inference.optsolve` runs.

    A subclass implements `solve` and sets `requires`. `ds.inference.optsolve` refuses a
    problem that cannot provide what the solver requires.

    Attributes:
        requires: The names of what `solve` needs from the problem, such as `"value_gradient"`, plus `"resume"` when `solve` takes `state=`.
    """

    #: The structure keys the solver needs, and ``RESUME`` when ``solve``
    #: takes ``state=``. The empty set takes every problem.
    requires: ClassVar[frozenset] = frozenset()

    @abstractmethod
    def solve(
        self,
        problem: Any,
        *,
        callbacks: tuple = (),
    ) -> "OptResult":
        """Minimize the problem and return the result.

        Args:
            problem: The [OptimizationProblem][diff_soup.inference.OptimizationProblem] to minimize.
            callbacks: The [Callback][diff_soup.inference.Callback] objects to call during the run.

        Returns:
            The [OptResult][diff_soup.inference.OptResult] of the run.
        """
        raise NotImplementedError


class Basis(eqx.Module):
    """Marker base for function-approximation bases.

    A ``Basis`` is an ``eqx.Module`` pytree that holds differentiable
    coefficient leaves.  Subclass this to mark a class as a function-
    approximation basis compatible with diff_soup's series and collocation
    solvers.

    No abstract methods are required; the class serves as a structural marker
    and a pytree node via the Equinox module system.
    """


__all__ = [
    "RESUME",
    "STRUCTURE_KEYS",
    "Basis",
    "LeastSquaresProblem",
    "OptSolver",
    "OptimizationProblem",
]
