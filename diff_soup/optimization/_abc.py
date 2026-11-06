"""Abstract base classes for ``diff_soup.optimization``.

This module defines the interfaces shared by the optimization solvers, the
problems they solve, and the function-approximation bases:

- [OptimizationProblem][diff_soup.optimization._abc.OptimizationProblem]: the
  numerical problem a solver reads, a pytree whose ``structure`` names what
  it provides (a value and gradient, a Hessian, Gauss-Newton statistics, a
  residual and Jacobian, constraints with their Jacobians and Lagrangian
  Hessians, the lanes of a lowered study). Every closure takes ``(x, args)``,
  with ``args`` a leaf of the problem.
- [LeastSquaresProblem][diff_soup.optimization._abc.LeastSquaresProblem]: an
  optimization problem whose objective is half the squared norm of a
  residual; ``LeastSquaresProblem.from_residual`` builds one from a plain
  residual function.
- [OptSolver][diff_soup.optimization._abc.OptSolver]: a solver, which names in
  ``requires`` the structure it needs.

Import only this module when you need type annotations: it does not import
any solver, problem, or fitting machinery, so it is safe to use anywhere in
the package without creating import cycles.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.optimization._binding import Bound, ignoring_args

if TYPE_CHECKING:
    from diff_soup.optimization.inference.fitting.result import OptResult


#: The structure an [OptimizationProblem][diff_soup.optimization._abc.OptimizationProblem]
#: can provide, each key a set of closures it then holds, every one taking ``args`` last:
#:
#: - ``"value_gradient"``: ``value(x, args)`` and ``value_and_gradient(x, args)``.
#: - ``"hessian"``: ``value_gradient_hessian(x, args)`` and ``hessian_vector(x, v, args)``.
#: - ``"gauss_newton"``: ``gauss_newton(x, args) -> (cost, gradient, matrix, failed)``.
#: - ``"residual_jacobian"``: ``residual(x, args)`` and ``residual_and_jacobian(x, args)``
#:   (a [LeastSquaresProblem][diff_soup.optimization._abc.LeastSquaresProblem]).
#: - ``"constraints"``: ``constraints(x, args) -> (equality, inequality)`` returns at least
#:   one row; an equality row holds at 0, an inequality row at or below 0.
#: - ``"constraint_jacobian"``: ``constraints_and_jacobians(x, args) -> (eq, ineq, J_eq,
#:   J_ineq)``, the dense Jacobians.
#: - ``"lagrangian_hessian"``: ``lagrangian_hessian(x, lam_eq, lam_ineq, args)`` and
#:   ``lagrangian_hessian_vector(x, lam_eq, lam_ineq, v, args)``, the exact second
#:   derivatives of ``f + lam_eq . c_E + lam_ineq . c_I``.
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

    `ds.optimization.optsolve` builds one from the problem it is given and passes it to
    `OptSolver.solve`. `structure` names the functions it provides, and asking for any
    other raises `ConfigurationError`.

    Every closure takes `args` last: `value(x, args)`, `constraints(x, args)`. The
    properties return them bound to the problem's own `args`, so `problem.value(x)` works,
    and a solver that compiles its loop takes the raw closure from `closure(name)` and
    passes `args` itself. A batch of problems that differ only in `args` compiles once.

    Attributes:
        x0: The starting point, a 1-D array.
        structure: The names of the functions this problem provides, such as `"value_gradient"`, `"hessian"` or `"gauss_newton"`.
        bounds: The `(lower, upper)` arrays aligned with `x0`, or `None` when unbounded.
        args: The pytree every closure takes last: limits, initial states, outer parameters. `None` for a problem whose closures read nothing else.
        n_equality: The number of equality rows `constraints` returns.
        n_inequality: The number of inequality rows `constraints` returns.
        constraint_scales: `(equality_scales, inequality_scales)`, one positive scale per row, or `None` for the solver's own scaling.
    """

    x0: Any
    structure: frozenset = eqx.field(static=True)
    bounds: Any = None
    args: Any = None
    n_equality: int = eqx.field(static=True, default=0)
    n_inequality: int = eqx.field(static=True, default=0)
    constraint_scales: Any = None
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
    _constraints_and_jacobians: Any = eqx.field(static=True, default=None)
    _lagrangian_hessian: Any = eqx.field(static=True, default=None)
    _lagrangian_hessian_vector: Any = eqx.field(static=True, default=None)
    _joint_view: Any = eqx.field(static=True, default=None)

    def __post_init__(self):
        """Derive what follows from what is given; a problem without rows gets empty ones.

        A value from a value and gradient, a Hessian-vector product from a Hessian, the
        constraint rows from the rows and Jacobians, and a Lagrangian Hessian-vector product
        from the Lagrangian Hessian.
        """
        if self._value is None and self._value_and_gradient is not None:
            self._value = _first_of(self._value_and_gradient)
        if self._hessian_vector is None and self._value_gradient_hessian is not None:
            self._hessian_vector = _hessian_times(self._value_gradient_hessian)
        if self._constraints is None and self._constraints_and_jacobians is not None:
            self._constraints = _rows_of(self._constraints_and_jacobians)
        if self.n_equality == 0 and self.n_inequality == 0:
            # No rows: the constraint closures are empty, and the Lagrangian is the objective.
            if self._constraints is None:
                self._constraints = _no_constraints
            if self._constraints_and_jacobians is None:
                self._constraints_and_jacobians = _no_constraint_jacobians
            if self._lagrangian_hessian is None and self._value_gradient_hessian is not None:
                self._lagrangian_hessian = _objective_hessian(self._value_gradient_hessian)
            implied = {"constraint_jacobian"}
            if self._lagrangian_hessian is not None:
                implied.add("lagrangian_hessian")
            self.structure = frozenset(self.structure) | implied
        if self._lagrangian_hessian_vector is None and self._lagrangian_hessian is not None:
            self._lagrangian_hessian_vector = _lagrangian_times(self._lagrangian_hessian)

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

    def closure(self, name: str) -> Callable:
        """The raw closure `name`, which takes `args` last, for a solver that passes `args` itself.

        Args:
            name: A property of this problem, such as `"value_and_gradient"` or `"constraints_and_jacobians"`.

        Returns:
            The stored closure, unbound.

        Raises:
            ConfigurationError: If the problem's structure does not provide `name`.
        """
        if name not in _KEY_OF:
            raise ConfigurationError(
                f"OptimizationProblem has no closure {name!r}; the closures are {sorted(_KEY_OF)}."
            )
        key = _KEY_OF[name]
        stored = getattr(self, "_" + name)
        if name == "is_valid":
            return _valid_everywhere if stored is None else stored
        if stored is None or (key not in self.structure and name not in _ALWAYS):
            raise _missing(self, key, name)
        return stored

    def with_args(self, args: Any) -> OptimizationProblem:
        """This problem with `args` replaced, as a lane of a batch holds it."""
        return eqx.tree_at(lambda problem: problem.args, self, args, is_leaf=_is_none)

    def with_constraints(
        self, constraints: Any, *, unflatten: Callable | None = None
    ) -> OptimizationProblem:
        """This problem under `constraints`, lowered to closures with AD Jacobians and Lagrangian Hessians.

        Args:
            constraints: The [Constraints][diff_soup.optimization.Constraints], or `None` for this problem unchanged.
            unflatten: The map from the flat vector to the view the constraint functions take; the flat vector itself when `None`.

        Returns:
            A problem whose structure adds `"constraints"`, `"constraint_jacobian"` and `"lagrangian_hessian"`.
        """
        if constraints is None:
            return self
        if self.n_equality or self.n_inequality:
            raise ConfigurationError(
                "OptimizationProblem.with_constraints: this problem has constraints already."
            )
        if "value_gradient" not in self.structure:
            raise _missing(self, "value_gradient", "with_constraints")
        # With the objective's own exact Hessian, only the constraint rows are differentiated
        # twice; otherwise the whole Lagrangian is, forward over reverse.
        own = "hessian" in self.structure
        view = _identity if unflatten is None else unflatten
        lowered = constraints.lower(None if own else self._value, view)
        hessian, hessian_vector = lowered.lagrangian_hessian, lowered.lagrangian_hessian_vector
        if own:
            hessian = _plus_objective_hessian(hessian, self._value_gradient_hessian)
            hessian_vector = _plus_objective_hessian_vector(hessian_vector, self._hessian_vector)
        keys = {"constraint_jacobian", "lagrangian_hessian"}
        if lowered.n_equality or lowered.n_inequality:
            keys.add("constraints")
        return dataclasses.replace(
            self,
            structure=frozenset(self.structure) | keys,
            n_equality=lowered.n_equality,
            n_inequality=lowered.n_inequality,
            constraint_scales=constraints.scales,
            _constraints=lowered.constraints,
            _constraints_and_jacobians=lowered.constraints_and_jacobians,
            _lagrangian_hessian=hessian,
            _lagrangian_hessian_vector=hessian_vector,
        )

    def _bound(self, name: str) -> Callable:
        """The closure `name` bound to this problem's `args`."""
        return Bound(self.closure(name), self.args)

    @property
    def n(self) -> int:
        """The number of decision variables, static."""
        return int(jnp.shape(self.x0)[-1])

    # The evaluation methods are properties returning the stored closures bound to
    # ``args``; two reads of one property are equal, so ``jax.jit(problem.value)`` read
    # twice traces once.

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
        """``x -> (value, gradient, Hessian)`` of the objective (``"hessian"``)."""
        return self._bound("value_gradient_hessian")

    @property
    def hessian_vector(self) -> Callable:
        """``(x, v) -> `` the Hessian of the objective at ``x`` times ``v`` (``"hessian"``)."""
        return self._bound("hessian_vector")

    @property
    def gauss_newton(self) -> Callable:
        """A function `gauss_newton(x)` returning the objective, its gradient, the Gauss-Newton matrix and a failure count.

        Available when `structure` holds `"gauss_newton"`.
        """
        return self._bound("gauss_newton")

    @property
    def is_valid(self) -> Callable:
        """``x -> `` whether ``x`` is a trial point the objective may be evaluated at.

        A scalar bool array; every point is valid for a problem without a
        trial predicate.
        """
        return _always_valid if self._is_valid is None else Bound(self._is_valid, self.args)

    @property
    def constraints(self) -> Callable:
        """``x -> (equality, inequality)`` constraint rows at ``x``, both 1-D.

        An equality row is satisfied at 0, an inequality row at or below 0.
        Both are empty for a problem without ``"constraints"``.
        """
        return self._bound("constraints")

    @property
    def constraints_and_jacobians(self) -> Callable:
        """``x -> (equality, inequality, J_eq, J_ineq)``, the rows and their dense Jacobians.

        Empty rows and ``(0, n)`` Jacobians for a problem without constraints.
        """
        return self._bound("constraints_and_jacobians")

    @property
    def lagrangian_hessian(self) -> Callable:
        """``(x, lam_eq, lam_ineq) -> `` the Hessian of ``f + lam_eq . c_E + lam_ineq . c_I`` (``"lagrangian_hessian"``)."""
        return self._bound("lagrangian_hessian")

    @property
    def lagrangian_hessian_vector(self) -> Callable:
        """``(x, lam_eq, lam_ineq, v) -> `` the Lagrangian Hessian times ``v`` (``"lagrangian_hessian"``)."""
        return self._bound("lagrangian_hessian_vector")

    def joint_view(self) -> Any:
        """The joint problem of the lowered study, with its level structure (``"batched_lanes"``)."""
        if "batched_lanes" not in self.structure:
            raise _missing(self, "batched_lanes", "joint_view")
        return self._joint_view()

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
            _value=ignoring_args(jax.jit(value)),
            _value_and_gradient=ignoring_args(jax.jit(jax.value_and_grad(value))),
            _value_gradient_hessian=ignoring_args(value_gradient_hessian),
            _hessian_vector=ignoring_args(hessian_vector),
            _is_valid=None if is_valid is None else ignoring_args(jax.jit(is_valid)),
        )


#: The closures each structure key needs.
_CLOSURES = {
    "value_gradient": ("_value", "_value_and_gradient"),
    "hessian": ("_value_gradient_hessian", "_hessian_vector"),
    "gauss_newton": ("_gauss_newton",),
    "constraints": ("_constraints",),
    "constraint_jacobian": ("_constraints_and_jacobians",),
    "lagrangian_hessian": ("_lagrangian_hessian", "_lagrangian_hessian_vector"),
    "batched_lanes": ("_joint_view",),
}

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
}

#: The closures every problem holds, with empty rows when it has no constraints.
_ALWAYS = frozenset({"constraints", "constraints_and_jacobians"})


def _identity(x: Any) -> Any:
    """The flat vector as the view constraint functions take, for a problem without one."""
    return x


def _objective_hessian(value_gradient_hessian: Callable) -> Callable:
    """The Lagrangian Hessian of a problem without constraints: the objective's."""

    def lagrangian_hessian(x, lam_eq, lam_ineq, args):
        return value_gradient_hessian(x, args)[2]

    return lagrangian_hessian


def _plus_objective_hessian(
    constraint_part: Callable, value_gradient_hessian: Callable
) -> Callable:
    """The Lagrangian Hessian as the objective's own Hessian plus the constraint terms'."""

    def lagrangian_hessian(x, lam_eq, lam_ineq, args):
        return value_gradient_hessian(x, args)[2] + constraint_part(x, lam_eq, lam_ineq, args)

    return lagrangian_hessian


def _plus_objective_hessian_vector(constraint_part: Callable, hessian_vector: Callable) -> Callable:
    """The Lagrangian Hessian-vector product as the objective's plus the constraint terms'."""

    def lagrangian_hessian_vector(x, lam_eq, lam_ineq, v, args):
        return hessian_vector(x, v, args) + constraint_part(x, lam_eq, lam_ineq, v, args)

    return lagrangian_hessian_vector


def _is_none(leaf: Any) -> bool:
    """Whether a pytree node is ``None``, which ``eqx.tree_at`` must see as a leaf to replace."""
    return leaf is None


def _always_valid(x: Any) -> Any:
    """The validity predicate of a problem without one: every point is valid."""
    return jnp.asarray(True)


def _valid_everywhere(x: Any, args: Any) -> Any:
    """The raw validity closure of a problem without a predicate: every point is valid."""
    return jnp.asarray(True)


def _no_constraints(x: Any, args: Any) -> tuple:
    """The constraint rows of a problem without constraints: two empty vectors."""
    empty = jnp.zeros((0,), dtype=jnp.result_type(x))
    return empty, empty


def _no_constraint_jacobians(x: Any, args: Any) -> tuple:
    """The rows and Jacobians of a problem without constraints: empty, with ``n`` columns."""
    empty = jnp.zeros((0,), dtype=jnp.result_type(x))
    jacobian = jnp.zeros((0, jnp.shape(x)[-1]), dtype=jnp.result_type(x))
    return empty, empty, jacobian, jacobian


def _rows_of(constraints_and_jacobians: Callable) -> Callable:
    """The constraint rows derived from a rows-and-Jacobians closure."""

    def constraints(x, args):
        equality, inequality, _, _ = constraints_and_jacobians(x, args)
        return equality, inequality

    return constraints


def _first_of(value_and_gradient: Callable) -> Callable:
    """The value function derived from a value-and-gradient closure."""

    def value(x, args):
        return value_and_gradient(x, args)[0]

    return value


def _hessian_times(value_gradient_hessian: Callable) -> Callable:
    """The Hessian-vector product derived from a value-gradient-Hessian closure."""

    def hessian_vector(x, v, args):
        return value_gradient_hessian(x, args)[2] @ v

    return hessian_vector


def _lagrangian_times(lagrangian_hessian: Callable) -> Callable:
    """The Lagrangian Hessian-vector product derived from the Lagrangian Hessian."""

    def lagrangian_hessian_vector(x, lam_eq, lam_ineq, v, args):
        return lagrangian_hessian(x, lam_eq, lam_ineq, args) @ v

    return lagrangian_hessian_vector


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

    def value(x, args):
        r = residual(x, args)
        return 0.5 * jnp.sum(r * r)

    return value


def _gradient_of(residual_and_jacobian: Callable) -> Callable:
    """The value and gradient ``J^T r`` derived from a residual-and-Jacobian closure."""

    def value_and_gradient(x, args):
        r, jacobian = residual_and_jacobian(x, args)
        return 0.5 * jnp.sum(r * r), jacobian.T @ r

    return value_and_gradient


def _gauss_newton_of(residual_and_jacobian: Callable) -> Callable:
    """The Gauss-Newton statistics ``(value, J^T r, J^T J, 0)`` derived from a residual-and-Jacobian closure."""

    def gauss_newton(x, args):
        r, jacobian = residual_and_jacobian(x, args)
        return 0.5 * jnp.sum(r * r), jacobian.T @ r, jacobian.T @ jacobian, 0

    return gauss_newton


class LeastSquaresProblem(OptimizationProblem):
    """An optimization problem whose objective is half the squared norm of a residual vector.

    `ds.optimization.optsolve` builds one from an `InferenceProblem`, and `from_residual` poses
    one from any JAX-traceable residual function. `ds.optimization.LevenbergMarquardt` needs
    this form.
    """

    _residual: Any = eqx.field(static=True, default=None)
    _residual_and_jacobian: Any = eqx.field(static=True, default=None)
    _segment_fns: tuple = eqx.field(static=True, default=())
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
        if not self._segment_fns and self._residual is not None:
            self._segment_fns = (self._residual,)
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
        return self._bound("residual")

    @property
    def residual_and_jacobian(self) -> Callable:
        """A function `residual_and_jacobian(x)` returning the residual and its Jacobian."""
        return self._bound("residual_and_jacobian")

    @property
    def segment_fns(self) -> tuple:
        """The residual functions of `x` whose outputs concatenate to `residual(x)`."""
        return tuple(Bound(fn, self.args) for fn in self._segment_fns)

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
            problem = ds.optimization.LeastSquaresProblem.from_residual(
                jnp.array([1.0, 1.0]), lambda x: x[0] * jnp.exp(-x[1] * t) - y)
            result = ds.optimization.optsolve(problem, ds.optimization.LevenbergMarquardt())
            ```
        """
        x0 = _as_vector(x0)
        residual_fn = jax.jit(residual)
        if jacobian is None:
            residual_and_jacobian = jax.jit(lambda x: (residual(x), jax.jacfwd(residual)(x)))
        else:
            residual_and_jacobian = jax.jit(lambda x: (residual(x), jacobian(x)))

        def objective(x):
            r = residual(x)
            return 0.5 * jnp.sum(r * r)

        value_and_gradient = jax.jit(
            lambda x: _gradient_of(ignoring_args(residual_and_jacobian))(x, None)
        )
        gauss_newton = jax.jit(
            lambda x: _gauss_newton_of(ignoring_args(residual_and_jacobian))(x, None)
        )

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
            _value=ignoring_args(jax.jit(objective)),
            _value_and_gradient=ignoring_args(value_and_gradient),
            _value_gradient_hessian=ignoring_args(value_gradient_hessian),
            _hessian_vector=ignoring_args(hessian_vector),
            _gauss_newton=ignoring_args(gauss_newton),
            _is_valid=None if is_valid is None else ignoring_args(jax.jit(is_valid)),
            _residual=ignoring_args(residual_fn),
            _residual_and_jacobian=ignoring_args(residual_and_jacobian),
            _segment_fns=tuple(map(ignoring_args, segment_fns)) or (ignoring_args(residual_fn),),
        )


class OptSolver(eqx.Module):
    """Base class for an optimizer that `ds.optimization.optsolve` runs.

    A subclass implements `solve` and sets `requires`. `ds.optimization.optsolve` refuses a
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
            problem: The [OptimizationProblem][diff_soup.optimization.OptimizationProblem] to minimize.
            callbacks: The [Callback][diff_soup.optimization.inference.Callback] objects to call during the run.

        Returns:
            The [OptResult][diff_soup.optimization.OptResult] of the run.
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
