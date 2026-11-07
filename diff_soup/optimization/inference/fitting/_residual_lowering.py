"""The residual problem an inference problem lowers to: its stacked residual, its loss and derivatives, its windowed Jacobian."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp

from diff_soup.optimization._abc import LeastSquaresProblem
from diff_soup.optimization._binding import ignoring_args
from diff_soup.optimization._derived import half_sum_of_squares
from diff_soup.optimization.inference.fitting._forward_seam import (
    _FAILED_SENTINEL,
)
from diff_soup.optimization.inference.fitting._residual_memory import (
    _MemoryCheckedHessian,
    _TiledJacobian,
)
from diff_soup.optimization.inference.fitting.bounds import resolve_decision_bounds


def build_stacked_residual_fns(segment_fns: tuple) -> tuple:
    """Build the jitted ``(r_fn, rj_fn)`` pair over a tuple of segment closures.

    ``r_fn(p)`` concatenates the per-segment residuals; ``rj_fn(p)`` returns
    ``(r, J)`` via forward-mode. Built ONCE per inference problem and cached, so a
    repeated ``solve()`` — every outer iteration of a penalty loop — reuses the
    compiled kernels instead of retracing the PDE residual. Regularizer rows are
    deliberately NOT part of these callables: they evaluate eagerly in the
    solvers, so penalty weights can change between solves without recompiling.
    """
    # Local import avoids a fitting-to-solver module cycle at import time.
    from diff_soup.optimization.solvers.tangent import value_and_jacfwd

    @jax.jit
    def _r_fn(p):
        return jnp.concatenate([fn(p) for fn in segment_fns])

    @jax.jit
    def _rj_fn(p):
        return value_and_jacfwd(lambda p_: _r_fn(p_), p)

    return _r_fn, _rj_fn


def _stacked_residual_fn(segment_fns: tuple, batched_r: Optional[Callable] = None) -> Callable:
    """Build a jitted data-residual closure without differentiating it."""
    if batched_r is not None:
        return batched_r

    @jax.jit
    def residual(x):
        return jnp.concatenate([fn(x) for fn in segment_fns])

    return residual


def _loss_fn_from_residual(residual_fn: Callable) -> Callable:
    """Build the scalar data-misfit loss from a residual closure."""

    @jax.jit
    def loss(x):
        residual = residual_fn(x)
        return half_sum_of_squares(residual)

    return loss


def _loss_and_grad_from_residual(residual_fn: Callable, ad: str) -> Callable:
    """Build one scalar-loss value-and-gradient seam."""
    if ad == "forward":
        from diff_soup.optimization.solvers.tangent import value_and_jacfwd

        @jax.jit
        def forward_loss_and_grad(x):
            residual, jacobian = value_and_jacfwd(residual_fn, x)
            loss = half_sum_of_squares(residual)
            grad = jacobian.T @ residual
            failed = jnp.any(jnp.abs(residual) >= _FAILED_SENTINEL)
            return loss, jnp.where(failed, jnp.zeros_like(grad), grad)

        return forward_loss_and_grad

    @jax.jit
    def reverse_loss_and_grad(x):
        def objective(point):
            residual = residual_fn(point)
            loss = half_sum_of_squares(residual)
            failed = jnp.any(jnp.abs(residual) >= _FAILED_SENTINEL)
            return loss, failed

        (loss, failed), grad = jax.value_and_grad(objective, has_aux=True)(x)
        # A zero cotangent entering a failed time loop can still encounter NaN
        # during replay. Select after the backward pass so it cannot escape.
        return loss, jnp.where(failed, jnp.zeros_like(grad), grad)

    return reverse_loss_and_grad


def _loss_grad_hessian_from_gradient(loss_and_grad_fn: Callable) -> Callable:
    """Build a forward derivative of a value-and-gradient seam."""

    @jax.jit
    def loss_grad_hessian(x):
        def gradient_with_aux(point):
            loss, grad = loss_and_grad_fn(point)
            return grad, (loss, grad)

        hessian, (loss, grad) = jax.jacfwd(gradient_with_aux, has_aux=True)(x)
        return loss, grad, hessian

    return _MemoryCheckedHessian(loss_grad_hessian, tile="full")


def _estimation_window_jacobian(problem: Any) -> Optional[_TiledJacobian]:
    """The structured Jacobian of an estimation problem over fitted window starts, or ``None``.

    Its residual is the data rows of its data problem followed by the MAP prior
    rows; the data rows take the data problem's window-block Jacobian.
    """
    data = getattr(problem, "diagnostic_problem", None)
    blocks = getattr(problem, "_prior_blocks", None)
    if data is None or data is problem or blocks is None:
        return None
    if not getattr(data, "fits_window_states", False) or data.batched_r_fn is not None:
        return None
    data_jacobian = data.stacked_rj_fn
    if not isinstance(data_jacobian, _TiledJacobian):
        return None

    def prior_rows(x):
        return jnp.concatenate(
            [jnp.zeros((0,), dtype=x.dtype)] + [prior.residual(x[c]) for c, prior in blocks]
        )

    def build(tile):
        data_rows = data_jacobian._build(tile)

        def residual_and_jacobian(x):
            residual, jacobian = data_rows(x)
            if not blocks:
                return residual, jacobian
            rows = prior_rows(x)
            return (
                jnp.concatenate((residual, rows)),
                jnp.concatenate((jacobian, jax.jacfwd(prior_rows)(x)), axis=0),
            )

        return residual_and_jacobian

    return _TiledJacobian(build)


def lower_residual_problem(
    problem: Any,
    optimizer: Any,
    *,
    evaluator: Any = None,
    lanes: Any = None,
    gauss_newton: Optional[Callable] = None,
    value: Optional[Callable] = None,
    regularization: Any = None,
    joint_view: Optional[Callable] = None,
) -> LeastSquaresProblem:
    """The [LeastSquaresProblem][diff_soup.optimization.LeastSquaresProblem] of a residual problem.

    ``problem`` is an inference problem of residual rows: an
    [InferenceProblem][diff_soup.optimization.inference.fitting.inference_problem.InferenceProblem],
    a joint problem, or a view of one. The result's residual is its
    ``batched_r_fn`` (else ``stacked_r_fn``, else the concatenation of its
    ``segment_fns``) with the matching Jacobian; its value, gradient and exact
    Hessian are ``evaluator.loss_fn``, ``evaluator.loss_and_grad_fn(ad)`` and
    ``evaluator.loss_grad_hessian_fn(composition)`` when the evaluator has
    them, evaluated on first use, and are derived from the residual
    otherwise. ``ad`` and ``composition`` come from ``optimizer`` (its ``ad``
    and ``composition``; a solver that requires ``"hessian"`` and names no
    ``ad`` differentiates its gradient in the mode the composition's inner
    derivative uses). The bounds are resolved once.

    Args:
        problem: The residual problem; its ``x0``, ``regularization``,
            ``bounds``, ``unflatten_fn``, ``prob_static``, ``segment_fns``,
            ``n_segments``, ``n_residuals`` and ``trial_is_valid`` are read.
        optimizer: The solver the problem is lowered for.
        evaluator: The object whose seams evaluate the objective; ``None``
            is ``problem``.
        lanes: The lowered study that evaluates the objective on its lanes,
            or ``None``; with one the structure holds ``"batched_lanes"``.
        gauss_newton: ``x -> (cost, J^T r, J^T J, failed)`` evaluated without
            forming the Jacobian, or ``None``: the lanes' ``gauss_newton_fn``
            when given, the reduced statistics of a sharded problem, else
            ``J^T J`` of the residual's Jacobian.
        value: ``x -> `` the objective, NaN where an evaluation failed, to use
            in place of ``evaluator.loss_fn``: the cost reduced across the
            mesh for a sharded problem; ``None`` reads ``evaluator.loss_fn``,
            or a sharded problem's ``evaluate_cost_fn``.
        regularization: The regularization the solver adds; ``None`` is
            ``problem.regularization``.
        joint_view: ``() -> `` the joint problem with its level structure,
            for ``"batched_lanes"``.
    """
    evaluator = problem if evaluator is None else evaluator
    requires = frozenset(getattr(optimizer, "requires", ()))
    asked = getattr(optimizer, "composition", "auto")
    resolve = getattr(evaluator, "resolved_hessian_composition", None)
    composition = None
    if "hessian" in requires and callable(resolve):
        composition = resolve(asked)
    ad = getattr(optimizer, "ad", None)
    if ad is None:
        ad = "forward" if composition not in (None, "forward-over-reverse") else "reverse"

    residual = getattr(problem, "batched_r_fn", None)
    residual_and_jacobian = getattr(problem, "batched_rj_fn", None)
    if residual is None or residual_and_jacobian is None:
        residual = getattr(problem, "stacked_r_fn", None)
        residual_and_jacobian = getattr(problem, "stacked_rj_fn", None)
    if residual is None or residual_and_jacobian is None:
        residual, residual_and_jacobian = build_stacked_residual_fns(tuple(problem.segment_fns))
    # Fitted window starts: the data rows take the window-block Jacobian.
    window_jacobian = _estimation_window_jacobian(problem)
    if window_jacobian is not None:
        residual_and_jacobian = window_jacobian
    if getattr(evaluator, "loss_fn", None) is None:
        evaluator = _ResidualSeams(residual)

    if value is None and lanes is None and getattr(problem, "is_sharded", False):
        evaluate_cost = getattr(problem, "evaluate_cost_fn", None)
        if evaluate_cost is not None:
            value = _reduced_cost(evaluate_cost)
    if value is None:

        def value(x):
            return evaluator.loss_fn(x)

    def value_and_gradient(x):
        return evaluator.loss_and_grad_fn(ad)(x)

    def value_gradient_hessian(x):
        return evaluator.loss_grad_hessian_fn(asked if composition is None else composition)(x)

    hessian_vector = None
    if lanes is None:

        def hessian_vector(x, v):
            gradient = evaluator.loss_and_grad_fn(ad)
            return jax.jvp(lambda y: gradient(y)[1], (x,), (v,))[1]

    predicate = getattr(problem, "trial_is_valid", None)
    is_valid = None
    if callable(predicate):

        def is_valid(x):
            return problem.trial_is_valid(x)

    structure = {"value_gradient", "residual_jacobian", "gauss_newton"}
    if getattr(evaluator, "supported_compositions", frozenset()):
        structure.add("hessian")
    if lanes is not None:
        structure.add("batched_lanes")
    reduced = gauss_newton is not None
    if gauss_newton is None and lanes is None and getattr(problem, "is_sharded", False):
        evaluate = getattr(problem, "evaluate_lsq_fn", None)
        if evaluate is not None:
            gauss_newton, reduced = _reduced_statistics(evaluate), True
    regularization = (
        getattr(problem, "regularization", None) if regularization is None else regularization
    )
    n_residuals = getattr(problem, "n_residuals", None)
    segment_fns = tuple(getattr(problem, "segment_fns", ()) or ())
    return LeastSquaresProblem(
        x0=problem.x0,
        structure=frozenset(structure),
        _unflatten=getattr(problem, "unflatten_fn", None),
        _static=getattr(problem, "prob_static", None),
        bounds=resolve_decision_bounds(problem),
        _regularization=regularization,
        lanes=lanes,
        ad=ad,
        hessian_composition=composition,
        n_segments=int(getattr(problem, "n_segments", 1) or 1),
        reduced_gauss_newton=reduced,
        # The closures read no args: an inference problem's lanes differ in their data,
        # which the lowering closes over.
        _value=ignoring_args(value),
        _value_and_gradient=ignoring_args(value_and_gradient),
        _value_gradient_hessian=ignoring_args(value_gradient_hessian),
        _hessian_vector=ignoring_args(hessian_vector),
        _gauss_newton=ignoring_args(gauss_newton),
        _is_valid=ignoring_args(is_valid),
        _joint_view=joint_view,
        _residual=ignoring_args(residual),
        _residual_and_jacobian=ignoring_args(residual_and_jacobian),
        _segment_fns=tuple(map(ignoring_args, segment_fns)),
        n_residuals=None if n_residuals is None else int(n_residuals),
    )


def _reduced_cost(evaluate_cost: Callable) -> Callable:
    """``x -> `` the cost from a reduced ``evaluate_cost_fn``, NaN where a child failed."""

    def value(x):
        cost, _rows, failed = evaluate_cost(jnp.asarray(x))
        return jnp.where(failed == 0, cost, jnp.nan)

    return value


def _reduced_statistics(evaluate: Callable) -> Callable:
    """``x -> (cost, J^T r, J^T J, failed)`` from a reduced ``evaluate_lsq_fn``."""

    def gauss_newton(x):
        cost, gradient, matrix, _rows, failed = evaluate(jnp.asarray(x))
        return cost, gradient, matrix, failed

    return gauss_newton


class _ResidualSeams:
    """The scalar seams of a residual problem that defines none of its own."""

    def __init__(self, residual: Callable):
        self.loss_fn = _loss_fn_from_residual(residual)
        self._residual = residual
        self._gradients: dict = {}
        self.supported_compositions = frozenset({"forward-over-forward"})

    def loss_and_grad_fn(self, ad: str = "reverse") -> Callable:
        if ad not in self._gradients:
            self._gradients[ad] = _loss_and_grad_from_residual(self._residual, ad)
        return self._gradients[ad]

    def loss_grad_hessian_fn(self, composition: str = "auto") -> Callable:
        if "hessian" not in self._gradients:
            self._gradients["hessian"] = _loss_grad_hessian_from_gradient(
                self.loss_and_grad_fn("forward")
            )
        return self._gradients["hessian"]
