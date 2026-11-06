"""The solve of one primal-dual KKT system, behind an interface its factorization can change.

The interior point and SQP steps solve

    [ W + Sigma + delta_w I    A^T       ] [dx  ]   [r_primal]
    [ A                        -delta_c I] [dlam] = [r_dual  ]

The dense backend takes the condensed route: a Cholesky factorization of the primal block,
`delta_w` raised until it succeeds, then the Schur complement on the constraint rows. A banded
backend on cuDSS replaces only this class when multiple shooting or transcription needs it.

The primal block alone factors only when `W + Sigma` is positive definite, which is more than
the correct inertia asks: a nonconvex `W` whose reduced Hessian is positive definite needs no
regularization. When the plain block fails, the backend adds `A^T diag(w) A` to it, the exact
row operation that condenses each constraint row into the primal block with a weight `w`.
That block and its Schur complement both factor exactly when the KKT matrix has the inertia
`(n, m, 0)` (for a weight large enough), so `delta_w` is raised only for true negative
curvature, as Ipopt's inertia correction does.

A solver factors once per step with
[KKTSolve.factor_regularized][diff_soup.optimization.solvers.KKTSolve.factor_regularized], which runs
Ipopt's inertia correction, and solves every right-hand side of that step (the step, its
second-order corrections) with
[KKTSolve.solve_factored][diff_soup.optimization.solvers.KKTSolve.solve_factored].
"""

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

# Ipopt's inertia-correction constants (Wächter and Biegler 2006, Algorithm IC).
_FIRST_GROWTH = 100.0
_SHRINK = 1.0 / 3.0
_DELTA_W_MIN = 1e-20
# A Cholesky pivot below this fraction of its diagonal entry is a zero pivot: the matrix is
# singular to working precision and the factorization does not certify the inertia.
_PIVOT_RTOL = 1e-13
# The augmented route weights each constraint row so that its term in the primal block is
# this many times the curvature of the primal block along the row.
_RHO_REL = 1e3

# The attempts of one inertia correction, in order.
_PLAIN = 0
_AUGMENTED = 1
_SINGULAR = 2


class KKTFactor(eqx.Module):
    """A factored KKT matrix, reusable for every right-hand side of one step.

    The factor solves the equivalent system `[[P + A^T W A, A^T], [A, -D / E]]` in `(dx, E dlam)`,
    where `W` is the row weight of the augmented route (0 on the plain route) and `E = 1 - W D`.

    Attributes:
        primal: `W + Sigma + delta_w I`, kept for iterative refinement, `(n, n)`.
        jacobian: `A`, `(m, n)`.
        dual: The dual diagonal `D`, `delta_c` included, `(m,)`.
        weight: The row weights `W`, `(m,)`.
        scale: `E`, `(m,)`.
        lower_primal: The Cholesky factor of `P + A^T W A`, `(n, n)`.
        half_jacobian: `L^{-1} A^T`, `(n, m)`.
        lower_schur: The Cholesky factor of `A (P + A^T W A)^{-1} A^T + D / E`, `(m, m)`.
        primal_ok: Whether the primal block factored.
        schur_ok: Whether the Schur complement factored.
    """

    primal: Any
    jacobian: Any
    dual: Any
    weight: Any
    scale: Any
    lower_primal: Any
    half_jacobian: Any
    lower_schur: Any
    primal_ok: Any
    schur_ok: Any

    @property
    def ok(self) -> Any:
        """Whether the factorization certified the inertia `(n, m, 0)`."""
        return self.primal_ok & self.schur_ok


class KKTSolve(eqx.Module):
    """A KKT backend, which solves one regularized system and reports whether the inertia came out right.

    A backend implements `factor` and `solve_factored`, and `solve` and the inertia correction
    are built on them.
    """

    @abc.abstractmethod
    def factor(
        self,
        primal_block: Any,
        constraint_jacobian: Any,
        delta_w: Any,
        delta_c: Any,
        augmented: Any = False,
    ) -> Any:
        """Factor the KKT matrix at one regularization.

        Args:
            primal_block: `W + Sigma`, `(n, n)`.
            constraint_jacobian: `A`, `(m, n)`.
            delta_w: The primal regularization, a scalar.
            delta_c: The dual regularization, a scalar or one entry per row, `(m,)`.
            augmented: Whether to condense the constraint rows into the primal block, which
                certifies the inertia of a nonconvex block.

        Returns:
            The backend's factor, with an `ok` flag.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def solve_factored(self, factor: Any, rhs_primal: Any, rhs_dual: Any) -> tuple[Any, Any]:
        """Solve with a factor from `factor`, returning `(dx, dlam)`."""
        raise NotImplementedError

    def solve(
        self,
        primal_block: Any,
        constraint_jacobian: Any,
        delta_w: Any,
        delta_c: Any,
        rhs_primal: Any,
        rhs_dual: Any,
    ) -> tuple[Any, Any, Any]:
        """Solve the KKT system at the given regularization.

        Args:
            primal_block: `W + Sigma`, the Lagrangian Hessian plus the barrier diagonal, `(n, n)`.
            constraint_jacobian: `A`, the rows of every active constraint, `(m, n)`.
            delta_w: The primal regularization added to the diagonal.
            delta_c: The dual regularization.
            rhs_primal: The primal right-hand side, `(n,)`.
            rhs_dual: The dual right-hand side, `(m,)`.

        Returns:
            `(dx, dlam, inertia_ok)`: the primal and dual steps, and whether the
                factorization certified the inertia `(n, m, 0)` at this `delta_w`.
        """
        plain = self.factor(primal_block, constraint_jacobian, delta_w, delta_c)
        augmented = self.factor(primal_block, constraint_jacobian, delta_w, delta_c, True)
        factor = jax.tree.map(lambda a, b: jnp.where(plain.ok, a, b), plain, augmented)
        dx, dlam = self.solve_factored(factor, rhs_primal, rhs_dual)
        return dx, dlam, factor.ok

    def factor_regularized(
        self,
        primal_block: Any,
        constraint_jacobian: Any,
        delta_c: Any,
        delta_c_singular: Any,
        delta_w_last: Any,
    ) -> tuple[Any, Any]:
        """Factor with Ipopt's inertia correction.

        The first attempt is the plain route at `delta_w = 0`, the second the augmented route.
        A primal block that still does not factor raises `delta_w`, from `delta_w_init` (or a
        third of `delta_w_last`) by `delta_w_factor` (or 100 the first time). A Schur
        complement that does not factor, a rank-deficient `A`, adds `delta_c_singular` to
        every row once, then raises `delta_w`.

        Args:
            primal_block: `W + Sigma`, `(n, n)`.
            constraint_jacobian: `A`, `(m, n)`.
            delta_c: The dual diagonal the system always carries, `(m,)`.
            delta_c_singular: The dual regularization added for a singular `A`, a scalar.
            delta_w_last: The last nonzero `delta_w`, 0 if none.

        Returns:
            `(factor, delta_w)`; `factor.ok` is false when `delta_w` passed `delta_w_max`.
        """

        def attempt(delta_w, route):
            extra = jnp.where(route == _SINGULAR, delta_c_singular, 0.0)
            return self.factor(
                primal_block, constraint_jacobian, delta_w, delta_c + extra, route != _PLAIN
            )

        def grow(delta_w):
            first = jnp.where(
                delta_w_last == 0.0,
                self.delta_w_init,
                jnp.maximum(_DELTA_W_MIN, _SHRINK * delta_w_last),
            )
            factor = jnp.where(delta_w_last == 0.0, _FIRST_GROWTH, self.delta_w_factor)
            return jnp.where(delta_w == 0.0, first, factor * delta_w)

        def cond(carry):
            factor, delta_w, _ = carry
            return ~factor.ok & (delta_w <= self.delta_w_max)

        def body(carry):
            factor, delta_w, route = carry
            raise_w = (route == _SINGULAR) | ((route == _AUGMENTED) & ~factor.primal_ok)
            next_route = jnp.where(
                route == _PLAIN,
                _AUGMENTED,
                jnp.where((route == _AUGMENTED) & factor.primal_ok, _SINGULAR, route),
            )
            delta_w = jnp.where(raise_w, grow(delta_w), delta_w)
            return attempt(delta_w, next_route), delta_w, next_route

        zero = jnp.zeros((), dtype=primal_block.dtype)
        route = jnp.asarray(_PLAIN, dtype=jnp.int32)
        factor, delta_w, _ = lax.while_loop(cond, body, (attempt(zero, route), zero, route))
        return factor, delta_w


def _cholesky(matrix: Any) -> tuple[Any, Any]:
    """The lower Cholesky factor of `matrix` and whether it certifies positive definiteness."""
    size = matrix.shape[-1]
    if size == 0:
        return matrix, jnp.asarray(True)
    lower = jnp.linalg.cholesky(matrix)
    pivots = jnp.diagonal(lower) ** 2
    scale = jnp.maximum(jnp.abs(jnp.diagonal(matrix)), jnp.finfo(matrix.dtype).tiny)
    ok = jnp.all(jnp.isfinite(lower)) & jnp.all(pivots > _PIVOT_RTOL * scale)
    identity = jnp.eye(size, dtype=matrix.dtype)
    return jnp.where(ok, lower, identity), ok


def _row_weights(primal: Any, jacobian: Any, dual: Any) -> tuple[Any, Any]:
    """`(W, E)` of the augmented route: each row weighted past the block's curvature along it."""
    tiny = jnp.finfo(primal.dtype).tiny
    norms = jnp.sum(jacobian * jacobian, axis=1)
    along = jnp.abs(jnp.sum(jacobian * (jacobian @ primal), axis=1)) / jnp.maximum(norms, tiny)
    floor = jnp.maximum(jnp.mean(jnp.abs(jnp.diagonal(primal))), tiny)
    rho = jnp.where(
        norms > 0.0, _RHO_REL * jnp.maximum(along, floor) / jnp.maximum(norms, tiny), 0.0
    )
    scale = 1.0 / (1.0 + rho * dual)
    return rho * scale, scale


class DenseKKT(KKTSolve):
    """The dense KKT solve, by Cholesky of the primal block and the Schur complement on the constraint rows.

    When the primal block does not factor, its diagonal is raised until it does, as in Ipopt's
    inertia correction.

    Args:
        delta_w_init: The first primal regularization tried, `1e-4` by default.
        delta_w_max: The largest primal regularization before the step fails, `1e40` by default.
        delta_w_factor: The factor the regularization grows by, `8.0` by default.
    """

    delta_w_init: float = eqx.field(static=True, default=1e-4)
    delta_w_max: float = eqx.field(static=True, default=1e40)
    delta_w_factor: float = eqx.field(static=True, default=8.0)

    def factor(
        self,
        primal_block: Any,
        constraint_jacobian: Any,
        delta_w: Any,
        delta_c: Any,
        augmented: Any = False,
    ) -> KKTFactor:
        """Cholesky of the (augmented) primal block, then of the Schur complement."""
        size = primal_block.shape[-1]
        rows = constraint_jacobian.shape[0]
        dtype = primal_block.dtype
        primal = primal_block + delta_w * jnp.eye(size, dtype=dtype)
        dual = jnp.broadcast_to(jnp.asarray(delta_c, dtype=dtype), (rows,))
        weight, scale = _row_weights(primal, constraint_jacobian, dual)
        weight = jnp.where(augmented, weight, 0.0)
        scale = jnp.where(augmented, scale, 1.0)
        lower, primal_ok = _cholesky(
            primal + constraint_jacobian.T @ (weight[:, None] * constraint_jacobian)
        )
        half = jax.scipy.linalg.solve_triangular(lower, constraint_jacobian.T, lower=True)
        schur, schur_ok = _cholesky(half.T @ half + jnp.diag(dual / scale))
        return KKTFactor(
            primal,
            constraint_jacobian,
            dual,
            weight,
            scale,
            lower,
            half,
            schur,
            primal_ok,
            schur_ok,
        )

    def _solve_once(self, factor: KKTFactor, rhs_primal: Any, rhs_dual: Any) -> tuple[Any, Any]:
        """One pass through the factors, without refinement."""
        lower, half, schur = factor.lower_primal, factor.half_jacobian, factor.lower_schur
        rhs = rhs_primal + factor.jacobian.T @ (factor.weight * rhs_dual)
        first = jax.scipy.linalg.solve_triangular(lower, rhs, lower=True)
        if schur.shape[-1] == 0:
            dmu = jnp.zeros((0,), dtype=rhs_primal.dtype)
        else:
            dmu = jax.scipy.linalg.cho_solve((schur, True), half.T @ first - rhs_dual)
        dx = jax.scipy.linalg.solve_triangular(lower.T, first - half @ dmu, lower=False)
        return dx, dmu / factor.scale

    def solve_factored(self, factor: KKTFactor, rhs_primal: Any, rhs_dual: Any) -> tuple[Any, Any]:
        """Solve by the condensed route with one step of iterative refinement on the original system."""
        dx, dlam = self._solve_once(factor, rhs_primal, rhs_dual)
        residual_primal = rhs_primal - factor.primal @ dx - factor.jacobian.T @ dlam
        residual_dual = rhs_dual - factor.jacobian @ dx + factor.dual * dlam
        cx, clam = self._solve_once(factor, residual_primal, residual_dual)
        return dx + cx, dlam + clam

    def solve(
        self,
        primal_block: Any,
        constraint_jacobian: Any,
        delta_w: Any,
        delta_c: Any,
        rhs_primal: Any,
        rhs_dual: Any,
    ) -> tuple[Any, Any, Any]:
        """Solve by the condensed route, as [KKTSolve.solve][diff_soup.optimization.solvers.KKTSolve.solve] describes."""
        return super().solve(
            primal_block, constraint_jacobian, delta_w, delta_c, rhs_primal, rhs_dual
        )
