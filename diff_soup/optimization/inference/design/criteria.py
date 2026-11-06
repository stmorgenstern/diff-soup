"""Experiment-design criteria: smooth scalars of a Fisher information matrix.

Every criterion takes the information matrix ``F`` of shape ``(n, n)`` over
the ``n`` analyzed parameters (from
[fisher_information][diff_soup.inference.design.fisher.fisher_information],
optionally with prior precision from
[fisher_with_prior][diff_soup.inference.design.criteria.fisher_with_prior]),
symmetrizes it to ``(F + F^T) / 2``, adds ``ridge * I`` and returns a scalar.
Each is built from log-determinants and Cholesky solves only, so ``jax.grad``
and ``jax.hessian`` with respect to ``F``, and through ``F`` with respect to
any design parameter it depends on, are defined everywhere on the positive
definite cone, repeated eigenvalues included (a differentiated
eigendecomposition is NaN there). Eigenvalues belong in host-side reports,
never on a gradient path.

* [d_optimal][diff_soup.inference.design.criteria.d_optimal]:
  ``log det F``, the log-volume of the information ellipsoid (larger is
  better).
* [ds_optimal][diff_soup.inference.design.criteria.ds_optimal]:
  ``log det F - log det F_nn``, the log-determinant of the Schur complement
  of the nuisance block: the information about a target subset with the
  other parameters profiled out (larger is better).
* [e_optimal_smooth][diff_soup.inference.design.criteria.e_optimal_smooth]:
  ``-(1/p) log tr(F^-p)``, a smooth lower surrogate of ``log lambda_min``,
  the worst-identified direction (larger is better).
* [integrated_variance][diff_soup.inference.design.criteria.integrated_variance]:
  ``tr(Sigma_target G)``, ``Sigma_target`` the target block of ``F^-1`` and
  ``G`` the Gram matrix of the block's basis: the integrated pointwise
  variance ``int Var[f(c)] dc`` of a function-valued parameter (smaller is
  better).
* [fisher_with_prior][diff_soup.inference.design.criteria.fisher_with_prior]:
  ``F + Lambda_0``, the information with a Gaussian prior's precision added.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array

__all__ = [
    "d_optimal",
    "ds_optimal",
    "e_optimal_smooth",
    "fisher_with_prior",
    "integrated_variance",
]


def _ridged(fim: Array, ridge: float) -> Array:
    """Return the symmetrized ``(fim + fimᵀ)/2 + ridge·I`` in ``fim``'s dtype.

    Symmetrizing first makes every criterion (and its gradient) invariant to
    the asymmetric numerical noise a computed FIM can carry.
    """
    fim = jnp.asarray(fim)
    fim = 0.5 * (fim + fim.T)
    if ridge == 0.0:
        return fim
    return fim + jnp.asarray(ridge, dtype=fim.dtype) * jnp.eye(fim.shape[-1], dtype=fim.dtype)


def d_optimal(fim: Array, *, ridge: float = 1e-10) -> Array:
    """Return the D-optimality criterion ``logdet(fim + ridge·I)``.

    Larger is better: the log-volume of the information ellipsoid. The ridge
    keeps the value and its gradient finite when the FIM is singular (an
    unidentifiable direction); report raw and regularized spectra side by side
    when the ridge is doing real work.

    Args:
        fim: Symmetric PSD matrix, shape ``(n, n)``.
        ridge: Small diagonal regularization added before the determinant.

    Returns:
        The scalar criterion value.
    """
    return jnp.linalg.slogdet(_ridged(fim, ridge))[1]


def ds_optimal(fim: Array, target: Any, *, ridge: float = 1e-10) -> Array:
    """Return the Ds-optimality criterion for a target parameter subset.

    ``logdet(F) − logdet(F_nuisance)``, which by the Schur determinant identity
    equals the log-determinant of the target block's Schur complement — the
    information about the target parameters after the nuisance parameters are
    profiled out. Use this when only a subset of ``wrt`` is of scientific
    interest and the rest must merely be tolerated.

    Args:
        fim: Symmetric PSD matrix, shape ``(n, n)``.
        target: Indices of the target parameters (a nonempty proper subset of
            ``range(n)``; static Python integers).
        ridge: Small diagonal regularization added before the determinants.

    Returns:
        The scalar criterion value.

    Raises:
        ValueError: If ``target`` is empty, out of range, contains duplicates,
            or covers every parameter (use
            [d_optimal][diff_soup.inference.design.criteria.d_optimal] for
            that).
    """
    fim = jnp.asarray(fim)
    n = fim.shape[-1]
    idx = tuple(int(i) for i in target)
    if len(idx) == 0:
        raise ValueError("ds_optimal: target must name at least one parameter index.")
    if len(set(idx)) != len(idx):
        raise ValueError(f"ds_optimal: target contains duplicate indices: {idx}.")
    if any(i < 0 or i >= n for i in idx):
        raise ValueError(f"ds_optimal: target indices {idx} out of range for a {n}-parameter FIM.")
    if len(idx) == n:
        raise ValueError(
            "ds_optimal: target covers every parameter; the nuisance block is "
            "empty. Use d_optimal for the full-parameter criterion."
        )
    nuisance = jnp.asarray([i for i in range(n) if i not in set(idx)])
    f = _ridged(fim, ridge)
    f_nn = f[jnp.ix_(nuisance, nuisance)]
    return jnp.linalg.slogdet(f)[1] - jnp.linalg.slogdet(f_nn)[1]


def e_optimal_smooth(fim: Array, *, p: int = 8, ridge: float = 1e-10) -> Array:
    """Return a smooth E-optimality surrogate ``−(1/p)·log tr((fim + ridge·I)^{−p})``.

    A smooth lower proxy of ``log λ_min`` that sharpens as ``p`` grows
    (``tr(F^{−p})^{−1/p} → λ_min``): maximizing it pushes up the
    worst-identified direction. Computed with one Cholesky factorization and
    ``p`` triangular solves — differentiable everywhere on the PD cone,
    including repeated eigenvalues.

    Args:
        fim: Symmetric PD matrix, shape ``(n, n)``.
        p: Sharpness of the surrogate (a small static integer; 8 is a good
            default — larger tracks ``log λ_min`` tighter but stiffens the
            design landscape).
        ridge: Small diagonal regularization added before factorizing.

    Returns:
        The scalar criterion value.

    Raises:
        ValueError: If ``p`` is not a positive integer.
    """
    if int(p) < 1:
        raise ValueError(f"e_optimal_smooth: p must be a positive integer; got {p}.")
    f = _ridged(fim, ridge)
    chol = jsl.cho_factor(f)
    x = jnp.eye(f.shape[-1], dtype=f.dtype)
    for _ in range(int(p)):
        x = jsl.cho_solve(chol, x)
    return -jnp.log(jnp.trace(x)) / int(p)


def integrated_variance(fim: Array, target: Any, *, gram: Array, ridge: float = 1e-10) -> Array:
    """Return the integrated pointwise variance of a function-valued block.

    For a function ``f(c) = Σ_i θ_i φ_i(c)`` whose coefficients occupy the
    ``target`` columns of the information matrix, the marginal coefficient
    covariance is the target block of ``fim⁻¹`` — which by the Schur identity
    already profiles the remaining (nuisance) parameters — and

    ``∫ Var[f(c)] dc  =  tr( Σ_target · G )``

    with ``G`` the basis Gram matrix over the function's own domain (e.g.
    ``Legendre.gram()``). Smaller is better: it is the I-optimality criterion
    for a function-valued parameter. Computed with one Cholesky factorization
    and column solves — no inverse and no eigendecomposition — so ``jax.grad``
    through a design parameterization is well-defined on the PD cone.

    Args:
        fim: Symmetric PD information matrix, shape ``(n, n)`` (add prior
            precision with
            [fisher_with_prior][diff_soup.inference.design.criteria.fisher_with_prior]
            first if the data alone
            leave null directions).
        target: Indices of the function block's coefficients (static Python
            integers; the whole parameter set is allowed).
        gram: The block's function-metric Gram matrix, shape ``(k, k)``.
        ridge: Small diagonal regularization added before factorizing.

    Returns:
        The scalar criterion value.

    Raises:
        ValueError: If ``target`` is empty, out of range, or contains
            duplicates, or ``gram`` does not match the block size.
    """
    fim = jnp.asarray(fim)
    n = fim.shape[-1]
    idx = tuple(int(i) for i in target)
    if len(idx) == 0:
        raise ValueError("integrated_variance: target must name at least one parameter index.")
    if len(set(idx)) != len(idx):
        raise ValueError(f"integrated_variance: target contains duplicate indices: {idx}.")
    if any(i < 0 or i >= n for i in idx):
        raise ValueError(
            f"integrated_variance: target indices {idx} out of range for a "
            f"{n}-parameter information matrix."
        )
    g = jnp.asarray(gram, dtype=fim.dtype)
    if g.shape != (len(idx), len(idx)):
        raise ValueError(
            f"integrated_variance: gram has shape {tuple(g.shape)} but the "
            f"target block has {len(idx)} coefficient(s); pass the block's own "
            "Gram matrix (e.g. Legendre.gram())."
        )
    f = _ridged(fim, ridge)
    chol = jsl.cho_factor(f)
    cols = (
        jnp.zeros((n, len(idx)), dtype=f.dtype).at[jnp.asarray(idx), jnp.arange(len(idx))].set(1.0)
    )
    sigma_block = jsl.cho_solve(chol, cols)[jnp.asarray(idx), :]
    return jnp.trace(sigma_block @ (0.5 * (g + g.T)))


def fisher_with_prior(fim: Array, prior_precision: Any) -> Array:
    """Return the Bayesian information ``fim + Λ₀``.

    Adds Gaussian prior precision to the Fisher information: a scalar adds
    ``λ·I`` (one isotropic precision), a vector adds ``diag(λ)`` (independent
    per-parameter priors), and a matrix adds ``Λ₀`` verbatim (a full prior
    covariance inverse). This is where parameter priors enter the design
    criteria.

    Args:
        fim: Symmetric PSD matrix, shape ``(n, n)``.
        prior_precision: Scalar, ``(n,)`` vector, or ``(n, n)`` matrix of prior
            precision (inverse prior covariance).

    Returns:
        The ``(n, n)`` regularized information matrix.
    """
    fim = jnp.asarray(fim)
    lam = jnp.asarray(prior_precision, dtype=fim.dtype)
    if lam.ndim == 0:
        return fim + lam * jnp.eye(fim.shape[-1], dtype=fim.dtype)
    if lam.ndim == 1:
        return fim + jnp.diag(lam)
    return fim + lam
