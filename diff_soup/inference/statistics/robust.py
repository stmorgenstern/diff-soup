"""robust.py — misspecification-robust frequentist covariance.

A sandwich covariance estimates the repeated-sampling variability of a
least-squares estimator when the declared weights may be wrong: the bread is
the Gauss-Newton curvature ``Jᵀ J`` and the meat is built from the observed
score contributions, so over- or under-stated noise inflates the result
instead of silently disappearing. It is a **frequentist repeated-sampling
covariance under stated independence assumptions** — it is not a posterior,
and neither it nor wider intervals repair biased parameters from a wrong
physical model.

Initial assumptions supported: independent observation rows, or independent
clusters of rows (every row of one entity correlated arbitrarily, distinct
entities independent).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup.inference.statistics.report import IdentifiedSubspace, ImproperPosteriorError

__all__ = ["RobustCovariance", "sandwich_covariance"]


@dataclasses.dataclass(frozen=True)
class RobustCovariance:
    """A sandwich covariance and the assumptions it stands on. Frozen.

    Attributes:
        covariance: The ``(n, n)`` robust covariance estimate.
        kind: ``"independent"`` (each residual row independent) or
            ``"cluster"`` (rows independent across clusters only).
        n_rows: Number of residual rows scored.
        n_clusters: Number of independent clusters (equals ``n_rows`` for the
            independent kind).
        interpretation: Plain-words statement of what the numbers mean.
    """

    covariance: np.ndarray
    kind: str
    n_rows: int
    n_clusters: int
    interpretation: str


def sandwich_covariance(
    residual: Any,
    x: Array,
    *,
    clusters: Optional[Any] = None,
    null_tol: float = 1e-10,
    names: tuple = (),
) -> RobustCovariance:
    """Estimate the estimator's repeated-sampling covariance by a sandwich.

    Computes ``B⁻¹ M B⁻¹`` with bread ``B = Jᵀ J`` and meat ``M`` assembled
    from the observed per-row (or per-cluster) score contributions
    ``g = Jᵀ r`` at ``x``. Solves use the SVD of ``J``; no matrix is ever
    inverted directly, and an undetermined direction raises rather than
    producing invented variances.

    Args:
        residual: The whitened residual closure ``x -> r`` (e.g. a
            problem's residual; rows already carry their declared ``1/σ``
            weights). Differentiated with ``jax.jacfwd``.
        x: The estimate to evaluate at (typically the converged fit).
        clusters: Optional integer label per residual row; rows sharing a
            label form one independent cluster. ``None`` treats every row as
            independent.
        null_tol: Relative information threshold for declaring a bread
            direction undetermined.
        names: Optional parameter labels for the failure report.

    Returns:
        A [RobustCovariance][diff_soup.inference.statistics.robust.RobustCovariance].

    Raises:
        ImproperPosteriorError: If the bread ``Jᵀ J`` is rank-deficient — the
            estimator itself is undetermined along the null directions.
    """
    x = jnp.asarray(x)
    r = np.asarray(residual(x))
    j = np.asarray(jax.jacfwd(residual)(x))
    n_rows, n_params = j.shape

    _, s_thin, vt = np.linalg.svd(j, full_matrices=True)
    s = np.zeros(n_params)
    s[: s_thin.size] = s_thin
    _determined(s**2, vt.T, null_tol=null_tol, names=names)

    if clusters is None:
        scores = j * r[:, None]
        n_clusters = n_rows
        kind = "independent"
        assumption = "independent observation rows"
    else:
        labels = np.asarray(clusters).reshape(-1)
        if labels.shape[0] != n_rows:
            raise ValueError(
                f"sandwich_covariance: {labels.shape[0]} cluster labels for {n_rows} residual rows."
            )
        uniq = np.unique(labels)
        scores = np.stack([(j[labels == c] * r[labels == c, None]).sum(axis=0) for c in uniq])
        n_clusters = int(uniq.shape[0])
        kind = "cluster"
        assumption = (
            f"{n_clusters} independent clusters; rows within a cluster may be "
            "arbitrarily correlated"
        )

    return _sandwich(s**2, vt.T, scores, n_rows=n_rows, kind=kind, assumption=assumption)


def _cluster_sandwich_covariance(
    bread: Any,
    scores: Any,
    *,
    n_rows: int,
    null_tol: float = 1e-10,
    names: tuple = (),
) -> RobustCovariance:
    """The cluster sandwich ``B⁻¹ M B⁻¹`` from the bread and the clusters' scores.

    The same estimate as ``sandwich_covariance(..., clusters=...)`` for a
    problem that sums ``Jᵀ J`` and every cluster's score ``J_cᵀ r_c`` itself
    (a lowered study's lanes) and forms no residual Jacobian: the solves use
    the eigendecomposition of the bread.

    Args:
        bread: The bread ``Jᵀ J``, ``(n, n)``.
        scores: One score ``J_cᵀ r_c`` per cluster, ``(C, n)``, in cluster
            order.
        n_rows: The number of residual rows scored.
        null_tol: Relative information threshold for declaring a bread
            direction undetermined.
        names: Optional parameter labels for the failure report.

    Returns:
        A [RobustCovariance][diff_soup.inference.statistics.robust.RobustCovariance].

    Raises:
        ImproperPosteriorError: If the bread is rank-deficient.
    """
    bread = np.asarray(bread, dtype=float)
    scores = np.asarray(scores, dtype=float)
    squares, v = np.linalg.eigh(0.5 * (bread + bread.T))
    order = np.argsort(squares)[::-1]
    squares, v = np.clip(squares[order], 0.0, None), v[:, order]
    _determined(squares, v, null_tol=null_tol, names=names)
    n_clusters = int(scores.shape[0])
    assumption = (
        f"{n_clusters} independent clusters; rows within a cluster may be arbitrarily correlated"
    )
    return _sandwich(squares, v, scores, n_rows=n_rows, kind="cluster", assumption=assumption)


def _determined(squares: np.ndarray, v: np.ndarray, *, null_tol: float, names: tuple) -> None:
    """Raise unless the bread ``V diag(squares) Vᵀ`` (descending) determines every direction."""
    n_params = v.shape[0]
    largest = float(squares[0]) if n_params else 0.0
    rank = int(np.sum(squares > null_tol * largest))
    if rank < n_params:
        raise ImproperPosteriorError(
            f"sandwich_covariance: the residual Jacobian leaves "
            f"{n_params - rank} of {n_params} parameter directions undetermined, "
            "so the estimator has no finite repeated-sampling covariance there. "
            "Pin or re-parameterize the undetermined directions first.",
            IdentifiedSubspace(
                directions=v[:, :rank],
                variances=1.0 / squares[:rank] if rank else np.zeros(0),
                null_directions=v[:, rank:],
                names=tuple(names),
            ),
        )


def _sandwich(
    squares: np.ndarray,
    v: np.ndarray,
    scores: np.ndarray,
    *,
    n_rows: int,
    kind: str,
    assumption: str,
) -> RobustCovariance:
    """``B⁻¹ M B⁻¹`` from the bread ``V diag(squares) Vᵀ`` and the scores."""
    meat = scores.T @ scores
    # bread_inv @ meat @ bread_inv via the factors: B^{-1} = V diag(1/s^2) V^T.
    half = (v / squares) @ (v.T @ meat)
    covariance = half @ (v / squares) @ v.T

    return RobustCovariance(
        covariance=covariance,
        kind=kind,
        n_rows=n_rows,
        n_clusters=int(scores.shape[0]),
        interpretation=(
            "frequentist repeated-sampling covariance of the least-squares "
            f"estimator under {assumption}; robust to a misstated noise scale, "
            "not a Bayesian posterior, and no protection against a biased "
            "physical model"
        ),
    )
