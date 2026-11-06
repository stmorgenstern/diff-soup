"""Handle-aware cluster-robust covariance for fitted Studies."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from diff_soup.inference.fitting.bounds import resolve_decision_bounds
from diff_soup.inference.fitting.result import OptResult
from diff_soup.inference.statistics.estimation import MAP, EstimationProblem
from diff_soup.inference.statistics.identifiability import (
    IdentifiabilityCoordinate,
    _require_problem_schema,
    _schema_coordinates,
)
from diff_soup.inference.statistics.robust import _cluster_sandwich_covariance, sandwich_covariance


@dataclasses.dataclass(frozen=True)
class ClusterResidualBlock:
    """One contiguous measurement block in the data residual's exact row order."""

    experiment_id: Any
    measurement: str
    cluster_id: Any
    row_slice: tuple[int, int]
    n_kept_frames: int
    rows_per_frame: int


@dataclasses.dataclass(frozen=True)
class ClusterCovarianceResult:
    """Auditable cluster-sandwich covariance in fitted physical coordinates.

    Rows from the same declared Study cluster are allowed arbitrary dependence;
    distinct cluster IDs are assumed independent.  This is an asymptotic
    frequentist repeated-sampling covariance of the fitted estimator (MLE or
    MAP, as ``objective_kind`` says), not a posterior covariance.
    """

    covariance: np.ndarray
    coordinates: tuple[IdentifiabilityCoordinate, ...]
    parameter_schema: Any
    schema_fingerprint: str
    study_fingerprint: str
    objective_kind: str
    evaluated_params: np.ndarray
    cluster: str
    cluster_ids: tuple
    cluster_row_counts: tuple[int, ...]
    row_cluster_ids: tuple
    residual_blocks: tuple[ClusterResidualBlock, ...]
    n_rows: int
    n_clusters: int
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    interpretation: str
    fit_status: str
    fit_converged: bool

    @property
    def names(self) -> tuple[str, ...]:
        """Owner-aware physical-coordinate names matching covariance axes."""
        return tuple(coordinate.name for coordinate in self.coordinates)


def _validate_problem_fit(
    problem: Any,
    fit: Any,
) -> tuple[Any, np.ndarray, tuple[IdentifiabilityCoordinate, ...], str, str]:
    """Validate that a fitted vector still belongs to this exact Study schema."""
    where = "cluster_covariance"
    if not isinstance(problem, EstimationProblem):
        raise TypeError(f"{where} first argument must be an InferenceProblem.")
    if not isinstance(fit, OptResult):
        raise TypeError(f"{where} fit must be an optimization OptResult.")
    if fit.parameter_schema is None:
        raise ValueError(
            f"{where} requires an OptResult with handle-based parameter schema "
            "provenance; fit the supplied InferenceProblem through diff_soup.inference.optsolve(...)."
        )

    schema = problem.parameter_schema
    _require_problem_schema(problem, fit.parameter_schema, where=where)
    objective_kind = type(problem.objective).__name__
    if type(fit.objective).__name__ != objective_kind:
        raise ValueError(
            f"{where}: fit objective is {type(fit.objective).__name__!r}, but the problem "
            f"objective is {objective_kind!r}."
        )
    uq_info = dict(fit.info).get("uq", {})
    if not isinstance(uq_info, Mapping):
        raise ValueError(f"{where}: OptResult has malformed UQ provenance metadata.")
    fit_fingerprint = uq_info.get("study_fingerprint")
    if fit_fingerprint != problem.study.fingerprint:
        raise ValueError(f"{where}: OptResult Study provenance does not match the problem.")
    if int(fit.n_failed_segments) != 0:
        raise ValueError(
            f"{where} refuses a fit with failed forward-solve segments; failed "
            f"indices are {tuple(fit.failed_segment_indices)}."
        )

    params = np.asarray(fit.params, dtype=float)
    expected_shape = (int(problem.n_params),)
    if params.shape != expected_shape:
        raise ValueError(
            f"{where}: fitted parameter vector has shape {params.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(params)):
        raise ValueError(f"{where}: fitted parameter vector is non-finite.")
    coordinates = _schema_coordinates(schema, params.size)
    status = getattr(fit.status, "value", str(fit.status))
    return schema, params, coordinates, objective_kind, status


def _cluster_residual_rows(
    problem: EstimationProblem,
    cluster: str,
) -> tuple[tuple, tuple[ClusterResidualBlock, ...]]:
    """Mirror Study -> child -> measurement -> C-order residual flattening."""
    if not isinstance(cluster, str) or not cluster:
        raise TypeError("cluster_covariance cluster must be a non-empty string.")
    if cluster not in problem.study.clusters:
        raise ValueError(
            f"cluster_covariance: unknown Study cluster {cluster!r}; available "
            f"clusters are {tuple(problem.study.clusters)}."
        )

    owner_ids = problem.study.clusters[cluster]
    row_ids: list[Any] = []
    blocks: list[ClusterResidualBlock] = []
    row_start = 0
    for experiment_id, experiment, owner_id in zip(
        problem.study.experiment_ids,
        problem.study.experiments,
        owner_ids,
    ):
        for measurement in experiment.measurements:
            shape = measurement.values_shape()
            if measurement.mask is None:
                n_kept_frames = int(shape[0])
            else:
                n_kept_frames = int(np.count_nonzero(np.asarray(measurement.mask, dtype=bool)))
            rows_per_frame = int(np.prod(shape[1:], dtype=int)) if len(shape) > 1 else 1
            n_rows = n_kept_frames * rows_per_frame
            row_stop = row_start + n_rows
            blocks.append(
                ClusterResidualBlock(
                    experiment_id=experiment_id,
                    measurement=measurement.name,
                    cluster_id=owner_id,
                    row_slice=(row_start, row_stop),
                    n_kept_frames=n_kept_frames,
                    rows_per_frame=rows_per_frame,
                )
            )
            row_ids.extend([owner_id] * n_rows)
            row_start = row_stop
    return tuple(row_ids), tuple(blocks)


def _integer_cluster_codes(row_ids: tuple) -> tuple[np.ndarray, tuple, tuple[int, ...]]:
    """Encode arbitrary hashable owner IDs without requiring them to be sortable."""
    ids: list[Any] = []
    lookup: dict[Any, int] = {}
    codes = np.empty(len(row_ids), dtype=int)
    for row, owner_id in enumerate(row_ids):
        if owner_id not in lookup:
            lookup[owner_id] = len(ids)
            ids.append(owner_id)
        codes[row] = lookup[owner_id]
    counts = tuple(int(value) for value in np.bincount(codes, minlength=len(ids)))
    return codes, tuple(ids), counts


def cluster_covariance(
    problem: EstimationProblem,
    fit: OptResult,
    *,
    cluster: str,
    null_tol: float = 1e-10,
) -> ClusterCovarianceResult:
    """Estimate the cluster-robust sandwich covariance of a fit, with each cluster an independent sampling unit.

    Rows within a cluster may be correlated, and the declared noise scale need not be right. For
    a `MAP` fit the result is the spread of the estimator, not a posterior.

    Args:
        problem: The [`InferenceProblem`][diff_soup.inference.InferenceProblem] or
            [`LoweredStudy`][diff_soup.inference.LoweredStudy] the fit was made on.
        fit: The [`OptResult`][diff_soup.inference.OptResult] of `ds.inference.optsolve` on
            `problem`.
        cluster: The name of a cluster of the problem's study, with at least two distinct labels.
        null_tol: The curvature, relative to the largest, at or below which a direction counts
            as undetermined, which raises `ImproperPosteriorError`.

    Returns:
        A `ClusterCovarianceResult` holding the `covariance`, the cluster labels, and warnings
            such as too few clusters.
    """
    schema, params, coordinates, objective_kind, status = _validate_problem_fit(problem, fit)
    penalized = isinstance(problem.objective, MAP)
    if bool(getattr(problem.diagnostic_problem, "is_sharded", False)):
        raise NotImplementedError(
            "cluster_covariance currently requires a materialized data residual "
            "and Jacobian; recompile the Study without a sharded layout."
        )

    row_ids, blocks = _cluster_residual_rows(problem, cluster)
    codes, cluster_ids, counts = _integer_cluster_codes(row_ids)
    if len(cluster_ids) < 2:
        raise ValueError(
            "cluster_covariance needs at least two independent clusters; one "
            "cluster has no between-cluster replication and can produce a "
            "spuriously zero score-meat matrix at an optimum."
        )
    names = tuple(coordinate.name for coordinate in coordinates)
    # A lowered study gives the bread and every experiment's score from its lanes; the
    # sandwich is then formed from them, with no residual Jacobian.
    lanes = getattr(problem, "_experiment_scores", None)
    pieces = lanes(params) if callable(lanes) else None
    jacobian = None
    if pieces is None and penalized:
        values, jacobian = problem.data_residual_jacobian(jnp.asarray(params))
        residual = np.ravel(np.asarray(values, dtype=float))
        jacobian = np.asarray(jacobian, dtype=float).reshape(residual.size, params.size)
        produced = residual.size
    elif pieces is None:
        residual = np.ravel(np.asarray(problem._data_r(jnp.asarray(params)), dtype=float))
        produced = residual.size
    else:
        bread, experiment_scores, experiment_rows = pieces
        residual = experiment_scores
        produced = int(np.sum(experiment_rows))
    if produced != len(row_ids):
        raise ValueError(
            "cluster_covariance: Study measurement shapes/masks imply "
            f"{len(row_ids)} residual rows, but the compiled data-only problem "
            f"produced {produced}. Refusing a potentially misaligned cluster "
            "covariance."
        )
    if not np.all(np.isfinite(residual)):
        raise ValueError("cluster_covariance: data residual is non-finite at the fitted point.")
    if jacobian is not None and not np.all(np.isfinite(jacobian)):
        raise ValueError(
            "cluster_covariance: the data residual's Jacobian is non-finite at the fitted point."
        )

    if pieces is None and not penalized:
        robust = sandwich_covariance(
            problem._data_r,
            jnp.asarray(params),
            clusters=codes,
            null_tol=null_tol,
            names=names,
        )
    else:
        if pieces is None:
            bread = jacobian.T @ jacobian
            scores = np.zeros((len(cluster_ids), params.size))
            np.add.at(scores, codes, jacobian * residual[:, None])
        else:
            position = {owner: index for index, owner in enumerate(cluster_ids)}
            owners = problem.study.clusters[cluster]
            scores = np.zeros((len(cluster_ids), experiment_scores.shape[1]))
            for experiment, owner in enumerate(owners):
                # An owner whose experiments have no rows holds no cluster and scores nothing.
                if owner in position:
                    scores[position[owner]] += experiment_scores[experiment]
        if penalized:
            # The bread of a MAP fit is the curvature of the objective it minimized: the
            # prior's Hessian joins the data's J^T J. The prior is the same in every
            # repeated sample, so it adds no score to the meat.
            prior = np.asarray(problem.prior_precision(jnp.asarray(params)), dtype=float)
            bread = np.asarray(bread, dtype=float) + prior
        robust = _cluster_sandwich_covariance(
            bread, scores, n_rows=produced, null_tol=null_tol, names=names
        )

    n_clusters = len(cluster_ids)
    caution: list[str] = []
    if not fit.converged:
        caution.append(
            f"The supplied fit did not converge (status {status!r}). A sandwich "
            "covariance assumes the estimator's score equation has been solved; "
            "this matrix is only a terminal-point diagnostic."
        )
    resolved_bounds = resolve_decision_bounds(problem)
    if resolved_bounds is not None:
        lower, upper = (np.asarray(value, dtype=float) for value in resolved_bounds)
        active = tuple(
            coordinates[index].name
            for index, value in enumerate(params)
            if (np.isfinite(lower[index]) and np.isclose(value, lower[index]))
            or (np.isfinite(upper[index]) and np.isclose(value, upper[index]))
        )
        if active:
            caution.append(
                f"Fitted coordinate(s) {active} are on a finite bound. The ordinary "
                "unconstrained sandwich/Wald approximation does not represent the "
                "one-sided or mixture asymptotics of a boundary-active estimator."
            )
    if n_clusters < 30:
        caution.append(
            f"Only {n_clusters} independent clusters are available. Cluster-sandwich "
            "covariance is a large-cluster approximation, and this implementation "
            "does not apply a small-sample or degrees-of-freedom correction; interval "
            "coverage may be poor. The value 30 is a cautionary rule of thumb, not "
            "an identifiability threshold."
        )
    if n_clusters <= len(coordinates):
        caution.append(
            f"The {n_clusters} clusters do not outnumber the {len(coordinates)} fitted "
            "coordinates. The cluster score meat has rank at most the number of "
            "clusters (and typically at most G-1 when an interior MLE score equation "
            "is solved), so covariance estimates are especially fragile even when the "
            "bread is full rank."
        )

    assumptions = (
        f"Study cluster {cluster!r} defines the independent sampling units; distinct "
        "cluster IDs are independent and rows within one ID may be arbitrarily correlated.",
        "Residual rows are whitened by the declared fixed Gaussian measurement scales.",
        "The estimator is locally regular and the number of independent clusters is "
        "large enough for the asymptotic sandwich approximation.",
        "No finite-cluster bias correction, leverage correction, or degrees-of-freedom "
        "adjustment is applied.",
        "Robust covariance addresses within-cluster error dependence and scale "
        "misspecification; it does not correct physical-model bias or a wrongly chosen "
        "clustering level.",
        (
            "The bread is the data's J^T J plus the prior's Hessian at the fit; only the "
            "data's scores enter the meat, the prior being the same in every repeated "
            "sample. The result is the spread of the MAP estimator, not a posterior, and "
            "it leaves out the estimator's bias toward the prior."
            if penalized
            else "Only response-data residuals enter the bread and meat."
        ),
    )
    if robust.n_rows != produced or robust.n_clusters != n_clusters:
        raise RuntimeError(
            "cluster_covariance: internal sandwich row/cluster accounting disagrees "
            "with Study provenance."
        )

    return ClusterCovarianceResult(
        covariance=np.asarray(robust.covariance),
        coordinates=coordinates,
        parameter_schema=schema,
        schema_fingerprint=schema.fingerprint,
        study_fingerprint=problem.study.fingerprint,
        objective_kind=objective_kind,
        evaluated_params=params.copy(),
        cluster=cluster,
        cluster_ids=cluster_ids,
        cluster_row_counts=counts,
        row_cluster_ids=row_ids,
        residual_blocks=blocks,
        n_rows=int(robust.n_rows),
        n_clusters=int(robust.n_clusters),
        assumptions=assumptions,
        warnings=tuple(caution),
        interpretation=(
            (
                "frequentist repeated-sampling covariance of the MAP estimator (the "
                "minimizer of the data's least squares plus the prior's penalty) under "
                f"{int(robust.n_clusters)} independent clusters; rows within a cluster may "
                "be arbitrarily correlated; the prior, held fixed across samples, narrows "
                "the spread through the bread and adds nothing to the meat; robust to a "
                "misstated noise scale, not a Bayesian posterior, and no protection "
                "against a biased physical model"
            )
            if penalized
            else robust.interpretation
        ),
        fit_status=status,
        fit_converged=bool(fit.converged),
    )


__all__ = [
    "ClusterResidualBlock",
    "ClusterCovarianceResult",
    "cluster_covariance",
]
