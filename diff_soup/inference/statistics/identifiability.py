"""Handle-aware, data-only local identifiability analysis.

This module is deliberately a thin facade over
``diff_soup.inference.statistics.identifiability_report``.  It is responsible for
selecting the *data* Jacobian from an
[InferenceProblem][diff_soup.inference.InferenceProblem], checking that
the fitted vector still belongs to the same handle-based parameter schema, and
labelling flat columns with their Recipe handles and cluster owners.  The SVD,
rank, null-direction, and correlation numerics stay in the existing report
kernel.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.inference.fitting.inference_problem import _is_structural_schema
from diff_soup.inference.fitting.result import OptResult
from diff_soup.inference.statistics.estimation import EstimationProblem
from diff_soup.inference.statistics.measurement import require_likelihoods
from diff_soup.inference.statistics.report import (
    IdentifiabilityReport as SpectrumReport,
)
from diff_soup.inference.statistics.report import (
    identifiability_report,
)


@dataclasses.dataclass(frozen=True)
class IdentifiabilityCoordinate:
    """Provenance for one column of the fitted physical parameter vector.

    ``owner`` is ``None`` for a shared handle and is the exact Study cluster
    owner for a varying handle.  ``component`` indexes a flattened array or
    function handle; scalar handles have component zero.
    """

    column: int
    name: str
    handle: Any
    owner: Any
    component: int


@dataclasses.dataclass(frozen=True)
class IdentifiabilityReport:
    """Auditable data-only local-identifiability evidence at one fitted point.

    ``raw`` analyzes the whitened response Jacobian with respect to the
    fitted physical coordinates.  ``scaled`` analyzes ``J @ diag(scales)``,
    corresponding to local perturbations ``theta = theta_hat + scales * z``.
    No prior-residual rows are present in either report, including for MAP fits.

    The result intentionally carries the handle-based schema and per-column handle
    provenance.  This makes owner-specific null-direction loadings inspectable
    without asking users to reconstruct flat optimizer indices.
    """

    raw: SpectrumReport
    scaled: SpectrumReport
    coordinates: tuple[IdentifiabilityCoordinate, ...]
    parameter_scales: np.ndarray
    parameter_schema: Any
    schema_fingerprint: str
    study_fingerprint: str
    objective_kind: str
    evaluated_params: np.ndarray
    n_data_rows: int
    data_residual_norm: float
    fit_status: str
    fit_converged: bool

    @property
    def names(self) -> tuple[str, ...]:
        """Owner-aware flat-coordinate names in Jacobian column order."""
        return tuple(coordinate.name for coordinate in self.coordinates)


def _coordinate_name(handle: Any, owner: Any, component: int, size: int) -> str:
    """Build the same readable owner suffix used by Study joint groups."""
    name = str(handle.name)
    if size != 1:
        name = f"{name}[{component}]"
    if owner is not None:
        name = f"{name}@{owner}"
    return name


def _schema_coordinates(schema: Any, n_params: int) -> tuple[IdentifiabilityCoordinate, ...]:
    """Compile and validate a complete schema-to-column provenance table."""
    by_column: list[IdentifiabilityCoordinate | None] = [None] * n_params
    for entry in schema.entries:
        size = int(entry.size)
        for occurrence in entry.occurrences:
            columns = tuple(int(column) for column in occurrence.columns)
            if len(columns) != size:
                raise ValueError(
                    f"identifiability: schema handle {entry.handle.name!r} occurrence "
                    f"has {len(columns)} columns; expected {size}."
                )
            for component, column in enumerate(columns):
                if column < 0 or column >= n_params:
                    raise ValueError(
                        f"identifiability: schema column {column} for handle "
                        f"{entry.handle.name!r} is outside a {n_params}-parameter fit."
                    )
                if by_column[column] is not None:
                    previous = by_column[column]
                    raise ValueError(
                        f"identifiability: schema column {column} is owned by both "
                        f"{previous.name!r} and {entry.handle.name!r}."
                    )
                by_column[column] = IdentifiabilityCoordinate(
                    column=column,
                    name=_coordinate_name(entry.handle, occurrence.owner, component, size),
                    handle=entry.handle,
                    owner=occurrence.owner,
                    component=component,
                )

    missing = tuple(index for index, value in enumerate(by_column) if value is None)
    if missing:
        raise ValueError(
            "identifiability: handle-based parameter schema does not own flat "
            f"parameter column(s) {missing}."
        )
    return tuple(by_column)  # type: ignore[arg-type]


def _schema_signature(schema: Any) -> tuple:
    """Return the handle/owner/column structure a fingerprint is meant to label."""
    return tuple(
        (
            entry.handle.identity,
            int(entry.size),
            tuple(
                (occurrence.owner, tuple(int(column) for column in occurrence.columns))
                for occurrence in entry.occurrences
            ),
        )
        for entry in schema.entries
    )


def _require_problem_schema(problem: Any, schema: Any, *, where: str) -> None:
    """Require a fit schema that labels this problem's own coordinates.

    A structural schema records one problem's own handle-to-column layout
    rather than the content of a declaration, so its fingerprint is never
    comparable with a Study fingerprint. The handle/owner/column structure is
    what validates such a schema; two declaration fingerprints are compared
    directly.

    Args:
        problem: The estimation problem the fit is being reused on.
        schema: The fit's parameter schema.
        where: The verb name the message speaks for.

    Raises:
        ValueError: If the two schemas label different structures, or two
            declaration fingerprints disagree.
    """
    problem_schema = problem.parameter_schema
    if _schema_signature(schema) != _schema_signature(problem_schema):
        raise ValueError(
            f"{where}: OptResult labels a different handle/owner/column structure "
            "than this problem; pass a fit produced from this problem."
        )
    if not _is_structural_schema(schema) and schema.fingerprint != problem_schema.fingerprint:
        raise ValueError(
            f"{where}: OptResult carries parameter schema fingerprint "
            f"{schema.fingerprint!r}, but this problem's schema fingerprint is "
            f"{problem_schema.fingerprint!r}."
        )


def _scale_block(value: Any, size: int, *, label: str) -> np.ndarray:
    """Broadcast one scalar or require one flattened scale per handle component."""
    block = np.asarray(value, dtype=float)
    if block.ndim == 0:
        return np.full(size, float(block))
    block = np.ravel(block)
    if block.shape != (size,):
        raise ValueError(
            f"identifiability: scale for {label} has {block.shape}; expected a scalar or ({size},)."
        )
    return block


def _parameter_scales(
    scales: Any,
    *,
    schema: Any,
    coordinates: tuple[IdentifiabilityCoordinate, ...],
) -> np.ndarray:
    """Resolve flat or handle-keyed scale declarations to column order."""
    n_params = len(coordinates)
    if scales is None:
        result = np.ones(n_params, dtype=float)
    elif not isinstance(scales, Mapping):
        result = np.asarray(scales, dtype=float)
        if result.shape != (n_params,):
            raise ValueError(
                f"identifiability: parameter_scales has shape {result.shape}; "
                f"expected ({n_params},) in handle-based schema column order."
            )
        result = result.copy()
    else:
        result = np.full(n_params, np.nan, dtype=float)
        assigned = np.zeros(n_params, dtype=bool)
        for key, value in scales.items():
            if isinstance(key, tuple) and len(key) == 2 and hasattr(key[0], "identity"):
                handle, requested_owner = key
                owner_is_explicit = True
            elif hasattr(key, "identity"):
                handle = key
                requested_owner = None
                owner_is_explicit = False
            else:
                raise TypeError(
                    "identifiability: parameter_scales mapping keys must be "
                    "Recipe handles or (handle, owner) pairs; flat names are not "
                    "accepted as parameter identities."
                )
            try:
                entry = schema.entry(handle)
            except KeyError as exc:
                raise ValueError(
                    f"identifiability: scale handle {getattr(handle, 'name', handle)!r} "
                    "does not belong to this fit's schema."
                ) from exc
            block = _scale_block(
                value,
                int(entry.size),
                label=(
                    f"{entry.handle.name!r}@{requested_owner!r}"
                    if owner_is_explicit
                    else repr(entry.handle.name)
                ),
            )
            occurrences = tuple(
                occurrence
                for occurrence in entry.occurrences
                if not owner_is_explicit or occurrence.owner == requested_owner
            )
            if not occurrences:
                raise ValueError(
                    f"identifiability: handle {entry.handle.name!r} has no schema "
                    f"occurrence owned by {requested_owner!r}."
                )
            for occurrence in occurrences:
                columns = np.asarray(occurrence.columns, dtype=int)
                overlap = columns[assigned[columns]]
                if overlap.size:
                    raise ValueError(
                        "identifiability: parameter_scales assigns flat column(s) "
                        f"{tuple(int(value) for value in overlap)} more than once."
                    )
                result[columns] = block
                assigned[columns] = True
        missing = tuple(coordinates[index].name for index in np.flatnonzero(~assigned))
        if missing:
            raise ValueError(
                "identifiability: parameter_scales mapping must cover every fitted "
                f"coordinate; missing {missing}."
            )

    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(
            "identifiability: every parameter scale must be finite and strictly positive."
        )
    return result


def _data_residual_jacobian(problem: Any, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate only the response residual and Jacobian exposed by a problem."""
    if bool(getattr(problem, "is_sharded", False)):
        raise NotImplementedError(
            "identifiability currently requires a materialized data Jacobian; "
            "a sharded Study exposes only reduced Gauss-Newton statistics. "
            "Recompile without a sharded layout for this local report."
        )

    x = jnp.asarray(params)
    residual_jacobian = getattr(problem, "batched_rj_fn", None)
    if residual_jacobian is None:
        residual_jacobian = getattr(problem, "stacked_rj_fn", None)
    if residual_jacobian is not None:
        residual, jacobian = residual_jacobian(x)
    else:
        segment_fns = tuple(getattr(problem, "segment_fns", ()) or ())
        if not segment_fns:
            raise TypeError(
                "identifiability: the problem's data-only problem exposes no "
                "residual/Jacobian closures."
            )
        residual_blocks = []
        jacobian_blocks = []
        for fn in segment_fns:
            residual_blocks.append(np.ravel(np.asarray(fn(x))))
            jacobian_blocks.append(np.asarray(jax.jacfwd(fn)(x)))
        residual = np.concatenate(residual_blocks)
        jacobian = np.concatenate(jacobian_blocks, axis=0)

    residual = np.ravel(np.asarray(residual, dtype=float))
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2:
        raise ValueError(f"identifiability: data Jacobian must be 2-D; got shape {jacobian.shape}.")
    if jacobian.shape[0] != residual.size:
        raise ValueError(
            "identifiability: data residual/Jacobian row mismatch: "
            f"{residual.size} residuals and {jacobian.shape[0]} Jacobian rows."
        )
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
        raise ValueError(
            "identifiability: data residual or Jacobian is non-finite at the fitted point."
        )
    return residual, jacobian


def identifiability(
    problem: EstimationProblem,
    fit: OptResult | Mapping,
    *,
    parameter_scales: Any = None,
    ridge: float = 1e-10,
    null_tol: float = 1e-10,
) -> IdentifiabilityReport:
    """Find which parameter directions the data determine at a fit, and which they leave undetermined.

    The prior and any regularization are left out, so neither can hide a direction the data
    leave open. The analysis is reported twice, in the parameters' own units and after scaling
    by `parameter_scales`.

    Args:
        problem: An [`InferenceProblem`][diff_soup.inference.InferenceProblem] whose measurements
            all declare a likelihood.
        fit: The [`OptResult`][diff_soup.inference.OptResult] of `ds.inference.optsolve` on
            `problem`, or a mapping from every fitted handle to its value.
        parameter_scales: Positive scales per fitted coordinate, as a vector or a mapping from
            handle or `(handle, owner)` to scale, or `None` for all ones.
        ridge: A small value added to each eigenvalue before inverting it, for the correlations
            of the report only.
        null_tol: The eigenvalue, relative to the largest, at or below which a direction counts
            as undetermined.

    Returns:
        An `IdentifiabilityReport` holding the analysis in raw and scaled coordinates.
    """
    if not isinstance(problem, EstimationProblem):
        raise TypeError("identifiability first argument must be an InferenceProblem.")
    require_likelihoods(problem, "identifiability")
    if isinstance(fit, Mapping):
        from diff_soup.inference.statistics.information import _point

        fit = OptResult(
            params=np.asarray(_point(problem, fit)),
            parameter_schema=problem.parameter_schema,
            objective=problem.objective,
        )
    if not isinstance(fit, OptResult):
        raise TypeError("identifiability requires a fit or a handle-valued point.")
    if fit.parameter_schema is None:
        raise ValueError(
            "identifiability requires an OptResult with handle-based parameter schema "
            "provenance; fit the supplied InferenceProblem through diff_soup.inference.optsolve(...)."
        )

    schema = problem.parameter_schema
    fit_schema = fit.parameter_schema
    _require_problem_schema(problem, fit_schema, where="identifiability")
    expected_objective = type(problem.objective).__name__
    if type(fit.objective).__name__ != expected_objective:
        raise ValueError(
            f"identifiability: fit objective is {type(fit.objective).__name__!r}, but the "
            f"problem objective is {expected_objective!r}."
        )
    fit_fingerprint = dict(fit.info).get("uq", {}).get("study_fingerprint")
    if fit_fingerprint is not None and fit_fingerprint != problem.study.fingerprint:
        raise ValueError("identifiability: OptResult Study provenance does not match the problem.")
    if int(fit.n_failed_segments) != 0:
        raise ValueError(
            "identifiability refuses a fit with failed forward-solve segments; "
            f"failed indices are {tuple(fit.failed_segment_indices)}."
        )

    params = np.asarray(fit.params, dtype=float)
    expected_shape = (int(problem.n_params),)
    if params.shape != expected_shape:
        raise ValueError(
            f"identifiability: fitted parameter vector has shape {params.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(params)):
        raise ValueError("identifiability: fitted parameter vector is non-finite.")

    coordinates = _schema_coordinates(schema, params.size)
    names = tuple(coordinate.name for coordinate in coordinates)
    scales = _parameter_scales(
        parameter_scales,
        schema=schema,
        coordinates=coordinates,
    )
    residual, jacobian = _data_residual_jacobian(problem.diagnostic_problem, params)
    if jacobian.shape[1] != params.size:
        raise ValueError(
            f"identifiability: data Jacobian has {jacobian.shape[1]} columns; "
            f"expected {params.size} from the handle-based parameter schema."
        )

    status = getattr(fit.status, "value", str(fit.status))
    common_provenance = {
        "study_fingerprint": problem.study.fingerprint,
        "schema_fingerprint": schema.fingerprint,
        "objective_kind": expected_objective,
        "information_source": "whitened measurement residuals only",
        "fit_status": status,
        "fit_converged": bool(fit.converged),
        "experiment_ids": problem.study.experiment_ids,
        "measurements": tuple(
            (
                experiment_id,
                tuple(measurement.name for measurement in experiment.measurements),
            )
            for experiment_id, experiment in zip(
                problem.study.experiment_ids, problem.study.experiments
            )
        ),
        "n_data_rows": int(residual.size),
    }
    raw_report = identifiability_report(
        whitened_jacobian=jacobian,
        names=names,
        ridge=ridge,
        null_tol=null_tol,
        evaluated_at={
            **common_provenance,
            "coordinate": "fitted physical parameter",
            "parameter_scales": None,
        },
    )
    scaled_report = identifiability_report(
        whitened_jacobian=jacobian * scales[None, :],
        names=names,
        ridge=ridge,
        null_tol=null_tol,
        evaluated_at={
            **common_provenance,
            "coordinate": "z where theta = theta_hat + parameter_scales * z",
            "parameter_scales": tuple(float(value) for value in scales),
        },
    )

    return IdentifiabilityReport(
        raw=raw_report,
        scaled=scaled_report,
        coordinates=coordinates,
        parameter_scales=scales,
        parameter_schema=schema,
        schema_fingerprint=schema.fingerprint,
        study_fingerprint=problem.study.fingerprint,
        objective_kind=expected_objective,
        evaluated_params=params.copy(),
        n_data_rows=int(residual.size),
        data_residual_norm=float(np.linalg.norm(residual)),
        fit_status=status,
        fit_converged=bool(fit.converged),
    )


__all__ = [
    "IdentifiabilityCoordinate",
    "IdentifiabilityReport",
    "identifiability",
]
