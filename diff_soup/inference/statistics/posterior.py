"""Posterior-sampling targets for estimation problems.

The fitting coordinate exposed by [InferenceProblem][diff_soup.inference.InferenceProblem] is the declared
physical Recipe coordinate. Posterior samplers instead need an unconstrained
coordinate when Recipe bounds are finite. ``posterior`` constructs that map
explicitly, includes its change-of-variables Jacobian, and returns the mature
``LogDensity`` target consumed by the existing RMH/NUTS implementation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np

from diff_soup.inference.fitting._layout import resolve_point
from diff_soup.inference.fitting.bounds import resolve_decision_bounds
from diff_soup.inference.fitting.result import OptResult
from diff_soup.inference.statistics.estimation import MAP, EstimationProblem
from diff_soup.inference.statistics.measurement import require_likelihoods
from diff_soup.inference.statistics.parameters import (
    BoundedAffine,
    Identity,
    Log,
    ParamBlock,
    Parameterization,
    Transform,
)
from diff_soup.inference.statistics.sample import LogDensity


def _scalar_layout(problem: EstimationProblem) -> tuple:
    """Return one unique handle/owner/component label per physical column."""
    layout: list[tuple[str, str] | None] = [None] * problem.n_params
    for entry in problem.parameter_schema.entries:
        for occurrence in entry.occurrences:
            for component, column in enumerate(occurrence.columns):
                name = str(entry.handle.name)
                if int(entry.size) != 1:
                    name = f"{name}[{component}]"
                if occurrence.owner is not None:
                    name = f"{name}@{occurrence.owner}"
                if layout[int(column)] is not None:
                    raise ValueError(f"posterior schema assigns column {column} more than once")
                layout[int(column)] = (name, str(entry.handle.name))
    missing = tuple(index for index, value in enumerate(layout) if value is None)
    if missing:
        raise ValueError(f"posterior schema does not own parameter column(s) {missing}")
    names = tuple(item[0] for item in layout)
    if len(set(names)) != len(names):
        raise ValueError(f"posterior scalar handle names are not unique: {names}")
    return tuple(layout)


def _schema_parameterization(
    problem: EstimationProblem,
    *,
    transforms: str | dict,
) -> Parameterization:
    """Build sampler coordinates from Recipe parameter handles rather than leaf names."""
    if not isinstance(transforms, (str, dict)):
        raise TypeError("posterior transforms must be 'auto', 'identity', or a mapping")
    if isinstance(transforms, str) and transforms not in ("auto", "identity"):
        raise ValueError("posterior transforms string must be 'auto' or 'identity'")
    requested = {} if isinstance(transforms, str) else dict(transforms)
    for key, value in requested.items():
        if not isinstance(key, str) or not key:
            raise TypeError("posterior transform keys must be non-empty handle names")
        if not isinstance(value, Transform):
            raise TypeError(
                f"posterior transform for {key!r} must be a Transform; got {type(value).__name__}"
            )

    resolved_bounds = resolve_decision_bounds(problem)
    if resolved_bounds is None:
        lower = np.full(problem.n_params, -np.inf)
        upper = np.full(problem.n_params, np.inf)
    else:
        lower = np.asarray(resolved_bounds[0], dtype=float)
        upper = np.asarray(resolved_bounds[1], dtype=float)

    used: set[str] = set()
    blocks = []
    for column, (name, handle_name) in enumerate(_scalar_layout(problem)):
        if name in requested:
            transform = requested[name]
            used.add(name)
        elif handle_name in requested:
            transform = requested[handle_name]
            used.add(handle_name)
        elif transforms == "auto":
            lo = float(lower[column])
            hi = float(upper[column])
            if np.isfinite(lo) and np.isfinite(hi):
                transform = BoundedAffine(lo, hi)
            elif lo == 0.0 and np.isposinf(hi):
                transform = Log()
            else:
                transform = Identity()
        else:
            transform = Identity()
        blocks.append(
            ParamBlock(
                name=name,
                index=slice(column, column + 1),
                transform=transform,
                shape=(),
                group=handle_name,
            )
        )
    unused = sorted(set(requested) - used)
    if unused:
        raise ValueError(f"posterior transforms name unknown coordinate/group(s) {unused}")
    return Parameterization(tuple(blocks))


def _validated_start(problem: EstimationProblem, start: Any) -> np.ndarray:
    """Resolve an array, mapping or OptResult start; a fit must share the problem's layout."""
    if start is None:
        value = np.asarray(problem.x0, dtype=float)
    elif isinstance(start, OptResult):
        if start.parameter_schema is None:
            raise ValueError(
                "posterior start OptResult has no handle-based parameter schema; "
                "fit this InferenceProblem through diff_soup.inference.optsolve first, "
                "or pass fit.params."
            )
        if type(start.objective).__name__ != "MAP":
            raise ValueError(
                f"posterior requires a MAP OptResult start; got {type(start.objective).__name__!r}."
            )
        if int(start.n_failed_segments) != 0:
            raise ValueError(
                "posterior refuses a start with failed forward segments; failed "
                f"indices are {tuple(start.failed_segment_indices)}."
            )
        value = np.asarray(resolve_point(problem, start, where="posterior start"), dtype=float)
    else:
        value = np.asarray(resolve_point(problem, start, where="posterior start"), dtype=float)
    if value.shape != (problem.n_params,):
        raise ValueError(
            f"posterior start has shape {value.shape}; expected ({problem.n_params},)."
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("posterior start must contain only finite values.")
    return value


def _move_inside_bounds(
    physical: np.ndarray,
    parameterization: Parameterization,
    *,
    fraction: float,
) -> np.ndarray:
    """Move an optimizer point infinitesimally inside open sampling transforms."""
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("posterior interior_fraction must lie strictly between 0 and 0.5.")
    result = np.asarray(physical, dtype=float).copy()
    for block in parameterization.blocks:
        transform = block.transform
        values = result[block.index]
        if isinstance(transform, BoundedAffine):
            width = transform.upper - transform.lower
            lower = transform.lower + fraction * width
            upper = transform.upper - fraction * width
            result[block.index] = np.clip(values, lower, upper)
        elif isinstance(transform, Log):
            if np.any(values <= 0.0):
                raise ValueError(
                    f"posterior start for positive block {block.name!r} contains "
                    f"non-positive value(s) {values.tolist()}."
                )
    return result


def posterior_target(
    problem: EstimationProblem,
    fit: Any,
    *,
    gradient: str | None = "forward",
    interior_fraction: float = 1.0e-8,
) -> LogDensity:
    """Build the normalized posterior target sampled by RMH or NUTS.

    The returned density lives in an unconstrained inference coordinate ``z``.
    ``Parameterization.from_inference_problem(..., transforms="auto")`` maps Recipe
    finite bounds with a smooth bounded-affine transform and leaves unbounded
    coordinates unchanged. The target is

    ``log_likelihood(x(z)) + log_prior(x(z)) + log|dx/dz|``.

    Sampling maps every draw back to the physical Recipe coordinate. The
    change-of-variables term preserves the posterior probability measure.

    With `gradient="reverse"` the reverse-mode gradient at the start must agree with the
    forward-mode one within 100 times the solvers' largest `rtol`, kept between 1e-3 and
    1e-1. A float32 solve counts as `rtol` 1e-3 or more.
    """
    if not isinstance(problem, EstimationProblem):
        raise TypeError("posterior_target requires an InferenceProblem.")
    require_likelihoods(problem, "posterior_target")
    if not isinstance(problem.objective, MAP):
        raise ValueError(
            "posterior sampling requires MAP() with a proper prior on every "
            "fitted handle; an MLE is not a normalized parameter distribution."
        )
    parameterization = _schema_parameterization(problem, transforms="auto")
    if parameterization.n_parameters != problem.n_params:
        raise ValueError(
            "posterior parameterization covers only "
            f"{parameterization.n_parameters} coordinates but the estimation "
            f"problem has {problem.n_params}; auxiliary decision variables need "
            "an explicit sampling parameterization."
        )
    physical_start = _move_inside_bounds(
        _validated_start(problem, fit),
        parameterization,
        fraction=float(interior_fraction),
    )
    inference_start = parameterization.inverse(jnp.asarray(physical_start))

    def log_density(z):
        physical = parameterization.transform(z)
        return (
            problem.log_likelihood(physical)
            + problem.log_prior(physical)
            + parameterization.log_abs_det_jacobian(z)
        )

    identity_coordinate = all(
        isinstance(block.transform, Identity) for block in parameterization.blocks
    )
    target = LogDensity.from_callable(
        log_density,
        init=inference_start,
        names=parameterization.names(),
        _coordinate="x" if identity_coordinate else "z",
        gradient=gradient,
        _source=problem,
    )
    return dataclasses.replace(
        target,
        _to_physical=None if identity_coordinate else parameterization.transform,
        _parameter_schema=problem.parameter_schema,
        _problem=problem,
        _fit=fit,
    )


__all__ = ["posterior_target"]
