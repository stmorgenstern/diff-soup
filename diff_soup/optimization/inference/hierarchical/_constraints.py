"""Named linear constraints and gauges, lowered to rows over the fitted vector.

A constraint or a gauge addresses parameter groups by name; the rows here
address the columns of the fitted vector, as the trust-region fit over the
lanes takes them.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._distributed import local_value
from diff_soup.inference.fitting.joint import JointInferenceProblem
from diff_soup.inference.statistics.estimation import EstimationProblem

Array = jax.Array


def _constraint_rows(
    joint: JointInferenceProblem | EstimationProblem,
    constraints: Any,
    gauges: Any,
    owner_weights: Array,
    dtype: Any,
) -> tuple[Array, Array]:
    """Lower named linear constraints and gauges to flat joint rows."""
    from diff_soup.inference.statistics.parameters import Gauge, LinearConstraint

    if constraints is None:
        constraints = ()
    if isinstance(constraints, LinearConstraint):
        constraints = (constraints,)
    constraints = tuple(constraints)
    if isinstance(gauges, Gauge):
        gauges = (gauges,)
    gauges = tuple(gauges or ())

    group_columns = dict(zip(joint.param_group_names, joint.param_group_cols))
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for constraint_index, constraint in enumerate(constraints):
        if not isinstance(constraint, LinearConstraint):
            raise TypeError(
                "TrustRegion constraints= takes LinearConstraint instances; "
                f"entry {constraint_index} is {type(constraint).__name__}. Pass a "
                "LinearConstraint or a sequence of them."
            )
        for name, coefficients in constraint.coefficients.items():
            if name not in group_columns:
                raise ValueError(
                    f"TrustRegion constraints= entry {constraint_index} addresses unknown "
                    f"parameter group {name!r}; name one of {tuple(group_columns)}."
                )
            width = len(group_columns[name])
            if coefficients.shape[1] != width:
                raise ValueError(
                    f"TrustRegion constraints= coefficients for group {name!r} have "
                    f"width {coefficients.shape[1]}; the group has {width} scalar(s), "
                    f"so the coefficients need {width} column(s)."
                )
        for row_index in range(constraint.n_rows):
            row = np.zeros((joint.n_params,), dtype=np.float64)
            for name, coefficients in constraint.coefficients.items():
                row[list(group_columns[name])] = local_value(coefficients)[row_index]
            rows.append(row)
            rhs.append(float(local_value(constraint.rhs)[row_index]))
    lowered_rows = [jnp.asarray(row, dtype=dtype) for row in rows]
    lowered_rhs = [jnp.asarray(value, dtype=dtype) for value in rhs]

    scalar_names = tuple(joint.theta_names)
    for gauge_index, gauge in enumerate(gauges):
        if not isinstance(gauge, Gauge):
            raise TypeError(
                "TrustRegion gauges= takes Gauge instances; "
                f"entry {gauge_index} is {type(gauge).__name__}. Pass a Gauge or a "
                "sequence of them."
            )
        if gauge.kind == "declared":
            continue
        row = jnp.zeros((joint.n_params,), dtype=dtype)
        if gauge.kind == "pin":
            matches = tuple(i for i, name in enumerate(scalar_names) if name == gauge.name)
            if len(matches) != 1:
                raise ValueError(
                    f"Gauge.pin({gauge.name!r}) must identify exactly one joint theta "
                    f"scalar; found {len(matches)} matches. Use a LinearConstraint on "
                    "a named joint group when scalar names repeat."
                )
            row = row.at[matches[0]].set(1.0)
            target = jnp.asarray(gauge.value, dtype=dtype)
        elif gauge.kind == "zero_mean":
            if gauge.name not in group_columns:
                raise ValueError(
                    f"Gauge.zero_mean addresses unknown joint group {gauge.name!r}; "
                    f"available groups are {tuple(group_columns)}."
                )
            columns = group_columns[gauge.name]
            if gauge.weights is None:
                within = jnp.full((len(columns),), 1.0 / len(columns), dtype=dtype)
            else:
                within = jnp.asarray(gauge.weights, dtype=dtype)
                if within.shape != (len(columns),):
                    raise ValueError(
                        f"Gauge.zero_mean weights for {gauge.name!r} must have shape "
                        f"{(len(columns),)}; got {within.shape}."
                    )
            row = row.at[jnp.asarray(columns)].set(within)
            target = jnp.zeros((), dtype=dtype)
        else:  # zero_mean_across
            joint._require_one_level("Gauge.zero_mean_across")
            base_outer = jnp.asarray(gauge.weights, dtype=dtype)
            owner_positions = []
            entries = tuple(gauge.owners.items())
            for name, functional in entries:
                if name not in group_columns:
                    raise ValueError(
                        f"Gauge.zero_mean_across addresses unknown joint group {name!r}; "
                        f"available groups are {tuple(group_columns)}."
                    )
                columns = tuple(group_columns[name])
                functional = jnp.asarray(functional, dtype=dtype)
                if functional.shape != (len(columns),):
                    raise ValueError(
                        "Gauge.zero_mean_across functional for joint group "
                        f"{name!r} must have shape {(len(columns),)}; got "
                        f"{functional.shape}."
                    )
                owner_matches = tuple(
                    owner_index
                    for owner_index, (lo, hi) in enumerate(joint.ownership_layout.owner_slices)
                    if all(lo <= column < hi for column in columns)
                )
                if len(owner_matches) != 1:
                    raise ValueError(
                        "Gauge.zero_mean_across may address only owner-tied joint "
                        f"groups; {name!r} does not resolve to exactly one owner block."
                    )
                owner_positions.append(owner_matches[0])
            effective = base_outer * owner_weights[jnp.asarray(owner_positions)]
            try:
                effective_host = local_value(effective)
            except (jax.errors.TracerArrayConversionError, TypeError):
                pass
            else:
                if not np.any(effective_host > 0.0):
                    raise ValueError(
                        "Gauge.zero_mean_across has zero effective weight after "
                        "cluster/bootstrap weights; its row would constrain nothing."
                    )
            effective = effective / jnp.sum(effective)
            for coefficient, (name, functional) in zip(effective, entries):
                columns = tuple(group_columns[name])
                row = row.at[jnp.asarray(columns)].set(
                    coefficient * jnp.asarray(functional, dtype=dtype)
                )
            target = jnp.zeros((), dtype=dtype)
        lowered_rows.append(row)
        lowered_rhs.append(target)

    if not lowered_rows:
        return jnp.zeros((0, joint.n_params), dtype=dtype), jnp.zeros((0,), dtype=dtype)
    return jnp.stack(lowered_rows), jnp.stack(lowered_rhs)
