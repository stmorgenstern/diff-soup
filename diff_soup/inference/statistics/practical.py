"""practical.py — the combined practical-identifiability report.

"Is this parameter identifiable?" has no unconditional yes/no answer: it
depends on the evaluated parameter point, the protocol, the noise level, and
on what precision would satisfy the person asking. [practical_report][diff_soup.inference.statistics.practical.practical_report]
therefore assembles the *evidence* — four sections, each an object the rest
of the package already knows how to compute — and never a verdict:

1. **Local sensitivity** — the information spectrum, correlations, and
   degenerate combinations at the fit (an
   [IdentifiabilityReport][diff_soup.inference.statistics.IdentifiabilityReport]).
2. **Likelihood geometry** — profile-likelihood summaries: intervals, open
   directions, and how each walk ended.
3. **Uncertainty contraction** — the posterior's marginal scales and
   data-null directions (from a [LaplacePosterior][diff_soup.inference.statistics.LaplacePosterior]).
4. **Model adequacy** — whether the residuals are consistent with the
   declared likelihood (a [ResidualReport][diff_soup.inference.statistics.ResidualReport]);
   uncertainty statements inherit the model's credibility.

The only pass/fail surface is [ToleranceStatus][diff_soup.inference.statistics.practical.ToleranceStatus], and it exists only
against tolerances the caller declares — each entry quotes the declared
tolerance next to the achieved value.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np

__all__ = ["PracticalReport", "ToleranceStatus", "practical_report"]


@dataclasses.dataclass(frozen=True)
class ToleranceStatus:
    """One user-declared precision requirement checked against the posterior.

    Attributes:
        name: The parameter the requirement addresses.
        tolerance: The declared maximum acceptable marginal standard
            deviation — the caller's number, quoted verbatim.
        value: The achieved posterior marginal standard deviation.
        within: Whether ``value <= tolerance``.
    """

    name: str
    tolerance: float
    value: float
    within: bool


@dataclasses.dataclass(frozen=True)
class PracticalReport:
    """Four sections of practical-identifiability evidence. Frozen.

    Deliberately carries no unconditional "identifiable" flag: every section
    is evidence relative to this experiment, this fit, and this noise model.
    The optional [status][diff_soup.inference.statistics.practical.PracticalReport.status] is the only pass/fail surface, and it exists
    only against tolerances the caller declared.

    Attributes:
        local_sensitivity: The information-spectrum report at the fit.
        likelihood_geometry: One summary dict per supplied profile: target,
            levels, interval at the largest level, open directions, and how
            each walk ended.
        uncertainty_contraction: Posterior marginal scales by name, the
            curvature kind, whether a prior informed it, and the number of
            data-null directions — or ``None`` when no posterior was given.
        model_adequacy: The residual-adequacy report, or ``None`` when the
            experiment carries no data.
        status: Tolerance checks (see [ToleranceStatus][diff_soup.inference.statistics.practical.ToleranceStatus]), or ``None``
            when no tolerances were declared.
        provenance: What evidence was and was not available, so a reader can
            tell an unexamined question from a settled one.
    """

    local_sensitivity: Any
    likelihood_geometry: tuple
    uncertainty_contraction: Optional[dict]
    model_adequacy: Optional[Any]
    status: Optional[tuple]
    provenance: dict


def practical_report(
    problem: Any,
    fit: Any,
    *,
    posterior: Any = None,
    profiles: tuple = (),
    tolerances: Optional[dict] = None,
    report: Any = None,
    experiment_id: Any = None,
) -> PracticalReport:
    """Collect the evidence on practical identifiability at a fit into one report.

    It holds the information spectrum, the supplied profiles and posterior, and the residual
    report, and names the evidence that was not supplied.

    Args:
        problem: An [`InferenceProblem`][diff_soup.inference.InferenceProblem] whose measurements
            all declare a likelihood.
        fit: The [`OptResult`][diff_soup.inference.OptResult] of `ds.inference.optsolve` on
            `problem`.
        posterior: The result of `laplace` at the fit, whose standard errors enter the report.
        profiles: Results of `profile`, whose intervals and open directions enter the report.
        tolerances: A mapping from parameter name to the largest acceptable standard error,
            which needs `posterior`.
        report: The result of `identifiability` at this fit, or `None` to compute it.
        experiment_id: The experiment whose residuals are checked, required when the problem has
            several.

    Returns:
        A `PracticalReport`, with each tolerance marked as met or not in `status`.
    """
    from diff_soup.inference.statistics.diagnostics import residual_report
    from diff_soup.inference.statistics.identifiability import identifiability
    from diff_soup.inference.statistics.measurement import require_likelihoods

    require_likelihoods(problem, "practical_report")
    x_hat = np.asarray(fit.params, dtype=float)
    missing: list = []

    local = identifiability(problem, fit) if report is None else report
    if local.study_fingerprint != problem.study.fingerprint or not np.array_equal(
        local.evaluated_params, x_hat
    ):
        raise ValueError("practical_report report must describe this problem at the supplied fit.")

    geometry = tuple(
        {
            "target": p.target,
            "levels": p.levels,
            "interval": p.interval(max(p.levels)),
            "open_directions": p.open_directions,
            "wing_end": dict(p.provenance.get("wing_end", {})),
        }
        for p in profiles
    )
    if not profiles:
        missing.append("no profile curves were supplied; likelihood geometry is unexamined")

    contraction = None
    if posterior is not None:
        names = tuple(posterior.names) or tuple(
            f"theta[{i}]" for i in range(int(np.asarray(posterior.mode).shape[0]))
        )
        sds = np.asarray(posterior.marginal_sd())
        contraction = {
            "marginal_sd": dict(zip(names, (float(s) for s in sds))),
            "curvature_kind": posterior.curvature_kind,
            "prior_informed": posterior.prior_spectrum is not None,
            "n_data_null": problem.n_params - local.raw.sensitivity_rank,
        }
    else:
        missing.append("no posterior was supplied; uncertainty contraction is unexamined")

    adequacy = residual_report(problem, fit, experiment_id=experiment_id)

    status = None
    if tolerances is not None:
        if posterior is None:
            raise ValueError(
                "practical_report tolerances require posterior= with a LaplacePosterior."
            )
        post_names = tuple(posterior.names)
        unknown = sorted(set(tolerances) - set(post_names))
        if unknown:
            raise ValueError(
                f"tolerances name unknown parameter(s) {unknown}; the posterior "
                f"carries {list(post_names)}."
            )
        entries = []
        for name, tol in tolerances.items():
            value = float(np.asarray(posterior.marginal_sd(name)).reshape(()))
            entries.append(
                ToleranceStatus(
                    name=name,
                    tolerance=float(tol),
                    value=value,
                    within=bool(value <= float(tol)),
                )
            )
        status = tuple(entries)

    return PracticalReport(
        local_sensitivity=local,
        likelihood_geometry=geometry,
        uncertainty_contraction=contraction,
        model_adequacy=adequacy,
        status=status,
        provenance={
            "evaluated_at": tuple(float(v) for v in x_hat),
            "missing_evidence": tuple(missing),
            "n_profiles": len(profiles),
            "tolerances_declared": tuple(tolerances) if tolerances else (),
        },
    )
