"""Experiment design for ``diff_soup.inference``: sensitivities, Fisher information, criteria.

[sensitivity_solve][diff_soup.inference.design.solve.sensitivity_solve] wires a
problem for sensitivity analysis: ``jax.jacfwd`` of the returned run gives the
parameter sensitivities ``S = ∂y/∂θ``, and any scalar criterion built from ``S``
can be reverse-differentiated with respect to design parameters.
[observable_sensitivities][diff_soup.inference.design.fisher.observable_sensitivities]
chains ``S`` through a measurement channel,
[fisher_information][diff_soup.inference.design.fisher.fisher_information]
assembles the Fisher information matrix, and
[fisher_with_prior][diff_soup.inference.design.criteria.fisher_with_prior] adds
Gaussian prior precision. The criteria ``d_optimal``, ``ds_optimal``,
``e_optimal_smooth``, and ``integrated_variance`` are smooth on every gradient
path (log-determinants and Cholesky solves; no eigendecomposition is ever
differentiated). ``t_optimal`` and ``DiscriminationEnsemble`` handle robust
model discrimination, the ``sequential_*`` verbs run checkpointable local
sequential design, and [parallel][diff_soup.inference.design.parallel] maps
criteria onto batched, multi-device, and multi-process execution.
"""

from diff_soup.inference.design.criteria import (
    d_optimal,
    ds_optimal,
    e_optimal_smooth,
    fisher_with_prior,
    integrated_variance,
)
from diff_soup.inference.design.discriminate import (
    DiscriminationEnsemble,
    t_optimal,
)
from diff_soup.inference.design.fisher import (
    fisher_information,
    observable_sensitivities,
    observable_trajectory,
)
from diff_soup.inference.design.parallel import (
    CampaignFailure,
    CampaignResult,
    CampaignTaskRecord,
    batched_criterion,
    checkpointed_map,
    choose_pattern,
    deterministic_weighted_reduce,
    robust_criterion,
    sharded_robust_gradient,
)
from diff_soup.inference.design.sequential import (
    CandidateDesign,
    ExperimentSpace,
    NextDesign,
    SequentialState,
    SequentialUpdate,
    replay_sequential,
    sequential_step,
    sequential_update,
)
from diff_soup.inference.design.solve import SensitivityRun, sensitivity_solve

__all__ = [
    # Sensitivity runs
    "sensitivity_solve",
    "SensitivityRun",
    # Fisher information
    "observable_trajectory",
    "observable_sensitivities",
    "fisher_information",
    "fisher_with_prior",
    # Design criteria
    "d_optimal",
    "ds_optimal",
    "e_optimal_smooth",
    "integrated_variance",
    # Model discrimination
    "t_optimal",
    "DiscriminationEnsemble",
    # Sequential design
    "CandidateDesign",
    "ExperimentSpace",
    "NextDesign",
    "SequentialState",
    "SequentialUpdate",
    "sequential_step",
    "sequential_update",
    "replay_sequential",
    # Batched / sharded / process-parallel execution
    "batched_criterion",
    "robust_criterion",
    "sharded_robust_gradient",
    "choose_pattern",
    "checkpointed_map",
    "deterministic_weighted_reduce",
    "CampaignFailure",
    "CampaignTaskRecord",
    "CampaignResult",
]
