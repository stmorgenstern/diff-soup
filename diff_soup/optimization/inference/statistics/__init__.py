"""Advanced statistical functions and numerical building blocks.

The documented fitting and analysis surface is ``ds.inference``. This
namespace provides advanced functions such as ``posterior_target``,
``whitening`` and ``sbc``, likelihood base classes, and result kernels.
"""

from diff_soup.inference.statistics.cluster import ClusterCovarianceResult, ClusterResidualBlock
from diff_soup.inference.statistics.compression import (
    AffineSummary,
    CompressionReport,
    compare_compression,
)
from diff_soup.inference.statistics.diagnostics import ChannelResiduals, McmcReport, ResidualReport
from diff_soup.inference.statistics.evidence import (
    EvidenceResult,
    PriorDominatedDirection,
    PriorSensitivity,
    laplace_evidence,
)
from diff_soup.inference.statistics.experiment import (
    CHANNEL_ROLES,
    Channel,
    ExperimentSet,
    Predictor,
)
from diff_soup.inference.statistics.identifiability import (
    IdentifiabilityCoordinate,
    IdentifiabilityReport,
)
from diff_soup.inference.statistics.information import InformationResult, assemble_information
from diff_soup.inference.statistics.laplace import LaplacePosterior, laplace_posterior
from diff_soup.inference.statistics.likelihoods import (
    CholeskyOutputInformation,
    DiagonalOutputInformation,
    Likelihood,
    OutputInformation,
)
from diff_soup.inference.statistics.parameters import (
    BoundedAffine,
    ConstrainedParameterization,
    ConvertedCovariance,
    Gauge,
    Identity,
    LinearConstraint,
    Log,
    Logit,
    ModeLabel,
    ParamBlock,
    Parameterization,
    ScaledTransform,
    Transform,
    constrained,
)
from diff_soup.inference.statistics.posterior import posterior_target
from diff_soup.inference.statistics.practical import PracticalReport, ToleranceStatus
from diff_soup.inference.statistics.predict import (
    ComparisonReport,
    HeldOutEvent,
    HeldOutPrediction,
    ModelScore,
    PredictiveBand,
    band_from_jacobian,
    compare_predictive,
    prior_predictive,
)
from diff_soup.inference.statistics.priors import (
    GaussianPrior,
    IndependentProduct,
    Prior,
    TransformedPrior,
    prior_from_regularizer,
    pullback_log_prob,
)
from diff_soup.inference.statistics.profile import At, ProfileRefit, ProfileResult
from diff_soup.inference.statistics.report import (
    FunctionBand,
    FunctionReport,
    IdentifiedSubspace,
    ImproperPosteriorError,
    NullDirection,
    fit_posterior,
    gauss_newton_covariance,
    identifiability_report,
    pointwise_variance,
)
from diff_soup.inference.statistics.robust import RobustCovariance, sandwich_covariance
from diff_soup.inference.statistics.sample import (
    Draws,
    GradientCheck,
    LogDensity,
    SbcReport,
    WhiteningTransform,
    log_posterior,
    sbc,
    whitening,
)
from diff_soup.inference.statistics.splits import (
    ConditioningRecord,
    CrossValidationSplit,
    HeldOutResponseGroup,
    grouped_kfold,
    leave_one_entity_out,
    one_standard_error,
    within_entity_holdout,
)

__all__ = [
    # Measurements, studies, estimation problems
    # Identifiability, cluster covariance
    "IdentifiabilityCoordinate",
    "IdentifiabilityReport",
    "ClusterResidualBlock",
    "ClusterCovarianceResult",
    # Posterior construction
    "posterior_target",
    # Experiment declaration (engine)
    "Channel",
    "ExperimentSet",
    "Predictor",
    "CHANNEL_ROLES",
    # Information and Laplace analysis
    "InformationResult",
    "assemble_information",
    "LaplacePosterior",
    "laplace_posterior",
    "laplace_evidence",
    "EvidenceResult",
    "PriorSensitivity",
    "PriorDominatedDirection",
    # Profile likelihood
    "ProfileResult",
    "ProfileRefit",
    "At",
    # Host-side reporting
    "identifiability_report",
    "NullDirection",
    "IdentifiedSubspace",
    "ImproperPosteriorError",
    "gauss_newton_covariance",
    "fit_posterior",
    "pointwise_variance",
    "FunctionReport",
    "FunctionBand",
    "PracticalReport",
    "ToleranceStatus",
    # Observable-compression comparisons
    "compare_compression",
    "CompressionReport",
    "AffineSummary",
    # Grouped resampling and leakage-safe split declarations
    "grouped_kfold",
    "leave_one_entity_out",
    "within_entity_holdout",
    "one_standard_error",
    "CrossValidationSplit",
    "HeldOutResponseGroup",
    "ConditioningRecord",
    # Prediction bands and prior-predictive checks
    "prior_predictive",
    "band_from_jacobian",
    "PredictiveBand",
    "compare_predictive",
    "HeldOutEvent",
    "HeldOutPrediction",
    "ModelScore",
    "ComparisonReport",
    # Adequacy and robustness
    "ResidualReport",
    "ChannelResiduals",
    "sandwich_covariance",
    "RobustCovariance",
    # Parameter coordinates
    "Parameterization",
    "ParamBlock",
    "Transform",
    "Identity",
    "Log",
    "Logit",
    "BoundedAffine",
    "ScaledTransform",
    "ConvertedCovariance",
    "ModeLabel",
    "LinearConstraint",
    "Gauge",
    "ConstrainedParameterization",
    "constrained",
    # Likelihoods
    "Likelihood",
    "OutputInformation",
    "DiagonalOutputInformation",
    "CholeskyOutputInformation",
    # Priors
    "Prior",
    "GaussianPrior",
    "TransformedPrior",
    "IndependentProduct",
    "pullback_log_prob",
    "prior_from_regularizer",
    # Normalized posterior targets, whitening, sampling, calibration, diagnostics
    "log_posterior",
    "LogDensity",
    "GradientCheck",
    "whitening",
    "WhiteningTransform",
    "Draws",
    "sbc",
    "SbcReport",
    "McmcReport",
]


# The constructor is public as posterior_target; the implementation module is private.
globals().pop("posterior", None)
