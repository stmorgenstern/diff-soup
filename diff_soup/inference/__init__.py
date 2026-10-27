"""Fit experiments and analyze uncertainty through Recipe parameter handles.

Declare Measurement data and an Experiment, construct an InferenceProblem,
and call optsolve. Information, Laplace, identifiability, profile, sample,
and predict use that same problem and its fit. Documented user-facing verbs
and likelihoods live here; statistics provides advanced building blocks.
``world()`` gives this process's rank and the number of processes of a run
launched as one MPI world, before ``ds.lower`` splits a study over them.
"""

from diff_soup.inference._abc import LeastSquaresProblem, OptimizationProblem, OptSolver
from diff_soup.inference.design.parallel import checkpointed_map
from diff_soup.inference.design.solve import sensitivity_solve
from diff_soup.inference.engine._transport import world
from diff_soup.inference.fitting._optsolve import optsolve
from diff_soup.inference.fitting.bounds import Bounds
from diff_soup.inference.fitting.callbacks import Callback, CheckpointCallback, LogCallback, StopFit
from diff_soup.inference.fitting.constrained import InequalityConstraints, constrained_optsolve
from diff_soup.inference.fitting.diagnostics import OptDiagnostics
from diff_soup.inference.fitting.inference_problem import (
    FixedInitialState,
    LearnedInitialState,
    LeastSquares,
    Multishoot,
    SingleShoot,
)
from diff_soup.inference.fitting.learnables import learnables
from diff_soup.inference.fitting.problem import InferenceProblem, LoweredStudy, prepare_windows
from diff_soup.inference.fitting.regularization import NoReg, Tikhonov
from diff_soup.inference.fitting.result import OptResult, TerminationStatus
from diff_soup.inference.fitting.scales import ConstantScale, HessianScale, JacScale, XScale
from diff_soup.inference.fitting.windows import Continuity, RestartStep, StateBasis, Windows
from diff_soup.inference.hierarchical.level_solvers import TrustRegion
from diff_soup.inference.solvers.bfgs import BFGS
from diff_soup.inference.solvers.gradient_descent import GradientDescent
from diff_soup.inference.solvers.levenberg_marquardt import LevenbergMarquardt
from diff_soup.inference.solvers.newton_trust_region import NewtonTrustRegion
from diff_soup.inference.solvers.optax_solver import Optax
from diff_soup.inference.statistics.bootstrap import (
    BootstrapBand,
    BootstrapInterval,
    BootstrapResult,
    bootstrap,
)
from diff_soup.inference.statistics.cluster import cluster_covariance
from diff_soup.inference.statistics.diagnostics import mcmc_report, residual_report
from diff_soup.inference.statistics.estimation import MAP, MLE
from diff_soup.inference.statistics.hierarchy import Hierarchy
from diff_soup.inference.statistics.identifiability import identifiability
from diff_soup.inference.statistics.information import information
from diff_soup.inference.statistics.laplace import laplace
from diff_soup.inference.statistics.likelihoods import (
    CorrelatedGaussian,
    Gaussian,
    LogNormal,
    MultiplicativeGaussian,
    Poisson,
    StudentT,
)
from diff_soup.inference.statistics.measurement import Experiment, Measurement
from diff_soup.inference.statistics.parameters import Gauge
from diff_soup.inference.statistics.practical import practical_report
from diff_soup.inference.statistics.predict import Prediction, PredictionBand, predict
from diff_soup.inference.statistics.priors import (
    GaussianCoefficientPrior,
    LogNormalPrior,
    NormalPrior,
)
from diff_soup.inference.statistics.profile import profile
from diff_soup.inference.statistics.report import function_report
from diff_soup.inference.statistics.sample import Draws, sample
from diff_soup.inference.statistics.study import Study, balance_owners

# The integrator's first-step policy at a window restart, declared on
# ``Multishoot(first_step=...)``: ``FirstStep.auto()``, ``.fixed(dt)``,
# ``.fraction(f)``, ``.hinted(...)``.
FirstStep = RestartStep

__all__ = [
    "StudentT",
    "CorrelatedGaussian",
    "Poisson",
    "LogNormal",
    "MultiplicativeGaussian",
    "GaussianCoefficientPrior",
    "residual_report",
    "predict",
    "Prediction",
    "PredictionBand",
    "practical_report",
    "function_report",
    # Fitting: the verb, problems, and composition nouns
    "optsolve",
    "InferenceProblem",
    "LoweredStudy",
    "prepare_windows",
    "checkpointed_map",
    "LeastSquares",
    "SingleShoot",
    "Multishoot",
    "Windows",
    "Continuity",
    "StateBasis",
    "FirstStep",
    "FixedInitialState",
    "LearnedInitialState",
    "learnables",
    "Bounds",
    "Tikhonov",
    "NoReg",
    "HessianScale",
    "XScale",
    "JacScale",
    "ConstantScale",
    "OptResult",
    "TerminationStatus",
    "OptDiagnostics",
    "LogCallback",
    "Callback",
    "CheckpointCallback",
    "StopFit",
    "Gauge",
    # Solver objects
    "LevenbergMarquardt",
    "NewtonTrustRegion",
    "BFGS",
    "GradientDescent",
    "TrustRegion",
    "Optax",
    "OptSolver",
    # Optimization problems
    "OptimizationProblem",
    "LeastSquaresProblem",
    # Constrained fitting
    "constrained_optsolve",
    "InequalityConstraints",
    # Statistics: measurements, studies, estimation problems
    "Measurement",
    "Experiment",
    "Study",
    "Hierarchy",
    "balance_owners",
    "MLE",
    "MAP",
    "Gaussian",
    "NormalPrior",
    "LogNormalPrior",
    # Statistics: analysis verbs
    "information",
    "laplace",
    "profile",
    "identifiability",
    "cluster_covariance",
    "bootstrap",
    "BootstrapResult",
    "BootstrapInterval",
    "BootstrapBand",
    "sample",
    "Draws",
    "mcmc_report",
    # Design
    "sensitivity_solve",
    # Runs of several processes
    "world",
]
