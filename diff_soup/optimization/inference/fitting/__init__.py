"""Numerical fitting engines, trajectory declarations, and optimizer results.

Use ``ds.inference.InferenceProblem`` with ``Experiment`` and ``Measurement``
to declare a fit. Structural leaf and joint classes remain internal to their
implementation modules.
"""

from diff_soup.inference.fitting._optsolve import optsolve
from diff_soup.inference.fitting.bounds import Bounds
from diff_soup.inference.fitting.callbacks import Callback, LogCallback
from diff_soup.inference.fitting.constrained import (
    ConstrainedOptResult,
    InequalityConstraints,
    constrained_optsolve,
)
from diff_soup.inference.fitting.diagnostics import IdentifiabilityWarning, OptDiagnostics
from diff_soup.inference.fitting.inference_problem import (
    FixedInitialState,
    LearnedInitialState,
    LeastSquares,
    Multishoot,
    SingleShoot,
)
from diff_soup.inference.fitting.learnables import learnables
from diff_soup.inference.fitting.regularization import NoReg, Regularizer, Tikhonov
from diff_soup.inference.fitting.result import (
    EvalCounts,
    OptInfo,
    OptResult,
    StepInfo,
    TerminationStatus,
    TrustRegionReport,
)
from diff_soup.inference.fitting.scales import ConstantScale, HessianScale, JacScale, XScale
from diff_soup.inference.fitting.windows import (
    Continuity,
    RestartStep,
    StateBasis,
    Windows,
)

__all__ = [
    # Formulations and composition nouns
    "LeastSquares",
    "SingleShoot",
    "Multishoot",
    "FixedInitialState",
    "LearnedInitialState",
    "learnables",
    # Windows and continuity
    "Windows",
    "Continuity",
    "StateBasis",
    "RestartStep",
    # Bounds, scaling, regularization
    "Bounds",
    "XScale",
    "JacScale",
    "HessianScale",
    "ConstantScale",
    "Regularizer",
    "Tikhonov",
    "NoReg",
    # Composer, constrained variant, results, diagnostics
    "optsolve",
    "constrained_optsolve",
    "InequalityConstraints",
    "ConstrainedOptResult",
    "OptResult",
    "OptInfo",
    "TrustRegionReport",
    "StepInfo",
    "EvalCounts",
    "TerminationStatus",
    "OptDiagnostics",
    "IdentifiabilityWarning",
    # Callbacks
    "Callback",
    "LogCallback",
]
