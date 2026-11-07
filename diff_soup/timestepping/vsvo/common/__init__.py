"""The engine-agnostic problem contract and error-weight laws the solver suite shares."""

from .driver import DriverConfig, run_engine
from .engine import Engine, StepReport

__all__ = [
    "Engine",
    "StepReport",
    "DriverConfig",
    "run_engine",
]
