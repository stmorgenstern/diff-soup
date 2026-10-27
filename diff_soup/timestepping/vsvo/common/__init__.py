"""The engine-agnostic problem contract and error-weight laws the solver suite shares."""

from .driver import DriverConfig, Engine, StepReport, run_engine

__all__ = [
    "Engine",
    "StepReport",
    "DriverConfig",
    "run_engine",
]
