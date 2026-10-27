"""diff_soup.timestepping: the scalar solver suite, events, and output policy.

This package exposes the public time-integration surface:

* The **scalar solver suite** — ``JNDF``: the variable-step / variable-order
  JAX NDF/BDF integrator class. It returns the shared ``Solution`` record.
  (The sibling engine classes ``JBDF`` / ``JIDA`` live at
  ``diff_soup.timestepping.vsvo.classes`` and are re-exported from the
  top-level ``diff_soup`` namespace alongside ``JNDF``.)
* **Events** — ``Event`` / ``EventSet`` / ``EventObservable`` / ``Bisection`` /
  ``terminate_steady_state``: the root-find / observation / termination surface.
* **Output policy** — ``SaveAt`` / ``SubSaveAt`` select which time samples a
  solve records; ``DenseOutputPolicy`` selects how those samples enforce
  algebraic consistency. ``CheckpointAdjoint`` is the checkpointed reverse-mode
  adjoint.
Solver problems expose their shared consistent-initialization machinery
through ``SolverView.consistent_initial_state``.

``Newton`` / ``BoxConstrainedActiveSet`` are re-exported here as the single
re-export point for the nonlinear swap surface.
"""

from diff_soup._adjoints import CheckpointAdjoint, ForwardAdjoint
from diff_soup.nonlinear._api import (
    BoxConstrainedActiveSet,
    Newton,
)
from diff_soup.timestepping.control_state import ControlState
from diff_soup.timestepping.events import (
    Bisection,
    Event,
    EventObservable,
    EventSet,
    terminate_steady_state,
)
from diff_soup.timestepping.saveat import DenseOutputPolicy, SaveAt, SubSaveAt
from diff_soup.timestepping.tolerances import ResolvedTolerances, ToleranceSpec

# The scalar solver suite and the shared output record are re-exported here
# under the same public names the top-level ``diff_soup`` namespace carries.
# They are resolved lazily so importing a timestepping submodule never eagerly
# pulls the solver classes back into a still-initializing ``diff_soup`` package.
_LAZY_REEXPORTS = {
    "JNDF": ("diff_soup.timestepping.vsvo.classes", "JNDF"),
    "DiffraxStepper": (
        "diff_soup.timestepping.diffrax_wrapper",
        "DiffraxStepper",
    ),
    "Solution": ("diff_soup.timestepping.vsvo.solution", "Solution"),
}


def __getattr__(name):
    """Resolve the lazily re-exported solver-suite names on first access."""
    target = _LAZY_REEXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = target
    return getattr(importlib.import_module(module_name), attr)


__all__ = [
    # ---- Scalar solver suite ----
    "JNDF",
    "DiffraxStepper",
    # ---- Nonlinear swap surface (single re-export rule) ----
    "Newton",
    "BoxConstrainedActiveSet",
    # ---- Output ----
    "Solution",
    "SaveAt",
    "SubSaveAt",
    "DenseOutputPolicy",
    "ToleranceSpec",
    "ResolvedTolerances",
    # ---- Differentiation strategies ----
    "ForwardAdjoint",
    "CheckpointAdjoint",
    "ControlState",
    # ---- Events ----
    "Event",
    "EventSet",
    "EventObservable",
    "Bisection",
    "terminate_steady_state",
]
