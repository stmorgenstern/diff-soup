"""Sentinel integers for the BDF engine's step-retry control flow.

This module threads two small, distinct integer code spaces through the
step-retry logic, giving every consumer (the corrector, the linear-setup
gate, the convergence/error-test ladder) one pinned set of values instead
of re-declaring them:

- [NonlinearFlag][] -- the values the corrector and the step loop pass to each
  other to say *why* a call is happening or *what* should happen next: an
  input reason for a corrector call, a mid-step control code, or a
  request to retry the step attempt.
- [StepFlag][] -- the termination codes a step attempt can end on: either
  the shared success code or one of the specific failure codes describing
  which stage aborted (a linear setup or solve, a right-hand-side
  evaluation, or a nonlinear solve).

The linear-setup hint describing why setup is being called on a given step
attempt (no prior failure this attempt, a prior failure with possibly-stale
Jacobian data, or a prior failure with current Jacobian data) is threaded
as plain booleans by the nonlinear-solve module rather than as a sentinel
class here.
"""

from __future__ import annotations


class NonlinearFlag:
    """Reason codes for a corrector call and control codes for the step loop."""

    # --- step-loop control codes ---
    DO_ERROR_TEST = 2
    """Run the local-error test next (the corrector converged)."""
    PREDICT_AGAIN = 3
    """Re-predict and retry the step attempt with updated parameters."""
    TRY_AGAIN = 5
    """Retry the corrector with a freshly rebuilt residual."""

    # --- input reason codes for a corrector call ---
    FIRST_CALL = 6
    """This is the first corrector call of the solve."""
    PREV_CONV_FAIL = 7
    """The previous attempt's corrector failed to converge."""
    PREV_PROJ_FAIL = 8
    """The previous attempt's projection step failed (projection is not
    implemented by this engine; the code is retained for a stable,
    contiguous value range)."""
    PREV_ERR_FAIL = 9
    """The previous attempt failed the local-error test."""

    # --- recoverable right-hand-side failure ---
    RHSFUNC_RECVR = 10
    """The right-hand-side evaluation failed recoverably."""

    PREV_PREC_FAIL = 11
    """The previous attempt's corrector failure was attributed to the
    linear-solve precision: retry at the promoted rung with a fresh setup,
    without shrinking the step."""


class StepFlag:
    """Step-attempt termination codes.

    A shared success code, or one of the specific failure codes describing
    which stage aborted.
    """

    STEP_SUCCESS = 0
    """The step attempt succeeded."""
    ERROR_TEST_FAIL = -3
    """The local-error test failed repeatedly enough to abort the step."""
    CONVERGENCE_FAIL = -4
    """The corrector failed to converge repeatedly enough to abort the
    step."""
    LINEAR_SETUP_FAIL = -6
    """A linear-solver setup call failed unrecoverably."""
    LINEAR_SOLVE_FAIL = -7
    """A linear-solver solve call failed unrecoverably."""
    RHS_FAIL = -8
    """The right-hand-side evaluation failed unrecoverably."""
    REPEATED_RHS_ERR = -10
    """The right-hand-side evaluation failed recoverably often enough to
    abort the step."""
    UNRECOVERABLE_RHS_ERR = -11
    """The right-hand-side evaluation reported a recoverable failure at a
    point in the step where that failure could not be handled."""
    NONLINEAR_FAIL = -16
    """The corrector reported a failure the step attempt cannot retry."""


__all__ = ["NonlinearFlag", "StepFlag"]
