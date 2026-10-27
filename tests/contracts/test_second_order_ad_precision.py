"""Second-order AD and precision policies: fp32 and mixed-mode (fp32_ir) derivative factors
under the checkpoint gradient and jax.hessian, and the full-precision checkpoint gradient
against the composable reference."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import diff_soup as ds
from tests.support.second_order import THETA, composable_reference, gradient_and_hessian, loss

#: Precision policies: a mixed mode (fp32_ir), the same with a full-precision reverse sweep,
#: and a half-precision factor (fp32).
_POLICIES = {
    "fp32_ir": ds.Precision(solve_mode="fp32_ir"),
    "reverse_fp64": ds.Precision(solve_mode="fp32_ir", reverse_sensitivity_solve_mode="fp64"),
    "fp32": ds.Precision(solve_mode="fp32"),
}
# Relative to the composable full-precision reference in the max norm: measured gradient
# 5.2e-14 / 5.3e-14 and Hessian 2.5e-13 / 2.5e-13 (fp32_ir / reverse_fp64), CPU VM 2026-10-22,
# pytest tests/contracts/test_second_order_ad_precision.py::test_a_mixed_precision_forward_factor_composes_under_the_hessian
_MIXED_BOUND = 5e-12
# measured gradient 2.4e-7 and Hessian 3.8e-7 (fp32), CPU VM 2026-10-22,
# pytest tests/contracts/test_second_order_ad_precision.py::test_a_half_precision_factor_composes_under_the_hessian_within_its_precision
_FP32_BOUND = 5e-6
# measured 1.1e-15 relative (fp64 gradient against the reference), CPU VM 2026-10-22,
# pytest tests/contracts/test_second_order_ad_precision.py::test_the_full_precision_checkpoint_gradient_equals_the_reference
_FULL_BOUND = 1e-12


def _relative(got, want):
    """The max-norm distance of ``got`` from ``want``, relative to ``want``."""
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.fixture(scope="module")
def reference():
    """The composable reverse-over-forward Hessian and forward gradient."""
    return composable_reference()


def _gradient_and_hessian(precision):
    """The checkpoint gradient and the forward pushes over it (the Hessian)."""
    return gradient_and_hessian(jax.grad(loss(ds.CheckpointAdjoint(), precision=precision)))


@pytest.mark.parametrize("policy", ["fp32_ir", "reverse_fp64"])
def test_a_mixed_precision_forward_factor_composes_under_the_hessian(reference, policy):
    """Under CheckpointAdjoint a mixed-mode (fp32_ir) forward sensitivity factor, with either
    reverse sweep, gives the full-precision gradient, and the forward pushes over it (the
    columns of jax.hessian) equal the reference Hessian: the implicit roots differentiate their
    rule, not the refinement."""
    value, hessian = _gradient_and_hessian(_POLICIES[policy])
    assert max(_relative(value, reference[1]), _relative(hessian, reference[0])) <= _MIXED_BOUND


def test_a_half_precision_factor_composes_under_the_hessian_within_its_precision(reference):
    """Under CheckpointAdjoint an fp32 factor gives the gradient and the Hessian of the
    full-precision reference to within the fp32 factor's accuracy."""
    value, hessian = _gradient_and_hessian(_POLICIES["fp32"])
    assert max(_relative(value, reference[1]), _relative(hessian, reference[0])) <= _FP32_BOUND


def test_the_full_precision_checkpoint_gradient_equals_the_reference(reference):
    """The reverse sweep of the checkpoint adjoint and the composable forward rule give the
    same gradient."""
    gradient = np.asarray(jax.grad(loss(ds.CheckpointAdjoint()))(THETA))
    assert _relative(gradient, reference[1]) <= _FULL_BOUND
