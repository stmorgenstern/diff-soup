"""Control: a control problem over a schedule solve, its horizon's end and its numeric problem."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from tests.support.problems import operational_cell

control = ds.optimization.control


def test_a_last_save_time_at_the_grids_end_is_the_horizons_end():
    """A grid scaled by a reference time ends one rounding past the sum of its durations; the
    last save time there is the horizon's end, so the horizon evaluates."""
    # Three intervals of a (k/3)**0.25 grid over 1 s, in units of 0.3 s: the grid's end,
    # 3.3333333333333335, lies one ulp past its durations added in order, 3.333333333333333.
    grid = np.linspace(0.0, 1.0, 4) ** 0.25
    problem = control.ControlProblem(
        problem=operational_cell().build(),
        controls=control.PiecewiseConstant(
            mode="current", intervals=3, duration=np.diff(grid) / 0.3
        ),
        cost=lambda sol, u, args: jnp.sum(sol.ys[-1]),
        save_times=grid[1:] / 0.3,
        initial_state=jnp.concatenate([jnp.full((4,), 2.0), jnp.zeros((1,))]),
        solver=ds.JIDA(linear="dense"),
    )
    numeric = problem.numeric_problem(ds.optimization.SQP())
    assert np.isfinite(numeric.value(jnp.ones(3)))
