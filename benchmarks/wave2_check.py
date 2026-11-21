"""Check of the second wave of lane features: observations of the global unknowns.

Not part of the benchmark suite: a hand-run check, printed and asserted.

    JAX_ENABLE_X64=1 python -m benchmarks.wave2_check                     # one process
    JAX_ENABLE_X64=1 mpirun -n 3 python -m mpi4py -m benchmarks.wave2_check world schedule
    JAX_ENABLE_X64=1 mpirun -n 3 python -m mpi4py -m benchmarks.wave2_check world link

1. Observations that read the global unknowns. The schedule of
   ``benchmarks.computed_start_check`` (a held current, a held voltage, a rest) on
   ``u' = -a u + I``, the current ``I`` a global algebraic unknown held by the model's
   operational table, three cells from computed starts ``u(0) = s``. Each cell observes
   the field ``u``, the global ``I`` directly (``observe(state, globals)``, also at the
   two times a segment ends, where the sample belongs to the segment that ends) and the
   voltage ``u - 2 I`` through the model's declaration ``"voltage"``, which reads ``I``.
   ``J^T J`` and ``J^T r`` against the closed-form rows (their Jacobian by differences of
   the closed form), the gradient against five-point differences of the cost, the
   gradient mode against the blocks, the reverse-mode gradient, and a joint fit. Then a
   linked pair whose source output is the global ``I``: two driven cells
   ``z' = -b z + y(t)`` on the piecewise-linear signal of the source's current, under
   ``reads="declared"``; the same comparisons, joint and nested fits. Then a lane
   without a schedule: ``u' = -a u`` with a global ``g`` closed by ``g = 2 u``, observed
   directly and through a declaration, against ``2 s e^{-a t}``, a fit on the lanes and
   a fit of one experiment in one shoot (``InferenceProblem``).

``world``: the schedule study (one cell per process) or the linked study over the three
processes of an MPI world: ``J^T J`` against the closed form, the gradient against
differences, fits. The byte comparison of a study without the new features is the
``baseline`` mode of ``benchmarks.computed_start_check`` on two versions of the library.
"""

# ruff: noqa: I001

from __future__ import annotations

import sys

import diff_soup as ds  # diff_soup before jax
import jax.numpy as jnp
import numpy as np

from benchmarks.computed_start_check import (
    A_TRUE,
    S_TRUE,
    SCHEDULE_CELLS,
    SCHEDULE_POINT,
    SCHEDULE_START,
    SCHEDULE_TIMES,
    SEGMENTS,
    SIGMA,
    _Report,
    _solver,
    differences,
    fit,
    relative,
)
from diff_soup.domains import cartesian_grid_1d
from diff_soup.optimization import inference as inf
from diff_soup.optimization.inference.approx import PiecewiseLinear

# The global current is also sampled at the two times a segment ends.
GLOBAL_TIMES = np.sort(np.concatenate([SCHEDULE_TIMES, [0.6, 1.2]]))
OUTPUT_TIMES = np.linspace(0.13, 1.93, 19)
DRIVEN_TIMES = np.linspace(0.3, 1.9, 9)
DRIVEN_B = {"d1": 1.3, "d2": 0.9}


def global_recipe(name, *, inert_b=False):
    """``u' = -a u + I`` under an operational table holding the global current ``I``.

    The current is held, or the voltage ``u - 2 I`` (the declaration ``"voltage"``, which
    reads ``I``). ``s`` is the start of ``u``, read by no equation; ``inert_b`` declares
    an unused rate ``b``. Returns ``(model, a, s)``.
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    a = spec.param("a", value=0.5, learnable=True, bounds=(0.05, 3.0))
    s = spec.param("s", value=1.0, learnable=True, bounds=(0.1, 5.0))
    if inert_b:
        spec.param("b", value=0.5, learnable=True, bounds=(0.05, 3.0))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.global_var("I_cell", initial=0.0, mass="algebraic")
    spec.observe.scalar(name="current", fn=lambda q: q["I_cell"])
    spec.observe.scalar(name="voltage", fn=lambda q: jnp.mean(q.u) - 2.0 * q["I_cell"])

    @spec.equation(writes="u", reads=("u", "I_cell"))
    def balance(q):
        return -q.p.a * q.u + q["I_cell"]

    spec.control.operational(
        name="terminal",
        current="current",
        voltage="voltage",
        current_unknown="I_cell",
        modes=("current", "voltage"),
    )
    return spec.build(), a, s


def truth_at(a, s, times):
    """``(u, I)`` at ``times`` in closed form; a time a segment ends belongs to that segment."""
    (_, i1, d1), (_, v2, d2), (_, i3, _) = SEGMENTS
    k = 0.5 - a  # u' = -a u + (u - V) / 2 while the voltage is held
    u1 = i1 / a + (s - i1 / a) * np.exp(-a * d1)
    u2 = v2 / (2 * k) + (u1 - v2 / (2 * k)) * np.exp(k * d2)
    us, currents = [], []
    for t in times:
        if t <= d1:
            u = i1 / a + (s - i1 / a) * np.exp(-a * t)
            current = i1
        elif t <= d1 + d2:
            u = v2 / (2 * k) + (u1 - v2 / (2 * k)) * np.exp(k * (t - d1))
            current = (u - v2) / 2
        else:
            u = i3 / a + (u2 - i3 / a) * np.exp(-a * (t - d1 - d2))
            current = i3
        us.append(u)
        currents.append(current)
    return np.asarray(us), np.asarray(currents)


def cell_series(a, s):
    """The observed series of one cell: ``u`` and ``u - 2 I`` at the schedule's times, ``I`` at ``GLOBAL_TIMES``."""
    u, current = truth_at(a, s, SCHEDULE_TIMES)
    _, current_all = truth_at(a, s, GLOBAL_TIMES)
    return u, current_all, u - 2.0 * current


def driven_truth(a, s, b):
    """``z`` at ``DRIVEN_TIMES`` from ``z = 0`` at the first output time.

    The signal is the piecewise-linear interpolant of the source's current at
    ``OUTPUT_TIMES``; on each piece it is ``alpha + beta t``, and
    ``z = c0 + c1 t + (z0 - c0) e^{-b t}`` with ``c1 = beta / b``, ``c0 = (alpha - c1) / b``.
    """
    knots = OUTPUT_TIMES
    values = truth_at(a, s, knots)[1]
    out = []
    for t in DRIVEN_TIMES:
        z = 0.0
        for left, right, start, end in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
            if left >= t:
                break
            beta = (end - start) / (right - left)
            span = min(right, t) - left
            c1 = beta / b
            c0 = (start - c1) / b
            z = c0 + c1 * span + (z - c0) * np.exp(-b * span)
        out.append(z)
    return np.asarray(out)


def _gaussian(name, times, values, observe):
    return inf.Measurement(
        name,
        times=jnp.asarray(times),
        values=jnp.asarray(values),
        observe=observe,
        likelihood=inf.Gaussian(sigma=SIGMA),
    )


def _cell_measurements(cell):
    u, current, voltage = cell_series(A_TRUE, S_TRUE[cell])
    return (
        _gaussian("u", SCHEDULE_TIMES, u, lambda state: state[0]),
        _gaussian("I", GLOBAL_TIMES, current, lambda state, globals: globals[0]),
        _gaussian("V", SCHEDULE_TIMES, voltage, "voltage"),
    )


def _computed_start(s):
    rest, unit = jnp.asarray([0.0, 0.0]), jnp.asarray([1.0, 0.0])
    return inf.ComputedInitialState(lambda theta: rest + unit * theta[s])


def _schedule():
    return ds.Schedule([ds.Segment(duration=d, **{kind: v}) for kind, v, d in SEGMENTS])


def global_study(name, solver=None):
    """Three cells sharing ``a``, each with its own start ``s``, observing ``u``, ``I`` and the voltage.

    Returns the study and its true point ``[a, s_c1, s_c2, s_c3]``.
    """
    model, a, s = global_recipe(name)
    experiments = tuple(
        inf.Experiment(
            model=model,
            solver=_solver() if solver is None else solver,
            t_span=(0.0, 2.0),
            initial_state=_computed_start(s),
            measurements=_cell_measurements(cell),
            id=cell,
            schedule=_schedule(),
            parameters={"cell": cell},
        )
        for cell in SCHEDULE_CELLS
    )
    hierarchy = inf.Hierarchy(levels=("cell",), shared=(a,), per={"cell": (s,)})
    truth = np.asarray([A_TRUE] + [S_TRUE[cell] for cell in SCHEDULE_CELLS])
    return inf.Study(experiments, parameters=hierarchy), truth


def global_rows(point):
    """The whitened rows of ``global_study`` at ``[a, s_c...]`` in closed form."""
    rows = []
    for k, cell in enumerate(SCHEDULE_CELLS):
        got = np.concatenate(cell_series(point[0], point[1 + k]))
        want = np.concatenate(cell_series(A_TRUE, S_TRUE[cell]))
        rows.append((got - want) / SIGMA)
    return np.concatenate(rows)


def linked_global_study(name):
    """A schedule cell whose global current, an output, drives ``z' = -b z + y(t)`` in two cells.

    ``a`` and ``s`` are shared, ``b`` one per cell; the source sits in a cell of its own
    and declares an unused ``b``. The driven models do not declare ``a`` or ``s``
    (``reads="declared"``). Returns the study and its true point
    ``[a, s, b_source, b_d1, b_d2]``.
    """
    model, a, s = global_recipe(f"{name}_source", inert_b=True)
    dspec = ds.problem(f"{name}_driven")
    dspec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    dspec.param("b", value=0.5, learnable=True, bounds=(0.05, 3.0))
    knots = jnp.asarray(OUTPUT_TIMES)
    dspec.function("y", PiecewiseLinear(knots, jnp.zeros(knots.shape)), learnable=False)
    dspec.field("z", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))

    @dspec.equation(writes="z", reads=("z",))
    def relax(q):
        return -q.p.b * q.z + q.fn.y(q.t)

    driven_model = dspec.build()
    s_true = S_TRUE["c1"]
    u, current, _ = cell_series(A_TRUE, s_true)
    experiments = [
        inf.Experiment(
            model=model,
            solver=_solver(),
            t_span=(0.0, 2.0),
            initial_state=_computed_start(s),
            measurements=(
                _gaussian("u", SCHEDULE_TIMES, u, lambda state: state[0]),
                _gaussian("I", GLOBAL_TIMES, current, lambda state, globals: globals[0]),
            ),
            id="source",
            schedule=_schedule(),
            outputs=(
                inf.Output("I", times=OUTPUT_TIMES, observe=lambda state, globals: globals[0]),
            ),
            parameters={"cell": "source"},
        )
    ]
    for cell, b_true in DRIVEN_B.items():
        experiments.append(
            inf.Experiment(
                model=driven_model,
                solver=_solver(),
                t_span=(float(OUTPUT_TIMES[0]), float(OUTPUT_TIMES[-1])),
                initial_state=dspec.assemble_state({}),
                measurements=(
                    _gaussian(
                        "z", DRIVEN_TIMES, driven_truth(A_TRUE, s_true, b_true), lambda st: st[0]
                    ),
                ),
                id=cell,
                links=(inf.Link(source="source", output="I", into="y"),),
                parameters={"cell": cell},
            )
        )
    source_b = experiments[0]._declarations["b"]
    hierarchy = inf.Hierarchy(levels=("cell",), shared=(a, s), per={"cell": (source_b,)})
    study = inf.Study(tuple(experiments), parameters=hierarchy, reads="declared")
    truth = np.asarray([A_TRUE, s_true, 0.5, DRIVEN_B["d1"], DRIVEN_B["d2"]])
    return study, truth


def linked_rows(point):
    """The whitened rows of ``linked_global_study`` at ``[a, s, b_source, b_d1, b_d2]``."""
    a, s = point[0], point[1]
    s_true = S_TRUE["c1"]
    got = cell_series(a, s)[:2]
    want = cell_series(A_TRUE, s_true)[:2]
    rows = [(np.concatenate(got) - np.concatenate(want)) / SIGMA]
    for k, (cell, b_true) in enumerate(DRIVEN_B.items()):
        rows.append(
            (driven_truth(a, s, point[3 + k]) - driven_truth(A_TRUE, s_true, b_true)) / SIGMA
        )
    return np.concatenate(rows)


def closed_form_blocks(rows_of, point, step=1e-6):
    """``J^T J`` and ``J^T r`` from closed-form rows, ``J`` by central differences of them."""
    rows = rows_of(point)
    columns = []
    for k in range(point.size):
        shift = np.zeros(point.size)
        shift[k] = step
        columns.append((rows_of(point + shift) - rows_of(point - shift)) / (2 * step))
    J = np.stack(columns, axis=1)
    return J.T @ J, J.T @ rows


CLOSURE_CELLS = ("c1", "c2")


def closure_study(name):
    """``u' = -a u`` with a global ``g`` closed by ``g = 2 u``, no schedule; two cells sharing ``a``.

    Each cell starts from its true ``u(0) = s`` and observes ``g`` twice: directly
    (``observe(state, globals)``) and through the declaration ``"g"``. Returns the study and
    the handle ``a``.
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    a = spec.param("a", value=0.5, learnable=True, bounds=(0.05, 3.0))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.global_var("g", initial=0.0)
    spec.observe.scalar(name="g", fn=lambda q: q["g"])

    @spec.equation(writes="u", reads=("u",))
    def decay(q):
        return -q.p.a * q.u

    spec.constraint.galvanostatic(
        name="closure",
        closes="g",
        current_fn=lambda q: q["g"] - 2.0 * jnp.mean(q.u),
        target_fn=lambda time: jnp.asarray(0.0),
    )
    model = spec.build()
    experiments = []
    for cell in CLOSURE_CELLS:
        values = 2.0 * S_TRUE[cell] * np.exp(-A_TRUE * SCHEDULE_TIMES)
        experiments.append(
            inf.Experiment(
                model=model,
                solver=_solver(),
                t_span=(0.0, 2.0),
                initial_state=spec.assemble_state({"u": S_TRUE[cell]}),
                measurements=(
                    _gaussian("g direct", SCHEDULE_TIMES, values, lambda st, globals: globals[0]),
                    _gaussian("g declared", SCHEDULE_TIMES, values, "g"),
                ),
                id=cell,
            )
        )
    return inf.Study(tuple(experiments), parameters=inf.Hierarchy(shared=(a,))), a


def closure_rows(point):
    """The whitened rows of ``closure_study`` at ``[a]``: ``2 s e^{-a t}`` twice per cell."""
    rows = []
    for cell in CLOSURE_CELLS:
        got = 2.0 * S_TRUE[cell] * np.exp(-point[0] * SCHEDULE_TIMES)
        want = 2.0 * S_TRUE[cell] * np.exp(-A_TRUE * SCHEDULE_TIMES)
        rows += [(got - want) / SIGMA] * 2
    return np.concatenate(rows)


LINK_POINT = np.asarray([0.7, 1.7, 0.5, 1.1, 1.0])
LINK_START = np.asarray([0.6, 1.2, 0.5, 0.7, 0.7])


def section_globals(report) -> None:
    """Section 1: observations of the global unknowns, in one process."""
    study, truth = global_study("wave2_globals")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient"))
    cost, gradient, matrix, failed = lowered.gauss_newton_fn(SCHEDULE_POINT)
    jtj, jtr = closed_form_blocks(global_rows, SCHEDULE_POINT)
    report["globals cost vs closed form"] = relative(
        cost, 0.5 * np.sum(global_rows(SCHEDULE_POINT) ** 2)
    )
    report["globals J^T J vs closed form"] = relative(matrix, jtj)
    report["globals J^T r vs closed form"] = relative(gradient, jtr)
    report["globals J^T r vs differences"] = relative(
        gradient, differences(lowered, SCHEDULE_POINT)
    )
    report["globals gradient mode vs blocks"] = relative(
        lowered.value_and_gradient(SCHEDULE_POINT).gradients[0], gradient
    )
    assert failed == 0
    for key in (
        "globals cost vs closed form",
        "globals J^T J vs closed form",
        "globals J^T r vs closed form",
        "globals J^T r vs differences",
        "globals gradient mode vs blocks",
    ):
        assert report[key] < 1e-6, (key, report[key])
    result = fit(lowered, SCHEDULE_START)
    report["globals joint fit error"] = relative(result.params, truth)
    assert report["globals joint fit error"] < 1e-6

    # The reverse-mode gradient, through a solver with a reverse adjoint.
    reverse = ds.JNDF.stock(linear="dense", rtol=1e-10, atol=1e-12, adjoint=ds.CheckpointAdjoint())
    study, _ = global_study("wave2_globals_reverse", solver=reverse)
    reversed_ = ds.lower(study)
    reversed_.prepare(modes=("value", "gradient"))
    report["globals reverse-mode gradient vs blocks"] = relative(
        reversed_.value_and_gradient(SCHEDULE_POINT).gradients[0], gradient
    )
    assert report["globals reverse-mode gradient vs blocks"] < 1e-6

    # A linked pair whose source output is the global current.
    study, link_truth = linked_global_study("wave2_linked")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient", "frozen"))
    cost, gradient, matrix, failed = lowered.gauss_newton_fn(LINK_POINT)
    jtj, jtr = closed_form_blocks(linked_rows, LINK_POINT)
    report["link cost vs closed form"] = relative(cost, 0.5 * np.sum(linked_rows(LINK_POINT) ** 2))
    report["link J^T J vs closed form"] = relative(matrix, jtj)
    report["link J^T r vs closed form"] = relative(gradient, jtr)
    report["link J^T r vs differences"] = relative(gradient, differences(lowered, LINK_POINT))
    assert failed == 0
    for key in (
        "link cost vs closed form",
        "link J^T J vs closed form",
        "link J^T r vs closed form",
        "link J^T r vs differences",
    ):
        assert report[key] < 1e-6, (key, report[key])
    for nested_at in (None, "cell"):
        result = fit(lowered, LINK_START, nested_at=nested_at)
        key = f"link {'nested' if nested_at else 'joint'} fit error"
        report[key] = relative(result.params, link_truth)
        assert report[key] < 1e-6, result.params

    # A lane without a schedule: a global g closed by g = 2 u, observed directly and
    # through a declaration; and the same experiment fitted in one shoot.
    study, model_a = closure_study("wave2_closure")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient"))
    point = np.asarray([0.7])
    cost, gradient, matrix, failed = lowered.gauss_newton_fn(point)
    jtj, jtr = closed_form_blocks(closure_rows, point)
    report["no schedule J^T J vs closed form"] = relative(matrix, jtj)
    report["no schedule J^T r vs closed form"] = relative(gradient, jtr)
    report["no schedule J^T r vs differences"] = relative(gradient, differences(lowered, point))
    assert failed == 0
    for key in ("J^T J vs closed form", "J^T r vs closed form", "J^T r vs differences"):
        assert report[f"no schedule {key}"] < 1e-6, (key, report[f"no schedule {key}"])
    report["no schedule joint fit error"] = relative(
        fit(lowered, np.asarray([0.4])).params, np.asarray([A_TRUE])
    )
    assert report["no schedule joint fit error"] < 1e-6
    shoot = inf.InferenceProblem(study.experiments[0], parameters=model_a)
    one_shoot = inf.optsolve(shoot, inf.LevenbergMarquardt(max_steps=40), diagnostics=False)
    report["no schedule one-shoot fit error"] = relative(one_shoot.params, np.asarray([A_TRUE]))
    assert report["no schedule one-shoot fit error"] < 1e-6

    # An observation of globals on a model without them is refused.
    spec = ds.problem("wave2_no_globals")
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    spec.param("a", value=0.5, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

    @spec.equation(writes="u", reads=("u",))
    def decay(q):
        return -q.p.a * q.u

    try:
        inf.Experiment(
            model=spec.build(),
            solver=_solver(),
            t_span=(0.0, 1.0),
            initial_state=spec.assemble_state({"u": 1.0}),
            measurements=(_gaussian("I", [0.5, 1.0], [0.0, 0.0], lambda st, globals: globals[0]),),
        )
    except ValueError as error:
        report["globals on a model without them refused"] = str(error)[:90]
    else:  # pragma: no cover
        raise AssertionError("an observation of globals on a model without them was not refused")


def one_process() -> None:
    """Every section that runs in one process."""
    report = _Report()
    section_globals(report)
    print("ALL CHECKS PASSED (one process)")


def world(kind: str) -> None:
    """``schedule`` (the globals study, one cell per process) or ``link`` over the MPI world."""
    from mpi4py import MPI

    if kind == "schedule":
        study, truth = global_study("wave2_world_globals")
        point, start, rows_of, nested = SCHEDULE_POINT, SCHEDULE_START, global_rows, ()
    else:
        study, truth = linked_global_study("wave2_world_linked")
        point, start, rows_of, nested = LINK_POINT, LINK_START, linked_rows, ("cell",)
    out = {}
    with ds.distributed.run(study) as lowered:
        report = lowered.prepare(modes=("value", "blocks", "gradient", "frozen"))
        _, gradient, matrix, failed = lowered.gauss_newton_fn(point)
        jtj, jtr = closed_form_blocks(rows_of, point)
        out["processes"] = lowered.processes
        out["ranks"] = sorted(int(r) for r in report.placement.rank_of.values())
        out["J^T J vs closed form"] = relative(matrix, jtj)
        out["J^T r vs closed form"] = relative(gradient, jtr)
        out["J^T r vs differences"] = relative(gradient, differences(lowered, point))
        out["failed"] = int(failed)
        for nested_at in (None, *nested):
            result = fit(lowered, start, nested_at=nested_at)
            out[f"{'nested' if nested_at else 'joint'} fit error"] = relative(result.params, truth)
    if MPI.COMM_WORLD.rank == 0:
        for key, value in out.items():
            print(f"{kind} world, {key}: {value}")
        # The placement may batch the two driven lanes, which share a program, on one process.
        assert out["processes"] == 3 and out["failed"] == 0
        assert out["ranks"] == [0, 1, 2] if kind == "schedule" else len(set(out["ranks"])) >= 2
        for key, value in out.items():
            if key.endswith(("closed form", "differences", "fit error")):
                assert value < 1e-6, (key, value)
        print(f"ALL CHECKS PASSED ({kind}, world of three)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "one"
    if mode == "world":
        world(sys.argv[2])
    else:
        {"one": one_process}[mode]()
