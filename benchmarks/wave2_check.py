"""Check of the second wave of lane features: observations of the global unknowns.

Not part of the benchmark suite: a hand-run check, printed and asserted.

    JAX_ENABLE_X64=1 python -m benchmarks.wave2_check                     # one process
    JAX_ENABLE_X64=1 mpirun -n 3 python -m mpi4py -m benchmarks.wave2_check world schedule
    JAX_ENABLE_X64=1 mpirun -n 3 python -m mpi4py -m benchmarks.wave2_check world link
    JAX_ENABLE_X64=1 mpirun -n 3 python -m mpi4py -m benchmarks.wave2_check world recipes
    JAX_ENABLE_X64=1 python -m benchmarks.wave2_check measure             # cost of a wide lane
    JAX_ENABLE_X64=1 python -m benchmarks.wave2_check saves               # ... by saved times
    JAX_ENABLE_X64=1 python -m benchmarks.wave2_check budget              # many saved times
    JAX_ENABLE_X64=1 python -m benchmarks.wave2_check repeat              # a solve repeats

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
   ``reads="declared"``; the same comparisons, joint and nested fits; ``predict`` of a
   driven cell against a solve with the source's series supplied by hand, its
   derivative against the closed form, and the residual reports and the fit's
   diagnostics, whose costs per measurement sum to the study's cost. Then a lane
   without a schedule: ``u' = -a u`` with a global ``g`` closed by ``g = 2 u``, observed
   directly and through a declaration, and ``u`` from the state and through a
   declaration, against the closed form, with unit scales and with a field of scale 10
   beside a global of scale 100 (the closures are written in internal units, the
   declarations report dimensional values, the state and ``globals`` hold internal
   ones); a fit on the lanes, and a fit of one experiment in one shoot
   (``InferenceProblem``).

2. A hierarchy over experiments of two recipes. A source of one recipe (the schedule cell,
   fitted ``a`` and ``s``) whose global current drives two cells of another
   (``z' = -(a + b) z + c_0 y(t) + c_1``); both recipes declare ``a``, shared, and the cells
   alone declare ``b`` and the array ``c`` of two entries, one value of each per cell. The
   hierarchy takes ``a`` and ``s`` from the source's recipe and ``b`` and ``c`` from the
   cells'; the source names no member of the level ``cell`` and sits directly under the
   study (``reads="declared"``). ``J^T J`` and ``J^T r`` against the closed form, the
   gradient against differences, joint and nested fits, each stopped and resumed from a
   checkpoint, the fitted vector's groups (each shared number once), ``predict`` of a cell
   and of the source, the residual reports, the costs per measurement and a fit's
   diagnostics, and the refusals of an
   experiment outside the level under ``reads="all"`` and of one that declares ``b``.

``budget``: a lane saves what its observations read, not its states. The wide lane on
2,000 cells observed at 4,000 times with 16 derivative directions: its states and their
tangents would take 4,000 x 2,000 x 17 doubles, 1.1 GB; the compiled ``blocks`` program
has to need less than 64 MB.

``repeat``: a solve on KLU repeats bit for bit whatever the process solved before it.
Damped oscillators ``u' = v + 0.01 u_xx + 0.1 r``, ``v' = -s(x) u - 50 v - u^3 + 10 sin 30 t``
on 200 cells with two algebraic unknowns per cell, ``0 = k e(x) r + w - u`` and
``0 = r + w / 2 - v``, whose pivots KLU takes on or off the diagonal by the size of
``k e(x)``: the ``value`` and ``blocks`` evaluations at ``k = 1``, again, and again after an
evaluation at ``k = 10^-3``, give the same bytes and the same work; the KLU counters of
each are printed (one full factorization per slot and evaluation, refactorizations
after it), and a digest of the bytes, to compare across processes and versions.

``world``: the schedule study (one cell per process), the linked study or the two-recipe study over the three
processes of an MPI world: ``J^T J`` against the closed form, the gradient against
differences, fits; for the linked study also ``predict`` and the residual reports. The byte comparison of a study without the new features is the
``baseline`` mode of ``benchmarks.computed_start_check`` on two versions of the library.
"""

# ruff: noqa: I001

from __future__ import annotations

import dataclasses
import sys
import warnings

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


def driven_recipe(name, values=None):
    """``z' = -b z + y(t)`` with ``y`` piecewise linear on ``OUTPUT_TIMES``, its values ``values`` (zeros).

    Returns ``(model, spec)``.
    """
    dspec = ds.problem(name)
    dspec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    dspec.param("b", value=0.5, learnable=True, bounds=(0.05, 3.0))
    knots = jnp.asarray(OUTPUT_TIMES)
    values = jnp.zeros(knots.shape) if values is None else jnp.asarray(values)
    dspec.function("y", PiecewiseLinear(knots, values), learnable=False)
    dspec.field("z", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))

    @dspec.equation(writes="z", reads=("z",))
    def relax(q):
        return -q.p.b * q.z + q.fn.y(q.t)

    return dspec.build(), dspec


def linked_global_study(name):
    """A schedule cell whose global current, an output, drives ``z' = -b z + y(t)`` in two cells.

    ``a`` and ``s`` are shared, ``b`` one per cell; the source sits in a cell of its own
    and declares an unused ``b``. The driven models do not declare ``a`` or ``s``
    (``reads="declared"``). Returns the study and its true point
    ``[a, s, b_source, b_d1, b_d2]``.
    """
    model, a, s = global_recipe(f"{name}_source", inert_b=True)
    driven_model, dspec = driven_recipe(f"{name}_driven")
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


TWO_RECIPE_B = {"d1": 1.3, "d2": 0.9}
TWO_RECIPE_C = {"d1": (0.8, 0.2), "d2": (1.2, -0.1)}


def recipe_b(name, values=None):
    """``z' = -(a + b) z + c_0 y(t) + c_1``, ``y`` piecewise linear on ``OUTPUT_TIMES`` (zeros).

    It declares ``a`` under the name and the form of the source's (``global_recipe``),
    a rate ``b`` and an array ``c`` of two entries. Returns ``(model, spec, a, b, c)``.
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    a = spec.param("a", value=0.5, learnable=True, bounds=(0.05, 3.0))
    b = spec.param("b", value=0.5, learnable=True, bounds=(0.05, 3.0))
    c = spec.param("c", value=jnp.asarray([1.0, 0.0]), learnable=True)
    knots = jnp.asarray(OUTPUT_TIMES)
    values = jnp.zeros(knots.shape) if values is None else jnp.asarray(values)
    spec.function("y", PiecewiseLinear(knots, values), learnable=False)
    spec.field("z", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))

    @spec.equation(writes="z", reads=("z",))
    def relax(q):
        return -(q.p.a + q.p.b) * q.z + q.p.c[0] * q.fn.y(q.t) + q.p.c[1]

    return spec.build(), spec, a, b, c


def recipe_b_truth(a, s, b, c):
    """``z`` of ``recipe_b`` at ``DRIVEN_TIMES`` from ``z = 0`` at the first output time.

    On each piece of the source's current, ``y = alpha + beta t`` and the forcing is
    ``A + B t`` with ``A = c_0 alpha + c_1``, ``B = c_0 beta``; with ``k = a + b``,
    ``z = C_0 + C_1 t + (z_0 - C_0) e^{-k t}``, ``C_1 = B / k``, ``C_0 = (A - C_1) / k``.
    """
    knots = OUTPUT_TIMES
    values = truth_at(a, s, knots)[1]
    k = a + b
    out = []
    for t in DRIVEN_TIMES:
        z = 0.0
        for left, right, start, end in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
            if left >= t:
                break
            span = min(right, t) - left
            slope = c[0] * (end - start) / (right - left)
            c1 = slope / k
            c0 = (c[0] * start + c[1] - c1) / k
            z = c0 + c1 * span + (z - c0) * np.exp(-k * span)
        out.append(z)
    return np.asarray(out)


def two_recipe_study(name):
    """A source of one recipe whose global current drives two cells of another.

    The source (``global_recipe``, fitted ``a`` and ``s``) names no member of the level
    ``cell``: its model declares none of the level's parameters, so under
    ``reads="declared"`` it sits directly under the study. The cells (``recipe_b``)
    declare ``a`` too, shared with the source; ``b`` and the array ``c`` take one value
    per cell. The hierarchy takes ``a`` and ``s`` from the source's recipe and ``b`` and
    ``c`` from the cells'. Returns the study and its true point
    ``[a, s, b_d1, c_d1 (2), b_d2, c_d2 (2)]``.
    """
    model, a, s = global_recipe(f"{name}_source")
    driven_model, dspec, _, b, c = recipe_b(f"{name}_cells")
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
        )
    ]
    for cell in TWO_RECIPE_B:
        z = recipe_b_truth(A_TRUE, s_true, TWO_RECIPE_B[cell], TWO_RECIPE_C[cell])
        experiments.append(
            inf.Experiment(
                model=driven_model,
                solver=_solver(),
                t_span=(float(OUTPUT_TIMES[0]), float(OUTPUT_TIMES[-1])),
                initial_state=dspec.assemble_state({}),
                measurements=(_gaussian("z", DRIVEN_TIMES, z, lambda st: st[0]),),
                id=cell,
                links=(inf.Link(source="source", output="I", into="y"),),
                parameters={"cell": cell},
            )
        )
    hierarchy = inf.Hierarchy(levels=("cell",), shared=(a, s), per={"cell": (b, c)})
    study = inf.Study(tuple(experiments), parameters=hierarchy, reads="declared")
    truth = np.asarray(
        [A_TRUE, s_true]
        + [value for cell in TWO_RECIPE_B for value in (TWO_RECIPE_B[cell], *TWO_RECIPE_C[cell])]
    )
    return study, truth


def two_recipe_rows(point):
    """The whitened rows of ``two_recipe_study`` at ``[a, s, b_d1, c_d1, b_d2, c_d2]``."""
    a, s = point[0], point[1]
    s_true = S_TRUE["c1"]
    got = cell_series(a, s)[:2]
    want = cell_series(A_TRUE, s_true)[:2]
    rows = [(np.concatenate(got) - np.concatenate(want)) / SIGMA]
    for k, cell in enumerate(TWO_RECIPE_B):
        own = point[2 + 3 * k : 5 + 3 * k]
        z = recipe_b_truth(a, s, own[0], own[1:])
        rows.append(
            (z - recipe_b_truth(A_TRUE, s_true, TWO_RECIPE_B[cell], TWO_RECIPE_C[cell])) / SIGMA
        )
    return np.concatenate(rows)


TWO_RECIPE_POINT = np.asarray([0.7, 1.7, 1.1, 0.9, 0.1, 1.0, 1.1, 0.0])
TWO_RECIPE_START = np.asarray([0.6, 1.2, 0.7, 1.0, 0.0, 0.7, 1.0, 0.0])


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


def link_reports(lowered, point):
    """What ``predict`` and the residual reports say of the linked study at ``point``.

    ``predict`` of the driven cell ``d1`` from two draws at ``point`` (its band's mean
    is the prediction there) against a direct solve of the driven model with the
    source's current at ``OUTPUT_TIMES`` supplied by hand (the closed form) and against
    the closed form of ``z``; the derivative of that prediction (the linearization
    ``predict`` makes from a fit) against central differences of the closed form, in
    every column, the source's ``a`` and ``s`` included; the residual report of every
    experiment and the cost per measurement against the study's cost. Returns the
    numbers by name.
    """
    import jax

    from diff_soup.optimization.inference.statistics._analysis import (
        measurement_costs,
        study_predictor,
    )

    a, s, b = point[0], point[1], point[3]
    out = {}
    draws = inf.Draws(
        values=jnp.asarray(np.stack([point, point]))[None],
        names=tuple(f"x{k}" for k in range(point.size)),
        info={},
        _parameter_schema=lowered.parameter_schema,
    )
    predicted = np.asarray(inf.predict(lowered, draws, experiment_id="d1")["z"].mean)
    source_current = truth_at(a, s, OUTPUT_TIMES)[1]
    model, spec = driven_recipe(f"wave2_by_hand_{lowered.processes}", source_current)
    by_hand = inf.Experiment(
        model=model,
        solver=_solver(),
        t_span=(float(OUTPUT_TIMES[0]), float(OUTPUT_TIMES[-1])),
        initial_state=spec.assemble_state({}),
        measurements=(
            _gaussian("z", DRIVEN_TIMES, np.zeros(DRIVEN_TIMES.shape), lambda st: st[0]),
        ),
    )
    solved, failed = by_hand.predictor()(jnp.asarray([b]))
    assert not bool(failed)
    out["predict d1 vs solve with the source's series by hand"] = relative(predicted, solved["z"])
    out["predict d1 vs closed form"] = relative(predicted, driven_truth(a, s, b))
    _, _, _, predict = study_predictor(lowered, "d1")
    jacobian = np.asarray(jax.jacfwd(lambda x: predict(x)[0]["z"])(jnp.asarray(point)))
    step, columns = 1e-6, []
    for k in range(point.size):
        shift = np.zeros(point.size)
        shift[k] = step
        plus, minus = point + shift, point - shift
        columns.append(
            (driven_truth(plus[0], plus[1], plus[3]) - driven_truth(minus[0], minus[1], minus[3]))
            / (2 * step)
        )
    out["predict d1 derivative vs closed form"] = relative(jacobian, np.stack(columns, axis=1))
    cost = float(lowered.value_and_gradient(point).values[0])
    reports = {
        experiment_id: inf.residual_report(lowered, point, experiment_id=experiment_id)
        for experiment_id in ("source", "d1", "d2")
    }
    summed = sum(c.cost for r in reports.values() for c in r.channels.values()) / 2
    out["residual reports, sum of costs vs study cost"] = relative(summed, cost)
    rows = (driven_truth(a, s, b) - driven_truth(A_TRUE, S_TRUE["c1"], DRIVEN_B["d1"])) / SIGMA
    out["residual report d1 vs closed form"] = relative(
        reports["d1"].channels["z"].cost, np.sum(rows**2)
    )
    costs, names = measurement_costs(lowered, point)
    out["cost per measurement, sum vs study cost"] = relative(sum(costs), cost)
    out["cost per measurement, names"] = names
    return out


def check_link_reports(report, numbers, label="link"):
    """Record ``link_reports`` numbers and assert each comparison."""
    for key, value in numbers.items():
        report[f"{label} {key}"] = value
        if key != "cost per measurement, names":
            assert value < 1e-6, (key, value)
    assert numbers["cost per measurement, names"] == (
        ("source", "u"),
        ("source", "I"),
        ("d1", "z"),
        ("d2", "z"),
    )


def check_link_diagnostics(report, result, label="link"):
    """Check a fit's diagnostics on the linked study.

    The costs per measurement sum to the fit's cost, and only the source's unused ``b``
    is flagged as not moving the residual.
    """
    diagnostics = result.diagnostics
    report[f"{label} fit diagnostics, cost per measurement minus cost"] = abs(
        sum(diagnostics.per_series_cost) - float(result.cost)
    )
    report[f"{label} fit diagnostics, warnings"] = diagnostics.warnings
    assert report[f"{label} fit diagnostics, cost per measurement minus cost"] < 1e-10
    assert len(diagnostics.warnings) == 1 and "('b@source',)" in diagnostics.warnings[0]
    assert diagnostics.series_names[2] == ("d1", "z")


CLOSURE_CELLS = ("c1", "c2")


def closure_study(name, field_scale=1.0, global_scale=1.0):
    """``u' = -a u`` with a global ``g`` closed by ``g = 2 u``, no schedule; two cells sharing ``a``.

    The equations and the closure are written, as every closure of the library, in the
    solver's internal units: the dimensional value of ``u`` is ``field_scale`` times its
    internal value, that of ``g`` is ``global_scale`` times its own. Each cell starts from
    its true dimensional ``u(0) = s`` and observes ``g`` and ``u`` four ways: ``g`` directly
    (``observe(state, globals)``, internal, ``2 u_int``), ``g`` through the declaration
    ``"g"`` (dimensional, ``global_scale * 2 u_int``), ``u`` from the state (internal) and
    through the declaration ``"u"`` (dimensional, ``s e^{-a t}``). Returns the study and
    the handle ``a``.
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    a = spec.param("a", value=0.5, learnable=True, bounds=(0.05, 3.0))
    spec.field("u", mass="differential", scale=field_scale, init=lambda x: jnp.ones_like(x))
    spec.global_var("g", initial=0.0, scale=global_scale)
    spec.observe.scalar(name="g", fn=lambda q: q["g"])
    spec.observe.scalar(name="u", fn=lambda q: jnp.mean(q.u))

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
        series = closure_series(A_TRUE, S_TRUE[cell], field_scale, global_scale)
        observes = (
            lambda st, globals: globals[0],
            "g",
            lambda st: st[0],
            "u",
        )
        names = ("g direct", "g declared", "u state", "u declared")
        experiments.append(
            inf.Experiment(
                model=model,
                solver=_solver(),
                t_span=(0.0, 2.0),
                initial_state=spec.assemble_state({"u": S_TRUE[cell]}),
                measurements=tuple(
                    _gaussian(label, SCHEDULE_TIMES, values, observe)
                    for label, values, observe in zip(names, series, observes)
                ),
                id=cell,
            )
        )
    return inf.Study(tuple(experiments), parameters=inf.Hierarchy(shared=(a,))), a


def closure_series(a, s, field_scale=1.0, global_scale=1.0):
    """The four series of one cell of ``closure_study`` at ``SCHEDULE_TIMES``, in closed form.

    ``u_int = s e^{-a t} / field_scale`` and ``g_int = 2 u_int``; the declarations report
    ``global_scale * g_int`` and ``field_scale * u_int``.
    """
    u_int = s * np.exp(-a * SCHEDULE_TIMES) / field_scale
    g_int = 2.0 * u_int
    return g_int, global_scale * g_int, u_int, field_scale * u_int


def closure_rows_of(field_scale=1.0, global_scale=1.0):
    """The whitened rows of ``closure_study`` at ``[a]`` in closed form, as a function of the point."""

    def rows(point):
        out = []
        for cell in CLOSURE_CELLS:
            got = closure_series(point[0], S_TRUE[cell], field_scale, global_scale)
            want = closure_series(A_TRUE, S_TRUE[cell], field_scale, global_scale)
            out += [(x - y) / SIGMA for x, y in zip(got, want)]
        return np.concatenate(out)

    return rows


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

    # predict and the residual reports of the linked study follow its links.
    check_link_reports(report, link_reports(lowered, LINK_POINT))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = inf.optsolve(lowered.with_start(LINK_START), inf.TrustRegion(max_steps=60))
    check_link_diagnostics(report, result)

    # A lane without a schedule: a global g closed by g = 2 u, observed directly and
    # through a declaration; with unit scales, and with a field of scale 10 beside a
    # global of scale 100; the unit-scale experiment also fitted in one shoot.
    point = np.asarray([0.7])
    for label, field_scale, global_scale in (
        ("no schedule", 1.0, 1.0),
        ("scaled (u 10, g 100)", 10.0, 100.0),
    ):
        study, model_a = closure_study(
            f"wave2_closure_{int(field_scale)}_{int(global_scale)}", field_scale, global_scale
        )
        rows_of = closure_rows_of(field_scale, global_scale)
        lowered = ds.lower(study)
        lowered.prepare(modes=("value", "blocks", "gradient"))
        cost, gradient, matrix, failed = lowered.gauss_newton_fn(point)
        jtj, jtr = closed_form_blocks(rows_of, point)
        report[f"{label} cost vs closed form"] = relative(cost, 0.5 * np.sum(rows_of(point) ** 2))
        report[f"{label} J^T J vs closed form"] = relative(matrix, jtj)
        report[f"{label} J^T r vs closed form"] = relative(gradient, jtr)
        report[f"{label} J^T r vs differences"] = relative(gradient, differences(lowered, point))
        assert failed == 0
        for key in (
            "cost vs closed form",
            "J^T J vs closed form",
            "J^T r vs closed form",
            "J^T r vs differences",
        ):
            assert report[f"{label} {key}"] < 1e-6, (label, key, report[f"{label} {key}"])
        report[f"{label} joint fit error"] = relative(
            fit(lowered, np.asarray([0.4])).params, np.asarray([A_TRUE])
        )
        assert report[f"{label} joint fit error"] < 1e-6
    study, model_a = closure_study("wave2_closure_shoot")
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


MEASURE_CELLS = 4000
MEASURE_TIMES = np.linspace(0.1, 2.0, 20)


def wide_study(name, width, cells=MEASURE_CELLS):
    """A wide lane: ``u' = D u_xx - k(x) u^3 + 4 sin(20 t) g(x)`` on ``cells`` cells, ``width`` rates.

    ``k`` is one learnable array of ``width`` entries, each the rate of one band of cells,
    so the lane has ``width`` derivative directions. Solved with KLU; the mean of ``u`` is
    observed at 20 times. Returns the study and its point (every rate at 1).
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(cells, 0.0, 1.0, method="fvm"))
    k = spec.param("k", value=jnp.ones(width), learnable=True)
    band = jnp.asarray(np.minimum(np.arange(cells) * width // cells, width - 1))
    bump = jnp.asarray(np.exp(-(((np.arange(cells) + 0.5) / cells - 0.5) ** 2) / 0.02))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

    @spec.equation(writes="u", reads=("u",))
    def reaction_diffusion(q):
        return 0.05 * q.ops.laplacian(q.u) - q.p.k[band] * q.u**3 + 4.0 * jnp.sin(20.0 * q.t) * bump

    model = spec.build()
    solver = ds.JNDF.stock(linear="klu", rtol=1e-8, atol=1e-10)
    values = np.zeros(MEASURE_TIMES.shape)
    experiment = inf.Experiment(
        model=model,
        solver=solver,
        t_span=(0.0, 2.0),
        initial_state=spec.assemble_state({"u": 1.0}),
        measurements=(_gaussian("mean", MEASURE_TIMES, values, lambda st: jnp.mean(st)),),
        id="wide",
    )
    return inf.Study((experiment,), parameters=ds.ParameterSet(shared=(k,))), np.ones(width)


def _timed(evaluate, repeats=3):
    """The smallest wall time of ``repeats`` warm calls, after one call that compiles."""
    import time

    evaluate()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        evaluate()
        best = min(best, time.perf_counter() - start)
    return best


def linear_costs(n, widths, repeats=20):
    """Warm wall time of one KLU factorization followed by one solve of ``w`` right-hand sides.

    On the pattern of a one-dimensional three-point stencil of ``n`` unknowns, the pattern
    of the wide lane's iteration matrix. Returns ``{w: seconds}``.
    """
    import time

    import jax

    from diff_soup.linear import KLU, SolveContext, SparseTopology, analyze, factor, solve

    index = np.arange(n)
    rows = np.concatenate([index, index[1:], index[:-1]])
    cols = np.concatenate([index, index[:-1], index[1:]])
    analysis = analyze(SparseTopology(rows, cols, n), backend=KLU())
    values = jnp.asarray(np.concatenate([np.full(n, 4.0), np.full(2 * n - 2, -1.0)]))

    def best(fn, *args):
        jax.block_until_ready(fn(*args))
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(fn(*args))
            times.append(time.perf_counter() - start)
        return min(times)

    ctx = SolveContext()
    solves = {}
    for width in widths:
        b = jnp.ones((width, n))
        solves[width] = best(
            jax.jit(lambda v, b: solve(factor(analysis, v, ctx=ctx), b)[0]), values, b
        )
    return solves


def measure(widths=(1, 8, 16, 32, 50)) -> None:
    """Where the time of one ``blocks`` evaluation of a wide lane goes, on KLU.

    Prints, per derivative width, the warm wall time of a ``value`` and a ``blocks``
    evaluation, the solver's work (step attempts, Newton iterations, its own
    factorizations) and the KLU extension's counters over one evaluation of each mode
    (numeric factorizations and refactorizations, of which full factorizations, and
    solve calls, each call one block of right-hand sides).
    """
    from diff_soup.linear.klu_backend import KLUBackend
    from diff_soup.optimization.inference.engine._request import Request

    rows = []
    for width in widths:
        study, point = wide_study(f"wave2_wide_{width}", width)
        lowered = ds.lower(study)
        lowered.prepare(modes=("value", "blocks"))
        counted = {}
        times = {}
        for mode in ("value", "blocks"):
            request = Request(mode=mode, points=point)

            def run(request=request):
                return lowered.evaluate(request)

            times[mode] = _timed(run)
            KLUBackend.reset_native_stats()
            results = run()
            counted[mode] = (dict(KLUBackend.native_stats()), results.work)
        work = counted["value"][1]
        row = {
            "width": width,
            "value s": times["value"],
            "blocks s": times["blocks"],
            "accepted": int(np.sum(work.accepts)),
            "rejected": int(np.sum(work.rejects)),
            "attempts": int(np.sum(work.trial_steps)),
            "newton": int(np.sum(work.newton_iters)),
            "solver factorizations": int(np.sum(work.factorizations)),
        }
        for mode in ("value", "blocks"):
            stats = counted[mode][0]
            row[f"{mode} klu factor+refactor"] = int(
                stats.get("factorizations", 0) + stats.get("refactorizations", 0)
            )
            row[f"{mode} klu of which klu_factor"] = int(stats.get("factorizations", 0))
            row[f"{mode} klu solve calls"] = int(stats.get("solves", 0))
        rows.append(row)
        print(row, flush=True)
    for width, seconds in linear_costs(MEASURE_CELLS, widths).items():
        print(
            f"KLU on {MEASURE_CELLS} unknowns, factor + one solve of {width} columns: {seconds:.2e} s"
        )
    base = rows[0]
    print("width, blocks - value (s), per direction over width 1 (s), per direction / value")
    for row in rows:
        extra = row["blocks s"] - row["value s"]
        slope = (
            (row["blocks s"] - base["blocks s"]) / (row["width"] - base["width"])
            if row["width"] != base["width"]
            else float("nan")
        )
        print(row["width"], round(extra, 4), round(slope, 5), round(slope / row["value s"], 4))


def measure_saves(saves=(5, 20, 80, 320), widths=(1, 16, 50)) -> None:
    """Warm wall time of a ``value`` and a ``blocks`` evaluation of the wide lane, by saved times and width.

    Also prints the cost of one derivative direction, in seconds and in value solves,
    from the widths' slope.
    """
    global MEASURE_TIMES
    from diff_soup.optimization.inference.engine._request import Request

    kept = MEASURE_TIMES
    print("saves, width, value s, blocks s")
    try:
        for count in saves:
            MEASURE_TIMES = np.linspace(0.1, 2.0, count)
            times = {}
            for width in widths:
                study, point = wide_study(f"wave2_saves_{count}_{width}", width)
                lowered = ds.lower(study)
                lowered.prepare(modes=("value", "blocks"))
                value = _timed(lambda: lowered.evaluate(Request(mode="value", points=point)))
                blocks = _timed(lambda: lowered.evaluate(Request(mode="blocks", points=point)))
                times[width] = (value, blocks)
                print(count, width, round(value, 4), round(blocks, 4), flush=True)
            (v0, b0), (_, b1) = times[widths[0]], times[widths[-1]]
            slope = (b1 - b0) / (widths[-1] - widths[0])
            print(f"saves {count}: one direction {slope:.4f} s = {slope / v0:.2f} value solves")
    finally:
        MEASURE_TIMES = kept


BUDGET_BYTES = 64 * 2**20


def measure_budget(cells=2000, count=4000, width=16) -> None:
    """The device memory and wall time of the wide lane at many sample times, against a budget.

    Prints the memory each compiled program needs (arguments, outputs,
    temporaries and code), the warm wall time of a ``value`` and a ``blocks``
    evaluation, and the bytes the states and their tangents would take; asserts
    that the ``blocks`` program fits ``BUDGET_BYTES`` and those states would not.
    """
    global MEASURE_TIMES
    from diff_soup.optimization.inference.engine._request import Request

    kept = MEASURE_TIMES
    try:
        MEASURE_TIMES = np.linspace(0.1, 2.0, count)
        study, point = wide_study("wave2_budget", width, cells=cells)
        lowered = ds.lower(study)
        report = lowered.prepare(modes=("value", "blocks"))
        needs = {record.mode: record.device_bytes for record in report.records}
        for mode in ("value", "blocks"):
            seconds = _timed(lambda mode=mode: lowered.evaluate(Request(mode=mode, points=point)))
            print(f"budget, {mode}: {needs[mode] / 2**20:.1f} MB, {seconds:.3f} s")
        states = count * cells * (1 + width) * 8
        print(
            f"budget, states and tangents would take {states / 2**20:.0f} MB; budget "
            f"{BUDGET_BYTES / 2**20:.0f} MB"
        )
        result = lowered.evaluate(Request(mode="blocks", points=point))
        assert np.all(result.status.ok) and np.all(np.isfinite(result.normal))
        assert needs["blocks"] < BUDGET_BYTES < states, (needs, states)
        print("ALL CHECKS PASSED (budget)")
    finally:
        MEASURE_TIMES = kept


def repeat_study(name, cells=200):
    """The study of the ``repeat`` mode (see the module docstring); its one parameter is ``k``."""
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(cells, 0.0, 1.0, method="fvm"))
    k = spec.param("k", value=1.0, learnable=True)
    stiffness = jnp.asarray(10.0 ** np.linspace(2.0, 6.0, cells))
    small = jnp.asarray(10.0 ** np.linspace(-6.0, -1.0, cells))
    for field, mass in (
        ("u", "differential"),
        ("v", "differential"),
        ("r", "algebraic"),
        ("w", "algebraic"),
    ):
        spec.field(field, mass=mass, scale=1.0, init=lambda x: jnp.zeros_like(x))

    @spec.equation(writes="u", reads=("u", "v", "r"))
    def position(q):
        return q.v + 0.01 * q.ops.laplacian(q.u) + 0.1 * q.r

    @spec.equation(writes="v", reads=("u", "v"))
    def velocity(q):
        return -stiffness * q.u - 50.0 * q.v - q.u**3 + 10.0 * jnp.sin(30.0 * q.t)

    @spec.equation(writes="r", reads=("r", "w", "u"))
    def first_constraint(q):
        return q.p.k * small * q.r + q.w - q.u

    @spec.equation(writes="w", reads=("r", "w", "v"))
    def second_constraint(q):
        return q.r + 0.5 * q.w - q.v

    times = np.linspace(0.05, 1.0, 20)
    experiment = inf.Experiment(
        model=spec.build(),
        solver=ds.JNDF.stock(linear="klu", rtol=1e-6, atol=1e-9),
        t_span=(0.0, 1.0),
        initial_state=spec.assemble_state({"u": 1.0, "v": 0.0, "r": 0.0, "w": 0.0}),
        measurements=(_gaussian("mean", times, np.zeros(times.shape), lambda st: jnp.mean(st)),),
        id="oscillators",
    )
    return inf.Study((experiment,), parameters=ds.ParameterSet(shared=(k,)))


def repeat() -> None:
    """A KLU solve gives the same bytes first, again, and after an unrelated solve."""
    import hashlib

    from diff_soup.linear.klu_backend import KLUBackend
    from diff_soup.optimization.inference.engine._request import Request

    lowered = ds.lower(repeat_study("wave2_repeat"))
    lowered.prepare(modes=("value", "blocks"))
    counters = ("factorizations", "refactorizations", "refactor_fallbacks", "rebuilds", "solves")

    def run(mode, k):
        KLUBackend.reset_native_stats()
        r = lowered.evaluate(Request(mode=mode, points=np.asarray([k])))
        stats = KLUBackend.native_stats()
        arrays = [r.cost, r.work.accepts, r.work.rejects, r.work.newton_iters]
        data = b"".join(
            np.asarray(a).tobytes() for a in arrays + [r.normal, r.gradient] if a is not None
        )
        steps = (int(np.sum(r.work.accepts)), int(np.sum(r.work.rejects)))
        return data, steps, {key: int(stats[key]) for key in counters}

    for mode in ("value", "blocks"):
        first, second = run(mode, 1.0), run(mode, 1.0)
        other = run(mode, 1e-3)
        third = run(mode, 1.0)
        for tag, (data, steps, stats) in (
            ("first", first),
            ("again", second),
            ("k = 1e-3", other),
            ("after k = 1e-3", third),
        ):
            digest = hashlib.sha1(data).hexdigest()[:12]
            print(
                f"repeat, {mode}, {tag}: steps (accepted, rejected) {steps}, digest {digest}, KLU {stats}"
            )
        assert second[0] == first[0] and third[0] == first[0], mode
        assert second[2] == first[2] == third[2], mode
    print("ALL CHECKS PASSED (repeat)")


def shape_study(name, observe, values):
    """``u' = -a u`` on one cell from ``u(0) = 1``, one measurement ``observe`` of ``values``.

    The model declares ``"u"`` (a scalar, the mean of ``u``) and ``"u field"`` (the
    field, of shape ``(1,)`` on the one cell). Returns the study.
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    a = spec.param("a", value=0.5, learnable=True, bounds=(0.05, 3.0))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.observe.scalar(name="u", fn=lambda q: jnp.mean(q.u))
    spec.observe.field(name="u field", fn=lambda q: q.u)

    @spec.equation(writes="u", reads=("u",))
    def decay(q):
        return -q.p.a * q.u

    experiment = inf.Experiment(
        model=spec.build(),
        solver=_solver(),
        t_span=(0.0, 2.0),
        initial_state=spec.assemble_state({"u": 1.0}),
        measurements=(_gaussian("u", SCHEDULE_TIMES, values, observe),),
        id="cell",
    )
    return inf.Study((experiment,), parameters=inf.Hierarchy(shared=(a,)))


def section_shapes(report) -> None:
    """A measurement's values have its observation's shape at each time, or it is refused.

    An observation of shape ``(1,)`` (the state of the one cell, by a function and by
    the declared field) measured with values of shape ``(times,)`` is refused when the
    study is lowered; with values of shape ``(times, 1)``, and a scalar observation
    (by a function and by the declared scalar) with values of shape ``(times,)``,
    give one cost, the closed form's.
    """
    values = np.exp(-A_TRUE * SCHEDULE_TIMES)
    point = np.asarray([0.7])
    expected = 0.5 * np.sum(((np.exp(-0.7 * SCHEDULE_TIMES) - values) / SIGMA) ** 2)
    for label, observe in (("function", lambda st: st), ("declared", "u field")):
        try:
            ds.lower(shape_study(f"wave2_shape_refused_{label}", observe, values))
        except ValueError as error:
            report[f"shape (1,) by {label} with values (times,) refused"] = str(error)
            assert "(1,)" in str(error) and "()" in str(error)
        else:  # pragma: no cover
            raise AssertionError(f"an observation of shape (1,) by {label} was not refused")
    for label, observe, given in (
        ("(1,) by function, values (times, 1)", lambda st: st, values[:, None]),
        ("(1,) declared, values (times, 1)", "u field", values[:, None]),
        ("scalar by function, values (times,)", lambda st: st[0], values),
        ("scalar declared, values (times,)", "u", values),
    ):
        lowered = ds.lower(shape_study(f"wave2_shape_{len(report)}", observe, given))
        lowered.prepare(modes=("value", "gradient"))
        cost = float(lowered.value_and_gradient(point).values[0])
        report[f"shape {label}, cost vs closed form"] = relative(cost, expected)
        assert report[f"shape {label}, cost vs closed form"] < 1e-6


def check_two_recipe_names(report, lowered, label="two recipes"):
    """The fitted vector names each shared number once and each cell's own numbers once per cell."""
    groups = tuple(lowered.param_group_names)
    columns = tuple(tuple(int(c) for c in cols) for cols in lowered.param_group_cols)
    report[f"{label} fitted groups"] = groups
    assert groups == ("a", "s", "b@d1", "c@d1", "b@d2", "c@d2"), groups
    assert columns == ((0,), (1,), (2,), (3, 4), (5,), (6, 7)), columns
    assert lowered.n_params == 8


def _stop_at_second_step():
    class StopAtSecondStep(inf.Callback):
        def __init__(self):
            self.steps = 0

        def on_step(self, info):
            self.steps += 1
            if self.steps == 2:
                raise inf.StopFit()

    return StopAtSecondStep()


def two_recipe_reports(lowered, point):
    """What ``predict``, the residual reports and the costs per measurement say of ``two_recipe_study`` at ``point``.

    ``predict`` of the cell ``d1`` from two draws at ``point`` and of the source's current
    against the closed form; the derivative of the cell's prediction against central
    differences of the closed form in every column, the source's ``a`` and ``s``
    included; the residual report of every experiment (``d1``'s cost against the closed
    form) and the costs per measurement, each summed against the study's cost. Returns
    the numbers by name.
    """
    import jax

    from diff_soup.optimization.inference.statistics._analysis import (
        measurement_costs,
        study_predictor,
    )

    def z_of(x):
        return recipe_b_truth(x[0], x[1], x[2], x[3:5])

    out = {}
    draws = inf.Draws(
        values=jnp.asarray(np.stack([point, point]))[None],
        names=tuple(f"x{k}" for k in range(point.size)),
        info={},
        _parameter_schema=lowered.parameter_schema,
    )
    predicted = np.asarray(inf.predict(lowered, draws, experiment_id="d1")["z"].mean)
    out["predict d1 vs closed form"] = relative(predicted, z_of(point))
    source = np.asarray(inf.predict(lowered, draws, experiment_id="source")["I"].mean)
    out["predict source current vs closed form"] = relative(
        source, truth_at(point[0], point[1], GLOBAL_TIMES)[1]
    )
    _, _, _, predict = study_predictor(lowered, "d1")
    jacobian = np.asarray(jax.jacfwd(lambda x: predict(x)[0]["z"])(jnp.asarray(point)))
    step, columns = 1e-6, []
    for k in range(point.size):
        shift = np.zeros(point.size)
        shift[k] = step
        columns.append((z_of(point + shift) - z_of(point - shift)) / (2 * step))
    out["predict d1 derivative vs closed form"] = relative(jacobian, np.stack(columns, axis=1))
    cost = float(lowered.value_and_gradient(point).values[0])
    reports = {
        experiment_id: inf.residual_report(lowered, point, experiment_id=experiment_id)
        for experiment_id in ("source", "d1", "d2")
    }
    summed = sum(c.cost for r in reports.values() for c in r.channels.values()) / 2
    out["residual reports, sum of costs vs study cost"] = relative(summed, cost)
    want = recipe_b_truth(A_TRUE, S_TRUE["c1"], TWO_RECIPE_B["d1"], TWO_RECIPE_C["d1"])
    out["residual report d1 vs closed form"] = relative(
        reports["d1"].channels["z"].cost, np.sum(((z_of(point) - want) / SIGMA) ** 2)
    )
    costs, names = measurement_costs(lowered, point)
    out["cost per measurement, sum vs study cost"] = relative(sum(costs), cost)
    assert names == (("source", "u"), ("source", "I"), ("d1", "z"), ("d2", "z")), names
    return out


def check_two_recipe_reports(report, lowered, start, label="two recipes"):
    """Record ``two_recipe_reports`` at ``TWO_RECIPE_POINT`` and a fit's diagnostics, and assert them.

    The fit's costs per measurement sum to its cost, and no parameter is flagged.
    """
    for key, value in two_recipe_reports(lowered, TWO_RECIPE_POINT).items():
        report[f"{label} {key}"] = value
        assert value < 1e-6, (key, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = inf.optsolve(lowered.with_start(start), inf.TrustRegion(max_steps=60))
    diagnostics = result.diagnostics
    report[f"{label} fit diagnostics, cost per measurement minus cost"] = abs(
        sum(diagnostics.per_series_cost) - float(result.cost)
    )
    report[f"{label} fit diagnostics, warnings"] = diagnostics.warnings
    assert report[f"{label} fit diagnostics, cost per measurement minus cost"] < 1e-10
    assert not diagnostics.warnings, diagnostics.warnings
    assert diagnostics.series_names == (("source", "u"), ("source", "I"), ("d1", "z"), ("d2", "z"))


def section_two_recipes(report) -> None:
    """A hierarchy over experiments of two recipes, the source outside the level ``cell``.

    ``two_recipe_study``: ``J^T J`` and ``J^T r`` against the closed form and the gradient
    against five-point differences; the gradient mode against the blocks; joint and nested
    (``nested_at="cell"``) fits from a wrong start; for each, a fit stopped after two steps
    resumed from its checkpoint to the uninterrupted fit's end; the fitted vector's groups.
    The driven lanes (``a``, ``b``, ``c``: four coordinates) are wider than their source
    (``a``, ``s``). Then the refusals: the source outside the level under ``reads="all"``,
    and an experiment outside the level whose model declares a parameter of the level.
    """
    import shutil
    import tempfile

    study, truth = two_recipe_study("wave2_two_recipes")
    assert study.clusters["cell"] == (None, "d1", "d2")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient", "frozen"))
    check_two_recipe_names(report, lowered)
    point = TWO_RECIPE_POINT
    cost, gradient, matrix, failed = lowered.gauss_newton_fn(point)
    jtj, jtr = closed_form_blocks(two_recipe_rows, point)
    report["two recipes cost vs closed form"] = relative(
        cost, 0.5 * np.sum(two_recipe_rows(point) ** 2)
    )
    report["two recipes J^T J vs closed form"] = relative(matrix, jtj)
    report["two recipes J^T r vs closed form"] = relative(gradient, jtr)
    report["two recipes J^T r vs differences"] = relative(gradient, differences(lowered, point))
    report["two recipes gradient mode vs blocks"] = relative(
        lowered.value_and_gradient(point).gradients[0], gradient
    )
    assert failed == 0
    for key in (
        "cost vs closed form",
        "J^T J vs closed form",
        "J^T r vs closed form",
        "J^T r vs differences",
        "gradient mode vs blocks",
    ):
        assert report[f"two recipes {key}"] < 1e-6, (key, report[f"two recipes {key}"])
    for nested_at in (None, "cell"):
        kind = "nested" if nested_at else "joint"
        straight = fit(lowered, TWO_RECIPE_START, nested_at=nested_at)
        report[f"two recipes {kind} fit error"] = relative(straight.params, truth)
        assert report[f"two recipes {kind} fit error"] < 1e-6, straight.params
        folder = tempfile.mkdtemp(prefix="wave2_two_recipes_")
        try:
            stopped = fit(
                lowered,
                TWO_RECIPE_START,
                nested_at=nested_at,
                callbacks=(_stop_at_second_step(),),
                checkpoint=folder,
            )
            assert stopped.status is inf.TerminationStatus.STOPPED
            resumed = fit(lowered, TWO_RECIPE_START, nested_at=nested_at, checkpoint=folder)
            report[f"two recipes {kind} checkpoint resumed == straight"] = bool(
                resumed.n_steps == straight.n_steps
                and np.array_equal(np.asarray(resumed.params), np.asarray(straight.params))
            )
            assert report[f"two recipes {kind} checkpoint resumed == straight"]
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    check_two_recipe_reports(report, lowered, TWO_RECIPE_START)

    # The refusals: outside the level under reads="all", and outside while declaring b.
    experiments = study.experiments
    try:
        inf.Study(experiments, parameters=study.parameters)
    except ValueError as error:
        # Under reads="all" every model declares every placed parameter; the source does not.
        report["two recipes outside under reads='all' refused"] = str(error)[:100]
        assert "absent from experiment 'source'" in str(error)
    else:  # pragma: no cover
        raise AssertionError("an experiment outside the levels under reads='all' was accepted")
    declares_b = (
        experiments[0],
        experiments[1],
        dataclasses.replace(experiments[2], parameters=None),
    )
    try:
        inf.Study(declares_b, parameters=study.parameters, reads="declared")
    except ValueError as error:
        report["two recipes outside declaring b refused"] = str(error)[:100]
        assert "'b'" in str(error) and "outside" in str(error)
    else:  # pragma: no cover
        raise AssertionError("an experiment outside the levels that declares b was accepted")


def offset_costs(lowered, offsets):
    """Make ``lowered`` add ``offsets[mode]`` to every lane's cost of a request in ``mode``.

    A hook of this check, not of the library: the instance's ``evaluate`` is wrapped, and
    ``offsets`` (a dict) is read at every request, so the caller changes it between fits.
    """
    plain = lowered.evaluate

    def evaluate(request):
        results = plain(request)
        offset = offsets.get(request.mode, 0.0)
        if offset:
            results = dataclasses.replace(results, cost=np.asarray(results.cost) + offset)
        return results

    object.__setattr__(lowered, "evaluate", evaluate)


def section_cost_modes(report) -> None:
    """The gain ratio compares the costs of two requests of one mode.

    On ``two_recipe_study``, joint and nested fits with every lane's cost of the ``blocks``
    requests raised and lowered by ``1e-4`` take the steps of the fits without an offset:
    the same accepted and rejected attempts at the same points and the same costs in the step
    log; the fit makes one request for the costs beyond those of the trials
    (``work["cost_passes"]``, at the start). With the cost of the trials' requests (``value``, and ``frozen`` in
    the nested fit) raised by ``1e-4`` per lane, every logged cost moves by three times
    that, and the attempts are those of the fit without an offset until the stop tests,
    which read the cost's size, end it.
    """
    study, _ = two_recipe_study("wave2_cost_modes")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient", "frozen"))
    offsets: dict = {}
    offset_costs(lowered, offsets)

    class Log(inf.Callback):
        def __init__(self):
            self.rows = []

        def on_step(self, info):
            self.rows.append((bool(info.accepted), float(info.cost), np.array(info.params)))

    def run(nested_at, given):
        offsets.clear()
        offsets.update(given)
        log = Log()
        result = fit(lowered, TWO_RECIPE_START, nested_at=nested_at, callbacks=(log,))
        return result, log.rows

    for nested_at in (None, "cell"):
        kind = "nested" if nested_at else "joint"
        plain, rows = run(nested_at, {})
        report[f"cost modes {kind}, accepted"] = [row[0] for row in rows]
        report[f"cost modes {kind}, cost requests at the start"] = int(plain.work["cost_passes"])
        assert plain.work["cost_passes"] == 1
        for offset in (1e-4, -1e-4):
            _, shifted = run(nested_at, {"blocks": offset})
            same = len(shifted) == len(rows) and all(
                a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2])
                for a, b in zip(rows, shifted)
            )
            report[f"cost modes {kind}, blocks costs {offset:+g}: the same steps and costs"] = same
            assert same
        _, raised = run(nested_at, {"value": 1e-4, "frozen": 1e-4})
        common = min(len(rows), len(raised))
        prefix = all(
            a[0] == b[0] and np.array_equal(a[2], b[2]) for a, b in zip(rows[:common], raised)
        )
        shifts = np.asarray([b[1] - a[1] for a, b in zip(rows[:common], raised)])
        report[f"cost modes {kind}, trial costs +1e-4: the same steps for"] = (
            f"{common} of {len(rows)}"
        )
        report[f"cost modes {kind}, trial costs +1e-4: logged cost shift minus 3e-4"] = float(
            np.max(np.abs(shifts - 3e-4))
        )
        assert prefix and common >= len(rows) - 3
        assert report[f"cost modes {kind}, trial costs +1e-4: logged cost shift minus 3e-4"] < 1e-12


def one_process() -> None:
    """Every section that runs in one process."""
    report = _Report()
    section_two_recipes(report)
    section_cost_modes(report)
    section_globals(report)
    section_shapes(report)
    print("ALL CHECKS PASSED (one process)")


def world_checkpoints(lowered, start) -> dict:
    """Joint and nested fits stopped after two steps and resumed from their checkpoints, in a run of several processes.

    Process 0 runs the fits (the others serve its requests), so the folder is its own.
    """
    import shutil
    import tempfile

    out = {}
    for nested_at in (None, "cell"):
        kind = "nested" if nested_at else "joint"
        folder = tempfile.mkdtemp(prefix="wave2_world_")
        try:
            straight = fit(lowered, start, nested_at=nested_at)
            stopped = fit(
                lowered,
                start,
                nested_at=nested_at,
                callbacks=(_stop_at_second_step(),),
                checkpoint=folder,
            )
            assert stopped.status is inf.TerminationStatus.STOPPED
            resumed = fit(lowered, start, nested_at=nested_at, checkpoint=folder)
            out[f"{kind} checkpoint resumed == straight"] = bool(
                resumed.n_steps == straight.n_steps
                and np.array_equal(np.asarray(resumed.params), np.asarray(straight.params))
            )
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    return out


def world(kind: str) -> None:
    """``schedule`` (the globals study, one cell per process), ``link`` or ``recipes`` over the MPI world."""
    from mpi4py import MPI

    if kind == "schedule":
        study, truth = global_study("wave2_world_globals")
        point, start, rows_of, nested = SCHEDULE_POINT, SCHEDULE_START, global_rows, ()
    elif kind == "recipes":
        study, truth = two_recipe_study("wave2_world_two_recipes")
        point, start, rows_of, nested = (
            TWO_RECIPE_POINT,
            TWO_RECIPE_START,
            two_recipe_rows,
            ("cell",),
        )
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
        if kind == "recipes":
            check_two_recipe_names(_Report(), lowered)
            check_two_recipe_reports(_Report(), lowered, start, label="two recipes world,")
            out.update(world_checkpoints(lowered, start))
        if kind == "link":
            linked = link_reports(lowered, point)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diagnosed = inf.optsolve(lowered.with_start(start), inf.TrustRegion(max_steps=60))
    if MPI.COMM_WORLD.rank == 0 and kind == "link":
        check_link_reports(_Report(), linked, label="link world,")
        check_link_diagnostics(_Report(), diagnosed, label="link world,")
    if MPI.COMM_WORLD.rank == 0:
        for key, value in out.items():
            print(f"{kind} world, {key}: {value}")
        # The placement may batch the two driven lanes, which share a program, on one process.
        assert out["processes"] == 3 and out["failed"] == 0
        assert out["ranks"] == [0, 1, 2] if kind == "schedule" else len(set(out["ranks"])) >= 2
        for key, value in out.items():
            if key.endswith(("closed form", "differences", "fit error")):
                assert value < 1e-6, (key, value)
            if key.endswith("resumed == straight"):
                assert value, key
        print(f"ALL CHECKS PASSED ({kind}, world of three)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "one"
    if mode == "world":
        world(sys.argv[2])
    else:
        {
            "one": one_process,
            "measure": measure,
            "saves": measure_saves,
            "budget": measure_budget,
            "repeat": repeat,
        }[mode]()
