"""Check of starting states computed from the parameters (``inf.ComputedInitialState``).

Not part of the benchmark suite: a hand-run check, printed and asserted.

    JAX_ENABLE_X64=1 python -m benchmarks.computed_start_check            # one process
    JAX_ENABLE_X64=1 mpirun -n 3 python -m mpi4py -m benchmarks.computed_start_check world cells
    JAX_ENABLE_X64=1 mpirun -n 3 python -m mpi4py -m benchmarks.computed_start_check world link
    JAX_ENABLE_X64=1 mpirun -n 3 python -m mpi4py -m benchmarks.computed_start_check world schedule
    JAX_ENABLE_X64=1 python -m benchmarks.computed_start_check baseline   # fixed starts only

One process:

- ``u' = -a u`` from ``u(0) = s``, two cells that share ``a`` and each fit their own ``s``.
  The solution is ``s e^{-a t}``, so the rows' Jacobian is known in closed form: the
  Gauss-Newton matrix ``J^T J`` and gradient ``J^T r`` of the lanes are compared with it,
  and the gradient with the five-point central difference of the cost. The joint and the
  nested trust region recover ``a`` and both ``s`` from a wrong start, and a fit stopped
  after two steps resumes from its checkpoint to the uninterrupted fit's end.
- The same decay with an algebraic unknown ``w = 2 u``, started from the guess ``w = 0``:
  the solve makes ``w`` consistent with the computed ``u(0) = s``, and the rows, which
  observe ``w``, have the derivative of ``2 s e^{-a t}``.
- The gradient mode in reverse mode (a solver with ``ds.CheckpointAdjoint()``) gives the
  forward blocks' gradient.
- ``reads="declared"``: a computed-start experiment beside one whose model does not
  declare ``s``; the gradient against differences and a fit.
- A link: a source with a computed start drives ``z' = -b z + y(t)``; the driven lane
  carries ``s`` through the link (gradient against differences, and a fit).
- A computed start that returns a constant gives the cost, gradient and Gauss-Newton
  matrix of the same study with that constant as a fixed start.

- A schedule (a held current, a held voltage, a rest) on ``u' = -a u + I`` with the current
  ``I`` an algebraic unknown, three cells from computed starts ``u(0) = s`` with the current
  guessed at zero: ``J^T J`` and ``J^T r`` against the closed-form rows (their Jacobian by
  differences of the closed form), the gradient against differences of the cost, a joint
  fit, and a constant computed start against the same fixed start.

``world``: the two-cell decay with a third cell, and the linked study, over the processes
of an MPI world (one lane per process): gradients against differences, joint and nested
fits. ``baseline``: the two-cell study with fixed starts, printed as JSON (fingerprint,
bucket identities, cost, gradient, matrix and fit, floats in hexadecimal) to compare two
versions of the library byte for byte.
"""

# ruff: noqa: I001

from __future__ import annotations

import json
import shutil
import sys
import tempfile

import diff_soup as ds  # diff_soup before jax
import jax.numpy as jnp
import numpy as np

from diff_soup.domains import cartesian_grid_1d
from diff_soup.optimization import inference as inf
from diff_soup.optimization.inference.approx import PiecewiseLinear

TIMES = np.linspace(0.25, 2.0, 8)
LINK_TIMES = np.linspace(0.0, 2.0, 21)
SIGMA = 0.01
A_TRUE = 0.8
S_TRUE = {"c1": 2.0, "c2": 1.2, "c3": 1.6}
STEP = 4e-3


def _solver():
    return ds.JNDF.stock(linear="dense", rtol=1e-10, atol=1e-12)


def decay_recipe(name, *, algebraic=False, declares_s=True, inert_b=False):
    """``u' = -a u``; with ``algebraic``, also ``0 = w - 2 u``. Returns ``(spec, model, a, s)``.

    ``s`` is the start of ``u``, declared and read by no equation; ``inert_b`` declares an
    unused rate ``b`` as well.
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    a = spec.param("a", value=0.5, learnable=True, bounds=(0.05, 3.0))
    s = spec.param("s", value=1.0, learnable=True, bounds=(0.1, 5.0)) if declares_s else None
    if inert_b:
        spec.param("b", value=0.5, learnable=True, bounds=(0.05, 3.0))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

    @spec.equation(writes="u", reads=("u",))
    def decay(q):
        return -q.p.a * q.u

    if algebraic:
        spec.field("w", mass="algebraic", scale=1.0, init=lambda x: jnp.zeros_like(x))

        @spec.equation(writes="w", reads=("u", "w"))
        def double(q):
            return q.w - 2.0 * q.u

    return spec, spec.build(), a, s


def _measurement(name, values, observe):
    return (
        inf.Measurement(
            name,
            times=jnp.asarray(TIMES),
            values=jnp.asarray(values),
            observe=observe,
            likelihood=inf.Gaussian(sigma=SIGMA),
        ),
    )


def computed(spec, s, **guesses):
    """The start ``u(0) = s``, the other fields at ``guesses``.

    The function holds arrays and the handle ``s`` only, which the study fingerprint
    encodes (a function holding the recipe ``spec`` itself would be refused there).
    """
    rest = spec.assemble_state({"u": 0.0, **guesses})
    unit = spec.assemble_state({"u": 1.0, **guesses}) - rest
    return inf.ComputedInitialState(lambda theta: rest + unit * theta[s])


def decay_experiment(spec, model, s, cell, *, start=None, algebraic=False, solver=None, **extra):
    """The decay of ``cell`` from its true ``s``, observed exactly; a computed start by default."""
    observe = (lambda state: state[1]) if algebraic else (lambda state: jnp.mean(state))
    scale = 2.0 if algebraic else 1.0
    values = scale * S_TRUE[cell] * np.exp(-A_TRUE * TIMES)
    if start is None:
        start = computed(spec, s, w=0.0) if algebraic else computed(spec, s)
    return inf.Experiment(
        model=model,
        solver=_solver() if solver is None else solver,
        t_span=(0.0, 2.0),
        initial_state=start,
        measurements=_measurement("x", values, observe),
        id=cell,
        parameters={"cell": cell},
        **extra,
    )


def cells_study(name, cells=("c1", "c2"), *, fixed=False, algebraic=False, solver=None):
    """Cells sharing ``a``, each with its own ``s`` (fitted, or fixed at its true value)."""
    spec, model, a, s = decay_recipe(name, algebraic=algebraic)
    experiments = []
    for cell in cells:
        start = None
        if fixed:
            start = spec.assemble_state({"u": S_TRUE[cell]})
        experiments.append(
            decay_experiment(spec, model, s, cell, start=start, algebraic=algebraic, solver=solver)
        )
    shared, per = ((a,), {"cell": (s,)})
    hierarchy = inf.Hierarchy(levels=("cell",), shared=shared, per=per)
    return inf.Study(tuple(experiments), parameters=hierarchy), (spec, a, s)


def analytic_blocks(point, cells, scale=1.0):
    """``J^T J`` and ``J^T r`` of the cells study at ``[a, s_c...]`` from ``s e^{-a t}``."""
    a = point[0]
    n = point.size
    jtj = np.zeros((n, n))
    jtr = np.zeros(n)
    for k, cell in enumerate(cells):
        s = point[1 + k]
        e = np.exp(-a * TIMES)
        r = scale * (s * e - S_TRUE[cell] * np.exp(-A_TRUE * TIMES)) / SIGMA
        J = np.zeros((TIMES.size, n))
        J[:, 0] = scale * (-TIMES * s * e) / SIGMA
        J[:, 1 + k] = scale * e / SIGMA
        jtj += J.T @ J
        jtr += J.T @ r
    return jtj, jtr


def differences(lowered, point, step=STEP):
    """The five-point central difference of the value cost in every column."""
    eye = np.eye(point.size)
    shifts = np.concatenate(
        [point + step * eye, point - step * eye, point + 2 * step * eye, point - 2 * step * eye]
    )
    values = np.asarray(lowered.value_and_gradient(shifts).values)
    n = point.size
    plus, minus, plus2, minus2 = (values[k * n : (k + 1) * n] for k in range(4))
    return (8.0 * (plus - minus) - (plus2 - minus2)) / (12.0 * step)


def relative(got, expected):
    """The largest absolute difference over the largest entry of ``expected``."""
    got, expected = np.asarray(got, float), np.asarray(expected, float)
    return float(np.max(np.abs(got - expected)) / max(np.max(np.abs(expected)), 1e-300))


def fit(lowered, start, nested_at=None, **options):
    """A joint (``nested_at=None``) or nested trust-region fit from ``start``."""
    return inf.optsolve(
        lowered.with_start(np.asarray(start)),
        inf.TrustRegion(max_steps=60, nested_at=nested_at),
        diagnostics=False,
        **options,
    )


def _driven_truth(a, s, b):
    """``z`` at ``TIMES`` on the piecewise-linear signal of the source mean ``s e^{-a t}``.

    On each piece the signal is ``alpha + beta t`` and ``z = c0 + c1 t + (z0 - c0) e^{-b t}``
    with ``c1 = beta / b`` and ``c0 = (alpha - c1) / b``.
    """
    knots, values = LINK_TIMES, s * np.exp(-a * LINK_TIMES)
    out = []
    for t in TIMES:
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


LINK_B = {"d1": 1.3, "d2": 0.9}


def linked_study(name):
    """A source ``u' = -a u`` from ``u(0) = s`` whose mean drives ``z' = -b z + y(t)`` in two cells.

    ``a`` and ``s`` are shared, ``b`` is one per cell; the source sits in a cell of its own,
    where its model declares ``b`` and does not use it. The driven models do not declare
    ``a`` or ``s`` (``reads="declared"``): their lanes read them through the link.
    Returns the study and its true point ``[a, s, b_source, b_d1, b_d2]``.
    """
    sspec, source_model, a, s = decay_recipe(f"{name}_source", inert_b=True)
    dspec = ds.problem(f"{name}_driven")
    dspec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    dspec.param("b", value=0.5, learnable=True, bounds=(0.05, 3.0))
    knots = jnp.asarray(LINK_TIMES)
    dspec.function("y", PiecewiseLinear(knots, jnp.zeros(knots.shape)), learnable=False)
    dspec.field("z", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))

    @dspec.equation(writes="z", reads=("z",))
    def relax(q):
        return -q.p.b * q.z + q.fn.y(q.t)

    driven_model = dspec.build()
    s_true = S_TRUE["c1"]
    mean = lambda state: jnp.mean(state)  # noqa: E731
    experiments = [
        inf.Experiment(
            model=source_model,
            solver=_solver(),
            t_span=(0.0, 2.0),
            initial_state=computed(sspec, s),
            measurements=_measurement("x", s_true * np.exp(-A_TRUE * TIMES), mean),
            id="source",
            outputs=(inf.Output("y", times=LINK_TIMES, observe=mean),),
            parameters={"cell": "source"},
        )
    ]
    for cell, b_true in LINK_B.items():
        experiments.append(
            inf.Experiment(
                model=driven_model,
                solver=_solver(),
                t_span=(0.0, 2.0),
                initial_state=dspec.assemble_state({}),
                measurements=_measurement("z", _driven_truth(A_TRUE, s_true, b_true), mean),
                id=cell,
                links=(inf.Link(source="source", output="y", into="y"),),
                parameters={"cell": cell},
            )
        )
    # A hierarchy takes its handles from one recipe: the source's b names the driven b.
    source_b = experiments[0]._declarations["b"]
    hierarchy = inf.Hierarchy(levels=("cell",), shared=(a, s), per={"cell": (source_b,)})
    study = inf.Study(tuple(experiments), parameters=hierarchy, reads="declared")
    truth = np.asarray([A_TRUE, s_true, 0.5, LINK_B["d1"], LINK_B["d2"]])
    return study, truth


# ---------------------------------------------------------------------------


# A schedule: a held current, a held voltage, then a rest at zero current.
SCHEDULE_TIMES = np.asarray([0.2, 0.4, 0.55, 0.7, 0.9, 1.15, 1.3, 1.6, 1.9])
SEGMENTS = (("current", 0.5, 0.6), ("voltage", 1.0, 0.6), ("current", 0.0, 0.8))
SCHEDULE_CELLS = ("c1", "c2", "c3")


def cell_recipe(name):
    """A one-node cell under an operational table, with a global and a field algebraic.

    ``u' = -a u + I``, the current ``I`` a global algebraic unknown that the
    operational table holds (current held, or voltage ``u - 2 I`` held), and an algebraic field
    ``w = u + I``. Returns ``(model, a, s)``, ``s`` the start of ``u``, declared and read by
    no equation. The measurements read the saved state ``(u, w)``: a global is not in it.
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    a = spec.param("a", value=0.5, learnable=True, bounds=(0.05, 3.0))
    s = spec.param("s", value=1.0, learnable=True, bounds=(0.1, 5.0))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.global_var("I_cell", initial=0.0, mass="algebraic")
    spec.observe.scalar(name="current", fn=lambda q: q["I_cell"])
    spec.observe.scalar(name="voltage", fn=lambda q: jnp.mean(q.u) - 2.0 * q["I_cell"])

    spec.field("w", mass="algebraic", scale=1.0, init=lambda x: jnp.zeros_like(x))

    @spec.equation(writes="u", reads=("u", "I_cell"))
    def balance(q):
        return -q.p.a * q.u + q["I_cell"]

    @spec.equation(writes="w", reads=("u", "w", "I_cell"))
    def sum_(q):
        return q.w - q.u - q["I_cell"]

    spec.control.operational(
        name="terminal",
        current="current",
        voltage="voltage",
        current_unknown="I_cell",
        modes=("current", "voltage"),
    )
    return spec.build(), a, s


def schedule_truth(a, s):
    """``(u, w)`` at ``SCHEDULE_TIMES`` in closed form, segment by segment."""
    (_, i1, d1), (_, v2, d2), (_, i3, _) = SEGMENTS
    k = 0.5 - a  # u' = -a u + (u - V) / 2 while the voltage is held
    u1 = i1 / a + (s - i1 / a) * np.exp(-a * d1)
    u2 = v2 / (2 * k) + (u1 - v2 / (2 * k)) * np.exp(k * d2)
    us, ws = [], []
    for t in SCHEDULE_TIMES:
        if t < d1:
            u = i1 / a + (s - i1 / a) * np.exp(-a * t)
            i = i1
        elif t < d1 + d2:
            u = v2 / (2 * k) + (u1 - v2 / (2 * k)) * np.exp(k * (t - d1))
            i = (u - v2) / 2
        else:
            u = i3 / a + (u2 - i3 / a) * np.exp(-a * (t - d1 - d2))
            i = i3
        us.append(u)
        ws.append(u + i)
    return np.asarray(us), np.asarray(ws)


def schedule_rows(point, truth):
    """The whitened rows of the schedule study at ``[a, s_c...]`` in closed form."""
    rows = []
    for k, cell in enumerate(SCHEDULE_CELLS):
        got = np.concatenate(schedule_truth(point[0], point[1 + k]))
        want = np.concatenate(schedule_truth(truth[0], truth[1 + k]))
        rows.append((got - want) / SIGMA)
    return np.concatenate(rows)


def schedule_study(name, starts=None):
    """Three cells following ``SEGMENTS``, sharing ``a``, each with its own start ``s``.

    ``starts`` maps a cell to its start; by default a ``ComputedInitialState`` reading ``s``,
    with ``w`` and the current guessed at zero (the solve makes them consistent).
    Returns the study and its true point ``[a, s_c1, s_c2, s_c3]``.
    """
    model, a, s = cell_recipe(name)
    schedule = ds.Schedule([ds.Segment(duration=d, **{kind: v}) for kind, v, d in SEGMENTS])
    rest, unit = jnp.asarray([0.0, 0.0, 0.0]), jnp.asarray([1.0, 0.0, 0.0])
    truth = np.asarray([A_TRUE] + [S_TRUE[cell] for cell in SCHEDULE_CELLS])
    experiments = []
    for k, cell in enumerate(SCHEDULE_CELLS):
        start = (starts or {}).get(cell)
        if start is None:
            start = inf.ComputedInitialState(lambda theta: rest + unit * theta[s])
        u, w = schedule_truth(A_TRUE, S_TRUE[cell])
        experiments.append(
            inf.Experiment(
                model=model,
                solver=_solver(),
                t_span=(0.0, 2.0),
                initial_state=start,
                measurements=tuple(
                    inf.Measurement(
                        name,
                        times=jnp.asarray(SCHEDULE_TIMES),
                        values=jnp.asarray(values),
                        observe=observe,
                        likelihood=inf.Gaussian(sigma=SIGMA),
                    )
                    for name, values, observe in (
                        ("u", u, lambda state: state[0]),
                        ("w", w, lambda state: state[1]),
                    )
                ),
                id=cell,
                schedule=schedule,
                parameters={"cell": cell},
            )
        )
    hierarchy = inf.Hierarchy(levels=("cell",), shared=(a,), per={"cell": (s,)})
    return inf.Study(tuple(experiments), parameters=hierarchy), truth


def closed_form_blocks(point, truth, step=1e-6):
    """``J^T J`` and ``J^T r`` from the closed-form rows, ``J`` by central differences of them."""
    rows = schedule_rows(point, truth)
    columns = []
    for k in range(point.size):
        shift = np.zeros(point.size)
        shift[k] = step
        columns.append(
            (schedule_rows(point + shift, truth) - schedule_rows(point - shift, truth)) / (2 * step)
        )
    J = np.stack(columns, axis=1)
    return J.T @ J, J.T @ rows


SCHEDULE_POINT = np.asarray([0.7, 1.7, 1.4, 1.5])
SCHEDULE_START = np.asarray([0.6, 1.0, 1.0, 1.0])


class _Report(dict):
    """A dict that prints every entry as it is set."""

    def __setitem__(self, key, value):
        print(f"{key}: {value}", flush=True)
        super().__setitem__(key, value)


def one_process() -> None:
    """Every check that runs in one process."""
    report = _Report()

    # 1. Two cells: closed-form blocks, differences, joint and nested fits, checkpoint.
    study, _ = cells_study("check_cells")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient"))
    truth = np.asarray([A_TRUE, S_TRUE["c1"], S_TRUE["c2"]])
    point = np.asarray([0.7, 1.7, 1.4])
    cost, gradient, matrix, failed = lowered.gauss_newton_fn(point)
    jtj, jtr = analytic_blocks(point, ("c1", "c2"))
    report["cells J^T J vs closed form"] = relative(matrix, jtj)
    report["cells J^T r vs closed form"] = relative(gradient, jtr)
    report["cells J^T r vs differences"] = relative(gradient, differences(lowered, point))
    report["cells gradient mode vs blocks"] = relative(
        lowered.value_and_gradient(point).gradients[0], gradient
    )
    assert failed == 0
    assert report["cells J^T J vs closed form"] < 1e-6
    assert report["cells J^T r vs closed form"] < 1e-6
    assert report["cells J^T r vs differences"] < 1e-6
    start = np.asarray([0.4, 1.0, 1.0])
    for nested_at in (None, "cell"):
        result = fit(lowered, start, nested_at=nested_at)
        key = f"cells {'nested' if nested_at else 'joint'} fit error"
        report[key] = relative(result.params, truth)
        assert report[key] < 1e-6, result.params

    class StopAtSecondStep(inf.Callback):
        def __init__(self):
            self.steps = 0

        def on_step(self, info):
            self.steps += 1
            if self.steps == 2:
                raise inf.StopFit()

    folder = tempfile.mkdtemp(prefix="computed_start_")
    try:
        straight = fit(lowered, start)
        stopped = fit(lowered, start, callbacks=(StopAtSecondStep(),), checkpoint=folder)
        assert stopped.status is inf.TerminationStatus.STOPPED
        resumed = fit(lowered, start, checkpoint=folder)
        report["checkpoint resumed == straight"] = bool(
            resumed.n_steps == straight.n_steps
            and np.array_equal(np.asarray(resumed.params), np.asarray(straight.params))
        )
        assert report["checkpoint resumed == straight"]
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    # The gradient mode in reverse mode, through a solver with a reverse adjoint.
    reverse = ds.JNDF.stock(linear="dense", rtol=1e-10, atol=1e-12, adjoint=ds.CheckpointAdjoint())
    study, _ = cells_study("check_cells_reverse", solver=reverse)
    reversed_ = ds.lower(study)
    reversed_.prepare(modes=("value", "gradient"))
    report["cells reverse-mode gradient vs blocks"] = relative(
        reversed_.value_and_gradient(point).gradients[0], gradient
    )
    assert report["cells reverse-mode gradient vs blocks"] < 1e-6

    # 2. An algebraic unknown made consistent with the computed start.
    study, _ = cells_study("check_algebraic", algebraic=True)
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient"))
    cost, gradient, matrix, failed = lowered.gauss_newton_fn(point)
    jtj, jtr = analytic_blocks(point, ("c1", "c2"), scale=2.0)
    report["algebraic J^T J vs closed form"] = relative(matrix, jtj)
    report["algebraic J^T r vs closed form"] = relative(gradient, jtr)
    report["algebraic J^T r vs differences"] = relative(gradient, differences(lowered, point))
    assert failed == 0
    assert report["algebraic J^T J vs closed form"] < 1e-6
    assert report["algebraic J^T r vs differences"] < 1e-6
    result = fit(lowered, start)
    report["algebraic joint fit error"] = relative(result.params, truth)
    assert report["algebraic joint fit error"] < 1e-6

    # 3. reads="declared": the start's parameter is read only by the experiment that declares it.
    spec, model, a, s = decay_recipe("check_reads")
    _plain_spec, plain_model, _a_plain, _ = decay_recipe("check_reads_plain", declares_s=False)
    plain = inf.Experiment(
        model=plain_model,
        solver=_solver(),
        t_span=(0.0, 2.0),
        initial_state=_plain_spec.assemble_state({"u": 1.0}),
        measurements=_measurement("x", np.exp(-A_TRUE * TIMES), lambda st: jnp.mean(st)),
        id="plain",
    )
    c1 = decay_experiment(spec, model, s, "c1")
    c1 = inf.Experiment(
        model=c1.model,
        solver=c1.solver,
        t_span=c1.t_span,
        initial_state=c1.initial_state,
        measurements=c1.measurements,
        id="c1",
    )
    study = inf.Study((c1, plain), parameters=inf.Hierarchy(shared=(a, s)), reads="declared")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient"))
    point2 = np.asarray([0.7, 1.7])
    _, gradient, _, failed = lowered.gauss_newton_fn(point2)
    widths = {bucket.lanes: bucket.width for bucket in lowered.plan().buckets}
    report["reads=declared widths"] = {str(k): v for k, v in widths.items()}
    report["reads=declared J^T r vs differences"] = relative(gradient, differences(lowered, point2))
    assert failed == 0
    assert report["reads=declared J^T r vs differences"] < 1e-6
    result = fit(lowered, np.asarray([0.4, 1.0]))
    report["reads=declared joint fit error"] = relative(
        result.params, np.asarray([A_TRUE, S_TRUE["c1"]])
    )
    assert report["reads=declared joint fit error"] < 1e-6

    # 4. A link from a source with a computed start.
    study, link_truth = linked_study("check_linked")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient", "frozen"))
    link_point = np.asarray([0.7, 1.7, 0.5, 1.1, 1.0])
    _, gradient, _, failed = lowered.gauss_newton_fn(link_point)
    report["link J^T r vs differences"] = relative(gradient, differences(lowered, link_point))
    assert failed == 0
    assert report["link J^T r vs differences"] < 1e-6
    for nested_at in (None, "cell"):
        result = fit(lowered, np.asarray([0.4, 1.0, 0.5, 0.6, 0.6]), nested_at=nested_at)
        key = f"link {'nested' if nested_at else 'joint'} fit error"
        report[key] = relative(result.params, link_truth)
        assert report[key] < 1e-6, result.params

    # 5. A computed start that returns a constant equals that constant as a fixed start.
    spec, model, a, s = decay_recipe("check_constant")
    constant = spec.assemble_state({"u": 1.7})
    pieces = []
    for start in (inf.ComputedInitialState(lambda theta: constant), constant):
        experiment = decay_experiment(spec, model, s, "c1", start=start)
        experiment = inf.Experiment(
            model=experiment.model,
            solver=experiment.solver,
            t_span=experiment.t_span,
            initial_state=start,
            measurements=experiment.measurements,
            id="c1",
        )
        lowered = ds.lower(inf.Study((experiment,), parameters=ds.ParameterSet(shared=(a, s))))
        lowered.prepare(modes=("blocks",))
        pieces.append(lowered.gauss_newton_fn(np.asarray([0.7, 1.3]))[:3])
    report["constant start == fixed start (max abs)"] = max(
        float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(*pieces)
    )
    assert report["constant start == fixed start (max abs)"] == 0.0

    # 6. Refusals.
    try:
        decay_experiment(
            spec,
            model,
            s,
            "c1",
            trajectory=inf.Multishoot(inf.Windows.uniform(2)),
        )
    except ValueError as error:
        report["multishoot refused"] = str(error)[:90]
    else:  # pragma: no cover
        raise AssertionError("a computed start under Multishoot was not refused")

    # 7. A schedule (held current, held voltage, rest) with an algebraic current, from a
    # computed start whose current guess the solve makes consistent.
    study, truth = schedule_study("check_schedule")
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient"))
    _, gradient, matrix, failed = lowered.gauss_newton_fn(SCHEDULE_POINT)
    jtj, jtr = closed_form_blocks(SCHEDULE_POINT, truth)
    report["schedule J^T J vs closed form"] = relative(matrix, jtj)
    report["schedule J^T r vs closed form"] = relative(gradient, jtr)
    report["schedule J^T r vs differences"] = relative(
        gradient, differences(lowered, SCHEDULE_POINT)
    )
    assert failed == 0
    assert report["schedule J^T J vs closed form"] < 1e-6
    assert report["schedule J^T r vs closed form"] < 1e-6
    assert report["schedule J^T r vs differences"] < 1e-6
    result = fit(lowered, SCHEDULE_START)
    report["schedule joint fit error"] = relative(result.params, truth)
    assert report["schedule joint fit error"] < 1e-6
    constant = jnp.asarray([1.7, 0.0, 0.0])
    pieces = []
    for start in (inf.ComputedInitialState(lambda theta: constant), constant):
        study, _ = schedule_study(
            "check_schedule_constant", starts={cell: start for cell in SCHEDULE_CELLS}
        )
        lowered = ds.lower(study)
        lowered.prepare(modes=("blocks",))
        pieces.append(lowered.gauss_newton_fn(SCHEDULE_POINT)[:3])
    report["schedule constant start == fixed start (max abs)"] = max(
        float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(*pieces)
    )
    assert report["schedule constant start == fixed start (max abs)"] == 0.0

    print("ALL CHECKS PASSED (one process)")


def world(kind: str) -> None:
    """One study over the processes of the MPI world: ``cells`` (three cells) or ``link``.

    A run of several processes lowers one study, so each kind is its own run.
    """
    from mpi4py import MPI

    out = {}
    if kind == "cells":
        study, _ = cells_study("check_world_cells", cells=("c1", "c2", "c3"))
        truth = np.asarray([A_TRUE, S_TRUE["c1"], S_TRUE["c2"], S_TRUE["c3"]])
        point = np.asarray([0.7, 1.7, 1.4, 1.5])
        start = np.asarray([0.4, 1.0, 1.0, 1.0])
    else:
        study, truth = linked_study("check_world_linked")
        point = np.asarray([0.7, 1.7, 0.5, 1.1, 1.0])
        start = np.asarray([0.4, 1.0, 0.5, 0.6, 0.6])
    with ds.distributed.run(study) as lowered:
        report = lowered.prepare(modes=("value", "blocks", "gradient", "frozen"))
        _, gradient, matrix, failed = lowered.gauss_newton_fn(point)
        out["processes"] = lowered.processes
        out["ranks"] = sorted(int(r) for r in report.placement.rank_of.values())
        if kind == "cells":
            jtj, _ = analytic_blocks(point, ("c1", "c2", "c3"))
            out["J^T J vs closed form"] = relative(matrix, jtj)
        out["J^T r vs differences"] = relative(gradient, differences(lowered, point))
        out["failed"] = int(failed)
        for nested_at in (None, "cell"):
            result = fit(lowered, start, nested_at=nested_at)
            out[f"{'nested' if nested_at else 'joint'} fit error"] = relative(result.params, truth)
    if MPI.COMM_WORLD.rank == 0:
        for key, value in out.items():
            print(f"{kind} world, {key}: {value}")
        assert out["processes"] == 3
        assert out["ranks"] == [0, 1, 2]
        assert out["failed"] == 0
        for key, value in out.items():
            if key.endswith(("closed form", "differences", "fit error")):
                assert value < 1e-6, (key, value)
        print(f"ALL CHECKS PASSED ({kind}, world of three)")


def world_schedule() -> None:
    """The schedule study over the processes of the MPI world, one cell per process."""
    from mpi4py import MPI

    study, truth = schedule_study("check_world_schedule")
    out = {}
    with ds.distributed.run(study) as lowered:
        report = lowered.prepare(modes=("value", "blocks", "gradient", "frozen"))
        _, gradient, matrix, failed = lowered.gauss_newton_fn(SCHEDULE_POINT)
        jtj, _ = closed_form_blocks(SCHEDULE_POINT, truth)
        out["processes"] = lowered.processes
        out["ranks"] = sorted(int(r) for r in report.placement.rank_of.values())
        out["J^T J vs closed form"] = relative(matrix, jtj)
        out["J^T r vs differences"] = relative(gradient, differences(lowered, SCHEDULE_POINT))
        out["failed"] = int(failed)
        out["joint fit error"] = relative(fit(lowered, SCHEDULE_START).params, truth)
    if MPI.COMM_WORLD.rank == 0:
        for key, value in out.items():
            print(f"schedule world, {key}: {value}")
        assert out["processes"] == 3 and out["ranks"] == [0, 1, 2] and out["failed"] == 0
        for key in ("J^T J vs closed form", "J^T r vs differences", "joint fit error"):
            assert out[key] < 1e-6, (key, out[key])
        print("ALL CHECKS PASSED (schedule, world of three)")


def baseline() -> None:
    """Two cells with fixed starts sharing ``a``, as JSON for a byte comparison across versions."""
    spec, model, a, _ = decay_recipe("check_baseline", declares_s=False)
    experiments = tuple(
        inf.Experiment(
            model=model,
            solver=_solver(),
            t_span=(0.0, 2.0),
            initial_state=spec.assemble_state({"u": S_TRUE[cell]}),
            measurements=_measurement(
                "x", S_TRUE[cell] * np.exp(-A_TRUE * TIMES), lambda st: jnp.mean(st)
            ),
            id=cell,
        )
        for cell in ("c1", "c2")
    )
    study = inf.Study(experiments, parameters=inf.Hierarchy(shared=(a,)))
    lowered = ds.lower(study)
    lowered.prepare(modes=("value", "blocks", "gradient"))
    cost, gradient, matrix, failed = lowered.gauss_newton_fn(np.asarray([0.6]))
    result = fit(lowered, np.asarray([0.4]))
    hexed = lambda x: [float(v).hex() for v in np.asarray(x).reshape(-1)]  # noqa: E731
    print(
        json.dumps(
            {
                "fingerprint": study.fingerprint,
                "identities": [b.full_identity.digest for b in lowered.plan().buckets],
                "cost": hexed(cost),
                "gradient": hexed(gradient),
                "matrix": hexed(matrix),
                "failed": int(failed),
                "fit": hexed(result.params),
                "steps": int(result.n_steps),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "one"
    if mode == "world" and sys.argv[2] == "schedule":
        world_schedule()
    elif mode == "world":
        world(sys.argv[2])
    else:
        {"one": one_process, "baseline": baseline}[mode]()
