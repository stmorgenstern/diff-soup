"""Check of a per-member scalar with a time derivative, on a padded family.

Not part of the benchmark suite: a hand-run check on a GPU with cuDSS, printed and asserted.

    JAX_ENABLE_X64=1 python benchmarks/member_scalar_check.py
    JAX_ENABLE_X64=1 python benchmarks/member_scalar_check.py pullbacks

``pullbacks`` prints only the cost of the iteration matrix's assembly (held rows,
pullbacks, colours) at 8 and 64 members whose scalar rows read shared host unknowns,
before and after the recolouring pass chose its shared columns again, and the matrix
against a dense ``jax.jacfwd`` (see ``pullbacks``).

The model. A one-cell domain holds a shared differential field ``film`` with
``d film/dt = g``, so ``film(t) = g t``. A family of 64 members, declared with
``nodes=`` so that half the members have 10 sites and half have 6, holds

- a site field ``c`` with ``dc/dt = D lap(c) - a_m (c - 1)`` (no flux at the ends);
- an algebraic member scalar ``rate``, the member's mean of that right-hand side; the
  diffusion conserves the mean, so ``rate(t) = -a_m (cbar_m(0) - 1) e^{-a_m t}``;
- a differential member scalar ``layer`` with
  ``d layer/dt = k smooth_abs(rate, eps) (film - layer)``, which reads the shared ``film``.

``layer(t)`` is compared with a reference from a fine Runge-Kutta quadrature of its
equation with the known ``rate(t)`` and ``film(t)`` (plain numpy), for JIDA, JBDF and JNDF
on cuDSS and on ``MemberSchur``. The printout also gives the mass diagonal on each kind of
row, the role tolerances, the iteration matrix against a dense ``jax.jacfwd``, the cost of
each solver (pattern nonzeros, colours, rows held out of the colouring, the shared system
of ``MemberSchur``, compile and warm solve times), and the forward derivative of
``layer(T)`` in ``k`` beside a central difference, both solves on capped steps, and beside
the reference sensitivity.
"""

# ruff: noqa: I001

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import diff_soup as ds  # diff_soup before jax
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.domains import cartesian_grid_1d
from diff_soup.linear._structured._member_elimination import member_schur_plan
from diff_soup.utils import smooth_abs

COUNT = 64
N_MAX, N_MIN = 10, 6
NODES = [N_MAX] * (COUNT // 2) + [N_MIN] * (COUNT // 2)
T_END = 2.0
K = 1.5  # uptake constant, the parameter differentiated
G = 0.5  # growth of the shared film
EPS = 0.05  # smoothing width of |rate|
DIFF = 0.02  # diffusion along a member
A = np.linspace(0.5, 2.0, COUNT)  # relaxation rate per member
AMP = np.linspace(-0.8, 0.8, COUNT)  # sets cbar(0) - 1, both signs
SAVE = np.linspace(0.2, T_END, 10)
RTOL, ATOL = 1e-8, 1e-10

failures: list = []


def check(label: str, value: float, limit: float) -> None:
    ok = bool(np.isfinite(value) and value <= limit)
    print(f"  {'pass' if ok else 'FAIL'}  {label}: {value:.3e} (limit {limit:.0e})")
    if not ok:
        failures.append(label)


def initial_profiles():
    """``c(0)`` on the padded ``(COUNT, N_MAX)`` shape, and each member's mean over its own sites."""
    c0 = np.ones((COUNT, N_MAX))
    for m, n in enumerate(NODES):
        x = (np.arange(n) + 0.5) / n
        c0[m, :n] = 1.0 + AMP[m] * (1.0 + 0.5 * np.cos(np.pi * x))
        c0[m, n:] = c0[m, n - 1]
    cbar0 = np.array([c0[m, :n].mean() for m, n in enumerate(NODES)])
    return c0, cbar0


def build():
    """The problem, the handle of ``k`` and the initial state."""
    spec = ds.problem("member_scalar_check").domain(
        "cell", cartesian_grid_1d(1, 0.0, 1.0, method="fvm")
    )
    film = spec.field("film", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))
    k = spec.param("k", value=K, learnable=True)
    spec.equation(writes="film", reads=("film",))(lambda q: G + 0.0 * q.film)

    names = ("c", "rate", "layer")
    members = spec.family(
        "p",
        count=COUNT,
        domain=cartesian_grid_1d(N_MAX, 0.0, 1.0, method="fvm"),
        nodes=NODES,
        params={"a": A},
        scale={n: 1.0 for n in names},
        residual_scale={n: 1.0 for n in names},
    )
    members.field_decl("c", mass="differential")
    members.field_decl("rate", mass="algebraic", shape=(1,))
    members.field_decl("layer", mass="differential", shape=(1,))

    def relax(q, m):
        return DIFF * q.ops.laplacian(q.c) - m.a * (q.c - 1.0)

    def rate_row(q, m):
        own = jnp.arange(N_MAX) < m.member_nodes
        mean = jnp.sum(jnp.where(own, relax(q, m), 0.0)) / m.member_nodes
        return q.rate - mean

    def layer_row(q, m):
        shared = jnp.reshape(q.film, (-1,))[0]
        return q.p.k * smooth_abs(q.rate, EPS) * (shared - q.layer)

    members.equation(writes="c", reads=("c",))(relax)
    members.equation(writes="rate", reads=("c", "rate"))(rate_row)
    members.equation(writes="layer", reads=("layer", "rate", "film"))(layer_row)
    spec.couple(film, onto="p", at=np.zeros(COUNT, dtype=np.int64))
    problem = spec.build()

    c0, cbar0 = initial_profiles()
    y0 = problem.assemble_state(
        {
            "film": jnp.zeros(1),
            "p.c": jnp.asarray(c0),
            "p.rate": jnp.asarray(-A * (cbar0 - 1.0))[:, None],
            "p.layer": jnp.zeros((COUNT, 1)),
        }
    )
    return problem, k, y0, cbar0


def reference(cbar0, times, steps=40_000):
    """``layer`` and ``d layer / dk`` at ``times`` by classical Runge-Kutta (numpy)."""

    def rhs(t, y):
        rate = -A * (cbar0 - 1.0) * np.exp(-A * t)
        s = np.sqrt(rate**2 + EPS**2) - EPS
        layer, sens = y
        dl = K * s * (G * t - layer)
        ds_ = s * (G * t - layer) - K * s * sens
        return np.stack([dl, ds_])

    h = T_END / steps
    y = np.zeros((2, COUNT))
    out, t, want = [], 0.0, list(times)
    for i in range(steps):
        k1 = rhs(t, y)
        k2 = rhs(t + h / 2, y + h / 2 * k1)
        k3 = rhs(t + h / 2, y + h / 2 * k2)
        k4 = rhs(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = (i + 1) * h
        while want and abs(t - want[0]) < h / 2:
            out.append(y.copy())
            want.pop(0)
    out = np.asarray(out)
    return out[:, 0], out[:, 1]


def build_count(count: int, tag: str = ""):
    """A host of two unknowns, ``ce`` and ``film``, and ``count`` members of ``N_MAX`` sites.

    Each member's ``rate`` row reads its sites and ``ce`` (a wide row, recovered by
    pullback), and its ``layer`` row reads ``rate``, ``layer`` and ``film``: every
    ``layer`` row reads the one host column ``film``. Returns the problem and ``theta``.
    """
    spec = ds.problem(f"member_scalar_pullbacks_{count}{tag}").domain(
        "cell", cartesian_grid_1d(1, 0.0, 1.0, method="fvm")
    )
    ce = spec.field("ce", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    film = spec.field("film", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))
    spec.param("k", value=K, learnable=True)
    spec.equation(writes="ce", reads=("ce",))(lambda q: -0.1 * q.ce)
    spec.equation(writes="film", reads=("film",))(lambda q: G + 0.0 * q.film)
    names = ("c", "rate", "layer")
    members = spec.family(
        "p",
        count=count,
        domain=cartesian_grid_1d(N_MAX, 0.0, 1.0, method="fvm"),
        params={"a": np.linspace(0.5, 2.0, count)},
        scale={n: 1.0 for n in names},
        residual_scale={n: 1.0 for n in names},
    )
    members.field_decl("c", mass="differential")
    members.field_decl("rate", mass="algebraic", shape=(1,))
    members.field_decl("layer", mass="differential", shape=(1,))

    def relax(q, m):
        return DIFF * q.ops.laplacian(q.c) - m.a * (q.c - 1.0)

    def rate_row(q, m):
        host = jnp.reshape(q.ce, (-1,))[0]
        return q.rate - jnp.mean(relax(q, m)) * host

    def layer_row(q, m):
        shared = jnp.reshape(q.film, (-1,))[0]
        return q.p.k * smooth_abs(q.rate, EPS) * (shared - q.layer)

    members.equation(writes="c", reads=("c",))(relax)
    members.equation(writes="rate", reads=("c", "rate", "ce"))(rate_row)
    members.equation(writes="layer", reads=("layer", "rate", "film"))(layer_row)
    spec.couple(ce, onto="p", at=np.zeros(count, dtype=np.int64))
    spec.couple(film, onto="p", at=np.zeros(count, dtype=np.int64))
    problem = spec.build()
    return problem, ds.default_theta(problem)


def pullbacks() -> int:
    """Held rows, pullbacks and colours at 8 and 64 members, and the iteration matrix.

    ``before``: the recolouring pass keeps the shared columns of the first choice (as
    diff-soup did up to 802bd5c, reproduced by replacing ``_share_columns``); ``now``: it
    chooses them again over the enlarged held rows. The assembled iteration matrix is
    compared with a dense ``jax.jacfwd`` of the residual for both.
    """
    from diff_soup.problem.implicit import linearize

    current = linearize._share_columns

    def first_choice_kept():
        first = {}

        def share(rows, cols, wide, narrow, n):
            if "shared" not in first:
                cost, split = current(rows, cols, wide, narrow, n)
                first["shared"] = split.shared
                return cost, split
            kept = np.isin(rows, wide) & ~np.isin(cols, first["shared"])
            groups = linearize._disjoint_groups(rows[kept], cols[kept], wide, n)
            return 0, linearize._WideRows(wide, groups, first["shared"])

        return share

    print(f"devices {jax.devices()}")
    print("members | rule   | n    | held rows | pullbacks | colours | max |assembled - dense|")
    counts = {}
    for count in (8, 64):
        for rule in ("before", "now"):
            linearize._share_columns = first_choice_kept() if rule == "before" else current
            try:
                problem, theta = build_count(count, rule)
                sv = ds.solver_view(problem)
                lin = sv.plan.linearization
                held = int(np.asarray(lin.held_rows).size)
                state, slope = jax.random.uniform(
                    jax.random.PRNGKey(3), (2, sv.n), minval=0.5, maxval=1.5
                )
                t, cj = jnp.asarray(0.3), jnp.asarray(7.0)
                values = sv.iteration_matrix_values(t, state, slope, theta, cj)
            finally:
                linearize._share_columns = current
            assembled = np.zeros((sv.n, sv.n))
            np.add.at(assembled, (sv.full_topology.rows, sv.full_topology.cols), np.asarray(values))
            dense = jax.jacfwd(lambda y: sv.residual(t, y, slope, theta))(state) + cj * jax.jacfwd(
                lambda yd: sv.residual(t, state, yd, theta)
            )(slope)
            err = float(np.abs(assembled - np.asarray(dense)).max())
            print(
                f"  {count:5d} | {rule:6s} | {sv.n:4d} | {held:9d} | {lin.pullback_count:9d} | "
                f"{lin.color_count:7d} | {err:.2e}"
            )
            check(f"{count} members, {rule}: max |assembled - dense|", err, 1e-12)
            counts[(count, rule)] = lin.pullback_count
    check("pullbacks at 64 members - at 8 (now)", float(abs(counts[(64, "now")] - counts[(8, "now")])), 0.0)
    print("\n" + ("all checks pass" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


def main() -> int:
    if sys.argv[1:] == ["pullbacks"]:
        return pullbacks()
    print(f"devices {jax.devices()}, cuDSS {ds.linear.cudss_available()}")
    problem, k_handle, y0, cbar0 = build()
    view = problem.family_view("p")
    theta0 = ds.default_theta(problem)
    n = int(problem.state_size())
    print(f"unknowns {n}: film 1, members {COUNT} with {N_MAX} or {N_MIN} sites, rate and layer")

    # -- roles ------------------------------------------------------------------------------
    print("\nroles")
    mass = np.asarray(problem.mass_diagonal())
    for name in ("c", "rate", "layer"):
        rows = np.asarray(view.field(jnp.asarray(mass), name))
        if name == "c":
            rows = np.concatenate([rows[m, :nm] for m, nm in enumerate(NODES)])
        print(f"  mass on {name} rows: min {rows.min():.3g}, max {rows.max():.3g}")
    layer_mass = np.asarray(view.field(jnp.asarray(mass), "layer"))
    rate_mass = np.asarray(view.field(jnp.asarray(mass), "rate"))
    check("mass of layer rows - 1", float(np.abs(layer_mass - 1.0).max()), 0.0)
    check("mass of rate rows", float(np.abs(rate_mass).max()), 0.0)
    n_alg = len(getattr(problem, "_alg_indices", ()))
    print(f"  algebraic rows {n_alg} (expected {COUNT}, the rate rows)")
    check("algebraic row count - 64", float(abs(n_alg - COUNT)), 0.0)
    try:
        from diff_soup.timestepping.tolerances import resolve_for_problem

        resolved = resolve_for_problem(
            ds.ToleranceSpec(rtol=RTOL, atol={"differential": 1e-10, "algebraic": 1e-7}),
            ds.solver_view(problem),
        )
        atol_layer = np.asarray(view.field(jnp.asarray(resolved.atol[:n]), "layer"))
        print(f"  role tolerance on layer rows: {np.unique(atol_layer)} (differential 1e-10)")
        check("layer atol - 1e-10", float(np.abs(atol_layer - 1e-10).max()), 0.0)
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        print(f"  role tolerances not resolved: {exc!r}")
        failures.append("role tolerances")

    # -- the iteration matrix ---------------------------------------------------------------
    print("\niteration matrix F_y + cj F_ydot against jax.jacfwd at a random state")
    sv = ds.solver_view(problem)
    state, slope = jax.random.uniform(jax.random.PRNGKey(3), (2, sv.n), minval=0.5, maxval=1.5)
    t, cj = jnp.asarray(0.3), jnp.asarray(7.0)
    values = sv.iteration_matrix_values(t, state, slope, theta0, cj)
    assembled = np.zeros((sv.n, sv.n))
    np.add.at(assembled, (sv.full_topology.rows, sv.full_topology.cols), np.asarray(values))
    dense = jax.jacfwd(lambda y: sv.residual(t, y, slope, theta0))(state) + cj * jax.jacfwd(
        lambda yd: sv.residual(t, state, yd, theta0)
    )(slope)
    check("max |assembled - dense|", float(np.abs(assembled - np.asarray(dense)).max()), 1e-12)

    # -- cost of the pattern ----------------------------------------------------------------
    print("\npattern")
    report = ds.topology_report(problem)
    plan_view = ds.solver_view(sv)
    held = np.asarray(getattr(plan_view.plan, "held_out_rows", ())).size
    print(
        f"  n {report.n}, nonzeros {report.nnz}, colours {report.color_count}, "
        f"pullbacks {report.pullback_count}, rows held out {held}, widest row {report.max_row_degree}"
    )
    try:
        layout = ds.linear.member_schur_layout(problem)
        plan = member_schur_plan(layout, sv.full_topology, "cyclic")
        print(
            f"  MemberSchur: {layout.n_members} members, sites {sorted(set(layout.lengths))}, "
            f"scalars per member {sorted(set(layout.scalar_counts))}; eliminated scalars per "
            f"member {plan.k}; shared system {plan.n_shared} unknowns, {plan.nnz_shared} nonzeros"
        )
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        print(f"  MemberSchur layout or plan FAILED: {exc!r}")
        failures.append("member_schur_layout")

    # -- accuracy, both solvers, three steppers ---------------------------------------------
    ref_layer, ref_sens = reference(cbar0, SAVE)
    ref_rate = -A[None, :] * (cbar0 - 1.0)[None, :] * np.exp(-A[None, :] * SAVE[:, None])
    saveat = ds.SaveAt(ts=jnp.asarray(SAVE), ordered=True)
    linears = {
        "cudss": lambda: ds.linear.CuDSS(),
        "member_schur": lambda: ds.linear.MemberSchur(
            ds.linear.member_schur_layout(problem), shared=ds.linear.CuDSS(), elimination="cyclic"
        ),
    }
    layers = {}
    print(f"\naccuracy at rtol {RTOL:g}, atol {ATOL:g}; errors over the {SAVE.size} save times")
    for stepper in ("JIDA", "JBDF", "JNDF"):
        for lin_name, make in linears.items():
            solver = getattr(ds, stepper).stock(linear=make(), rtol=RTOL, atol=ATOL)
            label = f"{stepper} {lin_name}"
            try:
                compiled = ds.compile(problem, solver, (0.0, T_END), y0, saveat=saveat, theta=theta0)
                sol = compiled((0.0, T_END), y0, theta=theta0)
                jax.block_until_ready(sol.ys)
                start = time.perf_counter()
                sol = compiled((0.0, T_END), y0, theta=theta0)
                jax.block_until_ready(sol.ys)
                warm = time.perf_counter() - start
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                print(f"  {label}: FAILED {exc!r}")
                failures.append(label)
                continue
            ys = np.asarray(sol.ys)
            layer = np.stack([np.asarray(view.field(jnp.asarray(y), "layer"))[:, 0] for y in ys])
            rate = np.stack([np.asarray(view.field(jnp.asarray(y), "rate"))[:, 0] for y in ys])
            cfull = [np.asarray(view.field(jnp.asarray(y), "c")) for y in ys]
            cbar = np.stack([[c[m, :nm].mean() for m, nm in enumerate(NODES)] for c in cfull])
            cbar_ref = 1.0 + (cbar0 - 1.0)[None, :] * np.exp(-A[None, :] * SAVE[:, None])
            film = ys[:, 0]
            layers[label] = layer
            print(
                f"  {label}: compile {compiled.compile_seconds:.1f} s, warm solve {warm * 1e3:.1f} ms, "
                f"accepted {int(sol.stats.accepts)}, rejected {int(sol.stats.rejects)}, "
                f"factorizations {int(sol.stats.factorizations)}"
            )
            print(
                f"    layer(T) range {layer[-1].min():.4f} .. {layer[-1].max():.4f}; "
                f"film(T) {film[-1]:.6f} (exact {G * T_END:.6f})"
            )
            check(f"{label} layer max abs error", float(np.abs(layer - ref_layer).max()), 1e-6)
            check(f"{label} rate max abs error", float(np.abs(rate - ref_rate).max()), 1e-6)
            check(f"{label} cbar max abs error", float(np.abs(cbar - cbar_ref).max()), 1e-6)
            check(f"{label} film max abs error", float(np.abs(film - G * SAVE).max()), 1e-8)
    print("\ncuDSS against MemberSchur, layer over the save times")
    for stepper in ("JIDA", "JBDF", "JNDF"):
        a, b = layers.get(f"{stepper} cudss"), layers.get(f"{stepper} member_schur")
        if a is not None and b is not None:
            check(f"{stepper} max |cudss - member_schur|", float(np.abs(a - b).max()), 1e-9)

    # -- derivative of layer(T) in k ----------------------------------------------------------
    hmax = T_END / 1000
    capped = dict(hmax=hmax, dt0=1e-5, max_steps=20_000)
    print(f"\nd layer(T) / dk on JIDA, steps capped at {hmax:g}, rtol 1e-6, atol 1e-9")
    for lin_name, make in linears.items():
        solver = ds.JIDA.stock(linear=make(), rtol=1e-6, atol=1e-9)

        def layer_end(k_value, solver=solver):
            theta = ds.default_theta(problem, {k_handle: k_value})
            sol = ds.pdesolve(problem, solver, (0.0, T_END), y0, theta=theta, **capped)
            return view.field(sol.ys[-1], "layer")[:, 0], sol.stats.accepts, sol.stats.rejects

        try:
            start = time.perf_counter()
            (value, _, _), (tangent, _, _) = jax.jvp(
                lambda kv: layer_end(kv), (jnp.asarray(K),), (jnp.asarray(1.0),)
            )
            tangent = np.asarray(jax.block_until_ready(tangent))
            first = time.perf_counter() - start
            step = 1e-3 * K
            plus, acc_p, rej_p = layer_end(jnp.asarray(K + step))
            minus, acc_m, rej_m = layer_end(jnp.asarray(K - step))
            central = (np.asarray(plus) - np.asarray(minus)) / (2 * step)
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            print(f"  {lin_name}: FAILED {exc!r}")
            failures.append(f"derivative {lin_name}")
            continue
        scale = np.abs(ref_sens[-1]).max()
        print(
            f"  {lin_name}: forward derivative (compile and run) {first:.1f} s; difference solves "
            f"accepted {int(acc_p)} / {int(acc_m)}, rejected {int(rej_p)} / {int(rej_m)}"
        )
        print(
            f"    dlayer/dk range {tangent.min():.6f} .. {tangent.max():.6f}; reference "
            f"{ref_sens[-1].min():.6f} .. {ref_sens[-1].max():.6f}"
        )
        check(
            f"{lin_name} |forward - central| / max|ref|",
            float(np.abs(tangent - central).max() / scale),
            1e-5,
        )
        check(
            f"{lin_name} |forward - reference| / max|ref|",
            float(np.abs(tangent - ref_sens[-1]).max() / scale),
            1e-3,
        )
        check(f"{lin_name} same accepted steps (difference)", float(abs(int(acc_p) - int(acc_m))), 0.0)

    print("\n" + ("all checks pass" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
