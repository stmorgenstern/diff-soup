"""Check of forward derivatives through a solve when a parameter enters a global's constraint.

Not part of the benchmark suite: a hand-run check, printed and asserted.

    JAX_ENABLE_X64=1 python benchmarks/global_constraint_derivative_check.py

The model. A field ``y`` on a uniform grid of the unit interval (finite volumes, no flux
at the ends), an algebraic global ``g`` and a differential global ``layer``:

- ``dy/dt = D lap(y) - k (y - g) - s p``;
- ``0 = sum_i V_i (p y_i - g)``, a ``spec.constraint.domain_integral`` that closes ``g``
  (the volumes sum to one, so ``g = p ybar``, ``ybar`` the mean of ``y``);
- ``d layer/dt = r (g - layer)``.

The diffusion conserves the mean and ``k`` is uniform, so the mean obeys
``ybar' = -a ybar - s p`` with ``a = k (1 - p)``: ``ybar = A e^{-a t} + B``,
``A = ybar(0) - B``, ``B = -s p / a``. Each node's departure from the mean does not
depend on ``p``, so ``dy_i/dp = dybar/dp`` at every node. With ``layer(0) = L0``,
``layer = L0 e^{-r t} + p B (1 - e^{-r t}) + p A r / (r - a) (e^{-a t} - e^{-r t})``.
The exact derivatives in ``p`` are ``jax.jacfwd`` of these closed forms.

Cases:

1. ``s = 0``: ``p`` enters only the constraint's integrand. The start state holds ``g``
   at its consistent value at the unperturbed ``p`` (as a seeded start does), so the
   start of ``g`` depends on ``p`` and the solve's consistent initialization supplies it.
2. ``s = 1/2``: ``p`` also enters every node's row of ``y``.
3. ``s = 0`` with ``g`` seeded at zero: a start that is consistent only after the
   initialization moves the algebraic unknown.

For each, the forward derivative of ``y``, ``g`` and ``layer`` at the save times
(``jax.jit(jax.jacfwd(...))`` of the solve, with ``theta.replace`` of the parameter, the way
a fit's derivative check differentiates a run) is printed beside the exact derivative and
beside central differences at steps ``h`` and ``h/2``, all on capped steps. The solver is
JIDA as an experiment's solver builds it (consistent initialization ``ya_ydp``, algebraic
rows out of the error test). The iteration matrix is also compared with a dense
``jax.jacfwd`` of the residual, and the rows the plan holds out of its colouring are
printed, since the constraint's row is one of them.
"""

# ruff: noqa: I001

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import diff_soup as ds  # diff_soup before jax
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.domains import cartesian_grid_1d

N = 40
D = 0.05
K = 2.0
R = 3.0
P0 = 0.4
S_SOURCE = 0.5
L0 = 0.1
T0, T_END = 0.0, 1.5
SAVE = np.linspace(0.25, T_END, 6)
RTOL, ATOL = 1e-9, 1e-11
HMAX = T_END / 3000
FD_STEP = 1e-3

failures: list = []


def check(label: str, value: float, limit: float) -> None:
    ok = bool(np.isfinite(value) and value <= limit)
    print(f"  {'pass' if ok else 'FAIL'}  {label}: {value:.3e} (limit {limit:.0e})")
    if not ok:
        failures.append(label)


def build(source: float):
    """The problem, the handle of ``p`` and ``y(0)``; ``source`` is ``s`` (0: ``p`` only in the constraint)."""
    spec = ds.problem(f"global_constraint_{'source' if source else 'plain'}").domain(
        "line", cartesian_grid_1d(N, 0.0, 1.0, method="fvm")
    )
    spec.field("y", mass="differential", scale=1.0)
    p = spec.param("gain", value=P0, learnable=True)
    spec.global_var("g", initial=0.0, scale=1.0)
    spec.global_var("layer", initial=L0, mass="differential", scale=1.0)

    if source:
        spec.equation(writes="y", reads=("y", "g"))(
            lambda q: D * q.ops.laplacian(q.y) - K * (q.y - q.g) - source * q.p.gain
        )
    else:
        spec.equation(writes="y", reads=("y", "g"))(
            lambda q: D * q.ops.laplacian(q.y) - K * (q.y - q.g)
        )
    spec.equation(writes="layer", reads=("layer", "g"))(lambda q: R * (q.g - q.layer))
    spec.constraint.domain_integral(
        name="g",
        field="y",
        integrand_fn=lambda q: q.p.gain * q.y - q.g,
        target_fn=lambda t: jnp.zeros_like(jnp.asarray(t, dtype=jnp.float64)),
        closes="g",
    )
    problem = spec.build()
    x = (np.arange(N) + 0.5) / N
    y0 = 1.0 + 0.5 * np.cos(np.pi * x) + 0.2 * x
    return problem, p, y0


def exact(p, source, ybar0):
    """``(ybar, g, layer)`` at ``SAVE`` in closed form, a function of ``p``."""
    t = jnp.asarray(SAVE)
    a = K * (1.0 - p)
    b = -source * p / a
    big_a = ybar0 - b
    ybar = big_a * jnp.exp(-a * t) + b
    g = p * ybar
    layer = (
        L0 * jnp.exp(-R * t)
        + p * b * (1.0 - jnp.exp(-R * t))
        + p * big_a * R / (R - a) * (jnp.exp(-a * t) - jnp.exp(-R * t))
    )
    return ybar, g, layer


def solver():
    """JIDA as an experiment's solver builds it."""
    linear = ds.linear.CuDSS() if ds.linear.cudss_available() else "auto"
    return ds.JIDA.stock(
        rtol=RTOL,
        atol=ATOL,
        linear=linear,
        consistent_ic=True,
        ic_mode="ya_ydp",
        suppress_alg=True,
        precision=ds.Precision(),
        ic_max_newton_iters=10,
    )


def run_case(label, source, g_seed):
    print(f"\n{label}")
    problem, handle, y_init = build(source)
    layout = {g["name"]: g for g in problem.global_layout()}
    fields = np.asarray(problem.assemble_state({"y": jnp.asarray(y_init)}), np.float64)
    globals0 = np.zeros(sum(g["size"] for g in layout.values()))
    ybar0 = float(np.mean(y_init))
    globals0[layout["g"]["offset"]] = P0 * ybar0 if g_seed == "consistent" else 0.0
    globals0[layout["layer"]["offset"]] = L0
    y0 = jnp.asarray(np.concatenate([fields, globals0]))
    y_slice = problem.field_slice("y")
    ig, il = layout["g"]["offset"], layout["layer"]["offset"]
    theta0 = ds.default_theta(problem)
    stepper = solver()
    saveat = ds.SaveAt(ts=jnp.asarray(SAVE))

    def observed(x):
        theta = theta0.replace(handle, jnp.asarray(P0, dtype=jnp.float64) + x[0])
        sol = stepper.solve(
            problem, (T0, T_END), y0, saveat=saveat, theta=theta,
            dt0=1e-6, hmax=HMAX, max_steps=40_000,
        )
        ys = sol.ys[:, y_slice]
        return jnp.concatenate([ys, sol.lambdas[:, ig : ig + 1], sol.lambdas[:, il : il + 1]], axis=1)

    value = jax.jit(observed)
    jac = jax.jit(jax.jacfwd(observed))
    zero = jnp.zeros(1)
    v0 = np.asarray(value(zero))
    forward = np.asarray(jac(zero))[..., 0]
    e = jnp.asarray([FD_STEP])
    fd = (np.asarray(value(e)) - np.asarray(value(-e))) / (2 * FD_STEP)
    fd2 = (np.asarray(value(e / 2)) - np.asarray(value(-e / 2))) / FD_STEP

    ybar_x, g_x, layer_x = exact(jnp.asarray(P0), source, ybar0)
    dybar, dg, dlayer = jax.jacfwd(lambda p: jnp.stack(exact(p, source, ybar0)))(jnp.asarray(P0))
    want_value = {"g": np.asarray(g_x), "layer": np.asarray(layer_x)}
    want = {
        "y": np.repeat(np.asarray(dybar)[:, None], N, axis=1),
        "g": np.asarray(dg),
        "layer": np.asarray(dlayer),
    }
    got = {"y": forward[:, :N], "g": forward[:, N], "layer": forward[:, N + 1]}
    diff = {"y": fd2[:, :N], "g": fd2[:, N], "layer": fd2[:, N + 1]}
    diff_h = {"y": fd[:, :N], "g": fd[:, N], "layer": fd[:, N + 1]}
    print(f"  values: max |g - exact| {np.abs(v0[:, N] - want_value['g']).max():.2e}, "
          f"max |layer - exact| {np.abs(v0[:, N + 1] - want_value['layer']).max():.2e}, "
          f"max |mean y - exact| {np.abs(v0[:, :N].mean(axis=1) - np.asarray(ybar_x)).max():.2e}")
    print("  t      | quantity | forward        exact          difference(h/2)")
    for i, t in enumerate(SAVE):
        for name in ("y", "g", "layer"):
            f = got[name][i] if name != "y" else got[name][i, 0]
            w = want[name][i] if name != "y" else want[name][i, 0]
            d = diff[name][i] if name != "y" else diff[name][i, 0]
            print(f"  {t:6.3f} | {name:8s} | {f: .8e} {w: .8e} {d: .8e}")
    for name in ("y", "g", "layer"):
        scale = float(np.abs(want[name]).max())
        check(f"{name}: max |forward - exact| / max |exact|", float(np.abs(got[name] - want[name]).max()) / scale, 1e-5)
        check(f"{name}: max |difference - exact| / max |exact|", float(np.abs(diff[name] - want[name]).max()) / scale, 1e-4)
        print(f"         {name}: max |difference(h) - difference(h/2)| / max |exact| "
              f"{float(np.abs(diff_h[name] - diff[name]).max()) / scale:.2e}")
    return problem, theta0


def matrix_check(problem, theta0):
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
    held = np.asarray(getattr(sv.plan, "held_out_rows", ()))
    print(f"  n {sv.n}, rows held out of the colouring {held.tolist()}")
    check("max |assembled - dense|", float(np.abs(assembled - np.asarray(dense)).max()), 1e-12)


def main() -> int:
    print(f"devices {jax.devices()}, cuDSS {ds.linear.cudss_available()}")
    print(f"N {N}, rtol {RTOL:g}, atol {ATOL:g}, largest step {HMAX:g}, difference step {FD_STEP:g}")
    problem, theta0 = run_case("1. p only in the constraint, g seeded consistent at p0", 0.0, "consistent")
    matrix_check(problem, theta0)
    run_case("2. p also in every node's row of y", S_SOURCE, "consistent")
    run_case("3. p only in the constraint, g seeded at zero", 0.0, "zero")
    print("\n" + ("all checks pass" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
