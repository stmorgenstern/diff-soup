"""MemberSchur against dense solves (B2 shared, round 2; throwaway). Usage: check_member_schur.py <dense|klu|cudss> <thomas|cyclic|native> <parts>
parts: comma list of main,complex,singular,topo"""

import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup.linear import (
    MemberSchur,
    MemberSchurLayout,
    SolveContext,
    SparseTopology,
    analyze,
    factor,
    refactor,
    solve,
    solve_checked,
)

TOL = 1e-10
WORST, FAILS = {}, []


class Sys:
    def __init__(self, entries, n, layout, bc, tag=""):
        keys = sorted(entries)
        self.rows = np.asarray([k[0] for k in keys])
        self.cols = np.asarray([k[1] for k in keys])
        self.base = np.asarray([entries[k] for k in keys], dtype=np.float64)
        self.n, self.layout, self.bc = n, layout, bc
        self.topology = SparseTopology(rows=self.rows, cols=self.cols, n_dofs=n, bc_indices=bc)
        self.index = {k: i for i, k in enumerate(keys)}

    def values(self, seed, dtype=np.float64):
        r = np.random.default_rng(seed)
        v = self.base * (1 + 0.05 * r.uniform(-1, 1, self.base.size))
        if np.issubdtype(dtype, np.complexfloating):
            v = v + 1j * 0.3 * self.base * r.uniform(-1, 1, self.base.size)
        return jnp.asarray(v.astype(dtype))


def system(n_members, b, *, links="none", seed=0, n_glob=3, bc=True, scalars=(0, 3), lengths=(3, 40)):
    rng = np.random.default_rng(seed)
    entries, members, scalar_idx = {}, [], []
    glob = list(range(n_glob))
    nxt = n_glob
    for m in range(n_members):
        n_i = int(rng.integers(lengths[0], lengths[1] + 1))
        k_i = int(rng.integers(scalars[0], scalars[1] + 1))
        if links != "none":
            k_i = max(k_i, 1)
        g = np.arange(nxt, nxt + n_i * b).reshape(n_i, b)
        nxt += n_i * b
        s = np.arange(nxt, nxt + k_i)
        nxt += k_i
        members.append(g)
        scalar_idx.append(s)
    n = nxt
    for g, s in zip(members, scalar_idx):
        n_i = g.shape[0]
        for i in range(n_i):
            for j in (i - 1, i, i + 1):
                if 0 <= j < n_i:
                    for p in range(b):
                        for q in range(b):
                            if i == j and p == q:
                                continue
                            if rng.uniform() < 0.8:
                                entries[(g[i, p], g[j, q])] = rng.uniform(-1, 1)
            for p in range(b):
                entries[(g[i, p], g[i, p])] = 4.0 * b + 3.0 + rng.uniform(0, 1)
                for c in glob:
                    entries[(g[i, p], c)] = rng.uniform(-0.3, 0.3)
                for q in s:
                    entries[(g[i, p], q)] = rng.uniform(-0.3, 0.3)
                    entries[(q, g[i, p])] = rng.uniform(-0.3, 0.3) / n_i
        for q in s:
            for r in s:
                entries[(q, r)] = rng.uniform(-0.2, 0.2)
            entries[(q, q)] = 3.0 + rng.uniform(0, 1)
            if glob:
                entries[(q, glob[1])] = rng.uniform(-0.3, 0.3)
        for p in range(b if glob else 0):
            entries[(glob[0], g[-1, p])] = rng.uniform(-0.5, 0.5)
        if len(s) and glob:
            entries[(glob[0], s[0])] = rng.uniform(-0.5, 0.5)
    for c in glob:
        for d in glob:
            entries[(c, d)] = rng.uniform(-0.5, 0.5) + (3.0 + n_members if c == d else 0.0)
    last = [s[-1] if len(s) else None for s in scalar_idx]
    edges = []
    if links == "chain":
        edges = [(m, m + 1) for m in range(n_members - 1)]
    elif links == "network":
        for m in range(n_members):
            for o in rng.choice(n_members, size=2, replace=False):
                if o != m:
                    edges.append((m, int(o)))
    for m, o in edges:
        entries[(last[m], last[o])] = rng.uniform(-0.3, 0.3)
        entries[(last[o], last[m])] = rng.uniform(-0.3, 0.3)
    bcs = None
    if bc:
        bcs = np.unique(np.concatenate([[glob[2]], [g[0, 0] for g in members[::3]]])).astype(np.int64)
    layout = MemberSchurLayout(members, scalars=scalar_idx, n_dofs=n)
    out = Sys(entries, n, layout, bcs)
    out.entries, out.members, out.glob = entries, members, glob
    return out


def dense(topology, v):
    a = jnp.zeros((topology.n_dofs,) * 2, v.dtype).at[topology.rows, topology.cols].add(v)
    if topology.bc_indices is not None:
        bc = np.asarray(topology.bc_indices)
        a = a.at[bc].set(0.0).at[bc, bc].set(1.0)
    return a


def close(name, got, want):
    got, want = np.asarray(got), np.asarray(want)
    err = float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-300))
    WORST[name] = max(WORST.get(name, 0.0), err)
    if not err <= TOL:
        print(f"FAIL {name}: {err:.2e}", flush=True)
        FAILS.append(name)


def run_case(label, shared, s, elim, *, sweeps=0, dtype=np.float64, ctx=None):
    backend = MemberSchur(s.layout, shared, elimination=elim, refinement_sweeps=sweeps)
    an = analyze(s.topology, backend=backend)
    topology, n = s.topology, s.n
    v1, v2 = s.values(1, dtype), s.values(2, dtype)
    a1, a2 = dense(topology, v1), dense(topology, v2)
    rng = np.random.default_rng(5)
    b = jnp.asarray(rng.uniform(-1, 1, n)).astype(dtype)
    B = jnp.asarray(rng.uniform(-1, 1, (3, n))).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        b = b + 1j * jnp.asarray(rng.uniform(-1, 1, n))
    fac = jax.jit(lambda v, o="A": factor(an, v, ctx=ctx, orientation=o), static_argnums=1)
    for orient, m in (("A", a1), ("AT", a1.T)):  # plain transposes
        f = fac(v1, orient)
        for tr, mm in ((False, m), (True, m.T)):
            x, _, rep = jax.jit(lambda f, b: solve_checked(f, b, transpose=tr))(f, b)
            assert int(rep.status) == 0, (label, orient, tr, int(rep.status))
            close(f"{label} solve {orient} T={tr}", x, jnp.linalg.solve(mm, b))
            X = jax.jit(lambda f, B: solve(f, B, transpose=tr)[0])(f, B)
            close(f"{label} multi {orient} T={tr}", X, jnp.linalg.solve(mm, B.T).T)
    f = factor(an, v1, ctx=ctx)
    close(f"{label} eager", solve(f, b)[0], jnp.linalg.solve(a1, b))
    step = jax.jit(lambda f, v, p: solve(refactor(f, v, p), b)[0])
    close(f"{label} refactor True", step(f, v2, jnp.asarray(True)), jnp.linalg.solve(a2, b))
    close(f"{label} refactor False", step(f, v2, jnp.asarray(False)), jnp.linalg.solve(a1, b))
    vs = jnp.stack([s.values(10 + i, dtype) for i in range(3)])

    def lane(v, p):
        g = factor(an, v, ctx=ctx)
        g2 = refactor(g, v * 1.1, p)
        return solve(g, b)[0], solve(g, b, transpose=True)[0], solve(g2, b)[0]

    xs, xts, x2s = jax.jit(jax.vmap(lane))(vs, jnp.asarray([True, False, True]))
    for i in range(3):
        ai = dense(topology, vs[i])
        close(f"{label} vmap", xs[i], jnp.linalg.solve(ai, b))
        close(f"{label} vmap T", xts[i], jnp.linalg.solve(ai.T, b))
        a_ref = dense(topology, vs[i] * 1.1) if i != 1 else ai
        close(f"{label} vmap refactor", x2s[i], jnp.linalg.solve(a_ref, b))
    if not np.issubdtype(dtype, np.complexfloating):
        def loss(v):
            y = solve(factor(an, v, ctx=ctx), b)[0]
            return jnp.sum(jnp.sin(y)) + 0.5 * y @ y

        def loss_ref(v):
            y = jnp.linalg.solve(dense(topology, v), b)
            return jnp.sum(jnp.sin(y)) + 0.5 * y @ y

        close(f"{label} grad", jax.jit(jax.grad(loss))(v1), jax.grad(loss_ref)(v1))
    return an


def grid_systems():
    for b in (1, 2):
        for links in ("none", "chain", "network"):
            yield f"b={b} {links}", system(7, b, links=links, seed=b * 10 + len(links))


def part_main(which, shared, elim, ctx, dtype, tag):
    for name, s in grid_systems():
        an = run_case(f"{which} {elim} {tag} {name}", shared, s, elim, dtype=dtype, ctx=ctx)
        print(f"{tag} {name}: n={s.n} n_shared={an.backend_analysis.core.n_shared}", flush=True)
    s = system(5, 2, links="chain", seed=3)
    run_case(f"{which} {elim} {tag} sweeps=2", shared, s, elim, sweeps=2, dtype=dtype, ctx=ctx)
    s = system(5, 1, n_glob=0, bc=False, seed=4)
    an = run_case(f"{which} {elim} {tag} n_shared=0", shared, s, elim, dtype=dtype, ctx=ctx)
    assert an.backend_analysis.core.n_shared == 0 and an.backend_analysis.shared is None


def statuses(an, v, b, ctx):
    f = factor(an, v, ctx=ctx)
    x, _, rep = solve_checked(f, b)
    xt, _, rept = solve_checked(f, b, transpose=True)
    return int(rep.status), bool(np.all(np.isnan(np.asarray(x)))), int(rept.status), bool(np.all(np.isnan(np.asarray(xt))))


def part_singular(which, shared, elim, ctx):
    # (a) zero pivot at the first grid point of member 0 (b=1, no links), whole matrix nonsingular
    s = system(4, 1, links="none", seed=21, bc=False, scalars=(1, 1))
    g = s.members[0]
    v = np.array(s.values(1))
    v[s.index[(g[0, 0], g[0, 0])]] = 0.0
    assert (g[0, 0], g[1, 0]) in s.index and (g[1, 0], g[0, 0]) in s.index
    v[s.index[(g[0, 0], g[1, 0])]] = 1.0
    v[s.index[(g[1, 0], g[0, 0])]] = 1.0
    a = np.asarray(dense(s.topology, jnp.asarray(v)))
    print(f"zero-pivot matrix: cond {np.linalg.cond(a):.2e}, x by dense finite: "
          f"{np.all(np.isfinite(np.linalg.solve(a, np.ones(s.n))))}", flush=True)
    b = jnp.ones(s.n)
    for sweeps in (0, 3):
        an = analyze(s.topology, backend=MemberSchur(s.layout, shared, elimination=elim, refinement_sweeps=sweeps))
        print(f"{which} {elim} zero pivot sweeps={sweeps}: (status, x all NaN, status T, xT all NaN) = "
              f"{statuses(an, jnp.asarray(v), b, ctx)}", flush=True)
    # (b) singular shared system: global row 1 zeroed (entries kept in the pattern)
    s = system(4, 1, links="chain", seed=22, bc=False)
    v = np.array(s.values(1))
    for (r, c), i in s.index.items():
        if r == s.glob[1]:
            v[i] = 0.0
    a = np.asarray(dense(s.topology, jnp.asarray(v)))
    print(f"singular-shared matrix: rank {np.linalg.matrix_rank(a)} of {s.n}", flush=True)
    for sweeps in (0, 3):
        an = analyze(s.topology, backend=MemberSchur(s.layout, shared, elimination=elim, refinement_sweeps=sweeps))
        print(f"{which} {elim} singular shared sweeps={sweeps}: {statuses(an, jnp.asarray(v), jnp.ones(s.n), ctx)}",
              flush=True)


def part_topo(which, shared, elim, ctx):
    s = system(6, 2, links="chain", seed=31)
    n = s.n
    # one unknown larger: trailing unknown n coupled to a global and to a grid unknown of member 0
    e = dict(s.entries)
    g00 = int(s.members[0][1, 0])
    e.update({(n, n): 5.0, (n, 0): 0.3, (0, n): -0.2, (g00, n): 0.25, (n, g00): -0.1})
    big = Sys(e, n + 1, s.layout, s.bc)
    an = run_case(f"{which} {elim} topo n+1", shared, big, elim, ctx=ctx)
    run_case(f"{which} {elim} topo n+1 complex", shared, big, elim, dtype=np.complex128, ctx=ctx)
    print(f"topo n+1: n_shared={an.backend_analysis.core.n_shared}", flush=True)
    # smaller: another system of size n - 7 (random sparse, diagonally dominant)
    m = n - 7
    rng = np.random.default_rng(32)
    e = {(i, i): 6.0 + rng.uniform() for i in range(m)}
    for _ in range(4 * m):
        i, j = rng.integers(0, m, 2)
        e[(int(i), int(j))] = e.get((int(i), int(j)), 0.0) + rng.uniform(-1, 1)
    small = Sys(e, m, s.layout, np.asarray([1, 5]))
    an = run_case(f"{which} {elim} topo smaller", shared, small, elim, ctx=ctx)
    run_case(f"{which} {elim} topo smaller complex", shared, small, elim, dtype=np.complex128, ctx=ctx)
    print(f"topo smaller: n={m} n_shared={an.backend_analysis.core.n_shared}", flush=True)


def main():
    which, elim, parts = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    shared = {"dense": ds.linear.Dense, "klu": ds.linear.KLU, "cudss": ds.linear.CuDSS}[which]()
    with SolveContext() as ctx:
        if "main" in parts:
            part_main(which, shared, elim, ctx, np.float64, "real")
        if "complex" in parts:
            part_main(which, shared, elim, ctx, np.complex128, "complex")
        if "singular" in parts:
            part_singular(which, shared, elim, ctx)
        if "topo" in parts:
            part_topo(which, shared, elim, ctx)
    for k, e in sorted(WORST.items()):
        print(f"{e:.2e}  {k}")
    print("worst", max(WORST.values()) if WORST else None, "cases", len(WORST), "fails", len(FAILS))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
