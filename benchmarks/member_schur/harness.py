"""Saved-matrix harness for the per-particle linear solver (B2): agreement and times against KLU and cuDSS.

Throwaway benchmark of the lithium iron phosphate overhaul (package B2, piece "harness").

    harness.py prepare <measure_dir> <out_dir>          split A_<P>_mid.npz into per-particle blocks (P = 240, 720)
    harness.py run --backend NAME [--sizes 240,720] [--variants base,ragged,...] [--time] [--reps 20]
                   [--blocks DIR] [--out DIR]
    harness.py table <json>...                          markdown tables (agreement, times)

Backends: klu, cudss, member_block, member_schur:<elimination> (thomas, cyclic, native).
Variants: base, ragged (some chains cut to 60..100 points), cplx<w> (A + 1j*w*diag(A)), chain, network,
          and <variant>+cplx<w> for a complex chain or network.
The reference is KLU when JAX runs on the processor and CuDSS when it runs on the graphics card. Every solve goes
through the public ``ds.linear`` calls; the harness's own NumPy and SciPy code only builds matrices and measures.
"""

import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import scipy.sparse as sp

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import diff_soup as ds  # noqa: E402

LIN = ds.linear
SCALAR_NAMES = ("mean", "rate", "drop")


# ----------------------------------------------------------------------------------------------- blocks


def prepare(measure_dir, out_dir, tags=("240", "720"), state="mid"):
    """Split each saved matrix into dense per-particle pieces (the split of measure/blocks.py)."""
    os.makedirs(out_dir, exist_ok=True)
    for tag in tags:
        z = np.load(f"{measure_dir}/A_{tag}_{state}.npz")
        n = int(z["n"])
        A = sp.csr_matrix(sp.coo_matrix((z["vals"], (z["rows"], z["cols"])), shape=(n, n)))
        m = np.load(f"{measure_dir}/maps_{tag}.npz")
        c = m["c"].reshape(m["c"].shape[0], -1)
        s = np.concatenate([m[k].reshape(c.shape[0], -1) for k in SCALAR_NAMES], axis=1)
        P, N = c.shape
        loc = np.concatenate([c.ravel(), s.ravel()])
        glob = np.setdiff1d(np.arange(n), loc)
        sub = lambda r, q: A[r][:, q].toarray()  # noqa: E731
        d = np.empty((P, N)); dl = np.zeros((P, N)); du = np.zeros((P, N))
        E = np.empty((P, N, 3)); F = np.empty((P, 3, N)); G = np.empty((P, 3, 3))
        offtri = 0
        for p in range(P):
            T = sub(c[p], c[p])
            d[p] = np.diag(T); dl[p, 1:] = np.diag(T, -1); du[p, :-1] = np.diag(T, 1)
            offtri += np.count_nonzero(T - np.diag(np.diag(T)) - np.diag(np.diag(T, 1), 1) - np.diag(np.diag(T, -1), -1))
            E[p] = sub(c[p], s[p]); F[p] = sub(s[p], c[p]); G[p] = sub(s[p], s[p])
        Ag = A[:, glob].tocsc()
        J = np.flatnonzero(np.abs(Ag[loc]).sum(axis=0).A1 > 0)
        I = np.flatnonzero(np.abs(A[glob][:, loc]).sum(axis=1).A1 > 0)
        Eg = np.stack([sub(c[p], glob[J]) for p in range(P)]); Gg = np.stack([sub(s[p], glob[J]) for p in range(P)])
        Hc = np.stack([sub(glob[I], c[p]) for p in range(P)]); Hl = np.stack([sub(glob[I], s[p]) for p in range(P)])
        D = sub(glob, glob)
        out = dict(dl=dl, d=d, du=du, E=E, F=F, G=G, Eg=Eg, Gg=Gg, Hc=Hc, Hl=Hl, D=D, I=I, J=J)
        B = assemble(out, np.full(P, N))[0]
        # The split must hold every entry: same count, same values up to the permutation.
        perm = np.concatenate([np.concatenate([c[p], s[p]]) for p in range(P)] + [glob])
        diff = abs(B - A[perm][:, perm]).max()
        print(f"prepare {tag}: P={P} N={N} n={n} nnz={A.nnz} assembled nnz={B.nnz} off-tridiagonal={offtri} "
              f"globals={len(glob)} J={J.tolist()} I={I.tolist()} max|B-PAP^T|={diff:.3e}", flush=True)
        assert offtri == 0 and diff == 0.0 and B.nnz == np.count_nonzero(A.data)  # A holds explicit zeros
        np.savez_compressed(f"{out_dir}/blocks_{tag}.npz", **out)


def load_blocks(blocks_dir, tag):
    z = np.load(f"{blocks_dir}/blocks_{tag}.npz")
    return {k: z[k] for k in z.files}


def grow(b0, P, seed=0):
    """The pieces of P particles: the saved ones repeated cyclically, repeats perturbed by 1e-3 relative.

    The global rows that sum over particles are scaled by P0/P, as measure/m3a.py scales them by 1/rep.
    """
    P0 = b0["d"].shape[0]
    if P == P0:
        return dict(b0)
    idx = np.arange(P) % P0
    rng = np.random.default_rng(seed)
    out = {}
    for k in ("dl", "d", "du", "E", "F", "G", "Eg", "Gg", "Hc", "Hl"):
        x = b0[k][idx].copy()
        pert = 1.0 + 1e-3 * rng.uniform(-1.0, 1.0, size=x.shape)
        pert[:P0] = 1.0  # the first copy is the saved matrix
        x *= pert
        if k in ("Hc", "Hl"):
            x *= P0 / P
        out[k] = x
    out["D"], out["I"], out["J"] = b0["D"], b0["I"], b0["J"]
    return out


def ragged(b, seed=1, frac=1 / 3):
    """Cut a third of the particles to 60..100 points: the chain on an evenly spaced subset of its points.

    Both ends are kept. The tridiagonal entries of the kept points are kept (the cut chain is again tridiagonal),
    and each scalar row is rescaled so that its total weight over the grid stays the same.
    """
    P, N = b["d"].shape
    rng = np.random.default_rng(seed)
    lengths = np.full(P, N)
    cut = rng.random(P) < frac
    lengths[cut] = rng.integers(60, N, size=int(cut.sum()))
    out = {k: v.copy() for k, v in b.items()}
    for p in np.flatnonzero(cut):
        L = lengths[p]
        K = np.unique(np.round(np.linspace(0, N - 1, L)).astype(int))
        assert K.size == L
        for k in ("d", "dl", "du"):
            v = np.zeros(N); v[:L] = b[k][p, K]; out[k][p] = v
        out["dl"][p, 0] = 0.0; out["du"][p, L - 1] = 0.0
        for k in ("E", "Eg"):
            v = np.zeros_like(b[k][p]); v[:L] = b[k][p, K]; out[k][p] = v
        F = b["F"][p]
        w = np.abs(F).sum(1) / np.maximum(np.abs(F[:, K]).sum(1), 1e-300)
        v = np.zeros_like(F); v[:, :L] = F[:, K] * w[:, None]; out["F"][p] = v
        v = np.zeros_like(b["Hc"][p]); v[:, :L] = b["Hc"][p][:, K]; out["Hc"][p] = v
    return out, lengths


def assemble(b, lengths, links=None):
    """Sparse matrix and layout of the pieces with particle p of length lengths[p]; ``links`` adds drop-drop entries.

    Unknowns: particle p is one block of lengths[p] grid points and its 3 scalars, then the globals.
    """
    P, N = b["d"].shape
    k = b["G"].shape[1]
    g = b["D"].shape[0]
    lengths = np.asarray(lengths)
    off = np.concatenate([[0], np.cumsum(lengths + k)])
    rows, cols, vals = [], [], []

    def add(r, q, v, mask=None):
        r, q, v = np.broadcast_arrays(r, q, v)
        keep = v != 0 if mask is None else (np.broadcast_to(mask, v.shape) & (v != 0))
        rows.append(r[keep]); cols.append(q[keep]); vals.append(v[keep])

    j = np.arange(N)[None]
    inside = j < lengths[:, None]                       # (P, N)
    i = off[:-1, None] + j                              # grid index; garbage where not inside (masked)
    add(i, i, b["d"], inside)
    add(i[:, 1:], i[:, :-1], b["dl"][:, 1:], inside[:, 1:])
    add(i[:, :-1], i[:, 1:], b["du"][:, :-1], inside[:, 1:])
    sidx = off[:-1, None] + lengths[:, None] + np.arange(k)[None]
    add(i[:, :, None], sidx[:, None, :], b["E"], inside[:, :, None])
    add(sidx[:, :, None], i[:, None, :], b["F"], inside[:, None, :])
    add(sidx[:, :, None], sidx[:, None, :], b["G"])
    n0 = off[-1]
    gI, gJ = n0 + b["I"], n0 + b["J"]
    add(i[:, :, None], gJ[None, None, :], b["Eg"], inside[:, :, None])
    add(sidx[:, :, None], gJ[None, None, :], b["Gg"])
    add(gI[None, :, None], i[:, None, :], b["Hc"], inside[:, None, :])
    add(gI[None, :, None], sidx[:, None, :], b["Hl"])
    gg = n0 + np.arange(g)
    add(gg[:, None], gg[None, :], b["D"])
    if links is not None:
        (lp, lq, lv) = links
        drop = sidx[:, 2]
        add(drop[lp], drop[lq], lv)
    n = int(n0 + g)
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    A.sum_duplicates()
    members = [(off[p] + np.arange(lengths[p]))[:, None] for p in range(P)]
    scalars = [sidx[p] for p in range(P)]
    return A, dict(members=members, scalars=scalars, n=n, globals=gg)


def make_links(b, kind, seed=2, strength=0.5):
    """Drop-drop links: neighbours (chain) or a random sparse graph of mean degree 4 (network).

    Each drop row gets off-diagonal links summing to ``strength`` times its own diagonal, of the diagonal's sign.
    """
    P = b["d"].shape[0]
    if kind == "chain":
        p = np.arange(P - 1)
        pairs = np.stack([p, p + 1], 1)
    else:
        rng = np.random.default_rng(seed)
        a = np.repeat(np.arange(P), 2)
        q = rng.integers(0, P, size=a.size)
        pairs = np.unique(np.sort(np.stack([a, q], 1), 1), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    lp = np.concatenate([pairs[:, 0], pairs[:, 1]])
    lq = np.concatenate([pairs[:, 1], pairs[:, 0]])
    deg = np.bincount(lp, minlength=P)
    diag = b["G"][:, 2, 2]
    lv = strength * diag[lp] / deg[lp]
    return lp, lq, lv


def build_case(blocks_dir, P, variant, cache={}):
    """(A, layout, info) of one case: P particles of 100 points, one variant."""
    parts = variant.split("+")
    base_tag = "240" if P == 240 else "720"
    if base_tag not in cache:
        cache[base_tag] = load_blocks(blocks_dir, base_tag)
    b = grow(cache[base_tag], P)
    N = b["d"].shape[1]
    lengths = np.full(P, N)
    links = None
    w = None
    for part in parts:
        if part == "base":
            pass
        elif part == "ragged":
            b, lengths = ragged(b)
        elif part in ("chain", "network"):
            links = make_links(b, part)
        elif part.startswith("cplx"):
            w = float(part[4:])
        else:
            raise SystemExit(f"unknown variant part {part!r}")
    A, lay = assemble(b, lengths, links)
    A_unlinked = assemble(b, lengths)[0] if links is not None else None
    if w is not None:
        cplx = lambda M: (M + 1j * w * sp.diags(M.diagonal())).tocsr()  # noqa: E731
        A = cplx(A)
        A_unlinked = None if A_unlinked is None else cplx(A_unlinked)
    lay["unlinked"] = A_unlinked
    info = dict(particles=P, nodes=N, variant=variant, n=lay["n"], nnz=int(A.nnz), min_len=int(lengths.min()),
                max_len=int(lengths.max()), n_cut=int((lengths < N).sum()), links=0 if links is None else len(links[0]),
                dtype=str(A.dtype))
    return A, lay, info


# ----------------------------------------------------------------------------------------------- backends


def on_gpu():
    return jax.default_backend() == "gpu"


def make_backend(name, lay, variant):
    """The backend object for ``name`` and the case's layout (the layout from the maps only)."""
    linked = any(p in ("chain", "network") for p in variant.split("+"))
    if name == "klu":
        return LIN.KLU()
    if name == "cudss":
        return LIN.CuDSS()
    if name == "member_block":
        layout = LIN.MemberBlockLayout(lay["members"], n_dofs=lay["n"])
        return LIN.MemberBlock(layout)
    if name.startswith("member_schur"):
        elim = name.split(":")[1] if ":" in name else "thomas"
        layout = LIN.MemberSchurLayout(lay["members"], scalars=lay["scalars"], n_dofs=lay["n"])
        if linked:
            shared = LIN.CuDSS() if on_gpu() else LIN.KLU()
        else:
            shared = LIN.Dense()
        return LIN.MemberSchur(layout, shared=shared, elimination=elim)
    raise SystemExit(f"unknown backend {name!r}")


def coo(A):
    A = A.tocoo()
    order = np.lexsort((A.col, A.row))
    return A.row[order].astype(np.int32), A.col[order].astype(np.int32), A.data[order]


class Solver:
    """analyze / factor / refactor / solve / transposed solve of one backend on one pattern, jitted."""

    def __init__(self, backend, rows, cols, n):
        self.topo = LIN.SparseTopology(rows=rows, cols=cols, n_dofs=n)
        self.backend = backend
        self.ctx = LIN.SolveContext()
        t0 = time.perf_counter()
        self.an = LIN.analyze(self.topo, backend=backend)
        self.analyze_s = time.perf_counter() - t0
        ctx = self.ctx
        an = self.an
        self.f_factor = jax.jit(lambda v: LIN.factor(an, v, ctx=ctx))
        self.f_refactor = jax.jit(lambda f, v: LIN.refactor(f, v, jnp.asarray(True)))
        self.f_solve = jax.jit(lambda f, r: LIN.solve(f, r)[0])
        self.f_solve_t = jax.jit(lambda f, r: LIN.solve(f, r, transpose=True)[0])
        self.f_checked = jax.jit(lambda f, r: LIN.solve_checked(f, r)[2].status)
        self.f_checked_t = jax.jit(lambda f, r: LIN.solve_checked(f, r, transpose=True)[2].status)


def berr(A, x, b):
    """Componentwise backward error max_i |b - A x|_i / (|A| |x| + |b|)_i."""
    r = np.abs(b - A @ x)
    den = abs(A) @ np.abs(x) + np.abs(b)
    return float(np.max(r / np.where(den > 0, den, 1.0)))


def rel(x, y):
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


def timeit(fn, reps):
    jax.block_until_ready(fn())
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e3, float(np.percentile(t, 10)) * 1e3, float(np.percentile(t, 90)) * 1e3


def run_case(name, A, lay, info, reps, do_time, ref_name):
    rows, cols, vals = coo(A)
    n = lay["n"]
    rng = np.random.default_rng(7)
    dt = A.dtype
    b = rng.standard_normal(n).astype(dt)
    c = rng.standard_normal(n).astype(dt)
    if np.iscomplexobj(b):
        b = b + 1j * rng.standard_normal(n); c = c + 1j * rng.standard_normal(n)
    out = dict(info, backend=name, reference=ref_name)
    V, bj, cj = jnp.asarray(vals), jnp.asarray(b), jnp.asarray(c)

    # Reference first (on the same device): KLU on the processor, CuDSS on the card.
    ref = Solver(make_backend(ref_name, lay, info["variant"]), rows, cols, n)
    fr = ref.f_factor(V)
    xr = np.asarray(ref.f_solve(fr, bj)); yr = np.asarray(ref.f_solve_t(fr, cj))
    out["ref_berr"] = berr(A, xr, b); out["ref_berr_T"] = berr(A.T.tocsr(), yr, c)
    del fr
    if lay.get("unlinked") is not None:
        # How much the links matter: the same right-hand side on the matrix without them.
        r0, c0, v0 = coo(lay["unlinked"])
        ref0 = Solver(make_backend(ref_name, lay, "base"), r0, c0, n)
        x0 = np.asarray(ref0.f_solve(ref0.f_factor(jnp.asarray(v0)), bj))
        out["link_effect"] = rel(x0, xr)

    try:
        s = Solver(make_backend(name, lay, info["variant"]), rows, cols, n)
        t0 = time.perf_counter()
        f = jax.block_until_ready(s.f_factor(V))
        out["first_factor_s"] = time.perf_counter() - t0
        x = np.asarray(s.f_solve(f, bj)); y = np.asarray(s.f_solve_t(f, cj))
        out["status"] = int(s.f_checked(f, bj)); out["status_T"] = int(s.f_checked_t(f, cj))
    except Exception as exc:  # recorded, the run goes on
        out["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        return out
    out["analyze_ms"] = s.analyze_s * 1e3
    out["relerr"] = rel(x, xr); out["relerr_T"] = rel(y, yr)
    out["berr"] = berr(A, x, b); out["berr_T"] = berr(A.T.tocsr(), y, c)
    out["finite"] = bool(np.isfinite(x).all() and np.isfinite(y).all())
    if do_time:
        # Analyse: cold calls on fresh topologies (analyze is memoised by content, so each repeat gets a new
        # SparseTopology object holding the same pattern and the memo is bypassed by a fresh backend object).
        ta = []
        for _ in range(min(reps, 5)):
            topo = LIN.SparseTopology(rows=rows.copy(), cols=cols.copy(), n_dofs=n)
            LIN._api._clear_analyses()
            t0 = time.perf_counter(); LIN.analyze(topo, backend=make_backend(name, lay, info["variant"]))
            ta.append(time.perf_counter() - t0)
        out["analyze_ms"] = float(np.median(ta)) * 1e3
        out["factor_ms"] = timeit(lambda: s.f_factor(V), reps)
        out["refactor_ms"] = timeit(lambda: s.f_refactor(f, V), reps)
        out["solve_ms"] = timeit(lambda: s.f_solve(f, bj), reps)
        out["solve_T_ms"] = timeit(lambda: s.f_solve_t(f, cj), reps)
    return out


def main_run(args):
    ref_name = "cudss" if on_gpu() else "klu"
    os.makedirs(args.out, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    variants = args.variants.split(",")
    dev = "gpu" if on_gpu() else "cpu"
    meta = dict(device=str(jax.devices()[0]), host=socket.gethostname(), argv=sys.argv,
                jax=jax.__version__, xla_flags=os.environ.get("XLA_FLAGS", ""),
                affinity=len(os.sched_getaffinity(0)))
    tag = args.backend.replace(":", "-")
    path = f"{args.out}/{dev}_{tag}_{args.label or '-'.join(map(str, sizes))}.json"
    results = []
    for P in sizes:
        for v in variants:
            A, lay, info = build_case(args.blocks, P, v)
            t0 = time.perf_counter()
            r = run_case(args.backend, A, lay, info, args.reps, args.time, ref_name)
            r.update(meta, wall_s=time.perf_counter() - t0, device_kind=dev)
            print(json.dumps({k: r[k] for k in r if k not in ("argv",)}), flush=True)
            results.append(r)
            json.dump(results, open(path, "w"), indent=1)
            jax.clear_caches()
    print("wrote", path)


# ----------------------------------------------------------------------------------------------- tables


def fmt(x):
    if x is None:
        return "–"
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, float):
        return f"{x:.1e}" if (x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e4)) else f"{x:.3g}"
    return str(x)


def main_table(paths):
    rows = []
    for p in paths:
        rows += json.load(open(p))
    print("| device | backend | particles | variant | n | relerr x | relerr x^T | berr x | berr x^T | ref berr x, x^T | link effect | status |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        st = r.get("error") or f"{r.get('status')}/{r.get('status_T')}"
        print(f"| {r['device_kind']} | {r['backend']} | {r['particles']} | {r['variant']} | {r['n']} | "
              f"{fmt(r.get('relerr'))} | {fmt(r.get('relerr_T'))} | {fmt(r.get('berr'))} | {fmt(r.get('berr_T'))} | "
              f"{fmt(r.get('ref_berr'))}, {fmt(r.get('ref_berr_T'))} | {fmt(r.get('link_effect'))} | {st} |")
    print()
    print("Times in ms, warm, median of the stated repeats (analyse: median of 5 cold calls).")
    print()
    print("| device | backend | particles | variant | analyse | factor | refactor | solve | solve^T |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if "factor_ms" not in r:
            continue
        print(f"| {r['device_kind']} | {r['backend']} | {r['particles']} | {r['variant']} | {fmt(r['analyze_ms'])} | "
              f"{fmt(r['factor_ms'])} | {fmt(r['refactor_ms'])} | {fmt(r['solve_ms'])} | {fmt(r['solve_T_ms'])} |")


def main_pivot(paths, variants=("base", "ragged", "cplx1", "chain", "network", "network+cplx1")):
    """Factor and solve times side by side, one row per device, size and variant, one column per backend."""
    rows = []
    for p in paths:
        rows += [r for r in json.load(open(p)) if "factor_ms" in r]
    backends = sorted({r["backend"] for r in rows}, key=lambda b: (b not in ("klu", "cudss"), b))
    key = lambda r: (r["device_kind"], r.get("host", "")[:8], r["particles"], r["variant"])  # noqa: E731
    table = {}
    for r in rows:
        table.setdefault(key(r), {})[r["backend"]] = r
    for what in ("factor_ms", "solve_ms", "solve_T_ms"):
        print(f"\n{what[:-3]} (ms, warm median)\n")
        print("| device | host | particles | variant | " + " | ".join(backends) + " |")
        print("|---|---|---:|---|" + "---:|" * len(backends))
        for k in sorted(table, key=lambda k: (k[0], k[1], k[2], variants.index(k[3]) if k[3] in variants else 99)):
            if k[3] not in variants:
                continue
            cells = [fmt(table[k][b][what]) if b in table[k] else "–" for b in backends]
            print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("measure"); p.add_argument("out")
    r = sub.add_parser("run")
    r.add_argument("--backend", required=True)
    r.add_argument("--sizes", default="240,720")
    r.add_argument("--variants", default="base,ragged,cplx0.1,cplx1,cplx10,chain,network,network+cplx1")
    r.add_argument("--time", action="store_true")
    r.add_argument("--reps", type=int, default=20)
    r.add_argument("--blocks", default=os.path.dirname(os.path.abspath(__file__)) + "/data")
    r.add_argument("--out", default=os.environ.get("OUT", "/drive2/research2/lfp-overhaul/store/B2"))
    r.add_argument("--label", default="")
    t = sub.add_parser("table"); t.add_argument("paths", nargs="+")
    t = sub.add_parser("pivot"); t.add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.measure, a.out)
    elif a.cmd == "run":
        main_run(a)
    elif a.cmd == "pivot":
        main_pivot(a.paths)
    else:
        main_table(a.paths)


if __name__ == "__main__":
    main()
