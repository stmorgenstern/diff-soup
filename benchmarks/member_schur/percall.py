"""Worker fable-solver: MemberSchur per call on a saved matrix (save_case.py) against KLU or cuDSS.

    python percall.py cases/mc50.npz --elimination native [--shared klu] [--cols 1,4] [--split] [--out r.json]

Warm medians in ms of factor, refactor, solve, transposed solve and multi-column solves; agreement with the
reference backend (KLU on the processor, cuDSS on the card); handle bytes; with --split the member pieces and the
count of top-level instructions (device kernels on a card) in each compiled piece; card memory by jax memory_stats.
"""
import argparse, json, os, re, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--elimination", default=None)
p.add_argument("--shared", default=None)
p.add_argument("--cols", default="1,4")
p.add_argument("--reps", type=int, default=20)
p.add_argument("--split", action="store_true")
p.add_argument("--no-ref", action="store_true")
p.add_argument("--complex", action="store_true")
p.add_argument("--out", default=None)
args = p.parse_args()

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diff_soup as ds
LIN = ds.linear
GPU = jax.devices()[0].platform == "gpu"
z = np.load(args.case)
rows, cols, n = z["rows"], z["cols"], int(z["n"])
lengths, b = z["lengths"], int(z["block"])
sites = z["sites"].reshape(-1, b)
cut = np.cumsum(lengths)[:-1]
members = np.split(sites, cut)
scalars = np.split(z["scalars"], np.cumsum(z["scalar_counts"])[:-1])
layout = LIN.MemberSchurLayout(members, scalars=scalars, n_dofs=int(z["layout_n"]))
bc = z["bc"]
topology = LIN.SparseTopology(rows=rows, cols=cols, n_dofs=n, bc_indices=bc if bc.size else None)
V = jnp.asarray(z["values"])
if args.complex:
    # the impedance system: the real matrix plus an imaginary shift on its diagonal
    V = V.astype(jnp.complex128) + 1j*jnp.asarray(np.where(rows == cols, 0.37*np.abs(z["values"]), 0.0))
elim = args.elimination or ("cyclic" if GPU else "native")
shared_name = args.shared or ("cudss" if GPU else "klu")
shared = {"klu": LIN.KLU, "dense": LIN.Dense, "cudss": LIN.CuDSS}[shared_name]()
ms = LIN.MemberSchur(layout, shared=shared, elimination=elim)
S, n_max = int(lengths.sum()), int(lengths.max())
res = dict(case=os.path.basename(args.case), device=str(jax.devices()[0]), elimination=elim, shared=shared_name,
           n=n, nnz=int(rows.size), members=int(lengths.size), sites=S, n_max=n_max, block=b,
           padded_fraction=float(1 - S/(lengths.size*n_max)), complex=bool(args.complex),
           uptime=os.popen("uptime").read().strip(), omp=os.environ.get("OMP_NUM_THREADS"))
print(json.dumps(res), flush=True)


def timeit(fn, reps=None):
    jax.block_until_ready(fn())
    t = []
    for _ in range(reps or args.reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn()); t.append(time.perf_counter() - t0)
    return float(np.median(t))*1e3


def nbytes(tree):
    return int(sum(np.prod(x.shape)*x.dtype.itemsize for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "shape")))


def card():
    if not GPU:
        return {}
    s = jax.devices()[0].memory_stats() or {}
    return {k: round(int(s[k])/2**20, 1) for k in ("bytes_in_use", "peak_bytes_in_use", "pool_bytes", "peak_pool_bytes") if k in s}


def kernels(fn, *a):
    try:
        text = jax.jit(fn).lower(*a).compile().as_text()
    except Exception as exc:  # noqa: BLE001
        return f"n/a ({type(exc).__name__})"
    entry = text[text.rfind("ENTRY"):]
    skip = ("parameter(", "constant(", " tuple(", "get-tuple-element(", "bitcast(")
    kinds = {}
    for ln in entry.splitlines():
        if " = " not in ln or any(s in ln for s in skip):
            continue
        m = re.search(r"= \S+ ([\w\-]+)\(", ln)
        kind = m.group(1) if m else "?"
        kinds[kind] = kinds.get(kind, 0) + 1
    return {"total": sum(kinds.values()), **kinds}


rng = np.random.default_rng(7)
col_counts = [int(c) for c in args.cols.split(",")]
rhs = {m: jnp.asarray(rng.standard_normal((m, n) if m > 1 else (n,))).astype(V.dtype) for m in col_counts}
backends = [("ms", ms)] + ([] if args.no_ref else [("ref", LIN.CuDSS() if GPU else LIN.KLU())])
out, xs = {}, {}
for name, backend in backends:
    ctx = LIN.SolveContext()
    t0 = time.perf_counter()
    an = LIN.analyze(topology, backend=backend)
    d = dict(analyze_ms=(time.perf_counter() - t0)*1e3, card_start=card())
    fac = jax.jit(lambda v: LIN.factor(an, v, ctx=ctx))
    ref_ = jax.jit(lambda f, v: LIN.refactor(f, v, jnp.asarray(True)))
    sol = jax.jit(lambda f, r: LIN.solve(f, r)[0])
    solt = jax.jit(lambda f, r: LIN.solve(f, r, transpose=True)[0])
    t0 = time.perf_counter()
    f = jax.block_until_ready(fac(V))
    d["first_factor_s"] = time.perf_counter() - t0
    d["status"] = int(np.asarray(f.status)) if hasattr(f, "status") else None
    d["handle_mb"] = nbytes(f)/2**20
    d["card_after_factor"] = card()
    d["factor_ms"] = timeit(lambda: fac(V))
    d["refactor_ms"] = timeit(lambda: ref_(f, V))
    xs[name] = {}
    for m in col_counts:
        r = rhs[m]
        xs[name][m] = (np.asarray(sol(f, r)), np.asarray(solt(f, r)))
        d[f"solve{m}_ms"] = timeit(lambda: sol(f, r))
        d[f"solveT{m}_ms"] = timeit(lambda: solt(f, r))
    d["card_end"] = card()
    out[name] = d
    print(name, type(backend).__name__, json.dumps(d), flush=True)
    del f
if not args.no_ref:
    rel = lambda a, r: float(np.linalg.norm(a - r)/np.linalg.norm(r))
    out["relerr"] = {m: [rel(xs["ms"][m][i], xs["ref"][m][i]) for i in range(2)] for m in col_counts}
    print("relerr (solve, transposed) per column count", out["relerr"], flush=True)

if args.split:
    from diff_soup.linear._structured import _member_elimination as me
    an = LIN.analyze(topology, backend=ms)
    plan = an.backend_analysis.core
    split = dict(nm=plan.nm, b=plan.b, k=plan.k, kc=plan.kc, n_shared=plan.n_shared, nnz_shared=plan.nnz_shared)
    elim_fn = lambda v: me.eliminate(plan, v)
    je = jax.jit(elim_fn)
    split["eliminate_ms"] = timeit(lambda: je(V))
    split["eliminate_kernels"] = kernels(elim_fn, V)
    tree, shared_vals, st = je(V)
    split["tree_mb"] = nbytes(tree)/2**20
    split["tree_leaves_mb"] = {k: round(nbytes(v)/2**20, 2) for k, v in tree._asdict().items() if v is not None} if hasattr(tree, "_asdict") else {}
    if plan.n_shared:
        san = an.backend_analysis.shared
        sctx = LIN.SolveContext()
        sf = jax.jit(lambda v: LIN.factor(san, v, ctx=sctx))
        split["shared_factor_ms"] = timeit(lambda: sf(shared_vals))
        fs = sf(shared_vals)
        ss = jax.jit(lambda f, r: LIN.solve(f, r)[0])
        rs = jnp.ones((plan.n_shared,), V.dtype)
        split["shared_solve_ms"] = timeit(lambda: ss(fs, rs))
    for m in col_counts:
        columns = (rhs[m].T if m > 1 else rhs[m][:, None])
        for tr in (False, True):
            red_fn = lambda t, c: me.reduce_rhs(plan, V, t, c, tr)
            red = jax.jit(red_fn)
            carry, rhs_s = red(tree, columns)
            back_fn = lambda t, c, x: me.back_substitute(plan, V, t, c, x, tr)
            back = jax.jit(back_fn)
            tag = f"m{m}{'T' if tr else ''}"
            split[f"reduce_{tag}_ms"] = timeit(lambda: red(tree, columns))
            split[f"back_{tag}_ms"] = timeit(lambda: back(tree, carry, rhs_s))
            if m == col_counts[0]:
                split[f"reduce_{tag}_kernels"] = kernels(red_fn, tree, columns)
                split[f"back_{tag}_kernels"] = kernels(back_fn, tree, carry, rhs_s)
    split["card_end"] = card()
    out["split"] = split
    print("split", json.dumps(split), flush=True)
res.update(out)
if args.out:
    with open(args.out, "w") as fh:
        json.dump(res, fh)
print("done", flush=True)
