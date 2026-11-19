"""Worker fable-solver: is the member elimination backward stable on a cell's matrices? (measurement)

    python robust.py cases/own200.npz [--eliminations native,thomas,cyclic] [--factors 1,0.01,100]

For every saved state and every multiple of cj: solves A x = b and A^T x = b with KLU and with MemberSchur, and
prints the componentwise backward error  max_i |r_i| / (|A||x| + |b|)_i  over all rows and the worst over the rows
of one member, for both; the relative difference of the two solutions; and, from each member's dense block
A_m = [[T, E], [F, G]] (numpy): the condition numbers of T and A_m, the smallest pivot of the unpivoted sweep
relative to its row, and the spread of the row scales of the member's scalar rows.
"""
import argparse, json
import numpy as np
import scipy.sparse as sp

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--eliminations", default="native")
p.add_argument("--factors", default="1,0.01,100")
p.add_argument("--blocks", action="store_true", help="per-member dense block diagnostics (numpy)")
args = p.parse_args()
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diff_soup as ds
LIN = ds.linear

z = np.load(args.case)
rows, cols, n, b = z["rows"], z["cols"], int(z["n"]), int(z["block"])
lengths, counts = z["lengths"], z["scalar_counts"]
sites = np.split(z["sites"].reshape(-1, b), np.cumsum(lengths)[:-1])
scal = np.split(z["scalars"], np.cumsum(counts)[:-1])
layout = LIN.MemberSchurLayout(sites, scalars=scal, n_dofs=int(z["layout_n"]))
topology = LIN.SparseTopology(rows=rows, cols=cols, n_dofs=n)
V1 = z["values_all"] if "values_all" in z else z["values"][None]
V2 = z["values_all2"] if "values_all2" in z else None
owner = np.full(n, -1)
for m, (s, c) in enumerate(zip(sites, scal)):
    owner[s.reshape(-1)] = m
    owner[c] = m
rng = np.random.default_rng(3)
rhs = rng.standard_normal(n)
GPU = jax.devices()[0].platform == "gpu"
ref = LIN.CuDSS() if GPU else LIN.KLU()
shared = LIN.CuDSS() if GPU else LIN.KLU()
backends = {"ref": ref}
for el in args.eliminations.split(","):
    backends[el] = LIN.MemberSchur(layout, shared=shared, elimination=el)
solvers = {}
for name, backend in backends.items():
    an = LIN.analyze(topology, backend=backend)
    ctx = LIN.SolveContext()
    def run(v, r, an=an, ctx=ctx):
        f = LIN.factor(an, v, ctx=ctx)
        x, _, rep = LIN.solve_checked(f, r)
        xt, _, _ = LIN.solve_checked(f, r, transpose=True)
        return x, xt, rep.status

    solvers[name] = jax.jit(run)


def backward(A, x, r):
    res = np.abs(A @ x - r)
    den = np.abs(A) @ np.abs(x) + np.abs(r)
    w = np.where(den > 0, res/np.where(den > 0, den, 1), 0.0)
    worst = np.zeros(lengths.size)
    np.maximum.at(worst, owner[owner >= 0], w[owner >= 0])
    return float(w.max()), float(worst.max()), int(worst.argmax())


def blocks(A):
    out = dict(cond_T=[], cond_A=[], min_pivot=[], scalar_row_spread=[])
    for s, c in zip(sites, scal):
        u = np.concatenate([s.reshape(-1), c])
        Am = A[u][:, u].toarray()
        nb = s.size
        T = Am[:nb, :nb]
        out["cond_T"].append(np.linalg.cond(T))
        out["cond_A"].append(np.linalg.cond(Am))
        if b == 1:
            d, lo, up = np.diag(T).copy(), np.r_[0., np.diag(T, -1)], np.r_[np.diag(T, 1), 0.]
            scale = np.abs(lo) + np.abs(d) + np.abs(up)
            piv, cp, worst = 0.0, 0.0, np.inf
            for i in range(nb):
                piv = d[i] - lo[i]*cp
                worst = min(worst, abs(piv)/scale[i])
                cp = up[i]/piv
            out["min_pivot"].append(worst)
        if c.size:
            rs = np.abs(Am[nb:]).max(axis=1)
            out["scalar_row_spread"].append(rs.max()/rs.min())
    return {k: (float(np.max(v)) if k != "min_pivot" else float(np.min(v))) for k, v in out.items() if v}


factors = [float(f) for f in args.factors.split(",")]
for i in range(V1.shape[0]):
    for f in factors:
        if f != 1 and V2 is None:
            continue
        v = V1[i] if f == 1 else V1[i] + (f - 1)*(V2[i] - V1[i])
        A = sp.csr_matrix((v, (rows, cols)), shape=(n, n))
        line = dict(state=i, cj_factor=f)
        xs = {}
        for name, fn in solvers.items():
            x, xt, status = fn(jnp.asarray(v), jnp.asarray(rhs))
            x, xt = np.asarray(x), np.asarray(xt)
            xs[name] = (x, xt)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(xt))):
                line[name] = dict(status=int(np.asarray(status)), finite=False)
                continue
            be, bm, who = backward(A, x, rhs)
            bet, bmt, _ = backward(A.T.tocsr(), xt, rhs)
            line[name] = dict(status=int(np.asarray(status)), berr=be, berr_member=bm, worst_member=who, berr_T=bet,
                              berr_T_member=bmt)
        for name in solvers:
            if name != "ref" and np.all(np.isfinite(xs[name][0])):
                line[name]["rel_to_ref"] = float(np.linalg.norm(xs[name][0] - xs["ref"][0])/np.linalg.norm(xs["ref"][0]))
                line[name]["rel_to_ref_T"] = float(np.linalg.norm(xs[name][1] - xs["ref"][1])/np.linalg.norm(xs["ref"][1]))
        if args.blocks:
            line["blocks"] = blocks(A)
        line["cond_est"] = None
        print(json.dumps(line), flush=True)
print("done", flush=True)
