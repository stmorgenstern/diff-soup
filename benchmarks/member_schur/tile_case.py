"""Worker fable-solver: a larger cell matrix made of copies of the members of a saved one (no batjax needed).

    python tile_case.py cases/mc50.npz --members 2000 --out cases/t2000.npz

Each new member is a copy of a random member of the saved case: its own entries, the shared columns it reads and
the shared rows that read it (those scaled by old/new member count, so the shared Schur complement keeps its
size). The shared unknowns are kept once. The result has the structure and values of a one-volume cell with
--members particles whose lengths are drawn from the saved case's lengths. A measurement helper.
"""
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--members", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
args = p.parse_args()
z = np.load(args.case)
rows, cols, vals, n, b = z["rows"].astype(np.int64), z["cols"].astype(np.int64), z["values"], int(z["n"]), int(z["block"])
assert z["bc"].size == 0, "a case with Dirichlet rows is not tiled"
lengths, counts = z["lengths"], z["scalar_counts"]
nm = lengths.size
site_cut, scal_cut = np.cumsum(lengths)*b, np.cumsum(counts)
sites, scal = z["sites"].reshape(-1).astype(np.int64), z["scalars"].astype(np.int64)
owner = np.full(n, -1, np.int64)
lid = np.full(n, -1, np.int64)
size = lengths*b + counts
for m in range(nm):
    s = sites[(site_cut[m] - lengths[m]*b):site_cut[m]]
    c = scal[(scal_cut[m] - counts[m]):scal_cut[m]]
    owner[s], owner[c] = m, m
    lid[s] = np.arange(s.size)            # site-major, field inside, as the layout lists them
    lid[c] = s.size + np.arange(c.size)
shared = np.flatnonzero(owner < 0)
ns = shared.size
pos = np.full(n, -1, np.int64)
pos[shared] = np.arange(ns)
rng = np.random.default_rng(args.seed)
M = args.members
src = rng.integers(0, nm, M)
base = ns + np.cumsum(size[src]) - size[src]
ro, co = owner[rows], owner[cols]
assert not np.any((ro >= 0) & (co >= 0) & (ro != co)), "a case with links between members is not tiled"
R, C, V = [], [], []
k = np.flatnonzero((ro < 0) & (co < 0))
R.append(pos[rows[k]]); C.append(pos[cols[k]]); V.append(vals[k])
for m in range(nm):
    copies = base[src == m]
    if not copies.size:
        continue
    k = np.flatnonzero((ro == m) & (co == m))
    R.append((copies[:, None] + lid[rows[k]][None, :]).ravel()); C.append((copies[:, None] + lid[cols[k]][None, :]).ravel())
    V.append(np.tile(vals[k], copies.size))
    k = np.flatnonzero((ro == m) & (co < 0))
    R.append((copies[:, None] + lid[rows[k]][None, :]).ravel()); C.append(np.tile(pos[cols[k]], copies.size))
    V.append(np.tile(vals[k], copies.size))
    k = np.flatnonzero((ro < 0) & (co == m))
    R.append(np.tile(pos[rows[k]], copies.size)); C.append((copies[:, None] + lid[cols[k]][None, :]).ravel())
    V.append(np.tile(vals[k]*(nm/M), copies.size))
R, C, V = np.concatenate(R), np.concatenate(C), np.concatenate(V)
order = np.lexsort((C, R))
new_sites = np.concatenate([base[j] + np.arange(lengths[src[j]]*b) for j in range(M)])
new_scal = np.concatenate([base[j] + lengths[src[j]]*b + np.arange(counts[src[j]]) for j in range(M)])
n_new = int(ns + size[src].sum())
np.savez(args.out, rows=R[order].astype(np.int32), cols=C[order].astype(np.int32), n=n_new, values=V[order],
         bc=np.zeros((0,), np.int64), cj=z["cj"], sites=new_sites.astype(np.int32), lengths=lengths[src],
         scalars=new_scal.astype(np.int32), scalar_counts=counts[src], block=b, layout_n=n_new)
L = lengths[src]
print(f"saved {args.out}: members {M} unknowns {n_new} entries {R.size} sites {int(L.sum())} n_max {int(L.max())} "
      f"padding {1 - L.sum()/(M*L.max()):.0%}", flush=True)
