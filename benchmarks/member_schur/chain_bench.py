"""Worker fable-solver: which chain kernel for ragged members on the card (throwaway measurement).

    python chain_bench.py --members 2000 --mean 100 [--block 1] [--cols 1,6]

M members with lognormal lengths as ONE flat ragged chain of S sites. Compares, per factor and per solve:
  cr    the library's block cyclic reduction on the flat chain (what elimination="cyclic" runs today, minus padding)
  pcr   parallel cyclic reduction on the flat chain: ceil(log2(n_max)) levels, every level one pass over all sites
  gtsv  jax.lax.linalg.tridiagonal_solve on the flat chain (cuSPARSE gtsv2 on the card; b = 1 only; no stored factor)
Prints warm medians in ms, the count of top-level device instructions in the compiled module, and agreement.
"""
import argparse, re, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from diff_soup.linear._structured._block_tridiagonal import block_tridiagonal_factor, block_tridiagonal_solve
from diff_soup.linear._structured._small_lu import bmm, small_lu_factor, small_lu_solve

p = argparse.ArgumentParser()
p.add_argument("--members", type=int, default=2000)
p.add_argument("--mean", type=float, default=100.0)
p.add_argument("--sigma", type=float, default=0.35)
p.add_argument("--block", type=int, default=1)
p.add_argument("--cols", default="1,6")
p.add_argument("--reps", type=int, default=15)
p.add_argument("--skip", default="")
args = p.parse_args()
rng = np.random.default_rng(0)
M, b = args.members, args.block
lengths = np.maximum(3, np.round(rng.lognormal(np.log(args.mean), args.sigma, M))).astype(np.int64)
S, n_max = int(lengths.sum()), int(lengths.max())
off = np.cumsum(lengths) - lengths
start = np.repeat(off, lengths).astype(np.int32)
stop = np.repeat(off + lengths, lengths).astype(np.int32)
print(f"devices {jax.devices()} M {M} S {S} n_max {n_max} mean {lengths.mean():.1f} padded would be {M*n_max} "
      f"({1-S/(M*n_max):.0%} padding) b {b}", flush=True)
lower = rng.uniform(-1, 1, (S, b, b)); upper = rng.uniform(-1, 1, (S, b, b))
diag = rng.uniform(-1, 1, (S, b, b)) + (4.0 * b + 1) * np.eye(b)
lower[start == np.arange(S)] = 0.0
upper[stop - 1 == np.arange(S)] = 0.0
lower, diag, upper = (jnp.asarray(a) for a in (lower, diag, upper))


def inv_blocks(d):
    if b == 1:
        return 1.0 / d
    lu, perm, _ = small_lu_factor(d)
    return small_lu_solve(lu, perm, jnp.broadcast_to(jnp.eye(b, dtype=d.dtype), d.shape))


def neighbours(d):
    pidx = jnp.arange(S, dtype=jnp.int32)
    li, ri = pidx - d, pidx + d
    lv, rv = li >= start, ri < stop
    return jnp.where(lv, li, 0), jnp.where(rv, ri, 0), lv[:, None, None], rv[:, None, None]


def pcr_factor(lower, diag, upper):
    levels, d = [], 1
    while d < n_max:
        li, ri, lv, rv = neighbours(d)
        dinv = inv_blocks(diag)
        alpha = jnp.where(lv, bmm(lower, dinv[li]), 0.0)
        gamma = jnp.where(rv, bmm(upper, dinv[ri]), 0.0)
        diag, lower, upper = (diag - bmm(alpha, upper[li]) - bmm(gamma, lower[ri]),
                              -bmm(alpha, lower[li]), -bmm(gamma, upper[ri]))
        levels.append((alpha, gamma))
        d *= 2
    return tuple(levels), inv_blocks(diag)


def pcr_solve(factor, rhs, transpose=False):
    levels, dinv = factor
    if not transpose:
        r, d = rhs, 1
        for alpha, gamma in levels:
            li, ri, _, _ = neighbours(d)
            r = r - bmm(alpha, r[li]) - bmm(gamma, r[ri])
            d *= 2
        return bmm(dinv, r)
    z = bmm(jnp.swapaxes(dinv, -1, -2), rhs)
    d = 2 ** (len(levels) - 1) if levels else 1
    for alpha, gamma in reversed(levels):
        li, ri, lv, rv = neighbours(d)
        at, gt = jnp.swapaxes(alpha, -1, -2), jnp.swapaxes(gamma, -1, -2)
        z = z - jnp.where(rv, bmm(at[ri], z[ri]), 0.0) - jnp.where(lv, bmm(gt[li], z[li]), 0.0)
        d //= 2
    return z


def gtsv_solve(lower, diag, upper, rhs, transpose=False):
    dl, d, du = lower[:, 0, 0], diag[:, 0, 0], upper[:, 0, 0]
    if transpose:
        dl, du = jnp.concatenate([jnp.zeros((1,), d.dtype), du[:-1]]), jnp.concatenate([dl[1:], jnp.zeros((1,), d.dtype)])
    return jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, 0, :])[:, None, :]


def timeit(fn, *a):
    jax.block_until_ready(fn(*a))
    t = []
    for _ in range(args.reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*a)); t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e3


def kernels(fn, *a):
    try:
        text = jax.jit(fn).lower(*a).compile().as_text()
    except Exception as exc:  # noqa: BLE001
        return f"n/a ({type(exc).__name__})"
    entry = text[text.rfind("ENTRY"):]
    skip = ("parameter(", "constant(", "tuple(", "get-tuple-element(", "bitcast(", "ROOT tuple")
    lines = [ln for ln in entry.splitlines() if " = " in ln and not any(s in ln for s in skip)]
    kinds = {}
    for ln in lines:
        m = re.search(r"= \S+ (\S+?)\(", ln)
        kind = m.group(1) if m else "?"
        if kind == "fusion":
            k2 = re.search(r"kind=(k\w+)", ln)
            kind = "fusion:" + (k2.group(1) if k2 else "?")
        kinds[kind] = kinds.get(kind, 0) + 1
    return f"{len(lines)} {kinds}"


def dense_ref(rhs, transpose):
    # reference by the sequential scalar/block Thomas on the host (numpy), small cases only
    lo, dg, up, r = (np.asarray(a) for a in (lower, diag, upper, rhs))
    if transpose:
        lo, up = np.concatenate([np.zeros_like(up[:1]), np.swapaxes(up[:-1], -1, -2)]), np.concatenate([np.swapaxes(lo[1:], -1, -2), np.zeros_like(lo[:1])])
        dg = np.swapaxes(dg, -1, -2)
    n = dg.shape[0]
    cp = np.zeros_like(dg); g = np.zeros_like(r)
    for i in range(n):
        piv = dg[i] - (lo[i] @ cp[i-1] if i else 0)
        cp[i] = np.linalg.solve(piv, up[i])
        g[i] = np.linalg.solve(piv, r[i] - (lo[i] @ g[i-1] if i else 0))
    x = g.copy()
    for i in range(n - 2, -1, -1):
        x[i] = g[i] - cp[i] @ x[i+1]
    return x


cols = [int(c) for c in args.cols.split(",")]
skip = args.skip.split(",")
rel = lambda a, r: float(np.max(np.abs(np.asarray(a) - r)) / np.max(np.abs(r)))
out = {}
if "cr" not in skip:
    f = jax.jit(block_tridiagonal_factor)
    fac, _ = f(lower, diag, upper)
    print("cr  factor ms", round(timeit(f, lower, diag, upper), 3), "kernels", kernels(block_tridiagonal_factor, lower, diag, upper),
          "bytes", sum(x.nbytes for x in jax.tree_util.tree_leaves(fac)), flush=True)
    for m in cols:
        rhs = jnp.asarray(rng.standard_normal((S, b, m)))
        for tr in (False, True):
            s = jax.jit(lambda fa, r: block_tridiagonal_solve(fa, r, transpose=tr))
            ref = dense_ref(rhs, tr) if S <= 300000 and m == 1 else None
            print(f"cr  solve m={m} T={tr} ms", round(timeit(s, fac, rhs), 3), "kernels",
                  kernels(lambda fa, r: block_tridiagonal_solve(fa, r, transpose=tr), fac, rhs),
                  "relerr", None if ref is None else rel(s(fac, rhs), ref), flush=True)
    del fac
if "pcr" not in skip:
    f = jax.jit(pcr_factor)
    fac = f(lower, diag, upper)
    print("pcr factor ms", round(timeit(f, lower, diag, upper), 3), "kernels", kernels(pcr_factor, lower, diag, upper),
          "levels", len(fac[0]), "bytes", sum(x.nbytes for x in jax.tree_util.tree_leaves(fac)), flush=True)
    for m in cols:
        rhs = jnp.asarray(rng.standard_normal((S, b, m)))
        for tr in (False, True):
            s = jax.jit(lambda fa, r: pcr_solve(fa, r, tr))
            ref = dense_ref(rhs, tr) if S <= 300000 and m == 1 else None
            print(f"pcr solve m={m} T={tr} ms", round(timeit(s, fac, rhs), 3), "kernels",
                  kernels(lambda fa, r: pcr_solve(fa, r, tr), fac, rhs),
                  "relerr", None if ref is None else rel(s(fac, rhs), ref), flush=True)
    del fac
if "gtsv" not in skip and b == 1:
    for m in cols:
        rhs = jnp.asarray(rng.standard_normal((S, b, m)))
        for tr in (False, True):
            s = jax.jit(lambda lo, dg, up, r: gtsv_solve(lo, dg, up, r, tr))
            try:
                ref = dense_ref(rhs, tr) if S <= 300000 and m == 1 else None
                print(f"gtsv solve m={m} T={tr} ms", round(timeit(s, lower, diag, upper, rhs), 3),
                      "relerr", None if ref is None else rel(s(lower, diag, upper, rhs), ref), flush=True)
            except Exception as exc:  # noqa: BLE001
                print("gtsv failed:", type(exc).__name__, str(exc)[:300], flush=True)
                break
# ---- member sums of site rows (S, w) -> (M, w): scatter-add, a pairwise tree of gathers, a padded gather-reduce
member_of_site = np.repeat(np.arange(M), lengths).astype(np.int32)


def tree_maps():
    maps, cur = [], lengths.copy()
    while cur.max() > 1:
        nxt = (cur + 1) // 2
        o_cur, o_nxt = np.cumsum(cur) - cur, np.cumsum(nxt) - nxt
        mem = np.repeat(np.arange(M), nxt)
        j = np.arange(int(nxt.sum())) - np.repeat(o_nxt, nxt)
        first = o_cur[mem] + 2 * j
        has = 2 * j + 1 < cur[mem]
        maps.append((first.astype(np.int32), np.where(has, first + 1, 0).astype(np.int32), has))
        cur = nxt
    return maps


TREE = tree_maps()
pad_idx = np.where(np.arange(n_max)[None, :] < lengths[:, None], off[:, None] + np.arange(n_max)[None, :], -1).astype(np.int32)


def sum_scatter(x):
    return jax.ops.segment_sum(x, member_of_site, num_segments=M, indices_are_sorted=True)


def sum_tree(x, barrier=True):
    for first, second, has in TREE:
        x = x[first] + jnp.where(has[:, None], x[second], 0.0)
        if barrier:
            x = jax.lax.optimization_barrier(x)
    return x


def sum_padded(x):
    return jnp.sum(jnp.where((pad_idx >= 0)[:, :, None], x[jnp.maximum(pad_idx, 0)], 0.0), axis=1)


for w in (3, 18):
    x = jnp.asarray(rng.standard_normal((S, w)))
    fa = jnp.asarray(rng.standard_normal((S, w)))
    ref = np.asarray(sum_scatter(x * fa))
    for name, fn in (("scatter", sum_scatter), ("tree", sum_tree), ("tree-nobarrier", lambda v: sum_tree(v, False)),
                     ("padded", sum_padded)):
        # the product feeds the sum, as in F.g: lets the compiler fuse it into the reduction
        g = lambda a, c, fn=fn: fn(a * c)
        j = jax.jit(g)
        try:
            print(f"member-sum {name} w={w} ms", round(timeit(j, x, fa), 3), "kernels", kernels(g, x, fa),
                  "relerr", rel(j(x, fa), ref), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"member-sum {name} failed", type(exc).__name__, str(exc)[:200], flush=True)
if jax.devices()[0].platform == "gpu":
    print("memory_stats", {k: v for k, v in (jax.devices()[0].memory_stats() or {}).items() if "bytes" in k or "pool" in k}, flush=True)
print("done", flush=True)
