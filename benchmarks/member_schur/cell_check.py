"""MemberSchur inside batjax's LFP half cell, against KLU or CuDSS (B2 cell check; not a test).

Usage, with batjax and this diff-soup on PYTHONPATH:
    python cell_check.py BACKEND NVOL NPART OUT [--sweeps K] [--noderiv] [--smoke]
BACKEND: klu | cudss | ms-dense | ms-klu | ms-cudss. Writes OUT.json and OUT.npz; compare with cell_compare.py.
"""
import sys, time, json, argparse
import numpy as np
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import diff_soup as ds
from diff_soup.linear._factor import SolveStatus
from models.lfp_halfcell.run import resized_config
from batjax.protocols import CC, Protocol, Voltage
from diff_soup.problem._ir._low_rank import member_schur_layout

ap = argparse.ArgumentParser()
ap.add_argument('backend'); ap.add_argument('nvol', type=int); ap.add_argument('npart', type=int); ap.add_argument('out')
ap.add_argument('--sweeps', type=int, default=0); ap.add_argument('--noderiv', action='store_true'); ap.add_argument('--smoke', action='store_true')
a = ap.parse_args()

built = resized_config(a.nvol, a.npart).build(solver='cudss' if a.backend in ('cudss', 'ms-cudss') else 'klu')
problem = built.problem
layout = member_schur_layout(problem)
if a.backend.startswith('ms-'):
    MS = ds.linear.MemberSchur
    shared = {'ms-dense': ds.linear.Dense, 'ms-klu': ds.linear.KLU, 'ms-cudss': ds.linear.CuDSS}[a.backend]()
    kw = dict(refinement_sweeps=a.sweeps) if a.sweeps else {}
    linear = MS(layout, shared=shared, **kw)
else:
    linear = None
if linear is not None:  # batjax's linear_backend refuses other backends; swap the stepper after build
    st = built.stepper
    built.stepper = ds.JIDA(rtol=st.rtol, atol=st.atol, linear=linear)
    built._integrators.clear()

# Count SINGULAR factorizations: wrap the factor hooks of the backend class in use.
singular = {'factor': 0, 'calls': 0}


def _count(status):
    s = np.asarray(status)
    singular['calls'] += int(s.size)
    singular['factor'] += int(np.sum(s == int(SolveStatus.SINGULAR)))


def wrap(cls):
    for name in ('factor_fun', 'refactor_fun'):
        orig = getattr(cls, name)
        if getattr(orig, '_wrapped', False):
            continue

        def make(orig):
            def f(self, *args, **kw):
                out = orig(self, *args, **kw)
                jax.debug.callback(_count, out[-2])
                return out
            f._wrapped = True
            return f
        setattr(cls, name, make(orig))


wrap(type(built.stepper.linear))

rec = dict(backend=a.backend, nvol=a.nvol, npart=a.npart, n=int(problem.state_size()), nnz=int(built.jacobian_nnz()),
           members=layout.n_members, device=str(jax.devices()[0]), sweeps=a.sweeps)
print(json.dumps(rec), flush=True)
dis = 60. if a.smoke else 3600/.5
times = np.linspace(0., dis, 1001)


def stats(sol, label):
    return [dict(step=label, accepts=int(r.stats.accepts), rejects=int(r.stats.rejects),
                 factorizations=int(r.stats.factorizations), newton_failures=int(r.stats.newton_failures),
                 t_end_s=float(r.terminal.t)*built.scales.t_ref) for r in sol.raw]


# Full discharge to batjax's test cutoff.
t0 = time.perf_counter()
sol = built.solve(Protocol(steps=[CC(crate=.5, duration=dis, until=Voltage(volts=2.5))]), save_times=times)
jax.block_until_ready(sol.voltage)
rec['solve_s'] = time.perf_counter()-t0
rec['steps'] = stats(sol, 'discharge')
rec['singular_discharge'] = dict(singular)
v, valid = [np.asarray(sol.voltage)], [np.asarray(sol.valid)]
# Full charge, a separate solve from the discharge's saved state at the last point at or above 2.9 V
# (charging from the 2.5 V cutoff state fails batjax's starting-state projection on KLU too).
raw = sol.raw[0]
ok = np.asarray(sol.valid) & (np.nan_to_num(np.asarray(sol.voltage)) >= 2.9)
i = int(np.nonzero(ok)[0][-1])
rec['charge_start'] = dict(t_s=float(times[i]), V=float(np.asarray(sol.voltage)[i]))
start = jnp.concatenate((raw.ys[i], raw.lambdas[i]))
try:
    t0 = time.perf_counter()
    ch = built.solve(Protocol(steps=[CC(crate=-.5, duration=dis, until=Voltage(volts=4.0))]), save_times=times,
                     initial_state=start)
    jax.block_until_ready(ch.voltage)
    rec['charge_s'] = time.perf_counter()-t0
    rec['steps'] += stats(ch, 'charge')
    v.append(np.asarray(ch.voltage)); valid.append(np.asarray(ch.valid))
except Exception as exc:  # recorded, not raised: the comparison of the discharge still stands
    rec['charge_error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
    v.append(np.full(times.size, np.nan)); valid.append(np.zeros(times.size, bool))
rec['singular'] = dict(singular)
print(json.dumps({k: rec.get(k) for k in ('solve_s', 'charge_s', 'charge_start', 'charge_error', 'steps', 'singular')}),
      flush=True)
np.savez(a.out + '.npz', v=np.concatenate(v), valid=np.concatenate(valid))
if not a.noderiv:
    short = Protocol(steps=[CC(crate=.5, duration=600.)])
    st_times = np.linspace(0., 600., 61)

    def cost(k, adjoint=None):
        theta = {'cathode.k0_scale': k}
        return jnp.sum(built.solve(short, theta=theta, save_times=st_times, adjoint=adjoint).voltage)
    t0 = time.perf_counter()
    _, fwd = jax.jvp(cost, (jnp.asarray(1.),), (jnp.asarray(1.),))
    rec['jvp'] = float(fwd)
    rec['jvp_s'] = time.perf_counter()-t0
    t0 = time.perf_counter()
    rec['grad'] = float(jax.grad(lambda k: cost(k, ds.CheckpointAdjoint()))(jnp.asarray(1.)))
    rec['grad_s'] = time.perf_counter()-t0
    print(json.dumps({k: rec[k] for k in ('jvp', 'grad', 'jvp_s', 'grad_s')}), flush=True)
rec['singular_total'] = dict(singular)
json.dump(rec, open(a.out + '.json', 'w'), indent=1)
print('DONE', flush=True)
