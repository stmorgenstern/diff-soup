"""cell_compare.py REF CAND (outputs of cell_check.py): max |dV| in mV on common valid save times, step counts, derivative relative differences."""
import sys, json
import numpy as np
r, c = sys.argv[1:3]
R, C = np.load(r + '.npz'), np.load(c + '.npz')
jr, jc = json.load(open(r + '.json')), json.load(open(c + '.json'))
ok = R['valid'] & C['valid']
out = dict(ref=jr['backend'], cand=jc['backend'], particles=jc['nvol']*jc['npart'], common=int(ok.sum()),
           only_ref=int((R['valid'] & ~C['valid']).sum()), only_cand=int((C['valid'] & ~R['valid']).sum()),
           max_dV_mV=float(np.max(np.abs(R['v'][ok]-C['v'][ok]))*1e3),
           steps_ref=jr['steps'], steps_cand=jc['steps'], singular_cand=jc.get('singular_total'),
           solve_s=(jr['solve_s'], jc['solve_s']))
for k in ('jvp', 'grad'):
    if k in jr and k in jc:
        out[k] = (jr[k], jc[k], abs(jc[k]-jr[k])/abs(jr[k]))
if 'jvp' in jc and 'grad' in jc:
    out['cand_jvp_vs_grad'] = abs(jc['jvp']-jc['grad'])/abs(jc['jvp'])
print(json.dumps(out, indent=1))
