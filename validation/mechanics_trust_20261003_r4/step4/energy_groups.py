"""Postprocess saved displacements with existing FE gradients; zero solves."""
from pathlib import Path
import gc,json,hashlib,time,sys
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4'
sys.path.insert(0,str(P))
import jax
import jax.numpy as jnp
import numpy as np
from hyperelastic_fem import make_density_hyperelastic_problem,neo_hookean_energy
ledger=json.loads((O/'execution.json').read_text());result={};start=time.monotonic()
for entry in ledger:
    tag=entry['tag'];f=O/(tag+'.json');rows=json.loads(f.read_text())['rows'] if f.exists() else json.loads((O/(tag+'.progress.json')).read_text())
    p=make_density_hyperelastic_problem(entry['N']);rho=np.asarray(p.rho);weights=np.asarray(p.JxW)[:,0,:]
    with np.load(O/(tag+'.displacement.npz')) as saved:
        assert np.array_equal(saved['points'],np.asarray(p.fe.points))
        assert np.allclose(saved['compression'],[r['compression'] for r in rows],rtol=0,atol=1e-14)
        groups=[]
        for row,u in zip(rows,saved['u']):
            F=jnp.eye(3)+p.fe.sol_to_grad(jnp.asarray(u))
            W=np.asarray(jax.vmap(neo_hookean_energy)(F.reshape((-1,3,3))).reshape(rho.shape))
            weighted=(1e-4+(1-1e-4)*rho)*W*weights;total=float(weighted.sum())
            assert abs(total-row['energy'])<1e-11
            parts={name:float(weighted[mask].sum()) for name,mask in
                  [('solid_rho_ge095',rho>=.95),('void_rho_le005',rho<=.05),('transition', (rho>.05)&(rho<.95))]}
            assert abs(sum(parts.values())-total)<1e-12
            groups.append({'compression':row['compression'],'energy':total,'energy_by_region':parts,
                           'fractions':{k:v/total if total>1e-14 else None for k,v in parts.items()}})
    result[tag]={'rows':groups,'actual_reference_Gauss_mapping':True,'saved_displacement_sha256':hashlib.sha256((O/(tag+'.displacement.npz')).read_bytes()).hexdigest()}
    del p,F,W,rho,weights,weighted;gc.collect();jax.clear_caches()
metadata={'definition':'Group actual weighted W by reference Gauss occupancy. These are diagnostic regions, not exact binary solid/void boundaries.',
          'cases':result,'postprocessing_seconds':time.monotonic()-start,'FEM_solves':0}
(O/'energy_groups.json').write_text(json.dumps(metadata,indent=2)+'\n');(O/'energy_groups.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({tag:v['rows'][-1] for tag,v in result.items()},indent=2))
