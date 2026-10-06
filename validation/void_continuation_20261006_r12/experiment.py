"""One candidate: existing-reference curves, saved material domain and mode.

No solver implementation and no additional loading path. Undefined active NH
states are reported as null full energies, never as positive-J partial sums.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import basix
from scipy.special import expit
import jax
import jax.numpy as jnp
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=Path('/home/xuehu/projects/tpms_jax'))
parser.add_argument('--experiment',type=Path,default=Path('/home/xuehu/projects/tpms_jax/validation/void_continuation_20261006_r12/frozen_field'))
parser.add_argument('--case-path',type=Path,required=True)
parser.add_argument('--frozen',action='store_true')
a=parser.parse_args();R=a.repo.resolve();O=a.experiment.resolve();O.mkdir(exist_ok=False)
sys.path.insert(0,str(R))
from jax_fem.basis import get_elements
from hyperelastic_fem import (objective_void_energy,objective_void_first_piola,void_nh_weight,void_nh_cutoff,objective_void_requires_positive_J)
from surface_distance import PeriodicSurfaceDistance
P=a.case_path.resolve();Q=R/'validation/large_compression_20261005_r6/quadratic_candidate'
A=R/'validation/large_compression_20261005_r6/abaqus'
G=R/'validation/thin_target_20261004_r5/gauss_field.npz'
L,T,N,ETA=10.,.5,32,1e-4
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def curve(rows):
    aa=np.array([r['compression'] for r in rows]);ff=-np.array([r['Fz_N'] for r in rows])
    aa,idx=np.unique(aa,return_index=True)
    return aa,ff[idx]

def compare(rec,rows,cfg,complete):
    old=read(Q/'T0p004_compact/result.json');oldcfg=read(Q/('T0p008_compact' if cfg['load_time_seconds']==.008 else 'T0p004_compact')/'input.json')
    shell=read(A/'explicit_T0p040/shell.json');standard=read(A/'standard/shell.json')
    fixed=['N','cell_size_mm','thickness_mm','E_MPa','nu','eta','interface_10_90_mm','mechanical_periodic_axes',
        'load_time_seconds','Gauss_field_sha256','element_type','element_degree','Gauss_points_per_cell',
        'mass_lumping','solid_density_tonne_per_mm3','void_mass_floor','total_steps','dt_seconds']
    for key in fixed:assert cfg[key]==oldcfg[key],(key,cfg[key],oldcfg[key])
    assert cfg['material_model']=='objective_void'
    curves={'Original Q2 27-point':curve(old['path']),
        'Shell slow Explicit':curve(shell['force_path']),'C2 continued virtual Q2':curve(rows)}
    aa,ff=curves['C2 continued virtual Q2'];refpeak=float(curve(standard['force_path'])[1].max())
    grid=np.linspace(.001,.2,1000);grid=grid[grid<=aa[-1]+1e-12]
    reference=np.interp(grid,*curves['Shell slow Explicit'])
    candidate=np.interp(grid,aa,ff)
    metric={'candidate_completed_20':complete,'fixed_physical_and_time_inputs_checked':fixed,
        'changed_factor':'Objective virtual energy only; same occupancy, mass and time algorithm',
        'last_accepted_compression':rows[-1]['compression'],'curve_RMS_over_Standard_peak':float(np.sqrt(np.mean((candidate-reference)**2))/refpeak),
        'curve_RMS_denominator_N':refpeak,'common_range':[float(grid[0]),float(grid[-1])],
        'body_seconds':rec['body_seconds'],'rejected_blocks':len(rec['rejected_blocks']),
        'minimum_monitored_actual_J':min(r['J_min'] for r in rows),
        'minimum_monitored_NH_active_J':min(r['required_positive_J_min'] for r in rows if r['required_positive_J_min'] is not None),
        'maximum_monitored_negative_J_points':max(r['negative_J_points'] for r in rows),
        'monitored_invalid_material_points':max(r['invalid_material_points'] for r in rows),
        'old_fast_response':read(Q/'comparison.json'),
        'reference_quality_certified':False,'physical_replacement_certified':False,'gradient20_certified':False,
        'new_rate_check_completed':False,'trusted_default_adopted':False}
    if complete:
        times=np.array([r['time'] for r in rows]);compression=np.array([r['compression'] for r in rows])
        forces=np.array([r['Fz_N'] for r in rows]);u=np.array([r['energy_N_mm'] for r in rows]);ke=np.array([r['KE_N_mm'] for r in rows])
        hold=times>=1.05*cfg['load_time_seconds']-1e-12;assert hold.sum()>=2
        mean=float(np.mean(forces[hold]));work=float(np.sum(.5*(forces[1:]+forces[:-1])*(-L*np.diff(compression))))
        dt=np.r_[0.,np.diff(times)];loading=(compression>=.01)&(times<=cfg['load_time_seconds']+1e-12)
        ratio=ke/np.maximum(u,1e-30)
        metric.update(hold_mean_Fz_N=mean,hold_samples=int(hold.sum()),
            hold_min_max_Fz_N=[float(forces[hold].min()),float(forces[hold].max())],
            terminal_force_difference_vs_shell=abs(mean/shell['hold_mean_Fz_N']-1),
            terminal_force_relative_change_vs_original_fast=mean/read(Q/'comparison.json')['hold_mean_Fz_N']-1,
            input_work_N_mm=work,input_work_difference_vs_shell=abs(work/shell['macro_work_from_RF_U_N_mm']-1),
            energy_work_gap=abs(u[-1]+ke[-1]-u[0]-ke[0]-work)/abs(work),
            loading_time_fraction_KE_below_5pct=float(np.sum(dt[loading]*(ratio[loading]<=.05))/np.sum(dt[loading])),
            terminal_energy_N_mm=float(u[-1]),terminal_KE_over_U=float(ratio[-1]),
            peak_magnitude_N=float(ff.max()),peak_compression=float(aa[ff.argmax()]),
            seconds_per_step_with_monitoring=rec['seconds_per_step_with_monitoring'],peak_RSS_GiB=rec['peak_RSS_GiB'],
            initial_dt_seconds=rec['initial_dt_seconds'],terminal_dt_seconds=rec['terminal_dt_seconds'])
        metric['response_target_only_pass']=bool(metric['terminal_force_difference_vs_shell']<=.1
            and metric['curve_RMS_over_Standard_peak']<=.1 and metric['input_work_difference_vs_shell']<=.1)
        metric['response_at_compressions']=[{'compression':c,'candidate_force_magnitude_N':float(np.interp(c,aa,ff)),
            'old_fast_force_magnitude_N':float(np.interp(c,*curves['Original Q2 27-point'])),
            'shell_force_magnitude_N':float(np.interp(c,*curves['Shell slow Explicit']))} for c in [.01,.05,.1,.15,.2]]
    else:metric.update(failure=rec['message'],terminal_force_difference_vs_shell=None,input_work_difference_vs_shell=None,response_target_only_pass=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name,(x,y) in curves.items():axes[0].plot(x*100,y,label=name,lw=1.6)
    axes[0].set(xlabel='Compression (%)',ylabel='Reaction magnitude (N)',title='Same physical inputs, XYZ periodic')
    axes[0].legend(fontsize=7)
    axes[1].plot([r['compression']*100 for r in rows],[r['KE_over_U']*100 for r in rows]);axes[1].axhline(5,color='gray',ls='--')
    axes[1].set(xlabel='Compression (%)',ylabel='KE / reference energy (%)',ylim=(0,20),title='Candidate monitored path')
    axes[2].plot([r['compression']*100 for r in rows],[r['J_min'] for r in rows],label='All points')
    axes[2].plot([r['compression']*100 for r in rows],[r['required_positive_J_min'] for r in rows],label='NH-active points')
    axes[2].axhline(0,color='gray',ls='--');axes[2].legend(fontsize=8)
    axes[2].set(xlabel='Compression (%)',ylabel='Minimum actual detF',title='27-point monitoring; dense probe separate')
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(O/'response.png',dpi=170);plt.close(fig)
    np.savetxt(O/'curves.csv',np.column_stack([grid]+[np.interp(grid,*curves[n]) for n in curves]),
        delimiter=',',header='compression,'+','.join(curves),comments='')
    write(O/'comparison.json',metric)
    return metric

def probe(q,row):
    D=O/'domain_probe';D.mkdir()
    with np.load(G) as f:vertices=f['surface_vertices'];triangles=f['surface_triangles'];labels=f['node_ids']
    distance=PeriodicSurfaceDistance(vertices,triangles)
    with np.load(Q/'gauss_field.npz') as f:oldphi=f['rho'];oldqp=f['physical_quad_points'];oldwt=f['JxW']
    family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
    local=np.rint(2*el.points[order]).astype(int)
    Wfun=jax.jit(jax.vmap(objective_void_energy));Pfun=jax.jit(jax.vmap(objective_void_first_piola))
    weightfun=jax.jit(jax.vmap(objective_void_requires_positive_J));cutfun=jax.jit(jax.vmap(void_nh_cutoff));rules={}
    bins=[('deep_void_le_0p001',-.000001,.001000000000001),('blend_lt_0p01',.001000000000001,.01),
        ('NH_0p01_to_0p05',.01,.05),('transition_0p05_to_0p95',.05,.95),('solid_ge_0p95',.95,1.000001)]
    for qorder in [4,8]:
        xi,wtref=basix.make_quadrature(cell,qorder);vals=el.tabulate(1,xi).take(order,axis=2)
        shape=vals[0,:,:,0];grad=vals[1:4,:,:,0].transpose(1,2,0)*N;weights=wtref/N**3
        result={'points_per_cell':len(xi),'Vf':0.,'raw_J_min':1e20,'raw_nonpositive_points':0,
            'raw_nonpositive_cells':0,'raw_nonpositive_phi_max':None,'required_positive_J_min':1e20,
            'invalid_material_points':0,'candidate_energy_N_mm':0.,'candidate_internal_Fz_N':0.,'continued_NH_points':0,
            'domain_bins':{name:{'points':0,'raw_J_min':None,'raw_nonpositive_points':0,
                'reference_volume_mm3':0.,'candidate_energy_N_mm':0.,'candidate_internal_Fz_N':0.} for name,_,_ in bins}}
        for begin in range(0,N**3,512):
            indices=np.arange(begin,min(begin+512,N**3));origins=np.stack([indices//N**2,(indices//N)%N,indices%N],axis=1)
            full=2*origins[:,None,:]+local[None,:,:];nodes=full/(2*N);qp=np.einsum('qn,cnd->cqd',shape,nodes)
            if qorder==4:
                assert np.max(abs(qp-oldqp[indices]))<2e-15
                rho=oldphi[indices];wt=oldwt[indices]
            else:
                d=distance.query(qp);rho=expit((T/(2*L)-d)/(.05/(2*np.log(9))/L));wt=np.broadcast_to(weights,rho.shape)
            grid=full%(2*N);ids=(grid[...,0]*(2*N)+grid[...,1])*(2*N)+grid[...,2]
            F=np.eye(3)+np.diag([0.,0.,-row['compression']])+np.einsum('cni,qnj->cqij',q[ids],grad)
            J=np.linalg.det(F);assert np.isfinite(J).all();bad=J<=0
            active=np.asarray(weightfun(jnp.asarray(rho.ravel()))).reshape(rho.shape)>0
            cutoff=np.asarray(cutfun(jnp.asarray(rho.ravel()))).reshape(rho.shape)
            result['continued_NH_points']+=int(((rho>.001)&(rho<.01)&(J<cutoff)).sum())
            result['Vf']+=float((rho*wt).sum());result['raw_J_min']=min(result['raw_J_min'],float(J.min()))
            result['raw_nonpositive_points']+=int(bad.sum());result['raw_nonpositive_cells']+=int(np.any(bad,axis=1).sum())
            if bad.any():result['raw_nonpositive_phi_max']=max(result['raw_nonpositive_phi_max'] or 0.,float(rho[bad].max()))
            if active.any():result['required_positive_J_min']=min(result['required_positive_J_min'],float(J[active].min()))
            result['invalid_material_points']+=int((bad&active).sum())
            valid=not (bad&active).any()
            if valid:
                W=np.asarray(Wfun(jnp.asarray(F.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape(rho.shape)
                P=np.asarray(Pfun(jnp.asarray(F.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape((*rho.shape,3,3))
                valid=np.isfinite(W).all() and np.isfinite(P).all()
            if valid and result['candidate_energy_N_mm'] is not None:
                result['candidate_energy_N_mm']+=float((W*wt).sum())*L**3
                result['candidate_internal_Fz_N']+=float((P[:,:,2,2]*wt).sum())*L**2
            elif not valid:result['candidate_energy_N_mm']=result['candidate_internal_Fz_N']=None
            for name,lo,hi in bins:
                mask=(rho>=lo)&(rho<hi);b=result['domain_bins'][name]
                b['points']+=int(mask.sum());b['raw_nonpositive_points']+=int((bad&mask).sum())
                if mask.any():b['raw_J_min']=min(b['raw_J_min'] if b['raw_J_min'] is not None else 1e20,float(J[mask].min()))
                b['reference_volume_mm3']+=float(wt[mask].sum())*L**3
                if valid:
                    b['candidate_energy_N_mm']+=float((W*wt)[mask].sum())*L**3
                    b['candidate_internal_Fz_N']+=float((P[:,:,2,2]*wt)[mask].sum())*L**2
        assert sum(b['points'] for b in result['domain_bins'].values())==N**3*len(xi)
        result['candidate_full_energy_defined']=result['candidate_energy_N_mm'] is not None
        if not result['candidate_full_energy_defined']:
            for b in result['domain_bins'].values():b['candidate_energy_N_mm']=b['candidate_internal_Fz_N']=None
        if qorder==4:
            assert result['candidate_full_energy_defined']
            result['energy_recovery_relative_error']=abs(result['candidate_energy_N_mm']/row['energy_N_mm']-1)
            result['internal_force_recovery_relative_error']=abs(result['candidate_internal_Fz_N']/row['internal_macro_Fz_N']-1)
            assert result['energy_recovery_relative_error']<1e-10 and result['internal_force_recovery_relative_error']<1e-10
        result['new_solve']=False;rules[str(qorder)]=result;write(D/f'rule_{qorder}.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='domain_bins'},indent=2),flush=True)
    # Reference-area weighted periodic midsurface translation comparison.
    cellindex=np.minimum(np.floor(vertices*N).astype(int),N-1);xi=vertices*N-cellindex
    shapes=el.tabulate(0,xi)[0,:,:,0][:,order]
    grid=(2*cellindex[:,None,:]+local[None,:,:])%(2*N)
    ids=(grid[...,0]*(2*N)+grid[...,1])*(2*N)+grid[...,2]
    area=np.linalg.norm(np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]]),axis=1)/2
    nodalarea=np.bincount(triangles.ravel(),weights=np.repeat(area/3,3),minlength=len(vertices))
    def fluct(q):
        w=np.einsum('pn,pni->pi',shapes,q[ids])*L
        return w-np.average(w,weights=nodalarea,axis=0)
    nw=fluct(q)
    with np.load(Q/'T0p004_compact/field.npz') as f:ow=fluct(f['q'])
    mode={'interpolation':'installed HEX27 basis on original midsurface nodes, reference area weighted, translation gauge removed',
        'comparison_at_current_compression':row['compression'],'scope':'Saved motion similarity, not stress or self-contact certification'}
    if abs(row['compression']-.2)<1e-12:
        with np.load(A/'explicit_T0p040/shell_field.npz') as f:lookup={int(label):u for label,u in zip(f['labels'],f['u_mm'])}
        sw=np.array([lookup[int(label)] for label in labels])-vertices*np.array([0.,0.,-.2])*L
        sw-=np.average(sw,weights=nodalarea,axis=0)
        dot=lambda x,y:float(np.sum(nodalarea[:,None]*x*y))
        for name,ref in [('old_fast',ow),('shell',sw)]:
            mode[name]={'fluctuation_vector_cosine':dot(nw,ref)/np.sqrt(dot(nw,nw)*dot(ref,ref)),
                'area_weighted_relative_fluctuation_difference':np.sqrt(dot(nw-ref,nw-ref)/dot(ref,ref))}
    write(O/'mode.json',mode);write(D/'result.json',{'rules':rules,'scope':'Same saved state, not new paths; actual raw folds remain explicit.'})
    return rules,mode

def main():
    start=time.perf_counter();assert not (O/'analysis_manifest.json').exists()
    complete=(P/'result.json').exists();record=P/('result.json' if complete else 'failure.json')
    rec=read(record);rows=rec['path'] if complete else rec['accepted_path'];cfg=read(P/'input.json')
    state=P/('field.npz' if complete else 'last_valid_field.npz')
    with np.load(state) as f:q=f['q'];assert int(f['N'])==N and abs(float(f['time'])-rows[-1]['time'])<1e-12
    files=[record,state,P/'input.json',Q/'gauss_field.npz',Q/'T0p004_compact/result.json',Q/'T0p004_compact/field.npz',
        A/'explicit_T0p040/shell.json',A/'standard/shell.json',A/'explicit_T0p040/shell_field.npz',G,
        R/'hyperelastic_fem.py',R/'scripts/thin_target_explicit.py',R/'surface_distance.py',R/'pixi.lock']
    before={str(p.relative_to(R)):sha(p) for p in files}
    write(O/'analysis_manifest.json',{'input_sha256':before,'experiment_sha256':sha(Path(__file__)),
        'environment':{'python':sys.version,'jax':jax.__version__,'numpy':np.__version__,'basix':basix.__version__},
        'complete_path':complete,'frozen_case':a.frozen,'material_domain':'strict actual J>0 for phi>=.01; C2 virtual continuation below .01','new_Abaqus_jobs':0,'new_extra_paths':0,'new_full_AD_jobs':0})
    metric=({'complete_path_is_frozen_old_result':True,'new_paths':0,'candidate_material_checked_on_old_q':True} if a.frozen else compare(rec,rows,cfg,complete));print('Inputs checked',flush=True)
    rules,mode=probe(q,rows[-1])
    metric.update(dense_sampling_J_min=rules['8']['raw_J_min'],dense_sampling_nonpositive_points=rules['8']['raw_nonpositive_points'],
        dense_sampling_required_positive_J_min=rules['8']['required_positive_J_min'],dense_sampling_invalid_material_points=rules['8']['invalid_material_points'],
        dense_sampling_material_defined=rules['8']['candidate_full_energy_defined'],mode=mode,
        analysis_seconds=time.perf_counter()-start)
    write(O/'comparison.json',metric)
    if a.frozen:
        assert rules['8']['candidate_full_energy_defined'] and rules['8']['invalid_material_points']==0
        assert rules['8']['raw_nonpositive_points']==17749
    assert before=={str(p.relative_to(R)):sha(p) for p in files}
    print(json.dumps(metric,indent=2),flush=True)
if __name__=='__main__':main()
