"""One fixed-state local quadrature study. No equilibrium, AD, or Abaqus job."""
from pathlib import Path
import sys,json,time,hashlib,argparse
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import numpy as np,basix
from scipy.special import expit
from jax_fem.basis import get_elements
from surface_distance import PeriodicSurfaceDistance
L=10.;N=32;T=.5;ETA=1e-4;ELL=.05/(2*np.log(9));E=10.;NU=.3
MU=E/(2*(1+NU));LAM=E*NU/((1+NU)*(1-2*NU));BUDGET=900.;BATCH=32
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def freeze():
    for n,h in json.loads((D/'frozen_before.json').read_text()).items():assert sha(R/n)==h,n
def geometry():
    c=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
    surface=PeriodicSurfaceDistance(c['surface_vertices'],c['surface_triangles'])
    return c,surface
def prepare():
    started=time.perf_counter();freeze();c,surface=geometry()
    m=np.load(R/'validation/initial_bias_mechanism_20261008_r32/local_energy.npz')['topmask']
    _,faces,_=surface.query(c['physical_quad_points'],details=True)
    selected=np.flatnonzero(np.any(m[faces]&(c['rho']>=.01),axis=1))
    assert len(selected)>0
    np.save(D/'selected_cells.npy',selected)
    cfg={'case':'diverse_04 r30 saved initial static equilibrium','N':N,'cell_size_mm':L,
      'thickness_mm':T,'E_MPa':E,'nu':NU,'eta':ETA,'interface_10_90_mm':.05,
      'selection':'Cell has >=1 original Gauss point with phi>=0.01 and nearest face in r32 fixed top20pct-area mask',
      'selected_cells':len(selected),'total_cells':N**3,'selected_cells_sha256':sha(D/'selected_cells.npy'),
      'dense_points_per_axis':[8,12],'dense_points_per_selected_cell':[512,1728],
      'budget_seconds':BUDGET,'batch_cells':BATCH,
      'gates':{'original_occupancy_max_abs':1e-10,'original_total_energy_relative':1e-10,
        'original_selected_energy_relative':1e-10,'constant_floor_energy_relative':1e-10,
        'dense_levels_relative_energy_volume_distance_moment':.01},
      'denominators':'original selected 27-point values for dense changes; 12-point-axis selected values for 8-vs12 stability; original full r30 energy for contribution scale only',
      'state_fixed':True,'outside_selection_densely_checked':False,'production_changed':False,
      'no_equilibrium_design_AD_or_Abaqus_job':True,'selection_preparation_seconds':time.perf_counter()-started}
    write('protocol.json',cfg);print(json.dumps(cfg,indent=2),flush=True)
def run():
    start=time.perf_counter();freeze();cfg=json.loads((D/'protocol.json').read_text());sel=np.load(D/'selected_cells.npy')
    assert sha(D/'selected_cells.npy')==cfg['selected_cells_sha256']
    c,surface=geometry();qp=c['physical_quad_points'];phi0=c['rho'];w0=c['JxW']
    state=np.load(R/'validation/initial_tangent_20261008_r30/linear_state.npz');q=state['q'];ci=state['class_ids'];H=state['H']
    family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
    local=np.rint(2*el.points[order]).astype(int);origins=2*np.indices((N,)*3).reshape(3,-1).T
    idx=origins[:,None,:]+local[None,:,:];cells=(idx[:,:,0]*65+idx[:,:,1])*65+idx[:,:,2]
    uc=q[ci[cells]];physical_origins=origins/64
    def grads(ref):return el.tabulate(1,ref)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*N
    def unit_energy(which,g):
        grad=H+np.einsum('cni,qnj->cqij',uc[which],g,optimize=True)
        ep=.5*(grad+grad.swapaxes(-1,-2))
        return .5*LAM*np.trace(ep,axis1=-2,axis2=-1)**2+MU*np.sum(ep*ep,axis=(-1,-2))
    def values(phi,d,w,unit):
        return np.stack([np.sum(phi*w,axis=1),np.sum(phi*w*(d*L)**2,axis=1),
          np.sum(ETA*unit*w,axis=1),np.sum((1-ETA)*phi*unit*w,axis=1)],axis=1)
    # Baseline is computed from the original cache before dense sampling.
    g0=grads(qp[0]*N);full_unit=unit_energy(np.arange(N**3),g0)
    full_energy=np.sum((ETA+(1-ETA)*phi0)*full_unit*w0)*L**3
    expected=json.loads((R/'validation/initial_tangent_20261008_r30/jax_result.json').read_text())['energy_N_mm']
    selected_qp=qp[sel];d0=surface.query(selected_qp)
    phi=expit((T/2-d0*L)/ELL);phierr=float(np.max(abs(phi-phi0[sel])))
    physical_w=w0[sel]*L**3
    original=values(phi0[sel],c['distance'][sel],physical_w,full_unit[sel])
    reproduction=values(phi,d0,physical_w,full_unit[sel]);baseline_error=abs(reproduction[:,2:].sum()/original[:,2:].sum()-1)
    validation={'original_occupancy_max_absolute_error':phierr,
      'original_full_energy_relative_error':float(abs(full_energy/expected-1)),
      'original_selected_energy_relative_error':float(baseline_error)}
    checks={'original_occupancy':phierr<=1e-10,'original_full_energy':bool(abs(full_energy/expected-1)<=1e-10),
      'original_selected_energy':bool(baseline_error<=1e-10)}
    write('baseline_validation.json',{'checks':checks,**validation,'full_original_energy_N_mm':float(full_energy),
      'selected_original_energy_fraction':float(original[:,2:].sum()/full_energy)})
    assert all(checks.values()),validation
    np.savez_compressed(D/'original_27point.npz',selected_cells=sel,per_cell=original,
        columns=np.array(['occupancy_volume_mm3','occupancy_distance_moment_mm5','floor_energy_N_mm','occupancy_energy_N_mm']))
    print(json.dumps({'stage':'baseline_valid','seconds':time.perf_counter()-start,'selected_cells':len(sel),
      'original_energy_fraction':float(original[:,2:].sum()/full_energy),'phi_error':phierr}),flush=True)
    stages=[];raw={}
    def summary(a):
        totals=a.sum(axis=0)
        return {'occupancy_volume_mm3':float(totals[0]),'occupancy_distance_moment_mm5':float(totals[1]),
          'floor_energy_N_mm':float(totals[2]),'occupancy_energy_N_mm':float(totals[3]),
          'total_energy_N_mm':float(totals[2]+totals[3]),
          'mean_distance_squared_mm2':float(totals[1]/totals[0])}
    for level in cfg['dense_points_per_axis']:
        started=time.perf_counter();z,wt=np.polynomial.legendre.leggauss(level);z=(z+1)/2;wt/=2
        ref=np.array([[a,b,c] for a in z for b in z for c in z]);weights=np.array([a*b*c for a in wt for b in wt for c in wt])*(L/N)**3
        g=grads(ref);out=np.zeros_like(original);done=0;last_print=started
        for pos in range(0,len(sel),BATCH):
            if time.perf_counter()-start>BUDGET:break
            which=sel[pos:pos+BATCH];points=physical_origins[which,None,:]+ref[None,:,:]/N
            d=surface.query(points);rho=expit((T/2-d*L)/ELL)
            out[pos:pos+len(which)]=values(rho,d,weights[None,:],unit_energy(which,g));done+=len(which)
            if time.perf_counter()-last_print>10:
                print(json.dumps({'stage':'dense','level':level,'completed_cells':done,'seconds':time.perf_counter()-start}),flush=True);last_print=time.perf_counter()
        complete=done==len(sel);raw[level]=out
        np.savez_compressed(D/f'dense_{level}.npz',selected_cells=sel,per_cell=out,completed_cells=done)
        row={'points_per_axis':level,'completed_cells':done,'complete':complete,'seconds':time.perf_counter()-started}
        if complete:row.update(summary(out))
        stages.append(row);write('progress.json',{'levels':stages,'seconds':time.perf_counter()-start})
        print(json.dumps(row),flush=True)
        if not complete:break
    base=summary(original);finished=len(stages)==2 and all(v['complete'] for v in stages)
    comparison={};stable=False
    if finished:
        dense8=summary(raw[8]);dense12=summary(raw[12])
        for key in ['total_energy_N_mm','occupancy_volume_mm3','occupancy_distance_moment_mm5']:
            comparison[key]={'dense8_over_original_minus_one':dense8[key]/base[key]-1,
              'dense12_over_original_minus_one':dense12[key]/base[key]-1,
              'dense8_over_dense12_minus_one':dense8[key]/dense12[key]-1}
        checks['constant_floor_energy_exact']=max(abs(dense8['floor_energy_N_mm']/base['floor_energy_N_mm']-1),
            abs(dense12['floor_energy_N_mm']/base['floor_energy_N_mm']-1))<=1e-10
        checks['dense_levels_stable']=all(abs(v['dense8_over_dense12_minus_one'])<=.01 for v in comparison.values())
        stable=checks['dense_levels_stable']
    checks['two_levels_completed']=finished;checks['budget']=time.perf_counter()-start<=BUDGET
    result={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,'case':cfg['case'],
      'state_fixed':True,'selected_cells':len(sel),'full_original_energy_N_mm':float(full_energy),
      'selected_original':base,'selected_original_energy_fraction':base['total_energy_N_mm']/float(full_energy),
      'levels':stages,'comparison':comparison,'local_dense_levels_stable':stable,
      'full_energy_scale_of_selected_reintegration_change':
          (stages[-1]['total_energy_N_mm']-base['total_energy_N_mm'])/float(full_energy) if finished else None,
      'outside_selected_cells_retained_original_rule_not_verified':True,
      'this_is_not_a_new_stiffness_force_or_equilibrium':True,'original_5pct_cause_certified':False,
      'no_FEM_Abaqus_AD_or_production_changes':True,'wall_seconds':time.perf_counter()-start}
    write('result.json',result);freeze();print(json.dumps(result,indent=2),flush=True)
    if result['status']!='ok':sys.exit(2)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
    (prepare if args.action=='prepare' else run)()
