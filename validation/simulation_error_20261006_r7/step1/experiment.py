"""Frozen-field diagnostics for accepted plan step 1; no mechanical solve.

Uses the installed HEX27 basis and shared material energy. The analytic stress
below is an independent recovery check, not an additional maintained FEM.
"""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import basix
from scipy.special import expit
from scipy.integrate import trapezoid
R = Path('/home/xuehu/projects/tpms_jax')
sys.path.insert(0, str(R))
from jax_fem.basis import get_elements
from hyperelastic_fem import neo_hookean_energy, first_piola, MU, KAPPA
from surface_distance import PeriodicSurfaceDistance
import jax
import jax.numpy as jnp

E = R/'validation/simulation_error_20261006_r7'
O = E/'step1'
Q = R/'validation/large_compression_20261005_r6/quadratic_candidate'
A = R/'validation/large_compression_20261005_r6/abaqus'
G = R/'validation/thin_target_20261004_r5/gauss_field.npz'
L, T, WIDTH, ETA = 10., .5, .05, 1e-4
ELL = WIDTH/(2*np.log(9))

def read(p): return json.loads(p.read_text())
def write(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def curves():
    records = {name: read(p) for name,p in {
        'Q2_fast':Q/'T0p004_compact/result.json', 'Q2_slow':Q/'T0p008_compact/result.json',
        'shell_slow':A/'explicit_T0p040/shell.json', 'shell_standard':A/'standard/shell.json'}.items()}
    data, summary = {}, {}
    for name,r in records.items():
        rows=r.get('path',r.get('force_path')); tload=.004 if name=='Q2_fast' else .008
        if name=='shell_slow': tload=.04
        rows=[z for z in rows if z['time']<=tload*(1+1e-9)] if name!='shell_standard' else rows
        # Keep the first occurrence at an equal compression; exclude the hold.
        aa=np.array([z['compression'] for z in rows]); ff=-np.array([z['Fz_N'] for z in rows])
        unique,idx=np.unique(aa,return_index=True); ff=ff[idx]; aa=unique
        assert np.all(np.diff(aa)>0) and aa[-1]>.1999
        data[name]=(aa,ff)
        k=(aa>=.002)&(aa<=.01)
        slope=float(np.dot(aa[k],ff[k])/np.dot(aa[k],aa[k]))
        peak=int(np.argmax(ff))
        summary[name]={'early_secant_fit_N_per_compression':slope,
                       'peak_N':float(ff[peak]),'peak_compression':float(aa[peak]),
                       'loading_work_N_mm':float(trapezoid(ff,aa)*L),
                       'loading_endpoint_N':float(ff[-1]),
                       'hold_mean_magnitude_N':-r['hold_mean_Fz_N'] if 'hold_mean_Fz_N' in r else None}
    grid=np.linspace(.001,.2,400)
    ref=np.interp(grid,*data['shell_slow']); norm=summary['shell_standard']['peak_N']
    phases={}
    for name in ['Q2_fast','Q2_slow','shell_standard']:
        f=np.interp(grid,*data[name]); d=f-ref
        phases[name]={}
        for label,lo,hi in [('early',.001,.05),('peak_region',.05,.12),('post_peak',.12,.2)]:
            m=(grid>=lo)&(grid<=hi)
            phases[name][label]={'RMS_over_Standard_peak':float(np.sqrt(np.mean(d[m]**2))/norm),
                                'mean_excess_force_N':float(np.mean(d[m]))}
        phases[name]['whole_RMS_over_Standard_peak']=float(np.sqrt(np.mean(d**2))/norm)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,(aa,ff) in data.items(): axes[0].plot(aa*100,ff,label=name)
    for name in ['Q2_slow','shell_standard']:
        axes[1].plot(grid*100,np.interp(grid,*data[name])-ref,label=name+' - shell_slow')
    axes[0].set(xlabel='Compression (%)',ylabel='Force magnitude (N)')
    axes[1].set(xlabel='Compression (%)',ylabel='Force difference (N)')
    for ax in axes: ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(O/'curves.png',dpi=160);plt.close(fig)
    np.savetxt(O/'curves.csv',np.column_stack([grid]+[np.interp(grid,*data[n]) for n in data]),
               delimiter=',',header='compression,'+','.join(data),comments='')
    return {'summary':summary,'phase_comparison':phases,
            'scope':'Loading-only diagnostic. Early fit uses 0.2%-1%; dynamic effects may affect it. Hold means and endpoint forces are separate.'}

def saved_field(name, phi, qp, weights):
    folder=Q/name;r=read(folder/'result.json');row=r['path'][-1]
    with np.load(folder/'field.npz') as f: q=f['q']; n=int(f['N']); savedtime=float(f['time'])
    assert n==32 and abs(savedtime-row['time'])<1e-12
    fam,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(fam,cell,degree)
    local=np.rint(2*el.points[order]).astype(int);xi=qp[0]*n
    gradients=el.tabulate(1,xi)[1:4,:,:,0].take(order,axis=2).transpose(1,2,0)*n
    expected=np.prod(np.where(abs(xi-.5)<1e-10,4/9,5/18),axis=1)/n**3
    werr=float(np.max(abs(weights-expected)))
    assert werr<1e-15
    totals={k:0. for k in ['U','U_dev','U_vol','F','floor_U','floor_F']}
    groups={k:{'points':0,'volume_mm3':0.,'U_N_mm':0.,'Fz_N':0.,'absolute_Pzz_integral_N':0.,
               'J_min':1e20} for k in ['soft_phi_lt_0p01','interface_0p01_to_0p9','solid_phi_ge_0p9']}
    stress_error=energy_error=0.; minimum=1e20
    checkP=jax.jit(jax.vmap(first_piola));checkW=jax.jit(jax.vmap(neo_hookean_energy))
    for start in range(0,n**3,1024):
        k=np.arange(start,min(start+1024,n**3));orig=np.stack([k//n**2,(k//n)%n,k%n],axis=1)
        grid=(2*orig[:,None,:]+local[None,:,:])%(2*n)
        ids=(grid[...,0]*(2*n)+grid[...,1])*(2*n)+grid[...,2]
        F=np.eye(3)+np.diag([0,0,-row['compression']])+np.einsum('cni,qnj->cqij',q[ids],gradients)
        J=np.linalg.det(F);assert np.all(J>0);minimum=min(minimum,float(J.min()))
        invT=np.swapaxes(np.linalg.inv(F),-1,-2);I1=np.sum(F*F,axis=(-2,-1))
        dev=MU/2*(J**(-2/3)*I1-3);vol=KAPPA/2*(J-1)**2;W=dev+vol
        P=MU*J[...,None,None]**(-2/3)*(F-I1[...,None,None]/3*invT)+KAPPA*(J*(J-1))[...,None,None]*invT
        # Sample every processed batch against the unchanged shared JAX kernel.
        sample=F.reshape(-1,3,3)[::173]
        stress_error=max(stress_error,float(np.max(abs(P.reshape(-1,3,3)[::173]-np.asarray(checkP(jnp.asarray(sample)))))))
        energy_error=max(energy_error,float(np.max(abs(W.ravel()[::173]-np.asarray(checkW(jnp.asarray(sample)))))))
        rho=phi[k];scale=ETA+(1-ETA)*rho;wt=weights[k]
        u=W*scale*wt*L**3;f=P[...,2,2]*scale*wt*L**2
        totals['U']+=float(u.sum());totals['F']+=float(f.sum())
        totals['U_dev']+=float((dev*scale*wt*L**3).sum());totals['U_vol']+=float((vol*scale*wt*L**3).sum())
        totals['floor_U']+=float((W*ETA*wt*L**3).sum());totals['floor_F']+=float((P[...,2,2]*ETA*wt*L**2).sum())
        masks=[rho<.01,(rho>=.01)&(rho<.9),rho>=.9]
        for g,m in zip(groups.values(),masks):
            g['points']+=int(m.sum());g['volume_mm3']+=float((m*wt*L**3).sum())
            g['U_N_mm']+=float(u[m].sum());g['Fz_N']+=float(f[m].sum())
            g['absolute_Pzz_integral_N']+=float(abs(f[m]).sum());g['J_min']=min(g['J_min'],float(J[m].min()))
    uerr=abs(totals['U']/row['energy_N_mm']-1)
    ferr=abs(totals['F']-row['internal_macro_Fz_N'])/abs(row['internal_macro_Fz_N'])
    assert uerr<1e-11 and ferr<1e-11 and abs(minimum-row['J_min'])<1e-12
    for g in groups.values():g['energy_fraction']=g['U_N_mm']/totals['U'];g['signed_force_fraction']=g['Fz_N']/totals['F']
    return {'totals':totals,'groups':groups,'relative_energy_record_error':uerr,'relative_internal_force_record_error':ferr,
            'shared_kernel_absolute_stress_error_MPa':stress_error,'shared_kernel_absolute_energy_error_MPa':energy_error,
            'weight_max_error':werr,'saved_time':savedtime,'J_min':minimum,
            'floor_force_fraction':totals['floor_F']/totals['F'],'floor_energy_fraction':totals['floor_U']/totals['U'],
            'scope':'Exact saved-state decomposition. Signed force contributions can cancel. Neither fractions nor floor removal estimate reequilibrated error.'}

def thickness(phi, distance, wt):
    binary=distance<=T/(2*L)
    def moments(rho):
        return {'M0_mm3':float(np.sum(rho*wt)*L**3),
                'M2_distance_mm5':float(np.sum(rho*wt*distance**2)*L**5)}
    qm={'smooth':moments(phi),'sharp_on_same_Gauss':moments(binary)}
    with np.load(G) as f: vertices=f['surface_vertices'];tri=f['surface_triangles']
    sides=vertices[tri];cross=np.cross(sides[:,1]-sides[:,0],sides[:,2]-sides[:,0]);area=np.linalg.norm(cross,axis=1)/2
    normals=cross/(2*area[:,None]);centers=sides.mean(axis=1)
    # Deterministic area-stratified sampling; a diagnostic, not a surface integral theorem.
    cdf=np.cumsum(area)/area.sum();selected=np.searchsorted(cdf,(np.arange(96)+.5)/96)
    z=np.linspace(-.45,.45,3601);query=centers[selected,None,:]+z[None,:,None]/L*normals[selected,None,:]
    d=PeriodicSurfaceDistance(vertices,tri).query(query)*L
    smooth=expit((T/2-d)/ELL);sharp=(d<=T/2).astype(float);flat=expit((T/2-abs(z))/ELL)
    profile={};samples=[]
    for label,occ in [('distance_sharp',sharp),('distance_smooth',smooth)]:
        m0=trapezoid(occ,z,axis=1);m1=trapezoid(occ*z,z,axis=1);m2=trapezoid(occ*z*z,z,axis=1)-m1*m1/m0
        for i in range(96):
            if label=='distance_sharp':samples.append({'facet_index':int(selected[i]),'center':centers[selected[i]].tolist()})
            samples[i][label+'_M0_mm']=float(m0[i]);samples[i][label+'_M2_mm3']=float(m2[i])
        profile[label]={'membrane_ratio_vs_t':np.quantile(m0/T,[0,.1,.5,.9,1]).tolist(),
                        'bending_ratio_vs_t3_over_12':np.quantile(m2/(T**3/12),[0,.1,.5,.9,1]).tolist(),
                        'M0_mean_mm':float(m0.mean()),'M2_mean_mm3':float(m2.mean())}
    flat0=float(trapezoid(flat,z));flat2=float(trapezoid(flat*z*z,z));theory2=T**3/12+np.pi**2*ELL**2*T/3
    assert abs(flat2/theory2-1)<1e-5
    qm['scope']='Global unsigned-distance second moment includes curvature and varying surface area; it is not an exact bending stiffness.'
    np.savez_compressed(O/'normal_profiles.npz',z_mm=z,distance_mm=d,smooth=smooth,selected_facets=selected)
    return {'same_Gauss_geometry_moments':qm,'normal_profiles':profile,'normal_profile_samples':samples,
            'flat_wall':{'ell_mm':ELL,'M0_mm':flat0,'M2_mm3':flat2,'analytic_M2_mm3':theory2,
                         'membrane_ratio':flat0/T,'bending_ratio':flat2/(T**3/12)},
            'scope':'Initial one-dimensional normal profiles; not nonlinear force predictions. Central second moments remove profile centroid shift; no curved-coordinate Jacobian used.'}

def main():
    started=time.perf_counter();O.mkdir(parents=True,exist_ok=False)
    inputs=[Q/'gauss_field.npz',G,Q/'T0p004_compact/result.json',Q/'T0p004_compact/field.npz',
            Q/'T0p008_compact/result.json',Q/'T0p008_compact/field.npz',A/'explicit_T0p040/shell.json',A/'standard/shell.json',
            R/'hyperelastic_fem.py',R/'surface_distance.py',R/'scripts/thin_target_explicit.py',R/'pixi.lock']
    before={str(p.relative_to(R)):sha(p) for p in inputs}
    manifest={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
              'input_sha256':before,'source_sha256':sha(Path(__file__)),
              'environment':{'python':sys.version,'numpy':np.__version__,'basix':basix.__version__,'jax':jax.__version__},
              'locked_physics':{'L_mm':L,'t_mm':T,'interface_10_90_mm':WIDTH,'eta':ETA,'mu_MPa':MU,'kappa_MPa':KAPPA,
                                'periodicity':'XYZ','macro_lateral_strain':0,'contact':False,'plasticity':False}}
    write(O/'manifest.json',manifest);shutil.copy2(__file__,O/'experiment.py')
    curve=curves();print('Curves recovered',flush=True)
    with np.load(Q/'gauss_field.npz') as f:phi=f['rho'];qp=f['physical_quad_points'];wt=f['JxW'];d=f['distance']
    fields={name:saved_field(name,phi,qp,wt) for name in ['T0p004_compact','T0p008_compact']}
    print('Both Gauss fields recovered',flush=True)
    geometry=thickness(phi,d,wt)
    assert before=={str(p.relative_to(R)):sha(p) for p in inputs}
    result={'status':'step1_diagnostics_complete','curves':curve,'fields':fields,'thickness':geometry,
            'body_seconds':time.perf_counter()-started,'protected_inputs_unchanged':True,
            'new_mechanical_jobs':0,'training_jobs':0}
    write(O/'result.json',result)
    print(json.dumps({'status':result['status'],'body_seconds':result['body_seconds'],
                      'slow_field':fields['T0p008_compact'],'thickness_summary':{k:v for k,v in geometry.items() if k!='normal_profile_samples'},
                      'curves':curve},indent=2),flush=True)

if __name__=='__main__':main()
