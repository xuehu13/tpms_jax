"""Native contact geometry audit and read-only G end kinematics decomposition."""
from pathlib import Path
import csv, hashlib, json, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path('/home/xuehu/projects/tpms_jax'); D = R/'validation/platen_xy_G_20261010_r51'
B = D/'diagnostics/native_shell_contact_20261010'; OLD = D/'diagnostics/shell_cap_arm_20261010'
sys.path.insert(0,str(D/'tools'))
from analyze_cap_arm_preparation import interpolate_jax, rotate
def read(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def initial_normal_envelope():
    s=np.load(R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz')
    v=s['surface_vertices']*10;t=s['surface_triangles'];tri=v[t]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    edges=np.sort(np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),axis=1);owners=np.tile(np.arange(len(t)),3)
    _,first,count=np.unique(edges,axis=0,return_index=True,return_counts=True)
    ix=first[count==1];edges=edges[ix];owners=owners[ix];rows=[]
    for side in [0,1]:
        use=np.all(np.isclose(v[edges,2],side*10,atol=1e-9),axis=1);e=edges[use];f=owners[use]
        length=np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1);z=.25*abs(normals[f,2]);j=np.argsort(z);cum=np.cumsum(length[j])/length.sum()
        rows.append({'side':'bottom' if side==0 else 'top','boundary_segments':len(e),'total_reference_rim_length_mm':float(length.sum()),
            'normal_extrusion_beyond_flat_plate_length_weighted_mean_mm':float(np.sum(length*z)/length.sum()),
            'weighted_median_mm':float(np.interp(.5,cum,z[j])),'weighted_95pct_mm':float(np.interp(.95,cum,z[j])),
            'max_mm':float(z.max()),'rim_length_fraction_above_0p01mm':float(length[z>.01].sum()/length.sum())})
    return {'scope':'Read-only initial reference facet-normal extrusion of .5mm shell at its untrimmed midsurface rim. Geometric illustration; not actual Abaqus contact thickness or new initial equilibrium. Facet normals, not interpolated shell directors. Current background solid is clipped by z=0 and10 planes.',
        'rows':rows,'new_G':False,'new_AD':False}
def main():
    assert not (B/'analysis/metrics.json').exists()
    cases=['default','self','noperimself']; rows=[]; seconds={a:0. for a in ['datacheck','solve','extract']}
    for case in cases:
        out=B/f'abaqus/{case}';cfg=read(B/f'input/{case}_protocol.json');r=read(out/'result.json')
        assert r['complete']
        sta=(out/'platen.sta').read_text(errors='replace')
        scales=re.findall(r'scaling factor was\s+([0-9.]+)',sta)
        sth=np.array([v['data'] for v in read(out/'contact_fields.json')[-1]['fields']['STH']])
        motion=np.load(B/'input/probe_mesh.npz')['final_u_mm']
        observed=np.array([r['last_fields']['U'][str(n)] for n in range(1,82)])
        assert np.max(abs(observed-motion))<1e-6
        assert 'No nodal position adjustments have been made.' in sta
        assert np.max(abs(sth-.5))<1e-6
        lo,hi=r['onset_plate_z_bracket_mm']
        rows.append({'case':case,'first_contact_height_low_mm':lo,'first_contact_height_high_mm':hi,
            'physical_flat_cut_height_mm':cfg['physical_flat_cut_highest_z_mm'],
            'contact_height_error_low_mm':lo-cfg['physical_flat_cut_highest_z_mm'],
            'contact_height_error_high_mm':hi-cfg['physical_flat_cut_highest_z_mm'],
            'reported_most_severe_contact_thickness_scale':float(scales[0]) if scales else 1.,
            'structural_thickness_min_mm':float(sth.min()),'structural_thickness_max_mm':float(sth.max()),
            'motion_max_error_mm':float(np.max(abs(observed-motion))),
            'geometry_gate_passed':r['local_geometry_gate_passed'],'final_prescribed_pose_contact_force_N':r['final_force_N']})
        for action in seconds:
            receipt=read(out/f'{action}_receipt.json');assert receipt['success'];seconds[action]+=receipt['wall_seconds']
    table(B/'analysis/native_geometry.csv',rows)
    m=dict(np.load(OLD/'input/cap_arm_wrapped_map.npz'));s=dict(np.load(OLD/'input/original_shell_pose.npz'))
    c=dict(np.load(D/'input/cut_face_contact.npz'));qz=np.load(D/'jax/short_2pct/last_valid_state.npz')
    par=m['parent_labels']-1;arm=m['reference_arm_mm'];area=m['area_mm2'];side=m['side'];phi=c['phi'][m['cap_indices']]
    u,F=interpolate_jax(s['xyz_mm'][par]/10,qz['q'],qz['macro'],True)
    UU,stretch,VH=np.linalg.svd(F);rot=UU@VH
    assert np.all(np.linalg.det(F)>0) and np.all(np.linalg.det(rot)>0)
    rigid=m['points_mm']+u*10+rotate(rot-np.eye(3),arm)
    affine=m['points_mm']+u*10+rotate(F-np.eye(3),arm)
    actual=c['points'][m['cap_indices']]*10+interpolate_jax(c['points'][m['cap_indices']],qz['q'],qz['macro'])*10
    # Only Z is compared: wrapping changes XY physical coordinates by periods.
    strain_z=affine[:,2]-rigid[:,2];remainder_z=actual[:,2]-affine[:,2];total_z=actual[:,2]-rigid[:,2]
    closure=float(np.max(abs(total_z-strain_z-remainder_z)))
    assert closure<1e-12
    kin=[]
    for sd in [0,1,2]:
        use=side==sd if sd!=2 else np.ones(len(side),bool);w=area[use]/area[use].sum()
        rms=lambda x:float(np.sqrt(np.sum(w*x[use]**2)))
        kin.append({'side':'bottom' if sd==0 else ('top' if sd==1 else 'both'),
            'rigid_actual_Z_RMS_mm':rms(total_z),'nonrigid_affine_correction_Z_RMS_mm':rms(strain_z),
            'affine_actual_Z_RMS_mm':rms(remainder_z),
            'affine_correction_variance_term_mm2':float(np.sum(w*strain_z[use]**2)),
            'remainder_variance_term_mm2':float(np.sum(w*remainder_z[use]**2)),
            'cross_variance_term_mm2':float(2*np.sum(w*strain_z[use]*remainder_z[use])),
            'max_abs_rigid_error_phi_ge_0p5_mm':float(np.max(abs(total_z[use&(phi>=.5)]))),
            'max_abs_affine_error_phi_ge_0p5_mm':float(np.max(abs(remainder_z[use&(phi>=.5)])))})
    table(B/'analysis/saved_G_kinematics.csv',kin)
    envelope=initial_normal_envelope()
    write(B/'analysis/G_initial_normal_envelope.json',envelope)
    np.savez_compressed(B/'analysis/saved_G_kinematics.npz',area_mm2=area,side=side,phi=phi,
        rigid_actual_Z_mm=total_z,nonrigid_affine_Z_mm=strain_z,affine_actual_Z_mm=remainder_z)
    metrics={'scope':'Three prescribed local Explicit geometry probes plus read-only decomposition of original G terminal state. No new deformable G path, no new JAX solve or design AD.',
        'native_contact_geometry':rows,'saved_G_kinematics':kin,'kinematic_decomposition_closure_error_mm':closure,
        'initial_G_normal_envelope':envelope,
        'decomposition_definition':'actual-rigid = (F-R)*arm + [actual-(parent+F*arm)]. The first includes local stretch/shear. The second includes non-affine Q2 variation and nearest-parent/surface-curvature effects; RMS terms are correlated, not additive percentages.',
        'native_geometry_gate_passed':any(r['geometry_gate_passed'] for r in rows),
        'production_adoption':False,'original_G_comparison_still_not_passed':True,
        'budget_seconds':seconds,'new_G_paths':0,'full20_submitted':False,'new_design_AD':False}
    write(B/'analysis/metrics.json',metrics)
    write(B/'analysis/budget_receipt.json',{**seconds,'sum_seconds':sum(seconds.values()),'scope':'Three datachecks, three prescribed-motion Explicit solves, three extractions; authoring and read-only analysis excluded.'})
    protected=read(B/'input/protected_before.json');changed=[p for p,h in protected.items() if sha(R/p)!=h]
    write(B/'analysis/integrity.json',{'protected_count':len(protected),'changed':changed,'passed':not changed})
    assert not changed,changed
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11.7,4.5),constrained_layout=True)
    labels=['Default','SELF','NOPERIMSELF'];heights=[(r['first_contact_height_low_mm']+r['first_contact_height_high_mm'])/2 for r in rows]
    axs[0].bar(labels,np.array(heights)*1000,color=['#3c6682','#797979','#b87837'])
    ref=rows[0]['physical_flat_cut_height_mm']*1000;axs[0].axhline(ref,color='#9c3434',ls='--',label=f'Physical flat end: {ref:.2f} um')
    for i,h in enumerate(heights):axs[0].text(i,h*1000+4,f'{h*1000:.2f}',ha='center',fontsize=9)
    axs[0].set(ylabel='Plate height at first contact (micrometre)',title='Native shell contact is not the same flat cut');axs[0].legend(fontsize=8)
    k=kin[-1];vals=np.array([k['rigid_actual_Z_RMS_mm'],k['affine_actual_Z_RMS_mm']])*1000
    axs[1].bar(['Rigid-arm approximation','Full F linear approximation'],vals,color=['#3c6682','#797979'],width=.55)
    for i,v in enumerate(vals):axs[1].text(i,v+max(vals)*.025,f'{v:.3f} um',ha='center')
    axs[1].set(ylabel='Occupied-area weighted Z error RMS (micrometre)',title='One-point reconstruction does not recover the cap',ylim=(0,max(vals)*1.25))
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Local geometry probes + saved-state analysis | no new G compression solution',fontsize=12)
    fig.savefig(B/'analysis/contact_geometry_and_kinematics.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,4.5),constrained_layout=True)
    for case,label,color in zip(cases,labels,['#3c6682','#797979','#b87837']):
        z=np.load(B/f'abaqus/{case}/field.npz');use=(z['time_s']>=.0002)&(z['time_s']<=.0012)
        ax.plot(z['plate_z_mm'][use]*1000,z['force_N'][use],label=label,color=color,lw=1.5)
    ax.axvline(ref,color='#9c3434',ls='--',label='Physical flat end first touch')
    ax.invert_xaxis();ax.set(xlabel='Descending plate Z (micrometre)',ylabel='Contact force in prescribed-motion probe (N)',
        title='Identical prescribed shell motion; only native thickness control changes')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.savefig(B/'analysis/native_contact_force.png',dpi=180);plt.close(fig)
    write(B/'status.json',{'stage':'three_prescribed_native_geometry_probes_complete_analyzed','active_jobs':[],
        'native_flat_cut_geometry_gate_passed':False,'native_contact_not_adopted':True,'new_G_paths':0,
        'original_G_comparison_passed':False,'full20_submitted':False,'new_design_AD':False})
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
