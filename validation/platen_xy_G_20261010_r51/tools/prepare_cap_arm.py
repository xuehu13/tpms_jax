"""r51-only finite-cap shell contact diagnostic. Production mechanics read-only."""
from pathlib import Path
import hashlib, json, itertools
import numpy as np
from scipy.spatial import cKDTree
R=Path('/home/xuehu/projects/tpms_jax')
D=R/'validation/platen_xy_G_20261010_r51'
B=D/'diagnostics/shell_cap_arm_20261010'
def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if B.exists(): raise FileExistsError(B)
    for n in ['input','abaqus/probe','abaqus/G','analysis']: (B/n).mkdir(parents=True,exist_ok=True)
    old=json.loads((D/'diagnostics/end_section_benchmark_20261010/input/jax_protected_before.json').read_text())
    for k,h in old.items(): assert sha(R/k)==h,k
    paths={R/k for k in old}
    for p in D.rglob('*'):
        if p.is_file() and p.suffix in ['.npz','.json','.csv','.png','.inp','.py'] and not p.is_relative_to(B) and p.name!='status.json' and '__pycache__' not in str(p): paths.add(p)
    write(B/'input/protected_before.json',{str(p.relative_to(R)):sha(p) for p in sorted(paths)})
    cfg=json.loads((D/'input/protocol.json').read_text())
    with np.load(D/'input/cut_face_contact.npz') as z: c=dict(z)
    with np.load(R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz') as z: v=z['surface_vertices']
    # Discard only numerical tails; no material/HRZ/cap changes on JAX side.
    use=c['phi']>=1e-10; selected=np.flatnonzero(use); points=c['points'][use]*10
    area=c['area_weight'][use]*100; side=c['side'][use]
    shifts=np.array(list(itertools.product([-1,0,1],repeat=2)))
    parent=np.empty(len(points),int); shift=np.empty((len(points),2),int)
    for s in (0,1):
        rim=np.flatnonzero(abs(v[:,2]-s)<1e-10)
        tiled=(v[rim,None,:2]+shifts[None,:,:]).reshape(-1,2)
        _,j=cKDTree(tiled).query(points[side==s,:2]/10)
        parent[side==s]=rim[j//len(shifts)]+1; shift[side==s]=shifts[j%len(shifts)]
    base=(D/'abaqus/platen_2pct.inp').read_text()
    nv=cfg['physical_nodes']; first=nv+13
    proxy_labels=np.arange(first,first+len(points))
    nextlabel=int(proxy_labels[-1])+1; ghosts={}; masters=[]; extra_nodes=[]; eq=[]
    for par,sh in zip(parent,shift):
        if not np.any(sh): masters.append(int(par)); continue
        key=(int(par),int(sh[0]),int(sh[1]))
        if key not in ghosts:
            ghosts[key]=nextlabel; p=v[par-1]*10+np.r_[sh*10,0]
            extra_nodes.append(f'{nextlabel},'+','.join(f'{x:.16g}' for x in p))
            for dof in range(1,7):
                terms=f'{nextlabel},{dof},1.,{par},{dof},-1.'; count=2
                if dof in (1,2) and sh[dof-1]:
                    terms+=f",{cfg['labels']['ax' if dof==1 else 'ay']},{dof},{-int(sh[dof-1])}."; count=3
                eq+=['*Equation',str(count),terms]
            nextlabel+=1
        masters.append(ghosts[key])
    extra_nodes=[f'{n},'+','.join(f'{x:.16g}' for x in p) for n,p in zip(proxy_labels,points)]+extra_nodes
    assert base.count('*Element, type=S3R, elset=WALL')==1
    text=base.replace('*Element, type=S3R, elset=WALL','\n'.join(extra_nodes)+'\n*Element, type=S3R, elset=WALL')
    # Explicit node-based pair area is always one. Encode kn*A in the force law!
    exact=cfg['contact_normal_stiffness_MPa_per_mm']*area
    stiffness=np.array([float(f'{x:.6g}') for x in exact])
    keys=sorted(set(stiffness.tolist())); props=[]; surfs=[]; pairs=[]
    for i,k in enumerate(keys):
        prop=f'CAP_K{i}'
        props += [f'*Surface Interaction, name={prop}','*Surface Behavior, pressure-overclosure=LINEAR',f'{k:.12g}',
                  '*Contact Damping, definition=CRITICAL DAMPING FRACTION','0.']
        for s in (0,1):
            ix=np.flatnonzero((stiffness==k)&(side==s))
            if not len(ix): continue
            surf=f'CAP_{s}_{i}'
            surfs += [f'*Surface, type=NODE, name={surf}']+[f'{proxy_labels[j]},1.' for j in ix]
            pairs += [f'*Contact Pair, interaction={prop}, mechanical constraint=PENALTY',f'{surf},PLATEN_'+('BOTTOM' if s==0 else 'TOP')]
    mpc=['*MPC']+[f'BEAM,{n},{p}' for n,p in zip(proxy_labels,masters)]
    insert='\n'.join(eq+mpc+surfs+props)+'\n'
    text=text.replace('*Surface Interaction, name=FRICTIONLESS',insert+'*Surface Interaction, name=FRICTIONLESS')
    oldpairs='*Contact Pair, interaction=FRICTIONLESS, mechanical constraint=PENALTY\nEND_BOTTOM,PLATEN_BOTTOM\nEND_TOP,PLATEN_TOP'
    assert text.count(oldpairs)==1
    text=text.replace(oldpairs,'\n'.join(pairs))
    # Add diagnostics; previous global outputs/physics remain as authored.
    text=text.replace('*Element Output, elset=WALL','*Contact Output\nCSTRESS,CDISP,CFORCE\n*Element Output, elset=WALL')
    text=text.replace('ALLIE,ALLSE,ALLKE,ALLAE,ALLVD,ALLWK,ETOTAL','ALLIE,ALLSE,ALLKE,ALLAE,ALLVD,ALLWK,ETOTAL,ALLPW,ALLCW')
    (B/'abaqus/G/platen_cap_arm.inp').write_text(text)
    origin=v[parent-1]*10+np.c_[shift*10,np.zeros(len(shift))]
    lever=points-origin
    # Geometric and rigid-motion virtual-work checks, independent of response.
    rng=np.random.default_rng(51); du=rng.normal(size=(len(points),3)); dw=rng.normal(size=du.shape)
    f=rng.normal(size=du.shape); dp=du+np.cross(dw,lever)
    lhs=float(np.sum(f*dp)); rhs=float(np.sum(f*du)+np.sum(np.cross(lever,f)*dw))
    audit={'proxy_count':len(points),'ghost_count':len(ghosts),'stiffness_groups':len(keys),'cap_contact_pairs':len(pairs)//2,
           'phi_tail_threshold':1e-10,'area_omitted_fraction':float(c['area_weight'][~use].sum()/c['area_weight'].sum()),
           'area_mm2':[float(area[side==s].sum()) for s in (0,1)],
           'max_stiffness_relative_rounding':float(np.max(abs(stiffness/exact-1))),
           'aggregate_stiffness_relative_rounding':[float(stiffness[side==s].sum()/exact[side==s].sum()-1) for s in (0,1)],
           'initial_gap_max_mm':float(np.max(abs(points[:,2]-10*side))),
           'area_weighted_arm_mm':float(np.sum(np.linalg.norm(lever,axis=1)*area)/area.sum()),
           'max_arm_phi_ge_0p5_mm':float(np.linalg.norm(lever[c['phi'][use]>=.5],axis=1).max()),
           'virtual_work_relative_error':abs(lhs-rhs)/max(abs(lhs),1e-30),'added_mass_tonne':0.,
           'caution':'Independent local rigid arms. Not a rigid cap joining shell rim nodes. Local shell stretch/thinning and smooth edge interpolation not represented.'}
    np.savez_compressed(B/'input/cap_arm_map.npz',cap_indices=selected,labels=proxy_labels,points_mm=points,parent_labels=parent,
                        shift_xy=shift,master_labels=masters,reference_arm_mm=lever,area_mm2=area,side=side,stiffness_N_mm=stiffness)
    protocol={**cfg,'case_id':'G_finite_cap_shell_arm_diagnostic','stage':'2pct explicit finite-width cap shell-arm diagnostic; not full-wall or self-contact',
              'shell_contact':'Cap sampling coordinates/weights shared with JAX; massless BEAM MPC arms following nearest periodic shell rim node; explicit force slopes kn*A, zero added contact damping',
              'no_self_contact':True,'input_sha256':sha(B/'abaqus/G/platen_cap_arm.inp'),
              'cap_map_sha256':sha(B/'input/cap_arm_map.npz'),'diagnostic_audit':audit,
              'single_task_budget_seconds':1450,'original_shell_peak_N':0.7241182387168918,
              'fixed_window_compression':[0.,.02],'compare_points':501,'hold_mean_window_s':[.0011,.0012],
              'provisional_short_gate':{'curve_RMS_each_denominator_max':.1,'hold_mean_abs_relative_bias_max':.1},
              'source_original_JAX_result':str(D/'jax/short_2pct/result.json'),
              'source_original_JAX_result_sha256':sha(D/'jax/short_2pct/result.json'),
              'changed_items':['shell contact support and rigid-arm moment transfer','automatic penalty replaced by area-weighted prescribed kn law','default penalty contact damping .03 replaced by zero to match JAX'],
              'remaining_differences':['shell plane-stress vs 3D continuum','rigid-arm local edge kinematics vs Q2 solid deformation','shell and HRZ projected inertia','shell nearest-node assignment and microscopic phi tails','current source triangulation difference'],
              'full20_submitted':False,'new_design_AD':False}
    write(B/'input/protocol.json',protocol);write(B/'input/input_audit.json',audit)
    # Tiny Explicit verification: a contact point at an offset rotates with a
    # prescribed shell node. Checks finite rotation, force units and moment.
    probe='''*Heading
r51 massless finite-rotation arm and contact slope probe
*Node
1,0,0,0
2,1,0,0
3,0,1,0
4,1,1,0
5,.25,0,0
10,0,0,0
11,-2,-2,0
12,2,-2,0
13,2,2,0
14,-2,2,0
*Element,type=S3R,elset=WALL
1,1,2,3
2,2,4,3
*Element,type=R3D4,elset=PLATE
3,11,12,13,14
*Rigid Body,ref node=10,elset=PLATE
*Material,name=BASE
*Density
1e-9
*Hyperelastic,Neo Hooke
1.923076923076923,.24
*Shell Section,elset=WALL,material=BASE
.5,5
*MPC
BEAM,5,1
*Surface,type=NODE,name=TIP
5,1.
*Surface,type=ELEMENT,name=GROUND
PLATE,SPOS
*Surface Interaction,name=K
*Surface Behavior,pressure-overclosure=LINEAR
2.
*Contact Damping,definition=CRITICAL DAMPING FRACTION
0.
*Boundary
1,1,2,0.
1,4,4,0.
1,6,6,0.
2,1,6,0.
3,1,6,0.
4,1,6,0.
10,1,6,0.
*Amplitude,name=A,definition=SMOOTH STEP
0.,0.,.001,1.,.0012,1.
*Step,name=PLATEN,nlgeom=YES
*Dynamic,Explicit
,.0012
*Contact Pair,interaction=K,mechanical constraint=PENALTY
TIP,GROUND
*Boundary,amplitude=A
1,3,3,-.01
1,5,5,.1
*Output,field,number interval=2
*Node Output
U,UR,RF,RM
*Contact Output
CSTRESS,CDISP,CFORCE
*Output,history,time interval=2e-6
*Energy Output
ALLIE,ALLKE,ALLAE,ALLVD,ALLWK,ETOTAL,ALLPW,ALLCW
*End Step
'''
    (B/'abaqus/probe/probe.inp').write_text(probe)
    write(B/'input/probe_protocol.json',{'stage':'Prescribed finite-rotation massless shell arm Explicit verification','load_time_s':.001,'hold_time_s':.0002})
    write(B/'status.json',{'stage':'prepared_probe_pending','active_jobs':[],'full20_submitted':False,'new_design_AD':False})
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
