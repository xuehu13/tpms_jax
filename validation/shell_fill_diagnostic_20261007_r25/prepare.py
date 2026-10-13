"""r25: one ordinary shell-in-solid near-zero control; no JAX solve.

Run from the formal WSL directory. All original shell nodes, triangles,
material, rotational periodic equations, and macro amplitude are reused.
The new translational space is deliberately tested, never assumed harmless.
"""
from pathlib import Path
import hashlib, json, math, shutil
import numpy as np

R = Path('/home/xuehu/projects/tpms_jax')
D = R / 'validation/shell_fill_diagnostic_20261007_r25'
W = Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
OLD = Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/shell_rate_20261007_r20_diverse04_explicit_T0p004')
P = OLD.parent / 'shell_fill_20261007_r25_nearzero'

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def blocks(s):
    out=[]
    for line in s.splitlines():
        if line.startswith('*') and not line.startswith('**'): out.append([line])
        elif out: out[-1].append(line)
    return out

def equation(terms):
    out=['*Equation',str(len(terms))]
    for start in range(0,len(terms),4):
        out.append(', '.join(f'{n}, {d}, {v:.16g}' for n,d,v in terms[start:start+4]))
    return out

def main():
    assert not P.exists(), 'Never reuse/overwrite a native job directory.'
    cfg=json.loads((OLD/'input.json').read_text())
    source=OLD/'thin_shell.inp'
    assert sha(source)==cfg['shell_inp_sha256']
    b=blocks(source.read_text())
    nblk=next(x for x in b if x[0].lower()=='*node')
    eblk=next(x for x in b if 'type=s3r' in x[0].lower())
    nodes={int(t[0]):np.array(list(map(float,t[1:4]))) for l in nblk[1:] if (t:=l.split(','))}
    tri=np.array([list(map(int,l.split(',')))[1:] for l in eblk[1:]],dtype=int)
    xyz=np.array([nodes[i] for i in range(1,cfg['physical_node_count']+1)])
    assert xyz.shape==(8372,3) and tri.shape==(15914,3)
    assert xyz.min()>=-1e-10 and xyz.max()<=10+1e-10
    area=float(np.linalg.norm(np.cross(xyz[tri[:,1]-1]-xyz[tri[:,0]-1],
                                      xyz[tri[:,2]-1]-xyz[tri[:,0]-1]),axis=1).sum()/2)
    N=32; L=10.; h=L/N; q=cfg['macro_control_label']; pin=q+1
    n0=100001; e0=100001
    def nid(i,j,k): return int(n0+(i*(N+1)+j)*(N+1)+k)
    # A stop at a=0.17 on the SAME 20%-in-0.004s quintic loading curve.
    lo,hi=0.,1.
    for _ in range(70):
        s=(lo+hi)/2; a=.2*(10*s**3-15*s**4+6*s**5)
        if a<.17: lo=s
        else: hi=s
    stop=.004*(lo+hi)/2
    ratio=1e-8; rho=1e-13
    shell_mass=area*.5*1e-9; added_mass=rho*L**3
    protocol={
        'research_question':'Does near-zero shell-in-solid embedding leave the original shell observed peak and curve sufficiently unchanged to interpret a subsequent filler-stiffness experiment?',
        'authorization':'2026-10-07 user explicitly resumed work and requested <=4 steps then execution',
        'plan_step':1,'status':'prepared_not_run','source_INP_sha256':sha(source),
        'host':{'type':'C3D8R','N':N,'elements':N**3,'nodes':(N+1)**3,'edge_mm':h,
                'purpose':'ordinary diagnostic host; NOT equivalent to HEX27/C2/HRZ JAX'},
        'filler':{'E_ratio_nearzero':ratio,'E_MPa':10*ratio,'nu':.3,
                  'density_tonne_per_mm3':rho,'planned_finite_E_ratio':1e-4,
                  'same_mass_for_both_stiffness_levels':True,'full_cube_including_overlap':True,
                  'material':'built-in compressible Neo-Hooke; actual geometry may distort and terminate'},
        'estimated_mass':{'shell_area_mm2':area,'shell_tonne':shell_mass,
                          'added_host_tonne':added_mass,'added_over_shell':added_mass/shell_mass,
                          'density_not_current_JAX_HRZ_distribution':True},
        'coupling':{'embedded':'all shell translations; original rotational equations retained',
                    'shell_translation_equations_removed':3*423,'shell_rotation_equations_retained':3*423,
                    'host_PBC':'XYZ, unique canonical representatives, lateral macro strain zero',
                    'gauge':'host interpolation at original shell pin position fixed to zero',
                    'absolute_exterior_tolerance_mm':1e-9,'roundoff_tolerance':0.,
                    'no_displacement_BC_or_translation_equation_on_embedded_shell':True},
        'loading':{'T20_seconds':.004,'a_stop':.17,'stop_seconds':stop,
                   'smoothstep_and_control_displacement_unchanged':True,'hold':False,
                   'no_contact':True,'no_plasticity':True,'no_mass_scaling':True},
        'predeclared_gate':{'purpose':'attribution screen, NOT a physical validation standard',
            'need_complete_to_a017':True,'observed_peak_shift_max_percentage_points':.5,
            'peak_force_relative_original_fast_peak_max':.05,
            'curve_RMS_original_fixed_2p25891N_max':.05,
            'curve_RMS_matched_fast_peak_max':.05,'window_compression':[.01,.17],
            'grid_points':1001,'fixed_denominator_N':2.25891,
            'energy_drift_max_relative_work':.01,'artificial_energy_max_relative_work':.05,
            'rationale':'peak gate is far smaller than the existing 4.1168pp mismatch; 5% curve/force screen avoids attributing a coupling effect to stiffness',
            'jump_KE_no_quasistatic_rejection':True,
            'failure_policy':'stop finite stiffness attribution; retain partial/failure; no VUEL, stabilization, or silent mesh sweep'},
        'future_finite_trial':{'launch_only_after_nearzero_gate_pass':True,
            'same_grid_coupling_mass_loading_and_outputs':True},
        'native_directory':str(P).replace('/mnt/e','E:'),
        'documentation':{'embedded':'https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-embeddedelement.htm',
            'version_scope':'available official 2025 documentation; installed 2026 syntax checked by datacheck'},
        'production_JAX_unchanged':True,'design_AD':False}
    D.mkdir(exist_ok=True)
    write(D/'protocol.json',protocol)  # Written before model/data check or solve.
    P.mkdir()
    out=['*Heading','r25 diverse04 nearzero embedded control; not JAX equivalent',
         '*Preprint, echo=NO, model=YES, history=NO',*nblk,
         f'{pin}, 0., 0., 0.','*Node']
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1): out.append(f'{nid(i,j,k)}, {i*h:.16g}, {j*h:.16g}, {k*h:.16g}')
    out+=eblk+['*Element, type=C3D8R, elset=FILLER']
    e=e0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                ns=[nid(i,j,k),nid(i+1,j,k),nid(i+1,j+1,k),nid(i,j+1,k),
                    nid(i,j,k+1),nid(i+1,j,k+1),nid(i+1,j+1,k+1),nid(i,j+1,k+1)]
                out.append(', '.join(map(str,[e]+ns)));e+=1
    for name,label in [('QZ',q),('PINCONTROL',pin)]: out+=['*Nset, nset='+name,str(label)]
    out+=['*Nset, nset=SHELLN, generate',f'1, {cfg["physical_node_count"]}, 1',
          '*Nset, nset=HOSTN, generate',f'{n0}, {nid(N,N,N)}, 1']
    for x in b:
        if x[0].lower().startswith(('*material','*density','*hyperelastic','*shell section')): out+=x
    out+=['*Material, name=SOFT','*Density',f'{rho:.16g}',
          '*Hyperelastic, Neo Hooke',f'{cfg["material"]["C10_MPa"]*ratio:.16g}, {cfg["material"]["D1_per_MPa"]/ratio:.16g}',
          '*Solid Section, elset=FILLER, material=SOFT',',',
          '*Embedded Element, host elset=FILLER, absolute exterior tolerance=1e-9, roundoff tolerance=0.',
          'WALL']
    eq_shell=0
    for x in b:
        if x[0].lower()=='*equation':
            first_dof=int(x[2].split(',')[1])
            if first_dof>=4:out+=x;eq_shell+=1
    assert eq_shell==3*423
    # The host's periodic duplicates only depend on interior canonical representatives.
    relations=[]
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                if N not in (i,j,k):continue
                r=nid(i%N,j%N,k%N);n=nid(i,j,k);dz=(k-k%N)*h
                relations.append([n,r,dz])
                for dof in (1,2,3):
                    terms=[(n,dof,1.),(r,dof,-1.)]
                    if dof==3 and dz:terms.append((q,3,-dz/L))
                    out+=equation(terms)
    # Fix only the global gauge at the same physical point as the old shell pin.
    p=nodes[cfg['pin_node_label']]/h;ijk=np.floor(p).astype(int);frac=p-ijk
    corners=[]
    for di in (0,1):
        for dj in (0,1):
            for dk in (0,1):
                wt=(frac[0] if di else 1-frac[0])*(frac[1] if dj else 1-frac[1])*(frac[2] if dk else 1-frac[2])
                corners.append((nid(*(ijk+np.array([di,dj,dk]))),float(wt)))
    corners.sort(key=lambda z:-z[1])
    for dof in (1,2,3):out+=equation([(n,dof,w) for n,w in corners]+[(pin,dof,-1.)])
    out+=['*Boundary','PINCONTROL, 1, 3, 0.','QZ, 1, 2, 0.',
          '*Amplitude, name=MACRO, definition=SMOOTH STEP, time=TOTAL TIME',
          '0., 0., 0.004, 1., 0.0044, 1.',
          '*Step, name=compress20, nlgeom=YES','*Dynamic, Explicit',f', {stop:.16g}',
          '*Boundary, amplitude=MACRO','QZ, 3, 3, -2.',
          '*Output, field, number interval=24','*Node Output, nset=SHELLN','U, UR',
          '*Node Output, nset=HOSTN','U','*Element Output, elset=FILLER','EVOL',
          '*Output, history, time interval=4.400000000000001e-06','*Energy Output',
          'ALLIE, ALLSE, ALLKE, ALLAE, ALLVD, ALLWK, ETOTAL',
          '*Node Output, nset=QZ','U3, RF3',
          '*Output, history, time interval=4.400000000000001e-06',
          '*Energy Output, elset=WALL','ALLIE, ALLSE, ALLKE, ALLAE, ALLVD',
          '*Output, history, time interval=4.400000000000001e-06',
          '*Energy Output, elset=FILLER','ALLIE, ALLSE, ALLKE, ALLAE, ALLVD','*End Step']
    (P/'shell_fill.inp').write_text('\n'.join(out)+'\n',encoding='ascii')
    newcfg={**cfg,'physical_node_count':cfg['physical_node_count'],'host_node_count':(N+1)**3,
        'host_label_start':n0,'host_label_end':nid(N,N,N),'host_N':N,'host_h_mm':h,
        'total_time_seconds':stop,'target_compression':.17,'hold_time_seconds':0.,
        'pin_control_label':pin,'input_kind':'nearzero_shell_embedded_C3D8R_control',
        'host_periodic_relations':relations,'original_source_INP_sha256':sha(source)}
    write(P/'input.json',newcfg)
    write(P/'protocol.json',protocol)
    for name in ('extract.py','run_windows.py'): shutil.copy2(D/name,P/name)
    production=['hyperelastic_fem.py','surface_distance.py','pbc.py','scripts/thin_target_explicit.py','pixi.toml','pixi.lock']
    freeze={str(R/n):sha(R/n) for n in production}
    freeze.update({str(OLD/n):sha(OLD/n) for n in ['thin_shell.inp','input.json','thin_shell.odb','shell.json']})
    write(D/'freeze_before.json',freeze)
    write(D/'preparation.json',{'INP_sha256':sha(P/'shell_fill.inp'),'input_json_sha256':sha(P/'input.json'),
        'nearzero_only':True,'host_PBC_relations':len(relations),'pin_weights':corners,
        'embedding_not_assumed_inert':True,'added_mass_fraction':added_mass/shell_mass,
        'shell_midsurface_and_connectivity_reused':True,'native_directory':str(P)})
    print(json.dumps({'prepared':str(P),'stop_seconds':stop,'shell_mass_tonne':shell_mass,
        'host_mass_fraction':added_mass/shell_mass,'host_relations':len(relations)},indent=2))

if __name__=='__main__':main()
