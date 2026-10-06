"""Target only failed endpoint brackets: distinguish first local exit/re-entry."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time
os.environ['JAX_PLATFORMS']='cpu'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
O=R/'validation/geometry_transfer_20261006_r15'
import numpy as np
from surface_distance import PeriodicSurfaceDistance
from scripts.prepare_thin_target import read_shell_mesh
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
cfg=json.loads((O/'input.json').read_text());old=json.loads((O/'preparation.json').read_text())
assert not old['step1_input_ready'] and old['band']['normal_bracket_failures']==4
assert not (O/'band_bracket_diagnosis.json').exists()
files=[O/x for x in ['input.json','preparation.json','gauss_field.npz','hrz_mass.npz','input/shell_mesh.inc']]
before={str(p):sha(p) for p in files};start=time.perf_counter();L=cfg['cell_size_mm'];t=cfg['thickness_mm']/L
vmm,f,_,_=read_shell_mesh(O/'input/shell_mesh.inc');v=vmm/L
cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
areas=np.linalg.norm(cross,axis=1)/2;normal=cross/(2*areas[:,None]);centers=v[f].mean(axis=1)
rng=np.random.default_rng(20261006);sample=rng.choice(len(f),512,replace=False,p=areas/areas.sum())
surface=PeriodicSurfaceDistance(v,f);failed=[]
for sign in [-1,1]:
    dist=surface.query(centers[sample]+sign*t*normal[sample])
    failed.extend((int(i),sign) for i in sample[dist<=t/2])
assert len(failed)==4
rows=[];s_values=np.linspace(0,t,201)
for face,sign in failed:
    pair=[]
    for direction_sign in [-1,1]:
        points=centers[face]+s_values[:,None]*direction_sign*normal[face]
        distance,faces,hits=surface.query(points,details=True);inside=distance<=t/2
        exits=np.where(inside[:-1]&~inside[1:])[0];entries=np.where(~inside[:-1]&inside[1:])[0]
        exit_distance=None
        if len(exits):
            i=int(exits[0]);lo=float(s_values[i]);hi=float(s_values[i+1])
            for _ in range(32):
                mid=(lo+hi)/2
                if float(surface.query(centers[face]+mid*direction_sign*normal[face]))<=t/2:lo=mid
                else:hi=mid
            exit_distance=(lo+hi)*L/2
        gap=None
        if len(exits) and len(entries):gap=float((s_values[entries[entries>exits[0]][0]]-s_values[exits[0]+1])*L) if np.any(entries>exits[0]) else None
        pair.append({'sign':direction_sign,'first_exit_mm':exit_distance,
         'exit_count_on_0_to_t':len(exits),'entry_count_on_0_to_t':len(entries),
         'positive_sampled_gap_before_reentry_mm':gap,'endpoint_inside_band':bool(inside[-1]),
         'ray_offset_mm':(s_values*L).tolist(),'nearest_distance_mm':(distance*L).tolist(),'nearest_face_ids':faces.tolist()})
    local_width=sum(p['first_exit_mm'] for p in pair) if all(p['first_exit_mm'] is not None for p in pair) else None
    relevant=next(p for p in pair if p['sign']==sign)
    proven=bool(relevant['first_exit_mm'] is not None and relevant['entry_count_on_0_to_t']>0 and
                relevant['positive_sampled_gap_before_reentry_mm'] is not None and relevant['positive_sampled_gap_before_reentry_mm']>0)
    rows.append({'face_id':face,'failed_sign':sign,'first_local_width_mm':local_width,
                 'failed_endpoint_explained_by_exit_then_reentry':proven,'pair':pair})
diagnosis={'work_kind':'Targeted geometry diagnostic only; no constant change or displacement solve',
 'failed_brackets':4,'rays':rows,'all_failed_endpoints_have_first_exit_then_reentry':all(r['failed_endpoint_explained_by_exit_then_reentry'] for r in rows),
 'corrected_failed_ray_local_widths_mm':[r['first_local_width_mm'] for r in rows],
 'sampling_note':'201 points on each of the four failed rays and opposite partner; first local exit bracket then bisection. Not global separation/contact certification.',
 'original_preparation_unchanged':True,'input_sha256':before,'source_sha256':sha(Path(__file__)),
 'seconds':time.perf_counter()-start,'new_FEM_solves':0,'new_Abaqus_jobs':0}
assert before=={str(p):sha(p) for p in files}
write(O/'band_bracket_diagnosis.json',diagnosis);shutil.copy2(__file__,O/'tools/diagnose_band.py')
shutil.copy2(Path(__file__).with_name('recover_report.py'),O/'tools/recover_report_fixed.py')
print(json.dumps({k:diagnosis[k] for k in ['all_failed_endpoints_have_first_exit_then_reentry','corrected_failed_ray_local_widths_mm','seconds']},indent=2),flush=True)
