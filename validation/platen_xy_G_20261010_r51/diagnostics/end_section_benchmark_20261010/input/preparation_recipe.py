"""Two same-solid static contact-support tests, local to r51; not a G replacement."""
from pathlib import Path
import json,hashlib,sys,shutil
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/platen_xy_G_20261010_r51'
B=D/'diagnostics/end_section_benchmark_20261010'
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not B.exists()
for name in ('input','abaqus/cap','abaqus/rim','analysis'): (B/name).mkdir(parents=True)
old=D/'diagnostics/contact_support_control_20261010/input/frozen_before.json'
protected=json.loads(old.read_text())
for p in (D/'diagnostics/contact_support_control_20261010').rglob('*'):
    if p.is_file() and p.suffix not in ('.pyc','.log'):protected[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
write(B/'input/frozen_before.json',protected)
shutil.copy2(D/'tools/run_abaqus.py',B/'input/run_abaqus_before.py')
protocol={'question':'Does finite-width end contact change force and rotation relative to centerline contact on exactly the same conforming 3D solid?',
 'scope':'Local inclined thin-wall cantilever mechanism test, not G, not replacement of its XY/free macro boundary conditions.',
 'height_mm':10.,'normal_thickness_mm':.5,'width_mm':1.,'midsurface_dx_dz':.2,
 'E_MPa':10.,'nu':.3,'C10_MPa':10/(4*1.3),'D1_MPa_inverse':.24,
 'solid_element':'C3D20R','nx_ny_nz':[4,1,40],
 'base':'z=0 clamped U1/U2/U3; U2=0 throughout for plane strain along extrusion',
 'loading':'Rigid horizontal top platen, U3=-.2mm; other rigid DOFs fixed; frictionless direct hard node-to-surface contact',
 'solver':'Abaqus Standard, static NLGEOM, no automatic stabilization or inertia',
 'changed_factor':'Only node membership/tributary area of secondary top contact surface; all solid nodes, elements, material, loading and constraints identical.',
 'cases':['cap','rim'],'new_solve_limit':2,'per_job_wall_limit_s':1450,
 'measurements':'Full reaction-displacement curve, external work, displacement, end rotation from deformed cross-section, eccentricity of contact normal force, reaction balance and penetration.',
 'comparison_grid_compression':[0,.02,201],
 'difference_denominator':'Full cap result at same displacement; RMS normalized by full-cap maximum load. Signed cap/rim difference also report relative rim at endpoint.',
 'causal_working_gate':'If full-cap force differs by >5% of rim at endpoint with matching loading, <1e-3 relative reaction imbalance and contact penetration<1e-4mm, contact support has a material effect in this benchmark. No transfer of magnitude to G.',
 'no_G_change':True,'no20pct':True,'no_new_AD':True}
write(B/'input/protocol.json',protocol)
nx,ny,nz=protocol['nx_ny_nz'];levels=np.array([2*nx+1,2*ny+1,2*nz+1])
offsets=np.array([(0,0,0),(2,0,0),(2,2,0),(0,2,0),(0,0,2),(2,0,2),(2,2,2),(0,2,2),
 (1,0,0),(2,1,0),(1,2,0),(0,1,0),(1,0,2),(2,1,2),(1,2,2),(0,1,2),(0,0,1),(2,0,1),(2,2,1),(0,2,1)])
cells_i=np.indices((nx,ny,nz)).reshape(3,-1).T*2
ijk=cells_i[:,None,:]+offsets[None,:,:];used=np.unique(ijk.reshape(-1,3),axis=0)
labels={tuple(q):i+1 for i,q in enumerate(used)}
cells=np.array([[labels[tuple(q)] for q in cell] for cell in ijk])
horizontal_width=.5*np.sqrt(1+.2**2)
z=used[:,2]/(2*nz)*10
xyz=np.c_[5+.2*(z-5)+horizontal_width*(used[:,0]/(2*nx)-.5),used[:,1]/(2*ny),z]
bottom=np.flatnonzero(used[:,2]==0)+1;top=np.flatnonzero(used[:,2]==2*nz)+1
rim=np.flatnonzero((used[:,2]==2*nz)&(used[:,0]==nx))+1
tip_left=np.flatnonzero((used[:,2]==2*nz)&(used[:,0]==0)&(used[:,1]==0))[0]+1
tip_right=np.flatnonzero((used[:,2]==2*nz)&(used[:,0]==2*nx)&(used[:,1]==0))[0]+1
tip_mid=np.flatnonzero((used[:,2]==2*nz)&(used[:,0]==nx)&(used[:,1]==0))[0]+1
nv=len(xyz);rp=nv+1;plate_nodes=list(range(nv+2,nv+6));plate_element=len(cells)+1
np.savez_compressed(B/'input/solid_mesh.npz',xyz_mm=xyz,cells=cells,indices=used,bottom=bottom,top=top,rim=rim)
def nset(name,values):return ['*Nset,nset='+name]+[','.join(map(str,values[i:i+16])) for i in range(0,len(values),16)]
common=['*Heading','r51 local end-section support benchmark; same bulk solid; static, not G','*Node']
common += [f'{i+1},'+','.join(f'{v:.16g}' for v in p) for i,p in enumerate(xyz)]
common += [f'{rp},6,.5,10']+[f'{lab},'+','.join(map(str,p)) for lab,p in zip(plate_nodes,[(0,-2,10),(0,3,10),(12,3,10),(12,-2,10)])]
common += ['*Element,type=C3D20R,elset=WALL']
for i,conn in enumerate(cells):common += [f'{i+1},'+','.join(map(str,conn[:15]))+',',','.join(map(str,conn[15:]))]
common += ['*Element,type=R3D4,elset=PLATE',f'{plate_element},'+','.join(map(str,plate_nodes)),f'*Rigid Body,ref node={rp},elset=PLATE']
for name,values in [('WALL_NODES',list(range(1,nv+1))),('BASE',bottom.tolist()),('TOP_ALL',top.tolist()),('TOP_RIM',rim.tolist()),('RP_TOP',[rp])]:common+=nset(name,values)
common += ['*Material,name=BASE_MAT','*Hyperelastic,Neo Hooke',f'{protocol["C10_MPa"]:.16g},.24','*Solid Section,elset=WALL,material=BASE_MAT',',',
 '*Surface,type=ELEMENT,name=PLATEN','PLATE,SPOS','*Surface Interaction,name=FRICTIONLESS','*Surface Behavior,pressure-overclosure=HARD',
 '*Boundary','BASE,1,3,0.','WALL_NODES,2,2,0.','RP_TOP,1,2,0.','RP_TOP,4,6,0.']
tail=['*Contact Pair,interaction=FRICTIONLESS,type=NODE TO SURFACE','END_CONTACT,PLATEN',
 '*Step,name=PRESS,nlgeom=YES,inc=1000','*Static','.005,1.,1e-8,.02','*Boundary','RP_TOP,3,3,-.2',
 '*Output,history,frequency=1','*Node Output,nset=RP_TOP','U3,RF3','*Energy Output','ALLIE,ALLSE,ALLWK,ALLAE',
 '*Output,field,frequency=1','*Node Output','U,RF','*Contact Output','CSTRESS,CDISP,CFORCE','*End Step']
for tag,contact_nodes in [('cap',top),('rim',rim)]:
    surface=['*Surface,type=NODE,name=END_CONTACT']+[f'{n},{horizontal_width/len(contact_nodes):.16g}' for n in contact_nodes]
    inp=B/'abaqus'/f'{tag}.inp';inp.write_text('\n'.join(common+surface+tail)+'\n')
    cfg={**protocol,'case':tag,'physical_nodes':nv,'labels':{'top':rp,'tip_mid':int(tip_mid),'tip_left':int(tip_left),'tip_right':int(tip_right)},
         'contact_nodes':contact_nodes.tolist(),'all_top_nodes':top.tolist(),'base_nodes':bottom.tolist(),'horizontal_width_mm':horizontal_width,
         'solid_mesh_sha256':hashlib.sha256((B/'input/solid_mesh.npz').read_bytes()).hexdigest(),'inp_sha256':hashlib.sha256(inp.read_bytes()).hexdigest()}
    write(B/'input'/f'{tag}.json',cfg)
assert np.all(np.linalg.det(np.array([[horizontal_width/nx,0,2/nz],[0,1,0],[0,0,10/nz]]))>0)
write(B/'status.json',{'stage':'prepared_two_static_same_solid_tests','active_jobs':[],'no_G_change':True,'no20pct':True,'no_new_AD':True})
# Existing launcher, one optional extraction script; no new general job runner.
p=D/'tools/run_abaqus.py';s=p.read_text();old="a.add_argument('--config',type=Path);x=a.parse_args()";new="a.add_argument('--config',type=Path);a.add_argument('--extract-script',type=Path);x=a.parse_args()"
assert s.count(old)==1;s=s.replace(old,new);old="shutil.copy2(D/'tools/extract_abaqus.py',native/'extract.py')";new="shutil.copy2(x.extract_script or D/'tools/extract_abaqus.py',native/'extract.py')";assert s.count(old)==1;p.write_text(s.replace(old,new))
print(json.dumps({'directory':str(B),'nodes':nv,'elements':len(cells),'full_cap_contact_nodes':len(top),'rim_contact_nodes':len(rim),'horizontal_width_mm':horizontal_width},indent=2))
