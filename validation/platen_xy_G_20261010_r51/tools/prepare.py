"""G-only input adapter. Original r50 geometry and material stay read-only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.special import expit
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from surface_distance import PeriodicSurfaceDistance

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def face_cache(n,phi=None):
    """Q2 material cut-face quadrature; no void stiffness floor in contact."""
    g=np.array([.5-np.sqrt(15)/10,.5,.5+np.sqrt(15)/10]);w=np.array([5/18,4/9,5/18])
    shape=lambda x:np.array([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)])
    ids=[];pts=[];sh=[];wa=[];sides=[];nn=2*n+1
    for side,iz in enumerate([0,2*n]):
        for ix in range(n):
            for iy in range(n):
                nodes=np.array([((2*ix+i)*nn+2*iy+j)*nn+iz for i in range(3) for j in range(3)])
                for i,x in enumerate(g):
                    for j,y in enumerate(g):
                        ids.append(nodes);pts.append([(ix+x)/n,(iy+y)/n,side])
                        sh.append(np.outer(shape(x),shape(y)).ravel());wa.append(w[i]*w[j]/n**2);sides.append(side)
    pts=np.array(pts);rho=np.ones(len(pts)) if phi is None else phi(pts)
    return dict(node_ids=np.array(ids),points=pts,shape=np.array(sh),area_weight=np.array(wa)*rho,phi=rho,side=np.array(sides))

def main():
    out=D/'input';cfgp=out/'protocol.json'
    if cfgp.exists():raise FileExistsError('Input already prepared; do not overwrite a frozen input')
    old=R/'validation/canonical_tpms_20261009_r50/G'
    cfg={'case_id':'G_platen_xy_r51','L_mm':10.,'thickness_mm':.5,'E_MPa':10.,'nu':.3,
         'density_tonne_mm3':1e-9,'eta':1e-4,'interface_10_90_mm':.05,'N':32,
         'contact_normal_stiffness_MPa_per_mm':320.,'contact_stiffness_rule':'10*E/background_h, selected before response',
         'target_compression':.02,'load_time_s':.001,'hold_time_s':.0002,
         'stage':'2pct cut-end prototype, not full-wall contact or 20pct certification',
         'lateral_macro':'free ax/ay, zero macro shear; dynamics includes projected mass coupling',
         'material_model':'objective_void','no_friction':True,'no_self_contact':True,'no_mass_scaling':True,
         'shell_contact':'node-based cut rims against rigid planes; zero contact thickness by node-based pair definition; section thickness .5 retained',
         'jax_contact':'reference cut-face quadrature weighted by continuous occupancy, no eta contact floor; constant plane normals',
         'remaining_difference':'shell rim discretization vs diffuse solid cap; different contact penalties and discrete mass, not identical models',
         'source_surface_sha256':sha(old/'surface.npz'),'source_shell_surface_sha256':sha(old/'surface_shell_quality.npz'),
         'original_Gauss_field':str(old/'gauss_field.npz'),'original_Gauss_sha256':sha(old/'gauss_field.npz')}
    with np.load(old/'surface.npz') as z:distance=PeriodicSurfaceDistance(z['surface_vertices'],z['surface_triangles'])
    ell=.05/10/(2*np.log(9));cache=face_cache(32,lambda x:expit((.5/20-distance.query(x))/ell))
    np.savez_compressed(out/'cut_face_contact.npz',**cache)
    cfg['contact_reference_area_mm2']=[float(cache['area_weight'][cache['side']==s].sum()*100) for s in [0,1]]
    with np.load(old/'surface_shell_quality.npz') as z:v=z['surface_vertices'];f=z['surface_triangles']
    groups={}
    for i,x in enumerate(v):
        key=tuple(np.round(np.r_[np.where(np.isclose(x[:2],1,atol=1e-10),0,x[:2]),x[2]],9));groups.setdefault(key,[]).append(i)
    rel=[]
    for g in groups.values():
        root=min(g)
        rel += [(i,root,np.rint(v[i,:2]-v[root,:2]).astype(int)) for i in g if i!=root]
    assert len(rel)==341
    slaves={i for i,_,_ in rel};assert not slaves & {j for _,j,_ in rel}
    for i,j,s in rel:np.testing.assert_allclose(v[i]-v[j],np.r_[s,0],atol=1e-9)
    rims=[np.where(np.isclose(v[:,2],s,atol=1e-10))[0]+1 for s in [0,1]]
    nv=len(v);ax,ay,bottom,top=range(nv+1,nv+5)
    lines=['*Heading','r51 G cut-end prototype; XY free lateral; frictionless rigid platens; NH .5mm','*Node']
    lines += [f'{i+1}, '+', '.join(f'{x:.16g}' for x in p*10) for i,p in enumerate(v)]
    lines += [f'{ax},15,5,5',f'{ay},5,15,5',f'{bottom},5,5,0',f'{top},5,5,10']
    # Two massless, prescribed rigid square facets. Top normal points downward.
    corners=[[-5,-5,0],[15,-5,0],[15,15,0],[-5,15,0],[-5,-5,10],[-5,15,10],[15,15,10],[15,-5,10]]
    lines += [f'{nv+5+i}, '+','.join(map(str,x)) for i,x in enumerate(corners)]
    lines += ['*Element, type=S3R, elset=WALL']
    lines += [f'{i+1}, '+','.join(str(j+1) for j in face) for i,face in enumerate(f)]
    lines += ['*Element, type=R3D4, elset=PLATE_BOTTOM',f'{len(f)+1}, '+','.join(str(nv+5+i) for i in range(4)),
              '*Element, type=R3D4, elset=PLATE_TOP',f'{len(f)+2}, '+','.join(str(nv+9+i) for i in range(4)),
              f'*Rigid Body, ref node={bottom}, elset=PLATE_BOTTOM',f'*Rigid Body, ref node={top}, elset=PLATE_TOP']
    def nset(name,ids):
        lines.append('*Nset, nset='+name)
        lines.extend(','.join(map(str,ids[i:i+16])) for i in range(0,len(ids),16))
    for name,ids in [('PHYSICAL',list(range(1,nv+1))),('RP_X',[ax]),('RP_Y',[ay]),('RP_BOTTOM',[bottom]),('RP_TOP',[top]),('RIM_BOTTOM',rims[0].tolist()),('RIM_TOP',rims[1].tolist())]:nset(name,ids)
    lines += ['*Material, name=BASE','*Density','1e-9','*Hyperelastic, Neo Hooke','1.923076923076923, .24',
              '*Shell Section, elset=WALL, material=BASE','.5,5']
    for i,j,s in rel:
        for dof in range(1,7):
            terms=f'{i+1},{dof},1.,{j+1},{dof},-1.';nt=2
            if dof in [1,2] and s[dof-1]:terms+=f',{ax if dof==1 else ay},{dof},{-int(s[dof-1])}.';nt=3
            lines += ['*Equation',str(nt),terms]
    lines += ['*Surface, type=NODE, name=END_BOTTOM','RIM_BOTTOM,','*Surface, type=NODE, name=END_TOP','RIM_TOP,',
              '*Surface, type=ELEMENT, name=PLATEN_BOTTOM','PLATE_BOTTOM,SPOS','*Surface, type=ELEMENT, name=PLATEN_TOP','PLATE_TOP,SPOS',
              '*Surface Interaction, name=FRICTIONLESS','*Surface Behavior, pressure-overclosure=HARD',
              '*Boundary','RP_BOTTOM,1,6,0.','RP_TOP,1,2,0.','RP_TOP,4,6,0.']
    # No wall node pin: horizontal net force is zero, initial rigid translations zero.
    # The two active macro DOFs remain free; unused dummy-node DOFs are inactive.
    T=cfg['load_time_s'];end=T+cfg['hold_time_s'];disp=-10*cfg['target_compression']
    lines += ['*Amplitude, name=LOAD, definition=SMOOTH STEP',f'0.,0.,{T},1.,{end},1.',
              '*Step, name=PLATEN, nlgeom=YES','*Dynamic, Explicit',f', {end}',
              '*Contact Pair, interaction=FRICTIONLESS, mechanical constraint=PENALTY','END_BOTTOM,PLATEN_BOTTOM','END_TOP,PLATEN_TOP',
              '*Boundary, amplitude=LOAD',f'RP_TOP,3,3,{disp}',
              '*Output, field, number interval=12','*Node Output','U,UR,RF','*Element Output, elset=WALL','S,LE,STH',
              '*Output, history, time interval=2e-6','*Energy Output','ALLIE,ALLSE,ALLKE,ALLAE,ALLVD,ALLWK,ETOTAL',
              '*Node Output, nset=RP_TOP','U3,RF3','*Node Output, nset=RP_BOTTOM','U3,RF3',
              '*Node Output, nset=RP_X','U1','*Node Output, nset=RP_Y','U2','*End Step']
    inp=D/'abaqus/platen_2pct.inp';inp.write_text('\n'.join(lines)+'\n')
    cfg.update(physical_nodes=nv,shell_elements=len(f),XY_relations=len(rel),XY_equations=6*len(rel),
               labels=dict(ax=ax,ay=ay,bottom=bottom,top=top),rim_node_count=[len(x) for x in rims],
               inp_sha256=sha(inp),wall_pin=None)
    write(cfgp,cfg);write(D/'checks/input_audit.json',{'XY_pairs':len(rel),'unique_slaves':len(slaves),'Z_periodic_pairs':0,
        'macro_affine_identity_verified':True,'no_wall_pin':True,'r50_original_unchanged':True,'contact_area_mm2':cfg['contact_reference_area_mm2']})
    write(D/'status.json',{'stage':'input_prepared','datacheck':'not_run','jax':'implementation_pending','full20_submitted':False})
    print(json.dumps(cfg,indent=2))
if __name__=='__main__':main()
