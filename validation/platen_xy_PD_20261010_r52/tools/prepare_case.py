"""P/D input adapter only; original geometry, G input and mechanics stay read-only."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit

B = Path(__file__).resolve().parents[1]
R = B.parents[1]
G = R/'validation/platen_xy_G_20261010_r51'
GE = G/'experiments/midsurface_rim_20pct_20261010'
sys.path[:0] = [str(G/'tools'), str(R)]
from prepare import face_cache
from contact_origin import shapes_at
from surface_distance import PeriodicSurfaceDistance

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['P','D']);a=ap.parse_args()
    E=B/a.kind;assert not (E/'input/protocol.json').exists()
    clock=json.loads((E/'checks/execution_clock.json').read_text())
    old=R/'validation/canonical_tpms_20261009_r50'/a.kind
    audit=json.loads((B/'checks/geometry_precheck.json').read_text())[a.kind]
    for name, digest in audit['source_sha256'].items():assert sha(old/name)==digest,(a.kind,name)
    for sub in ['input','abaqus','checks']:(E/sub).mkdir(parents=True,exist_ok=True)
    shell=Path(audit['shell_surface'])
    with np.load(shell) as z:v=z['surface_vertices'];faces=z['surface_triangles']
    with np.load(old/'surface.npz') as z:distance=PeriodicSurfaceDistance(z['surface_vertices'],z['surface_triangles'])
    ell=.05/10/(2*np.log(9))
    cap=face_cache(32, lambda x:expit((.5/20-distance.query(x))/ell))
    np.savez_compressed(E/'input/cut_face_contact.npz',**cap)
    pts=[];weights=[];sides=[];labels=[];assign=np.zeros(len(cap['points']),int);distances=np.zeros(len(assign))
    for s in (0,1):
        mask=cap['side']==s;rim=np.flatnonzero(np.isclose(v[:,2],s,atol=1e-10))
        dist,idx=cKDTree(v[rim,:2]).query(cap['points'][mask,:2])
        weight=np.bincount(idx,weights=cap['area_weight'][mask],minlength=len(rim))
        assign[mask]=len(pts)+idx;distances[mask]=dist
        pts.extend(v[rim]);weights.extend(weight);sides.extend([s]*len(rim));labels.extend((rim+1).tolist())
    pts=np.asarray(pts);ids,sh=shapes_at(pts,32)
    rimcache=dict(points=pts,node_ids=ids,shape=sh,area_weight=np.asarray(weights),side=np.asarray(sides),
                  phi=np.ones(len(pts)),shell_node_labels=np.asarray(labels))
    areas=[]
    for s in (0,1):
        original=cap['area_weight'][cap['side']==s].sum();actual=rimcache['area_weight'][rimcache['side']==s].sum()
        np.testing.assert_allclose(actual,original,rtol=1e-13)
        areas.append(float(actual*100))
    contact=E/'input/rimline_contact.npz';np.savez_compressed(contact,**rimcache)
    np.savez_compressed(E/'input/contact_assignment.npz',rim_assignment=assign,distance_normalized=distances)
    groups={}
    for i,x in enumerate(v):
        key=tuple(np.round(np.r_[np.where(np.isclose(x[:2],1,atol=1e-10),0,x[:2]),x[2]],9));groups.setdefault(key,[]).append(i)
    rel=[]
    for nodes in groups.values():
        root=min(nodes)
        rel.extend((i,root,np.rint(v[i,:2]-v[root,:2]).astype(int)) for i in nodes if i!=root)
    assert len(rel)==audit['xy_independent_relations']
    slaves={i for i,j,s in rel};assert len(slaves)==len(rel) and not slaves & {j for i,j,s in rel}
    for i,j,s in rel:np.testing.assert_allclose(v[i]-v[j],np.r_[s,0],atol=1e-9)
    nv=len(v);ax,ay,bottom,top=range(nv+1,nv+5)
    nsets={'PHYSICAL':list(range(1,nv+1)),'RP_X':[ax],'RP_Y':[ay],'RP_BOTTOM':[bottom],'RP_TOP':[top],
           'RIM_BOTTOM':(np.flatnonzero(np.isclose(v[:,2],0,atol=1e-10))+1).tolist(),
           'RIM_TOP':(np.flatnonzero(np.isclose(v[:,2],1,atol=1e-10))+1).tolist()}
    corners=[[-5,-5,0],[15,-5,0],[15,15,0],[-5,15,0],[-5,-5,10],[-5,15,10],[15,15,10],[15,-5,10]]
    node_lines=[f'{i+1}, '+', '.join(f'{x:.16g}' for x in p*10) for i,p in enumerate(v)]
    node_lines += [f'{ax},15,5,5',f'{ay},5,15,5',f'{bottom},5,5,0',f'{top},5,5,10']
    node_lines += [f'{nv+5+i}, '+','.join(map(str,x)) for i,x in enumerate(corners)]
    equations=[]
    for i,j,s in rel:
        for dof in range(1,7):
            terms=f'{i+1},{dof},1.,{j+1},{dof},-1.';nt=2
            if dof in (1,2) and s[dof-1]:terms+=f',{ax if dof==1 else ay},{dof},{-int(s[dof-1])}.';nt=3
            equations += ['*Equation',str(nt),terms]
    # Reuse all non-geometry cards verbatim from the completed G20 input.
    template=GE/'abaqus/platen_20pct.inp';blocks=[]
    for line in template.read_text().splitlines():
        if line.startswith('*') and not line.startswith('**'):blocks.append([line])
        else:blocks[-1].append(line)
    lines=[];inserted=False
    for block in blocks:
        keyword=block[0].lower()
        if keyword.startswith('*equation'):continue
        if keyword.startswith('*heading'):block=[block[0],f'r52 {a.kind}; same G20 controls; midsurface end contact; free XY']
        elif keyword=='*node':block=[block[0]]+node_lines
        elif keyword.split(',')[0]=='*element':
            if 'type=s3r' in keyword:block=[block[0]]+[f'{i+1}, '+','.join(str(j+1) for j in f) for i,f in enumerate(faces)]
            else:
                s=0 if 'plate_bottom' in keyword else 1
                block=[block[0],f'{len(faces)+s+1}, '+','.join(str(nv+5+4*s+i) for i in range(4))]
        elif keyword.startswith('*rigid body'):
            s=0 if 'plate_bottom' in keyword else 1
            block=[f'*Rigid Body, ref node={bottom if s==0 else top}, elset=PLATE_{"BOTTOM" if s==0 else "TOP"}']
        elif keyword.startswith('*nset'):
            name=block[0].split('nset=')[1].strip();ns=nsets[name]
            block=[block[0]]+[','.join(map(str,ns[i:i+16])) for i in range(0,len(ns),16)]
        if keyword.startswith('*surface') and not inserted:lines+=equations;inserted=True
        lines+=block
    text='\n'.join(lines)+'\n';source=template.read_text()
    # Loads/contact/output controls are an exact copy, not a separately tuned recipe.
    control_marker='*Surface, type=NODE, name=END_BOTTOM'
    assert text.split(control_marker,1)[1]==source.split(control_marker,1)[1]
    assert '*Shell Section, elset=WALL, material=BASE\n.5,5' in text
    assert text.count('*Equation')==6*len(rel) and 'RP_TOP,3,3,-2.' in text
    inp=E/'abaqus/platen_20pct.inp';inp.write_text(text)
    cfg=json.loads((GE/'input/protocol.json').read_text())
    for k in ['execution_plan','source_inp_sha256']:cfg.pop(k,None)
    cfg.update(case_id=f'{a.kind}_midsurface_end_platen_20pct_r52',kind=a.kind,
        stage=f'{a.kind} simplified midsurface-end platen20pct; not finite-width/self-contact or AD certification',
        original_Gauss_field=str(old/'gauss_field.npz'),original_Gauss_sha256=sha(old/'gauss_field.npz'),
        source_surface_sha256=sha(old/'surface.npz'),source_shell_surface_sha256=sha(shell),source_shell_surface=str(shell),
        contact_reference_area_mm2=areas,physical_nodes=nv,shell_elements=len(faces),XY_relations=len(rel),XY_equations=6*len(rel),
        labels=dict(ax=ax,ay=ay,bottom=bottom,top=top),rim_node_count=[len(nsets['RIM_BOTTOM']),len(nsets['RIM_TOP'])],
        contact_cache_path=str(contact),contact_cache_sha256=sha(contact),inp_sha256=sha(inp),
        template_G20_INP_sha256=sha(template),end_support_selection='all z=0/L midsurface nodes, including D tangencies; same as G',
        shell_contact='node-based midsurface end nodes against rigid planes; zero contact thickness; section .5 retained',
        jax_contact='own occupied cap weights aggregated at nearest matching shell end nodes; Q2 force distribution, reference area andkn conserved',
        execution_authorized=True,budget_seconds=json.loads((B/'protocol_planned.json').read_text())['per_geometry_budget_seconds'])
    write(E/'input/protocol.json',cfg)
    write(E/'checks/input_audit.json',{'kind':a.kind,'xy_pairs':len(rel),'unique_slaves':len(slaves),'Z_periodic_pairs':0,
        'macro_affine_identity_verified':True,'no_wall_pin':True,'G20_contact_load_output_tail_identical':True,
        'contact_area_mm2':areas,'rim_nodes':cfg['rim_node_count'],'area_conserved_per_end':True,
        'occupied_area_weighted_mapping_distance_mm':float(np.sum(cap['area_weight']*distances)/np.sum(cap['area_weight'])*10),
        'sources_unchanged':all(sha(old/n)==h for n,h in audit['source_sha256'].items()),
        'preparation_wall_seconds':time.time()-clock['started_unix']})
    print(json.dumps({'kind':a.kind,'stage':'input_prepared','area_mm2':areas,'rim_nodes':cfg['rim_node_count'],
                      'preparation_s':time.time()-clock['started_unix']}),flush=True)

if __name__=='__main__':main()
