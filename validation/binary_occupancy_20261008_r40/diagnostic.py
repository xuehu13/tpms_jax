"""Frozen binary-occupancy diagnostic, reusing r30/r34 initial operators."""
from pathlib import Path
import os, sys, json, hashlib, time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
R = Path('/home/xuehu/projects/tpms_jax')
D = Path(__file__).resolve().parent
sys.path.insert(0, str(R))

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p, obj): p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    for name, h in read(D/'frozen_before.json').items():
        assert sha(R/name) == h, name
def replace(source, old, new, count=1):
    assert source.count(old) == count, (old, source.count(old))
    return source.replace(old, new)
def execute(source, output):
    assert not output.exists()
    output.mkdir()
    write(output/'frozen_before.json', read(D/'frozen_before.json'))
    write(output/'source_adaptation.json', {
        'compiled_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'driver_sha256': sha(Path(__file__)),
        'source_is_in_memory_adaptation_of_frozen_prior_diagnostic': True,
        'production_source_changed': False})
    exec(compile(source, str(output/'adapted_source.py'), 'exec'),
         {'__file__': str(output/'adapted_source.py'), '__name__': '__main__'})

def binary27():
    source = (R/'validation/initial_tangent_20261008_r30/diagnostic.py').read_text()
    source = replace(source, "rho=cache['rho']; qp=cache['physical_quad_points']",
        "rho=(cache['distance']*L<=.25).astype(float); qp=cache['physical_quad_points']")
    source = replace(source, "'case':'diverse_04'", "'case':'diverse_04 binary Gauss occupancy'")
    source = replace(source, "'weighted_volume_mm3':float(jnp.sum(s*w[None,:])*L**3)",
        "'weighted_volume_mm3':float(jnp.sum(s*w[None,:])*L**3), 'occupancy_volume_mm3':float(np.sum(rho*cache['JxW'])*L**3), 'binary_occupancy_at_Gauss_not_voxel_cell_center':True")
    execute(source, D/'original27')

def fixed_check():
    import numpy as np, basix
    from jax_fem.basis import get_elements
    from hyperelastic_fem import MU, KAPPA
    from surface_distance import PeriodicSurfaceDistance
    tick=time.perf_counter()
    c=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
    st=np.load(D/'original27/linear_state.npz'); row=read(D/'original27/jax_result.json')
    assert row['status']=='ok'
    sel=np.load(R/'validation/local_quadrature_20261008_r33/selected_cells.npy')
    fam,cell,_,_,degree,order=get_elements('HEX27'); el=basix.create_element(fam,cell,degree)
    origins=2*np.indices((32,)*3).reshape(3,-1).T
    local=np.rint(el.points[order]*2).astype(int)
    xyz=origins[:,None,:]+local[None,:,:]
    cells=(xyz[:,:,0]*65+xyz[:,:,1])*65+xyz[:,:,2]
    uc=st['q'][st['class_ids'][cells[sel]]]; H=st['H']
    ref0=c['physical_quad_points'][0]*32
    g0=el.tabulate(1,ref0)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*32
    lam=KAPPA-2*MU/3
    def unit(g):
        v=H+np.einsum('cni,qnj->cqij',uc,g,optimize=True)
        ep=.5*(v+v.swapaxes(-1,-2))
        return .5*lam*np.trace(ep,axis1=-2,axis2=-1)**2+MU*np.sum(ep*ep,axis=(-1,-2))
    rho0=(c['distance']*10<=.25).astype(float)
    original=float(np.sum((1e-4+(1-1e-4)*rho0[sel])*unit(g0)*c['JxW'][sel])*1000)
    surface=PeriodicSurfaceDistance(c['surface_vertices'],c['surface_triangles'])
    levels=[]
    for level in [8,12]:
        z,w=np.polynomial.legendre.leggauss(level);z=(z+1)/2;w/=2
        ref=np.array([[a,b,c] for a in z for b in z for c in z])
        wt=np.array([a*b*c for a in w for b in w for c in w])/32**3
        gd=el.tabulate(1,ref)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*32
        total=volume=0.
        for pos in range(0,len(sel),32):
            assert time.perf_counter()-tick<900, 'Fixed-check budget exceeded; no more levels'
            which=sel[pos:pos+32]
            distance=surface.query(origins[which,None,:]/64+ref[None,:,:]/32)*10
            phi=(distance<=.25).astype(float)
            v=H+np.einsum('cni,qnj->cqij',uc[pos:pos+len(which)],gd,optimize=True)
            ep=.5*(v+v.swapaxes(-1,-2))
            u=.5*lam*np.trace(ep,axis1=-2,axis2=-1)**2+MU*np.sum(ep*ep,axis=(-1,-2))
            total+=float(np.sum((1e-4+(1-1e-4)*phi)*u*wt)*1000)
            volume+=float(np.sum(phi*wt)*1000)
        levels.append({'level_per_axis':level,'energy_N_mm':total,'occupancy_volume_mm3':volume})
        print(json.dumps({'stage':'fixed_binary_field',**levels[-1]}),flush=True)
    e=levels[0]['energy_N_mm']/levels[1]['energy_N_mm']-1
    v=levels[0]['occupancy_volume_mm3']/levels[1]['occupancy_volume_mm3']-1
    write(D/'fixed_check.json',{'status':'complete', 'selected_cells':len(sel),
        'original_selected_energy_N_mm':original,'full_original_energy_N_mm':row['energy_N_mm'],
        'original_selected_occupancy_volume_mm3':float(np.sum(rho0[sel]*c['JxW'][sel])*1000),
        'levels':levels,'relative_energy_8_12':e,'relative_volume_8_12':v,
        'partial_confirm_allowed':max(abs(e),abs(v))<=.01,
        'dense12_delta_scaled_by_original_full_energy':(levels[1]['energy_N_mm']-original)/row['energy_N_mm'],
        'outside_selected_not_checked':True,'sharp_interface_convergence_certified':False,
        'wall_seconds':time.perf_counter()-tick})

def mixed12():
    cfg=read(D/'protocol.json'); post=read(D/'fixed_check.json')
    assert post['partial_confirm_allowed'], 'Frozen 8/12 gate failed: do not confirm'
    source=(R/'validation/local_reequilibrium_20261008_r34/diagnostic.py').read_text()
    source=replace(source,"old=np.load(R/'validation/initial_tangent_20261008_r30/linear_state.npz')", "old=np.load(D.parent/'original27/linear_state.npz')")
    source=replace(source,"p=make_density_hyperelastic_problem(N,eta=ETA,rho_quad=c['rho']", "rho0=(c['distance']*L<=.25).astype(float)\np=make_density_hyperelastic_problem(N,eta=ETA,rho_quad=rho0")
    source=replace(source,"rho=expit((.25-surface.query(points)*L)/ELL)", "rho=(surface.query(points)*L<=.25).astype(float)")
    source=replace(source,"c['rho'][which]", "rho0[which]")
    source=replace(source,"qr=json.loads((Q/'result.json').read_text())", "qr=json.loads((D.parent/'fixed_check.json').read_text())")
    source=replace(source,"expected_delta=qr['levels'][-1]['total_energy_N_mm']-qr['selected_original']['total_energy_N_mm']", "expected_delta=qr['levels'][-1]['energy_N_mm']-qr['original_selected_energy_N_mm']")
    source=replace(source,"old_result=json.loads((R/'validation/initial_tangent_20261008_r30/jax_result.json').read_text())", "old_result=json.loads((D.parent/'original27/jax_result.json').read_text())")
    source=replace(source,"'case':'diverse_04 N32 initial static; only frozen r33 cells integrated densely'", "'case':'diverse_04 binary occupancy; same selected-12/rest-27 rule as smooth r34'")
    source=replace(source,"'r33_fixed_state_energy_reproduced_le_1e-9'", "'binary_fixed_state_energy_reproduced_le_1e-9'")
    source=replace(source,"'r33_delta_energy_N_mm'", "'binary_selected_delta_energy_N_mm'")
    source=replace(source,"'complete20pct_or_design_AD_certified':False", "'complete20pct_or_design_AD_certified':False, 'occupancy_volume_mm3':float(np.sum(rho0*c['JxW'])*L**3)+qr['levels'][-1]['occupancy_volume_mm3']-qr['original_selected_occupancy_volume_mm3']")
    output=D/'mixed12'; output.mkdir(exist_ok=False)
    write(output/'protocol.json',cfg)
    write(output/'frozen_before.json',read(D/'frozen_before.json'))
    write(output/'source_adaptation.json', {'compiled_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'driver_sha256':sha(Path(__file__)), 'reuse':'frozen r34 operator and solver; binary-specific fixed-field validation',
        'production_source_changed':False})
    exec(compile(source,str(output/'adapted_source.py'),'exec'),{'__file__':str(output/'adapted_source.py'),'__name__':'__main__'})

if __name__=='__main__':
    freeze()
    {'binary27':binary27,'check':fixed_check,'mixed12':mixed12}[sys.argv[1]]()
    freeze()
