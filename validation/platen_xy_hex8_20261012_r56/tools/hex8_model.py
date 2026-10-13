"""HEX8-only thin adapter; all material, contact and time equations remain shared."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
B=Path(__file__).resolve().parents[1]; R=B.parents[1]
G=R/'validation/platen_xy_G_20261010_r51'
sys.path[:0]=[str(R),str(R/'scripts'),str(G/'tools')]
import jax
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem

source=G/'tools/platen_model.py'
original=source.read_text()
old="n=cfg['N'];p.class_ids,_=periodic_node_classes(p.fe.points,2*n,2*n,2*n,periodic_axes=(0,1))"
new="n=cfg['N']*p.element_degree;p.class_ids,_=periodic_node_classes(p.fe.points,n,n,n,periodic_axes=(0,1))"
assert original.count(old)==1
ns={'__name__':'r56_shared_platen','__file__':str(source)}
exec(compile(original.replace(old,new),str(source),'exec'),ns)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def interpolate(points,p):
    """Q1 interpolation using actual mesh connectivity, independent of node numbering."""
    points=np.asarray(points); n=round(p.fe.num_cells**(1/3))
    cells=np.asarray(p.fe.cells); xyz=np.asarray(p.fe.points)[cells]
    origins=xyz.min(axis=1); keys=np.rint(origins*n).astype(int)
    lookup=np.empty((n,n,n),int);lookup[tuple(keys.T)]=np.arange(len(cells))
    ix=np.clip(np.floor(points*n+1e-12).astype(int),0,n-1)
    ci=lookup[tuple(ix.T)];local=(points-origins[ci])*n
    corners=np.rint((xyz[ci]-origins[ci,None,:])*n).astype(int)
    assert np.isin(corners,[0,1]).all() and np.all(local>=-1e-10) and np.all(local<=1+1e-10)
    shape=np.prod(np.where(corners==1,local[:,None,:],1-local[:,None,:]),axis=2)
    ids=cells[ci]
    np.testing.assert_allclose(shape.sum(1),1,atol=1e-13)
    np.testing.assert_allclose(np.einsum('cn,cnj->cj',shape,np.asarray(p.fe.points)[ids]),points,atol=1e-13)
    return ids,shape

class PlatenXY8(ns['PlatenXY']):
    def __init__(self,p,cfg,contact):
        assert p.element_degree==1 and p.fe.num_quads==8
        super().__init__(p,cfg,contact)
        # The parent compact path is Q2-only. Verify and establish the same
        # installed JAX-FEM cell kernel on regular Q1 Cartesian reference maps.
        grad0=np.asarray(p.shape_grads[0]);weight0=np.asarray(p.JxW[0])
        ge=we=0.;norm=np.max(abs(grad0))
        for s in range(0,p.fe.num_cells,512):
            ge=max(ge,float(np.max(abs(np.asarray(p.shape_grads[s:s+512])-grad0)))/norm)
            we=max(we,float(np.max(abs(np.asarray(p.JxW[s:s+512])/weight0-1))))
        assert max(ge,we)<5e-12
        self.common_geometry_relative_errors={'gradient':ge,'weight':we}
        self.kernel_geometry=(jax.device_put(p.physical_quad_points,jax.devices()[0]),jnp.asarray(grad0),jnp.asarray(weight0),jnp.asarray(p.v_grads_JxW[0]),self.scale)
        self.device_cells=jnp.asarray(p.fe.cells)
        self.force_for_geometry=jax.jit(self._force)
        self.force=lambda q,a:self.force_for_geometry(q,a,self.kernel_geometry)

def build(cfg):
    with np.load(cfg['original_Gauss_field']) as z:rho=z['rho'];qp=z['physical_quad_points'];qw=z['JxW']
    def field(p):
        np.testing.assert_array_equal(p.physical_quad_points,qp)
        np.testing.assert_array_equal(p.fe.JxW,qw)
        return rho
    with jax.default_device(jax.devices('cpu')[0]):
        p=make_density_hyperelastic_problem(cfg['N'],rho_quad=field,eta=cfg['eta'],periodic_axes=(0,1,2),element_degree=1,quadrature_order=2,material_model='objective_void')
    with np.load(cfg['contact_cache_path']) as z:contact=dict(z)
    return PlatenXY8(p,cfg,contact)

def preflight(p,ex,contact):
    n=ex.cfg['N'];ids=np.asarray(ex.ids)
    assert len(p.fe.points)==(n+1)**3 and ex.nc==n*n*(n+1)
    assert ex.mass_lumping=='row_sum' and ex.mass_conservation_error<1e-12
    assert np.min(ex.mass)>0
    # All opposite XY face points share classes; Z ends remain distinct.
    points=np.asarray(p.fe.points);lookup={tuple(np.rint(x*n).astype(int)):i for i,x in enumerate(points)}
    for axis in (0,1):
        for i in np.flatnonzero(np.isclose(points[:,axis],0)):
            key=np.rint(points[i]*n).astype(int);key[axis]=n
            assert ids[i]==ids[lookup[tuple(key)]]
    for i in np.flatnonzero(np.isclose(points[:,2],0)):
        key=np.rint(points[i]*n).astype(int);key[2]=n
        assert ids[i]!=ids[lookup[tuple(key)]]
    np.testing.assert_allclose(np.einsum('cn,cnj->cj',contact['shape'],points[contact['node_ids']]),contact['points'],atol=1e-13)
    # Shared compact kernel against the installed general residual, a nonzero
    # affine deformation represented through nodal Q1 interpolation.
    reps=np.unique(ids,return_index=True)[1];q=jnp.asarray(points[reps]*np.array([0,0,-1e-4]));a=jnp.array([3e-5,3e-5])
    H=jnp.diag(jnp.r_[a,0.]);direct=p.compute_residual_vars([q[ex.ids]],[jnp.broadcast_to(H,(*ex.scale.shape,3,3)),ex.scale],[])[0]
    compact=ex.force(q,a);jax.block_until_ready(compact)
    residual=float(jnp.linalg.norm(direct-compact)/jnp.maximum(jnp.linalg.norm(direct),1e-30))
    assert residual<1e-10
    # Q1 affine and translation reproduction, independent of material sampling.
    nodal=points*np.array([3e-5,3e-5,-1e-4])
    expected=np.diag([3e-5,3e-5,-1e-4])
    affine=np.einsum('cni,cqnj->cqij',nodal[np.asarray(p.fe.cells)],np.asarray(p.shape_grads))
    affine_error=float(np.max(abs(affine-expected)));assert affine_error<1e-12
    assert np.max(abs(np.asarray(p.shape_grads).sum(axis=2)))<1e-10
    rng=np.random.default_rng(56);fq=jnp.asarray(rng.normal(size=(ex.nc,3)));fa=jnp.asarray(rng.normal(size=2))
    sq,sa=ex.solve_mass(fq,fa);vf=ex.full_velocity(sq,sa)
    mr=ex.reduce(ex.nodem[:,None]*vf);ma=jnp.sum(ex.nodem[:,None]*vf[:,:2]*ex.b,axis=0)
    mass_residual=float(jnp.sqrt(jnp.sum((mr+fq)**2)+jnp.sum((ma+fa)**2))/jnp.sqrt(jnp.sum(fq**2)+jnp.sum(fa**2)))
    assert mass_residual<1e-10
    cq=jnp.zeros_like(q).at[:,2].set(1e-5*(jnp.asarray(points[reps,2])-.5));direction=jnp.asarray(rng.normal(size=q.shape));eps=1e-9
    cr=ex.contact(cq,0.)[0];virtual=float(jnp.sum(ex.reduce(cr)*direction))
    fd=float((ex.contact(cq+eps*direction,0.)[1]-ex.contact(cq-eps*direction,0.)[1])/(2*eps))
    work_error=abs(virtual-fd)/max(abs(virtual),1e-30);assert work_error<1e-6
    zero=ex.observe_free(ex.initial_state());assert abs(zero['energy_N_mm'])<1e-9 and zero['force_N']==0
    return {'pass':True,'compact_general_residual_relative_difference':residual,'affine_gradient_max_error':affine_error,
            'projected_mass_relative_residual':mass_residual,'contact_virtual_work_relative_error':work_error,'XY_classes':ex.nc,'zero_state':zero,
            'common_geometry_relative_errors':ex.common_geometry_relative_errors,'mass_conservation_error':ex.mass_conservation_error,
            'positive_lumped_mass':True,'XY_face_classes_equal':True,'Z_faces_not_periodic':True,
            'node_count':len(points),'Gauss_points_per_cell':p.fe.num_quads,'mass_lumping':ex.mass_lumping,
            'shared_platen_sha256':sha(source),'periodic_source_substitution':[old,new]}
