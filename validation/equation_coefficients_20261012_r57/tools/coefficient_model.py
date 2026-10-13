"""True diverse_28 equation coefficients; reuse r53 DesignPath mechanics."""
from pathlib import Path
import sys,json
from types import SimpleNamespace
R=Path('/home/xuehu/projects/tpms_jax')
B=R/'validation/equation_coefficients_20261012_r57'
sys.path[:0]=[str(B/'tools'),str(R/'validation/platen_xy_gradients_20261011_r53/tools')]
from design import DesignPath,jax,jnp,np
from platen_model import PlatenXY
from coefficient_geometry import CoefficientGeometry


class CoefficientPath(DesignPath):
    def __init__(self):
        # Absolute paths deliberately avoid touching the frozen r53 directory.
        super().__init__(str(B/'input/model.json'),str(R/'validation/platen_xy_gradients_20261011_r53/input/design_geometry.npz'))
        with np.load(B/'input/queries.npz') as z:
            queries=z['points']; self.ng=int(z['n_gauss']); self.gauss_shape=tuple(z['gauss_shape'])
            self.cap_raw_area=jnp.asarray(z['cap_raw_area']); self.assignment=jnp.asarray(z['assignment'])
        self.geometry=CoefficientGeometry(R/'validation/platen_xy_diverse_batch_20261012_r55/sources/diverse_28/surface.npz',queries)
        base=self.geometry.evaluate(np.array([-3.4,4.6,2.]))
        points=base['rim_points']; n=self.cfg['N']; self.rim_side=jnp.asarray(points[:,2].astype(int))
        self.rim_z=jnp.asarray(points[:,2]); self.local_cells=jnp.asarray(np.minimum(np.floor(points[:,:2]*n).astype(int),n-1))
        nn=2*n+1; offsets=np.indices((3,3)).reshape(2,-1).T
        ij=2*np.asarray(self.local_cells)[:,None,:]+offsets
        self.rim_ids=jnp.asarray((ij[:,:,0]*nn+ij[:,:,1])*nn+np.rint(points[:,2]*2*n).astype(int)[:,None])
        out=jax.ShapeDtypeStruct((len(queries)+len(points)*3,),jnp.float64)
        jacout=jax.ShapeDtypeStruct((len(queries)+len(points)*3,3),jnp.float64)
        def host(p,jac=False):
            d=self.geometry.evaluate(np.asarray(p))
            packed=np.r_[d['distance'],d['rim_points'].ravel()]
            return (packed,np.vstack([d['distance_jac'],d['rim_jac'].reshape((-1,3))])) if jac else packed
        @jax.custom_jvp
        def geometry(p):return jax.pure_callback(host,out,p)
        @geometry.defjvp
        def geometry_jvp(primals,tangents):
            p,=primals;dp,=tangents
            y,J=jax.pure_callback(lambda p:host(p,True),(out,jacout),p)
            return y,J@dp
        self.geometry_field=geometry

    def _fields(self,p):
        packed=self.geometry_field(p);distance=packed[:self.ng].reshape(self.gauss_shape)
        cap_distance=packed[self.ng:len(self.geometry.queries)]
        points=packed[len(self.geometry.queries):].reshape((-1,3))
        proxy=SimpleNamespace(ex=self.ex,distance=distance,cap_distance=cap_distance,
                              cap_raw_area=self.cap_raw_area,assignment=self.assignment)
        fields=DesignPath._fields(proxy,jnp.array([.5,.05]))
        local=points[:,:2]*self.cfg['N']-self.local_cells
        def basis(x):return jnp.stack([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)],axis=-1)
        sh=jnp.einsum('ni,nj->nij',basis(local[:,0]),basis(local[:,1])).reshape((-1,9))
        return (*fields[:-1],(fields[-1],sh))

    def _contact(self,q,t,data):
        area,sh=data;ex=self.ex
        proxy=SimpleNamespace(ids=ex.ids,cids=self.rim_ids,sh=sh,z0=self.rim_z,side=self.rim_side,
            phi=jnp.ones_like(area),points=ex.points,knL=ex.knL,L=ex.L,motion=ex.motion,area=area)
        return PlatenXY._contact(proxy,q,t)

    def partition_audit(self,p):
        from scipy.spatial import cKDTree
        g=self.geometry.evaluate(np.asarray(p));points=g['rim_points'];n=self.cfg['N']
        cells=np.minimum(np.floor(points[:,:2]*n).astype(int),n-1)
        if not np.array_equal(cells,np.asarray(self.local_cells)):raise ValueError('Rim crossed fixed background-face cell partition')
        new_assignment=np.zeros(len(self.geometry.queries)-self.ng,dtype=int)
        cap=self.geometry.queries[self.ng:]
        for side in (0,1):
            ids=np.where(points[:,2]==side)[0];mask=cap[:,2]==side
            _,index=cKDTree(points[ids,:2]).query(cap[mask,:2]);new_assignment[mask]=ids[index]
        changed=int(np.sum(new_assignment!=np.asarray(self.assignment)))
        return dict(changed_nearest_rim_assignments=changed,background_cells_unchanged=True,**g['audit'])
