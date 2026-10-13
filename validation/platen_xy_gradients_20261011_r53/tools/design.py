"""Parameter adapter for existing PlatenXY mechanics; no second material/solver."""
from pathlib import Path
from types import SimpleNamespace
import sys, json
import numpy as np
R=Path(__file__).resolve().parents[3];B=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'validation/platen_xy_G_20261010_r51/tools'),str(R),str(R/'scripts')]
from run_jax import build,estimate_initial_dt,jax,jnp
from platen_model import PlatenXY

class DesignPath:
    def __init__(self,config_path='input/protocol.json',geometry_path='input/design_geometry.npz'):
        self.cfg=json.loads((B/config_path).read_text())
        self.ex=build(self.cfg);self.ex.p.fe.JxW=np.asarray(self.ex.p.fe.JxW)
        with np.load(B/geometry_path) as z:
            self.distance=jnp.asarray(z['gauss_distance']);self.cap_distance=jnp.asarray(z['cap_distance'])
            self.cap_raw_area=jnp.asarray(z['cap_raw_area']);self.assignment=jnp.asarray(z['cap_assignment'])
        self.fields=jax.jit(self._fields)
        self.contact=jax.jit(self._contact)
        self.acceleration=jax.jit(self._acceleration)
        self.initial=jax.jit(self._initial)
        self.observe=jax.jit(self._observe)
        self.diagnostics=jax.jit(self._diagnostics)

    def _fields(self,p):
        ex=self.ex;t,w=p;ell=w/ex.L/(2*jnp.log(9.))
        phi=jax.nn.sigmoid((t/(2*ex.L)-self.distance)/ell)
        scale=ex.p.eta+(1-ex.p.eta)*phi
        scale,nodem,mass=ex.material_fields(scale)
        c=jnp.zeros((ex.nc,2)).at[ex.ids].add(nodem[:,None]*ex.b)
        d=jnp.sum(nodem[:,None]*ex.b**2,axis=0)
        schur=d-jnp.sum(c**2/mass[:,None],axis=0)
        cap_phi=jax.nn.sigmoid((t/(2*ex.L)-self.cap_distance)/ell)
        area=jnp.zeros_like(ex.area).at[self.assignment].add(self.cap_raw_area*cap_phi)
        return scale,nodem,mass,c,schur,area

    def _contact(self,q,t,area):
        ex=self.ex
        proxy=SimpleNamespace(ids=ex.ids,cids=ex.cids,sh=ex.sh,z0=ex.z0,side=ex.side,
                              phi=ex.phi,points=ex.points,knL=ex.knL,L=ex.L,motion=ex.motion,area=area)
        return PlatenXY._contact(proxy,q,t)

    def _acceleration(self,q,a,t,fields):
        ex=self.ex;scale,nodem,mass,c,schur,area=fields
        geometry=(*ex.kernel_geometry[:-1],scale)
        r=ex.force_for_geometry(q,a,geometry)+self._contact(q,t,area)[0]
        proxy=SimpleNamespace(c=c,mass=mass,schur=schur)
        return PlatenXY.solve_mass(proxy,ex.reduce(r),jnp.sum(r[:,:2]*ex.b,axis=0))

    def _initial(self,fields):
        ex=self.ex;q=jnp.zeros((ex.nc,3));a=jnp.zeros(2);t=jnp.array(0.)
        aq,aa=self._acceleration(q,a,t,fields)
        return q,a,jnp.zeros_like(q),jnp.zeros_like(a),aq,aa,t,jnp.array(0.)

    def step(self,state,fields,dt):
        q,a,vq,va,aq,aa,t,work=state;ex=self.ex
        ft0=self._contact(q,t,fields[-1])[2]
        vh=vq+.5*dt*aq;ah=va+.5*dt*aa
        q=q+dt*vh;a=a+dt*ah;tn=t+dt
        nq,na=self._acceleration(q,a,tn,fields)
        ft1=self._contact(q,tn,fields[-1])[2]
        work=work+.5*(ft0+ft1)*(ex.motion(tn)-ex.motion(t))*ex.L
        return q,a,vh+.5*dt*nq,ah+.5*dt*na,nq,na,tn,work

    def block(self,steps=64,remat=False):
        def block(state,fields,dt,count):
            def one(s,k):
                return jax.lax.cond(k<count,lambda x:self.step(x,fields,dt),lambda x:x,s),None
            body=jax.checkpoint(one) if remat else one
            return jax.lax.scan(body,state,jnp.arange(steps))[0]
        return block

    def _observe(self,state,fields):
        return jnp.array([state[-1],self._contact(state[0],state[-2],fields[-1])[2]])

    def _diagnostics(self,state,fields):
        ex=self.ex;scale=fields[0];weights=ex.kernel_geometry[2];grads=ex.kernel_geometry[1]
        q,a=state[:2];H=jnp.diag(jnp.r_[a,0.]);cells=q[ex.ids][ex.device_cells]
        # Statistics only; all actual J and virtual-domain energy are retained.
        batch=ex.force_batch_cells
        def part(x):
            v,s=x;F=jnp.eye(3)+H+jnp.einsum('cni,qnj->cqij',v,grads);J=jnp.linalg.det(F)
            required=ex.p.requires_positive_J(s)
            energy=jax.vmap(ex.p.material_energy)(F.reshape((-1,3,3)),s.ravel()).reshape(s.shape)
            return jnp.array([jnp.sum(energy*weights),jnp.min(J),jnp.min(jnp.where(required,J,jnp.inf)),
                              jnp.sum(required&(J<=0)),jnp.sum(J<=0),jnp.all(jnp.isfinite(energy))])
        values=jax.lax.map(part,(cells.reshape((-1,batch,*cells.shape[1:])),scale.reshape((-1,batch,scale.shape[-1]))))
        v=ex.full_velocity(state[2],state[3]);ke=.5*jnp.sum(fields[1][:,None]*v*v)*ex.L**3
        c=self._contact(q,state[-2],fields[-1])
        return jnp.array([jnp.sum(values[:,0])*ex.L**3,jnp.min(values[:,1]),jnp.min(values[:,2]),
                          jnp.sum(values[:,3]),jnp.sum(values[:,4]),jnp.all(values[:,5]>0),ke,c[2],c[3],c[1]*ex.L**3])

    def baseline_checks(self):
        ex=self.ex;fields=self.fields(jnp.array([.5,.05]))
        expected=(ex.scale,ex.nodem,ex.mass,ex.c,ex.schur,ex.area)
        errors={name:float(jnp.max(jnp.abs(x-y))/jnp.maximum(jnp.max(jnp.abs(y)),1e-30))
                for name,x,y in zip(['scale','nodem','mass','c','schur','area'],fields,expected)}
        old=R/self.cfg.get('baseline_state_path','validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010/jax/state_05pct.npz')
        with np.load(old) as z:q=jnp.asarray(z['q']);a=jnp.asarray(z['macro']);t=jnp.asarray(z['time'])
        original=ex.acceleration_free(q,a,t);actual=self.acceleration(q,a,t,fields)
        for i,name in enumerate(['acceleration_q','acceleration_macro']):
            errors[name]=float(jnp.max(jnp.abs(original[i]-actual[i]))/jnp.maximum(jnp.max(jnp.abs(original[i])),1e-30))
        errors['pass']=all(v<1e-9 for v in errors.values())
        return errors
