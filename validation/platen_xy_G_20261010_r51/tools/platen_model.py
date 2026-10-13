"""Experiment-specific XY/contact adapter; shared production material/residual."""
from pathlib import Path
import math,sys
import numpy as np
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R),str(R/'scripts')]
import jax
import jax.numpy as jnp
from thin_target_explicit import ExplicitXYZ
from pbc import periodic_node_classes
jax.config.update('jax_enable_x64',True)

class PlatenXY(ExplicitXYZ):
    def __init__(self,p,cfg,contact):
        n=cfg['N'];p.class_ids,_=periodic_node_classes(p.fe.points,2*n,2*n,2*n,periodic_axes=(0,1))
        p.fixed_class_id=None
        super().__init__(p,L=cfg['L_mm'],density=cfg['density_tonne_mm3'])
        self.cfg=cfg
        self.b=self.points[:,:2]-.5
        self.c=jnp.zeros((self.nc,2)).at[self.ids].add(self.nodem[:,None]*self.b)
        self.d=jnp.sum(self.nodem[:,None]*self.b**2,axis=0)
        self.schur=self.d-jnp.sum(self.c**2/self.mass[:,None],axis=0)
        if not np.all(np.asarray(self.schur)>0):raise ValueError('Singular projected macro inertia')
        self.cids=jnp.asarray(contact['node_ids']);self.sh=jnp.asarray(contact['shape'])
        self.area=jnp.asarray(contact['area_weight']);self.side=jnp.asarray(contact['side'])
        self.z0=jnp.asarray(contact['points'][:,2]);self.phi=jnp.asarray(contact['phi'])
        self.knL=cfg['contact_normal_stiffness_MPa_per_mm']*self.L
        self.contact=jax.jit(self._contact)
        self.acceleration_free=jax.jit(self._acceleration_free)
        self.stats_free=jax.jit(self._stats_free)

    def _force(self,q,a,geometry=None):
        H=jnp.diag(jnp.r_[a,0.]);geometry=self.kernel_geometry if geometry is None else geometry
        coords,grads,weights,vgrads,scale=geometry
        cells=q[self.ids][self.p.fe.cells].reshape((self.p.fe.num_cells,-1))
        batch=math.gcd(self.p.fe.num_cells,self.force_batch_cells)
        inputs=[cells,coords,jnp.broadcast_to(H,(*scale.shape,3,3)),scale]
        packed=[x.reshape((-1,batch,*x.shape[1:])) for x in inputs]
        def kernel(x):
            return self.p.kernel(x[0],x[1],jnp.broadcast_to(grads,(batch,*grads.shape)),
                jnp.broadcast_to(weights,(batch,*weights.shape)),jnp.broadcast_to(vgrads,(batch,*vgrads.shape)),x[2],x[3])
        values=jax.lax.map(kernel,packed).reshape(cells.shape)
        return self.p.compute_residual_vars_helper(values,[])[0]

    def motion(self,t):
        s=jnp.clip(t/self.cfg['load_time_s'],0.,1.)
        return self.cfg['target_compression']*(10*s**3-15*s**4+6*s**5)

    def _contact(self,q,t):
        z=self.z0+jnp.sum(q[self.ids][self.cids,2]*self.sh,axis=1)
        gap=jnp.where(self.side==0,z,1-self.motion(t)-z)
        pen=jnp.maximum(-gap,0.)
        sign=jnp.where(self.side==0,-1.,1.)
        f=self.knL*self.area*pen
        r=jnp.zeros_like(self.points).at[self.cids.ravel(),2].add((sign[:,None]*f[:,None]*self.sh).ravel())
        Uc=.5*self.knL*jnp.sum(self.area*pen**2)
        ft=jnp.sum(jnp.where(self.side==1,f,0.))*self.L**2
        fb=jnp.sum(jnp.where(self.side==0,f,0.))*self.L**2
        pmax=jnp.array([jnp.max(jnp.where((self.side==s)&(self.phi>=.01),pen,0.))*self.L for s in [0,1]])
        return r,Uc,ft,fb,pmax

    def solve_mass(self,fq,fa):
        """Exact Schur solve of B.T M B, not a diagonal macro mass guess."""
        aa=(-fa+jnp.sum(self.c*fq[:,:2]/self.mass[:,None],axis=0))/self.schur
        aq=-fq/self.mass[:,None]
        aq=aq.at[:,:2].add(-self.c*aa/self.mass[:,None])
        return aq,aa

    def _acceleration_free(self,q,a,t):
        r=self.force(q,a)+self._contact(q,t)[0]
        return self.solve_mass(self.reduce(r),jnp.sum(r[:,:2]*self.b,axis=0))

    def full_velocity(self,vq,va):
        return vq[self.ids]+jnp.pad(self.b*va,((0,0),(0,1)))

    def _stats_free(self,q,a):
        H=jnp.diag(jnp.r_[a,0.]);grads=self.kernel_geometry[1];weights=self.kernel_geometry[2]
        cells=q[self.ids][self.device_cells];scale=self.scale
        batch=math.gcd(self.p.fe.num_cells,self.force_batch_cells)
        packed=[x.reshape((-1,batch,*x.shape[1:])) for x in [cells,scale]]
        def part(x):
            F=jnp.eye(3)+H+jnp.einsum('cni,qnj->cqij',x[0],grads)
            J=jnp.linalg.det(F);required=self.p.requires_positive_J(x[1])
            W=jax.vmap(self.p.material_energy)(F.reshape((-1,3,3)),x[1].ravel()).reshape(x[1].shape)
            return jnp.array([jnp.sum(W*weights),jnp.min(J),jnp.min(jnp.where(required,J,jnp.inf)),
                jnp.sum(required&(J<=0)),jnp.sum(J<=0),jnp.all(jnp.isfinite(W))&jnp.all(jnp.isfinite(J))])
        v=jax.lax.map(part,packed)
        return jnp.array([jnp.sum(v[:,0]),jnp.min(v[:,1]),jnp.min(v[:,2]),jnp.sum(v[:,3]),jnp.sum(v[:,4]),jnp.all(v[:,5]>0)])

    def observe_free(self,state):
        q,a,vq,va,aq,aa,t=state
        s=np.asarray(self.stats_free(q,a));_,Uc,ft,fb,pen=self.contact(q,t)
        vel=self.full_velocity(vq,va)
        ke=float(.5*jnp.sum(self.nodem[:,None]*vel**2)*self.L**3)
        if not np.isfinite(s).all() or not s[5] or s[3]:raise ValueError('Nonfinite energy or nonpositive actual J in required material')
        return {'time':float(t),'compression':float(self.motion(t)),'force_N':float(ft),'bottom_force_N':float(fb),
            'ax_mm':float(a[0]*self.L),'ay_mm':float(a[1]*self.L),'energy_N_mm':float(s[0]*self.L**3),
            'contact_energy_N_mm':float(Uc*self.L**3),'KE_N_mm':ke,'KE_over_IE':ke/max(float(s[0]*self.L**3),1e-30),
            'required_positive_J_min':float(s[2]),'all_J_min':float(s[1]),'negative_void_J_points':int(s[4]),
            'cap_penetration_phi_ge_0p01_mm':np.asarray(pen).tolist(),
            'momentum_z_N_s':float(jnp.sum(self.nodem*vel[:,2])*self.L**2)}

    def initial_state(self):
        q=jnp.zeros((self.nc,3));a=jnp.zeros(2);t=jnp.array(0.)
        aq,aa=self.acceleration_free(q,a,t)
        return q,a,jnp.zeros_like(q),jnp.zeros_like(a),aq,aa,t

    def advance_block(self,steps=64):
        def run(state,dt,count):
            def active(s):
                q,a,vq,va,aq,aa,t=s
                vqh=vq+.5*dt*aq;vah=va+.5*dt*aa
                q=q+dt*vqh;a=a+dt*vah;t=t+dt
                nq,na=self._acceleration_free(q,a,t)
                return q,a,vqh+.5*dt*nq,vah+.5*dt*na,nq,na,t
            def one(s,k):return jax.lax.cond(k<count,active,lambda x:x,s),None
            return jax.lax.scan(one,state,jnp.arange(steps))[0]
        return jax.jit(run)
