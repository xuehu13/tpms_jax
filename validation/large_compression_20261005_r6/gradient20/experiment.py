"""20% elastic branch total thickness derivative, using existing JAX-FEM kernels.

Starts only after the Q2 rate/response gate. Explicit supplies a branch seed;
Newton/CG re-equilibrates it, then an implicit adjoint is checked by two newly
equilibrated thickness perturbations. No global tangent or new FEM is built.
This is an elastic endpoint derivative, not a history-dependent path gradient.
"""
from pathlib import Path
import argparse,json,sys,time,hashlib,shutil,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path('/home/xuehu/projects/tpms_jax');sys.path[:0]=[str(R),str(R/'scripts')]
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator,cg
from hyperelastic_fem import make_density_hyperelastic_problem
from thin_target_explicit import ExplicitXYZ
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--probe',action='store_true',help='Central residual/HVP/diagonal cost only; no certification')
parser.add_argument('--maxiter',type=int,default=4000)
parser.add_argument('--equilibrium-seed',type=Path,help='Reuse a saved central equilibrium, never change its thickness or compression')
parser.add_argument('--line-search',choices=['residual','energy'],default='residual')
a=parser.parse_args()
base=R/'validation/large_compression_20261005_r6/quadratic_candidate'
gate=json.loads((base/'rate_check.json').read_text())
assert gate['rate_targets_pass'] and gate['response_target_slow_pass'], 'Forward gate not passed'
a.output.mkdir(exist_ok=False);start=time.perf_counter()
def write(name,obj):
    (a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
with np.load(base/'gauss_field.npz') as f:
    rho=f['rho'];distance=jnp.asarray(f['distance']);points=f['physical_quad_points']
def field(p):
    assert np.array_equal(np.asarray(p.physical_quad_points),points)
    return rho
p=make_density_hyperelastic_problem(32,rho_quad=field,eta=1e-4,
                                    periodic_axes=(0,1,2),element_degree=2)
ex=ExplicitXYZ(p);del rho,points
L=10.;ell=.005/(2*np.log(9))
seed_path=a.equilibrium_seed or base/'T0p008_compact/field.npz'
with np.load(seed_path) as f:
    seed=np.asarray(f['q'])
    if a.equilibrium_seed:
        assert float(f['compression'])==.2 and float(f['thickness_mm'])==.5
shape=seed.shape;size=seed.size
def scale(t):return 1e-4+(1-1e-4)*jax.nn.sigmoid((t/(2*L)-distance)/ell)
def internal(q,t):
    geometry=(*ex.kernel_geometry[:4],scale(t))
    return ex._force(q,-.2,geometry)
def project(q):return q-jnp.mean(q,axis=0,keepdims=True)
def residual(q,t):return ex.reduce(internal(q,t))
def objective(q,t):return jnp.sum(internal(q,t)[:,2]*ex.points[:,2])*L**2
res=jax.jit(residual);val=jax.jit(objective)
hvp=jax.jit(lambda q,v,t:jax.jvp(lambda z:residual(z,t),(q,),(v,))[1])
stats=jax.jit(lambda q,t:ex._quadratic_stats(q[ex.ids],-.2,ex.kernel_geometry[1],
        ex.device_cells,ex.kernel_geometry[2][0],scale(t)))
# Q2 nodes in one Cartesian cell occupy three consecutive lattice levels.
# Modulo-four node colors remain distinct within a cell and across the XYZ
# periodic seam (64 intervals). AD products recover the exact diagonal.
idx=jnp.arange(ex.nc);colors=((idx//4096)%4*4+(idx//64)%4)*4+idx%4
cell_colors=np.sort(np.asarray(colors)[np.asarray(ex.ids)[p.fe.cells]],axis=1)
assert np.all(np.diff(cell_colors,axis=1)>0), 'Color probing does not isolate the tangent diagonal'
@jax.jit
def diagonal(q,t):
    def one(i,d):
        mask=(colors==i//3)[:,None]&(jnp.arange(3)==i%3)[None,:]
        v=mask.astype(q.dtype)
        return jnp.where(mask,hvp(q,v,t),d)
    return jax.lax.fori_loop(0,192,one,jnp.zeros_like(q))

records=[]
def linear(q,t,rhs,diag,label):
    qdev=jnp.asarray(q);rhs=np.asarray(project(jnp.asarray(rhs).reshape(shape))).ravel();iterations=0
    def mv(v):return np.asarray(project(hvp(qdev,project(jnp.asarray(v.reshape(shape))),t))).ravel()
    inv=1/np.asarray(diag).ravel()
    K=LinearOperator((size,size),matvec=mv,dtype=np.float64)
    def precondition(v):
        out=(inv*v).reshape(shape);return (out-out.mean(axis=0,keepdims=True)).ravel()
    M=LinearOperator((size,size),matvec=precondition,dtype=np.float64)
    def callback(_):
        nonlocal iterations
        iterations+=1
        if iterations%100==0:print(json.dumps({'linear':label,'iterations':iterations}),flush=True)
    t0=time.perf_counter()
    x,info=cg(K,rhs,M=M,rtol=1e-7,atol=0,maxiter=a.maxiter,callback=callback)
    err=float(np.linalg.norm(mv(x)-rhs)/max(np.linalg.norm(rhs),1e-30))
    record={'role':label,'iterations':iterations,'info':int(info),
            'relative_true_linear_residual':err,'seconds':time.perf_counter()-t0}
    records.append(record);write('linear_solves.json',records);print(json.dumps(record),flush=True)
    if info!=0 or err>2e-6:raise RuntimeError('Matrix-free CG did not satisfy the linear criterion')
    x=x.reshape(shape);return x-x.mean(axis=0,keepdims=True)

def equilibrate(initial,t,label):
    q=np.array(initial,copy=True);q-=q.mean(axis=0,keepdims=True);history=[]
    for k in range(9):
        r=np.asarray(res(jnp.asarray(q),t));norm=float(np.linalg.norm(r))
        minJ,U=stats(jnp.asarray(q),t);minJ=float(minJ);U=float(U)*L**3
        if not np.isfinite(q).all() or minJ<=0:raise RuntimeError('Invalid equilibrium trial')
        row={'iteration':k,'residual_l2':norm,'J_min':minJ,'energy_N_mm':U}
        history.append(row);print(label+' '+json.dumps(row),flush=True)
        write(label+'_progress.json',{'thickness_mm':t,'history':history})
        np.savez_compressed(a.output/(label+'_iterate.npz'),q=q,thickness_mm=t,
                            compression=.2,residual_l2=norm,equilibrated=norm<=1e-8)
        if norm<=1e-8:
            np.savez_compressed(a.output/(label+'_field.npz'),q=q,thickness_mm=t,compression=.2)
            return q,{'thickness_mm':t,'Fz_N':float(val(jnp.asarray(q),t)),
                      'residual_l2':norm,'J_min':minJ,'energy_N_mm':U,'history':history}
        if k==8:
            raise RuntimeError('Maximum equilibrium corrections reached; final checked state retained')
        d=np.asarray(diagonal(jnp.asarray(q),t))
        if not np.isfinite(d).all() or d.min()<=0:raise RuntimeError('Nonpositive tangent diagonal')
        step=linear(q,t,-r,d,label+'_Newton'+str(k))
        slope=float(np.sum(r*step))*L**3
        if a.line_search=='energy' and slope>=0:
            raise RuntimeError('Newton direction is not an energy descent direction')
        accepted=False
        for power in range(13):
            alpha=2.**(-power);trial=q+alpha*step
            Jtrial,Utrial=stats(jnp.asarray(trial),t)
            if float(Jtrial)<=0 or not np.isfinite(trial).all():continue
            rtrial=np.asarray(res(jnp.asarray(trial),t))
            ok=(np.linalg.norm(rtrial)<(1-1e-4*alpha)*norm if a.line_search=='residual'
                else float(Utrial)*L**3<=U+1e-4*alpha*slope)
            if ok:
                q=trial;accepted=True;break
        if not accepted:raise RuntimeError('No valid residual-reducing equilibrium step')
    raise RuntimeError('Maximum equilibrium corrections reached')

cfg={'target_compression':.2,'thickness_mm':.5,'perturbation_mm':.0025,
     'method':'implicit adjoint of re-equilibrated elastic branch; JAX-FEM residual/HVP, matrix-free CG',
     'static_translation_gauge':'zero-mean periodic fluctuation; equivalent elastic gauge, same XYZ macro loading',
     'history_dependent_path_gradient':False,'no_contact':True,'no_training':True,
     'seed_path':str(seed_path),'seed_sha256':hashlib.sha256(seed_path.read_bytes()).hexdigest(),
     'line_search':a.line_search,'forward_gate':gate,
     'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'source_sha256':{name:hashlib.sha256((R/name).read_bytes()).hexdigest()
                     for name in ['hyperelastic_fem.py','scripts/thin_target_explicit.py']}}
write('input.json',cfg);snapshot=a.output/'source_at_run';snapshot.mkdir()
shutil.copy2(__file__,snapshot/'gradient20_matrixfree.py')
for name in cfg['source_sha256']:shutil.copy2(R/name,snapshot/Path(name).name)
try:
    if a.probe:
        q=jnp.asarray(seed);r=np.asarray(res(q,.5));t0=time.perf_counter()
        d=np.asarray(diagonal(q,.5));seconds=time.perf_counter()-t0
        write('probe.json',{'residual_l2':float(np.linalg.norm(r)),'diagonal_min':float(d.min()),
                            'diagonal_seconds':seconds,'status':'cost_probe_not_gradient_validation'})
    else:
        q,central=equilibrate(seed,.5,'central');write('central.json',central)
        qdev=jnp.asarray(q);gq,gt=jax.jit(jax.grad(objective,argnums=(0,1)))(qdev,.5)
        Rt=jax.jit(lambda q,t:jax.jvp(lambda t:residual(q,t),(t,),(1.,))[1])(qdev,.5)
        diag=np.asarray(diagonal(qdev,.5));adj=linear(q,.5,np.asarray(gq),diag,'adjoint')
        derivative=float(gt-jnp.sum(jnp.asarray(adj)*Rt))
        write('adjoint.json',{'dFz_dt_N_per_mm':derivative,'gradient_check_pass':False,
                             'status':'adjoint_computed; independent perturbations pending'})
        _,minus=equilibrate(q,.4975,'minus');_,plus=equilibrate(q,.5025,'plus')
        fd=(plus['Fz_N']-minus['Fz_N'])/.005
        error=abs(derivative-fd)/max(abs(fd),1e-30)
        result={'status':'total_derivative_check_pass' if error<=.01 else 'gradient_check_failed',
                'central':central,'minus':minus,'plus':plus,
                'adjoint_dFz_dt_N_per_mm':derivative,'finite_difference_dFz_dt_N_per_mm':fd,
                'relative_gradient_difference':error,'gradient_check_limit':.01,
                'elapsed_seconds':time.perf_counter()-start,'linear_solves':records,
                'same_loading_path_design_perturbations_checked':False,
                'physical_replacement_certified':False,
                'scope':'local elastic equilibrium endpoint on selected XYZ/Q2 branch; not a compression-path or shape derivative'}
        write('result.json',result);print(json.dumps(result),flush=True)
except Exception as exc:
    write('failure.json',{'type':type(exc).__name__,'message':str(exc),
                          'elapsed_seconds':time.perf_counter()-start,'linear_solves':records})
    raise
