"""Independent two-level geometry FD, shared-field JVP and reverse duality."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import json,time,hashlib
import numpy as np
from design import B,R,jax,jnp,DesignPath
from prepare import face_cache

def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
start=time.time();root=B/'input/G_shape';protocol=json.loads((root/'geometry_protocol.json').read_text())
out=B/'checks/G_shape_geometry';out.mkdir(exist_ok=False)
with np.load(R/protocol['gauss_path']) as z:points=z['physical_quad_points'];old_d=z['distance']
cap=face_cache(32)
np.savez_compressed(root/'queries.npz',points=np.vstack([points.reshape(-1,3),cap['points']]),gauss_shape=old_d.shape)
from shape import ShapeDesignPath
model=ShapeDesignPath();geometry=model.shape_geometry
params=jnp.array([.5,.05,0.]);seed=jnp.array([0.,0.,1.])
baseline=model.baseline_checks()
if not baseline['pass']:raise ValueError('Shape baseline differs from the existing mechanics')
fields,dfields=jax.block_until_ready(jax.jvp(model.fields,(params,),(seed,)))
names=['scale','nodem','mass','macro_c','macro_schur','rim_area']
h,half=protocol['fd_steps'];samples={}
for a in [0.,h,-h,half,-half]:
    if time.time()-start>protocol['wall_budget_seconds']:raise TimeoutError('Geometry budget exhausted')
    d,dd,faces,audit=geometry.evaluate(a)
    f=jax.block_until_ready(model.fields(jnp.array([.5,.05,a])))
    samples[a]=(d,dd,faces,audit,tuple(np.asarray(x) for x in f))
    np.savez_compressed(out/f"geometry_{a:+.8f}.npz",distance=d,analytic_distance_derivative=dd,closest_triangle=faces)
    print(json.dumps(dict(shape_a=a,wall_s=time.time()-start,audit=audit)),flush=True)

def compare(ad,fd,fd2,baseline):
    norm=lambda x:float(np.linalg.norm(np.asarray(x).ravel()))
    scale=protocol['parameter_scale']/max(norm(baseline),1e-300)
    near=norm(fd2)*scale<protocol['nearzero_scaled_derivative_cutoff']
    err=norm(ad-fd2);delta=norm(fd-fd2);den=max(norm(fd2),1e-300)
    passed=(err*scale<=protocol['nearzero_scaled_error_tolerance'] and delta*scale<=protocol['nearzero_scaled_error_tolerance']) if near else (err/den<=.01 and delta/den<=.01)
    return dict(AD_L2=norm(ad),FD_half_L2=norm(fd2),AD_FD_relative_L2=err/den,
                FD_halving_relative_L2=delta/den,scaled_error=err*scale,
                nearzero=near,passed=passed)
rows={}
for i,name in enumerate(names):
    fd=(samples[h][4][i]-samples[-h][4][i])/(2*h)
    fd2=(samples[half][4][i]-samples[-half][4][i])/(2*half)
    rows[name]=compare(np.asarray(dfields[i]),fd,fd2,np.asarray(fields[i]))
d0,dd,face0,audit,_=samples[0.]
dfd=(samples[h][0]-samples[-h][0])/(2*h)
dfd2=(samples[half][0]-samples[-half][0])/(2*half)
# All-domain distance includes inactive medial-axis ties; material occupancy is the gate.
distance_check=compare(dd,dfd,dfd2,d0)
face_changes={str(a):int(np.count_nonzero(samples[a][2]!=face0)) for a in [h,-h,half,-half]}
with np.load(B/'input/design_geometry.npz') as z:
    old_all=np.r_[z['gauss_distance'].ravel(),z['cap_distance']]
baseline_distance_max=float(np.max(abs(d0-old_all)))
# A deterministic scalar contraction exercises the transposed custom JVP.
rng=np.random.default_rng(53)
cot=tuple(jnp.asarray(rng.normal(size=x.shape)/max(np.sqrt(x.size),1)) for x in fields)
_,pullback=jax.vjp(model.fields,params)
reverse=np.asarray(jax.block_until_ready(pullback(cot)[0]))
dual=float(sum(jnp.vdot(v,c) for v,c in zip(dfields,cot)))
duality_error=abs(reverse[2]-dual)/max(abs(dual),1e-30)
boundary=(np.isclose(geometry.vertices[:,2],0)|np.isclose(geometry.vertices[:,2],1))
rim_speed=float(np.max(abs(geometry.velocity[boundary])))
passed=baseline['pass'] and baseline_distance_max<1e-12 and rim_speed==0 and all(r['passed'] for r in rows.values()) and duality_error<1e-10
metrics=dict(passed=bool(passed),scope='Geometry/shared-field local derivative, no dynamic design derivative yet',
    baseline_equivalence=baseline,baseline_distance_max_normalized=baseline_distance_max,
    rim_vertex_velocity_max=rim_speed,field_derivatives=rows,distance_diagnostic=distance_check,
    closest_face_changes=face_changes,short_field_VJP_contraction=reverse[2],
    short_field_JVP_contraction=dual,field_duality_relative_error=duality_error,
    shape_rule_uses_FD=False,independent_geometries=True,wall_minutes=(time.time()-start)/60,
    geometry_protocol_sha256=hashlib.sha256((root/'geometry_protocol.json').read_bytes()).hexdigest())
write(out/'metrics.json',metrics)
text=['# G周期形态系数：几何求导链核对','',
      '几何层'+('通过' if passed else '未通过')+'预定门槛；尚无完整压缩响应的构型梯度结论。','',
      '以原G中面为基线，在归一化坐标中令 x → x + a sin(2πx) sin²(πz)，y、z不变。a无量纲，基线0，局部范围±0.01（最大移动0.1mm）。该形态族保持三角连接、周期性和上下端缘；它不保证变形后仍是严格极小曲面，也不代表所有解析系数均已验证。','',
      '距离采用实际移动三角面的最近点查询。唯一最近点处，距离导数等于负法向与最近点顶点速度插值的点积；占据、HRZ质量、XY投影惯性和端缘权重继续由原共享实现求导。差分步长0.0001及0.00005，每次重新查询实际移动的曲面。','',
      '| 字段 | AD/FD L2差 | 两档FD L2差 | 近零 | 通过 |','|---|---:|---:|---|---|']
for name,r in rows.items():text.append(f"| {name} | {100*r['AD_FD_relative_L2']:.6g}% | {100*r['FD_halving_relative_L2']:.6g}% | {r['nearzero']} | {r['passed']} |")
text+=['',f"基线距离最大差{baseline_distance_max:.4g}（归一化）；字段JVP/VJP对偶差{duality_error:.4g}；几何核对耗时{metrics['wall_minutes']:.2f}min。",'',
       '最近三角面可能切换，距离在多个最近点并存处未必光滑。原始全域距离差作为诊断保留，门槛按实际占据及质量等模型字段确定；不把远离实体的距离导数误差直接当作力学误差。完整动力学仍须独立JVP与两档FD，之后再验证完整反向。']
(out/'README.md').write_text('\n'.join(text)+'\n')
status=json.loads((B/'status.json').read_text());status.update(stage='G_shape_geometry_analyzed',active_jobs=[],
    shape_geometry='passed' if passed else 'not_passed',next_task='Benchmark G5 shape full-path derivative' if passed else 'Resolve geometry gate before FE')
write(B/'status.json',status)
print(json.dumps(metrics,indent=2),flush=True)
if not passed:raise SystemExit('Geometry derivative gate not passed; no dynamics submitted')
