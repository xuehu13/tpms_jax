"""Freeze G thickness-gradient inputs; no solver is launched by this file."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
from scipy.optimize import brentq
R=Path(__file__).resolve().parents[3]
B=Path(__file__).resolve().parents[1]
G=R/'validation/platen_xy_G_20261010_r51'
sys.path[:0]=[str(G/'tools'),str(R)]
from prepare import face_cache
from surface_distance import PeriodicSurfaceDistance

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    started=time.time()
    dest=B/'input';dest.mkdir(exist_ok=False)
    for name in ['checks','primal','AD','FD','analysis']: (B/name).mkdir(exist_ok=True)
    old=G/'experiments/midsurface_rim_20pct_20261010'
    cfg=json.loads((old/'input/protocol.json').read_text())
    surface=R/'validation/canonical_tpms_20261009_r50/G/surface.npz'
    with np.load(surface) as z:dist=PeriodicSurfaceDistance(z['surface_vertices'],z['surface_triangles'])
    cap=face_cache(cfg['N'])
    dcap=dist.query(cap['points'])
    with np.load(cfg['original_Gauss_field']) as z:d=z['distance']
    with np.load(G/'diagnostics/contact_origin_20261010/input/contact_assignment.npz') as z:assignment=z['rim_assignment']
    np.savez_compressed(dest/'design_geometry.npz',gauss_distance=d,cap_distance=dcap,
                        cap_raw_area=cap['area_weight'],cap_assignment=assignment)
    target=.05
    s=brentq(lambda s:.2*(10*s**3-15*s**4+6*s**5)-target,0,1,xtol=1e-14)
    cfg.update(case_id='G_platen_xy_gradient_r53',hold_time_s=0.,
        stage='Full path to5pct, fixed time discretization; thickness total derivative including HRZ and contact area',
        target_stop_compression=target,end_time_s=s*.004,
        gradient_parameters_mm=['thickness_mm','interface_10_90_mm'],baseline_parameters_mm=[.5,.05],
        active_parameter='thickness_mm',thickness_FD_steps_mm=[.0025,.00125],
        fixed_dt_max_s=5.8e-8,block_steps=64,
        loading='Prefix of original20pct/4ms ramp, no hold; no change of strain-rate definition',
        objective='Trapezoidal upper plate force versus travel at every accepted step; terminal total upper force auxiliary',
        derivative_scope='Fixed steps/time; no derivative of rejection, adaptive timestep or count decisions',
        includes=['Gauss occupancy','material/internal force','HRZ nodal mass','XY projected macro inertia','end contact occupied area','initial acceleration','all states and work quadrature'],
        thresholds=dict(nonzero_gradient_relative_AD_FD=.01,FD_h_halving_relative=.01,
                        nearzero_dimensionless_gradient_cutoff=1e-6,nearzero_absolute_scaled_error=1e-5,
                        objective_scales=[1.,1.],parameter_scale_mm=.5),
        budget=dict(benchmark_seconds=900,first_thickness_total_seconds=10800,
                    application='New cost budget selected for one thickness check, including compilation and4FD paths; stop and record if insufficient, no parameter changes'),
        batch_started_unix=started,
        execution_authorization='2026-10-11 user: 好的，按你的规划进行吧')
    protected=[R/'hyperelastic_fem.py',R/'scripts/thin_target_explicit.py',R/'surface_distance.py',R/'pbc.py',
               G/'tools/platen_model.py',G/'tools/run_jax.py',G/'tools/prepare.py',G/'tools/contact_origin.py',
               Path(cfg['original_Gauss_field']),Path(cfg['contact_cache_path']),old/'jax/result.json']
    write(dest/'protocol.json',cfg)
    write(B/'checks/source_guard_before.json',{str(p.relative_to(R)):sha(p) for p in protected})
    write(B/'status.json',dict(stage='inputs_frozen_benchmark_pending',active_jobs=[],step1='pending',step2='pending',step3='pending',step4='pending',new_reverse_AD=False))
    print(json.dumps(dict(end_time_s=cfg['end_time_s'],compression=target,gauss_points=d.size,cap_points=dcap.size)))

if __name__=='__main__':main()
