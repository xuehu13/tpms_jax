"""Geometry-only one-sided distance diagnosis. No dynamic solves or gate edits."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import csv,hashlib,json,time
import numpy as np
from scipy.special import expit
from shape import ShapeGeometry,B,R

start=time.time()
spec=json.loads((B/'input/G_shape_geometry_diagnostic.json').read_text())
for rel,h in spec['protected_sources'].items():
    if hashlib.sha256((R/rel).read_bytes()).hexdigest()!=h:raise ValueError('Protected bytes changed: '+rel)
out=B/'analysis/G5_shape_geometry_diagnostic';out.mkdir(exist_ok=False)
geom=ShapeGeometry();d0,g0,f0,_=geom.evaluate(0.)
ell=.05/10/(2*np.log(9));phi=expit((.5/(2*10)-d0)/ell)
weight=-(1-1e-4)*phi*(1-phi)/ell
analytic=weight*g0
norm=lambda x:float(np.linalg.norm(x))
denom=max(norm(analytic),1e-300)
rows=[]
for h in spec['geometry_only_fd_steps']:
    if time.time()-start>spec['wall_budget_seconds']:raise TimeoutError('Geometry diagnosis budget')
    dp,gp,fp,_=geom.evaluate(h);dm,gm,fm,_=geom.evaluate(-h)
    sp=1e-4+(1-1e-4)*expit((.025-dp)/ell)
    sm=1e-4+(1-1e-4)*expit((.025-dm)/ell)
    s0=1e-4+(1-1e-4)*phi
    right=(sp-s0)/h;left=(s0-sm)/h;central=(right+left)/2
    e=central-analytic
    relevant=(phi>1e-6)&(phi<1-1e-6)
    differing=(np.abs(gp-gm)>1e-3)&relevant
    row=dict(h=h,central_scale_relative_L2_vs_JVP=norm(e)/denom,
        right_scale_relative_L2_vs_JVP=norm(right-analytic)/denom,
        left_scale_relative_L2_vs_JVP=norm(left-analytic)/denom,
        left_right_scale_relative_L2=norm(right-left)/denom,
        exact_branch_derivative_difference_relative_L2=norm(weight*(gp-gm))/denom,
        relevant_queries=int(relevant.sum()),relevant_branch_derivatives_differ_by_more_than_1em3=int(differing.sum()),
        face_changes_plus=int(np.count_nonzero(fp!=f0)),face_changes_minus=int(np.count_nonzero(fm!=f0)),
        largest_absolute_scale_derivative_difference=float(np.max(np.abs(e))))
    rows.append(row);print(json.dumps(row),flush=True)
    if h==spec['geometry_only_fd_steps'][-1]:
        indices=np.argsort(np.abs(e))[-30:][::-1]
        points=[]
        for k in indices:
            points.append(dict(index=int(k),kind='Gauss' if k<geom.n_gauss else 'cap',
                x=float(geom.queries[k,0]),y=float(geom.queries[k,1]),z=float(geom.queries[k,2]),
                distance=float(d0[k]),occupancy=float(phi[k]),analytic_scale_derivative=float(analytic[k]),
                right_scale_slope=float(right[k]),left_scale_slope=float(left[k]),central_scale_slope=float(central[k]),
                baseline_distance_derivative=float(g0[k]),plus_distance_derivative=float(gp[k]),minus_distance_derivative=float(gm[k]),
                base_face=int(f0[k]),plus_face=int(fp[k]),minus_face=int(fm[k])))
        with (out/'largest_query_errors.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
protected=all(hashlib.sha256((R/rel).read_bytes()).hexdigest()==h for rel,h in spec['protected_sources'].items())
result=dict(scope='Geometry only: actual moved triangle surfaces, one-sided slopes, no time steps',
    original_full_response_passed=False,no_acceptance_change=True,rows=rows,
    largest_query_errors=points,protected_sources_unchanged=protected,wall_minutes=(time.time()-start)/60)
(out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
with (out/'convergence.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
