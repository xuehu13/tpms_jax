"""Read-only contact geometry and contact-only frequency probes for r51."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.special import expit
from scipy import sparse
from scipy.sparse.linalg import eigsh
D=Path(__file__).resolve().parents[1]
R=D.parents[1]
B=D/'diagnostics/contact_support_control_20261010'
sys.path[:0]=[str(R),str(D/'tools')]
from surface_distance import PeriodicSurfaceDistance
from run_jax import build

def main():
    paths=[('original',D/'input/cut_face_contact.npz'),
           ('halfway',B/'input/half_support_contact.npz'),
           ('rimline',D/'diagnostics/contact_origin_20261010/input/rimline_contact.npz')]
    with np.load(R/'validation/canonical_tpms_20261009_r50/G/surface.npz') as z:
        sd=PeriodicSurfaceDistance(z['surface_vertices'],z['surface_triangles'])
    geometry=[]
    for tag,p in paths:
        with np.load(p) as z:c=dict(z)
        phi=expit((.025-sd.query(c['points']))/(.005/(2*np.log(9))))
        w=c['area_weight'];area=w.sum()
        geometry.append({'case':tag,
            **{f'contact_weight_fraction_at_geometric_phi_lt_{name}':float(w[phi<cut].sum()/area) for name,cut in [('0p001',.001),('0p01',.01),('0p5',.5)]},
            'minimum_geometric_phi_at_nonzero_weight':float(phi[w>0].min()),
            'area_weighted_geometric_phi':float(np.sum(w*phi)/area)})
    cfg=json.loads((B/'input/cap_common_dt.json').read_text())
    ex=build(cfg);classes=np.asarray(ex.ids);mass=np.asarray(ex.mass);spectrum=[]
    for tag,p in paths:
        with np.load(p) as z:c=dict(z)
        active,inv=np.unique(classes[c['node_ids']],return_inverse=True)
        C=sparse.coo_matrix((c['shape'].ravel(),(np.repeat(np.arange(len(c['points'])),9),inv.ravel())),shape=(len(c['points']),len(active))).tocsr()
        A=C.multiply(np.sqrt(3200*c['area_weight'])[:,None]).multiply(1/np.sqrt(mass[active])[None,:])
        lam=float(eigsh((A.T@A).tocsr(),k=1,which='LA',tol=1e-8,return_eigenvectors=False)[0])
        spectrum.append({'case':tag,'contact_only_max_mass_normalized_eigenvalue_s_minus2':lam,
            'contact_only_dt_at_safety_0p3_s':float(.6/np.sqrt(lam)),
            'operator_active_mass_min':float(mass[active].min()),'active_boundary_classes':len(active)})
    for name,data,scope in [
        ('contact_location_occupancy',geometry,'Geometric occupancy at contact positions, unchanged volume Gauss field and HRZ; no FE trajectory.'),
        ('contact_only_spectrum',spectrum,'Both caps active, contact-only Z tangent on common HRZ class mass; not a global stable-step bound.')]:
        (B/'analysis'/f'{name}.json').write_text(json.dumps({'scope':scope,'cases':data},indent=2)+'\n')

if __name__=='__main__':main()
