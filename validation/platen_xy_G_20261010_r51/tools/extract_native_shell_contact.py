"""Read-only extraction for prescribed shell contact geometry probes."""
from pathlib import Path
import json, sys
import numpy as np
from odbAccess import openOdb
def values(v):
    try: return v.dataDouble
    except Exception: return v.data
odbpath, cfgpath, out = map(Path, sys.argv[1:]); assert not out.exists()
cfg = json.loads(cfgpath.read_text()); odb = openOdb(str(odbpath.resolve()), readOnly=True)
try:
    step = odb.steps['PLATEN']
    rp = [r for r in step.historyRegions.values() if 'RF3' in r.historyOutputs and 'U3' in r.historyOutputs]
    assert len(rp) == 1, len(rp)
    rf = np.array(rp[0].historyOutputs['RF3'].data); u = np.array(rp[0].historyOutputs['U3'].data)
    assert np.allclose(rf[:, 0], u[:, 0], atol=1e-12)
    force = -rf[:, 1]; z = cfg['plate_initial_z_mm'] + u[:, 1]
    active = np.flatnonzero((rf[:, 0] > cfg['rotation_end_s']) & (force > 1e-7))
    assert len(active) > 0
    i = int(active[0]); onset = [float(z[i]), float(z[i-1])]
    frames = []
    for fr in step.frames:
        ff = {'time_s': fr.frameValue, 'fields': {}}
        for key, field in fr.fieldOutputs.items():
            if key.startswith(('COPEN', 'CNORMF', 'CTHICK', 'STH')):
                ff['fields'][key] = [{'nodeLabel': v.nodeLabel if hasattr(v, 'nodeLabel') else None,
                    'elementLabel': v.elementLabel if hasattr(v, 'elementLabel') else None,
                    'instance': v.instance.name if v.instance else None,
                    'data': np.asarray(values(v)).tolist()} for v in field.values]
        frames.append(ff)
    fr = step.frames[-1]
    last = {}
    for key in ['U','UR','RF','RM']:
        last[key] = {str(v.nodeLabel): np.asarray(values(v)).tolist() for v in fr.fieldOutputs[key].values}
    result = {'case': cfg['case'], 'complete': abs(fr.frameValue - cfg['end_s']) < 1e-8,
        'onset_plate_z_bracket_mm': onset, 'physical_flat_cut_highest_z_mm': cfg['physical_flat_cut_highest_z_mm'],
        'onset_height_error_bracket_mm': [x-cfg['physical_flat_cut_highest_z_mm'] for x in onset],
        'local_geometry_gate_passed': bool(min(onset) - .001 <= cfg['physical_flat_cut_highest_z_mm'] <= max(onset) + .001),
        'final_force_N': float(force[-1]), 'history_rows': len(force), 'field_keys': list(fr.fieldOutputs.keys()),
        'last_fields': last, 'scope': cfg['scope'], 'new_G': False, 'full20_submitted': False, 'new_design_AD': False}
    np.savez_compressed(out.with_name('field.npz'), time_s=rf[:,0], force_N=force, plate_z_mm=z)
    out.with_name('contact_fields.json').write_text(json.dumps(frames, indent=2)+'\n')
finally: odb.close()
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='last_fields'}, indent=2))
