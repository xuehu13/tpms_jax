"""Prescribed local shell motion: audit native contact geometry, not a G solve."""
from pathlib import Path
import hashlib, json
import numpy as np
R = Path('/home/xuehu/projects/tpms_jax')
D = R / 'validation/platen_xy_G_20261010_r51'
B = D / 'diagnostics/native_shell_contact_20261010'
def write(p, v): p.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    B.mkdir(exist_ok=True)
    assert not (B / 'input/protected_before.json').exists(), 'Never overwrite a prepared experiment'
    for name in ['input', 'analysis', 'abaqus/default', 'abaqus/self', 'abaqus/noperimself']:
        (B / name).mkdir(parents=True, exist_ok=True)
    protected = {}
    previous = json.loads((D / 'diagnostics/shell_cap_arm_20261010/input/protected_before.json').read_text())
    # The preceding turn documented a read-only RF3 extraction sign fix.
    correction = json.loads((D / 'diagnostics/shell_cap_arm_20261010/input/probe_extraction_correction.json').read_text())
    prior_extractor = str((D / 'tools/extract_cap_arm_probe.py').relative_to(R))
    assert previous[prior_extractor] == correction['old_tool_sha256']
    previous[prior_extractor] = correction['new_tool_sha256']
    for name, digest in previous.items():
        assert sha(R / name) == digest, name
    for p in D.rglob('*'):
        if p.is_file() and not p.is_relative_to(B) and '__pycache__' not in str(p) and p.name not in ['README.md', 'status.json']:
            protected[str(p.relative_to(R))] = sha(p)
    for name in previous: protected[name] = sha(R / name)
    write(B / 'input/protected_before.json', protected)
    n = 8; angle = .1; thickness = .5
    points = np.array([[0., j / n, i / n - 1.] for i in range(n + 1) for j in range(n + 1)])
    cells = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j + 1; b = a + 1; c = a + n + 1; e = c + 1
            cells.extend([[a, b, c], [b, e, c]])
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    motion = points @ rotation.T - points
    common = ['*Heading', 'r51 local prescribed rigid shell rotation / native contact geometry probe', '*Node']
    common += [f'{i},' + ','.join(f'{x:.16g}' for x in p) for i, p in enumerate(points, 1)]
    common += ['1000,0,0,.3', '1001,-2,-2,.3', '1002,-2,2,.3', '1003,2,2,.3', '1004,2,-2,.3',
               '*Element,type=S3R,elset=WALL']
    common += [f'{i},' + ','.join(map(str, c)) for i, c in enumerate(cells, 1)]
    common += ['*Element,type=R3D4,elset=PLATE', '1000,1001,1002,1003,1004', '*Rigid Body,ref node=1000,elset=PLATE',
               '*Nset,nset=SHELLN,generate', '1,81,1', '*Nset,nset=RP', '1000',
               '*Material,name=BASE', '*Density', '1e-9', '*Hyperelastic,Neo Hooke', '1.923076923076923,.24',
               '*Shell Section,elset=WALL,material=BASE', '.5,5',
               '*Surface,type=ELEMENT,name=SHELL', 'WALL,SPOS', '*Surface,type=ELEMENT,name=PLATEN', 'PLATE,SPOS',
               '*Surface Interaction,name=K', '*Surface Behavior,pressure-overclosure=LINEAR', '320.',
               '*Contact Damping,definition=CRITICAL DAMPING FRACTION', '0.',
               '*Boundary', '1000,1,2,0.', '1000,4,6,0.',
               '*Amplitude,name=ROT,definition=SMOOTH STEP', '0.,0.,.0002,1.,.0013,1.',
               '*Amplitude,name=CLOSE,definition=SMOOTH STEP', '0.,0.,.0002,0.,.0012,1.,.0013,1.',
               '*Step,name=PLATEN,nlgeom=YES', '*Dynamic,Explicit', ',.0013',
               '*Contact', '*Contact Inclusions', 'SHELL,PLATEN', '*Contact Exclusions', 'SHELL,SHELL',
               '*Contact Property Assignment', 'SHELL,PLATEN,K',
               '*Contact Controls Assignment,ROTATIONAL TERMS=STRUCTURAL']
    for case in ['default', 'self', 'noperimself']:
        text = list(common)
        if case != 'default': text += [f'*Contact Controls Assignment,CONTACT THICKNESS REDUCTION={case.upper()}']
        text += ['*Boundary,amplitude=ROT']
        for i, u in enumerate(motion, 1):
            for dof in range(1, 7):
                value = u[dof - 1] if dof <= 3 else (angle if dof == 5 else 0.)
                text += [f'{i},{dof},{dof},{value:.16g}']
        text += ['*Boundary,amplitude=CLOSE', '1000,3,3,-.33', '*Output,field,number interval=130',
                 '*Node Output', 'U,UR,RF,RM', '*Contact Output', 'CSTRESS,CDISP,CFORCE',
                 '*Element Output,elset=WALL', 'STH', '*Output,history,time interval=2e-7',
                 '*Node Output,nset=RP', 'U3,RF3', '*Energy Output', 'ALLIE,ALLKE,ALLAE,ALLVD,ALLWK,ETOTAL,ALLPW,ALLCW', '*End Step']
        inp = B / f'abaqus/{case}/probe.inp'; inp.write_text('\n'.join(text) + '\n')
        cfg = {'case': case, 'scope': 'Prescribed local rigid motion; contact geometry only. Not free deformable shell, not G, no XY PBC or JAX forward path.',
               'solver': 'Abaqus Explicit', 'shell': 'S3R, .5mm, NH E10MPa nu.3, density1e-9 tonne/mm3',
               'shell_nodes': len(points), 'shell_elements': len(cells), 'mesh_side_mm': 1/n,
               'rotation_y_rad': angle, 'normal_thickness_mm': thickness,
               'plate_initial_z_mm': .3, 'plate_final_z_mm': -.03, 'plate_closure_mm': .33,
               'rotation_end_s': .0002, 'load_end_s': .0012, 'end_s': .0013,
               'physical_flat_cut_highest_z_mm': float(thickness/2 * np.sin(angle)),
               'contact': 'Element-based general contact, only shell/plate; no shell self-contact, kn320, zero contact damping, structural rotational terms.',
               'changed_factor': 'Only CONTACT THICKNESS REDUCTION control; default, SELF, NOPERIMSELF. The latter can introduce a nonphysical perimeter bull-nose and is diagnostic only.',
               'question': 'Does native shell contact represent the normal-thickness flat end after rotation? Can suppressing thickness reduction fix it without a perimeter extension?',
               'onset_measurement': 'First compressive RP force exceeding 1e-7N, bracketed by the previous history sample; report plate z interval, no fitted force acceptance.',
               'gate_before_G': 'Native onset height must agree with .5/2*sin(.1) within .001mm without a bull-nose or automatic initial adjustment. This is only a local geometry gate, not a G accuracy gate.',
               'input_sha256': sha(inp), 'new_G': False, 'full20_submitted': False, 'new_design_AD': False}
        write(B / f'input/{case}_protocol.json', cfg)
    np.savez_compressed(B / 'input/probe_mesh.npz', xyz_mm=points, cells=np.array(cells)-1, final_u_mm=motion, rotation=rotation)
    write(B / 'input/sources.json', {
        'native_geometry': 'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactgeneral.htm',
        'thickness_and_perimeter_controls': 'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactcontrolsassign.htm',
        'surface_thickness': 'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-surfacepropassign.htm',
        'note': 'Official public 2025 documentation used as algorithm reference; actual input tested using installed Abaqus 2026.'})
    write(B / 'status.json', {'stage': 'prepared_three_prescribed_geometry_probes', 'active_jobs': [], 'new_G_paths': 0, 'full20_submitted': False, 'new_design_AD': False})
    print(json.dumps({'directory': str(B), 'protected_files': len(protected), 'physical_flat_cut_highest_z_mm': float(thickness/2*np.sin(angle))}))
if __name__ == '__main__': main()
