"""Archive completed native jobs and compare global refinement indicators."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(package, output):
    output.mkdir(parents=True,exist_ok=True)
    cases=[]
    for path in sorted((package/'work').glob('*.acceptance.json')):
        report=json.loads(path.read_text()); case=report['case']
        expected=json.loads((package/(case+'.expected.json')).read_text())
        if report['status'] != 'ok' or not all(report['checks'].values()):
            raise ValueError('Consistency check failed for '+case)
        if sha(package/(case+'.inp')) != expected['input_sha256'] or report['input_sha256'] != expected['input_sha256']:
            raise ValueError('Input hash differs for '+case)
        sta=(package/'work'/(case+'.sta')).read_text()
        if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' not in sta:
            raise ValueError('Incomplete analysis '+case)
        diagnostics=[]
        distorted=0;out_of_core=False
        import re
        for suffix in ('.dat','.msg'):
            text=(package/'work'/(case+suffix)).read_text(errors='replace')
            for line in text.splitlines():
                if '***ERROR' in line:
                    raise ValueError('Abaqus error '+line)
                if '***WARNING' in line:
                    match=re.match(r'\s*\*\*\*WARNING:\s+(\d+) elements are distorted\.',line)
                    fallback=re.match(r'\s*\*\*\*WARNING: THE MEMORY LIMIT IS INSUFFICENT TO RUN THE PURE THREAD-BASED',line)
                    if fallback:
                        if 'TO THE HYBRID SOLVER FOR OUT-OF-CORE SOLUTION' not in text:
                            raise ValueError('Memory warning without documented fallback')
                        out_of_core=True;diagnostics.append(line);continue
                    if not match:
                        raise ValueError('Unresolved warning '+line)
                    distorted=int(match.group(1));diagnostics.append(line)
        dat_path=package/'work'/(case+'.dat')
        # Large DAT files contain one line per distorted element. Keep the
        # original beside the ODB, archive its hash and diagnostic excerpts.
        dat_lines=dat_path.read_text(errors='replace').splitlines()
        selected=set()
        for i,line in enumerate(dat_lines):
            if '***WARNING' in line or '***ERROR' in line or 'THE ANALYSIS HAS BEEN COMPLETED' in line:
                selected.update(range(max(0,i-2),min(len(dat_lines),i+5)))
        excerpt=[f'Original: {dat_path}',f'SHA256: {sha(dat_path)}',f'Bytes: {dat_path.stat().st_size}',
                 'Selected diagnostic lines; full raw DAT retained beside ODB.']
        excerpt += [f'{i+1}: {dat_lines[i]}' for i in sorted(selected)]
        (output/(case+'.dat_excerpt.txt')).write_text('\n'.join(excerpt)+'\n')
        for suffix in ('.acceptance.json','.sta','.msg','.console.txt','.extract.txt'):
            src=package/'work'/(case+suffix)
            if src.exists():shutil.copyfile(src,output/src.name)
        shutil.copyfile(package/(case+'.expected.json'),output/(case+'.expected.json'))
        diagnostic={'case':case,'errors':0,'warning_messages':len(diagnostics),
                    'distorted_elements':distorted,'quality_warning':distorted>0,
                    'out_of_core_solver':out_of_core,'convergence_claim':False,'warning_text':diagnostics}
        (output/(case+'.diagnostics.json')).write_text(json.dumps(diagnostic,indent=2)+'\n')
        if expected['model'] == 'uniform':
            shutil.copyfile(dat_path,output/dat_path.name)
            shutil.copyfile(package/(case+'.inp'),output/(case+'.inp'))
            shutil.copyfile(package/(case+'.mesh.npz'),output/(case+'.mesh.npz'))
        m=report['measured']
        cases.append({'case':case,'geometry_N':expected['geometry_N'],'refinement':expected['fe_refinement'],
                      'lateral':expected['lateral'],'nodes':expected['nodes'],'elements':expected['elements'],
                      'volume':m['volume_solid'],'Fz':m['macro_RF'][2],'energy':m['ALLSE'],
                      'ex':m['macro_displacements'][0],'ey':m['macro_displacements'][1],
                      'distorted_elements':distorted,'mean_ratio_min':expected['geometry']['mean_ratio_min'],
                      'input_sha256':expected['input_sha256'],'mesh_sha256':expected['mesh_sha256'],
                      'odb_path':str(package/'work'/(case+'.odb')),'odb_bytes':(package/'work'/(case+'.odb')).stat().st_size,
                      'dat_sha256':sha(dat_path),'dat_bytes':dat_path.stat().st_size,
                      'out_of_core_solver':out_of_core,'physical_consistency':'pass'})
    volume=json.loads((package/'binary_volume_reference.json').read_text())
    shutil.copyfile(package/'binary_volume_reference.json',output/'binary_volume_reference.json')
    fixed=sorted([r for r in cases if r['geometry_N'] and r['refinement'] == 0 and r['lateral']=='fixed'],key=lambda r:r['geometry_N'])
    geometry_changes=[{'from':a['geometry_N'],'to':b['geometry_N'],
                       'relative_Fz_change':abs(b['Fz']-a['Fz'])/abs(b['Fz']),
                       'denominator':'finer geometry response',
                       'volume_change':b['volume']-a['volume']} for a,b in zip(fixed,fixed[1:])]
    fe_changes=[]
    for n in sorted({r['geometry_N'] for r in cases if r['geometry_N']}):
        group=[r for r in cases if r['geometry_N']==n and r['lateral']=='fixed']
        if {r['refinement'] for r in group} >= {0,1}:
            a=next(r for r in group if r['refinement']==0);b=next(r for r in group if r['refinement']==1)
            if abs(a['volume']-b['volume']) > 1e-9:
                raise ValueError('FE refinement changed the binary geometry')
            fe_changes.append({'geometry_N':n,'relative_Fz_change':abs(b['Fz']-a['Fz'])/abs(b['Fz']),
                               'denominator':'refined FE response','volume_difference':b['volume']-a['volume']})
    reference_path=Path(__file__).resolve().parents[1]/'validation/m4_review/fixed.csv'
    rows=list(csv.DictReader(reference_path.open()))
    m4=next(r for r in rows if int(r['N'])==32 and float(r['beta'])==20 and float(r['emin_ratio'])==.001 and r['lateral']=='fixed')
    latest=fixed[-1];reference=float(m4['Fz_top'])
    difference={'projection_N':32,'projection_beta':20.,'projection_emin_ratio':.001,
                'projection_Fz':reference,'binary_geometry_N':latest['geometry_N'],'binary_Fz':latest['Fz'],
                'projection_excess_relative_to_binary':abs(reference)/abs(latest['Fz'])-1,
                'interpretation':'Finite-resolution model difference; both discretization effects remain. Not a pure beta error.'}
    free_comparison=None
    free_geometry=[];free_fe=[];free_screen=False
    free_reference=package/'projection_N32_relaxed_free.json'
    if free_reference.exists():
        free=json.loads(free_reference.read_text())
        if free['status'] != 'ok' or not all(free['checks'].values()):
            raise ValueError('Projection free-lateral reference failed consistency')
        shutil.copyfile(free_reference,output/free_reference.name)
        binary_free=max([r for r in cases if r['geometry_N'] and r['lateral']=='relaxed_free'],key=lambda r:r['geometry_N'])
        series=sorted([r for r in cases if r['geometry_N'] and r['lateral']=='relaxed_free' and r['refinement']==0],key=lambda r:r['geometry_N'])
        free_geometry=[{'from':a['geometry_N'],'to':b['geometry_N'],
                        'relative_Fz_change':abs(b['Fz']-a['Fz'])/abs(b['Fz']),
                        'denominator':'finer geometry response'} for a,b in zip(series,series[1:])]
        for refined in [r for r in cases if r['lateral']=='relaxed_free' and r['refinement']==1]:
            base=next(r for r in series if r['geometry_N']==refined['geometry_N'])
            if abs(base['volume']-refined['volume']) > 1e-9:
                raise ValueError('Free-lateral FE refinement changed geometry')
            free_fe.append({'geometry_N':refined['geometry_N'],
                            'relative_Fz_change':abs(refined['Fz']-base['Fz'])/abs(refined['Fz']),
                            'denominator':'refined FE response','volume_difference':refined['volume']-base['volume']})
        free_screen=bool(free_geometry and free_fe and free_geometry[-1]['relative_Fz_change']<.01 and free_fe[-1]['relative_Fz_change']<.01 and abs(binary_free['volume']-volume['mean'])/volume['mean']<.01)
        free_comparison={'projection_N':free['N'],'projection_Fz':free['Fz_top'],
                         'binary_geometry_N':binary_free['geometry_N'],'binary_Fz':binary_free['Fz'],
                         'projection_excess_relative_to_binary':abs(free['Fz_top'])/abs(binary_free['Fz'])-1,
                         'projection_eps':[free['eps_x'],free['eps_y']],
                         'binary_eps':[binary_free['ex'],binary_free['ey']],
                         'finite_resolution_comparison':True,'global_response_screening_passed':free_screen}
    limits={'geometry_response_change':.01,'fe_response_change':.01,'relative_volume_difference':.01}
    indicators={'last_geometry_Fz_change':geometry_changes[-1]['relative_Fz_change'],
                'last_fixed_geometry_FE_Fz_change':fe_changes[-1]['relative_Fz_change'] if fe_changes else None,
                'finest_volume_difference_from_sampling':abs(latest['volume']-volume['mean'])/volume['mean']}
    screen=indicators['last_geometry_Fz_change'] < .01 and indicators['last_fixed_geometry_FE_Fz_change'] is not None and indicators['last_fixed_geometry_FE_Fz_change'] < .01 and indicators['finest_volume_difference_from_sampling'] < .01
    summary={'cases':cases,'geometry_refinement':geometry_changes,'fixed_geometry_FE_refinement':fe_changes,
             'analytic_volume_sampling':volume,'projection_comparison_fixed':difference,
             'projection_comparison_free':free_comparison,
             'screening_limits':limits,'screening_indicators':indicators,'global_response_screening_passed':bool(screen),
             'rigorous_error_bound':False,'local_stress_convergence_claim':False,
             'unresolved_quality_warning':any(r['distorted_elements'] for r in cases),
             'free_geometry_refinement':free_geometry,'free_fixed_geometry_FE_refinement':free_fe,
             'free_lateral_global_response_screening_passed':free_screen}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    columns=['case','geometry_N','refinement','lateral','nodes','elements','volume','Fz','energy','ex','ey','distorted_elements','mean_ratio_min','physical_consistency']
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(cases)
    root=Path(__file__).resolve().parents[1]
    source_names=['binary_gyroid.py','scripts/prepare_abaqus_binary.py','scripts/extract_abaqus_binary.py',
                  'scripts/extract_uniform_baseline.py','scripts/estimate_binary_volume.py','scripts/run_abaqus_binary.ps1',
                  'scripts/capture_abaqus_binary.py','tests/test_abaqus_binary.py']
    source_names.append('scripts/capture_binary_projection_reference.py')
    source_names.append('scripts/plot_abaqus_binary.py')
    source_names.append('scripts/prepare_binary_lateral_pair.py')
    manifest={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'source_sha256':{name:sha(root/name) for name in source_names},
              'source_hash_scope':'Final scripts at evidence capture. Early inputs were prepared during development; analyzed INP/mesh hashes are authoritative.',
              'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
              'abaqus_release':'2026 (recorded in console logs)',
              'large_inputs_and_ODB_retained_at':str(package),'run_cases':len(cases)}
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'completed_cases':len(cases),'indicators':indicators,'comparison':difference,'screening_passed':bool(screen)},indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();capture(a.package,a.output)
