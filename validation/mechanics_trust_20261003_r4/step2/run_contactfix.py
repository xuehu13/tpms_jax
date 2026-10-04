"""Sequential bounded cases; original solvers/tools execute actual analysis."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/mechanics_trust_20261004_r4')
plan=json.loads((O/'plan.json').read_text());retry='--resume' in sys.argv;contactfix='--contactfix' in sys.argv
ledger=json.loads((O/'execution.json').read_text()) if retry else [];results=[]
if not retry:assert not (O/'execution.json').exists()
(O/('run_contactfix.py' if contactfix else ('run_resume.py' if retry else 'run.py'))).write_bytes(Path(__file__).read_bytes())
env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false','PYTHONDONTWRITEBYTECODE':'1'}


def load(f):
    return json.loads(f.read_text(encoding='utf-8-sig'))


def dump(f,value):
    f.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def win(f):
    return str(f).replace('/mnt/e/','E:/').replace('/','\\')


def execute(cmd,log,kind,case,timeout):
    t=time.monotonic();status=None
    with log.open('x') as stream:
        try:
            result=subprocess.run(cmd,cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            status=result.returncode
        except subprocess.TimeoutExpired:
            status='timeout'
    ledger.append({'case':case,'kind':kind,'command':cmd,'seconds':time.monotonic()-t,'exit_code':status,'log':str(log)})
    dump(O/'execution.json',ledger)
    print(json.dumps(ledger[-1]),flush=True)
    return status==0


for definition in plan['cases']:
    label,family,c=(definition[k] for k in ('label','family','c'))
    folder=O/label;folder.mkdir(exist_ok=retry)
    packages=[];record={**definition,'status':'preparing'}
    for g in (32,48):
        tag=f'G{g}_contactfix' if contactfix and label=='thin_gyroid' and g==48 else (f'G{g}_clipfix' if retry and label=='thin_gyroid' and g==48 else f'G{g}')
        package=A/label/tag
        cmd=[sys.executable,str(P/'scripts/prepare_abaqus_binary.py'),'--output',str(package),'--n',str(g),'--c',str(c),'--family',family,'--lateral','fixed']
        existing=next(package.glob('*.expected.json'),None) if package.exists() else None
        if not existing and not execute(cmd,folder/(tag+'.prepare.log'),'geometry_preparation',f'{label}_{tag}',1800):
            package.mkdir(parents=True,exist_ok=True)
            dump(package/'FAILED_BEFORE_SUBMISSION.json',{'stage':'geometry','log':str(folder/(tag+'.prepare.log')),'no_Abaqus_submission':True})
            record['status']='reference_not_established';record['failure_log']=str(folder/(tag+'.prepare.log'))
            break
        manifest=next(package.glob('*.expected.json'));ex=load(manifest)
        (package/'scripts').mkdir(exist_ok=bool(existing))
        for name in ('run_abaqus_binary.ps1','extract_abaqus_binary.py','extract_uniform_baseline.py','abaqus_mesh_quality.py'):
            if not existing:shutil.copy2(P/'scripts'/name,package/'scripts'/name)
        if not existing:shutil.copy2(manifest,folder/(tag+'.expected.json'))
        assert ex['c']==c and ex['family']==family
        packages.append((g,package,ex))
    if record['status']=='reference_not_established':
        results.append(record);dump(O/'progress.json',results);continue
    backgrounds=[]
    for n in (48,64):
        out=folder/f'N{n}.json'
        cmd=[sys.executable,str(P/'scripts/capture_binary_projection_reference.py'),'--N',str(n),'--beta','40','--emin-ratio','1e-4','--lateral','fixed','--solver','petsc','--c',str(c),'--family',family,'--out',str(out)]
        if n==64:cmd+=['--field-out',str(folder/'N64.displacement.npz')]
        if not out.exists() and not execute(cmd,folder/f'N{n}.log','background_forward',f'{label}_N{n}',1200):
            record['status']='background_failed';record['failure_log']=str(folder/f'N{n}.log');break
        backgrounds.append(load(out))
    if record['status']=='background_failed':
        results.append(record);dump(O/'progress.json',results);continue
    references=[]
    for g,package,ex in packages:
        cmd=['/mnt/c/Users/xuehu/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',win(package/'scripts/run_abaqus_binary.ps1'),'-PackageDirectory',win(package),'-Cases',ex['case'],'-Cpus','2','-Memory','8gb']
        if g==48:cmd+=['-QualityDiagnostics']
        accepted=package/'work'/(ex['case']+'.acceptance.json')
        joblog=folder/(f'G{g}.abaqus.retry.log' if retry else f'G{g}.abaqus.log')
        if not accepted.exists() and not execute(cmd,joblog,'Abaqus_analysis_and_datacheck',f'{label}_G{g}',1800):
            record['status']='reference_analysis_or_acceptance_failed';record['failure_log']=str(folder/f'G{g}.abaqus.log');break
        acceptance=load(package/'work'/(ex['case']+'.acceptance.json'))
        diagnostics=load(package/'work'/(ex['case']+'.diagnostics.json'))
        references.append({'G':g,'package':str(package),'case':ex['case'], 'K':acceptance['measured']['macro_RF'][2]/acceptance['measured']['macro_displacements'][2],
                           'acceptance':acceptance,'diagnostics':diagnostics,'expected':ex})
    if record['status'].endswith('failed'):
        results.append(record);dump(O/'progress.json',results);continue
    k48,k64=[b['Fz_top']/-.01 for b in backgrounds];kr32,kr48=[r['K'] for r in references]
    record.update(status='computed',K_N48=k48,K_N64=k64,K_G32=kr32,K_G48=kr48,
                  background_grid_indicator=abs(k48-k64)/k64,reference_grid_indicator=abs(kr32-kr48)/kr48,
                  model_difference=abs(k64-kr48)/kr48,
                  vf_background=backgrounds[1]['vf_int'],vf_same_gauss_binary=backgrounds[1]['vf_binary_ref'],
                  vf_reference=references[1]['acceptance']['measured']['volume_solid'])
    fine=references[1];package=Path(fine['package']);quality=load(package/'work'/(fine['case']+'.quality.json'))
    record['reference_quality']={k:quality[k] for k in ('quality_min','quality_count_p01','Abaqus_distorted','low_quality')}
    with np.load(folder/'N64.displacement.npz') as field, np.load(package/'work'/(fine['case']+'.nodal.npz')) as sample:
        grid=field['total_u'];H=field['H'];points=sample['points'];ur=sample['u'];axis=np.linspace(0,1,65)
        ub=RegularGridInterpolator((axis,axis,axis),grid)(np.clip(points,0,1))
        delta=ub-ur;delta[:,:2]-=delta[:,:2].mean(axis=0)
        wb=ub-points@H.T;wr=ur-points@H.T
        wb[:,:2]-=wb[:,:2].mean(axis=0);wr[:,:2]-=wr[:,:2].mean(axis=0)
        record['mode_diagnostics']={'sample_nodes':len(points),'rms_displacement_difference_over_axial_displacement':float(np.sqrt(np.mean(delta**2))/.01),
                                    'warp_rms_reference':float(np.sqrt(np.mean(wr**2))),
                                    'warp_difference_relative_rms':float(np.sqrt(np.mean((wb-wr)**2))/np.sqrt(np.mean(wr**2))),
                                    'warp_flattened_correlation':float(np.corrcoef(wb.ravel(),wr.ravel())[0,1]),
                                    'translation_alignment':'only free global x/y translations; axial fields unshifted'}
        np.savez_compressed(folder/'mode_sample.npz',points=points,background_u=ub,reference_u=ur,H=H)
    record['checks']={'background_grid<=1pct':record['background_grid_indicator']<=.01,'reference_grid<=1pct':record['reference_grid_indicator']<=.01,'model_difference<=2pct':record['model_difference']<=.02}
    record['status']='global_stiffness_screen_passed' if all(record['checks'].values()) else 'precision_insufficient'
    dump(folder/'comparison.json',record);results.append(record);dump(O/'progress.json',results)
    print(json.dumps(record,indent=2),flush=True)

# One predeclared mechanism check, not a parameter sweep or redefinition of acceptance.
candidate=next((r for r in results if r['status']=='precision_insufficient' and r['reference_grid_indicator']<=.01),None)
if candidate:
    label,family,c=(candidate[k] for k in ('label','family','c'));folder=O/label;out=folder/'N64_emin5_diagnostic.json'
    cmd=[sys.executable,str(P/'scripts/capture_binary_projection_reference.py'),'--N','64','--beta','40','--emin-ratio','1e-5','--lateral','fixed','--solver','petsc','--c',str(c),'--family',family,'--out',str(out)]
    assert not any(x['kind']=='conditional_soft_void_diagnostic' for x in ledger), 'Only one diagnostic is authorized'
    if execute(cmd,folder/'N64_emin5_diagnostic.log','conditional_soft_void_diagnostic',label,1200):
        row=load(out);k=row['Fz_top']/-.01
        candidate['soft_void_diagnostic']={'K_emin5':k,'relative_change_from_emin4':abs(k-candidate['K_N64'])/candidate['K_N64'],
                                           'difference_to_reference':abs(k-candidate['K_G48'])/candidate['K_G48'],'original_status_unchanged':True}
        dump(folder/'comparison.json',candidate)
dump(O/'summary.json',{'stage':'round4_step2_computed','cases':results,'operations':ledger,'steps3_4_started':False})
print('STEP2_FINITE_ELEMENT_RUNS_FINISHED',flush=True)
