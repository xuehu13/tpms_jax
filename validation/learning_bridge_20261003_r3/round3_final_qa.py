from pathlib import Path
import json,hashlib,subprocess,re,shutil
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an');O=R/'validation/learning_bridge_20261003_r3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial=json.loads((O/'initial_manifest.json').read_text());facts=json.loads((O/'summary.json').read_text())
counts={}
for n,digest in initial['frozen_manifests'].items():
 root=R/'validation'/n;assert sha(root/'source_manifest.json')==digest
 m=json.loads((root/'source_manifest.json').read_text())
 for name,h in m['experiment_sha256'].items():assert sha(root/name)==h,(n,name)
 counts[n]=len(m['experiment_sha256'])
for name,h in m['installed_jax_fem_sha256'].items():assert sha(R/'.pixi/envs/default/lib/python3.13/site-packages/jax_fem'/name)==h,name
for name,h in initial['source_before'].items():assert sha(O/'source_before'/name)==h,name
core=list(m['source_sha256']);changed=[name for name in core if sha(R/name)!=initial['source_before'][name]]
assert set(changed)=={'design_fem.py','voxel_field.py','binary_gyroid.py','scripts/prepare_abaqus_binary.py'},changed
for name in ['pixi.toml','pixi.lock','fem.py','pbc.py','density_fem.py','geometry.py']:
 assert sha(R/name)==initial['source_before'][name]
assert sha(R/'docs/RESEARCH_PLAN_ROUND3_COMPLETED.md')==initial['source_before']['docs/RESEARCH_PLAN.md']
for p in (O/'source_after').rglob('*'):
 if p.is_file():assert sha(p)==sha(R/p.relative_to(O/'source_after')),p
for local,formal in [('RESEARCH_BACKGROUND.md','RESEARCH_BACKGROUND.md'),('RESEARCH_PLAN.md','RESEARCH_PLAN.md'),('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md'),('FILE_MAP.md','FILE_MAP.md'),('PAPER_ROUTE.md','PAPER_ROUTE.md'),('ROUND3_REPORT.md','ROUND3_REPORT.md')]:
 t=(W/local).read_text().replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)').replace('(NEAR_TERM_REPORT.md)','(../validation/near_term_20261003/README.md)')
 assert t==(R/'docs'/formal).read_text(),local
for item in json.loads((W/'work/archive_20261002/transport_dirty_files_before.json').read_text()):assert sha(W/'work/tpms_jax'/item['path'])==item['sha256']
assert sha(R/'results/m4_numerical_study.csv')=='dc6c3c18881d75737cb99678e60376bc30f54b93f54504ea84a9ff8141bd0fb4'
assert '130 passed' in (O/'regression.log').read_text() and '3 passed' in (O/'new_checks.log').read_text()
assert facts['research_forward_count']==14 and facts['Abaqus_analysis_count']==1
assert facts['performance_span']<facts['background_three_times_mesh_lower_bound']
assert not (O/'step3').exists() and not (O/'step4').exists()
case='binary_gyroid_G32_R0_C3D10_fixed';base=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/learning_bridge_20261003_r3')
root=base/'ax035/G32';expected=json.loads((root/f'{case}.expected.json').read_text());a=json.loads((root/'work'/f'{case}.acceptance.json').read_text())
assert sha(root/f'{case}.inp')==expected['input_sha256']==a['input_sha256']
assert sha(root/f'{case}.mesh.npz')==expected['mesh_sha256']
assert a['status']=='ok' and all(a['checks'].values())
assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (root/'work'/f'{case}.sta').read_text()
assert not (base/'az035').exists() and not list((base/'ax035/G48').glob('*.inp')) and not list((base/'ax035/G48').rglob('*.odb'))
assert not list(base.rglob('*.lck'))
docs=[W/n for n in ['AGENTS.md','START_HERE.md','RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','PAPER_ROUTE.md','ROUND3_REPORT.md']]+[R/n for n in ['AGENTS.md','README.md','validation/README.md','docs/RESEARCH_BACKGROUND.md','docs/RESEARCH_PLAN.md','docs/RESEARCH_STATUS.md','docs/FILE_MAP.md','docs/PAPER_ROUTE.md','docs/ROUND3_REPORT.md']]
linkcount=0
for p in docs:
 for link in re.findall(r'\[[^\]]*\]\(([^\)]+)\)',p.read_text()):
  if link.startswith(('https:','http:','//','#')):continue
  link=link.split('#')[0].strip('<>')
  if link:assert (p.parent/link).exists(),(p,link);linkcount+=1
g=subprocess.run(['git','diff','--check'],cwd=R,capture_output=True,text=True);assert g.returncode==0,g.stdout+g.stderr
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();assert head==initial['HEAD']
qa={'passed':True,'frozen_experiment_files_checked':counts,'changed_existing_numerical_files':changed,'new_test_file':'tests/test_stationary_width.py','old_experiments_CSV_transport_environment_unchanged':True,'relative_links_checked':linkcount,'reading_copies_synchronized':True,'git_diff_check_passed':True,'HEAD_unchanged':True,'completed_Abaqus_case_hashes_and_checks_passed':True,'no_pending_lock_or_training_stage':True,'research_counts_and_stop_conditions_match':True}
(O/'verification.json').write_text(json.dumps(qa,indent=2)+'\n')
manifest={'phase':facts['phase'],'HEAD':head,'initial_source_sha256':initial['source_before'],'source_sha256':{n:sha(R/n) for n in core+['tests/test_stationary_width.py']},'installed_jax_fem_sha256':m['installed_jax_fem_sha256'],'document_sha256':{str(p):sha(p) for p in docs},'frozen_manifests':initial['frozen_manifests'],'Abaqus_artifact_sha256':{str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()},'experiment_sha256':{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='source_manifest.json'},'source_evolution':'stationary geometry interface before step1; binary periodic-width extension after step1; post-test changes are documentation/module description only; old pre-N64 runner added logging/resource guard after N32 without changing numerical algorithm','verification':qa}
(O/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copy2(O/'summary.json',W/'round3_summary.json');shutil.copy2(O/'verification.json',W/'round3_verification.json')
print(json.dumps(qa,indent=2));print('Third-round experiment file count',len(manifest['experiment_sha256']))
