"""Publish step3 facts and verify unchanged numerical/history evidence."""
from pathlib import Path
import hashlib,json,re,subprocess,shutil
W=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step3'
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/mechanics_trust_20261004_r4/step3_uniform')
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(f):return json.loads(f.read_text(encoding='utf-8-sig'))
def dump(f,v):f.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert not (HERE/'publish_receipt.json').exists()
summary=load(O/'summary.json');assert summary['stage']=='round4_step3_uniform_passed' and all(summary['checks'].values())
assert load(O/'regression_final.json')['exit_code']==0 and '144 passed' in (O/'regression_final.log').read_text()
baseline=load(O/'preservation_before.json');after={f:sha(Path(f)) for f in baseline['sha256']}
assert after==baseline['sha256'],'An existing numerical or historical file changed'
head=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip();assert head==baseline['HEAD']
old=load(P/'validation/mechanics_trust_20261003_r4/step2/verification.json')
old_artifacts={**old['old_artifact_sha256_after'],**old['new_primitive_package_sha256']}
old_after={f:sha(Path(f)) for f in old_artifacts};assert old_after==old_artifacts
frozen=[f for f in after if any('/validation/'+n+'/' in f for n in ('near_term_20261003','geometry_interface_20261003_r2','learning_bridge_20261003_r3'))]
prior_steps={n:sum('/mechanics_trust_20261003_r4/'+n+'/' in f for f in after) for n in ('step1','step2')}
assert len(frozen)==338
assert not list(A.rglob('*.lck'))
execution=load(O/'execution.json')
assert all(r['exit_code']==0 for r in execution)
assert sum(r['kind']=='JAX_path' for r in execution)==3
assert sum(r['kind']=='Abaqus_analysis_and_datacheck' for r in execution)==2
new_files=load(O/'new_source_files.json');new_source={n:sha(P/n) for n in new_files}
assert len(new_files)==4 and new_source==load(O/'source_at_run.json')
for name in new_files:
    f=P/name;saved=O/'source_after'/name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(f.read_bytes())
    assert not any(line.rstrip()!=line for line in f.read_text().splitlines()),'Trailing whitespace: '+name
new_artifacts={};jobs=[]
for tag in ('N4_d010','N4_d005'):
    package=A/tag;expected_path=next(package.glob('*.expected.json'));ex=load(expected_path);case=ex['case'];work=package/'work'
    assert sha(package/(case+'.inp'))==ex['input_sha256']
    acceptance=load(work/(case+'.acceptance.json'));assert acceptance['status']=='ok'
    assert len(acceptance['rows'])==len(ex['steps'])+1
    assert all(all(r['checks'].values()) for r in acceptance['rows'])
    assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (work/(case+'.sta')).read_text(errors='replace')
    diagnostics=load(work/(case+'.diagnostics.json'));check_diagnostics=load(work/(case+'_check.diagnostics.json'))
    assert diagnostics['errors']==0 and diagnostics['warning_messages']==0
    assert check_diagnostics['errors']==0 and check_diagnostics['warning_messages']==0
    assert 'ANALYSIS DATACHECK COMPLETE' in (work/(case+'_check.dat')).read_text(errors='replace')
    for f in package.rglob('*'):
        if f.is_file():new_artifacts[str(f)]=sha(f)
    jobs.append({'case':case,'completed_load_steps':len(ex['steps']),'saved_states':len(acceptance['rows']),
                 'analysis_and_datacheck_complete':True,'errors':0,'warnings':0,'physical_acceptance':'ok'})
newton={}
for name in ('N2_d010','N4_d010','N4_d005'):
    counts=[int(k) for k in re.findall(r'Timing summary .*? ([0-9]+) Newton iter',(O/(name+'.log')).read_text())]
    assert len(counts)==len(load(O/(name+'.json'))['rows'])
    assert counts[0]==3 and counts[-1]==3
    newton[name]={'all_stored_state_Newton_corrections':counts,'zero_and_twenty_percent_seeded':True}
dump(O/'verification.json',{'HEAD':head,'original_manifest_files_unchanged':len(after),'historical_frozen_files_unchanged':len(frozen),
    'prior_step_files_unchanged':prior_steps,'old_Abaqus_manifest_files_unchanged':len(old_after),
    'existing_numerical_modules_or_environment_changed':0,'new_source_files':new_source,
    'protected_sha256_after':after,'old_Abaqus_sha256_after':old_after,'new_Abaqus_sha256':new_artifacts,
    'completed_jobs':jobs,'Newton_evidence':newton,'final_test_count':144,'new_case_lock_files':0,'no_Git_commit_push_merge':True})
dump(O/'closure.json',{'date':'2026-10-04','stage':'round4_step3_closed_passed','checks':summary['checks'],
    'actual_counts':{'JAX_paths':3,'saved_JAX_equilibrium_states':86,'Abaqus_analyses':2,'datachecks':2,'new_physics_checks':6,
                     'final_maintenance_test_count':144,'TPMS_paths':0,'design_gradient_experiments':0,'training':0},
    'finite_strain_range':'0-20% uniform full solid only','source_change':'4 new files, no existing numerical edits',
    'does_not_certify':['TPMS finite strain','real material calibration','soft void behavior','buckling/contact','N64 nonlinear resources'],
    'next_step':'original step4: verified medium-wall Gyroid, lock interpolation/reference/resources first; staged 1-5-10%, conditional20%',
    'step4_started':False,'further_thin_reference_work_started':False})
mapping={'RESEARCH_BACKGROUND.md':'docs/RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'docs/RESEARCH_PLAN.md',
    'PROJECT_OVERVIEW.md':'docs/RESEARCH_STATUS.md','FILE_MAP.md':'docs/FILE_MAP.md','AGENTS.md':'AGENTS.md',
    'ROUND4_STEP3_REPORT.md':'docs/ROUND4_STEP3_REPORT.md'}
before=O/'documents_before_close';before.mkdir(exist_ok=False)
for name in list(mapping.values())+['README.md','validation/README.md']:
    f=P/name
    if f.exists():
        b=before/name;b.parent.mkdir(parents=True,exist_ok=True);b.write_bytes(f.read_bytes())
figure=P/'docs/figures/ROUND4_STEP3_UNIFORM.png';figure.parent.mkdir(exist_ok=True);shutil.copy2(O/'uniform_response.png',figure)
for src,name in mapping.items():
    text=(W/src).read_text().replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)')
    if name.startswith('docs/'):
        text=text.replace('(NEAR_TERM_REPORT.md)','(../validation/near_term_20261003/README.md)')
        text=text.replace('(work/mechanics_trust_20261004_r4_step3/uniform_response.png)','(figures/ROUND4_STEP3_UNIFORM.png)')
    if src=='AGENTS.md':
        for link in ('RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','ROUND3_REPORT.md','ROUND4_STEP1_REPORT.md','ROUND4_STEP2_REPORT.md','ROUND4_STEP3_REPORT.md'):
            text=text.replace('('+link+')','(docs/'+link+')')
        text=text.replace('此工作区主要是阅读副本和历史结果。','本目录为正式程序；Windows工作区为阅读副本和历史结果。')
    (P/name).write_text(text)
report=(W/'ROUND4_STEP3_REPORT.md').read_text()
for link in ('RESEARCH_PLAN.md','RESEARCH_BACKGROUND.md'):
    report=report.replace('('+link+')','(../../../docs/'+link+')')
report=report.replace('(work/mechanics_trust_20261004_r4_step3/uniform_response.png)','(uniform_response.png)')
(O/'README.md').write_text(report)
readme=P/'README.md';text=readme.read_text()
start=text.index('当前有限Gyroid');end=text.index('\n\n```bash',start)
replacement='''当前有限Gyroid小变形整体响应及专用导数、一个Primitive小变形整体响应已有证据。前三轮收口，旧停止项不自动续跑。第四轮[第1步](docs/ROUND4_STEP1_REPORT.md)参考审计完成，[第2步](docs/ROUND4_STEP2_REPORT.md)Primitive通过/薄壁参考未成立，[第3步](docs/ROUND4_STEP3_REPORT.md)完整均匀实体有限应变基准通过至20%。下一步是原第4步已验证中等壁厚Gyroid有限应变探查，待锁定输入/预算、尚未启动。均匀体通过不认证TPMS大压缩。训练和逆设计继续后置。综合报告是第四轮制定前的快照，最新状态以主规划为准。

正式路径 `/home/xuehu/projects/tpms_jax`；Pixi锁定环境，JAX-FEM PyPI 0.0.12，Abaqus在Windows。数值HEAD `5e1d03d`，阶段改动未提交。第2步扩展8个已有文件；第3步仅新增一个有限应变材料模块、两个捕获/提取脚本和一个物理测试文件，最终144项回归通过。原FEM/密度/PBC/梯度求解器及依赖环境不变。'''
readme.write_text(text[:start]+replacement+text[end:])
index=P/'validation/README.md';text=index.read_text()
needle='| --- | --- |\n';assert text.count(needle)==1
text=text.replace(needle,needle+'| [mechanics_trust_20261003_r4/step3](mechanics_trust_20261003_r4/step3/README.md) | 第四轮完整均匀实体有限应变基准通过至20%；3条JAX路径、2项分析及2项datacheck，144项回归通过；不认证TPMS大压缩 |\n')
text=text.replace('当前总数30','加第3步2项，当前总数32')
index.write_text(text)
subprocess.run(['git','-C',str(P),'diff','--check'],check=True)
active=[W/f for f in ('AGENTS.md','RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','START_HERE.md','ROUND4_STEP3_REPORT.md')]
active += [P/f for f in mapping.values()]+[P/'README.md',P/'validation/README.md',O/'README.md']
links=0
for f in active:
    for url in re.findall(r'\]\(([^)]+)\)',f.read_text()):
        if '://' in url or url.startswith('#'):continue
        assert (f.parent/url.split('#')[0]).resolve().exists(),(str(f),url)
        links+=1
assert all(sha(Path(f))==h for f,h in after.items())
assert all(sha(Path(f))==h for f,h in old_after.items())
assert all(sha(P/f)==h for f,h in new_source.items())
receipt={'date':'2026-10-04','HEAD':head,'synced_sha256':{f:sha(P/name) for f,name in mapping.items()},
    'Windows_report_sha256':sha(W/'ROUND4_STEP3_REPORT.md'),'closure_sha256':sha(O/'closure.json'),
    'verification_sha256':sha(O/'verification.json'),'active_relative_links_validated':links,
    'prior_step_files_unchanged':prior_steps,'old_Abaqus_manifest_files_unchanged':len(old_after),'no_commit_push_merge':True}
dump(HERE/'publish_receipt.json',receipt);dump(O/'publish_receipt.json',receipt)
(O/'publish.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'status':'step3_completed_published','tests':144,'historical_files':len(frozen),'prior_step_files':prior_steps,
                  'old_Abaqus_files':len(old_after),'existing_numerical_edits':0,'new_files':new_files,'links_checked':links},indent=2))
