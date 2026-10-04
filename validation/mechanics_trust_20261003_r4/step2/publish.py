"""Close stage2 with physical evidence and unchanged historical fingerprints."""
from pathlib import Path
import hashlib,json,re,subprocess,difflib,shutil
W=Path(__file__).resolve().parents[2];P=Path('/home/xuehu/projects/tpms_jax')
O=P/'validation/mechanics_trust_20261003_r4/step2'
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/mechanics_trust_20261004_r4')
HERE=Path(__file__).resolve().parent
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(f):return json.loads(f.read_text(encoding='utf-8-sig'))
def dump(f,value):f.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert not (HERE/'publish_receipt.json').exists(),'Publication only once'
assert load(O/'regression_final.json')['exit_code']==0
assert '138 passed' in (O/'regression_final.log').read_text()
baseline=load(O/'preservation_before.json');after={f:sha(Path(f)) for f in baseline['sha256']}
changed=[f for f,h in baseline['sha256'].items() if after[f]!=h]
allowed=['geometry.py','binary_gyroid.py','scripts/prepare_abaqus_binary.py','scripts/m4_numerical_study.py',
         'scripts/capture_binary_projection_reference.py','scripts/run_abaqus_binary.ps1','scripts/abaqus_mesh_quality.py','tests/test_abaqus_binary.py']
assert set(changed)=={str(P/f) for f in allowed},changed
frozen=[f for f in baseline['sha256'] if any('/validation/'+name+'/' in f for name in ('near_term_20261003','geometry_interface_20261003_r2','learning_bridge_20261003_r3'))]
step1=[f for f in baseline['sha256'] if '/mechanics_trust_20261003_r4/step1/' in f]
assert len(frozen)==338
old=load(P/'validation/mechanics_trust_20261003_r4/step1/verification.json')
artifacts={f:sha(Path(f)) for f in old['artifact_sha256']}
assert artifacts==old['artifact_sha256'] and len(artifacts)==112
head=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()
assert head==baseline['HEAD']
assert not list(A.rglob('*.lck'))
summary=load(O/'summary.json');rows=summary['cases'];ops=load(O/'execution.json')
assert [r['status'] for r in rows]==['reference_not_established','global_stiffness_screen_passed']
assert sum(r['kind']=='background_forward' and r['exit_code']==0 for r in ops)==2
assert sum(r['kind']=='Abaqus_analysis_and_datacheck' and r['exit_code']==0 for r in ops)==2
assert not any(r['kind']=='conditional_soft_void_diagnostic' for r in ops)
new_artifacts={};completed=[]
for g in (32,48):
    folder=A/f'primitive/G{g}';expected_path=next(folder.glob('*.expected.json'));ex=load(expected_path);case=ex['case'];work=folder/'work'
    assert sha(folder/(case+'.inp'))==ex['input_sha256']
    assert sha(folder/(case+'.mesh.npz'))==ex['mesh_sha256']
    accepted=load(work/(case+'.acceptance.json'))
    assert accepted['status']=='ok' and all(accepted['checks'].values())
    assert load(work/(case+'.diagnostics.json'))['errors']==0
    assert load(work/(case+'_check.diagnostics.json'))['errors']==0
    sta=(work/(case+'.sta')).read_text(errors='replace')
    assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta
    completed.append({'case':case,'sta_completion':True,'analysis_errors':0,'datacheck_errors':0,'physical_acceptance':'ok'})
    for f in folder.rglob('*'):
        if f.is_file():new_artifacts[str(f)]=sha(f)
for tag in ('G48','G48_clipfix','G48_contactfix'):
    folder=A/'thin_gyroid'/tag
    assert (folder/'FAILED_BEFORE_SUBMISSION.json').exists()
    assert not list(folder.glob('*.inp')) and not list(folder.rglob('*.odb'))
diff=[]
for name in allowed:
    f=P/name;b=O/'source_before'/name;a=O/'source_after'/name
    assert sha(b)==baseline['sha256'][str(f)]
    a.parent.mkdir(parents=True,exist_ok=True);a.write_bytes(f.read_bytes())
    diff.extend(difflib.unified_diff(b.read_text().splitlines(keepends=True),f.read_text().splitlines(keepends=True),fromfile='before/'+name,tofile='after/'+name))
(O/'source_changes.diff').write_text(''.join(diff))
dump(O/'verification.json',{'HEAD':head,'historical_frozen_files_unchanged':len(frozen),'step1_files_unchanged':len(step1),
    'before_manifest_files':len(after),'source_files_changed':allowed,'remaining_manifest_files_unchanged':len(after)-len(changed),
    'original_package_files_unchanged':len(artifacts),'source_sha256_after':{f:after[str(P/f)] for f in allowed},
    'protected_sha256_after':{f:h for f,h in after.items() if f not in changed},'old_artifact_sha256_after':artifacts,
    'new_primitive_package_sha256':new_artifacts,'completed_new_analyses':completed,'new_case_lock_files':0,
    'final_tests':{'passed':138,'log':'regression_final.log'},'no_Git_commit_push_merge':True})
dump(O/'closure.json',{'date':'2026-10-04','stage':'round4_step2_closed_under_stop_conditions','cases':rows,
    'actual_counts':{'background_forward':2,'Abaqus_analysis':2,'Abaqus_datacheck':2,'conditional_diagnostic':0,'gradient':0,'training':0},
    'thin_reference_preparation_failures':3,'thin_mechanics_runs':0,'thin_failure_stage':'reference clipping; no input or job submitted',
    'thin_original_threshold_vertex_candidate_gap_unfixed':True,'further_geometry_rebuild_this_round':False,
    'Primitive_reference_refinement_couples_geometry_and_FE':True,'reference_error_bound_or_local_stress_claim':False,
    'steps3_4_started':False,'next_step':'original step3: matched full-solid finite-strain benchmark; lock energy/loading/budget first'})
mapping={'RESEARCH_BACKGROUND.md':'docs/RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'docs/RESEARCH_PLAN.md',
    'PROJECT_OVERVIEW.md':'docs/RESEARCH_STATUS.md','FILE_MAP.md':'docs/FILE_MAP.md','AGENTS.md':'AGENTS.md',
    'ROUND4_STEP2_REPORT.md':'docs/ROUND4_STEP2_REPORT.md'}
before_dir=O/'documents_before_close';before_dir.mkdir(exist_ok=False)
for name in list(mapping.values())+['README.md','validation/README.md']:
    f=P/name
    if f.exists():
        b=before_dir/name;b.parent.mkdir(parents=True,exist_ok=True);b.write_bytes(f.read_bytes())
figure=P/'docs/figures/ROUND4_STEP2_PRIMITIVE.png';figure.parent.mkdir(exist_ok=True)
shutil.copy2(O/'primitive_response.png',figure)
for src,name in mapping.items():
    text=(W/src).read_text().replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)')
    if name.startswith('docs/'):
        text=text.replace('(NEAR_TERM_REPORT.md)','(../validation/near_term_20261003/README.md)')
        text=text.replace('(work/mechanics_trust_20261004_r4_step2/primitive_response.png)','(figures/ROUND4_STEP2_PRIMITIVE.png)')
    if src=='AGENTS.md':
        for link in ('RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','ROUND3_REPORT.md','ROUND4_STEP1_REPORT.md','ROUND4_STEP2_REPORT.md'):
            text=text.replace('('+link+')','(docs/'+link+')')
        text=text.replace('此工作区主要是阅读副本和历史结果。','本目录为正式程序；Windows工作区为阅读副本和历史结果。')
    (P/name).write_text(text)
report=(W/'ROUND4_STEP2_REPORT.md').read_text()
for link in ('RESEARCH_PLAN.md','RESEARCH_BACKGROUND.md'):
    report=report.replace('('+link+')','(../../../docs/'+link+')')
report=report.replace('(work/mechanics_trust_20261004_r4_step2/primitive_response.png)','(primitive_response.png)')
(O/'README.md').write_text(report)
readme=P/'README.md';text=readme.read_text()
start=text.index('当前小变形 Gyroid');end=text.index('\n\n```bash',start)
text=text[:start]+'''当前有限Gyroid小变形整体刚度及专用导数已有证据。前三轮收口，旧停止项不自动续跑。第四轮[第1步](docs/ROUND4_STEP1_REPORT.md)参考审计完成；[第2步](docs/ROUND4_STEP2_REPORT.md)Primitive整体响应工作筛查通过，薄壁Gyroid G48裁剪参考未成立、未进入力学计算。下一步是原第3步完整均匀实体有限应变基准，尚未启动。大压缩/整个构型族/网络反传尚未认证，训练和逆设计继续后置。综合报告是第四轮制定前的快照，最新状态以主规划为准。

正式路径 `/home/xuehu/projects/tpms_jax`；Pixi锁定环境，JAX-FEM PyPI 0.0.12，Abaqus在Windows。数值HEAD `5e1d03d`，阶段改动未提交。第2步最小扩展8个输入/几何/提取/测试文件，最终138项回归通过；原FEM/密度/PBC/梯度求解器及依赖环境不变。'''+text[end:]
readme.write_text(text)
index=P/'validation/README.md';text=index.read_text()
text=text.replace('零新增求解；第2步尚未启动','零新增求解；原始记录冻结')
text=text.replace('当前总数28','加第四轮第2步2项，当前总数30')
needle='| --- | --- |\n';assert text.count(needle)==1
text=text.replace(needle,needle+'| [mechanics_trust_20261003_r4/step2](mechanics_trust_20261003_r4/step2/README.md) | 第四轮新构型/薄壁筛查：Primitive整体响应通过，薄壁细参考未成立；2前向、2分析、2datacheck；最终138项回归通过 |\n')
index.write_text(text)
subprocess.run(['git','-C',str(P),'diff','--check'],check=True)
# Validate only active publication links; historical reports remain unchanged.
active=[W/f for f in ('AGENTS.md','RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','START_HERE.md','ROUND4_STEP2_REPORT.md')]
active+=[P/f for f in mapping.values()]+[P/'README.md',P/'validation/README.md',O/'README.md']
links=0
for f in active:
    for url in re.findall(r'\]\(([^)]+)\)',f.read_text()):
        if '://' in url or url.startswith('#'):continue
        path=(f.parent/url.split('#')[0]).resolve()
        assert path.exists(),(str(f),url)
        links+=1
assert all(sha(Path(f))==h for f,h in baseline['sha256'].items() if f not in changed)
assert all(sha(Path(f))==h for f,h in artifacts.items())
receipt={'date':'2026-10-04','HEAD':head,'synced_sha256':{f:sha(P/name) for f,name in mapping.items()},
    'report_sha256':sha(W/'ROUND4_STEP2_REPORT.md'),'root_and_formal_relative_links_validated':links,
    'closure_sha256':sha(O/'closure.json'),'verification_sha256':sha(O/'verification.json'),
    'no_new_jobs_after_closure':True,'no_commit_push_merge':True}
dump(HERE/'publish_receipt.json',receipt);dump(O/'publish_receipt.json',receipt)
(O/'publish.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'status':'step2_closed_and_published','frozen_files':len(frozen),'step1_files':len(step1),'old_package_files':len(artifacts),'changed_sources':allowed,'links_checked':links,'tests':138},ensure_ascii=False,indent=2))
