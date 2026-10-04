"""Publish the resource boundary and verify frozen evidence, zero new solves."""
from pathlib import Path
import hashlib,json,re,subprocess
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4';HERE=Path(__file__).resolve().parent;W=HERE.parents[1]
def load(f):return json.loads(f.read_text(encoding='utf-8-sig'))
def dump(f,v):f.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert not (HERE/'publish_receipt.json').exists()
summary=load(O/'summary.json');assert summary['stage']=='step4_resource_boundary'
assert summary['budget_counts']['JAX_paths_attempted']==3 and summary['budget_counts']['saved_equilibrium_states']==26
assert summary['budget_counts']['Abaqus_analyses']==0 and summary['TPMS_finite_strain_verified_interval'] is None
assert all(v['all_completed_checks_pass'] for v in summary['cases'])
assert load(O/'regression_final.json')['exit_code']==0 and '148 passed' in (O/'regression_final.log').read_text()
processes=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any('/scripts/finite_strain_gyroid.py' in line for line in processes),'Own scientific capture still running'
base=load(O/'preservation_before.json');after={f:sha(Path(f)) for f in base['sha256']}
changed={f for f,v in after.items() if v!=base['sha256'][f]};assert changed=={str(P/'hyperelastic_fem.py')}
assert sha(O/'source_before/hyperelastic_fem.py')==base['sha256'][str(P/'hyperelastic_fem.py')]
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==base['HEAD']
source=load(O/'source_at_run.json');assert all(sha(P/n)==v for n,v in source.items())
for n in source:
    f=O/'source_after'/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((P/n).read_bytes())
    assert not any(line.rstrip()!=line for line in (P/n).read_text().splitlines()),n
old=load(P/'validation/mechanics_trust_20261003_r4/step3/verification.json')
oldA={**old['old_Abaqus_sha256_after'],**old['new_Abaqus_sha256']}
oldAafter={f:sha(Path(f)) for f in oldA};assert oldAafter==oldA
plan=load(O/'plan.json')
assert all(sha(Path(f))==v for reference in plan['reference'].values() for f,v in reference['sha256'].items())
counts={n:sum('/mechanics_trust_20261003_r4/'+n+'/' in f for f in after) for n in ('step1','step2','step3')}
hist=sum(any('/validation/'+n+'/' in f for n in ('near_term_20261003','geometry_interface_20261003_r2','learning_bridge_20261003_r3')) for f in after);assert hist==338
assert not (O/'N48_d005.json').exists() and not (O/'N64_d005.json').exists()
details=load(O/'details.json');assert details['small_strain_N48']['pass_locked_1percent']
energy=load(O/'energy_groups.json');assert energy['FEM_solves']==0 and set(energy['cases'])=={v['tag'] for v in summary['cases']}
verification={'stage':'step4_frozen_history_and_source_verified','HEAD':base['HEAD'],'prior_files_unchanged_except_allowed_material':True,
    'protected_sha256_after':after,'allowed_modified_source':['hyperelastic_fem.py'],'new_source_sha256':source,
    'historical_files':hist,'prior_step_file_counts':counts,'old_Abaqus_sha256_after':oldAafter,
    'completed_state_checks':True,'regression_passed':148,'no_own_scientific_processes':True}
dump(O/'verification.json',verification)
closure={'date':'2026-10-04','status':'closed_at_configured_resource_boundary','planned_step':4,
    'actual':summary['budget_counts'],'N48':details['N48_resource_boundary'],
    'completed':{'small_strain_N48':details['small_strain_N48'],'coarse_paths_to_five_percent_internal_checks':True,'maintenance_tests':148,'added_physics_tests':4},
    'not_executed':['N64','half increment','independent nonlinear solid reference','10/20 percent extensions','training','design gradient'],
    'TPMS_nonlinear_precision_certified':False,'physical_compression_limit_identified':False,
    'next_stage':'Separately lock targeted same-model tangent linear-solve cost/configuration diagnosis, then restore required precision/reference work',
    'no_commit_push_merge':True}
dump(O/'closure.json',closure)
# Correct table adjacency and remove an obsolete current-stage prefix.
for name,oldtext,newtext in [('FILE_MAP.md','\n\n| ROUND4_STEP4_REPORT','\n| ROUND4_STEP4_REPORT'),
    ('PROJECT_OVERVIEW.md','\n\n| Gyroid有限应变探查','\n| Gyroid有限应变探查'),
    ('PROJECT_OVERVIEW.md','当前[主规划](RESEARCH_PLAN.md)第3步已通过','[主规划](RESEARCH_PLAN.md)第3步已通过')]:
    f=W/name;s=f.read_text()
    if oldtext in s:f.write_text(s.replace(oldtext,newtext))
    else:assert newtext in s
mapping={'RESEARCH_BACKGROUND.md':'docs/RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'docs/RESEARCH_PLAN.md',
         'PROJECT_OVERVIEW.md':'docs/RESEARCH_STATUS.md','FILE_MAP.md':'docs/FILE_MAP.md','ROUND4_STEP4_REPORT.md':'docs/ROUND4_STEP4_REPORT.md'}
for src,dst in mapping.items():
    s=(W/src).read_text().replace('](PROJECT_OVERVIEW.md)','](RESEARCH_STATUS.md)')
    s=s.replace('](NEAR_TERM_REPORT.md)','](../validation/near_term_20261003/README.md)')
    if src=='ROUND4_STEP4_REPORT.md':s=s.replace('work/mechanics_trust_20261004_r4_step4/gyroid_preflight.png','figures/ROUND4_STEP4_GYROID.png')
    (P/dst).write_text(s)
figure=P/'docs/figures/ROUND4_STEP4_GYROID.png';figure.write_bytes((HERE/'gyroid_preflight.png').read_bytes())
report=(W/'ROUND4_STEP4_REPORT.md').read_text();stage_report=report
for name in ('RESEARCH_PLAN.md','RESEARCH_BACKGROUND.md','ROUND4_STEP1_REPORT.md','ROUND4_STEP2_REPORT.md','ROUND4_STEP3_REPORT.md'):
    stage_report=stage_report.replace(']('+name+')','](../../../docs/'+name+')')
stage_report=stage_report.replace('work/mechanics_trust_20261004_r4_step4/gyroid_preflight.png','gyroid_preflight.png');(O/'README.md').write_text(stage_report)
s=(W/'AGENTS.md').read_text()
for name in ('RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','ROUND4_STEP1_REPORT.md','ROUND4_STEP2_REPORT.md','ROUND4_STEP3_REPORT.md','ROUND4_STEP4_REPORT.md','ROUND3_REPORT.md'):
    s=s.replace(']('+name+')','](docs/'+('RESEARCH_STATUS.md' if name=='PROJECT_OVERVIEW.md' else name)+')')
s=s.replace('此工作区主要是阅读副本和历史结果','本目录为正式程序；Windows工作区主要是阅读副本和历史结果');(P/'AGENTS.md').write_text(s)
f=P/'README.md';s=f.read_text();oldtext='下一步是原第4步已验证中等壁厚Gyroid有限应变探查，待锁定输入/预算、尚未启动。'
newtext='[第4步](docs/ROUND4_STEP4_REPORT.md)Gyroid预检查至5%内部通过，N48在0.5%点触及每点时间上限；第四轮已按条件收口，有限应变精度未认证，下一轮待锁定。'
assert oldtext in s or newtext in s;s=s.replace(oldtext,newtext)
s=s.replace('最终144项回归通过。原FEM/密度/PBC/梯度求解器及依赖环境不变。',
    '第4步扩展材料模块、增加Gyroid捕获及4项物理检查，最终148项回归通过。原FEM/密度/PBC/梯度求解器及依赖环境不变。');f.write_text(s)
f=P/'validation/README.md';s=f.read_text();key='| --- | --- |\n';assert key in s
if '[mechanics_trust_20261003_r4/step4]' not in s:
    s=s.replace(key,key+'| [mechanics_trust_20261003_r4/step4](mechanics_trust_20261003_r4/step4/README.md) | Gyroid预检查至5%，N48在0.5%点按资源停止；3条路径尝试/26状态、零新增Abaqus/训练/设计梯度，148项回归；有限应变精度未认证 |\n',1)
f.write_text(s)
check=subprocess.run(['git','-C',str(P),'diff','--check'],capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
active=[W/n for n in ('RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','START_HERE.md','AGENTS.md','ROUND4_STEP4_REPORT.md')]+[P/n for n in list(mapping.values())+['README.md','AGENTS.md','validation/README.md']]+[O/'README.md']
links=0
for f in active:
    s=f.read_text()
    for target in re.findall(r'\]\(([^)]+)\)',s):
        target=target.split('#',1)[0]
        if not target or '://' in target or target.startswith('http'):continue
        assert (f.parent/target).exists(),str(f)+' -> '+target
        links+=1
# Documents changed only; prior frozen bytes and numerical run snapshots still match.
assert all(sha(Path(f))==v for f,v in after.items())
assert all(sha(P/n)==v for n,v in source.items())
receipt={'date':'2026-10-04','status':'step4_resource_boundary_published','HEAD':base['HEAD'],
    'synced_sha256':{src:sha(P/dst) for src,dst in mapping.items()},'Windows_report_sha256':sha(W/'ROUND4_STEP4_REPORT.md'),
    'verification_sha256':sha(O/'verification.json'),'closure_sha256':sha(O/'closure.json'),'source_at_run_sha256':source,
    'relative_links_checked':links,'historical_files_unchanged':hist,'prior_step_files_unchanged':counts,
    'old_Abaqus_files_unchanged':len(oldA),'new_Abaqus_jobs':0,'tests':148,'no_commit_push_merge':True}
dump(HERE/'publish_receipt.json',receipt);dump(O/'publish_receipt.json',receipt);(O/'publish.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in receipt.items() if 'sha256' not in k},indent=2))
