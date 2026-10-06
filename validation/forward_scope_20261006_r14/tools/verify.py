"""Verify this documentation-only closure and preserve frozen evidence."""
from pathlib import Path
import hashlib, json, re, shutil, subprocess
from urllib.parse import unquote
R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
O=R/'validation/forward_scope_20261006_r14'
D=W/'work/forward_scope_20261006'
def read(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def git(*args): return subprocess.check_output(['git','-C',str(R),*args],text=True).strip()
# Correct two stale status lines in active indexes only.
(R/'docs/RESEARCH_STATUS.md').write_text((W/'PROJECT_OVERVIEW.md').read_text().replace('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md'))
p=R/'validation/README.md';s=p.read_text();old='当前下一项仅[主规划](../docs/RESEARCH_PLAN.md)第3步匹配厚度，以下目录为冻结实验索引，历史建议不生成任务。本轮仅文档/Windows工具整理，零新科学计算。'
assert s.count(old)==1
s=s.replace(old,'四步已完成，r13匹配厚度、r14只读范围收口。下一阶段仅看[主规划](../docs/RESEARCH_PLAN.md)的diverse_04单一几何建议，尚未提交新作业。以下目录为冻结实验索引，历史建议不生成任务；本轮零新力学/Abaqus/完整AD。')
p.write_text(s)
allowed={'AGENTS.md','README.md','docs/RESEARCH_BACKGROUND.md','docs/RESEARCH_PLAN.md','docs/RESEARCH_STATUS.md','docs/FILE_MAP.md','docs/TPMS_RESEARCH_REVIEW.md','scripts/README.md','validation/README.md'}
before=read(O/'before_manifest.json')
changed=[k for k,v in before.items() if not (R/k).is_file() or sha(R/k)!=v]
assert set(changed)<=allowed,changed
assert git('rev-parse','HEAD')==read(O/'input_manifest.json')['source_commit']
inputs=read(O/'input_manifest.json')['input_sha256']
assert all(Path(k).is_file() and sha(Path(k))==v for k,v in inputs.items())
frozen={}
for name in ['void_continuation_20261006_r12','thickness_range_20261006_r13']:
    m=read(R/'validation'/name/'evidence_manifest.json'); count=0
    for key in ['new_evidence_sha256','frozen_input_sha256']:
        for rel,v in m.get(key,{}).items():
            p=Path(rel) if Path(rel).is_absolute() else R/rel
            assert p.is_file() and sha(p)==v, (name,key,rel)
            count+=1
    frozen[name]=count
scope=read(O/'forward_scope.json');dec=read(O/'decision.json');gp=read(O/'gradient_gate_proposal.json')
assert [row['thickness_mm'] for row in scope['rows']]==[.45,.5,.55]
assert dec['main_plan_steps_completed']==[1,2,3,4]
assert not dec['gradient20_certified'] and not dec['strict_shell_quality_certified']
assert all(dec[k]==0 for k in ['new_mechanics_jobs','new_Abaqus_jobs','new_full_AD_jobs','new_training_jobs'])
assert gp['status']=='proposal_only_not_executed'
inventory=read(O/'geometry_inventory.json')
assert all(inventory[k]['basic_periodic_surface_screen_pass'] for k in ['diverse_04','diverse_05','diverse_28'])
assert read(O/'geometry_candidate.json')['status']=='selected_for_proposal_not_simulated'
missing=[];checked=0
def links(p):
    global checked
    for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        link=link.strip('<>')
        if '://' in link or link.startswith('#') or link.startswith('mailto:'):continue
        target=unquote(link.split('#')[0])
        if not target:continue
        checked+=1
        if not (p.parent/target).exists():missing.append((str(p),link))
for rel in allowed: links(R/rel)
links(O/'README.md')
for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','AGENTS.md','START_HERE.md','work/README.md']:links(W/name)
assert not missing,missing
subprocess.run(['git','-C',str(R),'diff','--check'],check=True)
receipt={'source_commit':git('rev-parse','HEAD'),'changed_old_files':sorted(changed),
 'unchanged_tracked_files':len(before)-len(changed),'unexpected_old_changes':[],
 'old_r12_r13_manifest_entries_verified':frozen,'source_and_original_geometry_inputs_verified':len(inputs),
 'markdown_local_links_checked':checked,'broken_local_links':[], 'git_diff_check':'pass',
 'new_mechanics_jobs':0,'new_Abaqus_jobs':0,'new_full_AD_jobs':0,'scientific_scope_assertions':'pass',
 'checks_kind':'document provenance and result reuse only; no repeated mechanics/unit tests because solver unchanged'}
shutil.copy2(__file__,O/'tools/verify.py')
write(O/'verification.json',receipt)
files={str(p.relative_to(R)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json'}
write(O/'evidence_manifest.json',{'source_commit':git('rev-parse','HEAD'),'new_evidence_sha256':files,
 'frozen_input_sha256':inputs,'old_scientific_evidence_unchanged':True,'new_mechanics_jobs':0})
write(D/'verification.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
