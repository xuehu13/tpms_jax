"""Prepare one duration-only shell diagnostic. No analysis is launched here."""
from pathlib import Path
import hashlib, json, shutil, difflib
R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=R/'validation/shell_rate_20261007_r20'
D.mkdir(exist_ok=False)
P=D/'abaqus/explicit_T0p004';P.mkdir(parents=True)
old=R/'validation/geometry_transfer_20261006_r15/abaqus/explicit_T0p040'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

freeze=[]
for name in ('geometry_transfer_20261006_r15','geometry_transfer_review_20261006_r16',
             'mechanism_20261007_r17','step_control_20261007_r18','dynamic_branch_review_20261007_r19'):
    for p in sorted((R/'validation'/name).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            freeze.append({'path':str(p.relative_to(R)),'sha256':sha(p)})
write(D/'frozen_before.json',freeze)
write(D/'source_before.json',[{'path':n,'sha256':sha(R/n)} for n in
      ('hyperelastic_fem.py','surface_distance.py','pbc.py','scripts/thin_target_explicit.py')])
for root,files,dest in ((W,['RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','AGENTS.md','START_HERE.md'],W/'history/before_shell_rate_20261007'),
    (R,['docs/RESEARCH_PLAN.md','docs/RESEARCH_STATUS.md','docs/FILE_MAP.md','AGENTS.md','README.md'],R/'docs/history/before_shell_rate_20261007')):
    dest.mkdir(exist_ok=False)
    for name in files:
        q=dest/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,q)

cfg=json.loads((old/'input.json').read_text());assert sha(old/'thin_shell.inp')==cfg['shell_inp_sha256']
raw=(old/'thin_shell.inp').read_bytes();new=raw
changes=[(b'0., 0., 0.04, 1., 0.044, 1.',b'0., 0., 0.004, 1., 0.0044, 1.'),
         (b', 0.044\n',b', 0.0044\n'),
         (b'time interval=4.400000000000001e-05',b'time interval=4.400000000000001e-06')]
for a,b in changes:
    assert new.count(a)==1,(a,new.count(a));new=new.replace(a,b)
reverse=new
for a,b in reversed(changes):reverse=reverse.replace(b,a)
assert reverse==raw
(P/'thin_shell.inp').write_bytes(new)
shutil.copy2(old/'extract_thin_explicit.py',P/'extract_thin_explicit.py')
assert sha(P/'extract_thin_explicit.py')==cfg['extractor_sha256']
cfg.update(load_time_seconds=.004,hold_time_seconds=.0004,total_time_seconds=.0044,
           shell_inp_sha256=sha(P/'thin_shell.inp'))
write(P/'input.json',cfg)
diff=''.join(difflib.unified_diff(raw.decode().splitlines(True),new.decode().splitlines(True),
                                fromfile='original_T0p040.inp',tofile='diagnostic_T0p004.inp'))
(D/'input_diff.patch').write_text(diff,encoding='utf-8')
shutil.copy2(R/'validation/mechanism_20261007_r17/extract_shell_modes.py',P/'extract_shell_modes.py')
write(D/'protocol.json',{'research_question':'Does a 10x faster loading shift the observed shell snap toward the existing JAX trajectory?',
    'user_authorization':'2026-10-07: 好的，开始按照规划进行吧', 'plan_step':2,
    'one_new_Abaqus_job':True,'new_JAX_forward':False,'design_AD':False,
    'only_physical_change':'load/hold duration divided by 10; smooth-step shape unchanged',
    'INP_changed_lines':3,'third_change':'history output interval divided by 10, not a physical parameter',
    'same_model_reverse_byte_check':True,'same_extractor_sha256':cfg['extractor_sha256'],
    'native_directory':'E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/shell_rate_20261007_r20_diverse04_explicit_T0p004',
    'job_settings':{'cpus':4,'double':'both','interactive':True},
    'target_compression':.2,'new_load_s':.004,'new_hold_s':.0004,
    'keep_original_quasistatic_checks_as_diagnostics':True,
    'JAX_comparison_limit':.17536302435107093,
    'windows_defined_before_run':{'prejump_compressions':[.01,.05,.10],
        'common_postjump_loading_compression':[.170,.175],
        'shell_hold_mean':'last half of each shell hold; no existing JAX hold comparison'},
    'mode_scope':'existing S3R field output frames, same reference surface; nearest compression labels retained',
    'next_single_factor_chosen_only_after_results':True})

for p in (W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md'):
    text=p.read_text(encoding='utf-8')
    text=text.replace('本轮第1步为只读调查/文档更新，第2～4步尚未执行，不自动提交新作业。',
        '第1步已完成r19。用户2026-10-07授权按规划推进；本轮执行第2步一次壳速率诊断（r20），第3/4步按证据条件推进，不自动扫描或启动AD。')
    text=text.replace('## 2. 下一项：一次复用壳模型的加载速率诊断','## 2. 执行中r20：一次复用壳模型的加载速率诊断')
    text=text.replace('当前没有运行中或排队作业。','准备阶段尚未提交作业，运行状态另见r20收据。')
    p.write_text(text,encoding='utf-8')
write(D/'preparation.json',{'frozen_files':len(freeze),'old_INP_sha256':sha(old/'thin_shell.inp'),
    'new_INP_sha256':sha(P/'thin_shell.inp'),'extractor_sha256':sha(P/'extract_thin_explicit.py'),
    'three_line_diff_verified':True,'reverse_to_original_byte_identical':True,'new_job_submitted':False})
print(diff);print(json.dumps({'frozen_files':len(freeze),'formal_package':str(P)}))
