from pathlib import Path
import json,re,hashlib,subprocess
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an');O=R/'validation/geometry_interface_20261003_r2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=W/'FILE_MAP.md';t=p.read_text().replace('含证据条件和预算；尚未执行','含证据条件和预算；四步已按判据收口').replace('文件夹状态（第二轮规划完成后）','文件夹状态（第二轮执行收口后）').replace('已同步到第二轮规划，作为当前入口','已同步第二轮执行结论，作为当前入口').replace('第二轮新增程序随实际需要出现，未预搭新架构','第二轮只扩展薄接口和显式阈值传递，没有新增 FEM 或训练架构');p.write_text(t,encoding='utf-8');(R/'docs/FILE_MAP.md').write_text(t.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'),encoding='utf-8')
p=R/'validation/README.md';t=p.read_text().replace('第二轮已制定规划，尚未新增实验目录或计算。当前工作只从主规划继续。','第二轮也已按判据收口，证据见 [geometry_interface_20261003_r2](geometry_interface_20261003_r2/README.md)，结论见 [第二轮报告](../docs/ROUND2_REPORT.md)。不自动续跑停止项或展开下一轮。');p.write_text(t)
p=R/'README.md';t=p.read_text().replace('；下一轮尚未制定／执行：几何输入保真、邻近几何精度、导数及成本，最后选择最小学习问题。当前任务和上限只读主规划。','。本轮覆盖几何输入保真、邻近几何精度、导数及成本，并选择最小后续问题；下一轮尚未制定／执行，当前边界只读主规划。');p.write_text(t)
for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','ROUND2_REPORT.md']:
 p=R/'docs'/name;t=p.read_text().replace('(NEAR_TERM_REPORT.md)','(../validation/near_term_20261003/README.md)');p.write_text(t)
transport=json.loads((W/'work/archive_20261002/transport_dirty_files_before.json').read_text())
for x in transport:assert sha(W/'work/tpms_jax'/x['path'])==x['sha256'],x['path']
manifest=json.loads((O/'source_manifest.json').read_text())
for n,h in manifest['source_sha256'].items():assert sha(R/n)==h,n
for n,h in manifest['experiment_sha256'].items():assert sha(O/n)==h,n
links=[]
for root,names in [(W,['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','ROUND2_REPORT.md','START_HERE.md','AGENTS.md']), (R/'docs',['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','ROUND2_REPORT.md'])]:
 for name in names:
  p=root/name
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if target.startswith(('https:','http:','//','#','chatgpt-')):continue
   q=p.parent/target.split('#')[0]
   assert q.exists(),(p,target)
   links.append({'file':str(p),'target':target})
for local,formal in [('RESEARCH_BACKGROUND.md','RESEARCH_BACKGROUND.md'),('RESEARCH_PLAN.md','RESEARCH_PLAN.md'),('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md'),('FILE_MAP.md','FILE_MAP.md'),('ROUND2_REPORT.md','ROUND2_REPORT.md')]:
 a=(W/local).read_text().replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)').replace('(NEAR_TERM_REPORT.md)','(../validation/near_term_20261003/README.md)');assert a==(R/'docs'/formal).read_text(),local
assert subprocess.run(['git','diff','--check'],cwd=R).returncode==0
assert not any((O/'step3').glob('N64/summary.json'))
result={'links_checked':len(links),'reading_copies_synchronized':5,'all_source_and_experiment_hashes_match':True,'transport_dirty_files_match_original_hashes':True,'git_diff_check_passed':True,'N64_adjoint_not_run':True,'documents_updated_only_after_compute':True}
(O/'final_qa.json').write_text(json.dumps(result,indent=2)+'\n')
(O/'round2_final_qa.py').write_bytes((W/'tmp/round2_final_qa.py').read_bytes())
manifest['document_sha256']={str(p):sha(p) for p in [R/'README.md',R/'AGENTS.md',R/'validation/README.md',*[(R/'docs'/n) for n in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','ROUND2_REPORT.md']]]}
manifest['experiment_sha256']={str(f.relative_to(O)):sha(f) for f in O.rglob('*') if f.is_file() and f.name!='source_manifest.json'}
(O/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(result,indent=2))
