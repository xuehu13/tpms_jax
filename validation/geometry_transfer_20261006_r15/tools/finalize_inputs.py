"""Close only input matching, preserving first failures and all raw scientific bytes."""
from pathlib import Path
import ast,hashlib,json,math,os,shutil,subprocess,textwrap
from types import SimpleNamespace
os.environ['JAX_PLATFORMS']='cpu'
R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
O=R/'validation/geometry_transfer_20261006_r15';D=W/'work/geometry_transfer_20261006'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
cfg=read(O/'input.json');prep=read(O/'preparation.json');diag=read(O/'band_bracket_diagnosis.json')
assert not prep['step1_input_ready'] and not prep['checks']['local_normal_band_brackets']
assert all(v for k,v in prep['checks'].items() if k!='local_normal_band_brackets')
assert diag['all_failed_endpoints_have_first_exit_then_reentry']
assert not (O/'step1_decision.json').exists()
# The new option changes metadata loading only; verify actual default and new readers.
current=(R/'scripts/thin_target_explicit.py').read_text()
old=subprocess.check_output(['git','show',cfg['base_commit']+':scripts/thin_target_explicit.py'],cwd=R,text=True)
cl=lambda s:ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='ExplicitXYZ'))
assert cl(old)==cl(current)
suffix=lambda s:s[s.index('        N=a.cells or cfg'):s.index("if __name__=='__main__':")]
assert suffix(old)==suffix(current)
import sys;sys.path.insert(0,str(R))
from hyperelastic_fem import MU,KAPPA
def load_fragment(code,a,begin):
    ns={'a':a,'json':json,'sha':sha,'math':math,'MU':MU,'KAPPA':KAPPA}
    fragment=textwrap.dedent(code[code.index(begin):code.index('        N=a.cells or cfg')])
    exec(compile(fragment,'existing entry metadata branch only','exec'),ns)
    return ns['cfg']
legacy_a=SimpleNamespace(case=R/'validation/thin_target_20261004_r5',case_input=None)
legacy_before=load_fragment(old,legacy_a,'        source=json.loads')
legacy_after=load_fragment(current,legacy_a,'        source_path=')
assert legacy_before==legacy_after
direct=load_fragment(current,SimpleNamespace(case=O,case_input=O/'input.json'),'        source_path=')
assert all(direct[k]==cfg[k] for k in ['case_id','N','cell_size_mm','thickness_mm','E_MPa','nu','eta','interface_10_90_mm'])
assert direct['source_case_input_sha256']==sha(O/'input.json')
help_run=subprocess.run([str(R/'.pixi/envs/default/bin/python'),'scripts/thin_target_explicit.py','--help'],cwd=R,text=True,capture_output=True)
assert help_run.returncode==0 and '--case-input' in help_run.stdout
(O/'entry_help.log').write_text(help_run.stdout+help_run.stderr)
decision={'status':'matched_inputs_ready_for_single_forward','main_plan_steps_completed':[1],'next_only':'step 2 single matched JAX/shell forward',
 'original_preparation_ready_flag':False,'original_failed_normal_brackets':4,'original_failure_kept':True,
 'targeted_diagnosis':'All four endpoint failures have a first local wall exit, positive sampled void gap, then reentry into another part of the periodic distance band.',
 'readiness_criterion_clarification':'A local wall-width diagnostic needs the first local exit. Being inside material again at an arbitrary full-t endpoint is not by itself a thickness mismatch. Keep the original gate result separate.',
 'affected_first_local_widths_mm':diag['corrected_failed_ray_local_widths_mm'],
 'maximum_affected_local_width_relative_indicator':max(diag['corrected_failed_ray_local_widths_mm'])/.5-1,
 'geometry_representation_scope':'Same midsurface and nominal 0.50mm target; shell normal section and smooth triangle-distance band are distinct approximations, not identical pointwise solids.',
 'global_offset_injectivity_or_contact_certified':False,'finite_normal_sampling_is_global_bound':False,
 'unchanged_physical_and_numerical_constants':True,'default_NH_unchanged':True,
 'explicit_class_AST_unchanged':True,'physical_run_body_after_metadata_unchanged':True,
 'legacy_metadata_reader_equivalent':True,'direct_case_reader_matches_declared_constants':True,'CLI_help_pass':True,
 'new_displacement_solves':0,'new_Abaqus_jobs':0,'new_full_AD_jobs':0,'gradient20_certified':False,
 'input_sha256':prep['input_sha256'],'source_sha256':cfg['solver_source_sha256'],
 'preparation_sha256':sha(O/'preparation.json'),'targeted_diagnosis_sha256':sha(O/'band_bracket_diagnosis.json')}
write(O/'step1_decision.json',decision)
write(O/'execution_notes.json',{'original_preparation_process_returncode':1,
 'original_failure':'NumPy boolean JSON serialization after arrays and native package were generated.',
 'first_recovery_returncode':1,'first_recovery_failure':'Active appended log was incorrectly included in its own immutability check; no input was changed.',
 'second_recovery_returncode':1,'second_recovery_failure':'Canonical report and receipt saved correctly; NumPy-bool console printing then failed. Its report also honestly retained the four geometric endpoint-bracket failures.',
 'report_and_readiness_verified_independently':True,'current_print_helpers_corrected_without_rebuilding_fields':True,
 'all_first_logs_and_attempt_sources_kept':True,'original_FEM_build_repeated':False,
 'original_runtime_and_peak_RSS_not_persisted':True,'no_displacement_or_Abaqus_jobs':True})
shutil.copy2(D/'prepare_inputs.py',O/'tools/prepare_inputs_fixed.py')

# Update existing active documents; no duplicate long progress report.
p=W/'RESEARCH_PLAN.md';s=p.read_text();s=s.replace(
 '更新2026-10-06；依据仓库3629cd7及r12–r14冻结证据。本次只更新状态、归档已完成工具和制定计划，零新力学/Abaqus/完整AD/训练。上一轮四步已完成；下面是新的唯一近期四步，当前均未执行，下一项仅第1步。',
 '更新2026-10-06；第1步匹配输入已完成，依据仓库7e9c642及新r15输入/诊断。当前第2/3/4步待执行，下一项仅第2步一次双方20%前向。本轮零位移求解/Abaqus作业/完整AD/训练；新输入与旧科学证据分别留存。')
s=s.replace('目前未创建实验或提交作业','已建立r15输入目录；未提交压缩作业')
s=s.replace('## 1. 建立双方匹配输入：待执行','## 1. 建立双方匹配输入：已完成（限定输入就绪）')
needle='## 2. 各完成一次20%前向：待执行'
extra='''r15已生成884736个真实HEX27 Gauss点占据、正HRZ质量和8372节点/15914个S3R单元的壳INP；423组XYZ关系/2538方程通过仿射加任意周期波动代入。占据积分体积分数13.2519%，壳面积×厚度名义值13.3032%，相对差−0.3854%，是表示指标，不是力误差。显式入口只加可选`--case-input`，直接读本轮参数JSON；旧默认读取、整个ExplicitXYZ类与物理推进主体一致，不伪造早期线性阶段。

原`preparation.json`仍为false：512个中面法向样本中4条在走出完整t时又落入实体带。针对这4条的首出口/孔隙间隙/再入口诊断已确认；局部首次离开两侧带得到宽度0.500–0.51155mm。判据澄清为检验中面附近的首次局部出口，而非要求任意全t末点一直处于孔隙；原失败不改写，新`step1_decision.json`给限定输入就绪。它不证明全局偏置单射或无接触，不把距离带与壳当完全相同三维实体，也不更改厚度/η/界面。

报告写出/恢复过程的类型、活动日志哈希与打印错误留原日志；数组、INP和原false报告未改。首次构建耗时/峰值未持久化，保持未知，不重复构建补数字。输入与原型源码/环境/记录见`validation/geometry_transfer_20261006_r15`。

'''
assert s.count(needle)==1;s=s.replace(needle,extra+needle)
s=s.replace('下一次推进从本轮第1步开始；本次仅规划与整理，没有任何新求解。','下一次推进从本轮第2步开始；输入就绪不是反力/屈曲响应通过，尚无新20%结果。')
p.write_text(s)
p=W/'PROJECT_OVERVIEW.md';s=p.read_text();s=s.replace(
 '更新2026-10-06；本次整理依据正式仓库3629cd7。上一轮四步及r14收口完成；新一轮diverse_04几何迁移四步已写入[唯一主规划](RESEARCH_PLAN.md)，均未执行，下一项仅第1步匹配输入。本次零新力学/Abaqus/完整AD/训练。',
 '更新2026-10-06；r15第1步匹配输入已完成，依据基线7e9c642。第2/3/4步待执行，下一项仅[唯一主规划](RESEARCH_PLAN.md)第2步双方20%前向。新增Gauss占据/质量/XYZ壳INP，零位移求解/Abaqus作业/完整AD/训练。')
s=s.replace('| 新第1步匹配输入 | 待执行；diverse_04的中面基本筛选已完成，新Gauss占据/HRZ质量和匹配XYZ壳参照尚未建立 |',
 '| 新第1步匹配输入 | 已完成限定输入就绪；884736个真实Gauss点、正HRZ质量、8372节点/15914个S3R单元、423组XYZ关系；原4条法向末点失败及首出口澄清分别保留 |')
s=s.replace('未改运行核、数值常数或科学原件。','材料、内力、HRZ、时间推进及数值常数未改；显式入口新加直接参数JSON选项，旧读取等价，旧科学原件不改。')
s=s.replace('此次只有状态/规划和Windows完成工具归档，','此前仅做状态/规划和完成工具归档；当前r15补匹配输入，')
s=s.replace('下一轮用一个不同几何补迁移证据；','不同几何已完成输入，第2步待给实际响应证据；')
s+='\nr15输入/限定就绪依据在`validation/geometry_transfer_20261006_r15`；新原生包位置见地图。占据体积分数与壳面积×厚度差约0.39%，不等于反力差。四条射线先退出局部壁、间隔孔隙再入带；宽度最大约+2.31%仅为有限局部指标，未认证全局无接触。首次构建计时/峰值未保存，未知保留。\n'
p.write_text(s)
p=W/'RESEARCH_BACKGROUND.md';s=p.read_text().replace(
 '当前唯一近期四步围绕diverse_04的一个匹配几何工况，均未执行，下一项为双方输入匹配。',
 '当前唯一近期四步围绕diverse_04的一个匹配几何工况；第1步限定输入就绪已完成，下一项为一次双方20%前向，当前没有新几何响应结果。')
p.write_text(s)
p=W/'START_HERE.md';s=p.read_text().replace('当前唯一近期四步为diverse_04单一几何匹配前向，均未执行，下一项仅第1步建立双方输入。',
 '当前diverse_04四步的第1步限定输入就绪已完成；原4条法向末点失败/首出口诊断分别留存。下一项仅第2步双方20%前向，尚无新压缩结果。')
p.write_text(s)
p=W/'AGENTS.md';s=p.read_text().replace('均未执行，下一项仅第1步，尚无新Gauss占据/XYZ参照或作业。',
 'r15第1步限定输入就绪已完成，下一项仅第2步；真实Gauss/HRZ/XYZ S3R INP已建立但零压缩作业。原preparation=false的4条法向末点失败保留，首出口—孔隙—再入带诊断后step1_decision给限定就绪，不认证全局偏置单射/无接触；不改厚度或常数。显式入口仅新增--case-input，旧读取等价，材料/内力/HRZ/推进主体不变。')
p.write_text(s)
p=W/'FILE_MAP.md';s=p.read_text().replace('当前第1步待执行','第1步输入已就绪，第2步待执行')
s=s.replace('| validation/forward_scope_20261006_r14/ |',
 '| validation/geometry_transfer_20261006_r15/ | diverse_04输入/Gauss占据/HRZ/XYZ壳包；原4条法向失败、首出口诊断、step1_decision限定就绪；零压缩/AD |\n| validation/forward_scope_20261006_r14/ |')
s=s.replace('新几何实验将按唯一规划在执行时建独立日期r15目录，当前不存在新场、INP、ODB或计算结果。',
 'r15已建匹配输入/Gauss占据与HRZ场及壳INP；当前没有新ODB或压缩响应。Windows `work/geometry_transfer_20261006`仅本步准备/恢复/诊断/同步收据；正式程序仍唯一共享FEM和显式入口。')
s+='\nr15正式位置：`/home/xuehu/projects/tpms_jax/validation/geometry_transfer_20261006_r15`。`gauss_field.npz`是真实27点占据，`hrz_mass.npz`是节点/周期质量，`surface_geometry.npz`为中面；`preparation.json`保留原false，`band_bracket_diagnosis.json`解释4条法向，`step1_decision.json`为当前限定就绪。规范壳包在其`abaqus/explicit_T0p040/`，同字节原生包在`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_transfer_20261006_r15_diverse04_explicit_T0p040`；无ODB。科学数组、首次日志/源码原字节保留，计时未知不补造。\n'
p.write_text(s)
p=W/'TPMS_RESEARCH_REVIEW.md';s=p.read_text().replace(
 '更新：2026-10-06，依据正式仓库3629cd7及r12–r14冻结证据。原四步已收口；此次更新状态、归档完成工具并将diverse_04单一几何迁移写入唯一近期四步，尚未建立新输入或求解。',
 '更新：2026-10-06，依据基线7e9c642及r15输入/诊断。原四步已收口；diverse_04单一几何迁移第1步已达限定输入就绪，未提交位移/Abaqus/AD作业。')
s=s.replace('当前唯一近期规划围绕一个几何迁移工况，保持t=0.50mm及所有既定材料/算法条件，均未执行。',
 '当前唯一近期规划围绕一个几何迁移工况，保持t=0.50mm及所有既定材料/算法条件；第1步输入已完成，第2步双方前向待执行。')
s=s.replace('基本中面检查不等于新仿真已经准备好：diverse_04还没有本项目t=0.50mm、匹配NH与XYZ的新壳参照或新Gauss距离/占据/质量场。下一阶段需要双方重新匹配；不能复用diverse_28占据、原XY约束、原厚度/塑性/压板结果。',
 'r15现已建立diverse_04的0.50mm新Gauss距离/占据/HRZ质量和匹配NH/XYZ壳INP；真正的壳响应尚未求解。没有复用diverse_28占据、原XY约束或原厚度/塑性/压板结果。原法向末点检查4条失败保留，首出口—孔隙—再入带诊断支持限定输入就绪；四条局部宽度0.500–0.51155mm、占据积分与壳面积×厚度相差约0.39%仅为几何指标，不是反力或全局无接触认证。显式入口只补直接参数JSON读取，整个物理推进主体与ExplicitXYZ类保持。')
p.write_text(s)

mapping={'RESEARCH_BACKGROUND.md':'docs/RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'docs/RESEARCH_PLAN.md',
 'PROJECT_OVERVIEW.md':'docs/RESEARCH_STATUS.md','FILE_MAP.md':'docs/FILE_MAP.md','TPMS_RESEARCH_REVIEW.md':'docs/TPMS_RESEARCH_REVIEW.md',
 'START_HERE.md':'README.md','AGENTS.md':'AGENTS.md'}
for src,dest in mapping.items():
    s=(W/src).read_text().replace('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md').replace('output/figures/','figures/')
    if dest=='docs/FILE_MAP.md':s=s.replace('[归档索引](history/completed_tools_20261006/README.md)','`Windows history/completed_tools_20261006/README.md`（Windows归档索引）')
    if dest=='docs/TPMS_RESEARCH_REVIEW.md':s=s.replace('figures/VOID_CONTINUATION_response.png','../validation/void_continuation_20261006_r12/response.png')
    if dest in ['AGENTS.md','README.md']:
        for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','THICKNESS_RANGE_PROGRESS.md','VOID_CONTINUATION_PROGRESS.md','PAPER_ROUTE.md']:
            s=s.replace(']('+name+')','](docs/'+name+')')
    (R/dest).write_text(s)
p=R/'scripts/README.md';s=p.read_text().replace('新一轮diverse_04单一几何匹配四步待执行，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第1步建立双方输入',
 'diverse_04第1步限定输入就绪已完成，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第2步一次双方前向')
s+='\n`thin_target_explicit.py --case-input <物理参数JSON>`可直接读取新几何参数，不要求伪造旧线性阶段；旧默认路径/读取保持。该入口验证E/ν、L和密度确实匹配现有固定材料/单位常数。仅元数据适配，ExplicitXYZ类及物理时间推进主体未改；r15输入就绪不等于新20%反力或梯度认证。\n'
p.write_text(s)
p=R/'validation/README.md';s=p.read_text().replace('新的唯一近期四步围绕diverse_04单一几何，均待执行，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第1步输入匹配，尚无新实验或作业。',
 'diverse_04的r15第1步匹配输入已达限定就绪，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第2步；没有新位移求解/Abaqus作业/完整AD。')
s=s.replace('| [forward_scope_20261006_r14]',
 '| [geometry_transfer_20261006_r15](geometry_transfer_20261006_r15/README.md) | diverse_04新Gauss/HRZ/XYZ壳输入；原法向末点失败保留、首出口诊断后限定输入就绪；零压缩/AD |\n| [forward_scope_20261006_r14]')
p.write_text(s)
(O/'README.md').write_text('''# diverse_04：第1步限定输入就绪，零压缩作业

科研问题：固定的0.50mm/NH/XYZ/HEX27/C²方法能否迁移到不同中面？本步只建立双方物理/离散输入，实际响应和梯度未验证。下一项按[唯一规划](../../docs/RESEARCH_PLAN.md)第2步一次双方20%前向。

- 原中面来自用户MS9/diverse_04；8372节点/15914三角面，L10mm、t0.50mm、E10MPa/ν0.3/密度10⁻⁹ tonne/mm³；无接触/塑性。研究候选明确objective_void，φ界限/η/界面/续接常数不变。
- gauss_field.npz：唯一Problem真实27点/884736点的距离占据；hrz_mass.npz：原HRZ节点/周期质量，均正、守恒；surface_geometry.npz：原中面。
- 占据积分Vf13.2519%，壳面积×厚度名义Vf13.3032%，相对−0.3854%；几何目标匹配但表示不同，不是力误差或精确三维实体认证。
- abaqus/explicit_T0p040/thin_shell.inp为规范包；XYZ423组关系/2538方程，用仿射和任意周期波动独立代入检查。原生同字节包在E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_transfer_20261006_r15_diverse04_explicit_T0p040；无ODB。
- preparation.json保持原false：4个全t末点仍在带内。band_bracket_diagnosis.json确认四条首出口→孔隙→再入带，首局部宽度0.500–0.51155mm；step1_decision.json据此给限定输入就绪。没有把旧失败改成通过，没有改变几何/数值常数；不认证全局偏置单射或无接触。

显式入口仅新增--case-input，直接物理JSON读取；旧读取输出等价，整个ExplicitXYZ类与物理推进主体相同。源/输入/环境及决定均有哈希。默认NH不改，当前20%新核AD未认证。

首次数组/INP生成完成后，报告np.bool写出失败；恢复时活动日志自哈希、最后np.bool打印先后失败，原日志/源码和false报告均留存。最终规范报告保存后只读核对，判据澄清另存decision，不重复FEM构建或修改原场。首次构建耗时/峰值未持久化，按未知报告；execution_notes.json说明边界。

tools是本步一次性生成/诊断/报告收据，不是活动FEM；不自动重跑。大数组、原日志/尝试脚本及原中面复制留本机，Git仅规范INP、关键JSON/摘要/源码。克隆不等于取得所有本机档案，位置见[地图](../../docs/FILE_MAP.md)。
''')
(O/'.gitignore').write_text('input/shell_mesh.inc\nabaqus/**/extract_thin_explicit.py\ntools/*attempt*.py\ntools/recover_report*.py\n')
shutil.copy2(__file__,O/'tools/finalize_inputs.py')
print(json.dumps({'step1':'matched inputs ready with targeted bracket clarification','next_only':'step2','new_displacement_solves':0,'new_Abaqus_jobs':0},ensure_ascii=False))
