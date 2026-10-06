"""Close the current four steps and record bounded future gates; no new solve."""
from pathlib import Path
import hashlib, json, shutil
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/forward_scope_20261006_r14'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def once(s,old,new):assert s.count(old)==1,old;return s.replace(old,new)
scope=read(O/'forward_scope.json');inv=read(O/'geometry_inventory.json')
assert inv['diverse_04']['basic_periodic_surface_screen_pass']
gradient={'status':'proposal_only_not_executed','new_full_AD_jobs':0,
 'research_question':'Does the repaired candidate provide an independently verified local thickness derivative of the entire declared 20% discrete path?',
 'case':'diverse_28 t=.5mm; frozen objective_void+C2, HEX27 N32/27, same XYZ/NH/eta/interface',
 'priority':'After one bounded geometry-transfer forward case; no automatic AD retry or training.',
 'primal_requirement':{'reuse':'r12 candidate center reference only, not the old NH AD path',
  'same_core':'ExplicitXYZ.material_fields, block and observables plus shared material/FEM',
  'accepted_schedule':'same accepted time grid across primal, JVP and perturbations; all hashes checked',
  'center_force_and_curve_relative_limit':1e-6,'controls':'No derivative of step rejection/dt-selection decisions; explicitly conditional on the fixed schedule.'},
 'design_chain':['cached reference distance -> thickness occupancy','occupancy -> stiffness and gate/cutoff',
  'occupancy -> HRZ nodal and periodic mass','mass-dependent affine inertia','every accepted time step and initial/half-step velocity convention',
  'signed reaction at 10%/15%, 20% last-half-hold mean and frozen normalized force loss'],
 'minimum_physical_difference':{'thicknesses_mm':[.4975,.5025],
  'delta_mm':.0025,'fixed_interface_width_mm':.05,
  'conditional_refinement_delta_mm':.00125,
  'refinement_policy':'Only if FD does not resolve a stable local derivative. At most one half-delta pair; do not keep shrinking until agreement.',
  'same_new_material_law_on_both_sides':True,'coarse_range_not_reused_as_local_FD':True},
 'gates':{'finite_primal_and_tangent':True,'actual_positive_J_in_uncontinued_NH':True,
  'complete_candidate_material_domain':'both perturbed paths and final dense probe; not positive-J subset energies',
  'same_mode_screen':'report possible branch change; similarity alone does not certify a derivative',
  'gradient_relative_limit':.01,'normalization_floor_N_per_mm':1e-8,
  'loss_gradient_relative_limit':.01,'loss_normalization_floor_per_mm':1e-8,
  'sign_agreement':'required for resolvable nonzero derivatives',
  'FD_half_delta_relative_limit_if_needed':.01},
 'bounded_execution':'First check the shared new-model primal and a short prefix; then at most one complete new-model directional JVP if those gates pass. No repeated full AD after failure.',
 'cost':'Two new physical paths about 2x10.7min before AD; old NH full JVP 49.77min is a cost precedent only, not a new-model runtime promise; one optional half-delta pair adds about 21min.',
 'pass_scope':'Local thickness derivative of one fixed discrete path; not shape derivative, adaptive-controller derivative, high-dimensional VJP, true-solid accuracy or completed inverse design.',
 'failure_action':'Preserve failure/domain/branch/numerical evidence, keep validated forward scope, identify a targeted mathematical issue before any next attempt.',
 'references':['https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html',
  'validation/large_compression_20261005_r6/gradient20_path_20261005/gradient_validation.json',
  'tests/test_objective_explicit.py']}
write(O/'gradient_gate_proposal.json',gradient)
candidate={'status':'selected_for_proposal_not_simulated','case':'diverse_04',
 'source':inv['diverse_04']['source_directory'],'geometry_screen':inv['diverse_04'],
 'reason':'Existing connected periodic oriented midsurface; different normal distribution and 15914 triangles vs 17986 in baseline. Chosen before new force results, not for a close reference curve.',
 'alternate_inventory_only':'diverse_05 also passes the basic surface screen; not a second queued job.',
 'future_fixed_conditions':'L10mm, t.5mm, XYZ zero lateral strain, NH E10 nu.3, frozen objective_void+C2 and numerical constants',
 'new_inputs_required':['new reference Gauss distances/occupancy and HRZ mass for this midsurface',
  'new matched S3R shell using this midsurface and recomputed XYZ periodic classes; original XY equations/results do not certify XYZ',
  'new input/source/environment provenance; never reuse diverse_28 occupancy or macro-control label blindly'],
 'future_order':['match and screen the physical inputs','one JAX and one slow shell 0-20% + hold','compare existing response, mode, energy/inertia and material-domain gates; report scope or failure'],
 'gates':'Use existing <=10% response/work targets, fixed RMS denominator, sampled material-domain and energy/inertia gates; report shell strict 1% energy quality separately.',
 'claim_limit':'One new periodic implicit-surface case would support transfer only to that geometry; not all TPMS families or a mathematical minimal-surface classification.'}
write(O/'geometry_candidate.json',candidate)
write(O/'decision.json',{'main_plan_steps_completed':[1,2,3,4],'work_kind':'read_only_scientific_synthesis_and_future_gate_selection',
 'new_mechanics_jobs':0,'new_Abaqus_jobs':0,'new_full_AD_jobs':0,'new_training_jobs':0,
 'scoped_forward_research_supported':True,'full_Abaqus_replacement_certified':False,
 'three_thickness_maxima':{k:scope[k] for k in ['maximum_hold_force_difference','maximum_curve_metric','maximum_work_difference']},
 'current_method_retained':'objective_void+C2 explicitly selected; nh default unchanged; no retuning',
 'strict_shell_quality_certified':False,'gradient20_certified':False,
 'next_recommended_case':'diverse_04, t.5mm, matched XYZ elastic no-contact; proposal only',
 'next_priority_order':['one geometry-transfer forward case','bounded new-model complete gradient gate if appropriate','training only after applicable physics/gradient evidence'],
 'reason_to_continue':'Three matched physical thickness samples and objective/domain/rate evidence support the forward route; full gradient-guided inverse-design feasibility remains conditional.'})

# Update the existing comprehensive explanation instead of adding another report.
p=W/'TPMS_RESEARCH_REVIEW.md';s=p.read_text()
s=once(s,'更新：2026-10-06。依据正式仓库5b232e7及其冻结证据。本次是文献复核、进度解释和文件整理，没有新增力学求解、Abaqus作业或完整AD。',
 '更新：2026-10-06，第4步收口。依据正式仓库10d7ce3及r12/r13冻结证据；本次只合并范围、只读筛选已有中面和提出未来梯度关口，零新力学/Abaqus/完整AD作业。r13上轮新增的两条JAX路径与两份匹配壳保留。')
s=once(s,'| 同方法覆盖相邻厚度和其他TPMS形态 | 未证明；下一项只检验少量相邻厚度，跨形态以后另定 |',
 '| 同方法覆盖相邻厚度 | 同一中面0.45/0.50/0.55mm三个采样点通过，最差反力/曲线/功差8.28%/7.48%/7.31%；不逐点认证连续区间 |\n| 其他曲面/构型/本构 | 尚未完成匹配验证；已有diverse_04候选中面，不能继承diverse_28响应或梯度认证 |')
s=once(s,'| 原完整厚度路径AD |',
 '| 匹配厚度范围r13 | 两个新厚度双方匹配至20%，0.50mm复用；曲线、功、模式及材料域工作门槛通过 | 同一构型三个采样点，不保证任意几何或有效局部导数 |\n| 原完整厚度路径AD |')
s=once(s,'约10%是本项目限定范围的工作目标，',
 '固定候选在0.45/0.50/0.55mm快路径上的反力差为6.80%/6.81%/8.28%，曲线指标3.55%/4.93%/7.48%，输入功差5.44%/5.97%/7.31%；每条约10.7分钟、零减步。末态中面波动相对差1.59%/1.82%/2.23%，运动方向余弦都超过0.9997。三份壳能量漂移1.17%/1.33%/1.48%，都未过原严格1%门槛，结论限有条件工程一致性。新厚度只有能量/惯性检查，独立快慢速率证据仅0.50mm；详见[冻结厚度报告](THICKNESS_RANGE_PROGRESS.md)。\n\n约10%是本项目限定范围的工作目标，')
s=once(s,'## 9. 下一步为什么这样安排', '## 9. 四步收口与下一阶段的科研判断')
start=s.index('第3步匹配0.45/0.55mm已完成',s.index('## 9.'));end=s.index('## 10. 文件与阅读口径')
s=s[:start]+'''当前四步已完成：定位→JAX单因素改善→有限真实厚度→范围/未来关口。结果支持继续这条研究路线，但要把“前向能算且工程响应接近”和“完整路径梯度能指导设计”分开。没有证据要求现在换方向，也没有可校准的成功概率或全面替代承诺。

| 当前用途 | 收口判断 |
| --- | --- |
| 同diverse_28中面，三个厚度采样点、0至20%周期弹性无接触整体响应 | 可作为科研前向工具继续使用；在已测条件下满足工作目标，输出完整曲线、功和模式 |
| 预测很接近的设计之间的细微排序 | 未有足够依据；约8%偏差不能直接当所有构型共同倍率，也不能保证区分几个百分点的性能差 |
| 局部应力、尖锐三维实体真值、接触/塑性压溃 | 未认证；不由整体力吻合或正采样J证明 |
| 梯度引导的20%逆设计 | 仍有关键缺口；局部材料/短块AD通过，完整新核路径、高维反向和形态链尚未通过 |

下一阶段优先一个几何迁移工况，保持t=0.50mm及所有既定材料/算法条件。已只读查看MS9下现有diverse_04、diverse_05和基准中面：三者都通过基本周期连接、边邻接和定向检查。建议先选diverse_04；其8372节点、15914三角面比基准9351/17986略小，曲面方向分布也不同，选择依据是现成输入与有区分力的几何，不是先看哪个力最接近。

方向指标是按原三角形面积平均的法向平方分量⟨n_x²,n_y²,n_z²⟩，三分量和为1，只描述曲面朝向，不是刚度。基准约(0.400,0.201,0.400)，diverse_04约(0.365,0.365,0.269)，diverse_05约(0.059,0.457,0.484)。因此候选提供不同几何方向，同时避免先开发新复杂网格。diverse_05只列库存，不排成第二个作业。

这些是用户已有周期隐式曲面的名称。diverse_28原表达式为−3.4cosXcosZ+4.6sinXsinYsinZ+2.0=0，不能直接称其为标准Gyroid或严格数学极小曲面。下一候选也只按实际中面检验，不把一个构型的成功扩成所有TPMS族。

基本中面检查不等于新仿真已经准备好：diverse_04还没有本项目t=0.50mm、匹配NH与XYZ的新壳参照或新Gauss距离/占据/质量场。下一阶段需要双方重新匹配；不能复用diverse_28占据、原XY约束、原厚度/塑性/压板结果。单个迁移工况结束后再决定扩大形态范围，不同时扫描构型与本构。

未来完整梯度最小关口已具体化，但本轮不启动：在diverse_28、t=0.50mm和固定新核上，从共享材料—HRZ质量—仿射惯性—全部接受时间步—反力/损失提取传播一个厚度方向JVP；用0.4975/0.5025mm独立完整路径核对。中央前向及JVP原值必须与同算法、同时间网格一致（旧1e−6门槛），10%/15%取样反力、20%末半段保载均值及原归一化三点损失的导数沿用1%门槛，非零导数符号一致。

厚度变化不是固定状态偏导，质量和动力学也必须一起求导。若差分尺度不能解析局部导数，最多补一次半步长±0.00125mm；不能不断减小到“碰巧吻合”。必须核对两侧材料求值域与分支，负J跨入未续接NH区不能称可用设计邻域。固定接受网格核对仅认证该离散路径，不包含拒绝/减步控制决策导数，不认证形态或高维反向。

先做新核共享前向和短前缀原值一致性，再至多一次完整方向JVP；任何入口/材料域/前缀失败就停在其证据上，不无限延长Newton或重复全AD。当前接口material_fields、block、observables和短块核对可复用，但旧NH结果不能当新路径结果。[JAX官方JVP与独立差分核对依据](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)支持这种验证思路，不替本项目完成有效梯度认证。

成本须单独核算：两条厚度差分前向约2×10.7分钟，旧完整JVP约49.77分钟只是成本先例，新核实际成本未测；条件性半步长对另约21分钟，高维反向未测。这里没有100秒限制，也不假定每个验证都必须15分钟以内。若新核导数仍失败，保留可用前向范围，判断是局部可微域、路径敏感性或实现问题，再提出一次针对性方案。

研究顺序仍仿真优先：一个几何迁移工况→按必要性进入限定梯度关口→之后才讨论训练/生成。材料可变但后置；仅改变E可能主要缩放力级，ν或本构改变需要重新匹配实体规律并核查虚域处理。执行输入、指标与停止条件只放[唯一主规划](RESEARCH_PLAN.md)，本文不维护另一套作业清单。收口证据在`validation/forward_scope_20261006_r14`，零新力学求解。

''' + s[end:]
p.write_text(s)

# Keep the completed four-step structure. Only a bounded future proposal follows.
p=W/'RESEARCH_PLAN.md';s=p.read_text();s=once(s,
 '当前第1/2/3步完成；第4步待收口，本轮新增2条JAX路径与2个匹配Abaqus壳作业，无完整AD/训练。',
 '当前四步均完成；第4步为证据收口与后续关口选择，零新力学/Abaqus/完整AD作业。r13两条新路径与两份壳是上轮完成证据，不重复。')
s=once(s,'## 4. 收口前向范围与未来梯度关口：当前唯一下一项',
 '## 4. 收口前向范围与未来梯度关口：已完成')
i=s.index('按第3步实测结论收口');j=s.index('## 留存与执行')
s=s[:i]+'''收口证据：`validation/forward_scope_20261006_r14`；解释集中在[综合说明第9节](TPMS_RESEARCH_REVIEW.md#9-四步收口与下一阶段的科研判断)。三厚度最差反力/曲线/功差8.28%/7.48%/7.31%，工作门槛通过；壳质量未严格认证，完整20%新核梯度未认证。保留原失败、实际虚域翻转、有限采样和无接触范围，不作全面Abaqus替代承诺。

### 下一阶段建议：一个几何变化，尚未提交作业

选择用户已有MS9/diverse_04中面（8372节点/15914三角面），基本周期连接/定向/接缝通过且方向分布不同。diverse_05只为库存，不自动加入。仍固定L10mm、t0.50mm、原NH/XYZ/候选常数/HEX27 N32/27点/HRZ；不改变本构、厚度或数值常数拟合原曲线。

后续推进时仅围绕这个工况拟定短期执行，不扩成广泛扫描：

1. 匹配输入：新中面真实Gauss距离/占据与质量；同中面0.50mm S3R壳、XYZ周期归并、NH和宏观加载。原XY/压板/塑性模型仅借中面，不继承响应认证。
2. 一次双方前向：JAX既定0.004s及慢壳0.040s，20%加原保载；保留真实稳定步长、材料域和物理保护。若实际惯性或失效要求，再提出一次针对性处理，不重开扫描。
3. 原门槛比较与范围结论：反力/固定分母RMS/功≤10%，峰值/模式另报，27/125点完整候选材料域、能量/惯性及壳原严格1%质量分别评价；失败原样保留，不从一构型外推任意形态。

这三项是下一阶段的有限建议，本轮未启动。输入预处理不复用diverse_28的占据或盲套其节点标签。依据见r14 geometry_candidate.json；基本中面检查不认证0.50mm实体带、无接触或力学精度。

### 未来梯度独立关口：方案已固定，本轮不执行

详见r14 gradient_gate_proposal.json。只考虑固定新核在diverse_28、t0.50mm的一个完整厚度方向，必须包括占据/门控、HRZ质量、仿射惯性、每个接受时间步和目标提取；原10%/15%NH认证不能继承。采用同一接受时间网格和共享入口，中央原值/曲线一致1e−6；±0.0025mm完整路径独立差分，10%/15%反力、20%保载均值及原三点归一化损失的AD/FD沿用1%，非零符号一致。

先过共享原值和短前缀，至多一次完整新核方向JVP；差分需解析性复核时最多追加±0.00125mm一对，不无限重试/减小步长或降门槛。两侧完整材料域、可能分支变化及同时间网格均要核对，不把固定状态偏导、粗厚度范围或错误末态伴随称有效AD。固定接受路径不含拒绝/减步决策导数。

成功也仅认证该局部厚度离散路径；形态/最近三角形、高维反向和训练另定。失败则保存范围结论与局部数学/实现问题，前向仍可继续；若最终目标必须20%有效梯度，再据证据决定调整方法或适用范围，不宣称必定成功。成本：独立两条前向约21分钟，旧完整JVP49.77分钟仅为先例，新核成本未测；不以预算停止冒充方法失败。

''' + s[j:]
p.write_text(s)

p=W/'PROJECT_OVERVIEW.md';s=p.read_text().replace('2026-10-06匹配厚度轮次完成；正式仓库基线dd66775。',
 '2026-10-06四步收口完成；正式仓库基线10d7ce3。本次零新力学/Abaqus/完整AD，复用r12/r13。')
s=once(s,'| 第4步收口 | 当前唯一下一项；依据实测范围选择有区分力的后续候选，无自动扫描/复杂模型 |',
 '| 第4步收口 | 已完成；明确前向范围、参考质量、梯度关口；只读选diverse_04为下一阶段候选，无新作业 |')
s=s.replace('下一候选优先一个几何变化，需要匹配中面/真实厚度和参照。',
 '已建议diverse_04的单一几何变化（基本周期中面通过，新占据/XYZ匹配参照待建立）；不启动其他构型/材料扫描。')
s+='\n本次收口证据：`validation/forward_scope_20261006_r14`，含范围矩阵、原中面只读筛选、候选与梯度方案。原程序、r6-r13科学字节不改；综合说明已修正仍把相邻厚度写成未验证的旧表述。\n'
p.write_text(s)
p=W/'RESEARCH_BACKGROUND.md';s=p.read_text();s=once(s,
 '当前只看主规划第4步的范围收口与后续关口选择；曲面/构型/本构是未来候选，不自动全面扫描，复杂独立三维参照后置。',
 '当前四步已收口；下一阶段优先已有diverse_04的一个匹配几何工况，尚未开始新前向。完整20%新核梯度方案已提出但未执行；曲面/构型/本构不自动全面扫描，复杂独立三维参照后置。')
p.write_text(s)
p=W/'START_HERE.md';s=p.read_text();s=s.replace('当前只看第4步收口。','第4步已收口，下一阶段只建议diverse_04的一个匹配几何工况，尚未提交新作业。');p.write_text(s)
p=W/'AGENTS.md';s=p.read_text();s=once(s,'前三步已执行；r13匹配同中面0.45/0.55mm壳/JAX完整20%，0.50mm复用，结果见THICKNESS_RANGE_PROGRESS.md。下一项仅第4步收口范围与未来关口；',
 '本轮四步已完成；r13匹配厚度、r14只读范围/未来关口收口，科学原件冻结。下一阶段优先diverse_04、t0.50mm的单一几何匹配前向，尚未建立新Gauss占据/XYZ参照或提交作业；未来新核完整梯度方案已提出但不自动启动。')
p.write_text(s)
p=W/'FILE_MAP.md';s=p.read_text().replace('唯一四步及当前第4步','四步已完成；唯一有限后续建议')
s=s.replace('| validation/thickness_range_20261006_r13/ |','| validation/forward_scope_20261006_r14/ | 只读四步收口：范围矩阵、三个原中面库存、diverse_04候选、未来完整梯度方案；零新力学 |\n| validation/thickness_range_20261006_r13/ |')
s+='\n第4步收口仅更新现有综合说明/唯一规划，不新增重复长报告或PDF。WSL r14下`forward_scope.json`为范围，`geometry_inventory.json`为只读原中面筛选，`geometry_candidate.json`为单一diverse_04候选，`gradient_gate_proposal.json`仅未来方案，`decision.json`/输入与验证收据为依据。Windows `work/forward_scope_20261006`为本次一次性收口/发布工具，非活动FEM。用户原中面位置为`F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_04/abaqus/ingredients/shell_mesh.inc`，未改。\n'
p.write_text(s)

mapping={'RESEARCH_BACKGROUND.md':'docs/RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'docs/RESEARCH_PLAN.md',
 'PROJECT_OVERVIEW.md':'docs/RESEARCH_STATUS.md','FILE_MAP.md':'docs/FILE_MAP.md','TPMS_RESEARCH_REVIEW.md':'docs/TPMS_RESEARCH_REVIEW.md',
 'START_HERE.md':'README.md','AGENTS.md':'AGENTS.md'}
for src,dest in mapping.items():
    s=(W/src).read_text().replace('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md').replace('output/figures/','figures/')
    if dest=='docs/FILE_MAP.md':s=s.replace('[归档索引](history/completed_tools_20261006/README.md)',
                                          '`Windows history/completed_tools_20261006/README.md`（Windows归档索引）')
    if dest=='docs/TPMS_RESEARCH_REVIEW.md':s=s.replace('figures/VOID_CONTINUATION_response.png','../validation/void_continuation_20261006_r12/response.png')
    if dest in ['AGENTS.md','README.md']:
        for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','THICKNESS_RANGE_PROGRESS.md','VOID_CONTINUATION_PROGRESS.md','PAPER_ROUTE.md']:
            s=s.replace(']('+name+')','](docs/'+name+')')
    (R/dest).write_text(s)
p=R/'validation/README.md';s=p.read_text();s=s.replace('| [thickness_range_20261006_r13]',
 '| [forward_scope_20261006_r14](forward_scope_20261006_r14/README.md) | 第4步只读收口、原中面筛选、diverse_04单候选及未来新核梯度方案；零新力学/Abaqus/AD |\n| [thickness_range_20261006_r13]')
s=s.replace('当前第1/2/3步已执行，第4步收口范围/未来关口','当前四步已收口，唯一规划中的下一阶段建议为diverse_04单一几何迁移')
s=s.replace('本次新增两条JAX前向及两个匹配壳Explicit，无完整AD/训练；详见r13。',
 'r13的两条JAX/两份匹配壳已冻结；本次r14仅只读收口和方案，无新力学/完整AD/训练。')
p.write_text(s)
p=R/'scripts/README.md';s=p.read_text().replace('下一项仅[主规划](../docs/RESEARCH_PLAN.md)第4步范围收口',
 '四步已收口，下一阶段仅看[主规划](../docs/RESEARCH_PLAN.md)的单一几何候选建议')
p.write_text(s)
(O/'README.md').write_text('''# 四步范围收口：零新求解

复用r12/r13，范围解释集中在[综合说明第9节](../../docs/TPMS_RESEARCH_REVIEW.md#9-四步收口与下一阶段的科研判断)，唯一执行建议看[主规划](../../docs/RESEARCH_PLAN.md)。

forward_scope.json为三个采样厚度/参照质量/前向成本；geometry_inventory.json只读用户现有三个中面，generic parser复用，未运行硬编码diverse_28的旧audit；geometry_candidate.json仅建议diverse_04（未新建Gauss场/XYZ参照/提交求解）；gradient_gate_proposal.json仅未来完整路径核对，不是认证或已运行AD；decision.json为收口判断。input_manifest/verification/evidence_manifest保留哈希，旧科学原件不改。

tools是一次性只读分析/同步/发布记录，不是新FEM或另一套实验框架。本轮零力学/Abaqus/全AD/训练，默认NH和候选常数不改，没有重复PDF/长报告。
''')
tools=O/'tools';tools.mkdir()
for name in ['analyze_scope.py','finish.py']:shutil.copy2(W/'work/forward_scope_20261006'/name,tools/name)
print('Current four steps closed; bounded geometry and gradient proposals only. No new mechanics.')
