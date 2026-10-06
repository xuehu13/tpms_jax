from pathlib import Path
import shutil,json
W=Path("/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an")
R=Path("/home/xuehu/projects/tpms_jax")
D=R/"validation/geometry_transfer_review_20261006_r16"
report=(W/"work/geometry_closure_20261006/report_source.md").read_text()
(D/"REVIEW.md").write_text(report)
reportW=report.replace("(../geometry_transfer_20261006_r15/STEP2.md)","(https://github.com/xuehu13/tpms_jax/blob/main/validation/geometry_transfer_20261006_r15/STEP2.md)")
for name in ("rejected_endpoint_diagnosis.json","partial_force_diagnosis.json","shell_path_diagnosis.json","scope_decision.json","verification.json"):
    reportW=reportW.replace("]("+name+")","](https://github.com/xuehu13/tpms_jax/blob/main/validation/geometry_transfer_review_20261006_r16/"+name+")")
reportW=reportW.replace("(existing_record_diagnosis.png)","(work/geometry_closure_20261006/existing_record_diagnosis.png)")
(W/"GEOMETRY_TRANSFER_REVIEW.md").write_text(reportW)
def exact(t,a,b):
    assert t.count(a)==1,(a,t.count(a))
    return t.replace(a,b)
link="[本轮诊断与收口](GEOMETRY_TRANSFER_REVIEW.md)"
x=W/"RESEARCH_PLAN.md";t=x.read_text()
t=exact(t,"第3/4步待执行，下一项仅第3步现有记录限定诊断，零新增作业/完整AD/训练。",
"第3/4步已完成现有记录限定诊断与范围收口（r16），第2步目标仍未通过。本轮零新增作业/求解器修复/完整AD/训练；问题处理留下一轮更新规划后决定。")
t=t.replace("## 3. 对照并定位实际差异：待执行（限定现有证据）","## 3. 对照并定位实际差异：已完成（限定现有证据）")
t=t.replace("## 4. 给出迁移范围与继续条件：待执行","## 4. 给出迁移范围与继续条件：已完成")
t=exact(t,"下一项仅利用现有记录作限定诊断；需要新修复或作业时先更新唯一规划，零自动重跑/AD/训练。",
"r16已完成现有记录限定诊断与范围收口；第2步完整目标未过，问题处理留下一轮，零自动重跑/AD/训练。")
t=exact(t,"下一次推进仅第3步限定诊断；当前没有可信的双方20%对照或新构型梯度认证。若需要新增修复/计算，先以实际问题更新本规划。",
"本轮执行已收口，后两步为限定诊断/范围判断完成，不等于第2步验证通过。当前没有可信的diverse_04双方20%对照或新构型梯度认证；无自动待执行作业。问题处理留下一轮针对实际证据更新本规划后决定。")
marker="## 4. 给出迁移范围与继续条件：已完成"
t=t.replace(marker,"第3步结论：转折前已存约2%～10%点差6.5%～7%；壳约11.416%反力降载，JAX13.765%仍增载，双方差异早于15.966%首次拒绝。第二拒绝末端的有限单元子集溢出集中深虚域，不能定位首次触发；第一/三拒绝场几乎全NaN。壳转折后惯性/黏性/人工能量突出，不能当严格准静态真值。完整20%误差/模式指标不计算，详见"+link+"。\n\n"+marker)
marker="## 已完成依据与执行规则"
t=t.replace(marker,"第4步结论：diverse_28三个厚度范围保持，diverse_04迁移不通过，完整新核20%梯度未认证。后续仅建议保全有效接受状态并定位首次异常、分别处理响应转折与参照质量，再决定匹配完整前向；不在本轮启动修复、速率、新构型或AD。用户已要求先完成后两步再处理问题，此四步到此收口。\n\n"+marker)
x.write_text(t)
x=W/"PROJECT_OVERVIEW.md";t=x.read_text()
t=t.replace("下一项仅[唯一主规划](RESEARCH_PLAN.md)第3步现有证据限定诊断；零追加作业/完整AD/训练。",
"第3/4步限定诊断与范围收口已完成（r16），第2步验证仍未通过；问题处理留下一轮，零追加作业/求解器修复/完整AD/训练。")
t=t.replace("| 新第3/4步 | 待执行：现有稀疏接受/拒绝场及完整壳诊断→迁移范围/继续条件；不自动重跑 |",
"| 新第3/4步 | 已完成r16：现有记录限定诊断与范围收口；不等于第2步通过，无自动待执行作业 |")
t=t.replace("不同几何前向尝试显示当前迁移限制，第3步待定位问题；","不同几何前向尝试及限定诊断显示当前迁移限制；")
t=t.replace("下一项仅利用现有记录作限定诊断；需要新修复或作业时先更新唯一规划，零自动重跑/AD/训练。",
"r16已完成后两步收口；后续需要针对实际问题更新唯一规划，零自动修复/重跑/AD/训练。")
t+="\n本轮新增诊断：约2%～10%的几个接受观察点力差6.5%～7%；壳约11.416%开始降载，JAX13.765%仍增载，差异早于首次拒绝。第二拒绝末端的有限节点单元子集出现深虚域溢出，未定位首次触发；整体末态/完整曲线仍缺失。只保留diverse_28原范围，不认证diverse_04迁移。详细证据、数字含义及继续条件见"+link+"；没有新前向、材料/算法修改或AD。\n"
x.write_text(t)
x=W/"RESEARCH_BACKGROUND.md";t=x.read_text().replace("下一项仅利用现有记录诊断，不改写旧范围或自动追加作业。",
"第3/4步现有记录限定诊断与范围收口已完成，转折差异和数值失稳分别待处理；不改写旧范围或自动追加作业。")
x.write_text(t)
x=W/"START_HERE.md";t=x.read_text().replace("下一项仅第3步现有记录限定诊断，不自动追加作业。",
"第3/4步限定诊断与范围收口已完成（r16），问题处理留下一轮决定，不自动修复或追加作业。")
t+="\n本轮主要发现与范围见"+link+"：双方响应转折差异早于JAX数值拒绝；未完成的20%验证不算通过。\n"
x.write_text(t)
x=W/"FILE_MAP.md";t=x.read_text()
t=t.replace("第1步完成，第2步目标未通过，第3步限定诊断待执行","第1步完成、第2步目标未过、第3/4步限定诊断与范围收口完成")
t=t.replace("| PAPER_ROUTE.md |","| GEOMETRY_TRANSFER_REVIEW.md | r16现有记录第3/4步诊断与范围；validation/geometry_transfer_review_20261006_r16/REVIEW.md |\n| PAPER_ROUTE.md |")
t=t.replace("| validation/geometry_transfer_20261006_r15/ |",
"| validation/geometry_transfer_review_20261006_r16/ | 新第3/4步只读后处理与收口；REVIEW.md、三份诊断、scope_decision、图与冻结哈希；没有新仿真或梯度 |\n| validation/geometry_transfer_20261006_r15/ |")
t+="\n后两步收口：r16的existing_record_review.py只读已有场/日志，复用共享材料核求值，无内力组装/时间推进；不是新求解器。rejected_endpoint_diagnosis只诊断拒绝子集，partial_force_diagnosis只有14个真实JAX观察对壳插值；不输出完整20%误差/模式。frozen_before.json保全旧证据，verification/evidence_manifest验证追溯；postprocess_attempt01保留Inf序列化失败，不是新仿真。Windows work/geometry_closure_20261006仅本次后处理/说明/发布工具；GEOMETRY_TRANSFER_REVIEW.md镜像正式REVIEW.md，图在该work目录。\n"
x.write_text(t)
x=W/"AGENTS.md";t=x.read_text()
t=exact(t,"下一项仅第3步现有记录限定诊断。",
"r16第3/4步现有记录限定诊断与范围收口已完成，第2步目标仍未通过，本轮无自动待执行作业。双方转折差异早于数值拒绝：壳约11.416%降载，JAX13.765%仍增载；第二拒绝末端有限单元子集溢出在深虚域，未定位首次触发，不排除含NaN单元NH越界。后续须针对实际问题更新唯一主规划再处理，不自动修复/新作业/AD。")
x.write_text(t)
x=W/"TPMS_RESEARCH_REVIEW.md";t=x.read_text()
t=t.replace("第3步限定诊断待执行，零完整AD。","r16第3/4步限定诊断与范围收口完成，第2步目标仍未通过，零完整AD。")
t=t.replace("| 其他曲面/构型/本构 | 尚未完成匹配验证；已有diverse_04候选中面，不能继承diverse_28响应或梯度认证 |",
"| 其他曲面/构型/本构 | diverse_04匹配输入后20%尝试未通过，已限定诊断收口；不能继承diverse_28响应或梯度认证 |")
t=t.replace("第1步输入完成，第2步目标未通过，第3步现有证据限定诊断待执行。",
"第1步输入完成，第2步目标未通过，第3/4步现有记录限定诊断与范围收口已完成；本轮无自动修复或待执行作业。")
t=t.replace("下一项仅利用现有记录作限定诊断；需要新修复或作业时先更新唯一规划，零自动重跑/AD/训练。",
"后两步已限定诊断与范围收口；问题处理留下一轮更新唯一规划后决定，零自动修复/重跑/AD/训练。")
t+="\n## 11. 新几何迁移的限定收口（r16）\n\n"+link+"给出原稀疏观察、拒绝场与完整壳历程的分析。重要区别是：转折前几个约2%～10%接受点力差6.5%～7%；壳约11.416%明显降载，JAX13.765%仍增载，差异早于15.966%附近非有限拒绝。第二拒绝末端的有限单元子集溢出集中深虚域，不能还原首个异常，也不能排除未可计算单元的NH越界。壳转折后惯性、人工能量和黏性耗散突出，未过质量门槛。\n\n第3/4步完成是诊断与范围收口完成，第2步可信20%匹配目标仍未过；不能给完整20%误差/末态模式，不能把解决数值失稳自动等同响应吻合。diverse_28三个厚度的有条件前向范围保持，跨几何和新核完整20%梯度仍有缺口。后续提出记录保全/首次异常、响应转折/参照质量等针对性继续条件，但本轮不修复、不追加作业或AD。\n"
x.write_text(t)
x=W/"PAPER_ROUTE.md";t=x.read_text()
t+="\n## r16限定诊断新增公开方法依据\n\n2026-10-06成功读取公开2025文档正文（实际软件2026，不认证所有版本差异）：[显式动力学理论](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm)的中央差分、稳定性、当前有效模量/频率与体积黏性；[能量平衡说明](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-ovwstatenerbal.htm)的内能包含人工能量、黏性耗散定义及总能量平衡。仅作为r16局部稳定性假设和质量指标解释，不证明本项目首次异常原因，不改变原验收门槛或要求立即细化网格。网页本次成功不改写前轮读取失败记录。\n"
x.write_text(t)
mapping={"RESEARCH_BACKGROUND.md":"docs/RESEARCH_BACKGROUND.md","RESEARCH_PLAN.md":"docs/RESEARCH_PLAN.md",
"PROJECT_OVERVIEW.md":"docs/RESEARCH_STATUS.md","FILE_MAP.md":"docs/FILE_MAP.md","TPMS_RESEARCH_REVIEW.md":"docs/TPMS_RESEARCH_REVIEW.md",
"START_HERE.md":"README.md","AGENTS.md":"AGENTS.md","PAPER_ROUTE.md":"docs/PAPER_ROUTE.md"}
for src,dest in mapping.items():
    t=(W/src).read_text().replace("PROJECT_OVERVIEW.md","RESEARCH_STATUS.md")
    if dest=="docs/FILE_MAP.md":t=t.replace("[归档索引](history/completed_tools_20261006/README.md)","Windows history/completed_tools_20261006/README.md（Windows归档索引）")
    if dest=="docs/TPMS_RESEARCH_REVIEW.md":t=t.replace("output/figures/VOID_CONTINUATION_response.png","../validation/void_continuation_20261006_r12/response.png")
    t=t.replace("](GEOMETRY_TRANSFER_REVIEW.md)","](../validation/geometry_transfer_review_20261006_r16/REVIEW.md)" if dest.startswith("docs/") else "](validation/geometry_transfer_review_20261006_r16/REVIEW.md)")
    if dest in ("AGENTS.md","README.md"):
        for name in ("RESEARCH_BACKGROUND.md","RESEARCH_PLAN.md","RESEARCH_STATUS.md","FILE_MAP.md","TPMS_RESEARCH_REVIEW.md","THICKNESS_RANGE_PROGRESS.md","VOID_CONTINUATION_PROGRESS.md","PAPER_ROUTE.md"):
            t=t.replace("]("+name+")","](docs/"+name+")")
    (R/dest).write_text(t)
for rel in ("scripts/README.md","validation/README.md"):
    x=R/rel;t=x.read_text()
    t=t.replace("下一项仅[主规划](../docs/RESEARCH_PLAN.md)第3步现有证据限定诊断","第3/4步现有记录限定诊断与范围收口已完成，后续仅在[主规划](../docs/RESEARCH_PLAN.md)更新后处理问题")
    t=t.replace("下一项仅[主规划](../docs/RESEARCH_PLAN.md)第3步现有记录诊断","第3/4步现有记录限定诊断与范围收口已完成，后续仅在[主规划](../docs/RESEARCH_PLAN.md)更新后处理问题")
    if rel=="validation/README.md":t+="\nr16：[现有记录诊断与范围收口](geometry_transfer_review_20261006_r16/REVIEW.md)，未使r15第2步验证通过；没有新前向/AD。\n"
    x.write_text(t)
shutil.copyfile(__file__,D/"update_current_docs.py")
print("Updated current reading entry points; original scientific records unchanged.")
