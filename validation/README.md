# 验证证据索引

方法、符号和结果解释见 [综合报告](../docs/TPMS_RESEARCH_REVIEW.md)，当前事实见 [状态](../docs/RESEARCH_STATUS.md)，执行只看 [主规划](../docs/RESEARCH_PLAN.md)。原四轮均已按判据/停止条件收口；子目录保留冻结记录，其中历史“下一步”不是自动待办。

已完成四步面向目标薄壁：恒厚表示、小变形壳对照、分段压缩、可信区间有效梯度。最新用户目标t/L=0.05，第1步表示完成；第2步原XY平端未过、唯一XYZ周期诊断通过；用户已统一XYZ；第3步1%/5%通过、10%未接受，第4步1%/5%厚度总梯度与两点损失通过，四步限定收口。旧中等厚度成本/N48/N64/半步计划已替换，不能从历史报告恢复待办。前三轮及第四轮实验的原始输入/结果/日志保持冻结；旧manifest记录当时版本，不改写匹配活动文档。

Git保存维护源码、关键JSON/CSV摘要、报告及必要图；source_before/source_at_run等源码快照、大型数组、重复停止尝试的INP和求解器原始日志留在本机原位、由.gitignore排除。克隆Git不等于下载完整实验档案；复核具体旧运行须按FILE_MAP取本机冻结证据/环境及原Abaqus包。已被Git跟踪的早期证据继续保留。

| 目录 | 已完成范围 |
| --- | --- |
| [thin_target_20261004_r5/step4_thickness_a01](thin_target_20261004_r5/step4_thickness_a01/README.md) | 1%/5%反力厚度总导数及两点损失；2伴随/6扰动/3停止，2项1%壳厚分析；157维护；5%独立壳导数未算，无训练/形态导数 |
| [thin_target_20261004_r5/step3_xyz](thin_target_20261004_r5/step3_xyz/README.md) | 5%薄壁XYZ有限应变；1%/5%初筛，10%未接受；3完成状态/4停止、4壳分析，155维护；无20%/梯度/训练 |
| [thin_target_20261004_r5/step2](thin_target_20261004_r5/step2/README.md) | 5%薄壁小变形；原平端未过，XYZ诊断差4.979%/模式一致初筛通过；2次JAX/2次壳；原记录冻结，用户后续已统一XYZ |
| [thin_target_20261004_r5](thin_target_20261004_r5/README.md) | 第1步恒厚距离带/真实Gauss/连接初筛，原记录冻结 |
| [mechanics_trust_20261003_r4/step4](mechanics_trust_20261003_r4/step4/README.md) | Gyroid预检查至5%，N48在0.5%点按资源停止；3条路径尝试/26状态、零新增Abaqus/训练/设计梯度，148项回归；有限应变精度未认证 |
| [mechanics_trust_20261003_r4/step3](mechanics_trust_20261003_r4/step3/README.md) | 第四轮完整均匀实体有限应变基准通过至20%；3条JAX路径、2项分析及2项datacheck，144项回归通过；不认证TPMS大压缩 |
| [mechanics_trust_20261003_r4/step2](mechanics_trust_20261003_r4/step2/README.md) | 第四轮新构型/薄壁筛查：Primitive整体响应通过，薄壁细参考未成立；2前向、2分析、2datacheck；最终138项回归通过 |
| [mechanics_trust_20261003_r4/step1](mechanics_trust_20261003_r4/step1/README.md) | 第四轮参考审计：14个旧包、实际INP/拓扑/映射检查通过；零新增求解；原始记录冻结 |
| [learning_bridge_20261003_r3](learning_bridge_20261003_r3/README.md) | N64专用梯度通过；新壁宽示例性能跨度不足／G48网格审查失败，第3/4步未启动 |
| [geometry_interface_20261003_r2](geometry_interface_20261003_r2/README.md) | 隐式输入三锚点与小／中网格梯度，通过与资源停止证据冻结 |
| [near_term_20261003](near_term_20261003/README.md) | 第一轮四步：解析前向通过；数组保真未达标并停止；全链导数仅作诊断通过 |
| [m4_review](m4_review/README.md) | M4-A 修复及 10 项固定横向参数数据 |
| [abaqus_uniform](abaqus_uniform/README.md) | 3 项原生 C3D8 解析基准 |
| [abaqus_element](abaqus_element/README.md) | 2 项 C3D8 单元矩阵诊断 |
| [abaqus_discrete](abaqus_discrete/README.md) | 3 项独立积分线性用户单元矩阵对照 |
| [abaqus_binary](abaqus_binary/README.md) | 13 项 C3D10，含 2 项均匀基准 |
| [abaqus_mesh_quality](abaqus_mesh_quality/README.md) | 2 项新作业及 6 个旧 ODB 诊断 |
| [projection_grid_20261002](projection_grid_20261002/README.md) | N48/N64 背景响应与求解路径核对 |
| [projection_effects_20261002](projection_effects_20261002/README.md) | 7 项 β/E_min/体积分数影响计算 |
| [design_gradient_20261002](design_gradient_20261002/README.md) | 3 组 N4/N8 固定横向参数梯度，132 次差分扰动 |

历史截至2026-10-02的Abaqus分析／矩阵诊断共23项；加第二轮4、第三轮1，加第四轮第2步2项，加第3步2项，截至第四轮合计32；本轮薄壁第2步另有2项、第3步另有4项、第4步另有2项，总计40项，datacheck另计。原23项索引及第一轮127项回归是历史快照。完整路径及哈希见 [23 项索引](research_audit_20261002.json)，较早的 [21 项快照](project_status_20261002_inventory.json) 保留。

大型 INP/ODB 在 E 盘。旧长报告可从 Git `6295ee7` 读取；历史 R/G 草案与 [范围纠正记录](scope_correction_20261003.json) 只供追溯。这些验证记录不能替代训练与逆向设计证据；具体后续只能由新的近期主规划决定。

## 2026-10-05状态更新

目标薄壁四步按限定范围完成；1%/5%前向及厚度梯度通过，10%差14.19%/壳质量未过，20%未提交。上述为冻结历史。唯一下一轮是docs/RESEARCH_PLAN的大压缩四步，均未启动；本次仅调查/整理，无新求解或维护测试。当前TPMS仍静力弹性，无物理Explicit、接触或塑性；官方示例不作本项目认证。新实验届时新目录，不修改旧输入/结果。
