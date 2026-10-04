# 验证证据索引

方法、符号和结果解释见 [综合报告](../docs/TPMS_RESEARCH_REVIEW.md)，当前事实见 [状态](../docs/RESEARCH_STATUS.md)，执行只看 [主规划](../docs/RESEARCH_PLAN.md)。三轮均已按判据收口；子目录保留冻结记录，其中历史“下一步”不是自动待办。

第四轮第1步在新目录审计旧输入/网格/响应，未新增有限元求解；前三轮实验目录、原始输入/结果/日志或旧清单保持冻结。旧manifest文档指纹指当时版本，对应历史快照保留；不能改写旧清单匹配活动文档。

| 目录 | 已完成范围 |
| --- | --- |
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

历史截至2026-10-02的Abaqus分析／矩阵诊断共23项；加第二轮4、第三轮1，加第四轮第2步2项，当前总数30，datacheck另计。原23项索引及第一轮127项回归是历史快照。完整路径及哈希见 [23 项索引](research_audit_20261002.json)，较早的 [21 项快照](project_status_20261002_inventory.json) 保留。

大型 INP/ODB 在 E 盘。旧长报告可从 Git `6295ee7` 读取；历史 R/G 草案与 [范围纠正记录](scope_correction_20261003.json) 只供追溯。这些验证记录不能替代训练与逆向设计证据；具体后续只能由新的近期主规划决定。
