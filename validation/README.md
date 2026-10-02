# 验证证据索引

完整研究路线见 [研究主规划](../docs/RESEARCH_PLAN.md)，当前证据见 [研究状态](../docs/RESEARCH_STATUS.md)，运行位置见 [文件地图](../docs/FILE_MAP.md)。验证是长期 TPMS 学习与逆向设计主线的第一阶段。

以下子目录是冻结实验记录。阶段 README 中的测试数和“下一步”仅代表当时；计算数据、原始日志、验收和指纹保留。当前工作从主规划继续。

| 目录 | 已完成范围 |
| --- | --- |
| [m4_review](m4_review/README.md) | M4-A 修复及 10 项固定横向参数数据 |
| [abaqus_uniform](abaqus_uniform/README.md) | 3 项原生 C3D8 解析基准 |
| [abaqus_element](abaqus_element/README.md) | 2 项 C3D8 单元矩阵诊断 |
| [abaqus_discrete](abaqus_discrete/README.md) | 3 项独立积分线性用户单元矩阵对照 |
| [abaqus_binary](abaqus_binary/README.md) | 13 项 C3D10，含 2 项均匀基准 |
| [abaqus_mesh_quality](abaqus_mesh_quality/README.md) | 2 项新作业及 6 个旧 ODB 诊断 |
| [projection_grid_20261002](projection_grid_20261002/README.md) | N48/N64 背景响应与求解路径核对 |
| [projection_effects_20261002](projection_effects_20261002/README.md) | 7 项 β/E_min/体积分数影响计算 |
| [design_gradient_20261002](design_gradient_20261002/README.md) | 3 组 N4/N8 固定横向参数梯度，132 次差分扰动 |

正式成功 Abaqus 作业共 23 项（含 2 项矩阵诊断），最新实际完整回归 125/125；文档整理未重跑。完整路径及哈希见 [23 项索引](research_audit_20261002.json)，较早的 [21 项快照](project_status_20261002_inventory.json) 保留。

大型 INP/ODB 在 E 盘。旧长报告可从 Git `6295ee7` 读取；历史 R/G 草案与 [范围纠正记录](scope_correction_20261003.json) 只供追溯。研究完成还需要多构型接口、训练、逆向设计和独立设计复核，不能以这些验证记录替代。
