# 验证证据索引

当前 TPMS 体素/学习/生成主线及 G1–G5 路线只看 [研究状态](../docs/RESEARCH_STATUS.md)；路径/脚本用途见 [文件地图](../docs/FILE_MAP.md)。

这里各子目录是当时实验的冻结证据。阶段 README 中的测试数量和“下一步”是历史记录；不要据此安排当前工作。计算数据、日志、验收和指纹不因本次整理而更改。

| 目录 | 层次 | 已完成范围 |
| --- | --- | --- |
| [m4_review](m4_review/README.md) | M4-A 审查修复 | 10 项固定横向参数数据；宏观功、Gauss 点、失败状态 |
| [abaqus_uniform](abaqus_uniform/README.md) | 均匀实体 | 3 项原生 C3D8 解析基准 |
| [abaqus_element](abaqus_element/README.md) | 单元算子诊断 | 2 项 C3D8 刚度矩阵诊断；识别与 JAX HEX8 算子差异 |
| [abaqus_discrete](abaqus_discrete/README.md) | 同离散方程 | 3 项独立积分线性用户单元矩阵对照 |
| [abaqus_binary](abaqus_binary/README.md) | 真正二值实体 | 13 项 C3D10（含 2 项均匀基准），整体响应加密筛查 |
| [abaqus_mesh_quality](abaqus_mesh_quality/README.md) | 同几何体网格质量 | 2 项新作业 + 6 个旧 ODB 诊断；累计 Abaqus 23 项 |
| [projection_grid_20261002](projection_grid_20261002/README.md) | 投影网格 | N48/N64 固定/自由响应；PETSc 路径核对 |
| [projection_effects_20261002](projection_effects_20261002/README.md) | 投影参数影响 | 7 项固定横向新增计算；β/E_min/体积分数 |
| [design_gradient_20261002](design_gradient_20261002/README.md) | 固定横向梯度 | 3 组 N4/N8、132 次差分扰动；最新完整回归 125/125 |

累计正式成功 Abaqus 作业为 23；最新一次实际全套回归为 125/125，非本次重跑。当前 [23 项索引](research_audit_20261002.json) 补齐原 [21 项历史快照](project_status_20261002_inventory.json) 之后的两项同几何体网格作业。datacheck/失败尝试/重复提取不计入。

根目录旧的长报告与进度文档现在保留为导航入口，其整理前版本可在 Git 提交 `6295ee7` 读取。原始大型 ODB/INP 留在本机 E 盘，各阶段机器清单标识实际路径和哈希。

2026-10-03：用户纠正了生成式目标。2026-10-02 审查中的“四参数恒体积优化”主线已撤销，原机器索引/历史报告保留以追溯。纠正记录见 [scope_correction_20261003.json](scope_correction_20261003.json)。
