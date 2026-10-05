# 文件地图：活动入口、正式程序与冻结证据

更新2026-10-06。正式程序在WSL `/home/xuehu/projects/tpms_jax`，Windows工作区为阅读/历史。本轮执行JAX积分单因素改进；只维护一个显式入口，共享材料/FEM核和原科学文件冻结。

## 现在读哪些文件

| Windows名称 | 用途；正式仓库位置 |
| --- | --- |
| START_HERE.md | 阅读入口；README.md |
| RESEARCH_BACKGROUND.md | 长期目标/边界；docs同名 |
| RESEARCH_PLAN.md | 唯一活动四步：定位保留、JAX改进优先、有限工况验证、范围/梯度关口；积分因素试完、候选未采纳、下一项虚域核验证；docs同名 |
| PROJECT_OVERVIEW.md | 事实状态；docs/RESEARCH_STATUS.md |
| TPMS_RESEARCH_REVIEW.md | 方法、符号、误差因素、客观可行性；docs同名 |
| SIMULATION_ERROR_PROGRESS.md | 冻结的定位/参照网格尝试；其后续建议已被主规划调整；docs同名 |
| JAX_INTEGRATION_PROGRESS.md | 本轮27/125点探针、64点新路径及资源处理；docs同名 |
| PAPER_ROUTE.md | 来源与阅读范围；docs同名 |
| FORWARD20_PROGRESS_REPORT.md | 冻结前向执行证据；docs同名，不当最新计划 |
| GRADIENT20_PROGRESS_REPORT.md | 冻结正负路径/AD失败证据；docs同名，下一项只看主规划 |
| AGENTS.md | 工作约定；正式根同名 |

## 唯一正式程序

以下为WSL仓库相对路径。源码由现有模块分工维护，不从Windows旧副本、validation/source_*或归档调度器运行。

| 路径 | 用途 |
| --- | --- |
| hyperelastic_fem.py | 唯一共享NH材料能/内力，HEX8/HEX27与周期参考几何 |
| pbc.py、fem.py、density_fem.py | 周期自由度与线弹性背景组装 |
| surface_distance.py | 周期中面距离、恒厚占据，未认证形态AD |
| geometry.py、volume.py、voxel_field.py、design_fem.py | 原几何/用料/场及专用已验证梯度，历史范围见报告 |
| scripts/thin_target_explicit.py | 当前XYZ中央差分/HRZ/回退减步、可追踪占据质量及响应 |
| scripts/thin_target_finite.py、thin_thickness_gradient.py | 原静力与厚度伴随入口，旧候选不冒充20%有效梯度 |
| scripts/extract_thin_explicit.py、extract_thin_finite.py | Abaqus只读提取 |
| scripts/README.md、pixi.toml、pixi.lock、tests/ | 入口说明、锁定环境及维护检验 |

## 本轮JAX单因素改进

根：`/home/xuehu/projects/tpms_jax/validation/jax_improvement_20261006_r8`。

| 位置 | 内容 |
| --- | --- |
| integration_probe/ | 原快/慢保存场27和125点采样；源/输入哈希、结果；负J点不计总材料响应 |
| source_before/、source_change.json | 原显式入口字节和有意源码变化 |
| uniform_q6/ | 已知周期波64点动态检查，通过 |
| device_transfer_probe/ | 实际点几何/CPU到GPU小模型接口检查，不算独立物理案例 |
| T0p004_q6/ | GPU参考几何显存不足，未开始时间推进；resource_failure.json |
| T0p004_q6_cpu_geometry/ | 64点代表薄壁完整20%及保载；未采纳，实际场/输入/源码/结果 |
| candidate_probe/ | 新20%保存场64/125点复查，125点仍有44软域负J |

Windows辅助执行/日志：`work/jax_improvement_20261006`，不是第二套FEM。旧r6/r7科学证据均保留原路径/字节。

## 当前主要结果

根：`/home/xuehu/projects/tpms_jax/validation/large_compression_20261005_r6`。

| 相对位置 | 内容 |
| --- | --- |
| quadratic_candidate/T0p004_compact、T0p008_compact | HEX27快/慢完整20%，主体15.59/30.86分钟，场/日志本机 |
| quadratic_candidate/rate_check.json、comparison.json、field_review.json | 约10%响应与速率通过，实体/软域及模式摘要；未认证严格替代 |
| background_bridge/ | C3D8同Gauss占据/保存位移桥接，非独立平衡曲线 |
| gradient20_path_20261005/adaptive_minus、adaptive_plus | 厚度正负完整路径 |
| gradient20_path_20261005/path_ad_full、gradient_validation.json | 完整JVP、10%/15%通过、20%未过、损失核对 |
| gradient20_path_20261005/path_stability_probe、short_stability_probe、tangent_localization | 局部高频/半步和Gauss敏感性定位，非唯一因果证明 |
| gradient20_path_20261005/evidence_manifest.json | 精选结果与本机大场/日志哈希、源码/环境追溯 |
| gradient20/endpoint* | 旧中心平衡/伴随候选、两次减薄Newton停止，冻结 |
| explicit/N64_T0p004*、static/a0p150000 | 原HEX8改善前基准与静力迭代阻塞 |
| abaqus/standard、explicit_T0p020、explicit_T0p040 | 三条壳参考摘要，能量质量仍有边界 |

目标薄壁基础在`validation/thin_target_20261004_r5`；第一至第四轮各日期实验冻结。早期M4最终CSV为`results/m4_numerical_study.csv`，Git忽略。原JSON中的阶段状态不自动覆盖最新解释或生成待办。

## 当前新实验：误差定位与三维参照准备

根：`/home/xuehu/projects/tpms_jax/validation/simulation_error_20261006_r7`。完整说明见[执行报告](SIMULATION_ERROR_PROGRESS.md)。

| 相对位置 | 内容 |
| --- | --- |
| step1/result.json、manifest.json、experiment.py | 四条旧曲线、快慢Q2 Gauss应力/能量/宏观力、膜厚/弯曲矩；主体18.18秒，无新完整求解 |
| step1/curves.csv、curves.png、normal_profiles.npz | 共同采样、曲线图、96法向剖面；NPZ留本机 |
| step1/faceted_feature_probe/ | 已有3处面片棱角指标的定点厚度诊断 |
| step2/geometry.json、sharp_surface.npz、sharp_surface.stl | 闭合周期裁剪尖锐距离带；大几何留本机，非力学参照 |
| step2/mesh_attempt、mesh_direct、surface_cleanup | 参数化不进展、直接体网格优化崩溃、焊接拓扑未过；全部保留，不自动重新运行 |
| evidence_manifest.json、README.md | 新文件/原输入/环境追溯与范围，完整日志本机保存 |

Windows工具及日志在`work/simulation_error_20261006`，是本轮调度/诊断记录，不维护第二套求解器。暂未新增Abaqus包或INP。

## Abaqus与用户资料

命令`E:/ABAQUS/2026/Commands/abaqus.bat`；作业包根`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus`。

| 包根相对位置 | 用途 |
| --- | --- |
| large_compression_20261005_r6_standard、_explicit_T0p020、_explicit_T0p040 | 当前壳20%原始INP/ODB/日志 |
| background_bridge_20261005/point_check、saved20 | 同Gauss材料场桥接，SDVINI/UHYPER |
| thin_target_20261004_r5_step2*、_step3_xyz_* | 原薄壁小变形/有限应变 |
| thin_target_20261005_r5_step4_t0p498000、_t0p502000 | 1%独立壳厚扰动，非20%梯度 |
| uniform1001及其他日期包 | 均匀体/材料场/二值等历史验证 |

用户中面`F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_28/abaqus`，原件不改。文献`C:/Users/xuehu/Desktop/tpms优化`，未移动或改写。大ODB和正式场仍在原位置，GitHub不含全部本机数据。

## 本次规范化与历史映射

| 位置 | 内容 |
| --- | --- |
| history/before_simulation_review_20261006/ | Windows活动说明修改前原字节；正式仓库独立保存在docs/history同名目录 |
| history/reading_checkout_20261006/tpms_jax/ | 原work/tpms_jax旧阅读代码副本，包括原Git/结果字节；不是正式运行入口 |
| history/git_transport/completed_20261006/ | 原work/gradient_path_20261005已发布transport.git和bundle；旧收据原路径按本表解释 |
| history/rendered_reports_20261005/TPMS_RESEARCH_REVIEW.pdf | 原output/pdf旧PDF，冻结快照 |
| 正式docs/history/rendered_reports_20261005/TPMS_RESEARCH_REVIEW.pdf | 原docs根旧PDF；最新解释只看Markdown |
| work/README.md | 完成调度器/摘录与当前整理工具的分类，不维护第二套FEM |
| work/simulation_review_20261006/ | 本次阅读、文档同步、归档/校验/发布工具与收据，零科学作业 |
| work/gradient_path_20261005/及其他日期work目录 | 已完成调度、诊断、发布收据，不自动重新运行 |
| history/scientific_reports_20261005/、其他before_* | 旧阶段和原说明；内部链接按原根目录上下文解释 |

本次归档均核对原字节哈希，路径收据`work/simulation_review_20261006/organization_receipt.json`。旧阅读副本/传输包可恢复，但不恢复成活动程序。科学原件、失败日志、用户论文和ODB不删除，归档报告不自动更新。本次没有新增重复PDF、框架或新求解器。

最新路线讨论工具与收据在work/jax_route_revision_20261006，零科学计算；原r7失败体网格仍在原位但不再是近期前置任务。正式活跃步骤只看RESEARCH_PLAN.md，科学文件位置不变。
