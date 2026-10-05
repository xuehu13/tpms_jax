# 文件地图

更新：2026-10-05。本Windows目录是阅读入口与历史材料；正式维护和计算只在WSL `/home/xuehu/projects/tpms_jax`。当前任务是原大压缩第4步的梯度验证，不是新开网格扫描或训练。

## 唯一活动入口

| Windows阅读文件 | 正式仓库位置 | 职责 |
| --- | --- | --- |
| [阅读入口](../README.md) | README.md | 进入当前解释与任务 |
| [背景](RESEARCH_BACKGROUND.md) | docs/RESEARCH_BACKGROUND.md | 长期问题、固定研究输入与证据约定 |
| [规划](RESEARCH_PLAN.md) | docs/RESEARCH_PLAN.md | 原第4步的四项执行任务 |
| [状态](RESEARCH_STATUS.md) | docs/RESEARCH_STATUS.md | 完成、未完成及实际成本 |
| [综合说明](TPMS_RESEARCH_REVIEW.md) | docs/TPMS_RESEARCH_REVIEW.md | 最新方法、数字解释、疑虑与判断 |
| [当前梯度执行](GRADIENT20_PROGRESS_REPORT.md) | docs/GRADIENT20_PROGRESS_REPORT.md | 本轮完整路径/梯度与成本 |
| [前向冻结记录](FORWARD20_PROGRESS_REPORT.md) | docs/FORWARD20_PROGRESS_REPORT.md | 截至51136a5的冻结证据；其当时下一步不是活动计划 |
| [文献索引](PAPER_ROUTE.md) | docs/PAPER_ROUTE.md | 阅读范围和可借鉴边界 |
| [历史说明](history/README.md) | docs/history/README.md（正式历史独立保存） | 旧报告/规划/传输材料 |
| AGENTS.md | AGENTS.md | 简洁工作约定 |

`output/pdf/TPMS_RESEARCH_REVIEW.pdf`是前次调查快照，本版Markdown才是最新综合说明；本次不新增重复PDF。Windows文件地图不意味着所有本机大数据都已上传GitHub。

## 正式程序

以下路径相对于WSL正式仓库；不从Windows `work/tpms_jax`旧阅读副本运行。

| 路径 | 用途 |
| --- | --- |
| hyperelastic_fem.py | 唯一共享Neo-Hookean材料核/残差组装；默认HEX8，可选HEX27，规则参考几何和批处理 |
| pbc.py、fem.py、density_fem.py | 周期自由度/既有线性背景材料场 |
| surface_distance.py | 周期中面距离、恒厚表示；不是已认证形态AD |
| geometry.py、volume.py、voxel_field.py、design_fem.py | 几何/积分用料/周期占据及已验证的专用梯度 |
| scripts/thin_target_explicit.py | 当前XYZ物理显式前向，中央差分/HRZ质量/块回退；材料/HRZ质量/响应可追踪，完整路径调用保存在本轮experiment.py，CLI厚度开关本身不是AD |
| scripts/thin_target_finite.py、scripts/thin_thickness_gradient.py | 旧静力路径及厚度伴随；不冒充20%有效梯度 |
| scripts/extract_thin_explicit.py、scripts/extract_thin_finite.py | Abaqus只读结果提取 |
| scripts/README.md、pixi.toml、pixi.lock、tests/ | 入口说明、锁定环境与维护检验 |

## 当前科学结果

实验根：`/home/xuehu/projects/tpms_jax/validation/large_compression_20261005_r6`。本次不改原输入、JSON、NPZ或日志。

| 相对目录/文件 | 内容及判断 |
| --- | --- |
| static/a0p150000、static/a0p200000 | 15%线性迭代阻塞；20%静力仅准备，不能叫20%求解失败 |
| abaqus/standard、explicit_T0p020、explicit_T0p040 | 三条无接触壳20%参考及提取，参考能量质量尚非严格通过 |
| explicit/N64_T0p004、N64_T0p004_adaptive | HEX8第一次保护停止及回退减步完成20%的原基准 |
| background_bridge/ | 原生C3D8同Gauss占据/指定保存位移桥接；非独立平衡曲线 |
| quadratic_candidate/T0p004_compact | 当前HEX27快路径；主体15.59分钟 |
| quadratic_candidate/T0p008_compact | 慢路径速率复核；主体30.86分钟 |
| quadratic_candidate/rate_check.json、rate_response.png | 速率及约10%响应目标通过；前向记录冻结；最新厚度路径导数见gradient20_path_20261005，严格替代未认证 |
| quadratic_wave/、quadratic_wave_runtime_geometry/ | 二次质量/波基础；不是TPMS实体精度证据 |
| gradient20/probe、endpoint、endpoint_energy | 中心平衡、候选伴随与两次减薄重平衡停止；当时的失败/候选冻结，不是本轮活动状态 |
| gradient20_path_20261005/ | 正负完整路径、同网格JVP、独立导数/曲线损失、原源码及运行快照；最新说明见新执行记录 |
| README.md、status.json | 本轮入口与原始执行状态；最新解释只在活动文档 |

原薄壁四步：`validation/thin_target_20261004_r5`；其根、step2/diagnostic_xyz、step3_xyz、step4_thickness_a01及_a05保留几何、小变形、有限应变、1%/5%梯度结果。对应阅读报告已移到[历史报告](history/scientific_reports_20261005/)。第一至第四轮历史实验仍在validation各日期目录，旧失败/条件项不自动成为待办。最终M4 CSV仍在正式`results/m4_numerical_study.csv`，Git忽略。

## Abaqus和用户资料

命令：`E:/ABAQUS/2026/Commands/abaqus.bat`。计算包根：`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus`；大ODB和原日志仍在那里，work/通常是作业目录，不搬动。

| 包根相对位置 | 内容 |
| --- | --- |
| large_compression_20261005_r6_standard、_explicit_T0p020、_explicit_T0p040 | 当前纯弹性XYZ壳20%原始作业 |
| background_bridge_20261005/point_check、saved20 | 原生C3D8单元点检及保存20%状态桥接 |
| thin_target_20261004_r5_step2、_step2_xyz、_step3_xyz_01、_05、_05_refined、_10 | 旧薄壁小变形/有限应变壳结果 |
| thin_target_20261005_r5_step4_t0p498000、_t0p502000 | 1%下独立壳厚扰动，不能当20%梯度验证 |
| uniform1001、element_comparison_20261002、discrete_comparison_verified_20261002及其他日期包 | 冻结完整体/单元/二值与历史诊断 |

用户中面参考：`F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_28/abaqus`。原厚0.328595mm、塑性/摩擦压板只作背景资料，当前不照搬。文献原件在`C:/Users/xuehu/Desktop/tpms优化`，未移动或改写。

## 整理与追溯

| 本工作区位置 | 内容 |
| --- | --- |
| history/before_feasibility_review_20261005/ | 本次修改前9份活动说明的原字节副本 |
| history/scientific_reports_20261005/ | 从根目录移入的11份旧阶段报告；内部相对链接仍按原根目录上下文解释 |
| history/git_transport/completed_20261005_later/ | 本次6项已发布bare仓库/bundle；旧发布收据保持原文 |
| history/git_transport/completed_20261005/及更早历史目录 | 前次归档，不恢复为活动源码 |
| history/before_gradient_path_20261005/ | 本轮修改前活动文档原字节；正式仓库独立保存 |
| work/gradient_path_20261005/ | 本轮调度/审阅/文档/发布工具，不维护第二套FEM |
| work/feasibility_review_20261005/ | 本次文档工具、文献/整理/同步/发布收据，零科学作业 |
| work/background_bridge_20261005/、forward20_step1_20261005/ | 已完成调度/分析工具及收据，正式入口只在WSL/scripts |
| work/large_compression_review_20261005/ | 前次23PDF身份及6篇选页摘录/预览，本次复用 |
| work/tpms_jax及历史缓存/source_at_run | 旧副本/运行源码快照；只读追溯，不维护第二套FEM |

本次旧新路径与SHA256：[整理收据](history/feasibility_review_20261005/windows_organization_receipt.json)。原始计算、停止日志、用户PDF和Abaqus作业不删除；已完成工具不重跑。正式仓库独立归档其修改前文件，不能拿Windows副本覆盖原历史。
