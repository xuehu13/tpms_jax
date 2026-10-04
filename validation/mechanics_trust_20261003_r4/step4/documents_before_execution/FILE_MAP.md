# 文件地图

更新：2026-10-04。本页说明位置；方法与数字看 [综合报告](TPMS_RESEARCH_REVIEW.md)、[第2步](ROUND4_STEP2_REPORT.md)及[第3步报告](ROUND4_STEP3_REPORT.md)，执行只看 [主规划](RESEARCH_PLAN.md)。

## 维护入口

| Windows阅读工作区 | 正式WSL文档 | 职责 |
| --- | --- | --- |
| START_HERE.md | README.md | 阅读入口 |
| TPMS_RESEARCH_REVIEW.md、output/pdf/TPMS_RESEARCH_REVIEW.pdf | docs/同名Markdown/PDF | 综合解释；PDF是可分享导出 |
| RESEARCH_BACKGROUND.md | docs/同名文件 | 稳定目的/约定 |
| RESEARCH_PLAN.md | docs/同名文件 | 第四轮唯一计划；第1/2/3步收口，下一步为已验证Gyroid的有限应变探查 |
| PROJECT_OVERVIEW.md | docs/RESEARCH_STATUS.md | 当前事实 |
| FILE_MAP.md、PAPER_ROUTE.md | docs/同名文件 | 位置/文献记录 |
| NEAR_TERM_REPORT.md、ROUND2_REPORT.md、ROUND3_REPORT.md | validation/near_term_20261003/README.md、docs/ROUND2_REPORT.md、docs/ROUND3_REPORT.md | 冻结轮次报告 |
| ROUND4_STEP1_REPORT.md | docs/同名文件及validation/mechanics_trust_20261003_r4/step1/README.md | 第四轮原始参考审计、比较定义与限制 |
| ROUND4_STEP2_REPORT.md | docs/同名文件及validation/mechanics_trust_20261003_r4/step2/README.md | Primitive整体响应对照、薄壁参考限制、成本与程序修订 |
| ROUND4_STEP3_REPORT.md | docs/同名文件及validation/mechanics_trust_20261003_r4/step3/README.md | 完整均匀实体有限应变到20%的算法基准与明确边界 |

正式目录 `/home/xuehu/projects/tpms_jax`；Windows访问 `\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax`。Python `.pixi/envs/default/bin/python`，pixi.toml/pixi.lock保留。`/home/xuehu/projects/jax-fem`只读参考，jax-fem-workspace为早期工作区。

## 核心程序

| 文件 | 职责 |
| --- | --- |
| geometry.py、volume.py | Gyroid/Primitive解析场、占据/体积 |
| fem.py、pbc.py | HEX8、小变形、周期约束 |
| hyperelastic_fem.py | Neo-Hookean有限应变材料/响应，复用周期运动学和JAX-FEM求解器；当前只有完整实体认证 |
| density_fem.py | 真实Gauss点材料场、宏观横向松弛 |
| design_fem.py | 当前参数、通用伴随及专用K/Vf导数 |
| voxel_field.py | 周期数组映射，先插值G再投影 |
| binary_gyroid.py | Gyroid/Primitive二值实体几何/网格及审查；第三轮壁宽G48及本轮薄壁G48未通过 |
| scripts/prepare_abaqus_binary.py、run_abaqus_binary.ps1、extract_abaqus_binary.py | 参考生成/运行/提取；提取复用extract_uniform_baseline.py |
| scripts/finite_strain_uniform.py、extract_finite_strain.py | 完整实体有限应变JAX路径、INP与完成ODB提取/检查；本轮研究入口 |
| tests/ | 维护检查；与研究工况分别计数 |

第2步最小扩展8个已有文件，138项回归通过；第3步仅新增一个材料模块、两个捕获/提取脚本和一个测试文件，最终144项回归通过。原FEM/密度/PBC/梯度求解器、环境不变，无FEM复制、新框架或新依赖。一次性执行/诊断程序保存在第四轮证据目录，不是另一套求解器；细节及未解决裁剪限制见第2/3步报告。

## 冻结证据与Abaqus

三轮正式证据在 `validation/near_term_20261003/`、`validation/geometry_interface_20261003_r2/`、`validation/learning_bridge_20261003_r3/`。包含输入/结果/日志、执行、source_before/source_after与source_manifest；它们是复现快照，不是维护代码，本次不移动/改写。旧manifest中文档SHA代表当时版本，历史文档保留，不改旧清单匹配新文档。

M4 CSV在 `results/m4_numerical_study.csv`，原10行、Git忽略、保持原样。三轮计划在 `docs/RESEARCH_PLAN_ROUND{1,2,3}_COMPLETED.md`，不是待办。验证总入口是正式 `validation/README.md`。

第四轮新证据在`validation/mechanics_trust_20261003_r4/step1/`：plan、audit程序/JSON、指纹核查及报告。第1步仅审计既有输入/网格/响应，没有新增求解，原三轮不改写。Windows一次性工具在`work/mechanics_trust_20261003_r4/step1/`，不是正式运行入口。

第2步证据在`validation/mechanics_trust_20261003_r4/step2/`：锁定plan、几何预检查、primitive/N48/N64结果与位移、比较/验收、thin_gyroid失败诊断、执行账本、最终closure、源码快照及历史指纹verification。Windows工具、图和发布收据在`work/mechanics_trust_20261004_r4_step2/`；原始数据以正式WSL和实际Abaqus包为准。

第3步证据在`validation/mechanics_trust_20261003_r4/step3/`：plan，三条N2/N4/减半路径JSON及位移，summary/details/图，原始Newton日志、执行账本、物理/维护检查、源程序快照及verification/closure。Windows工具、图与发布收据在`work/mechanics_trust_20261004_r4_step3/`。读取状态仍只从主规划和当前状态进入，不从一次性launch工具或历史报告自动启动下一步。

Abaqus根 `E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus`；命令 `E:\ABAQUS\2026\Commands\abaqus.bat`；旧壳参考 `F:\auto_abaqus\work`。

| Abaqus包目录 | 内容 |
| --- | --- |
| uniform1001/abaqus_uniform_baseline、element_comparison_20261002 | 均匀/单元矩阵 |
| discrete_comparison_20261002、discrete_comparison_verified_20261002 | N4/N16同离散 |
| binary_gyroid_20261002、mesh_quality_20261002 | 既有实体/质量/加密 |
| geometry_interface_20261003_r2/{c048,c060}/{G32,G48} | 第二轮常数锚点 |
| learning_bridge_20261003_r3/ax035/G32 | 第三轮成功分析 |
| learning_bridge_20261003_r3/ax035/G48/FAILED_BEFORE_SUBMISSION.json | 预处理失败、未提交；az035未建实体包 |
| mechanics_trust_20261004_r4/primitive/{G32,G48} | 本轮成功分析/对应datacheck、原始ODB、提取/验收；G48还有质量JSON及位移节点样本 |
| mechanics_trust_20261004_r4/thin_gyroid/G32 | 仅生成输入与网格，没有力学作业 |
| mechanics_trust_20261004_r4/thin_gyroid/{G48,G48_clipfix,G48_contactfix} | 三个提交前失败目录，原始日志在正式step2/thin_gyroid；不是结果 |
| mechanics_trust_20261004_r4/step3_uniform/{N4_d010,N4_d005} | 两条完整实体有限应变成功路径，21/41加载步；原始输入/ODB/dat/msg/sta、datacheck与验收；d010/d005为1/0.5个百分点增量 |

包根保留INP、mesh.npz、expected，work保留ODB/dat/msg/sta/提取/验收，scripts为实际执行副本。原始包不搬动。

## Windows整理后的目录

| 目录 | 用途 |
| --- | --- |
| history/reading_copies_before_review/ | 原根目录旧报告/图/汇总/日志副本/收据，字节不改，有索引与移动映射 |
| work/archive_20261003_documentation/tools/ | 原tmp顶层一次性脚本归档，不是正式执行程序 |
| work/documentation_review_20261003/ | 本次文档前快照、文献核查、PDF生成及QA |
| tmp/ | 旧阶段文档快照/文献缓存保留，不是执行入口 |
| output/pdf/ | 当前可分享PDF |
| outputs/ | 早期导出包/结果原位保留 |
| work/tpms_jax、work/*.bundle、work/archive_20261002/ | 历史Git传输/包/工具；原4个未提交文件保留 |

正式活动文档同步到WSL docs，冻结报告不改。整理核查见 `work/documentation_review_20261003/verification.json`，旧新路径/哈希映射见同目录 `organization_receipt.json`。第1/2/3步已收口，无本轮运行/排队作业；第4步未启动，历史停止项不自动续跑。制定前快照见 `work/planning_round4_20261003/receipt.json`，第1/2步同步见各自work目录publish_receipt，本步同步见`work/mechanics_trust_20261004_r4_step3/publish_receipt.json`。综合报告及其PDF保持制定第四轮前的原始快照；第2/3步报告为Markdown与科学图，不重复导出综合PDF。
