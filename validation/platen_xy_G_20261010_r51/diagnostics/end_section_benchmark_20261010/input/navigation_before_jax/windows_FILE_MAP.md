# 文件地图 当前入口与冻结证据

更新2026-10-10。R=`/home/xuehu/projects/tpms_jax`为唯一正式仓库，W为当前Windows工作区。旧调查收口，新G/P/D六项动态任务已结束，全部20%及保载。

## 当前入口

| Windows入口 | 用途 | 正式仓库 |
| --- | --- | --- |
| START_HERE.md | 导航 | R/README.md |
| RESEARCH_BACKGROUND.md | 长期问题与物理边界 | R/docs/RESEARCH_BACKGROUND.md |
| RESEARCH_PLAN.md | 当前r51四步方案、参数与执行边界 | R/docs/RESEARCH_PLAN.md |
| PROJECT_OVERVIEW.md | 实际范围与结论 | R/docs/RESEARCH_STATUS.md |
| TPMS_RESEARCH_REVIEW.md | 方法与有限可行性 | R/docs/TPMS_RESEARCH_REVIEW.md |
| MECHANISM_ANALYSIS.md | 已收口机制分析 | R/docs/MECHANISM_ANALYSIS.md |
| PAPER_ROUTE.md | 文献与实际阅读范围 | R/docs/PAPER_ROUTE.md |
| WEEKLY_RESEARCH_SUMMARY.md | 2026-10-07事实快照 | R/docs/WEEKLY_RESEARCH_SUMMARY.md |
| AGENTS.md | 执行边界 | R/AGENTS.md |

## 当前r51压板批次

- 用户请求的[学习快照ZIP](output/r51_JAX_model_learning_20261010.zip)及[中文阅读指南](output/r51_JAX_model_learning_20261010/README_先读.md)：67项源文件逐字节复制，94项包内文件，约29 MiB；有实际数组、代码、库许可证和校验清单。仅供学习，不是正式求解入口；打包未改WSL工程、未启动新仿真。制作配方在history/organization_r51_learning_20261010。
- 正式：R/validation/platen_xy_G_20261010_r51。tools、input、abaqus、jax、checks按用途分目录，根README/status为实验状态。
- Windows阅读副本：output/r51_platen_xy_G_20261010，不存第二套求解器。
- 入口备份：W/history/before_platen_xy_20261010_r51；正式旧入口在R/docs/history/before_platen_xy_20261010_r51。
- 原用户压板参照：F:/auto_abaqus/work/para_aly/Fine/T0p02/MS0/diverse_28/abaqus/physical.inp，只读，不照搬材料/壁厚/摩擦。

- 本轮实际完成2%＋保载，尚未通过约10%短程对照。阅读说明：[r51结果](output/r51_platen_xy_G_20261010/README.md)。传输与组织配方仅归档在W/history/organization_platen_xy_20261010_r51，不存第二套求解入口。

## 已完成r50批次

- 正式记录：R/validation/canonical_tpms_20261009_r50；G/P/D原始结果、protocol、预算收据和只读分析工具。
- [最终阅读副本](output/r50_canonical_tpms_20261009/README.md)：曲线、表格、能量、形变与逐项预算。
- r50完成记录保留；r51当前无活动计算，见本批status.json。G原壳首启ODB保留在E盘原_dynamic，修复网格的G/P另用_dynamic_quality；D为原_dynamic。

## 成品与历史证据

- [完整差异排查Word](output/TPMS背景有限元与Abaqus差异排查完整总结_20261009.docx)：r6至r49总索引，141页、64图、35表。
- [r47历史曲线图集和完整误差表](output/r47_simulation_atlas_20261009/README.md)：正式R/validation/simulation_atlas_20261009_r47，41路径/短段、22静态点、23图、CSV/Excel。
- [r48壳时长对照](output/r48_shell_rate_convergence_20261009/README.md)：正式R/validation/shell_rate_convergence_20261009_r48。
- [r49时间逻辑数据](output/r49_time_logic_audit_20261009/audit.json)：正式R/validation/time_logic_audit_20261009_r49/audit.json。
- [成品入口](output/README.md)：2026-10-07周报告/PPT与收口Word；日期快照不随新方案改写。

完整实验目录只见R/validation/README.md，逐轮细节在冻结报告和Word附录，本地图不重复列几十项状态。

## 正式程序与原件

生产：R/hyperelastic_fem.py、surface_distance.py、pbc.py、fem.py、scripts/thin_target_explicit.py；环境pixi.lock，用途见R/scripts/README.md。geometry.py含早期G/P阈值接口，不等于当前恒厚验证。

Abaqus命令`E:/ABAQUS/2026/Commands/abaqus.bat`，原INP/ODB根`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/`。原中面`F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_04/abaqus/ingredients/shell_mesh.inc`，diverse_28同级。论文`C:/Users/xuehu/Desktop/tpms优化/`，用户原Word/PPT`C:/Users/xuehu/Desktop/tpms总结1006/`，均不改。

## 本轮归档与凭据

W/history/organization_20261009_canonical_tpms_plan：

| 子目录或凭据 | 内容 |
| --- | --- |
| before_windows/ | 本轮修改前Windows入口 |
| before_wsl/ | 正式导航原文；R/docs/history/before_canonical_tpms_plan_20261009亦保留备份 |
| report_authoring/discrepancy_closure_20261009/ | 完成的Word制作、来源副本、渲染/QA，约94 MiB，原work目录 |
| reading_previews/pdfs/ | 23张旧页面预览，原tmp/pdfs |
| completed_tools/r47*、r48*、r49*/ | 7个已完成Windows辅助脚本；正式WSL配方原位 |
| archive_before.json、archive_receipt.json | 437归档文件、215原output文件的映射与哈希 |
| scientific_integrity.json | 原2243项科学/生产文件及补充程序基线核对，零新FEM/Abaqus/AD |

历史制作配方中的绝对路径是制作当时位置，重做旧报告前按映射恢复工作路径。此前各before_*与completed_tools_*保留，不重复合并或删除。Git不包含全部大场/日志/ODB，克隆不是完整实验备份。


本批过时的三个Windows传输草稿已移至history/organization_20261009_canonical_tpms_plan/completed_tools/r50_transfer_drafts，哈希核对保留；实际r50代码只见正式WSL目录。


本轮端部原因诊断：R/validation/platen_xy_G_20261010_r51/diagnostics/contact_origin_20261010；三个新增2%动态任务完成。Windows图表与指标：output/r51_platen_xy_G_20261010/contact_origin；说明补入原r51 README。T2ms原生ODB/INP在E盘同根platen_xy_G_20261010_r51_T2ms_diagnostic，原学习ZIP与原结果不更新或覆盖。


共同步长及映射诊断：R/validation/platen_xy_G_20261010_r51/diagnostics/contact_support_control_20261010；Windows阅读图表output/r51_platen_xy_G_20261010/contact_support_control。新增1条完整路径，半幅映射仅准备停止；沿用原r51说明，不新建第二份总报告。


同实体端部机制基准：R/validation/platen_xy_G_20261010_r51/diagnostics/end_section_benchmark_20261010；两项Abaqus静力完成，无新G或JAX路径。Windows图表/CSV/metrics在output/r51_platen_xy_G_20261010/end_section_benchmark，说明仍集中在原r51 README。原生INP/ODB分别在E盘同根r51_end_section_benchmark_20261010_cap_v2、r51_end_section_benchmark_20261010_rim；首轮cap目录仅为失败datacheck保留。制作配方归档到history/organization_platen_xy_20261010_r51/end_section_authoring，不是第二套求解入口。
