# 文件地图：当前入口与冻结证据

更新2026-10-06。正式程序仅WSL `/home/xuehu/projects/tpms_jax`，Windows是阅读/历史。详细旧地图已保存在`history/before_research_synthesis_20261006/FILE_MAP.md`，本页只列阅读及定位所需路径。

## 当前阅读入口

| 文件 | 唯一用途；正式仓库对应 |
| --- | --- |
| RESEARCH_BACKGROUND.md | 长期问题与边界；docs同名 |
| RESEARCH_PLAN.md | 新几何迁移唯一四步；第1步完成、第2步目标未过、第3/4步限定诊断与范围收口完成，旧四步只作完成依据；docs同名 |
| RESEARCH_STATUS.md | 完成/未完成事实；docs/RESEARCH_STATUS.md |
| TPMS_RESEARCH_REVIEW.md | 当前完整方法、数字解释、进度及可行性；docs同名 |
| GEOMETRY_TRANSFER_REVIEW.md | r16现有记录第3/4步诊断与范围；validation/geometry_transfer_review_20261006_r16/REVIEW.md |
| PAPER_ROUTE.md | 文献及实际阅读范围；docs同名 |
| START_HERE.md、AGENTS.md | 阅读入口与工作约定；README.md、根AGENTS.md |
| THICKNESS_RANGE_PROGRESS.md | r13匹配0.45/0.55mm完整20%及范围；docs同名 |
| VOID_CONTINUATION_PROGRESS.md | 最新r12执行细节，冻结原字节；docs同名 |
| 其余*_PROGRESS.md、*_REPORT.md | 冻结各轮执行证据，历史下一步不是待办；docs同名 |

## 正式程序与科学结果

以下路径均相对WSL正式根。维护现有一套FEM，历史入口仍可复现；此次不做无依据的代码搬迁或重构。

| 路径 | 用途 |
| --- | --- |
| hyperelastic_fem.py | 唯一共享NH/可选C²虚域材料、内力、HEX8/HEX27；默认NH保持 |
| scripts/thin_target_explicit.py | 当前XYZ中央差分/HRZ/回退减步入口，研究候选需明确选objective_void |
| surface_distance.py、pbc.py | 周期中面距离/真实Gauss占据、周期自由度 |
| fem.py、density_fem.py及其他原模块 | 原线性/几何/用料/专用设计计算，按既有范围维护 |
| scripts/README.md、tests/、pixi.toml/lock | 现有入口分类、维护检查、锁定环境 |
| validation/geometry_transfer_review_20261006_r16/ | 新第3/4步只读后处理与收口；REVIEW.md、三份诊断、scope_decision、图与冻结哈希；没有新仿真或梯度 |
| validation/geometry_transfer_20261006_r15/ | diverse_04匹配输入及第2步JAX中止/壳完整诊断；step1_decision与STEP2/step2_summary分别留输入与执行事实；零完整AD |
| validation/forward_scope_20261006_r14/ | 只读四步收口：范围矩阵、三个原中面库存、diverse_04候选、未来完整梯度方案；零新力学 |
| validation/thickness_range_20261006_r13/ | 匹配厚度输入、两条完整20%、两份壳参照、27/125点与模式、范围决定及哈希 |
| validation/void_continuation_20261006_r12/ | 当前C²候选：输入、45项检查、原6点、快慢20%、125点、速率、范围决定及证据哈希 |
| validation/large_compression_20261005_r6/ | 冻结原HEX8/HEX27、壳参照、背景桥接、20%原梯度失败与定位 |
| validation/thin_target_20261004_r5/ | 当前恒厚中面/Gauss基础及早期薄壁范围 |
| validation/simulation_error_20261006_r7/ | 保存场定位及失败三维体网格准备 |
| validation/jax_improvement_20261006_r8/ | 64点未采纳候选及27/125点检查 |
| validation/virtual_kernel_20261006_r9/ | 经典虚域转动缺陷，未采纳 |
| validation/objective_virtual_20261006_r10/ | 客观核原保存场验证，当时未接时间推进 |
| validation/objective_path_20261006_r11/ | 上一客观路径改善及6点混合域失败，保持 |
| validation/README.md、results/m4_numerical_study.csv | 全部实验索引；早期M4最终10工况CSV（Git忽略） |

r12的`T0p004/`、`T0p008/`含完整输入、结果、最终field.npz及运行源码；`*_analysis/`含对壳、模式和27/125点保存场检查；`decision.json`、`rate_check.json`、`evidence_manifest.json`记录验收及哈希。大场/日志/源码快照留本机，GitHub不含全部本机数据。

## Abaqus与用户资料

- 命令：`E:/ABAQUS/2026/Commands/abaqus.bat`。
- 包根：`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus`；现有20%壳包为`large_compression_20261005_r6_standard`、`_explicit_T0p020`、`_explicit_T0p040`，含INP/ODB/日志。
- 用户原中面：`F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_28/abaqus`。
- 用户论文：`C:/Users/xuehu/Desktop/tpms优化`；原件未移动/改写。
- 原模型和资料仅参考，不照搬原厚度、塑性或压板边界。r13新增0.45/0.55mm各一个Explicit匹配壳，原模型/资料不改。

## Windows历史与整理回执

已完成`work/<旧目录名>/`移动到`history/completed_tools_20261006/<旧目录名>/`，目录内部原字节保留，包括日志、摘录、transport.git和发布收据。此次补归档`research_synthesis_20261006`、`thickness_range_20261006`、`forward_scope_20261006`三个目录，143个文件哈希一致。旧脚本中的旧绝对路径仅作历史上下文，不能直接重跑；恢复定位先看Windows history/completed_tools_20261006/README.md（Windows归档索引）。早期完整映射/哈希现位于`history/completed_tools_20261006/research_synthesis_20261006/organization_receipt.json`；此次映射/哈希在`work/geometry_plan_20261006/organization_receipt.json`。

本次改写前活动说明在`history/before_research_synthesis_20261006/`。正式仓库备份/整理收据在`docs/history/research_synthesis_20261006/`。科学结果、用户论文、ODB和WSL运行源码没有移动或删除。

更早的`history/reading_checkout_20261006/tpms_jax`是旧阅读检出，`history/git_transport/`是原已完成传输，`history/rendered_reports_20261005/`为旧PDF。它们均非活动程序/最新说明，原详细映射可查备份地图。本轮不新增重复PDF。

## 当前新增证据

r13原生作业：`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/thickness_range_20261006_r13_t0p45_explicit_T0p040`及同名`t0p55`目录。ODB留该处；正式实验各厚度`abaqus/explicit_T0p040/`有提取和retention.json。各厚度`T0p004/`含场/完整结果、`analysis/`含27/125点/模式与comparison.json；一份规范INP和厚度单行差异可复现两侧输入。Windows `history/completed_tools_20261006/thickness_range_20261006`仅完成工具/发布收据，不是FEM入口。

旧第4步收口未新增重复长报告或PDF。WSL r14下`forward_scope.json`为范围，`geometry_inventory.json`为只读原中面筛选，`geometry_candidate.json`为单一diverse_04候选，`gradient_gate_proposal.json`仅未来方案，`decision.json`/输入与验证收据为依据。Windows `history/completed_tools_20261006/forward_scope_20261006`为完成的收口/发布工具，非活动FEM。用户原中面位置为`F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_04/abaqus/ingredients/shell_mesh.inc`，未改。

当前`work/geometry_plan_20261006`只放这次状态/规划/归档与发布收据，不是实验或求解器；正式对应`docs/history/geometry_plan_20261006/receipt.json`。r15已建匹配输入/Gauss占据与HRZ场及壳INP；当前有完整壳ODB/响应及JAX稀疏接受观察/拒绝场，没有有效JAX20%末态。Windows `work/geometry_transfer_20261006`仅本步准备/恢复/诊断/同步收据；正式程序仍唯一共享FEM和显式入口。

r15正式位置：`/home/xuehu/projects/tpms_jax/validation/geometry_transfer_20261006_r15`。`gauss_field.npz`是真实27点占据，`hrz_mass.npz`是节点/周期质量，`surface_geometry.npz`为中面；`preparation.json`保留原false，`band_bracket_diagnosis.json`解释4条法向，`step1_decision.json`为当前限定就绪。规范壳包在其`abaqus/explicit_T0p040/`，同字节原生包在`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_transfer_20261006_r15_diverse04_explicit_T0p040`；有一次壳ODB/日志。第1步科学数组/首次日志原字节保留，首次预处理计时未知不补造。

第2步：r15的T0p004为空输出目录及缓存启动失败收据；实际JAX尝试为T0p004_cpu_reference，含input/progress/rejected_blocks、3拒绝场和source_at_run，无result/有效末态。operator_stop.json保留停止前监测；accepted_logged_observations.json只提取14条真实稀疏接受观察，不等于完整曲线。abaqus/explicit_T0p040/results保留壳JSON/场/日志/retention，ODB留原生不重复复制。step2_summary.json和STEP2.md分别为结构化与可读事实。Windows work/geometry_forward_20261006仅本次调用/提取/留存/发布工具，非第二套FEM。

后两步收口：r16的existing_record_review.py只读已有场/日志，复用共享材料核求值，无内力组装/时间推进；不是新求解器。rejected_endpoint_diagnosis只诊断拒绝子集，partial_force_diagnosis只有14个真实JAX观察对壳插值；不输出完整20%误差/模式。frozen_before.json保全旧证据，verification/evidence_manifest验证追溯；postprocess_attempt01保留Inf序列化失败，不是新仿真。Windows work/geometry_closure_20261006仅本次后处理/说明/发布工具；GEOMETRY_TRANSFER_REVIEW.md镜像正式REVIEW.md，图在该work目录。
