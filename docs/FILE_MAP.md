# 文件地图与程序组织

[研究主规划](RESEARCH_PLAN.md) 规定研究问题和顺序；[研究状态](RESEARCH_STATUS.md) 记录已有证据。本页只说明文件用途，避免产生另一套待办。

## 正式位置

- WSL 程序：`/home/xuehu/projects/tpms_jax`；Windows 访问：`\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax`。
- Python：`.pixi/envs/default/bin/python`，由 `pixi.toml`/`pixi.lock` 管理，实际 JAX-FEM 为 PyPI 0.0.12。
- `/home/xuehu/projects/jax-fem` 只读参考，`/home/xuehu/projects/jax-fem-workspace` 是早期工作区。
- Abaqus 原始包：`E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus`；命令：`E:\ABAQUS\2026\Commands\abaqus.bat`。
- 旧壳模型：`F:\auto_abaqus\work`，仅作建模参考，旧参数无需照搬。

## 当前核心程序

| 模块 | 用途 |
| --- | --- |
| geometry.py / volume.py | Gyroid 隐式函数、投影与体积分数 |
| fem.py / pbc.py | HEX8 网格、基础弹性问题、周期约束 |
| density_fem.py | Gauss 点材料场与横向松弛 |
| design_fem.py | 已验证的四参数梯度适配；不是通用体素或网络接口 |
| binary_gyroid.py | 二值域与贴体网格参考 |

现有 7 个核心模块保持位置。阶段 2 按需要补体素适配，尽量复用现有求解器；阶段 4 实施时再建立必要的 `learning/` 数据/网络/训练/逆向入口，不预建空框架、不复制两套 FEM。每个实验只保留必要配置及结果清单。

## scripts/ 按用途查找

| 用途 | 入口 |
| --- | --- |
| 早期示例 | m1_volume.py；m2_*cube.py；m3_gyroid_first.py / m3_lateral_relaxation.py |
| 当前背景计算 | m4_numerical_study.py |
| 参数梯度检查 | check_design_gradients.py（读取已保存 plan.json） |
| 均匀 Abaqus 对照 | prepare_uniform_baseline.py / extract_uniform_baseline.py |
| 单元算子诊断 | abaqus_element_comparison.py |
| 同离散对照 | prepare_abaqus_discrete.py / run_abaqus_discrete.ps1 / extract_abaqus_discrete.py |
| 二值实体对照 | prepare_abaqus_binary.py / prepare_binary_lateral_pair.py / run_abaqus_binary.ps1 / extract_abaqus_binary.py |
| 体积、质量及证据 | estimate_binary_volume.py / abaqus_mesh_quality.py / capture_* / plot_abaqus_binary.py |

`prepare_*` 生成输入，`run_*` 提交，`extract_*` 读 ODB，`capture_*` 归档。各阶段目录内的 prepare.py/summarize.py 是固定实验的复现材料。调用准确参数见相应阶段 README；此处不提供尚未实现的训练/优化命令。

## 文档与数据

| 路径 | 职责 |
| --- | --- |
| AGENTS.md | 后续 agent 的阅读顺序与工作约定，不另定义研究路线 |
| README.md | 简短入口 |
| docs/RESEARCH_PLAN.md | 唯一完整研究规划，含长期训练/逆向设计与阶段完成条件 |
| docs/RESEARCH_STATUS.md | 当前阶段、已有结果和未验证范围 |
| docs/PAPER_ROUTE.md | 两篇论文的事实与方法关系 |
| docs/FILE_MAP.md | 文件索引与组织规则 |
| validation/README.md / validation/*/ | 冻结实验的证据索引、输入、日志和验收 |
| validation/research_audit_20261002.json | 历史 23 项作业索引；旧计划字段只供追溯 |
| validation/scope_correction_20261003.json | 历史范围纠正记录；不是当前执行计划 |
| tests/ | 最新实际完整回归 125 项；本次文档编辑未重跑 |
| results/ | 本机结果，Git 忽略；后续大型训练数据/检查点也采用本机存储加轻量清单 |

M4 原 `results/m4_numerical_study.csv` 与 `validation/m4_review/fixed.csv` 字节一致，SHA256 为 `dc6c3c18881d75737cb99678e60376bc30f54b93f54504ea84a9ff8141bd0fb4`。保留它及现有正式证据，新实验使用新结果目录。

## Abaqus 包索引

以下子目录均在上面的 E 盘根目录中，正式成功作业共 23 项（含 2 项矩阵诊断）。

| 包目录 | 作业数 | 内容 |
| --- | ---: | --- |
| uniform1001/abaqus_uniform_baseline | 3 | C3D8 均匀实体 |
| element_comparison_20261002 | 2 | 单元矩阵诊断 |
| discrete_comparison_20261002 | 1 | N4 同离散对照 |
| discrete_comparison_verified_20261002 | 2 | N16 固定/自由同离散对照 |
| binary_gyroid_20261002 | 13 | C3D10 均匀及二值实体加密 |
| mesh_quality_20261002 | 2 | G24 同几何网格重定位 |

包根保留输入和清单，`work/` 保留 ODB/求解日志/提取结果，`scripts/` 保留 Windows 执行副本。用包路径、工况和 INP 哈希识别作业。提取使用 Abaqus Python 和脚本绝对路径，JAX 使用 WSL Pixi Python。

## Windows 阅读工作区

`C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an`：

- START_HERE.md 是入口；RESEARCH_PLAN.md 是主规划的阅读副本；PROJECT_OVERVIEW.md 是当前状态副本；FILE_MAP.md/PAPER_ROUTE.md 与正式 docs 对应。
- 根目录各阶段 report/PNG/summary/log/receipt 为历史快照，保留原路径和字节，以维持图链接和既有哈希。
- work/archive_20261002/ 保留早期副本与报告；work/archive_20261003/ 保留较早草案。旧文件中的计划不覆盖主规划。
- work/tpms_jax 是 Git 传输副本，原未提交修改保留；不作为科研程序入口。work/*.bundle 仅用于传输。
- outputs/ 是早期导出包，tmp/ 是一次性整理工具；都不作为正式程序。原 ZIP 和 Abaqus 文件不移动。

此次只精简文档及入口，没有改变程序架构、锁定环境或原始结果。后续架构调整随实际阶段推进，先解决明确需求再抽取公共模块。
