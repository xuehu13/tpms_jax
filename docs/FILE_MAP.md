# 文件地图与运行边界

当前研究进度/待办只看 [RESEARCH_STATUS.md](RESEARCH_STATUS.md)，指定论文接口见 [PAPER_ROUTE.md](PAPER_ROUTE.md)。旧四参数优化 R1–R4 已撤销。以下是文件用途索引，不新增运行框架。

## 正式程序位置

- WSL：`/home/xuehu/projects/tpms_jax`。
- Windows 访问：`\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax`。
- 实际 Python：`.pixi/envs/default/bin/python`；环境由 `pixi.toml`/`pixi.lock` 管理。实际安装的 JAX-FEM 为 PyPI 0.0.12。
- `/home/xuehu/projects/jax-fem` 仅是只读参考源码，不是当前运行库；`/home/xuehu/projects/jax-fem-workspace` 是早期验证工作区，保持不动。

| 根目录模块 | 职责 |
| --- | --- |
| geometry.py | Gyroid 隐式函数、双 sigmoid 投影等几何/代理场定义 |
| volume.py | 几何/投影体积分数估计与标定 |
| fem.py | 网格、基础弹性问题与求解封装 |
| pbc.py | 周期约束与自由度映射 |
| density_fem.py | 真实 Gauss 点材料插值及宏观横向松弛 |
| design_fem.py | 已完成的四参数梯度检查适配器，保留参考；不是通用体素/扩散输入接口 |
| binary_gyroid.py | 二值域裁剪、网格审查与贴体网格构建 |

保留现有 7 个核心模块、23 个 Python 入口/辅助脚本、2 个 PowerShell 提交器、16 个测试文件。本次没有重命名数值程序，也没有建立新的配置系统。

## scripts/：按工作使用

| 用途 | 脚本 |
| --- | --- |
| 早期示例/复现 | m1_volume.py；m2_linear_cube.py、m2_periodic_cube.py、m2_xy_periodic_cube.py；m3_gyroid_first.py、m3_lateral_relaxation.py |
| 当前投影数值计算 | m4_numerical_study.py；仅需新算例时运行，输出使用新目录/文件，保留原十行 CSV |
| 当前固定横向梯度检查 | check_design_gradients.py；接收预先保存的 plan.json，不覆盖已有正式结果 |
| 均匀 Abaqus 输入/提取 | prepare_uniform_baseline.py、extract_uniform_baseline.py |
| 单元公式诊断 | abaqus_element_comparison.py |
| 同离散场输入/提交/提取 | prepare_abaqus_discrete.py、run_abaqus_discrete.ps1、extract_abaqus_discrete.py |
| 二值实体输入/提交/提取 | prepare_abaqus_binary.py、prepare_binary_lateral_pair.py、run_abaqus_binary.ps1、extract_abaqus_binary.py |
| 体积/质量及证据整理 | estimate_binary_volume.py、abaqus_mesh_quality.py、capture_m4_evidence.py、capture_abaqus_comparison.py、capture_abaqus_binary.py、capture_binary_projection_reference.py、plot_abaqus_binary.py |

`prepare_*` 生成输入；`run_*.ps1` 真正提交作业；`extract_*` 读取 ODB；`capture_*` 归档；各阶段 `summarize.py` 汇总已有证据。入口名称不能代替验收记录。`run_abaqus_binary.ps1 -ExtractOnly` 可复核已有 ODB，不重算。阶段目录中的 plan.json/prepare.py/summarize.py 属于该固定实验的复现材料，不是另一套核心程序。

## 数据与文档

| 路径 | 内容与规则 |
| --- | --- |
| README.md | 当前项目入口，仅显示最新有效状态 |
| docs/RESEARCH_STATUS.md | 用户明确的 TPMS 体素计算/学习/生成目标，G1–G5 当前路线 |
| docs/PAPER_ROUTE.md | 两篇指定论文的方法、表示/力学差异及接口任务 |
| docs/FILE_MAP.md | 本文件，程序/数据用途和平台边界 |
| validation/README.md | 9 个正式证据阶段的索引 |
| validation/research_audit_20261002.json | 历史 23 项作业索引及字节核查；其中 R1–R4 计划已撤销 |
| validation/scope_correction_20261003.json | 本次用户目标纠正与源文件/原始数据不变核查 |
| validation/*/ | 历史计划、结果、原始轻量日志、验收、图和指纹；阶段测试数/下一步仅代表当时 |
| tests/ | 最新一次实际完整回归 125 项；文档整理不重复启动计算测试 |
| results/ | 本机大型/试算结果，被 .gitignore 忽略；不等于 Git 归档。M4 原 CSV 和日志保留 |
| results/m4_numerical_study.csv | 当前修复后 10 行数据，与 validation/m4_review/fixed.csv 字节一致；SHA256 dc6c3c18881d75737cb99678e60376bc30f54b93f54504ea84a9ff8141bd0fb4 |
| /tmp/m4v3.log 等 | 早期临时日志，可能已清空；不作为唯一证据存放处 |

## 大型 Abaqus 原始文件（保持现有路径）

根目录：`E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus`。

| 子目录 | 已验收正式作业 | 文件用途 |
| --- | ---: | --- |
| uniform1001/abaqus_uniform_baseline | 3 | C3D8 均匀实体 |
| element_comparison_20261002 | 2 | C3D8 矩阵诊断 |
| discrete_comparison_20261002 | 1 | N4 同离散对照 |
| discrete_comparison_verified_20261002 | 2 | N16 固定/自由同离散对照 |
| binary_gyroid_20261002 | 13 | C3D10 均匀/真正二值实体与加密 |
| mesh_quality_20261002 | 2 | G24 同几何体网格重定位 |

包根通常保留 INP、expected.json、mesh/参数清单；包的 `work/` 保留 ODB、DAT/MSG/STA、提取结果；包内 `scripts/` 放已审查的 Windows 执行/提取副本。相同作业名在不同包中可能对应不同网格，必须用“包目录 + 工况 + INP 哈希”识别。

Windows Abaqus 命令为 `E:\ABAQUS\2026\Commands\abaqus.bat`。ODB 提取必须使用 Abaqus Python，JAX 程序使用 WSL Pixi Python。不要在包的 work/ 下用相对 `scripts/` 路径；应使用包根的绝对脚本路径。新求解使用新作业包，不覆盖正式结果。

## Windows 阅读工作区

位置：`C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an`。

- START_HERE.md / PROJECT_OVERVIEW.md：当前阅读入口及总览，来自正式 WSL 文档。
- 根目录各阶段 report/PNG/summary/test-log/merge-receipt：阶段快照；保留路径和内容以维持旧收据哈希与图链接。
- organization_receipt_20261002.json：95 个历史文件的旧/新位置与 SHA256、两个旧总报告快照的路径。
- work/archive_20261002/：原 work/ 松散副本和一次性脚本；不作为活动程序。status_snapshots/ 保留旧的长状态报告；本次整理工具也放在此处。
- work/tpms_jax：历史 Git 传输副本，原未提交改动保留；当前科研程序只使用 WSL 正式仓库。
- outputs/：早期下载/导出包及 M4 对照快照；README_START_HERE.md 说明历史范围。已有 ZIP/解压目录不移动、不重打包。

## 复现入口

在 WSL 中恢复锁定环境并按需运行测试：

```bash
cd /home/xuehu/projects/tpms_jax
pixi install
pixi run test
```

小网格梯度验证的全批复现会重新进行 132 次差分扰动，只在需要时运行，并指定新的目录：

```bash
.pixi/envs/default/bin/python scripts/check_design_gradients.py \
  --plan validation/design_gradient_20261002/plan.json --out-dir results/NEW_gradient_check
```

输入生成与 Abaqus 流程的准确参数见对应阶段 README。本次不提供尚未实现的优化命令；R2 实现后只增加该案例必要的最小入口。
