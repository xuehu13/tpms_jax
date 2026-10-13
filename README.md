# TPMS固定背景可导力学计算研究

科研结果截至2026-10-13；仓库同步于2026-10-14。研究目标是用JAX-FEM在固定背景网格上计算薄壁TPMS的压缩响应，并把力学目标对几何设计变量的梯度传回，为后续学习与逆向设计建立基础。当前仍先验证力学和梯度，训练后置。

共同中面端缘等效压板、XY周期与自由横向宏观伸缩下，G/P/diverse_04已有完整20%压缩的限定宏观通过证据；diverse_28/02未全部通过，05和D有覆盖限制。diverse_28真实方程三系数的完整0至10%加载功VJP通过双档实际几何差分核对，最大AD/FD差0.00285%；10%瞬时力尚未通过。原失败和未执行范围保持，不能推广为任意构型、完整壁面接触或全部力梯度已验证。

## 阅读入口

- [研究背景](docs/RESEARCH_BACKGROUND.md)
- [当前状态与通过范围](docs/RESEARCH_STATUS.md)
- [方法、假定与理论](docs/TPMS_RESEARCH_REVIEW.md)
- [全部实验索引](validation/README.md)
- [文件地图与本机档案](docs/FILE_MAP.md)
- [当前安排及暂停范围](docs/RESEARCH_PLAN.md)
- [G20%压板对照](validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010/README.md)
- [三系数10%完整梯度、曲线及误差](validation/equation_coefficients_10pct_20261013_r58/analysis/README.md)

## 程序与环境

共享材料/内力、几何距离和周期约束分别在`hyperelastic_fem.py`、`surface_distance.py`、`pbc.py`及`fem.py`。近期批次的`tools/`是轻量适配；梯度复用[r53分块反向](validation/platen_xy_gradients_20261011_r53/tools/reverse.py)，r57/r58接入真实方程系数，没有另建通用求解框架。

环境由`pixi.toml`与`pixi.lock`记录，平台为Linux/WSL；JAX配置包含CUDA12。基础检查可在仓库根目录执行：

```bash
pixi install
pixi run test
```

Abaqus 2026需另外安装并具备许可，不能由上述环境安装。正式执行目录是`/home/xuehu/projects/tpms_jax`。各批次保留当时的绝对路径、来源哈希、模型协议与成本；脚本不是任意克隆位置均可直接一键运行的产品入口。

## GitHub收录和复现边界

本次补齐r47至r58的有效程序、执行/验收协议、小型原始结果与曲线、分析图表和失败记录，并保留有代表性的Abaqus输入。仓库用于阅读方法、核对已发表到仓库的数值和维护一套程序；普通克隆不等于完整实验档案。

- 小型JSON/CSV结果、参数、门槛与图通常收录；历史失败或预算停止的结论不会因整理而删除。
- 大型动态检查点、状态和原占据积分场、ODB及原求解日志留本机；重复导航备份、旧程序副本、重复INP和排版临时物不上传。具体规则见[.gitignore](.gitignore)。
- 已收录13份冻结小型输入（G/P/D及diverse中面、r53距离缓存、r57查询数组），便于查看和保留精确三角化。r53/r57仍需本机Gauss场和接触缓存；r58还依赖r57各条路径自己的初始前缀、切向和检查点。因此不能凭小型结果JSON重建完整反向过程。须恢复原输入/前缀，或按原准备程序与冻结协议重新生成并重新验证；新生成数据不冒充历史原件。
- `status.json`、PID、运行路径及成本是完成时的历史记录，不表示克隆后有活动任务；某些旧报告中的“Git暂停”“本目录为阅读副本”保留当时表述，当前正式源码与证据入口以本仓库及当前文档为准。

Word/PPT、论文、用户原始Abaqus资料及Windows阅读副本留本机，位置见[文件地图](docs/FILE_MAP.md)。没有上传私人资料或安装环境。研究计算、heartbeat、Abaqus、训练继续暂停；本次只更新GitHub，不自动恢复旧建议。
