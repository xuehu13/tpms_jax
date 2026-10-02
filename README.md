# tpms_jax

三维 TPMS 隐式几何 → HEX8 体素有限元(JAX-FEM)→ 自动微分 → 结构逆设计的科研项目。

## 环境

使用 [Pixi](https://pixi.prefix.dev) 管理,环境配置见 `pixi.toml`,版本锁定见 `pixi.lock`。
当前 Pixi 环境安装 PyPI `jax-fem==0.0.12`；独立的 JAX-FEM 源码目录仅供参考。
实际安装版本、关键源码指纹及 M4 验证记录见
[`validation/m4_review/`](validation/m4_review/)。

```bash
pixi install        # 按 pixi.lock 创建/恢复环境
pixi run test       # 运行环境自检与有限元 smoke test
pixi run python     # 进入项目 Python 环境
```

## 状态

- 已完成：M0 环境、M1 几何/体积分数、M2 均匀弹性/周期边界、M3 积分点材料场/横向松弛、M4 数值敏感性。
- 已完成：Abaqus 2026 均匀基准、单元算子诊断、同离散问题对照及二值 Gyroid 贴体整体响应验证；已整合 main。
- 下一阶段：网格质量与局部应力诊断、投影模型差异分解、梯度验证及小规模逆设计。
- [完整阶段报告、文件位置与后续计划](validation/project_status_20261002.md)；[21 个正式 Abaqus 作业索引](validation/project_status_20261002_inventory.json)。

## 数值验证记录

- [M4-A 审查修复与 10 工况](validation/m4_review/)
- [Abaqus 2026 三个均匀实体基准](validation/abaqus_uniform/)
- [C3D8 与实际 JAX HEX8 单元矩阵比较](validation/abaqus_element/)
- [同一 Gauss 材料场的线性离散对照](validation/abaqus_discrete/)
- [二值 Gyroid C3D10 贴体实体与双工况加密对照](validation/abaqus_binary/)
- [同几何体网格质量对整体刚度的影响](validation/abaqus_mesh_quality/)
- [投影模型 N32/N48/N64 网格验证](validation/projection_grid_20261002/)
- [验证进度、范围和后续计划](validation/abaqus_validation_progress.md)

均匀仿射基准通过不代表 C3D8 与 JAX 的离散刚度相同；当前严格对照通过
独立积分的线性用户单元矩阵导入实现。二值 Gyroid 贴体模型完成 13 个原生
C3D10 作业：均匀基准、固定/自由横向的几何与 FE 加密；全局响应的 1%
筛查通过，局部应力和畸变单元质量仍需补充研究。二值阶段完整回归测试 102/102。

后续同几何体网格重定位完成两个新作业，固定/自由反力变化分别为
0.0065%/0.0053%，体积加权应力 p99 变化低于 0.1%。本阶段完整回归
108/108。当前整体刚度研究可继续，局部最大应力仍未验证收敛。

投影 β20 工况补至 N64，固定/自由 N48→64 反力变化 0.303%/0.322%，
达到 0.5% 网格筛查。与既有二值 G48 仍有 4.64%/4.91% 的有限分辨率模型差异。
本阶段完整回归 109/109；下一步分离投影/虚拟孔隙作用，再验证梯度。
