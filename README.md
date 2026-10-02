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

- M0:环境初始化与基础验证(2026-09)
- 里程碑规划:M1 TPMS 隐式几何与连续密度场 → M2 体素有限元 → M3 自动微分与逆设计

## 数值验证记录

- [M4-A 审查修复与 10 工况](validation/m4_review/)
- [Abaqus 2026 三个均匀实体基准](validation/abaqus_uniform/)
- [C3D8 与实际 JAX HEX8 单元矩阵比较](validation/abaqus_element/)
- [同一 Gauss 材料场的线性离散对照](validation/abaqus_discrete/)
- [二值 Gyroid C3D10 贴体实体与双工况加密对照](validation/abaqus_binary/)
- [验证进度、范围和后续计划](validation/abaqus_validation_progress.md)

均匀仿射基准通过不代表 C3D8 与 JAX 的离散刚度相同；当前严格对照通过
独立积分的线性用户单元矩阵导入实现。二值 Gyroid 贴体模型完成 13 个原生
C3D10 作业：均匀基准、固定/自由横向的几何与 FE 加密；全局响应的 1%
筛查通过，局部应力和畸变单元质量仍需补充研究。完整回归测试 102/102。
