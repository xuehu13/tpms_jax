# tpms_jax

三维 TPMS 隐式几何 → HEX8 体素有限元（JAX-FEM）→ 自动微分 → 结构逆设计的个人科研项目。

## 当前状态

已完成几何/前向模型、M4-A 审查、Abaqus 均匀基准、同离散方程对照、真正二值 Gyroid 整体响应、投影网格/参数检查，以及固定横向 N4/N8 一阶梯度验证。**尚未进行优化；论文新颖性尚待核查。**

数值基线：`6295ee7`。最新实际完整回归 **125/125**；正式成功 Abaqus 作业累计 **23 项（21 静力 + 2 矩阵）**。本次整理只改文档，不新增计算。历史阶段的较小测试数量不与当前数量相加。

- [研究目的、已完成/待办、限制及四阶段收口计划](docs/RESEARCH_STATUS.md)
- [程序、结果与本机路径地图](docs/FILE_MAP.md)
- [九个验证阶段的证据索引](validation/README.md)
- [当前 23 项 Abaqus 作业与整理核查记录](validation/research_audit_20261002.json)

下一步为 R1：明确论文问题和代表 N16 梯度复核；随后仅做一个固定横向、投影体积约束的四参数设计案例，再验证细网格及真实二值等体积收益。自由横向导数、USDFLD、全 XYZ 均匀化和局部应力目标列为可选扩展。

当前边界为 XY 周期和平整顶底压缩，结果是该工况的表观轴向刚度。严格同离散 Abaqus 对照采用独立积分后的线性用户单元矩阵；原生 C3D8+USDFLD 尚未验证。二值整体响应通过网格筛查，局部峰值应力仍未收敛。

## 环境与运行

使用 [Pixi](https://pixi.prefix.dev) 管理环境，配置/锁定文件为 pixi.toml / pixi.lock。实际运行库为 PyPI `jax-fem==0.0.12`；独立 JAX-FEM 源码目录仅供参考。

```bash
cd /home/xuehu/projects/tpms_jax
pixi install
pixi run test
```

正式结果见 validation/；results/ 为本机结果、不随 Git 上传。保留现有 INP/ODB 与验收证据，新计算使用新输出路径。
