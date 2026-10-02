# tpms_jax

**研究目标：用体素/固定背景网格表示 TPMS，通过经验证的 JAX-FEM 力学计算，为多构型与参数学习、性能驱动的逆向设计建立完整流程。**

当前处于线弹性精度验证阶段。已有同离散和二值实体对照，当前 Gyroid 背景模型对实体参考仍有约 4.6%–4.9% 响应差；尚未实施训练与逆向设计。

- [研究问题及六阶段主规划](docs/RESEARCH_PLAN.md)：今后推进以此为准。
- [当前状态与验证缺口](docs/RESEARCH_STATUS.md)。
- [程序、数据与运行位置](docs/FILE_MAP.md)。
- [已有验证证据](validation/README.md)。
- [两篇论文的方法参考](docs/PAPER_ROUTE.md)。

## 运行环境

正式目录 `/home/xuehu/projects/tpms_jax`。Pixi 锁定环境，实际 JAX-FEM 为 PyPI 0.0.12；Abaqus 在 Windows 运行。

```bash
cd /home/xuehu/projects/tpms_jax
pixi install
pixi run test
```

最新实际完整回归 125/125，数值代码基线 `6295ee7`；本次为文档更新，没有重跑。`validation/` 存轻量证据，`results/` 存本机结果，旧原始文件保留。
