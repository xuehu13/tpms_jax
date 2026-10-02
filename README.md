# tpms_jax

个人科研项目：**TPMS 体素表示 → JAX-FEM 力学计算 → 形态/性能学习 → 扩散模型等生成式逆设计 → 独立力学验证。**

2026-10-03 用户明确纠正研究目标：参考无条件扩散可导力学论文与混合 TPMS 3D-cGAN 论文。此前把“四参数恒体积刚度优化”列为主线的 R1–R4 计划已撤销。

## 当前状态与入口

已有 Gyroid 平滑代理场计算、均匀与同离散 Abaqus 对照、二值整体响应及低维梯度证据。**尚未建立通用体素输入/体素 AD 链、多形态训练库或生成模型。**

- [当前目标、已有工作和 G1–G5 后续路线](docs/RESEARCH_STATUS.md)
- [两篇指定论文的方法及接口对照](docs/PAPER_ROUTE.md)
- [程序和数据路径地图](docs/FILE_MAP.md)
- [历史验证证据索引](validation/README.md)

下一步是 G1：将任意 3D 体素/连续场接入已有 FEM 并做少量必要核查，然后进行多形态小批数据试验。四参数优化不作为前置，不继续扩展单一 Gyroid 的验证扫描。

数值程序基线 `6295ee7`；最新一次实际全套回归 125/125；成功正式 Abaqus 作业 23 项（21 静力 + 2 矩阵）。此次目标纠正只改文档，没有新计算或训练。

## 环境

正式 WSL 目录 `/home/xuehu/projects/tpms_jax`。Pixi 配置/锁定见 pixi.toml / pixi.lock，实际 JAX-FEM 为 PyPI 0.0.12；外部源码目录只供参考。

```bash
cd /home/xuehu/projects/tpms_jax
pixi install
pixi run test
```

validation/ 保存正式轻量证据；results/ 为本机计算数据。现有小应变/XY 边界响应不等于完整均匀化张量或压缩吸能标签。
