# tpms_jax

**当前任务：检验体素/固定背景网格方法下，JAX-FEM 能否准确计算 TPMS，并评估能否在用户需要的响应范围内替代此前的 Abaqus 壳单元计算。**

后续扩散模型/JAX 学习构型和参数是用途；当前先完成计算精度判断。异质材料研究、批量训练样本数及四参数优化不属于当前任务。

- [当前范围、原 B1–B4 规划与实际证据](docs/RESEARCH_STATUS.md)
- [程序与结果文件地图](docs/FILE_MAP.md)
- [历史验证索引](validation/README.md)
- [后续论文方法参考](docs/PAPER_ROUTE.md)

已有 Gyroid 背景模型、均匀基准、同离散对照及二值实体对照。B2 尚未找到 Abaqus 正式记录；B3 的 USDFLD 路径未完成，实际用了独立线性单元矩阵。现有二值对照仍有约 4.6%–4.9% 有限分辨率响应差，不能据此宣布原壳模型可替代。

下一步先核对原壳算例的几何、厚度、材料、加载和响应范围，匹配所比较的物理问题；按精度判断的需要补 B1–B4 缺口。此前 R1–R4/G1–G5 自动推进计划已撤销。

数值基线 `6295ee7`；最新实际完整回归 125/125，正式成功 Abaqus 作业 23 项。本次仅文档校正。

## 环境

正式 WSL 目录 `/home/xuehu/projects/tpms_jax`，Pixi 环境由 pixi.toml / pixi.lock 锁定，实际运行库为 PyPI jax-fem 0.0.12。

```bash
cd /home/xuehu/projects/tpms_jax
pixi install
pixi run test
```

validation/ 是正式轻量证据；results/ 为本机结果。已有数据和 Abaqus 原始文件保留。
