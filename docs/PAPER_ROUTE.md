# 文献依据与阅读边界

更新：2026-10-05。解释和研究判断集中在[综合说明](TPMS_RESEARCH_REVIEW.md)，下一项只看[主规划](RESEARCH_PLAN.md)。本页是来源索引，不再并列维护历次计划。

用户目录有23份PDF，前次已建立身份/哈希和选页摘录。本次重点复读两篇完整相关章节、视觉检查其代表页，并复用已有虚域/Explicit摘录；不是23篇全面精读。原件未改。旧详细阅读记录已原字节保存到`history/before_feasibility_review_20261005/PAPER_ROUTE.md`。

## 与当前判断直接相关

| 来源 | 本次读取/复用范围 | 用途及边界 |
| --- | --- | --- |
| [Qiu等TPMS壳/实体/体素比较，2024](https://doi.org/10.1016/j.ijmecsci.2023.108657) | 本地编号17，PDF11–13页，视觉检查12页 | 薄壁一两层体素可能遗漏连接；删除空域C3D8R，与当前光滑全背景不同，不证明普遍最优 |
| [Thillaithevan等周期非线性逆设计，2024](https://doi.org/10.1007/s00158-024-03761-7) | 本地04，4/5/7/8页，视觉检查4页 | 二维平面应变、修改SVK、SIMP软域、静力AD/伴随；支持路线，非3D TPMS20%显式认证 |
| [体素TPMS吸能，2026](https://doi.org/10.3390/ma19183904) | 复用本地16第5/6页及此前10–12页记录 | Explicit、弹塑性/接触、删除空体素；不能认证当前虚域或弹性梯度 |
| [条件数稳定化，2024](https://doi.org/10.1002/nme.7574) | 复用本地18第2/3页及此前2–5页记录 | 局部F条件指标和虚域稳定化；不是全局K诊断，项目未实现 |
| [Wang等能量插值，2014](https://orbit.dtu.dk/en/publications/interpolation-scheme-for-fictitious-domain-techniques-and-topolog/) | 作者摘要及此前Hu等能量过渡记录 | 固定背景大变形虚域风险；改变能量后需独立验证，不自动加模块 |
| [Bluhm等第三介质接触，2021](https://arxiv.org/abs/2010.14277) | 原作者摘要/方法说明 | 专门接触介质，不等于把孔隙填小模量材料；项目未认证接触 |
| [Abaqus一阶全积分弯曲](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-ctmfull.htm) | 官方正文 | 剪切锁定候选机制，不能单凭本项目改善作唯一因果结论 |
| [Abaqus显式理论](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm)、[能量判断](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-qsienergybal.htm) | 官方正文；公开2025版，实际使用2026 | 集中质量/中央差分/条件稳定性；低动能结合速率检查，不替代实体精度 |
| [官方JAX-FEM显式示例](https://github.com/deepmodeling/jax-fem/tree/main/applications/explicit_dynamics) | 示例及README | 三维弹性波基础，非TPMS认证；项目已实现自身XYZ物理显式入口 |
| [JAX JVP/VJP](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)、[checkpoint](https://docs.jax.dev/en/latest/gradient-checkpointing.html) | 官方正文 | 少参数路径JVP、未来多变量反向分块重算；当前完整路径AD未实现/验证 |
| [CG](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)、[MINRES](https://petsc.org/release/manualpages/KSP/KSPMINRES/) | 官方接口条件 | SPD与对称不定区别；更换线性算法不自动保证正确平衡分支 |

## 长期方法参考，暂不执行训练

[JAX-FEM原论文](https://doi.org/10.1016/j.cpc.2023.108802)与超弹性/梯度示例支持材料核和可微求解框架；本项目精度靠自己的对照建立。[Style-Constrained](https://doi.org/10.1002/nme.70376)用于生成先验与性能引导关系，[Hybrid TPMS GAN](https://doi.org/10.1016/j.ijmecsci.2026.111353)用于构型/参数生成；既有阅读范围保留在归档。本次没有复现网络，也不把软硬两相或楔形实体训练案例当薄壁压溃认证。

本次定位与校验：[阅读收据](history/feasibility_review_20261005/literature_review.json)。前次23PDF身份/6篇选页：`work/large_compression_review_20261005/literature_ledger.json`与`paper_*_selected.txt`。正式小型收据在`docs/history/feasibility_review_20261005/`；不复制用户PDF或整套摘录缓存。
