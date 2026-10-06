# 文献依据与实际阅读边界

更新2026-10-06。判断集中在[综合说明](TPMS_RESEARCH_REVIEW.md)，下一项只看[主规划](RESEARCH_PLAN.md)。这是来源索引，不维护第二份计划。

用户目录23PDF身份/哈希已建立，本次复用旧缓存，重点读五篇的相关方法/对比页；不是23篇全面精读，原件未改。阅读收据在work/simulation_review_20261006/literature_review.json，精选收据同步正式docs/history/simulation_review_20261006。

| 原作者来源 | 本次范围 | 结论用途与边界 |
| --- | --- | --- |
| [Qiu等TPMS壳/实体/体素，2024](https://doi.org/10.1016/j.ijmecsci.2023.108657) | 本地17，PDF11–13页，视觉12页图11/表5 | 薄壁弱连接及参照类型重要；二值C3D8R/塑性接触试样，非当前光滑Q2认证 |
| [Hu等几何投影非线性，2025](https://doi.org/10.1016/j.cma.2024.117636) | 本地03，复用2/3页，新读4–7页，视觉5/6页 | 能量插值式12–15核对；静力Newton/伴随，explicit topology不是显式动力学 |
| [Scherz等条件数稳定化，2024](https://doi.org/10.1002/nme.7574) | 本地18，复用2–5页，新读6–11页，视觉8页 | 局部F畸变及人工能方法；可能影响响应，不自动实施 |
| [JAX-FEM原论文，2023](https://doi.org/10.1016/j.cpc.2023.108802) | 本地05，PDF1/3/4/5页；复读方法/梯度基础 | 支持框架和材料AD，不认证本项目精度或性能速度 |
| [Wegert等非贴体自动形状微分，2025](https://doi.org/10.1016/j.cma.2025.118203) | 本地20，PDF1–4页，方法/适用范围 | 背景网格不必等于全域软填充；线弹性/Stokes示例，非20%TPMS证明 |
| [Thillaithevan等周期非线性逆设计，2024](https://doi.org/10.1007/s00158-024-03761-7) | 复用本地04第4/5/7/8页；本次官方摘要核查 | 二维平面应变、修改SVK、静力AD；方法可行性，非3D显式认证 |
| [Wang等能量插值，2014](https://doi.org/10.1016/j.cma.2014.03.021) | 作者/出版商摘要与介绍；完整原文未取得 | 固定背景软域大畸变和非线性/线性能量依据；具体式从Hu核对，不能混称原文精读 |
| [Schillinger等非线性FCM，2012](https://doi.org/10.1002/nme.3289) | 作者机构摘要/出版信息 | 高阶背景及非线性切界面困难；当前不是完整FCM |
| [Bluhm等第三介质接触](https://arxiv.org/abs/2010.14277) | 复用原作者方法/摘要 | 专门接触模型，不等于当前软孔隙 |

官方资料本次核查：[全积分弯曲](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-ctmfull.htm)、[超弹性](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-hyperelastic.htm)、[显式与数值黏性](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm)、[准静态能量](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-qsienergybal.htm)、[实体单元库](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-solidcont.htm)。公开文档2025版，实际程序Abaqus2026，不冒称核查2026所有差异。

[JAX-FEM显式示例](https://github.com/deepmodeling/jax-fem/tree/main/applications/explicit_dynamics)是弹性波基础，[JAX JVP/VJP](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)与[checkpoint](https://docs.jax.dev/en/latest/gradient-checkpointing.html)是AD工具依据；本项目完整单向JVP已执行，20%未过，多变量反向成本未验证。

用户[Style-Constrained扩散](https://doi.org/10.1002/nme.70376)与[Hybrid TPMS GAN](https://doi.org/10.1016/j.ijmecsci.2026.111353)是长期生成/逆设计动机，本次额外核查原作者预印本/出版商摘要，未重读全文、未复现网络，不将它们冒充当前薄壁前向证明。前次23PDF索引和六篇选页仍在work/large_compression_review_20261005；历史阅读记录不更新成当前任务。

本轮第2步核查[Gmsh 4.15.2官方手册](https://gmsh.info/doc/texinfo/)的离散曲面重新参数化与setPeriodic接口，实际安装版本4.15.2；核对上述Abaqus实体库的C3D10/C3D10M使用建议。没有重读23篇论文。Gmsh网格尝试失败见执行报告，软件功能说明不认证生成网格质量。
