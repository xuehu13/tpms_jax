# 文献依据与实际阅读边界

更新2026-10-06。综合判断见[当前说明](TPMS_RESEARCH_REVIEW.md)，执行仅看[主规划](RESEARCH_PLAN.md)。本页只记录来源，不维护任务。用户目录共23篇PDF；本轮复读以下四篇相关方法/比较页，并核查公开原作者来源，不宣称23篇全部精读或复现。

## 本轮直接复核

| 来源 | 实际范围 | 用途与限制 |
| --- | --- | --- |
| [Xue等JAX-FEM，2023](https://arxiv.org/abs/2212.00964) | 本地PDF3/4/5页，视觉4页图1和式13；作者摘要核查 | 一般固体对照、同能量材料AD；不是TPMS20%或速度保证 |
| [Qiu等TPMS壳/实体/体素，2024](https://doi.org/10.1016/j.ijmecsci.2023.108657) | 本地PDF11/12/13页，视觉12页图11/表5 | 薄弱连接、厚度与表示风险；二值体素/接触塑性试样，非当前光滑全背景认证 |
| [Hu等几何投影非线性，2025](https://doi.org/10.1016/j.cma.2024.117636) | 本地PDF5/6/7页，视觉5页式12–15 | 虚域能量、Newton/伴随依据；explicit topology不是显式动力学 |
| [Thillaithevan等周期非线性逆设计，2024](https://doi.org/10.1007/s00158-024-03761-7) | 本地PDF4/5/7/8页，视觉5页 | 周期目标曲线、二维修改SVK/静力AD；不认证三维显式薄壁 |
| [Han等材料插值比较，2026](https://link.springer.com/article/10.1007/s00158-026-04332-8) | 出版商开放正文§3.2.1、§3.3及结论，发表2026-04-29 | 虚域插值会改变有限应变敏感性；作为风险解释，不引入其新框架 |
| [Style-Constrained扩散](https://arxiv.org/abs/2601.06469) | 本轮作者摘要；前轮阅读范围复用 | 长期生成/可微物理动机，不认证当前力学或完整AD |

本轮四篇原PDF完整路径、SHA256、页数与渲染记录见Windows `work/research_synthesis_20261006/literature_review.json`；正式精选收据在`docs/history/research_synthesis_20261006/`。原PDF不改，摘录和页图留阅读工具目录，不重复上传原论文。

## 复用的已有阅读依据

| 来源 | 已有阅读边界与用途 |
| --- | --- |
| [Wang等虚域能量，2014](https://doi.org/10.1016/j.cma.2014.03.021) | 历史作者/出版商摘要，具体经典公式从Hu核对；本轮出版商403，未取得全文。不能称精读原文 |
| [Smith等Stable-NH，2018](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf) | r10原论文公式与客观核依据；当前占据混合/C²续接是本项目候选，非该论文的TPMS认证 |
| [Scherz等稳定化，2024](https://doi.org/10.1002/nme.7574) | 历史本地PDF6–11页及视觉8页；畸变/人工能风险，不自动实施 |
| [Wegert等非贴体形状AD，2025](https://doi.org/10.1016/j.cma.2025.118203) | 历史本地PDF1–4页；线弹性/Stokes范围，不认证20% |
| [非线性有限胞，2012](https://doi.org/10.1002/nme.3289) | 历史作者机构摘要；当前HEX27不是完整FCM |
| [第三介质接触](https://arxiv.org/abs/2010.14277) | 历史方法/摘要；专门接触模型，不等于软填孔隙 |
| [Hybrid TPMS GAN](https://doi.org/10.1016/j.ijmecsci.2026.111353) | 既有出版商摘要/生成背景，不在本轮复现网络 |

旧23篇身份索引和具体阅读记录移动到`history/completed_tools_20261006/large_compression_review_20261005/`与`simulation_review_20261006/`，按旧根上下文解析；本次改写前完整来源说明保留在`history/research_synthesis_20261006/before_documents/PAPER_ROUTE.md`。

Abaqus公开2025文档的超弹性、显式、弯曲和准静态能量说明作为既有方法参考，实际软件2026，不冒称核查其所有版本差异。本轮准静态能量网页未成功返回，使用已有读过的依据，未把网页失败称成仿真失败。JAX官方[JVP/VJP](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)与[checkpoint](https://docs.jax.dev/en/latest/gradient-checkpointing.html)、[显式示例](https://github.com/deepmodeling/jax-fem/tree/main/applications/explicit_dynamics)用于接口依据；不把工具可导当有效设计梯度认证。
