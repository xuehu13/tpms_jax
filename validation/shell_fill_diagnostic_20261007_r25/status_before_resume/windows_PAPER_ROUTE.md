# 文献依据与实际阅读边界

2026-10-07整理：科研暂停。下文各段日期和“本轮”沿当时阅读记录解释，不是当前待办；本次未新增论文阅读。

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

本轮四篇原PDF完整路径、SHA256、页数与渲染记录见Windows `history/completed_tools_20261006/research_synthesis_20261006/literature_review.json`；正式精选收据在`docs/history/research_synthesis_20261006/`。原PDF不改，摘录和页图留阅读工具目录，不重复上传原论文。

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

旧23篇身份索引和具体阅读记录移动到`history/completed_tools_20261006/large_compression_review_20261005/`与`simulation_review_20261006/`，按旧根上下文解析；本次改写前完整来源说明保留在`history/before_research_synthesis_20261006/PAPER_ROUTE.md`。

Abaqus公开2025文档的超弹性、显式、弯曲和准静态能量说明作为既有方法参考，实际软件2026，不冒称核查其所有版本差异。本轮准静态能量网页未成功返回，使用已有读过的依据，未把网页失败称成仿真失败。JAX官方[JVP/VJP](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)与[checkpoint](https://docs.jax.dev/en/latest/gradient-checkpointing.html)、[显式示例](https://github.com/deepmodeling/jax-fem/tree/main/applications/explicit_dynamics)用于接口依据；不把工具可导当有效设计梯度认证。

## r16限定诊断新增公开方法依据

2026-10-06成功读取公开2025文档正文（实际软件2026，不认证所有版本差异）：[显式动力学理论](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm)的中央差分、稳定性、当前有效模量/频率与体积黏性；[能量平衡说明](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-ovwstatenerbal.htm)的内能包含人工能量、黏性耗散定义及总能量平衡。仅作为r16局部稳定性假设和质量指标解释，不证明本项目首次异常原因，不改变原验收门槛或要求立即细化网格。网页本次成功不改写前轮读取失败记录。

## r19动态跳跃解释：2026-10-07新增核查

原来源/读取失败按历史保留。本轮实际复读Qiu本地PDF10～13页、视觉12/13页；Thillaithevan本地PDF4～8页、视觉5页，原件SHA256与页数在`validation/dynamic_branch_review_20261007_r19/literature_record.json`。Windows阅读页图/摘录在`history/completed_tools_20261007/branch_review_20261007/`，不复制论文原件，不宣称23篇全部读完。

| 公开原始来源 | 实际读取范围与用途 |
| --- | --- |
| [Liu/Gomez/Vella，JMPS2021](https://arxiv.org/abs/2010.07850) | 作者摘要及作者HTML引言/理论设置；纯弹性动态跳跃位置可随速率变化，拱/线性坡道不能直接量化本项目差异 |
| [Abaqus不稳定/后屈曲](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-postbuckling.htm) | 不稳定响应、分岔/触发及方法范围；不自动实施Riks/扰动 |
| [局部失稳示例](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-unstablestaticplate.htm) | 动态屈曲中储能转成动能；不继承其材料/边界 |
| [准静态能量](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-qsienergybal.htm) | 本轮正文成功读取；5%～10%为准静态经验指导，不是全部动态失败线；不改写前轮网页失败 |
| [Smooth Step公式](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-t-ampsmooth.htm) | 官方索引正文公式，与既有JAX五次加载公式及原壳INP核对 |
| [有限应变壳理论](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-finitestrainshells.htm) | 官方索引转动惯量/转动黏性段；实际安装2026默认细节未逐项认证 |

官方截面柱屈曲页本轮直接读取503，未称全文已读。仅使用上表可读取来源进行判断；网络失败不等于软件计算失败。文献支持机制候选，不代替项目验证；本轮未新增仿真、参数扫描或完整AD。

## r23同背景对照审查：2026-10-07

本轮本地Hu等CMAME435（2025）117636：PDF5～7完整方法页，文本与三页渲染图复核，前两页/8页作范围核对。原件SHA256见`validation/background_control_review_20261007_r23/literature_record.json`；摘录/页图归档Windows完成工具。实体/虚域能量过渡及Newton/伴随支持方法基础，不认证当前C²公式、薄壁显式或20%路径梯度。材料论文Han仅清点原件，本轮不新增其方法阅读/结论；既有阅读范围不变。

官方2025正文已查：C3D27 Standard变量节点库、Explicit索引、VUEL内力/质量/稳定步长接口、用户单元输出限制、VUMAT超弹性旋转系/能量、VUSDFLD逐增量重置。链接集中在MECHANISM_ANALYSIS.md；本机2026的新接口未编译/运行，不把公开旧版支持当实机通过。隐式动力学候选网页本轮返回Internal Error，未使用其正文作判断。

独立同背景相近只定位实现/模型这一层，不单归因软孔隙；误差归因方案属于项目推断，需后续受控试验。无全目录阅读、参数扫描、新前向或AD声明。
