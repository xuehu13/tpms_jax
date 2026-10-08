# 文献依据与实际阅读范围

更新2026-10-08：初始原因复盘后已执行至r39；本次新增平滑Heaviside方法的出版商摘要/引言检索，未重读其余全部论文。用户论文目录递归清点23篇；选择直接相关的方法/验证页，未宣称全部通读。文献支持机制与判别设计，不证明本项目根因。当前完整判断见[机制审查](MECHANISM_ANALYSIS.md)，任务只见[唯一规划](RESEARCH_PLAN.md)。

## 本次实际读取

| 原始来源 | 范围 | 对本项目的作用与限制 |
| --- | --- | --- |
| [Qiu等，IJMS2024，108657](https://doi.org/10.1016/j.ijmecsci.2023.108657) | 本地PDF1、11–13页文字，12页图11/表5及正文视觉复核；出版商摘要 | 二值薄壁可能丢连接并偏软，网格策略依赖构型/厚度；塑性/接触试样不是我们的连续背景认证 |
| [Hu等，CMAME2025，117636](https://doi.org/10.1016/j.cma.2024.117636) | 本地PDF1、5–7页文字，5页式11–21视觉复核 | 固定网格、平滑投影、虚域能量、Newton/伴随基础；非当前C²显式或20%AD认证 |
| [Wang等2025体素建模预印本](https://arxiv.org/abs/2506.04028) | 本地PDF1–3页文字，作者摘要核查 | 尺寸/网格几何质量联合控制；minimum Jacobian不能作我们detF的0.3阈值，不移植其收敛结论 |
| [Jiang等，Materials & Design2021，109655](https://doi.org/10.1016/j.matdes.2021.109655) | 本地PDF1–3页文字 | 体素设计/多尺度背景，未深入后续算例，不据此解释当前5% |
| [Rank等Shell FCM2011](https://doi.org/10.1016/j.cma.2011.06.005) | 出版商检索返回的摘要、引言和方法片段；直接正文403 | 薄壁相对笛卡尔胞过细时的困难；面参数域几何映射方案，不等价于当前Q2场 |
| [Rank等高阶实体/壳2005](https://doi.org/10.1016/j.cma.2004.07.042) | 出版商检索摘要/引言，直接正文403 | 区分实体离散与壳模型误差；无全文/项目精度认证 |
| [Schillinger等FCM综述2018](https://arxiv.org/abs/1807.01285) | 作者摘要 | 高阶位移/几何积分相互区分；当前并非完整FCM |
| [Abaqus壳元素说明](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-shellelem.htm) | “Thick/Thin”、有限应变、厚度应力、S3/S3R段落 | S3R允许横向剪切、常规壳厚度零应力假定；公开2025，实机2026，不认证所有版本差异 |
| [Gfrerer Trace-Finite-Cell，2021](https://link.springer.com/article/10.1007/s00466-020-01956-5) | 开放正文摘要、§1、§2.1 | 隐式曲面背景壳/C¹样条可行的线性方法；只作备选，未实施非线性/AD |
| [Xia等，CMAME2012，材料边界平滑敏感性](https://doi.org/10.1016/j.cma.2012.06.005) | 出版商检索摘要/引言，直接全文403 | 固定网格上平滑材料边界支持设计敏感性；非当前TPMS/显式/20%AD认证 |
| [Abaqus轴对称三维简化](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-dimension.htm) | 官方轴对称段及CAX8库检索片段 | 保留环向应变；用于r39轴对称三维实体，非平面应力或完整C3D网格 |
| [JAX-FEM原论文](https://arxiv.org/abs/2212.00964) | 本次作者摘要，复用旧方法页证据 | 一般固体与AD基础，不认证当前薄壁20% |

本地四篇路径、原件SHA256、页数和读取范围记录在正式 `docs/history/research_reassessment_20261008/local_reading_record.json`。本次未改/上传原PDF，少量摘录和阅读页图留Windows完成工具。Cornell薄壁会议合辑只用于检索线索，不作当前方法结论依据。

## 继承的历史依据，不冒称本次重读

先前来源明细原文已归档，下面保留稳定索引：

- [非线性周期逆设计，2024](https://doi.org/10.1007/s00158-024-03761-7)：本地4/5/7/8页；二维修改SVK、静力AD，非三维显式认证。
- [Han材料插值比较，2026](https://doi.org/10.1007/s00158-026-04332-8)：历史出版商§3.2.1、§3.3及结论；虚域影响有限应变敏感性，非当前零态根因。
- [Kudela等积分，2015](https://doi.org/10.1186/s40323-015-0031-y)、[体素FCM2020](https://doi.org/10.1007/s00419-020-01719-x)：历史背景/积分部分；支持独立检查积分，不要求反复加点。
- [Stable-NH，2018](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf)、[Wang虚域2014](https://doi.org/10.1016/j.cma.2014.03.021)：既有核/客观性基础，经典虚域具体式从Hu核对；Wang全文未取得。
- [非线性FCM2012](https://doi.org/10.1002/nme.3289)、[稳定化2024](https://doi.org/10.1002/nme.7574)、[非贴体形状AD2025](https://doi.org/10.1016/j.cma.2025.118203)：既有特定范围，不自动引入新框架。
- [Liu/Gomez/Vella动态跳跃2021](https://arxiv.org/abs/2010.07850)：历史作者摘要/引言/理论；速率可移动动态跳跃，不能直接量化当前TPMS峰错位。
- [第三介质接触](https://arxiv.org/abs/2010.14277)：专门接触方法，不等于低模量孔隙。
- [Style-Constrained扩散](https://arxiv.org/abs/2601.06469)、[Hybrid TPMS GAN](https://doi.org/10.1016/j.ijmecsci.2026.111353)：长期设计背景，训练后置，不作前向认证。

逐轮文献记录仍在原validation目录；2026-10-06详细身份索引在正式 `docs/history/research_synthesis_20261006/` 与Windows `history/completed_tools_20261006/`。失败网页/未读页保持原范围，不用成功检索覆盖原失败。
