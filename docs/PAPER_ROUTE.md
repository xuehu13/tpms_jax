# 文献阅读记录与方法关系

更新2026-10-04；首次核查2026-10-03；用户目录 `C:\Users\xuehu\Desktop\tpms优化`。23篇PDF身份/哈希复核，重点阅读/检索下面11篇方法和结果，另补官方文档和原始研究；不是全目录系统综述。页码为本地PDF页码，范围表示定位核查，不表示逐字精读全部内容。

解释集中在 [综合报告](TPMS_RESEARCH_REVIEW.md)相关方法与文献章节。本页是来源索引，不另立任务；文献结果不是项目验证。

| 原始研究 | 本地核查范围 | 可借鉴与不可外推之处 |
| --- | --- | --- |
| [JAX-FEM，2023](https://doi.org/10.1016/j.cpc.2023.108802) | 第3-5、7页 | 本构/隐式导数框架；不证明TPMS软孔隙精度，跨平台加速比不套用 |
| [PIMM，2021](https://doi.org/10.1007/s11837-021-04659-1) | 第3-5页；第4页视觉复核 | TPMS隐式投影/固定背景先例；单元中心密度、投影公式与我们不同 |
| [Qiu等TPMS比较，2024](https://doi.org/10.1016/j.ijmecsci.2023.108657) | 第8、10-12页§4；第12页图11/表5视觉复核 | 壳/贴体/体素优劣依赖壁厚和用途；一两层体素薄壁会失真，不支持体素普遍更快更准 |
| [TPMS Voxel Absorber，2026](https://doi.org/10.3390/ma19183904) | 第5-6、10-12页；第6页视觉复核 | 实体C3D8R、弹塑性/接触，66.7%压缩；孔隙无单元，不认证当前软孔隙 |
| [几何非线性投影，2025](https://doi.org/10.1016/j.cma.2024.117636) | 第1-4页 | 非线性、低刚度区和能量插值；项目未实现这些处理 |
| [条件数稳定化，2024](https://doi.org/10.1002/nme.7574) | 第1-4页 | 虚拟孔隙畸变的专门稳定化，不是已实现能力 |
| [非贴体/自动形状微分，2025](https://doi.org/10.1016/j.cma.2025.118203) | 第1-3、11页 | CutFEM/ghost penalty/孤立域，不是当前常规八点光滑背景 |
| [Style-Constrained，2026](https://doi.org/10.1002/nme.70376) | 第5-7页§2.2-2.3、第22-24页§3.5；第22页视觉及作者预印本互证 | 几何先验+冻结模型力学引导；3D软硬两相拉伸/剪切，不是真实孔隙TPMS压溃 |
| [Hybrid TPMS GAN，2026](https://doi.org/10.1016/j.ijmecsci.2026.111353) | 第3-4页§2.1-2.5 | 性能条件构型生成；32³是网络表示，FEM是C3D6楔形实体，未反传Abaqus |
| [Gen-Porous，2026](https://doi.org/10.1016/j.cad.2025.104020) | 第4-7页，重点第7页正文/视觉 | 已补正文：INR潜空间+神经无网格线弹性/占据；不是JAX-FEM大压缩认证 |
| [超弹性伴随+AD，2025](https://doi.org/10.1016/j.finel.2025.104440) | 第1-4页定位，灵敏度验证检索 | 伴随/AD与不同目标；专用K/Vf规则不能直接覆盖全部目标 |

补充：[有限应变能量插值（2014）](https://doi.org/10.1016/j.cma.2014.03.021)及[DTU作者摘要](https://orbit.dtu.dk/en/publications/interpolation-scheme-for-fictitious-domain-techniques-and-topolog/)、[FCM综述](https://arxiv.org/abs/1807.01285)、[JAX-FEM超弹性](https://deepmodeling.github.io/jax-fem/learn/hyperelasticity/example.html)与[梯度官方示例](https://deepmodeling.github.io/jax-fem/learn/compute_gradients/example.html)。本地 improved voxel/minimum Jacobian 论文及[作者预印本](https://arxiv.org/abs/2506.04028)仅补查摘要/方法线索，其中MJ=0.3不作为项目通用质量阈值。

Style的[作者预印本](https://arxiv.org/html/2601.06469v1)和[代码](https://github.com/CMSL-HKUST/genopt)只作方法参考，项目未运行/复现。旧DDIM/custom_vjp等来源保留在整理前快照，不新增训练工作。不同版本、几何和边界条件不能省略。

本次文件名/指纹/定位页在Windows `work/documentation_review_20261003/literature_ledger.json`，全目录身份索引仍在 `tmp/research_questions_20261003/index.json`；源PDF不改。正式副本在 `docs/history/review_20261003/review_20261003_literature.json`。页码和版本按本地文件及发行资料记录，网页抓取时间不等于发表时间。


## 2026-10-04补查：成本与有限应变问题

再次核对23篇原PDF的身份/哈希，不是全目录系统综述。补读Hu等几何非线性投影第1-4页、Thillaithevan等周期微结构非线性逆设计第4/5/7/8页、Scherz等条件数稳定化第1-4页；视觉复核各自第2/8/2页。定位/指纹在Windows `work/documentation_review_20261004/literature_ledger.json`，WSL副本在`docs/history/documentation_review_20261004/`。

新增重点：[周期微结构目标非线性响应，2024](https://doi.org/10.1007/s00158-024-03761-7)，作者[研究记录](https://research.birmingham.ac.uk/en/publications/inverse-design-of-periodic-microstructures-with-targeted-nonlinea/)及本地正文。文中二维周期单胞、修正St. Venant-Kirchhoff超弹性、SIMP/软孔隙及AD支持路线的理论可行性；第8页讨论孔隙刚度影响复核、收敛失败时插入中间应变，不能直接认证三维TPMS或解决本项目已观测的线性方程成本。本地PDF元数据有2025年生成/更新日期，出版年仍按原论文2024年记录。

Hu等能量过渡和Scherz等稳定化针对低刚度域的非线性问题，改变能量时需要独立验证。目前超时没有被归因于负J或畸变，因此不据此直接安装稳定化。Scherz论文的局部变形梯度条件数与全局切线矩阵条件数须分开。

实际安装代码核查：PETSc 3.25.1，当前GMRES/GAMG；KSP对象每次建立，缩减矩阵没有显式弹性近零模态/块信息。这些是候选原因，尚无新性能实验。新增原始文档：[GAMG](https://petsc.org/release/manualpages/PC/PCGAMG/)、[近零模态](https://petsc.org/release/manualpages/Mat/MatSetNearNullSpace/)、[CG条件](https://petsc.org/release/manualpages/KSP/KSPCG/)、[性能分析](https://petsc.org/release/manual/profiling/)、[真实残差](https://petsc.org/release/manualpages/KSP/KSPMonitorTrueResidual/)。2026-10-04网页release为3.26；启用前核对本机接口，本次不升级环境。

本次判断及其推理见综合报告第6节；这是用户薄壁纠正之前的判断，已被新主规划替换，不推导当前待办。扩散/GAN/INR仍是长期参考，训练没有启动。


## 2026-10-04薄壁目标纠正后补查

本次重读Qiu第11～13页（第12/13页视觉复核），明确薄壁一两层二值体素遗漏特征与壳的优势；它不认证当前光滑背景，也不能推出体素普遍不适用。重读Style期刊版第14～22页方法/案例，视觉复核第15页：2D超弹性为软/硬两相拉伸，2D塑性采用小应变J2，3D为软/硬两相拉伸/剪切；支持生成器与伴随连接，不证明薄壁孔隙大压缩。

核对Thillaithevan原论文：实际是修正St. Venant-Kirchhoff模型，不是此前文档误写的Neo-Hookean，已更正活动报告/文献表述，历史原字节不改。补读Ogawa等静态超弹性伴随/AD，支持通用损失需专门伴随，而非复用线性K的驻值规则。

原PDF身份和选页文本在`work/thin_tpms_feasibility_20261004/literature_reading.json`及同目录缓存；原PDF未改。官方[JAX-FEM超弹性](https://deepmodeling.github.io/jax-fem/learn/hyperelasticity/example.html)、[塑性](https://deepmodeling.github.io/jax-fem/learn/plasticity/example.html)示例是能力线索，不能代表项目已有有限应变塑性/接触认证。当前安装basis.py支持实体HEX8/HEX20/HEX27等，但本项目只维护已用HEX8，无壳单元；更高阶只是候选，不在本次自动实现。

## 2026-10-05：大压缩、Explicit与接触的针对性补充

本次核对桌面目录23份PDF的路径/哈希，重点读6篇相关章节并检查选定页面，不声称全面精读。原PDF未修改。账本/摘录在work/large_compression_review_20261005/literature_ledger.json及paper_*_selected.txt，页码为PDF页序。

| 本地编号 | 读取页 | 主线用途与限制 |
| --- | --- | --- |
| 03 几何投影非线性 | 2、3、4、5 | 虚域和非线性投影；不是薄壁/接触认证 |
| 04 周期微结构非线性逆设计 | 4、5、7、8 | 周期、隐式AD与软域处理；主要二维、修改SVK |
| 07 有限应变弹塑性 | 3、4、5、6 | 内变量/路径灵敏度；正式刊CMA449（2026）118445，DOI含2025 |
| 16 体素TPMS吸能 | 5、6、10、11、12 | Explicit/接触/塑性至约66.7%；空体素删除后实体C3D8R，非全背景软孔隙 |
| 17 体素/壳TPMS比较 | 11、12、13 | 薄壁一两层体素风险；不能直接认定本N64失败或通过 |
| 18 空域稳定化 | 2、3、4、5 | 变形梯度F条件指标；不是全局K条件数，稳定化需单独验证偏差 |

主文提供结果解释和近旁引用，不另维护第二份研究路线。新增主来源包括 [JAX-FEM Explicit](https://github.com/deepmodeling/jax-fem/tree/main/applications/explicit_dynamics)、[TMC源码](https://github.com/deepmodeling/jax-fem/blob/main/applications/third_medium_contact/example.py)、[Abaqus显式理论](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm)、[准静态能量](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-qsienergybal.htm)、[TMC自接触优化](https://link.springer.com/article/10.1007/s00466-023-02396-7)、[DiffIPC](https://huangzizhou.github.io/research/diffipc.html)。Abaqus可访问公开文档为2025版，2026实际作业设置尚未新验证。

本机官方只读checkout为9a79b4bb47460a90a6fafe26f1fbd58d2d3fed08，安装版0.0.12源码另核对。Explicit是示例式中央差分，不是当前TPMS命令选项；动态松弛人工质量不同于物理质量且周期未覆盖。官方TMC二维硬阈值/旋转历史不认证连续形态或全路径导数。[2026旋转TMC](https://doi.org/10.1016/j.cma.2026.118801)本次只取得摘要并结合源码，不声称全文阅读。本次没有运行官方示例或新增FEM。
