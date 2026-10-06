# diverse_04：第1步限定输入就绪，零压缩作业

科研问题：固定的0.50mm/NH/XYZ/HEX27/C²方法能否迁移到不同中面？本步只建立双方物理/离散输入，实际响应和梯度未验证。下一项按[唯一规划](../../docs/RESEARCH_PLAN.md)第2步一次双方20%前向。

- 原中面来自用户MS9/diverse_04；8372节点/15914三角面，L10mm、t0.50mm、E10MPa/ν0.3/密度10⁻⁹ tonne/mm³；无接触/塑性。研究候选明确objective_void，φ界限/η/界面/续接常数不变。
- gauss_field.npz：唯一Problem真实27点/884736点的距离占据；hrz_mass.npz：原HRZ节点/周期质量，均正、守恒；surface_geometry.npz：原中面。
- 占据积分Vf13.2519%，壳面积×厚度名义Vf13.3032%，相对−0.3854%；几何目标匹配但表示不同，不是力误差或精确三维实体认证。
- abaqus/explicit_T0p040/thin_shell.inp为规范包；XYZ423组关系/2538方程，用仿射和任意周期波动独立代入检查。原生同字节包在E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_transfer_20261006_r15_diverse04_explicit_T0p040；无ODB。
- preparation.json保持原false：4个全t末点仍在带内。band_bracket_diagnosis.json确认四条首出口→孔隙→再入带，首局部宽度0.500–0.51155mm；step1_decision.json据此给限定输入就绪。没有把旧失败改成通过，没有改变几何/数值常数；不认证全局偏置单射或无接触。

显式入口仅新增--case-input，直接物理JSON读取；旧读取输出等价，整个ExplicitXYZ类与物理推进主体相同。源/输入/环境及决定均有哈希。默认NH不改，当前20%新核AD未认证。

首次数组/INP生成完成后，报告np.bool写出失败；恢复时活动日志自哈希、最后np.bool打印先后失败，原日志/源码和false报告均留存。最终规范报告保存后只读核对，判据澄清另存decision，不重复FEM构建或修改原场。首次构建耗时/峰值未持久化，按未知报告；execution_notes.json说明边界。

tools是本步一次性生成/诊断/报告收据，不是活动FEM；不自动重跑。大数组、原日志/尝试脚本及原中面复制留本机，Git仅规范INP、关键JSON/摘要/源码。克隆不等于取得所有本机档案，位置见[地图](../../docs/FILE_MAP.md)。
