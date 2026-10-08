# 实验证据索引

更新2026-10-08。当前事实看[研究状态](../docs/RESEARCH_STATUS.md)，下一步只看[唯一规划](../docs/RESEARCH_PLAN.md)，原因审查见[机制分析](../docs/MECHANISM_ANALYSIS.md)。本次完成r39圆柱三方诊断：两个Abaqus初始参照通过；背景原相消检查失败保留，读场审计指向舍入，仅有限判读。无新TPMS长路径/20%/AD。每份以下报告保持当时范围，历史“下一步”不自动恢复。

Git保存维护源码、精选协议/结果与必要图；大数组、原日志、完整ODB留本机。克隆不是完整实验档案；实际位置见[地图](../docs/FILE_MAP.md)。r6共同保存态桥接不是独立路径；r24参考源未编译；r25近零填充已执行但8.35%失败，有限填充未执行。

| 原冻结证据 | 内容与范围 |
| --- | --- |
| [r39简单曲面三方](curved_patch_20261008_r39/REVIEW.md) | 解析/CAX8/S3R共同中面力；实体比壳高0.6209%，背景保存解偏软3.1837%；背景原检查失败保持，非TPMS根因认证 |
| [r36–r38兼容试探与界面陡峭度](interface_width_mixed_20261008_r38/REVIEW.md) | r36仅释放0.0023715%全能；r37两新宽度有效；r38同部分密积分窄界面对壳仍5.5301%，未认证根因/未采纳生产 |
| [r35穿厚度诊断](thickness_kinematics_20261008_r35/REVIEW.md) | 无新FEM/Abaqus/AD；主体法向差异保留，界面改善而K0仍高；候选运动学/模型差别，未认证根因 |
| [r33局部积分及r34干预](local_quadrature_20261008_r33/REVIEW.md) | r33固定状态两级局部积分稳定；r34只换此区域积分，初始有效平衡但对壳高6.5014%，未改善 |
| [r32初始机制](initial_bias_mechanism_20261008_r32/REVIEW.md) | 已有场后处理，膜内主导；同中面位移迹膜能差1.79%，非因果；其后局部积分与干预见r33/r34 |
| [r30初始静力](initial_tangent_20261008_r30/REVIEW.md) | 当前N32对同中面新Standard初始刚度高5.7025%，求解/能量/PBC通过；非20%或设计AD |
| [r31膜向补片](membrane_diagnostic_20261008_r31/REVIEW.md) | 一个平直补片泊松收缩及同高斯解析一致；对真实板采样偏软6.7956%，非TPMS偏硬根因 |
| [r29初始刚度](initial_stiffness_review_20261008_r29/REVIEW.md) | 仅读取既有初始动力/内部力、拟合与旧诊断，主因和新静力未验证；后续N32 |
| [r28 N64](n64_resolution_20261008_r28/REVIEW.md) | 接受到14.7320%，观察峰改善但覆盖段误差大；未完成20%/保载，原步长底线停止 |
| [r27弯曲](plate_bending_20261008_r27/REVIEW.md) | 平板采样误差有证据，不认证TPMS峰因 |
| [r26支撑](void_support_20261007_r26/REVIEW.md) | 只到13.0777%预算停，深虚域步长瓶颈非峰因证明 |
| [r25填充](shell_fill_diagnostic_20261007_r25/REVIEW.md) | 近零填料8.3500%失败，未覆盖峰；有限填充不提交 |
| [r24暂停准备](background_probe_20261007_r24/PAUSED.md) | 仅准备参考源，未编译/运行；27节点路线暂停 |
| [r23背景审查](background_control_review_20261007_r23/REVIEW.md) | 旧桥接缺口、资料与四步提议，未提交 |
| [r22细空间模式](fine_mode_20261007_r22/REVIEW.md) | 四方向200迭代未收敛；原中断/恢复保持，无临界认证 |
| [r21受限模式](critical_mode_20261007_r21/REVIEW.md) | HRZ分类，细空间残差大 |
| [r20壳速率](shell_rate_20261007_r20/REVIEW.md) | 快壳峰11.8241%，不足解释JAX15.9409% |
| [r19跳跃分析](dynamic_branch_review_20261007_r19/REVIEW.md) | 只读跳跃能量及文献，不改旧失败 |
| [r18步长控制](step_control_20261007_r18/REVIEW.md) | 一次受控路径至17.5363%预算停，无20%/AD |
| [r17首错](mechanism_20261007_r17/REVIEW.md) | 定位及短重放，不拼完整路径 |
| [r16迁移诊断](geometry_transfer_review_20261006_r16/REVIEW.md) | 原r15收口，未令20%通过 |
| [geometry_transfer_20261006_r15](geometry_transfer_20261006_r15/README.md) | diverse_04匹配输入及一次JAX中止/壳诊断；[第2步记录](geometry_transfer_20261006_r15/STEP2.md)保留限制，未认证迁移 |
| [forward_scope_20261006_r14](forward_scope_20261006_r14/README.md) | 第4步只读收口、原中面筛选、diverse_04单候选及未来新核梯度方案；零新力学/Abaqus/AD |
| [thickness_range_20261006_r13](thickness_range_20261006_r13/README.md) | 固定C²候选，0.45/0.55mm匹配壳与完整20%，0.50mm复用；反力/曲线/功、峰值/模式与27/125点见decision |
| [void_continuation_20261006_r12](void_continuation_20261006_r12/README.md) | C²混合虚域、45项核/回归、原6点局部AD/FD、快慢20%及新场125点；完整范围判断看decision |
| [objective_path_20261006_r11](objective_path_20261006_r11/README.md) | 一次20%客观虚域路径、35项检查；6.81%反力差，125点6个混合尾部材料域失败，未采纳默认 |
| [objective_virtual_20261006_r10](objective_virtual_20261006_r10/README.md) | 客观虚域纯核与原场验证通过；28项检查，无新路径/AD，下一项只看主规划 |
| [virtual_kernel_20261006_r9](virtual_kernel_20261006_r9/README.md) | 经典虚域核因转动缺陷未采纳，原科学证据冻结 |
| [jax_improvement_20261006_r8](../docs/JAX_INTEGRATION_PROGRESS.md) | 64点完整20%与125点场检查完成，积分候选未采纳 |
| [simulation_error_20261006_r7](simulation_error_20261006_r7/README.md) | 冻结旧顺序第1步误差定位完成，第2步恒厚边界就绪、体网格未过；无新Abaqus/FEM前向，第3/4步未执行 |
| [large_compression_20261005_r6](large_compression_20261005_r6/README.md) | 冻结20%壳诊断、静力阻塞、HEX8显式基准；原生保存状态桥接及HEX27改善，响应/速率工作目标通过；完整厚度JVP已执行，10%/15%通过、20%未过 |
| [thin_target_20261004_r5/step4_thickness_a01](thin_target_20261004_r5/step4_thickness_a01/README.md) | 1%/5%反力厚度总导数及两点损失；2伴随/6扰动/3停止，2项1%壳厚分析；157维护；5%独立壳导数未算，无训练/形态导数 |
| [thin_target_20261004_r5/step3_xyz](thin_target_20261004_r5/step3_xyz/README.md) | 5%薄壁XYZ有限应变；1%/5%初筛，10%未接受；3完成状态/4停止、4壳分析，155维护；无20%/梯度/训练 |
| [thin_target_20261004_r5/step2](thin_target_20261004_r5/step2/README.md) | 5%薄壁小变形；原平端未过，XYZ诊断差4.979%/模式一致初筛通过；2次JAX/2次壳；原记录冻结，用户后续已统一XYZ |
| [thin_target_20261004_r5](thin_target_20261004_r5/README.md) | 第1步恒厚距离带/真实Gauss/连接初筛，原记录冻结 |
| [mechanics_trust_20261003_r4/step4](mechanics_trust_20261003_r4/step4/README.md) | Gyroid预检查至5%，N48在0.5%点按资源停止；3条路径尝试/26状态、零新增Abaqus/训练/设计梯度，148项回归；有限应变精度未认证 |
| [mechanics_trust_20261003_r4/step3](mechanics_trust_20261003_r4/step3/README.md) | 第四轮完整均匀实体有限应变基准通过至20%；3条JAX路径、2项分析及2项datacheck，144项回归通过；不认证TPMS大压缩 |
| [mechanics_trust_20261003_r4/step2](mechanics_trust_20261003_r4/step2/README.md) | 第四轮新构型/薄壁筛查：Primitive整体响应通过，薄壁细参考未成立；2前向、2分析、2datacheck；最终138项回归通过 |
| [mechanics_trust_20261003_r4/step1](mechanics_trust_20261003_r4/step1/README.md) | 第四轮参考审计：14个旧包、实际INP/拓扑/映射检查通过；零新增求解；原始记录冻结 |
| [learning_bridge_20261003_r3](learning_bridge_20261003_r3/README.md) | N64专用梯度通过；新壁宽示例性能跨度不足／G48网格审查失败，第3/4步未启动 |
| [geometry_interface_20261003_r2](geometry_interface_20261003_r2/README.md) | 隐式输入三锚点与小／中网格梯度，通过与资源停止证据冻结 |
| [near_term_20261003](near_term_20261003/README.md) | 第一轮四步：解析前向通过；数组保真未达标并停止；全链导数仅作诊断通过 |
| [m4_review](m4_review/README.md) | M4-A 修复及 10 项固定横向参数数据 |
| [abaqus_uniform](abaqus_uniform/README.md) | 3 项原生 C3D8 解析基准 |
| [abaqus_element](abaqus_element/README.md) | 2 项 C3D8 单元矩阵诊断 |
| [abaqus_discrete](abaqus_discrete/README.md) | 3 项独立积分线性用户单元矩阵对照 |
| [abaqus_binary](abaqus_binary/README.md) | 13 项 C3D10，含 2 项均匀基准 |
| [abaqus_mesh_quality](abaqus_mesh_quality/README.md) | 2 项新作业及 6 个旧 ODB 诊断 |
| [projection_grid_20261002](projection_grid_20261002/README.md) | N48/N64 背景响应与求解路径核对 |
| [projection_effects_20261002](projection_effects_20261002/README.md) | 7 项 β/E_min/体积分数影响计算 |
| [design_gradient_20261002](design_gradient_20261002/README.md) | 3 组 N4/N8 固定横向参数梯度，132 次差分扰动 |
