# 实验索引

文件状态核对2026-10-14，科研结果截至10月13日。本次按用户授权同步GitHub的既有程序与小型证据，没有新增计算或队列；heartbeat、Abaqus和训练继续暂停。

## 近期批次：已有报告与实际范围

| 批次 | 报告入口 | 已完成内容及限制 |
|---|---|---|
| r51 | [G共同端缘压板](platen_xy_G_20261010_r51/README.md) | 接触支持诊断和共同端缘20%前向；原短程失败保留 |
| r52 | [P/D迁移](platen_xy_PD_20261010_r52/README.md) | P宏观目标通过；D参考壳约18.4%失败，不能评价完整20% |
| r53 | [参数梯度完整回顾](platen_xy_gradients_20261011_r53/analysis/parameter_summary/README.md) | 厚度/界面及旧形态a分目标证据；a瞬时力失败，含a完整VJP未启动 |
| r54 | [diverse_04](platen_xy_diverse04_20261012_r54/README.md) | 新加载模型20%及保载宏观目标通过 |
| r55 | [diverse_28/05/02](platen_xy_diverse_batch_20261012_r55/README.md) | 28/02未全过；05的HEX27只到19.36%预算停；另有历史Abaqus设置核查 |
| r56 | [四种HEX8](platen_xy_hex8_20261012_r56/README.md) | 四组完整20%更快，但三项目标均未全过，主线保留HEX27 |
| r57 | [真实方程系数0.5%](equation_coefficients_20261012_r57/README.md) | 三系数完整加载功VJP通过，c2瞬时力JVP通过 |
| r58 | [真实方程系数10%](equation_coefficients_10pct_20261013_r58/README.md) | 三系数完整加载功VJP通过，瞬时力未通过 |

具体数字及原失败只在各批次报告维护，当前范围见[研究状态](../docs/RESEARCH_STATUS.md)。Word等成品留本机，见[文件地图](../docs/FILE_MAP.md)。GitHub收录代表性INP；大数组、原日志与ODB留本机，复现边界见[仓库入口](../README.md#github收录和复现边界)。

## 原历史批次

| 原冻结证据 | 内容与范围 |
| --- | --- |
| [r50 G/P/D体周期成对对照](canonical_tpms_20261009_r50/README.md) | 六个20%及保载任务均完成；XYZ周期/横向宏观应变零，原对照范围冻结，不等同压板模型 |
| [r49时长逻辑核查](time_logic_audit_20261009_r49/audit.json) | 同秒单位与五次平滑加载一致，未发现隐藏时间缩放；未认证动态等价或时间离散收敛 |
| [r48壳加载时长](shell_rate_convergence_20261009_r48/README.md) | 同模型20/40/80ms峰稳定性、能量/模式、3图/4CSV；原预算与质量失败保持 |
| [r47历史图集与误差](simulation_atlas_20261009_r47/README.md) | 41份曲线/短段、22个静力点、23张图、7表Excel；完整/中断/局部/失败范围保留，零新求解 |
| [r46二值网格复盘](binary_mesh_reassessment_20261009_r46/REVIEW.md) | 同窗口/不同速率/反力惯性解释，一图与349项保护；零新FEM/Abaqus/AD，不改原关口 |
| [r45二值N64峰位](binary_n64_20261009_r45/REVIEW.md) | T1ms至13.59%预算停；无捕获峰/无N32，峰附近错位保持；0回退、全检查点与匹配壳保留，无AD |
| [r44二值完整20%](binary_full20_20261008_r44/REVIEW.md) | 从零20%+保载完整；精确预算接续，0回退，初始改善；曲线35.8966%/功40.2055%/均力47.8890%，不采纳生产/AD |
| [r43代表构型占据](binary_diverse28_20261008_r43/REVIEW.md) | 自身3896区8/12通过；同规则K降2.3435%、二值对壳1.0910%；重分布趋势重复，部分积分范围/无AD，生产保持 |
| [r42只读能量账目](initial_bias_reassessment_20261008_r42/README.md) | 既有同规则平滑/二值状态；固定权重与相容重平衡分解，原能量/恒等式通过，无新求解、非唯一主因认证 |
| [r40–r41占据与零态](binary_occupancy_20261008_r40/REVIEW.md) | r40同部分密积分K降2.1205%，对壳仍4.2430%；原27点1.4656%有积分敏感性。r41 diverse_28当前零态对匹配壳3.0033%；仅初始，未采纳生产 |
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
