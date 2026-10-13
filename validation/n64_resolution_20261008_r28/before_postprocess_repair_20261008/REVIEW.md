# r28：N64独立压缩已启动，总预算3小时

2026-10-08用户要求总计算不超过3小时，并授权调整加载时长。选0.002s加载/0.0002s保载；按初始成本估计约1.76小时，后期减步可能更久。

**同速率Abaqus壳已完成；N64从零路径已产生首批接受态，正在运行，尚无完整20%结果。** 壳原网格、材料和周期约束全部复用，只改3行时程/输出。缩短时程后壳观察峰12.1900%、5.340914N，相对0.004s壳位置变化0.3660个百分点、峰力变化2.121%；影响不大但不是零。原能量/人工能量与准静态关口未过照实保留，仍是条件性参照。

N64保持t0.5mm、L10mm、材料、界面、η、HEX27/27点、HRZ、XYZ周期波动和稳定步长保护。资源准备及存储等价已通过，真实Gauss占据/质量/dt重新计算。当前N64首批接受记录无无效NH点、无拒绝块；这里只是启动证据，不能评价完整压缩精度、接触或设计梯度。

从新壳启动起固定3h总墙钟，包含壳和N64初始化，预留120s提取。结束或预算停后，脚本自动保存实际接受窗口、comparison.json/response.png并更新本报告，不延长、不拼接、不重试。主要对照同速率壳；原N32及0.004s壳为明确时间标签的历史曲线。本轮同时改变分辨率和时程，不能将改善单独归因N64。

正式目录：/home/xuehu/projects/tpms_jax/validation/n64_resolution_20261008_r28。
实时进度：pipeline_progress.json以及full_from_zero/progress.json；真实接受路径/检查点在full_from_zero；预定规则protocol.json；预算launch_decision.json；原资源短测resource_result.json；启动核验start_verification.json。只有结束后才有本轮comparison.json。新壳原INP/ODB留在E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/n64_resolution_20261008_r28_shell_T0p002。

六生产/环境文件、安装库及旧关键证据字节保持。无新设计AD、训练、接触或GitHub提交。Windows完成的准备/传输工具已归档history/completed_tools_20261008/r28_support；正式程序仍只在WSL，原科学数据不搬迁。
