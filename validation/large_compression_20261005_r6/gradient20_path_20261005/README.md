# 第4步完整厚度路径及梯度

最新事实见[执行记录](../../../docs/GRADIENT20_PROGRESS_REPORT.md)。

完整正负路径及JVP已执行；10%/15%导数核对通过，20%保载及损失导数未通过。

## 留存与复核

维护的力学入口仍是scripts/thin_target_explicit.py；JVP运行快照在path_ad_full/experiment.py，独立比较在validate_gradient.py（发布时保存）。所有输入/锁定环境/核心源码哈希在input.json及evidence_manifest.json。后者记录本机大场/日志和必要Gauss缓存的哈希；GitHub不是包含全部大数组的计算档案。

这里的q是波动位移除以10mm，vhalf是相应的半步速度，time以秒计；thickness_mm及导数的厚度单位是mm。nodem/mass为参考体积和密度乘L²的归一化集中质量，不能直接当kg读。实体坐标和输出反力仍分别以mm和N解释。

本轮文件冻结后不直接覆盖重跑；新实验选新输出目录和显式输入引用。前向可复用正式CLI的target、--element-degree 2、--cells 32、--gauss-field、--thickness-mm、--load-time .004及--adaptive；旧experiment.py是执行快照，不是另一个长期维护的FEM。

验收对象为20%保载平均力及10/15/20%三点损失。瞬时导数图只作补充诊断；这不是全部可能路径目标、高维反向或Abaqus20%独立厚度导数认证。
