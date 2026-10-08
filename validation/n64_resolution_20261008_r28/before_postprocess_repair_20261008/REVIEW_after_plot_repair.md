# r28结果：只完成到14.7320%压缩，未完成20%/保载

本轮使用N64、HEX27/27点、同一0.5mm壁厚/材料/界面/HRZ与XYZ周期波动，加载从0.004s缩短到0.002s、保载0.0002s。用户固定总计算预算3小时；一次匹配壳参照和一次从零JAX尝试，不自动延长。未计算完整验收；仅报告实际共同窗口。

实际覆盖到压缩14.732049%、时间0.00126444608s。停止原因：Finite rollback reaches unchanged initial/16 safeguard。同速率壳观察峰12.1900%、5.340914N。JAX覆盖窗口内最大反力所在位置与数值见comparison.json；未完整时不把它认证为完整响应峰。

相对误差、实际J、dt、拒绝块、外功/总能量和高KE记录见comparison.json及response.png。壳原质量关口保持，不冒称真值。历史N32和0.004s壳仅作带明确时间标签的辅助曲线；本轮既改变空间分辨率又改变时程，不能只凭改善归因网格，不能将未完成当精度失败，也不认证设计梯度或任意构型替代。

正式位置：/home/xuehu/projects/tpms_jax/validation/n64_resolution_20261008_r28。完整/部分接受路径与检查点在full_from_zero；新壳原生INP/ODB在E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/n64_resolution_20261008_r28_shell_T0p002，轻量提取在abaqus/explicit_T0p002/results。protocol.json为预定规则，before_rate_revision保留缩短加载前协议，resource_result.json保留0.004s的初始成本测量。

六生产/环境文件、安装库、旧关键证据及原壳数据哈希保持：True。无训练、设计AD、新材料/阻尼/质量缩放/接触或GitHub提交。本次结果先用于判断能否改善复杂构型前向响应，后续只根据证据选择一项，不自动批量继续。
