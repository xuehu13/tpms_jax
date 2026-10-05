# 20%厚度总梯度：实际尝试，尚未认证

summary.json是当前解释入口。endpoint保存首个残差线搜索尝试，endpoint_energy保存复用中心平衡的能量线搜索重试；二者原始数据保持不变。probe只有成本/切线对角探针。

中心厚度0.5mm、压缩20%、HEX27 N32/XYZ成功重新平衡，伴随线性求解完成。减薄至0.4975mm未达到1e-8平衡判据，第二次最终校正方向不是能量下降方向；0.5025mm及±厚度完整加载路径未执行。候选导数不得用作已验证梯度。没有降低验收门槛或调整材料拟合。

experiment.py是能量重试实际源码的原字节留存，SHA256与endpoint_energy/input.json一致，调用正式hyperelastic_fem及thin_target_explicit；不维护第二套FEM。它只计算选定弹性端点分支，不是动态路径AD或通用已认证API。原先残差策略的实际源码在本机endpoint/source_at_run中冻结。大场、快照和日志不随Git全部上传。

实际重试命令（本机锁定环境、Gauss缓存均须存在；新输出目录）：

```sh
.pixi/envs/default/bin/python validation/large_compression_20261005_r6/gradient20/experiment.py --output /path/to/new_output --equilibrium-seed validation/large_compression_20261005_r6/gradient20/endpoint/central_field.npz --line-search energy --maxiter 4000
```

实际运行另外使用阅读工作区的资源监护和大型计算锁。上述独立实验命令本身不含监护，不能当成完整公开一键复现包。正式下一项只看docs/RESEARCH_PLAN.md的第4步，先做完整厚度扰动路径，不自动重跑这次已失败的Newton。
