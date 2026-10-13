# r53：薄壁TPMS完整路径梯度证据

更新2026-10-12。**本批已暂停，当前无活动任务，每30分钟自动检查已停止。**

详细数据与分析只维护一份：[梯度研究完整回顾、结果与判断](analysis/parameter_summary/README.md)。其中包含早期伴随／数组映射／薄壁梯度、当前17项核对、全部失败、8张有用图、参数单位／敏感度／成本和后续条件性建议。

厚度t在G5/约10/约12及P5的完整功／力梯度通过；G界面功在5/约10/约12通过，12%末端力未通过。一个固定端缘形态坐标a的G5加载功通过，补充核对差0.004001%；末端力9.54232%、FD减半14.9678%，仍未通过。G5实际完整反向仅覆盖t/w，包含a的完整VJP没有启动。原四步没有全部完成。

| 查阅内容 | 位置 |
|---|---|
| 完整总表／图／分析 | [analysis/parameter_summary](analysis/parameter_summary/README.md) |
| G5厚度原核对 | [analysis/README.md](analysis/README.md) |
| G5实际功反向t/w | AD/reverse_work/result.json（正式原件）；[反向说明](AD/reverse_work/README.md) |
| G5宽度功 | [analysis/interface5](analysis/interface5/README.md) |
| G10／G12厚度 | [G10](analysis/G10_thickness/README.md)、[G12](analysis/G12_thickness/README.md) |
| G10／G12宽度 | [G10](analysis/G10_interface/README.md)、[G12](analysis/G12_interface/README.md) |
| P5厚度 | [P5](analysis/P5_thickness/README.md) |
| 原形态与一次补充 | [原结果](analysis/G5_shape/README.md)、[补充结果](analysis/G5_shape_refined/README.md) |
| 已有状态／几何定位 | [保存态诊断](analysis/G5_shape_refined_diagnostic/README.md)、[几何诊断](analysis/G5_shape_geometry_diagnostic/README.md) |
| 实际进度／原协议／源码 | status.json、input/、tools/（正式WSL原件为准） |

正式位置`/home/xuehu/projects/tpms_jax/validation/platen_xy_gradients_20261011_r53`；Windows为阅读副本。原输入、源码、路径结果、大数组和失败保持。本次只做整理和已有数据绘图，无求解／AD／FD／Abaqus／训练／Git提交；旧报告的下一步不自动执行。
