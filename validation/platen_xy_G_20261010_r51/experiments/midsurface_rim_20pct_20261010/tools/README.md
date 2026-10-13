# 本次20%实验配方

run_jax_20.py复用r51/tools的模型、材料、内力及质量，只管理本次预算、保存和检查；start_abaqus_20.py用于本次原输入适配及提交。实验已完成，不自动重跑。

analyze_rim20.py从本次原始路径生成analysis/图表，修复过项目导入和ODB20%浮点端点；不改力、材料或加载。extract_rim20_frames.py只读ODB提取位移帧。

finalize_rim20.py是当次结束时的导航发布快照，后续文档已由整理批次更新；不要再次运行它覆盖当前文档。最新结果只看../README.md和../status.json。
