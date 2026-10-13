# r51工具用途索引

本目录保留本批唯一配方。当前无活动任务；脚本存在不表示应运行。研究范围只按主规划与新授权确定，旧协议及停止线保持，原输入/结果禁止原位覆盖。

| 用途 | 文件 | 当前定位 |
| --- | --- | --- |
| 原G基线 | prepare.py、platen_model.py、run_jax.py、run_abaqus.py、extract_abaqus.py、analyze.py | 已完成2%基线，不是自动续算入口 |
| 接触位置/步长诊断 | contact_origin.py、probe_contact_support.py、analyze_contact_origin.py、analyze_contact_control.py | 本轮已结束；人工映射不作生产修正 |
| 局部实体基准 | extract_end_section.py、analyze_end_section.py、run_end_section_jax.py、analyze_end_section_jax.py | 局部静力证据，不是G动态验证 |
| 辅助壳端部尝试 | prepare_cap_arm.py、prepare_cap_rigid.py、extract_cap_arm_probe.py、extract_cap_arm_source.py、analyze_cap_arm_preparation.py | 两探针完成，三G启动停止；不追加该方案 |
| 原生壳接触筛选 | prepare_native_shell_contact.py、extract_native_shell_contact.py、analyze_native_shell_contact.py | 三探针完成、几何门槛未过，不采纳 |

未执行的G壳辅助接触提取草稿已保存在对应diagnostics/shell_cap_arm_20261010/input/unexecuted_tools。Python源文件和历史数据留原位以保持复现路径，仅可再生成的__pycache__允许清理。程序改进结论见[本批总览](../README.md)，全部过程见[研究记录](../RESEARCH_LOG.md)。
