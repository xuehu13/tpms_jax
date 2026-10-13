# 2026-10-14文档与文件状态整理

本次只检查已有实验是否有总结、更新入口和归档已完成的临时排版材料。最新科研结果仍为2026-10-13的r58，未新增FEM、AD/FD、Abaqus、训练、定时检查或Git操作，没有修改Word/PPT成品。

已有综合总结共四份，覆盖最初方法、旧差异调查、10月7—12日过程及10月13日梯度进展，具体分工集中于../../output/README.md。r51至r58均有各自报告，本次补齐正式WSL近期批次索引，并保留通过、失败和不完整范围。

`before/windows`及`before/wsl`保存入口修改前原字节。`archive_manifest_before.json`保存4个归档目录中232个文件的原路径、新路径、大小和SHA256，另记录2个运行时junction的目标。

`artifact_qa`包含原tmp下的report_20261012_qa、gradient_report_20261013_qa、ppt_update_20261013及ppt_gradient_revision_20261013，总计73,414,875字节（约70 MiB）。搬移前后逐文件核对，不删除、不合并，不遍历junction目标。

原构建源码、收据内的tmp路径保留历史含义。若要复现排版，先确认没有活动任务和同名目录，按清单把指定目录恢复到原tmp路径，核对哈希后再显式运行成品source中的构建脚本。不得从归档批量恢复科研队列，也不要自动运行其中的整理或同步脚本。

`science_before.json`及`final_receipt.json`记录本轮保护核对：Windows实验阅读副本和成品、WSL共享核心源码与r51—r58的小型源码/协议/结果/报告。WSL大型状态、检查点和ODB没有搬移或重新全量哈希。`link_checks.json`记录当前入口链接核对。

本轮维护配方为maintenance.py、archive_completed_qa.ps1及sync_and_check.py，仅在本Windows工作区操作；它们不属于科研程序入口。正式WSL只保存文档备份和维护收据，232个排版临时文件不另复制。
