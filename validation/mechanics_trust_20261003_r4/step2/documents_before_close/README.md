# tpms_jax

可信的 TPMS 体素/固定背景压缩计算与有效梯度，服务构型/参数学习、生成及性能驱动逆设计。

先读 [综合报告](docs/TPMS_RESEARCH_REVIEW.md)，可分享 [PDF](docs/TPMS_RESEARCH_REVIEW.pdf)。继续科研按 [背景](docs/RESEARCH_BACKGROUND.md) → [唯一近期主规划](docs/RESEARCH_PLAN.md) → [状态](docs/RESEARCH_STATUS.md) → [文件地图](docs/FILE_MAP.md)。[文献记录](docs/PAPER_ROUTE.md)、[验证索引](validation/README.md)保留来源与证据。

当前小变形 Gyroid 整体刚度及专用导数已有有限范围证据；大压缩/多构型/网络反传/训练尚未认证。前三轮实验收口，旧停止项不自动续跑。第四轮[第1步参考审计](docs/ROUND4_STEP1_REPORT.md)完成，现有参考可在整体响应限制下复用；第2步尚未启动。训练和逆设计继续后置。综合报告为规划制定前的总结快照，最新执行状态以主规划为准。本次新增审计证据，没有新增有限元求解或数值源程序改动。

正式路径 `/home/xuehu/projects/tpms_jax`；Pixi锁定环境，JAX-FEM PyPI 0.0.12，Abaqus在Windows。数值HEAD `5e1d03d`，阶段改动未提交。原130回归与新增3检查分别通过，本次不重跑测试或更改环境。

```bash
cd /home/xuehu/projects/tpms_jax
pixi install
pixi run test
```

正式数值模块/测试在项目根及scripts/tests；冻结输入/结果/源码/日志在validation，M4 CSV在results，大型Abaqus包在E盘。历史快照不是第二套维护程序。
