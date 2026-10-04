# tpms_jax

可信的 TPMS 体素/固定背景压缩计算与有效梯度，服务构型/参数学习、生成及性能驱动逆设计。

先读 [综合报告](docs/TPMS_RESEARCH_REVIEW.md)，可分享 [PDF](docs/TPMS_RESEARCH_REVIEW.pdf)。继续科研按 [背景](docs/RESEARCH_BACKGROUND.md) → [唯一近期主规划](docs/RESEARCH_PLAN.md) → [状态](docs/RESEARCH_STATUS.md) → [文件地图](docs/FILE_MAP.md)。[文献记录](docs/PAPER_ROUTE.md)、[验证索引](validation/README.md)保留来源与证据。

当前有限Gyroid小变形整体刚度及专用导数已有证据。前三轮收口，旧停止项不自动续跑。第四轮[第1步](docs/ROUND4_STEP1_REPORT.md)参考审计完成；[第2步](docs/ROUND4_STEP2_REPORT.md)Primitive整体响应工作筛查通过，薄壁Gyroid G48裁剪参考未成立、未进入力学计算。下一步是原第3步完整均匀实体有限应变基准，尚未启动。大压缩/整个构型族/网络反传尚未认证，训练和逆设计继续后置。综合报告是第四轮制定前的快照，最新状态以主规划为准。

正式路径 `/home/xuehu/projects/tpms_jax`；Pixi锁定环境，JAX-FEM PyPI 0.0.12，Abaqus在Windows。数值HEAD `5e1d03d`，阶段改动未提交。第2步最小扩展8个输入/几何/提取/测试文件，最终138项回归通过；原FEM/密度/PBC/梯度求解器及依赖环境不变。

```bash
cd /home/xuehu/projects/tpms_jax
pixi install
pixi run test
```

正式数值模块/测试在项目根及scripts/tests；冻结输入/结果/源码/日志在validation，M4 CSV在results，大型Abaqus包在E盘。历史快照不是第二套维护程序。
