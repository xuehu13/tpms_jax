# r52轻量执行与后处理配方

正式材料、内力、HRZ投影质量和显式推进复用r51及仓库共享程序；此处不建立第二套有限元框架。

- prepare_case.py只把P/D自身几何、占据端面及XY配对适配到冻结G控制卡；run_jax.py保存一次受预算限制的原计划路径。
- run_batch.py是本轮P后D的Windows协调器。D壳失败后，continue_D.py只执行原计划尚未提交的D/JAX，不重跑Abaqus、不重置原时钟。均不作为自动重试入口。
- analyze_case.py重用G完整成对后处理；extract_partial_D.py及analyze_partial_D.py专门保留D失败历史／STA，主完整20%指标留空，只计算提交前固定的局部诊断窗口。
- finalize_batch.py只汇总已有数据、更新当前导航并归档Windows编辑配方，不提交求解或修改旧科学数据。

真实命令、启动时间、停止原因、源哈希与后处理变换分别见checks中的收据。协议和README不能替代原INP／数组／路径／ODB。不得仅因工具存在而再次运行；本轮完成后追加研究需另行安排。
