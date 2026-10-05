# 同物理薄壁HEX27候选

preparation.json说明32³二次单元、65³节点、3³Gauss点及同一中面距离带；prepare_at_run.py是实际准备输入。没有改变0.5mm厚度、界面宽度、eta、材料或XYZ加载。

T0p004_compact和T0p008_compact是两条完成的0至20%加保载路径；comparison.json、field_review.json、rate_check.json分别给出响应、模式/虚域和速率判断。三个响应指标9.45%/5.87%/6.87%为慢路径相对本次Abaqus壳诊断，不是真值误差认证。两张PNG在报告中引用。

execution_ledger.json保留资源探针及停止：probe_ready的32步不是一条完整薄壁曲线，其他失败目录不作力学成功或物理不可行证据。波检验在同轮quadratic_wave*目录，maintenance_checks.json和batching_verification.json说明维护/等价检查。

gauss_field.npz、终态场、资源日志和运行源码快照本机保留。维护入口只有根hyperelastic_fem.py与scripts/thin_target_explicit.py；此目录没有第二套FEM。全部旧输入/结果原字节冻结。
