"""Summarize the completed physical range without changing frozen evidence."""
from pathlib import Path
import hashlib, json, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/thickness_range_20261006_r13'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert not (O/'decision.json').exists()
baseline=R/'validation/void_continuation_20261006_r12'
metrics={.45:read(O/'t0p45/analysis/comparison.json'),.5:read(baseline/'T0p004_analysis/comparison.json'),
         .55:read(O/'t0p55/analysis/comparison.json')}
for t,m in metrics.items():assert m['candidate_completed_20']
new=[metrics[.45],metrics[.55]]
passed=all(m['scoped_engineering_response_pass'] for m in new)
decision={'main_plan_step':3,'completed_matched_thicknesses_mm':[.45,.55],'reused_thickness_mm':.5,
          'fixed_candidate':'objective_void+C2, HEX27 N32/27 points, HRZ, XYZ periodic zero lateral strain',
          'all_new_cases_meet_scoped_engineering_targets':passed,
          'maximum_hold_force_difference':max(m['terminal_force_difference_vs_shell'] for m in metrics.values()),
          'maximum_curve_RMS_over_fixed_Standard_peak':max(m['curve_RMS_over_Standard_peak'] for m in metrics.values()),
          'maximum_work_difference':max(m['input_work_difference_vs_shell'] for m in metrics.values()),
          'all_new_shell_quality_gates_pass':all(m['reference_quality_certified'] for m in new),
          'new_JAX_paths':2,'new_Abaqus_jobs':2,'new_full_AD_jobs':0,'new_training_jobs':0,
          'solver_changed':False,'numerical_constants_retuned':False,
          'thickness_range_is_derivative_certificate':False,'gradient20_certified':False,
          'scope':'Same diverse_28 midsurface, t=.45/.50/.55mm, L10mm, NH E10 nu.3, no contact/plasticity, XYZ 0-20%',
          'actual_virtual_folds_retained':True,'everywhere_in_time_domain_certified':False,
          'next_action':'Main plan step 4: consolidate scope and choose a bounded next test; no broad sweep or training.'}
write(O/'decision.json',decision)

fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,(t,m) in zip(axes,metrics.items()):
    if t==.5:
        j=read(baseline/'T0p004/result.json');s=read(R/'validation/large_compression_20261005_r6/abaqus/explicit_T0p040/shell.json')
    else:
        tag='t0p45' if t==.45 else 't0p55';j=read(O/tag/'T0p004/result.json');s=read(O/tag/'abaqus/explicit_T0p040/shell.json')
    for rows,name,style in [(j['path'],'JAX fixed candidate','-'),(s['force_path'],'Matched shell','--')]:
        aa=np.array([r['compression'] for r in rows]);ff=-np.array([r['Fz_N'] for r in rows]);x,ids=np.unique(aa,return_index=True)
        ax.plot(x*100,ff[ids],style,label=name,lw=1.5)
    ax.set(xlabel='Compression (%)',ylabel='Reaction magnitude (N)',title=f't = {t:.2f} mm')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    ax.text(.05,.05,f'Hold {100*m["terminal_force_difference_vs_shell"]:.2f}%\nCurve {100*m["curve_RMS_over_Standard_peak"]:.2f}%\nWork {100*m["input_work_difference_vs_shell"]:.2f}%',transform=ax.transAxes,fontsize=8)
fig.savefig(O/'response.png',dpi=180);plt.close(fig)

table='| 厚度 | JAX保载力幅值 | 壳保载力幅值 | 反力差 | 曲线指标 | 输入功差 | JAX主体时间 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
for t,m in metrics.items():
    shell_force=m.get('shell_hold_mean_Fz_N')
    if shell_force is None:shell_force=read(R/'validation/large_compression_20261005_r6/abaqus/explicit_T0p040/shell.json')['hold_mean_Fz_N']
    table+=f'| {t:.2f}mm | {abs(m["hold_mean_Fz_N"]):.6f}N | {abs(shell_force):.6f}N | {100*m["terminal_force_difference_vs_shell"]:.2f}% | {100*m["curve_RMS_over_Standard_peak"]:.2f}% | {100*m["input_work_difference_vs_shell"]:.2f}% | {m["body_seconds"]/60:.2f}min |\n'
quality='| 厚度 | JAX能量/功缺口 | 加载时长KE/U≤5% | 末态KE/U | 125点严格NH最小J | 125点虚域非正J数量 | 壳能量漂移 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
for t,m in metrics.items():
    drift=m.get('shell_energy_drift',read(R/'validation/large_compression_20261005_r6/abaqus/explicit_T0p040/shell.json')['global_energy_drift_relative_work'])
    quality+=f'| {t:.2f}mm | {100*m["energy_work_gap"]:.4f}% | {100*m["loading_time_fraction_KE_below_5pct"]:.2f}% | {100*m["terminal_KE_over_U"]:.6f}% | {m["dense_sampling_required_positive_J_min"]:.5f} | {m["dense_sampling_nonpositive_points"]} | {100*drift:.2f}% |\n'
peak='| 厚度 | JAX峰值 / 位置 | 壳峰值 / 位置 | 峰值差 | 中面波动向量余弦 | 中面波动相对差 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
for t,m in metrics.items():
    if t==.5:
        s=read(R/'validation/large_compression_20261005_r6/abaqus/explicit_T0p040/shell.json');xs=np.array([r['compression'] for r in s['force_path']]);ys=-np.array([r['Fz_N'] for r in s['force_path']]);sp=float(ys.max());sc=float(xs[ys.argmax()])
    else:sp=m['shell_peak_magnitude_N'];sc=m['shell_peak_compression']
    mode=m['mode']['shell']
    peak+=f'| {t:.2f}mm | {m["peak_magnitude_N"]:.5f}N / {100*m["peak_compression"]:.2f}% | {sp:.5f}N / {100*sc:.2f}% | {100*abs(m["peak_magnitude_N"]/sp-1):.2f}% | {mode["fluctuation_vector_cosine"]:.6f} | {100*mode["area_weighted_relative_fluctuation_difference"]:.2f}% |\n'
outcome=('两个新增厚度均通过固定的前向工作门槛，说明改善不是只在0.50mm这一点成立。' if passed else '有限厚度检验已执行，但至少一个新增厚度未通过原工作门槛；不能把代表点成功扩大为整个范围成功。')
scope=('同一中面在0.45至0.55mm附近的这三个采样厚度、0至20%弹性周期无接触响应，有条件地支持作为Abaqus的前向替代工具；不是连续整个区间逐点认证。' if passed else '目前只能保留已通过工况的前向范围，失效项见各工况comparison.json；不改η、界面或厚度拟合参照。')
report=f'''# 匹配厚度范围：固定方法的薄壁20%前向检验

2026-10-06，唯一主规划第3步。{outcome} {scope}

## 科研问题与输入

这轮回答固定背景方法在相邻实际薄壁厚度下是否仍有效，服务将来构型/参数探索，而非调整厚度来拟合原曲线。diverse_28同一周期中面、单胞边长L=10mm；厚度0.45/0.50/0.55mm分别为边长4.5%/5%/5.5%。0.50mm复用r12，两个新厚度同步改变壳截面和JAX距离带占据、HRZ质量。

基材仍均匀可压缩Neo-Hookean，E=10MPa、泊松比ν=0.3。软孔隙是几何/数值近似，不是异质材料。JAX固定HEX27 N32、27点、HRZ、objective_void+C²续接；η=1e−4、界面10%-90%宽0.05mm、φ界限0.001/0.01、最大续接截点0.1均不变。默认NH未改，无网格、积分、人工刚度或阻尼扫描。

Abaqus复用9351节点、17986个S3R单元、366组原XYZ周期关系和原中面；每份INP只改一行壳厚，逆替换能精确恢复原始字节。仍为三向周期波动、宏观横向应变零，压缩20%；不是压板试样。JAX加载0.004s，壳慢Explicit加载0.040s，均另保载10%时长；不同加载时间是既定准静态选择，不能称时间历史完全相同。新厚度速率独立加倍未执行，不继承0.50mm的速率认证。

## 响应与工作门槛

{table}
压缩力原始符号为负，表报幅值。保载力取末半段平均；反力差为|JAX−壳|/|壳|。曲线指标在0.1%-20%取1000个共同压缩点，力差均方根除以冻结Standard峰值2.258914N；它不是逐点最大相对误差。功为宏观反力沿位移的积分，单位N·mm。三项≤10%是本研究工作目标，不是物理精度认证。0.50mm表使用既有快路径，与两点相同加载时长；其既有慢路径反力/曲线/功差6.85%/5.05%/5.98%，速率证据保持。

两侧随真实厚度增加的力级与峰值变化一致，支持方法能表现厚度引起的响应变化。反力差从约6.8%上升至约8.3%，不能把它当作所有厚度/构型的统一校正倍率。曲线指标使用同一个固定分母，较厚工况力级较大也会影响该指标；不能据三点单独定位误差原因或认证局部设计导数。

![同一方法与各自匹配壳](output/figures/THICKNESS_RANGE_response.png)

{peak}
峰值位置为宏观压缩量。模式在原中面节点用安装版HEX27插值，移除宏观仿射运动与刚体平移，再按原三角形参考面积加权。余弦接近1说明运动方向相近，相对差衡量波动幅度/分布偏离；它们不认证局部应力或接触。

## 材料域、动力学与参照限制

{quality}
J=detF为实际局部体积比。严格NH区指φ≥0.01、不能使用负J延拓的区域；新工况材料求值数与27/125点完整能量、应力检查见原JSON。125点是同一最终保存位移场的密采样，共409.6万个点，不是额外求解或全过程处处有效证明。人工虚域非正J如实保留，完整候选能量不是只对正J子域求和。

KE/U是动能与弹性能之比，从1%压缩起按实际加载时长加权；末态与加载段分别报告。能量/功缺口<1%、末态KE/U<5%、加载时长中KE/U≤5%占比≥95%是原JAX门槛。新增点各自估算稳定时间步；其步长如相同也是计算结果，不是强制沿用。

两条新JAX路径均零减步，主体约10.7分钟；最终场分析各约23-26秒。两份原生壳Explicit主体989/984秒，即16.48/16.40分钟，均成功完成。壳各4CPU并行、JAX单GPU顺序运行；加载时间、单元和工作量不同，这些是本轮成本记录，不能当公平的软件速度比较。

壳自身仍按原1%全局能量漂移门槛评价，是否通过见shell_checks，不能放宽门槛将参照问题转成JAX通过。即便主响应接近，壳/背景运动学不同，缺少独立实体真值；结论是有条件工程一致性。实际自接触、周期镜像接触和塑性压溃均未认证，软孔隙不代替接触。

## 对长期目标和其他参数的判断

{scope} 当前可行性的积极证据是能够持续计算薄壁较大压缩、对匹配壳比较完整曲线并保留可微材料接口。不能承诺全面替代Abaqus。

曲面参数、构型和材料当然可以逐步扩大验证范围；这正是后续学习/逆设计需要的能力。应选择有区分力的变化，先查已有可用周期中面与参照，固定真实壁厚和算法；不预设四参数或大量样本。

| 后续候选因素 | 回答的问题与前提 |
| --- | --- |
| 同族曲面参数 | 曲率、薄弱连接变化后方法是否仍可信；需要新中面/距离场和匹配壳，解析阈值c不能当恒定壁厚 |
| 一个不同构型 | 弯曲/连通模式改变后能否迁移；优先已有可用周期壳中面，沿用已固定方法，不先开发复杂实体网格 |
| ν或另一弹性本构 | 体积/剪切比例或非线性材料不同后是否适用；双方真实实体规律必须匹配，虚域能量与稳定性另行验证 |
| E单项变化 | 同ν的NH静力能量/力有同比例缩放成分，证明力级可变不等于验证了新变形机制；显式还需考虑惯性及稳定步长 |

这是下一阶段的候选顺序与依据，不是本轮新增扫描清单。JAX-FEM论文支持可微材料和逆问题的基础能力，[原论文](https://arxiv.org/abs/2212.00964)，但没有替当前三维薄壁显式20%路径或任意本构完成认证。现有距离/最近面片预处理未认证形态AD，换构型算出力不等于已获得形态梯度。

新核局部/短块AD通过，但完整20%路径导数未认证；原NH20%JVP与FD符号相反的记录保留。本轮±10%厚度物理变化不是梯度检验，零新完整AD和训练。按唯一主规划第4步收口范围，再选择最小下一项；不提前建立训练框架。

## 文件与可追溯性

正式实验：`/home/xuehu/projects/tpms_jax/validation/thickness_range_20261006_r13`。`input.json`先固定方法/门槛，`preparation.json`记录原INP单行差异；`t0p45/`、`t0p55/`各含`T0p004/`、匹配`abaqus/explicit_T0p040/`与`analysis/`。完整场和日志留本机，JSON/分析工具/图及输入证据进Git。`decision.json`报告范围，`evidence_manifest.json`与`verification.json`核查哈希；r6-r12科学原件不覆盖。

原生壳ODB：`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/thickness_range_20261006_r13_t0p45_explicit_T0p040`、同名`t0p55`目录。Windows `work/thickness_range_20261006`仅一次性准备/提取/报告/发布收据，不是第二套FEM。准备时初次严格字节匹配因原INP的CRLF停下，随后保留原换行完成；原脚本/说明留存，没有改科学常数凑通过。
'''
(W/'THICKNESS_RANGE_PROGRESS.md').write_text(report)
shutil.copy2(W/'THICKNESS_RANGE_PROGRESS.md',R/'docs/THICKNESS_RANGE_PROGRESS.md')
figdir=W/'output/figures';figdir.mkdir(exist_ok=True,parents=True)
shutil.copy2(O/'response.png',figdir/'THICKNESS_RANGE_response.png')
shutil.copy2(O/'response.png',R/'docs/figures/THICKNESS_RANGE_response.png')
# Git documentation image path is repository-relative, Windows reading path differs.
p=R/'docs/THICKNESS_RANGE_PROGRESS.md';p.write_text(p.read_text().replace('output/figures/','figures/'))
print(json.dumps(decision,indent=2))
