"""Read-only numerical review of r55 saved paths; update derived report/navigation."""
from pathlib import Path
import csv,hashlib,json,shutil,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
START=time.perf_counter()
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
B=R/'validation/platen_xy_diverse_batch_20261012_r55';O=W/'output/r55_platen_xy_diverse_batch_20261012'
H=W/'history/execution_diverse_batch_20261012_r55/review_before';H.mkdir(parents=True,exist_ok=True)
C=B/'checks/result_review';C.mkdir(exist_ok=True);assert not (C/'receipt.json').exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
status=read(B/'status.json');assert status['batch_finished'] and not status['active_jobs']
cases=read(B/'protocol.json')['cases'];protected={}
for name,digest in read(B/'checks/source_guard_before.json').items():assert sha(R/name)==digest
for kind in cases:
 E=B/kind
 for root in [E/'input',E/'jax',E/'abaqus',B/'sources'/kind]:
  for p in root.glob('*'):
   if p.is_file() and p.suffix in ['.json','.inp','.npz']:protected[p.relative_to(B).as_posix()]=sha(p)
 for p in (E/'analysis').glob('*'):
  if p.is_file() and p.name!='README.md':protected[p.relative_to(B).as_posix()]=sha(p)
for p in (B/'tools').glob('*.py'):protected[p.relative_to(B).as_posix()]=sha(p)
write(C/'protected_before.json',protected)
for kind in cases:
 d=H/kind;d.mkdir(exist_ok=True)
 for q in [B/kind/'README.md',B/kind/'analysis/README.md']:
  shutil.copy2(q,d/('analysis_README.md' if q.parent.name=='analysis' else 'README.md'))
for p in [B/'README.md',B/'analysis/batch_metrics.json',W/'AGENTS.md',W/'RESEARCH_PLAN.md',W/'PROJECT_OVERVIEW.md',W/'FILE_MAP.md',W/'RESEARCH_BACKGROUND.md',W/'START_HERE.md']:
 name='batch_README.md' if p==B/'README.md' else p.name;shutil.copy2(p,H/name)
def arr(res,key):return np.array([r[key] for r in res['path']],float)
def average(res,key,lo,hi):
 t=arr(res,'time');xx=np.r_[lo,t[(t>lo)&(t<hi)],hi]
 yy=np.interp(xx,t,arr(res,key));return float(np.sum((yy[1:]+yy[:-1])*.5*np.diff(xx))/(hi-lo))
records={};dynamic=[]
for kind in cases:
 E=B/kind;p=read(E/'analysis/peak_metrics.json');m=read(E/'analysis/metrics.json')
 j=read(E/'jax/result.json');a=read(E/'abaqus/result.json');records[kind]=(p,m,j,a)
 common=min(arr(j,'time')[-1],arr(a,'time')[-1]);lo=.0041 if p['paired_complete20_and_hold'] else common-.0001
 hi=.0042 if p['paired_complete20_and_hold'] else common
 row={'case':kind,'window_s':[lo,hi],'window_scope':'fixed latter0.1ms hold' if p['paired_complete20_and_hold'] else 'last0.1ms of actual common loading; not hold',
  'JAX_upper_mean_N':average(j,'force_N',lo,hi),'JAX_bottom_mean_N':average(j,'bottom_force_N',lo,hi),
  'Abaqus_upper_mean_N':average(a,'force_N',lo,hi),'Abaqus_bottom_mean_N':average(a,'bottom_force_N',lo,hi)}
 for label in ['JAX','Abaqus']:row[label+'_mean_bottom_minus_upper_N']=row[label+'_bottom_mean_N']-row[label+'_upper_mean_N']
 row['JAX_sampled_momentum_change_N_s']=float(np.interp(hi,arr(j,'time'),arr(j,'momentum_z_N_s'))-np.interp(lo,arr(j,'time'),arr(j,'momentum_z_N_s')))
 row['JAX_sampled_net_impulse_N_s']=row['JAX_mean_bottom_minus_upper_N']*(hi-lo)
 row['interpretation']='Different upper/lower reactions measure net vertical acceleration in this dynamic model; no substitution of bottom force for the frozen upper-force target.'
 dynamic.append(row)
write(B/'analysis/reviewed_dynamic_windows.json',dynamic)
with (B/'analysis/reviewed_dynamic_windows.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(dynamic[0]));w.writeheader();w.writerows(dynamic)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
fig,axs=plt.subplots(2,3,figsize=(15,8.5),constrained_layout=True)
for col,kind in enumerate(cases):
 p,m,j,a=records[kind]
 for label,res,color,style in [('Abaqus',a,'#285c8e','-'),('JAX',j,'#c45b25','--')]:
  t=arr(res,'time');eps=arr(res,'compression');f=arr(res,'force_N');mask=t<.004-1e-13;end=min(.004,t[-1])
  mask=t<end-1e-13;xx=np.r_[eps[mask],np.interp(end,t,eps)]*100;yy=np.r_[f[mask],np.interp(end,t,f)]
  for ax in axs[:,col]:ax.plot(xx,yy,style,color=color,lw=1.2,label=label+(' (budget stop)' if not res['complete'] else ''))
 if not j['complete']:
  for ax in axs[:,col]:ax.axvline(p['common_compression_max']*100,ls=':',color='#777777',lw=1)
  axs[0,col].annotate('JAX stop19.36%\nwall budget',(p['common_compression_max']*100,1.9),xytext=(12.5,.3),arrowprops={'arrowstyle':'->','color':'#555'},fontsize=8)
 axs[0,col].set(title=kind,xlim=(0,20));axs[1,col].set(title='Initial0-6% (visual zoom)',xlim=(0,6))
 for ax in axs[:,col]:ax.set(xlabel='Engineering compression (%)',ylabel='Upper-platen force (N)');ax.grid(alpha=.2);ax.legend(fontsize=8)
 axs[1,col].set_ylim(bottom=0,top=max(np.max(arr(r,'force_N')[arr(r,'compression')<=.06]) for r in [j,a])*1.08)
fig.suptitle('r55 | same common-end loading | blue:Abaqus, orange:JAX | original forces, no scaling')
fig.savefig(B/'analysis/reviewed_force_comparison.png');plt.close(fig)
p28,m28,j28,a28=records['diverse_28'];p05,m05,j05,a05=records['diverse_05'];p02,m02,j02,a02=records['diverse_02']
analysis='''## 人工核对后的研究判断

本轮给出的结论不是“三种构型都算失败”，也不是“已经普遍达到10%”。三组Abaqus完成20%及保载；JAX的diverse_28和02也完整完成，05在65分钟单端预算到达时停止于19.36%。它没有非有限数值、主体翻转或失控减步的停止记录；保留完整有效前缀，但20%与保载尚无成对证据。三组总墙钟160.84 min，含自动准备／编译／检查／求解／提取／分析，下面的人工结果审阅成本另记，不重写原成本。

### 三种不同响应确实被计算出来

![完整加载与初始段，原曲线不缩放](reviewed_force_comparison.png)

第一排显示各自实际完成的加载路径，05竖线标出JAX预算停止位置；Abaqus05仍显示自身完整加载。第二排只是0–6%视觉放大，不增加验收窗口或修改既定指标。跨构型已覆盖持续增载、约11%显著降载、约4.5%显著降载并多次起伏的响应。这比只考察一种平顺曲线更有价值，但也揭示了当前方法的适用边界。

**diverse_28：变形接近，幅值持续偏高。** 两端曲线持续增载，无满足事前10%显著度规则的内部降载峰；20%处的最大力不能当作屈曲峰。2%、5%、10%、15%、20%处JAX总力分别偏高10.72%、12.54%、13.59%、13.79%、14.65%；曲线RMS／壳峰9.26%，功+13.99%，保载均力+14.86%。虽然RMS小于10%，另两个预定量未达标。20%附近中面总位移相对RMS差仅0.60%，说明宏观变形近，但位移指标包含受控位移与宏观分量，并不能认证局部应变、曲率或应力也只有0.60%误差。这组更像平顺加载中的系统性反力偏高，而非峰位错移。

**diverse_05：初段和主要峰接近，峰后仍分歧，末段因预算缺失。** 2%、5%、10%力差为−1.17%、+0.05%、+2.86%。原始主峰在JAX10.9007%／4.8232 N、Abaqus10.7701%／4.6405 N，差+0.1306个百分点／+3.94%；按既定平滑显著峰规则为10.90%与10.76%，差+0.14个百分点。必须区分“0.14个百分点”与“14%相对误差”。共同已算0–19.36%窗口的RMS／壳峰8.01%、功+1.43%，但15%瞬时力差−18.04%，不能用积分量接近掩盖瞬时差异。中面位移相对RMS差约5%／10%／15%时为4.85%／5.09%／12.18%，峰后局部响应差异增长。没有完整20%与保载结果，不能宣布本组完整通过或数值算法失败。

**diverse_02：首个显著峰接近，后屈曲动态路径差异更大。** 原始主峰JAX4.6867%／2.8749 N、Abaqus4.5497%／2.7741 N，差+0.1371个百分点／+3.63%；固定平滑规则下为4.66%和4.54%，差+0.12个百分点。2%力差−2.45%，说明早期增载可以接近；5%已经位于主要降载附近，瞬时力差+14.92%，不能笼统称为整个“小压缩区间”都准确。完整曲线RMS／壳峰15.01%、功+12.72%、上板保载均力+30.18%，未达到三个约10%工作目标。峰后多次起伏的时序和谷值不同，20%附近中面总位移相对RMS差10.75%。当前不能把峰位接近升级为整个后屈曲过程通过。

### 保载力差不能直接当成静态刚度差

保载只是停止移动压板，不表示结构已经静止。diverse_02预定后半段0.1 ms保载窗口：JAX上板均力1.9812 N、下板1.5609 N；Abaqus上板1.5218 N、下板1.5552 N。下板相对差约+0.37%，但JAX上下板均力相差约0.4203 N，而Abaqus约−0.0333 N。这说明同一短窗内两端的整体竖向加速度／动量变化不同。因此上板30.18%的差仍是本协议的真实未通过指标，但不能把它直接解释为静态承载能力偏高30%。下板恰接近也不能被事后改选为验收指标。冻结的上板、窗口、分母和未通过记录全部保持。

05在实际共同末时刻约3.358 ms，两端动能／内能约31.7%（JAX）和30.0%（Abaqus，同时间附近样本）；02完整末时刻约5.49%和5.96%。这些比值说明峰后是带惯性的动态响应，不是静态平衡曲线。不能仅凭动能升高否决跳跃现象，也不能据此自动证明全部差异均由惯性产生。动态窗口均力／动量与冲量计算另存reviewed_dynamic_windows.json，未增加任何求解。

### 能支持什么、尚不能支持什么

主体要求正体积的积分点在采样中最小实际J分别为28：0.7317、05：0.1142、02：0.3344，均正；三组没有拒绝推进块，JAX采样能量／功缺口分别约0.00091%、0.0161%、0.0198%。这些支持已保存路径具有基本数值一致性，但不是全过程、任意网格或真实接触的准确性证明。低占据虚域负J仍保留并计能，未做实际F／J夹断。05的停止是预定预算，02和28完成并不自动意味着力学验证通过。Abaqus本身也有离散误差；02全局能量漂移约1.008%，不能把参考当作无误差真值。

28／05／02的自身积分占据体积，相对壳中面面积×厚度仅差−0.35%／−0.71%／−0.57%。这使“几何总量相差十几个百分点”不受当前数据支持，但不能排除占据在厚度方向的分布、薄壁弯曲积分或局部运动学差异。总量接近不等于弯曲刚度接近。

目前最合理的研究判断是：固定背景网格方法已经能迁移到多种周期薄壁构型并生成有区别的受压响应，且在05和02上给出了接近的首次显著峰；但反力幅值精度和后屈曲动态路径还具有构型依赖性。结合已有G／P／diverse_04，受控前向可行性仍有证据，通用构型0–20%都在10%以内的结论尚不成立。本轮没有新梯度，已有参数梯度的通过范围与失败不改变；前向接近不自动认证设计梯度。

### 若继续，优先研究两类差异，暂不提交

下一轮应避免把所有现象归成同一个5%偏硬，更不宜用统一模量或厚度修正系数拟合。首先可为05剩余加载和保载另定预算，复用最后状态补齐范围，不能把当前未完成记作失败或通过。其次，28的平顺系统偏硬适合在无惯性初段定位薄壁有效刚度／积分／壳实体运动学及端缘传力；02和05的峰后差异则应结合既有上下板反力、动量、能量与变形，区分动态相位／惯性和分支差异。若另做时长或保载试验，需要独立冻结单一因素、保持原指标，不从结果中挑最接近的窗口。以上只是后续建议，本轮计算已停止，未自动增加任务。

本轮仍是共同中面端缘加载，无全壁面有限厚度接触或自接触。两端单元、罚律、接触插值和惯性离散不相同，不能声称已排除端部或所有物理模型影响；各具体来源仍需受控实验才可确立。原输入、源码、曲线、失败、指标与图片保持，新增人工审阅派生内容单独记录。
'''
master=(B/'README.md').read_text(encoding='utf-8')
master=master.replace('|未完整／待运行|','|JAX预算停在19.36%；壳完整|')
first=master.index('## 本轮可作的结论')
master=master[:first]+analysis
formal=master
for png in ['all_pairs_force.png','all_shapes_stress.png','reviewed_force_comparison.png']:formal=formal.replace(']('+png+')','](analysis/'+png+')')
(B/'README.md').write_text(formal,encoding='utf-8');(O/'README.md').write_text(master,encoding='utf-8')
for kind in cases:
 for p in [B/kind/'README.md',B/kind/'analysis/README.md',O/kind/'README.md']:
  s=p.read_text(encoding='utf-8')
  if kind=='diverse_05':
   s=s.replace('全曲线RMS／新壳峰值=8.012%','共同已完成0–19.36%窗口的曲线RMS／新壳共同窗口峰值=8.012%')
   s=s.replace('两端物理加载时长均4 ms、保载0.2 ms，五次平滑下压2 mm。','两端计划均4 ms加载、0.2 ms保载，五次平滑下压2 mm；Abaqus完成，JAX达到65分钟单端预算后停在19.36%，没有完成末段与保载。')
   s=s.replace('是否完整：JAX False、Abaqus True。','是否完整：JAX False、Abaqus True。JAX实际90,688步，因预定65分钟单端预算停止，不是非有限值或主体负J失败；总组预算100分钟与JAX单端65分钟是两个独立上限。')
  p.write_text(s,encoding='utf-8')
for p in (B/'analysis').glob('reviewed_*'):
 if p.is_file():shutil.copy2(p,O/p.name)
# Correct the current entry rather than leaving the three new results only in a banner.
q=W/'PROJECT_OVERVIEW.md';s=q.read_text(encoding='utf-8');s=s.replace('# 当前状态：diverse_04新加载对照已完成；梯度继续暂停','# 当前状态：r55三构型对照已收口；梯度继续暂停')
insert='''## 最新三构型对照r55

三组Abaqus完成；JAX28／02完整20%＋保载，05因65分钟单端预算停止在19.36%，不是数值崩溃。总墙钟160.84 min。28为平顺系统偏硬；05／02首次显著降载峰位置差0.14／0.12个百分点、原始主峰力差3.94%／3.63%，峰后仍有分歧。02短保载上下板反力未平衡，不能把上板30.18%差直接命名为静态刚度差；原上板验收未通过不改。[完整总结与图](output/r55_platen_xy_diverse_batch_20261012/README.md)。

| 构型 | 成对范围 | 曲线RMS／壳峰 | 加载功差 | 保载上板均力差 | 本轮判断 |
|---|---|---:|---:|---:|---|
| diverse_28 | 完整20%＋保载 | 9.26% | +13.99% | +14.86% | 幅值偏高，三个目标未全过 |
| diverse_05 | 共同到19.36% | 8.01% | +1.43% | 未计算 | 不作完整20%验收；主要峰接近 |
| diverse_02 | 完整20%＋保载 | 15.01% | +12.72% | +30.18% | 首个显著峰接近，峰后与短窗保载未通过 |

RMS不是逐点误差上限，05的量只覆盖实际共同窗口；约10%为限定工作目标，保留原始指标、失败和未完成记录。当前无活动队列，未启动额外前向、AD／FD、训练或Git。

'''
s=s.replace('## 当前新增对照r54',insert+'## 上轮对照r54');q.write_text(s,encoding='utf-8');shutil.copy2(q,R/'docs/RESEARCH_STATUS.md')
q=W/'RESEARCH_BACKGROUND.md';s=q.read_text(encoding='utf-8');s=s.replace('diverse_04共同端缘压板20%成对前向对照已完成；其范围见当前状态。','r54与r55共同端缘压板前向对照已收口；r55显示首次显著峰可接近，但幅值和后屈曲精度依赖构型，不能普遍承诺0–20%约10%以内；其范围见当前状态。');q.write_text(s,encoding='utf-8');shutil.copy2(q,R/'docs/RESEARCH_BACKGROUND.md')
q=W/'FILE_MAP.md';s=q.read_text(encoding='utf-8');s=s.replace('## 当前前向批次r54','## 已完成前向批次r54');s+='\n人工审阅补充在r55原README，新增reviewed_force_comparison.png及reviewed_dynamic_windows.json/CSV，原自动图和原始指标不覆盖。审阅前文档在W/history/execution_diverse_batch_20261012_r55/review_before；哈希及审阅收据在R/validation/platen_xy_diverse_batch_20261012_r55/checks/result_review。\n';q.write_text(s,encoding='utf-8');shutil.copy2(q,R/'docs/FILE_MAP.md')
q=W/'RESEARCH_PLAN.md';s=q.read_text(encoding='utf-8');s+='\n人工结果核对已完成，详细解释集中r55报告。后续只保留05续算补全、28平顺幅值来源、02／05动态峰后分歧三个建议，均未启动；原四步执行尝试已结束，但05完整20%验证尚缺，三组不全达约10%目标。\n';q.write_text(s,encoding='utf-8');shutil.copy2(q,R/'docs/RESEARCH_PLAN.md')
q=W/'AGENTS.md';s=q.read_text(encoding='utf-8');s=s.replace('三个宏观工作目标通过无。','三个宏观工作目标全部通过的新增构型：无；05预算停止不是数值崩溃。05／02首次显著峰位置差0.14／0.12个百分点，原始主峰力差3.94%／3.63%；峰后未普遍通过。02短保载上下板力不平衡，不能把上板30.18%差直接叫静态刚度差。');q.write_text(s,encoding='utf-8');shutil.copy2(q,R/'AGENTS.md')
assert all(sha(B/n)==h for n,h in protected.items())
for n,h in read(B/'checks/source_guard_before.json').items():assert sha(R/n)==h
write(C/'receipt.json',{'protected_r55_files':len(protected),'all_original_sources_inputs_results_metrics_figures_byte_unchanged':True,
 'prior93_files_unchanged':True,'review_seconds':time.perf_counter()-START,'new_dynamic_solve':0,'new_AD_FD':0,'new_git':0,
 'original_batch_minutes_unchanged':status['batch_minutes'],'derived_dynamic_windows':dynamic,
 'reports_updated':['README.md','docs/RESEARCH_STATUS.md','docs/RESEARCH_PLAN.md','docs/RESEARCH_BACKGROUND.md','docs/FILE_MAP.md'],
 'figure_layout_visually_reviewed':False})
print(json.dumps({'review_seconds':time.perf_counter()-START,'unchanged_protected_files':len(protected),'dynamic_windows':dynamic},ensure_ascii=False))
