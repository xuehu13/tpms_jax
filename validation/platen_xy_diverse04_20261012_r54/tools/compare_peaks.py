"""Compare existing pair results under frozen peak rules; no FEM/JAX calls."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];E=B/'diverse_04';O=E/'analysis'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
cfg=read(E/'input/protocol.json');T=cfg['load_time_s'];L=cfg['L_mm'];target=cfg['target_compression']
results={k:read(E/sub/'result.json') for k,sub in [('JAX','jax'),('Abaqus','abaqus')]}
def array(res,key):return np.array([r[key] for r in res['path']],float)
common_time=min(T,*[array(r,'time')[-1] for r in results.values()])
curves={};raw_loading={}
for name,r in results.items():
 t=array(r,'time');mask=t<common_time-1e-13
 eps=np.r_[array(r,'compression')[mask],np.interp(common_time,t,array(r,'compression'))]
 force=np.r_[array(r,'force_N')[mask],np.interp(common_time,t,array(r,'force_N'))]
 raw_loading[name]=(eps.copy(),force.copy(),np.r_[t[mask],common_time])
 if abs(eps[-1]-target)<1e-8:eps[-1]=target
 unique,rev=np.unique(eps[::-1],return_index=True);curves[name]=(unique,force[len(eps)-1-rev])
emax=min(x[0][-1] for x in curves.values());grid=np.linspace(0,emax,max(2,round(emax/.0002)+1))
force={name:np.interp(grid,x,f) for name,(x,f) in curves.items()}
smooth={name:uniform_filter1d(f,size=25,mode='nearest') for name,f in force.items()}
complete=all(r['complete'] for r in results.values()) and common_time>=T-1e-10
den=float(np.max(np.abs(raw_loading['Abaqus'][1])))
metrics={'scope':cfg['stage'],'paired_complete20_and_hold':bool(complete),'common_compression_max':float(emax),
 'common_time_s':float(common_time),'normalizer_N':den,
 'normalizer_scope':'new shell full loading peak' if complete else 'new shell common achieved loading window peak; not full20 acceptance',
 'raw_global_peak':{},'smoothed_peaks':{},'primary_curve_RMS_over_shell_peak':float(np.sqrt(np.mean((force['JAX']-force['Abaqus'])**2))/den),
 'peak_rules':read(B/'protocol.json')['metrics']['peak_rules'],'no_new_FEM_AD_FD':True}
for name,(eps,f,times) in raw_loading.items():
 idx=int(np.argmax(f));dt=np.diff(array(results[name],'time'))
 peak_time=float(times[idx]);at_endpoint=bool(eps[idx]>=emax-1e-8)
 metrics['raw_global_peak'][name]={'compression_percent':float(eps[idx]*100),'force_N':float(f[idx]),'time_s':peak_time,
  'at_loading_endpoint':at_endpoint,'sample_time_spacing_s_min_max':[float(dt.min()),float(dt.max())],
  'captured_interior_global_peak':bool(idx>0 and not at_endpoint)}
 f=smooth[name];prominence=.1*float(np.max(abs(f)));indices,props=find_peaks(f,prominence=prominence,distance=25)
 metrics['smoothed_peaks'][name]={'smoothing_points':25,'approx_width_percentage_points':float(25*(grid[1]-grid[0])*100),
  'prominence_threshold_N':prominence,'peaks':[{'compression_percent':float(grid[i]*100),'force_N':float(f[i]),'prominence_N':float(p)} for i,p in zip(indices,props['prominences'])]}
jp=metrics['raw_global_peak']['JAX'];ap=metrics['raw_global_peak']['Abaqus']
metrics['raw_global_peak_location_difference_percentage_points']=jp['compression_percent']-ap['compression_percent']
metrics['raw_global_peak_force_relative_bias']=jp['force_N']/ap['force_N']-1
for name in results:
 metrics['smoothed_peaks'][name]['first_prominent_interior_peak']=next(iter(metrics['smoothed_peaks'][name]['peaks']),None)
pjs=metrics['smoothed_peaks']['JAX']['first_prominent_interior_peak'];pas=metrics['smoothed_peaks']['Abaqus']['first_prominent_interior_peak']
metrics['first_prominent_peak_difference_percentage_points']=None if pjs is None or pas is None else pjs['compression_percent']-pas['compression_percent']
metrics['loading_work_N_mm']={name:float(np.trapezoid(f,x*L)) for name,(x,f) in curves.items()}
metrics['loading_work_relative_bias']=metrics['loading_work_N_mm']['JAX']/metrics['loading_work_N_mm']['Abaqus']-1
metrics['point_forces']=[]
for eps in [.02,.05,.10,.15,.20]:
 if eps>emax+1e-10:continue
 jf=float(np.interp(eps,*curves['JAX']));af=float(np.interp(eps,*curves['Abaqus']))
 metrics['point_forces'].append({'compression_percent':eps*100,'JAX_N':jf,'Abaqus_N':af,'relative_bias':jf/af-1 if abs(af)>1e-12 else None})
if complete:
 metrics['hold_mean_N']={}
 for name,r in results.items():
  t=array(r,'time');f=array(r,'force_N');lo=.0041;hi=.0042;x=np.r_[lo,t[(t>lo)&(t<hi)],hi]
  metrics['hold_mean_N'][name]=float(np.trapezoid(np.interp(x,t,f),x)/(hi-lo))
 metrics['hold_mean_relative_bias']=metrics['hold_mean_N']['JAX']/metrics['hold_mean_N']['Abaqus']-1
 metrics['three_working_targets_within10pct']=bool(metrics['primary_curve_RMS_over_shell_peak']<=.1 and abs(metrics['loading_work_relative_bias'])<=.1 and abs(metrics['hold_mean_relative_bias'])<=.1)
else:metrics['three_working_targets_within10pct']=None
for name,r in results.items():
 metrics[name+'_terminal_time_s']=float(array(r,'time')[-1]);metrics[name+'_complete']=r['complete']
metrics['source_sha256']={name:hashlib.sha256((E/sub/'result.json').read_bytes()).hexdigest() for name,sub in [('JAX','jax'),('Abaqus','abaqus')]}
write(O/'peak_metrics.json',metrics)
with (O/'peak_comparison.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['compression_percent','JAX_raw_interpolated_N','Abaqus_raw_interpolated_N','JAX_moving_mean_N','Abaqus_moving_mean_N'])
 w.writerows(zip(grid*100,force['JAX'],force['Abaqus'],smooth['JAX'],smooth['Abaqus']))
with (O/'peak_summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['solver','peak_definition','compression_percent','force_N','endpoint_flag'])
 for name,p in metrics['raw_global_peak'].items():w.writerow([name,'raw global max',p['compression_percent'],p['force_N'],p['at_loading_endpoint']])
 for name,p in metrics['smoothed_peaks'].items():
  for item in p['peaks']:w.writerow([name,'moving mean prominent interior peak',item['compression_percent'],item['force_N'],False])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
fig,axes=plt.subplots(2,1,figsize=(11.8,8.5),constrained_layout=True)
for name,col in [('Abaqus','#285c8e'),('JAX','#c45b25')]:
 eps,f=curves[name];axes[0].plot(eps*100,f,color=col,lw=1.25,label=name+' raw loading')
 p=metrics['raw_global_peak'][name];axes[0].scatter(p['compression_percent'],p['force_N'],color=col,s=40,zorder=4)
 axes[1].plot(grid*100,smooth[name],color=col,label=name+' fixed moving mean')
 for p in metrics['smoothed_peaks'][name]['peaks']:
  axes[1].scatter(p['compression_percent'],p['force_N'],color=col,s=35)
  axes[1].annotate(f"{p['compression_percent']:.2f}%",(p['compression_percent'],p['force_N']),xytext=(4,6),textcoords='offset points',fontsize=8,color=col)
for ax in axes:ax.set(xlabel='Engineering compression (%) = upper platen travel / 10 mm',ylabel='Total upper-platen compressive force (N)',xlim=(0,20));ax.grid(alpha=.2);ax.legend()
axes[0].set_title('Raw 0-to-20% loading; dots mark the observed global maxima')
axes[1].set_title('Diagnostic mean (~0.5 percentage points); dots mark predeclared prominent peaks')
fig.suptitle('diverse_04 | common midsurface-end platens | free XY PBC | 4 ms loading + 0.2 ms hold')
fig.savefig(O/'peak_and_curve_comparison.png');plt.close(fig)
rows='\n'.join(f"| {name} | {p['compression_percent']:.4f}% | {p['force_N']:.6f} | {'加载端点；不是内部峰' if p['at_loading_endpoint'] else '窗口内峰；是否屈曲需结合变形'} |" for name,p in metrics['raw_global_peak'].items())
pts='\n'.join(f"| {p['compression_percent']:.0f}% | {p['JAX_N']:.6f} | {p['Abaqus_N']:.6f} | {100*p['relative_bias']:+.2f}% |" for p in metrics['point_forces'] if p['relative_bias'] is not None)
mean='未形成完整保载对照' if not complete else f"保载后半段均力差{100*metrics['hold_mean_relative_bias']:+.3f}%"
prominent='至少一端未捕捉到满足预定显著性规则的内部峰；不计算两端首峰位置差。'
if pjs is not None and pas is not None:
 prominent=f"预定平滑规则下首个显著内部峰：JAX {pjs['compression_percent']:.3f}%，Abaqus {pas['compression_percent']:.3f}%，差{metrics['first_prominent_peak_difference_percentage_points']:+.3f}个百分点。"
text=f'''# diverse_04：共同中面端缘压板／XY周期／20%对照

2026-10-12。{'两端完整20%及保载完成。' if complete else '存在不完整路径，仅比较实际共同窗口，不能作完整20%验收。'} 本次主要比较上压板总反力随压板总压缩量的变化，原始峰、诊断平滑峰分别报告；没有新梯度或训练。

全曲线RMS／新壳峰值={100*metrics['primary_curve_RMS_over_shell_peak']:.3f}%；加载功差{100*metrics['loading_work_relative_bias']:+.3f}%；{mean}。当前共同窗口到{emax*100:.4f}%。分母为{metrics['normalizer_scope']} {den:.6f} N；RMS不是逐点误差上限。

## 峰位与曲线

| 求解器 | 原始全局最大力所在压缩量 | 最大力N | 类型 |
|---|---:|---:|---|
{rows}

原始全局最大力的位置差（JAX−Abaqus）{metrics['raw_global_peak_location_difference_percentage_points']:+.4f}个百分点，峰力差{100*metrics['raw_global_peak_force_relative_bias']:+.3f}%。若最大值在加载端点，不能把它命名为已捕捉的屈曲峰。{prominent}

![完整曲线与预先定义的峰](analysis/peak_and_curve_comparison.png)

平滑仅用于区分主要降载与局部振荡：固定工程压缩网格约0.02个百分点间隔、25点中心均值，显著峰要求prominence不小于自身曲线峰的10%，峰间至少25点。所有合格峰都列在peak_summary.csv，不事后挑最接近的一对；原始曲线指标不平滑、不缩放、不平移。该规则是曲线诊断，不能仅凭反力峰证明两个模型走了同一屈曲分支。

| 压缩量 | JAX上板总力N | Abaqus上板总力N | 相对差 |
|---|---:|---:|---:|
{pts}

## 建模条件与可比性

复用原diverse_04的8372节点／15914三角中面和原N32积分点占据场，不重新拟合几何。L10 mm、t0.5 mm、NH E10 MPa／ν0.3／密度1e−9 tonne/mm³。JAX仍为HEX27／27点、HRZ、η1e−4、界面0.05 mm、objective_void/C²；Abaqus为同中面的S3R壳、截面5点。两端物理加载时长均4 ms、保载0.2 ms，五次平滑下压2 mm。

仅XY周期，宏观横向伸缩自由、宏观剪切零，没有Z周期或壁节点额外固定。263条独立节点关系形成1578条壳方程；上下每端162个中面节点接触压板。JAX把自身连续占据端面面积聚合到同一批端缘位置，面积守恒、Q2插值、kn320 MPa/mm；Abaqus端缘节点接触忽略接触厚度，截面厚度仍保留。两端运动学、罚律、接触插值和惯性离散不同；不是同离散问题。

没有全壁面、自接触或周期镜像接触。实际F/J不夹断，主体正J保护及软域能量保留。允许物理跳跃与动能升高；数值失败、步长缩小和能量失衡另行报告。对照只回答共同简化加载下的计算能力，不认证真实完整有限厚度接触。

## 原始文件与解释边界

正式批次`/home/xuehu/projects/tpms_jax/validation/platen_xy_diverse04_20261012_r54/diverse_04`，参数在input/protocol.json、INP在abaqus/platen_20pct.inp、原路径在abaqus/result.json与jax/result.json。原生Abaqus目录`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/r54_diverse04_midsurface_end_20pct_20261012`，保留platen.inp／platen.odb及dat/msg/sta。

analysis/metrics.json沿用G/P/D对照口径；analysis/peak_metrics.json及CSV记录本次峰定义与原始来源哈希，costs.json记录含准备／编译／提取的实际成本。若有中断，只认实际保存的有效范围，不外推未计算峰。原XYZ结果保留，但旧时长、横向约束及停止范围可能不同，不能把新旧差异全部定量归因于取消Z周期或接触简化。
'''
for png,title in [('force_and_lateral.png','完整力及横向响应'),('energy_and_J.png','能量、实际J与动态指标'),('deformation_10_20pct.png','同一时间实际尺度变形')]:
 if (O/png).exists():text+=f'\n![{title}](analysis/{png})\n'
(E/'README.md').write_text(text)
(O/'README.md').write_text(text.replace('](analysis/',']('))
print(json.dumps({k:v for k,v in metrics.items() if k not in ['peak_rules','source_sha256']},ensure_ascii=False),flush=True)
