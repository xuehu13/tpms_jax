"""Freeze one authorized three-case forward batch; no solver submission here."""
from pathlib import Path
import ast, hashlib, json, shutil, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
W=Path(__file__).resolve().parents[2]
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
B=R/'validation/platen_xy_diverse_batch_20261012_r55'
O=W/'output/r55_platen_xy_diverse_batch_20261012'
LB='/home/xuehu/projects/tpms_jax/validation/'+B.name
cases={'diverse_28':'para_aly/Fine/T0p02/MS9/diverse_28',
       'diverse_05':'para_aly/Fine/T0p02/MS9/diverse_05',
       'diverse_02':'r3_smoke/batch_001/diverse_02'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
assert not B.exists() and not O.exists()
for p in [B/'checks',B/'tools',B/'sources',O]:p.mkdir(parents=True)
guard=json.loads((R/'validation/platen_xy_diverse04_20261012_r54/checks/source_guard_before.json').read_text())
for base in [R/'validation/platen_xy_diverse04_20261012_r54']:
 for p in base.rglob('*'):
  if p.is_file() and p.suffix in ['.py','.json','.inp'] and '__pycache__' not in str(p):guard[p.relative_to(R).as_posix()]=sha(p)
guard['validation/platen_xy_gradients_20261011_r53/status.json']=sha(R/'validation/platen_xy_gradients_20261011_r53/status.json')
for p in (W/'work/r55_stage').glob('*.py'):shutil.copy2(p,B/'tools'/p.name)
audit={}
for kind,relative in cases.items():
 src=Path('F:/auto_abaqus/work')/relative
 mesh=src/f'mesh/engine/cases/{kind}/shell/{kind}_shell.npz'
 cfg=json.loads((src/'case_used.json').read_text());L=float(cfg['unit_cell_size_mm']);assert L==10
 with np.load(mesh) as z:v=z['nodes']/L;f=z['triangles']
 cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);area=np.linalg.norm(cross,axis=1)/2
 assert np.isfinite(v).all() and area.min()>0 and f.min()==0 and f.max()<len(v)
 normal=cross/(2*area[:,None]);moment=np.einsum('f,fi,fj->ij',area,normal,normal)/area.sum()
 wrapped=np.where(np.isclose(v,1,atol=1e-10),0,v)
 _,inverse=np.unique(np.round(wrapped,9),axis=0,return_inverse=True);glued=inverse[f]
 directed=np.vstack([glued[:,[0,1]],glued[:,[1,2]],glued[:,[2,0]]])
 edges,which,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
 orient=np.bincount(which,weights=np.where(directed[:,0]<directed[:,1],1,-1))
 graph=coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])))
 comp=int(connected_components(graph,directed=False,return_labels=False));seams=[]
 for axis in range(3):
  lo=v[np.isclose(v[:,axis],0,atol=1e-10)].copy();hi=v[np.isclose(v[:,axis],1,atol=1e-10)].copy();lo[:,axis]=hi[:,axis]=0
  assert len(lo)==len(hi) and len(lo)>0
  gap=float(max(cKDTree(lo).query(hi)[0].max(),cKDTree(hi).query(lo)[0].max()));assert gap<1e-9
  seams.append({'axis':axis,'nodes_per_side':len(lo),'maximum_gap_over_L':gap})
 assert comp==1 and np.all(counts==2) and np.all(orient==0)
 groups={}
 for i,x in enumerate(v):groups.setdefault(tuple(np.round(np.r_[wrapped[i,:2],x[2]],9)),[]).append(i)
 relations=sum(len(g)-1 for g in groups.values())
 d=B/'sources'/kind;d.mkdir();np.savez_compressed(d/'surface.npz',surface_vertices=v,surface_triangles=f)
 for name in ['case_used.json',f'mesh/engine/cases/{kind}/shell/{kind}_shell_report.json']:
  shutil.copy2(src/name,d/Path(name).name)
 curve=src/'results/stress_strain.csv';hist=np.genfromtxt(curve,delimiter=',',names=True)
 i=hist['engineering_stress_MPa'].argmax()
 original={str(p):sha(p) for p in [mesh,src/'case_used.json',curve,src/'simulation_used.json'] if p.exists()}
 audit[kind]={'source_directory':str(src),'original_readonly_source_sha256':original,
  'source_directory_linux':'/mnt/f/auto_abaqus/work/'+relative,'implicit_expression':cfg['surface_expression'],
  'nodes':len(v),'triangles':len(f),'area_mm2':float(area.sum()*100),'normal_second_moment':moment.tolist(),
  'periodic_components':comp,'edges_not_two':int(np.sum(counts!=2)),'orientation_conflicts':int(np.sum(orient!=0)),
  'seams':seams,'xy_independent_relations':relations,'shell_surface':LB+'/sources/'+kind+'/surface.npz',
  'source_sha256':{'surface.npz':sha(d/'surface.npz')},'geometry_screen_pass':True,
  'historical_curve_selection_only':{'peak_compression':float(hist['engineering_strain'][i]),
    'peak_stress_MPa':float(hist['engineering_stress_MPa'][i]),'final_stress_MPa':float(hist['engineering_stress_MPa'][-1]),
    'not_current_reference':'Old material/rate/contact/mass scaling differ; selection only, no prediction of new response'},
  'selection_reason':'User-named existing periodic geometry, distinct orientation/area and historical drop; chosen before new results'}
 shutil.copy2(curve,d/'historical_stress_strain_selection_only.csv')
write(B/'checks/geometry_precheck.json',audit);write(O/'geometry_precheck.json',audit)
write(B/'checks/source_guard_before.json',guard)
p=json.loads((R/'validation/platen_xy_diverse04_20261012_r54/protocol.json').read_text())
p.update(request='Three paired 20% forward comparisons: diverse_28, diverse_05, diverse_02 under latest common midsurface-end loading',
 cases=list(cases),future_dynamic_jobs=6,execution_order=[k+'_'+s for k in cases for s in ['Abaqus','JAX']],
 authorization='User explicitly authorized preparation followed by 3-4 geometry tests; choose three named cases',
 cost_basis_minutes={'r54_diverse04_total':41.0129,'prior_G_total':87.08,'prior_P_total':44.59,'prior_D_total':80.12},
 maximum_batch_minutes=300,geometry_precheck='checks/geometry_precheck.json',
 implementation='Same r51 mechanics / r52 bounded runners and preparation; new geometry/Gauss caches and data-only adapters')
write(B/'protocol.json',p);write(B/'protocol_planned.json',p);write(O/'protocol.json',p)
write(B/'status.json',{'stage':'inputs_screened_queue_ready','execution_authorized':True,'active_jobs':[],
 'cases':{k:{'stage':'queued','dynamic_attempts':0} for k in cases},'queue':list(cases),'heartbeat':'PAUSED',
 'gradient_research':'paused; original failures and gate preserved','new_design_AD':False,'new_training':False,'new_git':False})
shutil.copy2(B/'status.json',O/'status.json')
archive=W/'history/execution_diverse_batch_20261012_r55/before';archive.mkdir(parents=True)
names=['AGENTS.md','RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','START_HERE.md']
for n in names:shutil.copy2(W/n,archive/n)
plan='''# 唯一主规划：三种diverse构型的共同端缘压板20%对照

2026-10-12用户新增授权：diverse_28、diverse_05、diverse_02共三组／六个前向任务，准备后顺序运行，不另行等待授权。采用与r54相同的压板／XY周期／自由横向宏观伸缩／共同中面端缘简化。每组100 min总预算（JAX65、Abaqus25、准备提取分析10）；总上限300 min，每端一次尝试，失败保留，不用物理或数值参数拟合参考。

1. 冻结原几何、独立输入、积分点占据、周期关系、共同端缘、参数和指标；保存原文件哈希。
2. 按diverse_28→diverse_05→diverse_02顺序执行，每组Abaqus后JAX：4 ms到20%＋0.2 ms保载，保留失败／有效部分路径及实际成本。
3. 同口径比较总力／名义应力、曲线与分段RMS、加载功、保载、原始最大值和预定显著峰、变形及能量。分母用每组新壳自身加载峰；不挑峰、不平移或缩放曲线。
4. 汇总三种构型的可迁移范围、差异和限制，更新当前导航和r55报告，完成即停止。

历史曲线仅用于选样，旧材料／时长／接触不作为本轮参考。共同端缘模型没有完整有限厚度壁面接触或自接触，宏观通过不等于局部模式通过。约10%为限定工作目标，壳不是真值。正式目录`/home/xuehu/projects/tpms_jax/validation/platen_xy_diverse_batch_20261012_r55`；[阅读报告](output/r55_platen_xy_diverse_batch_20261012/README.md)。

r54 diverse_04已完整完成：曲线RMS2.700%、功−1.031%、保载−4.706%，无满足预定规则的显著内部峰，旧证据保持。

原r53梯度四步未全部完成：第1／2步完成，第3步功通过、末端力失败，第4步总表完成但三参数完整VJP未启动。原门槛与失败保持，不扩展AD／FD、训练或Git；heartbeat tpms持续PAUSED。[梯度综合报告](output/r53_platen_xy_gradients_20261011/analysis/parameter_summary/README.md)。
'''
(W/'RESEARCH_PLAN.md').write_text(plan,encoding='utf-8');(R/'docs/RESEARCH_PLAN.md').write_text(plan,encoding='utf-8')
notice='2026-10-12新增授权：r55正在准备／执行diverse_28、diverse_05、diverse_02三组20%共同端缘压板对照；每组100 min，一次尝试。唯一执行队列见RESEARCH_PLAN.md；梯度与heartbeat仍暂停。旧“无活动／已停止”描述仅指r54结束时。\n\n'
for n,rn in [('PROJECT_OVERVIEW.md','docs/RESEARCH_STATUS.md'),('FILE_MAP.md','docs/FILE_MAP.md'),('START_HERE.md','README.md'),('AGENTS.md','AGENTS.md')]:
 q=W/n;s=q.read_text(encoding='utf-8');i=s.find('\n')+1;s=s[:i]+'\n'+notice+s[i:];q.write_text(s,encoding='utf-8');(R/rn).write_text(s,encoding='utf-8')
maptext='\n## 新增批次r55\n\n正式R/validation/platen_xy_diverse_batch_20261012_r55；阅读W/output/r55_platen_xy_diverse_batch_20261012；原生A/r55_<kind>_midsurface_end_20pct_20261012。sources按构型保存三角中面及实际Gauss场，各构型input／abaqus／jax／analysis隔离；共享力学实现仍为r51和仓库原程序。\n'
with (W/'FILE_MAP.md').open('a',encoding='utf-8') as f:f.write(maptext)
shutil.copy2(W/'FILE_MAP.md',R/'docs/FILE_MAP.md')
(B/'README.md').write_text('# r55：三种diverse构型的共同端缘压板对照\n\n已获授权，三组顺序执行。当前过程见status.json；协议、预算及指标见protocol.json。尚无新结果，不以历史曲线代替本轮计算。梯度／heartbeat暂停。\n',encoding='utf-8')
shutil.copy2(B/'README.md',O/'README.md')
write(B/'checks/setup_receipt.json',{'stage_source_sha256':{p.name:sha(p) for p in (B/'tools').glob('*.py')},'protected_old_files':len(guard),'cases':list(cases),'new_dynamic_jobs_at_setup':0,'time_unix':time.time()})
print(json.dumps({'geometry_screened':list(cases),'batch':str(B),'protected_files':len(guard)},ensure_ascii=False))
