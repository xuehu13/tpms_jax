from pathlib import Path
import sys,json,hashlib,shutil
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R));O=R/'validation/geometry_interface_20261003_r2';S=O/'step2'
from scripts.estimate_binary_volume import estimate
rows=[];aba=[]
for label,c in [('c048',.48),('c060',.60)]:
 a={N:json.loads((S/f'{label}_N{N}_analytic.json').read_text()) for N in [48,64]}
 mapped=json.loads((S/f'{label}_N64_M64.json').read_text());ks={N:abs(a[N]['Fz_top'])/.01 for N in [48,64]};km=abs(mapped['Fz_top'])/.01
 ref={}
 for G in [32,48]:
  package=Path(json.loads((S/f'{label}_G{G}.package.json').read_text())['package_wsl']);name=f'binary_gyroid_G{G}_R0_C3D10_fixed';work=package/'work'
  acceptance=json.loads((work/(name+'.acceptance.json')).read_text());diag=json.loads((work/(name+'.diagnostics.json')).read_text(encoding='utf-8-sig'));expected=json.loads((package/(name+'.expected.json')).read_text());exe=json.loads((package/'execution.json').read_text(encoding='utf-8-sig'))
  assert acceptance['status']=='ok' and exe['exit_code']==0
  assert hashlib.sha256((package/(name+'.inp')).read_bytes()).hexdigest()==expected['input_sha256']==acceptance['input_sha256']
  assert hashlib.sha256((package/(name+'.mesh.npz')).read_bytes()).hexdigest()==expected['mesh_sha256']
  assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (work/(name+'.sta')).read_text()
  for suffix in ['acceptance.json','diagnostics.json','sta','msg','dat','extract.txt','console.txt']:
   file=work/(name+'.'+suffix)
   if file.exists():shutil.copy2(file,S/f'{label}_G{G}.{suffix}')
  shutil.copy2(package/'execution.json',S/f'{label}_G{G}.execution.json')
  ref[G]=abs(acceptance['measured']['macro_RF'][2])/.01
  aba.append({'label':label,'G':G,'K':ref[G],'seconds':exe['seconds'],'diagnostics':diag,'elements':expected['elements'],'mesh_volume':expected['geometry']['volume'],'minimum_detJ':expected['geometry']['min_detJ'],'package':str(package),'files_sha256':{n:hashlib.sha256((work/(name+'.'+n)).read_bytes()).hexdigest() for n in ['odb','sta','dat','msg']}})
 volume=estimate(c=c);(S/f'{label}.binary_volume.json').write_text(json.dumps(volume,indent=2)+'\n')
 grid=abs(ks[48]-ks[64])/ks[64];rg=abs(ref[32]-ref[48])/ref[48];err=abs(ks[64]-ref[48])/ref[48];md=abs(km-ks[64])/ks[64];me=abs(km-ref[48])/ref[48]
 checks={'background_grid<=1pct':grid<=.01,'reference_grid<=1pct':rg<=.01,'analytic_vs_binary<=2pct':err<=.02,'mapped_vs_analytic<=1pct':md<=.01,'mapped_vs_binary<=2pct':me<=.02}
 rows.append({'label':label,'c':c,'K_N48':ks[48],'K_N64':ks[64],'K_M64_N64':km,'K_G32':ref[32],'K_G48':ref[48],'background_relative_change':grid,'reference_relative_change':rg,'analytic_binary_relative':err,'mapping_relative':md,'mapped_binary_relative':me,'vf_analytic_N64':a[64]['vf_int'],'vf_mapped_N64':mapped['vf_int'],'vf_binary_estimate':volume['mean'],'checks':checks,'analytic_anchor_passed':grid<=.01 and rg<=.01 and err<=.02,'array_anchor_passed':all(checks.values())})
(S/'summary.json').write_text(json.dumps({'anchors':rows,'abaqus':aba,'all_consistency_passed':True,'quality_limits':'Only global response; distorted elements and tiny positive Jacobians retained, not rigorous error bounds'},indent=2)+'\n');print(json.dumps(rows,indent=2))
ledger=json.loads((O/'execution.json').read_text())
for r in aba:
 case=f"{r['label']}_G{r['G']}"
 if not any(e['kind']=='abaqus' and e['case']==case for e in ledger):ledger.append({'stage':2,'case':case,'kind':'abaqus','seconds':r['seconds'],'exit_code':0})
(O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
