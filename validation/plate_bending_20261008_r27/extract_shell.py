"""Read one simple Standard bending ODB, never alter the model."""
from pathlib import Path
import json,sys,math
from odbAccess import openOdb
odb=openOdb(str(Path('plate.odb').resolve()),readOnly=True)
try:
 cfg=json.loads(Path('input.json').read_text());frame=odb.steps['SMALL_BENDING'].frames[-1];M=cfg['M_N_mm']
 def field(name):return {v.nodeLabel:[float(x) for x in v.data] for v in frame.fieldOutputs[name].values}
 u=field('U');ur=field('UR');rf=field('RF');rm=field('RM')
 Q=sum(p['weight']*ur[p['node']][1] for p in cfg['tip_weights'])
 tip_w=sum(p['weight']*u[p['node']][2] for p in cfg['tip_weights'])
 labels=list(range(1,cfg['ny']+2));assert all(n in rf and n in rm for n in labels), 'Declared root node labels missing from fields'
 reaction=[sum(rf[n][i] for n in labels) for i in range(3)];moment=[sum(rm[n][i] for n in labels) for i in range(3)]
 energy={}
 for reg in odb.steps['SMALL_BENDING'].historyRegions.values():
  for key,v in reg.historyOutputs.items():
   if key in ['ALLSE','ALLIE','ALLAE','ALLWK']:energy[key]=float(v.data[-1][1])
 C=Q/M;relative=C/cfg['analytic_compliance_per_N_mm']-1
 checks={'analytic_reference_5pct':abs(relative)<=.05,
   'reaction_moment':abs(moment[1]/M+1)<=1e-5,
   'reaction_force':math.sqrt(sum(x*x for x in reaction))/(M/cfg['L_mm'])<=1e-5,
   'artificial_energy':abs(energy.get('ALLAE',float('inf')))/max(energy.get('ALLSE',0),1e-30)<=.01,
   'finite_positive_compliance':math.isfinite(C) and C>0}
 result={'Q_rad':Q,'compliance_per_N_mm':C,'effective_EI_N_mm2':M*cfg['L_mm']/Q,
  'tip_average_uz_mm':tip_w,'reference_relative_compliance':relative,
  'root_reaction_force_N':reaction,'root_reaction_moment_N_mm':moment,'energies_N_mm':energy,
  'checks':checks,'reference_gate_pass':all(checks.values()),'scope':'one independent linear static S4R reference; no finite-strain path'}
 Path('shell.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
finally:odb.close()
