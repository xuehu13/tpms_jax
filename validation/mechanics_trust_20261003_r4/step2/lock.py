from pathlib import Path
import hashlib,json

P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
assert json.loads((O/'regression.json').read_text())['exit_code']==0
script=P/'scripts/run_abaqus_binary.ps1';text=script.read_text()
assert '[switch]$QualityDiagnostics' not in text
text=text.replace('[switch]$ExtractOnly','[switch]$ExtractOnly,\n [switch]$QualityDiagnostics')
needle='  Write-Output "$case passed physical consistency; convergence assessed separately"'
assert needle in text
text=text.replace(needle,'''  if($QualityDiagnostics){
   & $AbaqusCommand python (Join-Path $root 'scripts\\abaqus_mesh_quality.py') ($case+'.odb') --expected $expectedPath --out ($case+'.quality.json') --nodal-out ($case+'.nodal.npz') 2>&1 | Tee-Object -FilePath ($case+'.quality.console.txt')
   if($LASTEXITCODE -ne 0){throw 'Quality extraction failed'}
  }
'''+needle)
script.write_text(text)
script=P/'scripts/prepare_abaqus_binary.py'
script.write_text(script.read_text().replace('Generate binary sheet-Gyroid C3D10 inputs.','Generate binary Gyroid/Primitive sheet-solid C3D10 inputs.'))
definitions=json.loads((O/'geometry_precheck.json').read_text())
plan={'date':'2026-10-04','stage':'round4_step2_inputs_locked_before_mechanics',
      'cases':[{'label':d['label'],'family':d['family'],'c':d['c']} for d in definitions],
      'material':{'E_s':10.,'nu':.3,'emin_ratio':1e-4,'interpolation_power':1},
      'loading':{'L':1.,'eps_z':-.01,'periodic_axes':[0,1],'lateral':'fixed','flat_axial_faces':True},
      'background':{'N':[48,64],'beta':40,'input':'direct analytic at physical Gauss points','solver':'PETSc CG/GAMG'},
      'reference':{'G':[32,48],'element':'C3D10','void':'absent','fe_refinement':0,'surface':'piecewise-planar approximation of the same signed thresholds'},
      'budget':{'base_background_solves':4,'base_Abaqus_analyses':4,'datachecks':4,
                'conditional_extra_background_solves_max':1,'conditional':'If reference valid but global stiffness precision fails, one N64 Emin/Es=1e-5 test on the first failing case to distinguish soft-void contribution',
                'no_N96_or_parameter_sweep':True,'time_limit_each_forward_s':1200,'time_limit_each_geometry_or_Abaqus_s':1800,'Abaqus_cpus':2,'Abaqus_memory':'8gb'},
      'criteria':{'background_vs_reference_K_relative':.02,'background_grid_relative':.01,'reference_G32_G48_relative':.01,
                  'mode_check':'aligned translation, representative displacement/warp comparison; normalized RMS and correlation are diagnostics, visible incompatible modes prevent acceptance',
                  'reference_quality':'positive mapping and valid topology mandatory; report distortion energy/volume shares; no rigorous/local-stress claim'},
      'stop':'Invalid fine reference stops that geometry before mechanics; failed global precision is a reported limit, not a reason to fit c/beta/material',
      'precheck_sha256':hashlib.sha256((O/'geometry_precheck.json').read_bytes()).hexdigest(),
      'source_sha256':{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(P.glob('*.py'))+list((P/'scripts').glob('*.py'))+list((P/'scripts').glob('*.ps1'))}}
(O/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(O/'lock.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'locked_cases':plan['cases'],'base_background_solves':4,'base_Abaqus_analyses':4,'extra_diagnostic_max':1},indent=2))
