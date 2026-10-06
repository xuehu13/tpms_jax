"""Step 1 only: new physical case metadata and a minimal existing-entry adapter."""
from pathlib import Path
import ast,hashlib,json,shutil,subprocess
R=Path('/home/xuehu/projects/tpms_jax')
O=R/'validation/geometry_transfer_20261006_r15'
D=Path(__file__).resolve().parent
base='7e9c642be67ebdddcd44cdabb68fe86b7a4822e8'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==base
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
assert not O.exists()
O.mkdir();(O/'input').mkdir();(O/'tools').mkdir()
tracked=subprocess.check_output(['git','ls-files'],cwd=R,text=True).splitlines()
write(O/'before_manifest.json',{k:sha(R/k) for k in tracked})
inventory=json.loads((R/'validation/forward_scope_20261006_r14/geometry_inventory.json').read_text())['diverse_04']
src=Path(inventory['source_directory'])
assert sha(src/'case_used.json')==inventory['case_sha256']
assert sha(src/'abaqus/ingredients/shell_mesh.inc')==inventory['mesh_sha256']
assert inventory['basic_periodic_surface_screen_pass']
for s in ['case_used.json','abaqus/ingredients/shell_mesh.inc']:
    shutil.copy2(src/s,O/'input'/Path(s).name)
cfg={'case_id':'diverse04_t005','N':32,'cell_size_mm':10.,'thickness_mm':.5,
 'E_MPa':10.,'nu':.3,'eta':1e-4,'interface_10_90_mm':.05,
 'solid_density_tonne_per_mm3':1e-9,'mechanical_periodic_axes':[0,1,2],
 'target_compression':.2,'hold_fraction':.1,'material_model':'objective_void',
 'element_degree':2,'quadrature_order':4,'Gauss_points_per_cell':27,
 'JAX_load_time_seconds':.004,'shell_load_time_seconds':.040,
 'base_commit':base,'main_plan_step':1,
 'research_question':'Can the fixed candidate transfer to another existing periodic midsurface without retuning physical or numerical constants?',
 'scientific_scope':'Matched forward input readiness only; no displacement solve or derivative certification.',
 'only_physical_change':'Original midsurface diverse_28 -> diverse_04; physical thickness/material/boundary fixed',
 'reused_screen':inventory,'original_source_sha256':{s:sha(src/s) for s in ['case_used.json','abaqus/ingredients/shell_mesh.inc']},
 'gates':{'hold_force_difference':.1,'work_difference':.1,'legacy_RMS_denominator_N':2.2589142322540283,
  'legacy_RMS_limit':.1,'new_geometry_RMS_over_own_shell_loading_peak_limit':.1,
  'JAX_energy_work_gap':.01,'terminal_KE_over_U':.05,'loading_fraction_KE_below_5pct':.95,'shell_energy_drift':.01}}
write(O/'input.json',cfg)
p=R/'scripts/thin_target_explicit.py';s=p.read_text();original=s
old="        source=json.loads((a.case/'step2/diagnostic_xyz/input.json').read_text())"
new="""        source_path=getattr(a,'case_input',None) or a.case/'step2/diagnostic_xyz/input.json'
        source=json.loads(source_path.read_text())
        if getattr(a,'case_input',None) is not None:
            # Direct physical metadata for a new case; the solver's fixed
            # material/unit constants must actually match the declared input.
            mu=source['E_MPa']/(2*(1+source['nu']))
            kappa=source['E_MPa']/(3*(1-2*source['nu']))
            if (source['cell_size_mm']!=10. or not math.isclose(mu,MU,rel_tol=1e-12)
                    or not math.isclose(kappa,KAPPA,rel_tol=1e-12)
                    or source.get('solid_density_tonne_per_mm3',1e-9)!=1e-9):
                raise ValueError('Case metadata differs from the fixed material/unit/density constants')"""
assert s.count(old)==1;s=s.replace(old,new)
old="        cfg.update(source_linear_case_input_sha256=sha(a.case/'step2/diagnostic_xyz/input.json'),"
new="""        if getattr(a,'case_input',None) is None:
            cfg['source_linear_case_input_sha256']=sha(source_path)
        else:
            cfg.update(source_case_input_path=str(source_path),source_case_input_sha256=sha(source_path))
        cfg.update("""
assert s.count(old)==1;s=s.replace(old,new)
old="    p.add_argument('--output',type=Path,required=True);p.add_argument('--load-time',type=float,default=.02)"
new="    p.add_argument('--case-input',type=Path,help='Direct physical case metadata JSON; otherwise preserve the legacy linear-case source')\n"+old
assert s.count(old)==1;s=s.replace(old,new)
compile(s,str(p),'exec')
kernel=lambda text:ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ExplicitXYZ'))
assert kernel(original)==kernel(s)
p.write_text(s)
write(O/'entry_adapter.json',{'changed_file':str(p),'source_before_sha256':hashlib.sha256(original.encode()).hexdigest(),
 'source_after_sha256':sha(p),'change':'Optional direct physical case metadata and matching of fixed material/unit constants; legacy default preserved.',
 'material_mass_time_stepping_changed':False,'explicit_class_AST_unchanged':True,'new_FEM_implementation':False})
cfg['solver_source_sha256']={k:sha(R/k) for k in ['hyperelastic_fem.py','scripts/thin_target_explicit.py','surface_distance.py','pbc.py','fem.py','pixi.lock']}
write(O/'input.json',cfg)
shutil.copy2(__file__,O/'tools/setup.py')
print(json.dumps({'directory':str(O),'case':cfg['case_id'],'entry_adapter_only':True,'no_mechanics_jobs':True}))
