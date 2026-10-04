"""Lock finite-strain full-solid benchmark before any new FEM solve."""
from pathlib import Path
import hashlib,json,subprocess
W=Path(__file__).resolve().parents[2];P=Path('/home/xuehu/projects/tpms_jax')
O=P/'validation/mechanics_trust_20261003_r4/step3';O.mkdir(exist_ok=False)
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
files=list(P.glob('*.py'))+list((P/'scripts').glob('*'))+list((P/'tests').glob('*.py'))+[P/'pixi.toml',P/'pixi.lock',P/'results/m4_numerical_study.csv']
for name in ('near_term_20261003','geometry_interface_20261003_r2','learning_bridge_20261003_r3','mechanics_trust_20261003_r4/step1','mechanics_trust_20261003_r4/step2'):
    files += [f for f in (P/'validation'/name).rglob('*') if f.is_file()]
files=[f for f in files if f.is_file()]
dump=lambda f,v:f.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
dump(O/'preservation_before.json',{'HEAD':subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),'sha256':{str(f):sha(f) for f in files}})
mu=10/2.6;kappa=10/(3*.4)
plan={'date':'2026-10-04','stage':'round4_step3_locked_before_FEM','question':'Correct finite-strain kinematics/material/PBC/equilibrium/energy extraction, isolated from TPMS interfaces and voids',
    'material':{'E0':10.,'nu0':.3,'mu':mu,'kappa':kappa,'C10':mu/2,'D1':2/kappa,
                'W_reference':'mu/2*(J^(-2/3)*tr(F.T F)-3)+kappa/2*(J-1)^2','J':'det(F)>0','temperature_or_plasticity':False},
    'model':{'L':1.,'reference_volume':1.,'initial_axial_area':1.,'full_solid':True,'void_or_contact':False,'H':'diag(0,0,-compression)',
             'periodic_axes':[0,1],'macro_lateral':'fixed','flat_axial_faces':True,'free_fluctuations':'all unconstrained interior/side components; no full affine Dirichlet field'},
    'JAX_paths':[{'N':2,'increment':.01},{'N':4,'increment':.01},{'N':4,'increment':.005}],
    'Abaqus_paths':[{'N':4,'increment':.01},{'N':4,'increment':.005}],
    'path':'zero, 0.0001 small-strain point, then all uniform increments to 0.20; base22 / half42 stored states',
    'solver':'installed JAX-FEM Newton + scipy sparse linear solver; XY projection and pins reused; reduced-coordinate initial guess',
    'seed':'nonzero constraint-compatible fluctuation at zero and 20% to exercise Newton; subsequent points use continuation; no new solver implementation',
    'reference':'native full-integration C3D8, N4, NLGEOM=YES; NEO HOOKE constants exactly matched; one direct increment per named load step',
    'outputs':['uncorrected top/bottom nodal reaction','mean first Piola stress per reference volume','mean Cauchy stress per current volume','stored energy per reference volume','min/max detF','full physical displacement','reduced equilibrium','constraint errors','cost'],
    'criteria':{'path_force_difference_over_reference_max':.01,'path_energy_difference_over_reference_max':.01,'grid_or_step_path_change':.001,
                'work_vs_stored_energy_relative_final':.001,'small_strain_tangent_relative':.001,'JAX_analytic_force_energy_relative_max':1e-7,
                'reduced_residual_l2':1e-9,'JAX_constraint_error':1e-10,'Abaqus_displacement_error':1e-7,'positive_detF':True},
    'budget':{'scientific_JAX_paths':3,'equilibrium_states_including_zero':86,'Abaqus_analyses':2,'datachecks':2,
              'JAX_path_timeout_s':180,'Abaqus_each_timeout_s':600,'Abaqus_cpus':2,'Abaqus_memory':'2gb','target_background_N_gt4':False,
              'maintenance_checks_counted_separately':True,'TPMS_or_training_runs':0},
    'stop':'No TPMS finite-strain claim before full-solid checks pass. Preserve failed runs; repair implementation only, no material fitting/relaxed thresholds/extra sweep',
    'sources':{'JAX_FEM':'https://deepmodeling.github.io/jax-fem/learn/hyperelasticity/example.html','Abaqus_energy':'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-hyperelastic.htm'}}
dump(O/'plan.json',plan);(O/'lock.py').write_bytes(Path(__file__).read_bytes())
for root in (W,P/'docs'):
    f=root/'RESEARCH_PLAN.md';text=f.read_text()
    text=text.replace('第3步待锁定输入，第3/4步尚未启动','第3步输入/预算已锁定并执行中，第4步尚未启动')
    key='## 第3步：先在完整均匀实体上验证有限应变实现\n'
    assert key in text
    note='\n**2026-10-04已锁定并执行中：** 可压缩Neo-Hookean，E0=10、ν0=0.3，C10=1.923076923076923、D1=0.24；横向宏观固定、XY周期、平面z端面。N2/N4背景、1%增量及N4的0.5%增量，共3条路径；Abaqus N4原生C3D8作两种增量路径，共2项分析及2项datacheck。均压至20%，另含0.01%小应变点；使用非零初始扰动检查Newton，路径功不套用线性公式。锁定输入/指标/预算在正式`validation/mechanics_trust_20261003_r4/step3/plan.json`。本步无TPMS/孔隙/接触或训练，薄壁停止项不启动。\n'
    f.write_text(text.replace(key,key+note))
print(json.dumps({'stage':'step3_inputs_locked','mu':mu,'kappa':kappa,'C10':mu/2,'D1':2/kappa,'new_FEM_runs_yet':0},indent=2))
