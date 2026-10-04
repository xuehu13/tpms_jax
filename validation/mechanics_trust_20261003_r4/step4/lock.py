"""Lock the original step4 scope and protect prior evidence before solving."""
from pathlib import Path
import hashlib,json,subprocess
W=Path(__file__).resolve().parents[2];P=Path('/home/xuehu/projects/tpms_jax')
O=P/'validation/mechanics_trust_20261003_r4/step4';O.mkdir(exist_ok=False)
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus')
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def dump(f,v):f.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
files=list(P.glob('*.py'))+list((P/'scripts').glob('*'))+list((P/'tests').glob('*.py'))+[P/'pixi.toml',P/'pixi.lock',P/'results/m4_numerical_study.csv']
for n in ('near_term_20261003','geometry_interface_20261003_r2','learning_bridge_20261003_r3','mechanics_trust_20261003_r4/step1','mechanics_trust_20261003_r4/step2','mechanics_trust_20261003_r4/step3'):
    files += [f for f in (P/'validation'/n).rglob('*') if f.is_file()]
files=[f for f in files if f.is_file()]
old=json.loads((P/'validation/mechanics_trust_20261003_r4/step3/verification.json').read_text())
dump(O/'preservation_before.json',{'HEAD':subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),'sha256':{str(f):sha(f) for f in files}})
refs={}
for g in (32,48):
    package=A/'binary_gyroid_20261002';tag=f'binary_gyroid_G{g}_R0_C3D10_fixed'
    refs[str(g)]={'case':tag,'package':str(package),'sha256':{str(package/(tag+suffix)):sha(package/(tag+suffix)) for suffix in ('.inp','.mesh.npz','.expected.json')}}
plan={'date':'2026-10-04','stage':'round4_step4_locked_before_FEM',
 'question':'Finite-strain credibility and practical limit of the already verified medium-wall Gyroid background model',
 'geometry':{'family':'gyroid','c':.541062,'L':1.,'target':'abs(G)<=c','map':'density at actual reference-configuration Gauss points','beta':40.,'eta':1e-4,'no_geometry_calibration':True},
 'material':{'E0':10.,'nu0':.3,'W':'[eta+(1-eta)*rho(X)] * W_step3(F)','mu0':10/2.6,'kappa0':10/(3*.4),'occupancy_fixed_in_reference':True,'no_J_clipping_or_stabilization':True},
 'loading':{'H':'diag(0,0,-a)','XY_periodic':True,'flat_axial_faces':True,'macro_lateral':'fixed','no_contact_plasticity':True},
 'reference':refs,'reference_reuse':'Copy existing audited actual input/mesh; replace only material and load/output steps in new packages; no remeshing',
 'preflight':{'N':[16,32],'path':[0.,.0001,.005,.01,.015,.02,.025,.03,.035,.04,.045,.05],
     'purpose':'implementation, soft-domain distortion, measured cost only; not precision certification',
     'failure_diagnostic':'At most one same-N half-increment restart to distinguish loading/Newton overshoot; cannot certify fine-grid accuracy'},
 'precision':{'JAX_N':[48,64],'increment':.005,'first_limit':.05,'half_increment_N':64,'reference_G':[32,48],
     'conditional_extension':'.10 only after 0-.05 passes all checks; .20 only after .10, contact and instability branch evidence pass',
     'extension_increment':.005,'max_extension_paths':2},
 'criteria':{'force_path_difference_over_reference_max':.05,'energy_path_difference_over_reference_max':.05,
     'background_grid_path_change':.01,'reference_grid_path_change':.01,'half_increment_path_change':.01,
     'work_energy_gap':.01,'small_strain_secant_vs_old_linear':.01,
     'reduced_equilibrium_l2':1e-8,'reaction_balance_or_Piola_absolute':1e-8,'constraint_error':1e-10,
     'Abaqus_constraint_error':1e-7,'positive_detF':True,'soft_domain_J_min_stop':.1,
     'sample_mode_RMS_difference_over_compression':.05,'void_energy_fraction_trigger':.01},
 'budget':{'preflight_paths':2,'same_N_half_increment_diagnostic_max':1,'precision_paths_max':3,'conditional_extension_paths_max':2,
     'Abaqus_analyses_max':5,'datachecks_max':5,'JAX_host_RSS_limit_GiB':12.5,'JAX_GPU_used_limit_GiB':7.2,
     'host_available_floor_GiB':1.5,'per_JAX_path_timeout_s':1800,'per_state_timeout_s':300,
     'Newton_corrections_per_state_max':15,'linear_KSP_iterations_max':3000,
     'Abaqus_each_timeout_s':3600,'Abaqus_cpus':2,'Abaqus_memory':'8gb',
     'one_void_sensitivity_diagnostic_max':1,'maintenance_tests_separate':True,'training_design_gradient':0},
 'solver':{'Newton':'installed JAX-FEM, reduced-coordinate continuation','linear':'existing PETSc GMRES/GAMG; rtol1e-11, atol1e-13, error-if-unconverged',
     'globalization':'no new line-search/arc-length/contact solver; invalid trials retained as failures, one smaller-increment diagnostic allowed'},
 'stop':'Reference/precision resources infeasible, repeated invalid or severe soft-domain detF, nonconvergence, grid/step disagreement, instability or contact require boundary conclusion; no automatic stabilization or training',
 'literature':{'JAX_example':'https://deepmodeling.github.io/jax-fem/learn/hyperelasticity/example.html',
     'soft_domain_instability':'https://orbit.dtu.dk/en/publications/interpolation-scheme-for-fictitious-domain-techniques-and-topolog/'}}
dump(O/'plan.json',plan);(O/'lock.py').write_bytes(Path(__file__).read_bytes())
for root in (W,P/'docs'):
    f=root/'RESEARCH_PLAN.md';s=f.read_text();s=s.replace('第4步待锁定输入/预算、尚未启动','第4步输入/参考/预算已锁定，资源与畸变预检查执行中')
    key='## 第4步：对一个已验证Gyroid探查有限应变压缩的可信上限\n';assert key in s
    note='\n**2026-10-04已锁定并执行中：** c=0.541062、β=40、η=10^-4，初始E0=10/ν0=0.3；能量按`[η+(1−η)ρ(X)]W(F)`插值，ρ保持在初始Gauss坐标，沿用XY周期/横向宏观固定和平面z端面。先N16/N32至5%作实现/畸变/资源预检查，最多一项同N增量减半诊断；预检查不作精度结论。通过后才进入N48/N64、N64半步和旧G32/G48二值网格的独立新有限应变路径。5%/10%/条件20%顺序不变，先满足前段判据再扩展。资源、路径、指标和严格停止条件见正式`step4/plan.json`；不重建薄壁参考、不增加接触/稳定化/训练。\n'
    f.write_text(s.replace(key,key+note))
print(json.dumps({'status':'step4_locked','prior_files':len(files),'reference_meshes':[32,48],'FEM_runs_yet':0},indent=2))
