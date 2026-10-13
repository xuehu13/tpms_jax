"""Read saved clocks and source text; no FEM, ODB access, or solver imports."""
from pathlib import Path
import ast,hashlib,json,math,shutil

R=Path('/home/xuehu/projects/tpms_jax')
D=R/'validation/time_logic_audit_20261009_r49'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an/output/r49_time_logic_audit_20261009')

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def smooth(t,T):
    s=min(1.,max(0.,t/T));return .2*(10*s**3-15*s**4+6*s**5)

def main():
    D.mkdir(exist_ok=True);sources={}
    protected=read(R/'validation/shell_rate_convergence_20261009_r48/frozen_before.json')
    for p in (R/'validation/shell_rate_convergence_20261009_r48').rglob('*'):
        if p.is_file():protected[str(p.relative_to(R))]=sha(p)
    for rel in ['scripts/thin_target_explicit.py','hyperelastic_fem.py','pbc.py','fem.py']:
        protected[rel]=sha(R/rel)
    (D/'protected_before.json').write_text(json.dumps(protected,indent=2)+'\n')
    cat=read(R/'validation/simulation_atlas_20261009_r47/curve_catalog.json')
    selected={'04-B32','04-B64','04-J18','04-J64','28-C4','28-C8'}
    clock=[]
    for item in cat:
        if item['id'] not in selected and item['role']!='壳速率':continue
        p=R/item['source'];j=read(p);sources[item['source']]=sha(p)
        rows=j if isinstance(j,list) else j.get('path',j.get('force_path'))
        assert rows and all('time' in x and 'compression' in x for x in rows)
        T=item['T_ms']/1000
        err=max(abs(x['compression']-smooth(x['time'],T)) for x in rows)
        rec={'id':item['id'],'case':item['case'],'solver':item['solver'],'T_seconds':T,
             'records':len(rows),'time_from_seconds':rows[0]['time'],'time_to_seconds':rows[-1]['time'],
             'compression_end':rows[-1]['compression'],'max_compression_fraction_error_vs_shared_smoothstep':err,
             'max_compression_error_percentage_points':100*err,'monotone_time':all(b['time']>a['time'] for a,b in zip(rows,rows[1:])),
             'load_end_T_seconds':T,'load_plus_hold_end_1p1T_seconds':1.1*T,'source':item['source']}
        if 'dt_seconds' in rows[1]:
            counts=[(b['time']-a['time'])/b['dt_seconds'] for a,b in zip(rows,rows[1:])]
            integers=[round(x) for x in counts]
            total=math.fsum(n*b['dt_seconds'] for n,b in zip(integers,rows[1:]))
            rec.update(accepted_steps_reconstructed=sum(integers),block_increment_counts=sorted(set(integers)),
                max_count_distance_to_integer=max(abs(x-round(x)) for x in counts),
                sum_accepted_dt_seconds=total,time_accumulation_error_seconds=total-(rows[-1]['time']-rows[0]['time']),
                dt_min_seconds=min(x['dt_seconds'] for x in rows[1:]),dt_max_seconds=max(x['dt_seconds'] for x in rows[1:]))
        assert err<(1e-6 if item['solver'].startswith('Abaqus') else 1e-12),rec
        assert rec['monotone_time'],rec
        clock.append(rec)
    for tag in ['0p020000','0p080000']:
        b=R/('validation/shell_rate_convergence_20261009_r48/abaqus/explicit_T'+tag)
        p=b/'results/shell.json'
        if not p.exists():p=p.with_name('partial_shell.json')
        j=read(p);T=read(b/'input.json')['load_time_seconds'];sources[str(p.relative_to(R))]=sha(p)
        err=max(abs(x['compression']-smooth(x['time'],T)) for x in j['force_path'])
        assert err<1e-6
        clock.append({'id':'04-S'+str(T*1000),'solver':'Abaqus/Explicit','T_seconds':T,'records':len(j['force_path']),
            'time_to_seconds':j['force_path'][-1]['time'],'compression_end':j['compression'],
            'max_compression_fraction_error_vs_shared_smoothstep':err,'max_compression_error_percentage_points':err*100,
            'source':str(p.relative_to(R))})
    # Parse executed-source motion, rather than importing the FEM module.
    snapshot=R/'validation/binary_n64_20261009_r45/peak_N64/source_at_run/thin_target_explicit.py'
    tree=ast.parse(snapshot.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ExplicitXYZ')
    block=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='block')
    motion=next(n for n in block.body if isinstance(n,ast.FunctionDef) and n.name=='motion')
    class ScalarNP:
        @staticmethod
        def clip(x,a,b):return min(b,max(a,x))
        @staticmethod
        def array(x):return x
    ns={'jnp':ScalarNP,'load_time':.001,'wave':False}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[motion],type_ignores=[])),'frozen_scalar_motion','exec'),ns)
    motion_checks=[]
    for frac in [0.,.25,.5,.75,1.,1.1]:
        t=frac*.001;h,hd,hdd=ns['motion'](t)
        s=min(1.,max(0.,frac));assert abs(-h-smooth(t,.001))<1e-14
        motion_checks.append({'t_seconds':t,'fraction_of_T':frac,'compression':-h,'physical_top_displacement_mm':10*h,
            'physical_top_velocity_mm_per_s':10*hd,'physical_top_acceleration_mm_per_s2':10*hdd})
    controller_checks=[]
    for rel in ['validation/binary_n64_20261009_r45/effective_controller.py','validation/binary_full20_20261008_r44/effective_continuation_controller.py']:
        p=R/rel;s=p.read_text();sources[rel]=sha(p)
        checks={'motion_uses_physical_load_time':'ex.block(nominal,128,a.load_time,False)' in s,
            'runtime_actual_dt_passed':'step_dt=jnp.asarray(local_dt)' in s,
            'half_velocity_recenter_on_dt_change':'prior[1]+.5*(state_dt-local_dt)*acc' in s,
            'accepted_state_only_after_guard':'state=trial;state_dt=local_dt;done+=count' in s,
            'wall_clock_only_budget':'if time.time()>' in s,'tail_dt_targets_physical_end':'local_dt=(end-float(state[2]))/count' in s}
        assert all(checks.values());controller_checks.append({'source':rel,'checks':checks})
    # Actual INP definitions, not filenames or input JSON alone.
    inp=[]
    for rel,T in [('validation/binary_n64_20261009_r45/abaqus/explicit_T0p001000/thin_shell.inp',.001),
                  ('validation/shell_rate_20261007_r20/abaqus/explicit_T0p004/thin_shell.inp',.004)]:
        p=R/rel;lines=p.read_text().splitlines();sources[rel]=sha(p)
        ai=next(i for i,l in enumerate(lines) if l.lower().startswith('*amplitude'))
        di=next(i for i,l in enumerate(lines) if l.lower().startswith('*dynamic'))
        de=next(i for i,l in enumerate(lines) if l.lower().startswith('*density'))
        bi=next(i for i,l in enumerate(lines) if l.lower().startswith('*boundary, amplitude'))
        vals=[float(x.strip()) for x in lines[ai+1].split(',')]
        assert vals==[0.,0.,T,1.,1.1*T,1.] or max(abs(a-b) for a,b in zip(vals,[0.,0.,T,1.,1.1*T,1.]))<1e-15
        assert 'smooth step' in lines[ai].lower() and 'total time' in lines[ai].lower()
        assert float(lines[de+1])==1e-9 and float(lines[bi+1].split(',')[-1])==-2.
        assert sum(l.lower().startswith('*step,') for l in lines)==1
        assert not any(l.lower().startswith(('*mass scaling','*fixed mass scaling','*variable mass scaling')) for l in lines)
        inp.append({'source':rel,'amplitude_keyword':lines[ai],'amplitude_data':vals,'dynamic_period_seconds':float(lines[di+1].split(',')[-1]),
                    'density_tonne_per_mm3':float(lines[de+1]),'prescribed_control_displacement_mm':-2.,'single_step_total_time_zero_origin':True,'mass_scaling_keyword_absent':True})
    # Length normalization: Mphysical = L*Mcode and fint_physical = L^2*fint_code.
    E=10.;nu=.3;rho=1e-9;L=10.;mu=E/(2*(1+nu));K=E/(3*(1-2*nu))
    units={'length_mm':L,'stress_MPa':E,'density_tonne_per_mm3':rho,'time_unit_seconds':True,
        'rho_L_squared_code_mass_prefactor':rho*L**2,'unit_cube_physical_mass_tonne':rho*L**3,
        'longitudinal_wave_speed_physical_mm_per_s':math.sqrt((K+4*mu/3)/rho),
        'longitudinal_wave_speed_normalized_per_s':math.sqrt((K+4*mu/3)/(rho*L**2)),
        'length_normalization_does_not_rescale_time':True,
        'derivation':'X=L*x, U=L*w; Mphys=L*Mcode, fint_phys=L^2*fhat; dividing Mphys*L*wdd=fint_phys by L^2 gives Mcode=rho*L^2*mhat. Time remains seconds.'}
    continuity=read(R/'validation/binary_full20_20261008_r44/continuity_check.json');assert continuity['passed'] and continuity['retained_q_vhalf_time_dt']
    for rel in ['scripts/thin_target_explicit.py','hyperelastic_fem.py',str(snapshot.relative_to(R)),'validation/binary_full20_20261008_r44/continuity_check.json']:
        sources[rel]=sha(R/rel)
    bad=[p for p,h in protected.items() if sha(R/p)!=h];assert not bad,bad
    result={'scope':'Source and saved-history audit only; no new FEM/Abaqus/AD, no trajectories resumed',
        'finding':'Same seconds and same fifth-order smoothstep. No hidden step-count time rescaling found. Dynamic equivalence and time-discretization convergence are NOT certified.',
        'clock_checks':clock,'scalar_motion_from_executed_snapshot':motion_checks,'controller_checks':controller_checks,
        'variable_dt_rule':'Recenter vhalf by (dt_old-dt_new)*acc/2, then add dt_new*acc; together this is (dt_old+dt_new)*acc/2, matching the central-difference variable-increment rule. This algebra does not certify the whole constrained shell/solid dynamics.',
        'official_sources':[
            'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-absunits.htm',
            'https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-t-ampsmooth.htm',
            'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-expdynamic.htm'],
        'INP_checks':inp,'unit_derivation':units,'r44_exact_restart_check_preserved':continuity,
        'important_limits':['Earlier JAX4ms versus shell40ms comparisons used unequal T; labels do not imply same loading rate.',
          'Same physical T need not mean same ratio to bending/buckling periods: background mass/HRZ/void and shell inertia differ.',
          'JAX has no damping; Abaqus retains default bulk viscosity. These are dynamic model differences, not different definitions of seconds.',
          'Stable dt and correct cumulative time do not prove temporal accuracy or capture every instability; no new dt-convergence calculation in this audit.',
          'JAX compression comes from prescribed motion; agreement with that formula is bookkeeping evidence, not independent physical validation. Abaqus history uses extracted control U3.',
          'Observed block endpoints do not certify every internal state. Existing failures, quality flags and full20 AD failure remain unchanged.'],
        'source_sha256':sources,'protected_files':len(protected),'all_protected_unchanged':True,'new_FEM':0,'new_Abaqus':0,'AD':0}
    for p in [D/'audit.json',W/'audit.json']:p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    shutil.copy2(Path(__file__),D/'audit.py') if Path(__file__).resolve()!=D/'audit.py' else None
    print(json.dumps({'clocks':clock,'motion':motion_checks,'units':units,'protected':len(protected)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
