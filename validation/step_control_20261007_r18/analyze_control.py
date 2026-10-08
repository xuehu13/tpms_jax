"""Read-only analysis of the single r18 attempt; never advances a state."""
from pathlib import Path
import ast, copy, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/home/xuehu/projects/tpms_jax')
O=R/'validation/step_control_20261007_r18'
D=O/'controlled_forward'; A=O/'analysis'
assert (O/'forward_receipt.json').exists(), 'Wait for the one forward attempt to end'
A.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
path=read(D/'accepted_path.json'); bounds=read(D/'stability_path.json')
old=read(R/'validation/mechanism_20261007_r17/original_diagnostic/accepted_path.json')
shell=read(R/'validation/geometry_transfer_20261006_r15/abaqus/explicit_T0p040/results/shell.json')
sp=shell['force_path']; receipt=read(O/'forward_receipt.json')
completed=(D/'result.json').exists(); failure=read(D/'failure.json') if (D/'failure.json').exists() else None
assert len(path)==len(bounds)
t=np.array([p['time'] for p in path]); a=np.array([p['compression'] for p in path])
f=-np.array([p['Fz_N'] for p in path]); k=np.array([p['KE_over_U'] for p in path])
dt=np.array([p['dt_seconds'] for p in path]); rb=np.array([b['R_s_minus2'] for b in bounds])
safe=np.array([b['safe_dt_seconds'] for b in bounds]); nominal=read(D/'input.json')['initial_dt_seconds']
T=read(D/'input.json')['load_time_seconds'] if 'load_time_seconds' in read(D/'input.json') else .004
assert np.all(np.diff(t)>0) and np.all(dt>0)
assert all(p['J_finite'] and not p['invalid_material_points'] for p in path)
end_metric=dt[1:]*np.sqrt(rb[1:]); start_metric=dt[1:]*np.sqrt(rb[:-1])
assert max(end_metric)<=2+1e-12
assert max(start_metric)<=1.6+1e-12
assert np.all(np.diff(dt[1:])<=1e-18)
state_file=D/('field.npz' if completed else 'last_valid_field.npz')
state=np.load(state_file)
assert np.isfinite(state['q']).all()
assert abs(float(state['time'])-t[-1])<1e-12
assert abs(float(state['dt'])-dt[-1])<1e-18
weights=np.r_[0,np.diff(t)]; mask=(a>=.01)&(t<=T+1e-12)
fraction=float(np.sum(weights[mask]*(k[mask]<=.05))/np.sum(weights[mask])) if mask.any() else None
work=float(np.sum(.5*(f[:-1]+f[1:])*10*np.diff(a)))
energy_gap=abs(path[-1]['energy_N_mm']+path[-1]['KE_N_mm']-path[0]['energy_N_mm']-path[0]['KE_N_mm']-work)/max(abs(work),1e-30)
sa=np.array([p['compression'] for p in sp]); sf=-np.array([p['Fz_N'] for p in sp])
oa=np.array([p['compression'] for p in old]); of=-np.array([p['Fz_N'] for p in old])
table=[]
for x in [.01,.05,.10,.12,.14,.15,.16,.18,.20]:
    if x>a.max()+1e-12:continue
    fv=float(np.interp(x,a,f)); sv=float(np.interp(x,sa,sf))
    ov=float(np.interp(x,oa,of)) if x<=oa.max() else None
    table.append({'compression':x,'controlled_N':fv,'shell_N':sv,
        'relative_shell_difference':(fv-sv)/abs(sv),'original_JAX_N':ov,
        'relative_original_JAX_difference':None if ov is None else (fv-ov)/abs(ov)})
metrics={'complete_20pct_and_hold':completed,'stop':failure and {k:failure[k] for k in ('type','message')},
    'receipt':receipt,'accepted_observations':len(path),'accepted_final':path[-1],
    'min_required_J_over_accepted_endpoints':min(p['required_positive_J_min'] for p in path),
    'max_accepted_end_dt_sqrt_R':float(max(end_metric)),
    'max_accepted_start_dt_sqrt_R':float(max(start_metric)),
    'dt_nominal_s':nominal,'minimum_accepted_dt_s':float(min(dt[1:])),
    'minimum_accepted_dt_fraction':float(min(dt[1:])/nominal),
    'loading_low_KE_time_fraction_observed_range':fraction,
    'loading_high_KE_time_seconds':float(np.sum(weights[mask]*(k[mask]>.05))),
    'input_work_N_mm_observed_range':work,'energy_work_gap_observed_range':energy_gap,
    'common_compression_force_table':table,'new_abaqus_jobs':0,'new_design_AD_jobs':0,
    'reference_quality_checks':shell['checks'],
    'scope':'Observed accepted endpoints only; partial range is not full 20% certification. Shell quality failures remain.'}
if completed:
    hold=t>=1.05*T; grid=np.linspace(.001,.2,1000)
    rms=float(np.sqrt(np.mean((np.interp(grid,a,f)-np.interp(grid,sa,sf))**2)))
    fixed=read(R/'validation/thickness_range_20261006_r13/input.json')['gates']['fixed_curve_denominator_N']
    peak=float(max(sf)); holdf=float(np.mean(f[hold])); shellhold=-shell['hold_mean_Fz_N']
    metrics['full_metrics']={'hold_mean_force_N':holdf,'hold_force_relative_shell_difference':abs(holdf-shellhold)/abs(shellhold),
        'RMS_force_N':rms,'original_fixed_denominator_N':fixed,'RMS_over_original_fixed_denominator':rms/fixed,
        'matched_shell_peak_denominator_N':peak,'RMS_over_matched_shell_peak':rms/peak,
        'input_work_relative_shell_difference':abs(work-shell['macro_work_from_RF_U_N_mm'])/abs(shell['macro_work_from_RF_U_N_mm']),
        'loading_low_KE_time_fraction':fraction,'terminal_KE_over_U':float(k[-1]),'energy_work_gap':energy_gap}
else:
    metrics['full_metrics']=None
write(A/'controlled_analysis.json',metrics)

# Verify the default block equations exactly, after removing the optional dt interface.
def methods(tree):
    return {n.name:n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='ExplicitXYZ' for n in c.body if isinstance(n,ast.FunctionDef)}
current=methods(ast.parse((R/'scripts/thin_target_explicit.py').read_text()))
before=methods(ast.parse((O/'entry_before.py').read_text()))
class RestoreDefault(ast.NodeTransformer):
    def visit_FunctionDef(self,n):
        self.generic_visit(n)
        if n.args.args and n.args.args[-1].arg=='step_dt':n.args.args.pop(); n.args.defaults.pop()
        return n
    def visit_Assign(self,n):
        if any(isinstance(x,ast.Name) and x.id=='local_dt' for x in n.targets):return None
        return self.generic_visit(n)
    def visit_Name(self,n):
        return ast.copy_location(ast.Name(id='dt',ctx=n.ctx),n) if n.id=='local_dt' else n
    def visit_Call(self,n):
        self.generic_visit(n);n.keywords=[x for x in n.keywords if x.arg!='step_dt'];return n
normalized=RestoreDefault().visit(copy.deepcopy(current['block']))
assert ast.dump(normalized,include_attributes=False)==ast.dump(before['block'],include_attributes=False)
assert all(ast.dump(current[k],include_attributes=False)==ast.dump(v,include_attributes=False) for k,v in before.items() if k!='block')
frozen=read(O/'frozen_before.json')
if isinstance(frozen,dict) and 'files' in frozen:frozen=frozen['files']
assert isinstance(frozen,dict), type(frozen)
changed=[]
for name,value in frozen.items():
    p=Path(name) if Path(name).is_absolute() else R/name
    expected=value if isinstance(value,str) else value['sha256']
    if sha(p)!=expected:changed.append(str(p))
assert not changed,changed
write(A/'verification.json',{'frozen_files_checked':len(frozen),'frozen_files_changed':changed,
    'other_ExplicitXYZ_methods_AST_unchanged':True,'default_block_AST_exactly_unchanged_after_removing_optional_runtime_dt':True,
    'accepted_endpoint_frequency_limits_passed':True,'actual_terminal_state_dt_and_time_match_path':True,
    'scientific_results_sha256':{str(p.relative_to(R)):sha(p) for p in [state_file,D/'accepted_path.json',D/'stability_path.json',A/'controlled_analysis.json']}})

fig,ax=plt.subplots(3,1,figsize=(9,10),sharex=True)
ax[0].plot(sa*100,sf,label='Existing Abaqus shell (quality gates failed)',color='black')
ax[0].plot(oa*100,of,'--',label='Original JAX accepted endpoints',color='tab:gray')
ax[0].plot(a*100,f,label='State-bound JAX accepted endpoints',color='tab:blue')
ax[0].set_ylabel('Compression force (N)');ax[0].legend(fontsize=8);ax[0].grid(alpha=.25)
ax[1].plot(a[1:]*100,dt[1:]/nominal,label='Actual dt / initial dt')
ax[1].plot(a*100,safe/nominal,':',label='Frozen-state safe bound / initial dt')
ax[1].axhline(1/16,color='red',ls='--',label='Unchanged minimum safeguard')
ax[1].set_yscale('log');ax[1].set_ylabel('Time-step fraction');ax[1].legend(fontsize=8);ax[1].grid(alpha=.25)
ax[2].plot(a*100,k*100,label='KE / internal energy (%)')
ax[2].axhline(5,color='red',ls='--',label='Low-inertia criterion')
ax[2].set_ylim(0,max(10,min(50,float(max(k[a>=.01])*100)*1.1)))
ax[2].set_ylabel('Energy ratio (%)');ax[2].set_xlabel('Macroscopic compression (%)');ax[2].legend(fontsize=8);ax[2].grid(alpha=.25)
fig.suptitle('diverse_04: one controlled attempt; accepted endpoints, not full certification')
fig.tight_layout();fig.savefig(A/'controlled_response.png',dpi=170);plt.close(fig)
print(json.dumps(metrics,indent=2,ensure_ascii=False))
