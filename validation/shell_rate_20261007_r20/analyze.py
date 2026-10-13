"""Collect one completed shell job and compare frozen histories/fields; no FEM solve."""
from pathlib import Path
import ast,hashlib,json,shutil,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=R/'validation/shell_rate_20261007_r20';A=D/'analysis';A.mkdir(exist_ok=False)
P=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/shell_rate_20261007_r20_diverse04_explicit_T0p004')
results=D/'abaqus/explicit_T0p004/results';results.mkdir(exist_ok=True)
started=time.perf_counter()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
receipt=json.loads((P/'launch_receipt.json').read_text())
assert receipt['analysis_completed_successfully'] and receipt['shell_modes_exists']
for name in ('launch_manifest.json','launch_receipt.json','launch.log','extraction.log','mode_extraction.log',
             'shell.json','shell_field.npz','thin_shell.sta','thin_shell.dat','thin_shell.msg'):
    if (P/name).exists():shutil.copy2(P/name,results/name)
shutil.copytree(P/'shell_frames',results/'shell_frames',dirs_exist_ok=True)
write(results/'retention.json',{'ODB_retained_native':str(P/'thin_shell.odb'),
    'ODB_sha256':sha(P/'thin_shell.odb'),'ODB_not_duplicated':True,'old_science_unchanged':receipt['old_native_files_unchanged']})
slowfile=R/'validation/geometry_transfer_20261006_r15/abaqus/explicit_T0p040/results/shell.json'
jfile=R/'validation/step_control_20261007_r18/controlled_forward/accepted_path.json'
slow=json.loads(slowfile.read_text());fast=json.loads((results/'shell.json').read_text());j=json.loads(jfile.read_text())
def load(d,T):return [p for p in d['force_path'] if p['time']<=T]
sp=load(slow,.040);fp=load(fast,.004);jp=j
def peak(rows):return max((p for p in rows if p['compression']>=.01),key=lambda p:-p['Fz_N'])
spe,fpe,jpe=map(peak,(sp,fp,jp))
def curve(rows):return np.array([p['compression'] for p in rows]),-np.array([p['Fz_N'] for p in rows])
sx,sf=curve(sp);fx,ff=curve(fp);jx,jf=curve(jp)
shift=fpe['compression']-spe['compression'];oldgap=jpe['compression']-spe['compression'];newgap=jpe['compression']-fpe['compression']
table=[]
for a in (.01,.05,.10,.12,.14,.15,.16,.17,.175):
    table.append({'compression':a,'old_shell_N':float(np.interp(a,sx,sf)),
        'fast_shell_N':float(np.interp(a,fx,ff)),'existing_JAX_N':float(np.interp(a,jx,jf))})
grid=np.linspace(.01,j[-1]['compression'],1000)
def metrics(x,f):
    delta=np.interp(grid,jx,jf)-np.interp(grid,x,f)
    rms=float(np.sqrt(np.mean(delta**2)))
    return {'comparison_compression_range':[float(grid[0]),float(grid[-1])],
        'JAX_not_complete_20pct':True,'RMS_N':rms,'RMS_over_original_fixed_Standard_peak_2p25891N':rms/2.25891,
        'RMS_over_this_matched_shell_own_peak':rms/float(f.max()),
        'scope':'Raw compression alignment; no event shifting; partial dynamic curve, not old 20% acceptance.'}
wg=np.linspace(.170,.175,501)
windows={name:{'compression_range':[.170,.175], 'mean_N':float(np.mean(np.interp(wg,x,f))),
    'min_N':float(np.min(np.interp(wg,x,f))),'max_N':float(np.max(np.interp(wg,x,f))),
    'sampling':'501 uniform compression points, linear interpolation of retained observations; not a settled hold'}
    for name,x,f in (('old_shell',sx,sf),('fast_shell',fx,ff),('JAX_partial',jx,jf))}
comparison={'observed_peaks':{'old_shell':spe,'fast_shell':fpe,'existing_JAX':jpe},
    'shell_rate_peak_shift_percentage_points':100*shift,
    'original_JAX_shell_peak_gap_percentage_points':100*oldgap,
    'remaining_JAX_fast_shell_peak_gap_percentage_points':100*newgap,
    'fraction_original_peak_gap_reduced':shift/oldgap,
    'fraction_not_a_causal_variance_decomposition':True,
    'JAX_peak_force_relative_fast_shell_difference':(-jpe['Fz_N'])/(-fpe['Fz_N'])-1,
    'same_compression_force_table':table,'partial_raw_metrics':{'vs_old_shell':metrics(sx,sf),'vs_fast_shell':metrics(fx,ff)},
    'predefined_loading_window':windows,'shell_hold_means_N':{'old_shell':slow['hold_mean_Fz_N'],'fast_shell':fast['hold_mean_Fz_N']},
    'no_JAX_hold_or20pct_metric':True,'fast_shell_original_checks':fast['checks'],
    'quality':{name:{k:d[k] for k in ('KE_fraction_of_IE_terminal','loading_time_fraction_KE_below_5pct',
        'global_energy_drift_relative_work','artificial_energy_relative_max_work','energies_N_mm')} for name,d in [('old_shell',slow),('fast_shell',fast)]},
    'wall_seconds':receipt['wall_seconds'],
    'conclusion':'Rate difference has a measured but small effect on this shell peak. It is not a sufficient explanation of the much later JAX peak.',
    'not_a_more_accurate_quasistatic_reference':True}
write(A/'comparison.json',comparison)

# Reuse the existing pure Q2 interpolator by extracting only its AST function.
oldmode=R/'validation/mechanism_20261007_r17/shell_frames'
with np.load(oldmode/'reference.npz') as f:
    xyz=f['xyz_mm'];tri=f['triangles'];area=f['reference_area_weights_mm2'];normals=f['reference_vertex_normals']
with np.load(results/'shell_frames/reference.npz') as f:
    assert np.array_equal(xyz,f['xyz_mm']) and np.array_equal(tri,f['triangles'])
    assert np.array_equal(area,f['reference_area_weights_mm2'])
with np.load(R/'validation/geometry_transfer_20261006_r15/hrz_mass.npz') as f:ids=f['class_ids']
source=R/'validation/mechanism_20261007_r17/analyze_localization.py'
tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='interpolate')
namespace={'np':np,'sx':xyz,'L':10.,'N':32,'levels':65,'ids':ids,'area':area}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
interpolate=namespace['interpolate']
sm=json.loads((oldmode/'modes.json').read_text())['rows']
fm=json.loads((results/'shell_frames/modes.json').read_text())['rows']
def rms(w):return float(np.sqrt(np.average(np.sum(w*w,axis=1),weights=area)))
def cosine(w,z):return float(np.sum(area[:,None]*w*z)/np.sqrt(np.sum(area[:,None]*w*w)*np.sum(area[:,None]*z*z)))
states=R/'validation/step_control_20261007_r18/controlled_forward'
mode_rows=[];plotfields=[]
for name in ('accepted_a0.1000.npz','accepted_a0.1200.npz','accepted_a0.1400.npz','accepted_a0.1600.npz','last_valid_field.npz'):
    with np.load(states/name) as f:q=f['q'];t=float(f['time'])
    s=np.clip(t/.004,0,1);a=float(.2*(10*s**3-15*s**4+6*s**5));w,tr=interpolate(q)
    near_s=min(sm,key=lambda r:abs(r['compression']-a));near_f=min(fm,key=lambda r:abs(r['compression']-a))
    with np.load(oldmode/near_s['file']) as f:sw=f['fluctuation_mm'];su=f['u_mm']
    with np.load(results/'shell_frames'/near_f['file']) as f:fw=f['fluctuation_mm'];fu=f['u_mm']
    mode_rows.append({'JAX_state':name,'JAX_compression':a,'JAX_fluctuation_RMS_mm':rms(w),
        'old_shell_compression':near_s['compression'],'fast_shell_compression':near_f['compression'],
        'JAX_vs_old_cosine':cosine(w,sw),'JAX_vs_fast_cosine':cosine(w,fw),'old_vs_fast_cosine':cosine(sw,fw),
        'old_shell_RMS_mm':rms(sw),'fast_shell_RMS_mm':rms(fw),
        'scope':'Nearest retained frames may have unequal compression; fluctuation-pattern diagnostic, not an eigenmode.'})
    if name in ('accepted_a0.1000.npz','accepted_a0.1400.npz','last_valid_field.npz'):
        plotfields.append((a,[(xyz+su,sw,near_s['compression'],'Old shell'),
            (xyz+fu,fw,near_f['compression'],'Fast shell'),(xyz+w+xyz*np.array([0.,0.,-a]),w,a,'JAX r18')]))
write(A/'mode_comparison.json',{'rows':mode_rows,'Q2_interpolator_source':str(source.relative_to(R)),
    'source_sha256':sha(source),'reference_surface_exact_equal':True,'new_mechanical_tangents_or_FEM':False})

fig,axes=plt.subplots(2,1,figsize=(10,7.8),sharex=True,constrained_layout=True)
for x,f,label in ((sx,sf,'Shell load 0.040 s'),(fx,ff,'Shell load 0.004 s'),(jx,jf,'JAX load 0.004 s (partial)')):
    axes[0].plot(100*x,f,label=label)
for p,c,offset in ((spe,'C0',(-56,15)),(fpe,'C1',(8,15)),(jpe,'C2',(5,9))):
    axes[0].plot(100*p['compression'],-p['Fz_N'],'o',color=c)
    axes[0].annotate(f"{100*p['compression']:.3f}%",(100*p['compression'],-p['Fz_N']),xytext=offset,textcoords='offset points',color=c)
axes[0].set_ylabel('Compression force (N)');axes[0].legend()
for rows,c,label in ((sp,'C0','Old shell KE/ALLIE'),(fp,'C1','Fast shell KE/ALLIE')):
    shown=[p for p in rows if p['compression']>=.01]
    axes[1].plot([100*p['compression'] for p in shown],[100*p['ALLKE']/p['ALLIE'] for p in shown],color=c,label=label)
axes[1].plot(100*jx,[100*p['KE_over_U'] for p in jp],color='C2',label='JAX KE/U (different denominator)')
axes[1].set_ylabel('Kinetic / internal energy (%)');axes[1].set_xlabel('Macroscopic compression (%)');axes[1].legend(fontsize=9)
for ax in axes:ax.grid(alpha=.25);ax.set_xlim(1,20);ax.axvline(100*j[-1]['compression'],color='C2',linestyle='--',linewidth=.7)
fig.suptitle('diverse_04: one duration-only shell diagnostic\nSame S3R mesh/material/PBC/control; JAX retained through 17.536%, no JAX 20% data')
fig.savefig(A/'response.png',dpi=170);plt.close(fig)

limit=max(float(np.max(np.abs(np.einsum('ni,ni->n',w,normals)))) for _,fields in plotfields for _,w,_,_ in fields)
norm=Normalize(-limit,limit);cmap=plt.get_cmap('coolwarm');fig=plt.figure(figsize=(12,13.5))
for col,(a,fields) in enumerate(plotfields):
    for row,(current,w,actual,label) in enumerate(fields):
        normal=np.einsum('ni,ni->n',w,normals)
        ax=fig.add_subplot(3,3,row*3+col+1,projection='3d')
        ax.add_collection3d(Poly3DCollection(current[tri],facecolors=cmap(norm(normal[tri].mean(axis=1))),edgecolors='none'))
        ax.set(xlim=(0,10),ylim=(0,10),zlim=(0,10));ax.set_box_aspect((1,1,1));ax.view_init(24,-58)
        ax.set_title(f'{label}: {100*actual:.2f}%\nFluctuation RMS {rms(w):.3f} mm',fontsize=10,pad=14)
        ax.tick_params(labelsize=8)
fig.suptitle('Actual deformed midsurfaces, no amplification\nShared colors: reference-normal fluctuation. Nearest shell frames have different compression.',fontsize=11)
fig.subplots_adjust(left=.02,right=.90,bottom=.04,top=.90,wspace=.05,hspace=.38)
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.93,.20,.013,.55]));cb.set_label('Normal fluctuation (mm)')
fig.savefig(A/'modes.png',dpi=150);plt.close(fig)
shutil.copy2(A/'response.png',W/'output/figures/SHELL_RATE_response.png')
shutil.copy2(A/'modes.png',W/'output/figures/SHELL_RATE_modes.png')
write(A/'receipt.json',{'seconds':time.perf_counter()-started,'new_jobs':0,'new_design_AD':False,
    'new_FEM_or_tangent_evaluations':False,'JAX_source':str(jfile.relative_to(R)),
    'JAX_source_sha256':sha(jfile),'old_shell_source_sha256':sha(slowfile)})
print(json.dumps({'peak_shift_pp':100*shift,'gap_reduction_fraction':shift/oldgap,'remaining_gap_pp':100*newgap,
    'JAX_vs_fast_peak_force_difference':comparison['JAX_peak_force_relative_fast_shell_difference'],'mode_rows':mode_rows},indent=2))
