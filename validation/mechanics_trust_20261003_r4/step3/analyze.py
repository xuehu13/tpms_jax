"""Compare stored completed paths; no new equilibrium solve."""
from pathlib import Path
import json,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step3'
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/mechanics_trust_20261004_r4/step3_uniform')
def load(f):return json.loads(f.read_text(encoding='utf-8-sig'))
def keyed(rows):return {round(r['compression'],9):r for r in rows}
jax={tag:load(O/(tag+'.json')) for tag in ('N2_d010','N4_d010','N4_d005')}
aba={tag:load(next((A/tag/'work').glob('*.acceptance.json'))) for tag in ('N4_d010','N4_d005')}
assert all(r['status']=='ok' for r in list(jax.values())+list(aba.values()))
scaleF=max(abs(r['Fz_top']) for r in aba['N4_d005']['rows']);scaleU=max(abs(r['ALLSE']) for r in aba['N4_d005']['rows'])
def path_delta(left,right,energy_left='energy',energy_right='energy'):
    l=keyed(left['rows']);r=keyed(right['rows']);keys=sorted(set(l)&set(r))
    return {'common_states':len(keys),'force_over_reference_max':max(abs(l[k]['Fz_top']-r[k]['Fz_top']) for k in keys)/scaleF,
            'energy_over_reference_max':max(abs(l[k][energy_left]-r[k][energy_right]) for k in keys)/scaleU}
comparisons={'JAX_N4_vs_Abaqus_base':path_delta(jax['N4_d010'],aba['N4_d010'],'energy','ALLSE'),
             'JAX_N4_vs_Abaqus_half_increment':path_delta(jax['N4_d005'],aba['N4_d005'],'energy','ALLSE'),
             'JAX_grid_N2_N4':path_delta(jax['N2_d010'],jax['N4_d010']),
             'JAX_increment_halved':path_delta(jax['N4_d010'],jax['N4_d005']),
             'Abaqus_increment_halved':path_delta(aba['N4_d010'],aba['N4_d005'],'ALLSE','ALLSE')}
mode=[]
for tag in ('N4_d010','N4_d005'):
    with np.load(O/(tag+'.displacement.npz')) as jf,np.load(next((A/tag/'work').glob('*.acceptance.displacement.npz'))) as af:
        jp=jf['points'];ap=af['points'];mapping={tuple(np.round(p,12)):i for i,p in enumerate(jp)}
        order=[mapping[tuple(np.round(p,12))] for p in ap]
        assert np.allclose(jp[order],ap,atol=1e-12)
        assert np.allclose(jf['compression'],af['compression'],atol=1e-14)
        delta=jf['u'][:,order,:]-af['u']
        mode.append({'path':tag,'nodes':len(ap),'states':len(jf['compression']),
                     'max_absolute_displacement_difference':float(np.abs(delta).max()),
                     'rms_difference_at_twenty_percent_over_axial_displacement':float(np.sqrt(np.mean(delta[-1]**2))/.2)})
modulus=10*(1-.3)/((1+.3)*(1-2*.3));small=[]
for name,record in [(k,v) for k,v in jax.items()]+[('Abaqus_'+k,v) for k,v in aba.items()]:
    row=keyed(record['rows'])[.0001];secant=-row['Fz_top']/.0001
    small.append({'path':name,'small_compression':.0001,'secant':secant,'linear_constrained_modulus':modulus,'relative_difference':abs(secant-modulus)/modulus})
selected=[]
for a in (.01,.05,.1,.2):
    j=keyed(jax['N4_d005']['rows'])[a];ab=keyed(aba['N4_d005']['rows'])[a]
    selected.append({'compression':a,'JAX_Fz':j['Fz_top'],'Abaqus_Fz':ab['Fz_top'],'JAX_energy':j['energy'],'Abaqus_energy':ab['ALLSE'],
                     'JAX_J_min':j['J_min'],'Abaqus_J_min':ab['J_min']})
checks={'independent_force_path':all(v['force_over_reference_max']<=.01 for k,v in comparisons.items() if 'vs_Abaqus' in k),
        'independent_energy_path':all(v['energy_over_reference_max']<=.01 for k,v in comparisons.items() if 'vs_Abaqus' in k),
        'grid_and_increment':all(max(v['force_over_reference_max'],v['energy_over_reference_max'])<=.001 for k,v in comparisons.items() if 'vs_Abaqus' not in k),
        'small_strain_limit':all(v['relative_difference']<=.001 for v in small),
        'path_work':all(v['work']['relative_gap']<=.001 for v in list(jax.values())+list(aba.values())),
        'all_node_modes':all(v['max_absolute_displacement_difference']<=1e-7 for v in mode),
        'all_stored_physical_checks':all(all(all(row['checks'].values()) for row in v['rows']) for v in list(jax.values())+list(aba.values()))}
cost=[]
for name,v in jax.items():
    text=(O/(name+'.log')).read_text()
    # The installed solver logs the number of actual Newton corrections.
    counts=[int(x) for x in re.findall(r'\[TIMER\]\s+([0-9]+)\s+Newton',text)]
    cost.append({'path':name,'states':len(v['rows']),'runtime':v['runtime'],'initial_residual_at_seeded_points':[r['initial_residual_l2'] for r in v['rows'] if r['nonzero_seed']],
                 'work':v['work']})
summary={'stage':'round4_step3_uniform_passed' if all(checks.values()) else 'round4_step3_precision_or_implementation_failed',
         'checks':checks,'comparisons':comparisons,'small_strain':small,'node_modes':mode,'selected_points':selected,
         'cost':cost,'Abaqus_work':{k:v['work'] for k,v in aba.items()},'steps4_started':False,
         'limit':'Uniform affine full-solid algorithm/material/constraints benchmark only, no TPMS finite-strain certification'}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
row=jax['N4_d005']['rows'];a=np.array([r['compression'] for r in row])
reference=aba['N4_d005']['rows']
axes[0].plot(a*100,[-r['Fz_top'] for r in row],color='#2675aa',lw=2,label='JAX N4, 0.5% increment')
axes[0].plot([r['compression']*100 for r in reference][::2],[-r['Fz_top'] for r in reference][::2],ls='none',marker='o',ms=4,mfc='none',color='#dd9259',label='Abaqus C3D8')
axes[0].plot(a*100,modulus*a,'--',color='#666666',label='Small-strain tangent')
axes[0].set(xlabel='Compression / initial height (%)',ylabel='Compression force magnitude',title='Full solid; fixed macro lateral strain');axes[0].legend(fontsize=8)
energy=np.array([r['energy'] for r in row]);force=np.array([r['Fz_top'] for r in row])
work=np.r_[0,np.cumsum(-.5*(force[1:]+force[:-1])*np.diff(a))]
axes[1].plot(a*100,energy,color='#2675aa',lw=2,label='Stored energy')
axes[1].plot(a*100,work,'--',color='#dd9259',label='Integrated force-displacement work')
axes[1].plot(a*100,-force*a/2,':',color='#666666',label='0.5 F u (linear formula)')
axes[1].set(xlabel='Compression / initial height (%)',ylabel='Total energy',title='Nonlinear work check');axes[1].legend(fontsize=8)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'uniform_response.png',dpi=180);fig.savefig(Path(__file__).with_name('uniform_response.png'),dpi=180);plt.close(fig)
(O/'analyze.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary,indent=2))
if not all(checks.values()):raise SystemExit('Finite-strain full-solid benchmark did not pass')
