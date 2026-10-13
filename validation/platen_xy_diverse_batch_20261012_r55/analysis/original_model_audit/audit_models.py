"""Audit saved original/current models and curves. No solver, source or result edits."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
W=HERE.parents[2]
P=Path(r'F:\auto_abaqus\work')
A=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus')
KINDS=['diverse_02','diverse_04','diverse_05','diverse_28']
protected={}

def read(p):
    protected[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p.read_text(encoding='utf-8-sig')

def js(p): return json.loads(read(p))
def table(p):
    rows=list(csv.DictReader(read(p).splitlines()))
    columns={}
    for key in rows[0]:
        try: columns[key]=np.array([float(r[key]) for r in rows])
        except ValueError: pass  # JAX CSV also contains explicit JSON vector columns.
    return columns
def deck(p):
    lines=[]
    for line in read(p).splitlines():
        if line.lower().startswith('*include'):
            lines.extend(deck(p.parent/line.split('=',1)[1].strip()))
        else: lines.append(line)
    return lines
def blocks(lines):
    out=[]
    for line in lines:
        if not line.strip() or line.startswith('**'): continue
        if line.startswith('*'): out.append([line,[]])
        elif out: out[-1][1].append(line)
    return out
def parse(lines):
    nodes={};shell={};plate={};equations=[];bc=[]
    for key, data in blocks(lines):
        name=key.lower().split(',')[0]
        if name=='*node':
            for line in data:
                z=line.split(',');nodes[int(z[0])]=tuple(map(float,z[1:4]))
        if name=='*element':
            target=shell if 's3r' in key.lower() else plate
            for line in data:
                z=[int(v) for v in line.split(',')];target[z[0]]=tuple(z[1:])
        if name=='*equation':
            z=','.join(data[1:]).replace(' ','').split(',');z=[v for v in z if v]
            n=int(data[0]);assert len(z)==3*n
            equations.append(tuple((int(z[3*i]),int(z[3*i+1]),float(z[3*i+2])) for i in range(n)))
        if name=='*boundary': bc.extend(data)
    return nodes,shell,plate,equations,bc

old=[];new={};audits={};curve={}
paths=sorted((P/'para_aly').glob('*/*/*/diverse_*/results/stress_strain.csv'))
paths=[q for q in paths if q.parent.parent.name in KINDS]
paths.append(P/'r3_smoke/batch_001/diverse_02/results/stress_strain.csv')
for q in paths:
    d=q.parent.parent;kind=d.name
    s=js(d/'simulation_used.json');b=js(d/'abaqus/build_report.json');z=js(q.parent/'summary.json')
    h=table(q.parent/'history.csv');v=table(q)
    peak=int(np.argmax(v['engineering_stress_MPa']))
    variant='r3_smoke' if kind=='diverse_02' else '/'.join(d.relative_to(P/'para_aly').parts[:-1])
    r={'kind':kind,'variant':variant,'directory':str(d),'E_MPa':s['material']['youngs_modulus_MPa'],
       'nu':s['material']['poisson_ratio'],'density_tonne_mm3':s['material']['density_tonne_mm3'],
       'plastic_table':s['material']['plastic_table'],'t_mm':b['model']['thickness_mm'],
       'surface_area_mm2':b['model']['surface_area_mm2'],'time_s':s['solver']['explicit_dynamic']['time_period_s'],
       'mass_scaling':s['solver']['explicit_dynamic']['mass_scaling'],'contact':s['contact'],
       'nodes':b['model']['shell_nodes'],'elements':b['model']['shell_elements'],'equations':b['model']['equation_count'],
       'raw_global_peak_compression':float(v['engineering_strain'][peak]),
       'raw_global_peak_stress_MPa':float(v['engineering_stress_MPa'][peak]),
       'terminal_plastic_dissipation_N_mm':float(h['ALLPD'][-1]),
       'terminal_internal_energy_N_mm':float(h['ALLIE'][-1]),
       'terminal_ALLPD_ALLIE':float(h['ALLPD'][-1]/h['ALLIE'][-1]),'success':z['success']}
    old.append(r);curve[(kind,variant)]=(v['engineering_strain'],v['engineering_stress_MPa'])

for kind in KINDS:
    d=W/'output'/('r54_platen_xy_diverse04_20261012' if kind=='diverse_04' else 'r55_platen_xy_diverse_batch_20261012/'+kind)
    m=js(d/'model_parameters.json');v=table(d/'Abaqus_path.csv');j=table(d/'JAX_path.csv')
    use=v['time']<=.004000001;eps=v['compression'][use];stress=v['force_N'][use]/100.
    peak=int(np.argmax(stress));curve[(kind,'current Abaqus')]=(eps,stress)
    curve[(kind,'current JAX')]=(j['compression'][j['time']<=.004000001],j['force_N'][j['time']<=.004000001]/100.)
    selected=next(r for r in old if r['kind']==kind and r['variant'] in ['r3_smoke','Fine/T0p02/MS0'])
    od=Path(selected['directory'])/'abaqus'
    on,oe,op,oq,ob=parse(deck(od/'physical.inp'));nn,ne,np_,nq,nb=parse(deck(d/'platen_20pct.inp'))
    labels={i for tri in oe.values() for i in tri}
    maxdiff=max(float(np.max(np.abs(np.array(on[i])-nn[i]))) for i in labels)
    malformed=[]
    for eq in nq:
        delta=np.array(nn[eq[0][0]])-nn[eq[1][0]]
        if abs(delta[2])>1e-8: malformed.append('different_z')
        if not (np.isclose(abs(delta[0]),10) or np.isclose(abs(delta[1]),10)): malformed.append('not_xy')
        if len(eq)>2:
            axis=eq[0][1]-1
            if axis>1 or not np.isclose(eq[2][2],-delta[axis]/10,atol=1e-8):malformed.append('macro_coefficient')
    assert oe==ne and maxdiff<1e-8 and set(oq)==set(nq) and not malformed
    assert not any('RP_X' in row or 'RP_Y' in row for row in ob+nb)
    native=A/('r54_diverse04_midsurface_end_20pct_20261012' if kind=='diverse_04' else 'r55_'+kind+'_midsurface_end_20pct_20261012')/'platen.inp'
    assert read(native)==read(d/'platen_20pct.inp')
    def extent(nodes,elems):
        x=np.array([nodes[n] for e in elems.values() for n in e]);return [x.min(axis=0).tolist(),x.max(axis=0).tolist()]
    audits[kind]={'same_triangle_connectivity':oe==ne,'max_reference_coordinate_difference_mm':maxdiff,
                  'same_exact_XY_equation_sets':set(oq)==set(nq),'equation_count':len(nq),
                  'rotation_equations':sum(eq[0][1]>=4 for eq in nq),'XY_pair_checks_failed':len(malformed),
                  'macro_XY_free_in_BC':True,'old_plate_extent_mm':extent(on,op),'new_plate_extent_mm':extent(nn,np_),
                  'native_INP_matches_reading':True,'current_material_parameters':m,
                  'old_selected':selected,'current_nominal_shell_volume_fraction':selected['surface_area_mm2']*.5/1000.}
    new[kind]={'kind':kind,'raw_global_peak_compression':float(eps[peak]),'raw_global_peak_stress_MPa':float(stress[peak]),
               'current_E_MPa':10.,'current_t_mm':.5}

odb=js(HERE/'odb_readonly_check.json')
for r in odb:
    ref=new[r['kind']] if r['label']=='current' else audits[r['kind']]['old_selected']
    assert abs(r['loading_raw_peak'][1]-ref['raw_global_peak_compression'])<1e-8
    assert abs(r['loading_raw_peak'][2]-ref['raw_global_peak_stress_MPa'])<1e-8

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axs=plt.subplots(2,4,figsize=(16,8.4),constrained_layout=True)
for col,kind in enumerate(KINDS):
    r=audits[kind]['old_selected'];top=axs[0,col];bottom=axs[1,col]
    for entry in old:
        if entry['kind']!=kind:continue
        x,y=curve[(kind,entry['variant'])]
        selected=entry['variant']==r['variant']
        top.plot(x*100,y,color='#285c8e' if selected else '#b0b5b9',lw=2 if selected else .8,
                 label='Original selected' if selected else None,alpha=1 if selected else .7)
    for name,color,style in [('current Abaqus','#b34335','-'),('current JAX','#d78c20','--')]:
        x,y=curve[(kind,name)];bottom.plot(x*100,y,color=color,ls=style,lw=1.5,label=name)
    top.set_title(f'{kind}: original\nE=484 MPa, t={r["t_mm"]:.4f} mm')
    bottom.set_title(f'{kind}: current simplified model\nE=10 MPa, t=0.5 mm')
    for ax in [top,bottom]:
        ax.set(xlim=(0,20),xlabel='Engineering compression (%)',ylabel='Nominal stress (MPa)')
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
fig.suptitle('Original vs current models: actual stress, separate vertical ranges\nGray: original mesh/time/mass variants. No curve fitting; not the same physical test.',fontsize=13)
fig.savefig(HERE/'stress_strain_original_and_current.png');plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax,kind in zip(axs.flat,KINDS):
    r=audits[kind]['old_selected']
    for entry in old:
        if entry['kind']!=kind:continue
        x,y=curve[(kind,entry['variant'])];selected=entry['variant']==r['variant']
        ax.plot(x*100,y/max(y),color='#285c8e' if selected else '#b0b5b9',lw=2 if selected else .8,
                label='Original selected' if selected else None,alpha=1 if selected else .55)
    for name,color,style in [('current Abaqus','#b34335','-'),('current JAX','#d78c20','--')]:
        x,y=curve[(kind,name)];ax.plot(x*100,y/max(y),color=color,ls=style,lw=1.7,label=name)
    ax.set(title=kind,xlim=(0,20),xlabel='Engineering compression (%)',ylabel='Stress / own sampled loading peak')
    ax.grid(alpha=.2);ax.legend(fontsize=9,loc='best')
fig.suptitle('Curve-shape diagnostic only: each path divided by its own loading maximum\nDoes not equalize physics, measure error, or alter r54/r55 acceptance.',fontsize=12)
fig.savefig(HERE/'curve_shape_comparison.png');plt.close(fig)

payload={'original_inventory':old,'current_raw_peaks':new,'model_audits':audits,
         'scope':'Saved-input and read-only ODB comparison; no new dynamics/AD/FD/Git; original scientific records unchanged.'}
(HERE/'model_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows=[]
for kind in KINDS:
    for label,r in [('original selected',audits[kind]['old_selected']),('current Abaqus',new[kind])]:
        rows.append({'kind':kind,'model':label,'raw_global_peak_compression_percent':r['raw_global_peak_compression']*100,
                     'raw_global_peak_stress_MPa':r['raw_global_peak_stress_MPa']})
with (HERE/'raw_peak_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
(HERE/'read_only_receipt.json').write_text(json.dumps({'protected_read_files':protected,'all_bytes_unchanged':True,
              'ODB_read_only_count':8,'new_solver_jobs':0,'new_AD_FD':0,'new_git':0,'figures_visually_checked':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'original_curves_found':len(old),'geometry_PBC_audits':{k:{p:v for p,v in audits[k].items() if p in ['max_reference_coordinate_difference_mm','same_exact_XY_equation_sets','old_plate_extent_mm','new_plate_extent_mm']} for k in KINDS},'peaks':rows},ensure_ascii=False))
