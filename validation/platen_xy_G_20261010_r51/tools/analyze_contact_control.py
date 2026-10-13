"""Read frozen r51 outputs; compare common-dt support diagnostics, no FE solves."""
from pathlib import Path
import json, csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parents[1]
B = D/'diagnostics/contact_support_control_20261010'
OLD = D/'diagnostics/contact_origin_20261010'
sys.path.insert(0, str(D/'tools'))
from analyze_contact_origin import compare, series

def read(p): return json.loads(p.read_text())
def write(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

def loading(r, grid, T=.001):
    t,c,f = (series(r,k) for k in ('time','compression','force_N'))
    use=t<T
    return np.interp(grid,np.r_[c[use],.02],np.r_[f[use],np.interp(T,t,f)])

def interpolation(points):
    n=32; nn=65
    cell=np.minimum(np.floor(points*n).astype(int),n-1); loc=points*n-cell
    basis=lambda x:np.stack([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)],axis=-1)
    sh=np.einsum('ni,nj,nk->nijk',basis(loc[:,0]),basis(loc[:,1]),basis(loc[:,2])).reshape(-1,27)
    off=np.indices((3,3,3)).reshape(3,-1).T; nodes=2*cell[:,None,:]+off
    ids=((nodes[:,:,0]%(2*n))*(2*n)+nodes[:,:,1]%(2*n))*nn+nodes[:,:,2]
    return ids,sh

def support(state_path, cache_path, assignment, rim):
    with np.load(state_path) as z: q=z['q']; a=z['macro']
    with np.load(cache_path) as z: c=dict(z)
    def position(c):
        nodes=c['node_ids']; nn=65
        ij=nodes//nn; z=nodes%nn; ix=ij//nn; iy=ij%nn
        ids=((ix%64)*64+iy%64)*65+z
        u=np.einsum('cn,cnj->cj',c['shape'],q[ids]);u[:,:2]+=(c['points'][:,:2]-.5)*a
        return 10*(c['points']+u)
    x=position(c); xr=position(rim)
    pen=np.maximum(np.where(c['side']==0,-x[:,2],x[:,2]-9.8),0)
    f=320*c['area_weight']*100*pen
    lever=x[:,:2]-xr[assignment,:2]
    moment=np.zeros((len(xr),2)); np.add.at(moment,assignment,np.c_[lever[:,1],-lever[:,0]]*f[:,None])
    out=[]
    for s in (0,1):
        m=c['side']==s; rm=rim['side']==s; F=float(f[m].sum())
        out.append({'side':s,'terminal_force_N':F,
                    'sum_local_moment_magnitudes_over_force_mm':float(np.linalg.norm(moment[rm],axis=1).sum()/F),
                    'occupied_area_fraction_in_contact':float(c['area_weight'][m&(pen>0)].sum()/c['area_weight'][m].sum())})
    return {'scope':'Moments about assigned, currently displaced shell-rim interpolation points; sum of local magnitudes, not global net moment or isolated proof of moment causality.','sides':out}

def main():
    cfg=read(D/'input/protocol.json'); abq=read(D/'abaqus/result.json')
    baseline=read(D/'jax/short_2pct/result.json')
    line=read(OLD/'jax/rimline_T1ms/result.json')
    fine=read(B/'jax/cap_common_dt/result.json')
    half_path=B/'jax/half_support_common_dt/result.json'
    half=read(half_path) if half_path.exists() else None
    assert all(x['complete'] for x in (baseline,line,fine,abq))
    step=read(B/'input/protocol.json')['common_dt_s']
    assert all(abs(x['initial_dt_s']/step-1)<1e-13 and not x['rejections'] for x in (line,fine))
    with np.load(OLD/'input/rimline_contact.npz') as z: rim=dict(z)
    with np.load(OLD/'input/contact_assignment.npz') as z: assignment=z['rim_assignment']
    with np.load(D/'abaqus/field.npz') as z: verts=z['xyz_mm']/10; ua=z['u_mm']
    midsurface_ids,midsurface_shape=interpolation(verts)
    grid=np.linspace(0,.02,1001); fa=loading(abq,grid); denom=.7241182386842194
    definitions=[
        ('Original cap / original dt',baseline,D/'input/protocol.json',D/'jax/short_2pct',D/'input/cut_face_contact.npz',assignment,1.,'#8e9aa4',':'),
        ('Original cap / common dt',fine,B/'input/cap_common_dt.json',B/'jax/cap_common_dt',D/'input/cut_face_contact.npz',assignment,1.,'#28587b','-'),
        ('Rimline support / common dt',line,OLD/'input/rimline_protocol.json',OLD/'jax/rimline_T1ms',OLD/'input/rimline_contact.npz',np.arange(len(rim['points'])),0.,'#33765b','-'),
    ]
    if half is not None:
        assert half['complete']
        definitions.insert(2,('Halfway support / common dt',half,B/'input/half_support_common_dt.json',B/'jax/half_support_common_dt',B/'input/half_support_contact.npz',assignment,.5,'#b47727','-'))
    rows=[]; curves={}; fields={}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for label,r,cp,folder,cache,assignment_i,width,color,ls in definitions:
        row,_,_,_=compare(r,abq,read(cp),label,denom)
        fj=loading(r,grid); curves[label]=fj
        row.update(loading_RMS_1001_over_original_shell_peak=float(np.sqrt(np.mean((fj-fa)**2))/denom),
                   nominal_support_factor=width,steps=r['steps'])
        with np.load(folder/'last_valid_state.npz') as z: q=z['q']; a=z['macro']
        uj=np.einsum('cn,cnj->cj',midsurface_shape,q[midsurface_ids]);uj[:,:2]+=(verts[:,:2]-.5)*a;uj*=10
        fields[label]={'RMS_vector_mm':float(np.sqrt(np.mean(np.sum((uj-ua)**2,axis=1)))),
                       'RMS_over_shell_displacement_RMS':float(np.sqrt(np.sum((uj-ua)**2)/np.sum(ua**2))),
                       'maximum_vector_difference_mm':float(np.linalg.norm(uj-ua,axis=1).max())}
        row['terminal_displacement']=fields[label]
        row['terminal_support']=support(folder/'last_valid_state.npz',cache,assignment_i,rim)
        np.testing.assert_allclose(row['terminal_support']['sides'][1]['terminal_force_N'],r['path'][-1]['force_N'],rtol=1e-10)
        rows.append(row)
        axs[0].plot(grid*100,fj,color=color,ls=ls,label=label)
        axs[1].plot(series(r,'time')*1000,series(r,'force_N'),color=color,ls=ls,label=label)
    axs[0].plot(grid*100,fa,'k--',lw=1.7,label='Abaqus shell rim / T1ms')
    axs[1].plot(series(abq,'time')*1000,series(abq,'force_N'),'k--',lw=1.7,label='Abaqus shell rim / T1ms')
    axs[0].set(xlabel='Platen compression (%)',ylabel='Top compressive force (N)',title='Contact support diagnostic: unchanged G volume mechanics')
    axs[1].set(xlabel='Physical time (ms)',ylabel='Top compressive force (N)',title='2% ramp (1 ms) + hold (0.2 ms)')
    axs[1].axvline(1,color='.7',lw=.8)
    for ax in axs: ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(B/'analysis/common_dt_contact_comparison.png',dpi=190);plt.close(fig)
    control=np.max(abs(curves[definitions[1][0]]-curves[definitions[0][0]]))/denom
    result={'scope':'G 2pct end-contact diagnostics only; no physical finite-thickness certification, no20pct or newAD.',
            'common_initial_dt_s':step,'force_RMS_primary_grid_points':1001,
            'max_loading_cap_dt_change_over_shell_peak':float(control),
            'original_cap_hold_mean_relative_change_from_finer_dt':rows[1]['jax']['hold_top_N']/rows[0]['jax']['hold_top_N']-1,
            'cases':rows,
            'halfway_pre_dynamic_admission_failure':read(B/'jax/half_support_common_dt/admission_failure.json') if half is None else None,
            'limits':['Halfway support is artificial; material thickness is unchanged.',
                      'Support map alters contact positions and spatial discretization, not exclusively moment.',
                      'Pressure penalties differ between solvers; rimline matching is not a universal physical accuracy certificate.',
                      'Time-step control tests the current short path; it is not global time convergence.']}
    write(B/'analysis/metrics.json',result)
    with (B/'analysis/comparison.csv').open('w',newline='') as f:
        names=['case','nominal_support_factor','loading_RMS_1001_over_original_shell_peak','force_2pct_bias','hold_top_bias','work_bias','wall_minutes','initial_dt_s','steps','displacement_RMS']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for row in rows:
            record={k:row[k] for k in names if k in row}
            record.update(wall_minutes=row['jax']['wall_minutes'],initial_dt_s=row['jax']['initial_dt_s'],displacement_RMS=row['terminal_displacement']['RMS_over_shell_displacement_RMS'])
            w.writerow(record)
    np.savetxt(B/'analysis/loading_curves.csv',np.column_stack([grid,fa,*curves.values()]),delimiter=',',header=','.join(['compression','Abaqus shell',*curves.keys()]),comments='')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    print(json.dumps([{'case':r['case'],'RMS':r['loading_RMS_1001_over_original_shell_peak'],'hold_bias':r['hold_top_bias'],'force2_bias':r['force_2pct_bias'],'minutes':r['jax']['wall_minutes']} for r in rows],indent=2))

if __name__=='__main__': main()
