"""Read-only r51 contact-support and rate comparisons, using frozen outputs."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parents[1]
B=D/'diagnostics/contact_origin_20261010'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def series(r,k):return np.array([x[k] for x in r['path']])
def mean(t,y,lo,hi):
    x=np.r_[lo,t[(t>lo)&(t<hi)],hi]
    return float(np.trapezoid(np.interp(x,t,y),x)/(hi-lo))
def cap_diagnostic():
    cfg=read(D/'input/protocol.json');L=cfg['L_mm'];n=cfg['N'];nn=2*n+1
    with np.load(D/'input/cut_face_contact.npz') as z:cap=dict(z)
    with np.load(B/'input/rimline_contact.npz') as z:rim=dict(z)
    with np.load(B/'input/contact_assignment.npz') as z:assignment=z['rim_assignment']
    with np.load(D/'jax/short_2pct/last_valid_state.npz') as z:q=z['q'];a=z['macro']
    ii=np.indices((nn,nn,nn)).reshape(3,-1).T
    ids=((ii[:,0]%(2*n))*(2*n)+ii[:,1]%(2*n))*nn+ii[:,2]
    def displaced(c):
        u=np.einsum('cn,cnj->cj',c['shape'],q[ids[c['node_ids']]])
        u[:,:2]+=(c['points'][:,:2]-.5)*a
        return (c['points']+u)*L
    x=displaced(cap);xr=displaced(rim)
    pen=np.maximum(np.where(cap['side']==0,-x[:,2],x[:,2]-L*(1-cfg['target_compression'])),0)
    f=cfg['contact_normal_stiffness_MPa_per_mm']*cap['area_weight']*L**2*pen
    lever=x[:,:2]-xr[assignment,:2]
    moment=np.zeros((len(xr),2));np.add.at(moment,assignment,np.c_[lever[:,1],-lever[:,0]]*f[:,None])
    area_moment=np.zeros((len(xr),2));np.add.at(area_moment,assignment,np.c_[lever[:,1],-lever[:,0]]*cap['area_weight'][:,None])
    fg=np.bincount(assignment,weights=f,minlength=len(xr));ag=rim['area_weight']
    pressure_redistribution=moment-area_moment*np.divide(fg,ag,out=np.zeros_like(fg),where=ag>0)[:,None]
    metrics={'scope':'Saved terminal state of original cap-contact path only. Moments are about nearest shell rim nodes, using current lever arms; not an isolated causal test or physical shell nodal moments.','sides':[]}
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for side in (0,1):
        mask=cap['side']==side;rm=rim['side']==side
        F=float(f[mask].sum());M=float(np.linalg.norm(moment[rm],axis=1).sum());Md=float(np.linalg.norm(pressure_redistribution[rm],axis=1).sum())
        metrics['sides'].append({'side':side,'force_N':F,'sum_local_moment_magnitudes_N_mm':M,
              'equivalent_local_lever_mm':M/F,'pressure_redistribution_moment_N_mm':Md,
              'pressure_redistribution_equivalent_lever_mm':Md/F,
              'occupied_area_fraction_in_contact':float(cap['area_weight'][mask&(pen>0)].sum()/cap['area_weight'][mask].sum())})
        use=mask&(cap['phi']>=.01)
        plot=axs[side].scatter(cap['points'][use,0]*L,cap['points'][use,1]*L,c=(cfg['contact_normal_stiffness_MPa_per_mm']*pen*cap['phi'])[use],s=7,cmap='viridis',rasterized=True)
        axs[side].scatter(rim['points'][rm,0]*L,rim['points'][rm,1]*L,s=4,c='black',label='Shell midsurface rim nodes')
        axs[side].set(xlabel='Reference X (mm)',ylabel='Reference Y (mm)',title=('Bottom' if side==0 else 'Top')+' cap | terminal 2%')
        axs[side].set_aspect('equal');axs[side].legend(fontsize=7)
        fig.colorbar(plot,ax=axs[side],label='Occupancy-weighted nominal traction (MPa)',shrink=.8)
    fig.suptitle('Finite-width cap traction and shell rim contact locations',fontsize=12)
    fig.savefig(B/'analysis/terminal_contact_support.png',dpi=180);plt.close(fig)
    write(B/'analysis/terminal_support.json',metrics)
    return metrics
def compare(j,a,cfg,label,original_peak):
    assert j['complete'] and a['complete']
    T=cfg['load_time_s'];end=T+cfg['hold_time_s'];grid=np.linspace(0,cfg['target_compression'],501)
    def loading(r):
        t=series(r,'time');c=series(r,'compression');f=series(r,'force_N');m=t<T
        return np.interp(grid,np.r_[c[m],cfg['target_compression']],np.r_[f[m],np.interp(T,t,f)])
    fj,fa=loading(j),loading(a);peak=float(abs(fa).max());rms=float(np.sqrt(np.mean((fj-fa)**2)))
    row={'case':label,'scope':cfg['stage'],'loading_RMS_N':rms,'RMS_over_original_shell_peak':rms/original_peak,'RMS_over_matching_shell_peak':rms/peak,'matching_shell_peak_N':peak}
    for name,r in [('jax',j),('abaqus',a)]:
        t=series(r,'time');f=series(r,'force_N');bot=series(r,'bottom_force_N')
        row[name]={'force_1pct_N':float(np.interp(.01,grid,fj if name=='jax' else fa)),
          'force_2pct_N':float(np.interp(T,t,f)),
          'hold_top_N':mean(t,f,end-.0001,end),'hold_bottom_N':mean(t,bot,end-.0001,end),
          'work_N_mm':float(np.trapezoid(f,10*series(r,'compression'))),
          'terminal_ax_ay_mm':[r['path'][-1]['ax_mm'],r['path'][-1]['ay_mm']],
          'terminal_KE_over_IE':r['path'][-1].get('KE_over_IE',r.get('terminal_KE_over_IE'))}
        if name=='jax':
            work=np.r_[0,np.cumsum(.5*(f[1:]+f[:-1])*10*np.diff(series(r,'compression')))];den=max(float(abs(work).max()),1e-30)
            energy=series(r,'energy_N_mm')+series(r,'contact_energy_N_mm')+series(r,'KE_N_mm')
            momentum=series(r,'momentum_z_N_s');imbalance=bot-f
            impulse=np.r_[0,np.cumsum(.5*(imbalance[1:]+imbalance[:-1])*np.diff(t))]
            row[name].update(sampled_energy_defect_over_work=float(abs(energy-energy[0]-work).max()/den),
                sampled_impulse_defect_over_peak_momentum=float(abs(momentum-momentum[0]-impulse).max()/max(float(abs(momentum).max()),1e-30)),
                required_J_min=min(x['required_positive_J_min'] for x in r['path']),
                rejected_blocks=len(r['rejections']),wall_minutes=r['wall_seconds']/60,
                initial_dt_s=r['initial_dt_s'],terminal_dt_s=r['terminal_dt_s'])
        else:
            row[name].update(energy_drift_relative_work=r['energy_drift_relative_work'],XY_U_error_mm=r['XY_U_error_mm'],XY_UR_error_rad=r['XY_UR_error_rad'])
    row['hold_top_bias']=row['jax']['hold_top_N']/row['abaqus']['hold_top_N']-1
    row['force_2pct_bias']=row['jax']['force_2pct_N']/row['abaqus']['force_2pct_N']-1
    row['work_bias']=row['jax']['work_N_mm']/row['abaqus']['work_N_mm']-1
    return row,grid,fj,fa
def displacement_fields():
    with np.load(D/'abaqus/field.npz') as z:v=z['xyz_mm']/10;ua=z['u_mm']
    n=32;nn=65;cell=np.minimum(np.floor(v*n).astype(int),n-1);loc=v*n-cell
    shape=lambda x:np.stack([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)],axis=-1)
    sh=np.einsum('ni,nj,nk->nijk',shape(loc[:,0]),shape(loc[:,1]),shape(loc[:,2])).reshape(-1,27)
    off=np.indices((3,3,3)).reshape(3,-1).T;nodes=2*cell[:,None,:]+off
    ids=((nodes[:,:,0]%(2*n))*(2*n)+nodes[:,:,1]%(2*n))*nn+nodes[:,:,2]
    result={'scope':'Raw terminal displacement differences at existing shell midsurface nodes, Q2 interpolated from saved JAX state; no rigid alignment or scaling. This is a field diagnostic, not a new acceptance threshold.','cases':{}}
    for label,path in [('Original cap',D/'jax/short_2pct/last_valid_state.npz'),('Rimline diagnostic',B/'jax/rimline_T1ms/last_valid_state.npz')]:
        if not path.exists():continue
        with np.load(path) as z:q=z['q'];a=z['macro']
        uj=np.einsum('cn,cnj->cj',sh,q[ids]);uj[:,:2]+=(v[:,:2]-.5)*a;uj*=10
        diff=uj-ua
        result['cases'][label]={'RMS_component_mm':np.sqrt(np.mean(diff**2,axis=0)).tolist(),
            'RMS_vector_mm':float(np.sqrt(np.mean(np.sum(diff**2,axis=1)))),
            'RMS_vector_over_shell_displacement_RMS':float(np.sqrt(np.sum(diff**2)/np.sum(ua**2))),
            'maximum_vector_difference_mm':float(np.linalg.norm(diff,axis=1).max())}
    write(B/'analysis/displacement_fields.json',result)
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--terminal-only',action='store_true');args=ap.parse_args()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    support=cap_diagnostic()
    if args.terminal_only:print(json.dumps(support,indent=2));return
    basecfg=read(D/'input/protocol.json');basej=read(D/'jax/short_2pct/result.json');basea=read(D/'abaqus/result.json')
    peak=read(D/'analysis/metrics.json')['shell_loading_max_N']
    cases=[('Original cap / T1ms',basej,basea,basecfg)]
    for name,jp,ap,cp in [
      ('Rimline diagnostic / T1ms',B/'jax/rimline_T1ms/result.json',D/'abaqus/result.json',B/'input/rimline_protocol.json'),
      ('Original cap / T2ms',B/'jax/cap_T2ms/result.json',B/'abaqus/slow/result.json',B/'input/slow_protocol.json')]:
        if jp.exists() and ap.exists():
            j,a=read(jp),read(ap)
            if j.get('complete') and a.get('complete'):cases.append((name,j,a,read(cp)))
    rows=[];fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    colors=['#14608a','#a34545','#397c4e']
    for (label,j,a,cfg),color in zip(cases,colors):
        row,grid,fj,fa=compare(j,a,cfg,label,peak);rows.append(row)
        axs[0].plot(grid*100,fj,label='JAX '+label,color=color)
        if label!='Rimline diagnostic / T1ms':axs[0].plot(grid*100,fa,label=f"Abaqus shell rim / T{cfg['load_time_s']*1000:g}ms",color=color,ls='--')
        for tag,r,ls in [('JAX',j,'-'),('Abaqus',a,'--')]:
            if label=='Rimline diagnostic / T1ms' and tag=='Abaqus':continue
            legend=tag+' '+label if tag=='JAX' else f"Abaqus shell rim / T{cfg['load_time_s']*1000:g}ms"
            axs[1].plot(series(r,'time')/cfg['load_time_s'],series(r,'force_N'),color=color,ls=ls,label=legend)
    axs[0].set(xlabel='Platen compression (%)',ylabel='Top compressive force (N)',title='Same G material, geometry, N32, free XY macro')
    axs[1].set(xlabel='Time / ramp duration',ylabel='Top compressive force (N)',title='Hold is 0.2 ms in every physical path')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(B/'analysis/contact_origin_comparison.png',dpi=180);plt.close(fig)
    result={'full20_submitted':False,'new_design_AD':False,'cases':rows,'terminal_support':support,'displacement_fields':displacement_fields(),
            'interpretation_limits':['Rimline changes contact support and discretization, not only a scalar penalty.','Hold means remain dynamic, not static tangent stiffness.','Current forward comparison does not certify a geometry gradient.']}
    write(B/'analysis/metrics.json',result)
    with (B/'analysis/comparison.csv').open('w',newline='') as f:
        names=['case','RMS_over_original_shell_peak','RMS_over_matching_shell_peak','force_2pct_bias','hold_top_bias','work_bias']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows({k:r[k] for k in names} for r in rows)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
