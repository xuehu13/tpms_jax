"""Read-only paired short-path analysis; never submits or changes mechanics."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

D=Path(__file__).resolve().parents[1];R=D.parents[1]
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def arr(result,key):return np.array([x[key] for x in result['path']])
def integral(y,x):return float(np.sum(.5*(y[1:]+y[:-1])*np.diff(x)))
def mean_window(t,y,lo,hi):
    q=np.r_[lo,t[(t>lo)&(t<hi)],hi]
    return integral(np.interp(q,t,y),q)/(hi-lo)

def main():
    cfg=read(D/'input/protocol.json');j=read(D/'jax/short_2pct/result.json');a=read(D/'abaqus/result.json')
    assert j['complete'] and a['complete'],'Do not present interrupted paths as completed'
    out=D/'analysis';out.mkdir(exist_ok=True)
    T=cfg['load_time_s'];end=T+cfg['hold_time_s'];L=cfg['L_mm'];target=cfg['target_compression']
    jt,at=arr(j,'time'),arr(a,'time');jf,af=arr(j,'force_N'),arr(a,'force_N')
    jc,ac=arr(j,'compression'),arr(a,'compression')
    grid=np.linspace(0,target,501)
    def loading(c,t,y):
        m=t<T
        return np.interp(grid,np.r_[c[m],target],np.r_[y[m],np.interp(T,t,y)])
    fj,fa=loading(jc,jt,jf),loading(ac,at,af)
    shell_peak=float(np.max(np.abs(fa)));rms=float(np.sqrt(np.mean((fj-fa)**2))/shell_peak)
    hold_start=T+cfg['hold_time_s']/2
    hj=mean_window(jt,jf,hold_start,end);ha=mean_window(at,af,hold_start,end)
    jwork=integral(jf,L*jc);awork=integral(af,L*ac)
    je=arr(j,'energy_N_mm')+arr(j,'contact_energy_N_mm')+arr(j,'KE_N_mm')
    jw=np.r_[0,np.cumsum(.5*(jf[1:]+jf[:-1])*L*np.diff(jc))]
    impulse=np.r_[0,np.cumsum(.5*((arr(j,'bottom_force_N')-jf)[1:]+(arr(j,'bottom_force_N')-jf)[:-1])*np.diff(jt))]
    mom=arr(j,'momentum_z_N_s');pden=max(float(np.max(abs(mom))),1e-30)
    metrics={'scope':cfg['stage'],'full20_submitted':False,'new_design_AD':False,
      'load_RMS_over_shell_2pct_max':rms,'shell_loading_max_N':shell_peak,
      'force_at_1pct_N':{'jax':float(np.interp(.01,grid,fj)),'abaqus':float(np.interp(.01,grid,fa))},
      'force_at_2pct_N':{'jax':float(np.interp(T,jt,jf)),'abaqus':float(np.interp(T,at,af))},
      'last_half_hold_mean_force_N':{'jax':hj,'abaqus':ha,'relative_bias':hj/ha-1},
      'external_work_N_mm':{'jax_sampled_RF_U':jwork,'abaqus_sampled_RF_U':awork,'relative_bias':jwork/awork-1},
      'terminal_ax_ay_mm':{'jax':[j['path'][-1]['ax_mm'],j['path'][-1]['ay_mm']],
                             'abaqus':[a['path'][-1]['ax_mm'],a['path'][-1]['ay_mm']]},
      'terminal_contact_penetration_mm':{'jax_phi_ge_0p01':j['path'][-1]['cap_penetration_phi_ge_0p01_mm'],
                                         'abaqus_rim_nodes':a['terminal_rim_penetration_mm']},
      'jax_sampled_energy_defect_over_work':float(np.max(abs(je-je[0]-jw))/max(np.max(abs(jw)),1e-30)),
      'jax_sampled_impulse_defect_over_peak_momentum':float(np.max(abs(mom-mom[0]-impulse))/pden),
      'sampled_work_impulse_note':'Trapezoidal diagnostic on saved output, not stepwise conservation certification',
      'jax_terminal_KE_over_IE':j['path'][-1]['KE_over_IE'],
      'jax_min_required_J_in_saved_rows':min(x['required_positive_J_min'] for x in j['path']),
      'jax_max_negative_void_J_points_in_saved_rows':max(x['negative_void_J_points'] for x in j['path']),
      'jax_rejected_blocks':len(j['rejections']),
      'abaqus_XY_U_error_mm':a['XY_U_error_mm'],'abaqus_XY_UR_error_rad':a['XY_UR_error_rad'],
      'abaqus_energy_drift_relative_work':a['energy_drift_relative_work'],
      'abaqus_work_relative_ALLWK_error':a['work_relative_ALLWK_error'],
      'abaqus_terminal_KE_over_IE':a['terminal_KE_over_IE'],
      'wall_minutes':{'jax_including_preparation':j['wall_seconds']/60,
                      'abaqus_solve_launcher':read(D/'abaqus/solve_receipt.json')['wall_seconds']/60,
                      'abaqus_datacheck':read(D/'abaqus/datacheck_receipt.json')['wall_seconds']/60},
      'input_sha256':cfg['inp_sha256'],'initial_uniform_check':read(D/'checks/uniform_contact/result.json')['analytic_check_pass']}
    # Final-state coverage audit uses saved arrays only, with the exact Q2 grid numbering.
    n=cfg['N'];nn=2*n+1
    ii=np.indices((nn,nn,nn)).reshape(3,-1).T
    ids=((ii[:,0]%(2*n))*(2*n)+ii[:,1]%(2*n))*nn+ii[:,2]
    with np.load(D/'jax/short_2pct/last_valid_state.npz') as z:q=z['q']
    with np.load(D/'input/cut_face_contact.npz') as z:cap=dict(z)
    zcap=cap['points'][:,2]+np.sum(q[ids[cap['node_ids']],2]*cap['shape'],axis=1)
    pen=np.maximum(np.where(cap['side']==0,-zcap,zcap-(1-target)),0)*L
    per_side=[]
    for s in [0,1]:
        mask=cap['side']==s;area=cap['area_weight'][mask]
        per_side.append({'area_weighted_penetration_mm':float(np.sum(area*pen[mask])/np.sum(area)),
            'max_penetration_mm_by_phi_threshold':{str(th):float(np.max(pen[mask&(cap['phi']>=th)])) for th in [.01,.5,.9]},
            'integrated_force_N':float(cfg['contact_normal_stiffness_MPa_per_mm']*np.sum(area*pen[mask])*L**2)})
    np.testing.assert_allclose([x['integrated_force_N'] for x in per_side],
        [j['path'][-1]['bottom_force_N'],j['path'][-1]['force_N']],rtol=1e-10)
    with np.load(cfg['original_Gauss_field']) as z:rho=z['rho'];qp=z['physical_quad_points']
    cell=np.floor(qp[:,0,:]*n).astype(int);loc=qp[0]*n-cell[0]
    sh=lambda x:np.stack([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)],axis=-1)
    Q=np.einsum('qi,qj,qk->qijk',sh(loc[:,0]),sh(loc[:,1]),sh(loc[:,2])).reshape(27,27)
    offsets=np.indices((3,3,3)).reshape(3,-1).T;counts={str(th):0 for th in [.01,.5,.9]};max_pen=0.
    for start in range(0,len(cell),2048):
        base=2*cell[start:start+2048,None,:]+offsets
        cids=((base[:,:,0]%(2*n))*(2*n)+base[:,:,1]%(2*n))*nn+base[:,:,2]
        zgp=qp[start:start+2048,:,2]+q[cids,2]@Q.T
        gp_pen=np.maximum(np.maximum(-zgp,zgp-(1-target)),0)*L
        for th in [.01,.5,.9]:counts[str(th)]+=int(np.sum((rho[start:start+2048]>=th)&(gp_pen>1e-10)))
        active=rho[start:start+2048]>=.01
        max_pen=max(max_pen,float(np.max(np.where(active,gp_pen,0))))
    with np.load(D/'abaqus/field.npz') as z:v=z['xyz_mm'];u=z['u_mm']
    interior=(v[:,2]>1e-8)&(v[:,2]<L-1e-8);zs=(v+u)[:,2]
    outside=np.maximum(np.maximum(-zs,zs-L*(1-target)),0)
    coverage={'scope':'saved terminal state only; excludes finite shell-thickness envelope and self-contact',
              'jax_caps_bottom_top':per_side,'jax_volume_Gauss_points_outside_planes_by_phi_threshold':counts,
              'jax_volume_max_penetration_phi_ge_0p01_mm':max_pen,
              'shell_nonrim_midsurface_nodes_outside_planes':int(np.sum(interior&(outside>1e-8))),
              'shell_nonrim_midsurface_max_penetration_mm':float(np.max(np.where(interior,outside,0)))}
    write(out/'terminal_contact_coverage.json',coverage)
    metrics['terminal_contact_coverage']=coverage
    write(out/'metrics.json',metrics)
    for name,result in [('jax',j),('abaqus',a)]:
        rows=result['path']
        with (out/(name+'_response.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors={'JAX':'#14608a','Abaqus':'#c47123'}
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for label,t,c,f,result in [('JAX',jt,jc,jf,j),('Abaqus',at,ac,af,a)]:
        co=colors[label];mask=t<T
        axs[0,0].plot(np.r_[c[mask],target]*100,np.r_[f[mask],np.interp(T,t,f)],label=label,color=co)
        axs[0,1].plot(t*1000,f,label=label,color=co)
        axs[1,0].plot(t*1000,arr(result,'ax_mm'),label=label+' ax',color=co)
        axs[1,0].plot(t*1000,arr(result,'ay_mm'),label=label+' ay',color=co,ls='--')
        axs[1,1].plot(t*1000,f,color=co,label=label+' top')
        axs[1,1].plot(t*1000,arr(result,'bottom_force_N'),color=co,ls='--',label=label+' bottom')
    for ax in axs.flat:ax.grid(alpha=.22);ax.legend(fontsize=8)
    axs[0,0].set(xlabel='Compression (%)',ylabel='Compressive platen force (N)',title='G: loading to 2% (hold shown at right)')
    axs[0,1].set(xlabel='Time (ms)',ylabel='Top platen force (N)',title='Same 1 ms ramp + 0.2 ms hold')
    axs[1,0].set(xlabel='Time (ms)',ylabel='Lateral displacement jump (mm)',title='Free transverse macro displacement')
    axs[1,1].set(xlabel='Time (ms)',ylabel='Compressive contact force (N)',title='Top / bottom; inertia permits imbalance')
    fig.suptitle('r51 G | XY periodic | frictionless cut-end contact prototype',fontsize=13)
    fig.savefig(out/'paired_response.png');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    axs[0,0].plot(jt*1000,arr(j,'energy_N_mm'),label='JAX internal')
    axs[0,0].plot(at*1000,arr(a,'ALLIE'),label='Abaqus internal')
    axs[0,0].plot(jt*1000,arr(j,'KE_N_mm'),ls='--',label='JAX kinetic')
    axs[0,0].plot(at*1000,arr(a,'ALLKE'),ls='--',label='Abaqus kinetic')
    axs[0,0].set(ylabel='Energy (N mm)',title='Physical dynamics, no mass scaling')
    axs[0,1].plot(jt*1000,je-je[0],label='JAX internal + contact + kinetic')
    axs[0,1].plot(jt*1000,jw,label='JAX sampled external work',ls='--')
    axs[0,1].plot(at*1000,arr(a,'ETOTAL')-a['path'][0]['ETOTAL'],label='Abaqus total-energy drift')
    axs[0,1].set(ylabel='Energy (N mm)',title='Energy accounting; sampled work is approximate')
    pen=np.array([x['cap_penetration_phi_ge_0p01_mm'] for x in j['path']])
    axs[1,0].plot(jt*1000,pen[:,0]*1000,label='JAX bottom cap')
    axs[1,0].plot(jt*1000,pen[:,1]*1000,label='JAX top cap')
    axs[1,0].set(ylabel='Penetration (micrometre)',title='Only reference cap points with occupancy >= 0.01')
    axs[1,1].plot(jt*1000,mom,label='JAX actual z momentum')
    axs[1,1].plot(jt*1000,impulse,ls='--',label='Saved-output contact impulse')
    axs[1,1].set(ylabel='Momentum (N s)',title='Reaction imbalance versus momentum change')
    for ax in axs.flat:ax.set_xlabel('Time (ms)');ax.grid(alpha=.22);ax.legend(fontsize=8)
    fig.savefig(out/'energy_contact.png');plt.close(fig)
    with np.load(R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz') as z:
        vertices=z['surface_vertices']*L;faces=z['surface_triangles']
    fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d')
    ax.add_collection3d(Poly3DCollection(vertices[faces],facecolors='#c2cbd0',edgecolors='none',alpha=.8))
    for height,color in [(0,'#2d799c'),(L,'#c47123')]:
        square=np.array([[-1,-1,height],[11,-1,height],[11,11,height],[-1,11,height]])
        ax.add_collection3d(Poly3DCollection([square],facecolors=color,alpha=.2))
    ax.set(xlim=(-1,11),ylim=(-1,11),zlim=(0,10),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',
           title='G shell midsurface and platen planes\nXY periodic; transverse macro jumps free')
    ax.set_zlabel('');ax.text2D(1.08,.65,'Z (mm)',transform=ax.transAxes,rotation=90)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-60)
    fig.text(.5,.02,'Plates cropped in view; actual X,Y extent: -5 to 15 mm.',ha='center',fontsize=9)
    fig.savefig(out/'geometry_boundary.png',bbox_inches='tight');plt.close(fig)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
