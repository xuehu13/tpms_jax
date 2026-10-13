"""Packaging postmortem and saved-state geometry checks; no new G path."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/platen_xy_G_20261010_r51';B=D/'diagnostics/shell_cap_arm_20261010'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def rotate(Rm,a):return np.einsum('nij,nj->ni',Rm,a)
def rotation(theta):
    angle=np.linalg.norm(theta,axis=1);axis=np.divide(theta,angle[:,None],out=np.zeros_like(theta),where=angle[:,None]>0)
    K=np.zeros((len(theta),3,3));K[:,0,1]=-axis[:,2];K[:,0,2]=axis[:,1];K[:,1,0]=axis[:,2];K[:,1,2]=-axis[:,0];K[:,2,0]=-axis[:,1];K[:,2,1]=axis[:,0]
    return np.eye(3)+np.sin(angle)[:,None,None]*K+(1-np.cos(angle))[:,None,None]*(K@K)
def interpolate_jax(points,q,macro,gradient=False):
    n=32;nn=65;cell=np.minimum(np.floor(points*n).astype(int),n-1);loc=points*n-cell
    val=lambda x:np.stack([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)],axis=-1)
    der=lambda x:np.stack([4*x-3,4-8*x,4*x-1],axis=-1)*n
    sx,sy,sz=[val(loc[:,k]) for k in range(3)]
    sh=np.einsum('ni,nj,nk->nijk',sx,sy,sz).reshape(-1,27)
    off=np.indices((3,3,3)).reshape(3,-1).T;nodes=2*cell[:,None,:]+off
    ids=((nodes[:,:,0]%64)*64+nodes[:,:,1]%64)*65+nodes[:,:,2]
    qq=q[ids];u=np.einsum('cn,cnj->cj',sh,qq);u[:,:2]+=(points[:,:2]-.5)*macro
    if not gradient:return u
    parts=[np.einsum('ni,nj,nk->nijk',der(loc[:,0]),sy,sz),np.einsum('ni,nj,nk->nijk',sx,der(loc[:,1]),sz),np.einsum('ni,nj,nk->nijk',sx,sy,der(loc[:,2]))]
    grads=np.stack([x.reshape(-1,27) for x in parts],axis=-1)
    F=np.eye(3)+np.diag(np.r_[macro,0.])+np.einsum('cni,cnj->cij',qq,grads)
    return u,F
def main():
    cfg=read(B/'input/protocol_wrapped.json')
    with np.load(B/'input/cap_arm_wrapped_map.npz') as z:m=dict(z)
    with np.load(B/'input/original_shell_pose.npz') as z:s=dict(z)
    with np.load(D/'input/cut_face_contact.npz') as z:c=dict(z)
    with np.load(D/'jax/short_2pct/last_valid_state.npz') as z:q=z['q'];macro=z['macro']
    ix=m['parent_labels']-1;arms=m['reference_arm_mm'];phi=c['phi'][m['cap_indices']];area=m['area_mm2'];side=m['side'];kn=320.
    Rs=rotation(s['ur_rad'][ix]);xs=m['points_mm']+s['u_mm'][ix]+rotate(Rs,arms)-arms
    parent=s['xyz_mm'][ix]/10
    uj,F=interpolate_jax(parent,q,macro,True);UU,ss,Vh=np.linalg.svd(F);Rj=UU@Vh
    assert np.all(np.linalg.det(F)>0) and np.all(np.linalg.det(Rj)>0)
    xjr=m['points_mm']+uj*10+rotate(Rj,arms)-arms
    xj=c['points'][m['cap_indices']]*10+interpolate_jax(c['points'][m['cap_indices']],q,macro)*10
    def pen(x):return np.maximum(np.where(side==0,-x[:,2],x[:,2]-9.8),0)
    ps,pj,pjr=pen(xs),pen(xj),pen(xjr)
    rows=[]
    for name,p,x in [('Original shell pose + finite rigid cap replay',ps,xs),('Actual original JAX cap pose',pj,xj),('JAX pose + polar rigid cap replay',pjr,xjr)]:
        for sd in (0,1):
            use=side==sd;den=area[use].sum()
            rows.append({'case':name,'side':sd,'area_weighted_penetration_mm':float(np.sum(area[use]*p[use])/den),
                'max_penetration_phi_ge_0p01_mm':float(p[use&(phi>=.01)].max()),
                'max_penetration_phi_ge_0p5_mm':float(p[use&(phi>=.5)].max()),
                'occupied_contact_fraction':float(area[use&(p>0)].sum()/den),
                'fixed_pose_penalty_demand_N':float(np.sum(kn*area[use]*p[use]))})
    zerr=xjr[:,2]-xj[:,2]
    result={'scope':'READ-ONLY saved terminal states. Frozen-pose penalty demands are not equilibrated reactions, not new force curves and not replacement validation.',
        'G_new_dynamic_paths_complete':0,'G_attempts_stopped_before_time_advance':3,'original_G_comparison_still_not_passed':True,
        'saved_states_time_s':float(s['time_s']),'rows':rows,
        'rigid_cap_approximation_on_JAX':{'area_weighted_z_difference_RMS_mm':float(np.sqrt(np.sum(area*zerr**2)/area.sum())),
            'max_abs_z_difference_phi_ge_0p5_mm':float(abs(zerr[phi>=.5]).max()),
            'required_J_at_parent_min':float(np.linalg.det(F).min()),
            'note':'Rotation estimated from polar decomposition of Q2 F at original shell rim positions. Not an independent shell rotation solution; includes different stretch/shear kinematics.'},
        'shell_rim_rotation_max_degrees':float(np.linalg.norm(s['ur_rad'][np.unique(ix)],axis=1).max()*180/np.pi),
        'full20_submitted':False,'new_design_AD':False}
    original=read(D/'jax/short_2pct/result.json')['path'][-1]
    result['JAX_actual_cap_force_reconstruction_relative_error']=rows[3]['fixed_pose_penalty_demand_N']/original['force_N']-1
    assert abs(result['JAX_actual_cap_force_reconstruction_relative_error'])<1e-8
    write(B/'analysis/preparation_metrics.json',result)
    with (B/'analysis/saved_pose_cap_geometry.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(B/'analysis/saved_pose_cap_geometry.npz',cap_indices=m['cap_indices'],phi=phi,area_mm2=area,side=side,
        original_points_mm=c['points'][m['cap_indices']]*10,shell_cap_replay_x_mm=xs,JAX_cap_x_mm=xj,JAX_rigid_replay_x_mm=xjr,
        shell_replay_penetration_mm=ps,JAX_actual_penetration_mm=pj,JAX_rigid_replay_penetration_mm=pjr)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True);use=(side==1)&(phi>=.01);xy=c['points'][m['cap_indices']]*10
    vmax=max(float(ps[use].max()),float(pj[use].max()))*1000
    for ax,p,title in [(axs[0],ps,'Original shell pose: hypothetical finite cap'),(axs[1],pj,'Original JAX: actual finite cap')]:
        plot=ax.scatter(xy[use,0],xy[use,1],c=p[use]*1000,s=9,cmap='viridis',vmin=0,vmax=vmax)
        ax.set(title=title,xlabel='Reference X (mm)',ylabel='Reference Y (mm)');ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.colorbar(plot,ax=axs,label='Top cap penetration (micrometre)',shrink=.85)
    fig.suptitle('Saved terminal states only | 2% + hold | no new equilibrated G solution',fontsize=12)
    fig.savefig(B/'analysis/saved_pose_cap_penetration.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    mask=(side==1)&(phi>=.01);bins=np.linspace(0,max(float(ps[mask].max()),float(pj[mask].max()),float(pjr[mask].max()))*1000,45)
    for name,p,color in [('Shell pose / finite-cap replay',ps,'#8a8a8a'),('JAX actual cap',pj,'#14608a'),('JAX pose / rigid-cap replay',pjr,'#b36126')]:
        axs[0].hist(p[mask]*1000,bins=bins,weights=area[mask]/area[mask].sum(),histtype='step',lw=1.6,label=name,color=color)
    axs[0].set(xlabel='Top cap penetration (micrometre)',ylabel='Fraction of occupied reference area',title='Frozen geometries: no equilibrium re-solve');axs[0].legend(fontsize=8)
    p=axs[1].scatter(xy[mask,0],xy[mask,1],c=zerr[mask]*1000,s=9,cmap='coolwarm');axs[1].set_aspect('equal')
    axs[1].set(xlabel='Reference X (mm)',ylabel='Reference Y (mm)',title='Rigid-arm approximation differs from solid motion');fig.colorbar(p,ax=axs[1],label='Replayed minus actual JAX cap Z (micrometre)',shrink=.85)
    for ax in axs:ax.grid(alpha=.18)
    fig.savefig(B/'analysis/saved_pose_rigid_cap_limit.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
