"""Read existing frozen r15 records only; no FEM solve, replay, or design AD."""
import os
os.environ["JAX_PLATFORMS"]="cpu"
from pathlib import Path
import sys,json,hashlib,subprocess,time,shutil
import numpy as np
R=Path("/home/xuehu/projects/tpms_jax")
sys.path.insert(0,str(R))
D=R/"validation/geometry_transfer_review_20261006_r16"
P=R/"validation/geometry_transfer_20261006_r15"
BASE="450bc8f994e1ea1e9fb6f0576ae9e381e2e88e9b"
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=R,text=True).strip()==BASE
assert not subprocess.check_output(["git","status","--porcelain"],cwd=R,text=True).strip()
D.mkdir(exist_ok=False)
start=time.perf_counter()
def write(name,obj):
    (D/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
frozen=set(subprocess.check_output(["git","ls-files","validation"],cwd=R,text=True).splitlines())
for folder in ("void_continuation_20261006_r12","thickness_range_20261006_r13","forward_scope_20261006_r14","geometry_transfer_20261006_r15"):
    frozen.update(str(p.relative_to(R)) for p in (R/"validation"/folder).rglob("*") if p.is_file())
frozen.update(("hyperelastic_fem.py","scripts/thin_target_explicit.py","surface_distance.py","pbc.py","fem.py","pixi.lock"))
before={s:sha(R/s) for s in sorted(frozen)}
write("frozen_before.json",{"base_commit":BASE,"files":before})
shutil.copyfile(__file__,D/"existing_record_review.py")
gauss=np.load(P/"gauss_field.npz")
mass=np.load(P/"hrz_mass.npz")
rho=gauss["rho"]
eta=1e-4
scale=eta+(1-eta)*rho
phi=(scale-eta)/(1-eta)
from jax_fem.basis import get_elements
import basix,jax,jax.numpy as jnp
jax.config.update("jax_enable_x64",True)
from hyperelastic_fem import objective_void_energy,void_nh_cutoff
family,cell,face,_,degree,order=get_elements("HEX27")
el=basix.create_element(family,cell,degree)
xi,wt=basix.make_quadrature(cell,4)
tab=el.tabulate(1,xi).take(order,axis=2)
shape=tab[0,:,:,0]
grad=tab[1:4,:,:,0].transpose(1,2,0)*32
local=np.rint(2*el.points[order]).astype(int)
origins=2*np.indices((32,)*3).reshape(3,-1).T
nodes=origins[:,None,:]+local
nodeids=(nodes[...,0]*65+nodes[...,1])*65+nodes[...,2]
classids=mass["class_ids"][nodeids]
quad=np.einsum("qn,cnd->cqd",shape,nodes/64)
assert np.max(np.abs(quad-gauss["physical_quad_points"]))<1e-13
assert np.max(np.abs(wt[None,:]/32**3-gauss["JxW"]))<1e-14
assert np.array_equal(mass["class_ids"].reshape(65,65,65),
    ((np.indices((65,)*3)[0]%64)*64+np.indices((65,)*3)[1]%64)*64+np.indices((65,)*3)[2]%64)
wp=jax.jit(jax.vmap(jax.value_and_grad(objective_void_energy,0),in_axes=(0,0,None)))
rejections=json.loads((P/"T0p004_cpu_reference/rejected_blocks.json").read_text())
details=[]
for k,record in enumerate(rejections,1):
    f=P/f"T0p004_cpu_reference/rejected_block_{k:02d}.npz"
    z=np.load(f);q=z["q"];v=z["vhalf"];t=float(z["time"]);s=t/.004
    h=-.2*(10*s**3-15*s**4+6*s**5)
    qfinite=np.isfinite(q).all(1)
    cellfinite=qfinite[classids].all(1)
    complete=np.where(cellfinite)[0]
    row={"file":str(f.relative_to(R)),"sha256":sha(f),"accepted":False,
        "attempted_time_s":t,"compression":-h,
        "finite_q_scalar_count":int(np.isfinite(q).sum()),"q_scalar_count":q.size,
        "finite_q_class_count":int(qfinite.sum()),"class_count":len(q),
        "nan_q_scalar_count":int(np.isnan(q).sum()),"inf_q_scalar_count":int(np.isinf(q).sum()),
        "finite_vhalf_scalar_count":int(np.isfinite(v).sum()),
        "complete_finite_nodal_cells":len(complete),"candidate_finite_gauss_count":27*len(complete),
        "full_domain_energy_N_mm":None,
        "scope":"Rejected endpoint, finite-cell subset only. No valid final state, no full energy, no first-offending-step localization."}
    if len(complete):
        F=np.eye(3)[None,None,:,:]+np.diag([0,0,h])[None,None,:,:]+np.einsum("cni,qnj->cqij",q[classids[complete]],grad)
        flat=F.reshape(-1,3,3);pp=phi[complete].ravel()
        JJ=np.linalg.det(flat);norm=np.linalg.norm(flat,axis=(1,2))
        Wlist=[];Plist=[]
        for i in range(0,len(flat),4096):
            W,Pstress=wp(jnp.asarray(flat[i:i+4096]),jnp.asarray(pp[i:i+4096]),eta)
            Wlist.append(np.asarray(W));Plist.append(np.asarray(Pstress))
        W=np.concatenate(Wlist);stress=np.concatenate(Plist);pfinite=np.isfinite(stress).all((1,2));wfinite=np.isfinite(W)
        finiteF=np.isfinite(flat).all((1,2));required=pp>=.01
        def domain(mask):
            return {"points":int(mask.sum()),"nonpositive_J":int(np.sum(mask&(JJ<=0))),
                "nonfinite_J":int(np.sum(mask&~np.isfinite(JJ))),
                "nonfinite_energy":int(np.sum(mask&~wfinite)),"nonfinite_stress":int(np.sum(mask&~pfinite))}
        row["finite_subset"]={"finite_F_points":int(finiteF.sum()),"domains":{
            "deep_virtual_phi_le_0p001":domain(pp<=.001),
            "mixed_virtual_0p001_lt_phi_lt_0p01":domain((pp>.001)&(pp<.01)),
            "uncontinued_NH_phi_ge_0p01":domain(required)},
            "finite_J_min":float(np.min(JJ[np.isfinite(JJ)])),
            "finite_J_max":float(np.max(JJ[np.isfinite(JJ)])),
            "max_F_norm":float(norm.max()),
            "finite_energy_max":float(np.max(W[wfinite])),
            "finite_stress_max_absolute":float(np.max(np.abs(stress[pfinite])))}
        bad=np.where((required&(JJ<=0))|~wfinite|~pfinite)[0]
        row["finite_subset"]["offending_points_count"]=len(bad)
        row["finite_subset"]["first_12_by_cell_order"]=[
            {"cell":int(complete[i//27]),"gauss_index":int(i%27),
             "reference_xyz_normalized":quad[complete[i//27],i%27].tolist(),
             "phi":float(pp[i]),"J":float(JJ[i]) if np.isfinite(JJ[i]) else None,
             "energy_finite":bool(wfinite[i]),"stress_finite":bool(pfinite[i])}
            for i in bad[:12]]
    details.append(row)
    print("REJECTED_SUBSET",k,json.dumps({x:row[x] for x in ("finite_q_scalar_count","complete_finite_nodal_cells")}),flush=True)
write("rejected_endpoint_diagnosis.json",{"rows":details,
    "same_27_point_reference_verified":True,"used_shared_material_kernel":True,
    "no_force_assembly_or_time_advance":True,"no_design_gradient":True,
    "partial_cell_sum_not_used_as_full_energy":True})
shell=json.loads((P/"abaqus/explicit_T0p040/results/shell.json").read_text())
path=shell["force_path"];tt=np.array([r["time"] for r in path]);aa=np.array([r["compression"] for r in path])
ff=np.array([r["Fz_N"] for r in path])
IE=np.array([r["ALLIE"] for r in path]);KE=np.array([r["ALLKE"] for r in path]);WK=np.array([r["ALLWK"] for r in path])
AE=np.array([r["ALLAE"] for r in path]);VD=np.array([r["ALLVD"] for r in path]);ET=np.array([r["ETOTAL"] for r in path])
rat=KE/np.maximum(IE,1e-30);peakwork=float(np.max(np.abs(WK)));load=tt<=.04
loading=load&(aa>=.01)
weights=np.r_[0.,np.diff(tt)]
fraction=float(np.sum(weights[loading]*(rat[loading]<=.05))/np.sum(weights[loading]))
assert abs(fraction-shell["loading_time_fraction_KE_below_5pct"])<1e-12
drift=float(np.max(np.abs(ET-ET[0]))/peakwork);aefrac=float(np.max(np.abs(AE))/peakwork)
assert abs(drift-shell["global_energy_drift_relative_work"])<1e-12
assert abs(aefrac-shell["artificial_energy_relative_max_work"])<1e-12
loading_indices=np.where(load)[0];imax=loading_indices[np.argmax(np.abs(ff[load]))]
def sample(c):
    return {key:float(np.interp(c,aa[load],np.array([r[key] for r in path])[load]))
        for key in ("time","Fz_N","ALLIE","ALLSE","ALLKE","ALLAE","ALLVD","ALLWK","ETOTAL")}
intervals=[];mask=loading&(rat>.05)
for group in np.split(np.flatnonzero(mask),np.where(np.diff(np.flatnonzero(mask))>1)[0]+1):
    if len(group):intervals.append({"start_s":float(tt[group[0]]),"end_s":float(tt[group[-1]]),
        "start_compression":float(aa[group[0]]),"end_compression":float(aa[group[-1]]),
        "sample_count":len(group),"max_KE_over_IE":float(rat[group].max())})
accepted=json.loads((P/"accepted_logged_observations.json").read_text())["rows"]
pairs=[]
for row in accepted:
    sh=sample(row["compression"]);force=sh["Fz_N"]
    pairs.append({**row,"shell_at_same_compression_interpolated":sh,
        "signed_difference_N":row["Fz_N"]-force,
        "absolute_difference_N":abs(row["Fz_N"]-force),
        "difference_over_shell_force_magnitude":abs(row["Fz_N"]-force)/max(abs(force),1e-30),
        "dynamic_minus_internal_N":row["Fz_N"]-row["internal_macro_Fz_N"]})
jaxpre=[r for r in pairs if r["time"]<=rejections[0]["time"]+1e-15]
largest=max(jaxpre,key=lambda r:r["difference_over_shell_force_magnitude"])
write("partial_force_diagnosis.json",{"rows":pairs,"JAX_rows_are_original_sparse_observations":True,
    "shell_interpolation_is_diagnostic_only":True,"no_JAX_path_interpolation":True,
    "full_1000_point_RMS":None,"full_20pct_work_difference":None,"20pct_hold_force_difference":None,
    "mode_difference":None,"largest_prerejection_sample_relative_difference":largest})
write("shell_path_diagnosis.json",{"samples":len(path),"load_end_s":.04,"full_end_s":float(tt[-1]),
    "loading_peak_magnitude_N":float(abs(ff[imax])),"loading_peak_compression":float(aa[imax]),"loading_peak_time_s":float(tt[imax]),
    "loading_time_fraction_KE_below_5pct":fraction,"KE_high_sample_intervals":intervals,
    "energy_drift_over_peak_work":drift,"artificial_energy_over_peak_work":aefrac,
    "terminal_viscous_energy_over_external_work":float(VD[-1]/WK[-1]),
    "terminal_energy":shell["energies_N_mm"],"original_checks":shell["checks"],
    "samples_at_compression":{str(c):sample(c) for c in (.01,.05,.10,.125,.15,.15966239917005406,.175,.20)},
    "quality_metrics_reproduced":True,"no_rate_rerun":True,
    "limits":"Shell dynamics/viscosity/artificial energy are not isolated contributions to force error or certified quasistatic truth."})
sc=json.loads((R/"validation/forward_scope_20261006_r14/forward_scope.json").read_text())
write("scope_decision.json",{"steps_3_4":"completed_limited_existing_record_review",
    "step2_20pct_credible_transfer_goal_passed":False,"existing_diverse28_scope":sc,
    "diverse04_t0p50":{"JAX_complete":False,"shell_complete":True,"shell_strict_quality_pass":False,
        "JAX_last_monitored_compression":.1744483742919905,
        "first_rejected_interval_compression_start":rejections[0]["last_valid"]["compression"],
        "full_response_comparison":None,"migration_certified":False,"gradient20_certified":False},
    "decision":"Preserve diverse_28 scope. diverse_04 is not a validated replacement or gradient case. Continue only targeted mechanism work after user decides next round.",
    "next_conditions_proposal_only":[
        "Preserve complete accepted block path and last accepted state before any future attempted repair; do not relabel rejected fields.",
        "Localize the first invalid internal step/domain and distinguish local tangent/time-step loss from uncontinued NH crossing. Do not fit thickness, eta or damping.",
        "If shell is retained as reference, resolve its rate/energy/artificial-energy quality uncertainty before interpreting full-curve force differences.",
        "Only after a credible matched forward is available, assess full new-kernel design derivative on a consistent accepted time discretization; training remains later."],
    "no_new_forward_jobs":True,"no_solver_or_constant_changes":True,"no_full_AD_or_training":True})
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(11.4,7.8),layout="constrained")
ax=axs[0,0];ax.plot(100*aa[load],-ff[load],label="Shell loading (quality failed)",color="C1")
ap=np.array([r["compression"] for r in accepted]);af=np.array([r["Fz_N"] for r in accepted])
ax.scatter(100*ap,-af,label="JAX sparse accepted observations",s=23,color="C0",zorder=3)
ax.axvline(100*rejections[0]["last_valid"]["compression"],ls=":",color=".4",label="First rejected block starts")
ax.set(xlabel="Compression (%)",ylabel="Compression force magnitude (N)");ax.legend(fontsize=8)
ax=axs[0,1];ax.plot(tt/.04,100*rat,color="C1",label="Shell KE/IE")
ax.scatter(np.array([r["time"] for r in accepted])/.004,100*np.array([r["KE_over_U"] for r in accepted]),color="C0",label="JAX sparse KE/U")
ax.axhline(5,color=".4",ls="--");ax.set(xlabel="Time / each loading duration",ylabel="Kinetic / internal energy (%)",ylim=(0,32));ax.legend(fontsize=8)
ax=axs[1,0]
for vals,label in ((WK,"External work"),(IE,"Internal energy"),(VD,"Viscous dissipation"),(AE,"Artificial energy"),(KE,"Kinetic energy")):
    ax.plot(tt/.04,vals,label=label)
ax.set(xlabel="Shell time / loading duration",ylabel="Energy (N mm)");ax.legend(fontsize=8)
ax=axs[1,1]
ax.scatter(100*ap,[r["required_positive_J_min"] for r in accepted],color="C0",label="Min J in uncontinued NH")
ax.axhline(0,color=".4",ls="--")
ax.set(xlabel="Compression (%)",ylabel="Min local volume ratio J");ax.legend(fontsize=8)
fig.suptitle("diverse_04 t=0.50 mm: existing-record diagnosis only; no validated 20% JAX result",fontsize=11)
fig.savefig(D/"existing_record_diagnosis.png",dpi=170);plt.close(fig)
after={s:sha(R/s) for s in before}
assert before==after
write("verification.json",{"base_commit":BASE,"frozen_file_count":len(before),"all_frozen_bytes_unchanged":True,
    "maintained_sources_and_lock_unchanged":True,"quad_and_periodic_class_maps_verified":True,
    "shell_original_quality_metrics_reproduced":True,"elapsed_postprocess_seconds":time.perf_counter()-start,
    "no_time_advance_or_forward_job":True,"no_full_AD":True})
write("evidence_manifest.json",{"base_commit":BASE,"scope":"Existing-record steps 3 and 4 only",
    "input_files":{s:before[s] for s in before if "geometry_transfer_20261006_r15" in s or "forward_scope_20261006_r14" in s},
    "output_files":{str(f.relative_to(D)):sha(f) for f in D.iterdir() if f.is_file()},
    "run_python":sys.executable})
print("SUMMARY",json.dumps({"frozen_files":len(before),"shell_peak_N":abs(ff[imax]),"shell_peak_compression":aa[imax],
    "partial_largest_pre_rejection_difference":largest["difference_over_shell_force_magnitude"],"output":str(D)}),flush=True)
