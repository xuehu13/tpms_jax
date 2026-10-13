"""Repair constraint packaging only; preserve failed BEAM-MPC attempt."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/platen_xy_G_20261010_r51';B=D/'diagnostics/shell_cap_arm_20261010'
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    with np.load(B/'input/cap_arm_map.npz') as z:m=dict(z)
    cfg=json.loads((B/'input/protocol.json').read_text());text=(B/'abaqus/G/platen_cap_arm.inp').read_text()
    old=['*MPC']+[f'BEAM,{n},{p}' for n,p in zip(m['labels'],m['master_labels'])]
    rigid=[]
    for master in np.unique(m['master_labels']):
        name=f'ARM_RB_{master}';ns=m['labels'][m['master_labels']==master]
        rigid += [f'*Nset,nset={name}']+[','.join(map(str,ns[i:i+16])) for i in range(0,len(ns),16)]
        rigid += [f'*Rigid Body,ref node={master},tie nset={name},position=INPUT']
    assert text.count('\n'.join(old))==1;text=text.replace('\n'.join(old),'\n'.join(rigid))
    # A node-based penalty surface must not mix distinct rigid bodies.
    a=text.index('*Surface, type=NODE, name=CAP_');b=text.index('*Surface Interaction, name=CAP_K')
    surfs=[];pairs=[]
    keys=sorted(set(m['stiffness_N_mm'].tolist()))
    for i,k in enumerate(keys):
        for master in np.unique(m['master_labels'][m['stiffness_N_mm']==k]):
            ix=np.flatnonzero((m['stiffness_N_mm']==k)&(m['master_labels']==master));s=int(m['side'][ix[0]])
            assert np.all(m['side'][ix]==s)
            name=f'RC_{master}_{i}';surfs += [f'*Surface,type=NODE,name={name}']+[f'{m["labels"][j]},1.' for j in ix]
            pairs += [f'*Contact Pair,interaction=CAP_K{i},mechanical constraint=PENALTY',f'{name},PLATEN_'+('BOTTOM' if s==0 else 'TOP')]
    text=text[:a]+'\n'.join(surfs)+'\n'+text[b:]
    a=text.index('*Contact Pair, interaction=CAP_K');b=text.index('*Boundary, amplitude=LOAD',a)
    text=text[:a]+'\n'.join(pairs)+'\n'+text[b:]
    # All force/penetration data can be recovered from U and frozen k_i.
    text=text.replace('*Contact Output\nCSTRESS,CDISP,CFORCE\n','')
    out=B/'abaqus/G_rigid';out.mkdir(exist_ok=False);(out/'platen_cap_rigid.inp').write_text(text)
    cfg.update(case_id='G_finite_cap_shell_rigid_arm_diagnostic',
        input_sha256=hashlib.sha256(text.encode()).hexdigest(),constraint_encoding='Independent local rigid node groups; no shell element made rigid',
        shell_contact=cfg['shell_contact'].replace('BEAM MPC arms','local rigid-body arms'),
        repair='Same sampling, weights, force law, shell and PBC. Replaced BEAM MPC star constraints by local rigid bodies; each node-based surface belongs to only one body.',
        rigid_group_count=len(np.unique(m['master_labels'])),contact_pair_count=len(pairs)//2)
    write(B/'input/protocol_rigid.json',cfg)
    probe=(B/'abaqus/probe/probe.inp').read_text().replace('*MPC\nBEAM,5,1','*Nset,nset=ARM\n5\n*Rigid Body,ref node=1,tie nset=ARM,position=INPUT')
    out=B/'abaqus/probe_rigid';out.mkdir(exist_ok=False);(out/'probe.inp').write_text(probe)
    write(B/'input/probe_rigid_protocol.json',{'stage':'Rigid-body arm verification, same prescribed shell and force law','load_time_s':.001,'hold_time_s':.0002})
    write(B/'input/repair_record.json',{'failed_attempt':'abaqus/G','stopped_before_time_advance':True,
        'error':'Very large implicit constraint system: 4050 variables, 130114350 estimated operations','first_solve_seconds':25.875,
        'no_solver_protection_override':True,'rigid_groups':cfg['rigid_group_count'],'contact_pairs':cfg['contact_pair_count'],
        'physical_added_mass':0.,'remaining_mass_note':'Explicit rigid-body augmented inertia from attached deformable nodes must be recorded; no point mass/rotary inertia added.'})
    print(json.dumps({'rigid_groups':cfg['rigid_group_count'],'pairs':cfg['contact_pair_count']},indent=2))
def wrapped():
    """XY translations do not affect a frictionless infinite horizontal plane."""
    with np.load(B/'input/cap_arm_map.npz') as z:m=dict(z)
    oldtext=(B/'abaqus/G_rigid/platen_cap_rigid.inp').read_text()
    cfg=json.loads((B/'input/protocol_rigid.json').read_text())
    ghosts=set(m['master_labels'])-set(m['parent_labels']);lines=oldtext.splitlines();new=[];i=0
    # Remove only ghost nodes and their six equations; preserve original PBC.
    while i<len(lines):
        line=lines[i]
        if line=='*Equation' and int(lines[i+2].split(',')[0]) in ghosts:i+=3;continue
        if line and not line.startswith('*'):
            try:label=int(line.split(',')[0])
            except ValueError:label=-1
            if label in ghosts and i<lines.index('*Element, type=S3R, elset=WALL'):i+=1;continue
        new.append(line);i+=1
    text='\n'.join(new)+'\n'
    for label,p,sh in zip(m['labels'],m['points_mm'],m['shift_xy']):
        before=f'{label},'+','.join(f'{x:.16g}' for x in p)
        pp=p-np.r_[sh*10,0]
        after=f'{label},'+','.join(f'{x:.16g}' for x in pp)
        assert text.count(before+'\n')==1;text=text.replace(before+'\n',after+'\n')
    m['unwrapped_points_mm']=m['points_mm'].copy();m['points_mm']=m['points_mm']-np.c_[m['shift_xy']*10,np.zeros(len(m['labels']))]
    m['master_labels']=m['parent_labels'].copy()
    a=text.index('*Nset,nset=ARM_RB_');b=text.index('*Surface Interaction, name=CAP_K')
    rb=[];surfs=[];pairs=[];keys=sorted(set(m['stiffness_N_mm'].tolist()))
    for master in np.unique(m['master_labels']):
        ns=m['labels'][m['master_labels']==master];name=f'ARM_RB_{master}'
        rb += [f'*Nset,nset={name}']+[','.join(map(str,ns[i:i+16])) for i in range(0,len(ns),16)]
        rb += [f'*Rigid Body,ref node={master},tie nset={name},position=INPUT']
    for ki,k in enumerate(keys):
        for master in np.unique(m['master_labels'][m['stiffness_N_mm']==k]):
            ix=np.flatnonzero((m['stiffness_N_mm']==k)&(m['master_labels']==master));s=int(m['side'][ix[0]]);assert np.all(m['side'][ix]==s)
            name=f'RC_{master}_{ki}';surfs += [f'*Surface,type=NODE,name={name}']+[f'{m["labels"][j]},1.' for j in ix]
            pairs += [f'*Contact Pair,interaction=CAP_K{ki},mechanical constraint=PENALTY',f'{name},PLATEN_'+('BOTTOM' if s==0 else 'TOP')]
    text=text[:a]+'\n'.join(rb+surfs)+'\n'+text[b:]
    a=text.index('*Contact Pair,interaction=CAP_K');b=text.index('*Boundary, amplitude=LOAD',a);text=text[:a]+'\n'.join(pairs)+'\n'+text[b:]
    out=B/'abaqus/G_wrapped';out.mkdir(exist_ok=False);(out/'platen_cap_wrapped.inp').write_text(text)
    np.savez_compressed(B/'input/cap_arm_wrapped_map.npz',**m)
    rng=np.random.default_rng(51);u=rng.normal(size=m['points_mm'].shape);jump=rng.normal(size=2)
    ghostshift=np.c_[m['shift_xy']*jump,np.zeros(len(u))]
    gaperror=float(np.max(abs((m['unwrapped_points_mm']+u+ghostshift)[:,2]-(m['points_mm']+u)[:,2])))
    cfg.update(case_id='G_finite_cap_shell_wrapped_rigid_arm_diagnostic',input_sha256=hashlib.sha256(text.encode()).hexdigest(),
        cap_map_sha256=hashlib.sha256((B/'input/cap_arm_wrapped_map.npz').read_bytes()).hexdigest(),
        shell_contact=cfg['shell_contact']+'; XY periodic images folded by horizontal translations, identical normal gap and local arm',
        rigid_group_count=len(np.unique(m['master_labels'])),contact_pair_count=len(pairs)//2,
        periodic_image_fold={'normal_gap_error_mm':gaperror,'added_ghost_equations':0,'relative_arm_unchanged':True,
                             'only_valid_for':'constant horizontal plane normal, frictionless contact, plane covering all points; not friction or general obstacle'},
        previous_packaging_failures=['abaqus/G','abaqus/G_rigid'])
    assert gaperror==0;assert np.max(abs(m['points_mm'][:,:2]))<15 and np.min(m['points_mm'][:,:2])>-5
    write(B/'input/protocol_wrapped.json',cfg)
    print(json.dumps({'groups':cfg['rigid_group_count'],'pairs':cfg['contact_pair_count'],'folding':cfg['periodic_image_fold']},indent=2))
if __name__=='__main__':
    import sys
    wrapped() if '--wrap' in sys.argv else main()
