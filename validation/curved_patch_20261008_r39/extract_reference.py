"""Read one completed r39 Abaqus reference; retain failed validation results."""
from pathlib import Path
import argparse,json,math

def value(v):
    try:return v.dataDouble
    except Exception:return v.data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('odb',type=Path);ap.add_argument('input',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();cfg=json.loads(a.input.read_text());from odbAccess import openOdb
    odb=openOdb(str(a.odb.resolve()),readOnly=True)
    try:
        step=odb.steps['INITIAL'];frame=step.frames[-1];rp=cfg['control_label']
        data={v.nodeLabel:list(map(float,value(v))) for v in frame.fieldOutputs['U'].values}
        delta=data[rp][0];F=cfg['generalized_force_N']
        energy={n:[float(h.historyOutputs[n].data[-1][1]) for h in step.historyRegions.values() if n in h.historyOutputs] for n in ['ALLSE','ALLAE','ALLWK']}
        assert all(len(v)==1 for v in energy.values());energy={k:v[0] for k,v in energy.items()}
        s=(a.odb.parent/(a.odb.stem+'.inp')).read_text().splitlines();gaps=[]
        for i,line in enumerate(s):
            if line.lower().strip()=='*equation':
                fields=[x.strip() for x in s[i+2].split(',')];nt=int(s[i+1]);assert len(fields)==3*nt
                gaps.append(abs(sum(float(fields[3*k+2])*data[int(fields[3*k])][int(fields[3*k+1])-1] for k in range(nt))))
        total=energy['ALLSE']+energy['ALLAE'];work=.5*F*delta;error=abs(total/work-1);art=abs(energy['ALLAE'])/abs(work)
        checks={'completed':abs(frame.frameValue-1)<1e-8,'finite_positive':math.isfinite(delta) and delta>0,
                'energy_work':error<=1e-5,'artificial_energy':art<=.01,'equations':max(gaps,default=0)<=1e-9}
        result={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,'job':a.odb.stem,
                'K_N_per_mm':F/delta,'mid_radial_u_mm':delta,'generalized_force_N':F,'energies_N_mm':energy,
                'energy_work_relative_error':error,'artificial_energy_fraction':art,'max_equation_gap_mm':max(gaps,default=0),
                'odb_path':str(a.odb.resolve())}
    finally:odb.close()
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='ok':raise RuntimeError('Reference checks failed; preserve outcome')

if __name__=='__main__':main()
