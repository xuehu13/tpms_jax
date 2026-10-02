"""Run a separate M4 free-lateral N32 reference; preserve the original M4 CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.m4_numerical_study import evaluate_case

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    row=evaluate_case(32,20.,.001,'relaxed_free')
    if row['status'] != 'ok' or not all(row['checks'].values()):
        raise ValueError('M4 free-lateral consistency failed: '+str(row))
    root=Path(__file__).resolve().parents[1]
    row['solver']='Existing M4 default solver; global quantities only'
    row['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in ('scripts/m4_numerical_study.py','fem.py','density_fem.py','geometry.py','pbc.py')}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    print(json.dumps(row,indent=2),flush=True)
