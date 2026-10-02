"""Independent replicated scrambled Sobol estimate of analytic binary volume."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import qmc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from binary_gyroid import C,gyroid_numpy

def estimate(power=21,replicates=8):
    fractions=[]
    for seed in range(replicates):
        xyz=qmc.Sobol(3,scramble=True,seed=20261002+seed).random_base2(power)
        fractions.append(float(np.mean(np.abs(gyroid_numpy(xyz)) <= C)))
    return {'c':C,'method':'replicated scrambled Sobol, direct analytic |G|<=c',
            'power':power,'replicates':replicates,'samples_per_replicate':2**power,
            'fractions':fractions,'mean':float(np.mean(fractions)),
            'replicate_std':float(np.std(fractions,ddof=1)),
            'standard_error_of_mean':float(np.std(fractions,ddof=1)/np.sqrt(replicates)),
            'deterministic_error_bound':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();report=estimate();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
