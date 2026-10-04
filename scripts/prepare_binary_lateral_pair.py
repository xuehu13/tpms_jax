"""Create a free-lateral INP on exactly the analyzed fixed-lateral mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_abaqus_binary import prepare


def prepare_pair(expected_path,output):
    fixed=json.loads(expected_path.read_text())
    if fixed['model'] != 'binary_sheet_gyroid' or fixed['lateral'] != 'fixed':
        raise ValueError('Expected a fixed-lateral binary model')
    mesh_path=expected_path.with_name(fixed['case']+'.mesh.npz')
    if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != fixed['mesh_sha256']:
        raise ValueError('Source mesh hash differs')
    mesh=np.load(mesh_path)
    output.mkdir(parents=True,exist_ok=True)
    cache=output/(fixed['case']+'.linear_cache.npz')
    np.savez_compressed(cache,points=mesh['points'][:fixed['geometry']['nodes']],
                        cells=mesh['cells'][:,:4],geometry_N=fixed['geometry_N'],
                        c=fixed['c'],fe_refinement=fixed['fe_refinement'])
    result=prepare(output,fixed['geometry_N'],fixed['fe_refinement'],'relaxed_free',cached_mesh=cache,c=fixed['c'])
    paired=np.load(output/(result['case']+'.mesh.npz'))
    if not np.array_equal(paired['points'],mesh['points']) or not np.array_equal(paired['cells'],mesh['cells']):
        raise ValueError('Paired mesh differs from source fixed-lateral mesh')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-expected',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();prepare_pair(args.fixed_expected,args.output)
