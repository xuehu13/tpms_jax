"""Independent offline FE solves of the exported uniform constraints.

These are ordinary NumPy/SciPy solves, not Abaqus execution or acceptance.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

spec = importlib.util.spec_from_file_location(
    "prepare_uniform_baseline", Path(__file__).parents[1]/"scripts"/"prepare_uniform_baseline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("lateral", ["fixed", "relaxed_prescribed", "relaxed_free"])
def test_constraints_recover_analytic_uniform_response(n, lateral):
    nodes, cells, controls, equations, bcs = module.build_model(n, lateral)
    dofs = 3*len(nodes)+3
    def index(label, comp):
        return 3*(label-1)+comp-1 if label in nodes else 3*len(nodes)+controls.index(label)
    D = np.zeros((6,6))
    lam, mu = 10*0.3/(1.3*0.4), 10/2.6
    D[:3,:3] = lam
    D[np.arange(3),np.arange(3)] += 2*mu
    D[3:,3:] = mu*np.eye(3)
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    kr, kc, kv = [],[],[]
    for cell in cells:
        xyz = np.array([nodes[label] for label in cell])
        ke = np.zeros((24,24))
        for gx in (-1/np.sqrt(3),1/np.sqrt(3)):
            for gy in (-1/np.sqrt(3),1/np.sqrt(3)):
                for gz in (-1/np.sqrt(3),1/np.sqrt(3)):
                    g = np.array([gx,gy,gz])
                    gradients = np.empty((8,3))
                    for a in range(8):
                        for r in range(3):
                            other = [s for s in range(3) if s != r]
                            gradients[a,r] = signs[a,r]*np.prod(1+signs[a,other]*g[other])/8
                    jac = xyz.T@gradients
                    grad = gradients@np.linalg.inv(jac)
                    B = np.zeros((6,24))
                    for a,(dx,dy,dz) in enumerate(grad):
                        offset = 3*a
                        B[:,offset:offset+3] = [[dx,0,0],[0,dy,0],[0,0,dz],
                                                [dy,dx,0],[0,dz,dy],[dz,0,dx]]
                    ke += B.T@D@B*np.linalg.det(jac)
        cell_dofs = [index(label,comp) for label in cell for comp in (1,2,3)]
        kr.extend(np.repeat(cell_dofs,24))
        kc.extend(np.tile(cell_dofs,24))
        kv.extend(ke.ravel())
    K = coo_matrix((kv,(kr,kc)),shape=(dofs,dofs)).tocsr()
    ar, ac, av = [],[],[]
    for row,terms in enumerate(equations):
        for label,comp,coef in terms:
            ar.append(row); ac.append(index(label,comp)); av.append(coef)
    rhs = [0.0]*len(equations)
    for offset,(label,comp,value) in enumerate(bcs):
        ar.append(len(equations)+offset); ac.append(index(label,comp)); av.append(1.0)
        rhs.append(value)
    A = coo_matrix((av,(ar,ac)),shape=(len(rhs),dofs)).tocsr()
    system = bmat([[K,A.T],[A,csr_matrix((len(rhs),len(rhs)))]],format="csr")
    solution = spsolve(system,np.r_[np.zeros(dofs),rhs])
    u, multipliers = solution[:dofs],solution[dofs:]
    ex = 0.0 if lateral == "fixed" else 0.003
    strains = [ex,ex,-0.01]
    expected = np.zeros(dofs)
    for label,xyz in nodes.items():
        for comp in (1,2,3):
            expected[index(label,comp)] = xyz[comp-1]*strains[comp-1]
    for axis,label in enumerate(controls):
        expected[index(label,axis+1)] = strains[axis]
    assert np.max(np.abs(u-expected)) < 1e-11
    assert np.max(np.abs(A@u-rhs)) < 1e-11
    sigma_z = (lam+2*mu)*(-0.01)+lam*(2*ex)
    qz_bc = next(i for i,(label,comp,_) in enumerate(bcs) if label == controls[2])
    reaction = -multipliers[len(equations)+qz_bc]
    assert reaction == pytest.approx(sigma_z,rel=1e-10,abs=1e-12)
    assert 0.5*u@K@u == pytest.approx(0.5*sigma_z*(-0.01),rel=1e-10)
