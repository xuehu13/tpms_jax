"""Generate uniform XY-periodic Abaqus input decks; never submit a job.

Run with ordinary Python. No JAX, CAE, USDFLD, or compiler is needed.
"""
import argparse
import json
from pathlib import Path


def build_model(n, lateral):
    if n < 1 or lateral not in ("fixed", "relaxed_prescribed", "relaxed_free"):
        raise ValueError("Invalid grid or lateral condition")
    def node(i, j, k):
        return 1 + (i * (n + 1) + j) * (n + 1) + k
    nodes = {node(i, j, k): (i / n, j / n, k / n)
             for i in range(n + 1) for j in range(n + 1) for k in range(n + 1)}
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                cells.append([node(i,j,k), node(i+1,j,k), node(i+1,j+1,k),
                              node(i,j+1,k), node(i,j,k+1), node(i+1,j,k+1),
                              node(i+1,j+1,k+1), node(i,j+1,k+1)])
    qx, qy, qz = len(nodes) + 1, len(nodes) + 2, len(nodes) + 3
    equations = []
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                slave, master = node(i,j,k), node(i % n,j % n,k)
                if slave == master:
                    continue
                for comp in (1, 2, 3):
                    if comp == 3 and k in (0, n):
                        continue  # These z DOFs have bottom BCs or top equations.
                    terms = [(slave, comp, 1.0), (master, comp, -1.0)]
                    if comp == 1 and i == n:
                        terms.append((qx, 1, -1.0))
                    if comp == 2 and j == n:
                        terms.append((qy, 2, -1.0))
                    equations.append(terms)
    for i in range(n + 1):
        for j in range(n + 1):
            equations.append([(node(i,j,n), 3, 1.0), (qz, 3, -1.0)])
    bcs = [(node(i,j,0), 3, 0.0) for i in range(n + 1) for j in range(n + 1)]
    bcs += [(node(0,0,0), 1, 0.0), (node(0,0,0), 2, 0.0), (qz, 3, -0.01)]
    if lateral != "relaxed_free":
        strain = 0.0 if lateral == "fixed" else 0.003
        bcs += [(qx, 1, strain), (qy, 2, strain)]
    return nodes, cells, (qx,qy,qz), equations, bcs


def validate_model(n, model):
    """Check elimination restrictions and analytic affine field independently."""
    nodes, cells, controls, equations, bcs = model
    eliminated = [terms[0][:2] for terms in equations]
    bc_dofs = [(label,comp) for label,comp,_ in bcs]
    if len(set(eliminated)) != len(eliminated) or len(set(bc_dofs)) != len(bc_dofs):
        raise ValueError("Duplicate eliminated or boundary DOF")
    if set(eliminated) & set(bc_dofs):
        raise ValueError("Boundary condition on an eliminated DOF")
    later_refs = {term[:2] for terms in equations for term in terms[1:]}
    if set(eliminated) & later_refs:
        raise ValueError("Eliminated DOF referenced in another equation")
    for strain in ((0.0,0.0,-0.01),(0.003,0.003,-0.01)):
        values = {(label,comp+1): xyz[comp]*strain[comp]
                  for label,xyz in nodes.items() for comp in range(3)}
        values.update({(controls[i],i+1): strain[i] for i in range(3)})
        error = max(abs(sum(c*values[(label,comp)] for label,comp,c in terms))
                    for terms in equations)
        if error > 1e-14:
            raise ValueError("Affine displacement violates a constraint")
    if len(cells) != n**3 or len(nodes) != (n+1)**3:
        raise ValueError("Unexpected mesh size")
    # Regular-cell Jacobian at its center, evaluated without external packages.
    signs = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
             (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    for cell in cells:
        jac = [[sum(signs[a][r]*nodes[cell[a]][c]/8 for a in range(8))
                for c in range(3)] for r in range(3)]
        determinant = (jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])
                       -jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])
                       +jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]))
        if abs(determinant-1/(8*n**3)) > 1e-14:
            raise ValueError("Incorrect element orientation or volume")
    expected = 3*n*n*(n+1)-2*n*n-2
    free_dofs = 3*len(nodes)+3-len(equations)-len(bcs)
    if free_dofs not in (expected,expected+2):
        raise ValueError("Unexpected independent DOF count")
    return {"nodes":len(nodes), "elements":len(cells),
            "equations":len(equations), "independent_dofs":free_dofs,
            "min_detJ":1/(8*n**3), "total_volume":1.0,
            "checks":"unique eliminations, no BC overlap, affine jumps, positive Jacobians"}


def render(n, lateral, model):
    nodes, cells, controls, equations, bcs = model
    lines = ["*HEADING", "Uniform unit cube: XY periodic, flat axial faces",
             "** E=10, nu=0.3, eps_z=-0.01, pbc macro DOFs at QX/QY/QZ",
             "** Prepared only; no analysis has been submitted.", "*NODE, NSET=PHYSICAL"]
    lines += [f"{label}, {x:.16g}, {y:.16g}, {z:.16g}" for label,(x,y,z) in nodes.items()]
    for name,label in zip(("QX","QY","QZ"),controls):
        lines += [f"*NODE, NSET={name}", f"{label}, 0., 0., 0."]
    lines += ["*ELEMENT, TYPE=C3D8, ELSET=SOLID"]
    lines += [f"{index}, " + ", ".join(map(str,cell)) for index,cell in enumerate(cells,1)]
    for name,z in (("TOP",1.0),("BOTTOM",0.0)):
        ids = [str(label) for label,xyz in nodes.items() if xyz[2] == z]
        lines.append(f"*NSET, NSET={name}")
        lines += [", ".join(ids[a:a+16]) for a in range(0,len(ids),16)]
    lines += ["*MATERIAL, NAME=UNIFORM", "*ELASTIC", "10., 0.3",
              "*SOLID SECTION, ELSET=SOLID, MATERIAL=UNIFORM", ","]
    for terms in equations:
        lines += ["*EQUATION",str(len(terms)),
                  ", ".join(f"{label}, {comp}, {coef:.16g}" for label,comp,coef in terms)]
    lines += ["*STEP, NAME=COMPRESSION, NLGEOM=NO", "*STATIC", "1., 1.", "*BOUNDARY"]
    lines += [f"{label}, {comp}, {comp}, {value:.16g}" for label,comp,value in bcs]
    lines += ["*OUTPUT, FIELD, FREQUENCY=1", "*NODE OUTPUT", "U, RF",
              "*ELEMENT OUTPUT, ELSET=SOLID", "S, E, IVOL",
              "*OUTPUT, HISTORY, FREQUENCY=1", "*ENERGY OUTPUT", "ALLSE",
              "*NODE OUTPUT, NSET=QX", "U1, RF1", "*NODE OUTPUT, NSET=QY", "U2, RF2",
              "*NODE OUTPUT, NSET=QZ", "U3, RF3", "*END STEP"]
    return "\n".join(lines)+"\n"


def prepare(out, n=4):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"prepared_only":True, "abaqus_jobs_submitted":0,
                "n":n, "E":10.0, "nu":0.3, "length":1.0,
                "jax_reference_commit":"d75a7daf07dec927562e9c222833a1d6a61f8152",
                "relative_acceptance_tolerance":1e-6, "cases":{}}
    for lateral in ("fixed","relaxed_prescribed","relaxed_free"):
        model = build_model(n,lateral)
        checks = validate_model(n,model)
        fixed = lateral == "fixed"
        ez, ex, ey = -0.01, (0.0 if fixed else 0.003), (0.0 if fixed else 0.003)
        lam = 10*0.3/(1.3*0.4)
        mu = 10/2.6
        sigma = [lam*(ex+ey+ez)+2*mu*v for v in (ex,ey,ez)]
        energy = sum(s*e for s,e in zip(sigma,(ex,ey,ez)))/2
        case_name = "uniform_xy_"+lateral
        (out/(case_name+".inp")).write_text(render(n,lateral,model), encoding="ascii")
        manifest["cases"][case_name] = dict(lateral=lateral, controls=list(model[2]),
            eps=[ex,ey,ez], sigma_diagonal=sigma, Fz_top=sigma[2],
            energy=energy, offline_checks=checks)
    (out/"expected.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("validation/abaqus_uniform"))
    parser.add_argument("--n",type=int,default=4)
    args = parser.parse_args()
    prepare(args.out,args.n)
