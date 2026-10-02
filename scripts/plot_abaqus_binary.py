"""Static scientific figure from the archived real analysis results."""
import argparse,csv,json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from binary_gyroid import boundary_faces,gyroid_numpy

def plot(evidence,package,out):
    report=json.loads((evidence/'summary.json').read_text())
    mesh=np.load(package/'binary_gyroid_G24_R0_C3D10_fixed.mesh.npz')
    points,cells=mesh['points'],mesh['cells'][:,:4]
    faces=boundary_faces(cells);xyz=points[faces];centers=xyz.mean(axis=1)
    caps=np.zeros(len(faces),bool)
    for axis in range(3):
        caps |= np.all(xyz[:,:,axis]==0,axis=1)|np.all(xyz[:,:,axis]==1,axis=1)
    colors=np.where((gyroid_numpy(centers)>0)[:,None],np.array([.89,.48,.22,1.]),np.array([.15,.43,.66,1.]))
    colors[caps]=[.7,.72,.76,1.]
    fig=plt.figure(figsize=(13,4.4),layout='constrained')
    ax=fig.add_subplot(131,projection='3d')
    ax.add_collection3d(Poly3DCollection(xyz,facecolors=colors,linewidths=0,rasterized=True))
    ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='x',ylabel='y',zlabel='z')
    ax.set_box_aspect((1,1,1));ax.view_init(23,36)
    ax.set_title('Binary sheet Gyroid\nG=24 planar boundary, C3D10 solid',fontsize=10)
    cases=report['cases'];fixed=sorted([r for r in cases if r['geometry_N'] and r['refinement']==0 and r['lateral']=='fixed'],key=lambda r:r['geometry_N'])
    ax=fig.add_subplot(132)
    ax.plot([r['geometry_N'] for r in fixed],[r['volume'] for r in fixed],'o-',label='Meshed binary volume')
    ax.axhline(report['analytic_volume_sampling']['mean'],color='k',ls='--',label='Analytic binary (Sobol)')
    ax.set(xlabel='Geometry resolution G',ylabel='Solid / gross volume',title='Geometry convergence')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=fig.add_subplot(133)
    ax.plot([r['geometry_N'] for r in fixed],[abs(r['Fz']) for r in fixed],'o-',label='Binary, FE R=0')
    refined=sorted([r for r in cases if r['geometry_N'] and r['refinement']==1 and r['lateral']=='fixed'],key=lambda r:r['geometry_N'])
    ax.plot([r['geometry_N'] for r in refined],[abs(r['Fz']) for r in refined],'s',ms=7,label='Same binary geometry, FE R=1')
    m4=list(csv.DictReader((evidence.parent/'m4_review/fixed.csv').open()))
    m4=[r for r in m4 if float(r['beta'])==20 and float(r['emin_ratio'])==.001]
    ax.plot([int(r['N']) for r in m4],[abs(float(r['Fz_top'])) for r in m4],'x--',label='M4 projection, beta=20')
    ax.set(xlabel='G (binary) / N (projection)',ylabel='Axial reaction magnitude',title='Fixed lateral, eps_z = -0.01')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out,dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',required=True,type=Path);p.add_argument('--package',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();plot(a.evidence,a.package,a.out)
