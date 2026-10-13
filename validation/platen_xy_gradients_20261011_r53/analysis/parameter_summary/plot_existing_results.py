"""Plot frozen CSV evidence only. Does not import FEM, JAX, or a solver."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

S = Path(__file__).resolve().parent
def read(name):
    with (S/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
table = read('gradient_table.csv')
path = read('shape_saved_path.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':190})
blue,orange,green,red = '#234f78','#c37b23','#39765c','#a53232'
def finish(fig, name):
    fig.savefig(S/name,bbox_inches='tight',facecolor='white');plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(11.6,4.7),constrained_layout=True)
for ax,p in zip(axes,['t','w']):
    for obj,col,mark,label in [('loading_work',blue,'o','Loading work'),('terminal_upper_force',orange,'s','Terminal force')]:
        rows=[r for r in table if r['configuration']=='G' and r['parameter']==p and r['objective']==obj]
        rows.sort(key=lambda r:float(r['compression_actual_percent']))
        x=np.array([float(r['compression_actual_percent']) for r in rows]);
        ad=np.array([float(r['dimensionless_sensitivity']) for r in rows])
        fd=np.array([float(r['parameter_scale'])*float(r['FD_half'])/float(r['baseline_objective']) for r in rows])
        ax.plot(x,ad,color=col,marker=mark,label=label+' AD')
        ax.scatter(x,fd,facecolors='white',edgecolors=col,marker='D',s=48,zorder=4,label=label+' FD(h/2)')
        for xi,yi,r in zip(x,ad,rows):
            if r['passed']=='False':ax.scatter([xi],[yi],color=red,marker='x',s=100,zorder=5)
    if p=='t':
        for r in table:
            if r['configuration']=='P' and r['parameter']=='t':
                ax.scatter([5],[float(r['dimensionless_sensitivity'])],marker='^' if r['objective']=='loading_work' else 'v',s=75,color=green,zorder=4,
                    label='P5% '+('work' if r['objective']=='loading_work' else 'force'))
    ax.set_title('Physical thickness t' if p=='t' else 'Numerical interface width w')
    ax.set_xlabel('Compression (%)');ax.set_ylabel('Scaled sensitivity  S = p_scale / Q0 * dQ/dp')
    ax.set_xlim(4.6,12.4);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left' if p=='w' else 'lower right')
axes[1].text(.04,.04,'Red x: G12% force gradient failed the 1% gate.\nNo G5% width-force AD check.',transform=axes[1].transAxes,fontsize=8,color=red,bbox=dict(facecolor='white',edgecolor='none',alpha=.95))
fig.suptitle('Sensitivity at the validated endpoints; lines only connect tested endpoints',fontsize=12)
finish(fig,'parameter_sensitivity_vs_compression.png')

x=np.array([float(r['compression_percent']) for r in path])
fig,ax=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
for j,(key,unit,name) in enumerate([('work','N mm','Loading work'),('force','N','Upper platen force')]):
    ax[0,j].plot(x,[float(r['baseline_'+key+('_N_mm' if key=='work' else '_N')]) for r in path],color=blue)
    ax[0,j].set_title('Baseline '+name.lower());ax[0,j].set_ylabel(unit)
    for suffix,col,ls,label in [('AD_'+('N_mm' if key=='work' else 'N'),blue,'-','Complete-path JVP'),
                               ('FD_h',orange,'--','Actual geometry FD, h=1e-7'),
                               ('FD_half',green,':','Actual geometry FD, h=5e-8')]:
        ax[1,j].plot(x,[float(r[key+'_'+suffix]) for r in path],color=col,ls=ls,lw=1.2,label=label)
    ax[1,j].set_title('Shape derivative of '+name.lower());ax[1,j].set_ylabel('dQ/da ('+unit+')')
    ax[1,j].axhline(0,color='gray',lw=.6);ax[1,j].ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    ax[1,j].legend(fontsize=8,loc='best')
for a in ax.flat:a.grid(alpha=.2);a.set_xlabel('Compression (%)');a.set_xlim(0,5)
ax[1,1].text(.98,.97,'5% endpoint: AD/FD 9.54% (FAIL)\nFD-halving 14.97% (FAIL)',
    transform=ax[1,1].transAxes,ha='right',va='top',fontsize=8,color=red,
    bbox=dict(facecolor='white',edgecolor='#cccccc',alpha=.95))
fig.suptitle('G shape path: 388 logged samples from existing complete 0-to-5% runs',fontsize=12)
finish(fig,'shape_response_and_gradients.png')

labels=['Thickness / work','Thickness / force','Width / work','Width / force','Shape a / work','Shape a / force']
cols=[('G',5),('G',10),('G',12),('P',5)]
grid=np.zeros((6,4));notes=[['Not checked']*4 for _ in range(6)]
for r in table:
    if r['phase']=='G5_a_original':continue
    col=next((i for i,(g,c) in enumerate(cols) if r['configuration']==g and float(r['compression_nominal_percent'])==c),None)
    if col is None:continue
    row={'t':0,'w':2,'a':4}[r['parameter']]+(r['objective']!='loading_work')
    passed=r['passed']=='True';grid[row,col]=1 if passed else 2
    notes[row][col]=('PASS' if passed else 'FAIL')+'\nAD/FD '+format(float(r['AD_FD_percent']),'.3g')+'%\nFD/FD '+format(float(r['FD_halving_percent']),'.3g')+'%'
notes[3][0]='FD only\nNo AD check'
fig,ax=plt.subplots(figsize=(11,6.3),constrained_layout=True)
ax.imshow(grid,cmap=ListedColormap(['#ededed','#d9e7de','#efd7d5']),vmin=0,vmax=2,aspect='auto')
ax.set_xticks(range(4),['G 5%','G ~10%','G ~12%','P 5%']);ax.set_yticks(range(6),labels)
for i in range(6):
    for j in range(4):ax.text(j,i,notes[i][j],ha='center',va='center',fontsize=9,color=red if grid[i,j]==2 else '#25313b')
ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True)
ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='both',length=0)
ax.set_title('Current full-path coverage: both AD/FD and FD-halving must pass\nOriginal shape-force failure is retained separately; no shape VJP was run',pad=16)
finish(fig,'verification_coverage.png')

names=['gradient_table.csv','shape_saved_path.csv','plot_existing_results.py',
       'parameter_sensitivity_vs_compression.png','shape_response_and_gradients.png','verification_coverage.png']
(S/'plot_receipt.json').write_text(json.dumps({'scope':'Postprocessing of existing data only','new_FE_AD_FD':0,
    'source_and_output_sha256':{n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in names}},indent=2)+'\n')
print('Created 3 scientific figures from existing CSV data; no new calculations.')
