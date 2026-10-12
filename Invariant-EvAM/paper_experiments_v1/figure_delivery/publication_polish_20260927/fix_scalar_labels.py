from pathlib import Path
import sys,json
import pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;P=O.parents[1];sys.path.insert(0,str(P.parent/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'font.family':'Arial','font.size':7,'axes.labelsize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
n='G5_R6_diagnostic_value';D=P/'g5/final_results/plot_data';q=pd.read_csv(D/(n+'.tsv'),sep='\t');pts=pd.read_csv(D/(n+'_points.tsv'),sep='\t')
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];labels=['Fixed B','Fixed A','Random','Shuffled\ndiagnosis','New\nsource','D-opt','Largest\npanel']
f=plt.figure(figsize=(177/25.4,119/25.4));a=f.add_axes([.105,.24,.865,.61])
for j,r in enumerate(q.itertuples()):
 col=C[j%6];a.bar(j,r.gain,width=.62,facecolor='white',edgecolor=col,lw=1.6,zorder=2);a.errorbar(j,r.gain,yerr=[[r.gain-r.lo],[r.hi-r.gain]],fmt='none',ecolor='#333333',capsize=3,lw=.8,zorder=4)
 vals=pts[pts.reference==r.reference].sort_values('case').gain.values;assert len(vals)==6
 a.scatter(j+np.linspace(-.15,.15,6),vals,s=12,c=col,ec='white',lw=.3,zorder=5)
 star='***' if r.p_holm<.001 else '**' if r.p_holm<.01 else '*' if r.p_holm<.05 else 'ns'
 a.text(j,max(r.hi,max(vals))+.06,star,ha='center',va='bottom',fontsize=6.4)
a.axhline(0,c='#888888',lw=.65);a.set(xticks=range(len(q)),xticklabels=labels,ylabel='Target squared-error reduction',ylim=(-.12,max(max(q.hi),pts.gain.max())+.27));a.spines[['top','right']].set_visible(False)
a.text(.99,.99,'vs zero; Holm-adjusted',transform=a.transAxes,ha='right',va='top',fontsize=6.5,color='#333333')
for ext in ['pdf','svg']:f.savefig(P/'g5/final_figures'/f'{n}.{ext}',dpi=600)
plt.close(f)
