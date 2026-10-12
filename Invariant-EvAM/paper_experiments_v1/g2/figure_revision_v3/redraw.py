from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g2';V=G/'figure_revision_v3'
sys.path.insert(0,str(R/'src'))
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'svg.fonttype':'none','pdf.fonttype':42,'font.size':8})
for f in ['figures','data','previews']:(V/f).mkdir(exist_ok=True)
P=['#CC79A7','#D55E00','#E69F00','#009E73','#56B4E9','#0072B2'];methods=['B1','B2','B3','B4','B5','M'];designs=['strong','weak','nonidentifying'];T=[-.8,0,.15,.8];manifest=[]
def save(fig,name,d,ref,desc):
 for ext in ['pdf','svg']:fig.savefig(V/'figures'/f'{name}.{ext}',dpi=600)
 fig.savefig(V/'previews'/f'{name}.png',dpi=150);d.to_csv(V/'data'/f'{name}.tsv',sep='\t',index=False)
 manifest.append(dict(name=name,reference=ref,rationale=desc,size_mm=(fig.get_size_inches()*25.4).tolist(),dpi=600,palette=P,review_status='awaiting_visual_review',data_sha256=hashlib.sha256((V/'data'/f'{name}.tsv').read_bytes()).hexdigest()));plt.close(fig)
S=pd.read_csv(G/'results/G2_core_summary.tsv',sep='\t');S=S[S.method_id.isin(methods)].copy();assert len(S)==72
fig,ax=plt.subplots(figsize=(230/25.4,110/25.4));fig.subplots_adjust(left=.085,right=.985,bottom=.27,top=.80)
for mi,m in enumerate(methods):
 f=S[S.method_id==m].sort_values('cell_id');x=f.cell_id.to_numpy()+(mi-2.5)*.13;y=(1-f.conditional_coverage.to_numpy())*100
 ax.bar(x,y,width=.12,color=P[mi],edgecolor='#333333',lw=.45,label=m,zorder=2)
 ax.errorbar(x,y,yerr=np.maximum(0,[y-100*(1-f.coverage_wilson_high),100*(1-f.coverage_wilson_low)-y]),fmt='none',ecolor='#222222',elinewidth=.55,capsize=1,zorder=3)
 for xx,yy,(_,r) in zip(x,y,f.iterrows()):
  if m=='M':ax.text(xx,100*(1-r.coverage_wilson_low)+.65,f'{yy:.1f}',rotation=90,ha='center',va='bottom',fontsize=6,color='#222222')
ax.axhline(5,color='#555555',ls=(0,(3,2)),lw=.8,zorder=4)
ax.set(ylim=(0,34),xlim=(-.65,11.65),xticks=range(12),xticklabels=[f'{t:+.2g}' for d in designs for t in T],ylabel='Noncoverage among completed intervals (%)')
ax.spines[['top','right']].set_visible(False);ax.legend(ncol=6,loc='lower center',bbox_to_anchor=(.5,1.04),frameon=False)
for di,label in enumerate(['Complementary','Repeated family','Separate panels']):
 ax.text(di*4+1.5,-.16,label,ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
 ax.plot([di*4-.4,di*4+3.4],[-.09,-.09],transform=ax.get_xaxis_transform(),color='#555555',lw=.6,clip_on=False)
fig.text(.085,.045,'True interaction shown below each group; 500 planned replicates per condition. Dashed line: nominal 5% noncoverage.\nWhiskers: Wilson 95% Monte Carlo intervals. Separate-panel failures: B1 32, B2 0, B3 15, B4 70, B5 2, M 65.',fontsize=7,linespacing=1.5)
save(fig,'G2_v3_02_coverage_bars',S,'R21','72 categorical probabilities; plot 100% minus coverage to expose small deviations using a zero baseline, without a broken axis; replicate binary endpoints are not continuous clouds. Reproduce narrow dark-edge grouped bars, capped black errors, horizontal upper legend and zero baseline. No invented replicate scatter/P values; every method and condition retained.')
Q=pd.read_csv(G/'results/G2_paired_comparisons.tsv',sep='\t');rows=[]
for cell in range(12):
 a=Q[(Q.cell_id==cell)&(Q.comparison=='M_vs_B5')&(Q.metric=='correct_call')].iloc[0];b=Q[(Q.cell_id==cell)&(Q.comparison=='M_vs_B4')&(Q.metric=='correct_call')].iloc[0]
 rows.append(dict(cell=cell,design=designs[cell//4],truth=T[cell%4],x=100*a.paired_difference,y=100*b.paired_difference,x_low=100*a.paired_normal95_low,x_high=100*a.paired_normal95_high,y_low=100*b.paired_normal95_low,y_high=100*b.paired_normal95_high,n=500))
d=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(135/25.4,140/25.4));fig.subplots_adjust(left=.18,right=.96,bottom=.23,top=.93)
ax.axhline(0,color='#555555',lw=.8);ax.axvline(0,color='#555555',lw=.8);ax.plot([-15,105],[-15,105],color='#aaaaaa',ls=(0,(3,3)),lw=.8)
for design,c,m in zip(designs,['#0072B2','#009E73','#CC79A7'],['o','s','^']):
 f=d[d.design==design];ax.scatter(f.x,f.y,s=34,c=c,marker=m,edgecolor='#333333',lw=.6,label={'strong':'Complementary','weak':'Repeated family','nonidentifying':'Separate panels'}[design],zorder=4)
 for _,r in f.iterrows():
  if r.x==0 and r.y==0:continue
  ax.errorbar(r.x,r.y,xerr=[[r.x-r.x_low],[r.x_high-r.x]],yerr=[[r.y-r.y_low],[r.y_high-r.y]],fmt='none',ecolor=c,elinewidth=.6,capsize=1.5,alpha=.7,zorder=2)
  offset={0:(8,9),2:(-1,17),3:(10,7),4:(-32,12),6:(9,-12),7:(-36,12)}[int(r.cell)]
  ax.annotate(f'{r.truth:+.2g}',(r.x,r.y),xytext=offset,textcoords='offset points',fontsize=7,arrowprops=dict(arrowstyle='-',color='#777777',lw=.55))
ax.annotate('6 conditions at (0, 0)',(0,0),xytext=(19,-10),textcoords='offset points',fontsize=7,arrowprops=dict(arrowstyle='-',color='#777777',lw=.55))
ax.set(xlim=(-15,105),ylim=(-15,105),xlabel='Correct-call gain over B5 (percentage points)',ylabel='Correct-call gain over B4 (percentage points)');ax.set_aspect('equal');ax.legend(loc='lower right',fontsize=7,frameon=False)
fig.text(.18,.04,'One point per condition; labels give true interaction. No jitter.\nCrossbars: paired 95% Monte Carlo intervals; diagonal: equal gains.\nOriginal rates and all zero-discordance cases remain in source tables.',fontsize=7,linespacing=1.5)
save(fig,'G2_v3_03_paired_gain_scatter',d,'R13','Both axes are paired correct-call improvements for the same condition, an actual two-contrast dataset. Reproduce square frame, solid zero cross, pale equality diagonal, outlined colored points and leader labels. Add marginal paired uncertainty. Six exact zero points counted explicitly; sparse 12-condition data not fabricated into a dense cloud.')
B=pd.read_csv(G/'results/B6_matched_summary.tsv',sep='\t');fig,ax=plt.subplots(figsize=(170/25.4,113/25.4));fig.subplots_adjust(left=.13,right=.975,bottom=.27,top=.81)
rows=[]
for mi,(m,c) in enumerate([('B6_OFFICIAL','#56B4E9'),('M_MATCHED','#009E73')]):
 for ei,e in enumerate(['coverage','correct_calls']):
  f=B[(B.method_id==m)&(B.endpoint==e)].sort_values('truth');x=np.arange(4)+ei*4.7+(mi-.5)*.30;y=f.value.to_numpy()*100
  ax.bar(x,y,width=.27,color=c,edgecolor='#333333',lw=.65,label=('Official + Wald' if mi==0 else 'M + profile') if ei==0 else None)
  ax.errorbar(x,y,yerr=np.maximum(0,[y-100*f.wilson_low,100*f.wilson_high-y]),fmt='none',ecolor='#222222',elinewidth=.65,capsize=2)
  for xx,yy,(_,r) in zip(x,y,f.iterrows()):ax.text(xx,100*r.wilson_high+1.2,f'{yy:g}',ha='center',va='bottom',fontsize=6.5,rotation=90)
  rows.extend(f.to_dict('records'))
ax.set(ylim=(0,112),xticks=list(range(4))+list(np.arange(4)+4.7),xticklabels=[f'{t:+.2g}' for e in range(2) for t in T],ylabel='Replicates (%)');ax.spines[['top','right']].set_visible(False);ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.05),frameon=False)
for center,l in [(1.5,'Coverage'),(6.2,'Correct sign calls')]:ax.text(center,-.15,l,ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
fig.text(.13,.045,'True interaction below each group; 200 paired replicates per effect; no numerical failures.\nWhiskers: Wilson 95% Monte Carlo intervals. Point estimates agree; interval procedures differ.',fontsize=7,linespacing=1.5)
save(fig,'G2_v3_04_matched_bars',pd.DataFrame(rows),'R21','Two methods × four effects × two probability endpoints. Single shared zero-based axis preserves all 16 estimates; mimic dense categorical groups, dark outlines, black errors and upper legend. Binary raw points not artificially jittered; no fabricated significance marks.')
(V/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print('Generated 3 replacement figures')
