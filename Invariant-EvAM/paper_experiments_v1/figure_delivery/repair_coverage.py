from pathlib import Path
import sys,types,json,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];R=P.parent;OUT=Path('/tmp/figure_coverage_repair');OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(P/'reviewer_audit/redesign_r2'))
LOG=[]
def save(fig,g,name,df,ref,reason):
 for ext in ['pdf','svg']:fig.savefig(P/f'g{g}/final_figures/{name}.{ext}',dpi=600)
 fig.savefig(OUT/f'{name}.png',dpi=150)
 LOG.append(dict(group=f'g{g}',name=name,reference=ref,reason=reason,size_mm=(fig.get_size_inches()*25.4).tolist()))
 plt.close(fig)
def load(path,name,transform=lambda s:s):
 mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod;exec(compile(transform(path.read_text()),str(path),'exec'),mod.__dict__);return mod
# Restore the legacy alias endpoint: finite-sample recovery does not replace it.
def t1(s):
 s=s.split(' m=json.loads((folder/')[0]
 s=s.replace("for ext in ['pdf','svg']:fig.savefig(folder/f'{fid}.{ext}',dpi=600,facecolor='white')", "for ext in ['pdf','svg']:fig.savefig(G/'final_figures'/f'{fid}.{ext}',dpi=600,facecolor='white')")
 return s.replace("preview=G/'reviewer_audit/style_v3'", "preview=Path('/tmp/figure_coverage_repair')")
a=load(P/'g1/src/g1_plot.py','restore_g1',t1);a.plot_panel('G1A',P/'g1/figure_data/G1A');LOG.append(dict(group='g1',name='G1A',reference='R25',reason='Parameter alias/nonidentification endpoint not replaced by recovery MAE'))
b=load(P/'reviewer_audit/redesign_r2/plot_r2.py','restore_r2');b.save=lambda fig,g,name,df,ref,adapt:save(fig,g,name,df,ref,'Retained exact log-OR bias-bound branch; not generator-scale R3') if name.startswith('G3_R2_bias_') else plt.close(fig);b.g3()
# Reuse final line-chart implementation; suppress writes to frozen source tables/configs.
def t5(s):return '\n'.join(line for line in s.replace("OUT=G/'FINAL_FIGURES'", "OUT=G/'final_figures'").splitlines() if not line.startswith("(G/'configs/DISPLAY_SUBSAMPLING.json').write_text"))
c=load(P/'g5/src/g5_plot.py','g5_plot',t5);c.save=lambda fig,name,df,ref,endpoint,adapt:save(fig,5,name,df,ref,'Original multibudget simulation endpoint; high-cost anchored intervals do not replace it')
e=load(P/'g5/src/g5_plot_revision_v2.py','restore_g5',lambda s:'\n'.join(line for line in s.splitlines() if 'pd.DataFrame(tab).to_csv' not in line));e.outer()
# Previously unplotted retained endpoint: with vs without finite XS calibration.
d=pd.read_csv(P/'g4/results/G4_all_method_results.tsv',sep='\t');q=d[d.purpose.eq('perturbation') & ~d.fixed_n & d.method.isin(['M','M_NO_CAL'])].copy();rows=[]
f,ax=plt.subplots(figsize=(150/25.4,108/25.4));f.subplots_adjust(left=.14,right=.97,bottom=.25,top=.77);colors=['#0072B2','#E69F00'];designs=['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE']
for k,des in enumerate(designs):
 for j,cancer in enumerate(['LUAD','COAD']):
  v=q[q.cancer.eq(cancer)&q.design.eq(des)].pivot(index='replicate',columns='method',values='source_log_score');gain=v.M_NO_CAL-v.M;x=k+(j-.5)*.23;mean=gain.mean();err=1.96*gain.std(ddof=1)/np.sqrt(len(gain));ax.bar(x,mean,.19,fc=colors[j],ec='#222222',lw=.7,label=cancer if k==0 else None);ax.errorbar(x,mean,yerr=err,fmt='none',ecolor='#222222',capsize=2.4,lw=.7);batch=gain.groupby(gain.index//10).mean();ax.scatter(x+np.linspace(-.035,.035,len(batch)),batch,s=10,fc='white',ec='#333333',lw=.5,zorder=4)
  rows.extend(dict(cancer=cancer,design=des,replicate=int(i),gain=float(v)) for i,v in gain.items())
ax.spines[['top','right']].set_visible(False);ax.set_xticks(range(3),['Redundant','Heterogeneous','Missing joint path']);ax.axhline(0,c='#555555',lw=.65);ax.set_ylabel('Source log-score gain from calibration (nats)');ax.legend(loc='upper left',bbox_to_anchor=(0,1.24),ncol=2,frameon=True,fancybox=False);f.text(.14,.95,'G4 | Value of finite source calibration',fontsize=9,weight='bold');f.text(.14,.06,'50 paired perturbations per cell; caps: 1.96 seed-MCSE; dots: fixed batches of 10.\nThe same selected records; XS adds 80 source records (320 gene assays).\nAdditional-information ablation, not equal-cost algorithm superiority.',fontsize=6.2)
t=pd.DataFrame(rows);t.to_csv(P/'g4/figure_data/G4_finite_calibration_ablation.tsv',sep='\t',index=False);save(f,4,'G4_finite_calibration_ablation',t,'R21','Retained finite-calibration ablation was missing from selected figures')
(P/'figure_delivery/COVERAGE_REPAIRS.json').write_text(json.dumps(LOG,indent=2));print(json.dumps(LOG,indent=2))
