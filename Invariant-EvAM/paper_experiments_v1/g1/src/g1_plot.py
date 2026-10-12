"""G1 reference-led typesetting v3; frozen scientific outputs are read-only."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[3];G=ROOT/'paper_experiments_v1/g1'
sys.path.insert(0,str(ROOT/'src'))
from invariant_evam.figure_style import apply_publication_style
P=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7']
BLUE=mpl.colors.LinearSegmentedColormap.from_list('blue',['#ffffff',P[1],P[0]])
GREEN=mpl.colors.LinearSegmentedColormap.from_list('green',['#ffffff',P[2]])
SPECS={'G1A':('R25',89,104),'G1B':('R25',89,67),'G1C':('R20',89,113),'G1D':('R24',89,119),'G1E':('R20',89,58)}
CAP={
'G1A':'主终点改为原始目标interaction与兼容参数目标值配对，所有目标变化均为+0.4；右侧为K1最大观测概率残差，橙色表示残差>1e-8，仅为数值诊断，不是全局证明。源/观测误差配对旧图保留归档，不再作为主终点。20行分别为D3/D4完整模型各10个冻结draw，按模型、draw ID排序。空心圆为原始目标值，实心圆为兼容参数目标值；每对端点由线连接，线不是CI。目标坐标改变0.4。横轴线性尺度。仅显示优化端点，不称连续兼容路径。',
'G1B':'每个模型每场景60个v1.1 draw，固定1000×d事件测量proxy，K1。空心圆为single基准，实心圆为相应场景中位数；浅灰宽线为该场景Q1–Q3，不是CI。左侧细条长度表示identified比例，范围0–1。四场景全部保留，按冻结顺序排列。',
'G1C':'24个预冻结设计、两个稀疏模型，K1。每格60个draw。列按D3/D4分组，每组依次为新增rank比例、target rescue比例、target identified比例，均取0–1。四机制分块，设计按中性ID排序，全部144格保留。无显著性标记；数值见data.tsv。',
'G1D':'将既有原始结果展开到逐设计粒度：24设计×2稀疏模型×K0/K1=96格，每格60draw。面积正比identified比例，颜色表示target rescue比例；两者独立编码。零面积用小十字，缺失若出现用斜线方框。分块沿用冻结机制顺序，无按效果筛选。原360评价/格汇总表仍保留为data.tsv，本图使用data_by_design.tsv。',
'G1E':'每模型60draw；固定参数数、源Jacobian秩中位数、零空间维数中位数。D3/D4分块；计数顺序色，全部12格标注原始值，不对不同模型作显著性比较。'}
def axes(fig,rect):
 a=fig.add_axes(rect);a.tick_params(length=2.5,width=.65,pad=3);return a
def clean(a):
 a.spines[['top','right','left']].set_visible(False);a.spines['bottom'].set_linewidth(.65)
def topbar(fig,rect,cmap,label,maxval=1):
 ca=fig.add_axes(rect); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,maxval),cmap=cmap),cax=ca,orientation='horizontal')
 cb.set_ticks([0,.5,1] if maxval==1 else [0,4,8,12,16]);cb.ax.tick_params(length=2,width=.5,pad=2,labelsize=7);cb.outline.set_linewidth(.5)
 ca.set_title(label,fontsize=8,loc='left',pad=4)
def blocks(a,ncol):
 for k,label in enumerate(['Neither','Assay','Selection','Both']):
  y=k*6.65-.5;a.add_patch(Rectangle((-.5,y),ncol,6,fill=False,ec='#555555',lw=.65,clip_on=False))
  a.add_patch(Rectangle((-1.65,y),.36,6,fc='#eeeeee',ec='none',clip_on=False))
  a.text(-1.47,y+3,label,rotation=90,ha='center',va='center',fontsize=7,clip_on=False)
def plot_panel(fid,folder):
 apply_publication_style({'savefig.dpi':600,'font.size':8,'axes.labelsize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.65})
 ref,w,h=SPECS[fid];fig=plt.figure(figsize=(w/25.4,h/25.4));fig.text(.035,.977,fid[-1].lower(),va='top',fontsize=10.5,fontweight='bold')
 d=pd.read_csv(folder/'data.tsv',sep='\t')
 if fid=='G1A':
  d=pd.read_csv(folder/'data_target_endpoint.tsv',sep='\t');a=axes(fig,[.23,.16,.48,.73]);res=axes(fig,[.775,.16,.20,.73]);ys=[]
  for i,row in enumerate(d.sort_values(['model_id','parameter_draw_id']).itertuples()):
   y=i+(i//10)*.8;ys.append(y);v=row.target_original;u=row.target_alias
   a.axhline(y,color='#f0f0f0',lw=.45,zorder=0);a.plot([v,u],[y,y],color='#777777',lw=.8)
   a.plot(v,y,'o',ms=3.1,mfc='white',mec='#555555',mew=.65);a.plot(u,y,'o',ms=3.4,color=P[2])
   color=P[0] if row.diagnostic_match_le_1e8 else P[4]
   res.text(.5,y,f'{row.max_observed_error:.1e}',ha='center',va='center',fontsize=7,color=color)
  a.set_yticks(ys,[f'{i%10:02d}' for i in range(20)]);a.tick_params(axis='y',length=0)
  a.set_ylim(21,-.8);a.set_xlim(-.3,1.55);a.set_xticks([0,.5,1,1.5]);a.set_xlabel('Target interaction',labelpad=4);clean(a)
  res.set_ylim(21,-.8);res.set_xlim(0,1);res.axis('off');res.set_title('Max. |Δp obs.|',fontsize=7.5,pad=4)
  for start,label in [(0,'D3 full'),(10.8,'D4 full')]:
   a.text(-.24,1-(start+4.5+.8)/21.8,label,rotation=90,transform=a.transAxes,ha='center',va='center',fontsize=7.5)
  fig.legend(handles=[Line2D([],[],ls='',marker='o',ms=3.4,mfc='white',mec='#555555',label='Original'),Line2D([],[],ls='',marker='o',ms=3.4,color=P[2],label='Target +0.4')],loc='upper right',bbox_to_anchor=(.96,.984),ncol=2,columnspacing=.9,handletextpad=.35)
 elif fid=='G1B':
  a=axes(fig,[.43,.21,.52,.62]);side=axes(fig,[.349,.21,.042,.62]);ys=[];names=[]
  order=['single','duplicated','candidate_heterogeneous','pure_cross_view_failure']
  for mi,m in enumerate(['D3_SPARSE5','D4_SPARSE10']):
   q=d[d.model_id==m].set_index('scenario');base=q.loc['single','median_information']
   for j,s in enumerate(order):
    y=mi*4.8+j;ys.append(y);names.append(['Single','Duplicated','Heterogeneous','Cross-view loss'][j]);r=q.loc[s];color=[P[0],P[1],P[2],P[4]][j]
    a.axhline(y,color='#eeeeee',lw=.5);a.plot([r.q1_information,r.q3_information],[y,y],color='#dddddd',lw=3.5,solid_capstyle='round',zorder=1)
    a.plot([base,r.median_information],[y,y],color='#777777',lw=.8);a.plot(r.median_information,y,'o',ms=3.6,color=color);a.plot(base,y,'o',ms=2.8,mfc='white',mec='#555555',mew=.65)
    side.barh(y,1,color='#eeeeee',height=.42);side.barh(y,r.identified_fraction,color=color,height=.42)
  for x in [a,side]:x.set_ylim(8.4,-.65)
  side.set_yticks(ys,names);side.tick_params(axis='y',length=0,pad=4);side.set_xticks([]);side.spines[:].set_visible(False);side.set_title('ID',fontsize=7,pad=3)
  a.set_yticks([]);a.set_xlim(-1,31);a.set_xticks([0,10,20,30]);a.set_xlabel('Total target information',labelpad=4);clean(a)
  fig.text(.075,.70,'D3',rotation=90,fontsize=7.5,va='center');fig.text(.075,.37,'D4',rotation=90,fontsize=7.5,va='center')
  fig.legend(handles=[Line2D([],[],ls='',marker='o',ms=3.2,mfc='white',mec='#555555',label='Single'),Line2D([],[],ls='',marker='o',ms=3.2,color=P[2],label='Scenario'),Line2D([],[],color='#dddddd',lw=3.5,label='Q1–Q3')],loc='upper right',bbox_to_anchor=(.975,.99),ncol=3,columnspacing=.8,handletextpad=.35,handlelength=1)
 elif fid=='G1C':
  a=axes(fig,[.34,.19,.60,.65]);topbar(fig,[.34,.925,.60,.022],BLUE,'Fraction')
  for j,m in enumerate(['D3_SPARSE5','D4_SPARSE10']):
   q=d[d.model_id==m].sort_values('design_id')
   for i,row in enumerate(q.itertuples()):
    y=i+(i//6)*.65
    for k,key in enumerate(['rank_gain_fraction','target_rescue_fraction','target_identified_fraction']):
     x=j*3+k;v=getattr(row,key);a.add_patch(Rectangle((x-.5,y-.5),1,1,fc=BLUE(v),ec='#999999',lw=.3))
  blocks(a,6);a.axvline(2.5,color='#555555',lw=.8)
  a.set_xlim(-.5,5.5);a.set_ylim(25.65,-.55);a.set_yticks([i+(i//6)*.65 for i in range(24)],[f'D{i:02d}' for i in range(24)])
  a.set_xticks(range(6),['Rank gain','Rescue','Identified']*2,rotation=55,ha='right');a.tick_params(length=0,pad=3);a.spines[:].set_visible(False)
  a.text(1,-1.15,'D3 sparse',ha='center',fontsize=7.5);a.text(4,-1.15,'D4 sparse',ha='center',fontsize=7.5)
 elif fid=='G1D':
  d=pd.read_csv(folder/'data_by_design.tsv',sep='\t');a=axes(fig,[.31,.14,.43,.77])
  for j,(m,k) in enumerate([(m,k) for m in ['D3_SPARSE5','D4_SPARSE10'] for k in ['K0_KNOWN','K1_STRUCTURED']]):
   for i,row in enumerate(d[(d.model_id==m)&(d.operator_knowledge==k)].sort_values('design_id').itertuples()):
    y=i+(i//6)*.65;size=17*row.target_identified_fraction
    if size:a.scatter(j,y,s=size,c=[GREEN(row.target_rescue_fraction)],ec='#666666',lw=.4)
    else:a.plot(j,y,'+',color='#888888',ms=2,mew=.45)
  blocks(a,4);a.set_xlim(-.5,3.5);a.set_ylim(25.65,-.55);a.set_yticks([i+(i//6)*.65 for i in range(24)],[f'D{i:02d}' for i in range(24)]);a.set_xticks(range(4),['K0','K1','K0','K1'],rotation=90);a.tick_params(length=0,pad=3);a.spines[:].set_visible(False)
  a.text(.5,-1.2,'D3',ha='center',fontsize=7.5);a.text(2.5,-1.2,'D4',ha='center',fontsize=7.5)
  ca=fig.add_axes([.805,.64,.026,.17])
  cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,1),cmap=GREEN),cax=ca);cb.set_ticks([0,.5,1]);cb.ax.tick_params(labelsize=7,length=2,pad=2);cb.outline.set_linewidth(.5);ca.set_title('Rescue',fontsize=7,pad=5,loc='left')
  handles=[Line2D([],[],ls='',marker='o',markersize=np.sqrt(17*v),mfc='white',mec='#666666',mew=.4,label=f'{v:.0%}') for v in [.25,.5,.75,1]]
  handles.insert(0,Line2D([],[],ls='',marker='+',ms=2,color='#888888',label='0%'))
  fig.legend(handles=handles,title='Identified',title_fontsize=7,fontsize=7,loc='upper left',bbox_to_anchor=(.757,.58),borderaxespad=0,labelspacing=.55,handlelength=1,handletextpad=.45)
 elif fid=='G1E':
  a=axes(fig,[.37,.23,.56,.47]);topbar(fig,[.37,.85,.56,.035],BLUE,'Dimension',16)
  order=['D3_SPARSE5','D3_FULL9','D4_SPARSE10','D4_FULL16'];mat=d.set_index('model_id').loc[order,['parameter_count','source_rank','source_nullity']].to_numpy();ys=[0,1,2.45,3.45]
  for i,y in enumerate(ys):
   for j,v in enumerate(mat[i]):
    a.add_patch(Rectangle((j-.5,y-.5),1,1,fc=BLUE(v/16),ec='#777777',lw=.35));a.text(j,y,f'{v:.0f}',ha='center',va='center',fontsize=7.5,color='white' if v>=12 else '#222222')
  for y in [-.5,1.95]:a.add_patch(Rectangle((-.5,y),3,2,fill=False,ec='#555555',lw=.65,clip_on=False))
  a.set_xlim(-.5,2.5);a.set_ylim(3.96,-.51);a.set_yticks(ys,['D3 sparse','D3 full','D4 sparse','D4 full']);a.set_xticks(range(3),['Parameters','Rank','Nullity']);a.tick_params(length=0,pad=4);a.spines[:].set_visible(False)
 for ext in ['pdf','svg']:fig.savefig(folder/f'{fid}.{ext}',dpi=600,facecolor='white')
 preview=G/'reviewer_audit/style_v3';preview.mkdir(parents=True,exist_ok=True);fig.savefig(preview/f'{fid}.png',dpi=300);plt.close(fig)
 m=json.loads((folder/'manifest.json').read_text());datafile=folder/('data_target_endpoint.tsv' if fid=='G1A' else ('data_by_design.tsv' if fid=='G1D' else 'data.tsv'));m.update(primary_reference_id=ref,final_width_mm=w,final_height_mm=h,review_status='awaiting_visual_review',review_checks={k:False for k in m['review_checks']},plot_data_path=str(datafile.relative_to(ROOT)),data_source=str(datafile.relative_to(ROOT)),data_checksum=hashlib.sha256(datafile.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),review_record='paper_experiments_v1/g1/reports/FIGURE_STYLE_REVIEW_V3_CN.md',adaptations_from_reference=[CAP[fid]],reference_selection_reason=CAP[fid],interval_definition='G1B: Q1–Q3 across parameter draws, not CI; other panels: none',palette_mapping={'blue_scale':['white',P[1],P[0]],'green_scale':['white',P[2]],'single':P[0],'duplicate':P[1],'heterogeneous':P[2],'failure':P[4]})
 m['scale_and_transform']='Linear target coordinate in G1A; raw residuals in scientific notation; fractions/other axes linear'
 if fid=='G1D':m['n_definition']='60 draws per design-model-knowledge cell; 96 cells';m['unit_of_analysis']='design-model-knowledge cell'
 (folder/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 (folder/'caption_notes.md').write_text(f'# {fid} · style v3\n\n{CAP[fid]}\n\n参考{ref}；尺寸{w}×{h}mm。600dpi，独立PDF+SVG，矢量文字；刻度7pt，轴标题8.5pt，字母10.5pt。说明性段落移到本文件。仅既有数据重排，不新增实验或筛选成功结果。\n')
if __name__=='__main__':
 for fid in SPECS:plot_panel(fid,G/f'figure_data/{fid}')
