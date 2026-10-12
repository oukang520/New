"""Derive plot endpoints from frozen outputs; no simulations or selection."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];g=ROOT/'paper_experiments_v1/g1'
x=pd.read_csv(g/'results/G1_population_geometry.tsv',sep='\t')
x=x[x.model_id.isin(['D3_SPARSE5','D4_SPARSE10'])]
d=x.groupby(['model_id','operator_knowledge','design_family','design_id'],as_index=False).agg(target_identified_fraction=('primary_target_identified','mean'),target_rescue_fraction=('primary_target_rescued','mean'),n_evaluations=('parameter_draw_id','size'))
assert len(d)==96 and (d.n_evaluations==60).all()
d.to_csv(g/'figure_data/G1D/data_by_design.tsv',sep='\t',index=False)
cfg=json.loads((g/'configs/g1_protocol_v1.json').read_text());rows=[]
for line in (g/'results/G1_counterexamples.jsonl').read_text().splitlines():
 x=json.loads(line);m=cfg['models_core'][x['model_id']];idx=m['d']+m['edges'].index([1,0])
 rows.append({'model_id':x['model_id'],'parameter_draw_id':x['parameter_draw_id'],'target_original':x['theta_original'][idx],'target_alias':x['theta_alias'][idx],'target_change':x['target_change'],'max_source_error':x['max_abs_probability_error'],'max_observed_error':x['max_observed_error_K1'],'diagnostic_match_le_1e8':x['max_observed_error_K1']<=1e-8})
assert len(rows)==20
pd.DataFrame(rows).to_csv(g/'figure_data/G1A/data_target_endpoint.tsv',sep='\t',index=False)
