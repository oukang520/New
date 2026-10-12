"""Reproduce paired G2 correct-call differences; no model fitting or selection."""
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
out = P / 'g2/revision_r2/results'
d = pd.read_csv(out / 'G2_R2_raw.tsv', sep='\t')
rows = []
for (design, effect), q in d.groupby(['design', 'effect']):
    w = q.pivot(index='replicate', columns='method', values='correct').astype(float)
    for reference in ['OLD_BONFERRONI_BANDS', 'EXACT_PER_ENV_BONFERRONI', 'LOG_OR_WALD']:
        v = w['EXACT_STRATIFIED'] - w[reference]
        error = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        rows.append(dict(design=design, effect=effect, reference=reference,
                         n=len(v), correct_call_difference=v.mean(),
                         mc_lo=v.mean()-error, mc_hi=v.mean()+error))
pd.DataFrame(rows).to_csv(out / 'G2_R2_paired_summary.tsv', sep='\t', index=False)
print(f'Wrote {len(rows)} paired summaries')
