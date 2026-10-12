"""Render only the active G2_02 block; never rerun historical publishers."""
from pathlib import Path
import json
O = Path(__file__).resolve().parent
P = O.parents[1]
source = P / 'closure_r8/publication/plot.py'
text = source.read_text()
ns = {'__file__': str(source)}
exec(compile(text.split('# G1:')[0], str(source), 'exec'), ns)
ns['OUT'] = O / 'staging'
ns['D'] = O / 'plot_data'
for key in ['OUT', 'D']:
    ns[key].mkdir(exist_ok=True)
block = '# G2:' + text.split('# G2:', 1)[1].split('# G5 action', 1)[0]
exec(compile(block, str(source), 'exec'), ns)
(O / 'FIGURE_MANIFEST.json').write_text(json.dumps(ns['manifest'], ensure_ascii=False, indent=2))
print('G2_02 generated; awaiting visual review')
