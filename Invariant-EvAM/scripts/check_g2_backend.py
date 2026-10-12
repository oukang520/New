"""Numerical smoke check of the frozen G2 backend; writes no study outputs."""
from pathlib import Path
import json, sys
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'paper_experiments_v1/g2/src'))
from g2_engine import snapshot

theta = np.array([-1.0, -1.0, -1.0, 0.3, 0.1])
p, jacobian = snapshot(theta)
assert np.isfinite(p).all() and np.isfinite(jacobian).all()
assert (p >= 0).all() and abs(p.sum() - 1) < 1e-10
assert np.max(np.abs(jacobian.sum(axis=0))) < 1e-10
step = 1e-5
numeric = np.column_stack([
    (snapshot(theta + np.eye(5)[i] * step)[0]
     - snapshot(theta - np.eye(5)[i] * step)[0]) / (2 * step)
    for i in range(5)
])
error = float(np.max(np.abs(numeric - jacobian)))
assert error < 1e-7, error
print(json.dumps({'status': 'PASS', 'probability_sum': float(p.sum()),
                  'maximum_jacobian_error': error,
                  'scientific_experiments_rerun': False}, indent=2))
