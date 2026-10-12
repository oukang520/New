from __future__ import annotations

import itertools
import numpy as np

from phase2.jacobian_identifiability import functional_status


def functional_identifiability_map(model, result) -> list[dict[str, object]]:
    """Enumerate coordinate, interaction-ordering and pathway-sum functionals."""
    p = model.parameter_count
    eye = np.eye(p)
    rows: list[dict[str, object]] = []
    for j, name in enumerate(model.parameter_names):
        kind = "baseline" if j < model.d else "interaction"
        rows.append({"functional": name, "type": kind, "vector": eye[j],
                     **functional_status(eye[j], result)})
    interaction_indices = list(range(model.d, p))
    for j, k in itertools.combinations(interaction_indices, 2):
        vector = eye[j] - eye[k]
        rows.append({
            "functional": f"{model.parameter_names[j]}>{model.parameter_names[k]}",
            "type": "interaction_ordering", "vector": vector,
            **functional_status(vector, result),
        })
    # A pathway score is the mean of all incoming interactions for each target.
    for target in range(model.d):
        indices = [model.d + k for k, edge in enumerate(model.edges) if edge[0] == target]
        if not indices:
            continue
        vector = np.zeros(p); vector[indices] = 1.0 / len(indices)
        rows.append({
            "functional": f"mean_incoming_interaction_target_{target}",
            "type": "pathway_mean", "vector": vector,
            **functional_status(vector, result),
        })
    return rows


def recommend_environment(current_outputs, candidate_outputs: dict[str, tuple], tol: float = 1e-8):
    """Rank candidate environments lexicographically by rank then sigma_min."""
    from phase2.jacobian_identifiability import analyze_environments

    rows = []
    for name, output in candidate_outputs.items():
        result = analyze_environments([*current_outputs, output], tol=tol)
        rows.append({"candidate": name, "rank": result.rank,
                     "sigma_min": result.min_singular_value,
                     "condition_number": result.condition_number})
    return sorted(rows, key=lambda r: (r["rank"], r["sigma_min"]), reverse=True)
