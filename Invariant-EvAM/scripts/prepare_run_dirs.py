from pathlib import Path
root=Path(__file__).resolve().parents[1]/"paper_experiments_v1"
for group in ["g1","g2","g5"]:
    for name in ["development","runs","results","profile_refinement","profile_refinement_v2","fresh_confirmation","matched_size"]:
        (root/group/"revision_r8"/name).mkdir(parents=True,exist_ok=True)
print("R8 output directories exist; no files deleted or changed.")
