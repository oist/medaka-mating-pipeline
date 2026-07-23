"""Build and execute the inspectable review-quality/motion audit notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT = HERE / "observation_quality_and_motion_audit.ipynb"


def build() -> Path:
    cells = [
        nbformat.v4.new_markdown_cell(
            """# TL;DR

The completed gate-branch review does **not** establish that 6 mm is worse
than 8 mm. Human female identity and automatic SLEAP/geometry mapping quality
must be scored separately. Until the new structured evidence-quality field is
reviewed, the automatic threshold comparison remains inconclusive.
"""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import importlib.util
import json
import pandas as pd

audit_path = Path('report/20260718/k12_jump6_strict_mouth/gate_branch_diagnostic/audit_review_and_motion.py').resolve()
spec = importlib.util.spec_from_file_location('gate_motion_audit', audit_path)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)
result = audit.run()
result['summary']"""
        ),
        nbformat.v4.new_markdown_cell(
            """# Context & Methods

The 6 mm and controlled 8 mm reconstructions use the same per-frame 3D point
sets; their difference is temporal linking. The original linker uses only the
last accepted position because the optional velocity/tangent/acceleration
weights were zero.

This audit uses two independent checks:

1. Human evidence is eligible for tracker scoring only when the female is
   confirmed, both side views link to SLEAP, both views agree on geometry, and
   `evidence_quality=usable`. A `human_only` scene remains valuable human
   identity evidence but its SLEAP/geometry mapping is excluded.
2. Motion continuity is measured bidirectionally using up to 15 direct
   observations in the 30 frames before and after each branch. Velocity is the
   median of all pairwise 3D slopes, so the result uses more than two frames
   and is resistant to one tangent spike.

Motion is diagnostic evidence only. It cannot turn a false negative, shared
tail/node, or merged SLEAP instance into a valid observation.
"""
        ),
        nbformat.v4.new_code_cell(
            """pd.DataFrame({
    'input': ['human ledger', '6 mm track', '8 mm controlled track', 'branch transitions'],
    'path': [str(audit.LEDGER), str(audit.JUMP6), str(audit.JUMP8), str(audit.TRANSITIONS)],
})"""
        ),
        nbformat.v4.new_markdown_cell("# Data"),
        nbformat.v4.new_code_cell(
            """quality = result['quality']
quality_counts = pd.DataFrame({
    'measure': [
        'branches', 'annotated branches', 'two-view-consistent candidates',
        'explicit quality reviews', 'tracker-scoring eligible'
    ],
    'count': [
        len(quality), quality.annotation_frame.notna().sum(),
        quality.automatic_mapping_candidate.sum(),
        quality.evidence_quality.ne('unreviewed').sum(),
        quality.tracker_scoring_eligible.sum(),
    ],
})
quality_counts"""
        ),
        nbformat.v4.new_code_cell(
            """quality[[
    'branch_id', 'trigger_frame', 'annotation_frame', 'gt_status',
    'side1_selection_status', 'side2_selection_status',
    'geometry_id_consistency', 'evidence_quality',
    'observation_issue_flags', 'automatic_mapping_candidate',
    'tracker_scoring_eligible', 'notes'
]]"""
        ),
        nbformat.v4.new_markdown_cell("# Results"),
        nbformat.v4.new_code_cell(
            """motion = result['motion']
paired = motion.dropna(subset=[
    'jump6_robust_tangent_angle_deg', 'jump8_robust_tangent_angle_deg'
]).copy()
pd.DataFrame({
    'measure': ['paired changed-track boundaries', '6 mm lower tangent', '8 mm lower tangent',
                '6 mm tangent >90°', '8 mm tangent >90°'],
    'count': [len(paired),
              (paired.jump6_robust_tangent_angle_deg < paired.jump8_robust_tangent_angle_deg).sum(),
              (paired.jump8_robust_tangent_angle_deg < paired.jump6_robust_tangent_angle_deg).sum(),
              (paired.jump6_robust_tangent_angle_deg > 90).sum(),
              (paired.jump8_robust_tangent_angle_deg > 90).sum()],
})"""
        ),
        nbformat.v4.new_code_cell(
            """paired.assign(
    max_tangent_deg=paired[[
        'jump6_robust_tangent_angle_deg', 'jump8_robust_tangent_angle_deg'
    ]].max(axis=1)
).sort_values('max_tangent_deg', ascending=False)[[
    'branch_id', 'trigger_frame', 'six_id', 'old_eight_id', 'new_eight_id',
    'jump6_robust_tangent_angle_deg', 'jump8_robust_tangent_angle_deg',
    'jump6_forward_bridge_prediction_error_mm',
    'jump8_forward_bridge_prediction_error_mm'
]].head(20)"""
        ),
        nbformat.v4.new_markdown_cell(
            """# Takeaways

- The previous recommendation to prefer 8 mm is withdrawn: the new review
  demonstrates an observation-quality confound, and none of the legacy rows
  has yet been explicitly marked usable for tracker scoring.
- Robust multi-frame tangent evidence is mixed rather than uniformly favoring
  one gate. Large tangent discontinuities identify useful human-review scenes,
  but they are not ground truth when SLEAP itself is wrong.
- Use 8 mm (with the existing 80 mm hard cap) only as an outer candidate gate
  while testing motion-aware association. The identity choice should use
  robust history and eventually short lookahead; 6-versus-8 should be retested
  only on explicitly `usable` human evidence.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbformat.write(notebook, OUTPUT)
    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
    ).execute(cwd=str(REPO_ROOT))
    nbformat.write(executed, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
