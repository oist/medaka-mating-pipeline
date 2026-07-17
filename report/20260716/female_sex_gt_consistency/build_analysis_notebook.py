"""Build the reader-facing notebook for the female-track consistency audit."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "analysis_summary.json").read_text(encoding="utf-8"))
identity = SUMMARY["identity_consistency"]
mapping = SUMMARY["mapping"]
sensitivity = SUMMARY["sensitivity"]

notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "medaka-mating-pipeline (.venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Female annotation vs. round7 refractive track consistency"
    ),
    nbf.v4.new_markdown_cell(
        "## tl;dr\n\n"
        f"- The female annotation is **not consistent with one fixed geometry ID**: "
        f"the best fixed label is `{identity['best_fixed_geometry_id']}` at only "
        f"{identity['best_fixed_geometry_id_accuracy']:.1%} of the "
        f"{mapping['mapped_consensus_count']} mapped confirmed annotations.\n"
        f"- The sparse GT sequence contains **at least "
        f"{identity['observed_id_switch_count']} observed female-ID changes** across "
        f"{identity['observed_run_count']} runs.\n"
        f"- Annotation-to-geometry evidence is strong overall: "
        f"{mapping['two_view_agree_share']:.1%} two-view agreement and "
        f"{mapping['mapped_consensus_share_of_confirmed']:.1%} consensus mapping.\n"
        f"- The conclusion survives the strict 8 px sensitivity filter: the best "
        f"fixed ID reaches only {sensitivity['strict_two_view_best_fixed_accuracy']:.1%} "
        f"across {sensitivity['strict_two_view_count']} high-confidence rows."
    ),
    nbf.v4.new_markdown_cell(
        "## Context & Methods\n\n"
        "This diagnostic notebook compares the authoritative human female ledger "
        "with the exact round7 refractive `_3d.csv` named by the annotator. It "
        "validates keys and audit history, joins the selected geometry ID at each "
        "annotated frame, reprojects the selected 3D coordinate into each side view, "
        "reproduces the clean 8 mm provisional tracklet segmentation, and counts "
        "female geometry-ID runs and transitions.\n\n"
        "### Key Assumptions\n\n"
        "- Geometry IDs are provisional tracker labels, not biological identities.\n"
        "- A change between adjacent sparse GT samples is an observed lower bound; "
        "additional swaps can occur and reverse between samples.\n"
        "- The user-supplied GUI command states an 8 px gate, while the saved candidate "
        "summary records 80 px. The strict sensitivity subset therefore independently "
        "requires two-view agreement plus <=8 px click and reprojection distances."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "import pandas as pd\n\n"
        "cwd = Path.cwd().resolve()\n"
        "REPORT_DIR = (\n"
        "    cwd if cwd.name == 'female_sex_gt_consistency'\n"
        "    else (cwd / 'report/20260716/female_sex_gt_consistency').resolve()\n"
        ")\n"
        "sys.path.insert(0, str(REPORT_DIR))\n"
        "from female_track_consistency_analysis import run_analysis\n\n"
        "summary = run_analysis()\n"
        "summary['verdict']"
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        "pd.DataFrame({\n"
        "    'check': [\n"
        "        'Annotation rows', 'Confirmed female rows', 'Mapped consensus rows',\n"
        "        'Current track rows', 'Track fish IDs', 'Track backend',\n"
        "        'Provenance hash matches candidate build', 'Audit rows match canonical ledger'\n"
        "    ],\n"
        "    'value': [\n"
        "        summary['annotation']['row_count'],\n"
        "        summary['annotation']['confirmed_female_count'],\n"
        "        summary['mapping']['mapped_consensus_count'],\n"
        "        summary['track']['row_count'],\n"
        "        ', '.join(summary['track']['fish_ids']),\n"
        "        summary['track']['backend'],\n"
        "        summary['source_consistency']['provenance_sha256_matches'],\n"
        "        f\"{summary['audit']['canonical_rows_with_latest_audit_match']}/\"\n"
        "        f\"{summary['audit']['canonical_row_count']}\",\n"
        "    ]\n"
        "})"
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_markdown_cell(
        "### No fixed geometry ID represents the female\n\n"
        "Counts below use confirmed rows with a non-empty cross-view consensus ID."
    ),
    nbf.v4.new_code_cell(
        "pd.read_csv(REPORT_DIR / 'identity_counts.csv')"
    ),
    nbf.v4.new_markdown_cell(
        "### The dominant female label changes across the recording\n\n"
        "Temporal bins come from the saved candidate metadata; the two manual-jump "
        "rows have no bin and are retained separately."
    ),
    nbf.v4.new_code_cell(
        "pd.read_csv(REPORT_DIR / 'temporal_identity_counts.csv')"
    ),
    nbf.v4.new_markdown_cell(
        "### Strict sensitivity filtering does not rescue a fixed ID\n\n"
        "This filter removes one-view rows, disagreements, and every selected pose "
        "whose click or selected-ID reprojection exceeds 8 px in either side camera."
    ),
    nbf.v4.new_code_cell(
        "pd.DataFrame({\n"
        "    'subset': ['all mapped', 'strict two-view', 'very strict candidate'],\n"
        "    'n': [\n"
        "        summary['mapping']['mapped_consensus_count'],\n"
        "        summary['sensitivity']['strict_two_view_count'],\n"
        "        summary['sensitivity']['very_strict_count'],\n"
        "    ],\n"
        "    'best_fixed_id_accuracy': [\n"
        "        summary['identity_consistency']['best_fixed_geometry_id_accuracy'],\n"
        "        summary['sensitivity']['strict_two_view_best_fixed_accuracy'],\n"
        "        summary['sensitivity']['very_strict_best_fixed_accuracy'],\n"
        "    ],\n"
        "    'observed_switches': [\n"
        "        summary['identity_consistency']['observed_id_switch_count'],\n"
        "        summary['sensitivity']['strict_two_view_observed_switch_count'],\n"
        "        summary['sensitivity']['very_strict_observed_switch_count'],\n"
        "    ],\n"
        "})"
    ),
    nbf.v4.new_markdown_cell(
        "### Camera-link evidence is mostly strong, with a small wide-gate tail"
    ),
    nbf.v4.new_code_cell(
        "pd.read_csv(REPORT_DIR / 'reprojection_summary.csv')"
    ),
    nbf.v4.new_markdown_cell(
        "### Observed female-ID transition brackets\n\n"
        "These brackets localize changes only between consecutive annotated frames; "
        "they do not claim the exact swap frame."
    ),
    nbf.v4.new_code_cell(
        "pd.read_csv(REPORT_DIR / 'observed_female_id_transitions.csv')"
    ),
    nbf.v4.new_markdown_cell("## Takeaways"),
    nbf.v4.new_code_cell(
        "json.dumps({\n"
        "    'verdict': summary['verdict'],\n"
        "    'identity_consistency': summary['identity_consistency'],\n"
        "    'sensitivity': summary['sensitivity'],\n"
        "    'warnings': summary['warnings'],\n"
        "}, indent=2)"
    ),
    nbf.v4.new_markdown_cell(
        "The human annotations are suitable as sparse identity anchors after the "
        "annotator provenance is corrected. They should not be converted into a "
        "single `fish_id -> female` lookup. Use the transition brackets as supervision "
        "for identity stitching, and manually review the two cross-view disagreements "
        "plus the large side2 reprojection outlier at frame 246665 before freezing a "
        "gold-standard release."
    ),
]

nbf.write(notebook, HERE / "female_track_consistency_audit.ipynb")
print(HERE / "female_track_consistency_audit.ipynb")
