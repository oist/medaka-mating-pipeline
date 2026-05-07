import numpy as np
import pandas as pd

from scripts.visualization.rerun_viewer import (
    _estimate_json_to_csv_offset,
    _path_candidates,
)


def test_relinked_tag_resolves_relink_artifacts_first(tmp_path):
    best_candidates = _path_candidates(tmp_path, "exp_relinked", kind="best")
    pre_link_candidates = _path_candidates(tmp_path, "exp_relinked", kind="pre_link")
    decisions_candidates = _path_candidates(tmp_path, "exp_relinked", kind="decisions")

    assert best_candidates[0] == tmp_path / "exp_best_relinked.json"
    assert pre_link_candidates[0] == tmp_path / "exp_best_pre_link.json"
    assert decisions_candidates[0] == tmp_path / "exp_relink_decisions.csv"


def test_estimate_json_to_csv_offset_uses_matching_id_groups():
    df = pd.DataFrame(
        [
            {"frame": 10, "fish_id": "ID_1", "X": 11.0, "Y": 22.0, "Z": 33.0},
            {"frame": 10, "fish_id": "ID_2", "X": 14.0, "Y": 25.0, "Z": 36.0},
        ]
    )
    per_frame = {
        10: {
            "groups": {
                "ID_1": [1.0, 2.0, 3.0],
                "ID_2": [4.0, 5.0, 6.0],
            }
        }
    }

    offset = _estimate_json_to_csv_offset(df, per_frame, ["ID_1", "ID_2"])

    assert np.allclose(offset, [10.0, 20.0, 30.0])
