from pathlib import Path

import pandas as pd
import pytest

from scripts.visualization.plot_xyz_speed_qc import (
    load_tracks,
    parse_identity_specs,
)


def _write_geometry_csv(path: Path) -> None:
    rows = []
    for frame in range(3):
        for index, fish_id in enumerate(("ID_1", "ID_2", "ID_3")):
            rows.append(
                {
                    "frame": frame,
                    "fish_id": fish_id,
                    "X": frame + index,
                    "Y": index,
                    "Z": 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_tracks_accepts_geometry_id_column_and_display_order(tmp_path: Path):
    path = tmp_path / "geometry.csv"
    _write_geometry_csv(path)
    mapping = parse_identity_specs(["ID_3=H1", "ID_1=H2", "ID_2=H3"])

    tracks, fish_order = load_tracks(
        path,
        id_column="fish_id",
        identity_map=mapping,
    )

    assert fish_order == ("H1", "H2", "H3")
    assert set(tracks["id_final"]) == {"H1", "H2", "H3"}
    assert len(tracks) == 9


def test_identity_specs_reject_duplicate_display_labels():
    with pytest.raises(ValueError, match="Duplicate --identity display label"):
        parse_identity_specs(["ID_1=H1", "ID_2=H1"])
