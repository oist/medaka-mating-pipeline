import numpy as np
import pandas as pd

from scripts.visualization.render_4panel_video import (
    draw_tracklet_boundary_indicator,
    load_tracklet_boundary_events,
)


def test_boundary_loader_expands_start_and_end_display_windows(tmp_path):
    path = tmp_path / "top.tracklets.csv"
    pd.DataFrame(
        {
            "frame": [100, 101, 102, 110, 111],
            "tracklet_id": [3, 3, 3, 4, 4],
        }
    ).to_csv(path, index=False)

    by_view, exact = load_tracklet_boundary_events(
        [f"top_cam={path}"], hold_frames=2
    )
    assert len(exact) == 4
    assert any(e["kind"] == "START" and e["boundary_frame"] == 100 for e in by_view["top_cam"][98])
    assert any(e["kind"] == "END" and e["boundary_frame"] == 102 for e in by_view["top_cam"][104])
    assert 97 not in by_view["top_cam"]


def test_boundary_indicator_draws_visible_red_panel_border():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    draw_tracklet_boundary_indicator(
        image,
        [{"kind": "END", "tracklet_id": "7", "boundary_frame": 100}],
        source_frame=100,
    )
    assert image[:, :, 2].max() == 255
    assert int(image.sum()) > 0
