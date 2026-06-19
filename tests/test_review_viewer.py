from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.review_viewer import (
    Manifest, MethodSpec, MARKER_SHAPES, align_colors, build_events, draw_marker,
    draw_mask_contours, draw_predictions, load_manifest, mask_path_for,
)


def _ref_df():
    # fish 1 and 2 approach to distance 0 at frame 10 (a crossing).
    # rank_crossings casts ids to int, matching real final_fish_id values.
    rows = []
    for f in range(21):
        rows.append({"frame": f, "final_fish_id": 1, "X": float(f), "Y": 0.0, "Z": 0.0})
        rows.append({"frame": f, "final_fish_id": 2, "X": float(20 - f), "Y": 0.0, "Z": 0.0})
    return pd.DataFrame(rows)


def test_load_manifest_and_overrides(tmp_path):
    toml = tmp_path / "m.toml"
    toml.write_text(
        'experiment = "exp1"\n'
        'model_folder = "models"\n'
        '[videos]\n top = "t.mp4"\n side1 = "s1.mp4"\n side2 = "s2.mp4"\n'
        '[[methods]]\n label = "centroid"\n csv_3d = "c.csv"\n id_col = "final_fish_id"\n'
        '[[methods]]\n label = "idtracker"\n csv_3d = "i.csv"\n'
        '[events]\n source = "resolver"\n path = "ev.csv"\n reference_method = "centroid"\n'
        'gt_out = "gt.csv"\n',
        encoding="utf-8",
    )
    cfg = load_manifest(toml, {"max_events": 7, "gt_out": None})
    assert cfg.experiment == "exp1"
    assert [m.label for m in cfg.methods] == ["centroid", "idtracker"]
    assert cfg.methods[1].id_col == "final_fish_id"  # default applied
    assert cfg.event_source == "resolver" and cfg.reference_method == "centroid"
    assert cfg.max_events == 7  # override
    assert str(cfg.gt_out) == "gt.csv"  # None override ignored


def test_build_events_rank_crossings_finds_center():
    cfg = Manifest(
        experiment="e", videos={}, model_folder="m",
        methods=[MethodSpec("centroid", "c.csv", "final_fish_id")],
        gt_out="gt.csv", event_source="rank_crossings", reference_method="centroid",
        n_events=5, close_mm=18.0, window_s=0.05, fps=10.0,
    )
    events = build_events(cfg, {"centroid": _ref_df()})
    assert len(events) >= 1
    assert any(abs(e["center"] - 10) <= 1 for e in events)
    assert events[0]["source"] == "rank_crossings"


def test_build_events_resolver_reads_suspect(tmp_path):
    ev = tmp_path / "resolved.csv"
    pd.DataFrame([
        {"event_id": 0, "track_id_a": "A", "track_id_b": "B", "min_frame": 30,
         "min_distance": 2.0, "suspect_fish": "B", "decision": "suspect:B",
         "review_candidate": True},
        {"event_id": 1, "track_id_a": "A", "track_id_b": "B", "min_frame": 5,
         "min_distance": 1.0, "suspect_fish": "", "decision": "review",
         "review_candidate": False},
    ]).to_csv(ev, index=False)
    cfg = Manifest(
        experiment="e", videos={}, model_folder="m",
        methods=[MethodSpec("centroid", "c.csv")],
        gt_out="gt.csv", event_source="resolver", event_path=ev,
        reference_method="centroid", review_only=True,
    )
    events = build_events(cfg, {})
    # review_only keeps the one review_candidate row
    assert len(events) == 1
    assert events[0]["center"] == 30 and events[0]["suspect"] == "B"


def test_align_colors_matches_by_position():
    ref = pd.DataFrame([
        {"frame": 10, "final_fish_id": "A", "X": 0.0, "Y": 0.0, "Z": 0.0},
        {"frame": 10, "final_fish_id": "B", "X": 10.0, "Y": 0.0, "Z": 0.0},
    ])
    # method2 labels are swapped relative to position: X sits where ref-B is.
    m2 = pd.DataFrame([
        {"frame": 10, "final_fish_id": "X", "X": 10.0, "Y": 0.0, "Z": 0.0},
        {"frame": 10, "final_fish_id": "Y", "X": 0.0, "Y": 0.0, "Z": 0.0},
    ])
    id_cols = {"ref": "final_fish_id", "m2": "final_fish_id"}
    cmap = align_colors({"ref": ref, "m2": m2}, id_cols, "ref", center=10)
    assert cmap["ref"] == {"A": 0, "B": 1}
    # X is at B's position -> B's color (1); Y at A's position -> A's color (0)
    assert cmap["m2"]["X"] == 1
    assert cmap["m2"]["Y"] == 0


def test_align_colors_one_to_one_no_collision():
    ref = pd.DataFrame([{"frame": 10, "final_fish_id": "A", "X": 0.0, "Y": 0.0, "Z": 0.0}])
    # two method fish both nearest to the single reference fish
    m2 = pd.DataFrame([
        {"frame": 10, "final_fish_id": "X", "X": 0.0, "Y": 0.0, "Z": 0.0},
        {"frame": 10, "final_fish_id": "Y", "X": 1.0, "Y": 0.0, "Z": 0.0},
    ])
    cmap = align_colors({"ref": ref, "m2": m2}, {"ref": "final_fish_id", "m2": "final_fish_id"}, "ref", 10)
    assert cmap["m2"]["X"] == 0           # nearest claims the reference color
    assert cmap["m2"]["X"] != cmap["m2"]["Y"]  # one-to-one: no collision


def test_build_events_resolver_blank_suspect_stays_blank(tmp_path):
    ev = tmp_path / "r.csv"
    pd.DataFrame([
        {"event_id": 0, "track_id_a": "A", "track_id_b": "B", "min_frame": 5,
         "min_distance": 1.0, "suspect_fish": "", "decision": "review"},
        {"event_id": 1, "track_id_a": "A", "track_id_b": "B", "min_frame": 30,
         "min_distance": 2.0, "suspect_fish": "B", "decision": "suspect:B"},
    ]).to_csv(ev, index=False)
    cfg = Manifest(
        experiment="e", videos={}, model_folder="m",
        methods=[MethodSpec("centroid", "c.csv")],
        gt_out="gt.csv", event_source="resolver", event_path=ev,
        reference_method="centroid", review_only=False,
    )
    by = {e["center"]: e for e in build_events(cfg, {})}
    assert by[5]["suspect"] == ""     # blank read as NaN must NOT become "nan"
    assert by[30]["suspect"] == "B"


def test_draw_marker_each_shape_renders_distinctly():
    drawn = {}
    for shape in MARKER_SHAPES:
        fr = np.zeros((40, 40, 3), np.uint8)
        draw_marker(fr, 20, 20, (0, 255, 0), shape, r=10, t=2)
        assert fr.any(), f"{shape} drew nothing"
        drawn[shape] = fr
    # shapes are visually distinct (different non-zero pixel footprints)
    assert len({fr.tobytes() for fr in drawn.values()}) == len(MARKER_SHAPES)


def test_draw_predictions_and_mask_contours_render():
    fr = np.zeros((50, 50, 3), np.uint8)
    draw_predictions(fr, [np.array([[10.0, 10.0], [30.0, 30.0]])])
    assert fr.any()
    fr2 = np.zeros((50, 50, 3), np.uint8)
    mask = np.zeros((50, 50), np.uint8); mask[15:35, 15:35] = 255
    draw_mask_contours(fr2, mask)
    assert fr2.any()


def test_mask_path_for_formats_view_and_frame(tmp_path):
    p = mask_path_for(tmp_path, "{view}_cam/{frame:07d}.png", "top", 1234)
    assert p == tmp_path / "top_cam" / "0001234.png"
    p2 = mask_path_for(tmp_path, "{frame:07d}.png", "top", 7)
    assert p2 == tmp_path / "0000007.png"


def test_load_manifest_slp_dir_fallback(tmp_path):
    toml = tmp_path / "m.toml"
    toml.write_text(
        'experiment = "e"\n model_folder = "models"\n slp_dir = "TOP_SLP"\n gt_out = "gt.csv"\n'
        '[videos]\n top = "t.mp4"\n side1 = "s1.mp4"\n side2 = "s2.mp4"\n'
        '[[methods]]\n label = "a"\n csv_3d = "a.csv"\n'                       # inherits top-level slp_dir
        '[[methods]]\n label = "b"\n csv_3d = "b.csv"\n slp_dir = "OWN_SLP"\n'  # overrides
        '[[methods]]\n label = "c"\n csv_3d = "c.csv"\n masks_dir = "M"\n mask_pattern = "{view}/{frame:05d}.png"\n'
        '[events]\n source = "rank_crossings"\n',
        encoding="utf-8",
    )
    cfg = load_manifest(toml)
    by = {m.label: m for m in cfg.methods}
    assert str(by["a"].slp_dir) == "TOP_SLP"      # fallback
    assert str(by["b"].slp_dir) == "OWN_SLP"      # override
    assert str(by["c"].masks_dir) == "M" and by["c"].mask_pattern == "{view}/{frame:05d}.png"
