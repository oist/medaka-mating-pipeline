"""Smoke tests for scripts.sleap.mine_training_frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.sleap.mine_training_frames import (
    SIGNAL_WEIGHTS,
    _simple_nn_track,
    aggregate_experiment,
    select_candidates,
)


def test_simple_nn_track_links_consecutive_frames():
    rows = []
    for f in range(10):
        rows.append({"frame": f, "x": float(f), "y": 0.0})
        rows.append({"frame": f, "x": 100.0 - float(f), "y": 50.0})
    track_ids = _simple_nn_track(rows, gate_px=5.0)
    assert all(t >= 0 for t in track_ids)
    # Two stable tracks across the 10 frames.
    assert len(set(track_ids)) == 2
    # Even-indexed entries (the increasing-x track) all share an id.
    even_track = {track_ids[i] for i in range(0, len(rows), 2)}
    assert len(even_track) == 1


def test_simple_nn_track_breaks_track_when_gap_exceeds_max():
    rows = [
        {"frame": 0, "x": 0.0, "y": 0.0},
        {"frame": 100, "x": 0.0, "y": 0.0},  # huge temporal gap.
    ]
    track_ids = _simple_nn_track(rows, gate_px=5.0, max_gap_frames=30)
    assert track_ids[0] != track_ids[1]


def _synthetic_per_view(n_frames: int = 200, with_signals: bool = True) -> dict:
    """Build a tiny per-view payload mimicking the parquet feature schema."""
    df = pd.DataFrame({
        "frame": np.arange(n_frames),
        "n_inst": np.full(n_frames, 2, dtype=int),
        "max_score": np.full(n_frames, 0.95),
        "min_score": np.full(n_frames, 0.90),
        "x_first": np.linspace(100, 200, n_frames),
        "y_first": np.full(n_frames, 100.0),
        "score_first": np.full(n_frames, 0.95),
        "s1": np.zeros(n_frames, dtype=bool),
        "s2": np.zeros(n_frames, dtype=bool),
        "s5": np.zeros(n_frames, dtype=bool),
    })
    if with_signals:
        # Sprinkle a few signal-firing frames evenly across time.
        for k, idx in enumerate(np.linspace(0, n_frames - 1, 20).astype(int)):
            df.loc[idx, "s1" if k % 3 == 0 else "s2" if k % 3 == 1 else "s5"] = True
            if k % 3 == 2:
                df.loc[idx, "min_score"] = 0.4  # also S5-consistent
                df.loc[idx, "max_score"] = 0.95
    meta = {"video_filename": None, "bl_px_est": 30.0, "n_frames": n_frames}
    return {"df": df, "meta": meta}


def test_aggregate_and_select_returns_expected_schema():
    per_view = {v: _synthetic_per_view() for v in ("top_cam", "side1_cam", "side2_cam")}
    df = aggregate_experiment(
        "synth",
        per_view,
        pre_link_json_path=None,
        fps=166.0,
        skip_frame=0,
        time_bins=10,
    )
    expected_cols = {
        "experiment", "view", "frame", "n_inst",
        "n_top", "n_side1", "n_side2",
        "max_score", "min_score",
        "x_first", "y_first", "score_first",
        "s3_gap", "signals_fired", "n_signals",
        "suspicion_score", "time_bin", "time_sec",
    }
    assert expected_cols.issubset(df.columns)
    # All views have n_inst==2 → no S3 firing.
    assert (df["s3_gap"] == 0).all()
    assert df["n_signals"].max() >= 1

    selected = select_candidates(
        df, target_count=20, min_inter_pick_frames=5, time_bins=10
    )
    n_picked = int(selected["picked"].sum())
    assert 1 <= n_picked <= 20
    # Picks should distribute across views (best-effort across small fixture).
    picked = selected[selected["picked"]]
    assert picked["view"].nunique() >= 1


def test_signals_fire_for_score_dispersion():
    # n_inst=3 with widely spread scores triggers S5 in feature builder.
    # Here we directly synthesise a row where S5 should be set.
    n_frames = 50
    df = pd.DataFrame({
        "frame": np.arange(n_frames),
        "n_inst": np.full(n_frames, 3, dtype=int),
        "max_score": np.full(n_frames, 0.99),
        "min_score": np.full(n_frames, 0.30),  # dispersion 0.69 > 0.3, min<0.6
        "x_first": np.linspace(0, 10, n_frames),
        "y_first": np.zeros(n_frames),
        "score_first": np.full(n_frames, 0.99),
        "s1": np.zeros(n_frames, dtype=bool),
        "s2": np.zeros(n_frames, dtype=bool),
        "s5": np.ones(n_frames, dtype=bool),
    })
    per_view = {"top_cam": {"df": df, "meta": {}}}
    out = aggregate_experiment(
        "synth", per_view, pre_link_json_path=None, fps=166.0,
        skip_frame=0, time_bins=5,
    )
    assert (out["signals_fired"].str.contains("S5")).all()
    assert pytest.approx(out["suspicion_score"].iloc[0]) == SIGNAL_WEIGHTS["S5"]


def test_diversification_spreads_picks_across_time_bins():
    n_frames = 1000
    rows: list[dict] = []
    for f in range(n_frames):
        rows.append({
            "frame": f,
            "n_inst": 2,
            "max_score": 0.95,
            "min_score": 0.90,
            "x_first": 0.0, "y_first": 0.0, "score_first": 0.95,
            "s1": True,  # every frame has S1 firing.
            "s2": False,
            "s5": False,
        })
    df = pd.DataFrame(rows)
    per_view = {"top_cam": {"df": df, "meta": {}}}
    out = aggregate_experiment(
        "synth", per_view, pre_link_json_path=None, fps=166.0,
        skip_frame=0, time_bins=10,
    )
    selected = select_candidates(
        out, target_count=50, min_inter_pick_frames=5, time_bins=10
    )
    picked = selected[selected["picked"]]
    # With target=50 and 10 time bins, picks should occupy at least half the bins.
    assert picked["time_bin"].nunique() >= 5
