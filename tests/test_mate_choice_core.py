"""Unit tests for scripts/analysis/mate_choice_core.py (synthetic 3-fish tracks)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import mate_choice_core as mc


# --------------------------------------------------------------------- geometry helpers
def test_pairwise_distance_basic_and_nan():
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    b = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    d = mc.pairwise_distance(a, b)
    assert d[0] == pytest.approx(5.0)
    assert d[1] == pytest.approx(0.0)
    assert np.isnan(d[2])


def test_speed_first_frame_nan_and_value():
    a = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    v = mc.speed(a, fps=10.0)
    assert np.isnan(v[0])
    assert v[1] == pytest.approx(10.0)   # 1 mm * 10 fps
    assert v[2] == pytest.approx(0.0)


def test_runs_over_breaks_on_gap_and_nan():
    mask = np.array([1, 1, 1, 0, 1, 1, np.nan, 1, 1, 1], float)
    runs = mc.runs_over(mask, min_len=3)
    assert runs == [(0, 2), (7, 9)]   # the 2-long run (4,5) is dropped; NaN breaks


# ----------------------------------------------------------------------- track loading
def _synthetic_frame(n, posF, posM1, posM2):
    rows = []
    for f in range(n):
        for fid, p in (("F", posF), ("M1", posM1), ("M2", posM2)):
            rows.append({"frame": f, "id_final": fid, "X": p[f, 0], "Y": p[f, 1], "Z": p[f, 2]})
    return pd.DataFrame(rows)


def test_track_from_frame_roundtrip_and_missing_fish_nan():
    n = 5
    F = np.zeros((n, 3)); M1 = np.ones((n, 3)) * 5.0; M2 = np.ones((n, 3)) * 50.0
    pos, got_n = mc.track_from_frame(_synthetic_frame(n, F, M1, M2))
    assert got_n == n
    assert np.allclose(pos["F"], F)
    assert np.allclose(pos["M1"], M1)
    # a track with no M2 rows -> all NaN, not a crash
    df = _synthetic_frame(n, F, M1, M2)
    df = df[df.id_final != "M2"]
    pos2, _ = mc.track_from_frame(df)
    assert np.isnan(pos2["M2"]).all()


# --------------------------------------------------------------- association / mate choice
def _prefers_M1_track(n=2000):
    F = np.zeros((n, 3))
    M1 = np.zeros((n, 3)); M1[:, 0] = 5.0     # always 5 mm from F (within court range)
    M2 = np.zeros((n, 3)); M2[:, 0] = 100.0   # always far
    return mc.compute_distances({"F": F, "M1": M1, "M2": M2}, fps=mc.FPS)


def test_association_prefers_nearer_male():
    p = mc.BehaviourParams()
    a = mc.association_metrics(_prefers_M1_track(), p)
    assert a["preferred_male"] == "M1"
    assert a["association_index_M1"] == pytest.approx(1.0)
    assert a["closer_to_M1_frac"] == pytest.approx(1.0)
    assert a["time_near_M2_s"] == pytest.approx(0.0)


def test_courtship_bouts_one_sustained_bout():
    p = mc.BehaviourParams(min_bout=10)
    d = _prefers_M1_track(n=500)
    cb = mc.courtship_bouts(d, p)
    assert cb["M1"]["n_bouts"] == 1           # one continuous in-range run
    assert cb["M2"]["n_bouts"] == 0
    assert cb["M1"]["total_s"] == pytest.approx(500 / mc.FPS, abs=0.05)


# ----------------------------------------------------------------- aggression bouts
def test_aggression_bout_detected_with_initiator():
    n = 200
    F = np.full((n, 3), 100.0)                 # female parked far away
    M2 = np.zeros((n, 3)); M2[:, 0] = 10.0     # M2 static, 10 mm from origin (within agg_mm=25)
    M1 = np.zeros((n, 3))
    # M1 lunges fast: ±1.5 mm/frame jitter -> 1.5*166 = 249 mm/s > agg_speed=150, stays ~10 mm from M2
    M1[:, 1] = 1.5 * (np.arange(n) % 2)
    d = mc.compute_distances({"F": F, "M1": M1, "M2": M2})
    p = mc.BehaviourParams(min_bout=9)
    bouts = mc.aggression_bouts(d, p)
    assert len(bouts) >= 1
    assert bouts[0]["initiator"] == "M1"       # M1 is the moving (lunging) fish
    assert bouts[0]["min_d_mm"] < p.agg_mm
    summ = mc.aggression_summary(bouts)
    assert summ["initiator_M1"] >= 1


def test_no_aggression_when_males_far():
    n = 200
    F = np.full((n, 3), 0.0)
    M1 = np.zeros((n, 3)); M2 = np.full((n, 3), 80.0)   # 80 mm apart > agg_mm
    M1[:, 1] = np.linspace(0, 80, n)
    d = mc.compute_distances({"F": F, "M1": M1, "M2": M2})
    assert mc.aggression_bouts(d, mc.BehaviourParams()) == []


# ----------------------------------------------------------------- temporal dynamics
def test_temporal_preference_detects_switch_and_build():
    n = 2000
    sw = 800  # switch at 40% -> M1 dominates overall (last 60%)
    F = np.zeros((n, 3))
    M1 = np.zeros((n, 3)); M2 = np.zeros((n, 3))
    # early: F near M2; after the switch: F near M1  -> a switch that BUILDS toward M1
    M1[:sw, 0] = 100.0; M1[sw:, 0] = 5.0
    M2[:sw, 0] = 5.0;   M2[sw:, 0] = 100.0
    d = mc.compute_distances({"F": F, "M1": M1, "M2": M2})
    table, summary = mc.temporal_preference(d, mc.BehaviourParams(n_bins=10))
    assert len(table) == 10
    assert summary["overall_preferred_male"] == "M1"
    assert summary["late_pref_index_M1"] > summary["early_pref_index_M1"]
    assert summary["n_switch_bins"] >= 1          # early bins favour M2
    assert summary["trend"] == "builds"           # late preference for the chosen male strengthens


def test_temporal_preference_stable_when_constant():
    d = _prefers_M1_track(n=2000)
    _table, summary = mc.temporal_preference(d, mc.BehaviourParams(n_bins=8))
    assert summary["overall_preferred_male"] == "M1"
    assert summary["n_switch_bins"] == 0
    assert summary["early_predicts_choice"] is True


# ----------------------------------------------------------------- spawn + bout table
def test_spawn_attribution_picks_contact_male():
    d = _prefers_M1_track(n=1000)
    sp = mc.spawn_attribution(d, mc.BehaviourParams(), spawn_frame=500)
    assert sp["spawn_male"] == "M1"
    assert sp["contact"] is True
    assert mc.spawn_attribution(d, mc.BehaviourParams(), spawn_frame=None) is None
    assert mc.spawn_attribution(d, mc.BehaviourParams(), spawn_frame=99999) is None


def test_spawn_attribution_all_nan_window_does_not_silently_default():
    # female occluded across the spawn window -> must NOT silently pick a male.
    n = 1000
    F = np.full((n, 3), np.nan)
    M1 = np.zeros((n, 3)); M2 = np.full((n, 3), 50.0)
    d = mc.compute_distances({"F": F, "M1": M1, "M2": M2})
    sp = mc.spawn_attribution(d, mc.BehaviourParams(), spawn_frame=500)
    assert sp["spawn_male"] is None
    assert sp["contact"] is False
    # one male tracked, the other not -> pick the tracked one (no NaN comparison)
    F2 = np.zeros((n, 3))
    M2nan = np.full((n, 3), np.nan)
    d2 = mc.compute_distances({"F": F2, "M1": M1, "M2": M2nan})
    assert mc.spawn_attribution(d2, mc.BehaviourParams(), spawn_frame=500)["spawn_male"] == "M1"


def test_aggression_initiator_tie_goes_to_M2():
    # exact-tie mean speed -> attribute to M2 (matches original strict '>').
    n = 60
    F = np.full((n, 3), 100.0)
    M1 = np.zeros((n, 3)); M2 = np.zeros((n, 3)); M2[:, 0] = 10.0
    # identical motion for both males -> equal mean speed
    jit = 1.5 * (np.arange(n) % 2)
    M1[:, 1] = jit; M2[:, 1] = jit
    d = mc.compute_distances({"F": F, "M1": M1, "M2": M2})
    bouts = mc.aggression_bouts(d, mc.BehaviourParams(min_bout=9))
    assert bouts and all(b["initiator"] == "M2" for b in bouts)


def test_bout_table_and_breakdown_combine_types():
    n = 1500
    F = np.zeros((n, 3))
    M1 = np.zeros((n, 3)); M1[:, 0] = 5.0       # F-M1 courtship throughout (d=5 mm < court_mm)
    M2 = np.zeros((n, 3)); M2[:, 0] = 12.0      # M1-M2 ~12 mm apart (within agg_mm=25)
    M2[:, 1] = 1.5 * (np.arange(n) % 2)         # M2 fast jitter (249 mm/s > agg_speed) -> aggression
    d = mc.compute_distances({"F": F, "M1": M1, "M2": M2})
    p = mc.BehaviourParams(min_bout=20)
    tbl = mc.bout_table(d, p)
    assert set(tbl["type"]) == {"courtship", "aggression"}   # BOTH types must be present
    assert (tbl["type"] == "courtship").any()
    assert (tbl["type"] == "aggression").any()
    bd = mc.behaviour_breakdown(d, p)
    assert bd["n_courtship_bouts"] >= 1
    assert bd["n_aggression_bouts"] >= 1
    assert bd["courtship_total_s"] > 0
    # precomputed-bout path must match the recompute path (efficiency wiring is correct)
    cb = mc.courtship_bouts(d, p); agg = mc.aggression_bouts(d, p)
    assert mc.bout_table(d, p, courtship=cb, aggression=agg).equals(tbl)
