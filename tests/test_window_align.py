"""Tests for swap-corrected, co-visible identity labeling (window_align)."""
import numpy as np
import pytest

from sam2_id_experiment.src.id_model.window_align import (
    align_windows_by_appearance,
    select_covisible_mask,
    sex_track_from_window_anchors,
)


def _make_window(win, perm, centroids, rng, n=60, jitter=0.1):
    """perm[g] = true fish id assigned to within-window gid g in this window."""
    emb, gid, window = [], [], []
    for g in range(3):
        emb.append(centroids[perm[g]] + rng.normal(0, jitter, (n, centroids.shape[1])))
        gid += [g] * n
        window += [win] * n
    return np.vstack(emb), np.array(gid), np.array(window, dtype=object), np.array([perm[g] for g in gid])


def test_align_recovers_cross_window_permutation():
    rng = np.random.default_rng(0)
    C = np.eye(3, 8) * 5.0
    # window A: identity naming; window B: cyclic-permuted naming (a cross-window swap)
    eA, gA, wA, tA = _make_window("A", [0, 1, 2], C, rng)
    eB, gB, wB, tB = _make_window("B", [1, 2, 0], C, rng)
    emb = np.vstack([eA, eB])
    gid = np.concatenate([gA, gB])
    window = np.concatenate([wA, wB])
    true_fish = np.concatenate([tA, tB])

    aligned, info = align_windows_by_appearance(emb, window, gid)

    # ref is the most-separated window (tie -> first); both good, none dropped
    assert info["dropped"] == []
    # aligned ids must name the SAME physical fish across both windows
    assert (aligned == true_fish).mean() > 0.99
    # specifically: window B's raw gid is permuted but aligned matches window A's fish
    assert (aligned[window == "B"] == tB).mean() > 0.99


def test_drops_unseparated_bad_window():
    rng = np.random.default_rng(1)
    C = np.eye(3, 8) * 5.0
    eA, gA, wA, _ = _make_window("good", [0, 1, 2], C, rng)
    # bad window: all 3 fish collapse to one blob (no appearance separation)
    eBad, gBad, wBad = [], [], []
    for g in range(3):
        eBad.append(rng.normal(0, 0.1, (60, 8)))
        gBad += [g] * 60
        wBad += ["bad"] * 60
    emb = np.vstack([eA, np.vstack(eBad)])
    gid = np.concatenate([gA, np.array(gBad)])
    window = np.concatenate([wA, np.array(wBad, dtype=object)])

    aligned, info = align_windows_by_appearance(emb, window, gid)

    assert info["ref"] == "good"
    assert "bad" in info["dropped"]
    assert (aligned[window == "bad"] == -1).all()
    assert (aligned[window == "good"] >= 0).all()


def test_drops_window_missing_a_fish():
    # a window where fish id 2 never appears must be dropped, not aligned with a NaN centroid
    rng = np.random.default_rng(2)
    C = np.eye(3, 8) * 5.0
    eA, gA, wA, _ = _make_window("good", [0, 1, 2], C, rng)
    eM, gM, wM = [], [], []
    for g in (0, 1):  # fish id 2 absent
        eM.append(C[g] + rng.normal(0, 0.1, (40, 8)))
        gM += [g] * 40
        wM += ["miss"] * 40
    emb = np.vstack([eA, np.vstack(eM)])
    gid = np.concatenate([gA, np.array(gM)])
    window = np.concatenate([wA, np.array(wM, dtype=object)])

    aligned, info = align_windows_by_appearance(emb, window, gid)
    assert "miss" in info["dropped"]
    assert (aligned[window == "miss"] == -1).all()
    assert (aligned[window == "good"] >= 0).all()


def test_degenerate_when_nothing_separates():
    # all fish collapse to one blob in every window -> degenerate, everything dropped
    rng = np.random.default_rng(3)
    emb = rng.normal(0, 0.05, (180, 8))
    gid = np.array([0, 1, 2] * 60)
    window = np.array(["a"] * 90 + ["b"] * 90, dtype=object)
    aligned, info = align_windows_by_appearance(emb, window, gid, min_ref_sep=0.5)
    assert info["degenerate"] is True
    assert (aligned == -1).all()


def test_sex_track_from_window_anchors():
    # two anchors disagreeing on the female slot (a linker swap between them)
    frame = np.array([5, 5, 5, 50, 50, 50, 105, 105, 105])
    slot = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    spans = [(0, 10, 0), (100, 110, 2)]  # female = slot0 early, slot2 late
    sex, conf = sex_track_from_window_anchors(frame, slot, spans)
    # frame 5 inside window 0 -> female is slot 0, high confidence
    assert sex[:3].tolist() == ["F", "M", "M"] and (conf[:3] == "high").all()
    # frame 50 in the disagreeing gap -> low confidence, nearer anchor (window 0, slot 0)
    assert sex[3:6].tolist() == ["F", "M", "M"] and (conf[3:6] == "low").all()
    # frame 105 inside window 1 -> female is slot 2, high confidence
    assert sex[6:9].tolist() == ["M", "M", "F"] and (conf[6:9] == "high").all()


def test_sex_track_agreeing_anchors_are_med():
    frame = np.array([50, 50, 50])
    slot = np.array([0, 1, 2])
    spans = [(0, 10, 1), (100, 110, 1)]  # both anchors agree female = slot 1
    sex, conf = sex_track_from_window_anchors(frame, slot, spans)
    assert sex.tolist() == ["M", "F", "M"] and (conf == "med").all()


def test_select_covisible_mask():
    # frame 10: all 3 fish present (co-visible); frame 11: only 2 fish -> dropped
    view = ["s1", "s1", "s1", "s1", "s1"]
    frame = [10, 10, 10, 11, 11]
    gid = [0, 1, 2, 0, 1]
    mask = select_covisible_mask(view, frame, gid, num_fish=3)
    assert mask.tolist() == [True, True, True, False, False]


def test_select_covisible_separates_views():
    # same frame number on two views; each view independently needs all 3
    view = ["s1", "s1", "s1", "s2", "s2"]
    frame = [10, 10, 10, 10, 10]
    gid = [0, 1, 2, 0, 1]
    mask = select_covisible_mask(view, frame, gid, num_fish=3)
    assert mask[:3].all() and not mask[3:].any()
