import numpy as np
import pandas as pd

from scripts.review.postprocess import (
    NODE_BRIDGED,
    PostParams,
    acceleration,
    despike_series,
    dejump_series,
    postprocess_fish,
    postprocess_table,
)

FPS = 166.0


def swim(n=400, speed=0.4):
    t = np.arange(n)
    return np.stack([50 + speed * t, 60 + 5 * np.sin(t / 40), 30 + 0.01 * t], axis=1)


def test_spikes_of_one_to_several_frames_are_bridged():
    clean = swim()
    x = clean.copy()
    x[100] += [0, 6, 0]                 # one frame
    x[200:202] += [5, -4, 0]            # two frames
    x[300:306] += [0, 0, 8]             # six frames
    out, replaced, rows = despike_series(x, PostParams())
    assert np.abs(out - clean).max() < 0.3
    assert replaced[100] and replaced[200:202].all() and replaced[300:306].all()
    assert replaced.sum() <= 9 + 6         # at most a frame or so of slack per run
    assert acceleration(out).max() < 3.0
    assert sum(r[0] == "spike" and r[4] for r in rows) == 3


def test_tail_beat_and_fast_turn_are_left_alone():
    t = np.arange(600)
    beat = np.stack([4 * np.sin(2 * np.pi * 10 * t / FPS), 2 * np.sin(2 * np.pi * 20 * t / FPS), 0 * t], axis=1)
    out, replaced, _ = despike_series(beat, PostParams())
    assert not replaced.any() and np.allclose(out, beat)
    # Tail offset (18 mm) turning 180 degrees within 6 frames (an escape turn), then holding.
    ang = np.r_[np.zeros(100), np.linspace(0, np.pi, 7)[1:], np.full(100, np.pi)]
    turn = np.stack([18 * np.cos(ang), 18 * np.sin(ang), 0 * ang], axis=1)
    out, replaced, rows = despike_series(turn, PostParams())
    assert not replaced.any()
    assert all(not r[4] for r in rows)    # reported as a jerk at most, never replaced


def test_jump_is_spread_over_a_few_frames():
    x = swim(200)
    x[100:] += [20, 0, 0]               # a 20 mm step (wrong identity join)
    out, replaced, rows = dejump_series(x, PostParams())
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert steps.max() <= 6.0
    assert 4 <= replaced.sum() <= 12
    assert rows and rows[0][0] == "jump" and rows[0][4]


def test_body_follows_the_pectoral_and_sources_are_marked():
    n = 300
    pect = swim(n)
    mouth = pect + [7, 0, 0]
    tail = pect - [18, 0, 0]
    tail[150] += [0, 7, 0]              # tail spike
    pect[220:] += [15, 0, 0]            # jump of the whole fish ...
    mouth[220:] += [15, 0, 0]
    tail[220:] += [15, 0, 0]
    nodes = np.stack([mouth, pect, tail], axis=1)
    source = np.zeros((n, 3), np.int8)
    out, src, report = postprocess_fish(nodes, source, pect=1)
    assert np.allclose(out[:, 0] - out[:, 1], [7, 0, 0], atol=0.3)          # mouth keeps its offset
    assert np.abs(out[150, 2] - (out[150, 1] - [18, 0, 0])).max() < 0.3     # tail spike bridged
    assert src[150, 2] == NODE_BRIDGED and (src[:, 1] == NODE_BRIDGED).any()
    assert {r["kind"] for r in report if r["repaired"]} == {"spike", "jump"}


def test_table_position_becomes_the_cleaned_pectoral():
    n = 120
    rows = []
    for fid, label, off in (("ID_1", "F", 0.0), ("ID_2", "M1", 40.0)):
        p = swim(n) + off
        for t in range(n):
            r = {"frame": t, "fish_id": fid, "label": label, "X": p[t, 0] + 1, "Y": p[t, 1], "Z": p[t, 2]}
            for name, d in (("mouth", 7.0), ("pectoral", 0.0), ("tail", -18.0)):
                r.update({f"{name}_X": p[t, 0] + d, f"{name}_Y": p[t, 1], f"{name}_Z": p[t, 2], f"{name}_source": 0})
            rows.append(r)
    table = pd.DataFrame(rows).sort_values(["frame", "fish_id"]).reset_index(drop=True)
    table.loc[(table.frame == 60) & (table.fish_id == "ID_2"), "pectoral_Y"] += 6
    out, report = postprocess_table(table, ["mouth", "pectoral", "tail"])
    assert np.allclose(out["X"], out["pectoral_X"]) and np.allclose(out["centroid_X"], table["X"])
    fixed = out[(out.frame == 60) & (out.fish_id == "ID_2")].iloc[0]
    assert abs(fixed["pectoral_Y"] - table[(table.frame == 59) & (table.fish_id == "ID_2")]["pectoral_Y"].iloc[0]) < 0.5
    assert list(report["label"].unique()) == ["M1"] and report["repaired"].all()
