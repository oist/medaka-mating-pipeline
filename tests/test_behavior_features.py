"""Kinematic features, bout summaries, and trial discovery. No review package is read."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import behavior_features as F
from scripts.analysis.behavior_dataset import build_trial, find_package, locate_boris, parse_group
from scripts.analysis.behavior_report import (aligned_figure, competition_figure, features_figure, save_figure,
                                              timeline_figure)
from scripts.review.ledger import EVENT_COLUMNS


def _straight(n=8, base=(0.0, 0.0, 0.0)):
    """One fish, mouth ahead of the pectoral on +x, so the heading is +x and the bend is 180."""
    nodes = np.zeros((n, 1, 3, 3))
    nodes[:, 0, 0] = (base[0] + 5, base[1], base[2])
    nodes[:, 0, 1] = base
    nodes[:, 0, 2] = (base[0] - 5, base[1], base[2])
    return nodes


def test_heading_bend_and_body_frame():
    nodes = _straight()
    head = F.headings(nodes)
    assert head[0, 0, 0] == pytest.approx(1.0)
    assert F.bend_deg(nodes)[0, 0] == pytest.approx(180.0)
    nodes[0, 0, 2] = (0, 5, 0)  # tail on +y: a right angle at the pectoral
    assert F.bend_deg(nodes)[0] == pytest.approx(90.0)

    female = np.zeros((1, 3))
    heading = np.array([[1.0, 0.0, 0.0]])
    forward, lateral = F.body_offset(np.array([[5.0, 3.0, 0.0]]), female, heading)
    assert forward[0] == pytest.approx(5.0)
    assert lateral[0] == pytest.approx(-3.0)  # right is heading × up


def test_orbit_rate_of_a_steady_circle():
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    center = np.zeros((n, 3))
    orbiter = np.c_[np.cos(theta), np.sin(theta), np.zeros(n)]
    rate = F.orbit_rate(center, orbiter, fps=1.0)
    assert np.isnan(rate[0])
    assert rate[1:] == pytest.approx(18.0, abs=1e-6)  # 360 degrees over 20 frames


def test_named_nodes_follow_a_swap():
    nodes = np.zeros((6, 3, 3, 3))
    nodes[:, 0, :, 0], nodes[:, 1, :, 0], nodes[:, 2, :, 0] = 0, 10, 20
    owner = np.tile(np.arange(3), (6, 1))
    owner[3:, [1, 2]] = owner[3:, [2, 1]]
    labels = {"ID_1": "F", "ID_2": "M1", "ID_3": "M2"}
    named, valid = F.named_nodes(nodes, owner, ["ID_1", "ID_2", "ID_3"], labels)
    assert valid.all()
    assert named[0, 1, 0, 0] == pytest.approx(10)  # M1 starts on track 1
    assert named[4, 1, 0, 0] == pytest.approx(20)  # and is carried onto track 2
    assert named[4, 2, 0, 0] == pytest.approx(10)


def test_short_bout_uses_the_pad_and_follow_is_dropped():
    n = 100
    nodes = np.zeros((n, 3, 3, 3))
    for slot, x in enumerate((0.0, 40.0, 100.0)):
        nodes[:, slot, 0] = (x + 5, 0, 0)
        nodes[:, slot, 1] = (x, 0, 0)
        nodes[:, slot, 2] = (x - 5, 0, 0)
    table = F.kinematic_table(nodes, np.arange(n), fps=10.0)
    bouts = pd.DataFrame([
        {"start": 50, "end": 50, "subject": "m1tf", "who": "M1->F", "behavior": "attack",
         "category": "aggressive", "lane": 0, "point": True},
        {"start": 10, "end": 20, "subject": "m1tf", "who": "M1->F", "behavior": "follow",
         "category": "courtship", "lane": 0, "point": False},
    ])
    out = F.bout_features(table, bouts, fps=10.0, first=0)
    assert list(out["behavior"]) == ["attack"]
    assert out.loc[0, "window_start"] == 45  # 0.5 s at 10 fps, before the one-frame key
    assert out.loc[0, "dist"] == pytest.approx(40.0)


def _package(root: Path, experiment: str, n: int = 30) -> Path:
    folder = root / experiment / "pkg"
    ledger = folder / "id_pass"
    ledger.mkdir(parents=True)
    nodes = np.zeros((n, 3, 3, 3), np.float32)
    for slot, x in enumerate((0.0, 40.0, 90.0)):
        nodes[:, slot, 0] = (x + 5, 0, 0)
        nodes[:, slot, 1] = (x, 0, 0)
        nodes[:, slot, 2] = (x - 5, 0, 0)
    np.savez(folder / "review_data.npz", frames=np.arange(n), fish_ids=np.array(["ID_1", "ID_2", "ID_3"]),
             node_names=np.array(["mouth", "pectoral", "tail"]), nodes3d_filled=nodes,
             xyz=nodes[:, :, 1], view_present=np.ones((n, 3, 3), bool), is_interpolated=np.zeros((n, 3), bool))
    rows = [{"kind": "analysis_start", "fish": "*", "frame": 0, "value": "1"},
            {"kind": "label", "fish": "ID_1", "frame": 0, "value": "F"},
            {"kind": "label", "fish": "ID_2", "frame": 0, "value": "M1"},
            {"kind": "label", "fish": "ID_3", "frame": 0, "value": "M2"}]
    events = pd.DataFrame(rows)
    for column in EVENT_COLUMNS:
        if column not in events:
            events[column] = ""
    events[EVENT_COLUMNS].to_csv(ledger / "human_events.csv", index=False)
    pd.DataFrame([{"session": "s", "frame_start": 0, "frame_end": n - 1, "speed": 1, "mode": "play",
                   "wall_start": 0, "wall_end": 1}]).to_csv(ledger / "coverage.csv", index=False)
    proxy = folder.parent / "proxy"
    proxy.mkdir()
    (proxy / "proxy_manifest.json").write_text(json.dumps({
        "views": {"top_cam": {"source_fps": 10, "play_step": 2, "play_file": "top_cam.play.mp4"}}}), encoding="utf-8")
    return folder


def _boris(path: Path, experiment: str) -> None:
    path.write_text(
        "Observation id,Subject,Behavior,Behavioral category,Behavior type,Time,Image index,Media file name\n"
        f"{experiment},m1tf,quick circle,courtship,START,1,5,x\n"
        f"{experiment},m1tf,quick circle,courtship,STOP,2,15,x\n"
        f"{experiment},m1tf,follow,courtship,START,0,0,x\n"
        f"{experiment},m1tf,follow,courtship,STOP,0.5,4,x\n", encoding="utf-8")


def test_build_trial_names_fish_and_skips_follow(tmp_path):
    experiment = "20260101_P_9_1_FCMC"
    package = _package(tmp_path, experiment)
    boris = tmp_path / f"{experiment}.csv"
    _boris(boris, experiment)
    trial, _table = build_trial(package, boris, experiment)
    assert list(trial.bouts["behavior"]) == ["quick circle"]
    assert trial.bouts.loc[0, "dist"] == pytest.approx(40.0, abs=0.1)
    assert trial.provenance["usable_frames"] == 30
    for name, figure in (("timeline", timeline_figure), ("features", features_figure),
                         ("aligned", aligned_figure), ("competition", competition_figure)):
        save_figure(figure(trial), tmp_path / f"{name}.png")
        assert (tmp_path / f"{name}.png").is_file()


def test_package_and_boris_discovery(tmp_path):
    experiment = "20260113_P_1_2_FCMC"
    _package(tmp_path / "review", experiment)
    assert find_package(tmp_path / "review", experiment).name == "pkg"
    second = tmp_path / "review" / experiment / "other" / "id_pass"
    second.mkdir(parents=True)
    (second / "human_events.csv").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="expected one package"):
        find_package(tmp_path / "review", experiment)

    archive = tmp_path / "boris"
    archive.mkdir()
    (archive / "20260112_P_1_2_FCMC.csv").write_text("a", encoding="utf-8")
    found, how = locate_boris(experiment, [archive])
    assert how == "trial_key" and found.name.startswith("20260112")
    (archive / "20260111_P_1_2_FCMC.csv").write_text("b", encoding="utf-8")
    with pytest.raises(SystemExit, match="several BORIS"):
        locate_boris(experiment, [archive])


def test_review_bin_keeps_a_short_discard_and_only_the_six_behaviours_are_scored():
    status = np.ones(32, int)
    status[20] = 3
    assert list(F.review_bins(np.arange(32), status, bin_size=16)["review"]) == ["kept", "discarded"]
    n = 20
    nodes = np.zeros((n, 3, 3, 3))
    table = F.kinematic_table(nodes, np.arange(n), fps=10.0)
    bouts = pd.DataFrame([
        {"start": 2, "end": 4, "who": "M1->F", "behavior": "approach", "category": "courtship", "point": False},
        {"start": 8, "end": 8, "who": "M1->M2", "behavior": "attack", "category": "aggressive", "point": True},
    ])
    out = F.bout_features(table, bouts, fps=10.0, first=0)
    assert list(out["behavior"]) == ["attack"]
    assert out.loc[0, "target"] == "M2"


def test_mutual_who_is_not_read_as_directed():
    assert F.parse_who("M1<->M2") == ("M1", "M2", False)
    assert F.parse_who("M1->F") == ("M1", "F", True)


def test_parse_group_accepts_the_ways_it_gets_typed():
    assert parse_group("P-1") == parse_group("p1") == parse_group("P_1") == "P_1"
    with pytest.raises(SystemExit):
        parse_group("group 1")
