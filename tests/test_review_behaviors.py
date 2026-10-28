"""BORIS behaviour bouts for the review GUI's behaviour row."""

import os
import time

import pandas as pd

from scripts.review.behaviors import category_of, find_boris_csv, load_bouts, subject_label

BORIS_COLUMNS = ["Observation id", "Subject", "Behavior", "Behavioral category", "Behavior type", "Time",
                 "Image index", "Media file name"]


def write_boris(path, rows):
    """rows: (subject, behavior, category, type, frame)."""
    df = pd.DataFrame([{"Observation id": "exp", "Subject": s, "Behavior": b, "Behavioral category": c,
                        "Behavior type": t, "Time": f / 166.0, "Image index": f, "Media file name": "top.mp4"}
                       for s, b, c, t, f in rows], columns=BORIS_COLUMNS)
    df.to_csv(path, index=False)
    return path


def test_subjects_become_who_and_lane():
    assert subject_label("m1tf") == ("M1->F", 0)
    assert subject_label("m2tm1") == ("M2->M1", 1)
    assert subject_label("m1m2") == ("M1<->M2", 1)
    assert subject_label("f") == ("F", 2)
    assert subject_label("weird") == ("weird", 2)
    assert [category_of(c) for c in ("courtship", "Courtship", "aggressive", "Competitive", "Mating", "")] == \
        ["courtship", "courtship", "aggressive", "aggressive", "mating", "other"]


def test_load_bouts_pairs_start_stop_and_keeps_points(tmp_path):
    path = write_boris(tmp_path / "exp.csv", [
        ("m1tf", "follow", "courtship", "START", 100), ("m1tm2", "attack", "aggressive", "START", 120),
        ("m1tm2", "attack", "aggressive", "STOP", 130), ("m1tf", "follow", "courtship", "STOP", 200),
        ("f", "spawn", "courtship", "POINT", 250)])
    b = load_bouts(path)
    assert list(b["start"]) == [100, 120, 250] and list(b["end"]) == [200, 130, 250]
    assert list(b["who"]) == ["M1->F", "M1->M2", "F"] and list(b["lane"]) == [0, 1, 2]
    assert list(b["category"]) == ["courtship", "aggressive", "courtship"] and list(b["point"]) == [False, False, True]


def test_find_boris_csv_searches_folders_recursively_newest_first(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "deep").mkdir(parents=True)
    old = write_boris(tmp_path / "a" / "exp.csv", [])
    new = write_boris(tmp_path / "b" / "deep" / "exp.csv", [])
    past = time.time() - 100
    os.utime(old, (past, past))
    assert find_boris_csv("exp", [tmp_path / "missing", tmp_path]) == new
    assert find_boris_csv("exp", [old]) == old
    assert find_boris_csv("other", [tmp_path]) is None


def test_overlapping_bouts_get_sub_rows(tmp_path):
    from scripts.review.behaviors import bout_rows

    path = write_boris(tmp_path / "exp.csv", [
        ("m1tf", "follow", "courtship", "START", 100), ("m1tf", "quick circle", "courtship", "START", 150),
        ("m1tf", "quick circle", "courtship", "STOP", 160), ("m1tf", "follow", "courtship", "STOP", 300),
        ("m2tf", "approach", "courtship", "START", 155), ("m2tf", "approach", "courtship", "STOP", 170),
        ("m1tf", "wrap", "mating", "START", 310), ("m1tf", "wrap", "mating", "STOP", 320)])
    b = load_bouts(path)
    rows, sub = bout_rows(b)
    assert rows == [2, 1, 1]                      # three overlap at 155-160, capped at two rows
    by = {(r.behavior, r.start): sub[i] for i, r in zip(b.index, b.itertuples(index=False))}
    assert by[("follow", 100)] == 0 and by[("quick circle", 150)] == 1 and by[("wrap", 310)] == 0
