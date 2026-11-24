import numpy as np

from scripts.analysis.review_overview import colour_overlap
from scripts.review import id_segments as S


def test_discarded_time_split_across_machine_colours():
    # 0 safe, 1 orange, 2 red. The last two frames are before the analysis start.
    machine = np.array([2, 2, 2, 2, 1, 1, 0, 0, 2, 1], np.int8)
    status = np.array([
        S.UNUSABLE, S.UNUSABLE, S.WATCHED_SAFE, S.UNWATCHED,
        S.UNUSABLE, S.WATCHED_DANGER, S.UNUSABLE, S.WATCHED_SAFE,
        S.NOT_ANALYSED, S.NOT_ANALYSED,
    ], np.int8)
    row = colour_overlap(machine, status)
    # discarded frames: two red, one orange, one safe
    assert row["discarded_in_red_pct"] == 50.0
    assert row["discarded_in_orange_pct"] == 25.0
    assert row["discarded_in_safe_pct"] == 25.0
    # red analysed frames: four, two of them discarded
    assert row["red_discarded_pct"] == 50.0
    assert row["red_kept_pct"] == 25.0
    # three discarded runs, and only the opening pair is entirely red
    assert row["discarded_runs"] == 3
    assert row["discarded_runs_all_red"] == 1
