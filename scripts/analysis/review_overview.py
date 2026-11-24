"""All reviewed recordings on one page, and what the machine flagged against what the reviewer did.

Two figures, each written as PNG and PDF, plus the numbers behind the second as a csv::

    .venv\\Scripts\\python.exe -m scripts.analysis.review_overview --group K_1 --group P_1

* ``review_timeline``: one row per recording on a shared time axis. Review status
  (kept / discarded / unwatched) with its percentage, and the six behaviours above it.
  Each male has two lanes, toward the female and toward the other male. A second level
  for a crowded bout stays inside its lane.
* ``machine_vs_review``: for each recording, the machine's colours on one row and the
  review on the row below, on the same time axis. Red is a whole danger run whose
  closest approach was under 8 mm, not every frame under 8 mm. The csv gives the
  overlap: of the discarded time, how much is red, orange or safe, and the reverse.

Days of one group are repeated measures of the same fish. Nothing is written into a
review package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from scripts.analysis import behavior_features as F
from scripts.analysis.behavior_dataset import (Trial, ensure_trial, group_status, parse_group, pool,
                                               resolve_roots)
from scripts.analysis.behavior_report import (REVIEW_COLORS, REVIEW_LABELS, _blocks, _color, aligned_compare,
                                              save_figure)
from scripts.review import id_segments as S
from scripts.review.behaviors import default_search
from scripts.review.ledger import (COVERAGE_COLUMNS, EVENT_COLUMNS, analysis_start, read_csv, swap_list,
                                   watched_intervals)

RED, ORANGE, SAFE = "#d64545", "#f0a030", "#cfd8dc"

# Top to bottom: each male split into toward-the-female and toward-the-other-male.
# Pitch is wider than the shaded band, so a crowded bout stays inside its own lane.
_LANES = (
    ("M1_F", "M1->F", 5.25, "#d9e6f5", "#1f4b99"),
    ("M1_MM", "M1->M2", 4.20, "#e8eef6", "#1f4b99"),
    ("F", "female", 3.15, "#f3e3f6", "#7a3e9d"),
    ("M2_F", "M2->F", 2.10, "#dcefdc", "#1d7a32"),
    ("M2_MM", "M2->M1", 1.05, "#e7f3e7", "#1d7a32"),
    ("review", "review", 0.0, "#ececec", "#333333"),
)
_LANE_Y = {key: y for key, _label, y, _band, _ink in _LANES}
_BAND_HALF = 0.44
_BAR_H = 0.24
_DODGE = 0.13
_CODE_PATH = "scripts/analysis/review_overview.py"


def _day(trial: Trial) -> str:
    """``20260112_P_1_1_FCMC`` -> ``P_1_1``."""
    parts = trial.experiment.split("_")
    return "_".join(parts[1:4]) if len(parts) >= 4 and parts[0].isdigit() else trial.experiment


def _lane_key(actor: str, target: str) -> str:
    """Lane of one bout. A male aimed at the other male has his own row."""
    if actor == "M1":
        return "M1_MM" if target in ("M1", "M2") else "M1_F"
    if actor == "M2":
        return "M2_MM" if target in ("M1", "M2") else "M2_F"
    return "F"


def timeline_overview(trials: list[Trial]) -> Figure:
    fig = Figure(figsize=(22, 2.55 * len(trials) + 2.2), layout="constrained")
    # The day and the percentages sit in their own column, so a longer label cannot
    # shift that row's time axis relative to the rows above and below it.
    grid = fig.add_gridspec(len(trials) + 1, 2, width_ratios=(1.0, 0.16),
                            height_ratios=[1.0] * len(trials) + [0.16])
    axes: list = []
    longest = 0.0
    seen: list[str] = []
    y_top = _LANES[0][2]
    y_review = _LANE_Y["review"]
    labels = [label for _key, label, _y, _band, _ink in _LANES]
    ys = [y for _key, _label, y, _band, _ink in _LANES]
    for i, trial in enumerate(trials):
        ax = fig.add_subplot(grid[i, 0], sharex=axes[0] if axes else None)
        side = fig.add_subplot(grid[i, 1])
        axes.append(ax)
        fps = trial.fps
        frames = trial.traces["frame"].to_numpy(float)
        longest = max(longest, float(frames.max()) / fps / 60)
        for _key, _label, y, band, _ink in _LANES:
            ax.axhspan(y - _BAND_HALF, y + _BAND_HALF, facecolor=band, edgecolor="none", zorder=0)
        for start, width, code in _blocks(frames, trial.traces["review"].to_numpy(), fps):
            ax.barh(y_review, width, left=start, height=0.5, color=REVIEW_COLORS[code], linewidth=0, zorder=2)
        last_end: dict[str, float] = {}
        level: dict[str, int] = {}
        for bout in trial.bouts.sort_values("start").itertuples(index=False):
            male_target = bout.actor in ("M1", "M2") and bout.target in ("M1", "M2")
            lane = _lane_key(str(bout.actor), str(bout.target))
            y = _LANE_Y[lane]
            x = bout.start / fps / 60
            # bouts touching or overlapping the previous one in this lane use its other half
            crowded = x <= last_end.get(lane, -1e9) + 0.08
            level[lane] = 1 - level.get(lane, 0) if crowded else 0
            end_min = bout.end / fps / 60
            last_end[lane] = max(end_min, last_end.get(lane, -1e9)) if crowded else end_min
            y += _DODGE if level[lane] else -_DODGE
            if bout.behavior not in seen:
                seen.append(bout.behavior)
            if bout.seconds < 0.3:
                ax.plot(x, y, marker="x" if male_target else "|", color=_color(bout.behavior),
                        ms=7, mew=1.4, zorder=3)
            else:
                width = max((bout.end - bout.start + 1) / fps / 60, 0.02)
                ax.barh(y, width, left=x, height=_BAR_H, color=_color(bout.behavior),
                        edgecolor="white", linewidth=0.4, zorder=3)
                ax.plot([x, x], [y - _BAR_H / 2, y + _BAR_H / 2], color="black", lw=0.8,
                        solid_capstyle="butt", zorder=4)
        ax.set_yticks(ys, labels)
        for tick, (_key, _label, _y, _band, ink) in zip(ax.get_yticklabels(), _LANES):
            tick.set_fontsize(9)
            tick.set_fontweight("bold")
            tick.set_color(ink)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_ylim(y_review - _BAND_HALF - 0.06, y_top + _BAND_HALF + 0.06)
        ax.set_axisbelow(True)
        pct = trial.provenance.get("review_pct") or {}
        side.set_axis_off()
        side.text(0.0, 0.5, f"{_day(trial)}\nkept {pct.get('kept', 0):.1f}%\n"
                  f"discarded {pct.get('discarded', 0):.1f}%\nunwatched {pct.get('unwatched', 0):.1f}%",
                  transform=side.transAxes, fontsize=12, va="center", ha="left", family="monospace")
    axes[-1].set_xlim(0, longest * 1.01)
    axes[-1].set_xlabel("time (min)")
    note = fig.add_subplot(grid[-1, :])
    note.set_axis_off()
    note.text(0.0, 0.72, "Events shorter than 0.3 s aimed at the other male are crosses; "
              "other short events are vertical ticks.",
              transform=note.transAxes, fontsize=9, color="#333333", ha="left", va="center")
    note.text(0.0, 0.18, _CODE_PATH, transform=note.transAxes, fontsize=8,
              color="#9a9a9a", ha="left", va="center")
    handles = [Patch(facecolor=REVIEW_COLORS[c], edgecolor="none", label=REVIEW_LABELS[c])
               for c in ("kept", "discarded", "unwatched", "before")]
    handles += [Patch(facecolor=_color(b), edgecolor="none", label=F.BEHAVIOR_LABEL.get(b, b))
                for b in F.FOCUS_BEHAVIORS if b in seen]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Review status and behaviour, every reviewed recording\n"
                 "each male has two lanes: toward the female, and toward the other male; "
                 "percentages are of the analysed period; days of one group are the same fish")
    return fig


def _pct(count: int, total: int) -> float | None:
    """Percent, or ``None`` when the denominator is empty."""
    if total <= 0:
        return None
    return round(100.0 * count / total, 2)


def colour_overlap(machine: np.ndarray, status: np.ndarray) -> dict:
    """How kept and discarded frames sit on the machine colours, after the analysis start.

    ``machine`` is 0 safe, 1 orange, 2 red, one entry per package frame. ``status`` is
    ``id_segments.frame_status``. A discarded run is entirely red when every frame of it is red.
    """
    analysed = status != S.NOT_ANALYSED
    discarded = analysed & (status == S.UNUSABLE)
    kept = analysed & np.isin(status, (S.WATCHED_SAFE, S.WATCHED_DANGER))
    unwatched = analysed & (status == S.UNWATCHED)
    red = analysed & (machine == 2)
    orange = analysed & (machine == 1)
    safe = analysed & (machine == 0)
    n = int(analysed.sum())
    n_discarded, n_red, n_orange, n_safe = (int(m.sum()) for m in (discarded, red, orange, safe))
    runs = S.runs_of(discarded)
    all_red = int(sum(bool((machine[a:b] == 2).all()) for a, b in runs))
    return {
        "red_pct": _pct(n_red, n), "orange_pct": _pct(n_orange, n), "safe_pct": _pct(n_safe, n),
        "kept_pct": _pct(int(kept.sum()), n), "discarded_pct": _pct(n_discarded, n),
        "unwatched_pct": _pct(int(unwatched.sum()), n),
        "discarded_in_red_pct": _pct(int((discarded & red).sum()), n_discarded),
        "discarded_in_orange_pct": _pct(int((discarded & orange).sum()), n_discarded),
        "discarded_in_safe_pct": _pct(int((discarded & safe).sum()), n_discarded),
        "red_discarded_pct": _pct(int((red & discarded).sum()), n_red),
        "orange_discarded_pct": _pct(int((orange & discarded).sum()), n_orange),
        "safe_discarded_pct": _pct(int((safe & discarded).sum()), n_safe),
        "red_kept_pct": _pct(int((red & kept).sum()), n_red),
        "discarded_runs": int(len(runs)), "discarded_runs_all_red": all_red,
    }


def compare_recording(trial: Trial) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """One recording: the summary row, the frame numbers, machine colours and review status."""
    package = trial.package
    with np.load(package / "review_data.npz") as z:
        frames = np.asarray(z["frames"], int)
        fish_ids = [str(f) for f in z["fish_ids"]]
        xyz, present, interp = (np.asarray(z[k]) for k in ("xyz", "view_present", "is_interpolated"))
    ledger = package / "id_pass"
    events = read_csv(ledger / "human_events.csv", EVENT_COLUMNS)
    coverage = read_csv(ledger / "coverage.csv", COVERAGE_COLUMNS)
    fps, first, n = trial.fps, int(frames[0]), len(frames)
    danger = S.danger_map(xyz.astype(float), present, interp, fps)
    start = analysis_start(events)
    swaps = swap_list(events, fish_ids)
    status = S.frame_status(first, n, start, danger["danger"], danger["runs"],
                            watched_intervals(coverage), [f for f, _, _ in swaps],
                            S.manual_ranges(events), fps)
    machine = np.zeros(n, np.int8)
    machine[danger["danger"]] = 1
    machine[danger["high_frames"]] = 2
    s = 0 if start is None else max(0, int(start) - first)
    runs, high = danger["runs"], danger["high"]
    keep = np.flatnonzero(runs[:, 1] > s) if len(runs) else np.zeros(0, int)
    runs, high = runs[keep], high[keep]
    where = {"red": 0, "orange": 0, "safe": 0}
    hit_runs: set[int] = set()
    swap_at = [f - first for f, _, _ in swaps if f - first >= s]
    for t in swap_at:
        idx = np.flatnonzero((runs[:, 0] <= t) & (t < runs[:, 1])) if len(runs) else []
        if len(idx):
            hit_runs.add(int(idx[0]))
            where["red" if high[idx[0]] else "orange"] += 1
        else:
            where["safe"] += 1
    manual = [m for m in S.manual_ranges(events) if m[1] - first >= s]
    pose = [p for p in S.pose_ranges(events) if p[1] - first >= s]
    marked = np.zeros(n, bool)
    for a, b in manual:
        lo, hi = max(s, a - first), min(n, b - first + 1)
        if hi > lo:
            marked[lo:hi] = True
    n_marked = int(marked.sum())
    row = {"day": _day(trial), "experiment": trial.experiment, "fps": fps,
           "analysed_s": round((n - s) / fps, 1),
           "red_runs": int(high.sum()), "orange_runs": int((~high).sum()),
           **colour_overlap(machine, status),
           "x_s": round(n_marked / fps, 1),
           "x_in_red_pct": _pct(int((marked & (machine == 2)).sum()), n_marked),
           "x_in_orange_pct": _pct(int((marked & (machine == 1)).sum()), n_marked),
           "x_in_safe_pct": _pct(int((marked & (machine == 0)).sum()), n_marked),
           "swaps": len(swap_at), "swaps_in_red": where["red"], "swaps_in_orange": where["orange"],
           "swaps_in_safe": where["safe"], "x_ranges": len(manual), "pose_bad_ranges": len(pose),
           "red_runs_with_swap": sum(1 for i in hit_runs if high[i]),
           "orange_runs_with_swap": sum(1 for i in hit_runs if not high[i])}
    return row, frames, machine, status


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.1f}%"


def machine_figure(records: list[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]) -> Figure:
    """Machine colours on the upper row, the review on the lower row, one pair per recording."""
    machine_colors = {0: SAFE, 1: ORANGE, 2: RED}
    n = len(records)
    fig = Figure(figsize=(22, 1.45 * n + 2.6), layout="constrained")
    grid = fig.add_gridspec(n + 1, 2, width_ratios=(1.0, 0.24), height_ratios=[1.0] * n + [0.32])
    axes = []
    longest = 0.0
    for i, (row, frames, machine, status) in enumerate(records):
        ax = fig.add_subplot(grid[i, 0], sharex=axes[0] if axes else None)
        side = fig.add_subplot(grid[i, 1])
        axes.append(ax)
        human = np.full(len(status), "unwatched", dtype=object)
        human[status == S.NOT_ANALYSED] = "before"
        human[np.isin(status, (S.WATCHED_SAFE, S.WATCHED_DANGER))] = "kept"
        human[status == S.UNUSABLE] = "discarded"
        trial_fps = row["fps"]
        for start, width, code in _blocks(frames, machine, trial_fps):
            ax.barh(1, width, left=start, height=0.72, color=machine_colors[int(code)], linewidth=0, zorder=2)
        for start, width, code in _blocks(frames, human, trial_fps):
            ax.barh(0, width, left=start, height=0.72, color=REVIEW_COLORS[code], linewidth=0, zorder=2)
        analysed = np.flatnonzero(status != S.NOT_ANALYSED)
        if len(analysed):
            ax.axvline(frames[analysed[0]] / trial_fps / 60, color="#666666", lw=0.7, zorder=4)
        longest = max(longest, float(frames.max()) / trial_fps / 60)
        ax.set_yticks([1, 0], ["machine", "reviewer"])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, 1.7)
        side.set_axis_off()
        side.text(0.0, 0.98, row["day"], transform=side.transAxes, fontsize=12, fontweight="bold",
                  va="top", ha="left", family="monospace")
        side.text(0.0, 0.78,
                  "of discarded time\n"
                  f"  red {_fmt(row['discarded_in_red_pct'])}\n"
                  f"  orange {_fmt(row['discarded_in_orange_pct'])}\n"
                  f"  safe {_fmt(row['discarded_in_safe_pct'])}\n"
                  f"runs entirely red {row['discarded_runs_all_red']}/{row['discarded_runs']}\n"
                  f"of red time, discarded {_fmt(row['red_discarded_pct'])}\n"
                  f"of X marks, red {_fmt(row['x_in_red_pct'])}",
                  transform=side.transAxes, fontsize=8, va="top", ha="left", family="monospace")
    axes[-1].set_xlim(0, longest * 1.01)
    axes[-1].set_xlabel("time (min)")
    note = fig.add_subplot(grid[-1, :])
    note.set_axis_off()
    note.text(0.0, 0.5,
              "Upper row: machine. Lower row: reviewer. The line is the analysis start; "
              "the percentages are of the time after it.\n"
              "Red is a whole danger run whose closest approach was under 8 mm, "
              "so frames farther than 8 mm inside that run are red too.\n"
              "Orange is other danger: under 15 mm, unreliable 3D, or a safe gap shorter than 1 s, "
              "and the run never came under 8 mm.",
              transform=note.transAxes, fontsize=8, color="#333333", ha="left", va="center")
    handles = [Patch(facecolor=SAFE, edgecolor="none", label="machine safe"),
               Patch(facecolor=ORANGE, edgecolor="none", label="machine orange"),
               Patch(facecolor=RED, edgecolor="none", label="machine red")]
    handles += [Patch(facecolor=REVIEW_COLORS[c], edgecolor="none", label=REVIEW_LABELS[c])
                for c in ("kept", "discarded", "unwatched", "before")]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Machine judgment above, review below")
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Review overview of every reviewed recording of the given groups.")
    parser.add_argument("--group", action="append", default=[], help="Group key, repeatable: --group K_1 --group P_1")
    parser.add_argument("--out", type=Path, help="Output root. Default: <runs_dir>/behavior_features.")
    parser.add_argument("--paths-config", type=Path)
    args = parser.parse_args(argv)
    if not args.group:
        parser.error("give at least one --group")
    _config, review, out = resolve_roots(args.paths_config)
    out = args.out or out
    search = default_search(None)
    names: list[str] = []
    for group in args.group:
        ready, skipped = group_status(review, parse_group(group))
        names += ready
        for line in skipped:
            print("skipped", line)
    trials = [ensure_trial(review, out, name, search) for name in names]
    trials.sort(key=_day)
    folder = out / "overview"
    save_figure(timeline_overview(trials), folder / "review_timeline.png")
    by_group: dict[str, list[Trial]] = {}
    for trial in trials:
        by_group.setdefault("_".join(_day(trial).split("_")[:2]), []).append(trial)
    pooled = {name: pool(items, name) for name, items in by_group.items()}
    save_figure(aligned_compare(pooled), folder / "aligned_K_vs_P.png")
    records = [compare_recording(t) for t in trials]
    table = pd.DataFrame([row for row, _frames, _machine, _status in records]).drop(columns="fps")
    table.to_csv(folder / "machine_vs_review.csv", index=False)
    save_figure(machine_figure(records), folder / "machine_vs_review.png")
    show = ["day", "discarded_in_red_pct", "discarded_in_orange_pct", "discarded_in_safe_pct",
            "discarded_runs_all_red", "discarded_runs", "red_discarded_pct", "orange_discarded_pct",
            "safe_discarded_pct", "x_in_red_pct", "x_in_orange_pct", "x_in_safe_pct"]
    print(table[show].to_string(index=False))
    print(f"wrote {folder}")


if __name__ == "__main__":
    main()
