"""All reviewed recordings on one page, and what the machine flagged against what the reviewer did.

Two figures, each written as PNG and PDF, plus the numbers behind the second as a csv::

    .venv\\Scripts\\python.exe -m scripts.analysis.review_overview --group K_1 --group P_1

* ``review_timeline``: one row per recording on a shared time axis. Review status
  (kept / discarded / unwatched) with its percentage, and the six behaviours above it.
* ``machine_vs_review``: the machine's danger runs (high risk = closer than 8 mm, the
  GUI's red; other danger = orange) against the reviewer's identity swaps, hand-marked
  X ranges, and where the swaps fell. Wrong-pose marks (B) are counted in the csv only.

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
from scripts.review.ledger import EVENT_COLUMNS, analysis_start, read_csv, swap_list

RED, ORANGE, SAFE = "#d64545", "#f0a030", "#cfd8dc"


def _day(trial: Trial) -> str:
    """``20260112_P_1_1_FCMC`` -> ``P_1_1``."""
    parts = trial.experiment.split("_")
    return "_".join(parts[1:4]) if len(parts) >= 4 and parts[0].isdigit() else trial.experiment


def timeline_overview(trials: list[Trial]) -> Figure:
    fig = Figure(figsize=(22, 1.25 * len(trials) + 1.6), layout="constrained")
    axes = fig.subplots(len(trials), 1, sharex=True, squeeze=False)[:, 0]
    longest = 0.0
    seen: list[str] = []
    for ax, trial in zip(axes, trials):
        fps = trial.fps
        frames = trial.traces["frame"].to_numpy(float)
        longest = max(longest, float(frames.max()) / fps / 60)
        for start, width, code in _blocks(frames, trial.traces["review"].to_numpy(), fps):
            ax.barh(0, width, left=start, height=0.7, color=REVIEW_COLORS[code], linewidth=0)
        last_end: dict[str, float] = {}
        level: dict[str, int] = {}
        for bout in trial.bouts.sort_values("start").itertuples(index=False):
            male_target = bout.actor in ("M1", "M2") and bout.target in ("M1", "M2")
            y = {"M1": 3.0, "M2": 1.0}.get(bout.actor, 2.0)
            x = bout.start / fps / 60
            # bouts touching or overlapping the previous one in the same row go to a second level
            crowded = x <= last_end.get(bout.actor, -1e9) + 0.08
            level[bout.actor] = 1 - level.get(bout.actor, 0) if crowded else 0
            last_end[bout.actor] = (max(bout.end / fps / 60, last_end.get(bout.actor, -1e9))
                                    if crowded else bout.end / fps / 60)
            y += 0.18 if level[bout.actor] else -0.18
            if bout.behavior not in seen:
                seen.append(bout.behavior)
            if bout.seconds < 0.3:
                ax.plot(x, y, marker="x" if male_target else "|", color=_color(bout.behavior), ms=6, mew=1.3)
            else:
                width = max((bout.end - bout.start + 1) / fps / 60, 0.02)
                ax.barh(y, width, left=x, height=0.32, color=_color(bout.behavior),
                        hatch="///" if male_target else None, edgecolor="white", linewidth=0.5)
                ax.plot([x, x], [y - 0.16, y + 0.16], color="black", lw=0.8, solid_capstyle="butt")
        ax.set_yticks([0, 1, 2, 3], ["review", "M2", "female", "M1"], fontsize=7)
        ax.set_ylim(-0.6, 3.6)
        pct = trial.provenance.get("review_pct") or {}
        ax.text(1.005, 0.5, f"{_day(trial)}\nkept {pct.get('kept', 0):.1f}%\n"
                f"discarded {pct.get('discarded', 0):.1f}%\nunwatched {pct.get('unwatched', 0):.1f}%",
                transform=ax.transAxes, fontsize=7.5, va="center")
    axes[-1].set_xlim(0, longest * 1.01)
    axes[-1].set_xlabel("time (min)")
    handles = [Patch(facecolor=REVIEW_COLORS[c], edgecolor="none", label=REVIEW_LABELS[c])
               for c in ("kept", "discarded", "unwatched", "before")]
    handles += [Patch(facecolor=_color(b), edgecolor="none", label=F.BEHAVIOR_LABEL.get(b, b))
                for b in F.FOCUS_BEHAVIORS if b in seen]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", label="male-male: aimed at the other male (not the female)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Review status and behaviour, every reviewed recording\n"
                 "percentages are of the analysed period; events shorter than 0.3 s are ticks; "
                 "days of one group are the same fish")
    return fig


def machine_vs_review(trial: Trial) -> dict:
    """Machine danger runs against the reviewer's marks for one recording, after the analysis start."""
    package = trial.package
    with np.load(package / "review_data.npz") as z:
        frames = np.asarray(z["frames"], int)
        fish_ids = [str(f) for f in z["fish_ids"]]
        xyz, present, interp = (np.asarray(z[k]) for k in ("xyz", "view_present", "is_interpolated"))
    events = read_csv(package / "id_pass" / "human_events.csv", EVENT_COLUMNS)
    fps, first, n = trial.fps, int(frames[0]), len(frames)
    s = max(0, int(analysis_start(events)) - first)
    danger = S.danger_map(xyz.astype(float), present, interp, fps)
    runs, high = danger["runs"], danger["high"]
    keep = np.flatnonzero(runs[:, 1] > s) if len(runs) else np.zeros(0, int)
    runs, high = runs[keep], high[keep]
    clipped = np.maximum(runs, s) if len(runs) else runs
    length = (clipped[:, 1] - clipped[:, 0]) if len(runs) else np.zeros(0)
    analysed = max(1, n - s)
    swaps = [f - first for f, _, _ in swap_list(events, fish_ids) if f - first >= s]
    where = {"red": 0, "orange": 0, "safe": 0}
    hit_runs: set[int] = set()
    for t in swaps:
        idx = np.flatnonzero((runs[:, 0] <= t) & (t < runs[:, 1])) if len(runs) else []
        if len(idx):
            hit_runs.add(int(idx[0]))
            where["red" if high[idx[0]] else "orange"] += 1
        else:
            where["safe"] += 1
    manual = [m for m in S.manual_ranges(events) if m[1] - first >= s]
    pose = [p for p in S.pose_ranges(events) if p[1] - first >= s]
    red_hit = sum(1 for i in hit_runs if high[i])
    orange_hit = len(hit_runs) - red_hit
    return {"day": _day(trial), "experiment": trial.experiment,
            "analysed_s": round(analysed / fps, 1),
            "red_runs": int(high.sum()), "orange_runs": int((~high).sum()),
            "red_pct": round(100 * float(length[high].sum()) / analysed, 2),
            "orange_pct": round(100 * float(length[~high].sum()) / analysed, 2),
            "swaps": len(swaps), "swaps_in_red": where["red"], "swaps_in_orange": where["orange"],
            "swaps_in_safe": where["safe"], "x_ranges": len(manual), "pose_bad_ranges": len(pose),
            "red_runs_with_swap": red_hit, "orange_runs_with_swap": orange_hit,
            "discarded_pct": trial.provenance["review_pct"]["discarded"]}


def machine_figure(table: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(13.5, 0.6 * len(table) + 3.4), layout="constrained")
    a, b, c = fig.subplots(1, 3, sharey=True)
    y = np.arange(len(table))[::-1]
    a.barh(y, table["red_pct"], color=RED, label="high risk (< 8 mm), red")
    a.barh(y, table["orange_pct"], left=table["red_pct"], color=ORANGE, label="other danger, orange")
    for yi, r in zip(y, table.itertuples()):
        a.text(r.red_pct + r.orange_pct + 0.3, yi, f"{r.red_runs} red / {r.orange_runs} orange runs", va="center", fontsize=7.5)
    a.set_xlim(0, float((table["red_pct"] + table["orange_pct"]).max()) * 1.6)
    a.set_xlabel("share of the analysed time (%)")
    a.set_title("Machine: danger runs", fontsize=10)
    a.legend(fontsize=7.5, frameon=False, loc="lower right")
    a.set_yticks(y, table["day"])
    height = 0.36
    for k, (col, color, label) in enumerate((("swaps", "#3b6fb6", "identity swaps"),
                                             ("x_ranges", "#7a4aa0", "hand-marked X ranges"))):
        b.barh(y + (0.5 - k) * height, table[col], height=height, color=color, label=label)
    b.set_xlabel("count")
    b.set_title("Reviewer: marks", fontsize=10)
    b.legend(fontsize=7.5, frameon=False)
    left = np.zeros(len(table))
    for col, color, label in (("swaps_in_red", RED, "in a red run"), ("swaps_in_orange", ORANGE, "in an orange run"),
                              ("swaps_in_safe", SAFE, "outside danger")):
        c.barh(y, table[col], left=left, color=color, label=label)
        left = left + table[col].to_numpy()
    for yi, r in zip(y, table.itertuples()):
        c.text(r.swaps + 0.2, yi, f"red runs holding a swap: {r.red_runs_with_swap}/{r.red_runs}", va="center", fontsize=7.5)
    c.set_xlim(0, float(table["swaps"].max()) * 2.2 + 1)
    c.set_xlabel("identity swaps")
    c.set_title("Where the swaps fell", fontsize=10)
    c.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Machine flags against the reviewer's marks, after the analysis start\n"
                 "red = nearest pair under 8 mm, orange = other danger (under 15 mm or unreliable 3D)")
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
    table = pd.DataFrame([machine_vs_review(t) for t in trials])
    table.to_csv(folder / "machine_vs_review.csv", index=False)
    save_figure(machine_figure(table), folder / "machine_vs_review.png")
    print(table.drop(columns="experiment").to_string(index=False))
    print(f"wrote {folder}")


if __name__ == "__main__":
    main()
