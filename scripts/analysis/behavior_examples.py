"""Typical and atypical examples of each behaviour, for showing to someone who knows the behaviour.

One page per behaviour. Each column is one BORIS bout: distance, actor speed and heading
to the target from 5 s before the key to 6 s after it. The first columns are the bouts
closest to the behaviour's own median, the last the farthest, so the page shows both what
the usual one looks like and where the label gets stretched. Gaps are frames the review
discarded.

::

    .venv\\Scripts\\python.exe -m scripts.analysis.behavior_examples --group K_1 --group P_1

Typicality is a distance in robust units (median absolute deviation) over the bout medians
of distance, actor speed and heading, pooled over every day given. It says how unlike the
other bouts of the same label a bout is; it does not say the label is wrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from scripts.analysis import behavior_features as F
from scripts.analysis.behavior_dataset import Trial, ensure_trial, group_status, parse_group, resolve_roots
from scripts.analysis.behavior_report import _color, save_figure
from scripts.analysis.review_overview import _day
from scripts.review.behaviors import default_search

SCORE_FEATURES = ("dist", "speed_actor", "heading")
SHOW = (("dist", "distance (mm)"), ("speed_actor", "actor speed (mm/s)"), ("heading", "heading to target (deg)"))
PER_SIDE = 3
MIN_USABLE = 0.8
WINDOW = (-5.0, 6.0)


def _score(group: pd.DataFrame) -> pd.Series:
    z = []
    for column in SCORE_FEATURES:
        values = group[column].to_numpy(float)
        if not np.isfinite(values).sum() >= 3:
            continue
        mid = np.nanmedian(values)
        spread = np.nanmedian(np.abs(values - mid)) * 1.4826
        if not spread > 0:
            spread = float(np.nanstd(values)) or 1.0
        z.append(np.abs(values - mid) / spread)
    if not z:
        return pd.Series(np.nan, index=group.index)
    return pd.Series(np.nanmean(np.vstack(z), axis=0), index=group.index)


def choose(trials: list[Trial]) -> pd.DataFrame:
    """Typical and atypical bouts of every behaviour, with the day they came from."""
    bouts = pd.concat([t.bouts.assign(experiment=t.experiment, day=_day(t), fps=t.fps) for t in trials],
                      ignore_index=True)
    bouts = bouts[bouts["directed"].astype(bool) & (bouts["usable_frac"] >= MIN_USABLE)]
    rows = []
    for behavior in F.FOCUS_BEHAVIORS:
        group = bouts[bouts["behavior"] == behavior].copy()
        if group.empty:
            continue
        group["score"] = _score(group)
        group = group.sort_values("score", kind="stable", na_position="last")
        take = min(PER_SIDE, len(group))
        typical = group.head(take).assign(kind="typical")
        rest = group.iloc[take:]
        atypical = rest.tail(min(PER_SIDE, len(rest))).iloc[::-1].assign(kind="atypical")
        rows += [typical, atypical]
    columns = ["behavior", "kind", "day", "experiment", "who", "start", "seconds", "usable_frac", "score", "fps",
               "dist", "speed_actor", "heading"]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def example_figure(behavior: str, chosen: pd.DataFrame, curves: pd.DataFrame) -> Figure:
    picks = chosen[chosen["behavior"] == behavior].reset_index(drop=True)
    mine = curves[curves["behavior"] == behavior]
    shown = [(f, label) for f, label in SHOW if np.isfinite(mine.loc[mine["feature"] == f, "value"]).any()]
    if behavior == "spawn":
        shown = [(f, "female distance to nearer male (mm)" if f == "dist" else "female speed (mm/s)")
                 for f, _ in shown]
    fig = Figure(figsize=(max(2.3 * len(picks), 8.0) + 1.2, 2.5 * len(shown) + 1.6), layout="constrained")
    axes = fig.subplots(len(shown), len(picks), sharex=True, squeeze=False)
    for col, pick in enumerate(picks.itertuples(index=False)):
        part = curves[(curves["behavior"] == behavior) & (curves["experiment"] == pick.experiment)
                      & (curves["start"] == pick.start)]
        for row, (feature, label) in enumerate(shown):
            ax = axes[row, col]
            one = part[part["feature"] == feature].sort_values("t")
            one = one[(one["t"] >= WINDOW[0]) & (one["t"] <= WINDOW[1])]
            ax.axvspan(0, pick.seconds, color=_color(behavior), alpha=0.18, linewidth=0)
            ax.plot(one["t"], one["value"], color=_color(behavior), lw=1.1)
            ax.axvline(0, color="black", lw=0.6)
            ax.set_xlim(*WINDOW)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == len(shown) - 1:
                ax.set_xlabel("seconds from key", fontsize=8)
        minute = pick.start / pick.fps / 60
        axes[0, col].set_title(f"{pick.kind}\n{pick.day} {pick.who}\n{minute:.1f} min, {pick.seconds:.1f} s, "
                               f"usable {100 * pick.usable_frac:.0f}%", fontsize=7.5)
    fig.suptitle(f"{F.BEHAVIOR_LABEL.get(behavior, behavior)}: most and least typical bouts\n"
                 "shaded = labelled bout; gaps = discarded frames\n"
                 "typicality is relative to this label's other bouts", fontsize=10)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Typical and atypical bouts of each behaviour.")
    parser.add_argument("--group", action="append", default=[], help="Group key, repeatable.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--paths-config", type=Path)
    args = parser.parse_args(argv)
    if not args.group:
        parser.error("give at least one --group")
    _config, review, out = resolve_roots(args.paths_config)
    out = args.out or out
    search = default_search(None)
    names: list[str] = []
    for group in args.group:
        names += group_status(review, parse_group(group))[0]
    trials = sorted((ensure_trial(review, out, name, search) for name in names), key=_day)
    chosen = choose(trials)
    curves = pd.concat([t.aligned.assign(experiment=t.experiment) for t in trials], ignore_index=True)
    folder = out / "overview"
    folder.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(folder / "examples.csv", index=False)
    for behavior in chosen["behavior"].unique():
        save_figure(example_figure(behavior, chosen, curves), folder / f"examples_{behavior.replace(' ', '_')}.png")
    print(chosen.groupby(["behavior", "kind"]).size().to_string())
    print(f"wrote {folder}")


if __name__ == "__main__":
    main()
