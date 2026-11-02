"""Figures for the behaviour-feature tables, and the command that builds them.

One experiment and one group use the same figures. A group pools the days' bouts; the
days stay repeated measures of the same fish, and the figures say so. ``follow`` is
not drawn.

::

    .venv\\Scripts\\python.exe -m scripts.analysis.behavior_report --group P_1
    .venv\\Scripts\\python.exe -m scripts.analysis.behavior_report --group K_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from scripts.analysis import behavior_features as F
from scripts.analysis.behavior_dataset import (Trial, ensure_trial, group_status, parse_group, pool,
                                               resolve_roots)
from scripts.review.behaviors import default_search

COLORS = {"quick circle": "#2c7fb8", "incomplete qc": "#6baed6", "wrap": "#6a51a3", "spawn": "#222222",
          "attack": "#d7301f", "attempted att": "#fc8d59", "chase": "#e6550d", "parallel circle": "#2ca25f",
          "approach": "#74a9cf", "sneak": "#969696", "other": "#bdbdbd"}
LABELS = {"dist": "distance (mm)", "speed_actor": "actor speed (mm/s)", "speed_target": "target speed (mm/s)",
          "closing": "closing speed (mm/s)", "heading": "heading to target (deg)",
          "orbit": "orbit (deg/s)", "forward": "position in front of target (mm)",
          "lateral": "position to target's right (mm)", "mouth": "mouth-to-body (mm)",
          "bend": "body bend (deg)"}
ALIGNED_BEHAVIORS = F.FOCUS_BEHAVIORS
ALIGNED_FEATURES = ("dist", "heading", "orbit", "speed_actor")
DISTANCE_COLORS = {"dist_F_M1": "#1b9e77", "dist_F_M2": "#d95f02"}
REVIEW_COLORS = {"kept": "#3c9d4e", "discarded": "#d64545", "unwatched": "#9a9a9a", "before": "#f0f0f0"}
REVIEW_LABELS = {"kept": "kept", "discarded": "discarded", "unwatched": "unwatched",
                 "before": "before analysis"}


def _color(behavior: str) -> str:
    return COLORS.get(behavior, "#888888")


def _title(trial: Trial, text: str) -> str:
    days = trial.provenance.get("days")
    if days:
        head = f"{trial.experiment} ({len(days)} days, same fish)"
    else:
        head = trial.experiment
    if trial.review_open or trial.provenance.get("review_open"):
        head += " — review still open"
    return f"{head}\n{text}"


def save_figure(fig: Figure, path: Path, dpi: int = 160) -> None:
    """Write ``path`` as PNG and as PDF. ``path`` may already carry either suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".pdf"))


def _blocks(frames: np.ndarray, codes: np.ndarray, fps: float):
    """Consecutive bins of one review class, as (start minute, width minute, code)."""
    step = float(np.median(np.diff(frames))) if len(frames) > 1 else F.TRACE_BIN
    width = step / fps / 60
    out, i = [], 0
    while i < len(codes):
        j = i + 1
        while j < len(codes) and codes[j] == codes[i]:
            j += 1
        start = frames[i] / fps / 60
        out.append((start, (frames[j - 1] - frames[i]) / fps / 60 + width, codes[i]))
        i = j
    return out


def timeline_figure(trial: Trial) -> Figure:
    """One recording: review status, the six behaviours, and the female's distance to each male.

    M1 sits on the upper half of the behaviour row and M2 on the lower half. A hatch means
    the other fish is the male, so a male-male attack is distinct from one directed at her.
    Events shorter than 0.3 s are ticks; a bar of that duration would disappear on this axis.
    """
    fig = Figure(figsize=(11, 5.4), layout="constrained")
    status_ax, bout_ax, dist_ax = fig.subplots(3, 1, sharex=True, height_ratios=[0.28, 0.72, 1.35])
    fps = trial.fps
    traces = trial.traces
    frames = traces["frame"].to_numpy(float)
    if "review" in traces:
        for start, width, code in _blocks(frames, traces["review"].to_numpy(), fps):
            status_ax.barh(0, width, left=start, height=1, color=REVIEW_COLORS[code], linewidth=0)
    status_ax.set_ylim(-0.5, 0.5)
    status_ax.set_yticks([])
    pct = trial.provenance.get("review_pct") or {}
    handles = [Patch(facecolor=REVIEW_COLORS[code], edgecolor="none",
                     label=f"{REVIEW_LABELS[code]} {pct.get(key, 0):.1f}%")
               for code, key in (("kept", "kept"), ("discarded", "discarded"), ("unwatched", "unwatched"))]
    handles.append(Patch(facecolor=REVIEW_COLORS["before"], edgecolor="none",
                         label=REVIEW_LABELS["before"]))
    status_ax.set_ylabel("review")

    for bout in trial.bouts.itertuples(index=False):
        male_target = bout.actor in ("M1", "M2") and bout.target in ("M1", "M2")
        y = {"M1": 1.0, "M2": 0.0}.get(bout.actor, 0.5)
        x = bout.start / fps / 60
        color = _color(bout.behavior)
        if bout.seconds < 0.3:
            bout_ax.plot(x, y, marker="x" if male_target else "|", color=color, ms=9, mew=1.6)
        else:
            width = max((bout.end - bout.start + 1) / fps / 60, 0.015)
            bout_ax.barh(y, width, left=x, height=0.42, color=color, align="center",
                         hatch="///" if male_target else None, edgecolor="black", linewidth=0.2)
    bout_ax.set_yticks([0, 0.5, 1], ["M2", "female", "M1"])
    bout_ax.set_ylim(-0.45, 1.55)
    bout_ax.set_ylabel("behavior")
    seen = list(dict.fromkeys(trial.bouts["behavior"]))
    behavior_handles = [Patch(facecolor=_color(b), edgecolor="none", label=F.BEHAVIOR_LABEL.get(b, b)) for b in seen]
    behavior_handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                                  label="male-male: aimed at the other male (not the female)"))
    fig.legend(handles=handles + behavior_handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False)

    minutes = frames / fps / 60
    for column, label in (("dist_F_M1", "female–M1"), ("dist_F_M2", "female–M2")):
        dist_ax.plot(minutes, traces[column], color=DISTANCE_COLORS[column], lw=0.7, label=label)
    dist_ax.set_ylabel("distance (mm)")
    dist_ax.set_xlabel("time (min)")
    dist_ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle(_title(trial, "percentages are of the analysed period; events shorter than 0.3 s are ticks"))
    return fig


def features_figure(trial: Trial) -> Figure:
    """Bout medians of each feature. One point is one BORIS bout."""
    fig = Figure(figsize=(12, 7.2), layout="constrained")
    axes = fig.subplots(2, 5)
    bouts = trial.bouts
    behaviors = list(dict.fromkeys(bouts["behavior"]))
    rng = np.random.default_rng(0)
    for ax, feature in zip(axes.ravel(), F.BOUT_COLUMNS):
        data, labels, colors = [], [], []
        for behavior in behaviors:
            values = bouts.loc[bouts["behavior"] == behavior, feature].to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            data.append(values)
            labels.append(f"{F.BEHAVIOR_LABEL.get(behavior, behavior)} ({len(values)})")
            colors.append(_color(behavior))
        if data:
            ax.boxplot(data, showfliers=False, medianprops={"color": "black"})
            for i, values in enumerate(data):
                jitter = rng.uniform(-0.12, 0.12, len(values))
                ax.scatter(np.full(len(values), i + 1) + jitter, values, s=10, color=colors[i], zorder=3)
            ax.set_xticks(range(1, len(labels) + 1), labels, rotation=55, ha="right", fontsize=6)
        ax.set_title(LABELS[feature], fontsize=9)
    fig.suptitle(_title(trial, "one point is one bout; unusable frames are left out of its median"))
    return fig


def aligned_figure(trial: Trial) -> Figure:
    """Mean ± SEM aligned on the BORIS start. The line is at the key, not at the movement."""
    summary = F.summarise_aligned(trial.aligned)
    present = set(summary["behavior"]) if len(summary) else set()
    behaviors = [b for b in ALIGNED_BEHAVIORS if b in present]
    fig = Figure(figsize=(13.5, 7.5), layout="constrained")
    if not behaviors:
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no directed bouts", ha="center")
        return fig
    axes = fig.subplots(len(ALIGNED_FEATURES), len(behaviors), sharex=True, squeeze=False)
    for col, behavior in enumerate(behaviors):
        for row, feature in enumerate(ALIGNED_FEATURES):
            ax = axes[row, col]
            part = summary[(summary["behavior"] == behavior) & (summary["feature"] == feature)].sort_values("t")
            if len(part):
                mean, sem = part["mean"].to_numpy(float), part["sem"].to_numpy(float)
                ax.plot(part["t"], mean, color=_color(behavior), lw=1.2)
                ax.fill_between(part["t"], mean - sem, mean + sem, color=_color(behavior), alpha=0.25)
            ax.axvline(0, color="black", lw=0.5)
            if row == 0:
                n = int(part["n"].max()) if len(part) else 0
                ax.set_title(f"{F.BEHAVIOR_LABEL.get(behavior, behavior)} (n={n})", fontsize=9)
            if col == 0:
                ax.set_ylabel(LABELS[feature], fontsize=8)
            if row == len(ALIGNED_FEATURES) - 1:
                ax.set_xlabel("seconds from BORIS start")
    fig.suptitle(_title(trial, "line at the key; band is the SEM across bouts"))
    return fig


GROUP_COLORS = ("#2c7fb8", "#d95f02", "#4daf4a")


def aligned_compare(groups: dict[str, Trial]) -> Figure:
    """Mean +- SEM aligned on the BORIS start, one line per group.

    Groups are different fish and stay separate lines; pooling them would let the group
    with more bouts decide the mean.
    """
    summaries = {name: F.summarise_aligned(trial.aligned) for name, trial in groups.items()}
    present = set().union(*(set(s["behavior"]) for s in summaries.values() if len(s)))
    behaviors = [b for b in ALIGNED_BEHAVIORS if b in present]
    fig = Figure(figsize=(14.5, 8.0), layout="constrained")
    axes = fig.subplots(len(ALIGNED_FEATURES), len(behaviors), sharex=True, squeeze=False)
    for col, behavior in enumerate(behaviors):
        counts = []
        for row, feature in enumerate(ALIGNED_FEATURES):
            ax = axes[row, col]
            for color, (name, summary) in zip(GROUP_COLORS, summaries.items()):
                part = summary[(summary["behavior"] == behavior) & (summary["feature"] == feature)].sort_values("t")
                if not len(part):
                    continue
                mean, sem = part["mean"].to_numpy(float), part["sem"].to_numpy(float)
                ax.plot(part["t"], mean, color=color, lw=1.2, label=name)
                ax.fill_between(part["t"], mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)
                if row == 0:
                    counts.append(f"{name} n={int(part['n'].max())}")
            ax.axvline(0, color="black", lw=0.5)
            if col == 0:
                ax.set_ylabel(LABELS[feature], fontsize=8)
            if row == len(ALIGNED_FEATURES) - 1:
                ax.set_xlabel("seconds from BORIS start", fontsize=8)
        axes[0, col].set_title(f"{F.BEHAVIOR_LABEL.get(behavior, behavior)}\n" + ", ".join(counts), fontsize=8.5)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Aligned on the BORIS start, one line per group\n"
                 "band is the SEM across bouts; each group is one set of fish seen on three days")
    return fig


def competition_figure(trial: Trial) -> Figure:
    """Female distance change after each male-male bout, against who moved off."""
    fig = Figure(figsize=(8.0, 6.4), layout="constrained")
    ax = fig.add_subplot(111)
    races = trial.competition
    defined = races[races["retreater"].astype(str).isin(F.NAMES)] if len(races) else races
    groups = defined.groupby("behavior") if len(defined) else []
    for behavior, group in groups:
        ax.scatter(group["delta_stayer"], group["delta_retreater"], s=28, color=_color(behavior),
                   label=f"{F.BEHAVIOR_LABEL.get(behavior, behavior)} ({len(group)})")
    if len(defined):
        lo = float(np.nanmin(defined[["delta_stayer", "delta_retreater"]].to_numpy()))
        hi = float(np.nanmax(defined[["delta_stayer", "delta_retreater"]].to_numpy()))
        pad = max(5.0, 0.1 * (hi - lo))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=0.6)
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel("change in her distance to the male that stayed (mm)")
    ax.set_ylabel("change in her distance to the male that left (mm)")
    if len(defined):
        ax.legend(fontsize=8, frameon=False)
    n = len(defined)
    fig.suptitle(_title(trial, f"n={n} male-male bouts; negative means she ended closer"))
    return fig


def write_figures(trial: Trial, folder: Path, timeline: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if timeline and len(trial.traces):
        save_figure(timeline_figure(trial), folder / "timeline.png")
    save_figure(features_figure(trial), folder / "features.png")
    save_figure(aligned_figure(trial), folder / "aligned.png")
    save_figure(competition_figure(trial), folder / "competition.png")
    trial.separability.to_csv(folder / "separability.csv", index=False)
    trial.competition.to_csv(folder / "competition.csv", index=False)
    trial.bouts.to_csv(folder / "bouts.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build behaviour-feature tables and figures for one group or trial.")
    parser.add_argument("--group", help="Group key, e.g. P_1 or K_1. Every reviewed day of it is built.")
    parser.add_argument("--experiment", action="append", default=[], help="One experiment folder name. Repeatable.")
    parser.add_argument("--package", help="Package directory name, when an experiment has more than one ledger.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore saved tables and read the review packages again.")
    parser.add_argument("--out", type=Path, help="Output root. Default: <runs_dir>/behavior_features.")
    parser.add_argument("--paths-config", type=Path, help="paths toml. Default: paths.<user>.toml, else paths.local.toml.")
    args = parser.parse_args(argv)
    if not args.group and not args.experiment:
        parser.error("give --group or --experiment")
    _config, review, out = resolve_roots(args.paths_config)
    if args.out:
        out = args.out
    search = default_search(None)
    names = list(args.experiment)
    skipped: list[str] = []
    if args.group:
        group = parse_group(args.group)
        ready, skipped = group_status(review, group)
        names = ready + [n for n in names if n not in ready]
    trials = [ensure_trial(review, out, name, search, args.package, args.rebuild) for name in names]
    for trial in trials:
        write_figures(trial, out / trial.experiment, timeline=True)
        print(f"{trial.experiment}: {len(trial.bouts)} bouts, usable {trial.provenance.get('usable_pct')}%, "
              f"review open={trial.review_open}")
    if args.group and trials:
        group = parse_group(args.group)
        pooled = pool(trials, group)
        write_figures(pooled, out / group, timeline=False)
        (out / group / "days.txt").write_text("\n".join(pooled.provenance["days"] + skipped) + "\n", encoding="utf-8")
        print(f"{group}: pooled {len(trials)} days, skipped {len(skipped)}")
        for line in skipped:
            print(f"  skipped {line}")


if __name__ == "__main__":
    main()
