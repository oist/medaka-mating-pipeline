"""One recording against time, one set of rows per fish, with the review and the behaviours on top.

Per recording, two outputs (PNG and PDF each)::

    features_segments     the whole recording cut into six equal segments stacked on one page
    features_pages        the same six segments, one per page, larger (one PDF and one PNG each)

From the top of each segment:

* the review bar: kept, discarded, unwatched;
* the behaviour lane: one row per behaviour, named at the left, every event tagged with who
  does it to whom (``M1→F``: male 1 to the female; ``M2→M1``: male 2 to male 1);
* the female's rows (speed; distance to each male);
* male 1's rows (speed; heading toward the female and toward male 2; orbit around the female;
  direction of his velocity; angle swept around each other fish in the last 1 s; path curvature);
* male 2's rows, the same;
* the distance between the two males.

The colour of a line is the fish it is about: speed and orbit are the fish itself; in the
distance and heading rows the colour is the fish the measure points to. Tinted bands go over
every row: a behaviour in its colour, red where the review discarded the frames, grey where
nobody watched. Lines stop in discarded stretches.

::

    .venv\\Scripts\\python.exe -m scripts.analysis.behavior_timeline --group K_1 --group P_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.analysis import behavior_features as F
from scripts.analysis.behavior_dataset import Trial, ensure_trial, group_status, parse_group, resolve_roots
from scripts.analysis.behavior_report import REVIEW_COLORS, REVIEW_LABELS, _blocks, _color, save_figure
from scripts.analysis.review_overview import _day
from scripts.review.behaviors import default_search

FEMALE, M1, M2, BOTH = "#c2185b", "#1b9e77", "#d95f02", "#555555"
# (axis label, label colour, [(column, line colour), ...], clip);
# clip: pos = from 0, sym = about 0, angle = 0-180.
ROWS = (
    ("Female\nspeed (mm/s)", FEMALE, (("speed_F", FEMALE),), "pos"),
    ("Female\ndistance to Male 1 (green)\nand Male 2 (orange) (mm)", FEMALE,
     (("dist_F_M1", M1), ("dist_F_M2", M2)), "pos"),
    ("Male 1\nspeed (mm/s)", M1, (("speed_M1", M1),), "pos"),
    ("Male 1\nheading toward Female (pink)\nand Male 2 (orange) (deg)", M1,
     (("head_M1_to_F", FEMALE), ("head_M1_to_M2", M2)), "angle"),
    ("Male 1\norbit around Female (deg/s)", M1, (("orbit_M1_around_F", M1),), "sym"),
    ("Male 1\nvelocity direction to Female (pink)\nand Male 2 (orange) (deg)\n0 = toward, 90 = sideways", M1,
     (("vel_angle_M1_to_F", FEMALE), ("vel_angle_M1_to_M2", M2)), "angle"),
    ("Male 1\nangle swept in 1 s around\nFemale (pink) and Male 2 (orange) (deg)", M1,
     (("sweep_M1_around_F", FEMALE), ("sweep_M1_around_M2", M2)), "sweep"),
    ("Male 1\npath curvature (deg/mm)", M1, (("curve_M1", M1),), "curve"),
    ("Male 2\nspeed (mm/s)", M2, (("speed_M2", M2),), "pos"),
    ("Male 2\nheading toward Female (pink)\nand Male 1 (green) (deg)", M2,
     (("head_M2_to_F", FEMALE), ("head_M2_to_M1", M1)), "angle"),
    ("Male 2\norbit around Female (deg/s)", M2, (("orbit_M2_around_F", M2),), "sym"),
    ("Male 2\nvelocity direction to Female (pink)\nand Male 1 (green) (deg)\n0 = toward, 90 = sideways", M2,
     (("vel_angle_M2_to_F", FEMALE), ("vel_angle_M2_to_M1", M1)), "angle"),
    ("Male 2\nangle swept in 1 s around\nFemale (pink) and Male 1 (green) (deg)", M2,
     (("sweep_M2_around_F", FEMALE), ("sweep_M2_around_M1", M1)), "sweep"),
    ("Male 2\npath curvature (deg/mm)", M2, (("curve_M2", M2),), "curve"),
    ("Distance between\nMale 1 and Male 2 (mm)", BOTH, (("dist_M1_M2", BOTH),), "pos"),
)
LANE_ORDER = ("attack", "attempted att", "quick circle", "incomplete qc", "wrap", "spawn")
SEGMENTS = 6
ROW_INCH = 0.62
LANE_RATIO = 4.5


def _limits(trace) -> dict[str, tuple[float, float]]:
    """Axis limits per row from the whole recording, so every segment shares a scale."""
    out = {}
    for label, _color_, series, clip in ROWS:
        values = np.concatenate([trace[c].to_numpy(float) for c, _ in series])
        values = values[np.isfinite(values)]
        if clip == "angle":
            out[label] = (0.0, 180.0)
        elif not len(values):
            out[label] = (0.0, 1.0)
        elif clip == "pos":
            out[label] = (0.0, float(np.percentile(values, 99.5)) * 1.05)
        elif clip == "sweep":
            out[label] = (-400.0, 400.0)
        elif clip == "curve":
            out[label] = (0.0, 4.0)
        else:
            top = float(np.percentile(np.abs(values), 99.5))
            out[label] = (-top, top)
    return out


def _tag(bout) -> str:
    short = {"F": "F", "M1": "M1", "M2": "M2"}
    if bout.actor and bout.target:
        arrow = "↔" if not bout.directed else "→"
        return f"{short.get(bout.actor, bout.actor)}{arrow}{short.get(bout.target, bout.target)}"
    return short.get(bout.actor, bout.actor or "?")


def _minutes(frame: int, fps: float) -> float:
    """Clock time of one frame. A bout runs from its START frame to its STOP frame."""
    return frame / fps / 60


def _bands(ax, trial: Trial, t0: float, t1: float, minimum: float) -> None:
    fps = trial.fps
    frames = trial.traces["frame"].to_numpy(float)
    for start, width, code in _blocks(frames, trial.traces["review"].to_numpy(), fps):
        if code in ("discarded", "unwatched") and start < t1 and start + width > t0:
            ax.axvspan(start, start + max(width, minimum), color=REVIEW_COLORS[code],
                       alpha=0.28 if code == "discarded" else 0.35, linewidth=0, zorder=0)
    ordered = trial.bouts.sort_values(["start", "end"], kind="mergesort")
    for bout in ordered.itertuples(index=False):
        x0, x1 = _minutes(bout.start, fps), _minutes(bout.end, fps)
        if x1 <= x0 or x0 > t1 or x1 < t0:
            continue
        ax.axvspan(x0, x1, color=_color(bout.behavior), alpha=0.18, linewidth=0, zorder=1)


def draw_block(fig, trial: Trial, t0: float, t1: float, limits) -> None:
    """Review bar, behaviour lane and the per-fish rows for minutes ``t0..t1`` on ``fig``."""
    fps = trial.fps
    traces = trial.traces
    minutes = traces["frame"].to_numpy(float) / fps / 60
    keep = (minutes >= t0 - 0.05) & (minutes <= t1 + 0.05)
    minimum = (t1 - t0) * 0.0015
    ratios = [0.3, LANE_RATIO] + [1.0] * len(ROWS)
    axes = fig.subplots(len(ROWS) + 2, 1, sharex=True, height_ratios=ratios)
    status, lane, rows = axes[0], axes[1], axes[2:]

    for start, width, code in _blocks(traces["frame"].to_numpy(float), traces["review"].to_numpy(), fps):
        status.barh(0, width, left=start, height=1, color=REVIEW_COLORS[code], linewidth=0)
    status.set_ylim(-0.5, 0.5)
    status.set_yticks([])
    status.set_ylabel("review", fontsize=8, rotation=0, ha="right", va="center")

    n = len(LANE_ORDER)
    y_of = {b: n - 1 - i for i, b in enumerate(LANE_ORDER)}
    ordered = trial.bouts.sort_values(["start", "end"], kind="mergesort")
    for bout in ordered.itertuples(index=False):
        y = y_of.get(bout.behavior)
        if y is None:
            continue
        x0, x1 = _minutes(bout.start, fps), _minutes(bout.end, fps)
        if x0 > t1 or x1 < t0:
            continue
        color = _color(bout.behavior)
        if x1 > x0:
            lane.barh(y, x1 - x0, left=x0, height=0.62, align="center", color=color, linewidth=0, zorder=2)
        else:
            lane.plot([x0, x0], [y - 0.31, y + 0.31], color=color, lw=1.1, solid_capstyle="butt", zorder=3)
        lane.text(x0, y + 0.34, _tag(bout), fontsize=7, rotation=90, ha="left", va="bottom", color="black")
    lane.set_yticks([y_of[b] for b in LANE_ORDER], [F.BEHAVIOR_LABEL[b] for b in LANE_ORDER], fontsize=8)
    lane.set_ylim(-0.5, n - 0.5)
    for y in range(n - 1):
        lane.axhline(y + 0.5, color="#dddddd", lw=0.5)
    lane.set_ylabel("behaviour\n(tag: who → whom)", fontsize=8, rotation=0, ha="right", va="center")

    for ax, (label, label_color, series, _clip) in zip(rows, ROWS):
        _bands(ax, trial, t0, t1, minimum)
        for column, color in series:
            ax.plot(minutes[keep], traces[column].to_numpy(float)[keep], color=color, lw=0.7, zorder=3)
        ax.set_ylim(*limits[label])
        ax.set_ylabel(label, fontsize=8, color=label_color, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
    rows[-1].set_xlim(t0, t1)
    rows[-1].set_xlabel("time (min)", fontsize=9)


def _legend(fig, trial: Trial) -> None:
    seen = [b for b in LANE_ORDER if b in set(trial.bouts["behavior"])]
    handles = [Patch(facecolor=REVIEW_COLORS[c], edgecolor="none", label=REVIEW_LABELS[c])
               for c in ("kept", "discarded", "unwatched", "before")]
    handles += [Patch(facecolor=_color(b), edgecolor="none", label=F.BEHAVIOR_LABEL[b]) for b in seen]
    handles += [Line2D([], [], color=c, lw=2, label=name)
                for c, name in ((FEMALE, "Female"), (M1, "Male 1"), (M2, "Male 2"),
                                (BOTH, "between the two males"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8.5, frameon=False)


def _title(trial: Trial) -> str:
    pct = trial.provenance.get("review_pct") or {}
    return (f"{trial.experiment}: kept {pct.get('kept', 0):.1f}%, discarded {pct.get('discarded', 0):.1f}%, "
            f"unwatched {pct.get('unwatched', 0):.1f}%\n"
            "line colour = the fish the measure is about or points to; tint over every row = behaviour (its colour), "
            "red = discarded, grey = unwatched; lines stop in discarded stretches")


def _edges(trial: Trial) -> list[tuple[float, float]]:
    total = float(trial.traces["frame"].max()) / trial.fps / 60
    width = total / SEGMENTS
    return [(k * width, (k + 1) * width) for k in range(SEGMENTS)]


def segments_figure(trial: Trial, limits) -> Figure:
    height = (0.3 + LANE_RATIO + len(ROWS)) * ROW_INCH
    fig = Figure(figsize=(22, height * SEGMENTS + 1.8), layout="constrained")
    for sub, (t0, t1) in zip(fig.subfigures(SEGMENTS, 1), _edges(trial)):
        draw_block(sub, trial, t0, t1, limits)
    _legend(fig, trial)
    fig.suptitle(_title(trial), fontsize=11)
    return fig


def page_figure(trial: Trial, limits, t0: float, t1: float) -> Figure:
    fig = Figure(figsize=(22, 13), layout="constrained")
    draw_block(fig, trial, t0, t1, limits)
    _legend(fig, trial)
    fig.suptitle(_title(trial) + f"\n{t0:.1f}-{t1:.1f} min", fontsize=11)
    return fig


def _bin_median(frames: np.ndarray, values: np.ndarray, step: int = 16) -> np.ndarray:
    bin_id = np.asarray(frames) // step
    n_bins = int(bin_id.max()) + 1
    out = np.full(n_bins, np.nan)
    order = np.argsort(bin_id, kind="mergesort")
    bids, vals = bin_id[order], values[order]
    cuts = np.flatnonzero(np.diff(bids)) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(bids)]
    for s, e in zip(starts, ends):
        part = vals[s:e]
        part = part[np.isfinite(part)]
        if len(part):
            out[int(bids[s])] = float(np.median(part))
    return out


def attach_circle_columns(trial: Trial) -> None:
    """Add velocity direction, 1 s swept angle and path curvature onto ``trial.traces``.

    Saved traces predate these columns. They are the actor's own path, plus that
    actor relative to the other fish: 0° velocity direction is straight toward
    the other, and the swept angle keeps its sign.
    """
    needed = [c for _label, _color, series, _clip in ROWS for c, _ in series]
    if all(c in trial.traces.columns for c in needed):
        return
    import json

    from scripts.review import id_segments as S
    from scripts.review.ledger import (COVERAGE_COLUMNS, EVENT_COLUMNS, analysis_start, owner_series,
                                       read_csv, static_labels, swap_list, watched_intervals)

    package = Path(trial.provenance["package"])
    led = package / "id_pass"
    with np.load(package / "review_data.npz") as z:
        frames = np.asarray(z["frames"], int)
        fish_ids = [str(f) for f in z["fish_ids"]]
        key = "nodes3d_filled" if "nodes3d_filled" in z.files else "nodes3d"
        nodes = np.asarray(z[key], float)
        xyz = np.asarray(z["xyz"], float)
        present = np.asarray(z["view_present"])
        interp = np.asarray(z["is_interpolated"])
    manifest = json.loads((package.parent / "proxy" / "proxy_manifest.json").read_text(encoding="utf-8"))
    fps = float(manifest["views"]["top_cam"]["source_fps"])
    events = read_csv(led / "human_events.csv", EVENT_COLUMNS)
    coverage = read_csv(led / "coverage.csv", COVERAGE_COLUMNS)
    labels = static_labels(events, fish_ids)
    start = analysis_start(events)
    swaps = swap_list(events, fish_ids)
    danger = S.danger_map(xyz, present, interp, fps)
    status = S.frame_status(int(frames[0]), len(frames), start, danger["danger"], danger["runs"],
                            watched_intervals(coverage), [f for f, _, _ in swaps], S.manual_ranges(events), fps)
    usable = np.isin(status, (S.WATCHED_SAFE, S.WATCHED_DANGER))
    named, valid = F.named_nodes(nodes, owner_series(len(fish_ids), swaps, frames), fish_ids, labels)
    named[~usable | ~valid] = np.nan
    pos = F.centroids(named)
    slot = {name: i for i, name in enumerate(F.NAMES)}
    full: dict[str, np.ndarray] = {}
    for actor in F.NAMES:
        full[f"curve_{actor}"] = F.path_curvature_deg_per_mm(
            pos[:, slot[actor]], stride=max(1, int(round(0.1 * fps))))
    for actor, target in F.DIRECTED:
        orbit = F.orbit_rate(pos[:, slot[target]], pos[:, slot[actor]], fps)
        full[f"orbit_{actor}_around_{target}"] = orbit
        full[f"vel_angle_{actor}_to_{target}"] = F.velocity_bearing_deg(pos[:, slot[actor]], pos[:, slot[target]])
        full[f"sweep_{actor}_around_{target}"] = F.angle_swept_deg(orbit, fps)
    binned = {name: _bin_median(frames, values) for name, values in full.items()}
    index = trial.traces["frame"].to_numpy(int) // 16
    for name, values in binned.items():
        out = np.full(len(index), np.nan)
        ok = (index >= 0) & (index < len(values))
        out[ok] = values[index[ok]]
        trial.traces[name] = out


def write_trial(trial: Trial, folder: Path) -> int:
    attach_circle_columns(trial)
    limits = _limits(trial.traces)
    save_figure(segments_figure(trial, limits), folder / "features_segments.png", dpi=100)
    pages = folder / "features_pages"
    pages.mkdir(parents=True, exist_ok=True)
    for old in pages.glob("page_*.png"):
        old.unlink()
    with PdfPages(folder / "features_pages.pdf") as pdf:
        for k, (t0, t1) in enumerate(_edges(trial), start=1):
            fig = page_figure(trial, limits, t0, t1)
            fig.savefig(pages / f"page_{k}_{t0:04.1f}-{t1:04.1f}min.png", dpi=120)
            pdf.savefig(fig)
    return SEGMENTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Feature timelines for every reviewed recording of the groups.")
    parser.add_argument("--group", action="append", default=[], help="Group key, repeatable.")
    parser.add_argument("--experiment", action="append", default=[], help="One experiment folder name.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--paths-config", type=Path)
    args = parser.parse_args(argv)
    if not args.group and not args.experiment:
        parser.error("give --group or --experiment")
    _config, review, out = resolve_roots(args.paths_config)
    out = args.out or out
    search = default_search(None)
    names = list(args.experiment)
    for group in args.group:
        names += group_status(review, parse_group(group))[0]
    for name in names:
        trial = ensure_trial(review, out, name, search)
        pages = write_trial(trial, out / trial.experiment)
        print(f"{trial.experiment}: one segments page, {pages} pages")


if __name__ == "__main__":
    main()

