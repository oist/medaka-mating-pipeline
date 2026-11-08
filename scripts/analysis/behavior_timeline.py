"""One recording against time, one set of rows per fish, with the review and the behaviours on top.

Per recording, two outputs (PNG and PDF each)::

    features_segments     the whole recording cut into six equal segments stacked on one page
    features_pages        the same six segments, one per page, larger (one PDF and one PNG each)

From the top of each segment:

* the review bar: kept, discarded, unwatched;
* the behaviour lane: one row per behaviour, named at the left, every event tagged with who
  does it to whom (``M1→F``: male 1 to the female; ``M2→M1``: male 2 to male 1);
* speed of each fish;
* distance from each male to the female, with the 12 mm and 40 mm levels marked;
* distance between the two males;
* each male's height minus the female's (negative: he is below her);
* how fast the female–male gap is shrinking;
* each male's 3D swim direction toward the female, and toward the other male
  (0° straight at that fish, 90° sideways);
* each male's body heading toward the female, and toward the other male
  (where the snout points, separate from where he is swimming);
* net turn of each male around the female over the last 1 s;
* net turn of the line between the two males over the last 1 s (one series:
  the opposite direction is the same number);
* the female's speed component away from each male;
* each male's speed component away from the other male.

The colour of a line is the fish that series is about. Tinted bands go over every row:
a behaviour in its colour, red where the review discarded the frames, grey where nobody
watched. Lines stop in discarded stretches.

Rows that repeat one geometric quantity are not drawn. The angular velocity of a male
about the female is the angular velocity of the line that joins them, and the opposite
ordering of the two males has that same rate; the 1 s net turn is that rate added up,
so the figure keeps the sum and not the rate. Path curvature is omitted: its time index
is wrong, and a bend in one fish's own path does not say which fish is at the centre.

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
# (axis label, label colour, [(column, line colour), ...], clip, guide levels).
# clip: pos = from 0, sym = about 0, angle = 0-180, sweep = -360..360.
# A guide is a dashed horizontal line (the 12/40 mm band, the tangent angles, or zero).
ROWS = (
    ("Speed (mm/s)\nFemale (pink), Male 1 (green), Male 2 (orange)", BOTH,
     (("speed_F", FEMALE), ("speed_M1", M1), ("speed_M2", M2)), "pos", ()),
    ("Distance to the female (mm)\nMale 1 (green), Male 2 (orange)\ndashed lines at 12 and 40", FEMALE,
     (("dist_F_M1", M1), ("dist_F_M2", M2)), "pos", (12.0, 40.0)),
    ("Distance between\nthe two males (mm)", BOTH, (("dist_M1_M2", BOTH),), "pos", ()),
    ("Male height minus female height (mm)\nnegative = that male is below her", BOTH,
     (("zrel_M1_F", M1), ("zrel_M2_F", M2)), "sym", (0.0,)),
    ("Closing speed of the female-male gap (mm/s)\npositive = the distance is shrinking", BOTH,
     (("close_F_M1", M1), ("close_F_M2", M2)), "sym", (0.0,)),
    ("Swim direction toward the female (deg)\n"
     "0 = straight at her, 90 = sideways, 180 = away\n3D velocity; dashed lines at 60, 90, 120", BOTH,
     (("bear3d_M1_to_F", M1), ("bear3d_M2_to_F", M2)), "angle", (60.0, 90.0, 120.0)),
    ("Swim direction toward the other male (deg)\n0 = straight at him, 90 = sideways, 180 = away", BOTH,
     (("bear3d_M1_to_M2", M1), ("bear3d_M2_to_M1", M2)), "angle", (90.0,)),
    ("Body heading toward the female (deg)\nsnout axis; 0 = pointing at her", BOTH,
     (("head_M1_to_F", M1), ("head_M2_to_F", M2)), "angle", ()),
    ("Body heading toward the other male (deg)\nsnout axis; 0 = pointing at him", BOTH,
     (("head_M1_to_M2", M1), ("head_M2_to_M1", M2)), "angle", ()),
    ("Net turn around the female\nin the last 1 s (deg)\n"
     "positive = counterclockwise; +360 or -360 = one full circle", BOTH,
     (("sweep_M1_around_F", M1), ("sweep_M2_around_F", M2)), "sweep", (0.0,)),
    ("Net turn between the two males\nin the last 1 s (deg)\n"
     "Male 1 about Male 2; the reverse is the same number", BOTH,
     (("sweep_M1_around_M2", BOTH),), "sweep", (0.0,)),
    ("Female speed away from each male (mm/s)\npositive = she is moving away from that male", FEMALE,
     (("away_F_from_M1", M1), ("away_F_from_M2", M2)), "sym", (0.0,)),
    ("Male speed away from the other male (mm/s)\npositive = he is moving away", BOTH,
     (("away_M1_from_M2", M1), ("away_M2_from_M1", M2)), "sym", (0.0,)),
)
LANE_ORDER = ("attack", "attempted att", "quick circle", "incomplete qc", "wrap", "spawn")
SEGMENTS = 6
ROW_INCH = 0.78
LANE_RATIO = 4.5


def _limits(trace) -> dict[str, tuple[float, float]]:
    """Axis limits per row from the whole recording, so every segment shares a scale."""
    out = {}
    for label, _color_, series, clip, _guides in ROWS:
        values = np.concatenate([trace[c].to_numpy(float) for c, _ in series])
        values = values[np.isfinite(values)]
        if clip == "angle":
            out[label] = (0.0, 180.0)
        elif not len(values):
            out[label] = (0.0, 1.0)
        elif clip == "pos":
            out[label] = (0.0, float(np.percentile(values, 99.5)) * 1.05)
        elif clip == "sweep":
            out[label] = (-360.0, 360.0)
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

    for ax, (label, label_color, series, _clip, guides) in zip(rows, ROWS):
        _bands(ax, trial, t0, t1, minimum)
        for level in guides:
            ax.axhline(level, color="#4a4a4a", lw=0.6, ls=(0, (3, 2)), zorder=2)
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
    return (f"{trial.experiment}: kept {pct.get('kept', 0):.1f}%, "
            f"discarded {pct.get('discarded', 0):.1f}%, unwatched {pct.get('unwatched', 0):.1f}%")


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
    fig.suptitle(_title(trial) + f"    {t0:.1f}-{t1:.1f} min", fontsize=11)
    return fig


def _decimate_median(values: np.ndarray, step: int = 16) -> np.ndarray:
    """Median of each contiguous block, the same blocks as ``behavior_features.decimate``."""
    out = np.full(int(np.ceil(len(values) / step)), np.nan)
    for i, a in enumerate(range(0, len(values), step)):
        part = values[a:a + step]
        part = part[np.isfinite(part)]
        if len(part):
            out[i] = float(np.median(part))
    return out


def _pair_distance(pos: np.ndarray, i: int, j: int) -> np.ndarray:
    dist = np.linalg.norm(pos[:, i] - pos[:, j], axis=-1)
    dist = np.asarray(dist, float)
    missing = ~np.isfinite(pos[:, i]).all(1) | ~np.isfinite(pos[:, j]).all(1)
    dist[missing] = np.nan
    return dist


def attach_circle_columns(trial: Trial) -> None:
    """Add the rows that are not in the saved trace table.

    Saved traces already carry speed, distance, body heading and the 1 s net turn.
    This fills height difference, gap closing, 3D swim direction and the away
    component, and rebuilds the 1 s net turn when that column is absent.
    """
    needed = [c for _label, _color, series, _clip, _guides in ROWS for c, _ in series]
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
    female = slot["F"]
    for actor in ("M1", "M2"):
        height = np.asarray(pos[:, slot[actor], 2] - pos[:, female, 2], float)
        missing = ~np.isfinite(pos[:, slot[actor], 2]) | ~np.isfinite(pos[:, female, 2])
        height[missing] = np.nan
        full[f"zrel_{actor}_F"] = height
        full[f"close_F_{actor}"] = F.closing_speed(_pair_distance(pos, female, slot[actor]), fps)
        full[f"away_F_from_{actor}"] = F.away_speed(pos[:, female], pos[:, slot[actor]], fps)
        full[f"bear3d_{actor}_to_F"] = F.velocity_bearing_3d_deg(pos[:, slot[actor]], pos[:, female])
    for actor, other in (("M1", "M2"), ("M2", "M1")):
        full[f"bear3d_{actor}_to_{other}"] = F.velocity_bearing_3d_deg(pos[:, slot[actor]], pos[:, slot[other]])
        full[f"away_{actor}_from_{other}"] = F.away_speed(pos[:, slot[actor]], pos[:, slot[other]], fps)
    for actor, target in (("M1", "F"), ("M2", "F"), ("M1", "M2")):
        orbit = F.orbit_rate(pos[:, slot[target]], pos[:, slot[actor]], fps)
        full[f"sweep_{actor}_around_{target}"] = F.angle_swept_deg(orbit, fps)
    n_rows = len(trial.traces)
    for name, values in full.items():
        if name in trial.traces.columns:
            continue
        column = _decimate_median(values)
        if len(column) != n_rows:
            raise RuntimeError(f"{name}: {len(column)} bins, trace has {n_rows} rows")
        trial.traces[name] = column


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

