"""Timeline figures of a recording: what the reviewer marked, what the machine flagged, and how they agree.

    python -m scripts.review.id_figure <review_root>/<exp>/<tag> [--open] [--wait]

Two figures on one time scale, never mixed:

* ``id_human.png`` - only the reviewer's work: what was watched, the ID changes (4/5/6), the unusable (X)
  and wrong-pose (B) ranges. The machine's danger runs play no part.
* ``id_machine.png`` - only the machine's intervals (danger under ``safe_mm`` or unreliable 3D, high risk
  under ``high_risk_mm``), over the same frames. Its subtitle holds the agreement with the human marks,
  also written to ``id_agreement.json``.

Reviewed folders read ``<folder>/id_pass/`` if that pass was watched, else the folder's own ledger.
Unreviewed folders get the machine figure only. ``--wait`` polls until the package has been built, so a
figure can be requested before a batch reaches it.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
from pathlib import Path

import numpy as np

from scripts.review import id_segments as S
from scripts.review.ledger import (COVERAGE_COLUMNS, EVENT_COLUMNS, analysis_start, read_csv, swap_list,
                                   watched_intervals)

HUMAN_FIGURE = "id_human.png"
MACHINE_FIGURE = "id_machine.png"
AGREEMENT_FILE = "id_agreement.json"
ROW_SECONDS = 300.0
WAIT_SECONDS = 60.0
LOOK_BACK_S = 3.0          # an ID change is often pressed after the encounter: also count the machine's danger this far back
MARK_DRAW_S = 1.0          # marks are drawn at least this wide so a short one stays visible; percentages use real frames
SAFE, DANGER, CRITICAL, NEUTRAL = "#0ca30c", "#fab219", "#d03b3b", "#e1e0d9"     # status palette; neutral = gridline
POSE, CHANGE = "#c77dff", "#7a1414"
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
H_UNWATCHED, H_WATCHED, H_POSE, H_UNUSABLE, H_CHANGE = 0, 1, 2, 3, 4
HUMAN = ((H_WATCHED, "Watched, nothing marked", SAFE), (H_POSE, "Wrong pose (B), identity right", POSE),
         (H_UNUSABLE, "Marked unusable (X)", CRITICAL), (H_CHANGE, "ID change (4/5/6), +-0.5 s", CHANGE),
         (H_UNWATCHED, "Not watched", NEUTRAL))
MACHINE = ((1, "Safe", SAFE), (2, "Danger", DANGER), (3, "High risk", CRITICAL))


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def machine_codes(danger_map: dict) -> np.ndarray:
    """``(F,)`` 1 safe, 2 danger, 3 high risk."""
    return (1 + danger_map["danger"].astype(np.int8) + danger_map["high_frames"].astype(np.int8)).astype(np.int8)


def shares(codes: np.ndarray, categories) -> list[tuple[str, str, float]]:
    """``(label, color, percent)`` of the categories, over the frames in any of them."""
    known = np.isin(codes, [c for c, _l, _k in categories])
    total = max(1, int(known.sum()))
    return [(label, color, 100 * float((codes == c).sum()) / total) for c, label, color in categories]


def draw(path: Path, codes: np.ndarray, categories, first: int, fps: float, title: str, subtitle: str,
         min_draw_s: float = 0.0) -> Path:
    """Write the figure: the percentages, then the recording wrapped into 5 minute rows on one time scale."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    last_s = (first + len(codes)) / fps
    n_rows = int(np.ceil(last_s / ROW_SECONDS))
    height = 2.5 + 0.62 * n_rows
    fig = Figure(figsize=(14, height), dpi=130, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    fig.text(0.03, 1 - 0.32 / height, title, fontsize=15, fontweight="bold", color=INK, va="center")
    fig.text(0.03, 1 - 0.52 / height, "\n".join(textwrap.wrap(subtitle, 165)), fontsize=10, color=INK_2, va="top")
    present = [s for s in shares(codes, categories) if round(s[2], 1) > 0 or s[1] != NEUTRAL]
    for i, (label, color, pct) in enumerate(present):
        x = 0.03 + i * 0.94 / max(4, len(present))
        y = 1 - 1.4 / height
        fig.patches.append(__rect(fig, x, y - 0.11 / height, 0.012, 0.22 / height, color))
        fig.text(x + 0.018, y, f"{pct:.1f}%", fontsize=20, fontweight="bold", color=INK, va="center")
        fig.text(x + 0.018, y - 0.27 / height, label, fontsize=10, color=INK_2, va="center")
    top = 1 - 1.9 / height
    bottom = 0.42 / height
    grid = fig.add_gridspec(n_rows, 1, left=0.07, right=0.985, top=top, bottom=bottom, hspace=0.75)
    for row in range(n_rows):
        ax = fig.add_subplot(grid[row])
        t0 = row * ROW_SECONDS
        for code, _label, color in sorted(categories, key=lambda c: c[2] != NEUTRAL):    # grey first, marks on top
            spans = S.runs_of(codes == code)
            if len(spans):
                ax.broken_barh([((first + a) / fps, max((b - a) / fps, min_draw_s)) for a, b in spans], (0, 1),
                               facecolors=color, edgecolors="none")
        ax.set_xlim(t0, t0 + ROW_SECONDS)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(np.arange(t0, t0 + ROW_SECONDS + 1, 60))
        ax.set_xticklabels([mmss(x) for x in np.arange(t0, t0 + ROW_SECONDS + 1, 60)], fontsize=8, color=MUTED)
        ax.tick_params(axis="x", length=3, color="#c3c2b7", pad=2)
        ax.set_facecolor("#f0efec")
        for side in ("top", "left", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel(f"{row * 5}-{row * 5 + 5} min", rotation=0, ha="right", va="center", fontsize=9, color=INK_2,
                      labelpad=8)
    note = "x: time in the recording (mm:ss), 5 minutes per row"
    if (codes == S.NOT_ANALYSED).any():
        note += "; empty at the start = before the analysis start"
    if min_draw_s:
        note += f"; marks drawn at least {min_draw_s:g} s wide"
    fig.text(0.985, 0.012, note, fontsize=8, color=MUTED, ha="right")
    fig.savefig(path, facecolor=SURFACE)
    return path


def __rect(fig, x, y, w, h, color):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=color, edgecolor="none", figure=fig)


def _span(first: int, n: int, lo: int, hi: int) -> slice:
    """Index slice of the inclusive frames ``lo..hi`` of a recording starting at ``first``."""
    return slice(max(0, lo - first), max(0, min(n, hi - first + 1)))


def human_codes(first: int, n: int, start: int | None, watched: list[tuple[int, int]], changes: list[int],
                manual: list[tuple[int, int]], pose: list[tuple[int, int, list[str]]], fps: float) -> np.ndarray:
    """``(n,)`` what the reviewer did, frame by frame; the machine's danger runs are not used.

    Later rules win: watched, then wrong pose, then unusable, then an ID change (+-``CHANGE_MARGIN_S``
    around the frame it was placed at). Nothing before the analysis start counts.
    """
    codes = np.full(n, H_UNWATCHED, np.int8)
    for a, b in watched:
        codes[_span(first, n, a, b)] = H_WATCHED
    for lo, hi, _fish in pose:
        codes[_span(first, n, lo, hi)] = H_POSE
    for lo, hi in manual:
        codes[_span(first, n, lo, hi)] = H_UNUSABLE
    margin = int(round(S.CHANGE_MARGIN_S * fps))
    for f in changes:
        codes[_span(first, n, f - margin, f + margin)] = H_CHANGE
    if start is not None:
        codes[:max(0, start - first)] = S.NOT_ANALYSED
    return codes


def machine_review_codes(dmap: dict, first: int, start: int | None) -> np.ndarray:
    """:func:`machine_codes` over the analysed frames only, the same ones the human figure covers."""
    codes = machine_codes(dmap)
    if start is not None:
        codes[:max(0, start - first)] = S.NOT_ANALYSED
    return codes


def agreement(dmap: dict, first: int, fps: float, start: int | None, changes: list[int],
              manual: list[tuple[int, int]]) -> dict:
    """How the human marks and the machine's danger runs line up, from the analysis start.

    An ID change counts as inside danger when a danger frame lies within ``+-CHANGE_MARGIN_S`` of the frame
    it was placed at (``strict``) or within the ``LOOK_BACK_S`` before it (``lenient``: the key is often
    pressed after the encounter). An X range counts when it touches danger. Danger runs are counted as
    holding a human mark when a change or an X range falls inside them (+- the margin).
    """
    n = len(dmap["danger"])
    lo0 = 0 if start is None else max(0, start - first)
    danger, high = dmap["danger"].copy(), dmap["high_frames"].copy()
    danger[:lo0], high[:lo0] = False, False
    margin, back = int(round(S.CHANGE_MARGIN_S * fps)), int(round(LOOK_BACK_S * fps))
    ts = [c - first for c in changes if c - first >= lo0]
    xs = [(max(a - first, lo0), b - first) for a, b in manual if b - first >= lo0]

    def touches(mask: np.ndarray, a: int, b: int) -> bool:
        return bool(mask[max(a, 0):max(b + 1, 0)].any())

    out = {
        "from_frame": first + lo0, "id_changes": len(ts), "unusable_ranges": len(xs),
        "changes_in_danger_strict": sum(touches(danger, t - margin, t + margin) for t in ts),
        "changes_in_danger_lenient": sum(touches(danger, t - back, t + margin) for t in ts),
        "changes_in_high_strict": sum(touches(high, t - margin, t + margin) for t in ts),
        "changes_in_high_lenient": sum(touches(high, t - back, t + margin) for t in ts),
        "unusable_touching_danger": sum(touches(danger, a, b) for a, b in xs),
        "unusable_touching_high": sum(touches(high, a, b) for a, b in xs),
    }
    marks = np.zeros(n, bool)
    for t in ts:
        marks[max(0, t - margin):t + margin + 1] = True
    for a, b in xs:
        marks[a:b + 1] = True
    runs = [(a, b, h) for (a, b), h in zip(dmap["runs"], dmap["high"]) if b > lo0]
    held = [bool(marks[max(a - margin, 0):b + margin].any()) for a, b, _h in runs]
    is_high = [bool(h) for _a, _b, h in runs]
    out.update({
        "danger_runs": len(runs), "danger_runs_with_mark": sum(held),
        "high_runs": sum(is_high), "high_runs_with_mark": sum(h and k for h, k in zip(is_high, held)),
        "danger_seconds": round(float(danger.sum()) / fps, 1), "marked_seconds": round(float(marks.sum()) / fps, 1),
        "marked_seconds_in_danger": round(float((marks & danger).sum()) / fps, 1),
    })
    return out


def human_subtitle(codes: np.ndarray, changes: int, manual: int, pose: int, start: int | None) -> str:
    analysed = max(1, int((codes != S.NOT_ANALYSED).sum()))
    watched = 100 * float(np.isin(codes, (H_WATCHED, H_POSE, H_UNUSABLE, H_CHANGE)).sum()) / analysed
    return (f"Reviewer only: {watched:.1f}% watched, {changes} ID changes, {manual} unusable (X) ranges, {pose} wrong-pose (B) "
            f"marks. The machine's danger runs are not used here. Percentages of the analysed time "
            f"from frame {start if start is not None else 'first'}.")


def machine_subtitle(dmap: dict, params: S.SegmentParams, a: dict | None) -> str:
    n_runs, n_high = (len(dmap["runs"]), int(dmap["high"].sum())) if a is None else (a["danger_runs"], a["high_runs"])
    text = (f"Machine only: danger = under {params.safe_mm:g} mm or unreliable 3D; high risk = under "
            f"{params.high_risk_mm:g} mm. {n_runs} danger runs, {n_high} high risk"
            f"{'' if a is None else f' (from frame {a['from_frame']}, like the reviewer figure)'}. ")
    if a is None:
        return text + "Whole recording; not reviewed, so no human marks to compare."
    n_c, n_x = a["id_changes"], a["unusable_ranges"]
    return (text + f"Against the reviewer: ID changes in danger {a['changes_in_danger_strict']}/{n_c} (+-0.5 s), "
            f"{a['changes_in_danger_lenient']}/{n_c} counting {LOOK_BACK_S:g} s back; in high risk "
            f"{a['changes_in_high_strict']}/{n_c} and {a['changes_in_high_lenient']}/{n_c}. Unusable (X) ranges touching "
            f"danger {a['unusable_touching_danger']}/{n_x}. Danger runs holding a human mark "
            f"{a['danger_runs_with_mark']}/{a['danger_runs']}, high-risk runs {a['high_runs_with_mark']}/{a['high_runs']}.")


def write_figures(out_dir: Path, dmap: dict, events, coverage, first: int, n: int, fps: float, title: str,
                  params: S.SegmentParams, fish_ids: list[str]) -> list[Path]:
    """The human and the machine figure and ``id_agreement.json`` in ``out_dir``; returns the figure paths."""
    out_dir = Path(out_dir)
    swaps = [f for f, _a, _b in swap_list(events, fish_ids)]
    start = analysis_start(events)
    manual, pose = S.manual_ranges(events), S.pose_ranges(events)
    human = human_codes(first, n, start, watched_intervals(coverage), swaps, manual, pose, fps)
    agree = agreement(dmap, first, fps, start, swaps, manual)
    (out_dir / AGREEMENT_FILE).write_text(json.dumps(agree, indent=2), encoding="utf-8")
    return [
        draw(out_dir / HUMAN_FIGURE, human, HUMAN, first, fps, title + "  -  reviewer",
             human_subtitle(human, len(swaps), len(manual), len(pose), start), MARK_DRAW_S),
        draw(out_dir / MACHINE_FIGURE, machine_review_codes(dmap, first, start), MACHINE, first, fps,
             title + "  -  machine", machine_subtitle(dmap, params, agree)),
    ]


def build(folder: Path, params: S.SegmentParams = S.SegmentParams()) -> list[Path]:
    """Figures of ``folder``: reviewer and machine if it was reviewed, else the machine's only."""
    folder = Path(folder)
    manifest = json.loads((folder / "review_manifest.json").read_text(encoding="utf-8"))
    fps = float(manifest.get("fps") or 166.0)
    with np.load(folder / "review_data.npz") as z:
        frames, fish_ids = z["frames"], [str(f) for f in z["fish_ids"]]
        xyz, present, interp = z["xyz"].astype(float), z["view_present"], z["is_interpolated"]
    dmap = S.danger_map(xyz, present, interp, fps, params)
    first = int(frames[0])
    source, coverage, events = None, None, None
    for candidate in (folder / S.LEDGER_FOLDER, folder):
        coverage = read_csv(candidate / "coverage.csv", COVERAGE_COLUMNS)
        if len(coverage):
            source, events = candidate, read_csv(candidate / "human_events.csv", EVENT_COLUMNS)
            break
    name = f"{manifest.get('experiment', folder.parent.name)}  /  {folder.name}"
    if source is not None:
        return write_figures(source, dmap, events, coverage, first, len(frames), fps, name, params, fish_ids)
    codes = machine_codes(dmap)
    safe_runs = S.runs_of(codes == 1)
    lengths = (safe_runs[:, 1] - safe_runs[:, 0]) / fps
    sub = (machine_subtitle(dmap, params, None) + f" {len(safe_runs)} safe intervals, median "
           f"{np.median(lengths):.1f} s, longest {lengths.max():.0f} s.")
    return [draw(folder / MACHINE_FIGURE, codes, MACHINE, first, fps, name + "  -  machine", sub)]


def wait_for(folder: Path) -> None:
    while not (Path(folder) / "review_manifest.json").is_file():
        time.sleep(WAIT_SECONDS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    p.add_argument("--open", action="store_true", help="open the figure when it is written")
    p.add_argument("--wait", action="store_true", help="wait until the package has been built")
    args = p.parse_args(argv)
    if args.wait:
        wait_for(args.folder)
    for out in build(args.folder):
        print(out, flush=True)
        if args.open:
            os.startfile(out)      # noqa: S606 - Windows, the reviewer's own image viewer
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
