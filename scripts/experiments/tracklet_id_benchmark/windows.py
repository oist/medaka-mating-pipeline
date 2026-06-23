from __future__ import annotations

"""Select a small taxonomy of pilot windows for the identity benchmark.

A *window* is a contiguous frame range `[start, end]` plus a `kind` describing
what makes it interesting. The taxonomy deliberately spans the easy and the hard
cases so a comparison is not dominated by either:

  * ``early_fp``         single->few fish + false-positive phantoms (clip start)
  * ``easy_separated``   a long stretch with no hard-core event (upper bound)
  * ``crossing_clear``   around a resolver ``velocity_consistent`` (clean) pass
  * ``dwell_ambiguous``  around a long ``ambiguous`` event (the unresolvable core)
  * ``suspect_swap``     one window per resolver ``suspect_swap`` (likely real swap)

The selection is pure (``select_windows``) so it is unit-tested without disk I/O;
``main`` is a thin CLI that loads the resolved-events CSV and writes ``windows.csv``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))


def _pad(center: int, half: int, lo: int, hi: int) -> tuple[int, int]:
    s = max(lo, center - half)
    e = min(hi, center + half)
    return int(s), int(e)


def select_windows(
    events: pd.DataFrame,
    *,
    total_frames: int,
    fps: float = 166.0,
    pad_frames: int = 500,
    long_event_frames: int = 166,
    n_suspect: int = 4,
    n_clear: int = 1,
    n_ambiguous: int = 1,
    early_fp_end: int = 8000,
    seed: int = 0,
) -> list[dict]:
    """Return an ordered list of window dicts spanning the event taxonomy.

    ``events`` is the resolver output (columns: pair,start,end,n_frames,min_mm,
    verdict,reason,...). ``total_frames`` bounds the windows. Deterministic given
    ``seed`` (used only to pick representative clear/ambiguous events).
    """
    rng = np.random.default_rng(seed)
    lo, hi = 0, int(total_frames) - 1
    out: list[dict] = []

    def add(name, kind, center, source_event="", note="", half=None):
        s, e = _pad(int(center), pad_frames if half is None else half, lo, hi)
        out.append({"name": name, "kind": kind, "start": s, "end": e,
                    "center": int(center), "n_frames": e - s + 1,
                    "source_event": source_event, "note": note})

    # 1) early false-positive / single->three transition window
    add("early_fp", "early_fp", min(early_fp_end // 2, hi),
        note="clip start: single/few fish, FP phantoms before all 3 are added",
        half=max(pad_frames, early_fp_end // 2))

    ev = events.copy()
    if "reason" not in ev.columns:
        ev["reason"] = ""
    ev["n_frames"] = pd.to_numeric(ev.get("n_frames", 0), errors="coerce").fillna(0)
    ev["center"] = ((pd.to_numeric(ev["start"], errors="coerce")
                     + pd.to_numeric(ev["end"], errors="coerce")) / 2)

    # 2) easy separated window: the largest gap between consecutive events,
    #    away from the early-FP zone.
    starts = np.sort(pd.to_numeric(ev["start"], errors="coerce").dropna().to_numpy())
    starts = starts[starts > early_fp_end]
    if len(starts) >= 2:
        gaps = np.diff(starts)
        gi = int(np.argmax(gaps))
        mid = int((starts[gi] + starts[gi + 1]) / 2)
        add("easy_separated", "easy_separated", mid,
            note=f"longest event-free gap ({int(gaps[gi])} fr) — upper-bound sanity")
    else:
        add("easy_separated", "easy_separated", min(hi, early_fp_end + 20000),
            note="fallback easy window (few events)")

    # 3) clear crossings (clean passes the resolver auto-cleared)
    clears = ev[ev["reason"] == "velocity_consistent"].sort_values("n_frames", ascending=False)
    for i, (_, r) in enumerate(clears.head(n_clear).iterrows()):
        add(f"crossing_clear_{i}", "crossing_clear", r["center"],
            source_event=f"{r.get('pair','')}@{int(r['start'])}-{int(r['end'])}",
            note=f"resolver clear, n_frames={int(r['n_frames'])}, min_mm={r.get('min_mm')}")

    # 4) long ambiguous dwells (the unresolvable hard core)
    amb = ev[(ev["reason"] == "ambiguous") & (ev["n_frames"] >= long_event_frames)]
    amb = amb.sort_values("n_frames", ascending=False)
    for i, (_, r) in enumerate(amb.head(n_ambiguous).iterrows()):
        add(f"dwell_ambiguous_{i}", "dwell_ambiguous", r["center"],
            source_event=f"{r.get('pair','')}@{int(r['start'])}-{int(r['end'])}",
            note=f"long ambiguous dwell, n_frames={int(r['n_frames'])}, min_mm={r.get('min_mm')}")

    # 5) suspect swaps (likely real id errors) — highest review priority
    susp = ev[ev["reason"] == "suspect_swap"].sort_values("n_frames", ascending=False)
    for i, (_, r) in enumerate(susp.head(n_suspect).iterrows()):
        add(f"suspect_swap_{i}", "suspect_swap", r["center"],
            source_event=f"{r.get('pair','')}@{int(r['start'])}-{int(r['end'])}",
            note=f"velocity contradicts linker, n_frames={int(r['n_frames'])}, min_mm={r.get('min_mm')}")

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select pilot windows for the identity benchmark.")
    p.add_argument("--events-csv", required=True, help="hardcore_events_resolved.csv")
    p.add_argument("--total-frames", type=int, required=True)
    p.add_argument("--out", required=True, help="windows.csv to write")
    p.add_argument("--pad-frames", type=int, default=500)
    p.add_argument("--n-suspect", type=int, default=4)
    p.add_argument("--early-fp-end", type=int, default=8000)
    p.add_argument("--fps", type=float, default=166.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    events = pd.read_csv(args.events_csv)
    windows = select_windows(events, total_frames=args.total_frames, fps=args.fps,
                             pad_frames=args.pad_frames, n_suspect=args.n_suspect,
                             early_fp_end=args.early_fp_end)
    df = pd.DataFrame(windows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"selected {len(df)} windows -> {args.out}")
    print(df[["name", "kind", "start", "end", "n_frames"]].to_string(index=False))


if __name__ == "__main__":
    main()
