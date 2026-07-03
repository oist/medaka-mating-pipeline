"""TASK A3 - ranked close-approach event worklist for K_1_2 (read-only analysis)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.reconstruction_3d.calibration import Calibration  # noqa: E402

BACKBONE = Path("C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d.csv")
SWAPS = Path("C:/udmt_validation/qc/_swaps.txt")
CAL_DIR = REPO / "calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models"
OUT = Path(__file__).resolve().parent / "event_worklist.csv"

FPS = 166.0
DIST_THRESH_MM = 18.0
MERGE_GAP_FRAMES = 10
IDS = ["ID_1", "ID_2", "ID_3"]
PAIRS = [("ID_1", "ID_2"), ("ID_1", "ID_3"), ("ID_2", "ID_3")]
# Reprojection-overlap pixel threshold per view (cameras ~ tank 20cm wide in frame)
PX_OVERLAP_THRESH = 25.0


def load_backbone():
    df = pd.read_csv(BACKBONE, usecols=["frame", "fish_id", "X", "Y", "Z"])
    # pivot to per-fish wide arrays indexed by frame
    frames = np.arange(df.frame.max() + 1)
    pos = {}
    for fid in IDS:
        sub = df[df.fish_id == fid].set_index("frame")[["X", "Y", "Z"]]
        arr = np.full((len(frames), 3), np.nan)
        arr[sub.index.values] = sub.values
        pos[fid] = arr
    return frames, pos


def parse_swaps():
    rows = []
    pat = re.compile(r"SWAP\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    for line in SWAPS.read_text().splitlines():
        m = pat.search(line)
        if m:
            a, b, s, e, ln = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
            rows.append((frozenset((a, b)), s, e))
    return rows


def main():
    frames, pos = load_backbone()
    n = len(frames)

    # pairwise distances
    dist = {}
    for a, b in PAIRS:
        d = np.linalg.norm(pos[a] - pos[b], axis=1)
        dist[(a, b)] = d

    # close-approach mask per pair, then events per pair
    cal = Calibration(CAL_DIR)
    swaps = parse_swaps()

    events = []
    for (a, b) in PAIRS:
        d = dist[(a, b)]
        close = d < DIST_THRESH_MM  # NaN -> False
        close = np.nan_to_num(close, nan=False).astype(bool) if close.dtype != bool else close
        # build contiguous spans
        idx = np.where(close)[0]
        if len(idx) == 0:
            continue
        breaks = np.where(np.diff(idx) > 1)[0]
        spans = []
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k - prev > 1:
                spans.append((start, prev))
                start = k
            prev = k
        spans.append((start, prev))
        # merge spans separated by < MERGE_GAP_FRAMES
        merged = []
        for s, e in spans:
            if merged and s - merged[-1][1] - 1 < MERGE_GAP_FRAMES:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        for s, e in merged:
            events.append(_event_metrics(a, b, s, e, pos, dist, cal, swaps))

    cols = [
        "rank", "start_frame", "end_frame", "duration_frames", "duration_s",
        "pair", "fish_a", "fish_b", "min_distance_mm", "mean_distance_mm",
        "antiparallel", "heading_dot_at_min", "net_crossing", "dominant_axis",
        "overlap_views", "n_overlap_views", "overlaps_parsed_swap", "suspicion",
    ]
    edf = pd.DataFrame(events)
    edf["suspicion"] = edf["duration_s"] * np.where(edf["net_crossing"], 2.0, 1.0) * np.where(edf["antiparallel"], 1.5, 1.0)
    edf = edf.sort_values("suspicion", ascending=False).reset_index(drop=True)
    edf.insert(0, "rank", np.arange(1, len(edf) + 1))
    edf = edf[cols]
    edf.to_csv(OUT, index=False)

    # summary
    summary = {
        "total_events": len(edf),
        "dur_ge_0.5s": int((edf.duration_s >= 0.5).sum()),
        "dur_ge_1s": int((edf.duration_s >= 1.0).sum()),
        "net_crossing": int(edf.net_crossing.sum()),
        "antiparallel": int(edf.antiparallel.sum()),
        "overlaps_parsed_swap": int(edf.overlaps_parsed_swap.sum()),
    }
    print("SUMMARY", summary)
    print(edf.head(15).to_string())


def _heading(pos_arr, f):
    """unit velocity at frame f using +/-3 frame window."""
    lo = max(0, f - 3)
    hi = min(len(pos_arr) - 1, f + 3)
    v = pos_arr[hi] - pos_arr[lo]
    nrm = np.linalg.norm(v)
    if nrm < 1e-9 or np.isnan(nrm):
        return None
    return v / nrm


def _event_metrics(a, b, s, e, pos, dist, cal, swaps):
    d = dist[(a, b)][s:e + 1]
    dur_f = int(e - s + 1)
    fmin = s + int(np.nanargmin(d))
    dmin = float(np.nanmin(d))
    dmean = float(np.nanmean(d))

    # antiparallel vs parallel at min-distance frame (heading dot)
    ha = _heading(pos[a], fmin)
    hb = _heading(pos[b], fmin)
    if ha is None or hb is None:
        hdot = np.nan
        antiparallel = False
    else:
        hdot = float(np.dot(ha, hb))
        antiparallel = hdot < 0.0

    # net-crossing: relative spatial order swaps on the axis with largest separation change
    pa_s, pb_s = pos[a][s], pos[b][s]
    pa_e, pb_e = pos[a][e], pos[b][e]
    rel_s = pa_s - pb_s
    rel_e = pa_e - pb_e
    valid = np.isfinite(rel_s) & np.isfinite(rel_e)
    if valid.any():
        dom = int(np.nanargmax(np.abs(rel_s)))  # dominant axis = largest separation at start
        net_crossing = bool(np.sign(rel_s[dom]) != np.sign(rel_e[dom]) and abs(rel_s[dom]) > 1e-6 and abs(rel_e[dom]) > 1e-6)
        dominant_axis = ["X", "Y", "Z"][dom]
    else:
        net_crossing = False
        dominant_axis = "NA"

    # camera-view overlap via reprojection at min-distance frame
    overlap_views, n_ov = _overlap_views(a, b, fmin, pos, cal)

    sw_match = any((fs == frozenset((a, b))) and not (e < ss or s > se) for (fs, ss, se) in swaps)

    return dict(
        start_frame=int(s), end_frame=int(e), duration_frames=dur_f,
        duration_s=round(dur_f / FPS, 4), pair=f"{a}|{b}", fish_a=a, fish_b=b,
        min_distance_mm=round(dmin, 3), mean_distance_mm=round(dmean, 3),
        antiparallel=antiparallel, heading_dot_at_min=round(hdot, 4) if np.isfinite(hdot) else np.nan,
        net_crossing=net_crossing, dominant_axis=dominant_axis,
        overlap_views=overlap_views, n_overlap_views=n_ov,
        overlaps_parsed_swap=sw_match,
    )


def _overlap_views(a, b, f, pos, cal):
    pa, pb = pos[a][f], pos[b][f]
    if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
        return "", 0
    proj = cal.compute_imCoords_from_XYZ(np.vstack([pa, pb]))  # (2,3,2)
    view_names = ["top", "side1", "side2"]
    overlaps = []
    for ci, name in enumerate(view_names):
        px_d = np.linalg.norm(proj[0, ci] - proj[1, ci])
        if px_d < PX_OVERLAP_THRESH:
            overlaps.append(name)
    return "+".join(overlaps), len(overlaps)


if __name__ == "__main__":
    main()
