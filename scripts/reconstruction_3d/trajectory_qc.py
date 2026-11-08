"""Flag anomalous stretches of a 3D reconstruction from its trajectories alone.

Reads a review package's ``review_data.npz`` (pectoral anchor ``xyz``, filled nodes, views, residuals) and,
for an m3dfull package, medaka-3d's ``c30_summary.json`` (slot joins and refused boundaries). Every check
uses one fixed parameter set for all experiments, so group differences are not processing differences.

Two categories:

* ``identity`` - where a name may have moved to another fish: ``jump`` (pectoral step), ``close`` (two fish
  closer than ``close_mm``), ``gap`` (long interpolation), ``few_views`` (fewer than two views),
  ``c30_join`` (the stitcher re-permuted slots here), ``c30_refusal`` (it could not decide).
* ``geometry`` - where the position or body is doubtful: ``spike`` (acceleration), ``local_outlier``
  (far off the running median, robust z per fish), ``body_length`` (mouth-pectoral-tail path length off
  the fish's median), ``out_of_volume``, ``residual``.

Outputs, in the package folder:

* ``trajectory_anomalies.csv`` - one row per raw finding (type, category, fish, start, end, peak, score);
* ``trajectory_anomaly_intervals.csv`` - padded, merged intervals over all fish;
* ``trajectory_flags.npz`` - ``flags`` (frames x fish) bit mask, bit ``i`` = ``type_names[i]``;
* ``trajectory_qc_summary.json`` - parameters, counts, flagged share per type after the analysis start and,
  when the package has an identity pass, how the flags cover the reviewer's marks (``score_against_review``).

Flags are hints for review and for sensitivity analyses; they change no coordinate.

CLI::

    python -m scripts.reconstruction_3d.trajectory_qc <review_package_dir> [--c30-summary <json>]
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.review.review_data import runs

#: Bit order of ``trajectory_flags.npz``; append only, so old flag files keep their meaning.
TYPES = ("jump", "close", "gap", "few_views", "c30_join", "c30_refusal",
         "spike", "local_outlier", "body_length", "out_of_volume", "residual")
IDENTITY_TYPES = frozenset(TYPES[:6])
ROW_COLUMNS = ["type", "category", "fish", "start", "end", "peak", "score", "details"]
INTERVAL_COLUMNS = ["interval_id", "frame_start", "frame_end", "categories", "types", "fish", "n_findings"]


@dataclass(frozen=True)
class QCParams:
    max_step_mm: float = 6.0          # pectoral step per frame (~1 m/s at 166 fps); P_1_1 p99.9 is 4.2
    max_acc_mm: float = 3.0           # mm/frame^2, as review.postprocess.PostParams.acc_mm; P_1_1 p99 is 1.3
    local_window: int = 9             # frames of the centred running median a position is compared with
    local_z: float = 8.0              # robust z of the distance from that median, per fish ...
    local_floor_mm: float = 1.5       # ... and at least this far
    length_low: float = 0.65          # body path length / the fish's median (P_1_1 p0.1 ~0.69, p99.9 ~1.14)
    length_high: float = 1.25
    length_min_frames: int = 3
    volume_mm: tuple[float, float, float] = (192.0, 192.0, 186.0)   # calibrated volume from the origin
    volume_margin_mm: float = 10.0
    gap_min_frames: int = 17          # interpolated for at least 0.1 s
    few_views_min_frames: int = 17    # seen in fewer than two views for at least 0.1 s
    residual_mm2: float = 10.0        # median node residual; P_1_1 p99.9 is ~28
    residual_min_frames: int = 5
    close_mm: float = 10.0            # P_1_1 id pass: 8 / 10 / 15 mm catch 10 / 11 / 11 of 13 name changes
    pad_frames: int = 17              # merged intervals are padded by this much on both sides


def _robust_z(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return np.full_like(x, np.nan)
    med = np.median(x[ok])
    mad = 1.4826 * np.median(np.abs(x[ok] - med))
    return (x - med) / max(mad, 1e-9)


def _local_deviation(x: np.ndarray, window: int) -> np.ndarray:
    """Distance of each position from the centred running median of its track."""
    med = pd.DataFrame(x).rolling(window, center=True, min_periods=window // 2 + 1).median().to_numpy()
    return np.linalg.norm(x - med, axis=1)


def body_path_length(nodes: np.ndarray, pect: int) -> np.ndarray:
    """``(F, fish)`` sum of the node-to-pectoral distances (bending changes it little)."""
    seg = [np.linalg.norm(nodes[:, :, n] - nodes[:, :, pect], axis=-1) for n in range(nodes.shape[2]) if n != pect]
    return np.sum(seg, axis=0)


def read_c30(summary: Path, lam: float) -> tuple[list[dict], list[dict]]:
    """``(joins, refusals)`` of one lambda arm of a ``c30_summary.json``."""
    arms = json.loads(Path(summary).read_text(encoding="utf-8"))["arms"]
    arm = arms[f"{lam:g}"]
    return arm.get("join_list", []), arm.get("refusal_list", [])


def find_anomalies(frames: np.ndarray, fish_ids: list[str], xyz: np.ndarray, nodes: np.ndarray,
                   node_source: np.ndarray, pect: int, present: np.ndarray, is_interp: np.ndarray,
                   node_resid: np.ndarray, p: QCParams, c30: tuple[list, list] | None = None) -> pd.DataFrame:
    """Raw findings, one row each (``ROW_COLUMNS``)."""
    rows: list[dict] = []

    def add(kind, a, b, fish, score, details, peak=None):
        rows.append({"type": kind, "category": "identity" if kind in IDENTITY_TYPES else "geometry",
                     "fish": fish, "start": int(frames[a]), "end": int(frames[b]),
                     "peak": int(frames[a if peak is None else peak]), "score": float(score), "details": details})

    def add_runs(kind, mask, values, fish, fmt, min_len=1, use_max=True):
        for a, b in runs(mask, min_len):
            seg = values[a:b + 1]
            k = a + int(np.nanargmax(seg) if use_max else np.nanargmin(seg))
            add(kind, a, b, fish, values[k], fmt.format(v=values[k], n=b - a + 1), k)

    xyz = np.asarray(xyz, float)
    n_frames, n_fish = xyz.shape[:2]
    lengths = body_path_length(np.asarray(nodes, float), pect)
    measured = (node_source == 0).all(axis=2)
    lo = np.array([-p.volume_margin_mm] * 3)
    hi = np.array(p.volume_mm) + p.volume_margin_mm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, fid in enumerate(fish_ids):
            x = xyz[:, k]
            step = np.r_[np.nan, np.linalg.norm(np.diff(x, axis=0), axis=1)]
            add_runs("jump", np.nan_to_num(step) > p.max_step_mm, step, fid, "step {v:.1f} mm/frame")
            acc = np.r_[np.nan, np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1), np.nan]
            add_runs("spike", np.nan_to_num(acc) > p.max_acc_mm, acc, fid, "acceleration {v:.1f} mm/frame^2")
            dev = _local_deviation(x, p.local_window)
            z = _robust_z(dev)
            add_runs("local_outlier", (np.nan_to_num(z) > p.local_z) & (np.nan_to_num(dev) > p.local_floor_mm),
                     dev, fid, "{v:.1f} mm off the running median")
            ok = measured[:, k] & np.isfinite(lengths[:, k])
            if ok.sum() > 100:
                ratio = lengths[:, k] / np.median(lengths[ok, k])
                bad = ok & ((ratio < p.length_low) | (ratio > p.length_high))
                off = np.abs(np.log(np.where(bad, ratio, 1.0)))
                add_runs("body_length", bad, off, fid, "body length off the median by up to {v:.2f} (log ratio)",
                         p.length_min_frames)
            outside = np.nanmax(np.maximum(lo - x, x - hi), axis=1)
            add_runs("out_of_volume", np.nan_to_num(outside, nan=-1) > 0, outside, fid,
                     "{v:.1f} mm outside the calibrated volume")
            add_runs("gap", is_interp[:, k], np.ones(n_frames), fid, "interpolated {n} frames", p.gap_min_frames)
            few = (present[:, k].sum(axis=1) < 2) & np.isfinite(x).all(axis=1) & ~is_interp[:, k]
            views = present[:, k].sum(axis=1).astype(float)
            add_runs("few_views", few, views, fid, "fewer than two views for {n} frames", p.few_views_min_frames,
                     use_max=False)
            med = np.nanmedian(node_resid[:, k], axis=1)
            add_runs("residual", np.nan_to_num(med) > p.residual_mm2, med, fid,
                     "node residual up to {v:.1f} mm^2", p.residual_min_frames)
        for i in range(n_fish):
            for j in range(i + 1, n_fish):
                d = np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1)
                add_runs("close", np.nan_to_num(d, nan=np.inf) < p.close_mm, d, f"{fish_ids[i]}|{fish_ids[j]}",
                         "min distance {v:.1f} mm", use_max=False)

    if c30 is not None:
        index = {int(f): t for t, f in enumerate(frames)}
        joins, refusals = c30
        for j in joins:
            t = index.get(int(j["frame"]))
            if t is not None:
                add("c30_join", t, t, "", j.get("margin_mm") or 0.0,
                    f"slots re-permuted {j.get('permutation')} by {j.get('evidence')}")
        for r in refusals:
            a, b = r["zone"] if "zone" in r else (r["frame"], r["frame"])
            ta, tb = index.get(int(a)), index.get(int(b))
            if ta is not None and tb is not None:
                add("c30_refusal", ta, tb, "", r.get("margin_mm") or 0.0, f"stitch refused: {r['reason']}")
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def flag_mask(rows: pd.DataFrame, frames: np.ndarray, fish_ids: list[str]) -> np.ndarray:
    """``(F, fish)`` uint16 bit mask; a pair finding sets both fish, one without a fish sets all."""
    flags = np.zeros((len(frames), len(fish_ids)), np.uint16)
    index = {f: k for k, f in enumerate(fish_ids)}
    f0 = int(frames[0])
    for r in rows.itertuples(index=False):
        fish = [index[f] for f in str(r.fish).split("|") if f in index] if r.fish else list(range(len(fish_ids)))
        flags[r.start - f0:r.end - f0 + 1, fish] |= np.uint16(1 << TYPES.index(r.type))
    return flags


def merge_intervals(rows: pd.DataFrame, pad: int, first: int, last: int) -> pd.DataFrame:
    """Pad and union the findings over all fish (``INTERVAL_COLUMNS``)."""
    if rows.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)
    merged: list[dict] = []
    for r in rows.sort_values("start").itertuples(index=False):
        a, b = max(first, r.start - pad), min(last, r.end + pad)
        if merged and a <= merged[-1]["frame_end"]:
            m = merged[-1]
            m["frame_end"] = max(m["frame_end"], b)
        else:
            m = {"frame_start": a, "frame_end": b, "members": []}
            merged.append(m)
        m["members"].append(r)
    out = []
    for i, m in enumerate(merged):
        members = m["members"]
        fish = {f for r in members for f in str(r.fish).split("|") if f}
        out.append({"interval_id": f"T{i + 1:05d}", "frame_start": m["frame_start"], "frame_end": m["frame_end"],
                    "categories": ";".join(sorted({r.category for r in members})),
                    "types": ";".join(sorted({r.type for r in members})),
                    "fish": ";".join(sorted(fish)) or "*", "n_findings": len(members)})
    return pd.DataFrame(out, columns=INTERVAL_COLUMNS)


def _event_ranges(events: pd.DataFrame, kind: str) -> list[tuple[int, int]]:
    e = events[events["kind"] == kind]
    end = e["frame_end"].fillna(e["frame"]).astype(int)
    return [(int(min(a, b)), int(max(a, b))) for a, b in zip(e["frame"].astype(int), end)]


def score_against_review(intervals: pd.DataFrame, events: pd.DataFrame, start: int, last: int, fps: float,
                         tolerance_s: float = 0.5) -> dict:
    """How the merged intervals cover an identity pass's marks after ``start``.

    ``swap`` events are frames where the reviewer changed a name; ``danger`` ranges they marked unusable;
    ``pose_bad`` ranges where the body was wrong. A swap counts as covered by an ``identity`` interval within
    ``tolerance_s``; a range as covered when any interval of the matching category overlaps it.
    """
    tol = round(tolerance_s * fps)
    n = last - start + 1

    def mask(category: str | None) -> np.ndarray:
        m = np.zeros(n, bool)
        sel = intervals if category is None else intervals[intervals["categories"].str.contains(category)]
        for a, b in zip(sel["frame_start"], sel["frame_end"]):
            lo, hi = max(a, start) - start, min(b, last) - start
            if hi >= lo:
                m[lo:hi + 1] = True
        return m

    identity, geometry, anything = mask("identity"), mask("geometry"), mask(None)

    def hit(m: np.ndarray, a: int, b: int, pad: int = 0) -> bool:
        lo, hi = max(a - pad, start) - start, min(b + pad, last) - start
        return hi >= lo and bool(m[lo:hi + 1].any())

    events = events[events["frame"] >= start]
    swaps = sorted(set(events.loc[events["kind"] == "swap", "frame"].astype(int)))
    # Shares of frames inside the padded, merged intervals.
    out = {"from_frame": start, "tolerance_s": tolerance_s,
           "flagged_pct": round(100 * anything.mean(), 2), "identity_flagged_pct": round(100 * identity.mean(), 2),
           "geometry_flagged_pct": round(100 * geometry.mean(), 2),
           "swaps": len(swaps), "swaps_covered": sum(hit(identity, f, f, tol) for f in swaps),
           "swaps_missed": [f for f in swaps if not hit(identity, f, f, tol)]}
    for kind, m in (("danger", anything), ("pose_bad", geometry)):
        ranges = _event_ranges(events, kind)
        out[f"{kind}_ranges"] = len(ranges)
        out[f"{kind}_covered"] = sum(hit(m, a, b) for a, b in ranges)
    return out


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    write(tmp)
    os.replace(tmp, path)


def run_qc(package_dir: str | Path, *, c30_summary: str | Path | None = None, lam: float = 0.767,
           params: QCParams | None = None, log=print) -> dict:
    """Write the four outputs into ``package_dir``; returns the summary."""
    package_dir = Path(package_dir)
    p = params or QCParams()
    with np.load(package_dir / "review_data.npz") as z:
        frames, fish_ids = z["frames"], [str(f) for f in z["fish_ids"]]
        node_names = [str(n) for n in z["node_names"]]
        data = {k: z[k] for k in ("xyz", "nodes3d_filled", "node_source", "view_present", "is_interpolated",
                                  "node_resid")}
    manifest = json.loads((package_dir / "review_manifest.json").read_text(encoding="utf-8"))
    fps = float(manifest.get("fps", 166.0))
    c30 = read_c30(Path(c30_summary), lam) if c30_summary and Path(c30_summary).is_file() else None
    rows = find_anomalies(frames, fish_ids, data["xyz"], data["nodes3d_filled"], data["node_source"],
                          node_names.index("pectoral"), data["view_present"], data["is_interpolated"],
                          data["node_resid"], p, c30)
    first, last = int(frames[0]), int(frames[-1])
    intervals = merge_intervals(rows, p.pad_frames, first, last)
    flags = flag_mask(rows, frames, fish_ids)

    events_path = package_dir / "id_pass" / "human_events.csv"
    events = pd.read_csv(events_path) if events_path.is_file() else None
    start = first
    if events is not None and (events["kind"] == "analysis_start").any():
        start = int(events.loc[events["kind"] == "analysis_start", "frame"].iloc[-1])
    after = frames >= start
    summary = {
        "package": str(package_dir), "params": asdict(p), "fps": fps, "lambda_mm": lam,
        "c30_summary": str(c30_summary) if c30 is not None else None,
        "analysis_start": start, "findings": int(len(rows)), "intervals": int(len(intervals)),
        "by_type": {t: {"findings": int((rows["type"] == t).sum()),
                        "fish_frames_pct": round(100 * float(((flags[after] >> i) & 1).mean()), 3)}
                    for i, t in enumerate(TYPES)},
        "frames_flagged_unpadded_pct": round(100 * float(flags[after].any(axis=1).mean()), 2),
        "review": score_against_review(intervals, events, start, last, fps) if events is not None else None,
    }
    _atomic_write(package_dir / "trajectory_anomalies.csv", lambda f: rows.to_csv(f, index=False))
    _atomic_write(package_dir / "trajectory_anomaly_intervals.csv", lambda f: intervals.to_csv(f, index=False))
    _atomic_write(package_dir / "trajectory_flags.npz",
                  lambda f: np.savez_compressed(f, frames=frames, fish_ids=np.array(fish_ids), flags=flags,
                                                type_names=np.array(TYPES)))
    _atomic_write(package_dir / "trajectory_qc_summary.json",
                  lambda f: f.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"))
    log(f"[trajectory_qc] {package_dir.name}: {len(rows)} findings, {len(intervals)} intervals, "
        f"{summary['frames_flagged_unpadded_pct']}% of frames flagged (unpadded) after {start}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package_dir", type=Path)
    ap.add_argument("--c30-summary", type=Path, default=None, help="medaka-3d c30_summary.json (m3dfull only)")
    ap.add_argument("--lambda-mm", type=float, default=0.767)
    args = ap.parse_args(argv)
    summary = run_qc(args.package_dir, c30_summary=args.c30_summary, lam=args.lambda_mm)
    print(json.dumps({k: summary[k] for k in ("findings", "intervals", "frames_flagged_unpadded_pct", "review")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
