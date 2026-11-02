"""Who closes the gap before spawning: the female, or the male.

Uses the identity-pass labels on a review package (mouth, pectoral, tail in
3D) and the BORIS spawn wrap. The predictor window ends when the wrap that
contains the spawn begins, and every wrap bout is left out, so contact during
mating is not counted as approach.

Her contribution on a frame is the component of her pectoral step along the
direction toward that male. His contribution is the component of his step
along the direction toward her. Heading is her mouth-minus-tail axis.

    .venv\\Scripts\\python.exe -m scripts.analysis.female_direction
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.review import id_segments as S
from scripts.review.ledger import (
    COVERAGE_COLUMNS,
    EVENT_COLUMNS,
    analysis_start,
    owner_series,
    read_csv,
    static_labels,
    swap_list,
    watched_intervals,
)

FPS = 166.0
MAX_STEP_MM = 6.0
HEAD_LO_MM = 15.0
HEAD_HI_MM = 40.0
NEAR_LO_MM = 15.0
NEAR_HI_MM = 100.0
ROLES = ("F", "M1", "M2")

ROOT = Path(__file__).resolve().parents[2]
BORIS_BUILD = ROOT / "report" / "20260630" / "boris_cohort" / "build_boris_cohort_data.py"

# Paths are absolute because the packages live on the bucket, not in the repo.
EXPERIMENTS = [
    ("K_1_1", r"Z:\medaka\review_packages\20251225_K_1_1_FCMC\s16_bottomup_3node_uncapped_m3dfull",
     ROOT / "archive" / "boris" / "20251225_K_1_1_FCMC.csv"),
    ("K_1_2", r"Z:\medaka\review_packages\20251225_K_1_2_FCMC\s16_bottomup_3node_uncapped_m3dfull",
     ROOT / "archive" / "boris" / "20251225_K_1_2_FCMC.csv"),
    ("K_1_3", r"Z:\medaka\review_packages\20251225_K_1_3_FCMC\s16_bottomup_3node_uncapped_m3dfull",
     ROOT / "archive" / "boris" / "20251225_K_1_3_FCMC.csv"),
    ("P_1_1", r"Z:\medaka\review_packages\20260112_P_1_1_FCMC\s14_bottomup_3node_reviewed_m3dfull",
     ROOT / "archive" / "boris" / "p2" / "20260112_P_1_1_FCMC.csv"),
    ("P_1_2", r"Z:\medaka\review_packages\20260113_P_1_2_FCMC\s14_bottomup_3node_reviewed_m3dfull",
     ROOT / "archive" / "boris" / "p2" / "20260112_P_1_2_FCMC.csv"),
    ("P_1_3", r"Z:\medaka\review_packages\20260112_P_1_3_FCMC\s14_bottomup_3node_reviewed_m3dfull",
     ROOT / "archive" / "boris" / "p2" / "20260112_P_1_3_FCMC.csv"),
]


def _boris_parser():
    spec = importlib.util.spec_from_file_location("boris_build", BORIS_BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_boris_file


def spawn_and_wraps(parse_boris_file, path: Path) -> tuple[float | None, int | None, str | None, np.ndarray]:
    """Spawn time, spawn frame, the male in the wrap that holds the spawn, and wrap frame ranges."""
    events, _meta = parse_boris_file(path)
    if events.empty:
        return None, None, None, np.zeros((0, 2), int)
    wraps = events[events["behavior"].str.lower().eq("wrap")].copy()
    ranges = []
    for _, row in wraps.iterrows():
        a, b = row["start_frame"], row["stop_frame"]
        if np.isfinite(a) and np.isfinite(b):
            ranges.append((int(min(a, b)), int(max(a, b))))
    spawn = events[events["behavior"].str.lower().eq("spawn")]
    if spawn.empty:
        return None, None, None, np.asarray(ranges, int).reshape(-1, 2)
    sp_t = float(spawn["start_time_s"].iloc[0])
    sp_f = int(spawn["start_frame"].iloc[0]) if np.isfinite(spawn["start_frame"].iloc[0]) else int(round(sp_t * FPS))
    holder = None
    best = None
    for _, row in wraps.iterrows():
        a, b = float(row["start_time_s"]), float(row["stop_time_s"])
        if a <= sp_t <= b:
            dur = float(row["duration_s"]) if np.isfinite(row["duration_s"]) else 0.0
            if best is None or dur > best:
                best = dur
                holder = str(row["subject"])
    male = {"m1tf": "M1", "m2tf": "M2"}.get(holder or "")
    return sp_t, sp_f, male, np.asarray(ranges, int).reshape(-1, 2)


def courtship_before(parse_boris_file, path: Path, stop_s: float | None) -> tuple[float, float]:
    events, _meta = parse_boris_file(path)
    if events.empty:
        return 0.0, 0.0
    court = events[events["category"].str.lower().eq("courtship") & events["subject"].isin(["m1tf", "m2tf"])]
    out = {}
    for subject in ("m1tf", "m2tf"):
        total = 0.0
        for _, row in court[court["subject"].eq(subject)].iterrows():
            a, b = float(row["start_time_s"]), float(row["stop_time_s"])
            dur = float(row["duration_s"]) if np.isfinite(row["duration_s"]) else 0.0
            if stop_s is None or b <= stop_s:
                total += dur
            elif a < stop_s:
                total += max(0.0, stop_s - a)
        out[subject] = total
    return out["m1tf"], out["m2tf"]


def in_ranges(frames: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    out = np.zeros(len(frames), dtype=bool)
    for a, b in ranges:
        out |= (frames >= a) & (frames <= b)
    return out


def gather(track_value: np.ndarray, role_of_track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``track_value (T, K, ...)`` gathered into roles F, M1, M2. Second return is valid (exactly one track)."""
    t = len(role_of_track)
    valid = np.ones(t, dtype=bool)
    index = np.zeros(t, dtype=int)
    pieces = []
    for role in range(3):
        match = role_of_track == role
        count = match.sum(axis=1)
        valid &= count == 1
        index = np.where(count == 1, match.argmax(axis=1), 0)
        pieces.append(track_value[np.arange(t), index])
    stacked = np.stack(pieces, axis=1)
    return stacked, valid


def analyse(folder: Path, boris: Path, parse_boris_file) -> dict:
    events = read_csv(folder / "id_pass" / "human_events.csv", EVENT_COLUMNS)
    coverage = read_csv(folder / "id_pass" / "coverage.csv", COVERAGE_COLUMNS)
    with np.load(folder / "review_data.npz") as z:
        frames = np.asarray(z["frames"])
        fish_ids = [str(x) for x in z["fish_ids"]]
        nodes = np.asarray(z["nodes3d_filled"], dtype=np.float32)
        source = np.asarray(z["node_source"])
        xyz = np.asarray(z["xyz"], dtype=np.float64)
        present = np.asarray(z["view_present"]) if "view_present" in z.files else np.isfinite(z["kp2d"]).all(axis=-1).any(axis=-1)
        interp = np.asarray(z["is_interpolated"]) if "is_interpolated" in z.files else np.zeros(xyz.shape[:2], dtype=bool)
        head_i = int(z["head_node"]) if "head_node" in z.files else 0
        tail_i = int(z["tail_node"]) if "tail_node" in z.files else 2
        names = [str(x) for x in z["node_names"]] if "node_names" in z.files else ["mouth", "pectoral", "tail"]
    pect_i = names.index("pectoral")

    labels = static_labels(events, fish_ids)
    swaps = swap_list(events, fish_ids)
    owner = owner_series(len(fish_ids), swaps, frames)
    # owner[t, k] is the corrected fish sitting on track k. Labels name corrected fish.
    role_of_track = np.full(owner.shape, -1, dtype=np.int8)
    for corrected, fish in enumerate(fish_ids):
        name = labels.get(fish, "")
        if name in ROLES:
            role_of_track[owner == corrected] = ROLES.index(name)

    start = analysis_start(events)
    danger = S.danger_map(xyz, present, interp, FPS)
    status = S.frame_status(
        int(frames[0]), len(frames), start, danger["danger"], danger["runs"],
        watched_intervals(coverage), [f for f, _a, _b in swaps],
        S.manual_ranges(events), FPS,
    )
    usable = np.isin(status, (S.WATCHED_SAFE, S.WATCHED_DANGER))

    pose = np.zeros(owner.shape, dtype=bool)
    id_index = {fish: i for i, fish in enumerate(fish_ids)}
    for a, b, fishes in S.pose_ranges(events):
        lo = int(np.searchsorted(frames, a, side="left"))
        hi = int(np.searchsorted(frames, b, side="right"))
        for fish in fishes:
            if fish in id_index:
                pose[lo:hi, id_index[fish]] = True

    sp_t, sp_f, spawn_male, wrap_ranges = spawn_and_wraps(parse_boris_file, boris)
    wrap_start_s = None
    if sp_t is not None and len(wrap_ranges):
        # The wrap whose frames contain the spawn; predictor ends at its first frame.
        holds = [(a, b) for a, b in wrap_ranges if a <= sp_f <= b]
        if holds:
            wrap_start_s = holds[0][0] / FPS
            end_frame = holds[0][0]
        else:
            end_frame = sp_f
            wrap_start_s = sp_t
    else:
        end_frame = int(frames[-1]) + 1

    window = usable & (frames < end_frame) & ~in_ranges(frames, wrap_ranges)
    pos, named = gather(nodes[:, :, pect_i, :], role_of_track)
    window &= named & np.isfinite(pos).all(axis=(1, 2))

    measured = np.isin(source[:, :, head_i], (0, 1)) & np.isin(source[:, :, tail_i], (0, 1))
    head_vec = nodes[:, :, head_i, :] - nodes[:, :, tail_i, :]
    head_len = np.linalg.norm(head_vec, axis=-1)
    head_ok_track = measured & (head_len >= HEAD_LO_MM) & (head_len <= HEAD_HI_MM) & ~pose
    match_f = role_of_track == 0
    female_track = np.where(match_f.sum(axis=1) == 1, match_f.argmax(axis=1), -1)
    heading_track_ok = np.zeros(len(frames), dtype=bool)
    has_track = female_track >= 0
    heading_track_ok[has_track] = head_ok_track[np.arange(len(frames))[has_track], female_track[has_track]]
    heads, _ = gather(head_vec, role_of_track)
    head_f = heads[:, 0, :]

    row = {
        "experiment": folder.parent.name,
        "spawn_male": spawn_male or "",
        "spawn_s": None if sp_t is None else round(sp_t, 1),
        "window_s": round(float(window.sum()) / FPS, 1),
        "labels": " ".join(f"{labels.get(f) or '?'}={f}" for f in fish_ids),
        "swaps": len(swaps),
        "flips": int((events["kind"] == "flip").sum()),
    }
    court_m1, court_m2 = courtship_before(parse_boris_file, boris, wrap_start_s if spawn_male else None)
    row["boris_court_M1_s"] = round(court_m1, 1)
    row["boris_court_M2_s"] = round(court_m2, 1)

    her_net = []
    him_net = []
    her_pos = []
    point = []
    for male in (1, 2):
        u = pos[:, male, :] - pos[:, 0, :]
        dist = np.linalg.norm(u, axis=1)
        step_f = np.diff(pos[:, 0, :], axis=0)
        step_m = np.diff(pos[:, male, :], axis=0)
        u0 = u[:-1]
        dist0 = dist[:-1]
        ok = (
            window[:-1] & window[1:]
            & (np.diff(frames) == 1)
            & np.isfinite(dist0)
            & (dist0 > 1.0)
            & (np.linalg.norm(step_f, axis=1) <= MAX_STEP_MM)
            & (np.linalg.norm(step_m, axis=1) <= MAX_STEP_MM)
        )
        uh = np.divide(u0, dist0[:, None], out=np.zeros_like(u0), where=dist0[:, None] > 1.0)
        her = np.einsum("ij,ij->i", uh, step_f)
        him = -np.einsum("ij,ij->i", uh, step_m)
        n = max(int(ok.sum()), 1)
        her_net.append(float(her[ok].sum()) / n * FPS)
        him_net.append(float(him[ok].sum()) / n * FPS)
        her_pos.append(float(her[ok].clip(min=0).sum()) / n * FPS)

        point.append(dist)

    d1, d2 = point
    both = window & np.isfinite(d1) & np.isfinite(d2)
    nearer_m1 = int((both & (d1 < d2)).sum())
    row["nearer_M1_frac"] = round(nearer_m1 / max(int(both.sum()), 1), 3)
    nearer_dist = np.minimum(d1, d2)
    orient = (
        window & heading_track_ok & named
        & np.isfinite(nearer_dist)
        & (nearer_dist >= NEAR_LO_MM)
        & (nearer_dist <= NEAR_HI_MM)
    )
    hnorm = np.linalg.norm(head_f, axis=1)
    hh = np.divide(head_f, hnorm[:, None], out=np.zeros_like(head_f), where=hnorm[:, None] > 0)
    cos1 = np.einsum("ij,ij->i", hh, pos[:, 1, :] - pos[:, 0, :])
    cos2 = np.einsum("ij,ij->i", hh, pos[:, 2, :] - pos[:, 0, :])
    # The einsum above is heading · (male - female), not the cosine. Divide by distance.
    cos1 = np.divide(cos1, d1, out=np.zeros_like(cos1), where=np.isfinite(d1) & (d1 > 1))
    cos2 = np.divide(cos2, d2, out=np.zeros_like(cos2), where=np.isfinite(d2) & (d2 > 1))
    n_orient = int(orient.sum())
    toward_m1 = int(((cos1 > cos2) & orient).sum())
    toward_clear = int((orient & (np.maximum(cos1, cos2) > 0.5) & (np.abs(cos1 - cos2) > 0.15)).sum())
    clear_m1 = int((orient & (cos1 > cos2) & (cos1 > 0.5) & ((cos1 - cos2) > 0.15)).sum())
    # In front: the male lies within 60 degrees of her head axis, not merely less behind her.
    row["in_front_M1_frac"] = round(float((orient & (cos1 > 0.5)).sum()) / n_orient, 3) if n_orient else None
    row["in_front_M2_frac"] = round(float((orient & (cos2 > 0.5)).sum()) / n_orient, 3) if n_orient else None
    row["mean_cos_M1"] = round(float(cos1[orient].mean()), 3) if n_orient else None
    row["mean_cos_M2"] = round(float(cos2[orient].mean()), 3) if n_orient else None

    row["her_net_M1_mm_s"] = round(her_net[0], 3)
    row["her_net_M2_mm_s"] = round(her_net[1], 3)
    row["him_net_M1_mm_s"] = round(him_net[0], 3)
    row["him_net_M2_mm_s"] = round(him_net[1], 3)
    row["her_approach_M1_mm_s"] = round(her_pos[0], 3)
    row["her_approach_M2_mm_s"] = round(her_pos[1], 3)
    row["heading_s"] = round(n_orient / FPS, 1)
    row["heading_M1_frac"] = round(toward_m1 / n_orient, 3) if n_orient else None
    row["heading_clear_M1_frac"] = round(clear_m1 / toward_clear, 3) if toward_clear else None
    row["heading_clear_s"] = round(toward_clear / FPS, 1)
    return row


def preference_timeline(n_bins: int = 10, court_mm: float = 30.0, duration_s: float = 2100.0) -> list[dict]:
    """Poster metric: share of <court_mm time spent with the male who is in the spawn wrap.

    Ten equal bins across the recording, identity-pass labels, usable frames only.
    """
    parse_boris_file = _boris_parser()
    series = []
    for name, folder, boris in EXPERIMENTS:
        folder = Path(folder)
        events = read_csv(folder / "id_pass" / "human_events.csv", EVENT_COLUMNS)
        coverage = read_csv(folder / "id_pass" / "coverage.csv", COVERAGE_COLUMNS)
        with np.load(folder / "review_data.npz") as z:
            frames = np.asarray(z["frames"])
            fish_ids = [str(x) for x in z["fish_ids"]]
            nodes = np.asarray(z["nodes3d_filled"], dtype=np.float32)
            xyz = np.asarray(z["xyz"], dtype=np.float64)
            present = np.asarray(z["view_present"]) if "view_present" in z.files else np.isfinite(z["kp2d"]).all(axis=-1).any(axis=-1)
            interp = np.asarray(z["is_interpolated"]) if "is_interpolated" in z.files else np.zeros(xyz.shape[:2], dtype=bool)
            names = [str(x) for x in z["node_names"]] if "node_names" in z.files else ["mouth", "pectoral", "tail"]
        pect_i = names.index("pectoral")
        labels = static_labels(events, fish_ids)
        swaps = swap_list(events, fish_ids)
        owner = owner_series(len(fish_ids), swaps, frames)
        role_of_track = np.full(owner.shape, -1, dtype=np.int8)
        for corrected, fish in enumerate(fish_ids):
            role_name = labels.get(fish, "")
            if role_name in ROLES:
                role_of_track[owner == corrected] = ROLES.index(role_name)
        start = analysis_start(events)
        danger = S.danger_map(xyz, present, interp, FPS)
        status = S.frame_status(
            int(frames[0]), len(frames), start, danger["danger"], danger["runs"],
            watched_intervals(coverage), [f for f, _a, _b in swaps],
            S.manual_ranges(events), FPS,
        )
        usable = np.isin(status, (S.WATCHED_SAFE, S.WATCHED_DANGER))
        _sp_t, _sp_f, spawn_male, _wraps = spawn_and_wraps(parse_boris_file, boris)
        pos, named = gather(nodes[:, :, pect_i, :], role_of_track)
        ok = usable & named & np.isfinite(pos).all(axis=(1, 2))
        d1 = np.linalg.norm(pos[:, 1, :] - pos[:, 0, :], axis=1)
        d2 = np.linalg.norm(pos[:, 2, :] - pos[:, 0, :], axis=1)
        if spawn_male == "M1":
            d_mate, d_other = d1, d2
        elif spawn_male == "M2":
            d_mate, d_other = d2, d1
        else:
            d_mate, d_other = d1, d2
        bins = []
        width = duration_s / n_bins
        for b in range(n_bins):
            t0, t1 = b * width, (b + 1) * width
            sel = ok & (frames >= t0 * FPS) & (frames < t1 * FPS)
            mate_s = float(np.count_nonzero(sel & (d_mate < court_mm))) / FPS
            other_s = float(np.count_nonzero(sel & (d_other < court_mm))) / FPS
            total = mate_s + other_s
            bins.append({
                "t": round((t0 + t1) / 2, 1),
                "index": round(mate_s / total, 4) if total else None,
                "mate_s": round(mate_s, 1),
                "other_s": round(other_s, 1),
            })
        series.append({
            "name": name,
            "group": "K" if name.startswith("K") else "P",
            "spawn_s": None if _sp_t is None else round(_sp_t, 1),
            "spawn_male": spawn_male or "",
            "bins": bins,
        })
    return series


def _side(a: float, b: float, eps: float = 0.05) -> str:
    if abs(a - b) <= eps:
        return "tie"
    return "M1" if a > b else "M2"


def main() -> None:
    parse_boris_file = _boris_parser()
    rows = []
    for name, folder, boris in EXPERIMENTS:
        print(f"reading {name} ...", flush=True)
        row = analyse(Path(folder), boris, parse_boris_file)
        row["name"] = name
        row["her_net_side"] = _side(row["her_net_M1_mm_s"], row["her_net_M2_mm_s"])
        row["him_net_side"] = _side(row["him_net_M1_mm_s"], row["him_net_M2_mm_s"])
        row["her_approach_side"] = _side(row["her_approach_M1_mm_s"], row["her_approach_M2_mm_s"], eps=0.02)
        if row["heading_M1_frac"] is None:
            row["heading_side"] = ""
        else:
            row["heading_side"] = _side(row["heading_M1_frac"], 1 - row["heading_M1_frac"], eps=0.02)
        row["boris_side"] = _side(row["boris_court_M1_s"], row["boris_court_M2_s"], eps=1.0)
        row["nearer_side"] = "M1" if row["nearer_M1_frac"] >= 0.5 else "M2"
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    dest = ROOT / "runs" / "female_direction" / "summary.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
