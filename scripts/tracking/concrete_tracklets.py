from __future__ import annotations

"""Build CONSERVATIVE, high-confidence NN tracklets from SLEAP predictions.

Unlike ``track_simple_nn`` (which tries to *maintain* continuous IDs and bridges
through close contact -- the behaviour that lets identity drift smoothly onto the
wrong fish at a crossing), this tool only links a detection to a tracklet when the
assignment is UNAMBIGUOUS, and otherwise BREAKS the tracklet. A frame is
"trackable" only when:

  * exactly ``num_fish`` detections are present (after an optional top-K-by-score
    filter to drop reflections / spurious extras),
  * every detection scores >= ``min_score``,
  * the closest pair of fish is >= ``sep_bl`` body lengths apart (well separated),

and a frame-to-frame link is committed only when the one-to-one NN match is within
``jump_bl`` body lengths. Any violation ends the active tracklets; clean tracking
resumes as FRESH tracklets once the fish separate again.

The result is a set of "concrete" tracklets (each a contiguous, high-confidence
segment) plus the headline number: how many FRAGMENTS the video breaks into.
Downstream (3D geometry, appearance, manual review) can then stitch them.

Run::

    python -m scripts.tracking.concrete_tracklets <exp> --paths-config configs/paths.local.toml
    python -m scripts.tracking.concrete_tracklets --slp <top_cam.slp> --sep-bl 1.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths


def _min_pairwise(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return np.inf
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def build_concrete_tracklets(
    frames_data: dict[int, list[tuple[float, float, float]]],
    *,
    num_fish: int,
    body_length: float,
    sep_bl: float = 1.5,
    jump_bl: float = 1.0,
    min_score: float = 0.3,
    max_gap: int = 0,
    min_sep_by_frame: dict[int, float] | None = None,
    sep_threshold: float | None = None,
) -> tuple[list[dict], dict]:
    """frames_data: {frame: [(x, y, score), ...]} (already top-K filtered).

    A tracklet extends only on a *clean* frame (exactly num_fish detections, all
    scoring >= min_score, closest pair >= sep). A frame whose present detections
    come within `sep` (close contact) BREAKS the tracklet -- that's a crossing.
    A non-clean frame that is NOT close contact (a detection dropout / wrong count
    / low score, with the present fish still apart) is a bridgeable gap: tracklets
    stay alive for up to `max_gap` such frames and re-link when clean resumes.

    Returns (tracklets, stats). Each tracklet: {id, start, end, length, points}.
    """
    # separation threshold: an explicit value (e.g. a dimensionless stIFD gate in
    # sigma-units) or, by default, sep_bl body lengths. The jump gate stays in px.
    sep = float(sep_threshold) if sep_threshold is not None else float(sep_bl) * float(body_length)
    jump = float(jump_bl) * float(body_length)
    tracklets: list[dict] = []
    active: dict[int, dict] = {}     # id -> {"last": (x,y), "pts": [(f,x,y,s)]}
    next_id = 0
    last_clean: int | None = None
    n_clean = 0
    n_break = 0
    n_bridged = 0
    n_wrong = 0       # wrong detection count (reflections / misses)
    n_close = 0       # right count but a pair < sep (genuine close contact)
    n_lowscore = 0    # right count, separated, but a low-score detection

    def close():
        nonlocal active
        for t in active.values():
            pts = t["pts"]
            tracklets.append({
                "id": t["id"], "start": pts[0][0], "end": pts[-1][0],
                "length": len(pts), "points": pts,
            })
        active = {}

    def start_new(fi, dets):
        nonlocal next_id
        for x, y, s in dets:
            active[next_id] = {"id": next_id, "last": (x, y), "pts": [(fi, x, y, s)]}
            next_id += 1

    for fi in sorted(frames_data):
        dets = frames_data[fi]
        pts = np.array([[x, y] for x, y, _ in dets], dtype=float) if dets else np.empty((0, 2))
        # separation = externally-supplied (e.g. body-axis segment distance) or, by
        # default, the pairwise distance between the (x,y) carried in frames_data.
        pmin = (float(min_sep_by_frame.get(fi, np.inf))
                if min_sep_by_frame is not None else _min_pairwise(pts))
        clean = (
            len(dets) == num_fish
            and all(s >= min_score for _, _, s in dets)
            and pmin >= sep
        )
        is_close = len(pts) >= 2 and pmin < sep
        if len(dets) != num_fish:
            n_wrong += 1
        elif pmin < sep:
            n_close += 1
        elif not all(s >= min_score for _, _, s in dets):
            n_lowscore += 1
        if clean:
            n_clean += 1
            gap = (fi - last_clean) if last_clean is not None else None
            if active and gap is not None and gap <= max_gap + 1:
                tids = list(active.keys())
                tpos = np.array([active[t]["last"] for t in tids], dtype=float)
                cost = np.linalg.norm(tpos[:, None, :] - pts[None, :, :], axis=2)
                ri, ci = linear_sum_assignment(cost)
                if len(ri) == num_fish and all(cost[r, c] <= jump * gap for r, c in zip(ri, ci)):
                    for r, c in zip(ri, ci):
                        t = active[tids[r]]
                        x, y, s = dets[c]
                        t["last"] = (x, y)
                        t["pts"].append((fi, x, y, s))
                    if gap > 1:
                        n_bridged += 1
                else:
                    n_break += 1
                    close()
                    start_new(fi, dets)
            else:
                if active:                  # gap too long -> the run ended
                    n_break += 1
                close()
                start_new(fi, dets)
            last_clean = fi
        elif is_close:                      # crossing -> break immediately
            if active:
                n_break += 1
                close()
        else:                               # detection dropout / glitch -> bridgeable
            if active and last_clean is not None and (fi - last_clean) > max_gap:
                n_break += 1
                close()
    close()

    total = len(frames_data)
    lens = np.array([t["length"] for t in tracklets]) if tracklets else np.array([0])
    covered = int(lens.sum())
    stats = {
        "num_fish": num_fish,
        "body_length_px": round(float(body_length), 1),
        "sep_threshold_px": round(sep, 1),
        "jump_gate_px": round(jump, 1),
        "min_score": min_score,
        "max_gap": max_gap,
        "total_frames": total,
        "clean_frames": n_clean,
        "clean_frac": round(n_clean / max(total, 1), 3),
        "nonclean_wrong_count_frac": round(n_wrong / max(total, 1), 3),
        "nonclean_close_contact_frac": round(n_close / max(total, 1), 3),
        "nonclean_low_score_frac": round(n_lowscore / max(total, 1), 3),
        "n_fragments": len(tracklets),
        "fragments_per_fish": round(len(tracklets) / max(num_fish, 1), 1),
        "break_events": n_break,
        "bridged_gaps": n_bridged,
        "frag_len_frames_median": int(np.median(lens)),
        "frag_len_frames_mean": round(float(lens.mean()), 1),
        "frag_len_frames_max": int(lens.max()),
        "coverage_frac_of_fish_frames": round(covered / max(num_fish * total, 1), 3),
    }
    return tracklets, stats


def _nominal_speed(frames_data):
    """Median per-detection nearest-neighbour displacement (px/frame) -- a scene
    motion scale used for sigma when a fish has no established velocity yet."""
    prev = None
    sp = []
    for f in sorted(frames_data):
        cur = [(d[0], d[1]) for d in frames_data[f]]
        if prev:
            for c in cur:
                best = min((np.hypot(c[0] - p[0], c[1] - p[1]) for p in prev), default=np.inf)
                if np.isfinite(best):
                    sp.append(best)
        prev = cur
    return float(np.median(sp)) if sp else 1.0


def build_tracklets_edge_extended(
    frames_data: dict[int, list[tuple[float, float, float]]],
    *,
    num_fish: int,
    body_length: float,
    tau: float = 2.0,
    sigma0: float = 2.0,
    nominal_speed: float | None = None,
    jump_bl: float = 1.0,
    min_score: float = 0.3,
    max_gap: int = 5,
    sep_smooth_window: int = 5,
) -> tuple[list[dict], dict]:
    """Core-seed + per-tracklet velocity-standardized edge-extension.

    Tracklets are SEEDED only from individually-unambiguous detections (a fish clearly
    separated from the others) and EXTENDED frame-by-frame, per fish, using the
    tracklet's own velocity to predict-and-assign. A tracklet breaks only when ITS
    detection turns ambiguous:

        stIFD_i = (distance from det i to the nearest OTHER detection)
                  / sqrt(sigma_self^2 + sigma_rival^2),     sigma = sigma0 + speed

    where speed is the tracklet's own velocity (for a tracked fish) or nominal_speed
    otherwise. Break when stIFD_i < tau. So an uninvolved fish keeps tracking through a
    pairwise close approach -- unlike the frame-level gate, which breaks all tracks at
    once. frames_data: {frame: [(x, y, score), ...]} (positions = the bind anchor).

    The break test uses a rolling MEDIAN of the recent per-tracklet stIFD over
    `sep_smooth_window` frames, so a single-frame separation dip (localization jitter)
    doesn't cause break/reseed flicker; set sep_smooth_window=1 to disable.
    """
    if nominal_speed is None:
        nominal_speed = _nominal_speed(frames_data)
    jump = float(jump_bl) * float(body_length)
    tracklets: list[dict] = []
    active: dict[int, dict] = {}
    next_id = n_seed = n_break = frames_full = 0

    def close(t):
        pts = t["pts"]
        tracklets.append({"id": t["id"], "start": pts[0][0], "end": pts[-1][0],
                          "length": len(pts), "points": pts})

    for f in sorted(frames_data):
        dets = frames_data[f]
        n = len(dets)
        P = np.array([[d[0], d[1]] for d in dets], dtype=float) if n else np.empty((0, 2))
        scores = [d[2] for d in dets]
        nod = np.full(n, np.inf)            # nearest-other distance per detection
        nearest = np.full(n, -1, dtype=int)
        if n >= 2:
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            np.fill_diagonal(D, np.inf)
            nod = D.min(axis=1)
            nearest = D.argmin(axis=1)

        tids = list(active)
        assign = {}
        if tids and n:
            preds = np.array([[active[t]["last"][0] + active[t]["vel"][0] * (f - active[t]["lf"]),
                               active[t]["last"][1] + active[t]["vel"][1] * (f - active[t]["lf"])]
                              for t in tids], dtype=float)
            cost = np.linalg.norm(preds[:, None, :] - P[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                gap = max(1, f - active[tids[r]]["lf"])
                if cost[r, c] <= jump * gap and gap <= max_gap + 1:
                    assign[tids[r]] = int(c)
        vel_by_tid = {t: active[t]["vel"] for t in tids}   # snapshot (tracklets get deleted below)
        claimed = set(assign.values())
        extended = 0
        for t in tids:
            tr = active[t]
            if t in assign:
                i = assign[t]
                sig_t = sigma0 + float(np.hypot(*vel_by_tid[t]))
                sig_r = sigma0 + nominal_speed
                if n >= 2:
                    j = int(nearest[i])
                    for t2, c2 in assign.items():
                        if c2 == j:
                            sig_r = sigma0 + float(np.hypot(*vel_by_tid[t2]))
                            break
                    stifd = nod[i] / float(np.hypot(sig_t, sig_r))
                else:
                    stifd = np.inf
                # rolling-median over the tracklet's recent stIFD -> no single-frame flicker
                s = stifd if np.isfinite(stifd) else 1e3
                window = (tr["seps"] + [s])[-sep_smooth_window:]
                smoothed = float(np.median(window))
                if smoothed >= tau and scores[i] >= min_score:
                    x, y = float(P[i][0]), float(P[i][1])
                    gap = max(1, f - tr["lf"])
                    tr["vel"] = ((x - tr["last"][0]) / gap, (y - tr["last"][1]) / gap)
                    tr["last"] = (x, y); tr["lf"] = f
                    tr["pts"].append((f, x, y, scores[i]))
                    tr["seps"] = window
                    extended += 1
                else:
                    close(tr); del active[t]; n_break += 1
            elif f - tr["lf"] > max_gap:
                close(tr); del active[t]; n_break += 1
        # seed new tracklets from unclaimed, individually-unambiguous detections
        for i in range(n):
            if i in claimed or len(active) >= num_fish or scores[i] < min_score:
                continue
            if nod[i] / (np.sqrt(2.0) * (sigma0 + nominal_speed)) >= tau:
                x, y = float(P[i][0]), float(P[i][1])
                active[next_id] = {"id": next_id, "pts": [(f, x, y, scores[i])],
                                   "last": (x, y), "lf": f, "vel": (0.0, 0.0), "seps": []}
                next_id += 1; n_seed += 1; extended += 1
        if extended == num_fish:
            frames_full += 1
    for t in list(active.values()):
        close(t)

    total = len(frames_data)
    lens = np.array([t["length"] for t in tracklets]) if tracklets else np.array([0])
    covered = int(lens.sum())
    stats = {
        "tau": tau, "sigma0": sigma0, "sep_smooth_window": sep_smooth_window,
        "nominal_speed_px": round(float(nominal_speed), 2),
        "total_frames": total, "n_fragments": len(tracklets), "n_seeds": n_seed,
        "all_tracked_frames": frames_full,
        "all_tracked_frac": round(frames_full / max(total, 1), 3),
        "frag_len_frames_median": int(np.median(lens)),
        "frag_len_frames_mean": round(float(lens.mean()), 1),
        "frag_len_frames_max": int(lens.max()),
        "coverage_frac_of_fish_frames": round(covered / max(num_fish * total, 1), 3),
    }
    return tracklets, stats


def _load_frames_data(slp_path: Path, *, num_fish: int, top_k: bool,
                      start_frame: int, end_frame: int | None):
    """Load per-frame detections (centroid + score) and estimate body length."""
    import sleap_io as sio
    labels = sio.load_slp(str(slp_path))
    nodes = [n.name for n in labels.skeletons[0].nodes]
    mouth_i = nodes.index("mouth") if "mouth" in nodes else 0
    tail_i = nodes.index("tail") if "tail" in nodes else (1 if len(nodes) > 1 else 0)

    frames_data: dict[int, list[tuple[float, float, float]]] = {}
    bls: list[float] = []
    n_wrong_count = 0
    lfs = sorted(labels.labeled_frames, key=lambda f: f.frame_idx)
    for f in lfs:
        fi = int(f.frame_idx)
        if fi < start_frame or (end_frame is not None and fi > end_frame):
            continue
        dets = []
        for inst in f.instances:
            pn = inst.numpy()[:, :2]
            mask = np.isfinite(pn).all(axis=1)
            if not mask.any():
                continue
            centroid = pn[mask].mean(axis=0)
            score = getattr(inst, "score", None)
            score = float(score) if score is not None and np.isfinite(score) else 0.0
            dets.append((float(centroid[0]), float(centroid[1]), score))
            if mouth_i < len(pn) and tail_i < len(pn):
                m, t = pn[mouth_i], pn[tail_i]
                if np.isfinite(m).all() and np.isfinite(t).all():
                    bls.append(float(np.linalg.norm(m - t)))
        if len(dets) != num_fish:
            n_wrong_count += 1
        if top_k and len(dets) > num_fish:
            dets = sorted(dets, key=lambda d: d[2], reverse=True)[:num_fish]
        frames_data[fi] = dets
    body_length = float(np.median(bls)) if bls else float("nan")
    return frames_data, body_length, n_wrong_count


def write_tracklets_csv(tracklets: list[dict], out_path: Path) -> None:
    rows = []
    for t in tracklets:
        for fi, x, y, s in t["points"]:
            rows.append({"frame": fi, "tracklet_id": t["id"], "x": x, "y": y, "score": s})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["frame", "tracklet_id", "x", "y", "score"]).to_csv(out_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Conservative high-confidence NN tracklets.")
    p.add_argument("experiment", nargs="?", default=None)
    p.add_argument("--view", default="top_cam")
    p.add_argument("--slp", default=None)
    p.add_argument("--output", default=None, help="optional tracklets CSV")
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--sep-bl", type=float, default=1.5, help="min inter-fish separation in body lengths")
    p.add_argument("--jump-bl", type=float, default=1.0, help="max per-frame link distance in body lengths")
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--max-gap", type=int, default=0,
                   help="bridge up to this many non-close-contact dropout frames within a tracklet")
    p.add_argument("--body-length", type=float, default=None, help="override estimated body length (px)")
    p.add_argument("--no-top-k", action="store_true", help="keep all detections (don't drop extras by score)")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--fps", type=float, default=166.0)
    p.add_argument("--sweep", action="store_true", help="also report fragment counts over sep-bl in {1.0,1.5,2.0}")
    p.add_argument("--paths-config", default=None)
    p.add_argument("--sleap-output-dir", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    slp_path = Path(args.slp) if args.slp else None
    if slp_path is None:
        if args.experiment is None:
            raise SystemExit("pass an experiment name or --slp")
        local = load_local_paths(args.paths_config)
        out_dir = choose_path(args.sleap_output_dir, local.sleap_output_dir if local else None)
        slp_path = Path(out_dir) / args.experiment / f"{args.view}.slp"
    if not slp_path.exists():
        raise SystemExit(f"slp not found: {slp_path}")

    print(f"Loading {slp_path} ...")
    frames_data, bl_est, n_wrong = _load_frames_data(
        slp_path, num_fish=args.num_fish, top_k=not args.no_top_k,
        start_frame=args.start_frame, end_frame=args.end_frame,
    )
    bl = args.body_length if args.body_length else bl_est
    print(f"frames={len(frames_data)}  body_length~{bl:.1f}px  "
          f"frames_with_!= {args.num_fish}_raw_detections={n_wrong} "
          f"({100*n_wrong/max(len(frames_data),1):.0f}%)")

    tracklets, stats = build_concrete_tracklets(
        frames_data, num_fish=args.num_fish, body_length=bl,
        sep_bl=args.sep_bl, jump_bl=args.jump_bl, min_score=args.min_score,
        max_gap=args.max_gap,
    )
    print("\n=== concrete tracklet stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  frag_len_seconds_median: {stats['frag_len_frames_median']/args.fps:.2f}")
    print(f"  frag_len_seconds_mean:   {stats['frag_len_frames_mean']/args.fps:.2f}")

    if args.sweep:
        print("\n=== sep-bl sweep (n_fragments / clean_frac / median_frag_frames) ===")
        for s in (1.0, 1.5, 2.0):
            _, st = build_concrete_tracklets(
                frames_data, num_fish=args.num_fish, body_length=bl,
                sep_bl=s, jump_bl=args.jump_bl, min_score=args.min_score,
                max_gap=args.max_gap)
            print(f"  sep_bl={s}: {st['n_fragments']} frags  "
                  f"clean={st['clean_frac']}  median={st['frag_len_frames_median']}fr")

    if args.output:
        write_tracklets_csv(tracklets, Path(args.output))
        print(f"\nwrote tracklets -> {args.output}")


if __name__ == "__main__":
    main()
