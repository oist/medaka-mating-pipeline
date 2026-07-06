from __future__ import annotations

"""
Compare a trained top-down pipeline vs a bottom-up model for robustness on medaka.

Round-7 (retrain_20260704) trains BOTH strategies from scratch on the same
round-1..6 annotation set and this script scores them head-to-head on the
IDENTICAL frames:

* top-down   = centroid (anchor=mouth) -> centered_instance, resolved from the
  two ``best.ckpt`` run dirs under ``<round_dir>/<view>/runs/``.
* bottom-up  = a single full-frame confmaps+PAFs model, resolved from the
  ``bottomup`` run dir under the same ``runs/`` folder.

What it does
------------
1. Loads the merged ground-truth labels from ``<round_dir>/<view>/labels.pkg.slp``
   (an embedded-frame package, so inference runs offline from the pkg itself).
2. Builds a DETERMINISTIC eval subset (fixed SHA1 hash of ``(video_index,
   frame_idx)``) so both strategies are scored on exactly the same frames.
   ``--difficult-only`` further restricts to close-contact frames (two GT fish
   whose centroids are within ``--close-contact-px``), which are the real
   round-6b stress test.
3. Runs top-down inference and bottom-up inference on those frames via the
   sleap-nn Python API (``sleap_nn.predict.run_inference`` with
   ``input_labels=<subset>``; no shelling out).
4. Matches predicted instances to GT (Hungarian on centroid distance) and
   computes, per strategy: detected vs GT counts (missed / extra fish),
   mean & p95 mouth error, tail error, OKS, a mouth<->tail SWAP rate, and a
   close-contact merge rate (two GT fish collapsed to one prediction).
5. Writes ``<round_dir>/eval_td_vs_bu_<view>.json`` + a short markdown table
   (``...<view>.md``) and prints a verdict line.

Leakage caveat
--------------
ALL of these frames were used to train BOTH strategies, so this is NOT a
held-out generalisation measurement. It measures (a) relative in-sample
robustness and (b) behaviour on the difficult close-contact frames -- i.e.
"which architecture degrades more gracefully where the fish overlap", not
absolute accuracy on unseen data.

Usage
-----
    python -m scripts.sleap.eval_topdown_vs_bottomup \
        --round-dir Z:/StephensU/medaka/sleap_models/retrain_20260704 \
        --view top --limit 300 --device cuda

    # only the close-contact stress frames:
    python -m scripts.sleap.eval_topdown_vs_bottomup \
        --round-dir Z:/StephensU/medaka/sleap_models/retrain_20260704 \
        --view side --difficult-only --device cuda

NOTE: this script only runs INFERENCE (needs the trained best.ckpt run dirs).
It does not train anything. The pure matching/metric helpers below are import-
safe without sleap_nn / torch so ``tests/test_eval_topdown_vs_bottomup.py`` can
exercise them on CPU in CI.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Pure geometry / matching / metric helpers (NO sleap_nn / torch imports here,
# so the pytest can import and exercise them on CPU without a model or GPU).
# ---------------------------------------------------------------------------

MOUTH = "mouth"
TAIL = "tail"


def instance_centroid(points: np.ndarray) -> np.ndarray:
    """Centroid of an instance = mean over visible (non-NaN) nodes.

    Args:
        points: (n_nodes, 2) array; missing nodes are NaN.

    Returns:
        (2,) centroid, or (nan, nan) if no node is visible.
    """
    pts = np.asarray(points, dtype=float)[:, :2]
    if np.all(np.isnan(pts)):
        return np.array([np.nan, np.nan])
    return np.nanmean(pts, axis=0)


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def centroid_cost_matrix(gt_pts: list[np.ndarray], pr_pts: list[np.ndarray]) -> np.ndarray:
    """Pairwise centroid Euclidean distance, shape (n_gt, n_pr).

    Instances whose centroid is undefined contribute an ``inf`` row/column so
    they never win a match.
    """
    gt_c = [instance_centroid(p) for p in gt_pts]
    pr_c = [instance_centroid(p) for p in pr_pts]
    cost = np.full((len(gt_c), len(pr_c)), np.inf)
    for i, gc in enumerate(gt_c):
        if np.any(np.isnan(gc)):
            continue
        for j, pc in enumerate(pr_c):
            if np.any(np.isnan(pc)):
                continue
            cost[i, j] = _euclid(gc, pc)
    return cost


def hungarian_match(
    cost: np.ndarray, gate: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """One-to-one match on ``cost`` (Hungarian), gated by ``gate`` pixels.

    Returns ``(matches, unmatched_gt, unmatched_pr)`` where matches is a list of
    ``(gt_idx, pr_idx, dist)`` with ``dist <= gate``.
    """
    n_gt, n_pr = cost.shape
    matched_gt: set[int] = set()
    matched_pr: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    if n_gt and n_pr:
        # linear_sum_assignment cannot ingest inf; replace with a large finite
        # sentinel and reject over-gate pairs afterwards.
        finite = np.where(np.isinf(cost), 1e12, cost)
        rows, cols = linear_sum_assignment(finite)
        for r, c in zip(rows, cols):
            d = float(cost[r, c])
            if np.isfinite(d) and d <= gate:
                matches.append((int(r), int(c), d))
                matched_gt.add(int(r))
                matched_pr.add(int(c))
    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pr = [j for j in range(n_pr) if j not in matched_pr]
    return matches, unmatched_gt, unmatched_pr


def node_error(gt_pt: np.ndarray, pr_pt: np.ndarray) -> Optional[float]:
    """Euclidean error for a single node, or None if either point is missing."""
    g = np.asarray(gt_pt, float)[:2]
    p = np.asarray(pr_pt, float)[:2]
    if np.any(np.isnan(g)) or np.any(np.isnan(p)):
        return None
    return _euclid(g, p)


def is_mouth_tail_swap(
    gt_mouth: np.ndarray,
    gt_tail: np.ndarray,
    pr_mouth: np.ndarray,
    pr_tail: np.ndarray,
) -> Optional[bool]:
    """True if the prediction has mouth/tail swapped relative to GT.

    A swap = predicted mouth is closer to the GT *tail* than to the GT mouth AND
    predicted tail is closer to the GT *mouth* than to the GT tail. Returns None
    if any of the four points is missing (cannot decide).
    """
    for pt in (gt_mouth, gt_tail, pr_mouth, pr_tail):
        if np.any(np.isnan(np.asarray(pt, float)[:2])):
            return None
    d_pm_gm = _euclid(pr_mouth, gt_mouth)
    d_pm_gt = _euclid(pr_mouth, gt_tail)
    d_pt_gm = _euclid(pr_tail, gt_mouth)
    d_pt_gt = _euclid(pr_tail, gt_tail)
    return (d_pm_gt < d_pm_gm) and (d_pt_gm < d_pt_gt)


def count_merged_gt(
    gt_pts: list[np.ndarray], pr_pts: list[np.ndarray], gate: float
) -> int:
    """Count GT fish collapsed onto a shared prediction (close-contact merge).

    Assign each GT to its NEAREST prediction (many-to-one allowed, within
    ``gate``). Any predicted instance that is the nearest for >= 2 GT fish has
    "merged" them; every GT beyond the first on such a prediction counts as a
    merged GT.
    """
    cost = centroid_cost_matrix(gt_pts, pr_pts)
    if cost.size == 0:
        return 0
    nearest_pr_for_gt: dict[int, int] = {}
    for gi in range(cost.shape[0]):
        row = cost[gi]
        if not np.any(np.isfinite(row)):
            continue
        pj = int(np.argmin(row))
        if np.isfinite(row[pj]) and row[pj] <= gate:
            nearest_pr_for_gt[gi] = pj
    from collections import Counter

    counts = Counter(nearest_pr_for_gt.values())
    return int(sum(c - 1 for c in counts.values() if c >= 2))


@dataclass
class MetricAccumulator:
    """Running metric tallies for one strategy across all matched frames."""

    n_frames: int = 0
    n_gt: int = 0
    n_pr: int = 0
    n_matched: int = 0
    n_missed: int = 0
    n_extra: int = 0
    n_merged_gt: int = 0
    n_swap_decidable: int = 0
    n_swap: int = 0
    mouth_errs: list[float] = field(default_factory=list)
    tail_errs: list[float] = field(default_factory=list)
    oks_vals: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        def _mean(x: list[float]) -> float:
            return float(np.mean(x)) if x else float("nan")

        def _p95(x: list[float]) -> float:
            return float(np.percentile(x, 95)) if x else float("nan")

        return {
            "n_frames": self.n_frames,
            "n_gt": self.n_gt,
            "n_detected": self.n_pr,
            "n_matched": self.n_matched,
            "n_missed": self.n_missed,
            "n_extra": self.n_extra,
            "n_merged_gt": self.n_merged_gt,
            "detect_rate": (self.n_matched / self.n_gt) if self.n_gt else float("nan"),
            "miss_rate": (self.n_missed / self.n_gt) if self.n_gt else float("nan"),
            "extra_rate": (self.n_extra / self.n_pr) if self.n_pr else float("nan"),
            "merge_rate": (self.n_merged_gt / self.n_gt) if self.n_gt else float("nan"),
            "swap_rate": (
                self.n_swap / self.n_swap_decidable
                if self.n_swap_decidable
                else float("nan")
            ),
            "mouth_err_mean": _mean(self.mouth_errs),
            "mouth_err_p95": _p95(self.mouth_errs),
            "tail_err_mean": _mean(self.tail_errs),
            "tail_err_p95": _p95(self.tail_errs),
            "oks_mean": _mean(self.oks_vals),
        }


def accumulate_frame(
    acc: MetricAccumulator,
    gt_pts: list[np.ndarray],
    pr_pts: list[np.ndarray],
    mouth_idx: int,
    tail_idx: int,
    gate: float,
    oks_fn=None,
) -> None:
    """Match one frame's instances and fold the per-pair metrics into ``acc``.

    ``gt_pts`` / ``pr_pts`` are lists of (n_nodes, 2) arrays. ``oks_fn`` is an
    optional callable ``(gt(1,n,2), pr(1,n,2)) -> (1,1) array``; if None, OKS is
    skipped (keeps the helper usable in tests without sleap_nn).
    """
    acc.n_frames += 1
    acc.n_gt += len(gt_pts)
    acc.n_pr += len(pr_pts)

    cost = centroid_cost_matrix(gt_pts, pr_pts)
    matches, unmatched_gt, unmatched_pr = hungarian_match(cost, gate)
    acc.n_matched += len(matches)
    acc.n_missed += len(unmatched_gt)
    acc.n_extra += len(unmatched_pr)
    acc.n_merged_gt += count_merged_gt(gt_pts, pr_pts, gate)

    for gi, pj, _d in matches:
        g = np.asarray(gt_pts[gi], float)
        p = np.asarray(pr_pts[pj], float)
        me = node_error(g[mouth_idx], p[mouth_idx])
        te = node_error(g[tail_idx], p[tail_idx])
        if me is not None:
            acc.mouth_errs.append(me)
        if te is not None:
            acc.tail_errs.append(te)
        swap = is_mouth_tail_swap(
            g[mouth_idx], g[tail_idx], p[mouth_idx], p[tail_idx]
        )
        if swap is not None:
            acc.n_swap_decidable += 1
            if swap:
                acc.n_swap += 1
        if oks_fn is not None:
            try:
                oks = float(oks_fn(g[None, :, :2], p[None, :, :2])[0, 0])
                if np.isfinite(oks):
                    acc.oks_vals.append(oks)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Deterministic eval-subset selection (also pure / testable).
# ---------------------------------------------------------------------------


def frame_hash(video_index: int, frame_idx: int) -> int:
    """Stable, order-independent hash of a (video, frame) key."""
    key = f"{int(video_index)}:{int(frame_idx)}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest(), 16)


def select_eval_indices(
    keys: list[tuple[int, int]], limit: Optional[int]
) -> list[int]:
    """Deterministically pick eval-frame positions from ``keys``.

    ``keys`` is the list of ``(video_index, frame_idx)`` for every candidate
    frame (already restricted by --difficult-only, in original order). We sort
    positions by ``frame_hash`` and keep the first ``limit`` (or all if None).
    The result is a set of ORIGINAL positions so both strategies score the same
    frames regardless of model.
    """
    order = sorted(range(len(keys)), key=lambda i: frame_hash(*keys[i]))
    if limit is not None and limit >= 0:
        order = order[:limit]
    return sorted(order)


def min_pairwise_centroid_distance(gt_pts: list[np.ndarray]) -> float:
    """Smallest centroid-to-centroid distance among GT instances (inf if <2)."""
    cents = [instance_centroid(p) for p in gt_pts]
    cents = [c for c in cents if not np.any(np.isnan(c))]
    if len(cents) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            best = min(best, _euclid(cents[i], cents[j]))
    return best


# ---------------------------------------------------------------------------
# sleap-nn-backed parts (imported lazily; NOT exercised by the pytest).
# ---------------------------------------------------------------------------

VIEW_TO_BUCKET = {"top": "top", "side": "side"}


def resolve_run_dir(view_dir: Path, wanted_type: str) -> Path:
    """Find the run dir under ``<view_dir>/runs`` whose config head == wanted_type.

    ``wanted_type`` is one of 'centroid', 'centered_instance', 'bottomup'.
    Robust to the round-tag / n<N> suffix naming (globs all runs and reads each
    training_config.yaml).
    """
    from omegaconf import OmegaConf
    from sleap_nn.config.utils import get_model_type_from_cfg

    runs = view_dir / "runs"
    if not runs.is_dir():
        raise SystemExit(
            f"no runs/ directory under {view_dir} (has the {wanted_type} model "
            f"been trained yet?)"
        )
    candidates: list[Path] = []
    for cand in sorted(runs.glob("*")):
        cfg = cand / "training_config.yaml"
        if not cfg.exists():
            continue
        try:
            model_type = get_model_type_from_cfg(OmegaConf.load(cfg.as_posix()))
        except Exception:
            continue
        if model_type == wanted_type:
            if not (cand / "best.ckpt").exists():
                raise SystemExit(
                    f"run dir {cand} is a {wanted_type} model but has no best.ckpt "
                    f"(training incomplete?)"
                )
            candidates.append(cand)
    if not candidates:
        raise SystemExit(
            f"no trained '{wanted_type}' run found under {runs}. Expected a run "
            f"dir whose training_config.yaml head_config is '{wanted_type}'."
        )
    if len(candidates) > 1:
        print(f"[warn] multiple '{wanted_type}' runs; using {candidates[-1].name}")
    return candidates[-1]


def build_subset_labels(labels, positions: list[int]):
    """Build a sio.Labels containing only ``positions`` labeled frames."""
    import sleap_io as sio

    sub = [labels.labeled_frames[i] for i in positions]
    return sio.Labels(
        labeled_frames=sub,
        videos=labels.videos,
        skeletons=labels.skeletons,
    )


def run_strategy(
    *,
    strategy: str,
    model_paths: list[Path],
    subset_labels,
    max_instances: int,
    batch_size: int,
    device: str,
):
    """Run one strategy's inference on the subset; return a predicted sio.Labels."""
    from sleap_nn.predict import run_inference

    print(f"\n--- {strategy} inference ({len(model_paths)} model dir(s)) ---")
    for mp in model_paths:
        print(f"    model: {mp}")
    out = run_inference(
        input_labels=subset_labels,
        model_paths=[str(m) for m in model_paths],
        only_labeled_frames=True,
        max_instances=max_instances,
        batch_size=batch_size,
        device=device,
        make_labels=True,
        output_path=None,
    )
    return out


def _instances_as_arrays(instances) -> list[np.ndarray]:
    arrs = []
    for inst in instances:
        pts = np.asarray(inst.numpy(), float)[:, :2]
        arrs.append(pts)
    return arrs


def evaluate_strategy(
    *,
    strategy: str,
    gt_subset_labels,
    pred_labels,
    mouth_idx: int,
    tail_idx: int,
    gate: float,
) -> dict:
    """Match predicted labels to GT by ``(video object, frame_idx)``.

    We deliberately do NOT use ``find_frame_pairs`` / ``labels.match`` here:
    those pair frames by ``video.filename`` + ``frame_idx``, but every embedded
    frame in the merged training pkg shares the SAME filename
    (``labels.pkg.slp``). Frames mined from different source clips (K_1_1 /
    K_1_2 / K_1_3) then collide on identical ``(filename, frame_idx)`` keys and
    most pairs are dropped, producing a spurious ~86% "missed" rate.

    ``run_inference`` reuses the same ``sio.Video`` objects as the input subset,
    so ``(id(video), frame_idx)`` is an exact, collision-free key. GT frames with
    no predicted counterpart are counted as fully missed.
    """
    from sleap_nn.evaluation import compute_oks

    acc = MetricAccumulator()
    pred_by_key: dict[tuple[int, int], object] = {}
    for pr_lf in pred_labels.labeled_frames:
        pred_by_key[(id(pr_lf.video), int(pr_lf.frame_idx))] = pr_lf

    # Fallback: if video objects were not preserved (0 key hits) but frame
    # counts line up, pair positionally — run_inference keeps input order.
    gt_frames = list(gt_subset_labels.labeled_frames)
    hits = sum(
        1 for gt_lf in gt_frames
        if (id(gt_lf.video), int(gt_lf.frame_idx)) in pred_by_key
    )
    positional = hits == 0 and len(pred_labels.labeled_frames) == len(gt_frames)

    for i, gt_lf in enumerate(gt_frames):
        gt_pts = _instances_as_arrays(gt_lf.instances)
        if not gt_pts:
            continue
        if positional:
            pr_lf = pred_labels.labeled_frames[i]
        else:
            pr_lf = pred_by_key.get((id(gt_lf.video), int(gt_lf.frame_idx)))
        pr_pts = _instances_as_arrays(pr_lf.instances) if pr_lf is not None else []
        accumulate_frame(
            acc, gt_pts, pr_pts, mouth_idx, tail_idx, gate, oks_fn=compute_oks
        )
    summ = acc.summary()
    summ["strategy"] = strategy
    return summ


# ---------------------------------------------------------------------------
# Verdict + reporting.
# ---------------------------------------------------------------------------

# (metric_key, "lower"|"higher", human label) -- robustness-critical metrics.
VERDICT_METRICS = [
    ("swap_rate", "lower", "mouth<->tail swap rate"),
    ("merge_rate", "lower", "close-contact merge rate"),
    ("miss_rate", "lower", "missed-fish rate"),
    ("extra_rate", "lower", "extra-fish rate"),
    ("mouth_err_p95", "lower", "mouth error p95"),
    ("tail_err_p95", "lower", "tail error p95"),
    ("oks_mean", "higher", "mean OKS"),
]


def decide_verdict(td: dict, bu: dict) -> dict:
    td_wins = 0
    bu_wins = 0
    per_metric = []
    for key, direction, label in VERDICT_METRICS:
        a = td.get(key, float("nan"))
        b = bu.get(key, float("nan"))
        winner = "tie"
        if np.isfinite(a) and np.isfinite(b) and a != b:
            better_is_smaller = direction == "lower"
            td_better = (a < b) if better_is_smaller else (a > b)
            winner = "top_down" if td_better else "bottom_up"
            if winner == "top_down":
                td_wins += 1
            else:
                bu_wins += 1
        per_metric.append(
            {"metric": key, "label": label, "top_down": a, "bottom_up": b, "winner": winner}
        )
    if td_wins > bu_wins:
        overall = "top_down"
    elif bu_wins > td_wins:
        overall = "bottom_up"
    else:
        overall = "tie"
    return {
        "overall": overall,
        "top_down_wins": td_wins,
        "bottom_up_wins": bu_wins,
        "per_metric": per_metric,
    }


def _fmt(v: float) -> str:
    if not isinstance(v, (int, float)) or not np.isfinite(v):
        return "n/a"
    return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"


def build_markdown(view: str, td: dict, bu: dict, verdict: dict, meta: dict) -> str:
    rows = [
        ("frames scored", "n_frames"),
        ("GT fish", "n_gt"),
        ("detected fish", "n_detected"),
        ("missed rate", "miss_rate"),
        ("extra rate", "extra_rate"),
        ("merge rate", "merge_rate"),
        ("swap rate", "swap_rate"),
        ("mouth err mean", "mouth_err_mean"),
        ("mouth err p95", "mouth_err_p95"),
        ("tail err mean", "tail_err_mean"),
        ("tail err p95", "tail_err_p95"),
        ("mean OKS", "oks_mean"),
    ]
    lines = [
        f"# Top-down vs bottom-up -- view `{view}`",
        "",
        f"- round dir: `{meta['round_dir']}`",
        f"- difficult-only: {meta['difficult_only']} "
        f"(close-contact <= {meta['close_contact_px']} px)",
        f"- frames scored: {td.get('n_frames')} (deterministic hash subset, "
        f"limit={meta['limit']})",
        f"- match gate: {meta['match_gate_px']} px",
        "",
        "| metric | top-down | bottom-up |",
        "|---|---|---|",
    ]
    for label, key in rows:
        lines.append(f"| {label} | {_fmt(td.get(key))} | {_fmt(bu.get(key))} |")
    lines.append("")
    lines.append(
        f"**Verdict: {verdict['overall']}** "
        f"(top-down wins {verdict['top_down_wins']}, "
        f"bottom-up wins {verdict['bottom_up_wins']} of "
        f"{len(VERDICT_METRICS)} robustness metrics)"
    )
    lines.append("")
    lines.append("> Leakage caveat: all frames were used to train both models; "
                 "this measures relative in-sample robustness + difficult-frame "
                 "behaviour, not held-out generalisation.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--round-dir", required=True, type=Path,
                    help="Round dir containing <view>/labels.pkg.slp and <view>/runs/.")
    ap.add_argument("--view", required=True, choices=["top", "side"])
    ap.add_argument("--difficult-only", action="store_true",
                    help="Restrict to close-contact frames (round-6b stress test).")
    ap.add_argument("--close-contact-px", type=float, default=80.0,
                    help="GT centroid gap (px) below which a frame is 'close-contact'. "
                         "Default 80 (< one body length). Data-driven proxy for the "
                         "round-6b difficult set.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap eval frames (deterministic lowest-hash subset). "
                         "Omit for all frames.")
    ap.add_argument("--match-gate-px", type=float, default=150.0,
                    help="Max GT<->pred centroid distance to count as a match.")
    ap.add_argument("--max-instances", type=int, default=3,
                    help="Max instances per frame (3 fish).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda",
                    help="Torch device: 'cuda', 'cpu', 'auto', 'cuda:0', ...")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Override output JSON path "
                         "(default <round_dir>/eval_td_vs_bu_<view>.json).")
    args = ap.parse_args(argv)

    round_dir: Path = args.round_dir
    view: str = args.view
    view_dir = round_dir / view
    gt_path = view_dir / "labels.pkg.slp"
    if not gt_path.exists():
        raise SystemExit(f"ground-truth labels not found: {gt_path}")

    # Resolve run dirs up front so we fail fast with a clear message.
    centroid_dir = resolve_run_dir(view_dir, "centroid")
    instance_dir = resolve_run_dir(view_dir, "centered_instance")
    bottomup_dir = resolve_run_dir(view_dir, "bottomup")
    print(f"top-down   : {centroid_dir.name} + {instance_dir.name}")
    print(f"bottom-up  : {bottomup_dir.name}")

    import sleap_io as sio

    print(f"\nloading GT: {gt_path}")
    labels = sio.load_slp(str(gt_path))
    node_names = [n.name for n in labels.skeletons[0].nodes]
    if MOUTH not in node_names or TAIL not in node_names:
        raise SystemExit(
            f"skeleton must contain '{MOUTH}' and '{TAIL}' nodes; got {node_names}"
        )
    mouth_idx = node_names.index(MOUTH)
    tail_idx = node_names.index(TAIL)
    vid_index = {id(v): i for i, v in enumerate(labels.videos)}

    # Candidate frames (optionally restricted to close-contact / difficult).
    # The deterministic-selection key is (video_index, frame_idx) in the GT
    # labels; it is only used to pick WHICH frames to score, not to match.
    cand_positions: list[int] = []
    keys: list[tuple[int, int]] = []
    for i, lf in enumerate(labels.labeled_frames):
        gt_pts = _instances_as_arrays(lf.user_instances)
        if not gt_pts:
            continue
        if args.difficult_only:
            if min_pairwise_centroid_distance(gt_pts) > args.close_contact_px:
                continue
        cand_positions.append(i)
        keys.append((vid_index.get(id(lf.video), -1), int(lf.frame_idx)))

    if not cand_positions:
        raise SystemExit(
            "no candidate frames after filtering "
            f"(difficult_only={args.difficult_only}, "
            f"close_contact_px={args.close_contact_px})."
        )

    sel_local = select_eval_indices(keys, args.limit)
    eval_positions = [cand_positions[i] for i in sel_local]
    print(f"candidate frames: {len(cand_positions)}  ->  eval subset: "
          f"{len(eval_positions)} (limit={args.limit})")

    # One GT subset object for matching (find_frame_pairs reads user_instances).
    gt_subset = build_subset_labels(labels, eval_positions)

    # --- top-down (build a fresh input subset so inference can't consume the
    #     matching copy) ---
    td_pred = run_strategy(
        strategy="top_down",
        model_paths=[centroid_dir, instance_dir],
        subset_labels=build_subset_labels(labels, eval_positions),
        max_instances=args.max_instances,
        batch_size=args.batch_size,
        device=args.device,
    )
    td = evaluate_strategy(
        strategy="top_down", gt_subset_labels=gt_subset, pred_labels=td_pred,
        mouth_idx=mouth_idx, tail_idx=tail_idx, gate=args.match_gate_px,
    )

    # --- bottom-up ---
    bu_pred = run_strategy(
        strategy="bottom_up",
        model_paths=[bottomup_dir],
        subset_labels=build_subset_labels(labels, eval_positions),
        max_instances=args.max_instances,
        batch_size=args.batch_size,
        device=args.device,
    )
    bu = evaluate_strategy(
        strategy="bottom_up", gt_subset_labels=gt_subset, pred_labels=bu_pred,
        mouth_idx=mouth_idx, tail_idx=tail_idx, gate=args.match_gate_px,
    )

    verdict = decide_verdict(td, bu)

    meta = {
        "round_dir": str(round_dir),
        "view": view,
        "gt_path": str(gt_path),
        "difficult_only": args.difficult_only,
        "close_contact_px": args.close_contact_px,
        "limit": args.limit,
        "match_gate_px": args.match_gate_px,
        "max_instances": args.max_instances,
        "n_eval_frames": len(eval_positions),
        "top_down_run_dirs": [str(centroid_dir), str(instance_dir)],
        "bottom_up_run_dir": str(bottomup_dir),
    }
    caveats = [
        "All eval frames were used to train BOTH strategies -> in-sample; this is "
        "relative robustness + difficult-frame behaviour, NOT held-out generalisation.",
        "--difficult-only uses a data-driven close-contact proxy (min GT centroid "
        "gap <= close_contact_px), which approximates but is not identical to the "
        "exact round-6b difficult frame set.",
        "Identity is NOT evaluated here (Hungarian is per-frame on geometry); this "
        "scores detection + keypoint placement robustness, not track identity.",
    ]

    result = {
        "meta": meta,
        "strategies": {"top_down": td, "bottom_up": bu},
        "verdict": verdict,
        "caveats": caveats,
    }

    out_json = args.out_json or (round_dir / f"eval_td_vs_bu_{view}.json")
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(view, td, bu, verdict, meta), encoding="utf-8")

    print("\n=== summary (top-down | bottom-up) ===")
    for label, key in [
        ("missed rate", "miss_rate"), ("extra rate", "extra_rate"),
        ("merge rate", "merge_rate"), ("swap rate", "swap_rate"),
        ("mouth p95", "mouth_err_p95"), ("tail p95", "tail_err_p95"),
        ("mean OKS", "oks_mean"),
    ]:
        print(f"  {label:<12} {_fmt(td.get(key)):>10} | {_fmt(bu.get(key)):>10}")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")
    print(
        f"\nVERDICT [{view}]: {verdict['overall'].upper()} is more robust "
        f"(top-down {verdict['top_down_wins']} vs bottom-up "
        f"{verdict['bottom_up_wins']} of {len(VERDICT_METRICS)} metrics)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
