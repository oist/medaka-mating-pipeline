from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
import pandas as pd


def _sample_pairs(items: list[int], n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    all_pairs = list(itertools.combinations(items, 2))
    if len(all_pairs) <= n:
        return all_pairs
    idx = rng.choice(len(all_pairs), size=n, replace=False)
    return [all_pairs[int(i)] for i in idx]


def sample_constrained_pairs(
    crop_metadata: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    """Self-supervised pairs from must-link / cannot-link constraints, no IDs needed.

    Refines :func:`sample_positive_negative_pairs`. The original draws negatives from
    *span-overlapping* tracklets, which can mislabel two fragments of the SAME fish as
    "different" whenever their spans overlap but they never share a frame (a false
    negative that poisons the embedding). Here negatives are drawn ONLY from crops in
    the **same frame** but **different tracklets** -- two instances in one frame are
    provably different fish, so the cannot-link is exact.

    Constraints:
      * must-link  (label 1) -- crops in the same clean segment / tracklet,
      * cannot-link (label 0) -- crops in the same frame, different tracklet.

    Optional area-spike guard: drop crops whose ``area_ratio_prev`` exceeds
    ``max_area_ratio`` (a merged/occluded mask whose area jumped), per the user's
    "exclude segments with anomalous area" precaution.

    ``crop_metadata`` needs columns ``crop_id, frame, tracklet_id, qc_status`` (and
    optionally ``qc_segment_id``, ``area_ratio_prev``). Returns
    ``DataFrame[crop_a_id, crop_b_id, label, reason]``.
    """
    clean = crop_metadata[crop_metadata["qc_status"] == "clean_train"].copy()
    max_area_ratio = config.get("max_area_ratio")
    if max_area_ratio is not None and "area_ratio_prev" in clean.columns:
        ar = pd.to_numeric(clean["area_ratio_prev"], errors="coerce")
        clean = clean[~(ar > float(max_area_ratio))].copy()
    if clean.empty:
        return pd.DataFrame(columns=["crop_a_id", "crop_b_id", "label", "reason"])

    rng = np.random.default_rng(int(config.get("random_seed", 13)))
    n_pos = int(config.get("positive_pairs_per_tracklet", 1000))
    n_neg_per_frame = int(config.get("negative_pairs_per_frame", 6))
    n_neg_total = config.get("max_negative_pairs")

    rows: list[dict] = []
    # must-link positives: same clean segment (preferred) else tracklet
    pos_col = "qc_segment_id" if "qc_segment_id" in clean.columns else "tracklet_id"
    pos_src = clean[clean[pos_col].astype(int) >= 0] if pos_col == "qc_segment_id" else clean
    for group_id, grp in pos_src.groupby(pos_col):
        crop_ids = grp.sort_values("frame")["crop_id"].astype(int).tolist()
        tracklet_ids = grp["tracklet_id"].unique()
        tid = int(tracklet_ids[0]) if len(tracklet_ids) else -1
        for a, b in _sample_pairs(crop_ids, n_pos, rng):
            rows.append({"crop_a_id": int(a), "crop_b_id": int(b), "label": 1,
                         "reason": f"same_segment:{int(group_id)}:tracklet:{tid}"})

    # cannot-link negatives: same frame, different tracklet
    neg_rows: list[dict] = []
    for frame, grp in clean.groupby("frame"):
        recs = grp[["crop_id", "tracklet_id"]].astype(int).to_dict("records")
        frame_pairs = [
            (a["crop_id"], b["crop_id"], a["tracklet_id"], b["tracklet_id"])
            for a, b in itertools.combinations(recs, 2)
            if a["tracklet_id"] != b["tracklet_id"]
        ]
        if len(frame_pairs) > n_neg_per_frame:
            idx = rng.choice(len(frame_pairs), size=n_neg_per_frame, replace=False)
            frame_pairs = [frame_pairs[int(i)] for i in idx]
        for a, b, ta, tb in frame_pairs:
            neg_rows.append({"crop_a_id": int(a), "crop_b_id": int(b), "label": 0,
                             "reason": f"same_frame:{int(frame)}:tracklets:{ta}:{tb}"})
    if n_neg_total is not None and len(neg_rows) > int(n_neg_total):
        idx = rng.choice(len(neg_rows), size=int(n_neg_total), replace=False)
        neg_rows = [neg_rows[int(i)] for i in idx]
    rows.extend(neg_rows)

    pairs = pd.DataFrame(rows, columns=["crop_a_id", "crop_b_id", "label", "reason"])
    if not pairs.empty:
        pairs = pairs.sample(frac=1.0, random_state=int(config.get("random_seed", 13))).reset_index(drop=True)
    return pairs


def sample_positive_negative_pairs(
    crop_metadata: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    clean = crop_metadata[crop_metadata["qc_status"] == "clean_train"].copy()
    if clean.empty:
        return pd.DataFrame(columns=["crop_a_id", "crop_b_id", "label", "reason"])

    rng = np.random.default_rng(int(config.get("random_seed", 13)))
    n_pos = int(config.get("positive_pairs_per_tracklet", 1000))
    n_neg = int(config.get("negative_pairs_per_tracklet_pair", 500))

    rows: list[dict] = []
    positive_group_col = "qc_segment_id" if "qc_segment_id" in clean.columns else "tracklet_id"
    if positive_group_col == "qc_segment_id":
        clean = clean[clean["qc_segment_id"].astype(int) >= 0].copy()
    by_positive_group = {
        int(group_id): grp.sort_values("frame")["crop_id"].astype(int).tolist()
        for group_id, grp in clean.groupby(positive_group_col)
    }
    for group_id, crop_ids in by_positive_group.items():
        tracklet_ids = clean[clean[positive_group_col].astype(int) == group_id]["tracklet_id"].unique()
        tracklet_id = int(tracklet_ids[0]) if len(tracklet_ids) else -1
        for a, b in _sample_pairs(crop_ids, n_pos, rng):
            rows.append({
                "crop_a_id": a,
                "crop_b_id": b,
                "label": 1,
                "reason": f"same_clean_segment:{group_id}:tracklet:{tracklet_id}",
            })

    spans = clean.groupby("tracklet_id")["frame"].agg(["min", "max"]).reset_index()
    by_track = {
        int(tid): grp.sort_values("frame")["crop_id"].astype(int).tolist()
        for tid, grp in clean.groupby("tracklet_id")
    }
    for left, right in itertools.combinations(spans.to_dict("records"), 2):
        if int(left["max"]) < int(right["min"]) or int(right["max"]) < int(left["min"]):
            continue
        a_ids = by_track[int(left["tracklet_id"])]
        b_ids = by_track[int(right["tracklet_id"])]
        all_pairs = [(a, b) for a in a_ids for b in b_ids]
        if len(all_pairs) > n_neg:
            idx = rng.choice(len(all_pairs), size=n_neg, replace=False)
            all_pairs = [all_pairs[int(i)] for i in idx]
        for a, b in all_pairs:
            rows.append({
                "crop_a_id": int(a),
                "crop_b_id": int(b),
                "label": 0,
                "reason": f"coexisting_clean_tracklets:{int(left['tracklet_id'])}:{int(right['tracklet_id'])}",
            })

    pairs = pd.DataFrame(rows, columns=["crop_a_id", "crop_b_id", "label", "reason"])
    if not pairs.empty:
        pairs = pairs.sample(frac=1.0, random_state=int(config.get("random_seed", 13))).reset_index(drop=True)
    return pairs
