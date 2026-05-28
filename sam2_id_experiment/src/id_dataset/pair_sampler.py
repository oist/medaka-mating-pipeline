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
