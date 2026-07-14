from __future__ import annotations

"""Read-only audit of the historical K_1_2 appearance-classifier dataset.

The script never modifies its input datasets or checkpoints.  It reconstructs the
historical train/validation assignments, enriches the corrected crop table with the
raw-pool and tracklet-bind provenance that still exists, and writes all results to a
new output directory.

The audit is intentionally conservative: ``clean_train`` is an automatic pipeline
status, not an independently validated biological-ID label.  Samples are therefore
reported as ``ambiguous`` unless an objective structural defect makes them ``reject``.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import pandas as pd


EXPERIMENT = "20251225_K_1_2_FCMC"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="corrected crop dataset")
    p.add_argument("--raw-dataset", required=True, help="pre-correction pooled crop dataset")
    p.add_argument("--tracklet-bound-csv", required=True)
    p.add_argument("--model-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--near-duplicate-hamming", type=int, default=4)
    p.add_argument("--skip-visuals", action="store_true")
    return p


def _female_split(meta: pd.DataFrame, val_frac: float = 0.25) -> pd.Series:
    """Exactly reproduce scripts/37_train_female_classifier.py (seed 0)."""
    rng = np.random.default_rng(0)
    tracks = meta["view_tracklet"].unique()
    n_val = max(1, int(len(tracks) * val_frac))
    val_tracks = set(int(v) for v in rng.choice(tracks, size=n_val, replace=False))
    return pd.Series(
        np.where(meta["view_tracklet"].astype(int).isin(val_tracks), "validation", "train"),
        index=meta.index,
        name="female_model_split",
    )


def _male_split(meta: pd.DataFrame, val_frac: float = 0.2) -> pd.Series:
    """Exactly reproduce scripts/39_train_identity_classifier.py (seed 13)."""
    male = meta[meta["global_id"].astype(int) != 1].copy()
    male["label_idx"] = male["global_id"].astype(int).map({0: 0, 2: 1})
    rng = np.random.default_rng(13)
    val_groups: set[int] = set()
    group_labels = male.groupby("view_tracklet")["label_idx"].agg(lambda s: int(s.mode().iloc[0]))
    for _label, groups in group_labels.groupby(group_labels):
        group_ids = groups.index.to_numpy().copy()
        rng.shuffle(group_ids)
        n_val = int(round(len(group_ids) * val_frac))
        if len(group_ids) > 1:
            n_val = min(max(n_val, 1), len(group_ids) - 1)
        else:
            n_val = 0
        val_groups.update(int(v) for v in group_ids[:n_val])
    out = pd.Series("not_applicable", index=meta.index, name="male_model_split")
    out.loc[male.index] = np.where(
        male["view_tracklet"].astype(int).isin(val_groups), "validation", "train"
    )
    return out


def _load_crops(dataset: Path) -> tuple[h5py.File, h5py.Dataset]:
    h5 = h5py.File(dataset / "crops.h5", "r")
    key = "crops" if "crops" in h5 else "images"
    return h5, h5[key]


def _hash_crops(crops: h5py.Dataset, batch_size: int = 256) -> tuple[list[str], np.ndarray]:
    exact: list[str] = []
    dhash = np.zeros(len(crops), dtype=np.uint64)
    for start in range(0, len(crops), batch_size):
        batch = np.asarray(crops[start : start + batch_size], dtype=np.uint8)
        for offset, image in enumerate(batch):
            exact.append(hashlib.sha256(image.tobytes()).hexdigest())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
            value = 0
            for bit in bits:
                value = (value << 1) | int(bit)
            dhash[start + offset] = np.uint64(value)
    return exact, dhash


class _BKNode:
    def __init__(self, value: int) -> None:
        self.value = int(value)
        self.children: dict[int, _BKNode] = {}

    def add(self, value: int) -> None:
        node = self
        value = int(value)
        while True:
            distance = (node.value ^ value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value)
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        stack = [self]
        while stack:
            node = stack.pop()
            distance = (node.value ^ int(value)).bit_count()
            if distance <= radius:
                result.append((distance, node.value))
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node.children.items() if low <= edge <= high)
        return result


def _near_duplicate_matches(
    meta: pd.DataFrame,
    hashes: np.ndarray,
    split_col: str,
    radius: int,
) -> pd.DataFrame:
    use = meta[meta[split_col].isin(["train", "validation"])].copy()
    train_ids = use.index[use[split_col] == "train"].to_numpy(dtype=int)
    val_ids = use.index[use[split_col] == "validation"].to_numpy(dtype=int)
    by_hash: dict[int, list[int]] = defaultdict(list)
    for idx in val_ids:
        by_hash[int(hashes[idx])].append(int(idx))
    unique_hashes = list(by_hash)
    if not unique_hashes:
        return pd.DataFrame()
    tree = _BKNode(unique_hashes[0])
    for value in unique_hashes[1:]:
        tree.add(value)
    rows = []
    for train_idx in train_ids:
        matches = tree.query(int(hashes[train_idx]), radius)
        if not matches:
            continue
        distance, value = min(matches, key=lambda item: item[0])
        val_idx = by_hash[value][0]
        rows.append(
            {
                "split_definition": split_col,
                "train_crop_id": int(train_idx),
                "validation_crop_id": int(val_idx),
                "dhash_hamming": int(distance),
                "same_frame": bool(meta.at[train_idx, "frame"] == meta.at[val_idx, "frame"]),
                "frame_delta": int(abs(meta.at[train_idx, "frame"] - meta.at[val_idx, "frame"])),
                "same_corrected_identity": bool(
                    meta.at[train_idx, "global_id"] == meta.at[val_idx, "global_id"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _split_metrics(meta: pd.DataFrame, split_col: str) -> dict:
    use = meta[meta[split_col].isin(["train", "validation"])].copy()
    train = use[use[split_col] == "train"]
    val = use[use[split_col] == "validation"]
    train_frames = set(zip(train["view"], train["window"], train["frame"]))
    val_frames = set(zip(val["view"], val["window"], val["frame"]))
    same_frames = train_frames & val_frames

    def affected_within(delta: int, left: pd.DataFrame, right: pd.DataFrame) -> int:
        affected = 0
        right_groups = {
            key: np.sort(group["frame"].unique())
            for key, group in right.groupby(["view", "window"], sort=False)
        }
        for key, group in left.groupby(["view", "window"], sort=False):
            targets = right_groups.get(key)
            if targets is None or not len(targets):
                continue
            frames = group["frame"].to_numpy()
            pos = np.searchsorted(targets, frames)
            best = np.full(len(frames), np.iinfo(np.int64).max, dtype=np.int64)
            valid = pos < len(targets)
            best[valid] = np.minimum(best[valid], np.abs(targets[pos[valid]] - frames[valid]))
            valid = pos > 0
            best[valid] = np.minimum(best[valid], np.abs(targets[pos[valid] - 1] - frames[valid]))
            affected += int((best <= delta).sum())
        return affected

    return {
        "split": split_col,
        "n_train_crops": int(len(train)),
        "n_validation_crops": int(len(val)),
        "n_train_tracklets": int(train["view_tracklet"].nunique()),
        "n_validation_tracklets": int(val["view_tracklet"].nunique()),
        "tracklet_overlap": int(
            len(set(train["view_tracklet"].astype(int)) & set(val["view_tracklet"].astype(int)))
        ),
        "same_frame_keys_across_splits": int(len(same_frames)),
        "train_crops_on_shared_frames": int(
            sum((r.view, r.window, r.frame) in same_frames for r in train.itertuples())
        ),
        "validation_crops_on_shared_frames": int(
            sum((r.view, r.window, r.frame) in same_frames for r in val.itertuples())
        ),
        "train_crops_with_validation_frame_within_1": affected_within(1, train, val),
        "validation_crops_with_train_frame_within_1": affected_within(1, val, train),
        "train_crops_with_validation_frame_within_10": affected_within(10, train, val),
        "validation_crops_with_train_frame_within_10": affected_within(10, val, train),
    }


def _checkpoint_inventory(model_root: Path) -> pd.DataFrame:
    import torch

    names = [
        "model_all",
        "model_resnet_full",
        "model_female",
        "model_female_clean",
        "model_male_clean",
        "model_male_identity_classifier_masked",
    ]
    rows = []
    for name in names:
        path = model_root / name / "model.pt"
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", {})
        keys = list(state)
        if any("embedder.net.layer4" in key for key in keys):
            verified = "torchvision ResNet18 -> 128-D L2 embedding -> linear classifier"
        elif any(key.startswith("net.layer4") for key in keys):
            verified = "torchvision ResNet18 -> 128-D L2 embedding"
        elif any("backbone.6" in key or key.startswith("backbone") for key in keys):
            verified = "SmallMedakaEmbeddingNet: 3 convolution blocks -> projection"
        else:
            verified = "unresolved from checkpoint keys"
        head_shape = next(
            (list(value.shape) for key, value in state.items() if key.endswith("head.weight")), None
        )
        rows.append(
            {
                "model": name,
                "path": str(path),
                "checkpoint_arch": checkpoint.get("arch", "not_recorded"),
                "embedding_dim": checkpoint.get("embedding_dim", "not_recorded"),
                "verified_architecture": verified,
                "classifier_head_weight_shape": json.dumps(head_shape),
                "parameter_count": int(sum(value.numel() for value in state.values())),
                "checkpoint_sha256": _file_sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _file_sha256(path: Path, block: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _contact_sheet(
    crops: h5py.Dataset,
    meta: pd.DataFrame,
    indices: Iterable[int],
    path: Path,
    columns: int = 6,
) -> None:
    ids = [int(v) for v in indices]
    if not ids:
        return
    tile_w, tile_h = 220, 180
    rows = int(np.ceil(len(ids) / columns))
    canvas = np.full((rows * tile_h, columns * tile_w, 3), 245, np.uint8)
    for pos, idx in enumerate(ids):
        image = np.asarray(crops[idx], dtype=np.uint8)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        row, col = divmod(pos, columns)
        x0, y0 = col * tile_w, row * tile_h
        canvas[y0 : y0 + 128, x0 : x0 + 128] = image
        record = meta.loc[idx]
        lines = [
            f"crop {idx} {record['audit_confidence_state']}",
            f"{record['window']} f{int(record['frame'])}",
            f"gid {int(record['global_id'])} trk {int(record['tracklet_id'])}",
        ]
        for line_no, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (x0 + 2, y0 + 143 + line_no * 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)


def _qc_video(crops: h5py.Dataset, meta: pd.DataFrame, indices: list[int], path: Path) -> None:
    if not indices:
        return
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (640, 360))
    for idx in indices:
        frame = np.full((360, 640, 3), 245, dtype=np.uint8)
        image = cv2.resize(np.asarray(crops[idx], dtype=np.uint8), (300, 300))
        frame[30:330, 20:320] = image
        record = meta.loc[idx]
        lines = [
            f"crop_id: {idx}",
            f"{record['view']} / {record['window']}",
            f"frame: {int(record['frame'])}",
            f"tracklet: {int(record['tracklet_id'])}",
            f"corrected ID: {int(record['global_id'])}",
            f"state: {record['audit_confidence_state']}",
        ]
        for line_no, line in enumerate(lines):
            cv2.putText(frame, line, (345, 75 + line_no * 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (20, 20, 20), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset)
    raw_dataset = Path(args.raw_dataset)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(dataset / "crop_metadata.parquet").reset_index(drop=True)
    raw = pd.read_parquet(raw_dataset / "crop_metadata.parquet").reset_index(drop=True)
    raw_lookup = raw[["crop_id", "local_crop_id", "view", "window", "frame", "tracklet_id"]].rename(
        columns={"crop_id": "source_pool_crop_id"}
    )
    provenance = meta.merge(
        raw_lookup,
        on=["local_crop_id", "view", "window", "frame", "tracklet_id"],
        how="left",
        validate="one_to_one",
    )
    provenance.insert(0, "experiment", EXPERIMENT)
    provenance.insert(1, "corrected_dataset", str(dataset))
    provenance["female_model_split"] = _female_split(provenance)
    provenance["male_model_split"] = _male_split(provenance)

    tracklet = pd.read_csv(args.tracklet_bound_csv)
    tracklet = tracklet.rename(
        columns={
            "x": "sleap_track_x",
            "y": "sleap_track_y",
            "score": "sleap_score",
            "global_id": "geometry_global_id_at_frame",
        }
    )
    provenance = provenance.merge(
        tracklet,
        on=["frame", "tracklet_id"],
        how="left",
        validate="many_to_one",
    )
    provenance["mask_centroid_to_sleap_point_px"] = np.hypot(
        provenance["x"] - provenance["sleap_track_x"],
        provenance["y"] - provenance["sleap_track_y"],
    )
    key = ["view", "window", "frame", "tracklet_id"]
    provenance["same_tracklet_frame_crop_count"] = provenance.groupby(key)["crop_id"].transform("size")
    provenance["raw_label_matches_frame_geometry"] = (
        provenance["raw_global_id"] == provenance["geometry_global_id_at_frame"]
    )
    reasons = []
    states = []
    for row in provenance.itertuples():
        row_reasons = []
        reject = False
        if pd.isna(row.source_pool_crop_id) or pd.isna(row.geometry_global_id_at_frame):
            row_reasons.append("missing_source_join")
            reject = True
        if not bool(row.raw_label_matches_frame_geometry):
            row_reasons.append("raw_label_disagrees_with_frame_geometry")
            reject = True
        if int(row.same_tracklet_frame_crop_count) > 1:
            row_reasons.append("multiple_masks_assigned_to_same_tracklet_frame")
            reject = True
        if bool(row.low_conf):
            row_reasons.append("tracklet_bind_low_conf")
        if float(row.mask_centroid_to_sleap_point_px) > 40.0:
            row_reasons.append("mask_sleap_centroid_distance_gt_40px")
        row_reasons.append("not_independently_validated")
        states.append("reject" if reject else "ambiguous")
        reasons.append(";".join(row_reasons))
    provenance["audit_confidence_state"] = states
    provenance["audit_confidence_reason"] = reasons

    h5, crops = _load_crops(dataset)
    try:
        exact_hashes, dhashes = _hash_crops(crops)
        provenance["crop_sha256"] = exact_hashes
        provenance["crop_dhash64"] = [f"{int(v):016x}" for v in dhashes]

        split_summaries = []
        near_tables = []
        for split_col in ["female_model_split", "male_model_split"]:
            metrics = _split_metrics(provenance, split_col)
            use = provenance[provenance[split_col].isin(["train", "validation"])]
            exact_split_counts = use.groupby("crop_sha256")[split_col].nunique()
            metrics["exact_pixel_hashes_crossing_splits"] = int((exact_split_counts > 1).sum())
            near = _near_duplicate_matches(
                provenance, dhashes, split_col, int(args.near_duplicate_hamming)
            )
            metrics["train_crops_with_near_duplicate_in_validation"] = int(len(near))
            metrics["near_duplicate_hamming_threshold"] = int(args.near_duplicate_hamming)
            split_summaries.append(metrics)
            if not near.empty:
                near_tables.append(near)
        near_all = pd.concat(near_tables, ignore_index=True) if near_tables else pd.DataFrame()

        tracklet_consistency = (
            provenance.groupby(["view", "view_tracklet", "tracklet_id"], as_index=False)
            .agg(
                n_crops=("crop_id", "size"),
                frame_start=("frame", "min"),
                frame_end=("frame", "max"),
                corrected_identity_nunique=("global_id", "nunique"),
                raw_identity_nunique=("raw_global_id", "nunique"),
                frame_geometry_identity_nunique=("geometry_global_id_at_frame", "nunique"),
                n_low_conf=("low_conf", "sum"),
                n_duplicate_assignment_crops=(
                    "same_tracklet_frame_crop_count", lambda s: int((s > 1).sum())
                ),
            )
        )

        state_summary = (
            provenance.groupby(["audit_confidence_state", "view", "global_id"], as_index=False)
            .size()
            .rename(columns={"global_id": "corrected_identity", "size": "n_samples"})
        )
        camera_balance = (
            provenance.groupby(["view", "window", "global_id"], as_index=False)
            .size()
            .rename(columns={"global_id": "corrected_identity", "size": "n_samples"})
        )
        duplicate_groups = (
            provenance.groupby(key, as_index=False)
            .agg(n_crops=("crop_id", "size"), crop_ids=("crop_id", lambda s: ";".join(map(str, s))))
        )
        duplicate_groups = duplicate_groups[duplicate_groups["n_crops"] > 1].copy()

        model_inventory = _checkpoint_inventory(Path(args.model_root))
        summary = {
            "experiment": EXPERIMENT,
            "dataset": str(dataset),
            "raw_dataset": str(raw_dataset),
            "n_samples": int(len(provenance)),
            "n_tracklets": int(provenance["view_tracklet"].nunique()),
            "views": {str(k): int(v) for k, v in provenance["view"].value_counts().items()},
            "windows": {str(k): int(v) for k, v in provenance["window"].value_counts().items()},
            "corrected_identity_counts": {
                str(k): int(v) for k, v in provenance["global_id"].value_counts().sort_index().items()
            },
            "confidence_state_counts": {
                str(k): int(v) for k, v in provenance["audit_confidence_state"].value_counts().items()
            },
            "source_pool_join_missing": int(provenance["source_pool_crop_id"].isna().sum()),
            "tracklet_bind_join_missing": int(provenance["geometry_global_id_at_frame"].isna().sum()),
            "raw_label_frame_geometry_mismatches": int(
                (~provenance["raw_label_matches_frame_geometry"]).sum()
            ),
            "low_conf_tracklet_bind_samples": int(provenance["low_conf"].sum()),
            "mask_sleap_centroid_distance_px": {
                str(q): float(provenance["mask_centroid_to_sleap_point_px"].quantile(q))
                for q in [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
            },
            "mask_sleap_centroid_distance_gt_40px": int(
                (provenance["mask_centroid_to_sleap_point_px"] > 40).sum()
            ),
            "duplicate_tracklet_frame_groups": int(len(duplicate_groups)),
            "samples_in_duplicate_tracklet_frame_groups": int(
                (provenance["same_tracklet_frame_crop_count"] > 1).sum()
            ),
            "frame_crop_multiplicity": {
                str(k): int(v)
                for k, v in provenance.groupby(["view", "window", "frame"]).size().value_counts().sort_index().items()
            },
            "tracklets_with_multiple_corrected_labels": int(
                (tracklet_consistency["corrected_identity_nunique"] > 1).sum()
            ),
            "tracklets_with_multiple_frame_geometry_labels": int(
                (tracklet_consistency["frame_geometry_identity_nunique"] > 1).sum()
            ),
            "split_audits": split_summaries,
            "provenance_limitations": [
                "The corrected metadata does not preserve the Detectron2 mask index, bbox, score, or mask-file path.",
                "The original video path and exact command line were not persisted in the dataset.",
                "No human or independently measured biological-ID review state is present.",
                "The x/y columns are mask centroids, not the SLEAP point; SLEAP coordinates were recovered from the matching tracklet-bind table.",
            ],
        }

        provenance.to_csv(out / "sample_provenance.csv", index=False)
        provenance.to_parquet(out / "sample_provenance.parquet", index=False)
        tracklet_consistency.to_csv(out / "tracklet_label_consistency.csv", index=False)
        state_summary.to_csv(out / "confidence_summary_by_identity_camera.csv", index=False)
        camera_balance.to_csv(out / "camera_window_identity_balance.csv", index=False)
        duplicate_groups.to_csv(out / "duplicate_mask_assignments.csv", index=False)
        near_all.to_csv(out / "cross_split_near_duplicate_matches.csv", index=False)
        model_inventory.to_csv(out / "model_architecture_inventory.csv", index=False)
        (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if not args.skip_visuals:
            balanced = []
            for (_window, _gid), group in provenance.groupby(["window", "global_id"]):
                balanced.extend(group.sample(min(4, len(group)), random_state=17).index.tolist())
            _contact_sheet(crops, provenance, balanced, out / "contact_sheet_balanced_ambiguous.png")
            rejected = provenance.index[provenance["audit_confidence_state"] == "reject"].tolist()
            _contact_sheet(crops, provenance, rejected[:60], out / "contact_sheet_structural_rejects.png")
            video_ids = []
            for _gid, group in provenance.groupby("global_id"):
                video_ids.extend(group.sort_values("frame").iloc[:: max(1, len(group) // 20)].index[:20])
            _qc_video(crops, provenance, [int(v) for v in video_ids], out / "qc_crop_sequence.mp4")
    finally:
        h5.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
