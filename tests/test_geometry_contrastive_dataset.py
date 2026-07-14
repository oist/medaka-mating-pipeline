from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import (
    MaskInstance,
    bbox_from_mask,
    write_mask_instances_h5,
)
from sam2_id_experiment.src.id_model.geometry_contrastive import (
    load_explicit_pairs,
    load_samples,
)


builder = importlib.import_module(
    "sam2_id_experiment.scripts.48_build_geometry_contrastive_dataset"
)


def test_assign_splits_uses_complete_embargo_blocks():
    assigned = builder.assign_splits(
        range(500), [], block_size=100, embargo_frames=100,
        fractions=(0.6, 0.2, 0.2),
    )
    by_block = assigned.groupby("split_block_id")["split"].nunique()
    assert by_block.max() == 1
    assert assigned.groupby("split")["split_block_id"].nunique().to_dict() == {
        "embargo": 2,
        "test": 1,
        "train": 1,
        "calibration": 1,
    }
    train_max = assigned.loc[assigned["split"] == "train", "frame"].max()
    calibration_min = assigned.loc[assigned["split"] == "calibration", "frame"].min()
    assert calibration_min - train_max - 1 >= 100


def test_mask_store_reads_original_chunks_through_manifest(tmp_path):
    chunks = []
    for start, frame in ((0, 1), (10, 11)):
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:8, 5:10] = True
        path = tmp_path / f"top_cam.masks_{start}-{start + 9}.h5"
        write_mask_instances_h5(
            path,
            [
                MaskInstance(
                    frame=frame,
                    instance_id=2,
                    sam2_track_id=7,
                    mask=mask,
                    bbox=bbox_from_mask(mask),
                    score=0.9,
                )
            ],
            image_shape=(16, 16),
        )
        chunks.append(
            {"start_frame": start, "end_frame": start + 9, "path": path.name}
        )
    manifest = tmp_path / "top_cam.mask_chunks.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "sam2_mask_chunk_manifest_v1",
                "view": "top_cam",
                "chunks": chunks,
            }
        ),
        encoding="utf-8",
    )
    store = builder.MaskStore(manifest, view="top_cam")
    try:
        first = store.read(1, 2)
        second = store.read(11, 2)
        assert first is not None and second is not None
        assert np.array_equal(first[0], second[0])
        assert store.read(9, 2) is None
    finally:
        store.close()


def test_build_pairs_obeys_local_constraint_semantics():
    samples = pd.DataFrame(
        [
            # node A remains one positive fragment even though the frame-local
            # geometry source changes.
            {"sample_id": 0, "experiment": "e", "window": "w", "split": "train",
             "view": "top", "frame": 0, "node_id": "top:A", "geometry_source_idx_local": 0},
            {"sample_id": 1, "experiment": "e", "window": "w", "split": "train",
             "view": "top", "frame": 10, "node_id": "top:A", "geometry_source_idx_local": 1},
            {"sample_id": 2, "experiment": "e", "window": "w", "split": "train",
             "view": "side", "frame": 0, "node_id": "side:B", "geometry_source_idx_local": 0},
            {"sample_id": 3, "experiment": "e", "window": "w", "split": "train",
             "view": "top", "frame": 0, "node_id": "top:C", "geometry_source_idx_local": 2},
        ]
    )
    samples["local_fragment_id"] = ["frag_a", "frag_a", "frag_b", "frag_c"]
    positives, negatives = builder.build_pairs(
        samples, min_positive_delta=5, max_positive_delta=20,
        max_positive_per_fragment=10, include_cross_view_negatives=False,
        random_seed=1,
    )
    temporal = positives[positives["pair_type"] == "same_local_fragment"]
    assert {tuple(sorted(pair)) for pair in temporal[["sample_a_id", "sample_b_id"]].to_numpy()} == {(0, 1)}
    assert temporal["geometry_source_idx_local"].tolist() == [-1]
    synchronized = positives[positives["pair_type"] == "synchronized_cross_camera_same_source"]
    assert {tuple(sorted(pair)) for pair in synchronized[["sample_a_id", "sample_b_id"]].to_numpy()} == {(0, 2)}
    assert synchronized["geometry_source_idx_local"].tolist() == [0]
    assert set(negatives["pair_type"]) == {"simultaneous_same_view_different_source"}
    assert {tuple(sorted(pair)) for pair in negatives[["sample_a_id", "sample_b_id"]].to_numpy()} == {(0, 3)}


def _write_synthetic_inputs(root: Path) -> list[str]:
    experiment = "synthetic_fish"
    frames = range(12)
    views = ("top_cam", "side1_cam")
    colors = ((40, 100, 220), (60, 210, 80), (220, 90, 50))
    centers = ((8, 8), (16, 16), (24, 24))

    best_frames = {}
    identified_rows = []
    disposition_rows = []
    view_args: list[str] = []
    for view_index, view in enumerate(views):
        image_dir = root / f"{view}_images"
        image_dir.mkdir()
        instances = []
        association_rows = []
        for frame in frames:
            image = np.full((32, 32, 3), 17, dtype=np.uint8)
            for source, ((cx, cy), color) in enumerate(zip(centers, colors)):
                image[cy - 2:cy + 3, cx - 3:cx + 4] = color
                mask = np.zeros((32, 32), dtype=bool)
                mask[cy - 2:cy + 3, cx - 3:cx + 4] = True
                instances.append(MaskInstance(
                    frame=frame, instance_id=source, sam2_track_id=100 + source,
                    mask=mask, bbox=bbox_from_mask(mask), score=0.95,
                ))
                node = f"{view}:{source}"
                identified_rows.append({
                    "frame": frame, "view": view, "node_id": node,
                    "tracklet_id": source, "x": cx, "y": cy,
                    # Deliberately present in the source; the builder must ignore it.
                    "final_fish_id": f"DO_NOT_COPY_{source}",
                    "component_id": f"component_{view}_{source}",
                })
                disposition_rows.append({
                    "frame": frame, "view": view, "tracklet_node": node,
                    "selected_source_idx": source,
                    "disposition": "assigned_2d_valid_3d", "synthetic_view": -1,
                    "x": cx, "y": cy,
                })
                association_rows.append({
                    "frame": frame, "tracklet_id": source, "x": cx, "y": cy,
                    "matched_mask_instance_id": source,
                    "matched_mask_track_id": 100 + source, "mask_score": 0.95,
                    "n_candidate_masks": 1, "n_skeletons_for_mask": 1,
                    "split_endpoints_across_masks": False,
                    "silhouette_status": "strong_match",
                    "silhouette_evidence_state": "support", "mutual_confidence": 0.9,
                })
            assert cv2.imwrite(str(image_dir / f"frame_{frame:06d}.png"), image)
        masks_path = root / f"{view}_masks.h5"
        write_mask_instances_h5(masks_path, instances, image_shape=(32, 32))
        association_path = root / f"{view}_associations.csv"
        pd.DataFrame(association_rows).to_csv(association_path, index=False)
        view_args.extend(["--view-input", view, str(image_dir), str(masks_path), str(association_path)])

    for frame in frames:
        best_frames[str(frame)] = {
            "groups": {"a": [0.0 + frame * 0.1, 0.0, 0.0],
                       "b": [30.0 + frame * 0.1, 0.0, 0.0],
                       "c": [60.0 + frame * 0.1, 0.0, 0.0]},
            "errors": {"a": 5.0, "b": 6.0, "c": 7.0},
            "recovered": [False, False, False],
            "synthetic_views": [-1, -1, -1],
            "dropped_high_residual": [],
        }
    best_path = root / "best.json"
    best_path.write_text(json.dumps({"num_individuals": 3, "frames": best_frames}), encoding="utf-8")
    identified_path = root / "identified.csv"
    disposition_path = root / "dispositions.csv"
    pd.DataFrame(identified_rows).to_csv(identified_path, index=False)
    pd.DataFrame(disposition_rows).to_csv(disposition_path, index=False)
    return [
        "--experiment", experiment,
        "--identified-2d", str(identified_path),
        "--dispositions", str(disposition_path),
        "--best-pre-link-json", str(best_path),
        *view_args,
    ]


def test_synthetic_smoke_writes_black_background_local_constraints(tmp_path):
    output = tmp_path / "dataset"
    argv = _write_synthetic_inputs(tmp_path) + [
        "--window", "development_a:0:3:train",
        "--window", "calibration_a:4:7:calibration",
        "--window", "heldout_a:8:11:test",
        "--embargo-frames", "0",
        "--output-dir", str(output),
        "--sample-every", "1",
        "--min-positive-frame-delta", "2",
        "--max-positive-frame-delta", "6",
        "--contact-sheet-samples", "8",
    ]
    builder.main(argv)

    samples = pd.read_parquet(output / "samples.parquet")
    positives = pd.read_parquet(output / "positive_pairs.parquet")
    negatives = pd.read_parquet(output / "negative_pairs.parquet")
    assert len(samples) == 12 * 3 * 2
    assert samples["crop_index"].tolist() == samples["sample_id"].tolist()
    assert "final_fish_id" not in samples.columns
    assert samples["geometry_source_idx_local"].isin([0, 1, 2]).all()
    assert set(positives["pair_label"]) == {"same"}
    assert set(negatives["pair_label"]) == {"different"}
    left = samples.set_index("sample_id").loc[negatives["sample_a_id"]]
    right = samples.set_index("sample_id").loc[negatives["sample_b_id"]]
    assert np.array_equal(left["frame"].to_numpy(), right["frame"].to_numpy())
    assert np.array_equal(left["view"].to_numpy(), right["view"].to_numpy())

    with h5py.File(output / "crops.h5", "r") as h5:
        crops = np.asarray(h5["crops"])
        assert crops.dtype == np.uint8
        assert crops.shape == (72, 128, 128, 3)
        assert h5.attrs["color_order"] == "RGB"
        assert np.any(crops == 0)
        assert np.any(crops != 0)
        assert np.any(np.all(crops == np.asarray([220, 100, 40], dtype=np.uint8), axis=-1))
        # Corners are outside every mask and must be black, not source background.
        assert np.all(crops[:, 0, 0] == 0)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "No global biological identity labels" in summary["identity_semantics"]
    assert (output / "contact_sheet.png").exists()

    trainer_samples, _ = load_samples(output)
    trainer_pairs, audit, _ = load_explicit_pairs(
        output, trainer_samples, min_positive_frame_gap=2,
    )
    assert len(trainer_pairs) == audit["retained"]
    assert set(trainer_pairs["pair_subtype"]) == {
        "same_fragment", "cross_camera", "simultaneous_frame",
    }
