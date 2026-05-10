import numpy as np
import pandas as pd

from scripts.tracking.top_camera_identity import (
    BlobCandidate,
    assign_fragment_identities,
    bootstrap_identity_labels,
    classify_fragment_feature,
    component_candidates,
    candidate_mask_to_full_frame,
    filter_candidates_by_sleap,
    foreground_mask_from_frame,
    keep_best_candidate_per_sleap_detection,
    link_blob_fragments,
    merge_same_frame_same_id_candidates,
    save_edited_candidate_mask,
    smart_fill_region_from_click,
    suppress_overlapping_candidates,
)


def _ellipse_frame(centers):
    import cv2

    frame = np.full((120, 180, 3), 120, dtype=np.uint8)
    for cx, cy in centers:
        cv2.ellipse(frame, (cx, cy), (12, 5), 20, 0, 360, (30, 30, 30), -1)
    return frame


def test_background_mask_extracts_three_fish_like_blobs():
    background = np.full((120, 180), 120, dtype=np.uint8)
    frame = _ellipse_frame([(30, 40), (90, 70), (145, 35)])

    mask = foreground_mask_from_frame(frame, background, threshold=20)
    candidates = component_candidates(
        mask,
        frame=0,
        source="bgsub",
        candidate_prefix="t",
        min_area=40,
        max_area=1000,
    )

    assert len(candidates) == 3
    assert sorted(round(c.x) for c in candidates) == [30, 90, 145]


def test_sleap_proximity_filter_keeps_nearby_blob_and_rejects_reflection():
    candidates = [
        BlobCandidate("near", 0, 50.0, 50.0, 100, 40, 40, 20, 10, "", "bgsub", -1, 0.0),
        BlobCandidate("far", 0, 150.0, 100.0, 100, 140, 95, 20, 10, "", "bgsub", -1, 0.0),
    ]

    kept = filter_candidates_by_sleap(candidates, [(52.0, 51.0, 0.9)], max_distance_px=20)

    assert [c.candidate_id for c in kept] == ["near"]
    assert kept[0].near_sleap_det == 0
    assert kept[0].quality_score > 0


def test_overlap_suppression_keeps_best_candidate_per_sleap_detection():
    candidates = [
        BlobCandidate("weak", 0, 50.0, 50.0, 100, 40, 40, 20, 10, "", "bgsub", 0, 2.0),
        BlobCandidate("strong", 0, 52.0, 49.0, 120, 42, 39, 20, 10, "", "sam", 0, 5.0),
        BlobCandidate("other", 0, 120.0, 50.0, 100, 110, 40, 20, 10, "", "bgsub", 1, 1.0),
    ]

    kept = suppress_overlapping_candidates(candidates, max_center_distance_px=10)

    assert [c.candidate_id for c in kept] == ["other", "strong"]


def test_keep_best_candidate_per_sleap_detection_limits_to_three_fish_slots():
    candidates = [
        BlobCandidate("weak0", 0, 50.0, 50.0, 100, 40, 40, 20, 10, "", "bgsub", 0, 2.0),
        BlobCandidate("strong0", 0, 52.0, 49.0, 120, 42, 39, 20, 10, "", "sam", 0, 5.0),
        BlobCandidate("one1", 0, 120.0, 50.0, 100, 110, 40, 20, 10, "", "bgsub", 1, 1.0),
        BlobCandidate("one2", 0, 160.0, 50.0, 100, 150, 40, 20, 10, "", "bgsub", 2, 1.0),
    ]

    kept = keep_best_candidate_per_sleap_detection(candidates)

    assert [c.candidate_id for c in kept] == ["one1", "one2", "strong0"]


def test_fragment_linking_preserves_smooth_motion_and_splits_crossing_jump():
    rows = []
    for frame in range(5):
        rows.append(
            {
                "candidate_id": f"a{frame}",
                "frame": frame,
                "x": float(10 + frame),
                "y": 20.0,
                "area": 100,
                "bbox_x": 0,
                "bbox_y": 0,
                "bbox_w": 10,
                "bbox_h": 10,
                "mask_path": "",
                "source": "bgsub",
                "near_sleap_det": 0,
                "quality_score": 1.0,
                "status": "accept",
                "id": "",
            }
        )
    rows.append({**rows[-1], "candidate_id": "jump", "frame": 6, "x": 200.0})
    df = pd.DataFrame(rows)

    linked = link_blob_fragments(df, max_distance_px=10, max_gap_frames=2)

    smooth_ids = set(linked[linked["candidate_id"].str.startswith("a")]["fragment_id"])
    jump_id = int(linked[linked["candidate_id"] == "jump"]["fragment_id"].iloc[0])
    assert len(smooth_ids) == 1
    assert jump_id not in smooth_ids


def test_fragment_linking_uses_bbox_overlap_and_respects_manual_id_conflict():
    rows = [
        {
            "candidate_id": "a0",
            "frame": 0,
            "x": 10.0,
            "y": 10.0,
            "area": 100,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": 20,
            "bbox_h": 20,
            "mask_path": "",
            "source": "bgsub",
            "near_sleap_det": 0,
            "quality_score": 1.0,
            "status": "accept",
            "id": "ID_1",
        },
        {
            "candidate_id": "a1",
            "frame": 1,
            "x": 25.0,
            "y": 10.0,
            "area": 100,
            "bbox_x": 15,
            "bbox_y": 0,
            "bbox_w": 20,
            "bbox_h": 20,
            "mask_path": "",
            "source": "bgsub",
            "near_sleap_det": 0,
            "quality_score": 1.0,
            "status": "accept",
            "id": "ID_1",
        },
        {
            "candidate_id": "conflict",
            "frame": 2,
            "x": 40.0,
            "y": 10.0,
            "area": 100,
            "bbox_x": 30,
            "bbox_y": 0,
            "bbox_w": 20,
            "bbox_h": 20,
            "mask_path": "",
            "source": "bgsub",
            "near_sleap_det": 0,
            "quality_score": 1.0,
            "status": "accept",
            "id": "ID_2",
        },
    ]
    linked = link_blob_fragments(pd.DataFrame(rows), max_distance_px=20)

    first = int(linked[linked["candidate_id"] == "a0"]["fragment_id"].iloc[0])
    second = int(linked[linked["candidate_id"] == "a1"]["fragment_id"].iloc[0])
    conflict = int(linked[linked["candidate_id"] == "conflict"]["fragment_id"].iloc[0])

    assert first == second
    assert conflict != first


def test_fragment_linking_stitches_short_same_id_gaps():
    rows = []
    for candidate_id, frame, x in [
        ("a0", 0, 10.0),
        ("a1", 1, 20.0),
        ("a2", 6, 70.0),
    ]:
        rows.append(
            {
                "candidate_id": candidate_id,
                "frame": frame,
                "x": x,
                "y": 10.0,
                "area": 100,
                "bbox_x": int(x - 5),
                "bbox_y": 5,
                "bbox_w": 10,
                "bbox_h": 10,
                "mask_path": "",
                "source": "bgsub",
                "near_sleap_det": 0,
                "quality_score": 1.0,
                "status": "accept",
                "id": "ID_1",
            }
        )

    linked = link_blob_fragments(
        pd.DataFrame(rows),
        max_distance_px=12,
        max_gap_frames=1,
        stitch_gap_frames=10,
    )

    assert linked["fragment_id"].nunique() == 1


def test_identity_assignment_leaves_low_margin_fragment_unknown():
    centroids = {
        "ID_1": np.array([1.0, 0.0], dtype=np.float32),
        "ID_2": np.array([0.98, 0.02], dtype=np.float32),
    }
    feature = np.array([0.99, 0.01], dtype=np.float32)

    identity, distance, margin, reason = classify_fragment_feature(
        feature,
        centroids,
        min_margin=0.1,
        max_distance=2.0,
    )

    assert identity == "unknown"
    assert distance is not None
    assert margin is not None and margin < 0.1
    assert reason == "low_margin"


def test_assign_fragment_identities_accepts_clear_centroid_match():
    centroids = {
        "ID_1": np.array([1.0, 0.0], dtype=np.float32),
        "ID_2": np.array([0.0, 1.0], dtype=np.float32),
    }
    assignments = assign_fragment_identities(
        {5: np.array([0.99, 0.01], dtype=np.float32)},
        centroids,
        min_margin=0.1,
        max_distance=2.0,
    )

    assert assignments[5][0] == "ID_1"
    assert assignments[5][3] == ""


def test_save_edited_candidate_mask_updates_geometry_and_artifacts(tmp_path):
    import cv2

    frame = np.full((80, 100, 3), 120, dtype=np.uint8)
    df = pd.DataFrame(
        [
            {
                "candidate_id": "c0",
                "frame": 0,
                "x": 25.0,
                "y": 25.0,
                "area": 25,
                "bbox_x": 20,
                "bbox_y": 20,
                "bbox_w": 10,
                "bbox_h": 10,
                "mask_path": "masks/c0.png",
                "crop_path": "crops/c0.png",
                "source": "bgsub",
                "near_sleap_det": 0,
                "quality_score": 1.0,
                "status": "review",
                "id": "",
                "bad_reason": "",
            }
        ]
    )
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(full_mask, (18, 20), (34, 32), 255, -1)

    save_edited_candidate_mask(df, 0, full_mask, frame, csv_dir=tmp_path)
    reloaded = candidate_mask_to_full_frame(df.loc[0], csv_dir=tmp_path, frame_shape=frame.shape)

    assert int(df.at[0, "area"]) == int((full_mask > 0).sum())
    assert int(df.at[0, "bbox_x"]) == 18
    assert int(df.at[0, "bbox_w"]) == 17
    assert (tmp_path / "masks" / "c0.png").exists()
    assert (tmp_path / "crops" / "c0.png").exists()
    assert np.array_equal(reloaded > 0, full_mask > 0)


def test_merge_same_frame_same_id_candidates_combines_split_manual_blobs(tmp_path):
    import cv2

    crops_dir = tmp_path / "crops"
    masks_dir = tmp_path / "masks"
    crops_dir.mkdir()
    masks_dir.mkdir()
    for name, value in [("left", 90), ("right", 180)]:
        cv2.imwrite(str(crops_dir / f"{name}.png"), np.full((10, 10, 3), value, dtype=np.uint8))
        cv2.imwrite(str(masks_dir / f"{name}.png"), np.full((10, 10), 255, dtype=np.uint8))

    df = pd.DataFrame(
        [
            {
                "candidate_id": "left",
                "frame": 0,
                "x": 5.0,
                "y": 5.0,
                "area": 100,
                "bbox_x": 0,
                "bbox_y": 0,
                "bbox_w": 10,
                "bbox_h": 10,
                "mask_path": "masks/left.png",
                "crop_path": "crops/left.png",
                "source": "bgsub",
                "near_sleap_det": 0,
                "quality_score": 1.0,
                "status": "accept",
                "id": "ID_1",
                "id_source": "manual",
            },
            {
                "candidate_id": "right",
                "frame": 0,
                "x": 15.0,
                "y": 5.0,
                "area": 100,
                "bbox_x": 10,
                "bbox_y": 0,
                "bbox_w": 10,
                "bbox_h": 10,
                "mask_path": "masks/right.png",
                "crop_path": "crops/right.png",
                "source": "bgsub",
                "near_sleap_det": 0,
                "quality_score": 1.0,
                "status": "accept",
                "id": "ID_1",
                "id_source": "manual",
            },
        ]
    )

    merged, report = merge_same_frame_same_id_candidates(df, csv_dir=tmp_path)

    assert report == {"n_merged_groups": 1, "n_merged_parts": 2}
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["id"] == "ID_1"
    assert int(row["bbox_w"]) == 20
    assert int(row["area"]) == 200
    assert (tmp_path / row["mask_path"]).exists()
    assert (tmp_path / row["crop_path"]).exists()


def test_smart_fill_region_from_click_keeps_connected_fish_blob():
    import cv2

    frame = np.full((120, 180, 3), 150, dtype=np.uint8)
    cv2.ellipse(frame, (55, 60), (24, 8), 15, 0, 360, (45, 45, 45), -1)
    cv2.ellipse(frame, (125, 60), (24, 8), 15, 0, 360, (45, 45, 45), -1)

    fill = smart_fill_region_from_click(frame, (55, 60), search_radius=45, tolerance=12)

    assert fill[60, 55] > 0
    assert fill[60, 125] == 0
    assert 250 < int((fill > 0).sum()) < 900


def test_bootstrap_identity_labels_prefills_unlabeled_fragments(tmp_path):
    import cv2

    crops_dir = tmp_path / "crops"
    masks_dir = tmp_path / "masks"
    crops_dir.mkdir()
    masks_dir.mkdir()

    def write_crop(name: str, pattern: str) -> None:
        img = np.full((32, 32, 3), 30, dtype=np.uint8)
        if pattern == "left":
            img[:, :12] = 220
        elif pattern == "right":
            img[:, 20:] = 220
        elif pattern == "diag":
            for i in range(32):
                img[max(0, i - 2) : min(32, i + 3), i] = 220
        cv2.imwrite(str(crops_dir / f"{name}.png"), img)
        cv2.imwrite(str(masks_dir / f"{name}.png"), np.full((32, 32), 255, dtype=np.uint8))

    rows = []
    specs = [
        ("seed1", 0, 10.0, "ID_1", "left"),
        ("seed2", 0, 60.0, "ID_2", "right"),
        ("seed3", 0, 110.0, "ID_3", "diag"),
        ("u1", 10, 10.0, "", "left"),
        ("u2", 10, 60.0, "", "right"),
        ("u3", 10, 110.0, "", "diag"),
    ]
    for name, frame, x, identity, pattern in specs:
        write_crop(name, pattern)
        rows.append(
            {
                "candidate_id": name,
                "frame": frame,
                "x": x,
                "y": 20.0,
                "area": 100,
                "bbox_x": int(x),
                "bbox_y": 10,
                "bbox_w": 32,
                "bbox_h": 32,
                "mask_path": f"masks/{name}.png",
                "crop_path": f"crops/{name}.png",
                "source": "bgsub",
                "near_sleap_det": 0,
                "quality_score": 1.0,
                "status": "accept" if identity else "review",
                "id": identity,
                "id_source": "manual" if identity else "",
                "bad_reason": "",
            }
        )
    csv_path = tmp_path / "blob_candidates.reviewed.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = bootstrap_identity_labels(candidates_csv=csv_path, min_seed_samples=1)
    out = pd.read_csv(result["output_csv"])

    assigned = dict(zip(out["candidate_id"], out["id"]))
    assert assigned["u1"] == "ID_1"
    assert assigned["u2"] == "ID_2"
    assert assigned["u3"] == "ID_3"
    assert set(out[out["candidate_id"].str.startswith("u")]["id_source"]) == {"bootstrap"}
    assert result["n_auto_labeled_rows"] == 3
