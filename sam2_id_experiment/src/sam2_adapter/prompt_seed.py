from __future__ import annotations

from pathlib import Path

import numpy as np


def seed_prompts_from_mask_h5(
    mask_path: str | Path,
    *,
    source_frame: int,
    target_local_frame: int = 0,
    min_area: int = 50,
    points_per_mask: int = 3,
    negative_points_per_other_mask: int = 1,
) -> tuple[list[dict], int]:
    """Create SAM2 point prompts from masks at the nearest available frame."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for SAM2 mask prompt seeding.") from exc

    mask_path = Path(mask_path)
    with h5py.File(mask_path, "r") as h5:
        frames = sorted(int(k.split("_")[1]) for k in h5.keys() if k.startswith("frame_"))
        if not frames:
            raise ValueError(f"No frame groups found in {mask_path}")
        selected_frame = min(frames, key=lambda frame: abs(frame - int(source_frame)))
        frame_group = h5[f"frame_{selected_frame:06d}"]

        mask_points: list[tuple[int, list[list[int]], list[list[int]]]] = []
        for instance_key in sorted(k for k in frame_group.keys() if k.startswith("instance_")):
            instance = frame_group[instance_key]
            track_id = int(instance.attrs.get("sam2_track_id", -1))
            if track_id < 0:
                continue
            mask = np.asarray(instance["mask"][...]).astype(bool)
            area = int(mask.sum())
            if area < int(min_area):
                continue
            points = mask_prompt_points(mask, n_points=max(1, int(points_per_mask)))
            if not points:
                continue
            negative_points = mask_prompt_points(
                mask,
                n_points=max(1, int(negative_points_per_other_mask)),
            )
            mask_points.append((track_id, points, negative_points))

    prompts: list[dict] = []
    for track_id, positive_points, _negative_points in mask_points:
        negatives: list[list[int]] = []
        if negative_points_per_other_mask > 0:
            for other_track_id, _other_positive, other_negative in mask_points:
                if int(other_track_id) == int(track_id):
                    continue
                negatives.extend(other_negative[:negative_points_per_other_mask])
        prompts.append({
            "obj_id": track_id,
            "frame": int(target_local_frame),
            "points": positive_points + negatives,
            "labels": [1] * len(positive_points) + [0] * len(negatives),
        })
    prompts.sort(key=lambda prompt: int(prompt["obj_id"]))
    return prompts, int(selected_frame)


def mask_prompt_points(mask: np.ndarray, *, n_points: int = 3) -> list[list[int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []

    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    center = coords.mean(axis=0)
    if len(coords) < 3 or n_points == 1:
        points = [_nearest_mask_pixel(coords, center)]
    else:
        centered = coords - center
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        projection = centered @ axis
        quantiles = np.linspace(0.25, 0.75, n_points)
        points = [
            _nearest_mask_pixel(coords, coords[int(np.argmin(np.abs(projection - np.quantile(projection, q))))])
            for q in quantiles
        ]

    unique: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for point in points:
        key = (int(point[0]), int(point[1]))
        if key not in seen:
            seen.add(key)
            unique.append([int(point[0]), int(point[1])])
    return unique


def _nearest_mask_pixel(coords: np.ndarray, target: np.ndarray) -> list[int]:
    idx = int(np.argmin(np.sum((coords - target) ** 2, axis=1)))
    x, y = coords[idx]
    return [int(round(x)), int(round(y))]
