from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class MaskInstance:
    frame: int
    instance_id: int
    sam2_track_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def write_mask_instances_h5(
    path: str | Path,
    instances: Iterable[MaskInstance],
    *,
    image_shape: tuple[int, int] | None = None,
    overwrite: bool = True,
) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for SAM2 mask HDF5 I/O.") from exc

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(out_path, mode) as h5:
        h5.attrs["schema_version"] = "sam2_id_masks_v1"
        if image_shape is not None:
            h5.attrs["image_height"] = int(image_shape[0])
            h5.attrs["image_width"] = int(image_shape[1])

        per_frame_counts: dict[int, int] = {}
        for inst in instances:
            frame = int(inst.frame)
            idx = per_frame_counts.get(frame, 0)
            per_frame_counts[frame] = idx + 1
            frame_group = h5.require_group(f"frame_{frame:06d}")
            group = frame_group.create_group(f"instance_{idx:03d}")

            mask = np.asarray(inst.mask).astype(np.uint8)
            bbox = inst.bbox if inst.bbox is not None else bbox_from_mask(mask)
            group.create_dataset("mask", data=mask, compression="gzip")
            group.create_dataset("bbox", data=np.asarray(bbox, dtype=np.int32))
            group.attrs["frame"] = frame
            group.attrs["instance_id"] = int(inst.instance_id)
            group.attrs["sam2_track_id"] = int(inst.sam2_track_id)
            group.attrs["score"] = float(inst.score)


def iter_mask_instances_h5(path: str | Path) -> Iterator[MaskInstance]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for SAM2 mask HDF5 I/O.") from exc

    with h5py.File(Path(path), "r") as h5:
        for frame_key in sorted(k for k in h5.keys() if k.startswith("frame_")):
            frame_group = h5[frame_key]
            for inst_key in sorted(k for k in frame_group.keys() if k.startswith("instance_")):
                group = frame_group[inst_key]
                mask = np.asarray(group["mask"][...]).astype(bool)
                bbox_arr = np.asarray(group["bbox"][...], dtype=np.int32)
                yield MaskInstance(
                    frame=int(group.attrs.get("frame", int(frame_key.split("_")[1]))),
                    instance_id=int(group.attrs.get("instance_id", int(inst_key.split("_")[1]))),
                    sam2_track_id=int(group.attrs.get("sam2_track_id", -1)),
                    mask=mask,
                    bbox=tuple(int(v) for v in bbox_arr.tolist()),
                    score=float(group.attrs.get("score", np.nan)),
                )


def read_mask_instances_h5(path: str | Path) -> list[MaskInstance]:
    return list(iter_mask_instances_h5(path))

