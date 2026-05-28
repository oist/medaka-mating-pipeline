from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import iter_mask_instances_h5


STATUS_COLORS = {
    "clean_train": (40, 210, 90),
    "usable_inference": (255, 190, 60),
    "manual_review": (40, 130, 255),
    "reject": (50, 50, 230),
}
WARNING_COLOR = (0, 255, 255)


def make_qc_overlay_video(
    *,
    video_path: str | Path,
    mask_path: str | Path,
    tracklets_qc: pd.DataFrame,
    output_path: str | Path,
    assignments: pd.DataFrame | None = None,
    max_frames: int | None = None,
) -> Path:
    import cv2

    by_frame: dict[int, list] = defaultdict(list)
    for inst in iter_mask_instances_h5(mask_path):
        by_frame[int(inst.frame)].append(inst)

    qc_lookup = {
        (int(r["frame"]), int(r["tracklet_id"])): r
        for r in tracklets_qc.to_dict("records")
    }
    assignment_lookup = {}
    if assignments is not None and not assignments.empty:
        assignment_lookup = {
            int(r["tracklet_id"]): r.get("assigned_identity", r.get("cluster_id", "NA"))
            for r in assignments.to_dict("records")
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        if by_frame:
            first_frame = min(by_frame)
            last_frame = max(by_frame)
        else:
            first_frame = 0
            last_frame = -1
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        frame_idx = first_frame
        frames_written = 0
        while True:
            if frame_idx > last_frame:
                break
            if max_frames is not None and frames_written >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            overlay = frame.copy()
            for inst in by_frame.get(frame_idx, []):
                tracklet_id = int(inst.sam2_track_id if inst.sam2_track_id >= 0 else inst.instance_id)
                row = qc_lookup.get((frame_idx, tracklet_id), {})
                status = str(row.get("qc_status", "usable_inference"))
                auto_qc_reason = str(row.get("auto_qc_reason", "") or "")
                auto_qc_flag = bool(row.get("auto_qc_flag", False)) or bool(auto_qc_reason)
                color = STATUS_COLORS.get(status, (230, 230, 230))
                mask = np.asarray(inst.mask, dtype=np.uint8)
                color_img = np.zeros_like(frame)
                color_img[:, :] = color
                overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.65, color_img, 0.35, 0), overlay)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_color = WARNING_COLOR if auto_qc_flag else color
                contour_thickness = 4 if auto_qc_flag else 2
                cv2.drawContours(overlay, contours, -1, contour_color, contour_thickness, cv2.LINE_AA)
                x0, y0, _x1, _y1 = inst.bbox
                assigned = assignment_lookup.get(tracklet_id, "")
                label = f"SAM2 {tracklet_id} {status}"
                if assigned != "":
                    label += f" ID {assigned}"
                label_y = max(15, int(y0) - 5)
                cv2.putText(overlay, label, (int(x0), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                if auto_qc_reason:
                    cv2.putText(
                        overlay,
                        auto_qc_reason,
                        (int(x0), label_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        WARNING_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
            writer.write(overlay)
            frame_idx += 1
            frames_written += 1
    finally:
        cap.release()
        writer.release()
    return out_path
