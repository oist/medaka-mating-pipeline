"""Render native-resolution 2x2 QC videos for mixed female tracklets.

The three source-camera panels remain at their native resolution. The fourth
panel records the provisional tracklet, sparse female-GT evidence, and the
piecewise-nearest GT assignment used to label the current female. Refractive
3D projections are used only to assign geometry IDs to raw SLEAP poses. The
visible arrows follow each matched SLEAP mouth-tail axis so they stay on the
observed fish image.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from scripts.analysis.female_sex_gt import (match_geometry_to_instances,
                                            visible_node_centroids)
from scripts.analysis.geometry_swap_review import IndexedSlpPredictions
from scripts.reconstruction_3d.calibration import Calibration


SOURCE_VIDEOS = {
    "top_cam": Path("top_cam") / "cam0_top.mp4",
    "side1_cam": Path("side1_cam") / "cam1_side.mp4",
    "side2_cam": Path("side2_cam") / "cam2_side.mp4",
}
CAMERA_INDEX = {"top_cam": 0, "side1_cam": 1, "side2_cam": 2}
PANEL_POSITION = {
    "top_cam": (0, 0),
    "side1_cam": (0, 1),
    "side2_cam": (1, 0),
}
ID_COLORS_BGR = {
    "ID_1": (255, 180, 30),
    "ID_2": (30, 170, 255),
    "ID_3": (90, 220, 120),
}
FEMALE_COLOR_BGR = (255, 80, 255)
UNASSIGNED_COLOR_BGR = (155, 155, 155)
@dataclass(frozen=True)
class Scene:
    tracklet_id: str
    geometry_id: str
    tracklet_start: int
    tracklet_end: int
    render_start: int
    render_end: int
    evidence_frames: tuple[int, ...]
    evidence_ids: tuple[str, ...]


def safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_")


def sampled_source_frames(
    start_frame: int,
    end_frame: int,
    *,
    source_fps: float,
    output_fps: float,
) -> np.ndarray:
    """Return monotonic source frames sampled at the requested playback FPS."""
    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")
    if source_fps <= 0 or output_fps <= 0 or output_fps > source_fps:
        raise ValueError("FPS values must satisfy 0 < output_fps <= source_fps")
    duration = (end_frame - start_frame) / source_fps
    count = int(np.floor(duration * output_fps)) + 1
    frames = np.rint(
        start_frame + np.arange(count, dtype=float) * source_fps / output_fps
    ).astype(np.int64)
    frames = frames[(frames >= start_frame) & (frames <= end_frame)]
    return np.unique(frames)


def nearest_female_assignment(
    frame: int,
    annotation_frames: np.ndarray,
    annotation_ids: np.ndarray,
) -> tuple[str, int]:
    """Use nearest sparse GT; ties prefer the earlier annotation."""
    if not len(annotation_frames):
        raise ValueError("no female annotations are available")
    right = int(np.searchsorted(annotation_frames, frame, side="left"))
    if right <= 0:
        index = 0
    elif right >= len(annotation_frames):
        index = len(annotation_frames) - 1
    else:
        left = right - 1
        index = left if frame - annotation_frames[left] <= annotation_frames[right] - frame else right
    return str(annotation_ids[index]), int(annotation_frames[index])


def load_strict_female_gt(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    table = pd.read_csv(path)
    numeric = [
        "frame",
        "side1_reprojection_px",
        "side2_reprojection_px",
        "side1_click_to_pose_segment_px",
        "side2_click_to_pose_segment_px",
    ]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    strict = table.loc[
        table["gt_status"].eq("confirmed")
        & table["sex_label"].eq("F")
        & table["geometry_id_consistency"].eq("agree")
        & table["side1_reprojection_px"].le(8)
        & table["side2_reprojection_px"].le(8)
        & table["side1_click_to_pose_segment_px"].le(8)
        & table["side2_click_to_pose_segment_px"].le(8)
    ].copy()
    strict["frame"] = strict["frame"].astype(np.int64)
    strict.sort_values("frame", kind="stable", inplace=True)
    return (
        strict["frame"].to_numpy(dtype=np.int64),
        strict["consensus_geometry_id"].astype(str).to_numpy(),
        strict,
    )


def load_scenes(
    mixed_tracklets_csv: Path,
    mixed_evidence_csv: Path,
    *,
    subset: str,
    context_frames: int,
    maximum_frame: int,
    requested_tracklets: set[str],
) -> list[Scene]:
    tracklets = pd.read_csv(mixed_tracklets_csv)
    evidence = pd.read_csv(mixed_evidence_csv)
    tracklets = tracklets.loc[
        tracklets["subset"].eq(subset)
        & tracklets["mixed_female_nonfemale"].astype(bool)
    ].copy()
    evidence = evidence.loc[evidence["subset"].eq(subset)].copy()
    if requested_tracklets:
        tracklets = tracklets.loc[
            tracklets["clean_tracklet_id"].isin(requested_tracklets)
        ]
    scenes: list[Scene] = []
    for row in tracklets.drop_duplicates("clean_tracklet_id").itertuples():
        item = evidence.loc[
            evidence["clean_tracklet_id"].eq(row.clean_tracklet_id)
        ].sort_values("frame", kind="stable")
        scenes.append(
            Scene(
                tracklet_id=str(row.clean_tracklet_id),
                geometry_id=str(row.geometry_source_id),
                tracklet_start=int(row.tracklet_start_frame),
                tracklet_end=int(row.tracklet_end_frame),
                render_start=max(0, int(row.tracklet_start_frame) - context_frames),
                render_end=min(
                    maximum_frame, int(row.tracklet_end_frame) + context_frames
                ),
                evidence_frames=tuple(item["frame"].astype(int)),
                evidence_ids=tuple(item["consensus_geometry_id"].astype(str)),
            )
        )
    scenes.sort(key=lambda scene: scene.tracklet_start)
    missing = requested_tracklets - {scene.tracklet_id for scene in scenes}
    if missing:
        raise ValueError(f"requested tracklets not found in {subset}: {sorted(missing)}")
    return scenes


def project_geometry(
    geometry_csv: Path,
    calibration_model_dir: Path,
    frames: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    geometry = pd.read_csv(
        geometry_csv,
        usecols=["frame", "fish_id", "X", "Y", "Z"],
        low_memory=False,
    )
    geometry["frame"] = pd.to_numeric(geometry["frame"], errors="raise").astype(
        np.int64
    )
    geometry = geometry.loc[geometry["frame"].isin(set(map(int, frames)))].copy()
    geometry["fish_id"] = geometry["fish_id"].astype(str)
    xyz = geometry[["X", "Y", "Z"]].to_numpy(dtype=float)
    finite = np.isfinite(xyz).all(axis=1)
    projected = np.full((len(geometry), 3, 2), np.nan, dtype=float)
    calibration = Calibration(calibration_model_dir, backend="refractive")
    if finite.any():
        projected[finite] = np.asarray(
            calibration.project_world(xyz[finite]), dtype=float
        )
    output: dict[int, dict[str, np.ndarray]] = {}
    for row_index, row in enumerate(geometry.itertuples()):
        output.setdefault(int(row.frame), {})[str(row.fish_id)] = projected[row_index]
    return output


def _outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_identity_arrows(
    image: np.ndarray,
    projected_by_id: dict[str, np.ndarray],
    instances: np.ndarray | None,
    *,
    camera_index: int,
    female_geometry_id: str,
    match_gate_px: float,
) -> tuple[int, int, int, list[float]]:
    """Draw IDs on matched raw poses and return match diagnostics.

    The projection-to-centroid match determines identity only. Arrow endpoints
    come from the observed SLEAP tail and mouth coordinates.
    """
    height, width = image.shape[:2]
    projected_for_view: dict[str, np.ndarray] = {}
    for geometry_id, all_views in projected_by_id.items():
        values = np.asarray(all_views, dtype=float)
        if values.shape != (3, 2):
            continue
        point = values[camera_index]
        if np.isfinite(point).all():
            projected_for_view[str(geometry_id)] = point

    matches = match_geometry_to_instances(
        projected_for_view,
        instances,
        gate_px=match_gate_px,
    )
    if instances is None:
        return 0, len(projected_for_view), 0, []
    poses = np.asarray(instances, dtype=float)
    centers = visible_node_centroids(poses)
    distances: list[float] = []

    unassigned_count = 0
    for instance_index, pose in enumerate(poses):
        if instance_index in matches or len(pose) < 2:
            continue
        mouth = np.asarray(pose[0], dtype=float)
        tail = np.asarray(pose[1], dtype=float)
        if not (np.isfinite(mouth).all() and np.isfinite(tail).all()):
            continue
        start = np.rint(tail).astype(int)
        target = np.rint(mouth).astype(int)
        if not (
            0 <= start[0] < width
            and 0 <= start[1] < height
            and 0 <= target[0] < width
            and 0 <= target[1] < height
        ):
            continue
        cv2.arrowedLine(
            image,
            tuple(start),
            tuple(target),
            (0, 0, 0),
            3,
            cv2.LINE_AA,
            0,
            0.13,
        )
        cv2.arrowedLine(
            image,
            tuple(start),
            tuple(target),
            UNASSIGNED_COLOR_BGR,
            1,
            cv2.LINE_AA,
            0,
            0.13,
        )
        unassigned_count += 1

    for instance_index, geometry_id in sorted(matches.items(), key=lambda item: item[1]):
        if instance_index < 0 or instance_index >= len(poses):
            continue
        pose = poses[instance_index]
        if len(pose) < 2:
            continue
        mouth = np.asarray(pose[0], dtype=float)
        tail = np.asarray(pose[1], dtype=float)
        if not (np.isfinite(mouth).all() and np.isfinite(tail).all()):
            continue
        if geometry_id in projected_for_view and np.isfinite(centers[instance_index]).all():
            distances.append(
                float(np.linalg.norm(projected_for_view[geometry_id] - centers[instance_index]))
            )

        start = np.rint(tail).astype(int)
        target = np.rint(mouth).astype(int)
        if not (
            0 <= start[0] < width
            and 0 <= start[1] < height
            and 0 <= target[0] < width
            and 0 <= target[1] < height
        ):
            continue
        label = "F" if geometry_id == female_geometry_id else geometry_id
        color = (
            FEMALE_COLOR_BGR
            if geometry_id == female_geometry_id
            else ID_COLORS_BGR.get(geometry_id, (230, 230, 230))
        )
        cv2.arrowedLine(
            image,
            tuple(start),
            tuple(target),
            (0, 0, 0),
            3,
            cv2.LINE_AA,
            0,
            0.13,
        )
        cv2.arrowedLine(
            image,
            tuple(start),
            tuple(target),
            color,
            1,
            cv2.LINE_AA,
            0,
            0.13,
        )

        direction = mouth - tail
        norm = float(np.linalg.norm(direction))
        normal = np.array([-direction[1], direction[0]], dtype=float)
        if norm > 1e-6:
            normal *= 13.0 / norm
        else:
            normal = np.array([8.0, -8.0])
        label_origin = np.rint(mouth + normal).astype(int)
        label_origin[0] = int(np.clip(label_origin[0], 4, width - 52))
        label_origin[1] = int(np.clip(label_origin[1], 16, height - 4))
        _outlined_text(
            image,
            label,
            tuple(label_origin),
            color=color,
            scale=0.42,
            thickness=1,
        )
    return len(distances), len(projected_for_view), unassigned_count, distances


def _put_info_line(
    panel: np.ndarray,
    text: str,
    y: int,
    *,
    color: tuple[int, int, int] = (225, 225, 225),
    scale: float = 0.58,
) -> None:
    cv2.putText(
        panel,
        text,
        (52, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_info_panel(
    panel_size: int,
    scene: Scene,
    *,
    source_frame: int,
    source_fps: float,
    female_geometry_id: str,
    nearest_gt_frame: int,
) -> np.ndarray:
    panel = np.full((panel_size, panel_size, 3), 20, dtype=np.uint8)
    _put_info_line(panel, "MIXED FEMALE TRACKLET QC", 62, color=(255, 255, 255), scale=0.72)
    _put_info_line(panel, scene.tracklet_id, 118, color=ID_COLORS_BGR.get(scene.geometry_id, (230, 230, 230)), scale=0.68)
    _put_info_line(
        panel,
        f"tracklet span  {scene.tracklet_start:,} - {scene.tracklet_end:,}",
        176,
    )
    _put_info_line(
        panel,
        f"render span    {scene.render_start:,} - {scene.render_end:,}  (+/-5 s)",
        218,
    )
    _put_info_line(
        panel,
        f"current        frame {source_frame:,}   t={source_frame/source_fps:8.3f} s",
        282,
        color=(255, 255, 255),
    )
    status = "F" if female_geometry_id == scene.geometry_id else "NOT-F"
    status_color = FEMALE_COLOR_BGR if status == "F" else (100, 180, 255)
    _put_info_line(
        panel,
        f"tracklet state {status}   (nearest-GT F={female_geometry_id})",
        340,
        color=status_color,
        scale=0.66,
    )
    _put_info_line(
        panel,
        f"GT source      frame {nearest_gt_frame:,}   midpoint inference",
        386,
        color=(185, 185, 185),
        scale=0.51,
    )
    _put_info_line(panel, "Strict female-GT evidence inside tracklet:", 466, color=(245, 245, 245), scale=0.55)
    for index, (frame, identity) in enumerate(zip(scene.evidence_frames, scene.evidence_ids)):
        _put_info_line(
            panel,
            f"f{frame:,}: F={identity}",
            508 + index * 36,
            color=(220, 220, 220),
            scale=0.49,
        )

    _put_info_line(panel, "Overlay legend", 790, color=(245, 245, 245), scale=0.57)
    _put_info_line(panel, "F  = nearest strict female GT", 830, color=FEMALE_COLOR_BGR, scale=0.50)
    _put_info_line(panel, "ID = current geometry ID of each male", 866, color=(205, 205, 205), scale=0.50)
    _put_info_line(panel, "Gray arrows = unassigned raw SLEAP predictions", 902, color=UNASSIGNED_COLOR_BGR, scale=0.46)
    _put_info_line(panel, "All arrows follow raw SLEAP mouth-tail poses", 938, color=(170, 170, 170), scale=0.46)

    x0, x1, y = 70, panel_size - 70, panel_size - 120
    cv2.line(panel, (x0, y), (x1, y), (90, 90, 90), 2, cv2.LINE_AA)

    def x_for(frame: int) -> int:
        fraction = (frame - scene.render_start) / max(1, scene.render_end - scene.render_start)
        return int(round(x0 + np.clip(fraction, 0.0, 1.0) * (x1 - x0)))

    cv2.line(
        panel,
        (x_for(scene.tracklet_start), y),
        (x_for(scene.tracklet_end), y),
        ID_COLORS_BGR.get(scene.geometry_id, (220, 220, 220)),
        8,
        cv2.LINE_AA,
    )
    for frame, identity in zip(scene.evidence_frames, scene.evidence_ids):
        color = FEMALE_COLOR_BGR if identity == scene.geometry_id else (100, 180, 255)
        x = x_for(frame)
        cv2.line(panel, (x, y - 22), (x, y + 22), color, 3, cv2.LINE_AA)
    current_x = x_for(source_frame)
    cv2.line(panel, (current_x, y - 36), (current_x, y + 36), (255, 255, 255), 2, cv2.LINE_AA)
    _put_info_line(panel, "context", panel_size - 62, color=(130, 130, 130), scale=0.44)
    return panel


def open_captures(video_dir: Path) -> tuple[dict[str, cv2.VideoCapture], int, int, int]:
    captures: dict[str, cv2.VideoCapture] = {}
    width = height = frame_count = None
    for view, relative in SOURCE_VIDEOS.items():
        path = video_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing source video: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video: {path}")
        this_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        this_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        this_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width is None:
            width, height, frame_count = this_width, this_height, this_count
        elif (this_width, this_height, this_count) != (width, height, frame_count):
            raise ValueError("source videos must have identical dimensions and frame counts")
        captures[view] = capture
    assert width is not None and height is not None and frame_count is not None
    if width != height:
        raise ValueError("native 2x2 renderer currently requires square camera frames")
    return captures, width, height, frame_count


def read_frame(
    capture: cv2.VideoCapture,
    target_frame: int,
    next_frame: int,
) -> tuple[np.ndarray, int]:
    while next_frame < target_frame:
        if not capture.grab():
            raise RuntimeError(f"failed to advance video to frame {target_frame}")
        next_frame += 1
    ok, image = capture.read()
    if not ok or image is None:
        raise RuntimeError(f"failed to decode source frame {target_frame}")
    return image, next_frame + 1


def ffmpeg_command(
    ffmpeg: str,
    output: Path,
    *,
    width: int,
    height: int,
    fps: float,
    encoder: str,
    crf: int,
) -> list[str]:
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
    ]
    if encoder == "h264_nvenc":
        command += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(crf),
            "-b:v",
            "0",
        ]
    else:
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(crf),
        ]
    command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    return command


def render_scene(
    scene: Scene,
    *,
    video_dir: Path,
    slp_sources: dict[str, IndexedSlpPredictions],
    projection_by_frame: dict[int, dict[str, np.ndarray]],
    annotation_frames: np.ndarray,
    annotation_ids: np.ndarray,
    source_fps: float,
    output_fps: float,
    output_dir: Path,
    encoder: str,
    crf: int,
    slp_match_gate_px: float,
    max_output_frames: int | None,
) -> dict[str, Any]:
    captures, panel_width, panel_height, frame_count = open_captures(video_dir)
    if scene.render_end >= frame_count:
        raise ValueError(f"scene ends beyond video frame count: {scene.render_end}")
    source_frames = sampled_source_frames(
        scene.render_start,
        scene.render_end,
        source_fps=source_fps,
        output_fps=output_fps,
    )
    if max_output_frames is not None:
        source_frames = source_frames[:max_output_frames]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{safe_name(scene.tracklet_id)}_f{scene.render_start:09d}_{scene.render_end:09d}"
        "_native_2x2.mp4"
    )
    log_path = output.with_suffix(".ffmpeg.log")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    command = ffmpeg_command(
        ffmpeg,
        output,
        width=panel_width * 2,
        height=panel_height * 2,
        fps=output_fps,
        encoder=encoder,
        crf=crf,
    )
    for capture in captures.values():
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_frames[0]))
    next_frames = {view: int(source_frames[0]) for view in captures}
    match_counts = {view: 0 for view in SOURCE_VIDEOS}
    projection_counts = {view: 0 for view in SOURCE_VIDEOS}
    unassigned_counts = {view: 0 for view in SOURCE_VIDEOS}
    match_distances = {view: [] for view in SOURCE_VIDEOS}
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_handle,
        )
        assert process.stdin is not None
        try:
            for output_index, source_frame_value in enumerate(source_frames):
                source_frame = int(source_frame_value)
                female_id, gt_frame = nearest_female_assignment(
                    source_frame, annotation_frames, annotation_ids
                )
                canvas = np.full(
                    (panel_height * 2, panel_width * 2, 3), 20, dtype=np.uint8
                )
                projections = projection_by_frame.get(source_frame, {})
                for view in SOURCE_VIDEOS:
                    image, next_frames[view] = read_frame(
                        captures[view], source_frame, next_frames[view]
                    )
                    instances = slp_sources[view].get(source_frame, default=None)
                    matched, projected, unassigned, distances = draw_identity_arrows(
                        image,
                        projections,
                        instances,
                        camera_index=CAMERA_INDEX[view],
                        female_geometry_id=female_id,
                        match_gate_px=slp_match_gate_px,
                    )
                    match_counts[view] += matched
                    projection_counts[view] += projected
                    unassigned_counts[view] += unassigned
                    match_distances[view].extend(distances)
                    _outlined_text(
                        image,
                        view,
                        (18, 30),
                        color=(245, 245, 245),
                        scale=0.50,
                        thickness=1,
                    )
                    row, column = PANEL_POSITION[view]
                    canvas[
                        row * panel_height : (row + 1) * panel_height,
                        column * panel_width : (column + 1) * panel_width,
                    ] = image
                canvas[
                    panel_height : panel_height * 2,
                    panel_width : panel_width * 2,
                ] = draw_info_panel(
                    panel_width,
                    scene,
                    source_frame=source_frame,
                    source_fps=source_fps,
                    female_geometry_id=female_id,
                    nearest_gt_frame=gt_frame,
                )
                process.stdin.write(canvas.tobytes())
                if output_index and output_index % 300 == 0:
                    elapsed = time.time() - started
                    rate = (output_index + 1) / max(elapsed, 1e-9)
                    remaining = (len(source_frames) - output_index - 1) / max(rate, 1e-9)
                    print(
                        f"  {scene.tracklet_id}: {output_index + 1}/{len(source_frames)} "
                        f"({rate:.1f} fps, ETA {remaining/60:.1f} min)",
                        flush=True,
                    )
            process.stdin.close()
            return_code = process.wait()
        except BaseException:
            process.stdin.close()
            process.terminate()
            process.wait(timeout=10)
            raise
        finally:
            for capture in captures.values():
                capture.release()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {return_code}; see {log_path}"
        )
    elapsed = time.time() - started
    print(
        f"Wrote {output} ({len(source_frames)} frames, {elapsed/60:.1f} min render)",
        flush=True,
    )
    return {
        "tracklet_id": scene.tracklet_id,
        "geometry_id": scene.geometry_id,
        "tracklet_start": scene.tracklet_start,
        "tracklet_end": scene.tracklet_end,
        "render_start": scene.render_start,
        "render_end": scene.render_end,
        "context_seconds": (scene.tracklet_start - scene.render_start) / source_fps,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "source_frame_count": int(len(source_frames)),
        "output_width": panel_width * 2,
        "output_height": panel_height * 2,
        "female_assignment": "piecewise nearest strict GT; ties use earlier frame",
        "overlay_position_source": "raw SLEAP mouth-tail pose matched to refractive projection",
        "slp_match_gate_px": slp_match_gate_px,
        "slp_match_diagnostics": {
            view: {
                "matched_poses": match_counts[view],
                "finite_projected_poses": projection_counts[view],
                "unassigned_pose_arrows": unassigned_counts[view],
                "match_rate": (
                    match_counts[view] / projection_counts[view]
                    if projection_counts[view]
                    else None
                ),
                "projection_to_sleap_centroid_px": {
                    "median": (
                        float(np.median(match_distances[view]))
                        if match_distances[view]
                        else None
                    ),
                    "p95": (
                        float(np.percentile(match_distances[view], 95))
                        if match_distances[view]
                        else None
                    ),
                    "max": (
                        float(np.max(match_distances[view]))
                        if match_distances[view]
                        else None
                    ),
                },
            }
            for view in SOURCE_VIDEOS
        },
        "evidence": [
            {"frame": frame, "female_geometry_id": identity}
            for frame, identity in zip(scene.evidence_frames, scene.evidence_ids)
        ],
        "output": str(output.resolve()),
        "ffmpeg_log": str(log_path.resolve()),
        "render_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--slp-dir", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--calibration-model-dir", type=Path, required=True)
    parser.add_argument("--annotation-details-csv", type=Path, required=True)
    parser.add_argument("--mixed-tracklets-csv", type=Path, required=True)
    parser.add_argument("--mixed-evidence-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", default="strict_two_view_8px")
    parser.add_argument("--tracklet", action="append", default=[])
    parser.add_argument("--context-seconds", type=float, default=5.0)
    parser.add_argument("--source-fps", type=float, default=166.0)
    parser.add_argument("--output-fps", type=float, default=60.0)
    parser.add_argument(
        "--encoder", choices=("h264_nvenc", "libx264"), default="h264_nvenc"
    )
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument(
        "--slp-match-gate-px",
        type=float,
        default=80.0,
        help="Maximum projection-to-SLEAP centroid distance for identity matching.",
    )
    parser.add_argument(
        "--max-output-frames",
        type=int,
        default=None,
        help="Testing only: stop each scene after this many output frames.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.context_seconds < 0:
        raise ValueError("--context-seconds must be non-negative")
    if args.slp_match_gate_px <= 0:
        raise ValueError("--slp-match-gate-px must be positive")
    annotation_frames, annotation_ids, _strict = load_strict_female_gt(
        args.annotation_details_csv
    )
    captures, panel_width, panel_height, frame_count = open_captures(args.video_dir)
    for capture in captures.values():
        capture.release()
    context_frames = int(round(args.context_seconds * args.source_fps))
    scenes = load_scenes(
        args.mixed_tracklets_csv,
        args.mixed_evidence_csv,
        subset=args.subset,
        context_frames=context_frames,
        maximum_frame=frame_count - 1,
        requested_tracklets=set(args.tracklet),
    )
    if not scenes:
        raise ValueError("no mixed tracklet scenes matched the requested subset")
    all_source_frames = np.unique(
        np.concatenate(
            [
                sampled_source_frames(
                    scene.render_start,
                    scene.render_end,
                    source_fps=args.source_fps,
                    output_fps=args.output_fps,
                )[: args.max_output_frames]
                if args.max_output_frames is not None
                else sampled_source_frames(
                    scene.render_start,
                    scene.render_end,
                    source_fps=args.source_fps,
                    output_fps=args.output_fps,
                )
                for scene in scenes
            ]
        )
    )
    print(
        f"Preparing {len(scenes)} scenes at native {panel_width}x{panel_height} per view; "
        f"{len(all_source_frames):,} distinct source frames",
        flush=True,
    )
    projections = project_geometry(
        args.geometry_csv, args.calibration_model_dir, all_source_frames
    )
    slp_sources: dict[str, IndexedSlpPredictions] = {}
    try:
        for view in SOURCE_VIDEOS:
            path = args.slp_dir / f"{view}.slp"
            if not path.is_file():
                raise FileNotFoundError(f"missing SLP predictions: {path}")
            slp_sources[view] = IndexedSlpPredictions(
                path, node_names=("mouth", "tail")
            )
        results = []
        for scene in scenes:
            results.append(
                render_scene(
                    scene,
                    video_dir=args.video_dir,
                    slp_sources=slp_sources,
                    projection_by_frame=projections,
                    annotation_frames=annotation_frames,
                    annotation_ids=annotation_ids,
                    source_fps=args.source_fps,
                    output_fps=args.output_fps,
                    output_dir=args.output_dir,
                    encoder=args.encoder,
                    crf=args.crf,
                    slp_match_gate_px=args.slp_match_gate_px,
                    max_output_frames=args.max_output_frames,
                )
            )
    finally:
        for source in slp_sources.values():
            source.close()
    manifest = {
        "schema_version": 1,
        "layout": "native-resolution 2x2: top, side1, side2, GT information",
        "panel_width": panel_width,
        "panel_height": panel_height,
        "overlay": {
            "female_label": "F",
            "male_labels": "current geometry IDs",
            "female_assignment": "piecewise nearest strict GT; midpoint transitions",
            "position_source": "raw SLEAP mouth-tail pose matched to refractive projection",
            "unassigned_predictions": "thin gray mouth-tail arrows without text labels",
            "slp_match_gate_px": args.slp_match_gate_px,
            "line_thickness_px": 1,
            "font_scale": 0.42,
        },
        "scenes": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "female_tracklet_scene_videos_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
