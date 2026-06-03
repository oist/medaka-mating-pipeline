from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sam2_id_experiment.src.sam2_adapter.mask_io import (
    MaskInstance,
    bbox_from_mask,
    write_mask_instances_h5,
)


@dataclass(frozen=True)
class FrameWindowInfo:
    frame_dir: Path
    original_shape: tuple[int, int]
    inference_shape: tuple[int, int]
    sampled_frames: list[int]
    cache_used: bool


def run_sam2_video_segmentation(*, video_path: Path, output_path: Path, config: dict) -> dict:
    """Run the configured segmentation backend and write pseudo-mask HDF5.

    ``runner=opencv_bgsub`` is a dependency-light bootstrap for the first
    top-camera window. ``runner=sam2_video_predictor`` uses Meta's SAM2 video
    predictor and requires manual or mined initial prompts in the config.
    """
    runner = str(config.get("runner", "sam2_video_predictor")).lower()
    if runner in {"opencv_bgsub", "bgsub", "background_subtraction"}:
        return _run_opencv_bgsub(video_path=video_path, output_path=output_path, config=config)
    if runner in {"sam2", "sam2_video", "sam2_video_predictor"}:
        return _run_sam2_video_predictor(video_path=video_path, output_path=output_path, config=config)
    raise ValueError(f"Unknown segmentation runner: {runner}")


def _frame_window(config: dict, *, total_frames: int | None = None) -> tuple[int, int]:
    start_frame = int(config.get("start_frame") or 0)
    end_value = config.get("end_frame")
    if end_value is None:
        if total_frames is None or total_frames <= 0:
            raise ValueError("end_frame is required when video frame count is unknown.")
        end_frame = total_frames - 1
    else:
        end_frame = int(end_value)
    if end_frame < start_frame:
        raise ValueError(f"end_frame {end_frame} is before start_frame {start_frame}")
    return start_frame, end_frame


def _run_opencv_bgsub(*, video_path: Path, output_path: Path, config: dict) -> dict:
    import cv2

    bg_cfg = dict(config.get("opencv_bgsub") or {})
    threshold = int(bg_cfg.get("threshold", 24))
    open_px = int(bg_cfg.get("open_px", 3))
    close_px = int(bg_cfg.get("close_px", 7))
    min_area = int(bg_cfg.get("min_area", 80))
    max_area = int(bg_cfg.get("max_area", 8000))
    background_samples = int(bg_cfg.get("background_samples", 80))
    max_track_distance = float(bg_cfg.get("max_track_distance_px", 80))
    n_expected = int(config.get("n_animals_expected") or 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame, end_frame = _frame_window(config, total_frames=total_frames)

    background = _median_background(
        cap,
        start_frame=start_frame,
        end_frame=end_frame,
        max_samples=background_samples,
    )
    instances: list[MaskInstance] = []
    active_tracks: dict[int, tuple[float, float, int]] = {}
    next_track_id = 0
    image_shape: tuple[int, int] | None = None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            image_shape = frame_bgr.shape[:2]
            fg = _foreground_mask(
                frame_bgr,
                background,
                threshold=threshold,
                open_px=open_px,
                close_px=close_px,
            )
            components = _component_masks(
                fg,
                min_area=min_area,
                max_area=max_area,
            )
            components.sort(key=lambda c: c["area"], reverse=True)
            if n_expected > 0:
                components = components[:n_expected]
            assignments = _assign_tracks(
                components,
                active_tracks,
                next_track_id=next_track_id,
                frame_idx=frame_idx,
                max_distance=max_track_distance,
            )
            next_track_id = assignments["next_track_id"]
            active_tracks = assignments["active_tracks"]
            for instance_id, comp in enumerate(components):
                track_id = int(comp["track_id"])
                mask = comp["mask"].astype(bool)
                instances.append(MaskInstance(
                    frame=frame_idx,
                    instance_id=instance_id,
                    sam2_track_id=track_id,
                    mask=mask,
                    bbox=bbox_from_mask(mask),
                    score=float(comp.get("score", 0.90)),
                ))
    finally:
        cap.release()

    write_mask_instances_h5(output_path, instances, image_shape=image_shape)
    return {
        "runner": "opencv_bgsub",
        "n_instances": len(instances),
        "n_frames_requested": int(end_frame - start_frame + 1),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "background_samples": background_samples,
    }


def _median_background(cap, *, start_frame: int, end_frame: int, max_samples: int) -> np.ndarray:
    import cv2

    frames = np.linspace(start_frame, end_frame, max(1, max_samples)).round().astype(int)
    samples: list[np.ndarray] = []
    for frame_idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if ok:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if not samples:
        raise RuntimeError("No frames read while estimating background.")
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def _foreground_mask(
    frame_bgr: np.ndarray,
    background_gray: np.ndarray,
    *,
    threshold: int,
    open_px: int,
    close_px: int,
) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background_gray)
    _ret, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    if open_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _component_masks(mask: np.ndarray, *, min_area: int, max_area: int) -> list[dict]:
    import cv2

    n, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    components: list[dict] = []
    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area or area > max_area or w <= 0 or h <= 0:
            continue
        comp_mask = labels == label
        cx, cy = centroids[label]
        components.append({
            "mask": comp_mask,
            "centroid": (float(cx), float(cy)),
            "area": area,
            "bbox": (x, y, x + w, y + h),
            "score": 0.90,
        })
    return components


def _assign_tracks(
    components: list[dict],
    active_tracks: dict[int, tuple[float, float, int]],
    *,
    next_track_id: int,
    frame_idx: int,
    max_distance: float,
) -> dict:
    updated: dict[int, tuple[float, float, int]] = {}
    unmatched_tracks = set(active_tracks)
    unmatched_components = set(range(len(components)))
    costs: list[tuple[float, int, int]] = []
    for track_id, (tx, ty, last_frame) in active_tracks.items():
        frame_gap = max(1, frame_idx - last_frame)
        for comp_idx, comp in enumerate(components):
            cx, cy = comp["centroid"]
            dist = float(np.hypot(cx - tx, cy - ty))
            if dist <= max_distance * frame_gap:
                costs.append((dist, track_id, comp_idx))
    costs.sort()
    for _dist, track_id, comp_idx in costs:
        if track_id not in unmatched_tracks or comp_idx not in unmatched_components:
            continue
        components[comp_idx]["track_id"] = track_id
        cx, cy = components[comp_idx]["centroid"]
        updated[track_id] = (cx, cy, frame_idx)
        unmatched_tracks.remove(track_id)
        unmatched_components.remove(comp_idx)

    for comp_idx in sorted(unmatched_components):
        track_id = next_track_id
        next_track_id += 1
        components[comp_idx]["track_id"] = track_id
        cx, cy = components[comp_idx]["centroid"]
        updated[track_id] = (cx, cy, frame_idx)

    # Keep unmatched tracks alive briefly, so one missed frame does not
    # immediately force a new provisional id.
    for track_id in unmatched_tracks:
        tx, ty, last_frame = active_tracks[track_id]
        if frame_idx - last_frame <= 5:
            updated[track_id] = (tx, ty, last_frame)
    return {"active_tracks": updated, "next_track_id": next_track_id}


def _run_sam2_video_predictor(*, video_path: Path, output_path: Path, config: dict) -> dict:
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 is not installed. Install Meta's facebookresearch/sam2 package "
            "or use sam2.runner: opencv_bgsub for the bootstrap mask pass."
        ) from exc

    prompts = list(config.get("prompts") or [])
    if not prompts:
        raise ValueError(
            "sam2_video_predictor needs sam2.prompts in the config. Add one "
            "initial-frame positive point prompt per fish, or keep runner: opencv_bgsub."
        )

    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = str(config["checkpoint"])
    model_cfg = str(config["model_cfg"])
    max_side_px = int(config.get("max_side_px") or 0)
    jpeg_quality = int(config.get("jpeg_quality") or 95)
    frame_stride = max(1, int(config.get("frame_stride") or 1))
    vos_optimized = bool(config.get("vos_optimized", False))
    if vos_optimized and importlib.util.find_spec("triton") is None:
        print(
            "sam2.vos_optimized=true requires Triton for torch.compile, but Triton "
            "is not installed. Falling back to vos_optimized=false for this run."
        )
        vos_optimized = False
    async_loading_frames = bool(config.get("async_loading_frames", False))
    offload_video_to_cpu = bool(config.get("offload_video_to_cpu", False))
    offload_state_to_cpu = bool(config.get("offload_state_to_cpu", False))
    start_frame, end_frame = _frame_window(config)

    with _frame_window_context(video_path, config, start_frame, end_frame, frame_stride, max_side_px, jpeg_quality) as frame_info:
        image_shape = frame_info.original_shape
        infer_shape = frame_info.inference_shape
        scale_x = infer_shape[1] / image_shape[1]
        scale_y = infer_shape[0] / image_shape[0]
        abs_to_local = {frame: idx for idx, frame in enumerate(frame_info.sampled_frames)}
        predictor = build_sam2_video_predictor(
            model_cfg,
            checkpoint,
            device=device,
            vos_optimized=vos_optimized,
        )
        instances: list[MaskInstance] = []
        prompt_adjustments: list[dict] = []
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else _null_context()
        )
        with torch.inference_mode(), autocast_ctx:
            state = predictor.init_state(
                video_path=str(frame_info.frame_dir),
                async_loading_frames=async_loading_frames,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )
            for prompt in prompts:
                points = np.asarray(prompt["points"], dtype=np.float32).copy()
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y
                labels = np.asarray(prompt.get("labels", [1] * len(points)), dtype=np.int32)
                requested_local = int(prompt.get("frame", 0))
                requested_abs = start_frame + requested_local
                sampled_abs = _nearest_sampled_frame(requested_abs, frame_info.sampled_frames)
                sampled_local = abs_to_local[sampled_abs]
                if sampled_abs != requested_abs:
                    prompt_adjustments.append({
                        "obj_id": int(prompt["obj_id"]),
                        "requested_frame": requested_abs,
                        "sampled_frame": sampled_abs,
                    })
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=sampled_local,
                    obj_id=int(prompt["obj_id"]),
                    points=points,
                    labels=labels,
                )

            for local_frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                frame_idx = int(frame_info.sampled_frames[int(local_frame_idx)])
                masks = (mask_logits > 0).detach().cpu().numpy()
                for instance_id, obj_id in enumerate(obj_ids):
                    mask = np.squeeze(masks[instance_id]).astype(bool)
                    if mask.shape != image_shape:
                        mask = _resize_bool_mask(mask, image_shape)
                    instances.append(MaskInstance(
                        frame=frame_idx,
                        instance_id=instance_id,
                        sam2_track_id=int(obj_id),
                        mask=mask,
                        bbox=bbox_from_mask(mask),
                        score=1.0,
                    ))

    write_mask_instances_h5(output_path, instances, image_shape=image_shape)
    return {
        "runner": "sam2_video_predictor",
        "n_instances": len(instances),
        "n_prompts": len(prompts),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "original_shape": list(image_shape),
        "inference_shape": list(infer_shape),
        "max_side_px": max_side_px,
        "frame_stride": frame_stride,
        "n_sampled_frames": len(frame_info.sampled_frames),
        "cache_used": frame_info.cache_used,
        "vos_optimized": vos_optimized,
        "async_loading_frames": async_loading_frames,
        "prompt_frame_adjustments": prompt_adjustments,
    }


class _frame_window_context:
    def __init__(
        self,
        video_path: Path,
        config: dict,
        start_frame: int,
        end_frame: int,
        frame_stride: int,
        max_side_px: int,
        jpeg_quality: int,
    ) -> None:
        self.video_path = video_path
        self.config = config
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_stride = frame_stride
        self.max_side_px = max_side_px
        self.jpeg_quality = jpeg_quality
        self._tmp: tempfile.TemporaryDirectory | None = None
        self.info: FrameWindowInfo | None = None

    def __enter__(self) -> FrameWindowInfo:
        cache_dir_value = self.config.get("frame_cache_dir")
        cache_frames = bool(self.config.get("cache_frames", False) and cache_dir_value)
        refresh = bool(self.config.get("refresh_frame_cache", False))
        if cache_frames:
            frame_dir = _resolve_frame_cache_dir(
                str(cache_dir_value),
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                frame_stride=self.frame_stride,
                max_side_px=self.max_side_px,
            )
        else:
            self._tmp = tempfile.TemporaryDirectory(prefix="sam2_frames_")
            frame_dir = Path(self._tmp.name)
        self.info = _extract_frame_window_to_jpegs(
            self.video_path,
            frame_dir,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            frame_stride=self.frame_stride,
            max_side_px=self.max_side_px,
            jpeg_quality=self.jpeg_quality,
            refresh=refresh,
        )
        return self.info

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._tmp is not None:
            self._tmp.cleanup()
        return False


def _resolve_frame_cache_dir(
    template: str,
    *,
    start_frame: int,
    end_frame: int,
    frame_stride: int,
    max_side_px: int,
) -> Path:
    rendered = template.format(
        start_frame=start_frame,
        end_frame=end_frame,
        frame_stride=frame_stride,
        max_side_px=max_side_px,
    )
    return Path(rendered) / f"f{start_frame}-{end_frame}_s{frame_stride}_m{max_side_px or 'orig'}"


def _sampled_frames(start_frame: int, end_frame: int, frame_stride: int) -> list[int]:
    frames = list(range(start_frame, end_frame + 1, max(1, frame_stride)))
    if not frames or frames[-1] != end_frame:
        frames.append(end_frame)
    return frames


def _nearest_sampled_frame(frame: int, sampled_frames: list[int]) -> int:
    return min(sampled_frames, key=lambda f: abs(f - frame))


def _extract_frame_window_to_jpegs(
    video_path: Path,
    frame_dir: Path,
    *,
    start_frame: int,
    end_frame: int,
    frame_stride: int = 1,
    max_side_px: int = 0,
    jpeg_quality: int = 95,
    refresh: bool = False,
) -> FrameWindowInfo:
    import cv2

    frame_dir.mkdir(parents=True, exist_ok=True)
    sampled_frames = _sampled_frames(start_frame, end_frame, frame_stride)
    manifest_path = frame_dir / "manifest.json"
    if not refresh and _frame_cache_is_complete(manifest_path, frame_dir, sampled_frames):
        manifest = _read_frame_cache_manifest(manifest_path)
        return FrameWindowInfo(
            frame_dir=frame_dir,
            original_shape=tuple(manifest["original_shape"]),
            inference_shape=tuple(manifest["inference_shape"]),
            sampled_frames=[int(v) for v in manifest["sampled_frames"]],
            cache_used=True,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    original_shape: tuple[int, int] | None = None
    inference_shape: tuple[int, int] | None = None
    params = [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, jpeg_quality)))]
    try:
        for local_idx, frame_idx in enumerate(sampled_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            original_shape = frame.shape[:2]
            frame_out = _resize_frame_for_sam2(frame, max_side_px=max_side_px)
            inference_shape = frame_out.shape[:2]
            cv2.imwrite(str(frame_dir / f"{local_idx:06d}.jpg"), frame_out, params)
    finally:
        cap.release()
    if original_shape is None or inference_shape is None:
        raise RuntimeError(f"No frames extracted from {video_path}")
    manifest = {
        "original_shape": original_shape,
        "inference_shape": inference_shape,
        "sampled_frames": sampled_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_stride": frame_stride,
        "max_side_px": max_side_px,
    }
    _write_frame_cache_manifest(manifest_path, manifest)
    return FrameWindowInfo(
        frame_dir=frame_dir,
        original_shape=original_shape,
        inference_shape=inference_shape,
        sampled_frames=sampled_frames,
        cache_used=False,
    )


def _frame_cache_is_complete(manifest_path: Path, frame_dir: Path, sampled_frames: list[int]) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = _read_frame_cache_manifest(manifest_path)
    except (OSError, ValueError, KeyError):
        return False
    if [int(v) for v in manifest.get("sampled_frames", [])] != sampled_frames:
        return False
    return all((frame_dir / f"{i:06d}.jpg").exists() for i in range(len(sampled_frames)))


def _read_frame_cache_manifest(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame_cache_manifest(path: Path, manifest: dict) -> None:
    import json

    serializable = dict(manifest)
    serializable["original_shape"] = list(serializable["original_shape"])
    serializable["inference_shape"] = list(serializable["inference_shape"])
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")


def _resize_frame_for_sam2(frame: np.ndarray, *, max_side_px: int) -> np.ndarray:
    if max_side_px <= 0:
        return frame
    import cv2

    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side_px:
        return frame
    scale = max_side_px / float(longest)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    out = cv2.resize(
        mask.astype(np.uint8),
        (int(shape[1]), int(shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )
    return out.astype(bool)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
