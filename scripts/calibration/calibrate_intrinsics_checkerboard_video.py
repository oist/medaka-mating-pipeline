"""Calibrate dry camera intrinsics from the 10 mm and 25 mm air videos.

The script mines checkerboard observations densely, retains a pose-diverse
subset, rejects high reprojection-error views, and compares Brown-5 and
rational distortion on held-out frames.  All reported pixel errors refer to
the native video resolution, even though pattern discovery uses a downscaled
image for speed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAMERAS = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


@dataclass
class Observation:
    dataset: str
    frame: int
    pattern_cols: int
    pattern_rows: int
    square_mm: float
    corners: np.ndarray
    descriptor: np.ndarray
    sharpness: float

    @property
    def key(self) -> str:
        return f"{self.dataset}:f{self.frame}"

    def object_points(self) -> np.ndarray:
        grid = np.zeros((self.pattern_cols * self.pattern_rows, 3), np.float32)
        grid[:, :2] = (
            np.mgrid[0 : self.pattern_cols, 0 : self.pattern_rows]
            .T.reshape(-1, 2)
            .astype(np.float32)
            * self.square_mm
        )
        return grid


def _detect(gray: np.ndarray, scale: float) -> np.ndarray | None:
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pattern = _detect.pattern  # type: ignore[attr-defined]
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    if not ok or corners is None:
        return None
    corners = corners.astype(np.float32) / float(scale)
    cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4),
    )
    return corners.reshape(-1, 2)


def _descriptor(corners: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    center = corners.mean(axis=0) / np.array([width, height], dtype=float)
    rect = cv2.minAreaRect(corners.astype(np.float32))
    rw, rh = rect[1]
    area_fraction = max(rw * rh / float(width * height), 1e-7)
    cols = _descriptor.pattern_cols  # type: ignore[attr-defined]
    top_axis = corners[cols - 1] - corners[0]
    angle = np.arctan2(top_axis[1], top_axis[0])
    spread = np.std(corners / np.array([width, height]), axis=0)
    return np.array(
        [center[0], center[1], np.log(area_fraction), np.sin(angle), np.cos(angle), spread[0], spread[1]],
        dtype=float,
    )


def mine_video(
    video_path: Path,
    *,
    dataset: str,
    pattern_cols: int,
    pattern_rows: int,
    square_mm: float,
    stride: int,
    discovery_scale: float,
) -> tuple[list[Observation], tuple[int, int], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    observations: list[Observation] = []
    index = -1
    _detect.pattern = (pattern_cols, pattern_rows)  # type: ignore[attr-defined]
    _descriptor.pattern_cols = pattern_cols  # type: ignore[attr-defined]
    while True:
        ok = cap.grab()
        if not ok:
            break
        index += 1
        if index % stride:
            continue
        ok, bgr = cap.retrieve()
        if not ok:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners = _detect(gray, discovery_scale)
        if corners is None:
            continue
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        observations.append(
            Observation(
                dataset=dataset,
                frame=index,
                pattern_cols=pattern_cols,
                pattern_rows=pattern_rows,
                square_mm=square_mm,
                corners=corners,
                descriptor=_descriptor(corners, (width, height)),
                sharpness=sharpness,
            )
        )
    cap.release()
    return observations, (width, height), frame_count


def select_pose_diverse(observations: Sequence[Observation], maximum: int) -> list[Observation]:
    """Greedy farthest-point selection in normalized image-pose space."""
    if len(observations) <= maximum:
        return list(observations)
    x = np.stack([o.descriptor for o in observations])
    med = np.median(x, axis=0)
    scale = np.median(np.abs(x - med), axis=0)
    scale[scale < 1e-4] = 1.0
    z = (x - med) / scale
    sharp = np.asarray([o.sharpness for o in observations])
    chosen = [int(np.argmax(sharp))]
    nearest = np.linalg.norm(z - z[chosen[0]], axis=1)
    while len(chosen) < maximum:
        score = nearest.copy()
        score[chosen] = -np.inf
        candidate = int(np.argmax(score))
        chosen.append(candidate)
        nearest = np.minimum(nearest, np.linalg.norm(z - z[candidate], axis=1))
    return [observations[i] for i in chosen]


def _calibrate(
    observations: Sequence[Observation],
    image_size: tuple[int, int],
    model: str,
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    flags = 0
    if model == "rational":
        flags |= cv2.CALIB_RATIONAL_MODEL
    rms, k, d, rvecs, tvecs = cv2.calibrateCamera(
        [o.object_points() for o in observations],
        [o.corners.reshape(-1, 1, 2).astype(np.float32) for o in observations],
        image_size,
        None,
        None,
        flags=flags,
    )
    return float(rms), k, d.reshape(-1), rvecs, tvecs


def _view_errors(
    observations: Sequence[Observation],
    k: np.ndarray,
    d: np.ndarray,
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
) -> np.ndarray:
    errors = []
    for obs, rvec, tvec in zip(observations, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(obs.object_points(), rvec, tvec, k, d)
        delta = projected.reshape(-1, 2) - obs.corners
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return np.asarray(errors)


def calibrate_with_rejection(
    observations: Sequence[Observation], image_size: tuple[int, int], model: str
) -> tuple[list[Observation], float, np.ndarray, np.ndarray, np.ndarray]:
    kept = list(observations)
    for _ in range(4):
        rms, k, d, rvecs, tvecs = _calibrate(kept, image_size, model)
        per_view = _view_errors(kept, k, d, rvecs, tvecs)
        median = float(np.median(per_view))
        mad = float(np.median(np.abs(per_view - median)))
        threshold = max(0.45, median + 3.5 * 1.4826 * max(mad, 0.01))
        mask = per_view <= threshold
        if mask.all() or int(mask.sum()) < 12:
            break
        kept = [o for o, good in zip(kept, mask, strict=True) if good]
    rms, k, d, rvecs, tvecs = _calibrate(kept, image_size, model)
    per_view = _view_errors(kept, k, d, rvecs, tvecs)
    return kept, rms, k, d, per_view


def heldout_errors(
    observations: Iterable[Observation], k: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    point_errors: list[float] = []
    rows: list[dict[str, object]] = []
    for obs in observations:
        # A checkerboard is planar. IPPE returns both plane-pose branches; use
        # the branch with the smaller native-pixel reprojection error instead
        # of allowing iterative PnP to settle on the wrong branch.
        result = cv2.solvePnPGeneric(
            obs.object_points(),
            obs.corners.astype(np.float32),
            k,
            d,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not result[0]:
            continue
        candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for rvec, tvec in zip(result[1], result[2], strict=True):
            projected, _ = cv2.projectPoints(obs.object_points(), rvec, tvec, k, d)
            errors = np.linalg.norm(projected.reshape(-1, 2) - obs.corners, axis=1)
            candidates.append((errors, rvec, tvec))
        errors, _rvec, _tvec = min(candidates, key=lambda item: float(np.mean(item[0] ** 2)))
        point_errors.extend(errors.tolist())
        rows.append(
            {
                "observation": obs.key,
                "dataset": obs.dataset,
                "frame": obs.frame,
                "rms_px": float(np.sqrt(np.mean(errors**2))),
                "median_px": float(np.median(errors)),
                "p95_px": float(np.percentile(errors, 95)),
            }
        )
    return np.asarray(point_errors), rows


def _partition(selected: Sequence[Observation]) -> tuple[list[Observation], list[Observation]]:
    train: list[Observation] = []
    test: list[Observation] = []
    by_dataset: dict[str, list[Observation]] = {}
    for obs in selected:
        by_dataset.setdefault(obs.dataset, []).append(obs)
    for group in by_dataset.values():
        ordered = sorted(group, key=lambda o: o.frame)
        for index, obs in enumerate(ordered):
            (test if index % 5 == 2 else train).append(obs)
    return train, test


def _summary_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"n": 0, "median_px": None, "rms_px": None, "p95_px": None}
    return {
        "n": int(len(values)),
        "median_px": float(np.median(values)),
        "rms_px": float(np.sqrt(np.mean(values**2))),
        "p95_px": float(np.percentile(values, 95)),
    }


def _plot_qc(
    camera: str,
    observations: Sequence[Observation],
    selected: Sequence[Observation],
    heldout_by_model: dict[str, np.ndarray],
    image_size: tuple[int, int],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 5.3), layout="constrained")
    ax0 = fig.add_subplot(1, 2, 1)
    palette = ("#2563A6", "#D99024", "#C45A7A", "#6E7F40", "#7A5AA6")
    colors = {
        dataset: palette[index % len(palette)]
        for index, dataset in enumerate(sorted({o.dataset for o in observations}))
    }
    for dataset in sorted({o.dataset for o in observations}):
        mined = [o for o in observations if o.dataset == dataset]
        kept = [o for o in selected if o.dataset == dataset]
        ax0.scatter(
            [o.descriptor[0] * image_size[0] for o in mined],
            [o.descriptor[1] * image_size[1] for o in mined],
            s=10,
            alpha=0.18,
            color=colors[dataset],
        )
        ax0.scatter(
            [o.descriptor[0] * image_size[0] for o in kept],
            [o.descriptor[1] * image_size[1] for o in kept],
            s=25,
            facecolors="none",
            edgecolors=colors[dataset],
            linewidths=0.9,
            label=f"{dataset.replace('_', ' ')} selected (n={len(kept)})",
        )
    ax0.set(xlim=(0, image_size[0]), ylim=(image_size[1], 0), xlabel="image x (px)", ylabel="image y (px)")
    ax0.set_title(f"{camera}: checkerboard-center coverage")
    ax0.grid(color="#D9D9D9", linewidth=0.6)
    ax0.legend(frameon=False, fontsize=8)

    ax1 = fig.add_subplot(1, 2, 2)
    bins = np.linspace(0, max(1.0, max((v.max(initial=0.0) for v in heldout_by_model.values()))), 25)
    styles = {"brown5": ("#2563A6", "-"), "rational": ("#D99024", "--")}
    for model, values in heldout_by_model.items():
        ax1.hist(values, bins=bins, histtype="step", linewidth=2.0, color=styles[model][0], linestyle=styles[model][1], label=f"{model} (n={len(values)})")
    ax1.set(xlabel="held-out corner error (px)", ylabel="corners")
    ax1.set_title("Held-out reprojection error")
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax1.legend(frameon=False)
    fig.suptitle("Dry intrinsic calibration QC", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--discovery-scale", type=float, default=0.5)
    parser.add_argument("--max-poses-per-board", type=int, default=70)
    parser.add_argument(
        "--dataset-spec",
        action="append",
        default=None,
        metavar="LABEL,FOLDER,COLS,ROWS,SQUARE_MM",
        help=(
            "checkerboard dataset relative to --air-root; repeat to combine datasets. "
            "Defaults to the legacy 10 mm and 25 mm air recordings"
        ),
    )
    parser.add_argument("--cameras", nargs="+", choices=tuple(CAMERAS), default=list(CAMERAS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_spec:
        specs = []
        for raw in args.dataset_spec:
            fields = [field.strip() for field in raw.split(",")]
            if len(fields) != 5:
                raise ValueError(
                    f"Invalid --dataset-spec {raw!r}; expected LABEL,FOLDER,COLS,ROWS,SQUARE_MM"
                )
            label, folder, cols, rows, square = fields
            specs.append((label, folder, int(cols), int(rows), float(square)))
    else:
        specs = [
            ("air_10mm", "calib_checkerboard_10mm_air", 7, 6, 10.0),
            ("air_25mm", "calib_checkerboard_25mm_air", 9, 7, 25.0),
        ]
    combined_report: dict[str, object] = {
        "air_root": str(args.air_root),
        "native_error_units": "pixels",
        "cameras": {},
    }
    all_corner_rows: list[dict[str, object]] = []
    for camera in args.cameras:
        all_obs: list[Observation] = []
        image_size: tuple[int, int] | None = None
        mined_counts: dict[str, int] = {}
        frame_counts: dict[str, int] = {}
        for dataset, folder, cols, rows, square in specs:
            video = args.air_root / folder / camera / CAMERAS[camera]
            print(f"[{camera}] mining {dataset}: {video}", flush=True)
            obs, size, n_frames = mine_video(
                video,
                dataset=dataset,
                pattern_cols=cols,
                pattern_rows=rows,
                square_mm=square,
                stride=args.stride,
                discovery_scale=args.discovery_scale,
            )
            if image_size is not None and size != image_size:
                raise ValueError(f"Image-size mismatch for {camera}: {image_size} vs {size}")
            image_size = size
            mined_counts[dataset] = len(obs)
            frame_counts[dataset] = n_frames
            all_obs.extend(obs)
            print(f"[{camera}] {dataset}: detected {len(obs)} sampled frames", flush=True)
        assert image_size is not None
        selected: list[Observation] = []
        for dataset, _folder, _cols, _rows, _square in specs:
            selected.extend(
                select_pose_diverse([o for o in all_obs if o.dataset == dataset], args.max_poses_per_board)
            )
        train, test = _partition(selected)
        if len(train) < 12 or len(test) < 3:
            raise RuntimeError(f"Insufficient observations for {camera}: train={len(train)}, test={len(test)}")
        model_results: dict[str, dict[str, object]] = {}
        heldout_by_model: dict[str, np.ndarray] = {}
        for model in ("brown5", "rational"):
            kept, train_rms, k, d, train_view_error = calibrate_with_rejection(train, image_size, model)
            test_error, test_rows = heldout_errors(test, k, d)
            heldout_by_model[model] = test_error
            model_results[model] = {
                "train_selected_views": len(train),
                "train_kept_views": len(kept),
                "train_calibrate_rms_px": train_rms,
                "train_per_view": _summary_stats(train_view_error),
                "heldout_points": _summary_stats(test_error),
                "heldout_views": test_rows,
                "K": k.tolist(),
                "distortion": d.tolist(),
            }
        brown_p95 = float(model_results["brown5"]["heldout_points"]["p95_px"])  # type: ignore[index]
        rational_p95 = float(model_results["rational"]["heldout_points"]["p95_px"])  # type: ignore[index]
        chosen_model = "rational" if rational_p95 + 0.02 < brown_p95 else "brown5"
        final_kept, final_rms, final_k, final_d, final_per_view = calibrate_with_rejection(
            selected, image_size, chosen_model
        )
        npz_path = args.output_dir / f"intrinsics_{camera}.npz"
        np.savez_compressed(
            npz_path,
            K=final_k,
            distortion=final_d,
            image_size=np.asarray(image_size),
            distortion_model=np.asarray(chosen_model),
        )
        qc_path = args.output_dir / f"intrinsics_{camera}_qc.png"
        _plot_qc(camera, all_obs, selected, heldout_by_model, image_size, qc_path)
        selected_keys = {o.key for o in selected}
        final_keys = {o.key for o in final_kept}
        for obs in all_obs:
            for corner_id, xy in enumerate(obs.corners):
                all_corner_rows.append(
                    {
                        "camera": camera,
                        "dataset": obs.dataset,
                        "frame": obs.frame,
                        "corner_id": corner_id,
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                        "selected": obs.key in selected_keys,
                        "final_kept": obs.key in final_keys,
                    }
                )
        camera_report = {
            "image_size": list(image_size),
            "video_frame_counts": frame_counts,
            "sample_stride_frames": args.stride,
            "mined_views": mined_counts,
            "selected_views": {
                dataset: sum(o.dataset == dataset for o in selected) for dataset in mined_counts
            },
            "heldout_views": len(test),
            "candidate_models": model_results,
            "chosen_model": chosen_model,
            "selection_rule": "rational only if held-out corner P95 improves by >0.02 px",
            "final_views": len(final_kept),
            "final_calibrate_rms_px": final_rms,
            "final_per_view": _summary_stats(final_per_view),
            "K": final_k.tolist(),
            "distortion": final_d.tolist(),
            "outputs": {"npz": str(npz_path), "qc_figure": str(qc_path)},
        }
        combined_report["cameras"][camera] = camera_report  # type: ignore[index]
        print(
            f"[{camera}] chose {chosen_model}; held-out P95 brown={brown_p95:.3f}px, "
            f"rational={rational_p95:.3f}px; final RMS={final_rms:.3f}px",
            flush=True,
        )
    corners_path = args.output_dir / "checkerboard_corners.csv"
    pd.DataFrame(all_corner_rows).to_csv(corners_path, index=False)
    combined_report["outputs"] = {"corners_csv": str(corners_path)}
    report_path = args.output_dir / "intrinsics_summary.json"
    report_path.write_text(json.dumps(combined_report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
