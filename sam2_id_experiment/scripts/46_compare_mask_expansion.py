from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare Detectron2 body masks, several dilations, and SAM2 masks on a "
            "bounded set of shared frames. SAM2 is a pseudo-reference, not fin ground truth."
        )
    )
    p.add_argument("--detectron-masks", required=True)
    p.add_argument("--sam-masks", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--experiment", default="")
    p.add_argument("--view", default="")
    p.add_argument("--video", default="", help="Optional source video recorded in the annotation manifest.")
    p.add_argument("--radii", default="0,2,4,8,12")
    p.add_argument("--max-frames", type=int, default=24)
    p.add_argument("--max-contact-frames", type=int, default=8)
    return p


def _frame_keys(path: Path) -> dict[int, str]:
    import h5py

    with h5py.File(path, "r") as h5:
        return {
            int(key.removeprefix("frame_")): key
            for key in h5.keys()
            if str(key).startswith("frame_")
        }


def _image_paths(image_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted([*image_dir.glob("*.jpg"), *image_dir.glob("*.png")]):
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        if digits:
            out[int(digits)] = path
    return out


def _sample_evenly(values: list[int], n: int) -> list[int]:
    if len(values) <= n:
        return values
    indices = np.linspace(0, len(values) - 1, n).round().astype(int)
    return [values[int(i)] for i in sorted(set(indices.tolist()))]


def _read_frame(path: Path, key: str) -> list[dict]:
    import h5py

    out: list[dict] = []
    with h5py.File(path, "r") as h5:
        group = h5[key]
        for name in sorted(group.keys()):
            item = group[name]
            out.append({
                "name": str(name),
                "instance_id": int(item.attrs.get("instance_id", -1)),
                "track_id": int(item.attrs.get("sam2_track_id", -1)),
                "score": float(item.attrs.get("score", np.nan)),
                "mask": np.asarray(item["mask"][...], dtype=bool),
            })
    return out


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _greedy_matches(left: list[dict], right: list[dict]) -> list[tuple[int, int, float]]:
    candidates = [
        (_iou(a["mask"], b["mask"]), i, j)
        for i, a in enumerate(left)
        for j, b in enumerate(right)
    ]
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, i, j in sorted(candidates, reverse=True):
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append((i, j, score))
    return sorted(matches)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    import cv2

    size = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _pairwise_overlap_pixels(masks: list[np.ndarray]) -> int:
    total = 0
    for i, left in enumerate(masks):
        for right in masks[i + 1 :]:
            total += int(np.logical_and(left, right).sum())
    return total


def _metrics_for_match(
    *,
    frame: int,
    det: dict,
    sam: dict,
    det_index: int,
    sam_index: int,
    all_det: list[dict],
    radius: int,
) -> dict:
    base = det["mask"]
    candidate = _dilate(base, radius)
    reference = sam["mask"]
    added = np.logical_and(candidate, ~base)
    other_original = np.zeros_like(base)
    for index, other in enumerate(all_det):
        if index != det_index:
            other_original |= other["mask"]
    inter = int(np.logical_and(candidate, reference).sum())
    added_inside_reference = int(np.logical_and(added, reference).sum())
    added_on_other = int(np.logical_and(added, other_original).sum())
    return {
        "frame": int(frame),
        "detectron_instance_id": int(det["instance_id"]),
        "detectron_index": int(det_index),
        "detectron_score": float(det["score"]),
        "sam_track_id": int(sam["track_id"]),
        "sam_index": int(sam_index),
        "radius_px": int(radius),
        "base_area_px": int(base.sum()),
        "candidate_area_px": int(candidate.sum()),
        "sam_area_px": int(reference.sum()),
        "area_increase_fraction": _safe_ratio(int(candidate.sum()) - int(base.sum()), int(base.sum())),
        "iou_vs_sam": _iou(candidate, reference),
        "coverage_of_sam": _safe_ratio(inter, int(reference.sum())),
        "precision_vs_sam": _safe_ratio(inter, int(candidate.sum())),
        "added_area_px": int(added.sum()),
        "added_inside_sam_px": added_inside_reference,
        "added_inside_sam_fraction": _safe_ratio(added_inside_reference, int(added.sum())),
        "added_outside_sam_fraction": _safe_ratio(int(added.sum()) - added_inside_reference, int(added.sum())),
        "added_on_other_detectron_px": added_on_other,
        "added_on_other_detectron_fraction": _safe_ratio(added_on_other, int(added.sum())),
    }


def _aggregate(rows: list[dict], radius: int) -> dict:
    subset = [row for row in rows if int(row["radius_px"]) == radius]
    fields = [
        "area_increase_fraction",
        "iou_vs_sam",
        "coverage_of_sam",
        "precision_vs_sam",
        "added_inside_sam_fraction",
        "added_outside_sam_fraction",
        "added_on_other_detectron_fraction",
    ]
    result: dict[str, object] = {"radius_px": radius, "n_matched_instances": len(subset)}
    for field in fields:
        values = np.asarray([float(row[field]) for row in subset], dtype=float)
        result[f"mean_{field}"] = float(np.mean(values)) if len(values) else None
        result[f"median_{field}"] = float(np.median(values)) if len(values) else None
    return result


def _draw_panel(image: np.ndarray, masks: list[np.ndarray], label: str, color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    canvas = image.copy()
    overlay = np.zeros_like(canvas)
    for mask in masks:
        overlay[mask] = color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
    canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.25, 0.0)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 1)
    return cv2.resize(canvas, (280, 280), interpolation=cv2.INTER_AREA)


def _contact_sheet(
    records: list[dict],
    radii: list[int],
    output_path: Path,
    max_frames: int,
) -> None:
    import cv2

    selected = records[:max_frames]
    if not selected:
        return
    rows = []
    for record in selected:
        image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        panels = [_draw_panel(image, [], f"frame {record['frame']}", (0, 0, 0))]
        panels.append(_draw_panel(image, record["det_masks"], "Detectron2 r=0", (0, 220, 255)))
        for radius in radii:
            if radius <= 0:
                continue
            panels.append(
                _draw_panel(image, [_dilate(mask, radius) for mask in record["det_masks"]], f"dilate r={radius}", (255, 180, 0))
            )
        panels.append(_draw_panel(image, record["sam_masks"], "SAM2 pseudo-ref", (60, 230, 80)))
        rows.append(np.hstack(panels))
    if rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), np.vstack(rows))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Fin-mask expansion proxy comparison",
        "",
        f"Experiment: `{summary['experiment'] or 'unspecified'}`  ",
        f"View: `{summary['view'] or 'unspecified'}`  ",
        f"Sampled frames: {summary['n_sampled_frames']}  ",
        f"Matched mask pairs: {summary['n_matched_pairs']}",
        "",
        "## Interpretation boundary",
        "",
        "SAM2 masks are an independently generated pseudo-reference, not hand-labeled fin ground truth. "
        "The comparison can quantify expansion, agreement, and contact contamination risk. It cannot prove "
        "that added pixels are fins. Visual review and fin-inclusive labels remain required.",
        "",
        "## Results",
        "",
        "| radius (px) | area + | IoU vs SAM | SAM coverage | precision vs SAM | added inside SAM | added outside SAM | added on other fish |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_radius"]:
        def pct(name: str) -> str:
            value = row.get(name)
            return "n/a" if value is None else f"{100 * float(value):.1f}%"

        def val(name: str) -> str:
            value = row.get(name)
            return "n/a" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {row['radius_px']} | {pct('mean_area_increase_fraction')} | {val('mean_iou_vs_sam')} | "
            f"{pct('mean_coverage_of_sam')} | {pct('mean_precision_vs_sam')} | "
            f"{pct('mean_added_inside_sam_fraction')} | {pct('mean_added_outside_sam_fraction')} | "
            f"{pct('mean_added_on_other_detectron_fraction')} |"
        )
    lines.extend([
        "",
        "## Contact-risk frames",
        "",
        "The JSON summary contains per-radius counts of sampled frames where independently dilated instance masks overlap. "
        "Those frames should be prioritized for annotation because dilation can import a neighbor's pixels.",
        "",
        "## Outputs",
        "",
        "- `mask_expansion_metrics.csv`: one row per matched mask and radius.",
        "- `mask_expansion_summary.json`: aggregates and contact-risk counts.",
        "- `mask_expansion_contact_sheet.png`: raw, Detectron2, dilations, and SAM2 overlays.",
        "- `annotation_candidates.csv`: provenance-preserving inputs for the SAM-assisted GUI.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    det_path = Path(args.detectron_masks).resolve()
    sam_path = Path(args.sam_masks).resolve()
    image_dir = Path(args.image_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    radii = sorted(set(int(value.strip()) for value in str(args.radii).split(",") if value.strip()))
    if 0 not in radii:
        radii.insert(0, 0)

    det_keys = _frame_keys(det_path)
    sam_keys = _frame_keys(sam_path)
    images = _image_paths(image_dir)
    shared = sorted(set(det_keys) & set(sam_keys) & set(images))
    frames = _sample_evenly(shared, max(1, int(args.max_frames)))
    if not frames:
        raise RuntimeError("No shared frame ids were found across both mask files and the image directory.")

    rows: list[dict] = []
    visual_records: list[dict] = []
    manifest: list[dict] = []
    contact_counts = {str(radius): 0 for radius in radii}
    matched_pairs = 0
    for frame in frames:
        detections = _read_frame(det_path, det_keys[frame])
        sam_masks = _read_frame(sam_path, sam_keys[frame])
        matches = _greedy_matches(detections, sam_masks)
        matched_pairs += len(matches)
        for radius in radii:
            if _pairwise_overlap_pixels([_dilate(item["mask"], radius) for item in detections]) > 0:
                contact_counts[str(radius)] += 1
        for det_index, sam_index, base_iou in matches:
            for radius in radii:
                row = _metrics_for_match(
                    frame=frame,
                    det=detections[det_index],
                    sam=sam_masks[sam_index],
                    det_index=det_index,
                    sam_index=sam_index,
                    all_det=detections,
                    radius=radius,
                )
                row["base_match_iou"] = float(base_iou)
                rows.append(row)
            manifest.append({
                "sample_id": f"{args.view or 'view'}_f{frame}_det{detections[det_index]['instance_id']}",
                "experiment": str(args.experiment),
                "view": str(args.view),
                "frame": int(frame),
                "image_path": str(images[frame]),
                "video_path": str(args.video),
                "masks_h5": str(det_path),
                "mask_instance_id": int(detections[det_index]["instance_id"]),
                "matched_sam_h5": str(sam_path),
                "matched_sam_track_id": int(sam_masks[sam_index]["track_id"]),
                "base_match_iou": float(base_iou),
            })
        visual_records.append({
            "frame": frame,
            "image_path": images[frame],
            "det_masks": [item["mask"] for item in detections],
            "sam_masks": [item["mask"] for item in sam_masks],
        })

    summary = {
        "experiment": str(args.experiment),
        "view": str(args.view),
        "detectron_masks": str(det_path),
        "sam_masks": str(sam_path),
        "image_dir": str(image_dir),
        "n_available_shared_frames": len(shared),
        "n_sampled_frames": len(frames),
        "sampled_frames": frames,
        "n_matched_pairs": matched_pairs,
        "radii_px": radii,
        "by_radius": [_aggregate(rows, radius) for radius in radii],
        "frames_with_any_dilated_instance_overlap": contact_counts,
        "reference_limit": "SAM2 is a pseudo-reference and does not establish fin ground truth.",
    }
    _write_csv(out_dir / "mask_expansion_metrics.csv", rows)
    _write_csv(out_dir / "annotation_candidates.csv", manifest)
    (out_dir / "mask_expansion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(out_dir / "FIN_SEGMENTATION_COMPARISON.md", summary)
    _contact_sheet(
        visual_records,
        radii,
        out_dir / "mask_expansion_contact_sheet.png",
        max_frames=max(1, int(args.max_contact_frames)),
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
