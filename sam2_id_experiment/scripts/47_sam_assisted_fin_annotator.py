from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


WINDOW_NAME = "Medaka SAM2 body/fin annotator"
REGIONS = ("body", "fins")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Review Detectron2/SLEAP-seeded fish instances and create SAM2-assisted body/fin masks."
    )
    p.add_argument(
        "--manifest",
        required=True,
        action="append",
        help="CSV made by 46_compare_mask_expansion.py or equivalent; repeat to review multiple views.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sam-model-cfg", default="sam2_hiera_l.yaml")
    p.add_argument("--checkpoint", default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--display-scale", type=float, default=0.75)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "view", "frame", "image_path"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    for row in rows:
        row["frame"] = int(row["frame"])
        if row.get("mask_instance_id", "") != "":
            row["mask_instance_id"] = int(row["mask_instance_id"])
    return rows


def _read_frame_masks(path: str | Path, frame: int) -> list[dict]:
    import h5py

    if not path or not Path(path).exists():
        return []
    key = f"frame_{int(frame)}"
    out: list[dict] = []
    with h5py.File(path, "r") as h5:
        if key not in h5:
            return []
        for name in sorted(h5[key].keys()):
            item = h5[key][name]
            out.append({
                "instance_id": int(item.attrs.get("instance_id", -1)),
                "track_id": int(item.attrs.get("sam2_track_id", -1)),
                "score": float(item.attrs.get("score", np.nan)),
                "mask": np.asarray(item["mask"][...], dtype=bool),
            })
    return out


def _bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


@dataclass
class RegionState:
    points: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    choice: int = 0

    @property
    def mask(self) -> np.ndarray | None:
        if not self.masks:
            return None
        return self.masks[max(0, min(self.choice, len(self.masks) - 1))]

    @property
    def score(self) -> float | None:
        if not self.scores:
            return None
        return float(self.scores[max(0, min(self.choice, len(self.scores) - 1))])


class SamPredictor:
    def __init__(self, model_cfg: str, checkpoint: str, device: str) -> None:
        self.model_cfg = str(model_cfg)
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = str(device)
        self._predictor = None
        self._image_key: str | None = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(self.model_cfg, self.checkpoint, device=self.device, mode="eval")
        self._predictor = SAM2ImagePredictor(model)
        return self._predictor

    def set_image(self, image_rgb: np.ndarray, image_key: str) -> None:
        predictor = self._load()
        if image_key != self._image_key:
            predictor.set_image(image_rgb)
            self._image_key = image_key

    def predict(
        self,
        *,
        image_rgb: np.ndarray,
        image_key: str,
        points: list[list[float]],
        labels: list[int],
        box: np.ndarray | None,
    ) -> tuple[list[np.ndarray], list[float]]:
        self.set_image(image_rgb, image_key)
        coords = np.asarray(points, dtype=np.float32) if points else None
        point_labels = np.asarray(labels, dtype=np.int32) if labels else None
        masks, scores, _logits = self._predictor.predict(
            point_coords=coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        order = np.argsort(np.asarray(scores))[::-1]
        return [np.asarray(masks[i], dtype=bool) for i in order], [float(scores[i]) for i in order]


class Annotator:
    def __init__(self, args: argparse.Namespace, rows: list[dict]) -> None:
        import cv2

        self.cv2 = cv2
        self.args = args
        self.rows = rows
        self.index = max(0, min(int(args.start_index), len(rows) - 1))
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = SamPredictor(args.sam_model_cfg, args.checkpoint, args.device)
        self.image: np.ndarray | None = None
        self.frame_masks: list[dict] = []
        self.selected_mask_index = 0
        self.region = "body"
        self.states = {name: RegionState() for name in REGIONS}
        self.instance_select_mode = False
        self.status = ""
        self.load_current()

    @property
    def row(self) -> dict:
        return self.rows[self.index]

    @property
    def state(self) -> RegionState:
        return self.states[self.region]

    @property
    def selected_mask(self) -> np.ndarray | None:
        if not self.frame_masks:
            return None
        return self.frame_masks[self.selected_mask_index]["mask"]

    def load_current(self) -> None:
        image_path = Path(self.row["image_path"])
        self.image = self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Could not read candidate image: {image_path}")
        self.frame_masks = _read_frame_masks(self.row.get("masks_h5", ""), int(self.row["frame"]))
        desired = self.row.get("mask_instance_id", "")
        self.selected_mask_index = 0
        for index, item in enumerate(self.frame_masks):
            if desired != "" and int(item["instance_id"]) == int(desired):
                self.selected_mask_index = index
                break
        self.states = {name: RegionState() for name in REGIONS}
        if self.row.get("sleap_x", "") != "" and self.row.get("sleap_y", "") != "":
            self.states["body"].points.append([float(self.row["sleap_x"]), float(self.row["sleap_y"])])
            self.states["body"].labels.append(1)
        self.region = "body"
        self.instance_select_mode = False
        self.status = "Loaded candidate. Add points, then P to preview."

    def preview(self) -> None:
        box = _bbox(self.selected_mask) if self.selected_mask is not None else None
        if not self.state.points and box is None:
            self.status = "Add a positive point or select a Detectron2 instance first."
            return
        masks, scores = self.predictor.predict(
            image_rgb=self.cv2.cvtColor(self.image, self.cv2.COLOR_BGR2RGB),
            image_key=str(Path(self.row["image_path"]).resolve()),
            points=self.state.points,
            labels=self.state.labels,
            box=box,
        )
        self.state.masks = masks
        self.state.scores = scores
        self.state.choice = 0
        self.status = f"{self.region}: SAM2 preview ready; score={self.state.score:.3f}. [/] changes candidate."

    def add_point(self, x: int, y: int, label: int) -> None:
        self.state.points.append([float(x), float(y)])
        self.state.labels.append(int(label))
        self.preview()

    def select_instance_at(self, x: int, y: int) -> None:
        hits = [i for i, item in enumerate(self.frame_masks) if bool(item["mask"][y, x])]
        if hits:
            self.selected_mask_index = hits[0]
        elif self.frame_masks:
            distances = []
            for item in self.frame_masks:
                ys, xs = np.nonzero(item["mask"])
                distances.append(float(np.hypot(xs.mean() - x, ys.mean() - y)) if len(xs) else np.inf)
            self.selected_mask_index = int(np.argmin(distances))
        self.instance_select_mode = False
        self.status = f"Selected Detectron2 instance {self.selected_instance_id()}."

    def selected_instance_id(self) -> int | None:
        if not self.frame_masks:
            return None
        return int(self.frame_masks[self.selected_mask_index]["instance_id"])

    def draw(self) -> np.ndarray:
        canvas = self.image.copy()
        if self.selected_mask is not None:
            overlay = np.zeros_like(canvas)
            overlay[self.selected_mask] = (0, 210, 255)
            canvas = self.cv2.addWeighted(canvas, 1.0, overlay, 0.22, 0.0)
        colors = {"body": (255, 110, 20), "fins": (220, 50, 220)}
        for name, state in self.states.items():
            if state.mask is not None:
                overlay = np.zeros_like(canvas)
                overlay[state.mask] = colors[name]
                canvas = self.cv2.addWeighted(canvas, 1.0, overlay, 0.32, 0.0)
            for (x, y), label in zip(state.points, state.labels):
                color = (40, 230, 40) if label == 1 else (30, 30, 240)
                self.cv2.circle(canvas, (round(x), round(y)), 7, color, -1)
                self.cv2.circle(canvas, (round(x), round(y)), 9, (255, 255, 255), 1)
        header = (
            f"{self.index + 1}/{len(self.rows)} {self.row['view']} frame={self.row['frame']} "
            f"sample={self.row['sample_id']} D2={self.selected_instance_id()} region={self.region}"
        )
        hint = "L/R click +/- | B body F fins | P preview | U undo | I select D2 | 1 clean 2 ambiguous 3 reject | N/PgDn next"
        for y, text in [(28, header), (56, hint), (84, self.status)]:
            self.cv2.putText(canvas, text, (12, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3)
            self.cv2.putText(canvas, text, (12, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 10, 10), 1)
        return self.cv2.resize(
            canvas,
            None,
            fx=float(self.args.display_scale),
            fy=float(self.args.display_scale),
            interpolation=self.cv2.INTER_AREA,
        )

    def save(self, decision: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        stem = f"{_safe_token(str(self.row['sample_id']))}__{timestamp}"
        artifact_dir = self.output_dir / "artifacts"
        record_dir = self.output_dir / "records"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        record_dir.mkdir(parents=True, exist_ok=True)
        mask_paths: dict[str, str] = {}
        areas: dict[str, int] = {}
        for name, state in self.states.items():
            if state.mask is None:
                mask_paths[name] = ""
                areas[name] = 0
                continue
            path = artifact_dir / f"{stem}__{name}.png"
            self.cv2.imwrite(str(path), state.mask.astype(np.uint8) * 255)
            mask_paths[name] = str(path)
            areas[name] = int(state.mask.sum())
        combined = np.zeros(self.image.shape[:2], dtype=bool)
        for state in self.states.values():
            if state.mask is not None:
                combined |= state.mask
        combined_path = ""
        if combined.any():
            path = artifact_dir / f"{stem}__combined.png"
            self.cv2.imwrite(str(path), combined.astype(np.uint8) * 255)
            combined_path = str(path)

        image_path = Path(self.row["image_path"]).resolve()
        selected = self.frame_masks[self.selected_mask_index] if self.frame_masks else None
        record = {
            "schema_version": "medaka_sam2_body_fin_annotation_v1",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "sample": dict(self.row),
            "image_sha256": _sha256(image_path),
            "selected_detectron_instance_id": int(selected["instance_id"]) if selected else None,
            "selected_detectron_track_id": int(selected["track_id"]) if selected else None,
            "selected_detectron_score": float(selected["score"]) if selected else None,
            "sam2": {
                "model_cfg": self.args.sam_model_cfg,
                "checkpoint": str(Path(self.args.checkpoint).resolve()),
                "device": self.args.device,
                "box_from_detectron_mask": _bbox(self.selected_mask).tolist() if self.selected_mask is not None else None,
            },
            "regions": {
                name: {
                    "points_xy": state.points,
                    "point_labels": state.labels,
                    "sam_candidate_index": state.choice,
                    "sam_score": state.score,
                    "mask_path": mask_paths[name],
                    "area_px": areas[name],
                }
                for name, state in self.states.items()
            },
            "combined_mask_path": combined_path,
            "combined_area_px": int(combined.sum()),
        }
        record_path = record_dir / f"{stem}.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        self._append_index(record_path, record)
        self.status = f"Saved {decision}: {record_path.name}"
        return record_path

    def _append_index(self, record_path: Path, record: dict) -> None:
        path = self.output_dir / "annotation_records.csv"
        row = {
            "saved_at_utc": record["saved_at_utc"],
            "sample_id": self.row["sample_id"],
            "experiment": self.row.get("experiment", ""),
            "view": self.row["view"],
            "frame": self.row["frame"],
            "decision": record["decision"],
            "selected_detectron_instance_id": record["selected_detectron_instance_id"],
            "body_area_px": record["regions"]["body"]["area_px"],
            "fins_area_px": record["regions"]["fins"]["area_px"],
            "combined_area_px": record["combined_area_px"],
            "record_json": str(record_path),
        }
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def move(self, step: int) -> None:
        self.index = (self.index + step) % len(self.rows)
        self.load_current()

    def mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        x = int(round(x / float(self.args.display_scale)))
        y = int(round(y / float(self.args.display_scale)))
        if not (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]):
            return
        if self.instance_select_mode and event == self.cv2.EVENT_LBUTTONDOWN:
            self.select_instance_at(x, y)
        elif event == self.cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y, 1)
        elif event == self.cv2.EVENT_RBUTTONDOWN:
            self.add_point(x, y, 0)

    def run(self) -> None:
        self.cv2.namedWindow(WINDOW_NAME, self.cv2.WINDOW_AUTOSIZE)
        self.cv2.setMouseCallback(WINDOW_NAME, self.mouse)
        try:
            while True:
                self.cv2.imshow(WINDOW_NAME, self.draw())
                key = self.cv2.waitKeyEx(40)
                if key < 0:
                    continue
                low = key & 0xFF
                if low in {ord("q"), 27}:
                    break
                if low == ord("b"):
                    self.region = "body"
                elif low == ord("f"):
                    self.region = "fins"
                elif low == ord("p"):
                    self.preview()
                elif low == ord("u"):
                    if self.state.points:
                        self.state.points.pop()
                        self.state.labels.pop()
                        self.preview()
                elif low == ord("r"):
                    self.states[self.region] = RegionState()
                    self.status = f"Cleared {self.region}."
                elif low == ord("i"):
                    self.instance_select_mode = True
                    self.status = "Instance selection: left-click inside/near the desired Detectron2 mask."
                elif low == ord("[") and self.state.masks:
                    self.state.choice = (self.state.choice - 1) % len(self.state.masks)
                elif low == ord("]") and self.state.masks:
                    self.state.choice = (self.state.choice + 1) % len(self.state.masks)
                elif low == ord("1"):
                    self.save("clean")
                    self.move(1)
                elif low == ord("2"):
                    self.save("ambiguous")
                    self.move(1)
                elif low == ord("3"):
                    self.save("reject")
                    self.move(1)
                elif low == ord("n") or key in {2555904, 34}:
                    self.move(1)
                elif key in {2424832, 33}:
                    self.move(-1)
                elif low == ord("x") and self.frame_masks:
                    self.selected_mask_index = (self.selected_mask_index + 1) % len(self.frame_masks)
                    self.status = f"Selected Detectron2 instance {self.selected_instance_id()}."
        finally:
            self.cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = [row for manifest in args.manifest for row in read_manifest(manifest)]
    if not rows:
        raise RuntimeError("The annotation manifest has no rows.")
    missing = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {missing[0]}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {args.checkpoint}")
    summary = {
        "manifests": [str(Path(path).resolve()) for path in args.manifest],
        "output_dir": str(Path(args.output_dir).resolve()),
        "n_candidates": len(rows),
        "views": sorted({str(row["view"]) for row in rows}),
        "sam_model_cfg": args.sam_model_cfg,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    Annotator(args, rows).run()


if __name__ == "__main__":
    main()
