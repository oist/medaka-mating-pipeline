from __future__ import annotations

import json
import shutil
from pathlib import Path


def merge_coco_exports(*, coco_dirs: list[str | Path], output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    merged = {
        "info": {"description": "Merged medaka SAM2 QC pseudo-label COCO dataset", "version": "0.1"},
        "licenses": [],
        "categories": [{"id": 1, "name": "medaka", "supercategory": "fish"}],
        "images": [],
        "annotations": [],
    }
    next_image_id = 1
    next_ann_id = 1
    for dataset_idx, coco_dir in enumerate(coco_dirs):
        coco_dir = Path(coco_dir)
        data = json.loads((coco_dir / "annotations.json").read_text(encoding="utf-8"))
        id_map: dict[int, int] = {}
        prefix = f"d{dataset_idx:02d}"
        for image in data.get("images", []):
            old_id = int(image["id"])
            new_id = next_image_id
            next_image_id += 1
            id_map[old_id] = new_id
            src_name = str(image["file_name"])
            dst_name = f"{prefix}_{Path(src_name).name}"
            shutil.copy2(coco_dir / "images" / src_name, image_dir / dst_name)
            out_image = dict(image)
            out_image["id"] = new_id
            out_image["file_name"] = dst_name
            merged["images"].append(out_image)
        for ann in data.get("annotations", []):
            old_image_id = int(ann["image_id"])
            if old_image_id not in id_map:
                continue
            out_ann = dict(ann)
            out_ann["id"] = next_ann_id
            next_ann_id += 1
            out_ann["image_id"] = id_map[old_image_id]
            merged["annotations"].append(out_ann)

    annotation_path = output_dir / "annotations.json"
    annotation_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    summary = {
        "annotation_path": str(annotation_path),
        "image_dir": str(image_dir),
        "n_images": len(merged["images"]),
        "n_annotations": len(merged["annotations"]),
        "source_datasets": [str(Path(path)) for path in coco_dirs],
    }
    (output_dir / "coco_merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
