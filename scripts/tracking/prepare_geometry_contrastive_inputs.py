from __future__ import annotations

"""Assemble disjoint geometry-contrastive blocks without rewriting source masks.

The JSON manifest names already reviewed identified-2D, disposition, pre-link,
and per-view silhouette-association artifacts. Rows are filtered to each locked
frame range, component IDs are namespaced by block, and disjoint pre-link frame
records are merged. Original files remain read-only.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def _filter_frames(table: pd.DataFrame, start: int, end: int, label: str) -> pd.DataFrame:
    if "frame" not in table.columns:
        raise ValueError(f"{label} lacks frame")
    frames = pd.to_numeric(table["frame"], errors="raise").astype(int)
    return table.loc[frames.between(int(start), int(end))].copy()


def _write_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False)


def assemble(manifest_path: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "geometry_contrastive_input_manifest_v1":
        raise ValueError("unsupported input manifest schema_version")
    blocks = list(payload.get("blocks", []))
    if not blocks:
        raise ValueError("input manifest contains no blocks")
    ordered = sorted(blocks, key=lambda row: int(row["start_frame"]))
    for left, right in zip(ordered, ordered[1:]):
        if int(left["end_frame"]) >= int(right["start_frame"]):
            raise ValueError("input manifest blocks overlap")

    identified_parts = []
    disposition_parts = []
    association_parts: dict[str, list[pd.DataFrame]] = {}
    merged_frames: dict[str, object] = {}
    best_template: dict | None = None
    block_summary = []
    for block in ordered:
        name = str(block["name"])
        start = int(block["start_frame"])
        end = int(block["end_frame"])
        identified_path = Path(block["identified_2d"])
        disposition_path = Path(block["dispositions"])
        best_path = Path(block["best_pre_link_json"])
        for path in (identified_path, disposition_path, best_path):
            if not path.exists():
                raise FileNotFoundError(path)

        identified = _filter_frames(_read_table(identified_path), start, end, f"{name} identified")
        if "component_id" not in identified.columns:
            raise ValueError(f"{name} identified table lacks component_id")
        if identified["component_id"].isna().any():
            raise ValueError(f"{name} identified table has null component_id")
        identified["component_id"] = (
            name + "::" + identified["component_id"].astype(str).str.strip()
        )
        if identified["component_id"].str.endswith("::").any():
            raise ValueError(f"{name} identified table has blank component_id")
        identified["source_window_provenance"] = name
        identified_parts.append(identified)

        dispositions = _filter_frames(
            _read_table(disposition_path), start, end, f"{name} dispositions"
        )
        dispositions["source_window_provenance"] = name
        disposition_parts.append(dispositions)

        best = json.loads(best_path.read_text(encoding="utf-8"))
        best_template = dict(best) if best_template is None else best_template
        frames = best.get("frames", best)
        selected_frames = {
            str(frame): value
            for frame, value in frames.items()
            if start <= int(frame) <= end
        }
        duplicate_frames = set(selected_frames) & set(merged_frames)
        if duplicate_frames:
            raise ValueError(f"pre-link frame overlap in block {name}")
        merged_frames.update(selected_frames)

        association_counts = {}
        for view, raw_path in block.get("associations", {}).items():
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(path)
            table = _filter_frames(_read_table(path), start, end, f"{name} {view} associations")
            table["source_window_provenance"] = name
            association_parts.setdefault(str(view), []).append(table)
            association_counts[str(view)] = int(len(table))
        block_summary.append(
            {
                "name": name,
                "start_frame": start,
                "end_frame": end,
                "identified_rows": int(len(identified)),
                "disposition_rows": int(len(dispositions)),
                "best_pre_link_frames": int(len(selected_frames)),
                "association_rows": association_counts,
            }
        )

    expected_views = set(association_parts)
    if not expected_views:
        raise ValueError("input manifest contains no association views")
    if any(set(block.get("associations", {})) != expected_views for block in ordered):
        raise ValueError("every block must provide the same association views")

    output_dir.mkdir(parents=True, exist_ok=True)
    identified = pd.concat(identified_parts, ignore_index=True, sort=False)
    dispositions = pd.concat(disposition_parts, ignore_index=True, sort=False)
    _write_table(identified, output_dir / "identified_2d.csv")
    _write_table(dispositions, output_dir / "detection_dispositions.csv")
    for view, parts in sorted(association_parts.items()):
        _write_table(
            pd.concat(parts, ignore_index=True, sort=False),
            output_dir / f"{view}.silhouette_fused_tracklets.csv",
        )
    merged_best = dict(best_template or {})
    merged_best["frames"] = dict(sorted(merged_frames.items(), key=lambda item: int(item[0])))
    merged_best["assembled_from_disjoint_blocks"] = [row["name"] for row in block_summary]
    (output_dir / "best_pre_link.json").write_text(
        json.dumps(merged_best, separators=(",", ":")), encoding="utf-8"
    )
    summary = {
        "schema_version": "geometry_contrastive_assembled_inputs_v1",
        "experiment": payload.get("experiment"),
        "manifest": str(manifest_path),
        "blocks": block_summary,
        "identified_rows": int(len(identified)),
        "disposition_rows": int(len(dispositions)),
        "best_pre_link_frames": int(len(merged_frames)),
        "association_views": sorted(expected_views),
        "component_ids_namespaced": True,
        "source_masks_rewritten": False,
    }
    (output_dir / "assembly_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(assemble(args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
