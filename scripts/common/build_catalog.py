from __future__ import annotations

"""
Aggregate every per-experiment `experiment_metadata.json` under
`medaka_data_dir` into two files at the same level:

    catalog.jsonl   lossless aggregation (one manifest per line)
    catalog.csv     flattened tabular view with convenience columns

Usage:
    python -m scripts.common.build_catalog \\
        --paths-config configs/paths.local.toml

The per-folder manifests remain the source of truth; this tool just caches
their aggregate into two formats useful for filtering and pivoting.

Convenience columns in the CSV (derived from the `sleap` section if present):
    merged_all          — all three views have a merged .slp file
    merged_top,         — per-view merge booleans
    merged_side1,
    merged_side2
    n_chunks_top,       — number of chunk .slp files per view
    n_chunks_side1,
    n_chunks_side2
    frames_<view>_recorded — from each camera's metadata.csv
    frames_<view>_merged   — from the merged .slp (after slp_merge)
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.experiment_metadata import MANIFEST_FILENAME, VIEWS
from scripts.common.local_paths import choose_path, load_local_paths


DEFAULT_CATALOG_BASENAME = "catalog"


def collect_manifests(data_root: Path) -> list[dict]:
    """Return every manifest under `data_root`, sorted by folder name."""
    manifests: list[dict] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifests.append(json.load(fh))
        except json.JSONDecodeError as exc:
            print(f"WARN: could not parse {manifest_path}: {exc}", file=sys.stderr)
    return manifests


def derive_convenience_fields(manifest: dict) -> dict:
    """
    Flat helper columns for the CSV view — quick merge-status + frame counts
    per view without digging through nested JSON.
    """
    sleap_section = manifest.get("sleap") or {}
    sleap_views = sleap_section.get("views", {}) or {}
    raw_views = manifest.get("views", {}) or {}

    out: dict[str, object] = {"has_manifest": True}
    per_view_merged: dict[str, bool] = {}
    for view in VIEWS:
        sleap_info = sleap_views.get(view) or {}
        per_view_merged[view] = bool(sleap_info.get("merged"))
        out[f"n_chunks_{view.replace('_cam', '')}"] = len(sleap_info.get("chunks") or [])
        out[f"frames_{view}_merged"] = sleap_info.get("total_frames")
        out[f"frames_{view}_recorded"] = (raw_views.get(view) or {}).get("total_frames")

    out["merged_top"] = per_view_merged.get("top_cam", False)
    out["merged_side1"] = per_view_merged.get("side1_cam", False)
    out["merged_side2"] = per_view_merged.get("side2_cam", False)
    out["merged_all"] = all(per_view_merged.get(v, False) for v in VIEWS)
    return out


def write_jsonl(manifests: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for m in manifests:
            fh.write(json.dumps(m, ensure_ascii=False, default=str))
            fh.write("\n")


def write_csv(manifests: list[dict], path: Path) -> None:
    import pandas as pd

    rows: list[dict] = []
    for m in manifests:
        flat = pd.json_normalize(m, sep=".").to_dict(orient="records")[0]
        flat.update(derive_convenience_fields(m))
        rows.append(flat)

    df = pd.DataFrame(rows)

    priority = [
        "folder_name",
        "experiment.start_date",
        "experiment.group",
        "experiment.trial",
        "experiment.day",
        "experiment.paradigm",
        "merged_all",
        "merged_top",
        "merged_side1",
        "merged_side2",
        "n_chunks_top",
        "n_chunks_side1",
        "n_chunks_side2",
        "frames_top_cam_recorded",
        "frames_top_cam_merged",
        "frames_side1_cam_recorded",
        "frames_side1_cam_merged",
        "frames_side2_cam_recorded",
        "frames_side2_cam_merged",
    ]
    ordered = [c for c in priority if c in df.columns] + [
        c for c in df.columns if c not in priority
    ]
    df = df.reindex(columns=ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_catalog(data_root: Path, output_dir: Path, basename: str) -> dict:
    manifests = collect_manifests(data_root)
    jsonl_path = output_dir / f"{basename}.jsonl"
    csv_path = output_dir / f"{basename}.csv"
    write_jsonl(manifests, jsonl_path)
    write_csv(manifests, csv_path)

    merged_count = sum(1 for m in manifests if (m.get("sleap") or {}).get("views"))
    return {
        "n_experiments": len(manifests),
        "n_with_sleap": merged_count,
        "jsonl": jsonl_path,
        "csv": csv_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-experiment manifests into catalog.jsonl + catalog.csv."
        )
    )
    parser.add_argument("--medaka-data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write catalog files (default: medaka_data_dir).",
    )
    parser.add_argument(
        "--basename",
        default=DEFAULT_CATALOG_BASENAME,
        help=f"Output file basename (default: {DEFAULT_CATALOG_BASENAME}).",
    )
    parser.add_argument("--paths-config", default=None)
    args = parser.parse_args(argv)

    local = load_local_paths(args.paths_config)
    root_str = choose_path(
        args.medaka_data_dir, local.medaka_data_dir if local else None
    )
    if root_str is None:
        parser.error(
            "medaka_data_dir is required. Pass --medaka-data-dir or set it in "
            "configs/paths.local.toml."
        )
    root = Path(root_str)
    out_dir = Path(args.output_dir) if args.output_dir else root

    summary = build_catalog(root, out_dir, args.basename)
    print(f"Wrote {summary['n_experiments']} experiments "
          f"({summary['n_with_sleap']} with SLEAP merge info):")
    print(f"  {summary['jsonl']}")
    print(f"  {summary['csv']}")


if __name__ == "__main__":
    main()
