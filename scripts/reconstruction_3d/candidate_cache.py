from __future__ import annotations

"""Compact, resumable SLEAP candidate caches for windowed reconstruction.

The merged SLEAP files are expensive to materialize and historically were
decoded repeatedly by coordinate loading and tracklet-to-slot matching.  This
module performs that object-heavy decode once per view and stores only the
fields needed by the windowed mouth-tracklet pipeline in chunked HDF5 arrays.

Candidate slots use the exact false-positive filtering and confidence ordering
from :mod:`scripts.reconstruction_3d.load_2d_slp`.  A strict mouth contract is
the default: instances without a finite mouth are omitted, while finite tails
and their confidence remain available as optional secondary evidence.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from scripts.common.local_paths import load_local_paths
from scripts.reconstruction_3d.load_2d_slp import _instances_in_slot_order


SCHEMA_VERSION = "medaka_candidate_cache_v1"
VIEW_NAMES = ("top_cam", "side1_cam", "side2_cam")


@dataclass(frozen=True)
class CandidateWindow:
    """One inclusive frame slice from a candidate cache."""

    frame_start: int
    frame_end: int
    mouth_xy: np.ndarray
    tail_xy: np.ndarray
    instance_score: np.ndarray
    mouth_score: np.ndarray
    tail_score: np.ndarray
    visibility: np.ndarray
    source_instance: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.frame_end - self.frame_start + 1)


def _plain_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def _source_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sha256(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _node_scores(instance: object, node_count: int) -> np.ndarray:
    """Return node-level confidence when exposed by the sleap-io version."""
    for attribute in ("scores", "point_scores", "node_scores"):
        values = getattr(instance, attribute, None)
        if values is None:
            continue
        array = np.asarray(values, dtype=float).reshape(-1)
        if len(array) >= node_count:
            return array[:node_count]
    return np.full(node_count, np.nan, dtype=float)


def cache_matches_source(
    cache_path: str | Path,
    source_path: str | Path,
    *,
    view: str | None = None,
    num_candidates: int | None = None,
    frame_count: int | None = None,
    anchor_node: str | None = None,
    require_anchor_node: bool | None = None,
    filter_fp: bool | None = None,
    min_span_px: float | None = None,
    dup_dist_px: float | None = None,
) -> bool:
    """Return true only for a complete cache with matching source and policy.

    Optional build parameters are deliberately checked by callers that intend
    to reuse a cache.  This prevents a six-slot strict-mouth run from silently
    accepting an otherwise valid cache made with different filtering rules.
    """
    import h5py

    cache_path = _plain_path(cache_path)
    source_path = _plain_path(source_path)
    if not cache_path.exists() or not source_path.exists():
        return False
    source = _source_metadata(source_path)
    try:
        with h5py.File(cache_path, "r") as handle:
            matches = bool(
                handle.attrs.get("schema_version", "") == SCHEMA_VERSION
                and int(handle.attrs.get("complete", 0)) == 1
                and int(handle.attrs.get("source_size_bytes", -1))
                == source["size_bytes"]
                and int(handle.attrs.get("source_mtime_ns", -1))
                == source["mtime_ns"]
            )
            expected = {
                "view": view,
                "num_candidates": num_candidates,
                "frame_count": frame_count,
                "anchor_node": anchor_node,
                "require_anchor_node": (
                    None if require_anchor_node is None else int(require_anchor_node)
                ),
                "filter_fp": None if filter_fp is None else int(filter_fp),
                "min_span_px": min_span_px,
                "dup_dist_px": dup_dist_px,
            }
            for name, value in expected.items():
                if value is None:
                    continue
                cached = handle.attrs.get(name)
                if isinstance(value, float):
                    matches &= bool(np.isclose(float(cached), value, atol=0.0, rtol=0.0))
                else:
                    matches &= str(cached) == str(value)
            return matches
    except OSError:
        return False


def build_view_cache(
    source_path: str | Path,
    cache_path: str | Path,
    *,
    view: str,
    num_candidates: int = 6,
    frame_count: int | None = None,
    anchor_node: str = "mouth",
    require_anchor_node: bool = True,
    filter_fp: bool = True,
    min_span_px: float = 15.0,
    dup_dist_px: float = 12.0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Decode one merged SLP once and atomically write its compact cache."""
    import h5py
    import sleap_io as sio

    if view not in VIEW_NAMES:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEW_NAMES}")
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    source_path = _plain_path(source_path)
    cache_path = _plain_path(cache_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not overwrite and cache_matches_source(
        cache_path,
        source_path,
        view=view,
        num_candidates=num_candidates,
        frame_count=frame_count,
        anchor_node=anchor_node,
        require_anchor_node=require_anchor_node,
        filter_fp=filter_fp,
        min_span_px=min_span_px,
        dup_dist_px=dup_dist_px,
    ):
        return read_cache_metadata(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    started = time.perf_counter()
    labels = sio.load_slp(str(source_path))
    labeled_frames = labels.labeled_frames
    inferred_count = (
        max((int(frame.frame_idx) for frame in labeled_frames), default=-1) + 1
    )
    total_frames = int(frame_count if frame_count is not None else inferred_count)
    if total_frames < inferred_count:
        raise ValueError(
            f"frame_count={total_frames} truncates labeled frame {inferred_count - 1}"
        )
    if not labels.skeletons:
        raise ValueError(f"SLP has no skeleton: {source_path}")
    node_names = [node.name for node in labels.skeletons[0].nodes]
    if anchor_node not in node_names:
        raise ValueError(
            f"anchor node {anchor_node!r} not found in {source_path}: {node_names}"
        )
    mouth_index = node_names.index("mouth") if "mouth" in node_names else None
    tail_index = node_names.index("tail") if "tail" in node_names else None

    shape_xy = (total_frames, num_candidates, 2)
    shape_value = (total_frames, num_candidates)
    # PredictedPointsArray uses float64.  Preserve it for the verified baseline;
    # float32 is a separate performance ablation because a rounded residual can
    # alter a near-tied candidate-set decision.
    mouth_xy = np.full(shape_xy, np.nan, dtype=np.float64)
    tail_xy = np.full(shape_xy, np.nan, dtype=np.float64)
    instance_score = np.full(shape_value, np.nan, dtype=np.float64)
    mouth_score = np.full(shape_value, np.nan, dtype=np.float64)
    tail_score = np.full(shape_value, np.nan, dtype=np.float64)
    visibility = np.zeros(shape_value, dtype=np.uint8)
    source_instance = np.full(shape_value, -1, dtype=np.int16)
    frame_present = np.zeros(total_frames, dtype=np.uint8)

    kept = 0
    raw_instances = 0
    mouth_missing = 0
    tail_missing = 0
    for labeled_frame in labeled_frames:
        frame_index = int(labeled_frame.frame_idx)
        if frame_index < 0 or frame_index >= total_frames:
            continue
        frame_present[frame_index] = 1
        instances = list(labeled_frame.instances)
        raw_instances += len(instances)
        source_index_by_object = {id(instance): i for i, instance in enumerate(instances)}
        ordered = _instances_in_slot_order(
            instances,
            node_names=node_names,
            anchor_node=anchor_node,
            require_anchor_node=require_anchor_node,
            filter_fp=filter_fp,
            min_span_px=min_span_px,
            dup_dist_px=dup_dist_px,
        )[:num_candidates]
        for slot, instance in enumerate(ordered):
            points = np.asarray(instance.numpy(), dtype=float)
            scores = _node_scores(instance, len(node_names))
            instance_value = getattr(instance, "score", np.nan)
            try:
                instance_score[frame_index, slot] = float(instance_value)
            except (TypeError, ValueError):
                pass
            source_instance[frame_index, slot] = int(
                source_index_by_object.get(id(instance), -1)
            )
            if mouth_index is not None and mouth_index < len(points):
                point = points[mouth_index]
                if np.isfinite(point).all():
                    mouth_xy[frame_index, slot] = point
                    visibility[frame_index, slot] |= np.uint8(1)
                    if mouth_index < len(scores):
                        mouth_score[frame_index, slot] = scores[mouth_index]
                else:
                    mouth_missing += 1
            if tail_index is not None and tail_index < len(points):
                point = points[tail_index]
                if np.isfinite(point).all():
                    tail_xy[frame_index, slot] = point
                    visibility[frame_index, slot] |= np.uint8(2)
                    if tail_index < len(scores):
                        tail_score[frame_index, slot] = scores[tail_index]
                else:
                    tail_missing += 1
            kept += 1

    source = _source_metadata(source_path)
    chunks_xy = (min(4096, total_frames), num_candidates, 2)
    chunks_value = (min(4096, total_frames), num_candidates)
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "complete": 0,
                "view": view,
                "source_path": source["path"],
                "source_size_bytes": source["size_bytes"],
                "source_mtime_ns": source["mtime_ns"],
                "frame_count": total_frames,
                "num_candidates": num_candidates,
                "anchor_node": anchor_node,
                "require_anchor_node": int(require_anchor_node),
                "filter_fp": int(filter_fp),
                "min_span_px": float(min_span_px),
                "dup_dist_px": float(dup_dist_px),
                "node_names_json": json.dumps(node_names),
                "coordinate_dtype": "float64",
                "raw_instance_count": raw_instances,
                "kept_instance_count": kept,
                "mouth_missing_after_filter": mouth_missing,
                "tail_missing_after_filter": tail_missing,
            }
        )
        compression = "lzf"
        handle.create_dataset(
            "mouth_xy", data=mouth_xy, chunks=chunks_xy, compression=compression
        )
        handle.create_dataset(
            "tail_xy", data=tail_xy, chunks=chunks_xy, compression=compression
        )
        for name, data in (
            ("instance_score", instance_score),
            ("mouth_score", mouth_score),
            ("tail_score", tail_score),
            ("visibility", visibility),
            ("source_instance", source_instance),
        ):
            handle.create_dataset(
                name, data=data, chunks=chunks_value, compression=compression
            )
        handle.create_dataset(
            "frame_present",
            data=frame_present,
            chunks=(min(16384, total_frames),),
            compression=compression,
        )
        handle.attrs["decode_write_seconds"] = float(time.perf_counter() - started)
        handle.attrs["complete"] = 1
        handle.flush()
    os.replace(temporary, cache_path)
    metadata = read_cache_metadata(cache_path)
    return metadata


def read_cache_metadata(
    cache_path: str | Path, *, include_content_hash: bool = True
) -> dict[str, object]:
    import h5py

    cache_path = _plain_path(cache_path)
    with h5py.File(cache_path, "r") as handle:
        metadata = {
            str(key): (
                value.item() if isinstance(value, np.generic) else value
            )
            for key, value in handle.attrs.items()
        }
    metadata["cache_path"] = str(cache_path.resolve())
    metadata["cache_size_bytes"] = int(cache_path.stat().st_size)
    if include_content_hash:
        metadata["cache_sha256"] = _sha256(cache_path)
    return metadata


def load_candidate_window(
    cache_path: str | Path, frame_start: int, frame_end: int
) -> CandidateWindow:
    """Load one inclusive bounded slice without materializing the full cache."""
    import h5py

    if frame_end < frame_start:
        raise ValueError("frame_end must be greater than or equal to frame_start")
    cache_path = _plain_path(cache_path)
    with h5py.File(cache_path, "r") as handle:
        total = int(handle.attrs["frame_count"])
        if frame_start < 0 or frame_end >= total:
            raise IndexError(
                f"requested frames {frame_start}..{frame_end}, cache has 0..{total - 1}"
            )
        selection = slice(frame_start, frame_end + 1)
        arrays = {
            name: np.asarray(handle[name][selection])
            for name in (
                "mouth_xy",
                "tail_xy",
                "instance_score",
                "mouth_score",
                "tail_score",
                "visibility",
                "source_instance",
            )
        }
    return CandidateWindow(frame_start=frame_start, frame_end=frame_end, **arrays)


def write_manifest(
    cache_paths: Mapping[str, str | Path], output_path: str | Path
) -> dict[str, object]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "views": {
            view: read_cache_metadata(path)
            for view, path in sorted(cache_paths.items())
        },
    }
    output_path = _plain_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _resolve_sources(
    experiment: str,
    round_tag: str,
    paths_config: str | Path,
    source_root: str | Path | None,
) -> dict[str, Path]:
    if source_root is None:
        paths = load_local_paths(paths_config)
        if paths is None or paths.sleap_output_dir is None:
            raise ValueError(
                "sleap_output_dir is missing; pass --source-root or configure "
                f"it in {paths_config}"
            )
        source_root = paths.sleap_output_dir
    root = _plain_path(source_root) / experiment / round_tag
    return {view: root / f"{view}.slp" for view in VIEW_NAMES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--round-tag", required=True)
    parser.add_argument("--paths-config", default="configs/paths.local.toml")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    sources = _resolve_sources(
        args.experiment, args.round_tag, args.paths_config, args.source_root
    )
    cache_paths: dict[str, Path] = {}
    for view in VIEW_NAMES:
        cache_path = args.output_dir / f"{view}.candidates.h5"
        metadata = build_view_cache(
            sources[view],
            cache_path,
            view=view,
            num_candidates=args.num_candidates,
            frame_count=args.frame_count,
            overwrite=args.overwrite,
        )
        cache_paths[view] = cache_path
        print(json.dumps(metadata, sort_keys=True))
    write_manifest(cache_paths, args.output_dir / "candidate_cache_manifest.json")


if __name__ == "__main__":
    main()
