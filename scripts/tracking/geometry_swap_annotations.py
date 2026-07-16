"""Replay geometry-ID review events and export conservative clean tracklets.

The geometry reconstruction is treated as immutable. Review annotations are a
small event ledger with independent mappings for each camera and the 3D track
panel. Only the ``3d`` (plus legacy-global) mapping changes geometry-tracklet
exports. Ambiguous, interpolated, and missing observations are never included
in the clean training tracklets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


EVENT_TYPES = (
    "pair_swap",
    "false_positive",
    "ambiguous",
    "track_break",
    "mapping_anchor",
    "reviewed_no_swap",
    "bad_mouth",
    "bad_tail",
    "wrong_link",
    "review_progress",
)
EVENT_SCOPES = ("all", "top_cam", "side1_cam", "side2_cam", "3d")
CAMERA_SCOPES = ("top_cam", "side1_cam", "side2_cam")
NODE_QC_EVENT_TYPES = ("bad_mouth", "bad_tail", "wrong_link")
EVENT_COLUMNS = (
    "event_id",
    "event_type",
    "view_scope",
    "frame_start",
    "frame_end",
    "slot_a",
    "slot_b",
    "mapping_after",
    "confidence",
    "notes",
    "candidate_source",
    "candidate_id",
    "created_at",
)


@dataclass(frozen=True)
class TrackletExportConfig:
    id_column: str = "fish_id"
    max_gap_frames: int = 0


def infer_slot_map(values: Iterable[object]) -> dict[str, str]:
    """Map sorted source IDs to display slots G1..Gn."""
    source = sorted({str(value) for value in values if pd.notna(value)})
    if not source:
        raise ValueError("no finite geometry IDs were found")
    return {value: f"G{index + 1}" for index, value in enumerate(source)}


def default_identity_mapping(slots: Iterable[str]) -> dict[str, str]:
    return {
        slot: f"H{index + 1}"
        for index, slot in enumerate(sorted({str(slot) for slot in slots}))
    }


def format_mapping(mapping: Mapping[str, str]) -> str:
    return ";".join(f"{slot}={mapping[slot]}" for slot in sorted(mapping))


def parse_mapping(value: object, slots: Iterable[str]) -> dict[str, str]:
    expected = {str(slot) for slot in slots}
    parts: dict[str, str] = {}
    for item in str(value or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid mapping item {item!r}; expected SLOT=IDENTITY")
        slot, identity = (part.strip() for part in item.split("=", 1))
        if not slot or not identity:
            raise ValueError(f"invalid mapping item {item!r}")
        parts[slot] = identity
    if set(parts) != expected:
        raise ValueError(
            "mapping anchor must define exactly "
            + ", ".join(sorted(expected))
        )
    if len(set(parts.values())) != len(parts):
        raise ValueError("mapping anchor identities must be one-to-one")
    return parts


def _integer_series(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain finite integer frames")
    return numeric.astype(np.int64)


def _optional_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalize_events(events: pd.DataFrame | None, slots: Iterable[str]) -> pd.DataFrame:
    """Return a validated event table in deterministic replay order."""
    slot_set = {str(slot) for slot in slots}
    if events is None or events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    table = events.copy()
    for column in EVENT_COLUMNS:
        if column not in table:
            table[column] = ""
    table = table[list(EVENT_COLUMNS)]
    table["event_type"] = table["event_type"].astype(str).str.strip()
    bad_types = sorted(set(table["event_type"]) - set(EVENT_TYPES))
    if bad_types:
        raise ValueError(f"unknown event types: {bad_types}")
    table["view_scope"] = table["view_scope"].fillna("").astype(str).str.strip()
    # Ledgers written before per-view review had global semantics.
    table.loc[table["view_scope"].eq(""), "view_scope"] = "all"
    bad_scopes = sorted(set(table["view_scope"]) - set(EVENT_SCOPES))
    if bad_scopes:
        raise ValueError(f"unknown view scopes: {bad_scopes}")
    table["frame_start"] = _integer_series(table["frame_start"], "frame_start")
    missing_end = table["frame_end"].isna() | table["frame_end"].astype(str).str.strip().eq("")
    table.loc[missing_end, "frame_end"] = table.loc[missing_end, "frame_start"]
    table["frame_end"] = _integer_series(table["frame_end"], "frame_end")
    if (table["frame_end"] < table["frame_start"]).any():
        raise ValueError("event frame_end cannot precede frame_start")

    for index, row in table.iterrows():
        event_type = row["event_type"]
        if event_type == "pair_swap":
            a, b = str(row["slot_a"]).strip(), str(row["slot_b"]).strip()
            if a not in slot_set or b not in slot_set or a == b:
                raise ValueError(f"invalid pair_swap slots at row {index}: {a!r}, {b!r}")
            if row["frame_start"] != row["frame_end"]:
                raise ValueError("pair_swap must occur at one transition frame")
        elif event_type == "false_positive":
            slot = str(row["slot_a"]).strip()
            if slot not in slot_set:
                raise ValueError(f"invalid false_positive slot at row {index}: {slot!r}")
            if row["view_scope"] not in CAMERA_SCOPES:
                raise ValueError("false_positive events must target one camera view")
        elif event_type in NODE_QC_EVENT_TYPES:
            slot = str(row["slot_a"]).strip()
            if slot not in slot_set:
                raise ValueError(
                    f"invalid {event_type} slot at row {index}: {slot!r}"
                )
            if row["view_scope"] not in CAMERA_SCOPES:
                raise ValueError(
                    f"{event_type} events must target one camera view"
                )
            if _optional_text(row["slot_b"]):
                raise ValueError(
                    f"{event_type} events target exactly one slot; "
                    "slot_b must be empty"
                )
        elif event_type == "review_progress":
            if row["view_scope"] not in CAMERA_SCOPES:
                raise ValueError(
                    "review_progress events must target one camera view"
                )
            if row["frame_start"] != row["frame_end"]:
                raise ValueError(
                    "review_progress must record one checkpoint frame"
                )
            if _optional_text(row["slot_a"]) or _optional_text(row["slot_b"]):
                raise ValueError(
                    "review_progress is camera-scoped; slot_a and slot_b "
                    "must be empty"
                )
        elif event_type == "mapping_anchor":
            parse_mapping(row["mapping_after"], slot_set)
            if row["frame_start"] != row["frame_end"]:
                raise ValueError("mapping_anchor must occur at one frame")

    event_ids = table["event_id"].astype(str).str.strip()
    if event_ids.eq("").any():
        next_ids = [f"E{index + 1:06d}" for index in range(len(table))]
        table.loc[event_ids.eq(""), "event_id"] = np.asarray(next_ids, dtype=object)[
            event_ids.eq("").to_numpy()
        ]
    if table["event_id"].astype(str).duplicated().any():
        raise ValueError("event_id values must be unique")
    type_order = {name: index for index, name in enumerate(EVENT_TYPES)}
    table["_type_order"] = table["event_type"].map(type_order)
    table["_input_order"] = np.arange(len(table))
    table.sort_values(
        ["frame_start", "_type_order", "_input_order"], kind="stable", inplace=True
    )
    return table.drop(columns=["_type_order", "_input_order"]).reset_index(drop=True)


def events_for_scope(
    events: pd.DataFrame | None,
    slots: Iterable[str],
    view_scope: str,
) -> pd.DataFrame:
    """Return legacy-global plus target-specific events for one review panel."""
    if view_scope not in EVENT_SCOPES or view_scope == "all":
        raise ValueError(
            f"view_scope must be one of {', '.join(EVENT_SCOPES[1:])}, got {view_scope!r}"
        )
    normalized = normalize_events(events, slots)
    return normalized[
        normalized["view_scope"].isin(("all", view_scope))
    ].reset_index(drop=True)


def false_positive_slots_at_frame(
    events: pd.DataFrame | None,
    slots: Iterable[str],
    frame: int,
    *,
    view_scope: str,
) -> dict[str, list[str]]:
    """Return camera-view geometry slots whose matched SLEAP pose is an FP."""
    scoped = events_for_scope(events, slots, view_scope)
    active = scoped[
        scoped["event_type"].eq("false_positive")
        & scoped["frame_start"].le(int(frame))
        & scoped["frame_end"].ge(int(frame))
    ]
    result: dict[str, list[str]] = {}
    for row in active.itertuples(index=False):
        result.setdefault(str(row.slot_a), []).append(str(row.event_id))
    return result


def node_qc_events_at_frame(
    events: pd.DataFrame | None,
    slots: Iterable[str],
    frame: int,
    *,
    view_scope: str,
) -> dict[str, dict[str, list[str]]]:
    """Return active node/link QC event IDs grouped by geometry slot.

    The result is shaped for direct GUI use::

        {"G2": {"bad_mouth": ["E000012"], "wrong_link": ["E000013"]}}

    Events are active throughout their inclusive frame interval. Node QC is
    deliberately camera-specific because the same 3D slot can have a good
    pose match in one view and a bad node or mouth-tail link in another.
    """
    if view_scope not in CAMERA_SCOPES:
        raise ValueError(
            "view_scope must be one camera scope: "
            + ", ".join(CAMERA_SCOPES)
        )
    scoped = events_for_scope(events, slots, view_scope)
    active = scoped[
        scoped["event_type"].isin(NODE_QC_EVENT_TYPES)
        & scoped["frame_start"].le(int(frame))
        & scoped["frame_end"].ge(int(frame))
    ]
    result: dict[str, dict[str, list[str]]] = {}
    for row in active.itertuples(index=False):
        by_type = result.setdefault(str(row.slot_a), {})
        by_type.setdefault(str(row.event_type), []).append(str(row.event_id))
    return result


def review_progress_checkpoint(
    events: pd.DataFrame | None,
    slots: Iterable[str],
    *,
    view_scope: str,
) -> dict[str, object] | None:
    """Return the latest explicit checkpoint for one camera review scope.

    GUI-written ledgers are normalized by frame after every edit, so row order
    is not creation order. ISO ``created_at`` timestamps therefore determine
    the latest checkpoint. If timestamps are absent (for example in a hand-
    written ledger), the furthest checkpoint frame is returned.
    """
    if view_scope not in CAMERA_SCOPES:
        raise ValueError(
            "view_scope must be one camera scope: "
            + ", ".join(CAMERA_SCOPES)
        )
    normalized = normalize_events(events, slots)
    checkpoints = normalized[
        normalized["event_type"].eq("review_progress")
        & normalized["view_scope"].eq(view_scope)
    ].copy()
    if checkpoints.empty:
        return None

    created = pd.to_datetime(checkpoints["created_at"], errors="coerce", utc=True)
    if created.notna().any():
        latest_timestamp = created.max()
        row = checkpoints.loc[created.eq(latest_timestamp)].iloc[-1]
    else:
        # normalize_events orders by frame, then deterministic event order.
        row = checkpoints.iloc[-1]
    return {
        "event_id": str(row["event_id"]),
        "view_scope": str(row["view_scope"]),
        "frame": int(row["frame_start"]),
        "created_at": _optional_text(row["created_at"]),
        "notes": _optional_text(row["notes"]),
    }


def _truthy(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.lower().isin({"1", "true", "yes", "y"})


def mapping_at_frame(
    events: pd.DataFrame | None,
    slots: Iterable[str],
    frame: int,
    *,
    initial_mapping: Mapping[str, str] | None = None,
    view_scope: str = "3d",
) -> tuple[dict[str, str], list[str]]:
    """Return the reviewed slot mapping and active ambiguity event IDs."""
    mapping, active, _resolved = identity_state_at_frame(
        events,
        slots,
        frame,
        initial_mapping=initial_mapping,
        view_scope=view_scope,
    )
    return mapping, active


def identity_state_at_frame(
    events: pd.DataFrame | None,
    slots: Iterable[str],
    frame: int,
    *,
    initial_mapping: Mapping[str, str] | None = None,
    view_scope: str = "3d",
) -> tuple[dict[str, str], list[str], bool]:
    """Return mapping, active ambiguities, and whether identity is resolved.

    Once an ambiguous interval ends, identity remains unresolved until a
    complete ``mapping_anchor`` is recorded.  This prevents the pre-gap mapping
    from being silently propagated through an uncertain crossing.
    """
    slot_list = sorted({str(slot) for slot in slots})
    mapping = dict(initial_mapping or default_identity_mapping(slot_list))
    normalized = events_for_scope(events, slot_list, view_scope)
    active_ambiguity: list[str] = []
    resolved = True
    for _, event in normalized.iterrows():
        start, end = int(event["frame_start"]), int(event["frame_end"])
        if event["event_type"] == "ambiguous" and start <= frame <= end:
            active_ambiguity.append(str(event["event_id"]))
        if event["event_type"] == "ambiguous" and end < frame:
            resolved = False
        if start > frame:
            continue
        if event["event_type"] == "pair_swap":
            a, b = str(event["slot_a"]), str(event["slot_b"])
            mapping[a], mapping[b] = mapping[b], mapping[a]
        elif event["event_type"] == "mapping_anchor":
            mapping = parse_mapping(event["mapping_after"], slot_list)
            resolved = True
    return mapping, active_ambiguity, resolved


def replay_annotations(
    tracks: pd.DataFrame,
    events: pd.DataFrame | None,
    *,
    id_column: str = "fish_id",
    slot_map: Mapping[str, str] | None = None,
    initial_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Apply review events without changing geometry positions.

    A pair swap at frame ``f`` changes the mapping for observations at ``f``
    and later.  Ambiguous intervals set ``human_identity`` to missing.  The
    output preserves every source row and adds review/audit columns.
    """
    required = {"frame", id_column, "X", "Y", "Z"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"geometry track is missing columns: {sorted(missing)}")
    result = tracks.copy()
    result["frame"] = _integer_series(result["frame"], "frame")
    source_ids = result[id_column].astype(str)
    resolved_slot_map = dict(slot_map or infer_slot_map(source_ids))
    unknown = sorted(set(source_ids) - set(resolved_slot_map))
    if unknown:
        raise ValueError(f"slot_map is missing source IDs: {unknown}")
    result["geometry_source_id"] = source_ids
    result["geometry_slot"] = source_ids.map(resolved_slot_map)
    slots = sorted(result["geometry_slot"].unique())
    mapping = dict(initial_mapping or default_identity_mapping(slots))
    if set(mapping) != set(slots) or len(set(mapping.values())) != len(mapping):
        raise ValueError("initial_mapping must be one-to-one and cover every geometry slot")
    normalized = events_for_scope(events, slots, "3d")

    changes: dict[int, list[pd.Series]] = {}
    invalidations: set[int] = set()
    ambiguous: list[tuple[int, int, str]] = []
    for _, event in normalized.iterrows():
        frame = int(event["frame_start"])
        if event["event_type"] in {"pair_swap", "mapping_anchor"}:
            changes.setdefault(frame, []).append(event)
        if event["event_type"] == "ambiguous":
            end = int(event["frame_end"])
            ambiguous.append((frame, end, str(event["event_id"])))
            invalidations.add(end + 1)

    mapping_by_frame: dict[int, dict[str, str]] = {}
    resolved_by_frame: dict[int, bool] = {}
    mapping_resolved = True
    for frame in sorted(result["frame"].unique()):
        if int(frame) in invalidations:
            mapping_resolved = False
        for event in changes.get(int(frame), []):
            if event["event_type"] == "pair_swap":
                a, b = str(event["slot_a"]), str(event["slot_b"])
                mapping[a], mapping[b] = mapping[b], mapping[a]
            else:
                mapping = parse_mapping(event["mapping_after"], slots)
                mapping_resolved = True
        mapping_by_frame[int(frame)] = dict(mapping)
        resolved_by_frame[int(frame)] = mapping_resolved

    identities = []
    mapping_states = []
    for frame, slot in zip(result["frame"], result["geometry_slot"], strict=True):
        state = mapping_by_frame[int(frame)]
        identities.append(state[str(slot)])
        mapping_states.append(format_mapping(state))
    result["human_identity"] = identities
    result["mapping_state"] = mapping_states
    result["mapping_resolved"] = result["frame"].map(resolved_by_frame).astype(bool)
    result["label_status"] = "clean"
    result["annotation_event_id"] = ""

    if "is_interpolated" in result:
        interpolated = _truthy(result["is_interpolated"])
        result.loc[interpolated, "label_status"] = "interpolated"
    finite = np.isfinite(result[["X", "Y", "Z"]].to_numpy(dtype=float)).all(axis=1)
    result.loc[~finite, "label_status"] = "missing"
    result.loc[~finite, "human_identity"] = pd.NA
    unresolved = ~result["mapping_resolved"] & finite
    result.loc[unresolved, "label_status"] = "unresolved"
    result.loc[unresolved, "human_identity"] = pd.NA

    for start, end, event_id in ambiguous:
        mask = result["frame"].between(start, end)
        result.loc[mask, "label_status"] = "ambiguous"
        result.loc[mask, "human_identity"] = pd.NA
        existing = result.loc[mask, "annotation_event_id"].astype(str)
        result.loc[mask, "annotation_event_id"] = np.where(
            existing.eq(""), event_id, existing + ";" + event_id
        )
    return result.sort_values(["frame", "geometry_slot"], kind="stable").reset_index(drop=True)


def build_clean_tracklets(
    corrected: pd.DataFrame,
    events: pd.DataFrame | None,
    *,
    max_gap_frames: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split clean observations into deterministic contiguous fragments."""
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be >= 0")
    required = {
        "frame",
        "geometry_slot",
        "human_identity",
        "label_status",
        "X",
        "Y",
        "Z",
    }
    missing = required - set(corrected)
    if missing:
        raise ValueError(f"corrected table is missing columns: {sorted(missing)}")
    slots = sorted(corrected["geometry_slot"].dropna().astype(str).unique())
    normalized = events_for_scope(events, slots, "3d")
    break_frames = set(
        normalized.loc[
            normalized["event_type"].isin({"pair_swap", "track_break", "mapping_anchor"}),
            "frame_start",
        ].astype(int)
    )
    clean = corrected[
        corrected["label_status"].eq("clean")
        & corrected["human_identity"].notna()
        & np.isfinite(corrected[["X", "Y", "Z"]].to_numpy(dtype=float)).all(axis=1)
    ].copy()
    clean.sort_values(["human_identity", "frame", "geometry_slot"], kind="stable", inplace=True)

    next_id = 1
    ids = pd.Series(index=clean.index, dtype=object)
    for _identity, group in clean.groupby("human_identity", sort=True, observed=True):
        previous_frame: int | None = None
        previous_slot: str | None = None
        current_id = ""
        for index, row in group.iterrows():
            frame, slot = int(row["frame"]), str(row["geometry_slot"])
            split = (
                previous_frame is None
                or frame - previous_frame > max_gap_frames + 1
                or slot != previous_slot
                or frame in break_frames
            )
            if split:
                current_id = f"T{next_id:06d}"
                next_id += 1
            ids.loc[index] = current_id
            previous_frame, previous_slot = frame, slot
    clean["clean_tracklet_id"] = ids
    clean.sort_values(["frame", "clean_tracklet_id"], kind="stable", inplace=True)
    clean.reset_index(drop=True, inplace=True)

    if clean.empty:
        manifest = pd.DataFrame(
            columns=[
                "clean_tracklet_id",
                "human_identity",
                "geometry_slot",
                "start_frame",
                "end_frame",
                "n_frames",
            ]
        )
    else:
        manifest = (
            clean.groupby("clean_tracklet_id", observed=True, sort=True)
            .agg(
                human_identity=("human_identity", "first"),
                geometry_slot=("geometry_slot", "first"),
                start_frame=("frame", "min"),
                end_frame=("frame", "max"),
                n_frames=("frame", "size"),
            )
            .reset_index()
        )
    return clean, manifest


def project_clean_tracklets(clean: pd.DataFrame, calibration_model_dir: Path) -> pd.DataFrame:
    """Project clean 3D fragments into all views with explicit provenance."""
    from scripts.reconstruction_3d.calibration import Calibration

    columns = [
        "frame",
        "clean_tracklet_id",
        "human_identity",
        "geometry_slot",
        "view",
        "x",
        "y",
        "coordinate_source",
    ]
    if clean.empty:
        return pd.DataFrame(columns=columns)
    calibration = Calibration(str(calibration_model_dir))
    xyz = clean[["X", "Y", "Z"]].to_numpy(dtype=float)
    projected = calibration.compute_imCoords_from_XYZ(xyz)
    views = ("top_cam", "side1_cam", "side2_cam")
    parts = []
    base = clean[["frame", "clean_tracklet_id", "human_identity", "geometry_slot"]]
    for view_index, view in enumerate(views):
        part = base.copy()
        part["view"] = view
        part["x"] = projected[:, view_index, 0]
        part["y"] = projected[:, view_index, 1]
        part["coordinate_source"] = "calibration_reprojection"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[columns]


def build_tracklet_constraints(clean: pd.DataFrame) -> pd.DataFrame:
    """Return unique cannot-link fragment pairs that coexist in a frame."""
    rows: set[tuple[str, str]] = set()
    for _, frame in clean.groupby("frame", observed=True, sort=False):
        ids = sorted(frame["clean_tracklet_id"].dropna().astype(str).unique())
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                rows.add((left, right))
    return pd.DataFrame(sorted(rows), columns=["tracklet_a", "tracklet_b"])


def export_reviewed_tracklets(
    *,
    geometry_csv: Path,
    events_csv: Path,
    output_dir: Path,
    calibration_model_dir: Path | None = None,
    id_column: str = "fish_id",
    slot_map: Mapping[str, str] | None = None,
    max_gap_frames: int = 0,
    write_constraints: bool = False,
) -> dict[str, object]:
    tracks = pd.read_csv(geometry_csv, low_memory=False)
    events = pd.read_csv(events_csv, keep_default_na=False) if events_csv.exists() else None
    corrected = replay_annotations(
        tracks, events, id_column=id_column, slot_map=slot_map
    )
    clean, manifest = build_clean_tracklets(
        corrected, events, max_gap_frames=max_gap_frames
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "corrected_tracks": output_dir / "corrected_geometry_tracks.csv",
        "clean_3d_tracklets": output_dir / "clean_3d_tracklets.csv",
        "tracklet_manifest": output_dir / "tracklet_manifest.csv",
    }
    corrected.to_csv(paths["corrected_tracks"], index=False)
    clean.to_csv(paths["clean_3d_tracklets"], index=False)
    manifest.to_csv(paths["tracklet_manifest"], index=False)
    if calibration_model_dir is not None:
        projected = project_clean_tracklets(clean, calibration_model_dir)
        paths["clean_2d_tracklets"] = output_dir / "clean_2d_tracklets.csv"
        projected.to_csv(paths["clean_2d_tracklets"], index=False)
    if write_constraints:
        constraints = build_tracklet_constraints(clean)
        paths["cannot_link_tracklets"] = output_dir / "cannot_link_tracklets.csv"
        constraints.to_csv(paths["cannot_link_tracklets"], index=False)

    status_counts = corrected["label_status"].value_counts().sort_index().to_dict()
    normalized_events = normalize_events(
        events, sorted(corrected["geometry_slot"].dropna().astype(str).unique())
    )
    scope_counts = normalized_events["view_scope"].value_counts().sort_index().to_dict()
    summary = {
        "schema_version": "geometry_swap_tracklets_v2",
        "geometry_csv": str(geometry_csv),
        "events_csv": str(events_csv),
        "id_column": id_column,
        "rows": int(len(corrected)),
        "frames": int(corrected["frame"].nunique()),
        "events": int(0 if events is None else len(events)),
        "event_scope_counts": {str(key): int(value) for key, value in scope_counts.items()},
        "clean_rows": int(len(clean)),
        "clean_tracklets": int(len(manifest)),
        "label_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    summary_path = output_dir / "clean_tracklet_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["outputs"]["summary"] = str(summary_path)
    return summary
