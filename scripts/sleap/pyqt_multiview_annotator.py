from __future__ import annotations

"""
Multi-view annotation GUI (PyQt5) for the medaka mating pipeline.

Designed around the actual workflow rather than a generic image-annotation
tool: 3 independent camera views are tiled with **calibration-synced**
zoom + pan, the user adds fish as **instances** (mouth → tail = one fish),
and the GUI **auto-suggests fish IDs** via cross-view calibration rather
than asking the user to remember which color is which fish.

Key concepts
------------

- ``focus_xyz``: a 3D point in the tank coordinate system. The 3 view panes
  always show the region around its projection through the calibration
  model. Click any view to set the focus to where you clicked (the click's
  2D position is back-projected by holding the un-clicked axes constant).
- ``zoom_factor``: uniform multiplier. Applies the same display scale to
  each pane so a fish looks the same physical size across views.
- **Instance**: a 2-keypoint object (mouth, tail) per view. ``Add instance``
  mode is a state machine — the next click places the mouth, the click
  after that places the tail and commits. Repeat in each view to fully
  annotate one fish.
- **Auto ID**: when an instance is committed in a view, the GUI projects
  its centroid into the other views via the calibration's pairwise models
  and ranks any existing instances there by reprojection error. The top
  candidate's fish ID is suggested in a popup.

Frame navigation, save, and skeleton handling reuse the round-2/round-3
pkg.slp data layout (mouth → tail edge).

CLI
---
    python -m scripts.sleap.pyqt_multiview_annotator \\
        --in  C:/.../annotation_seed_round3.pkg.slp \\
        --candidate-slp C:/.../prediction_candidates.slp \\
        --out C:/.../annotation_seed_round3_v3.pkg.slp \\
        --paths-config configs/paths.local.toml

``--in`` is always the authoritative annotation package. Predictions supplied
through ``--candidate-slp`` are a review-only overlay: merely viewing them
never converts them into user instances. Dragging/toggling a candidate or
pressing A accepts it; pressing R rejects the remaining candidates in the
current synchronized frame. Review decisions are persisted in provenance.
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIEWS: Tuple[str, ...] = ("top_cam", "side1_cam", "side2_cam")
VIEW_INDEX = {v: i for i, v in enumerate(VIEWS)}

VIEW_TO_VIDEO = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}

NODE_NAMES: Tuple[str, ...] = ("mouth", "tail")

# Per-fish RGB (matches Rerun viewer scheme). We don't pre-allocate IDs for
# more than three fish — extra instances simply get a grey palette.
FISH_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (31, 119, 180),    # ID_1 blue
    1: (255, 127, 14),    # ID_2 orange
    2: (44, 160, 44),     # ID_3 green
}
FALLBACK_COLOR = (170, 170, 170)

# Visual radius of the mouth/tail circles in image pixels. Keep Large at the
# historical 8 px so existing users see no change until they choose a smaller
# option from the toolbar.
KEYPOINT_SIZE_RADII_PX: Dict[str, float] = {
    "Small": 4.0,
    "Medium": 6.0,
    "Large": 8.0,
}
DEFAULT_KEYPOINT_SIZE = "Large"


def keypoint_radius(size_name: str) -> float:
    """Return the configured keypoint-circle radius in image pixels."""
    try:
        return KEYPOINT_SIZE_RADII_PX[size_name]
    except KeyError as exc:
        choices = ", ".join(KEYPOINT_SIZE_RADII_PX)
        raise ValueError(
            f"Unknown keypoint size {size_name!r}; choose one of: {choices}"
        ) from exc


def instance_fit_zoom(
    points_by_view: Dict[str, List[Tuple[float, float]]],
    focus_by_view: Dict[str, Tuple[float, float]],
    viewport_sizes: Dict[str, Tuple[int, int]],
    *,
    display_margin_px: float = 32.0,
    minimum_scene_span_px: float = 160.0,
) -> Optional[float]:
    """Return one zoom that fits every labeled point in its camera pane.

    The panes share a zoom, so the tightest horizontal or vertical constraint
    across all populated views wins. ``minimum_scene_span_px`` leaves useful
    image context when a frame contains only one point or a very small fish.
    """
    candidates: List[float] = []
    for view in VIEWS:
        raw_points = points_by_view.get(view, [])
        if not raw_points or view not in focus_by_view:
            continue
        points = np.asarray(raw_points, dtype=float).reshape(-1, 2)
        points = points[np.all(np.isfinite(points), axis=1)]
        focus = np.asarray(focus_by_view[view], dtype=float)
        if not len(points) or focus.shape != (2,) or not np.all(np.isfinite(focus)):
            continue

        viewport_w, viewport_h = viewport_sizes.get(view, (0, 0))
        usable_w = max(1.0, float(viewport_w) - 2.0 * display_margin_px)
        usable_h = max(1.0, float(viewport_h) - 2.0 * display_margin_px)
        max_dx = float(np.max(np.abs(points[:, 0] - focus[0])))
        max_dy = float(np.max(np.abs(points[:, 1] - focus[1])))
        scene_w = max(minimum_scene_span_px, 2.0 * max_dx)
        scene_h = max(minimum_scene_span_px, 2.0 * max_dy)
        candidates.extend((usable_w / scene_w, usable_h / scene_h))

    if not candidates:
        return None
    return float(np.clip(min(candidates), 0.05, 8.0))


def full_save_path(output_path: Path) -> Path:
    """Return the documented portable-save name for a quick-save path."""
    name = output_path.name
    if name.lower().endswith(".pkg.slp"):
        return output_path.with_name(name[: -len(".pkg.slp")] + "_full.pkg.slp")
    return output_path.with_name(output_path.stem + "_full" + output_path.suffix)


def copy_labels_provenance(labels) -> dict:
    """Return a detached copy of package provenance for saved annotations."""
    return deepcopy(dict(getattr(labels, "provenance", {}) or {}))


PREDICTION_REVIEW_PROVENANCE_KEY = "prediction_candidate_review"
PREDICTION_REVIEW_SCHEMA = "prediction-candidate-review-v2"
REVIEW_SLOT_TRACK_PREFIX = "prediction-review-slot-v1:"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a candidate artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_view_key(experiment: str, frame_idx: int, view: str) -> str:
    """Return the stable provenance key for one reviewed camera frame."""
    return f"{experiment}|{int(frame_idx)}|{view}"


def review_slot_key(
    experiment: str,
    frame_idx: int,
    view: str,
    fish_idx: int,
) -> str:
    """Return the stable provenance key for one candidate fish/view slot."""
    return (
        f"{experiment}|{int(frame_idx)}|{view}|"
        f"fish_slot={int(fish_idx)}"
    )


def review_slot_track_name(
    experiment: str,
    frame_idx: int,
    view: str,
    fish_idx: int,
) -> str:
    """Return an item-unique Track name that preserves a review slot."""
    return (
        f"{REVIEW_SLOT_TRACK_PREFIX}{quote(str(experiment), safe='')}:"
        f"{int(frame_idx)}:{quote(str(view), safe='')}:{int(fish_idx)}"
    )


def review_slot_index_from_instance(instance) -> Optional[int]:
    """Read a stable review slot index from an instance Track, if present."""
    track = getattr(instance, "track", None)
    name = str(getattr(track, "name", "") or "")
    if not name.startswith(REVIEW_SLOT_TRACK_PREFIX):
        return None
    value = name.rsplit(":", 1)[-1]
    try:
        slot = int(value)
    except ValueError:
        return None
    return slot if slot >= 0 else None


def reviewed_view_keys_from_provenance(labels) -> set[str]:
    """Read reviewed camera-frame keys from an annotation package."""
    provenance = copy_labels_provenance(labels)
    review = provenance.get(PREDICTION_REVIEW_PROVENANCE_KEY, {})
    if not isinstance(review, dict):
        return set()
    values = review.get("reviewed_views", [])
    return {str(value) for value in values if str(value).strip()}


def candidate_artifact_from_provenance(labels) -> dict:
    """Return the persisted prediction-candidate artifact record, if any."""
    provenance = copy_labels_provenance(labels)
    review = provenance.get(PREDICTION_REVIEW_PROVENANCE_KEY, {})
    if not isinstance(review, dict):
        return {}
    artifact = review.get("candidate_artifact", {})
    return deepcopy(artifact) if isinstance(artifact, dict) else {}


def review_slot_decisions_from_provenance(
    labels,
) -> Tuple[set[str], set[str]]:
    """Return accepted and rejected instance-review slot keys."""
    provenance = copy_labels_provenance(labels)
    review = provenance.get(PREDICTION_REVIEW_PROVENANCE_KEY, {})
    if not isinstance(review, dict):
        return set(), set()
    instance_review = review.get("instance_review", {})
    if not isinstance(instance_review, dict):
        return set(), set()
    accepted = {
        str(value)
        for value in instance_review.get("accepted_slots", [])
        if str(value).strip()
    }
    rejected = {
        str(value)
        for value in instance_review.get("rejected_slots", [])
        if str(value).strip()
    }
    # Accepted wins if malformed provenance contains the same slot in both.
    rejected.difference_update(accepted)
    return accepted, rejected


def candidate_artifact_record(path: Path) -> dict:
    """Build the persisted identity record for a candidate SLP."""
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def candidate_artifact_mismatch(
    persisted: dict,
    current: dict,
) -> Optional[str]:
    """Explain an unsafe resume/candidate mismatch, or return ``None``."""
    old_digest = str((persisted or {}).get("sha256", "")).strip().lower()
    new_digest = str((current or {}).get("sha256", "")).strip().lower()
    if old_digest and new_digest and old_digest != new_digest:
        return (
            "the resume file contains review decisions for a different "
            "candidate SLP (SHA-256 mismatch)"
        )
    return None


def ranked_instances(labeled_frame) -> list:
    """Return instances in the deterministic rank order used by the GUI."""
    return sorted(
        list(getattr(labeled_frame, "instances", []) or []),
        key=lambda inst: -(getattr(inst, "score", 0.0) or 0.0),
    )


def instances_by_review_slot(
    labeled_frame,
    n_slots: int,
) -> Dict[int, object]:
    """Place tracked instances in their saved slots, then rank untracked."""
    assignments: Dict[int, object] = {}
    untracked = []
    for instance in ranked_instances(labeled_frame):
        slot = review_slot_index_from_instance(instance)
        if slot is None or slot >= n_slots or slot in assignments:
            untracked.append(instance)
        else:
            assignments[slot] = instance
    empty_slots = [
        fish_idx
        for fish_idx in range(n_slots)
        if fish_idx not in assignments
    ]
    for fish_idx, instance in zip(empty_slots, untracked):
        assignments[fish_idx] = instance
    return assignments


def build_instance_review_sequence(
    authoritative_group_keys,
    candidate_groups,
    authoritative_groups=None,
    accepted_slot_keys=None,
    rejected_slot_keys=None,
) -> List[Tuple[Tuple[str, int], str, int]]:
    """Flatten incomplete per-view candidate items in authoritative order."""
    authoritative_groups = dict(authoritative_groups or {})
    decided_slot_keys = set(accepted_slot_keys or ()) | set(
        rejected_slot_keys or ()
    )
    sequence: List[Tuple[Tuple[str, int], str, int]] = []
    for key in authoritative_group_keys:
        candidate_views = dict(candidate_groups.get(key, {}))
        authoritative_views = dict(authoritative_groups.get(key, {}))
        for view in VIEWS:
            candidate_lf = candidate_views.get(view)
            if candidate_lf is None:
                continue
            candidates = ranked_instances(candidate_lf)
            authoritative = instances_by_review_slot(
                authoritative_views.get(view),
                len(candidates),
            )
            for fish_idx, candidate in enumerate(candidates):
                if (
                    review_slot_key(
                        key[0],
                        key[1],
                        view,
                        fish_idx,
                    )
                    in decided_slot_keys
                ):
                    continue
                mouth, _mouth_vis, tail, _tail_vis = (
                    instance_xy_vis_pair(candidate)
                )
                if mouth is None and tail is None:
                    continue
                authoritative_instance = authoritative.get(fish_idx)
                if authoritative_instance is None:
                    sequence.append((key, view, fish_idx))
                    continue
                auth_mouth, _mouth_vis, auth_tail, _tail_vis = (
                    instance_xy_vis_pair(authoritative_instance)
                )
                if auth_mouth is None or auth_tail is None:
                    sequence.append((key, view, fish_idx))
    return sequence


def fish_points_by_view(
    fish: List[FishKeypoints],
    fish_indices,
) -> Dict[str, List[Tuple[float, float]]]:
    """Collect finite mouth/tail points for selected fish slots."""
    points_by_view: Dict[str, List[Tuple[float, float]]] = {
        view: [] for view in VIEWS
    }
    for fish_idx in fish_indices:
        if fish_idx < 0 or fish_idx >= len(fish):
            continue
        keypoints = fish[fish_idx]
        for node_points in (keypoints.mouth, keypoints.tail):
            for view in VIEWS:
                point = node_points.get(view)
                if point is None:
                    continue
                xy = np.asarray(point, dtype=float)
                if xy.shape == (2,) and np.all(np.isfinite(xy)):
                    points_by_view[view].append(
                        (float(xy[0]), float(xy[1]))
                    )
    return points_by_view


# ---------------------------------------------------------------------------
# Data model — kept small and reused by the napari prototype too.
# ---------------------------------------------------------------------------

@dataclass
class FishKeypoints:
    """One fish across all three views (mouth, tail per view).

    For each (view, node) pair we keep a position (``None`` if not labeled
    in that view) and a visibility flag (``True`` if the keypoint is plainly
    visible at that position; ``False`` if the user marked it as occluded /
    hidden — the position is still useful for training but the model is
    told the keypoint isn't directly observable).
    """
    fish_id: Optional[int] = None
    mouth: Dict[str, Optional[Tuple[float, float]]] = field(
        default_factory=lambda: {v: None for v in VIEWS}
    )
    tail: Dict[str, Optional[Tuple[float, float]]] = field(
        default_factory=lambda: {v: None for v in VIEWS}
    )
    mouth_visible: Dict[str, bool] = field(
        default_factory=lambda: {v: True for v in VIEWS}
    )
    tail_visible: Dict[str, bool] = field(
        default_factory=lambda: {v: True for v in VIEWS}
    )
    # Candidate coordinates are visible/editable but are never serialized as
    # user labels until accepted explicitly or modified by the annotator.
    candidate_by_view: Dict[str, bool] = field(
        default_factory=lambda: {v: False for v in VIEWS}
    )
    accepted_by_view: Dict[str, bool] = field(
        default_factory=lambda: {v: False for v in VIEWS}
    )


def accept_fish_view(keypoints: FishKeypoints, view: str) -> bool:
    """Accept one populated candidate view and return whether it changed."""
    populated = (
        keypoints.mouth.get(view) is not None
        or keypoints.tail.get(view) is not None
    )
    if not populated:
        return False
    changed = not keypoints.accepted_by_view.get(view, False)
    keypoints.accepted_by_view[view] = True
    return changed


def fish_view_is_saveable(keypoints: FishKeypoints, view: str) -> bool:
    """Return whether this camera-view instance may become a user label."""
    populated = (
        keypoints.mouth.get(view) is not None
        or keypoints.tail.get(view) is not None
    )
    if not populated:
        return False
    if keypoints.candidate_by_view.get(view, False):
        return bool(keypoints.accepted_by_view.get(view, False))
    return True


def reject_candidate_fish_view(
    keypoints: FishKeypoints,
    view: str,
) -> bool:
    """Clear one unaccepted candidate view and return whether it changed."""
    if (
        not keypoints.candidate_by_view.get(view, False)
        or keypoints.accepted_by_view.get(view, False)
    ):
        return False
    keypoints.mouth[view] = None
    keypoints.tail[view] = None
    keypoints.candidate_by_view[view] = False
    keypoints.accepted_by_view[view] = False
    return True


def first_empty_fish_slot(fish: List[FishKeypoints]) -> Optional[int]:
    """Return the first completely unused fish slot, if one exists."""
    for fish_index, keypoints in enumerate(fish):
        if all(
            point is None
            for view in VIEWS
            for point in (
                keypoints.mouth.get(view),
                keypoints.tail.get(view),
            )
        ):
            return fish_index
    return None


@dataclass
class FrameGroup:
    experiment: str
    frame_idx: int
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    fish: List[FishKeypoints] = field(default_factory=list)
    src_lfs: Dict[str, "sio.LabeledFrame"] = field(default_factory=dict)
    reviewed_views: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Helpers shared with the napari prototype
# ---------------------------------------------------------------------------

def _video_source_path(video) -> str:
    """Walk the ``source_video`` chain on a sleap-io Video and return the
    deepest path that points at an actual on-disk video file (mp4 / avi /
    mov / mkv).

    sleap-io's ``save_slp`` with ``embed=…`` creates a chain at every
    save: ``new_pkg.source_video → previous_pkg → … → original.mp4``. We
    need the original path so :func:`classify_view` and
    :func:`experiment_from_video` can extract the (experiment, view)
    tags. Just reading ``video.filename`` or ``backend.source_filename``
    only gives the most recent step in the chain.
    """
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    visited: set[int] = set()
    cur = video
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        backend = getattr(cur, "backend", None)
        if backend is not None:
            sf = getattr(backend, "source_filename", None)
            if sf and str(sf).lower().endswith(video_exts):
                return str(sf)
        fname = getattr(cur, "filename", None)
        if fname and str(fname).lower().endswith(video_exts):
            return str(fname)
        cur = getattr(cur, "source_video", None)
    # Fallback: return whatever the top-level filename / backend says, even
    # if it's another .slp — caller will likely fail to classify, which is
    # the right signal (this video doesn't trace back to a known mp4).
    fallback_backend = getattr(video, "backend", None)
    fb = getattr(fallback_backend, "source_filename", None) if fallback_backend else None
    if fb:
        return str(fb)
    return str(getattr(video, "filename", "")) or ""


def classify_view(fname: str) -> Optional[str]:
    s = str(fname).lower().replace("\\", "/")
    for tag in VIEWS:
        if f"/{tag}/" in s:
            return tag
    if "side1" in s:
        return "side1_cam"
    if "side2" in s:
        return "side2_cam"
    if "top_cam" in s or "cam0_top" in s:
        return "top_cam"
    return None


def experiment_from_video(fname: str) -> Optional[str]:
    parts = Path(str(fname).replace("\\", "/")).parts
    for i, p in enumerate(parts):
        if p in VIEWS and i > 0:
            return parts[i - 1]
    return None


def instance_xy_pair(inst):
    """Return ``(mouth_xy, tail_xy)`` as in the original API."""
    pts = inst.numpy()
    mouth = (
        (float(pts[0, 0]), float(pts[0, 1]))
        if pts.shape[0] > 0 and np.isfinite(pts[0]).all()
        else None
    )
    tail = (
        (float(pts[1, 0]), float(pts[1, 1]))
        if pts.shape[0] > 1 and np.isfinite(pts[1]).all()
        else None
    )
    return mouth, tail


def instance_xy_vis_pair(inst):
    """Return ``(mouth_xy, mouth_visible, tail_xy, tail_visible)``.

    Reads the structured ``inst.points`` array so the per-node ``visible``
    flag is preserved (sleap-io keeps it on every Instance).
    """
    pts = getattr(inst, "points", None)
    if pts is not None and len(pts) >= 1:
        m_xy = pts[0]["xy"]
        if np.isfinite(m_xy).all():
            mouth = (float(m_xy[0]), float(m_xy[1]))
            mouth_vis = bool(pts[0]["visible"])
        else:
            mouth, mouth_vis = None, True
    else:
        mouth, mouth_vis = None, True
    if pts is not None and len(pts) >= 2:
        t_xy = pts[1]["xy"]
        if np.isfinite(t_xy).all():
            tail = (float(t_xy[0]), float(t_xy[1]))
            tail_vis = bool(pts[1]["visible"])
        else:
            tail, tail_vis = None, True
    else:
        tail, tail_vis = None, True
    return mouth, mouth_vis, tail, tail_vis


def _instance_centroid(mouth, tail) -> Optional[Tuple[float, float]]:
    pts = []
    if mouth is not None:
        pts.append(mouth)
    if tail is not None:
        pts.append(tail)
    if not pts:
        return None
    arr = np.array(pts, dtype=float)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def predict_missing_view(
    cal: Calibration,
    pts_by_view: Dict[str, Optional[Tuple[float, float]]],
) -> Dict[str, Optional[Tuple[float, float]]]:
    """Project a missing view from the two observed views.

    This uses the public calibration API so it works for both the historical
    polynomial model (which completes the missing view with its helper model)
    and the fixed-interface refractive model (which intersects the two water
    rays directly). The returned point is editable annotation assistance, not
    a user-verified label.
    """
    out = dict(pts_by_view)
    have = [v for v in VIEWS if pts_by_view[v] is not None]
    if len(have) != 2:
        return out
    missing = [v for v in VIEWS if v not in have][0]
    triplet = np.full((3, 2), np.nan, dtype=float)
    for view in have:
        triplet[VIEW_INDEX[view]] = np.asarray(pts_by_view[view], dtype=float)
    try:
        xyz = np.asarray(cal.triangulate(triplet), dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            return out
        projected = np.asarray(cal.project_world(xyz), dtype=float)
        arr = projected[VIEW_INDEX[missing]]
        if arr.shape != (2,) or not np.all(np.isfinite(arr)):
            return out
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return out
    out[missing] = (float(arr[0]), float(arr[1]))
    return out


def triangulate_xyz(cal: Calibration, pts: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Triangulate populated views and return 3D coordinates in millimetres."""
    triplet = np.array([list(pts[v]) for v in VIEWS], dtype=float)
    return np.asarray(cal.triangulate(triplet), dtype=float)


def reproject_xyz(cal: Calibration, xyz: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Project a 3D point into each view through the selected backend."""
    out = np.asarray(cal.project_world(np.asarray(xyz, dtype=float)), dtype=float)
    if out.shape != (3, 2) or not np.all(np.isfinite(out)):
        raise RuntimeError("Calibration returned invalid three-view projections.")
    return {v: (float(out[VIEW_INDEX[v], 0]), float(out[VIEW_INDEX[v], 1])) for v in VIEWS}


def back_project_view_click(
    cal: Calibration,
    view: str,
    click_xy: Tuple[float, float],
    focus_xyz: np.ndarray,
) -> np.ndarray:
    """Take a 2D click in *view* and update ``focus_xyz`` so its projection
    in that view lands on ``click_xy``. The other coordinate(s) of
    ``focus_xyz`` are kept fixed.

    Approximation: solve a small 2D root-finding problem by Newton-stepping
    along the in-plane axes for that view (cam0=XY, cam1=XZ, cam2=YZ) until
    the projected point matches the click within a few pixels.
    """
    in_plane_axes = {
        "top_cam": (0, 1),
        "side1_cam": (0, 2),
        "side2_cam": (1, 2),
    }[view]

    target = np.asarray(click_xy, dtype=float)
    xyz = np.asarray(focus_xyz, dtype=float).copy()

    for _ in range(8):
        proj = cal.compute_imCoords_from_XYZ(xyz.reshape(1, 3))[0]
        cur = proj[VIEW_INDEX[view]]
        err = target - cur
        if np.linalg.norm(err) < 0.5:
            break
        # Numerical Jacobian d(view 2D) / d(in-plane 3D axes), step 1mm.
        eps = 1.0
        J = np.zeros((2, 2))
        for col, ax in enumerate(in_plane_axes):
            xyz_p = xyz.copy()
            xyz_p[ax] += eps
            proj_p = cal.compute_imCoords_from_XYZ(xyz_p.reshape(1, 3))[0]
            J[:, col] = (proj_p[VIEW_INDEX[view]] - cur) / eps
        if abs(np.linalg.det(J)) < 1e-6:
            break
        delta = np.linalg.solve(J, err)
        for col, ax in enumerate(in_plane_axes):
            xyz[ax] += float(delta[col])
        # Clamp to a reasonable tank-extension cube to keep the polynomial
        # in its trained domain.
        # Loose clamp: well outside the calibrated tank cube so the user
        # can pan/zoom near the edges without the focus_xyz hitting a wall
        # and "snapping back" relative to the user's view.
        xyz = np.clip(xyz, -200.0, 400.0)
    return xyz


# ---------------------------------------------------------------------------
# Source data: groups
# ---------------------------------------------------------------------------

def build_groups(labels):
    groups: Dict[Tuple[str, int], Dict[str, "sio.LabeledFrame"]] = defaultdict(dict)
    for lf in labels.labeled_frames:
        fname = _video_source_path(lf.video)
        view = classify_view(fname)
        exp = experiment_from_video(fname)
        if view is None or exp is None:
            continue
        groups[(exp, int(lf.frame_idx))][view] = lf
    return dict(groups)


def filter_groups(groups, *, min_views: int = 1):
    keys = [k for k, gv in groups.items() if len(gv) >= min_views]
    keys.sort()
    return keys


def parse_missing_instance_target(
    value: str,
) -> Tuple[Tuple[str, int], str]:
    """Parse ``EXPERIMENT:FRAME:VIEW`` for targeted missing-fish review."""
    try:
        experiment, frame_text, view = value.rsplit(":", 2)
        frame_idx = int(frame_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "missing-instance target must be EXPERIMENT:FRAME:VIEW"
        ) from exc
    if not experiment:
        raise ValueError("missing-instance target has an empty experiment")
    if view not in VIEWS:
        raise ValueError(
            f"unknown view {view!r}; choose one of: {', '.join(VIEWS)}"
        )
    return (experiment, frame_idx), view


def build_missing_instance_sequence(
    groups,
    targets: List[Tuple[Tuple[str, int], str]],
    *,
    expected_instances: int,
) -> List[Tuple[Tuple[str, int], str]]:
    """Queue one mouth→tail task per deficient instance in requested views.

    Counts are evaluated after resume labels have been merged, so relaunching
    the same command skips additions that were already saved successfully.
    """
    if expected_instances < 1:
        raise ValueError("expected_instances must be at least 1")
    sequence: List[Tuple[Tuple[str, int], str]] = []
    seen = set()
    for key, view in targets:
        target = (key, view)
        if target in seen:
            raise ValueError(
                f"duplicate missing-instance target: {key[0]}:{key[1]}:{view}"
            )
        seen.add(target)
        frames_by_view = groups.get(key)
        if frames_by_view is None:
            raise ValueError(
                f"missing-instance target group is absent: {key[0]}:{key[1]}"
            )
        lf = frames_by_view.get(view)
        if lf is None:
            raise ValueError(
                "missing-instance target view is absent: "
                f"{key[0]}:{key[1]}:{view}"
            )
        deficit = expected_instances - len(lf.instances)
        if deficit < 0:
            raise ValueError(
                f"{key[0]}:{key[1]}:{view} has {len(lf.instances)} "
                f"instances, above expected {expected_instances}"
            )
        sequence.extend([target] * deficit)
    return sequence


def _merge_resume_labels(seed_labels, resume_labels):
    """Overlay annotations from ``resume_labels`` (a previous quick-save)
    onto ``seed_labels``.

    For every LF in resume keyed by ``(experiment, frame_idx, view)``, the
    matching seed LF's ``instances`` are replaced with the resume LF's.
    The seed's video object is retained so the GUI keeps reading frame
    data from the seed's embedded HDF5 (resume files are usually
    quick-saves that don't embed and would force network reads).

    Resume LFs that don't match any seed group are appended as-is — useful
    if the user added entirely new instances at indices the seed didn't
    cover.
    """
    import sleap_io as sio

    resume_index = {}
    for lf in resume_labels.labeled_frames:
        sp = _video_source_path(lf.video)
        view = classify_view(sp)
        exp = experiment_from_video(sp)
        if view is None or exp is None:
            continue
        resume_index[(exp, int(lf.frame_idx), view)] = lf

    new_lfs = []
    overlaid = 0
    seen = set()
    for lf in seed_labels.labeled_frames:
        sp = _video_source_path(lf.video)
        view = classify_view(sp)
        exp = experiment_from_video(sp)
        key = (
            (exp, int(lf.frame_idx), view) if view and exp else None
        )
        if key is not None and key in resume_index:
            r_lf = resume_index[key]
            new_lfs.append(sio.LabeledFrame(
                video=lf.video,           # keep seed's video for fast frame reads
                frame_idx=lf.frame_idx,
                instances=list(r_lf.instances),
            ))
            overlaid += 1
            seen.add(key)
        else:
            new_lfs.append(lf)

    # Carry over resume LFs that have no seed counterpart.
    appended = 0
    for key, r_lf in resume_index.items():
        if key in seen:
            continue
        new_lfs.append(r_lf)
        appended += 1

    print(f"  overlaid={overlaid} LF(s); appended={appended} new LF(s).")

    provenance = copy_labels_provenance(seed_labels)
    resume_review = copy_labels_provenance(resume_labels).get(
        PREDICTION_REVIEW_PROVENANCE_KEY
    )
    if isinstance(resume_review, dict):
        provenance[PREDICTION_REVIEW_PROVENANCE_KEY] = deepcopy(
            resume_review
        )

    merged_tracks = []
    seen_track_ids = set()
    for track in (
        list(getattr(seed_labels, "tracks", []) or [])
        + list(getattr(resume_labels, "tracks", []) or [])
        + [
            getattr(instance, "track", None)
            for lf in new_lfs
            for instance in lf.instances
        ]
    ):
        if track is None or id(track) in seen_track_ids:
            continue
        seen_track_ids.add(id(track))
        merged_tracks.append(track)

    return sio.Labels(
        skeletons=seed_labels.skeletons,
        videos=list({id(lf.video): lf.video for lf in new_lfs}.values()),
        labeled_frames=new_lfs,
        tracks=merged_tracks,
        suggestions=seed_labels.suggestions,
        provenance=provenance,
    )


def load_video_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    import cv2

    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def load_image_for_view(
    lf,
    *,
    medaka_data_dir: Path,
    experiment: str,
    view: str,
    frame_idx: int,
) -> Optional[np.ndarray]:
    import cv2

    if lf is not None and lf.video is not None:
        try:
            frame = lf.video[int(lf.frame_idx)]
            if isinstance(frame, np.ndarray) and frame.size > 0:
                if frame.ndim == 3 and frame.shape[2] == 3:
                    return frame.copy()
                if frame.ndim == 2:
                    return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass
    return load_video_frame(
        medaka_data_dir / experiment / view / VIEW_TO_VIDEO[view],
        int(frame_idx),
    )


def build_frame_group(
    key,
    src_lfs,
    *,
    medaka_data_dir,
    cal,
    candidate_src_lfs=None,
    reviewed_views=None,
    accepted_slot_keys=None,
    rejected_slot_keys=None,
):
    exp, frame_idx = key
    candidate_src_lfs = dict(candidate_src_lfs or {})
    accepted_slot_keys = set(accepted_slot_keys or ())
    rejected_slot_keys = set(rejected_slot_keys or ())
    grp = FrameGroup(
        experiment=exp,
        frame_idx=frame_idx,
        src_lfs=dict(src_lfs),
        reviewed_views=set(reviewed_views or ()),
    )
    for view in VIEWS:
        lf = src_lfs.get(view)
        img = load_image_for_view(
            lf,
            medaka_data_dir=medaka_data_dir,
            experiment=exp,
            view=view,
            frame_idx=frame_idx,
        )
        if img is None:
            img = np.full((1200, 1200, 3), 30, dtype=np.uint8)
        grp.images[view] = img

    # Up to NUM_FISH instances per view; align by detection-score order.
    n_fish = 0
    for frames_by_view in (src_lfs, candidate_src_lfs):
        for view in VIEWS:
            lf = frames_by_view.get(view)
            if lf is not None:
                n_fish = max(n_fish, len(lf.instances))
                tracked_slots = [
                    review_slot_index_from_instance(instance)
                    for instance in lf.instances
                ]
                tracked_slots = [
                    slot for slot in tracked_slots if slot is not None
                ]
                if tracked_slots:
                    n_fish = max(n_fish, max(tracked_slots) + 1)
    n_fish = max(1, n_fish)
    grp.fish = [FishKeypoints(fish_id=i) for i in range(n_fish)]

    # Authoritative input instances are already user labels and stay
    # saveable. Candidate instances are added only to otherwise empty slots.
    for view, lf in src_lfs.items():
        if lf is None:
            continue
        assignments = instances_by_review_slot(lf, n_fish)
        for i, inst in sorted(assignments.items()):
            mouth, mouth_vis, tail, tail_vis = instance_xy_vis_pair(inst)
            grp.fish[i].mouth[view] = mouth
            grp.fish[i].tail[view] = tail
            grp.fish[i].mouth_visible[view] = mouth_vis
            grp.fish[i].tail_visible[view] = tail_vis
            grp.fish[i].accepted_by_view[view] = True

    for view, lf in candidate_src_lfs.items():
        if lf is None or view in grp.reviewed_views:
            continue
        ranked = ranked_instances(lf)
        for i, inst in enumerate(ranked[:n_fish]):
            slot_key = review_slot_key(exp, frame_idx, view, i)
            if (
                slot_key in accepted_slot_keys
                or slot_key in rejected_slot_keys
            ):
                continue
            keypoints = grp.fish[i]
            if (
                keypoints.mouth.get(view) is not None
                or keypoints.tail.get(view) is not None
            ):
                continue
            mouth, mouth_vis, tail, tail_vis = instance_xy_vis_pair(inst)
            if mouth is None and tail is None:
                continue
            keypoints.mouth[view] = mouth
            keypoints.tail[view] = tail
            keypoints.mouth_visible[view] = mouth_vis
            keypoints.tail_visible[view] = tail_vis
            keypoints.candidate_by_view[view] = True
            keypoints.accepted_by_view[view] = False
    return grp


# ---------------------------------------------------------------------------
# Qt view pane
# ---------------------------------------------------------------------------

def _qt():
    from PyQt5 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


class ViewPane:
    """One camera view: image + keypoint overlays + zoom/pan tied to a
    shared focus_xyz.

    Implemented over QGraphicsView so we can scale freely and overlay the
    keypoints as items.
    """

    def __init__(self, view_name: str, on_click, on_pan, on_wheel=None):
        QtCore, QtGui, QtWidgets = _qt()
        self.view_name = view_name
        self._on_click = on_click  # callable(view_name, x_image, y_image)
        # Navigation callbacks receive the *actual* scene center produced by
        # this pane's QGraphicsView transform. The main window back-projects
        # that center to synchronize the other two cameras.
        self._on_pan = on_pan      # callable(view_name, center_x, center_y)
        self._on_wheel = on_wheel  # callable(view_name, center_x, center_y, zoom)
        self._press_pos = None     # for click-vs-drag detection
        self._press_in_drag = False

        self.scene = QtWidgets.QGraphicsScene()
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.view.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.view.setMouseTracking(True)
        # Mouse events are emitted by QGraphicsView's viewport widget. Setting
        # tracking only on the outer view leaves drag moves missing on some
        # Windows/PyQt builds.
        self.view.viewport().setMouseTracking(True)

        # Capture mouse events on the view itself so our click handler runs
        # regardless of which scene item is under the cursor.
        self._event_filter = _make_event_filter(self)
        self.view.viewport().installEventFilter(self._event_filter)

        self.image_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self.keypoint_items: List[QtWidgets.QGraphicsItem] = []
        self.label_item: Optional[QtWidgets.QGraphicsTextItem] = None
        self._set_label(view_name)

        self.view.setMinimumSize(400, 400)

    # -------------------- Public API --------------------

    def set_image(self, img_rgb: np.ndarray) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        h, w = img_rgb.shape[:2]
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=2)
        # Ensure contiguous uint8 RGB array.
        if img_rgb.dtype != np.uint8:
            img_rgb = img_rgb.astype(np.uint8)
        if not img_rgb.flags["C_CONTIGUOUS"]:
            img_rgb = np.ascontiguousarray(img_rgb)
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(
            img_rgb.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg)

        if self.image_item is None:
            self.image_item = self.scene.addPixmap(pix)
            self.image_item.setZValue(0)
        else:
            self.image_item.setPixmap(pix)
        self.scene.setSceneRect(0, 0, w, h)
        self._reposition_label()

    def set_focus(self, focus_xy: Tuple[float, float], pixels_per_image_px: float) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self.view.resetTransform()
        self.view.scale(pixels_per_image_px, pixels_per_image_px)
        # Center the view on the focus image-space coordinate.
        self.view.centerOn(QtCore.QPointF(focus_xy[0], focus_xy[1]))
        self._reposition_label()

    def current_center(self) -> Tuple[float, float]:
        """Return the image-space point currently at the viewport center."""
        center = self.view.viewport().rect().center()
        scene_point = self.view.mapToScene(center)
        return float(scene_point.x()), float(scene_point.y())

    def current_zoom(self) -> float:
        """Return display pixels per image pixel for this pane."""
        return float(abs(self.view.transform().m11()))

    def zoom_at_viewport_pos(
        self,
        viewport_pos,
        factor: float,
        *,
        min_zoom: float = 0.05,
        max_zoom: float = 8.0,
    ) -> Tuple[Tuple[float, float], float]:
        """Scale while keeping the image pixel under ``viewport_pos`` fixed.

        QGraphicsView's transformation-anchor behavior is sensitive to hidden
        scrollbar ranges and platform event timing. This method enforces the
        cursor invariant explicitly: measure the scene point under the cursor,
        scale, then shift the scene center by the measured residual.
        """
        QtCore, QtGui, QtWidgets = _qt()
        old_zoom = self.current_zoom()
        if old_zoom <= 0:
            old_zoom = 1.0
        target_zoom = float(np.clip(old_zoom * factor, min_zoom, max_zoom))
        if abs(target_zoom - old_zoom) < 1e-12:
            return self.current_center(), old_zoom

        anchor_before = self.view.mapToScene(viewport_pos)
        self.view.scale(target_zoom / old_zoom, target_zoom / old_zoom)
        anchor_after = self.view.mapToScene(viewport_pos)
        center_after_scale = self.view.mapToScene(
            self.view.viewport().rect().center()
        )
        corrected_center = center_after_scale + (anchor_before - anchor_after)
        self.view.centerOn(corrected_center)
        return self.current_center(), target_zoom

    def pan_by_viewport_delta(
        self,
        dx_px: float,
        dy_px: float,
    ) -> Tuple[float, float]:
        """Move the displayed image with a mouse drag and return its center."""
        QtCore, QtGui, QtWidgets = _qt()
        zoom = max(self.current_zoom(), 1e-12)
        center_x, center_y = self.current_center()
        self.view.centerOn(
            QtCore.QPointF(
                center_x - float(dx_px) / zoom,
                center_y - float(dy_px) / zoom,
            )
        )
        return self.current_center()

    def clear_keypoints(self) -> None:
        for item in self.keypoint_items:
            try:
                self.scene.removeItem(item)
            except Exception:
                pass
        self.keypoint_items.clear()

    def add_keypoint(
        self,
        x: float,
        y: float,
        *,
        color_rgb: Tuple[int, int, int],
        radius: float = 6.0,
        filled: bool = True,
        fill_alpha: int = 110,
        z: float = 1.0,
        label: Optional[str] = None,
    ) -> None:
        """Static (non-draggable) keypoint marker. Use ``add_draggable_keypoint``
        for editable mouth/tail markers."""
        QtCore, QtGui, QtWidgets = _qt()
        pen = QtGui.QPen(QtGui.QColor(*color_rgb))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        if filled:
            brush = QtGui.QBrush(QtGui.QColor(*color_rgb, fill_alpha))
        else:
            brush = QtGui.QBrush(QtCore.Qt.NoBrush)
        ellipse = self.scene.addEllipse(
            x - radius, y - radius, 2 * radius, 2 * radius, pen, brush
        )
        ellipse.setZValue(z)
        self.keypoint_items.append(ellipse)
        if label:
            text = self.scene.addText(label)
            text.setDefaultTextColor(QtGui.QColor(*color_rgb))
            font = text.font()
            font.setBold(True)
            text.setFont(font)
            text.setPos(x + radius + 2, y - radius - 2)
            text.setZValue(z + 0.1)
            self.keypoint_items.append(text)

    def add_draggable_keypoint(
        self,
        x: float,
        y: float,
        *,
        color_rgb: Tuple[int, int, int],
        radius: float = 8.0,
        fill_alpha: int = 110,
        view_name: str,
        fish_idx: int,
        node: str,
        on_moved,
        on_release,
        on_delete,
        on_toggle_visible,
        on_request_swap,
        label: Optional[str] = None,
        candidate: bool = False,
    ) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        item = _make_keypoint_item(
            x, y, radius,
            view_name=view_name, fish_idx=fish_idx, node=node,
            color_rgb=color_rgb, fill_alpha=fill_alpha,
            on_moved=on_moved, on_release=on_release, on_delete=on_delete,
            on_toggle_visible=on_toggle_visible,
            on_request_swap=on_request_swap,
            candidate=candidate,
            z=2.0,
        )
        self.scene.addItem(item)
        self.keypoint_items.append(item)
        if label:
            text = self.scene.addText(label)
            text.setDefaultTextColor(QtGui.QColor(*color_rgb))
            font = text.font()
            font.setBold(True)
            text.setFont(font)
            text.setPos(x + radius + 2, y - radius - 2)
            text.setZValue(2.1)
            self.keypoint_items.append(text)

    def add_halo(
        self,
        x: float,
        y: float,
        *,
        color_rgb: Tuple[int, int, int],
        radius: float = 16.0,
        width: float = 2.5,
    ) -> None:
        """Hollow ring drawn behind a keypoint to convey reprojection error."""
        QtCore, QtGui, QtWidgets = _qt()
        pen = QtGui.QPen(QtGui.QColor(*color_rgb, 230))
        pen.setWidthF(width)
        pen.setCosmetic(True)
        ring = self.scene.addEllipse(
            x - radius, y - radius, 2 * radius, 2 * radius, pen,
            QtGui.QBrush(QtCore.Qt.NoBrush),
        )
        ring.setZValue(1.5)
        self.keypoint_items.append(ring)

    def add_line(self, p1, p2, color_rgb, width=1.5):
        QtCore, QtGui, QtWidgets = _qt()
        pen = QtGui.QPen(QtGui.QColor(*color_rgb))
        pen.setWidthF(width)
        pen.setCosmetic(True)
        line = self.scene.addLine(p1[0], p1[1], p2[0], p2[1], pen)
        line.setZValue(0.5)
        self.keypoint_items.append(line)

    def add_arrow(
        self,
        p_tail,
        p_mouth,
        color_rgb,
        width=1.5,
        head_len=13.0,
        head_deg=26.0,
        tip_gap=10.0,
        candidate: bool = False,
    ):
        """Body-axis edge drawn as an arrow pointing tail -> mouth.

        The arrowhead sits just short of the mouth marker (backed off by
        ``tip_gap`` so it clears the keypoint circle). Because the head marks
        the mouth end, a mouth<->tail flip reverses the arrow and is obvious at
        a glance -- the main thing an annotator is hunting for at crossings.
        Sizes are in scene (image-pixel) units so they scale with zoom like the
        keypoints; the pen is cosmetic so the stroke stays crisp.
        """
        import math
        QtCore, QtGui, QtWidgets = _qt()
        xt, yt = float(p_tail[0]), float(p_tail[1])
        xm, ym = float(p_mouth[0]), float(p_mouth[1])
        pen = QtGui.QPen(QtGui.QColor(*color_rgb))
        pen.setWidthF(width)
        pen.setCosmetic(True)
        if candidate:
            pen.setStyle(QtCore.Qt.DashLine)
        dx, dy = xm - xt, ym - yt
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return
        ux, uy = dx / L, dy / L
        # Shorten the head geometry for very short (curled-fish) edges so the
        # arrowhead never overshoots the tail.
        tg = min(tip_gap, 0.4 * L)
        hl = min(head_len, 0.5 * L)
        tipx, tipy = xm - ux * tg, ym - uy * tg          # arrowhead apex
        shaft = self.scene.addLine(xt, yt, tipx, tipy, pen)
        shaft.setZValue(0.5)
        self.keypoint_items.append(shaft)
        ang = math.atan2(uy, ux)
        ha = math.radians(head_deg)
        for s in (+1.0, -1.0):
            bx = tipx - hl * math.cos(ang + s * ha)
            by = tipy - hl * math.sin(ang + s * ha)
            barb = self.scene.addLine(tipx, tipy, bx, by, pen)
            barb.setZValue(0.5)
            self.keypoint_items.append(barb)

    # -------------------- internals --------------------

    def _set_label(self, name: str) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        if self.label_item is None:
            self.label_item = self.scene.addText(name)
            font = self.label_item.font()
            font.setPointSize(14)
            font.setBold(True)
            self.label_item.setFont(font)
            self.label_item.setDefaultTextColor(QtGui.QColor(220, 220, 220))
            self.label_item.setZValue(2)
        self.label_item.setPlainText(name)
        self._reposition_label()

    def _reposition_label(self):
        if self.label_item is None:
            return
        self.label_item.setPos(8, 8)

    # -------------------- mouse events --------------------

    def _handle_press(self, ev):
        QtCore, QtGui, QtWidgets = _qt()
        if ev.button() != QtCore.Qt.LeftButton:
            return
        # If the click landed on a draggable keypoint, defer to Qt's
        # built-in item drag and don't track the press at the view level.
        scene_pt = self.view.mapToScene(ev.pos())
        item = self.scene.itemAt(scene_pt, self.view.transform())
        if (
            item is not None
            and item is not self.image_item
            and item is not self.label_item
            and (item.flags() & QtWidgets.QGraphicsItem.ItemIsMovable)
        ):
            self._press_pos = None
            return
        self._press_pos = (ev.pos().x(), ev.pos().y())
        # Set to True the first time movement crosses the click→drag
        # threshold; subsequent moves emit pan events on every pixel
        # without re-checking. Reset on release.
        self._press_in_drag = False

    def _handle_move(self, ev):
        QtCore, QtGui, QtWidgets = _qt()
        if self._press_pos is None:
            return
        # Some Windows/PyQt viewport paths report ``NoButton`` on move
        # events even while the left-button press is still active. The
        # press/release handlers already own the gesture lifecycle, so
        # `_press_pos` is the reliable drag-state signal here.
        dx = ev.pos().x() - self._press_pos[0]
        dy = ev.pos().y() - self._press_pos[1]
        if not self._press_in_drag:
            # Wait until the user has moved enough that this is clearly
            # a drag gesture, not a click. After that, emit every move.
            if abs(dx) <= 4 and abs(dy) <= 4:
                return
            self._press_in_drag = True
        if dx == 0 and dy == 0:
            return
        center_x, center_y = self.pan_by_viewport_delta(dx, dy)
        self._on_pan(self.view_name, center_x, center_y)
        self._press_pos = (ev.pos().x(), ev.pos().y())

    def _handle_release(self, ev):
        QtCore, QtGui, QtWidgets = _qt()
        if ev.button() == QtCore.Qt.LeftButton and self._press_pos is not None:
            # If the user never crossed the drag threshold, treat the press
            # as a plain click (focus-set / add-instance).
            if not self._press_in_drag:
                scene_pt = self.view.mapToScene(ev.pos())
                self._on_click(self.view_name, float(scene_pt.x()), float(scene_pt.y()))
            self._press_pos = None
            self._press_in_drag = False

    def _handle_wheel(self, ev):
        if self._on_wheel is None:
            return False
        delta_y = ev.angleDelta().y()
        if delta_y == 0:
            delta_y = ev.pixelDelta().y()
        if delta_y == 0:
            return True
        factor = 1.0 + 0.0015 * abs(delta_y)
        if delta_y < 0:
            factor = 1.0 / factor
        (center_x, center_y), zoom = self.zoom_at_viewport_pos(
            ev.pos(), factor
        )
        self._on_wheel(self.view_name, center_x, center_y, zoom)
        return True


def _make_keypoint_item(
    x: float,
    y: float,
    radius: float,
    *,
    view_name: str,
    fish_idx: int,
    node: str,
    color_rgb: Tuple[int, int, int],
    fill_alpha: int = 110,
    on_moved,
    on_release,
    on_delete,
    on_toggle_visible,
    on_request_swap,
    candidate: bool = False,
    z: float = 1.0,
):
    """Create a movable :class:`QGraphicsEllipseItem` that streams its
    position back via callbacks. Right-click pops up a delete menu.
    """
    QtCore, QtGui, QtWidgets = _qt()

    class _KeypointItem(QtWidgets.QGraphicsEllipseItem):
        def __init__(self):
            super().__init__(-radius, -radius, 2 * radius, 2 * radius)
            self.setFlag(self.ItemIsMovable, True)
            self.setFlag(self.ItemIsSelectable, True)
            self.setFlag(self.ItemSendsScenePositionChanges, True)
            self.setAcceptedMouseButtons(
                QtCore.Qt.LeftButton | QtCore.Qt.RightButton
            )
            pen = QtGui.QPen(QtGui.QColor(*color_rgb))
            pen.setWidthF(2.0)
            pen.setCosmetic(True)
            if candidate:
                pen.setStyle(QtCore.Qt.DashLine)
            self.setPen(pen)
            self.setBrush(QtGui.QBrush(QtGui.QColor(*color_rgb, fill_alpha)))
            self.setPos(x, y)
            self.setZValue(z)
            # Tag with metadata for the drag/delete handlers.
            self._view_name = view_name
            self._fish_idx = fish_idx
            self._node = node
            self._on_moved = on_moved
            self._on_release = on_release
            self._on_delete = on_delete
            self._on_toggle_visible = on_toggle_visible
            self._on_request_swap = on_request_swap
            self._dragging = False

        def itemChange(self, change, value):
            if change == self.ItemScenePositionHasChanged and self._dragging:
                pos = self.pos()
                self._on_moved(
                    self._view_name, self._fish_idx, self._node,
                    float(pos.x()), float(pos.y()),
                )
            return super().itemChange(change, value)

        def mousePressEvent(self, ev):
            if ev.button() == QtCore.Qt.RightButton:
                ctrl = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
                if not ctrl:
                    # Plain right-click = toggle hidden/visible (most common
                    # operation, no menu).
                    self._on_toggle_visible(
                        self._view_name, self._fish_idx, self._node,
                    )
                    ev.accept()
                    return
                # Ctrl + right-click = destructive / structural menu.
                menu = QtWidgets.QMenu()
                act_swap = menu.addAction(
                    f"Swap ID_{self._fish_idx + 1} ↔ another (across all views)…"
                )
                menu.addSeparator()
                act_kp = menu.addAction(
                    f"Delete this {self._node} (ID_{self._fish_idx + 1}, "
                    f"{self._view_name})"
                )
                act_inst = menu.addAction(
                    f"Delete instance (ID_{self._fish_idx + 1} mouth+tail "
                    f"in {self._view_name})"
                )
                menu.addSeparator()
                act_fish = menu.addAction(
                    f"Delete ID_{self._fish_idx + 1} from ALL views"
                )
                chosen = menu.exec_(ev.screenPos())
                if chosen is None:
                    ev.accept()
                    return
                if chosen is act_swap:
                    self._on_request_swap(
                        self._view_name, self._fish_idx,
                    )
                elif chosen is act_kp:
                    self._on_delete(self._view_name, self._fish_idx, self._node, "keypoint")
                elif chosen is act_inst:
                    self._on_delete(self._view_name, self._fish_idx, None, "instance")
                elif chosen is act_fish:
                    self._on_delete(self._view_name, self._fish_idx, None, "fish")
                ev.accept()
                return
            self._dragging = True
            super().mousePressEvent(ev)

        def mouseReleaseEvent(self, ev):
            super().mouseReleaseEvent(ev)
            if self._dragging:
                self._dragging = False
                self._on_release(
                    self._view_name, self._fish_idx, self._node,
                )

    return _KeypointItem()


def _error_color(err_px: float) -> Tuple[int, int, int]:
    """Map per-view reprojection error to RGB: green→yellow→red."""
    if not np.isfinite(err_px):
        return (160, 160, 160)
    if err_px < 3.0:
        return (60, 200, 80)
    if err_px < 8.0:
        return (220, 200, 30)
    return (230, 60, 50)


def _make_event_filter(pane: ViewPane):
    """Create a QObject subclass that forwards viewport mouse events to the
    owning :class:`ViewPane`. Done via a closure so we can defer the import
    of Qt classes until call time.
    """
    QtCore, QtGui, QtWidgets = _qt()

    class _Filter(QtCore.QObject):
        def eventFilter(self, _obj, ev):
            et = ev.type()
            if et == QtCore.QEvent.MouseButtonPress:
                pane._handle_press(ev)
            elif et == QtCore.QEvent.MouseMove:
                pane._handle_move(ev)
            elif et == QtCore.QEvent.MouseButtonRelease:
                pane._handle_release(ev)
            elif et == QtCore.QEvent.Wheel:
                # Consume the event so QGraphicsView doesn't also try to
                # scroll vertically when the user wheels for zoom.
                if pane._handle_wheel(ev):
                    return True
            return False

    return _Filter()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MultiviewWindow:
    """Top-level annotation window."""

    def __init__(
        self,
        *,
        labels,
        cal: Calibration,
        group_keys: List[Tuple[str, int]],
        groups_src,
        medaka_data_dir: Path,
        out_path: Path,
        allow_calibration_fill: bool = True,
        candidate_groups=None,
        candidate_artifact: Optional[dict] = None,
        instance_review: bool = False,
        missing_instance_sequence=None,
    ):
        QtCore, QtGui, QtWidgets = _qt()
        self.labels = labels
        self.cal = cal
        self.group_keys = group_keys
        self.groups_src = groups_src
        self.medaka_data_dir = medaka_data_dir
        self.out_path = out_path
        self.allow_calibration_fill = bool(allow_calibration_fill)
        self.candidate_groups = dict(candidate_groups or {})
        self.candidate_artifact = deepcopy(candidate_artifact or {})
        self.candidate_review_enabled = bool(self.candidate_artifact)
        self.instance_review = bool(instance_review)
        self.missing_instance_sequence = list(
            missing_instance_sequence or []
        )
        self.missing_instance_idx = 0
        self.missing_instance_review = bool(
            self.missing_instance_sequence
        )
        if self.instance_review and not self.candidate_review_enabled:
            raise ValueError(
                "instance review requires a prediction candidate overlay"
            )
        if self.instance_review and self.missing_instance_review:
            raise ValueError(
                "instance review and missing-instance review are exclusive"
            )
        self.reviewed_view_keys = reviewed_view_keys_from_provenance(labels)
        (
            self.accepted_slot_keys,
            self.rejected_slot_keys,
        ) = review_slot_decisions_from_provenance(labels)
        self.instance_review_sequence = (
            build_instance_review_sequence(
                group_keys,
                self.candidate_groups,
                self.groups_src,
                self.accepted_slot_keys,
                self.rejected_slot_keys,
            )
            if self.instance_review
            else []
        )
        if self.instance_review and not self.instance_review_sequence:
            raise ValueError(
                "candidate overlay has no populated fish slots to review"
            )
        self.instance_review_idx = 0
        self._group_index_by_key = {
            key: index for index, key in enumerate(self.group_keys)
        }

        self.cur_idx: int = 0
        if self.instance_review_sequence:
            first_key, _first_view, _first_fish_idx = (
                self.instance_review_sequence[0]
            )
            self.cur_idx = self._group_index_by_key[first_key]
        elif self.missing_instance_sequence:
            first_key, _first_view = self.missing_instance_sequence[0]
            self.cur_idx = self._group_index_by_key[first_key]
        self.cur_group: Optional[FrameGroup] = None
        # Tank center as initial focus (mm).
        self.focus_xyz: np.ndarray = np.array([100.0, 100.0, 100.0])
        self.zoom: float = 0.5  # display pixels per image pixel
        self.keypoint_size: str = DEFAULT_KEYPOINT_SIZE
        # State machine for instance entry: None | "place_mouth" | "place_tail"
        self.add_mode: Optional[str] = None
        self.pending_view: Optional[str] = None
        self.pending_mouth: Optional[Tuple[float, float]] = None
        self.pending_fish_idx: Optional[int] = None
        # Track which view the user last touched. F uses this as the anchor
        # for cross-view re-matching: the latest edits are presumed correct,
        # so other views are re-paired to follow them.
        self.last_edited_view: Optional[str] = None
        # Per-group edit cache so navigating away doesn't drop the user's
        # changes. Keyed by ``(experiment, frame_idx)``. Built on demand;
        # rebuilt from the source LFs only when first visited.
        self._edited_groups: Dict[Tuple[str, int], FrameGroup] = {}

        # Build window.
        self.window = QtWidgets.QMainWindow()
        title = "Medaka multi-view annotator (PyQt)"
        if getattr(self, "instance_review", False):
            title += " — instance-by-instance prediction review"
        elif self.missing_instance_review:
            title += " — add missing fish"
        elif self.candidate_review_enabled:
            title += " — human-in-the-loop prediction review"
        elif not self.allow_calibration_fill:
            title += " — independent manual points"
        self.window.setWindowTitle(title)
        self.window.resize(1500, 900)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        # Toolbar.
        self.toolbar = QtWidgets.QHBoxLayout()
        self._build_toolbar(self.toolbar)
        root_layout.addLayout(self.toolbar)

        # Three view panes.
        self.panes: Dict[str, ViewPane] = {}
        view_row = QtWidgets.QHBoxLayout()
        for view in VIEWS:
            pane = ViewPane(
                view, self.on_pane_click, self.on_pane_pan, self.on_pane_wheel,
            )
            self.panes[view] = pane
            view_row.addWidget(pane.view)
        root_layout.addLayout(view_row, stretch=1)

        # Status bar.
        self.status = self.window.statusBar()
        self.status.showMessage("Ready.")

        # Right-side help panel.
        self._add_help_dock()

        # Belt-and-braces: an application-level key filter that catches the
        # most-used shortcuts regardless of widget focus. QShortcut doesn't
        # always win against QGraphicsView's keyboard handling.
        self._install_global_key_filter()

        # Initial render.
        self._render_current()

        # Safety net: the annotator otherwise only persists on Ctrl+S (no
        # auto-save), so closing the window silently discards edits. Add a
        # periodic quiet autosave plus a save when the app is about to quit;
        # both write the same quick-save file that auto-resume reloads next
        # open, so forgetting Ctrl+S no longer loses work.
        QtCore, QtGui, QtWidgets = _qt()
        self._autosave_timer = QtCore.QTimer(self.window)
        self._autosave_timer.setInterval(120_000)  # 2 minutes
        self._autosave_timer.timeout.connect(self._autosave_quiet)
        self._autosave_timer.start()
        _app = QtWidgets.QApplication.instance()
        if _app is not None:
            _app.aboutToQuit.connect(self._autosave_quiet)

    def _autosave_quiet(self):
        """Autosave (periodic + on app exit) so edits survive even if the user
        never presses Ctrl+S. Silent on success; failures are surfaced/logged
        by :meth:`_safe_save`. Writes the quick-save file auto-resume reloads.
        """
        if self.cur_group is None and not self._edited_groups:
            return
        self._safe_save("autosave", self.out_path, embed=False)

    def _install_global_key_filter(self):
        QtCore, QtGui, QtWidgets = _qt()
        win = self  # closure capture

        class _KeyFilter(QtCore.QObject):
            def eventFilter(self, obj, ev):
                if ev.type() != QtCore.QEvent.KeyPress:
                    return False
                key = ev.key()
                mods = ev.modifiers()
                # Only handle "no modifier" presses so Ctrl+S etc. still work
                # via QShortcut.
                no_mods = mods == QtCore.Qt.NoModifier
                if no_mods and key == QtCore.Qt.Key_F:
                    win.fill_missing_views()
                    return True
                if no_mods and key == QtCore.Qt.Key_I:
                    win.add_btn.toggle()
                    return True
                if no_mods and key == QtCore.Qt.Key_Left:
                    win.prev_frame()
                    return True
                if no_mods and key == QtCore.Qt.Key_Right:
                    win.next_frame()
                    return True
                if no_mods and key == QtCore.Qt.Key_Escape:
                    win.cancel_add()
                    return True
                if no_mods and key == QtCore.Qt.Key_Equal:
                    win.bump_zoom(1.25)
                    return True
                if no_mods and key == QtCore.Qt.Key_Minus:
                    win.bump_zoom(0.8)
                    return True
                if no_mods and key == QtCore.Qt.Key_0:
                    win.fit_zoom()
                    return True
                return False

        self._global_key_filter = _KeyFilter()
        QtWidgets.QApplication.instance().installEventFilter(self._global_key_filter)

    # -------------------- toolbar --------------------

    def _build_toolbar(self, layout):
        QtCore, QtGui, QtWidgets = _qt()

        prev_btn = QtWidgets.QToolButton()
        prev_btn.setText("◀ Prev (Left)")
        prev_btn.clicked.connect(self.prev_frame)
        layout.addWidget(prev_btn)

        next_btn = QtWidgets.QToolButton()
        next_btn.setText("Next (Right) ▶")
        next_btn.clicked.connect(self.next_frame)
        layout.addWidget(next_btn)

        layout.addSpacing(20)

        self.add_btn = QtWidgets.QToolButton()
        self.add_btn.setText("Add instance (I)")
        self.add_btn.setCheckable(True)
        self.add_btn.toggled.connect(self.toggle_add_mode)
        layout.addWidget(self.add_btn)

        if self.candidate_review_enabled:
            accept_btn = QtWidgets.QToolButton()
            accept_btn.setText(
                "Accept active fish (A)"
                if self.instance_review
                else "Accept candidates (A)"
            )
            accept_btn.setToolTip(
                (
                    "Accept the one active per-view fish candidate."
                )
                if self.instance_review
                else (
                    "Accept every visible prediction candidate in this "
                    "synchronized frame group."
                )
            )
            accept_btn.clicked.connect(self.accept_current_candidates)
            layout.addWidget(accept_btn)

            reject_btn = QtWidgets.QToolButton()
            reject_btn.setText(
                "Reject active fish (R)"
                if self.instance_review
                else "Reject candidates (R)"
            )
            reject_btn.setToolTip(
                (
                    "Reject the one active per-view fish candidate. Other "
                    "candidates remain undecided."
                )
                if self.instance_review
                else (
                    "Reject every unaccepted prediction candidate in this "
                    "synchronized frame group. Rejections persist across "
                    "resume."
                )
            )
            reject_btn.clicked.connect(self.reject_current_candidates)
            layout.addWidget(reject_btn)

        zoom_in_btn = QtWidgets.QToolButton()
        zoom_in_btn.setText("Zoom + (=)")
        zoom_in_btn.clicked.connect(lambda: self.bump_zoom(1.25))
        layout.addWidget(zoom_in_btn)

        zoom_out_btn = QtWidgets.QToolButton()
        zoom_out_btn.setText("Zoom − (-)")
        zoom_out_btn.clicked.connect(lambda: self.bump_zoom(0.8))
        layout.addWidget(zoom_out_btn)

        zoom_fit_btn = QtWidgets.QToolButton()
        zoom_fit_btn.setText("Fit view (0)")
        zoom_fit_btn.clicked.connect(self.fit_zoom)
        layout.addWidget(zoom_fit_btn)

        fit_instances_btn = QtWidgets.QToolButton()
        fit_instances_btn.setText(
            "Fit active (Ctrl+0)"
            if self.instance_review
            else "Fit instances (Ctrl+0)"
        )
        fit_instances_btn.setToolTip(
            (
                "Center and zoom tightly on the active review instance."
                if self.instance_review
                else (
                    "Center and zoom all three views to the current "
                    "frame's fish."
                )
            )
        )
        fit_instances_btn.clicked.connect(self.fit_instances)
        layout.addWidget(fit_instances_btn)

        keypoint_size_label = QtWidgets.QLabel("Keypoints:")
        layout.addWidget(keypoint_size_label)

        self.keypoint_size_combo = QtWidgets.QComboBox()
        self.keypoint_size_combo.addItems(list(KEYPOINT_SIZE_RADII_PX))
        self.keypoint_size_combo.setCurrentText(self.keypoint_size)
        self.keypoint_size_combo.setToolTip(
            "Change the mouth/tail circle size in all three views."
        )
        self.keypoint_size_combo.currentTextChanged.connect(
            self.set_keypoint_size
        )
        layout.addWidget(self.keypoint_size_combo)

        layout.addStretch(1)

        self.frame_label = QtWidgets.QLabel("")
        layout.addWidget(self.frame_label)

        save_btn = QtWidgets.QToolButton()
        save_btn.setText("Save (Ctrl+S)")
        save_btn.setToolTip(
            "Quick save — references the seed's videos. Fast (~seconds). "
            "Use this for working iterations."
        )
        save_btn.clicked.connect(self.save)
        layout.addWidget(save_btn)

        save_full_btn = QtWidgets.QToolButton()
        save_full_btn.setText("Save full (Ctrl+Shift+S)")
        save_full_btn.setToolTip(
            "Full self-contained save (writes <out>_full.pkg.slp). "
            "Re-embeds every user-labeled frame so the file is portable "
            "across machines. Slower."
        )
        save_full_btn.clicked.connect(self.save_full)
        layout.addWidget(save_full_btn)

        # Keyboard shortcuts on the main window.
        self._make_shortcut("Left", self.prev_frame)
        self._make_shortcut("Right", self.next_frame)
        self._make_shortcut("I", lambda: self.add_btn.toggle())
        self._make_shortcut("=", lambda: self.bump_zoom(1.25))
        self._make_shortcut("+", lambda: self.bump_zoom(1.25))
        self._make_shortcut("-", lambda: self.bump_zoom(0.8))
        self._make_shortcut("0", self.fit_zoom)
        self._make_shortcut("Ctrl+0", self.fit_instances)
        self._make_shortcut("Ctrl+S", self.save)
        self._make_shortcut("Ctrl+Shift+S", self.save_full)
        self._make_shortcut("Escape", self.cancel_add)
        self._make_shortcut("F", self.fill_missing_views)
        if self.candidate_review_enabled:
            self._make_shortcut("A", self.accept_current_candidates)
            self._make_shortcut("R", self.reject_current_candidates)

    def set_keypoint_size(self, size_name: str) -> None:
        """Apply a toolbar-selected marker size without altering labels."""
        keypoint_radius(size_name)  # Validate before changing GUI state.
        self.keypoint_size = size_name
        self._draw_keypoints()
        self.status.showMessage(f"Keypoint circles: {size_name}.", 2500)

    def _active_review_item(
        self,
    ) -> Optional[Tuple[Tuple[str, int], str, int]]:
        """Return the selected per-view review item, if enabled."""
        if not self.instance_review_sequence:
            return None
        return self.instance_review_sequence[self.instance_review_idx]

    def _active_missing_instance_item(
        self,
    ) -> Optional[Tuple[Tuple[str, int], str]]:
        """Return the selected deficient view, if targeted add mode is on."""
        sequence = getattr(self, "missing_instance_sequence", [])
        if not sequence:
            return None
        return sequence[self.missing_instance_idx]

    def _active_fish_idx(self) -> Optional[int]:
        """Return the selected review slot, or ``None`` in frame mode."""
        item = self._active_review_item()
        return item[2] if item is not None else None

    def _active_review_view(self) -> Optional[str]:
        """Return the selected camera view, or ``None`` in frame mode."""
        item = self._active_review_item()
        if item is not None:
            return item[1]
        missing_item = self._active_missing_instance_item()
        return missing_item[1] if missing_item is not None else None

    def _mark_fish_view_decision(
        self,
        fish_idx: int,
        view: str,
        decision: str,
    ) -> None:
        """Persist an edit at view level or instance-slot level."""
        if not self.instance_review:
            self._mark_view_reviewed(view)
            return
        if self.cur_group is None:
            return
        key = review_slot_key(
            self.cur_group.experiment,
            self.cur_group.frame_idx,
            view,
            fish_idx,
        )
        if decision == "accepted":
            self.accepted_slot_keys.add(key)
            self.rejected_slot_keys.discard(key)
        elif decision == "rejected":
            self.rejected_slot_keys.add(key)
            self.accepted_slot_keys.discard(key)
        else:
            raise ValueError(f"Unknown review decision: {decision!r}")

    def accept_current_candidates(self) -> None:
        """Accept candidates in the current frame or active fish slot."""
        if self.cur_group is None:
            return
        active_fish_idx = self._active_fish_idx()
        fish_items = (
            [(active_fish_idx, self.cur_group.fish[active_fish_idx])]
            if active_fish_idx is not None
            and active_fish_idx < len(self.cur_group.fish)
            else list(enumerate(self.cur_group.fish))
        )
        active_view = self._active_review_view()
        views = [active_view] if active_view is not None else VIEWS
        accepted = 0
        incomplete = 0
        for fish_idx, keypoints in fish_items:
            for view in views:
                if self.instance_review and (
                    keypoints.mouth.get(view) is None
                    or keypoints.tail.get(view) is None
                ):
                    incomplete += 1
                    continue
                if not keypoints.candidate_by_view.get(view, False):
                    continue
                if accept_fish_view(keypoints, view):
                    accepted += 1
                self._mark_fish_view_decision(
                    fish_idx, view, "accepted"
                )
        self._draw_keypoints()
        self._update_status()
        if incomplete:
            self.status.showMessage(
                "Active candidate is partial. Press I, then click the "
                "missing mouth or tail before accepting it.",
                6000,
            )
            return
        self.status.showMessage(
            (
                f"Accepted active fish in {accepted} view(s)."
                if self.instance_review
                else f"Accepted {accepted} candidate view-instance(s)."
            ),
            4000,
        )

    def reject_current_candidates(self) -> None:
        """Reject candidates in the current frame or active fish slot."""
        if self.cur_group is None:
            return
        active_fish_idx = self._active_fish_idx()
        fish_items = (
            [(active_fish_idx, self.cur_group.fish[active_fish_idx])]
            if active_fish_idx is not None
            and active_fish_idx < len(self.cur_group.fish)
            else list(enumerate(self.cur_group.fish))
        )
        active_view = self._active_review_view()
        views = [active_view] if active_view is not None else VIEWS
        rejected = 0
        for fish_idx, keypoints in fish_items:
            for view in views:
                if reject_candidate_fish_view(keypoints, view):
                    self._mark_fish_view_decision(
                        fish_idx, view, "rejected"
                    )
                    rejected += 1
        self._draw_keypoints()
        self._update_status()
        self.status.showMessage(
            (
                f"Rejected active fish in {rejected} view(s)."
                if self.instance_review
                else f"Rejected {rejected} candidate view-instance(s)."
            ),
            4000,
        )

    def _make_shortcut(self, key, handler):
        QtCore, QtGui, QtWidgets = _qt()
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self.window)
        # ApplicationShortcut so the shortcut fires even when focus is on a
        # child widget (the QGraphicsView panes capture keyboard input
        # otherwise).
        sc.setContext(QtCore.Qt.ApplicationShortcut)
        sc.activated.connect(handler)

    def _add_help_dock(self):
        QtCore, QtGui, QtWidgets = _qt()
        if self.instance_review:
            fill_help = (
                "<b>F is disabled</b> in instance review because its "
                "frame-wide re-matching would modify hidden fish slots.<br>"
            )
        elif self.missing_instance_review:
            fill_help = (
                "<b>Targeted missing-fish mode</b>: only the deficient view "
                "is shown. Click mouth, then tail; the addition is saved and "
                "the GUI advances automatically.<br>"
            )
        elif not self.allow_calibration_fill and self.candidate_review_enabled:
            fill_help = (
                "<b>F is disabled</b>: review/correct the displayed model "
                "candidates directly, or add missing points manually.<br>"
            )
        elif not self.allow_calibration_fill:
            fill_help = (
                "<b>F is disabled</b> for this independent benchmark; every "
                "mouth and tail coordinate must be clicked manually.<br>"
            )
        else:
            fill_help = (
                "<b>F</b>: re-match all fish across views via calibration "
                "(anchored on the most recently edited view) and re-predict "
                "the keypoints in the OTHER views from the new matches. The "
                "anchor view's keypoints are kept as-is; other views are "
                "Hungarian-paired by reprojection error and the empty slots "
                "are filled.<br>"
            )
        if self.instance_review:
            candidate_help = (
                "<b>Instance review</b>: one camera pane and one candidate "
                "are shown and tightly centered. <kbd>←</kbd>/<kbd>→</kbd> "
                "move through per-view instances across frames. Gray dashed "
                "points are unsaved candidates. Drag/change or press "
                "<kbd>A</kbd> to accept this item; <kbd>R</kbd> rejects only "
                "this item. Other candidates remain undecided and reviewable "
                "after resume."
                "<br><br>"
            )
        elif self.candidate_review_enabled:
            candidate_help = (
                "<b>Prediction review</b>: gray dashed instances are "
                "candidates, not saved annotations. Drag or change a "
                "candidate to accept it; <kbd>A</kbd> accepts all candidates "
                "in this synchronized frame. <kbd>R</kbd> rejects all "
                "still-gray candidates. Acceptance and rejection persist "
                "across resume.<br><br>"
            )
        else:
            candidate_help = ""
        navigation_help = (
            "fish candidate.<br>"
            if self.instance_review
            else "frame.<br>"
        )
        text = (
            "<h3>Multi-view annotator</h3>"
            "<b>Synced focus</b>: click any view → focus point moves to "
            "where you clicked; the other 2 views follow.<br>"
            "Drag = pan (in-plane axes for that view). Wheel = zoom both views.<br><br>"
            "<b>Add instance (I)</b>: enter add mode. Next click = mouth; "
            "click after that = tail. The two clicks must be in the same "
            "view. After committing, candidate IDs are computed via "
            "calibration and one is shown in a popup.<br><br>"
            + candidate_help
            + fill_help
            + "<b>Right-click a keypoint</b>: toggle hidden ↔ visible "
            "immediately. Hidden keypoints are drawn with the alternate "
            "fill and saved with <code>visible=False</code>.<br>"
            "<b>Ctrl + right-click a keypoint</b>: menu with Swap ID, "
            "delete-this-keypoint, delete-this-instance, "
            "delete-fish-across-all-views.<br><br>"
            "<b>Navigation</b>: <kbd>←</kbd> previous, <kbd>→</kbd> next "
            + navigation_help
            + "<b>Zoom</b>: <kbd>=</kbd>/<kbd>-</kbd>; <kbd>0</kbd> fits the "
            "view and <kbd>Ctrl+0</kbd> fits all current instances.<br>"
            "<b>Keypoints</b>: choose Small, Medium, or Large in the toolbar "
            "to change the mouth/tail circle size in all three views.<br>"
            "<b>Save</b>: <kbd>Ctrl+S</kbd> = quick save (references seed). "
            "<kbd>Ctrl+Shift+S</kbd> = full save with embedded frames "
            "(<code>&lt;out&gt;_full.pkg.slp</code>) — for sharing.<br>"
            "<b>Auto-resume</b>: when you re-open the GUI it loads the seed, "
            "then overlays annotations from the previous quick-save "
            "(<code>--out</code>) if it exists. Pass <code>--no-resume</code> "
            "to start fresh from the seed."
        )
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setMinimumWidth(280)
        dock = QtWidgets.QDockWidget("Help", self.window)
        dock.setWidget(label)
        self.window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    # -------------------- frame nav --------------------

    def _render_current(self):
        if not self.group_keys:
            self.status.showMessage("No groups in this seed.")
            return
        # Stash the current group's edits before swapping it out.
        self._stash_current_group()
        key = self.group_keys[self.cur_idx]
        if key in self._edited_groups:
            self.cur_group = self._edited_groups[key]
            # Images were dropped to save RAM when the group was last
            # stashed; reload from sources. Keypoints are preserved.
            self._reload_images_into_group(self.cur_group)
        else:
            reviewed_views = {
                view
                for view in VIEWS
                if review_view_key(key[0], key[1], view)
                in self.reviewed_view_keys
            }
            self.cur_group = build_frame_group(
                key,
                self.groups_src[key],
                medaka_data_dir=self.medaka_data_dir,
                cal=self.cal,
                candidate_src_lfs=self.candidate_groups.get(key, {}),
                reviewed_views=reviewed_views,
                accepted_slot_keys=self.accepted_slot_keys,
                rejected_slot_keys=self.rejected_slot_keys,
            )
        for view, pane in self.panes.items():
            pane.set_image(self.cur_group.images[view])
            pane.view.setVisible(
                not (
                    self.instance_review
                    or self.missing_instance_review
                )
                or view == self._active_review_view()
            )
        if self.instance_review:
            self.fit_active_instance()
        elif self.missing_instance_review:
            self.fit_active_missing_view()
        else:
            self._sync_views()
        self._draw_keypoints()
        if self.instance_review:
            active_fish_idx = self._active_fish_idx()
            active_view = self._active_review_view()
            self.frame_label.setText(
                f"[instance {self.instance_review_idx + 1}/"
                f"{len(self.instance_review_sequence)}] "
                f"{key[0]} @ frame {key[1]} | "
                f"{active_view} candidate {int(active_fish_idx) + 1}"
            )
        elif self.missing_instance_review:
            active_view = self._active_review_view()
            self.frame_label.setText(
                f"[missing fish {self.missing_instance_idx + 1}/"
                f"{len(self.missing_instance_sequence)}] "
                f"{key[0]} @ frame {key[1]} | {active_view}"
            )
            self._arm_missing_instance_add()
        else:
            self.frame_label.setText(
                f"[{self.cur_idx + 1}/{len(self.group_keys)}] "
                f"{key[0]} @ frame {key[1]}"
            )
        self._update_status()

    def _stash_current_group(self) -> None:
        """Cache the current group (without its image data) so the user's
        edits survive navigating to other frames."""
        if self.cur_group is None:
            return
        key = (self.cur_group.experiment, self.cur_group.frame_idx)
        # Drop the images to keep memory bounded — they'll be reloaded if
        # the user comes back.
        self.cur_group.images = {}
        self._edited_groups[key] = self.cur_group

    def _reload_images_into_group(self, grp: FrameGroup) -> None:
        for view in VIEWS:
            lf = grp.src_lfs.get(view)
            img = load_image_for_view(
                lf,
                medaka_data_dir=self.medaka_data_dir,
                experiment=grp.experiment,
                view=view,
                frame_idx=grp.frame_idx,
            )
            if img is None:
                img = np.full((1200, 1200, 3), 30, dtype=np.uint8)
            grp.images[view] = img

    def next_frame(self):
        if self.instance_review_sequence:
            self.instance_review_idx = min(
                self.instance_review_idx + 1,
                len(self.instance_review_sequence) - 1,
            )
            key, _view, _fish_idx = self.instance_review_sequence[
                self.instance_review_idx
            ]
            self.cur_idx = self._group_index_by_key[key]
        elif self.missing_instance_sequence:
            self.missing_instance_idx = min(
                self.missing_instance_idx + 1,
                len(self.missing_instance_sequence) - 1,
            )
            key, _view = self.missing_instance_sequence[
                self.missing_instance_idx
            ]
            self.cur_idx = self._group_index_by_key[key]
        else:
            self.cur_idx = min(
                self.cur_idx + 1, len(self.group_keys) - 1
            )
        self.cancel_add()
        self._render_current()

    def prev_frame(self):
        if self.instance_review_sequence:
            self.instance_review_idx = max(
                self.instance_review_idx - 1, 0
            )
            key, _view, _fish_idx = self.instance_review_sequence[
                self.instance_review_idx
            ]
            self.cur_idx = self._group_index_by_key[key]
        elif self.missing_instance_sequence:
            self.missing_instance_idx = max(
                self.missing_instance_idx - 1, 0
            )
            key, _view = self.missing_instance_sequence[
                self.missing_instance_idx
            ]
            self.cur_idx = self._group_index_by_key[key]
        else:
            self.cur_idx = max(self.cur_idx - 1, 0)
        self.cancel_add()
        self._render_current()

    # -------------------- focus / zoom sync --------------------

    def _sync_views(self, *, skip_view: Optional[str] = None):
        focus_proj = self.cal.compute_imCoords_from_XYZ(
            self.focus_xyz.reshape(1, 3)
        )[0]  # shape (3, 2)
        for i, view in enumerate(VIEWS):
            if view == skip_view:
                continue
            self.panes[view].set_focus(
                (float(focus_proj[i, 0]), float(focus_proj[i, 1])),
                self.zoom,
            )

    def bump_zoom(self, factor):
        self.zoom = float(np.clip(self.zoom * factor, 0.05, 8.0))
        if self.instance_review or self.missing_instance_review:
            view = self._active_review_view()
            pane = self.panes.get(view) if view is not None else None
            if pane is not None:
                # Preserve the exact center chosen by the instance fitter.
                # A calibration projection would move this single-view
                # review away from the active detection.
                pane.set_focus(pane.current_center(), self.zoom)
                self._update_status()
                return
        self._sync_views()
        self._update_status()

    def fit_zoom(self):
        if self.missing_instance_review:
            self.fit_active_missing_view()
            self._update_status()
            return
        # Pick a zoom that fits ~250 image-px in the smallest pane.
        if not self.panes:
            return
        avail = []
        for pane in self.panes.values():
            r = pane.view.viewport().size()
            avail.append(min(r.width(), r.height()))
        target_disp_px = max(200, min(avail) - 40)
        self.zoom = float(target_disp_px / 1200.0)
        self._sync_views()
        self._update_status()

    def _mark_view_reviewed(self, view: str) -> None:
        if self.cur_group is None:
            return
        self.cur_group.reviewed_views.add(view)
        self.reviewed_view_keys.add(
            review_view_key(
                self.cur_group.experiment,
                self.cur_group.frame_idx,
                view,
            )
        )

    def fit_active_instance(self) -> None:
        """Tightly center the sole visible pane on the active candidate."""
        if self.cur_group is None or not self.panes:
            return
        fish_idx = self._active_fish_idx()
        view = self._active_review_view()
        if fish_idx is None or view is None:
            return
        points_by_view = fish_points_by_view(
            self.cur_group.fish,
            [fish_idx],
        )
        points = points_by_view[view]
        if not points:
            return

        points_array = np.asarray(points, dtype=float)
        center_array = (
            np.min(points_array, axis=0) + np.max(points_array, axis=0)
        ) / 2.0
        center = (float(center_array[0]), float(center_array[1]))
        focus_by_view = {view: center}
        viewport_sizes = {
            view: (
                self.panes[view].view.viewport().size().width(),
                self.panes[view].view.viewport().size().height(),
            )
        }
        zoom = instance_fit_zoom(
            {view: points},
            focus_by_view,
            viewport_sizes,
            display_margin_px=48.0,
            minimum_scene_span_px=120.0,
        )
        if zoom is not None:
            self.zoom = zoom
        self.panes[view].set_focus(center, self.zoom)

    def fit_instances(self):
        """Center and zoom the synchronized panes around current instances."""
        if self.cur_group is None or not self.panes:
            return
        if self.instance_review:
            self.fit_active_instance()
            self._update_status()
            return
        if self.missing_instance_review:
            self.fit_active_missing_view()
            self._update_status()
            return

        fish_indices = range(len(self.cur_group.fish))
        points_by_view = fish_points_by_view(
            self.cur_group.fish,
            fish_indices,
        )

        populated_views = [view for view in VIEWS if points_by_view[view]]
        if not populated_views:
            self.status.showMessage("No instances on this frame to fit.", 3000)
            return

        # A bounding-box midpoint is preferable to the arithmetic mean here:
        # it gives equal room to the outermost fish. Triangulating those view
        # centers produces one physically consistent focus for the synced panes.
        centers: Dict[str, Tuple[float, float]] = {}
        for view in populated_views:
            points = np.asarray(points_by_view[view], dtype=float)
            center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
            centers[view] = (float(center[0]), float(center[1]))

        new_focus: Optional[np.ndarray] = None
        if len(populated_views) >= 2:
            observations = np.full((3, 2), np.nan, dtype=float)
            for view, center in centers.items():
                observations[VIEW_INDEX[view]] = center
            try:
                candidate = np.asarray(self.cal.triangulate(observations), dtype=float)
                if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                    new_focus = candidate
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass

        if new_focus is None:
            # One-view frames (or calibrations without two-view support) can
            # still center accurately in the most populated pane.
            anchor = max(populated_views, key=lambda view: len(points_by_view[view]))
            new_focus = back_project_view_click(
                self.cal, anchor, centers[anchor], self.focus_xyz
            )
        self.focus_xyz = new_focus

        projected = np.asarray(
            self.cal.compute_imCoords_from_XYZ(self.focus_xyz.reshape(1, 3))[0],
            dtype=float,
        )
        focus_by_view = {
            view: (
                float(projected[VIEW_INDEX[view], 0]),
                float(projected[VIEW_INDEX[view], 1]),
            )
            for view in VIEWS
        }
        viewport_sizes = {
            view: (
                pane.view.viewport().size().width(),
                pane.view.viewport().size().height(),
            )
            for view, pane in self.panes.items()
        }
        zoom = instance_fit_zoom(
            points_by_view, focus_by_view, viewport_sizes
        )
        if zoom is None:
            return
        self.zoom = zoom
        self._sync_views()
        self._update_status()

    def fit_active_missing_view(self) -> None:
        """Fit the whole deficient camera frame so the missing fish is visible."""
        if self.cur_group is None:
            return
        view = self._active_review_view()
        pane = self.panes.get(view) if view is not None else None
        image = self.cur_group.images.get(view) if view is not None else None
        if pane is None or image is None or image.ndim < 2:
            return
        image_h, image_w = image.shape[:2]
        viewport = pane.view.viewport().size()
        usable_w = max(1.0, float(viewport.width()) - 24.0)
        usable_h = max(1.0, float(viewport.height()) - 24.0)
        self.zoom = float(
            np.clip(
                min(usable_w / image_w, usable_h / image_h),
                0.05,
                8.0,
            )
        )
        pane.set_focus((image_w / 2.0, image_h / 2.0), self.zoom)

    def _arm_missing_instance_add(self) -> None:
        """Force mouth→tail add mode for the active deficient view."""
        if not self.missing_instance_review:
            return
        self.add_btn.blockSignals(True)
        self.add_btn.setChecked(True)
        self.add_btn.blockSignals(False)
        self.add_mode = "place_mouth"
        self.pending_view = None
        self.pending_mouth = None
        self.pending_fish_idx = None

    # -------------------- click + pan handlers --------------------

    def on_pane_click(self, view: str, x_img: float, y_img: float):
        if self.add_mode == "place_missing_mouth":
            self._place_missing_node(view, "mouth", x_img, y_img)
        elif self.add_mode == "place_missing_tail":
            self._place_missing_node(view, "tail", x_img, y_img)
        elif self.add_mode == "place_mouth":
            self._place_mouth(view, x_img, y_img)
        elif self.add_mode == "place_tail":
            self._place_tail(view, x_img, y_img)
        else:
            # Default: clicking sets the focus point (back-projected).
            self.focus_xyz = back_project_view_click(
                self.cal, view, (x_img, y_img), self.focus_xyz
            )
            self._sync_views()
            self._update_status()

    def on_pane_wheel(
        self,
        view: str,
        center_x: float,
        center_y: float,
        zoom: float,
    ):
        """Synchronize other views after cursor-anchored local zoom."""
        self.focus_xyz = back_project_view_click(
            self.cal, view, (center_x, center_y), self.focus_xyz
        )
        self.zoom = float(zoom)
        # The active pane has already applied an exact cursor-preserving Qt
        # transform. Keep it untouched and project its new center to peers.
        self._sync_views(skip_view=view)
        self._update_status()

    def on_pane_pan(self, view: str, center_x: float, center_y: float):
        """Synchronize other views after a direct mouse-drag translation."""
        self.focus_xyz = back_project_view_click(
            self.cal, view, (center_x, center_y), self.focus_xyz
        )
        self._sync_views(skip_view=view)

    # -------------------- instance add --------------------

    def toggle_add_mode(self, on: bool):
        if on:
            active_fish_idx = self._active_fish_idx()
            active_view = self._active_review_view()
            if (
                active_fish_idx is not None
                and active_view is not None
                and self.cur_group is not None
            ):
                keypoints = self.cur_group.fish[active_fish_idx]
                mouth_missing = keypoints.mouth.get(active_view) is None
                tail_missing = keypoints.tail.get(active_view) is None
                if mouth_missing != tail_missing:
                    missing_node = (
                        "mouth" if mouth_missing else "tail"
                    )
                    self.add_mode = f"place_missing_{missing_node}"
                    self.pending_view = active_view
                    self.pending_mouth = None
                    self.pending_fish_idx = active_fish_idx
                    self.status.showMessage(
                        f"COMPLETE ACTIVE CANDIDATE: click the missing "
                        f"{missing_node.upper()} in {active_view}.",
                        0,
                    )
                    return
            self.add_mode = "place_mouth"
            self.pending_view = None
            self.pending_mouth = None
            self.pending_fish_idx = None
            self.status.showMessage(
                "ADD INSTANCE: click on a view to place MOUTH.", 0
            )
        else:
            self.cancel_add()

    def cancel_add(self):
        self.add_mode = None
        self.pending_view = None
        self.pending_mouth = None
        self.pending_fish_idx = None
        if self.add_btn.isChecked():
            self.add_btn.blockSignals(True)
            self.add_btn.setChecked(False)
            self.add_btn.blockSignals(False)
        self._update_status()

    def _place_mouth(self, view: str, x: float, y: float):
        active_view = self._active_review_view()
        if (
            self.missing_instance_review
            and view != active_view
        ):
            self.status.showMessage(
                f"Add the missing fish in {active_view}.", 4000
            )
            return
        self.pending_view = view
        self.pending_mouth = (x, y)
        self.add_mode = "place_tail"
        self.status.showMessage(
            f"Mouth placed in {view}. Click in the SAME view to place TAIL "
            f"(or Esc to cancel).",
            0,
        )
        self._draw_keypoints()  # show the in-progress mouth marker

    def _place_tail(self, view: str, x: float, y: float):
        if view != self.pending_view:
            self.status.showMessage(
                f"Tail must be in the same view as mouth ({self.pending_view}). "
                "Click again or Esc to cancel.",
                4000,
            )
            return
        mouth = self.pending_mouth
        tail = (x, y)

        # Candidates = fish that have annotations in some other view AND
        # don't yet have an annotation in `view`. Already-claimed IDs in
        # this view are filtered out so we never offer a duplicate match.
        candidates = self._unassigned_fish_for_view(view)

        fish_idx: Optional[int] = None
        decision_msg: str = ""

        if self.missing_instance_review:
            # Identity is not part of this task: choose the first fish slot
            # that is empty in the active view and avoid a calibration popup.
            # The benchmark finalizer strips tracks after per-view QC.
            fish_idx = candidates[0][0] if candidates else None
            decision_msg = (
                f"assigned to open ID_{fish_idx + 1}"
                if fish_idx is not None
                else "new fish"
            )
        elif self.instance_review:
            fish_idx = self._active_fish_idx()
            decision_msg = (
                f"assigned to active candidate {int(fish_idx) + 1}"
            )
        elif len(candidates) == 1:
            # Definitive: there's only one fish this could plausibly be.
            fish_idx = candidates[0][0]
            decision_msg = f"auto-assigned ID_{fish_idx + 1} (only candidate left)"
        elif len(candidates) >= 2:
            # Try calibration auto-match restricted to the unassigned set.
            candidate_idxs = [c[0] for c in candidates]
            cal_match = self._suggest_fish_id(
                view, mouth, tail, restrict_to=candidate_idxs
            )
            if cal_match is not None:
                fish_idx = cal_match
                decision_msg = f"auto-assigned ID_{fish_idx + 1} (calibration match)"
            else:
                picked = self._ask_user_for_id(candidates, view)
                if picked == "cancel":
                    self.add_mode = "place_mouth"
                    self.pending_view = None
                    self.pending_mouth = None
                    self._draw_keypoints()
                    self.status.showMessage("Cancelled — pending mouth dropped.", 3000)
                    return
                fish_idx = picked  # int or None (None = new fish)
                decision_msg = (
                    f"user picked ID_{fish_idx + 1}"
                    if fish_idx is not None else "user picked New fish"
                )
        # candidates is empty -> first instance for this view in this group;
        # we'll fall through to "new fish" below.

        if fish_idx is None:
            empty_slot = first_empty_fish_slot(self.cur_group.fish)
            if empty_slot is None:
                fish_idx = len(self.cur_group.fish)
                self.cur_group.fish.append(FishKeypoints(fish_id=fish_idx))
            else:
                fish_idx = empty_slot
            decision_msg = decision_msg or f"new fish ID_{fish_idx + 1}"
        else:
            while fish_idx >= len(self.cur_group.fish):
                self.cur_group.fish.append(
                    FishKeypoints(fish_id=len(self.cur_group.fish))
                )

        self.cur_group.fish[fish_idx].mouth[view] = mouth
        self.cur_group.fish[fish_idx].tail[view] = tail
        self.cur_group.fish[fish_idx].candidate_by_view[view] = False
        self.cur_group.fish[fish_idx].accepted_by_view[view] = True
        self._mark_fish_view_decision(fish_idx, view, "accepted")
        self.last_edited_view = view
        self.status.showMessage(
            f"Instance placed in {view}: {decision_msg}.", 4000
        )
        if self.missing_instance_review:
            self._finish_missing_instance()
            return
        # Reset state machine but stay in add mode for next instance.
        self.add_mode = "place_mouth"
        self.pending_view = None
        self.pending_mouth = None
        self._draw_keypoints()
        self._update_status()

    def _unassigned_fish_for_view(self, view: str) -> List[Tuple[int, str]]:
        """Return ``(fish_idx, hint_text)`` for each fish that:
        - has at least one keypoint in some view other than ``view``, AND
        - does NOT yet have any keypoint in ``view``.

        These are the IDs the next instance in ``view`` could plausibly
        merge with. Already-claimed IDs in ``view`` are excluded so we never
        offer a duplicate.
        """
        out: List[Tuple[int, str]] = []
        for fish_idx, fk in enumerate(self.cur_group.fish):
            already_in_view = (
                fk.mouth[view] is not None or fk.tail[view] is not None
            )
            if already_in_view:
                continue
            other_view_bits: List[str] = []
            for v in VIEWS:
                if v == view:
                    continue
                if fk.mouth[v] is not None or fk.tail[v] is not None:
                    other_view_bits.append(v.split("_")[0])
            if other_view_bits:
                out.append((fish_idx, "+".join(other_view_bits)))
        return out

    def _ask_user_for_id(
        self, candidates: List[Tuple[int, str]], view: str
    ):
        """Show a dialog with one button per existing fish + a "New" button +
        Cancel. Returns the selected fish_idx (int), ``None`` for new, or the
        sentinel string ``"cancel"`` if dismissed.
        """
        QtCore, QtGui, QtWidgets = _qt()
        dlg = QtWidgets.QDialog(self.window)
        dlg.setWindowTitle(f"Assign ID for new {view} instance")
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(QtWidgets.QLabel(
            f"Calibration can't auto-match yet (need 2 other views populated "
            f"for some fish). Pick which existing fish this new instance is:"
        ))
        result: Dict[str, object] = {"value": "cancel"}
        for fish_idx, hint in candidates:
            color = self._color_for(fish_idx)
            btn = QtWidgets.QPushButton(f"ID_{fish_idx + 1}   (already has {hint})")
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgb({color[0]},{color[1]},{color[2]}); "
                f"color: white; font-weight: bold; padding: 6px 12px; }}"
            )
            def _pick(_checked=False, idx=fish_idx, r=result, d=dlg):
                r["value"] = idx
                d.accept()
            btn.clicked.connect(_pick)
            layout.addWidget(btn)
        new_btn = QtWidgets.QPushButton("New fish (next available ID)")
        def _new(_checked=False, r=result, d=dlg):
            r["value"] = None
            d.accept()
        new_btn.clicked.connect(_new)
        layout.addWidget(new_btn)
        cancel_btn = QtWidgets.QPushButton("Cancel — drop the pending instance")
        def _cancel(_checked=False, d=dlg):
            d.reject()
        cancel_btn.clicked.connect(_cancel)
        layout.addWidget(cancel_btn)
        if dlg.exec_() == QtWidgets.QDialog.Rejected:
            return "cancel"
        return result["value"]

    def _suggest_fish_id(
        self,
        view: str,
        mouth: Tuple[float, float],
        tail: Tuple[float, float],
        *,
        restrict_to: Optional[List[int]] = None,
    ) -> Optional[int]:
        """Rank existing fish for the new instance in ``view``. Uses
        whichever calibration scoring is available:

        - 2 other views populated -> predict our view via pairwise model and
          score by pixel distance (most reliable).
        - 1 other view populated  -> hypothesize new annotation belongs to
          this fish, predict the unobserved 3rd view via pairwise model,
          triangulate the full triplet, reproject through XYZ_to_imCoords,
          score by reprojection error on the 2 *observed* views.

        Threshold for accepting an auto-match is per-mode: 80 px for the
        2-view case, 60 px (sum of two views) for the 1-view case.
        """
        new_centroid = (
            (mouth[0] + tail[0]) / 2.0,
            (mouth[1] + tail[1]) / 2.0,
        )
        best_idx: Optional[int] = None
        best_err = float("inf")
        best_mode: Optional[str] = None
        candidate_iter = (
            restrict_to if restrict_to is not None
            else range(len(self.cur_group.fish))
        )
        for fish_idx in candidate_iter:
            if fish_idx >= len(self.cur_group.fish):
                continue
            fk = self.cur_group.fish[fish_idx]
            cents: Dict[str, Optional[Tuple[float, float]]] = {}
            for v in VIEWS:
                if v == view:
                    cents[v] = None
                else:
                    cents[v] = _instance_centroid(fk.mouth[v], fk.tail[v])
            other_views = [v for v in VIEWS if v != view and cents[v] is not None]

            if len(other_views) >= 2:
                # 2-other-views path.
                pred = predict_missing_view(self.cal, cents)
                pred_xy = pred[view]
                if pred_xy is None:
                    continue
                d = float(np.hypot(
                    pred_xy[0] - new_centroid[0], pred_xy[1] - new_centroid[1]
                ))
                mode = "calibration2"
            elif len(other_views) == 1:
                # 1-other-view path: synthesise the 3rd view, triangulate,
                # reproject, score by error on the 2 observed views.
                other = other_views[0]
                third = [v for v in VIEWS if v != view and v != other][0]
                hypo = {view: new_centroid, other: cents[other], third: None}
                pred = predict_missing_view(self.cal, hypo)
                third_xy = pred[third]
                if third_xy is None:
                    continue
                triplet = np.array(
                    [
                        list(new_centroid) if v == view
                        else (list(cents[other]) if v == other else list(third_xy))
                        for v in VIEWS
                    ],
                    dtype=float,
                )
                try:
                    xyz = self.cal.compute_XYZ_from_imageCoord_triplet(triplet)
                except Exception:
                    continue
                if (
                    not np.all(np.isfinite(xyz))
                    or np.any(xyz < -30) or np.any(xyz > 230)
                ):
                    continue
                try:
                    reproj = reproject_xyz(self.cal, xyz)
                except RuntimeError:
                    continue
                err_v = float(np.hypot(
                    new_centroid[0] - reproj[view][0],
                    new_centroid[1] - reproj[view][1],
                ))
                err_w = float(np.hypot(
                    cents[other][0] - reproj[other][0],
                    cents[other][1] - reproj[other][1],
                ))
                d = err_v + err_w
                mode = "calibration1"
            else:
                continue

            if d < best_err:
                best_err = d
                best_idx = fish_idx
                best_mode = mode

        if best_idx is None:
            return None
        threshold = 80.0 if best_mode == "calibration2" else 60.0
        if best_err <= threshold:
            return best_idx
        return None

    # -------------------- drawing --------------------

    def _color_for(self, fish_id: int):
        return FISH_COLORS.get(fish_id, FALLBACK_COLOR)

    def _draw_keypoints(self):
        for pane in self.panes.values():
            pane.clear_keypoints()
        if self.cur_group is None:
            return

        marker_radius = keypoint_radius(self.keypoint_size)
        halo_radius = marker_radius + 6.0
        active_fish_idx = self._active_fish_idx()
        fish_indices = (
            [active_fish_idx]
            if active_fish_idx is not None
            else range(len(self.cur_group.fish))
        )

        # Pre-compute reprojection errors per fish/node for halo coloring.
        # Only when all 3 views are populated.
        errors_by_fnv: Dict[Tuple[int, str, str], float] = {}
        if not self.instance_review:
            for fish_idx in fish_indices:
                if fish_idx >= len(self.cur_group.fish):
                    continue
                fk = self.cur_group.fish[fish_idx]
                for node, kp_dict in (
                    ("mouth", fk.mouth),
                    ("tail", fk.tail),
                ):
                    if not all(kp_dict[v] is not None for v in VIEWS):
                        continue
                    pts = {v: kp_dict[v] for v in VIEWS}
                    xyz = triangulate_xyz(self.cal, pts)
                    if not np.all(np.isfinite(xyz)):
                        continue
                    try:
                        reproj = reproject_xyz(self.cal, xyz)
                    except RuntimeError:
                        continue
                    for v in VIEWS:
                        err = float(np.hypot(
                            pts[v][0] - reproj[v][0],
                            pts[v][1] - reproj[v][1],
                        ))
                        errors_by_fnv[(fish_idx, node, v)] = err

        for view, pane in self.panes.items():
            if (
                (
                    self.instance_review
                    or self.missing_instance_review
                )
                and view != self._active_review_view()
            ):
                continue
            for fish_idx in fish_indices:
                if fish_idx >= len(self.cur_group.fish):
                    continue
                fk = self.cur_group.fish[fish_idx]
                candidate = (
                    fk.candidate_by_view.get(view, False)
                    and not fk.accepted_by_view.get(view, False)
                )
                color = (
                    (190, 190, 190)
                    if candidate
                    else self._color_for(fish_idx)
                )
                m = fk.mouth.get(view)
                t = fk.tail.get(view)
                # Halos go behind the keypoints (lower z).
                err_m = errors_by_fnv.get((fish_idx, "mouth", view))
                if err_m is not None and m is not None:
                    pane.add_halo(
                        m[0], m[1],
                        color_rgb=_error_color(err_m),
                        radius=halo_radius,
                    )
                err_t = errors_by_fnv.get((fish_idx, "tail", view))
                if err_t is not None and t is not None:
                    pane.add_halo(
                        t[0], t[1],
                        color_rgb=_error_color(err_t),
                        radius=halo_radius,
                    )

                if m is not None:
                    m_vis = fk.mouth_visible.get(view, True)
                    # Hidden keypoints: no fill (just an outline) so they
                    # don't visually compete with visible ones.
                    pane.add_draggable_keypoint(
                        m[0], m[1],
                        color_rgb=color, radius=marker_radius,
                        fill_alpha=110 if m_vis else 0,
                        view_name=view, fish_idx=fish_idx, node="mouth",
                        on_moved=self._on_keypoint_moved,
                        on_release=self._on_keypoint_drag_release,
                        on_delete=self._on_keypoint_delete,
                        on_toggle_visible=self._on_keypoint_toggle_visible,
                        on_request_swap=self._on_keypoint_request_swap,
                        candidate=candidate,
                        label=(
                            (
                                f"CAND ID_{fish_idx + 1}"
                                if candidate
                                else f"ID_{fish_idx + 1}"
                            )
                            + ("" if m_vis else " (hidden)")
                        ),
                    )
                if t is not None:
                    t_vis = fk.tail_visible.get(view, True)
                    # Tail is normally hollow; for hidden tail draw a small
                    # central dot to differentiate from visible-tail's empty
                    # circle.
                    pane.add_draggable_keypoint(
                        t[0], t[1],
                        color_rgb=color, radius=marker_radius,
                        fill_alpha=0 if t_vis else 110,
                        view_name=view, fish_idx=fish_idx, node="tail",
                        on_moved=self._on_keypoint_moved,
                        on_release=self._on_keypoint_drag_release,
                        on_delete=self._on_keypoint_delete,
                        on_toggle_visible=self._on_keypoint_toggle_visible,
                        on_request_swap=self._on_keypoint_request_swap,
                        candidate=candidate,
                        label=("(hidden)" if not t_vis else None),
                    )
                if m is not None and t is not None:
                    # Arrow tail -> mouth: a mouth/tail flip reverses it, so
                    # swaps jump out at close-contact crossings.
                    pane.add_arrow(
                        t,
                        m,
                        color_rgb=color,
                        width=1.5,
                        candidate=candidate,
                    )
            # Pending mouth marker if mid-add.
            if (
                self.add_mode == "place_tail"
                and self.pending_view == view
                and self.pending_mouth is not None
            ):
                pane.add_keypoint(
                    self.pending_mouth[0], self.pending_mouth[1],
                    color_rgb=(255, 255, 255),
                    radius=marker_radius + 2.0,
                    filled=False,
                    label="(pending)",
                )

    def _on_keypoint_moved(self, view: str, fish_idx: int, node: str, x: float, y: float):
        """Called continuously while the user drags an existing keypoint.
        We only update the FrameGroup's data + status bar; we do NOT redraw
        (re-creating the items would break the in-flight drag).
        """
        if self.cur_group is None or fish_idx >= len(self.cur_group.fish):
            return
        fk = self.cur_group.fish[fish_idx]
        target = fk.mouth if node == "mouth" else fk.tail
        target[view] = (float(x), float(y))
        accept_fish_view(fk, view)
        self._mark_fish_view_decision(fish_idx, view, "accepted")
        self.last_edited_view = view
        self._update_status()

    def _on_keypoint_drag_release(self, view: str, fish_idx: int, node: str):
        """Called once after a drag ends. Full redraw so error halos refresh."""
        self._draw_keypoints()

    def _on_keypoint_request_swap(self, view: str, fish_idx: int):
        """Show a dialog asking which other fish to swap IDs with. The swap
        applies across all views — useful when calibration auto-match (or
        the round-2/3 predictions) paired the wrong existing fish.
        """
        if self.cur_group is None:
            return
        # Other fish that have at least one keypoint anywhere — swapping with
        # an empty fish slot just renames it, which is also fine but rarely
        # what's wanted, so we list them last.
        candidates: List[Tuple[int, str, bool]] = []
        active_view = self._active_review_view()
        for i, fk in enumerate(self.cur_group.fish):
            if i == fish_idx:
                continue
            if active_view is not None:
                has_data = (
                    fk.mouth.get(active_view) is not None
                    or fk.tail.get(active_view) is not None
                )
                hint = (
                    f"({active_view})" if has_data else "(empty slot)"
                )
            else:
                has_data = any(
                    fk.mouth.get(v) is not None
                    or fk.tail.get(v) is not None
                    for v in VIEWS
                )
                n_views = sum(
                    1 for v in VIEWS
                    if fk.mouth.get(v) is not None
                    or fk.tail.get(v) is not None
                )
                hint = (
                    f"({n_views}/3 views)"
                    if has_data
                    else "(empty slot)"
                )
            candidates.append((i, hint, has_data))
        if not candidates:
            self.status.showMessage("No other fish to swap with.", 3000)
            return
        candidates.sort(key=lambda c: (-(1 if c[2] else 0), c[0]))

        QtCore, QtGui, QtWidgets = _qt()
        dlg = QtWidgets.QDialog(self.window)
        dlg.setWindowTitle(f"Swap ID_{fish_idx + 1} with…")
        layout = QtWidgets.QVBoxLayout(dlg)
        scope_text = (
            f"in {active_view}"
            if active_view is not None
            else "across all 3 views"
        )
        layout.addWidget(QtWidgets.QLabel(
            f"Swap <b>ID_{fish_idx + 1}</b> ↔ which fish? "
            f"(applies {scope_text})"
        ))
        result: Dict[str, Optional[int]] = {"other": None}
        for other_idx, hint, _has in candidates:
            color = self._color_for(other_idx)
            btn = QtWidgets.QPushButton(f"ID_{other_idx + 1}  {hint}")
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgb({color[0]},{color[1]},{color[2]}); "
                f"color: white; font-weight: bold; padding: 6px 12px; }}"
            )
            def _pick(_checked=False, idx=other_idx, r=result, d=dlg):
                r["other"] = idx
                d.accept()
            btn.clicked.connect(_pick)
            layout.addWidget(btn)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(dlg.reject)
        layout.addWidget(cancel)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        other = result["other"]
        if other is None:
            return
        # Swap the keypoint dicts between the two fish slots. Keep the
        # ``fish_id`` field aligned with the slot index so colors stay
        # tied to position (and downstream code is happy).
        a = self.cur_group.fish[fish_idx]
        b = self.cur_group.fish[other]
        if active_view is not None:
            swap_views = [active_view]
            for values_a, values_b in (
                (a.mouth, b.mouth),
                (a.tail, b.tail),
                (a.mouth_visible, b.mouth_visible),
                (a.tail_visible, b.tail_visible),
                (a.candidate_by_view, b.candidate_by_view),
                (a.accepted_by_view, b.accepted_by_view),
            ):
                values_a[active_view], values_b[active_view] = (
                    values_b[active_view],
                    values_a[active_view],
                )
        else:
            swap_views = list(VIEWS)
            a.mouth, b.mouth = b.mouth, a.mouth
            a.tail, b.tail = b.tail, a.tail
            a.mouth_visible, b.mouth_visible = (
                b.mouth_visible,
                a.mouth_visible,
            )
            a.tail_visible, b.tail_visible = (
                b.tail_visible,
                a.tail_visible,
            )
            a.candidate_by_view, b.candidate_by_view = (
                b.candidate_by_view,
                a.candidate_by_view,
            )
            a.accepted_by_view, b.accepted_by_view = (
                b.accepted_by_view,
                a.accepted_by_view,
            )
        for swapped_idx, keypoints in ((fish_idx, a), (other, b)):
            for candidate_view in swap_views:
                if (
                    keypoints.mouth.get(candidate_view) is not None
                    or keypoints.tail.get(candidate_view) is not None
                ):
                    accept_fish_view(keypoints, candidate_view)
                    self._mark_fish_view_decision(
                        swapped_idx,
                        candidate_view,
                        "accepted",
                    )
        self._draw_keypoints()
        self._update_status()

    def _finish_missing_instance(self) -> None:
        """Persist one targeted addition before moving to the next target."""
        if not self._safe_save(
            "targeted missing-fish autosave",
            self.out_path,
            embed=False,
        ):
            # Stay on the same target and re-arm only after the user has seen
            # the save failure. This prevents an unsaved click from advancing.
            self._arm_missing_instance_add()
            return
        if self.missing_instance_idx + 1 >= len(
            self.missing_instance_sequence
        ):
            self.cancel_add()
            self.status.showMessage(
                "All targeted missing fish were saved. You can close this "
                "window.",
                0,
            )
            return
        self.missing_instance_idx += 1
        key, _view = self.missing_instance_sequence[
            self.missing_instance_idx
        ]
        self.cur_idx = self._group_index_by_key[key]
        self._render_current()

    def _place_missing_node(
        self,
        view: str,
        node: str,
        x: float,
        y: float,
    ) -> None:
        """Complete the sole missing node on the active review item."""
        fish_idx = self._active_fish_idx()
        active_view = self._active_review_view()
        if (
            self.cur_group is None
            or fish_idx is None
            or view != active_view
            or node not in NODE_NAMES
        ):
            self.status.showMessage(
                f"Click the active {active_view} pane to complete this item.",
                4000,
            )
            return
        keypoints = self.cur_group.fish[fish_idx]
        target = keypoints.mouth if node == "mouth" else keypoints.tail
        target[view] = (float(x), float(y))
        visibility = (
            keypoints.mouth_visible
            if node == "mouth"
            else keypoints.tail_visible
        )
        visibility[view] = True
        keypoints.candidate_by_view[view] = False
        keypoints.accepted_by_view[view] = True
        self._mark_fish_view_decision(
            fish_idx,
            view,
            "accepted",
        )
        self.last_edited_view = view
        self.cancel_add()
        self._draw_keypoints()
        self._update_status()
        self.status.showMessage(
            f"Completed and accepted candidate {fish_idx + 1}: "
            f"placed {node} in {view}.",
            5000,
        )
        self.status.showMessage(
            f"Swapped ID_{fish_idx + 1} ↔ ID_{other + 1} {scope_text}.",
            5000,
        )

    def _on_keypoint_toggle_visible(self, view: str, fish_idx: int, node: str):
        """Flip the visibility flag for a single (fish, view, node)."""
        if self.cur_group is None or fish_idx >= len(self.cur_group.fish):
            return
        fk = self.cur_group.fish[fish_idx]
        vis_dict = fk.mouth_visible if node == "mouth" else fk.tail_visible
        vis_dict[view] = not vis_dict.get(view, True)
        accept_fish_view(fk, view)
        self._mark_fish_view_decision(fish_idx, view, "accepted")
        new_state = "hidden" if not vis_dict[view] else "visible"
        self._draw_keypoints()
        self.status.showMessage(
            f"ID_{fish_idx + 1} {node} in {view} marked as {new_state}.", 4000
        )

    def _on_keypoint_delete(self, view: str, fish_idx: int, node: Optional[str], scope: str):
        """Right-click delete from a keypoint context menu.

        scope:
            "keypoint" - drop only this view's mouth or tail (use ``node``).
            "instance" - drop both mouth and tail in this view.
            "fish"     - drop the fish entirely (all 3 views).
        """
        if self.cur_group is None or fish_idx >= len(self.cur_group.fish):
            return
        fk = self.cur_group.fish[fish_idx]
        if scope == "keypoint" and node in NODE_NAMES:
            (fk.mouth if node == "mouth" else fk.tail)[view] = None
            populated = (
                fk.mouth.get(view) is not None
                or fk.tail.get(view) is not None
            )
            fk.accepted_by_view[view] = populated
            self._mark_fish_view_decision(
                fish_idx,
                view,
                "accepted" if populated else "rejected",
            )
            self.last_edited_view = view
            msg = f"Deleted ID_{fish_idx + 1} {node} in {view}."
        elif scope == "instance":
            fk.mouth[view] = None
            fk.tail[view] = None
            fk.candidate_by_view[view] = False
            fk.accepted_by_view[view] = False
            self._mark_fish_view_decision(
                fish_idx, view, "rejected"
            )
            self.last_edited_view = view
            msg = f"Deleted ID_{fish_idx + 1} instance in {view}."
        elif scope == "fish":
            for v in VIEWS:
                populated = (
                    fk.mouth.get(v) is not None
                    or fk.tail.get(v) is not None
                )
                fk.mouth[v] = None
                fk.tail[v] = None
                fk.candidate_by_view[v] = False
                fk.accepted_by_view[v] = False
                if populated:
                    self._mark_fish_view_decision(
                        fish_idx, v, "rejected"
                    )
            msg = f"Deleted ID_{fish_idx + 1} from all views."
        else:
            return
        self._draw_keypoints()
        self._update_status()
        self.status.showMessage(msg, 4000)

    # -------------------- fill / errors --------------------

    def fill_missing_views(self):
        """Re-match all fish across views via calibration, anchored on the
        most recently edited view, then re-predict the keypoints in the
        other views from those (now consistent) matches.

        Steps:
          1. Pick the *anchor* view: the view the user touched last (added,
             dragged, or deleted). Falls back to whichever view has the most
             detections if the user hasn't edited yet.
          2. The anchor view's instances become the new fish slots — slot
             order is preserved so colors don't shuffle for the user.
          3. For each *other* view, build a pairwise reprojection-error cost
             matrix between anchor's instances and the view's existing
             instances. Hungarian assign to find the best pairing. The
             paired instances are slotted into the matching anchor slot.
             Unmatched anchor slots stay empty in that view.
          4. Predict any still-empty (mouth, tail) entries via the pairwise
             calibration model. Those predicted points are flagged
             ``visible=True`` (they're a geometric best-guess; the user can
             toggle to hidden if they're an inference rather than visible).
        """
        if getattr(self, "instance_review", False):
            self.status.showMessage(
                "F is disabled in instance review because it would change "
                "hidden fish slots.",
                5000,
            )
            return
        if not self.allow_calibration_fill:
            self.status.showMessage(
                "F is disabled: independent benchmark coordinates must be "
                "placed manually.",
                5000,
            )
            return
        if self.cur_group is None:
            self.status.showMessage("F: no group loaded.", 3000)
            return

        # Step 1: pick anchor view.
        det_per_view = {v: self._detections_in_view(v) for v in VIEWS}
        nonempty_views = [v for v in VIEWS if det_per_view[v]]
        if not nonempty_views:
            self.status.showMessage("F: no annotations yet.", 3000)
            return
        if (
            self.last_edited_view in nonempty_views
        ):
            anchor = self.last_edited_view
        else:
            anchor = max(nonempty_views, key=lambda v: len(det_per_view[v]))

        # Step 2: anchor's instances seed the new fish slots.
        new_fish: List[FishKeypoints] = []
        for det in det_per_view[anchor]:
            fk = FishKeypoints()
            fk.mouth[anchor] = det["mouth"]
            fk.tail[anchor] = det["tail"]
            fk.mouth_visible[anchor] = det["mouth_vis"]
            fk.tail_visible[anchor] = det["tail_vis"]
            new_fish.append(fk)

        # Step 3: match each other view's instances to anchor slots via
        # pairwise reprojection cost + Hungarian assignment.
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:  # pragma: no cover - scipy is in deps
            self.status.showMessage("F: scipy required for re-matching.", 4000)
            return

        rematch_summary: List[str] = []
        for view in VIEWS:
            if view == anchor or not det_per_view[view]:
                continue
            n_anc = len(new_fish)
            n_oth = len(det_per_view[view])
            if n_anc == 0 or n_oth == 0:
                continue
            cost = np.full((n_anc, n_oth), 1e6, dtype=float)
            for i, fk_a in enumerate(new_fish):
                cent_a = self._fish_centroid_in_view(fk_a, anchor)
                if cent_a is None:
                    continue
                for j, det_o in enumerate(det_per_view[view]):
                    cent_o = self._inst_centroid(det_o)
                    if cent_o is None:
                        continue
                    err = self._pair_reprojection_error(
                        anchor, cent_a, view, cent_o,
                    )
                    if err is not None and np.isfinite(err):
                        cost[i, j] = err
            # Hungarian (rectangular OK).
            try:
                rows, cols = linear_sum_assignment(cost)
            except ValueError:
                continue
            paired = 0
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                det_o = det_per_view[view][int(j)]
                new_fish[int(i)].mouth[view] = det_o["mouth"]
                new_fish[int(i)].tail[view] = det_o["tail"]
                new_fish[int(i)].mouth_visible[view] = det_o["mouth_vis"]
                new_fish[int(i)].tail_visible[view] = det_o["tail_vis"]
                paired += 1
            unmatched_oth = n_oth - paired
            rematch_summary.append(
                f"{view}: paired {paired}/{n_oth}"
                + (f" (+{unmatched_oth} unmatched)" if unmatched_oth else "")
            )

        # Step 4: predict any still-empty views via pairwise model.
        n_filled = 0
        for fk in new_fish:
            pred_m = predict_missing_view(self.cal, fk.mouth)
            pred_t = predict_missing_view(self.cal, fk.tail)
            for v in VIEWS:
                if fk.mouth[v] is None and pred_m[v] is not None:
                    fk.mouth[v] = pred_m[v]
                    fk.mouth_visible[v] = True
                    n_filled += 1
                if fk.tail[v] is None and pred_t[v] is not None:
                    fk.tail[v] = pred_t[v]
                    fk.tail_visible[v] = True
                    n_filled += 1

        self.cur_group.fish = new_fish
        for fish_idx, keypoints in enumerate(self.cur_group.fish):
            for populated_view in VIEWS:
                if (
                    keypoints.mouth.get(populated_view) is not None
                    or keypoints.tail.get(populated_view) is not None
                ):
                    keypoints.accepted_by_view[populated_view] = True
                    self._mark_fish_view_decision(
                        fish_idx,
                        populated_view,
                        "accepted",
                    )
        self._draw_keypoints()
        self._update_status()
        msg = (
            f"F: anchored on {anchor}, {len(new_fish)} fish slot(s); "
            + ("; ".join(rematch_summary) if rematch_summary else "no other views to match")
            + f"; filled {n_filled} keypoint(s) via calibration."
        )
        self.status.showMessage(msg, 7000)

    # -------------------- helpers used by F (re-match + fill) --------------------

    def _detections_in_view(self, view: str) -> List[Dict]:
        """List of (mouth, tail, *_vis) dicts for every fish that has at
        least one keypoint in this view."""
        out: List[Dict] = []
        if self.cur_group is None:
            return out
        for fk in self.cur_group.fish:
            m = fk.mouth.get(view)
            t = fk.tail.get(view)
            if m is None and t is None:
                continue
            out.append({
                "mouth": m,
                "tail": t,
                "mouth_vis": fk.mouth_visible.get(view, True),
                "tail_vis": fk.tail_visible.get(view, True),
            })
        return out

    def _fish_centroid_in_view(self, fk: FishKeypoints, view: str):
        m = fk.mouth.get(view)
        t = fk.tail.get(view)
        return _instance_centroid(m, t)

    def _inst_centroid(self, det: Dict):
        return _instance_centroid(det.get("mouth"), det.get("tail"))

    def _pair_reprojection_error(
        self,
        view_a: str,
        cent_a: Tuple[float, float],
        view_b: str,
        cent_b: Tuple[float, float],
    ) -> Optional[float]:
        """Pairwise reprojection consistency: predict the 3rd view from
        (cent_a, cent_b), triangulate the full triplet, reproject through
        XYZ_to_imCoords, return the sum of pixel errors on the 2 observed
        views. Lower = better (a, b) correspondence.
        """
        third = [v for v in VIEWS if v != view_a and v != view_b][0]
        cents = {view_a: cent_a, view_b: cent_b, third: None}
        pred = predict_missing_view(self.cal, cents)
        if pred[third] is None:
            return None
        triplet = np.array(
            [
                list(cent_a) if v == view_a
                else (list(cent_b) if v == view_b else list(pred[third]))
                for v in VIEWS
            ],
            dtype=float,
        )
        try:
            xyz = self.cal.compute_XYZ_from_imageCoord_triplet(triplet)
        except Exception:
            return None
        if (
            not np.all(np.isfinite(xyz))
            or np.any(xyz < -50) or np.any(xyz > 250)
        ):
            return None
        try:
            reproj = reproject_xyz(self.cal, xyz)
        except RuntimeError:
            return None
        err_a = float(np.hypot(
            cent_a[0] - reproj[view_a][0], cent_a[1] - reproj[view_a][1],
        ))
        err_b = float(np.hypot(
            cent_b[0] - reproj[view_b][0], cent_b[1] - reproj[view_b][1],
        ))
        return err_a + err_b

    def _update_status(self):
        if self.cur_group is None:
            return
        bits = []
        active_fish_idx = self._active_fish_idx()
        active_view = self._active_review_view()
        if active_fish_idx is not None and active_view is not None:
            slot_key = review_slot_key(
                self.cur_group.experiment,
                self.cur_group.frame_idx,
                active_view,
                active_fish_idx,
            )
            if slot_key in self.accepted_slot_keys:
                decision = "accepted"
            elif slot_key in self.rejected_slot_keys:
                decision = "rejected"
            else:
                decision = "undecided"
            bits.append(
                f"instance {self.instance_review_idx + 1}/"
                f"{len(self.instance_review_sequence)}"
            )
            bits.append(
                f"frame={self.cur_group.frame_idx} "
                f"view={active_view} "
                f"candidate={active_fish_idx + 1} [{decision}]"
            )
        elif self.missing_instance_review and active_view is not None:
            bits.append(
                f"missing fish {self.missing_instance_idx + 1}/"
                f"{len(self.missing_instance_sequence)}"
            )
            bits.append(
                f"frame={self.cur_group.frame_idx} view={active_view}"
            )
        if not (
            self.instance_review
            or self.missing_instance_review
        ):
            bits.append(f"focus_xyz=({self.focus_xyz[0]:.0f},"
                        f"{self.focus_xyz[1]:.0f},"
                        f"{self.focus_xyz[2]:.0f})mm")
        bits.append(f"zoom={self.zoom:.2f}x")
        fish_items = (
            []
            if self.instance_review
            else list(enumerate(self.cur_group.fish))
        )
        for fish_idx, fk in fish_items:
            for node, kp in (("mouth", fk.mouth), ("tail", fk.tail)):
                full = [v for v in VIEWS if kp[v] is not None]
                if len(full) == 3:
                    pts = {v: kp[v] for v in VIEWS}
                    xyz = triangulate_xyz(self.cal, pts)
                    if np.all(np.isfinite(xyz)):
                        try:
                            reproj = reproject_xyz(self.cal, xyz)
                            errs = {
                                v: float(np.hypot(
                                    pts[v][0] - reproj[v][0],
                                    pts[v][1] - reproj[v][1],
                                )) for v in VIEWS
                            }
                            err_str = "/".join(f"{errs[v]:.1f}" for v in VIEWS)
                            bits.append(
                                f"f{fish_idx + 1}.{node} err={err_str}px"
                            )
                        except RuntimeError:
                            pass
        if self.add_mode == "place_mouth":
            bits.append("[ADD: click MOUTH]")
        elif self.add_mode == "place_tail":
            bits.append(f"[ADD: click TAIL in {self.pending_view}]")
        elif self.add_mode == "place_missing_mouth":
            bits.append(f"[COMPLETE: click MOUTH in {self.pending_view}]")
        elif self.add_mode == "place_missing_tail":
            bits.append(f"[COMPLETE: click TAIL in {self.pending_view}]")
        self.status.showMessage("  |  ".join(bits))

    # -------------------- save --------------------

    def save(self):
        """Quick save — references the seed's existing embedded frames.

        Finishes in seconds. The saved pkg.slp is *not* portable on its
        own: it references video objects from the seed pkg.slp, so a
        colleague would still need that seed file (and matching frames)
        to open it. Use :meth:`save_full` to write a self-contained copy.
        """
        self._safe_save("quick", self.out_path, embed=False)

    def save_full(self):
        """Full save — re-embed every user-labeled frame so the file is
        portable on its own. Slower (re-reads frames from sources) but the
        output can be copied to another machine without depending on the
        seed pkg.slp or the original video files.

        Output path is the quick-save path with ``_full`` appended, so the
        quick-save file isn't clobbered.
        """
        full_path = full_save_path(self.out_path)
        # ``embed="user"`` only embeds frames of LFs that have at least
        # one Instance; with the expanded seed many of our LFs are
        # placeholder-empty for missing-view slots the user hasn't
        # touched yet, and "user" leaves them unembedded — that breaks
        # portability. ``embed="all"`` embeds every LF's frame so the
        # output is fully self-contained.
        self._safe_save("full", full_path, embed="all")

    def _safe_save(self, label: str, out_path: Path, *, embed):
        """Run :meth:`_do_save` with a try/except that:
        - Surfaces a short error in the status bar.
        - Writes the full traceback to ``<out_path.parent>/save_error.log``
          so the user can paste it back when something goes wrong.
        """
        try:
            self._do_save(out_path=out_path, embed=embed)
        except Exception as exc:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            print(tb)
            log_path = out_path.parent / "save_error.log"
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as fh:
                    fh.write(f"Save ({label}) failed for {out_path}\n\n")
                    fh.write(tb)
            except Exception:
                log_path = None
            extra = f"  See {log_path}" if log_path else ""
            self.status.showMessage(
                f"Save ({label}) FAILED: {exc!s}.{extra}", 30000
            )
            return False
        return True

    def _do_save(self, *, out_path: Path, embed):
        import sleap_io as sio

        # Make sure the in-memory current group is in the edit cache too.
        if self.cur_group is not None:
            key = (self.cur_group.experiment, self.cur_group.frame_idx)
            self._edited_groups[key] = self.cur_group

        skel = self.labels.skeletons[0]

        # Map source-video path -> Video object so we can dedupe and reuse
        # video objects across LFs. Use the *source* path (the Z:\ video
        # behind the embedded frames), since the embedded-package path is
        # identical for all 75 videos in the seed.
        video_by_src: Dict[str, "sio.Video"] = {}
        for v in self.labels.videos:
            src = _video_source_path(v) or str(v.filename)
            video_by_src.setdefault(src, v)

        def _resolve_video(view: str, exp: str, src_lf):
            """Return a Video object for (exp, view), reusing existing ones."""
            if src_lf is not None and src_lf.video is not None:
                src = _video_source_path(src_lf.video)
                if src:
                    return video_by_src.setdefault(src, src_lf.video)
                return src_lf.video
            # No source LF: fall back to medaka_data_dir's video file.
            on_disk = (
                self.medaka_data_dir / exp / view / VIEW_TO_VIDEO[view]
            )
            cached = video_by_src.get(str(on_disk))
            if cached is not None:
                return cached
            try:
                vid = sio.load_video(str(on_disk))
                video_by_src[str(on_disk)] = vid
                return vid
            except Exception:
                return None

        # Index original LFs so we can drop only the ones the user actually
        # touched and keep everything else as-is.
        lf_index: Dict[Tuple[str, int, str], "sio.LabeledFrame"] = {}
        for lf in self.labels.labeled_frames:
            src_path = _video_source_path(lf.video)
            view = classify_view(src_path)
            exp = experiment_from_video(src_path)
            if view is None or exp is None:
                continue
            lf_index[(exp, int(lf.frame_idx), view)] = lf

        replaced_keys: set[Tuple[str, int, str]] = set()
        new_lfs: List["sio.LabeledFrame"] = []
        track_by_name = {
            str(getattr(track, "name", "") or ""): track
            for track in list(getattr(self.labels, "tracks", []) or [])
            if str(getattr(track, "name", "") or "")
        }

        # Iterate *all* edited groups, not just the current one.
        for (exp, frame_idx), grp in self._edited_groups.items():
            for view in VIEWS:
                slot_keys = [
                    review_slot_key(
                        exp,
                        frame_idx,
                        view,
                        fish_idx,
                    )
                    for fish_idx in range(len(grp.fish))
                ]
                has_slot_decision = any(
                    key in self.accepted_slot_keys
                    or key in self.rejected_slot_keys
                    for key in slot_keys
                )
                if self.instance_review:
                    should_replace = (
                        view in grp.reviewed_views
                        or has_slot_decision
                    )
                else:
                    should_replace = (
                        not self.candidate_review_enabled
                        or view in grp.reviewed_views
                    )
                if not should_replace:
                    # Merely visiting a candidate view is not acceptance.
                    # Leave the authoritative input LF untouched.
                    continue
                src_lf = grp.src_lfs.get(view)
                target_video = _resolve_video(view, exp, src_lf)
                if target_video is None:
                    continue
                instances = []
                for fish_idx, fk in enumerate(grp.fish):
                    if not fish_view_is_saveable(fk, view):
                        continue
                    m = fk.mouth.get(view)
                    t = fk.tail.get(view)
                    m_vis = bool(fk.mouth_visible.get(view, True)) if m is not None else False
                    t_vis = bool(fk.tail_visible.get(view, True)) if t is not None else False
                    pts = np.array(
                        [
                            [m[0] if m is not None else np.nan,
                             m[1] if m is not None else np.nan,
                             1.0 if m_vis else 0.0],
                            [t[0] if t is not None else np.nan,
                             t[1] if t is not None else np.nan,
                             1.0 if t_vis else 0.0],
                        ],
                        dtype=float,
                    )
                    track = None
                    slot_key = review_slot_key(
                        exp,
                        frame_idx,
                        view,
                        fish_idx,
                    )
                    if (
                        self.instance_review
                        and slot_key in self.accepted_slot_keys
                    ):
                        track_name = review_slot_track_name(
                            exp,
                            frame_idx,
                            view,
                            fish_idx,
                        )
                        track = track_by_name.get(track_name)
                        if track is None:
                            track = sio.Track(name=track_name)
                            track_by_name[track_name] = track
                    instances.append(
                        sio.Instance.from_numpy(
                            points_data=pts,
                            skeleton=skel,
                            track=track,
                        )
                    )
                if not instances:
                    # User removed every fish from this view of this group;
                    # in candidate-review mode preserve an explicit empty LF
                    # so resume overlays the rejection instead of resurrecting
                    # the candidate.
                    replaced_keys.add((exp, frame_idx, view))
                    if self.candidate_review_enabled:
                        new_lfs.append(
                            sio.LabeledFrame(
                                video=target_video,
                                frame_idx=frame_idx,
                                instances=[],
                            )
                        )
                    continue
                new_lfs.append(
                    sio.LabeledFrame(
                        video=target_video,
                        frame_idx=frame_idx,
                        instances=instances,
                    )
                )
                replaced_keys.add((exp, frame_idx, view))

        kept_lfs = [
            lf for k, lf in lf_index.items() if k not in replaced_keys
        ]
        merged_lfs = kept_lfs + new_lfs
        suggestions = [
            sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
            for lf in merged_lfs
        ]
        # Collect *every* video object actually referenced by an LF in the
        # merged set. The earlier dedupe-by-source-path logic could drop a
        # second Video object that an LF still pointed at, leaving the
        # saved file with stale video indices that crash on reload.
        merged_videos = list(
            {id(lf.video): lf.video for lf in merged_lfs}.values()
        )
        merged_tracks = []
        seen_track_ids = set()
        for track in (
            list(getattr(self.labels, "tracks", []) or [])
            + [
                getattr(instance, "track", None)
                for lf in merged_lfs
                for instance in lf.instances
            ]
        ):
            if track is None or id(track) in seen_track_ids:
                continue
            seen_track_ids.add(id(track))
            merged_tracks.append(track)
        provenance = copy_labels_provenance(self.labels)
        if self.candidate_review_enabled:
            provenance[PREDICTION_REVIEW_PROVENANCE_KEY] = {
                "schema_version": PREDICTION_REVIEW_SCHEMA,
                "candidate_artifact": deepcopy(self.candidate_artifact),
                "reviewed_views": sorted(self.reviewed_view_keys),
                "reviewed_view_count": len(self.reviewed_view_keys),
                "instance_review": {
                    "enabled": bool(self.instance_review),
                    "unit": "experiment-frame-view-ranked-candidate",
                    "accepted_slots": sorted(self.accepted_slot_keys),
                    "rejected_slots": sorted(self.rejected_slot_keys),
                    "accepted_slot_count": len(self.accepted_slot_keys),
                    "rejected_slot_count": len(self.rejected_slot_keys),
                    "track_name_prefix": REVIEW_SLOT_TRACK_PREFIX,
                },
                "safety_policy": {
                    "unreviewed_candidates_saved_as_user_instances": False,
                    "candidate_edit_implies_acceptance": True,
                    "reviewed_rejection_saved_as_explicit_empty_frame": True,
                    "instance_decisions_do_not_review_sibling_slots": True,
                },
            }
        merged = sio.Labels(
            skeletons=[skel],
            videos=merged_videos,
            labeled_frames=merged_lfs,
            tracks=merged_tracks,
            suggestions=suggestions,
            provenance=provenance,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        QtWidgets_app = _qt()[2].QApplication.instance()
        kind = "full (embedded)" if embed else "quick (no embed)"
        self.status.showMessage(
            f"Saving {kind} -> {out_path} …", 0
        )
        if QtWidgets_app is not None:
            QtWidgets_app.processEvents()
        sio.save_slp(merged, str(out_path), embed=embed)
        self.status.showMessage(
            f"Saved {kind}: {len(merged_lfs)} LFs "
            f"({len(self._edited_groups)} edited groups) -> {out_path}",
            10000,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input", type=Path, required=True)
    ap.add_argument(
        "--candidate-slp",
        type=Path,
        default=None,
        help=(
            "Prediction SLP to display as a review-only overlay. --in stays "
            "authoritative; untouched candidates are never saved as user "
            "instances."
        ),
    )
    ap.add_argument(
        "--instance-review",
        action="store_true",
        help=(
            "Review one per-view prediction instance at a time. Requires "
            "--candidate-slp; Left/Right navigate items and A/R decide only "
            "the active candidate."
        ),
    )
    ap.add_argument(
        "--missing-instance-target",
        action="append",
        default=[],
        metavar="EXPERIMENT:FRAME:VIEW",
        help=(
            "Show only this deficient camera frame, force mouth->tail add "
            "mode, save immediately after commit, and advance. Repeat once "
            "per deficient view."
        ),
    )
    ap.add_argument(
        "--expected-instances",
        type=int,
        default=3,
        help=(
            "Required per-view instance count for --missing-instance-target "
            "(default: 3). Saved targets are skipped on resume."
        ),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--paths-config", default=None)
    ap.add_argument("--medaka-data-dir", default=None)
    ap.add_argument("--calibration-model-dir", default=None)
    ap.add_argument(
        "--calibration-backend",
        choices=("auto", "refractive", "polynomial"),
        default="auto",
        help=(
            "Calibration geometry. 'auto' prefers a complete refractive JSON "
            "bundle and otherwise uses the historical polynomial joblibs."
        ),
    )
    ap.add_argument("--min-views", type=int, default=1)
    ap.add_argument(
        "--resume", type=Path, default=None,
        help=(
            "pkg.slp to merge on top of the seed at startup so you "
            "continue from a previous session. Defaults to ``--out`` "
            "when that file already exists. Pass ``--no-resume`` to "
            "ignore an existing --out file."
        ),
    )
    ap.add_argument(
        "--no-resume", action="store_true",
        help="Don't auto-resume from --out even if it exists.",
    )
    ap.add_argument(
        "--independent",
        action="store_true",
        help=(
            "Disable calibration coordinate fill. Cross-view view navigation "
            "and ID suggestions remain available, but all mouth/tail "
            "coordinates must be clicked manually."
        ),
    )
    args = ap.parse_args(argv)

    if args.instance_review and args.candidate_slp is None:
        ap.error("--instance-review requires --candidate-slp.")
    if args.instance_review and args.missing_instance_target:
        ap.error(
            "--instance-review and --missing-instance-target are exclusive."
        )
    if args.candidate_slp is not None and args.missing_instance_target:
        ap.error(
            "--candidate-slp and --missing-instance-target are exclusive."
        )
    if args.expected_instances < 1:
        ap.error("--expected-instances must be at least 1.")
    try:
        missing_instance_targets = [
            parse_missing_instance_target(value)
            for value in args.missing_instance_target
        ]
    except ValueError as exc:
        ap.error(str(exc))
    if not args.input.is_file():
        ap.error(f"--in does not exist: {args.input}")
    if (
        missing_instance_targets
        and args.input.resolve() == args.out.resolve()
    ):
        ap.error(
            "--out must not overwrite --in in targeted missing-fish mode."
        )
    if args.candidate_slp is not None:
        if not args.candidate_slp.is_file():
            ap.error(
                f"--candidate-slp does not exist: {args.candidate_slp}"
            )
        if args.candidate_slp.resolve() == args.input.resolve():
            ap.error("--candidate-slp must be distinct from authoritative --in.")
        if args.candidate_slp.resolve() == args.out.resolve():
            ap.error("--out must not overwrite --candidate-slp.")

    local = load_local_paths(args.paths_config)
    medaka_data_dir = choose_path(
        args.medaka_data_dir, local.medaka_data_dir if local else None
    )
    calib_dir = choose_path(
        args.calibration_model_dir, local.calibration_model_dir if local else None
    )
    if medaka_data_dir is None:
        ap.error("medaka_data_dir is required.")
    if calib_dir is None:
        ap.error("calibration_model_dir is required.")

    import sleap_io as sio

    print(f"Loading {args.input} ...")
    labels = sio.load_slp(str(args.input))
    print(f"  {len(labels.labeled_frames)} LFs, {len(labels.videos)} videos.")

    # Auto-resume: if --out already exists (from a previous quick save),
    # overlay its LFs onto the seed so the user picks up where they left off.
    resume_path: Optional[Path] = args.resume
    if resume_path is None and not args.no_resume and args.out.exists():
        resume_path = args.out
    if resume_path is not None and resume_path.exists():
        print(f"Resuming from {resume_path} (overlaying onto seed)...")
        try:
            resume = sio.load_slp(str(resume_path))
            labels = _merge_resume_labels(labels, resume)
            print(f"  merged: {len(labels.labeled_frames)} LFs after overlay.")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: failed to load resume file: {exc!s}")
            print(f"  continuing with the seed only.")

    candidate_groups = {}
    candidate_artifact = {}
    if args.candidate_slp is not None:
        candidate_artifact = candidate_artifact_record(args.candidate_slp)
        persisted_artifact = candidate_artifact_from_provenance(labels)
        mismatch = candidate_artifact_mismatch(
            persisted_artifact,
            candidate_artifact,
        )
        if mismatch:
            ap.error(
                f"{mismatch}. Use the original --candidate-slp, choose a "
                "different --out, or pass --no-resume to start a new review."
            )

        print(f"Loading candidate overlay {args.candidate_slp} ...")
        candidate_labels = sio.load_slp(
            str(args.candidate_slp),
            open_videos=False,
        )
        all_candidate_groups = build_groups(candidate_labels)
        authoritative_keys = set(build_groups(labels))
        candidate_groups = {
            key: frames_by_view
            for key, frames_by_view in all_candidate_groups.items()
            if key in authoritative_keys
        }
        matched_views = sum(
            len(frames_by_view)
            for frames_by_view in candidate_groups.values()
        )
        ignored_groups = len(all_candidate_groups) - len(candidate_groups)
        print(
            f"  {matched_views} candidate view-frame(s) matched "
            f"{len(candidate_groups)} authoritative group(s); "
            f"{ignored_groups} non-authoritative group(s) ignored."
        )

    cal = Calibration(str(calib_dir), backend=args.calibration_backend)
    if not cal.capabilities.get("projection", False):
        ap.error("calibration must support 3D-to-2D projection for synced views.")
    if not cal.capabilities.get("two_view_triangulation", False):
        ap.error("calibration must support two-view triangulation for missing-view assistance.")
    print(
        f"  calibration: {cal.backend_name} backend; "
        f"score unit={cal.score_unit}; model={cal.model_folder}"
    )

    groups = build_groups(labels)
    keys = filter_groups(groups, min_views=args.min_views)
    print(f"  {len(keys)} groups (min_views={args.min_views}).")
    missing_instance_sequence = []
    if missing_instance_targets:
        try:
            missing_instance_sequence = build_missing_instance_sequence(
                groups,
                missing_instance_targets,
                expected_instances=args.expected_instances,
            )
        except ValueError as exc:
            ap.error(str(exc))
        if not missing_instance_sequence:
            ap.error(
                "All --missing-instance-target views already have the "
                f"expected {args.expected_instances} instances."
            )
        print(
            f"  {len(missing_instance_sequence)} missing instance(s) queued "
            f"across {len(missing_instance_targets)} requested view(s)."
        )
    if args.instance_review:
        accepted_slot_keys, rejected_slot_keys = (
            review_slot_decisions_from_provenance(labels)
        )
        review_sequence = build_instance_review_sequence(
            keys,
            candidate_groups,
            groups,
            accepted_slot_keys,
            rejected_slot_keys,
        )
        if not review_sequence:
            ap.error(
                "--instance-review found no incomplete authoritative "
                "per-view candidate items."
            )
        print(
            f"  {len(review_sequence)} per-view candidate instance(s) "
            "queued for review."
        )

    # Launch Qt.
    QtCore, QtGui, QtWidgets = _qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    win = MultiviewWindow(
        labels=labels,
        cal=cal,
        group_keys=keys,
        groups_src=groups,
        medaka_data_dir=Path(medaka_data_dir),
        out_path=args.out,
        allow_calibration_fill=not args.independent,
        candidate_groups=candidate_groups,
        candidate_artifact=candidate_artifact,
        instance_review=args.instance_review,
        missing_instance_sequence=missing_instance_sequence,
    )
    win.window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
