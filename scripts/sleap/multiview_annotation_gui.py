from __future__ import annotations

"""
Multi-view annotation GUI (napari) for the medaka mating pipeline.

What it does
------------
- Loads a round-3 ``annotation_seed.pkg.slp`` (the 999-frame seed combining
  500 user-corrected round-2 frames + 499 round-3 bad-frame predictions).
- Groups labelled frames by ``(experiment, frame_idx)``. By default, every
  group is presented for editing (``--min-views 1``), so round-2 single-view
  picks get expanded to all three views just like round-3 picks. Pass
  ``--min-views 2`` to skip round-2 frames and focus on the round-3 picks
  that already have multiple views.
- For each group, ensures all 3 views are present by loading any missing-view
  video frame from ``medaka_data_dir`` so the user can annotate it from
  scratch with calibration help.
- Displays the 3 views tiled side-by-side in napari with the existing
  predicted/user keypoints overlaid as an editable Points layer.
- On every points-layer edit, recomputes:
    * Cross-view predictions (when a fish has 2 views with both keypoints,
      predict the 3rd via the calibration's pairwise models).
    * Reprojection error (when all 3 views have both keypoints, triangulate
      and report the per-view RMS pixel error).
  Both are shown in the status bar.

How to annotate
---------------
The Points layer starts in **add** mode. Each time you click on a view
the GUI inserts a keypoint with the current "active fish" + "active node"
attached and figures out which view you clicked from the x position. Hold
the keys below to set the active fish/node before clicking. The status
bar at the bottom of napari always shows the current target.

Switch the Points layer to **select** mode (toolbar 4th icon, or press
``2`` while focus is in the Points layer's toolbar) to drag existing
points or delete them with ``Delete``.

Keyboard shortcuts (work anywhere in the viewer)
------------------------------------------------
- ``[`` / ``]``     previous / next (exp, frame_idx) group
- ``q`` / ``w`` / ``e``   active fish = ID_1 / ID_2 / ID_3
- ``a``             active node = mouth
- ``s``             active node = tail
- ``f``             fill missing views via calibration pairwise model
- ``Ctrl+S``        save merged annotations to ``--out``

Per-fish color: ID_1 blue, ID_2 orange, ID_3 green.
Mouth = filled disc; tail = ring (open circle).

CLI
---
    python -m scripts.sleap.multiview_annotation_gui \\
        --in "C:/Users/.../annotation_seed_round3.pkg.slp" \\
        --calibration-model-dir "<from paths.local.toml>" \\
        --paths-config configs/paths.local.toml \\
        --out "C:/Users/.../annotation_seed_round3_v2.pkg.slp"
"""

import argparse
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration


VIEWS: Tuple[str, ...] = ("top_cam", "side1_cam", "side2_cam")
VIEW_INDEX = {v: i for i, v in enumerate(VIEWS)}
VIEW_TO_VIDEO = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}

# Per-fish RGB colors aligned with the Rerun viewer scheme.
FISH_COLORS_RGB: Dict[int, Tuple[float, float, float]] = {
    0: (0.122, 0.467, 0.706),   # ID_1 blue
    1: (1.000, 0.498, 0.055),   # ID_2 orange
    2: (0.173, 0.627, 0.173),   # ID_3 green
}
PRED_COLOR_RGB = (0.7, 0.7, 0.7)  # auto-predicted points

NODE_NAMES = ("mouth", "tail")
NUM_FISH = 3


def copy_labels_provenance(labels) -> dict:
    """Return an independent copy of package provenance for save operations."""
    return deepcopy(dict(getattr(labels, "provenance", {}) or {}))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FishKeypoints:
    """Mouth/tail per view for one fish at one frame.

    ``mouth[view]`` is None if the keypoint is missing in that view, else a
    ``(x, y)`` tuple. Same for ``tail``.
    """
    mouth: Dict[str, Optional[Tuple[float, float]]] = field(
        default_factory=lambda: {v: None for v in VIEWS}
    )
    tail: Dict[str, Optional[Tuple[float, float]]] = field(
        default_factory=lambda: {v: None for v in VIEWS}
    )
    predicted_mouth: Dict[str, Optional[Tuple[float, float]]] = field(
        default_factory=lambda: {v: None for v in VIEWS}
    )
    predicted_tail: Dict[str, Optional[Tuple[float, float]]] = field(
        default_factory=lambda: {v: None for v in VIEWS}
    )


@dataclass
class FrameGroup:
    """One annotation unit: 3 views at a single ``(experiment, frame_idx)``."""
    experiment: str
    frame_idx: int
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    fish: List[FishKeypoints] = field(default_factory=list)
    # Source LF per view so we can rebuild user-instance objects on save.
    src_lfs: Dict[str, "sio.LabeledFrame"] = field(default_factory=dict)


def clone_fish_keypoints(
    fish: List[FishKeypoints],
) -> List[FishKeypoints]:
    """Copy editable keypoints without retaining the large frame images."""
    return [
        FishKeypoints(
            mouth=dict(item.mouth),
            tail=dict(item.tail),
            predicted_mouth=dict(item.predicted_mouth),
            predicted_tail=dict(item.predicted_tail),
        )
        for item in fish
    ]


def fish_have_annotations(fish: List[FishKeypoints]) -> bool:
    """Return whether any real mouth/tail point is present."""
    return any(
        point is not None
        for item in fish
        for by_view in (item.mouth, item.tail)
        for point in by_view.values()
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _video_source_path(video) -> str:
    """Return the original on-disk path of a sleap-io Video, even when frames
    are embedded in a pkg.slp (in which case ``video.filename`` points at the
    package file rather than the source video)."""
    src = getattr(video, "source_video", None)
    if src is not None and getattr(src, "filename", None):
        return str(src.filename)
    backend = getattr(video, "backend", None)
    src_fname = getattr(backend, "source_filename", None) if backend is not None else None
    if src_fname:
        return str(src_fname)
    return str(getattr(video, "filename", "")) or ""


def classify_view(fname: str) -> Optional[str]:
    s = str(fname).lower().replace("\\", "/")
    for tag in ("top_cam", "side1_cam", "side2_cam"):
        if f"/{tag}/" in s or s.endswith(f"/{tag}.slp") or s.endswith(f"\\{tag}.slp"):
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


def instance_xy_pair(inst) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
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


def instance_centroid(inst) -> Optional[Tuple[float, float]]:
    pts = inst.numpy()
    mask = np.isfinite(pts).all(axis=1)
    if not mask.any():
        return None
    return tuple(pts[mask].mean(axis=0).tolist())


# ---------------------------------------------------------------------------
# Group construction
# ---------------------------------------------------------------------------

def build_groups(labels) -> Dict[Tuple[str, int], Dict[str, "sio.LabeledFrame"]]:
    """Group labelled frames by ``(experiment, frame_idx)``.

    Each value maps view name -> LabeledFrame (or absent if the view has no
    LF for that frame_idx).
    """
    groups: Dict[Tuple[str, int], Dict[str, "sio.LabeledFrame"]] = defaultdict(dict)
    for lf in labels.labeled_frames:
        fname = _video_source_path(lf.video)
        view = classify_view(fname)
        exp = experiment_from_video(fname)
        if view is None or exp is None:
            continue
        groups[(exp, int(lf.frame_idx))][view] = lf
    return dict(groups)


def filter_groups(
    groups: Dict[Tuple[str, int], Dict[str, "sio.LabeledFrame"]],
    *,
    min_views: int = 1,
) -> List[Tuple[str, int]]:
    """Keep only group keys with at least ``min_views`` views populated.

    Default ``min_views=1`` means every (exp, frame_idx) with any annotation
    is exposed to the GUI — so round-2 single-view picks get expanded to all
    three views with calibration help. Pass ``min_views=2`` to focus only on
    round-3 (already multi-view) picks.
    """
    keys = [k for k, gv in groups.items() if len(gv) >= min_views]
    keys.sort()
    return keys


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

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


def load_embedded_or_disk_image(
    lf,
    *,
    medaka_data_dir: Path,
    experiment: str,
    view: str,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """Best-effort: read from the LF's embedded video first; fall back to
    disk. ``frame_idx`` is required (LFs may be ``None`` for views the seed
    didn't include, in which case we still want to load the frame from
    ``medaka_data_dir``)."""
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
    # Fall back to medaka_data_dir for views with no LF or whose embedded
    # frame couldn't be decoded.
    video_path = medaka_data_dir / experiment / view / VIEW_TO_VIDEO[view]
    return load_video_frame(video_path, int(frame_idx))


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def predict_missing_view(
    cal: Calibration,
    pts_by_view: Dict[str, Optional[Tuple[float, float]]],
) -> Dict[str, Optional[Tuple[float, float]]]:
    """For each view that's None, predict its 2D position from the other two
    via the calibration's pairwise models. Returns predictions per view; views
    with valid input are echoed back as their input value.
    """
    out: Dict[str, Optional[Tuple[float, float]]] = {v: pts_by_view[v] for v in VIEWS}
    have = [v for v in VIEWS if pts_by_view[v] is not None]
    if len(have) != 2:
        return out
    missing = [v for v in VIEWS if v not in have][0]
    p_top = pts_by_view["top_cam"]
    p_s1 = pts_by_view["side1_cam"]
    p_s2 = pts_by_view["side2_cam"]
    try:
        if missing == "top_cam":
            arr = cal.compute_XY_imcoords_from_XZ_YZ(np.array(p_s1), np.array(p_s2))
        elif missing == "side1_cam":
            arr = cal.compute_XZ_imcoords_from_XY_YZ(np.array(p_top), np.array(p_s2))
        else:  # side2_cam
            arr = cal.compute_YZ_imcoords_from_XZ_XY(np.array(p_s1), np.array(p_top))
    except RuntimeError:
        return out
    out[missing] = (float(arr[0]), float(arr[1]))
    return out


def triangulate_and_reproject(
    cal: Calibration,
    pts_by_view: Dict[str, Tuple[float, float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Triangulate the (top, side1, side2) image-point triplet to XYZ and
    reproject to all three views. Returns ``(xyz, per_view_pixel_error)``.
    """
    triplet = np.array(
        [
            list(pts_by_view["top_cam"]),
            list(pts_by_view["side1_cam"]),
            list(pts_by_view["side2_cam"]),
        ],
        dtype=float,
    )
    xyz = cal.compute_XYZ_from_imageCoord_triplet(triplet)
    errors: Dict[str, float] = {v: float("nan") for v in VIEWS}
    if cal._model_xyz2im is not None and np.all(np.isfinite(xyz)):
        reproj = cal._model_xyz2im.predict(xyz.reshape(1, -1))
        if reproj.ndim == 2 and reproj.size == 6:
            reproj = reproj.reshape(3, 2)
        else:
            reproj = np.asarray(reproj).reshape(-1, 3, 2)[0]
        for v in VIEWS:
            i = VIEW_INDEX[v]
            d = triplet[i] - reproj[i]
            errors[v] = float(np.linalg.norm(d))
    return xyz, errors


# ---------------------------------------------------------------------------
# Group expansion: load + seed all 3 views
# ---------------------------------------------------------------------------

def build_frame_group(
    key: Tuple[str, int],
    src_lfs: Dict[str, "sio.LabeledFrame"],
    *,
    medaka_data_dir: Path,
    cal: Calibration,
) -> FrameGroup:
    """Materialise one (exp, frame_idx) group: load images for all 3 views,
    extract existing keypoints, compute cross-view predictions for missing
    keypoints.
    """
    exp, frame_idx = key
    group = FrameGroup(experiment=exp, frame_idx=frame_idx, src_lfs=dict(src_lfs))

    for view in VIEWS:
        lf = src_lfs.get(view)
        img = load_embedded_or_disk_image(
            lf,
            medaka_data_dir=medaka_data_dir,
            experiment=exp,
            view=view,
            frame_idx=frame_idx,
        )
        if img is None:
            img = np.zeros((1200, 1200, 3), dtype=np.uint8)
        group.images[view] = img

    # Up to NUM_FISH instances per view; sort by detection score so
    # corresponding instances across views align by index. The round-2
    # predictions already keep that ordering.
    n_fish = 0
    for view in VIEWS:
        lf = src_lfs.get(view)
        if lf is not None:
            n_fish = max(n_fish, len(lf.instances))
    n_fish = min(NUM_FISH, n_fish or NUM_FISH)
    group.fish = [FishKeypoints() for _ in range(n_fish)]

    for view, lf in src_lfs.items():
        if lf is None:
            continue
        ranked = sorted(
            lf.instances,
            key=lambda inst: -(getattr(inst, "score", 0.0) or 0.0),
        )
        for i, inst in enumerate(ranked[:n_fish]):
            mouth, tail = instance_xy_pair(inst)
            group.fish[i].mouth[view] = mouth
            group.fish[i].tail[view] = tail

    # Predictions are computed on demand (via the 'f' hotkey) so the user
    # always sees the pristine state of what's annotated.
    return group


# ---------------------------------------------------------------------------
# napari integration
# ---------------------------------------------------------------------------

class MultiviewAnnotator:
    """Tiles all three views into a single napari canvas as one hstacked
    image, with one shared Points layer whose x-coordinates have view
    offsets baked in. Click handlers route by world-x to the correct view.

    World layout (in column pixels):
        top_cam:    x in [0, W_top]
        side1_cam:  x in [W_top, W_top + W_s1]
        side2_cam:  x in [W_top + W_s1, W_top + W_s1 + W_s2]

    Each view is padded to the canvas's max height so views with different
    aspect ratios still tile cleanly side by side.
    """

    def __init__(
        self,
        *,
        labels,
        cal: Calibration,
        group_keys: List[Tuple[str, int]],
        groups_src: Dict[Tuple[str, int], Dict[str, "sio.LabeledFrame"]],
        medaka_data_dir: Path,
        out_path: Path,
        autosave: bool = False,
        allow_calibration_fill: bool = True,
        embed_output: str = "user",
    ) -> None:
        self.labels = labels
        self.cal = cal
        self.group_keys = group_keys
        self.groups_src = groups_src
        self.medaka_data_dir = medaka_data_dir
        self.out_path = out_path
        self.autosave = bool(autosave)
        self.allow_calibration_fill = bool(allow_calibration_fill)
        self.embed_output = embed_output
        self.cur_idx: int = 0
        self.cur_group: Optional[FrameGroup] = None
        # Keypoint-only cache. Without this, navigating away rebuilds the
        # frame from the input SLP and silently loses unsaved blank-set edits.
        self.group_edits: Dict[Tuple[str, int], List[FishKeypoints]] = {}
        # Cached x-offsets per view, refreshed each frame group.
        self._x_offsets: Dict[str, float] = {}
        self._view_widths: Dict[str, float] = {}
        # Active state for click-to-add: which fish + which node the next
        # click will be assigned to. Updated via keyboard shortcuts.
        self.active_fish: int = 0
        self.active_node: str = "mouth"
        # Number of points present after the most recent rebuild. Used to
        # detect newly-added points in events.data.
        self._last_n_points: int = 0
        self._suspend_data_handler: bool = False

        self.viewer = None
        self.image_layer = None  # single hstacked image
        self.points_layer = None

    # -------------------- viewer setup --------------------

    def show(self) -> None:
        import napari

        self.viewer = napari.Viewer(title="Medaka multi-view annotator")
        # Single hstacked image layer (data set in _render_current).
        self.image_layer = self.viewer.add_image(
            np.zeros((1200, 3600, 3), dtype=np.uint8),
            name="views (top | side1 | side2)",
            rgb=True,
            blending="opaque",
        )

        # Editable points layer. Default mode = "add" so a click adds a
        # keypoint at the cursor with the current active_fish / active_node.
        self.points_layer = self.viewer.add_points(
            np.empty((0, 2)),
            name="keypoints",
            size=14,
            symbol="disc",
            border_width=0.15,
            properties={
                "view": np.empty(0, dtype="<U10"),
                "fish_id": np.empty(0, dtype=int),
                "node": np.empty(0, dtype="<U6"),
            },
            face_color="cyan",
            border_color="white",
        )
        # Start in add mode so the user can place keypoints by clicking.
        self.points_layer.mode = "add"
        self.viewer.layers.selection.active = self.points_layer

        self._add_help_widget()
        self._bind_keys()
        self._render_current()
        self.points_layer.events.data.connect(self._on_points_changed)
        napari.run()

    # -------------------- frame nav + render --------------------

    def _cache_current_group(self) -> None:
        if self.cur_group is None:
            return
        key = (self.cur_group.experiment, self.cur_group.frame_idx)
        self.group_edits[key] = clone_fish_keypoints(self.cur_group.fish)

    def _render_current(self) -> None:
        if not self.group_keys:
            self._set_status("No multi-view groups in this seed.")
            return
        key = self.group_keys[self.cur_idx]
        self.cur_group = build_frame_group(
            key,
            self.groups_src[key],
            medaka_data_dir=self.medaka_data_dir,
            cal=self.cal,
        )
        if key in self.group_edits:
            self.cur_group.fish = clone_fish_keypoints(self.group_edits[key])
        self._update_image_layers()
        self._populate_points_from_group()
        self._update_status_bar()

    def _update_image_layers(self) -> None:
        # Pad each view image to the same height; hstack into one canvas.
        imgs = [self.cur_group.images[v] for v in VIEWS]
        if any(img.ndim == 2 for img in imgs):
            imgs = [
                img if img.ndim == 3 else np.stack([img] * 3, axis=2)
                for img in imgs
            ]
        max_h = max(img.shape[0] for img in imgs)
        padded: List[np.ndarray] = []
        offset_x = 0.0
        self._x_offsets = {}
        self._view_widths = {}
        for view, img in zip(VIEWS, imgs):
            h, w = img.shape[:2]
            if h < max_h:
                pad = np.zeros((max_h - h, w, img.shape[2]), dtype=img.dtype)
                img = np.concatenate([img, pad], axis=0)
            padded.append(img)
            self._x_offsets[view] = offset_x
            self._view_widths[view] = float(w)
            offset_x += float(w)
        canvas = np.concatenate(padded, axis=1)
        self.image_layer.data = canvas
        self.viewer.reset_view()

    def _view_from_world_x(self, x: float) -> Optional[str]:
        for view in VIEWS:
            x0 = self._x_offsets[view]
            x1 = x0 + self._view_widths[view]
            if x0 <= x < x1:
                return view
        return None

    def _populate_points_from_group(self) -> None:
        """Build the unified points dataset from the current FrameGroup.

        Only *real* (i.e. user-placed or already-confirmed) keypoints are
        added. Calibration predictions live separately and are folded in
        only when the user presses ``f`` (see ``_fill_missing_views``).
        """
        rows: List[Tuple[float, float]] = []
        views: List[str] = []
        fish_ids: List[int] = []
        nodes: List[str] = []

        for fish_idx, fk in enumerate(self.cur_group.fish):
            for node_name, kp_dict in (("mouth", fk.mouth), ("tail", fk.tail)):
                for view in VIEWS:
                    real = kp_dict[view]
                    if real is None:
                        continue
                    rows.append((float(real[1]), float(real[0]) + self._x_offsets[view]))
                    views.append(view)
                    fish_ids.append(fish_idx)
                    nodes.append(node_name)

        data = np.array(rows, dtype=float) if rows else np.empty((0, 2))
        self._suspend_data_handler = True
        try:
            self.points_layer.data = data
            self.points_layer.properties = {
                "view": np.array(views, dtype="<U10"),
                "fish_id": np.array(fish_ids, dtype=int),
                "node": np.array(nodes, dtype="<U6"),
            }
            self._restyle_points()
            self._last_n_points = len(data)
        finally:
            self._suspend_data_handler = False

    def _restyle_points(self) -> None:
        """Set per-point face color (by fish_id) and symbol (disc=mouth,
        ring=tail)."""
        n = len(self.points_layer.data)
        if n == 0:
            return
        face_colors = np.zeros((n, 4))
        border_colors = np.zeros((n, 4))
        symbols = np.empty(n, dtype="<U16")
        for i in range(n):
            fish_id = int(self.points_layer.properties["fish_id"][i])
            node = str(self.points_layer.properties["node"][i])
            color = FISH_COLORS_RGB.get(fish_id, (0.5, 0.5, 0.5))
            face_colors[i] = (*color, 1.0)
            border_colors[i] = (1.0, 1.0, 1.0, 1.0)
            symbols[i] = "disc" if node == "mouth" else "ring"
        self.points_layer.face_color = face_colors
        self.points_layer.border_color = border_colors
        self.points_layer.symbol = symbols

    # -------------------- editing handlers --------------------

    def _on_points_changed(self, event=None) -> None:  # noqa: ARG002
        """Triggered after the user drags/adds/deletes a point.

        For newly-added points (data length > _last_n_points), assign
        ``view`` (from world-x), ``fish_id`` (from active state), ``node``
        (from active state). Then sync the FrameGroup to whatever is
        currently in the layer.
        """
        if self._suspend_data_handler or self.cur_group is None:
            return

        data = np.asarray(self.points_layer.data)
        n = len(data)

        # Editable copies of properties (we may need to write defaults for
        # any newly-added rows).
        props = dict(self.points_layer.properties or {})
        for k, dtype in (("view", "<U10"), ("fish_id", int), ("node", "<U6")):
            arr = np.asarray(props.get(k, np.empty(0, dtype=dtype)))
            # napari pads new rows with default ("" / 0) — overwrite for added rows.
            if len(arr) < n:
                pad_n = n - len(arr)
                if k == "view":
                    pad = np.array(["?"] * pad_n, dtype=dtype)
                elif k == "fish_id":
                    pad = np.full(pad_n, -1, dtype=dtype)
                else:  # node
                    pad = np.array(["?"] * pad_n, dtype=dtype)
                arr = np.concatenate([arr, pad])
            props[k] = arr

        # Fill metadata for any rows that don't have it (new clicks).
        for i in range(n):
            if str(props["view"][i]) not in VIEWS:
                view = self._view_from_world_x(float(data[i, 1]))
                if view is None:
                    # Click was outside any view tile; keep the point but
                    # park it as top_cam so we don't lose it. User can drag.
                    view = "top_cam"
                props["view"][i] = view
            if int(props["fish_id"][i]) < 0:
                props["fish_id"][i] = self.active_fish
            if str(props["node"][i]) not in NODE_NAMES:
                props["node"][i] = self.active_node

        # Push potentially-updated metadata back to the layer (without
        # triggering recursive handler runs).
        self._suspend_data_handler = True
        try:
            self.points_layer.properties = {
                "view": np.array(props["view"], dtype="<U10"),
                "fish_id": np.array(props["fish_id"], dtype=int),
                "node": np.array(props["node"], dtype="<U6"),
            }
        finally:
            self._suspend_data_handler = False

        # Sync FrameGroup from layer state (treating every visible point as
        # a real annotation).
        for fk in self.cur_group.fish:
            fk.mouth = {v: None for v in VIEWS}
            fk.tail = {v: None for v in VIEWS}
        n_fish = len(self.cur_group.fish)
        for i in range(n):
            view = str(props["view"][i])
            fish_id = int(props["fish_id"][i])
            node = str(props["node"][i])
            if view not in VIEWS or node not in NODE_NAMES:
                continue
            if fish_id >= n_fish:
                # User picked a higher fish slot than we expected; grow.
                self.cur_group.fish.extend(
                    FishKeypoints() for _ in range(fish_id - n_fish + 1)
                )
                n_fish = len(self.cur_group.fish)
            x = float(data[i, 1] - self._x_offsets[view])
            y = float(data[i, 0])
            fk = self.cur_group.fish[fish_id]
            (fk.mouth if node == "mouth" else fk.tail)[view] = (x, y)

        self._restyle_points()
        self._last_n_points = n
        self._update_status_bar()

    # -------------------- status bar / errors --------------------

    def _update_status_bar(self) -> None:
        if self.cur_group is None:
            return
        key = (self.cur_group.experiment, self.cur_group.frame_idx)
        active = (
            f"ACTIVE fish={self.active_fish + 1} (ID_{self.active_fish + 1})"
            f" node={self.active_node}"
        )
        bits = [
            f"[{self.cur_idx + 1}/{len(self.group_keys)}]",
            f"{key[0]} f={key[1]}",
            active,
            (
                f"cached={sum(fish_have_annotations(v) for v in self.group_edits.values())}"
                f"/{len(self.group_keys)}"
            ),
        ]
        for fish_idx, fk in enumerate(self.cur_group.fish):
            for node_name, kp in (("mouth", fk.mouth), ("tail", fk.tail)):
                full_views = [v for v in VIEWS if kp[v] is not None]
                if len(full_views) == 3:
                    pts = {v: kp[v] for v in VIEWS}
                    _, errs = triangulate_and_reproject(self.cal, pts)
                    err_str = "/".join(
                        f"{errs[v]:.1f}" for v in VIEWS
                    )
                    bits.append(f"f{fish_idx + 1}.{node_name} err={err_str}px")
        self._set_status("  |  ".join(bits))

    def _set_status(self, msg: str) -> None:
        if self.viewer is not None:
            self.viewer.status = msg

    def _add_help_widget(self) -> None:
        """Dock a static help label so the user sees the hotkeys at all times."""
        from qtpy.QtWidgets import QLabel

        fill_help = (
            "<b>Fill missing views</b>: disabled for independent annotation.<br>"
            if not self.allow_calibration_fill
            else (
                "<b>Fill missing views</b>: <kbd>f</kbd> (predicts the 3rd view "
                "for any fish that has both keypoints in 2 views).<br>"
            )
        )
        save_help = (
            "<b>Autosave</b>: enabled on navigation. "
            "<kbd>Ctrl+S</kbd> also saves immediately.<br>"
            if self.autosave
            else "<b>Save</b>: <kbd>Ctrl+S</kbd> writes the merged labels.<br>"
        )
        help_text = (
            "<h3>Multi-view annotator</h3>"
            "<b>Layout</b>: top | side1 | side2 hstacked.<br>"
            "<br>"
            "<b>Add a keypoint</b>: keep the Points layer in <i>add</i> mode "
            "(plus icon) and click on a view. The current ACTIVE fish + node "
            "(see status bar) is attached.<br>"
            "<b>Move a keypoint</b>: switch the Points layer to <i>select</i> "
            "mode (arrow icon), drag the point.<br>"
            "<b>Delete</b>: select-mode + Delete key.<br>"
            "<br>"
            "<b>Active fish</b>: <kbd>q</kbd>=ID_1 (blue), "
            "<kbd>w</kbd>=ID_2 (orange), <kbd>e</kbd>=ID_3 (green)<br>"
            "<b>Active node</b>: <kbd>a</kbd>=mouth (disc), "
            "<kbd>s</kbd>=tail (ring)<br>"
            "<br>"
            "<b>Frame nav</b>: <kbd>[</kbd>=prev, <kbd>]</kbd>=next<br>"
            + fill_help
            + save_help
            + "<br>"
            "When all 3 views have a keypoint for some fish/node the status "
            "bar shows the per-view reprojection error in pixels."
        )
        label = QLabel(help_text)
        label.setWordWrap(True)
        label.setMinimumWidth(280)
        self.viewer.window.add_dock_widget(label, area="right", name="Help")

    def _fill_missing_views(self) -> None:
        """For each fish, predict any view's mouth/tail that isn't yet
        annotated using the other 2 views via the calibration pairwise
        model. The predictions are inserted as regular keypoints — the user
        can drag to refine or delete if wrong.
        """
        if not self.allow_calibration_fill:
            self._set_status(
                "Calibration fill is disabled for independent locked-test annotation."
            )
            return
        if self.cur_group is None:
            return
        added = 0
        for fk in self.cur_group.fish:
            for node_name in ("mouth", "tail"):
                kp = fk.mouth if node_name == "mouth" else fk.tail
                pred = predict_missing_view(self.cal, kp)
                for view in VIEWS:
                    if kp[view] is None and pred[view] is not None:
                        kp[view] = pred[view]
                        added += 1
        self._populate_points_from_group()
        self._update_status_bar()
        if added:
            self._set_status(f"Filled {added} missing-view keypoint(s) from calibration.")
        else:
            self._set_status(
                "No views were filled. A fish needs both a mouth (or tail) "
                "in two views before the third can be predicted."
            )

    # -------------------- key bindings --------------------

    def _bind_keys(self) -> None:
        def navigate(delta: int) -> None:
            if self.autosave:
                self.save()
            else:
                self._cache_current_group()
            self.cur_idx = max(
                0,
                min(self.cur_idx + delta, len(self.group_keys) - 1),
            )
            self._render_current()

        @self.viewer.bind_key("BracketRight", overwrite=True)
        def next_frame(_v):  # noqa: ARG001
            navigate(1)

        @self.viewer.bind_key("BracketLeft", overwrite=True)
        def prev_frame(_v):  # noqa: ARG001
            navigate(-1)

        # Fish selection: q/w/e -> ID_1 / ID_2 / ID_3.
        for key, fish_idx in (("q", 0), ("w", 1), ("e", 2)):
            def make_set_fish(idx):
                def _set(_v):  # noqa: ARG001
                    self.active_fish = idx
                    self._update_status_bar()
                return _set
            self.viewer.bind_key(key, overwrite=True)(make_set_fish(fish_idx))

        # Node selection: a -> mouth, s -> tail.
        @self.viewer.bind_key("a", overwrite=True)
        def set_mouth(_v):  # noqa: ARG001
            self.active_node = "mouth"
            self._update_status_bar()

        @self.viewer.bind_key("s", overwrite=True)
        def set_tail(_v):  # noqa: ARG001
            self.active_node = "tail"
            self._update_status_bar()

        # Fill missing views via calibration pairwise model.
        @self.viewer.bind_key("f", overwrite=True)
        def fill_missing(_v):  # noqa: ARG001
            self._fill_missing_views()

        @self.viewer.bind_key("Control-S", overwrite=True)
        def save_now(_v):  # noqa: ARG001
            self.save()

    # -------------------- save --------------------

    def save(self) -> None:
        """Write merged annotations back to ``self.out_path``.

        Strategy:
          - Every original LabeledFrame outside the multi-view groups is kept
            verbatim (round-2 single-view frames).
          - For each multi-view group, build/replace 3 LFs (one per view),
            each with up to ``NUM_FISH`` Instance objects (mouth+tail).
          - SuggestionFrame entries cover every saved LF.
        """
        import sleap_io as sio

        # Pull latest edits from the live napari layer into `cur_group` if open.
        if self.cur_group is not None and self.points_layer is not None:
            self._on_points_changed()
        self._cache_current_group()

        skel = self.labels.skeletons[0]
        # Map source-video filename -> Video object so we keep a single
        # Video instance per file across the saved Labels.
        video_by_name: Dict[str, "sio.Video"] = {}
        for v in self.labels.videos:
            video_by_name[str(v.filename)] = v

        # Index original LFs by (experiment, frame_idx, view) so we can
        # detect which need replacement.
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

        for key in self.group_keys:
            exp, frame_idx = key
            if (
                self.cur_group is not None
                and (self.cur_group.experiment, self.cur_group.frame_idx) == key
            ):
                grp = self.cur_group
            else:
                grp = build_frame_group(
                    key,
                    self.groups_src[key],
                    medaka_data_dir=self.medaka_data_dir,
                    cal=self.cal,
                )
                if key in self.group_edits:
                    grp.fish = clone_fish_keypoints(self.group_edits[key])
            for view in VIEWS:
                # Determine the target Video for this LF: prefer the existing
                # source video for this (exp, view); fall back to the on-disk
                # video.
                src_lf = grp.src_lfs.get(view)
                if src_lf is not None:
                    target_video = video_by_name.setdefault(
                        str(src_lf.video.filename), src_lf.video
                    )
                else:
                    on_disk = (
                        self.medaka_data_dir / exp / view / VIEW_TO_VIDEO[view]
                    )
                    target_video = video_by_name.get(str(on_disk))
                    if target_video is None:
                        try:
                            target_video = sio.load_video(str(on_disk))
                        except Exception:
                            target_video = None
                        if target_video is not None:
                            video_by_name[str(on_disk)] = target_video

                if target_video is None:
                    # Without a usable video object we cannot persist this view.
                    continue

                instances = []
                for fk in grp.fish:
                    m = fk.mouth[view]
                    t = fk.tail[view]
                    if m is None and t is None:
                        continue
                    pts = np.array(
                        [
                            list(m) if m is not None else [np.nan, np.nan],
                            list(t) if t is not None else [np.nan, np.nan],
                        ],
                        dtype=float,
                    )
                    instances.append(sio.Instance.from_numpy(points_data=pts, skeleton=skel))
                # Preserve an explicit blank LF too. This makes deletion of all
                # instances durable and keeps blank annotation manifests
                # resumable without dropping unvisited frames.
                new_lfs.append(
                    sio.LabeledFrame(
                        video=target_video,
                        frame_idx=frame_idx,
                        instances=instances,
                    )
                )
                replaced_keys.add((exp, frame_idx, view))

        # Carry over any LF that wasn't replaced (round-2 single-view frames
        # plus any group-frame view we couldn't write).
        kept_lfs: List["sio.LabeledFrame"] = []
        for (exp, frame_idx, view), lf in lf_index.items():
            if (exp, frame_idx, view) in replaced_keys:
                continue
            kept_lfs.append(lf)

        merged_lfs = kept_lfs + new_lfs
        suggestions = [
            sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
            for lf in merged_lfs
        ]
        merged = sio.Labels(
            skeletons=[skel],
            videos=list({id(v): v for v in video_by_name.values()}.values()),
            labeled_frames=merged_lfs,
            suggestions=suggestions,
            provenance=copy_labels_provenance(self.labels),
        )
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(merged_lfs)} LFs ({len(new_lfs)} replaced, "
              f"{len(kept_lfs)} kept) -> {self.out_path}")
        embed = False if self.embed_output == "none" else self.embed_output
        sio.save_slp(merged, str(self.out_path), embed=embed)
        print("  done.")
        self._set_status(f"Saved {len(merged_lfs)} LFs to {self.out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input", type=Path, required=True,
                    help="Round-3 annotation_seed.pkg.slp.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output pkg.slp.")
    ap.add_argument("--paths-config", default=None,
                    help="paths.local.toml (for medaka_data_dir + calibration).")
    ap.add_argument("--medaka-data-dir", default=None,
                    help="Override medaka_data_dir from paths.local.toml.")
    ap.add_argument("--calibration-model-dir", default=None,
                    help="Override calibration_model_dir from paths.local.toml.")
    ap.add_argument("--min-views", type=int, default=1,
                    help="Only show (exp, frame_idx) groups with at least N "
                    "views already populated. Default 1 = include round-2 "
                    "single-view picks (so the GUI also lets you fill the "
                    "other 2 views in for those).")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If --out already exists, load it instead of --in.",
    )
    ap.add_argument(
        "--autosave",
        action="store_true",
        help="Save before previous/next navigation; Ctrl+S also saves immediately.",
    )
    ap.add_argument(
        "--independent",
        action="store_true",
        help="Disable calibration-based missing-view fill for a locked test set.",
    )
    ap.add_argument(
        "--embed-output",
        choices=("user", "none"),
        default="user",
        help=(
            "Embed annotated images in --out ('user') or retain raw-video "
            "references ('none'). Use none for fast autosave."
        ),
    )
    args = ap.parse_args(argv)

    local = load_local_paths(args.paths_config)
    medaka_data_dir = choose_path(
        args.medaka_data_dir, local.medaka_data_dir if local else None,
    )
    calib_dir = choose_path(
        args.calibration_model_dir,
        local.calibration_model_dir if local else None,
    )
    if medaka_data_dir is None:
        ap.error("medaka_data_dir is required.")
    if calib_dir is None:
        ap.error("calibration_model_dir is required.")

    import sleap_io as sio

    input_path = args.out if args.resume and args.out.exists() else args.input
    print(f"Loading {input_path} ...")
    labels = sio.load_slp(str(input_path))
    print(f"  {len(labels.labeled_frames)} LFs, {len(labels.videos)} videos.")

    cal = Calibration(str(calib_dir))
    if cal._model_xy_from_xz_yz is None:
        print("Warning: no xz_yz_to_xy.joblib in calibration; can't predict cam0 from sides.")

    groups = build_groups(labels)
    keys = filter_groups(groups, min_views=args.min_views)
    n_views_hist: Dict[int, int] = {}
    for k in keys:
        n = len(groups[k])
        n_views_hist[n] = n_views_hist.get(n, 0) + 1
    print(
        f"  {len(keys)} (exp, frame_idx) groups (min_views={args.min_views})"
        f" out of {len(groups)} total."
    )
    print(
        "  views-per-group: "
        + ", ".join(f"{n}:{c}" for n, c in sorted(n_views_hist.items()))
    )

    if not keys:
        print("No multi-view groups to annotate. Exiting.")
        return

    annotator = MultiviewAnnotator(
        labels=labels,
        cal=cal,
        group_keys=keys,
        groups_src=groups,
        medaka_data_dir=Path(medaka_data_dir),
        out_path=args.out,
        autosave=args.autosave,
        allow_calibration_fill=not args.independent,
        embed_output=args.embed_output,
    )
    annotator.show()


if __name__ == "__main__":
    main()
