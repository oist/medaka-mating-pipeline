from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import iter_mask_instances_h5


REVIEW_LABELS = (
    "good",
    "partial",
    "uncertain",
    "merged",
    "missed",
    "wrong_object",
    "background",
)

REVIEW_SHORTCUTS = {
    "G": "good",
    "P": "partial",
    "U": "uncertain",
    "M": "merged",
    "D": "missed",
    "X": "wrong_object",
    "B": "background",
    "C": "",
}


def read_review_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    return pd.read_parquet(path)


def build_review_template(
    tracklets_qc: pd.DataFrame,
    *,
    reference_points: pd.DataFrame | None = None,
    max_reference_distance_px: float = 120.0,
) -> pd.DataFrame:
    required = {"frame", "tracklet_id"}
    missing = required - set(tracklets_qc.columns)
    if missing:
        raise ValueError(f"tracklets_qc missing columns: {', '.join(sorted(missing))}")

    cols = [
        "frame",
        "tracklet_id",
        "review_label",
        "assigned_identity",
        "identity_review_label",
        "review_note",
        "qc_status",
        "qc_reason",
        "centroid_x",
        "centroid_y",
        "reference_identity",
        "reference_distance_px",
    ]
    rows = tracklets_qc.copy()
    rows = rows.sort_values(["frame", "tracklet_id"]).drop_duplicates(["frame", "tracklet_id"])
    for col in cols:
        if col not in rows.columns:
            rows[col] = ""
    rows["review_label"] = rows["review_label"].fillna("").astype(str)
    rows["identity_review_label"] = rows["identity_review_label"].fillna("").astype(str)
    rows["review_note"] = rows["review_note"].fillna("").astype(str)

    if reference_points is not None and not reference_points.empty:
        rows = _attach_nearest_reference(rows, reference_points, max_distance_px=max_reference_distance_px)

    return rows[cols].copy()


def write_review_template(
    *,
    tracklets_qc: pd.DataFrame,
    output_csv: str | Path,
    reference_points: pd.DataFrame | None = None,
    max_reference_distance_px: float = 120.0,
) -> dict:
    template = build_review_template(
        tracklets_qc,
        reference_points=reference_points,
        max_reference_distance_px=max_reference_distance_px,
    )
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    return {
        "output_csv": str(path),
        "n_rows": int(len(template)),
        "review_labels": list(REVIEW_LABELS),
        "review_shortcuts": dict(REVIEW_SHORTCUTS),
        "n_reference_labeled_rows": int(template["reference_identity"].fillna("").astype(str).ne("").sum()),
    }


def set_review_labels(
    review: pd.DataFrame,
    *,
    frame: int,
    tracklet_ids: list[int],
    review_label: str,
) -> int:
    if review_label and review_label not in REVIEW_LABELS:
        raise ValueError(f"Unknown review_label {review_label!r}; expected one of {', '.join(REVIEW_LABELS)}")
    if not tracklet_ids:
        return 0
    required = {"frame", "tracklet_id", "review_label"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"review table missing columns: {', '.join(sorted(missing))}")
    tids = {int(v) for v in tracklet_ids}
    mask = review["frame"].astype(int).eq(int(frame)) & review["tracklet_id"].astype(int).isin(tids)
    review.loc[mask, "review_label"] = str(review_label)
    return int(mask.sum())


def tracklet_ids_for_frame(review: pd.DataFrame, *, frame: int) -> list[int]:
    required = {"frame", "tracklet_id"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"review table missing columns: {', '.join(sorted(missing))}")
    rows = review[review["frame"].astype(int).eq(int(frame))]
    return sorted(int(v) for v in rows["tracklet_id"].dropna().astype(int).unique())


def save_review_table(review: pd.DataFrame, output_csv: str | Path) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(path, index=False)


def launch_napari_review(
    *,
    video_path: str | Path,
    mask_path: str | Path,
    tracklets_qc: pd.DataFrame,
    output_csv: str | Path,
    reference_points: pd.DataFrame | None = None,
    max_frames: int | None = 300,
    max_reference_distance_px: float = 120.0,
) -> dict:
    output_path = Path(output_csv)
    if output_path.exists():
        review = read_review_table(output_path)
        summary = _review_summary(review, output_path, existing_review_file=True)
    else:
        summary = write_review_template(
            tracklets_qc=tracklets_qc,
            output_csv=output_path,
            reference_points=reference_points,
            max_reference_distance_px=max_reference_distance_px,
        )
        review = read_review_table(output_path)
    try:
        import cv2
        import napari
    except ImportError as exc:
        summary["napari_launched"] = False
        summary["napari_error"] = f"missing_dependency: {exc}"
        return summary

    frames = sorted(int(v) for v in tracklets_qc["frame"].dropna().astype(int).unique())
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    images = _load_video_frames(video_path, frames)
    labels = _labels_stack(mask_path, frames, images.shape[1:3])

    viewer = napari.Viewer(title="Medaka side-camera tracklet review")
    viewer.add_image(images, name="video", rgb=True)
    labels_layer = viewer.add_labels(labels, name="tracklet_masks")
    if reference_points is not None and not reference_points.empty:
        points = _points_layer(reference_points, frames)
        if len(points):
            viewer.add_points(points, name="projected_3d_identity", size=8, face_color="yellow")
    _add_review_dock_widget(
        viewer=viewer,
        labels_layer=labels_layer,
        review=review,
        output_csv=output_path,
        frames=frames,
    )
    napari.run()
    summary["napari_launched"] = True
    summary.update(_review_summary(read_review_table(output_path), output_path, existing_review_file=True))
    return summary


def _review_summary(review: pd.DataFrame, output_csv: Path, *, existing_review_file: bool) -> dict:
    return {
        "output_csv": str(output_csv),
        "n_rows": int(len(review)),
        "review_labels": list(REVIEW_LABELS),
        "review_shortcuts": dict(REVIEW_SHORTCUTS),
        "existing_review_file": bool(existing_review_file),
        "n_review_labeled_rows": int(review.get("review_label", pd.Series(dtype=str)).fillna("").astype(str).ne("").sum()),
        "n_reference_labeled_rows": int(review.get("reference_identity", pd.Series(dtype=str)).fillna("").astype(str).ne("").sum()),
    }


def _add_review_dock_widget(*, viewer, labels_layer, review: pd.DataFrame, output_csv: Path, frames: list[int]) -> None:
    from qtpy import QtCore, QtWidgets

    class ReviewDock(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.review = review
            self.output_csv = output_csv
            self.frames = frames

            layout = QtWidgets.QVBoxLayout(self)
            self.frame_label = QtWidgets.QLabel("")
            self.status_label = QtWidgets.QLabel("")
            self.status_label.setWordWrap(True)
            layout.addWidget(self.frame_label)
            shortcut_hint = QtWidgets.QLabel(_shortcut_hint_text())
            shortcut_hint.setWordWrap(True)
            layout.addWidget(shortcut_hint)

            self.table = QtWidgets.QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels([
                "tracklet_id", "qc_status", "reference_identity", "review_label", "qc_reason"
            ])
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            self.table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(self.table)

            button_grid = QtWidgets.QGridLayout()
            for idx, label in enumerate(REVIEW_LABELS):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(lambda _checked=False, value=label: self.apply_label(value))
                button_grid.addWidget(button, idx // 2, idx % 2)
            clear_button = QtWidgets.QPushButton("clear")
            clear_button.clicked.connect(lambda _checked=False: self.apply_label(""))
            button_grid.addWidget(clear_button, (len(REVIEW_LABELS) + 1) // 2, 0)
            layout.addLayout(button_grid)

            save_button = QtWidgets.QPushButton("Save CSV")
            save_button.clicked.connect(self.save)
            layout.addWidget(save_button)
            layout.addWidget(self.status_label)
            self._install_panel_shortcuts()
            self.refresh()

        def _install_panel_shortcuts(self) -> None:
            save_action = QtWidgets.QAction(self)
            save_action.setShortcut("Ctrl+S")
            save_action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
            save_action.triggered.connect(self.save)
            self.addAction(save_action)

        def current_frame(self) -> int:
            if not self.frames:
                return 0
            step = int(viewer.dims.current_step[0]) if viewer.dims.current_step else 0
            step = max(0, min(step, len(self.frames) - 1))
            return int(self.frames[step])

        def selected_tracklet_ids(self) -> list[int]:
            rows = {index.row() for index in self.table.selectedIndexes()}
            tids: list[int] = []
            for row in sorted(rows):
                item = self.table.item(row, 0)
                if item is not None:
                    tids.append(int(item.data(QtCore.Qt.UserRole)))
            if tids:
                return tids
            selected_label = int(getattr(labels_layer, "selected_label", 0) or 0)
            return [selected_label - 1] if selected_label > 0 else []

        def apply_label(self, label: str, *, all_current_frame_if_none: bool = False) -> None:
            tids = self.selected_tracklet_ids()
            if not tids and all_current_frame_if_none:
                tids = tracklet_ids_for_frame(self.review, frame=self.current_frame())
            if not tids:
                self.status_label.setText("Select one or more table rows, or select a mask label.")
                return
            n_changed = set_review_labels(
                self.review,
                frame=self.current_frame(),
                tracklet_ids=tids,
                review_label=label,
            )
            self.save()
            self.refresh()
            self.status_label.setText(f"Saved {label or 'blank'} for {n_changed} row(s) at frame {self.current_frame()}.")

        def save(self) -> None:
            save_review_table(self.review, self.output_csv)
            n_labeled = int(self.review["review_label"].fillna("").astype(str).ne("").sum())
            self.status_label.setText(f"Saved {self.output_csv} ({n_labeled} labeled rows).")

        def refresh(self, *_args) -> None:
            frame = self.current_frame()
            frame_rows = self.review[self.review["frame"].astype(int).eq(frame)].copy()
            frame_rows.sort_values("tracklet_id", inplace=True)
            self.frame_label.setText(f"Frame {frame}: select row(s), click a label, then Save CSV.")
            self.table.setRowCount(len(frame_rows))
            for row_idx, row in enumerate(frame_rows.to_dict("records")):
                values = [
                    int(row["tracklet_id"]),
                    str(row.get("qc_status", "")),
                    str(row.get("reference_identity", "")),
                    str(row.get("review_label", "")),
                    str(row.get("qc_reason", "")),
                ]
                for col_idx, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    if col_idx == 0:
                        item.setData(QtCore.Qt.UserRole, int(row["tracklet_id"]))
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    self.table.setItem(row_idx, col_idx, item)
            self.table.resizeColumnsToContents()

    dock = ReviewDock()
    for key, label in REVIEW_SHORTCUTS.items():
        viewer.bind_key(
            key.lower(),
            lambda _viewer, value=label: dock.apply_label(value, all_current_frame_if_none=True),
            overwrite=True,
        )
        viewer.bind_key(
            key,
            lambda _viewer, value=label: dock.apply_label(value, all_current_frame_if_none=True),
            overwrite=True,
        )
    viewer.bind_key("Control-S", lambda _viewer: dock.save(), overwrite=True)
    viewer.dims.events.current_step.connect(dock.refresh)
    if hasattr(labels_layer, "events") and hasattr(labels_layer.events, "selected_label"):
        labels_layer.events.selected_label.connect(dock.refresh)
    viewer.window.add_dock_widget(dock, area="right", name="Tracklet Review")


def _shortcut_hint_text() -> str:
    parts = [
        "Shortcuts:",
        "G good",
        "P partial",
        "U uncertain",
        "M merged",
        "D missed",
        "X wrong_object",
        "B background",
        "C clear",
        "Ctrl+S save",
    ]
    return "  ".join(parts)


def _attach_nearest_reference(
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    max_distance_px: float,
) -> pd.DataFrame:
    ref = reference.copy()
    if "identity" not in ref.columns:
        if "final_fish_id" in ref.columns:
            ref["identity"] = ref["final_fish_id"]
        elif "assigned_identity" in ref.columns:
            ref["identity"] = ref["assigned_identity"]
        else:
            return rows
    if not {"frame", "x", "y"}.issubset(ref.columns):
        return rows

    by_frame = {int(frame): group.copy() for frame, group in ref.groupby("frame", sort=False)}
    identities: list[str] = []
    distances: list[float] = []
    for row in rows.to_dict("records"):
        frame_ref = by_frame.get(int(row["frame"]))
        if frame_ref is None or frame_ref.empty or not _finite_xy(row):
            identities.append("")
            distances.append(np.nan)
            continue
        xy = frame_ref[["x", "y"]].to_numpy(dtype=float)
        distance = np.hypot(xy[:, 0] - float(row["centroid_x"]), xy[:, 1] - float(row["centroid_y"]))
        best = int(np.argmin(distance))
        if float(distance[best]) <= float(max_distance_px):
            identities.append(str(frame_ref.iloc[best]["identity"]))
            distances.append(float(distance[best]))
        else:
            identities.append("")
            distances.append(float(distance[best]))
    out = rows.copy()
    out["reference_identity"] = identities
    out["reference_distance_px"] = distances
    return out


def _finite_xy(row: dict) -> bool:
    try:
        return np.isfinite(float(row["centroid_x"])) and np.isfinite(float(row["centroid_y"]))
    except (KeyError, TypeError, ValueError):
        return False


def _load_video_frames(video_path: str | Path, frames: list[int]) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    loaded: list[np.ndarray] = []
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cap.read()
            if ok:
                loaded.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not loaded:
        raise RuntimeError("No review frames could be loaded.")
    return np.stack(loaded, axis=0)


def _labels_stack(mask_path: str | Path, frames: list[int], shape: tuple[int, int]) -> np.ndarray:
    frame_index = {frame: i for i, frame in enumerate(frames)}
    labels = np.zeros((len(frames), int(shape[0]), int(shape[1])), dtype=np.int32)
    by_frame: dict[int, list] = defaultdict(list)
    for inst in iter_mask_instances_h5(mask_path):
        if int(inst.frame) in frame_index:
            by_frame[int(inst.frame)].append(inst)
    for frame, instances in by_frame.items():
        plane = labels[frame_index[frame]]
        for inst in instances:
            tracklet_id = int(inst.sam2_track_id if inst.sam2_track_id >= 0 else inst.instance_id)
            mask = np.asarray(inst.mask, dtype=bool)
            if mask.shape != plane.shape:
                continue
            plane[mask] = tracklet_id + 1
    return labels


def _points_layer(reference: pd.DataFrame, frames: list[int]) -> np.ndarray:
    frame_index = {frame: i for i, frame in enumerate(frames)}
    rows = []
    for row in reference.to_dict("records"):
        frame = int(row["frame"])
        if frame in frame_index:
            rows.append([frame_index[frame], float(row["y"]), float(row["x"])])
    return np.asarray(rows, dtype=float)
