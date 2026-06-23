from __future__ import annotations

"""Build the script-33 point table (``<view>.tracklet_bound.csv`` shape) directly
from a 3D-identified reconstruction CSV, by reprojecting each geometry fish slot
into every requested camera view.

Why this exists
---------------
``33_build_segid_dataset`` links Detectron2 masks to *per-frame fish points* that
carry a stable geometry slot (``global_id``). On the K_1_2 pilot those points came
from the heavy concrete-tracklet -> 3D-bind workflow (``bind_tracklets_3d.py``),
written as ``<view>.tracklet_bound.csv``. For a new experiment that lacks those
artifacts, we can synthesize an equivalent table far more cheaply and with tighter
integration: the canonical identity already lives in ``<tag>_3d_identified.csv``
(geometry ``fish_id`` = ID_1/ID_2/ID_3, one row per fish per frame), so we just
reproject each fish's finite ``(X, Y, Z)`` into each view via the same calibration
``reconstruct_from_slp`` trusts.

The geometry slot mapping matches how the female anchor (``38_apply_female_anchor``)
joins sex back onto the 3D CSV: ``global_id k <-> fish_id ID_{k+1}``. So slot 0 is
ID_1, slot 1 is ID_2, slot 2 is ID_3. We set ``tracklet_id == global_id`` (one
tracklet per slot per view): at *apply* time the only thing that matters is that
every crop carries the correct geometry slot, which per-slot aggregation then votes
on. Frames where two fish reproject into the same Detectron2 mask are still dropped
by ``33`` (its "merged mask" rule), so the anchor only ever votes on cleanly
separated crops -- exactly the validated regime.
"""

import re
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

# Output column order mirrors the real ``<view>.tracklet_bound.csv`` so the
# downstream scripts (33 reads frame/x/y/tracklet_id; 34 reads tracklet_id/global_id)
# consume it unchanged.
SLOT_POINT_COLUMNS = ["frame", "x", "y", "score", "global_id", "tracklet_id"]

_FISH_ID_RE = re.compile(r"(\d+)")


class _Projector(Protocol):
    def project_world_to_view(self, xyz: np.ndarray, view: str) -> np.ndarray:
        """Reproject (N, 3) world coords to (N, 2) pixel coords for one view."""


def slot_from_fish_id(fish_id) -> int:
    """Map a geometry ``fish_id`` to its zero-based slot.

    ``ID_1 -> 0``, ``ID_2 -> 1``, ``ID_3 -> 2``. Plain integers are assumed to be
    already zero-based slot indices and returned unchanged. The canonical
    ``_3d_identified.csv`` uses the ``ID_<n>`` form, so we key off the trailing
    integer of the string and subtract one. Raises ``ValueError`` for anything we
    cannot map, so a schema surprise fails loudly instead of mislabelling sex.
    """
    if isinstance(fish_id, (int, np.integer)):
        return int(fish_id)
    s = str(fish_id).strip()
    m = _FISH_ID_RE.search(s)
    if m is None:
        raise ValueError(f"cannot derive geometry slot from fish_id {fish_id!r}")
    n = int(m.group(1))
    # ID_1/ID_2/ID_3 -> 0/1/2
    return n - 1 if s.upper().startswith("ID") else n


def build_slot_points(
    identified_df: pd.DataFrame,
    projector: _Projector,
    views: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Reproject every finite fish position into each view.

    Parameters
    ----------
    identified_df:
        A ``<tag>_3d_identified.csv`` frame with at least ``frame, fish_id, X, Y,
        Z``. Rows with a non-finite position are skipped (cannot be reprojected).
    projector:
        Anything exposing ``project_world_to_view(xyz, view) -> (N, 2)`` (the
        ``scripts.reconstruction_3d.calibration.Calibration`` does).
    views:
        Camera views to project into, e.g. ``["side1_cam", "side2_cam"]``.

    Returns
    -------
    dict view -> DataFrame with columns :data:`SLOT_POINT_COLUMNS`.
    """
    required = {"frame", "fish_id", "X", "Y", "Z"}
    missing = required - set(identified_df.columns)
    if missing:
        raise ValueError(f"identified_3d_csv missing columns: {sorted(missing)}")

    df = identified_df[["frame", "fish_id", "X", "Y", "Z"]].copy()
    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    finite = np.isfinite(xyz).all(axis=1)
    df = df.loc[finite].reset_index(drop=True)
    xyz = xyz[finite]
    slots = df["fish_id"].map(slot_from_fish_id).to_numpy(dtype=int)
    frames = df["frame"].to_numpy(dtype=int)

    out: dict[str, pd.DataFrame] = {}
    for view in views:
        if len(xyz) == 0:
            out[view] = pd.DataFrame(columns=SLOT_POINT_COLUMNS)
            continue
        proj = np.asarray(projector.project_world_to_view(xyz, view), dtype=float)
        if proj.ndim != 2 or proj.shape[1] != 2 or len(proj) != len(xyz):
            raise ValueError(
                f"project_world_to_view({view!r}) returned shape {proj.shape}, "
                f"expected ({len(xyz)}, 2)"
            )
        view_df = pd.DataFrame(
            {
                "frame": frames,
                "x": proj[:, 0],
                "y": proj[:, 1],
                "score": 1.0,
                "global_id": slots,
                "tracklet_id": slots,
            }
        )
        # Drop any reprojection that came back non-finite (defensive).
        view_df = view_df[np.isfinite(view_df["x"]) & np.isfinite(view_df["y"])]
        out[view] = view_df.sort_values(["frame", "global_id"]).reset_index(drop=True)[
            SLOT_POINT_COLUMNS
        ]
    return out
