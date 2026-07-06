"""Step 6 (Goal 5) — gravid-belly GT spot-check montage.

The dual-line female call is appearance-internal-consistency-validated (0.99) but NOT externally grounded:
the model's notion of "female" comes from its K_1_2 training anchor. This renders a labeled side-view
montage of the called F / M1 / M2 at well-separated high-confidence frames across the timeline (+ the
spawn frame), so the biological female (visibly distended gravid belly) can be checked against the
appearance call by eye. Rows = frames, columns = F | M1 | M2 (consistent order).

Output: gt_montage.png (+ the chosen frames). Crops are fixed boxes around each fish's side1 tracklet
point (no model needed). Pure rendering on the interp track + local side video.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from sam2_id_experiment.src.id_model import per_frame_resolve as pfr  # noqa: E402

TRACK = HERE / "step2" / "round5_260611_3d_individual_id_interp.csv"
SIDE1_TB = Path(r"C:\udmt_validation\tracklet_bind_extend_smooth\side1_cam.tracklet_bound.csv")
SIDE1_VID = Path(r"C:\Users\machi\AppData\Local\Temp\claude\c--Users-machi-Documents-GitHub-medaka-mating-pipeline\b07af347-293e-4fed-a208-e52ffcd113b6\scratchpad\videos\side1_cam\cam1_side.mp4")
MASKS = Path(r"Z:\StephensU\medaka\detectron2_output\20251225_K_1_2_FCMC")
CHUNKS = [(0, 43574), (43575, 87149), (87150, 130724), (130725, 174299),
          (174300, 217874), (217875, 261449), (261450, 305024), (305025, 348599)]
OUT = HERE / "step2" / "gt_montage.png"

SPAWN = 135888                       # 818.6s * 166 fps (BORIS ethogram spawn)
CELL = 240                           # montage cell size (px); crops tight-framed by mask bbox
COLORS = {"F": (255, 0, 255), "M1": (255, 255, 0), "M2": (0, 255, 255)}   # BGR: F magenta, M1 cyan, M2 yellow
ORDER = ["F", "M1", "M2"]
N_SPREAD = 9


def select_frames(track):
    """Well-separated, all-3-high-confidence frames spread across the timeline, + the spawn frame."""
    piv = track.pivot_table(index="frame", columns="geometry_slot", values=["X", "Y", "Z"])
    frames = piv.index.to_numpy()
    P = {s: np.stack([piv[("X", s)], piv[("Y", s)], piv[("Z", s)]], 1) for s in (0, 1, 2)}
    dmin = np.full(len(frames), np.inf)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        dmin = np.minimum(dmin, np.linalg.norm(P[a] - P[b], axis=1))
    conf_all_high = track.groupby("frame").id_conf.apply(lambda s: (s == "high").all())
    ok = conf_all_high.reindex(frames).fillna(False).to_numpy() & (dmin > 40) & np.isfinite(dmin)
    good = frames[ok]
    if len(good) >= N_SPREAD:
        idx = np.linspace(0, len(good) - 1, N_SPREAD).astype(int)
        chosen = sorted(set(good[idx].tolist()) | {SPAWN})
    else:
        chosen = sorted(set(good.tolist()) | {SPAWN})
    return chosen


def fish_slots(track, frame):
    g = track[track.frame == frame]
    return {row.id_final: int(row.geometry_slot) for row in g.itertuples()}


def main():
    track = pd.read_csv(TRACK, usecols=["frame", "geometry_slot", "X", "Y", "Z", "id_final", "id_conf"])
    tb = pd.read_csv(SIDE1_TB, usecols=["frame", "x", "y", "global_id"])
    pts = {}
    for f, x, y, g in zip(tb.frame, tb.x, tb.y, tb.global_id):
        pts.setdefault(int(f), {})[int(g)] = (float(x), float(y))

    # only spread frames where all 3 slots have a side1 point (avoids empty cells); spawn kept regardless
    spread = [f for f in select_frames(track) if f == SPAWN or len(pts.get(f, {})) == 3]
    chunks = {(cs, ce): h5py.File(MASKS / f"side1_cam.masks_{cs}-{ce}.h5", "r") for cs, ce in CHUNKS}

    def chunk_h5(f):
        return next((h for (cs, ce), h in chunks.items() if cs <= f <= ce), None)

    cap = cv2.VideoCapture(str(SIDE1_VID))
    rows = []
    for fr in spread:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        slots = fish_slots(track, fr)
        h5 = chunk_h5(fr)
        masks = pfr.read_frame_masks(h5, fr) if h5 is not None else []
        framepts = [(x, y, g) for g, (x, y) in pts.get(fr, {}).items()]
        gid2mask = dict(pfr.link_gid(masks, framepts)) if masks else {}
        cells = []
        for fid in ORDER:
            cell = np.full((CELL, CELL, 3), 40, np.uint8)
            sl = slots.get(fid)
            if sl is not None and sl in gid2mask:                       # tight bbox from the mask
                cell = pfr.square_crop(img, masks[gid2mask[sl]][1], CELL, pad_frac=0.35)
            elif sl is not None and sl in pts.get(fr, {}):              # fallback: fixed box on the point
                cx, cy = (int(v) for v in pts[fr][sl]); H, W = img.shape[:2]
                crop = img[max(0, cy - 90):min(H, cy + 90), max(0, cx - 90):min(W, cx + 90)]
                if crop.size:
                    cell = cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_AREA)
            cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), COLORS[fid], 4)
            cv2.putText(cell, fid, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[fid], 2, cv2.LINE_AA)
            cells.append(cell)
        label = np.full((CELL, 90, 3), 20, np.uint8)
        tag = f"f{fr}" + ("\nSPAWN" if fr == SPAWN else "")
        for i, t in enumerate(tag.split("\n")):
            cv2.putText(label, t, (4, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack([label] + cells))
    cap.release()
    for h in chunks.values():
        h.close()
    if not rows:
        raise SystemExit("no frames rendered")
    header = np.full((40, rows[0].shape[1], 3), 20, np.uint8)
    for i, fid in enumerate(["frame"] + ORDER):
        cv2.putText(header, fid, (10 + i * (90 if i == 0 else CELL), 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    montage = np.vstack([header] + rows)
    cv2.imwrite(str(OUT), montage)
    print(f"frames: {spread}")
    print(f"wrote {OUT}  ({montage.shape[1]}x{montage.shape[0]})")


if __name__ == "__main__":
    main()
