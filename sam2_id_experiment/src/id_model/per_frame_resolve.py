"""Per-frame appearance resolution of a 3-fish (1F + 2M) track — the production form of the
dual-line Line-B "step 2/3" pipeline validated on K_1_2.

Line A (3D geometry) gives positions + a swap-prone per-frame ``global_id`` (geometry slot 0/1/2 <->
fish_id ID_1/2/3). Line B (appearance) runs clean classifiers on Detectron2-mask crops and pins, per
frame, which geometry slot is the female and which two are M1/M2 — correcting the linker's crossings.

This module holds the **pure** resolution/fusion logic (NumPy/pandas, no I/O), shared by the research
scripts (``runs/dual_line/lineB_fulltimeline/step2_*``/``step3_fuse``) and the production stage
(``45_per_frame_resolve`` / stage 5 of ``40_female_anchor_pipeline``). The mask+video inference loop
that produces the per-frame probability tables is :func:`per_frame_class_prob` at the bottom (kept here
so both callers share one implementation; it needs cv2/h5py and is exercised end-to-end, not unit-tested).

Conventions: geometry slot k <-> fish_id ID_{k+1}. Female model head -> P(female). Male model classes
[gid_0, gid_2]; head -> P(gid_2) == P(M_beta == M2). M_alpha == M1 == gid_0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = (0, 1, 2)


# --------------------------------------------------------------------------- #
# Pure resolution / fusion core (unit-tested)                                 #
# --------------------------------------------------------------------------- #
def forward_fill_slot(anchor_frames, anchor_slots, n_frames: int, high_dist: int):
    """Forward-fill per-anchor slot values to frame resolution by nearest preceding anchor.

    Returns ``(slot_ff, conf)`` where ``slot_ff[f]`` is the slot of the nearest preceding confident
    anchor and ``conf[f]`` is ``"high"`` iff ``f`` is within ``high_dist`` frames of any anchor, else
    ``"low"``. With no anchors -> all ``-1`` / ``"low"``.
    """
    allf = np.arange(n_frames)
    if len(anchor_frames) == 0:
        return np.full(n_frames, -1, dtype=int), np.array(["low"] * n_frames, dtype=object)
    af = np.asarray(anchor_frames)
    asl = np.asarray(anchor_slots, dtype=int)
    order = np.argsort(af)
    af, asl = af[order], asl[order]
    prev = np.clip(np.searchsorted(af, allf, side="right") - 1, 0, len(af) - 1)
    nxt = np.clip(np.searchsorted(af, allf), 0, len(af) - 1)
    slot_ff = asl[prev]
    nearest_d = np.minimum(np.abs(allf - af[prev]), np.abs(allf - af[nxt]))
    conf = np.where(nearest_d <= high_dist, "high", "low")
    return slot_ff, conf


def confident_anchors(frames, slots, margin, margin_thr: float):
    """Select (frame, slot) rows with ``slot >= 0`` and ``margin > margin_thr``, sorted by frame."""
    frames = np.asarray(frames)
    slots = np.asarray(slots, dtype=int)
    margin = np.asarray(margin, dtype=float)
    keep = (slots >= 0) & (margin > margin_thr)
    order = np.argsort(frames[keep])
    return frames[keep][order], slots[keep][order]


def female_slot_from_probs(p_slots):
    """``p_slots``: (3,) P(female) per geometry slot (NaN = slot absent). Return (female_slot, margin).

    female_slot = argmax over present slots; margin = top1 - top2 (NaN if < 2 slots present, -1 slot if
    none present). Mirrors the per-frame female call in step2_per_frame_female.
    """
    arr = np.asarray(p_slots, dtype=float)
    present = ~np.isnan(arr)
    if not present.any():
        return -1, np.nan
    filled = np.where(np.isnan(arr), -1.0, arr)
    fs = int(filled.argmax())
    if present.sum() >= 2:
        srt = np.sort(filled)
        margin = float(srt[-1] - srt[-2])
    else:
        margin = np.nan
    return fs, margin


def beta_from_probs(p_slots, female_slot: int):
    """``p_slots``: (3,) P(M2) per slot; ``female_slot``: the female's slot. Return (beta_slot, margin).

    Among the two NON-female slots, M2 (beta) = argmax P(M2); margin = |diff|. (-1, NaN) if either male
    crop is absent. M1 (alpha) is the remaining non-female slot.
    """
    arr = np.asarray(p_slots, dtype=float)
    if female_slot < 0:
        return -1, np.nan
    others = [s for s in SLOTS if s != female_slot]
    p0, p1 = arr[others[0]], arr[others[1]]
    if np.isnan(p0) or np.isnan(p1):
        return -1, np.nan
    return (others[0], p0 - p1) if p0 >= p1 else (others[1], p1 - p0)


def resolve_conflict_beta(beta_ff, female_ff):
    """Where the (independently) forward-filled M2 slot collides with the female slot, fall back to a
    geometry default (gid_2 unless the female is there, else gid_0) and flag it. Return
    ``(beta_final, conflict_mask)``.
    """
    beta_ff = np.asarray(beta_ff, dtype=int)
    female_ff = np.asarray(female_ff, dtype=int)
    conflict = beta_ff == female_ff
    geo_beta = np.where(female_ff != 2, 2, 0)
    beta_final = np.where(conflict, geo_beta, beta_ff)
    return beta_final, conflict


def alpha_from(female_ff, beta_final):
    """The remaining (M1) slot: slots sum to 0+1+2 = 3, so alpha = 3 - female - beta."""
    return 3 - np.asarray(female_ff, dtype=int) - np.asarray(beta_final, dtype=int)


def resolve_male_track(per_frame_male: pd.DataFrame, female_ff, n_frames: int,
                       margin_thr: float, high_dist: int, pb_cols=("pb_slot0", "pb_slot1", "pb_slot2")):
    """Resolve M1/M2 to frame resolution from per-frame P(M2) per slot + the (dense) female slot.

    ``per_frame_male`` has columns ``frame`` + ``pb_cols`` (P(M2) per geometry slot). Returns a dict
    with ``beta_final`` (M2 slot per frame), ``alpha_final`` (M1 slot), ``male_conf`` ("high"/"low"),
    and the confident-anchor arrays (``anchor_frames``, ``anchor_betas``) for downstream fusion.
    """
    female_ff = np.asarray(female_ff, dtype=int)
    pb = per_frame_male[list(pb_cols)].to_numpy()
    mf = per_frame_male["frame"].to_numpy().astype(int)
    fs_at = female_ff[mf]
    beta_slot = np.full(len(per_frame_male), -1, dtype=int)
    margin = np.full(len(per_frame_male), np.nan)
    for i in range(len(per_frame_male)):
        b, m = beta_from_probs(pb[i], int(fs_at[i]))
        beta_slot[i], margin[i] = b, m
    af, abeta = confident_anchors(mf, beta_slot, margin, margin_thr)
    beta_ff_raw, male_conf = forward_fill_slot(af, abeta, n_frames, high_dist)
    beta_final, conflict = resolve_conflict_beta(beta_ff_raw, female_ff)
    male_conf = np.where(conflict, "low", male_conf)
    return {"beta_final": beta_final, "alpha_final": alpha_from(female_ff, beta_final),
            "male_conf": np.asarray(male_conf, dtype=object),
            "anchor_frames": af, "anchor_betas": abeta}


def resolve_female_track(per_frame_female: pd.DataFrame, n_frames: int, margin_thr: float,
                         high_dist: int, p_cols=("p_slot0", "p_slot1", "p_slot2")):
    """Resolve the female slot to frame resolution from per-frame P(female) per slot (step2 female).

    ``per_frame_female`` has columns ``frame`` + ``p_cols``. Returns ``female_ff`` (female slot per
    frame), ``app_conf`` ("high"/"low"), and the confident anchors.
    """
    p = per_frame_female[list(p_cols)].to_numpy()
    fr = per_frame_female["frame"].to_numpy().astype(int)
    fslot = np.full(len(p), -1, dtype=int)
    marg = np.full(len(p), np.nan)
    for i in range(len(p)):
        fslot[i], marg[i] = female_slot_from_probs(p[i])
    af, asl = confident_anchors(fr, fslot, marg, margin_thr)
    female_ff, conf = forward_fill_slot(af, asl, n_frames, high_dist)
    return {"female_ff": female_ff, "app_conf": np.asarray(conf, dtype=object),
            "anchor_frames": af, "anchor_slots": asl}


def top_male_anchors(per_frame_top: pd.DataFrame, female_ff, margin_thr: float,
                     tm_cols=("tm_slot0", "tm_slot1", "tm_slot2")):
    """Confident TOP M2 anchors (frames, betas) for :func:`fuse_side_top`, from per-frame top P(M2)."""
    female_ff = np.asarray(female_ff, dtype=int)
    tm = per_frame_top[list(tm_cols)].to_numpy()
    tf = per_frame_top["frame"].to_numpy().astype(int)
    tbeta = np.full(len(tm), -1, dtype=int)
    tmarg = np.full(len(tm), np.nan)
    for i in range(len(tm)):
        tbeta[i], tmarg[i] = beta_from_probs(tm[i], int(female_ff[tf[i]]))
    return confident_anchors(tf, tbeta, tmarg, margin_thr)


def fuse_side_top(side_beta_ff, side_conf, top_anchor_frames, top_anchor_betas, female_ff,
                  n_frames: int, high_dist: int):
    """Side-primary, layered fusion: keep the side male resolution byte-identical where it is confident;
    fill ONLY the side-low frames with the top vote (when within ``high_dist`` of a confident top anchor).
    A blind side+top average degrades side-confident frames; this gate strictly adds coverage.

    Returns ``(beta_final, male_conf)`` after conflict resolution against the female slot.
    """
    side_beta_ff = np.asarray(side_beta_ff, dtype=int)
    side_conf = np.asarray(side_conf, dtype=object)
    female_ff = np.asarray(female_ff, dtype=int)
    is_side_high = side_conf == "high"
    top_beta_ff, _ = forward_fill_slot(top_anchor_frames, top_anchor_betas, n_frames, high_dist)
    allf = np.arange(n_frames)
    if len(top_anchor_frames):
        taf = np.sort(np.asarray(top_anchor_frames))
        tprev = np.clip(np.searchsorted(taf, allf, side="right") - 1, 0, len(taf) - 1)
        tnxt = np.clip(np.searchsorted(taf, allf), 0, len(taf) - 1)
        top_d = np.minimum(np.abs(allf - taf[tprev]), np.abs(allf - taf[tnxt]))
    else:
        top_d = np.full(n_frames, 1 << 30)
    top_fills = (~is_side_high) & (top_d <= high_dist) & (top_beta_ff >= 0)
    beta_ff = np.where(is_side_high, side_beta_ff, np.where(top_fills, top_beta_ff, side_beta_ff))
    male_conf = np.where(is_side_high | top_fills, "high", "low")
    beta_final, conflict = resolve_conflict_beta(beta_ff, female_ff)
    male_conf = np.where(conflict, "low", male_conf)
    return beta_final, np.asarray(male_conf, dtype=object)


def assemble_individual_labels(frames_arr, slots_arr, female_ff, beta_final, n_frames: int):
    """Per (frame, geometry_slot) -> individual_id in {F, M1, M2}. ``frames_arr``/``slots_arr`` are the
    row-aligned frame and geometry_slot of an identified-3D table; female/beta are dense per-frame slots.
    """
    female_ff = np.asarray(female_ff, dtype=int)
    beta_final = np.asarray(beta_final, dtype=int)
    slot_label = np.empty((n_frames, 3), dtype=object)
    for s in SLOTS:
        slot_label[:, s] = np.where(s == female_ff, "F", np.where(s == beta_final, "M2", "M1"))
    return slot_label[np.asarray(frames_arr), np.asarray(slots_arr, dtype=int)]


def _runs_of(mask):
    """Maximal runs of True in a boolean array -> list of (start, end) inclusive."""
    idx = np.where(np.asarray(mask))[0]
    if not len(idx):
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(r[0]), int(r[-1])) for r in np.split(idx, splits)]


def bridge_low_spans(slot_ff, conf, n_frames: int):
    """Classify each low-confidence run by its bounding confident anchors (Goal 4 interpolation).

    A run of non-"high" frames between two "high" anchors with the SAME slot is **bridged** — the
    identity is certain (no swap occurred), it merely lacked a nearby confident frame. A run between
    anchors with DIFFERENT slots is **ambiguous** (a swap happened somewhere in the gap). A run touching
    the timeline start/end (only one bounding anchor) is **edge** (fillable by the nearest anchor).

    ``slot_ff`` is the forward-filled slot (already = the preceding anchor's slot in each run). Returns
    ``(id_source, runs)``: ``id_source`` is an (n_frames,) array in {"high","bridged","edge","ambiguous"};
    ``runs`` is a list of ``(start, end, kind, before_slot, after_slot)``.
    """
    slot_ff = np.asarray(slot_ff, dtype=int)
    conf = np.asarray(conf, dtype=object)
    hi = conf == "high"
    id_source = np.where(hi, "high", "ambiguous").astype(object)
    runs = []
    for (s, e) in _runs_of(~hi):
        before = int(slot_ff[s - 1]) if s - 1 >= 0 and hi[s - 1] else -1
        after = int(slot_ff[e + 1]) if e + 1 < n_frames and hi[e + 1] else -1
        if before >= 0 and after >= 0:
            kind = "bridged" if before == after else "ambiguous"
        else:
            kind = "edge"
        id_source[s:e + 1] = kind
        runs.append((s, e, kind, before, after))
    return id_source, runs


def localize_ambiguous_swaps(slot_ff, runs, pair_dist):
    """Place the swap inside each **ambiguous** run at the frame of closest 3D approach between the two
    slots involved (the crossing). ``pair_dist`` maps an unordered slot pair frozenset({a,b}) -> an
    (n_frames,) distance array. Within a run [s,e] with before-slot ``a`` and after-slot ``b`` (a!=b),
    forward-fill already set the whole run to ``a``; this sets [t*..e] to ``b`` where t* = argmin
    distance(a,b) over [s,e]. Returns updated ``slot_ff`` (copy) and the list of localized swap frames.
    """
    slot_ff = np.asarray(slot_ff, dtype=int).copy()
    swaps = []
    for (s, e, kind, a, b) in runs:
        if kind != "ambiguous" or a < 0 or b < 0 or a == b:
            continue
        key = frozenset((a, b))
        if key not in pair_dist:
            continue
        seg = np.asarray(pair_dist[key])[s:e + 1]
        finite = np.isfinite(seg)
        if not finite.any():
            continue
        seg = np.where(finite, seg, np.inf)
        tstar = s + int(np.argmin(seg))     # crossing = closest approach
        slot_ff[tstar:e + 1] = b
        swaps.append(tstar)
    return slot_ff, swaps


def female_dense_from_track(track: pd.DataFrame, n_frames: int, slot_col="app_female_slot"):
    """Dense per-frame female slot from an identified-3D table that already carries ``slot_col``."""
    female_ff = np.full(n_frames, -1, dtype=int)
    pf = track.drop_duplicates("frame")
    female_ff[pf.frame.to_numpy()] = pf[slot_col].to_numpy().astype(int)
    return female_ff


def appearance_swap_across_event(slot_a: int, slot_b: int, start: int, end: int,
                                 label_fs, conf, search: int = 300):
    """Independent appearance adjudication of a hard-core event for the slot pair (a, b).

    A hard-core event over ``[start, end]`` is a place the 3D linker MIGHT have swapped the two slots.
    Read the per-frame appearance individual_id of slots a and b at the nearest HIGH-confidence frame
    before ``start`` and after ``end`` (within ``search`` frames): if the (a, b) labels are unchanged
    the appearance says NO swap; if they exchanged it says SWAP; otherwise (no confident neighbour
    within ``search``, or the labels neither stayed nor cleanly exchanged — e.g. only one of the pair
    moved, or one slot is the female) UNCERTAIN.

    ``label_fs``: (n_frames, 3) array of individual_id ("F"/"M1"/"M2") per (frame, geometry_slot).
    ``conf``: (n_frames,) array of "high"/"low". Returns
    ``(verdict, (before_a, before_b), (after_a, after_b))`` with verdict in {no_swap, swap, uncertain}.
    """
    n = len(conf)
    bi = next((f for f in range(min(start - 1, n - 1), max(start - 1 - search, -1), -1) if conf[f] == "high"), None)
    ai = next((f for f in range(max(end + 1, 0), min(end + 1 + search, n)) if conf[f] == "high"), None)
    if bi is None or ai is None:
        return "uncertain", None, None
    ba, bb = label_fs[bi, slot_a], label_fs[bi, slot_b]
    aa, ab = label_fs[ai, slot_a], label_fs[ai, slot_b]
    if ba == aa and bb == ab:
        verdict = "no_swap"
    elif ba == ab and bb == aa:
        verdict = "swap"
    else:
        verdict = "uncertain"
    return verdict, (ba, bb), (aa, ab)


def dense_label_and_conf(track: pd.DataFrame, n_frames: int,
                         id_col: str = "individual_id", conf_col: str = "male_conf"):
    """Build the (n_frames, 3) individual_id grid + (n_frames,) confidence from an individual-ID track
    (cols ``frame``, ``geometry_slot``, ``id_col``, ``conf_col``) for :func:`appearance_swap_across_event`.
    """
    label_fs = np.full((n_frames, 3), "", dtype=object)
    fr = track.frame.to_numpy().astype(int)
    sl = track.geometry_slot.to_numpy().astype(int)
    label_fs[fr, sl] = track[id_col].to_numpy()
    conf = np.array(["low"] * n_frames, dtype=object)
    pf = track.drop_duplicates("frame")
    conf[pf.frame.to_numpy().astype(int)] = pf[conf_col].to_numpy()
    return label_fs, conf


_ID_CODE = {"F": 0, "M1": 1, "M2": 2}
_ID_NAME = np.array(["F", "M1", "M2"], dtype=object)


def debounce_teleport_swaps(id_at_slot, pos_at_slot, jump_mm: float = 60.0, max_flicker: int = 166):
    """Revert physically-impossible brief id swaps in a resolved 3-fish track (motion continuity).

    A correct ``id_final`` follows each physical fish, so its 3D position is continuous. A "teleport"
    -- a fish's position jumping > ``jump_mm`` in a single frame (e.g. 200 mm at 166 fps = 33 m/s) --
    only happens when two ids are confidently assigned to the WRONG geometry slots at a near-crossing.
    This finds short runs (<= ``max_flicker`` frames) where the per-frame slot->id permutation deviates
    from a value it both LEAVES and RETURNS to (a flicker) AND the deviation caused a teleport, and
    reverts them to the bracketing permutation. Sustained (non-reverting) swaps are left untouched.

    ``id_at_slot``: (n_frames, 3) int id code (0=F,1=M1,2=M2) per (frame, geometry_slot) -- assumed a
    full permutation each frame (the track has exactly 1F+1M1+1M2/frame). ``pos_at_slot``:
    (n_frames, 3, 3) xyz per (frame, slot), NaN where a slot has no finite 3D position. Returns the
    corrected (n_frames, 3) ``id_at_slot`` and the list of reverted ``(start, end)`` spans (inclusive).
    """
    id_at_slot = np.asarray(id_at_slot, dtype=int)
    pos_at_slot = np.asarray(pos_at_slot, dtype=float)
    n = id_at_slot.shape[0]
    pos_by_id = np.full((n, 3, 3), np.nan)            # position of each id per frame
    rows = np.arange(n)
    for c in range(3):
        slot_c = np.argmax(id_at_slot == c, axis=1)
        pos_by_id[:, c, :] = pos_at_slot[rows, slot_c, :]
    perm = id_at_slot[:, 0] * 9 + id_at_slot[:, 1] * 3 + id_at_slot[:, 2]
    bnd = np.r_[0, np.where(np.diff(perm) != 0)[0] + 1, n]   # run boundaries [bnd[i], bnd[i+1])
    out = id_at_slot.copy()
    reverted = []
    for i in range(1, len(bnd) - 2):
        s, e = int(bnd[i]), int(bnd[i + 1])
        if perm[bnd[i - 1]] != perm[e] or (e - s) > max_flicker or s == 0:
            continue                                  # not bracketed by the same perm / too long
        jumps = np.linalg.norm(pos_by_id[s] - pos_by_id[s - 1], axis=1)
        if not (np.nanmax(jumps) > jump_mm):          # entry was continuous -> a real change, keep
            continue
        out[s:e] = id_at_slot[bnd[i - 1]]             # revert to the bracketing permutation
        reverted.append((s, e - 1))
    return out, reverted


def debounce_track_teleports(track, n_frames: int, id_col: str = "id_final",
                             jump_mm: float = 60.0, max_flicker: int = 166):
    """Apply :func:`debounce_teleport_swaps` to an individual-ID track DataFrame (cols ``frame``,
    ``geometry_slot``, ``id_col``, ``X``, ``Y``, ``Z``). Returns ``(new_id_per_row, reverted_spans)``
    where ``new_id_per_row`` is row-aligned to ``track`` (the corrected F/M1/M2 label)."""
    fr = track.frame.to_numpy().astype(int)
    sl = track.geometry_slot.to_numpy().astype(int)
    id_at_slot = np.full((n_frames, 3), -1, dtype=int)
    id_at_slot[fr, sl] = track[id_col].map(_ID_CODE).to_numpy()
    pos_at_slot = np.full((n_frames, 3, 3), np.nan)
    pos_at_slot[fr, sl, 0] = track.X.to_numpy()
    pos_at_slot[fr, sl, 1] = track.Y.to_numpy()
    pos_at_slot[fr, sl, 2] = track.Z.to_numpy()
    corrected, reverted = debounce_teleport_swaps(id_at_slot, pos_at_slot, jump_mm, max_flicker)
    return _ID_NAME[corrected[fr, sl]], reverted


# --------------------------------------------------------------------------- #
# Inference loop (I/O; produces the per-frame probability tables)             #
# --------------------------------------------------------------------------- #
def load_classifier(path, device: str):
    """Load a 2-class EmbeddingClassifier (female or male head). ``forward_head`` gives P(class 1):
    P(female) for the female model, P(M2=gid_2) for the male model."""
    import torch
    from sam2_id_experiment.src.id_model.resnet_embedder import EmbeddingClassifier, ResNet18Embedder
    ck = torch.load(path, map_location=device)
    m = EmbeddingClassifier(ResNet18Embedder(embedding_dim=int(ck.get("embedding_dim", 128)), pretrained=False), 2)
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval()


def forward_head(model, crops_uint8, device: str, batch_size: int = 512):
    """crops_uint8: (N,H,W,3) uint8 BGR -> (N,) P(class 1). Use as the ``forward_fn`` for
    :func:`per_frame_class_prob`: ``lambda m, c: forward_head(m, c, device)``."""
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(crops_uint8), batch_size):
            x = torch.from_numpy(crops_uint8[s:s + batch_size].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
            out.append(torch.softmax(model(x), 1)[:, 1].cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros(0)


def square_crop(img, bbox, size: int = 128, pad_frac: float = 0.15):
    """Production crop: mask-bbox (+pad) -> square -> resize, UNMASKED (``33._square_crop``).

    CRITICAL: this must stay mask-bbox->resize unmasked. A raw-128/masked crop collapses the classifier
    (AUC 0.982 -> ~0.5; see dual-line Phase 1b).
    """
    import cv2
    x0, y0, x1, y1 = [int(v) for v in bbox]
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = int(max(w, h) * (1 + pad_frac) / 2) + 1
    H, W = img.shape[:2]
    xa, xb = max(0, int(cx - half)), min(W, int(cx + half))
    ya, yb = max(0, int(cy - half)), min(H, int(cy + half))
    crop = img[ya:yb, xa:xb]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def link_gid(masks, points, gate: float = 40.0):
    """``masks``: [(mask_bool, bbox)]; ``points``: [(x,y,gid)] -> [(gid, mask_idx)].

    A mask gets the single tracklet point inside it; if none inside, the nearest point within ``gate``
    px of the mask centroid; masks with >=2 points inside are merged and dropped. (== ``33.link_frame``.)
    """
    out = []
    for mi, (m, _b) in enumerate(masks):
        H, W = m.shape
        inside = [g for x, y, g in points
                  if 0 <= int(round(y)) < H and 0 <= int(round(x)) < W and m[int(round(y)), int(round(x))]]
        if len(inside) == 1:
            out.append((inside[0], mi))
        elif len(inside) == 0:
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            cx, cy = xs.mean(), ys.mean()
            best, bd = None, gate
            for x, y, g in points:
                d = float(np.hypot(x - cx, y - cy))
                if d < bd:
                    best, bd = g, d
            if best is not None:
                out.append((best, mi))
    return out


def read_frame_masks(h5, frame: int):
    """Random-access one frame's masks from a chunk h5 (no full-chunk decompress)."""
    key = f"frame_{frame:06d}"
    if key not in h5:
        return []
    g = h5[key]
    return [(np.asarray(g[ik]["mask"][...], bool), tuple(int(v) for v in g[ik]["bbox"][...]))
            for ik in g if ik.startswith("instance_")]


def discover_chunks(mask_dir: Path, view: str):
    """Find ``<view>.masks_<S>-<E>.h5`` chunks -> sorted [((S,E), path)]."""
    import re
    out = []
    for p in Path(mask_dir).glob(f"{view}.masks_*.h5"):
        m = re.search(r"masks_(\d+)-(\d+)\.h5$", p.name)
        if m:
            out.append(((int(m.group(1)), int(m.group(2))), p))
    return sorted(out)


def _load_points(tracklet_csv: Path, lo: int, hi: int):
    df = pd.read_csv(tracklet_csv, usecols=["frame", "x", "y", "global_id"])
    df = df[(df.frame >= lo) & (df.frame <= hi) & df.global_id.isin(SLOTS)]
    pts: dict[int, list] = {}
    for f, x, y, g in zip(df.frame, df.x, df.y, df.global_id):
        pts.setdefault(int(f), []).append((float(x), float(y), int(g)))
    return pts


def per_frame_class_prob(views, mask_dir, video_paths, tracklet_csvs, model, forward_fn, *,
                         start: int, end: int, stride: int = 10, crop_size: int = 128,
                         gate: float = 40.0, batch: int = 4096, prob_prefix: str = "p"):
    """Run ``model`` per frame on mask crops across [start,end], multiview-averaging P over ``views``.

    ``views``: list of view names. ``video_paths``/``tracklet_csvs``: {view: path}. ``forward_fn(model,
    crops_uint8)`` -> (N,) probabilities (e.g. P(female) or P(M2)). Returns a per-frame wide DataFrame:
    ``frame, <prefix>_slot0/1/2, n_slots`` (multiview-mean prob per geometry slot).
    """
    import cv2
    import h5py
    recs = {"frame": [], "gid": [], "p": []}
    for view in views:
        pts = _load_points(Path(tracklet_csvs[view]), start, end)
        chunks = {rng: h5py.File(p, "r") for rng, p in discover_chunks(mask_dir, view)
                  if not (rng[1] < start or rng[0] > end)}
        if not chunks:
            raise FileNotFoundError(
                f"no Detectron2 mask chunks '{view}.masks_<S>-<E>.h5' overlapping [{start},{end}] "
                f"in {mask_dir} (check --masks-dir points at the full-length saion fan-out for {view})")

        def chunk_h5(f):
            for (cs, ce), h in chunks.items():
                if cs <= f <= ce:
                    return h
            return None

        cap = cv2.VideoCapture(str(video_paths[view]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        buf_crops, buf_meta = [], []

        def flush():
            if not buf_crops:
                return
            p = forward_fn(model, np.asarray(buf_crops, np.uint8))
            for (fr, g), pv in zip(buf_meta, p):
                recs["frame"].append(fr); recs["gid"].append(g); recs["p"].append(float(pv))
            buf_crops.clear(); buf_meta.clear()

        f = start
        while f <= end:
            sampled = (f - start) % stride == 0
            ok, img = (cap.read() if sampled else (cap.grab(), None))
            if not ok:
                break
            if sampled and img is not None and f in pts:
                h5 = chunk_h5(f)
                if h5 is not None:
                    masks = read_frame_masks(h5, f)
                    if masks:
                        for gid, mi in link_gid(masks, pts[f], gate):
                            buf_crops.append(square_crop(img, masks[mi][1], crop_size))
                            buf_meta.append((f, gid))
                            if len(buf_crops) >= batch:
                                flush()
            f += 1
        flush()
        cap.release()
        for h in chunks.values():
            h.close()

    df = pd.DataFrame(recs)
    pivot = df.groupby(["frame", "gid"]).p.mean().reset_index()
    wide = pivot.pivot(index="frame", columns="gid", values="p")
    for g in SLOTS:
        if g not in wide.columns:
            wide[g] = np.nan
    wide = wide[list(SLOTS)]
    arr = wide.to_numpy()
    return pd.DataFrame({"frame": wide.index,
                         f"{prob_prefix}_slot0": arr[:, 0], f"{prob_prefix}_slot1": arr[:, 1],
                         f"{prob_prefix}_slot2": arr[:, 2], "n_slots": np.sum(~np.isnan(arr), axis=1)})
