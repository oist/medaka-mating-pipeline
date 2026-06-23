from __future__ import annotations

"""Swap-corrected, co-visible identity labeling for per-video seg-ID training.

The 3D-bind ``global_id`` is reliable WITHIN a window but its *naming* is permuted
BETWEEN the scattered training windows (measured: cross-window RAW accuracy ~chance
while Hungarian-REMAP ~1.0 -- the appearance persists, only the geometry label is
swapped). Pooling windows under the raw ``global_id`` therefore mixes individuals and
cripples the classifier.

This module fixes that without any manual labels:

  * :func:`select_covisible_mask` -- keep only frames where all ``num_fish`` fish are
    simultaneously present (co-visible), where the within-window 3D-bind identity is
    unambiguous;
  * :func:`align_windows_by_appearance` -- align each window's per-fish appearance
    centroids to a reference window via a Hungarian bijection, so the returned
    ``aligned_id`` names the same physical fish across all windows; windows whose fish
    do not separate (bad-apple, e.g. heavy-occlusion windows) are dropped to ``-1``.

The reference window is the one whose 3 fish are most separated in embedding space,
making it the most trustworthy anchor for the cross-window naming.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def select_covisible_mask(view, frame, gid, num_fish: int = 3) -> np.ndarray:
    """Boolean mask keeping crops in (view, frame) groups that contain all ``num_fish``
    distinct fish ids -- i.e. frames where every fish is cleanly present at once."""
    df = pd.DataFrame({"view": np.asarray(view), "frame": np.asarray(frame), "gid": np.asarray(gid)})
    nuniq = df.groupby(["view", "frame"]).gid.transform("nunique")
    return (nuniq >= num_fish).to_numpy()


def sex_track_from_window_anchors(frame, slot, window_female_slot, global_female_slot=None):
    """Per-row sex label for a 3D track by following the female's geometry slot.

    Because the 3D linker re-labels identities at crossings, the female occupies
    DIFFERENT geometry slots in different parts of the track. The appearance model
    anchors her slot only inside the sampled windows; this propagates those anchors to
    the whole track and labels the uncertainty honestly:

      * a frame inside an anchored window  -> that window's female slot   (conf ``high``);
      * a frame between two windows that AGREE on the slot                (conf ``med``);
      * a frame between two windows that DISAGREE (a linker swap happened in the gap, at
        an unknown frame) -> the nearer window's slot                     (conf ``low``);
      * before the first / after the last window -> nearest window's slot (conf ``med``).

    Parameters
    ----------
    frame, slot : (N,) arrays over the track rows (one row per fish per frame); ``slot``
        is the zero-based geometry slot (``fish_id ID_{k+1}`` -> ``k``).
    window_female_slot : list of ``(start, end, female_slot)`` inclusive spans.
    global_female_slot : fallback slot when there are no windows at all.

    Returns
    -------
    sex : (N,) array of ``"F"``/``"M"``.
    confidence : (N,) array of ``"high"``/``"med"``/``"low"``.
    """
    frame = np.asarray(frame)
    slot = np.asarray(slot)
    spans = sorted(window_female_slot, key=lambda t: t[0])
    if not spans:
        gf = 0 if global_female_slot is None else int(global_female_slot)
        return np.where(slot == gf, "F", "M"), np.full(len(frame), "low", dtype=object)
    starts = np.array([s for s, _, _ in spans])
    ends = np.array([e for _, e, _ in spans])
    fslots = np.array([fs for _, _, fs in spans])
    if global_female_slot is None:
        global_female_slot = int(np.bincount(fslots).argmax())

    uf = np.unique(frame)
    fem = np.empty(len(uf), dtype=int)
    conf = np.empty(len(uf), dtype=object)
    for i, f in enumerate(uf):
        inside = np.where((starts <= f) & (ends >= f))[0]
        if len(inside):
            fem[i], conf[i] = fslots[inside[0]], "high"
            continue
        prev = np.where(ends < f)[0]
        nxt = np.where(starts > f)[0]
        if len(prev) and len(nxt):
            ps, ns = fslots[prev[-1]], fslots[nxt[0]]
            if ps == ns:
                fem[i], conf[i] = ps, "med"
            else:  # a swap occurred somewhere in the gap; pick the nearer anchor
                near_prev = (f - ends[prev[-1]]) <= (starts[nxt[0]] - f)
                fem[i], conf[i] = (ps if near_prev else ns), "low"
        elif len(prev):
            fem[i], conf[i] = fslots[prev[-1]], "med"
        elif len(nxt):
            fem[i], conf[i] = fslots[nxt[0]], "med"
        else:
            fem[i], conf[i] = global_female_slot, "low"
    f2fem = dict(zip(uf.tolist(), fem.tolist()))
    f2conf = dict(zip(uf.tolist(), conf.tolist()))
    row_fem = np.array([f2fem[int(f)] for f in frame])
    sex = np.where(slot == row_fem, "F", "M")
    confidence = np.array([f2conf[int(f)] for f in frame], dtype=object)
    return sex, confidence


def _centroids(emb_sub: np.ndarray, gid_sub: np.ndarray, labels) -> np.ndarray:
    return np.stack([emb_sub[gid_sub == g].mean(axis=0) for g in labels])


def _min_pairwise_sep(cs: np.ndarray) -> float:
    n = len(cs)
    return float(min(np.linalg.norm(cs[i] - cs[j]) for i in range(n) for j in range(i + 1, n)))


def align_windows_by_appearance(emb, window, gid, num_fish: int = 3,
                                drop_sep_frac: float = 0.5, min_ref_sep: float = 0.1):
    """Align per-window ``gid`` naming to a reference window by appearance bijection.

    Parameters
    ----------
    emb : (N, D) array of L2-ish embeddings, one row per crop.
    window : (N,) window key per crop.
    gid : (N,) within-window fish id per crop, assumed in ``range(num_fish)``.
    drop_sep_frac : a window is dropped (aligned -> -1) if its fish are separated by
        less than ``drop_sep_frac`` of the reference window's separation.
    min_ref_sep : absolute floor on the reference window's fish separation. If even the
        best window is below this, the embedding does not separate the fish anywhere, so
        the result is flagged ``degenerate`` and everything is dropped (no silent guess).

    Returns
    -------
    aligned : (N,) int array of cross-window-consistent fish ids (``-1`` = dropped).
    info : dict with ``ref``, per-window ``seps``, ``dropped``, ``thr``, ``n_windows``,
        ``degenerate``.
    """
    emb = np.asarray(emb)
    window = np.asarray(window)
    gid = np.asarray(gid)
    labels = list(range(num_fish))
    wins = sorted(set(window.tolist()))
    aligned = np.full(len(gid), -1, dtype=int)

    # A window is usable only if every fish id is actually present in it -- otherwise its
    # centroids would contain NaN and the Hungarian bijection would be meaningless. Such
    # windows are dropped, not silently aligned.
    usable = [w for w in wins if all(np.any(gid[window == w] == g) for g in labels)]
    missing = [w for w in wins if w not in usable]
    if not usable:
        return aligned, {"ref": None, "seps": {}, "dropped": list(wins), "thr": 0.0,
                         "n_windows": len(wins), "degenerate": True}

    cen = {w: _centroids(emb[window == w], gid[window == w], labels) for w in usable}
    seps = {w: _min_pairwise_sep(cen[w]) for w in usable}
    ref = max(seps, key=seps.get)
    degenerate = seps[ref] < float(min_ref_sep)
    thr = drop_sep_frac * seps[ref]
    dropped = list(missing)

    if degenerate:
        dropped = list(wins)
    else:
        refc = cen[ref]
        for w in usable:
            if seps[w] < thr:
                dropped.append(w)
                continue
            cost = np.linalg.norm(cen[w][:, None, :] - refc[None, :, :], axis=2)  # (gid_w, gid_ref)
            ri, ci = linear_sum_assignment(cost)
            mapping = {labels[r]: labels[c] for r, c in zip(ri, ci)}
            sel = window == w
            aligned[sel] = np.array([mapping[g] for g in gid[sel]], dtype=int)
    info = {"ref": ref, "seps": seps, "dropped": dropped, "thr": float(thr),
            "n_windows": len(wins), "degenerate": bool(degenerate)}
    return aligned, info
