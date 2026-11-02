"""Kinematic features of one named 3-fish trial, summarised on BORIS bouts.

Nodes are mouth, pectoral and tail, in millimetres. A frame the identity pass did not
accept (unwatched, unusable, or before the analysis start) is NaN. Nothing here fills
those frames in. Hand-marked wrong poses (B) are not used: those frames count as normal.

``follow`` is omitted from the comparison: it is most of the annotation and is not the
question. Those frames are also left out of the baseline, so baseline windows are time
with no BORIS annotation at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

def _nanmin(values, axis=None):
    """``np.nanmin`` without the warning an all-NaN slice raises."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(values, axis=axis)


def _nanmean(values, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


NAMES = ("F", "M1", "M2")
# Ordered pairs used for directed features. The first fish is the one being described.
DIRECTED = (("M1", "F"), ("M2", "F"), ("F", "M1"), ("F", "M2"), ("M1", "M2"), ("M2", "M1"))
SYMMETRIC = (("F", "M1"), ("F", "M2"), ("M1", "M2"))
BOUT_COLUMNS = ("dist", "speed_actor", "speed_target", "closing", "heading", "orbit",
                "forward", "lateral", "mouth", "bend")
# The only behaviours the comparison draws or scores. Male-male and male-female attacks are
# both kept; the target fish distinguishes them. Everything else, including follow, stays
# out of the comparison and out of the baseline's complement: its frames are still excluded
# from baseline windows.
FOCUS_BEHAVIORS = ("quick circle", "incomplete qc", "attack", "attempted att", "wrap", "spawn")
BEHAVIOR_LABEL = {"quick circle": "quick circle", "incomplete qc": "incomplete quick circle",
                  "attack": "attack", "attempted att": "attempted attack", "wrap": "wrap", "spawn": "spawn"}
# K-group BORIS files use another ethogram; the experimenter maps Courtship to quick circle and
# Dash to attack. Everything else (Chase, Touch, Jerk, ...) keeps its own name and stays out of
# the six. Lower-casing makes ``Wrap`` and ``Spawn`` the same behaviours as P's.
BEHAVIOR_ALIAS = {"courtship": "quick circle", "dash": "attack"}


def normalise_behaviors(bouts: pd.DataFrame) -> pd.DataFrame:
    """Lower-case behaviour names and apply ``BEHAVIOR_ALIAS``."""
    out = bouts.copy()
    names = out["behavior"].astype(str).str.strip().str.lower()
    out["behavior"] = names.replace(BEHAVIOR_ALIAS)
    return out


MALE_MALE = ("attack", "attempted att")
SKIP_BEHAVIORS = ("follow",)
SHORT_BOUT_S = 0.2          # shorter than this (and point events): features use the pad before the key
SHORT_PAD_S = 0.5
COMPETITION_AFTER_S = 2.0
BASELINE_LIMIT = 200
TRACE_BIN = 16              # ~10 Hz at 166 fps, for the exported traces


def centroids(nodes: np.ndarray) -> np.ndarray:
    """Mean of the finite nodes. ``nodes`` is (..., N, 3); the result drops the node axis."""
    mask = np.isfinite(nodes).all(axis=-1)
    total = np.where(mask[..., None], nodes, 0.0).sum(axis=-2)
    count = mask.sum(axis=-1)
    out = np.full(nodes.shape[:-2] + (3,), np.nan)
    good = count > 0
    out[good] = total[good] / count[good, None]
    return out


def headings(nodes: np.ndarray, mouth: int = 0, pectoral: int = 1) -> np.ndarray:
    """Unit vector from the pectoral node to the mouth. ``nodes`` (..., N, 3) -> (..., 3)."""
    v = nodes[..., mouth, :] - nodes[..., pectoral, :]
    norm = np.linalg.norm(v, axis=-1)
    out = np.full(v.shape, np.nan)
    good = norm > 1e-6
    out[good] = v[good] / norm[good, None]
    return out


def bend_deg(nodes: np.ndarray, mouth: int = 0, pectoral: int = 1, tail: int = 2) -> np.ndarray:
    """Angle at the pectoral node, in degrees. A straight fish is 180."""
    a = nodes[..., mouth, :] - nodes[..., pectoral, :]
    b = nodes[..., tail, :] - nodes[..., pectoral, :]
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = np.sum(a * b, axis=-1) / (na * nb)
    out = np.full(na.shape, np.nan)
    good = (na > 1e-6) & (nb > 1e-6)
    out[good] = np.degrees(np.arccos(np.clip(cos[good], -1.0, 1.0)))
    return out


def speeds(pos: np.ndarray, fps: float) -> np.ndarray:
    """``pos`` (F, K, 3) -> (F, K) mm/s. The first frame, and any step touching NaN, is NaN."""
    out = np.full(pos.shape[:2], np.nan)
    dist = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    good = np.isfinite(dist)
    out[1:][good] = dist[good] * fps
    return out


def named_nodes(nodes: np.ndarray, owner: np.ndarray, fish_ids: list[str],
                labels: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Track nodes ``(F, K, N, 3)`` rearranged to ``(F, 3, N, 3)`` in order F, M1, M2.

    ``owner[t, k]`` is the fish index whose data lies on track ``k`` (``ledger.owner_series``).
    ``labels`` maps that fish's id to a name. A frame is valid only when the three names
    sit on three different tracks; every other frame is NaN.
    """
    n = nodes.shape[0]
    index = {name: i for i, name in enumerate(NAMES)}
    by_fish = np.array([index.get(str(labels.get(fid, "")), -1) for fid in fish_ids])
    named = by_fish[np.asarray(owner)]
    out = np.full((n, 3, nodes.shape[2], 3), np.nan)
    valid = np.ones(n, dtype=bool)
    for slot in range(3):
        hit = named == slot
        count = hit.sum(axis=1)
        valid &= count == 1
        take = np.flatnonzero(count == 1)
        out[take, slot] = nodes[take, hit.argmax(axis=1)[take]]
    out[~valid] = np.nan
    return out, valid


def _angle_deg(heading: np.ndarray, direction: np.ndarray) -> np.ndarray:
    nh, nd = np.linalg.norm(heading, axis=-1), np.linalg.norm(direction, axis=-1)
    cos = np.sum(heading * direction, axis=-1) / (nh * nd)
    out = np.full(nh.shape, np.nan)
    good = (nh > 1e-6) & (nd > 1e-6) & np.isfinite(cos)
    out[good] = np.degrees(np.arccos(np.clip(cos[good], -1.0, 1.0)))
    return out


def body_offset(pos_other: np.ndarray, pos_self: np.ndarray, heading_self: np.ndarray):
    """Where ``other`` is in ``self``'s frame: forward along the heading, lateral to its right.

    Right is ``heading × up`` with up = +Z, so the sign is fixed by the tank axis.
    """
    right = np.cross(heading_self, np.array([0.0, 0.0, 1.0]))
    norm = np.linalg.norm(right, axis=-1)
    offset = pos_other - pos_self
    forward, lateral = np.full(len(offset), np.nan), np.full(len(offset), np.nan)
    good = (norm > 1e-6) & np.isfinite(offset).all(1) & np.isfinite(heading_self).all(1)
    forward[good] = np.sum(offset[good] * heading_self[good], axis=1)
    lateral[good] = np.sum(offset[good] * (right[good] / norm[good, None]), axis=1)
    return forward, lateral


def velocity_bearing_deg(pos_self: np.ndarray, pos_other: np.ndarray) -> np.ndarray:
    """Angle between ``self``'s horizontal velocity and the line toward ``other``.

    0° is straight toward the other fish, 90° is sideways, 180° is straight away.
    """
    vel = np.full_like(pos_self, np.nan)
    vel[1:] = pos_self[1:] - pos_self[:-1]
    return _angle_deg(vel[..., :2], (pos_other - pos_self)[..., :2])


def angle_swept_deg(orbit_deg_s: np.ndarray, fps: float, window_s: float = 1.0) -> np.ndarray:
    """Signed angle ``orbit_deg_s`` accumulates over the last ``window_s`` seconds.

    A full circle kept in one direction for one second is ±360°. A window that is
    more than half empty is left missing.
    """
    delta = np.asarray(orbit_deg_s, float) / fps
    n = max(1, int(round(window_s * fps)))
    finite = np.isfinite(delta)
    total = np.cumsum(np.where(finite, delta, 0.0))
    count = np.cumsum(finite.astype(int))
    out = np.full(len(delta), np.nan)
    if n >= len(delta):
        return out
    swept = total[n:] - total[:-n]
    covered = count[n:] - count[:-n]
    good = covered >= 0.5 * n
    out[n:][good] = swept[good]
    return out


def path_curvature_deg_per_mm(pos: np.ndarray, min_step_mm: float = 1.0, stride: int = 1) -> np.ndarray:
    """Turning of the horizontal velocity per millimetre travelled.

    A circle of radius R has curvature (180/π)/R. ``stride`` skips frames so a single
    noisy step is not read as a tight circle. Steps shorter than ``min_step_mm``
    are left missing.
    """
    stride = max(1, int(stride))
    step = pos[stride:, :2] - pos[:-stride, :2]
    dist = np.linalg.norm(step, axis=1)
    ang = np.arctan2(step[:, 1], step[:, 0])
    turn = np.degrees((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    out = np.full(len(pos), np.nan)
    good = (dist[:-1] >= min_step_mm) & (dist[1:] >= min_step_mm) & np.isfinite(turn)
    at = (np.flatnonzero(good) + 1) * stride
    keep = at < len(out)
    out[at[keep]] = np.abs(turn[good][keep]) / dist[1:][good][keep]
    return out


def orbit_rate(pos_center: np.ndarray, pos_orbiter: np.ndarray, fps: float) -> np.ndarray:
    """Signed horizontal angular rate of ``orbiter`` about ``center``, degrees per second."""
    rel = pos_orbiter[:, :2] - pos_center[:, :2]
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    finite = np.isfinite(ang) & (np.linalg.norm(rel, axis=1) > 1e-6)
    out = np.full(len(ang), np.nan)
    delta = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    good = finite[1:] & finite[:-1]
    out[1:][good] = np.degrees(delta[good]) * fps
    return out


def closing_speed(distance: np.ndarray, fps: float) -> np.ndarray:
    """Positive when the distance is shrinking, mm/s."""
    out = np.full(len(distance), np.nan)
    delta = np.diff(distance)
    good = np.isfinite(delta)
    out[1:][good] = -delta[good] * fps
    return out


def away_speed(pos_self: np.ndarray, pos_other: np.ndarray, fps: float) -> np.ndarray:
    """Positive when ``self`` is moving away from ``other``, mm/s."""
    rel = pos_other - pos_self
    dist = np.linalg.norm(rel, axis=-1)
    direction = np.divide(rel, dist[:, None], out=np.full_like(rel, np.nan), where=(dist > 1e-6)[:, None])
    vel = np.full_like(pos_self, np.nan)
    vel[1:] = (pos_self[1:] - pos_self[:-1]) * fps
    out = np.full(len(pos_self), np.nan)
    good = np.isfinite(vel).all(1) & np.isfinite(direction).all(1)
    out[good] = -np.sum(vel[good] * direction[good], axis=1)
    return out


def mouth_to_body(nodes: np.ndarray, actor: int, target: int, mouth: int) -> np.ndarray:
    """Distance from ``actor``'s mouth to the nearest finite node of ``target``."""
    point = nodes[:, actor, mouth, :]
    body = nodes[:, target]
    dist = np.linalg.norm(body - point[:, None, :], axis=-1)
    return _nanmin(dist, axis=1)


def kinematic_table(nodes: np.ndarray, frames: np.ndarray, fps: float, mouth: int = 0,
                    pectoral: int = 1, tail: int = 2) -> pd.DataFrame:
    """One row per frame. ``nodes`` is ``(F, 3, N, 3)`` in order F, M1, M2, already masked."""
    pos = centroids(nodes)
    head = headings(nodes, mouth, pectoral)
    bend = bend_deg(nodes, mouth, pectoral, tail)
    speed = speeds(pos, fps)
    slot = {name: i for i, name in enumerate(NAMES)}
    cols: dict[str, np.ndarray] = {"frame": np.asarray(frames)}
    for name in NAMES:
        cols[f"speed_{name}"] = speed[:, slot[name]]
        cols[f"bend_{name}"] = bend[:, slot[name]]
    for a, b in SYMMETRIC:
        dist = np.linalg.norm(pos[:, slot[a]] - pos[:, slot[b]], axis=-1)
        dist[~np.isfinite(pos[:, slot[a]]).all(1) | ~np.isfinite(pos[:, slot[b]]).all(1)] = np.nan
        cols[f"dist_{a}_{b}"] = dist
        cols[f"close_{a}_{b}"] = closing_speed(dist, fps)
    for a, b in DIRECTED:
        direction = pos[:, slot[b]] - pos[:, slot[a]]
        cols[f"head_{a}_to_{b}"] = _angle_deg(head[:, slot[a]], direction)
        forward, lateral = body_offset(pos[:, slot[a]], pos[:, slot[b]], head[:, slot[b]])
        cols[f"fwd_{a}_on_{b}"] = forward
        cols[f"lat_{a}_on_{b}"] = lateral
        cols[f"orbit_{a}_around_{b}"] = orbit_rate(pos[:, slot[b]], pos[:, slot[a]], fps)
        cols[f"vel_angle_{a}_to_{b}"] = velocity_bearing_deg(pos[:, slot[a]], pos[:, slot[b]])
        cols[f"sweep_{a}_around_{b}"] = angle_swept_deg(cols[f"orbit_{a}_around_{b}"], fps)
        cols[f"mouth_{a}_to_{b}"] = mouth_to_body(nodes, slot[a], slot[b], mouth)
    for name in NAMES:
        cols[f"curve_{name}"] = path_curvature_deg_per_mm(
            pos[:, slot[name]], stride=max(1, int(round(0.1 * fps))))
    for a, b in (("M1", "M2"), ("M2", "M1")):
        cols[f"away_{a}_from_{b}"] = away_speed(pos[:, slot[a]], pos[:, slot[b]], fps)
    return pd.DataFrame(cols)


def parse_who(who: str) -> tuple[str | None, str | None, bool]:
    """``M1->F`` -> (M1, F, directed); ``M1<->M2`` -> (M1, M2, mutual); ``F`` -> (F, None, directed)."""
    text = str(who)
    # ``<->`` contains ``->``, so the mutual form has to be tested first.
    if "<->" in text:
        actor, target = text.split("<->", 1)
        return actor, target, False
    if "->" in text:
        actor, target = text.split("->", 1)
        return actor, target, True
    if text in NAMES:
        return text, None, True
    return None, None, False


def _median(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan")


def _pair(actor: str, target: str) -> tuple[str, str]:
    """Column order of a symmetric pair (``dist_F_M1``, never ``dist_M1_F``)."""
    return (actor, target) if (actor, target) in SYMMETRIC or (actor, target) == ("F", "M1") else (
        (target, actor) if (target, actor) in SYMMETRIC else (actor, target))


def bout_features(table: pd.DataFrame, bouts: pd.DataFrame, fps: float, first: int,
                  skip: tuple[str, ...] = SKIP_BEHAVIORS,
                  keep: tuple[str, ...] = FOCUS_BEHAVIORS) -> pd.DataFrame:
    """One row per BORIS bout with the median of each feature over its window.

    A point event, or a bout shorter than ``SHORT_BOUT_S``, takes its features from the
    ``SHORT_PAD_S`` before the key as well: an attack key is shorter than the lunge.
    ``usable_frac`` is the share of that window whose actor-target distance (or the
    actor's speed, when the bout names one fish) is finite.
    """
    frames = table["frame"].to_numpy()
    rows = []
    for bout in bouts.itertuples(index=False):
        if bout.behavior in skip or bout.behavior not in keep:
            continue
        actor, target, directed = parse_who(bout.who)
        start, end = int(bout.start), int(bout.end)
        window = start
        if bool(bout.point) or (end - start + 1) / fps < SHORT_BOUT_S:
            window = max(first, start - int(round(SHORT_PAD_S * fps)))
        lo, hi = np.searchsorted(frames, window), np.searchsorted(frames, end, side="right")
        part = table.iloc[lo:hi]
        row = {"start": start, "end": end, "window_start": window, "seconds": round((end - start + 1) / fps, 3),
               "who": bout.who, "behavior": bout.behavior, "category": bout.category,
               "actor": actor or "", "target": target or "", "directed": directed, "n_frames": int(hi - lo)}
        if actor and target and directed:
            pair = _pair(actor, target)
            dist = part[f"dist_{pair[0]}_{pair[1]}"].to_numpy()
            row.update(dist=_median(dist), speed_actor=_median(part[f"speed_{actor}"]),
                       speed_target=_median(part[f"speed_{target}"]),
                       closing=_median(part[f"close_{pair[0]}_{pair[1]}"]),
                       heading=_median(part[f"head_{actor}_to_{target}"]),
                       orbit=_median(part[f"orbit_{actor}_around_{target}"]),
                       forward=_median(part[f"fwd_{actor}_on_{target}"]),
                       lateral=_median(part[f"lat_{actor}_on_{target}"]),
                       mouth=_median(part[f"mouth_{actor}_to_{target}"]),
                       bend=_median(part[f"bend_{actor}"]),
                       usable_frac=float(np.isfinite(dist).mean()) if len(dist) else float("nan"))
        elif actor and target and not directed:
            pair = _pair(actor, target)
            dist = part[f"dist_{pair[0]}_{pair[1]}"].to_numpy()
            row.update(dist=_median(dist),
                       speed_actor=_median(_nanmean([part[f"speed_{actor}"], part[f"speed_{target}"]], axis=0)),
                       speed_target=float("nan"), closing=_median(part[f"close_{pair[0]}_{pair[1]}"]),
                       heading=_median(_nanmin([part[f"head_{actor}_to_{target}"],
                                                part[f"head_{target}_to_{actor}"]], axis=0)),
                       orbit=_median(_nanmean([np.abs(part[f"orbit_{actor}_around_{target}"]),
                                               np.abs(part[f"orbit_{target}_around_{actor}"])], axis=0)),
                       forward=float("nan"), lateral=float("nan"),
                       mouth=_median(_nanmin([part[f"mouth_{actor}_to_{target}"],
                                             part[f"mouth_{target}_to_{actor}"]], axis=0)),
                       bend=_median(_nanmean([part[f"bend_{actor}"], part[f"bend_{target}"]], axis=0)),
                       usable_frac=float(np.isfinite(dist).mean()) if len(dist) else float("nan"))
        elif actor:
            speed = part[f"speed_{actor}"].to_numpy()
            nearest = _nanmin([part[f"dist_{_pair(actor, other)[0]}_{_pair(actor, other)[1]}"]
                               for other in NAMES if other != actor], axis=0) if actor == "F" else speed
            row.update(dist=_median(nearest) if actor == "F" else float("nan"), speed_actor=_median(speed),
                       speed_target=float("nan"), closing=float("nan"), heading=float("nan"),
                       orbit=float("nan"), forward=float("nan"), lateral=float("nan"), mouth=float("nan"),
                       bend=_median(part[f"bend_{actor}"]),
                       usable_frac=float(np.isfinite(speed).mean()) if len(speed) else float("nan"))
            if actor == "F":
                d1, d2 = _median(part["dist_F_M1"]), _median(part["dist_F_M2"])
                row["dist_F_M1"], row["dist_F_M2"] = d1, d2
                row["nearer_male"] = ("M1" if d1 < d2 else "M2") if np.isfinite([d1, d2]).all() else ""
        else:
            row.update({c: float("nan") for c in BOUT_COLUMNS})
            row["usable_frac"] = float("nan")
        rows.append(row)
    columns = ["start", "end", "window_start", "seconds", "who", "behavior", "category", "actor", "target",
               "directed", "n_frames", "usable_frac", *BOUT_COLUMNS, "dist_F_M1", "dist_F_M2", "nearer_male"]
    return pd.DataFrame(rows, columns=columns)


def _eligible(n: np.ndarray, bouts: pd.DataFrame, frames: np.ndarray) -> np.ndarray:
    """Frames whose features are finite and that no BORIS bout, follow included, covers."""
    ok = np.asarray(n, bool).copy()
    for bout in bouts.itertuples(index=False):
        lo, hi = np.searchsorted(frames, int(bout.start)), np.searchsorted(frames, int(bout.end), side="right")
        ok[lo:hi] = False
    return ok


def _window_medians(values: np.ndarray, eligible: np.ndarray, width: int, limit: int, seed: int) -> np.ndarray:
    width = max(1, int(width))
    starts, t = [], 0
    while t + width <= len(values):
        if eligible[t:t + width].all():
            starts.append(t)
            t += width
        else:
            t += 1
    if len(starts) > limit:
        starts = list(np.random.default_rng(seed).choice(starts, limit, replace=False))
    out = [np.nanmedian(values[s:s + width]) for s in starts if np.isfinite(values[s:s + width]).any()]
    return np.asarray(out, float)


def _auc(sample: np.ndarray, baseline: np.ndarray) -> float:
    """Probability that a bout value exceeds a baseline value. NaN below 3 of either."""
    a, b = sample[np.isfinite(sample)], baseline[np.isfinite(baseline)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    from scipy.stats import mannwhitneyu

    return float(mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)))


def separability(bouts: pd.DataFrame, table: pd.DataFrame, all_bouts: pd.DataFrame,
                 finite: np.ndarray, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per behaviour and feature: bout medians against duration-matched baseline windows.

    ``finite`` marks frames that may enter a baseline window. ``all_bouts`` includes the
    behaviours left out of the comparison, so their frames stay out of the baseline too.
    The second frame holds the baseline window values, so several days can be pooled
    without the per-frame tables. AUC is descriptive: one group's bouts are not
    independent trials.
    """
    frames = table["frame"].to_numpy()
    eligible = _eligible(finite, all_bouts, frames)
    rows, samples = [], []
    for behavior, group in bouts.groupby("behavior", sort=False):
        width = max(1, int(np.nanmedian((group["end"] - group["window_start"] + 1).to_numpy())))
        for column in BOUT_COLUMNS:
            sample, base = [], []
            keys = ["actor", "target", "directed"]
            for pair, sub in group.fillna({"actor": "", "target": ""}).groupby(keys, sort=False):
                series = frame_feature(table, column, pair[0], pair[1], bool(pair[2]))
                sample.append(sub[column].to_numpy(float))
                base.append(_window_medians(series, eligible, width, BASELINE_LIMIT, seed))
            sample = np.concatenate(sample) if sample else np.array([])
            base = np.concatenate(base) if base else np.array([])
            rows.append({"behavior": behavior, "feature": column, "n_bouts": int(np.isfinite(sample).sum()),
                         "n_baseline": int(base.size), "median_bout": _median(sample),
                         "median_baseline": _median(base), "auc": _auc(sample, base)})
            samples.extend({"behavior": behavior, "feature": column, "value": float(v)} for v in base)
    columns = ["behavior", "feature", "n_bouts", "n_baseline", "median_bout", "median_baseline", "auc"]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(samples, columns=["behavior", "feature", "value"])


def separability_from_bouts(bouts: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """The same summary as :func:`separability`, from saved bout rows and baseline windows."""
    rows = []
    for behavior, group in bouts.groupby("behavior", sort=False):
        for column in BOUT_COLUMNS:
            sample = group[column].to_numpy(float)
            base = samples[(samples["behavior"] == behavior) & (samples["feature"] == column)]["value"].to_numpy(float)
            rows.append({"behavior": behavior, "feature": column, "n_bouts": int(np.isfinite(sample).sum()),
                         "n_baseline": int(np.isfinite(base).sum()), "median_bout": _median(sample),
                         "median_baseline": _median(base), "auc": _auc(sample, base)})
    columns = ["behavior", "feature", "n_bouts", "n_baseline", "median_bout", "median_baseline", "auc"]
    return pd.DataFrame(rows, columns=columns)


def frame_feature(table: pd.DataFrame, column: str, actor: str, target: str, directed: bool) -> np.ndarray:
    """The per-frame series a bout median of ``column`` was taken from, for this pair.

    The baseline is drawn from this series, so a male-male distance is compared with
    male-male distance and a circle toward the female with that male's distance to her.
    """
    n = len(table)
    actor, target = str(actor), str(target)
    if target in ("", "None", "nan"):
        if column == "speed_actor" and actor in NAMES:
            return table[f"speed_{actor}"].to_numpy(float)
        if column == "bend" and actor in NAMES:
            return table[f"bend_{actor}"].to_numpy(float)
        if column == "dist" and actor == "F":
            return _nanmin([table["dist_F_M1"], table["dist_F_M2"]], axis=0)
        return np.full(n, np.nan)
    pair = _pair(actor, target)
    dist = table[f"dist_{pair[0]}_{pair[1]}"].to_numpy(float)
    if not directed:
        mutual = {"dist": dist, "closing": table[f"close_{pair[0]}_{pair[1]}"].to_numpy(float),
                  "speed_actor": _nanmean([table[f"speed_{actor}"], table[f"speed_{target}"]], axis=0),
                  "heading": _nanmin([table[f"head_{actor}_to_{target}"], table[f"head_{target}_to_{actor}"]], axis=0),
                  "orbit": _nanmean([np.abs(table[f"orbit_{actor}_around_{target}"]),
                                     np.abs(table[f"orbit_{target}_around_{actor}"])], axis=0),
                  "mouth": _nanmin([table[f"mouth_{actor}_to_{target}"],
                                    table[f"mouth_{target}_to_{actor}"]], axis=0),
                  "bend": _nanmean([table[f"bend_{actor}"], table[f"bend_{target}"]], axis=0)}
        return mutual.get(column, np.full(n, np.nan))
    name = {"dist": f"dist_{pair[0]}_{pair[1]}", "speed_actor": f"speed_{actor}",
            "speed_target": f"speed_{target}", "closing": f"close_{pair[0]}_{pair[1]}",
            "heading": f"head_{actor}_to_{target}", "orbit": f"orbit_{actor}_around_{target}",
            "forward": f"fwd_{actor}_on_{target}", "lateral": f"lat_{actor}_on_{target}",
            "mouth": f"mouth_{actor}_to_{target}", "bend": f"bend_{actor}"}[column]
    return table[name].to_numpy(float)


def competition(table: pd.DataFrame, bouts: pd.DataFrame, fps: float,
                after_s: float = COMPETITION_AFTER_S) -> pd.DataFrame:
    """Each male-male attack, chase or parallel circle, and where the female is afterwards.

    The male with the larger away-speed in the ``after_s`` after the bout is the one who
    moved off. ``delta_*`` is the female's median distance to that male in the after window
    minus her median distance during the bout: negative means she ended up closer.
    """
    frames = table["frame"].to_numpy()
    pad = int(round(after_s * fps))
    rows = []
    for bout in bouts.itertuples(index=False):
        if bout.behavior not in MALE_MALE:
            continue
        actor, target, directed = parse_who(bout.who)
        if {actor, target} != {"M1", "M2"}:
            continue
        lo, hi = np.searchsorted(frames, int(bout.window_start)), np.searchsorted(frames, int(bout.end), side="right")
        a0, a1 = np.searchsorted(frames, int(bout.end) + 1), np.searchsorted(frames, int(bout.end) + pad, side="right")
        during, after = table.iloc[lo:hi], table.iloc[a0:a1]
        off = {name: _median(after[f"away_{name}_from_{other}"]) for name, other in (("M1", "M2"), ("M2", "M1"))}
        if not np.isfinite(list(off.values())).all() or abs(off["M1"] - off["M2"]) < 1.0:
            retreater = stayer = ""
        elif off["M1"] > off["M2"]:
            retreater, stayer = "M1", "M2"
        else:
            retreater, stayer = "M2", "M1"
        def delta(male: str) -> float:
            pair = _pair("F", male)
            return _median(after[f"dist_{pair[0]}_{pair[1]}"]) - _median(during[f"dist_{pair[0]}_{pair[1]}"])

        rows.append({"start": int(bout.start), "end": int(bout.end), "behavior": bout.behavior, "who": bout.who,
                     "initiator": actor if directed else "", "retreater": retreater, "stayer": stayer,
                     "away_M1": off["M1"], "away_M2": off["M2"],
                     "delta_retreater": delta(retreater) if retreater else float("nan"),
                     "delta_stayer": delta(stayer) if stayer else float("nan"),
                     "after_frames": int(a1 - a0)})
    columns = ["start", "end", "behavior", "who", "initiator", "retreater", "stayer", "away_M1", "away_M2",
               "delta_retreater", "delta_stayer", "after_frames"]
    return pd.DataFrame(rows, columns=columns)


ALIGN_BEFORE_S, ALIGN_AFTER_S = 10.0, 6.0


def aligned(table: pd.DataFrame, bouts: pd.DataFrame, fps: float, before_s: float = ALIGN_BEFORE_S,
            after_s: float = ALIGN_AFTER_S, step: int = TRACE_BIN) -> pd.DataFrame:
    """Features aligned on the BORIS start, one row per bout, feature and time.

    Time is in seconds, negative before the key. Mutual bouts are skipped: they have no
    single actor-target feature. A bout of one fish (spawn) keeps the columns that are
    defined for it (her distance to the nearer male, her speed). Days are pooled by
    concatenating these rows and then :func:`summarise_aligned`.
    """
    frames = table["frame"].to_numpy()
    times = np.arange(-before_s, after_s + 1e-9, step / fps)
    offsets = np.round(times * fps).astype(int)
    rows = []
    directed = bouts[bouts["directed"].astype(bool)]
    for bout in directed.itertuples(index=False):
        idx = np.searchsorted(frames, int(bout.start) + offsets)
        inside = (idx >= 0) & (idx < len(frames))
        for column in ("dist", "speed_actor", "heading", "orbit", "forward", "mouth", "closing"):
            values = np.full(len(times), np.nan)
            series = frame_feature(table, column, bout.actor, bout.target, True)
            values[inside] = series[idx[inside]]
            for t, value in zip(times, values):
                rows.append({"behavior": bout.behavior, "feature": column, "start": int(bout.start),
                             "t": round(float(t), 4), "value": float(value) if np.isfinite(value) else np.nan})
    return pd.DataFrame(rows, columns=["behavior", "feature", "start", "t", "value"])


def summarise_aligned(aligned_rows: pd.DataFrame) -> pd.DataFrame:
    """Mean and SEM across bouts at each aligned time."""
    if aligned_rows.empty:
        return pd.DataFrame(columns=["behavior", "feature", "t", "mean", "sem", "n"])
    rows = []
    for (behavior, feature, t), group in aligned_rows.groupby(["behavior", "feature", "t"], sort=False):
        values = group["value"].to_numpy(float)
        n = int(np.isfinite(values).sum())
        mean = float(np.nanmean(values)) if n else float("nan")
        sem = float(np.nanstd(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append({"behavior": behavior, "feature": feature, "t": t, "mean": mean, "sem": sem, "n": n})
    return pd.DataFrame(rows)


def review_bins(frames: np.ndarray, status: np.ndarray, bin_size: int = TRACE_BIN) -> pd.DataFrame:
    """One review class per timeline bin.

    Codes are the identity-pass frame status: -1 before the analysis start, 0 not watched,
    1 or 2 watched and usable, 3 unusable. A bin takes the strongest class inside it, so a
    short discarded stretch is not averaged away.
    """
    frames, status = np.asarray(frames), np.asarray(status)
    rows = []
    for a in range(0, len(frames), bin_size):
        part = status[a:a + bin_size]
        if np.any(part == 3):
            code = "discarded"
        elif np.any((part == 1) | (part == 2)):
            code = "kept"
        elif np.any(part == 0):
            code = "unwatched"
        else:
            code = "before"
        rows.append({"frame": int(frames[a]), "review": code})
    return pd.DataFrame(rows, columns=["frame", "review"])


def decimate(table: pd.DataFrame, columns: list[str], bin_size: int = TRACE_BIN) -> pd.DataFrame:
    """Median of each column in non-overlapping bins, for the timeline."""
    frames = table["frame"].to_numpy()
    rows = []
    for a in range(0, len(table), bin_size):
        part = table.iloc[a:a + bin_size]
        row = {"frame": int(frames[a])}
        for column in columns:
            row[column] = _median(part[column].to_numpy(float))
        rows.append(row)
    return pd.DataFrame(rows)


TRACE_COLUMNS = ["dist_F_M1", "dist_F_M2", "dist_M1_M2", "speed_F", "speed_M1", "speed_M2",
                 "head_M1_to_F", "head_M2_to_F", "orbit_M1_around_F", "orbit_M2_around_F",
                 "mouth_M1_to_F", "mouth_M2_to_F", "bend_F", "bend_M1", "bend_M2",
                 "head_M1_to_M2", "head_M2_to_M1",
                 "vel_angle_M1_to_F", "vel_angle_M1_to_M2", "vel_angle_M2_to_F", "vel_angle_M2_to_M1",
                 "vel_angle_F_to_M1", "vel_angle_F_to_M2",
                 "sweep_M1_around_F", "sweep_M1_around_M2", "sweep_M2_around_F", "sweep_M2_around_M1",
                 "sweep_F_around_M1", "sweep_F_around_M2",
                 "curve_F", "curve_M1", "curve_M2"]
