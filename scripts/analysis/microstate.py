"""Shared code for the pair-state / Infomap notebooks (``archive/notebooks/ms0*.ipynb``).

Method after O'Shaughnessy et al., PRX Life 2, 043006 (2024): per fish pair the state is
``(D, theta_a, theta_b)`` (pectoral-point distance and two planar heading angles), binned,
turned into a lag-tau transition matrix and clustered with Infomap.

Rules enforced here, not in the notebooks:

* The split is by **group** and is frozen in ``split.csv``. A test-group trial cannot be
  loaded unless ``final=True`` is passed.
* Gaps are only interpolated inside a continuous segment; frames the review marked
  unusable ("blocked") are never filled and never bridged.
* Every figure is written as ``S<stage>_<nn>_<name>.png`` and listed in ``figure_index.csv``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter

NAMES = ("F", "M1", "M2")
PAIRS = (("F", "M1"), ("F", "M2"), ("M1", "M2"))
PAIR_NAMES = tuple(f"{a}-{b}" for a, b in PAIRS)
PAIR_SETS = {
    "F-M1": ["F-M1"],
    "F-M2": ["F-M2"],
    "M1-M2": ["M1-M2"],
    "F-M pooled": ["F-M1", "F-M2"],
}
MOUTH, PECT, TAIL = 0, 1, 2  # node order of the 3-node skeleton
OUT_NAME = "microstate_infomap"
SPLIT_SEED = 20261106
MIN_XY_MM = 1.0
TAU_GRID = (1, 2, 3, 4, 6, 8, 12, 17, 25, 35, 50, 70, 100, 166, 330, 664)

_EXP = re.compile(r"^(\d{8})_([A-Z]+_\d+)_(\d+)_FCMC$")


# ---------------------------------------------------------------- paths and split
def roots() -> dict:
    from scripts.analysis.behavior_dataset import resolve_roots

    config, review, out = resolve_roots(None)
    out = Path(out) / OUT_NAME
    (out / "_data").mkdir(parents=True, exist_ok=True)
    return {"config": Path(config), "review": Path(review), "out": out, "data": out / "_data"}


def parse_experiment(experiment: str) -> tuple[str, int]:
    """``20251225_K_1_2_FCMC`` -> (``K_1``, 2)."""
    m = _EXP.match(experiment)
    if not m:
        raise ValueError(f"not a standard experiment folder name: {experiment}")
    return m.group(2), int(m.group(3))


def ledger_experiments() -> list[str]:
    """Experiments with exactly one package that holds an identity-pass ledger."""
    review = roots()["review"]
    names = sorted({p.parents[2].name for p in review.glob("*/*/id_pass/human_events.csv")})
    return [n for n in names if _EXP.match(n)]


def split_table() -> pd.DataFrame:
    path = roots()["out"] / "split.csv"
    if not path.exists():
        raise FileNotFoundError("split.csv missing")
    return pd.read_csv(path)


def write_split(assignments: list[tuple[str, str]], rule: str) -> pd.DataFrame:
    """Replace ``split.csv``. ``assignments`` is ``(experiment, "dev"|"test")`` in the order to keep."""
    rows = []
    for experiment, split in assignments:
        if split not in ("dev", "test"):
            raise ValueError(split)
        group, day = parse_experiment(experiment)
        rows.append({"experiment": experiment, "group": group, "day": day, "split": split,
                     "rule": rule, "created": time.strftime("%Y-%m-%d %H:%M")})
    table = pd.DataFrame(rows)
    table.to_csv(roots()["out"] / "split.csv", index=False)
    return table


def freeze_split(groups: list[str], test_fraction: float = 0.25, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Group-level split used before 2026-11-06. A hand-written per-video split is left unchanged."""
    path = roots()["out"] / "split.csv"
    if path.exists() and "experiment" in pd.read_csv(path, nrows=0).columns:
        return split_table()
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["group", "split", "rule", "created"])
    new = sorted(set(groups) - set(old["group"]))
    if new:
        rank = sorted(new, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        if old.empty:
            n_test = max(1, round(test_fraction * len(rank))) if len(rank) > 1 else 0
        else:
            n_test = sum(int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), 16) % 4 == 0 for g in rank)
        rows = [{"group": g, "split": "test" if i < n_test else "dev",
                 "rule": f"sha256 rank, seed {seed}", "created": time.strftime("%Y-%m-%d %H:%M")}
                for i, g in enumerate(rank)]
        old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        old.to_csv(path, index=False)
    return old


def split_of(experiment: str) -> str:
    table = split_table()
    if "experiment" in table.columns:
        hit = table.loc[table["experiment"] == experiment, "split"]
        if hit.empty:
            raise KeyError(f"{experiment} is not in the frozen split")
        return str(hit.iloc[0])
    group, _ = parse_experiment(experiment)
    by_group = table.set_index("group")
    if group not in by_group.index:
        raise KeyError(f"group {group} is not in the frozen split")
    return str(by_group.loc[group, "split"])


def trials(split: str) -> list[str]:
    table = split_table()
    if "experiment" not in table.columns:
        return [e for e in ledger_experiments() if split_of(e) == split]
    have = set(ledger_experiments())
    return [e for e in table.loc[table["split"] == split, "experiment"] if e in have]


def require_access(experiment: str, final: bool) -> None:
    if split_of(experiment) == "test" and not final:
        raise PermissionError(f"{experiment} is in the test group; it is only read with final=True")


# ---------------------------------------------------------------- params and figures
def load_params() -> dict:
    p = roots()["data"] / "params.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def update_params(stage: str, values: dict) -> dict:
    params = load_params()
    params[stage] = values
    (roots()["data"] / "params.json").write_text(json.dumps(params, indent=2, default=float), encoding="utf-8")
    return params


def save_fig(fig, stage: int, number: int, name: str, notebook: str = "") -> Path:
    import matplotlib.pyplot as plt

    out = roots()["out"]
    fname = f"S{stage}_{number:02d}_{name}.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    index = out / "figure_index.csv"
    rows = pd.read_csv(index) if index.exists() else pd.DataFrame(columns=["file", "stage", "number", "name", "notebook", "written"])
    rows = rows[rows["file"] != fname]
    rows.loc[len(rows)] = [fname, stage, number, name, notebook, time.strftime("%Y-%m-%d %H:%M")]
    rows.sort_values("file").to_csv(index, index=False)
    plt.close(fig)
    return out / fname


# ---------------------------------------------------------------- loading one trial
def segment_ids(blocked: np.ndarray) -> np.ndarray:
    """Continuous-segment id of every frame; -1 on blocked frames."""
    rising = np.diff(np.concatenate([[False], blocked]).astype(np.int8)) == 1
    seg = np.cumsum(rising)
    seg[blocked] = -1
    return seg


def load_trial(experiment: str, node_key: str = "nodes3d", final: bool = False, keep_tracks: bool = False) -> dict:
    """Named (F, M1, M2) 3D nodes in mm, NaN on unusable frames, plus review bookkeeping."""
    require_access(experiment, final)
    from scripts.analysis import behavior_features as F
    from scripts.analysis.behavior_dataset import find_package
    from scripts.review import id_segments as S
    from scripts.review.ledger import (COVERAGE_COLUMNS, EVENT_COLUMNS, analysis_start, owner_series, read_csv,
                                       static_labels, swap_list, watched_intervals)

    package = find_package(roots()["review"], experiment)
    led = package / "id_pass"
    with np.load(package / "review_data.npz") as z:
        frames = np.asarray(z["frames"], int)
        fish_ids = [str(f) for f in z["fish_ids"]]
        node_names = [str(n) for n in z["node_names"]]
        nodes = np.asarray(z[node_key], float)
        xyz = np.asarray(z["xyz"], float)
        present = np.asarray(z["view_present"])
        interp = np.asarray(z["is_interpolated"])
    if node_names != ["mouth", "pectoral", "tail"]:
        raise RuntimeError(f"unexpected node order {node_names}")
    if not np.all(np.diff(frames) == 1):
        raise RuntimeError("frames are not contiguous")
    manifest = json.loads((package.parent / "proxy" / "proxy_manifest.json").read_text(encoding="utf-8"))
    fps = float(manifest["views"]["top_cam"]["source_fps"])
    events = read_csv(led / "human_events.csv", EVENT_COLUMNS)
    coverage = read_csv(led / "coverage.csv", COVERAGE_COLUMNS)
    labels = static_labels(events, fish_ids)
    start = analysis_start(events)
    swaps = swap_list(events, fish_ids)
    danger = S.danger_map(xyz, present, interp, fps)
    status = S.frame_status(int(frames[0]), len(frames), start, danger["danger"], danger["runs"],
                            watched_intervals(coverage), [f for f, _, _ in swaps], S.manual_ranges(events), fps)
    usable = np.isin(status, (S.WATCHED_SAFE, S.WATCHED_DANGER))
    owner = owner_series(len(fish_ids), swaps, frames)
    named, valid = F.named_nodes(nodes, owner, fish_ids, labels)
    blocked = ~usable | ~valid
    named[blocked] = np.nan
    out = {
        "experiment": experiment, "package": package, "fps": fps, "frames": frames, "fish_ids": fish_ids,
        "labels": labels, "named": named, "blocked": blocked, "seg": segment_ids(blocked),
        "usable_frac": float((~blocked).mean()), "swap_frames": [f for f, _, _ in swaps],
        "analysis_start": int(start), "node_key": node_key,
    }
    if keep_tracks:
        out["tracks"] = nodes
    return out


# ---------------------------------------------------------------- stage 1: skeleton cleaning
def skeleton_lengths(nodes: np.ndarray) -> dict:
    """Head-pectoral, pectoral-tail, head-tail lengths ``(T, 3)`` in the node units."""
    return {
        "head_pec": np.linalg.norm(nodes[:, :, MOUTH] - nodes[:, :, PECT], axis=-1),
        "pec_tail": np.linalg.norm(nodes[:, :, PECT] - nodes[:, :, TAIL], axis=-1),
        "head_tail": np.linalg.norm(nodes[:, :, MOUTH] - nodes[:, :, TAIL], axis=-1),
    }


def suggest_k(ratio: np.ndarray, rel: float = 0.01, step: float = 0.05) -> tuple[float, float]:
    """Valley estimate on the histogram of length/median: first bin beside the main peak whose smoothed
    density drops below ``rel`` of the peak, going up (long tail) and down (collapsed skeleton)."""
    r = ratio[np.isfinite(ratio)]
    edges = np.arange(0, 4 + step, step)
    h, _ = np.histogram(r, bins=edges, density=True)
    h = np.convolve(h, np.ones(3) / 3, mode="same")
    mode = int(np.argmax(h))
    up = next((i for i in range(mode, len(h)) if h[i] <= rel * h[mode]), len(h) - 1)
    dn = next((i for i in range(mode, -1, -1) if h[i] <= rel * h[mode]), 0)
    return float(edges[dn]), float(edges[up + 1])


def clean_skeleton(nodes: np.ndarray, k_lo: dict, k_hi: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """NaN the whole skeleton of a fish on frames where head-pec or pec-tail leaves ``[k_lo, k_hi] * median``.

    Returns ``(cleaned, removed (T,3), medians)``. Medians are per fish within this trial.
    """
    L = skeleton_lengths(nodes)
    med = {k: np.nanmedian(L[k], axis=0) for k in ("head_pec", "pec_tail", "head_tail")}
    bad = np.zeros(L["head_pec"].shape, bool)
    for k in ("head_pec", "pec_tail"):
        with np.errstate(invalid="ignore"):
            bad |= (L[k] > k_hi[k] * med[k]) | (L[k] < k_lo[k] * med[k])
    out = nodes.copy()
    for f in range(3):
        out[bad[:, f], f] = np.nan
    return out, bad, med


# ---------------------------------------------------------------- stage 2/3: gaps and smoothing
def nan_runs(isnan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.concatenate([[0], isnan.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def fill_gaps(x: np.ndarray, limit: int, blocked: np.ndarray) -> np.ndarray:
    """Linear fill of NaN runs of at most ``limit`` frames, per column, never across ``blocked`` frames."""
    shape = x.shape
    flat = x.reshape(len(x), -1).copy()
    ref = x.reshape(len(x), -1)
    csum = np.concatenate([[0], np.cumsum(blocked)])
    n = len(x)
    for c in range(flat.shape[1]):
        starts, ends = nan_runs(~np.isfinite(ref[:, c]))
        for s, e in zip(starts, ends):
            if s == 0 or e == n or e - s > limit or csum[e] - csum[s] > 0:
                continue
            t = np.arange(s, e)
            flat[s:e, c] = np.interp(t, [s - 1, e], [ref[s - 1, c], ref[e, c]])
    return flat.reshape(shape)


def gap_lengths(nodes: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """Lengths of NaN runs per fish (skeleton-level) that are not touching a blocked frame."""
    out = []
    csum = np.concatenate([[0], np.cumsum(blocked)])
    for f in range(nodes.shape[1]):
        miss = ~np.isfinite(nodes[:, f]).all(axis=(1, 2))
        s, e = nan_runs(miss)
        out.extend(int(b - a) for a, b in zip(s, e) if a > 0 and b < len(miss) and csum[b] - csum[a] == 0)
    return np.asarray(out, int)


def smooth_sg(nodes: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Savitzky-Golay inside continuous finite runs of each fish; half a window is dropped at both ends."""
    if window <= order or window % 2 == 0:
        raise ValueError("window must be odd and larger than the polynomial order")
    out = np.full_like(nodes, np.nan)
    h = window // 2
    for f in range(nodes.shape[1]):
        ok = np.isfinite(nodes[:, f]).all(axis=(1, 2))
        starts, ends = nan_runs(ok)  # runs of True = finite runs
        for a, b in zip(starts, ends):
            if b - a < window:
                continue
            seg = nodes[a:b, f].reshape(b - a, -1)
            y = savgol_filter(seg, window, order, axis=0, mode="interp").reshape(b - a, *nodes.shape[2:])
            out[a + h:b - h, f] = y[h:b - a - h]
    return out


# ---------------------------------------------------------------- stage 4: pair coordinates
def signed_planar_angle(heading: np.ndarray, line: np.ndarray, min_xy: float = MIN_XY_MM) -> np.ndarray:
    """Signed XY angle (rad, ccw positive) from ``heading`` to ``line``; NaN if either is shorter than ``min_xy``."""
    h, r = heading[..., :2], line[..., :2]
    nh, nr = np.linalg.norm(h, axis=-1), np.linalg.norm(r, axis=-1)
    ang = np.arctan2(h[..., 0] * r[..., 1] - h[..., 1] * r[..., 0], np.sum(h * r, axis=-1))
    out = np.full(nh.shape, np.nan)
    good = (nh >= min_xy) & (nr >= min_xy) & np.isfinite(ang)
    out[good] = ang[good]
    return out


def pair_coordinates(nodes: np.ndarray) -> dict:
    """``{pair: (D_mm, theta_a, theta_b)}``; theta in rad, 0 = facing the other fish, +-pi = facing away."""
    slot = {n: i for i, n in enumerate(NAMES)}
    pec = nodes[:, :, PECT]
    heading = nodes[:, :, MOUTH] - pec
    out = {}
    for a, b in PAIRS:
        ia, ib = slot[a], slot[b]
        delta = pec[:, ib] - pec[:, ia]
        dist = np.linalg.norm(delta, axis=-1)
        fin = np.isfinite(pec[:, ia]).all(1) & np.isfinite(pec[:, ib]).all(1)
        out[f"{a}-{b}"] = (np.where(fin, dist, np.nan), signed_planar_angle(heading[:, ia], delta),
                           signed_planar_angle(heading[:, ib], -delta))
    return out


# ---------------------------------------------------------------- whole chain for one trial
def default_params() -> dict:
    p = load_params()
    need = ["stage1", "stage2", "stage3"]
    miss = [k for k in need if k not in p]
    if miss:
        raise RuntimeError(f"params.json lacks {miss}: run ms01 first")
    return p


def process_trial(experiment: str, params: dict | None = None, final: bool = False) -> dict:
    """Stages 1-4 with the frozen parameters. Returns smoothed nodes, pair coordinates in mm and body length."""
    params = params or default_params()
    tr = load_trial(experiment, final=final)
    s1, s2, s3 = params["stage1"], params["stage2"], params["stage3"]
    nodes, removed, med = clean_skeleton(tr["named"], s1["k_lo"], s1["k_hi"])
    nodes = fill_gaps(nodes, int(s2["fill_limit"]), tr["blocked"])
    nodes = smooth_sg(nodes, int(s3["window"]), int(s3.get("order", 2)))
    bl = med["head_tail"]
    pairs = pair_coordinates(nodes)
    return {"experiment": experiment, "fps": tr["fps"], "seg": tr["seg"], "blocked": tr["blocked"], "nodes": nodes,
            "body_length_mm": bl, "pairs": pairs, "removed_frac": removed.mean(0), "usable_frac": tr["usable_frac"],
            "frames": tr["frames"]}


def pair_units(proc: dict, pair: str, unit: str = "BL") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    D, ta, tb = proc["pairs"][pair]
    if unit == "BL":
        a, b = pair.split("-")
        bl = proc["body_length_mm"]
        D = D / (0.5 * (bl[NAMES.index(a)] + bl[NAMES.index(b)]))
    return D, ta, tb


# ---------------------------------------------------------------- stage 5/6: states and transitions
def bin_states(D: np.ndarray, ta: np.ndarray, tb: np.ndarray, n: int, dmax: float) -> np.ndarray:
    """State id in ``[0, n**3)``; -1 where any coordinate is missing or D >= dmax."""
    ok = np.isfinite(D) & np.isfinite(ta) & np.isfinite(tb) & (D < dmax)
    di = np.clip(np.floor(np.where(ok, D, 0) / dmax * n).astype(int), 0, n - 1)
    ai = np.clip(np.floor((np.where(ok, ta, 0) + np.pi) / (2 * np.pi) * n).astype(int), 0, n - 1)
    bi = np.clip(np.floor((np.where(ok, tb, 0) + np.pi) / (2 * np.pi) * n).astype(int), 0, n - 1)
    return np.where(ok, (di * n + ai) * n + bi, -1)


def transition_counts(states: np.ndarray, seg: np.ndarray, tau: int, n_states: int) -> tuple[sparse.csr_matrix, int]:
    """All (t, t+tau) with both states valid and both frames in the same continuous segment."""
    i, j = states[:-tau], states[tau:]
    ok = (i >= 0) & (j >= 0) & (seg[:-tau] == seg[tau:]) & (seg[:-tau] >= 0)
    m = sparse.coo_matrix((np.ones(ok.sum(), np.float64), (i[ok], j[ok])), shape=(n_states, n_states)).tocsr()
    return m, int(ok.sum())


def pooled_counts(items: list[tuple[np.ndarray, np.ndarray]], tau: int, n_states: int):
    total = sparse.csr_matrix((n_states, n_states))
    npair = 0
    for states, seg in items:
        m, k = transition_counts(states, seg, tau, n_states)
        total = total + m
        npair += k
    return total, npair


def infomap_labels(counts: sparse.csr_matrix, trials: int = 10, seed: int = 0) -> np.ndarray:
    """Cluster id per state (-1 for states never visited) on the row-normalised directed weighted graph."""
    import random

    import igraph as ig

    c = counts.tocsr()
    row = np.asarray(c.sum(axis=1)).ravel()
    P = sparse.diags(1.0 / np.where(row > 0, row, 1)) @ c
    P = P.tocoo()
    used = np.flatnonzero((np.asarray(c.sum(axis=0)).ravel() + row) > 0)
    remap = -np.ones(c.shape[0], int)
    remap[used] = np.arange(len(used))
    g = ig.Graph(n=len(used), edges=list(zip(remap[P.row].tolist(), remap[P.col].tolist())), directed=True)
    g.es["weight"] = P.data.tolist()
    random.seed(seed)
    membership = g.community_infomap(edge_weights="weight", trials=trials).membership
    labels = -np.ones(c.shape[0], int)
    labels[used] = membership
    return labels


def _infomap_job(args):
    counts, trials, seed = args
    return infomap_labels(counts, trials, seed)


def infomap_parallel(jobs: list, workers: int = 12) -> list[np.ndarray]:
    """Run ``(counts, trials, seed)`` jobs in separate processes (Infomap is single-threaded)."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return list(ex.map(_infomap_job, jobs))


def cluster_metrics(labels: np.ndarray, state_frames: np.ndarray) -> dict:
    used = labels >= 0
    k = labels[used]
    n_clusters = int(len(np.unique(k)))
    sizes = np.bincount(k) if len(k) else np.array([0])
    fr = np.bincount(k, weights=state_frames[used]) if len(k) else np.array([0.0])
    return {
        "n_clusters": n_clusters, "occupied_states": int(used.sum()),
        "largest_state_frac": float(sizes.max() / max(used.sum(), 1)),
        "largest_frame_frac": float(fr.max() / max(fr.sum(), 1)),
        "mean_states_per_cluster": float(used.sum() / max(n_clusters, 1)),
        "singletons": int((sizes == 1).sum()),
    }


def frame_nmi(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    """NMI between two state labellings, weighted by frame counts per state, on states labelled in both."""
    ok = (a >= 0) & (b >= 0) & (weights > 0)
    if ok.sum() < 2:
        return float("nan")
    return _weighted_nmi(a[ok], b[ok], weights[ok])


def _weighted_nmi(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((len(ua), len(ub)))
    np.add.at(joint, (ia, ib), w)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return 1.0 if ha == hb == 0 else float(2 * mi / (ha + hb)) if (ha + hb) > 0 else 0.0


def state_frame_counts(items: list[tuple[np.ndarray, np.ndarray]], n_states: int) -> np.ndarray:
    out = np.zeros(n_states)
    for states, _ in items:
        v = states[states >= 0]
        out += np.bincount(v, minlength=n_states)
    return out


# ---------------------------------------------------------------- stage 8: autocorrelation
# 14 pose coordinates after removing the pair centroid and the yaw about z.
# Order: mouth_a xyz, tail_a xyz, mouth_b xyz, tail_b xyz, pectoral_b x, pectoral_b z.
# pectoral_a and pectoral_b's y are fixed by the two constraints and are not stored.
POSE_COLUMNS = (
    "mouth_a_x", "mouth_a_y", "mouth_a_z", "tail_a_x", "tail_a_y", "tail_a_z",
    "mouth_b_x", "mouth_b_y", "mouth_b_z", "tail_b_x", "tail_b_y", "tail_b_z",
    "pectoral_b_x", "pectoral_b_z",
)
_RECONSTRUCT = None  # 18 x 14, built once. 18 = 6 points x (x, y, z), point order mouth_a, pec_a, tail_a, mouth_b, pec_b, tail_b.


def _reconstruct_matrix() -> np.ndarray:
    """Map the 14 stored coordinates back to 6 centred, yaw-aligned points."""
    global _RECONSTRUCT
    if _RECONSTRUCT is not None:
        return _RECONSTRUCT
    m = np.zeros((18, 14))
    # stored blocks copy straight across: mouth_a, tail_a, mouth_b, tail_b
    for src, dst in ((0, 0), (3, 6), (6, 9), (9, 15)):
        m[dst:dst + 3, src:src + 3] = np.eye(3)
    m[12, 12] = 1.0   # pectoral_b x
    m[14, 13] = 1.0   # pectoral_b z
    # pectoral_a = -sum of the other points, with pectoral_b y = pectoral_a y
    for col, sign in ((0, -1), (3, -1), (6, -1), (9, -1), (12, -1)):
        m[3, col] += sign          # pec_a x
    for col in (2, 5, 8, 11, 13):
        m[5, col] = -1.0           # pec_a z
    for col in (1, 4, 7, 10):
        m[4, col] = -0.5           # pec_a y
        m[13, col] = -0.5          # pec_b y
    _RECONSTRUCT = m
    return m


def reconstruct_pose(feat: np.ndarray) -> np.ndarray:
    """``(..., 14)`` -> ``(..., 6, 3)`` centred, yaw-aligned points."""
    return (np.asarray(feat, float) @ _reconstruct_matrix().T).reshape(np.shape(feat)[:-1] + (6, 3))


def aligned_configuration(points: np.ndarray, min_xy: float = MIN_XY_MM) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 6, 3)`` keypoints -> ``(T, 18)`` point coordinates with the pair centroid removed and the
    pectoral-to-pectoral direction yawed onto +x. Four degrees of freedom are gone: the flattened
    coordinates of a valid frame sum to zero on each axis, and the two pectorals share one y.
    """
    feat, valid = align_pair(points, min_xy)
    flat = reconstruct_pose(feat).reshape(feat.shape[0], 18)
    flat[~valid] = np.nan
    return flat, valid


def align_pair(points: np.ndarray, min_xy: float = MIN_XY_MM) -> tuple[np.ndarray, np.ndarray]:
    """Centre a pair and yaw-align it. ``points`` is ``(T, 6, 3)`` in order mouth, pectoral, tail of fish a then b.

    The reference direction is the horizontal pectoral-to-pectoral vector, rotated onto +x.
    Returns ``(feat (T, 14), valid)``. Frames with a missing point or a horizontal separation below ``min_xy`` are invalid.
    """
    finite = np.isfinite(points).all(axis=(1, 2))
    centre = np.nanmean(np.where(finite[:, None, None], points, np.nan), axis=1)
    rel = points - centre[:, None, :]
    ref = rel[:, 4, :2] - rel[:, 1, :2]
    length = np.linalg.norm(ref, axis=1)
    valid = finite & (length >= min_xy)
    ang = np.arctan2(ref[:, 1], ref[:, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.empty_like(rel[..., :2])
    rot[..., 0] = c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1]
    rot[..., 1] = s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]
    aligned = np.concatenate([rot, rel[..., 2:]], axis=-1)
    ma, pa, ta, mb, pb, tb = (aligned[:, i] for i in range(6))
    feat = np.concatenate([ma, ta, mb, tb, pb[:, :1], pb[:, 2:3]], axis=1)
    feat[~valid] = np.nan
    return feat, valid


def pair_points(nodes: np.ndarray, pair: str) -> np.ndarray:
    """``(T, 6, 3)`` keypoints of one pair, fish a then fish b, from named nodes ``(T, 3, 3, 3)``."""
    a, b = pair.split("-")
    ia, ib = NAMES.index(a), NAMES.index(b)
    cols = (MOUTH, PECT, TAIL)
    return np.concatenate([nodes[:, ia][:, cols], nodes[:, ib][:, cols]], axis=1)


def bend_angle(mouth: np.ndarray, pec: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Angle at the pectoral point, radians. π is a straight body."""
    u, v = mouth - pec, tail - pec
    nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    cos = np.sum(u * v, axis=-1) / (nu * nv)
    out = np.full(nu.shape, np.nan)
    good = (nu > 1e-6) & (nv > 1e-6) & np.isfinite(cos)
    out[good] = np.arccos(np.clip(cos[good], -1.0, 1.0))
    return out


def pose_scalars(nodes: np.ndarray, pair: str) -> dict[str, np.ndarray]:
    """Interpretable scalars for one pair: distance, two headings, two bends, two elevations."""
    D, ta, tb = pair_coordinates(nodes)[pair]
    a, b = pair.split("-")
    ia, ib = NAMES.index(a), NAMES.index(b)
    out = {"D": D, "theta_a": ta, "theta_b": tb,
           "bend_a": bend_angle(nodes[:, ia, MOUTH], nodes[:, ia, PECT], nodes[:, ia, TAIL]),
           "bend_b": bend_angle(nodes[:, ib, MOUTH], nodes[:, ib, PECT], nodes[:, ib, TAIL])}
    for name, i in (("elev_a", ia), ("elev_b", ib)):
        h = nodes[:, i, MOUTH] - nodes[:, i, PECT]
        n = np.linalg.norm(h, axis=-1)
        elev = np.full(n.shape, np.nan)
        good = n > 1e-6
        elev[good] = np.arcsin(np.clip(h[good, 2] / n[good], -1.0, 1.0))
        out[name] = elev
    return out


def kept_components(model: dict, tol: float = 1e-6) -> np.ndarray:
    """Indices of components that are not the removed centroid or yaw. ``tol`` is relative to the largest eigenvalue."""
    ev = model["eigenvalues"]
    return np.flatnonzero(ev > tol * ev[0])


def pca_fit(feat: np.ndarray) -> dict:
    """PCA of the valid rows. Eigenvectors are columns, largest first. Works for the 14-parameter chart and the 18 point coordinates."""
    X = feat[np.isfinite(feat).all(1)]
    if len(X) < feat.shape[1]:
        raise ValueError(f"only {len(X)} valid frames")
    mu = X.mean(axis=0)
    xc = X - mu
    cov = (xc.T @ xc) / (len(X) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    total = float(evals.sum())
    return {"mean": mu, "eigenvalues": evals, "eigenvectors": evecs,
            "variance_fraction": evals / total, "n_frames": int(len(X))}


def pca_scores(feat: np.ndarray, model: dict) -> np.ndarray:
    """Scores with the same NaN pattern as ``feat``. Each column is one component, not scaled."""
    scores = np.full(feat.shape, np.nan)
    ok = np.isfinite(feat).all(1)
    scores[ok] = (feat[ok] - model["mean"]) @ model["eigenvectors"]
    return scores


def loadings_18(model: dict) -> np.ndarray:
    """``(18, n)`` displacement of the six aligned points per unit of each component score."""
    return _reconstruct_matrix() @ model["eigenvectors"]


SCALAR_CANDIDATES = ("D", "theta_a", "theta_b", "bend_a", "bend_b")


def correlate_modes(scores: np.ndarray, scalars: dict[str, np.ndarray], seg: np.ndarray) -> pd.DataFrame:
    """Pearson r of every component with every scalar, on frames inside a continuous segment."""
    use = (seg >= 0) & np.isfinite(scores).all(1)
    rows = []
    for j in range(scores.shape[1]):
        x = scores[use, j]
        x = (x - x.mean()) / x.std()
        for name, values in scalars.items():
            y = values[use]
            m = np.isfinite(y)
            r = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 100 else float("nan")
            rows.append({"component": j + 1, "variable": name, "r": r})
    return pd.DataFrame(rows)


def correlation_time_s(x: np.ndarray, seg: np.ndarray, fps: float, lags: np.ndarray) -> tuple[float, np.ndarray]:
    """Unit-variance autocorrelation and the lag where it first drops below 1/e, in seconds.

    Dividing by the standard deviation does not change a correlation; it is done so the series
    matches the paper's normalisation. Pairs that cross a gap are excluded by ``acf``.
    """
    ok = np.isfinite(x) & (seg >= 0)
    z = np.full_like(x, np.nan, dtype=float)
    z[ok] = (x[ok] - np.mean(x[ok])) / np.std(x[ok])
    curve = acf(z, seg, lags)
    below = np.flatnonzero(curve < 1.0 / np.e)
    tau = float(lags[below[0]] / fps) if len(below) else float("nan")
    return tau, curve


def speed_groups(tau_s: np.ndarray, min_ratio: float = 2.0) -> list[list[int]]:
    """Group component indices (0-based) from slow to fast. A new group starts when the correlation
    time falls by ``min_ratio`` or more. Returns [] when no component has a finite time."""
    order = [int(i) for i in np.argsort(-np.where(np.isfinite(tau_s), tau_s, -np.inf)) if np.isfinite(tau_s[i])]
    if not order:
        return []
    groups, cur = [], [order[0]]
    for prev, nxt in zip(order, order[1:]):
        if tau_s[prev] / tau_s[nxt] >= min_ratio:
            groups.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    groups.append(cur)
    return groups


def representative_of(group: list[int], corr: pd.DataFrame, min_abs: float = 0.5) -> dict:
    """The interpretable scalar most correlated with any component in ``group`` (1-based in the result)."""
    part = corr[corr["component"].isin([i + 1 for i in group]) & corr["variable"].isin(SCALAR_CANDIDATES)]
    part = part.dropna(subset=["r"])
    if part.empty:
        return {"variable": None, "component": None, "r": None}
    row = part.loc[part["r"].abs().idxmax()]
    if abs(row["r"]) < min_abs:
        return {"variable": None, "component": int(row["component"]), "r": float(row["r"])}
    return {"variable": str(row["variable"]), "component": int(row["component"]), "r": float(row["r"])}


def acf(x: np.ndarray, seg: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Autocorrelation at ``lags`` using only frame pairs in the same continuous segment."""
    ok = np.isfinite(x) & (seg >= 0)
    mu, var = np.nanmean(x[ok]), np.nanvar(x[ok])
    out = np.full(len(lags), np.nan)
    for k, lag in enumerate(lags):
        a, b = x[:-lag], x[lag:]
        m = np.isfinite(a) & np.isfinite(b) & (seg[:-lag] == seg[lag:]) & (seg[:-lag] >= 0)
        if m.sum() > 100:
            out[k] = np.mean((a[m] - mu) * (b[m] - mu)) / var
    return out


#: Behaviour names used throughout this analysis. ``att`` = attack + attempted attack (BORIS "dash"),
#: ``qc`` = quick circle + incomplete quick circle (BORIS "courtship"). Other names are kept as they are.
BEHAVIOR_NAMES = {
    "dash": "att", "attack": "att", "attempted att": "att", "attempted attack": "att",
    "courtship": "qc", "quick circle": "qc", "incomplete qc": "qc", "incomplete quick circle": "qc",
}


def canonical_behavior(name: str) -> str:
    n = str(name).strip().lower()
    return BEHAVIOR_NAMES.get(n, n)


def bouts_of(experiment: str) -> pd.DataFrame:
    """BORIS bouts of an experiment (frames). Callers must have checked access to the experiment."""
    from scripts.analysis.behavior_dataset import locate_boris
    from scripts.review.behaviors import default_search, load_bouts

    path, _ = locate_boris(experiment, default_search())
    return load_bouts(path)


def frame_behavior_labels(bouts: pd.DataFrame, pair: str, n_frames: int, offset: int = 0) -> np.ndarray:
    """One behaviour name per frame for ``pair`` (``none`` when no bout of that pair covers it).

    A bout belongs to a pair when ``who`` is ``X->Y`` with {X, Y} equal to the pair.
    """
    want = set(pair.split("-"))
    lab = np.full(n_frames, "none", dtype=object)
    for r in bouts.sort_values("start").itertuples():
        who = str(r.who)
        if "->" not in who or set(who.split("->")) != want:
            continue
        s, e = int(r.start) - offset, int(r.end) - offset
        s, e = max(s, 0), min(e + 1, n_frames)
        if e > s:
            seg = lab[s:e]
            seg[seg == "none"] = canonical_behavior(r.behavior)
    return lab
