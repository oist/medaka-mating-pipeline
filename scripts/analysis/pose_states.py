"""Helpers for the pose-PCA / microstate / Infomap notebook
(``archive/notebooks/experiment_workflows/20261107_KP_pose_pca_states.ipynb``).

The notebook owns the parameters and the plotting; this module holds the parts that are
easier to test outside a notebook:

* the flat output layout and its naming rule
  ``<step>_<name>__<unit>__<params>__<set>.<ext>`` (``RunLayout``, ``param_label``),
  including the one function every figure is saved through (``RunLayout.save_fig``);
* pair / triple coordinate frames and the fixed feature order;
* PCA, segment-wise autocorrelation, binning, transition counting, Infomap;
* the side-by-side comparison images and the single-file ``index.html``.
"""

from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter

FISH = ("F", "M1", "M2")
NODES = ("head", "pec", "tail")  # node order of the 3-node skeleton (mouth, pectoral, tail)
HEAD, PEC, TAIL = 0, 1, 2
PAIR_UNITS = {"FM1": ("F", "M1"), "FM2": ("F", "M2"), "M1M2": ("M1", "M2"), "M2M1": ("M2", "M1")}
UNIT_ORDER = ("FM1", "FM2", "M1M2", "M2M1", "triple", "all")
SET_ORDER = ("K", "P", "KP")
FIGURE_INDEX_COLUMNS = ["file", "folder", "step", "name", "unit", "params", "set", "title", "written"]


# ---------------------------------------------------------------- naming
def _fmt_value(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return "_".join(_fmt_value(v) for v in value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def param_label(params) -> str:
    """``{"tau": 7, "bins": 10, "pcs": [1, 2, 3]}`` -> ``tau7-bins10-pcs1_2_3``.

    ``True`` adds the bare key (``norm_bl``); ``None`` and ``False`` are left out; nothing
    left gives ``base``. A string is returned unchanged (already a label).
    """
    if params is None:
        return "base"
    if isinstance(params, str):
        return params or "base"
    parts = []
    for key, value in params.items():
        if value is None or value is False:
            continue
        if value is True:
            parts.append(str(key))
        else:
            parts.append(f"{key}{_fmt_value(value)}")
    return "-".join(parts) if parts else "base"


_TOKEN = re.compile(r"^([a-z_]*[a-z])(.*)$")


def parse_label(label: str) -> dict:
    """Inverse of ``param_label`` up to value types: ``tau7-norm_bl`` -> ``{"tau": "7", "norm_bl": ""}``."""
    if not label or label == "base":
        return {}
    out = {}
    for token in label.split("-"):
        m = _TOKEN.match(token)
        if m:
            out[m.group(1)] = m.group(2)
        else:
            out[token] = ""
    return out


def join_label(parsed: dict) -> str:
    parts = [f"{k}{v}" for k, v in parsed.items()]
    return "-".join(parts) if parts else "base"


def file_stem(step: str, name: str, unit: str, params, set_name: str) -> str:
    step = str(step)
    if step.isdigit():
        step = f"{int(step):02d}"
    for field in (name, unit, set_name):
        if "__" in str(field):
            raise ValueError(f"'__' is reserved as field separator: {field!r}")
    return f"{step}_{name}__{unit}__{param_label(params)}__{set_name}"


def split_stem(stem: str) -> dict | None:
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    head, unit, params, set_name = parts
    m = re.match(r"^(\d{2}[a-z]?)_(.+)$", head)
    if not m:
        return None
    return {"step": m.group(1), "name": m.group(2), "unit": unit, "params": params, "set": set_name}


class RunLayout:
    """Flat folder of one run: ``figures/ thumbs/ compare/ data/ validation/`` + index files."""

    def __init__(self, root: Path, dpi: int = 200, thumb_width: int = 400):
        self.root = Path(root)
        self.dpi = int(dpi)
        self.thumb_width = int(thumb_width)
        for sub in ("figures", "thumbs", "compare", "data", "validation"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.figures = self.root / "figures"
        self.thumbs = self.root / "thumbs"
        self.compare = self.root / "compare"
        self.data = self.root / "data"
        self.validation = self.root / "validation"
        self.index_csv = self.root / "figures_index.csv"
        self.params_json = self.root / "params.json"
        self.log_json = self.data / "run_log.json"

    # -- figures
    def save_fig(self, fig, step, name, unit, params, set_name, dpi: int | None = None,
                 folder: str = "figures", title: str | None = None, verbose: bool = True) -> Path:
        """The only place a figure is written: png + thumbnail + one row in ``figures_index.csv``."""
        import matplotlib.pyplot as plt
        from PIL import Image

        stem = file_stem(step, name, unit, params, set_name)
        out_dir = self.root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}.png"
        if title is None:
            title = _figure_title(fig)
        fig.savefig(path, dpi=int(dpi or self.dpi), bbox_inches="tight")
        plt.close(fig)
        with Image.open(path) as im:
            w, h = im.size
            tw = min(self.thumb_width, w)
            thumb = im.convert("RGB").resize((tw, max(1, round(h * tw / w))), Image.LANCZOS)
            thumb.save(self.thumbs / f"{stem}.png", optimize=True)
        info = split_stem(stem)
        row = {"file": f"{stem}.png", "folder": folder, "step": info["step"], "name": info["name"],
               "unit": info["unit"], "params": info["params"], "set": info["set"],
               "title": (title or "").replace("\n", " | "), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        index = self.read_index()
        index = index[~((index["file"] == row["file"]) & (index["folder"] == folder))]
        index = pd.concat([index, pd.DataFrame([row])], ignore_index=True)
        index.sort_values(["folder", "file"]).to_csv(self.index_csv, index=False)
        if verbose:
            print(f"  fig {folder}/{stem}.png")
        return path

    def read_index(self) -> pd.DataFrame:
        if self.index_csv.exists():
            return pd.read_csv(self.index_csv, dtype=str, keep_default_na=False)
        return pd.DataFrame(columns=FIGURE_INDEX_COLUMNS)

    # -- data
    def path(self, step, name, unit, params, set_name, ext: str, folder: str = "data") -> Path:
        out = self.root / folder
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{file_stem(step, name, unit, params, set_name)}.{ext}"

    def save_csv(self, df: pd.DataFrame, step, name, unit, params, set_name, folder="data", verbose=True) -> Path:
        p = self.path(step, name, unit, params, set_name, "csv", folder)
        df.to_csv(p, index=False)
        if verbose:
            print(f"  csv {folder}/{p.name}  rows {len(df)}")
        return p

    def load_csv(self, step, name, unit, params, set_name, folder="data") -> pd.DataFrame:
        p = self.path(step, name, unit, params, set_name, "csv", folder)
        if not p.exists():
            raise FileNotFoundError(f"{p.name} is missing; run step {step} first")
        return pd.read_csv(p)

    def save_npz(self, arrays: dict, step, name, unit, params, set_name, verbose=True) -> Path:
        p = self.path(step, name, unit, params, set_name, "npz")
        np.savez(p, **arrays)
        if verbose:
            print(f"  npz data/{p.name}")
        return p

    def load_npz(self, step, name, unit, params, set_name) -> dict:
        p = self.path(step, name, unit, params, set_name, "npz")
        if not p.exists():
            raise FileNotFoundError(f"{p.name} is missing; run step {step} first")
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save_json(self, obj, step, name, unit, params, set_name) -> Path:
        p = self.path(step, name, unit, params, set_name, "json")
        p.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
        print(f"  json data/{p.name}")
        return p

    def load_json(self, step, name, unit, params, set_name):
        p = self.path(step, name, unit, params, set_name, "json")
        if not p.exists():
            raise FileNotFoundError(f"{p.name} is missing; run step {step} first")
        return json.loads(p.read_text(encoding="utf-8"))

    # -- params.json and the run log
    def update_params(self, section: str, values) -> dict:
        current = json.loads(self.params_json.read_text(encoding="utf-8")) if self.params_json.exists() else {}
        current[section] = values
        self.params_json.write_text(json.dumps(current, indent=2, default=_json_default), encoding="utf-8")
        return current

    def log(self, step: str, status: str, reason: str = "") -> None:
        """``status`` is ``done`` or ``skipped``; Step 8 prints the skipped ones."""
        entries = json.loads(self.log_json.read_text(encoding="utf-8")) if self.log_json.exists() else []
        entries = [e for e in entries if not (e["step"] == step and e["reason"] == reason)]
        entries.append({"step": step, "status": status, "reason": reason,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.log_json.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def clear_log(self, step: str) -> None:
        if not self.log_json.exists():
            return
        entries = json.loads(self.log_json.read_text(encoding="utf-8"))
        entries = [e for e in entries if e["step"] != step]
        self.log_json.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def read_log(self) -> list:
        return json.loads(self.log_json.read_text(encoding="utf-8")) if self.log_json.exists() else []


def _figure_title(fig) -> str:
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        return sup.get_text()
    for ax in fig.axes:
        if ax.get_title():
            return ax.get_title()
    return ""


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------- runs and segments
def bool_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) of every run of True."""
    mask = np.asarray(mask, dtype=bool)
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def segment_ids(valid: np.ndarray, video: np.ndarray | None = None) -> np.ndarray:
    """Id of the continuous valid segment of each frame (-1 outside). A video boundary always splits."""
    valid = np.asarray(valid, dtype=bool).copy()
    seg = -np.ones(len(valid), dtype=np.int64)
    if video is None:
        video = np.zeros(len(valid), dtype=np.int64)
    video = np.asarray(video)
    brk = np.zeros(len(valid), dtype=bool)
    brk[1:] = video[1:] != video[:-1]
    starts, ends = bool_runs(valid)
    sid = 0
    for a, b in zip(starts, ends):
        cuts = [a] + [int(i) for i in np.flatnonzero(brk[a + 1:b]) + a + 1] + [b]
        for s, e in zip(cuts[:-1], cuts[1:]):
            seg[s:e] = sid
            sid += 1
    return seg


def segment_table(seg: np.ndarray, video_names: np.ndarray, frames: np.ndarray, fps: np.ndarray) -> pd.DataFrame:
    ids = seg[seg >= 0]
    if len(ids) == 0:
        return pd.DataFrame(columns=["video", "segment", "start_frame", "end_frame", "n_frames", "seconds"])
    change = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(seg)]])
    rows = []
    for a, b in zip(starts, ends):
        if seg[a] < 0:
            continue
        rows.append({"video": str(video_names[a]), "segment": int(seg[a]), "start_frame": int(frames[a]),
                     "end_frame": int(frames[b - 1]), "n_frames": int(b - a), "seconds": float((b - a) / fps[a])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- preprocessing
def smooth_sg_keep_ends(nodes: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay per fish inside each finite run (``mode='interp'`` at the run ends).

    Runs shorter than ``window`` are left unsmoothed. NaN frames stay NaN.
    """
    if window <= order or window % 2 == 0:
        raise ValueError("SG window must be odd and larger than the polynomial order")
    out = np.array(nodes, dtype=np.float64, copy=True)
    for f in range(nodes.shape[1]):
        ok = np.isfinite(nodes[:, f]).all(axis=(1, 2))
        for a, b in zip(*bool_runs(ok)):
            if b - a < window:
                continue
            seg = nodes[a:b, f].reshape(b - a, -1)
            out[a:b, f] = savgol_filter(seg, window, order, axis=0, mode="interp").reshape(b - a, *nodes.shape[2:])
    return out


def skeleton_distances(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """head-pec and pec-tail distance ``(T, n_fish)``."""
    head, pec, tail = nodes[:, :, HEAD], nodes[:, :, PEC], nodes[:, :, TAIL]
    return np.linalg.norm(head - pec, axis=-1), np.linalg.norm(pec - tail, axis=-1)


def body_lengths(nodes: np.ndarray) -> np.ndarray:
    hp, pt = skeleton_distances(nodes)
    return np.nanmedian(hp, axis=0) + np.nanmedian(pt, axis=0)


# ---------------------------------------------------------------- coordinate frames and features
def pair_points(nodes: np.ndarray, unit: str) -> np.ndarray:
    """``(T, 6, 3)``: fish 1 head, pec, tail, then fish 2 head, pec, tail."""
    a, b = PAIR_UNITS[unit]
    ia, ib = FISH.index(a), FISH.index(b)
    return np.concatenate([nodes[:, ia], nodes[:, ib]], axis=1)


def _rotate_xy(rel: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Rotate ``rel (T, n, 3)`` so the horizontal unit vector ``(ux, uy)`` becomes +X'. Z' = Z."""
    xp = rel[..., 0] * ux[:, None] + rel[..., 1] * uy[:, None]
    yp = -rel[..., 0] * uy[:, None] + rel[..., 1] * ux[:, None]
    return np.stack([xp, yp, rel[..., 2]], axis=-1)


def align_pair(points: np.ndarray, min_xy: float) -> tuple[np.ndarray, np.ndarray]:
    """Co-rotating pair frame. Returns ``(pose (T, 6, 3), valid (T,))``.

    Origin = midpoint of the two pectoral points; Z' = Z; X' = horizontal direction
    fish-1 pec -> fish-2 pec; Y' = Z' x X' (right-handed). Frames whose horizontal
    pec-pec distance is below ``min_xy`` have no defined X' and are invalid.
    """
    pec1, pec2 = points[:, 1], points[:, 4]
    origin = 0.5 * (pec1 + pec2)
    d = pec2[:, :2] - pec1[:, :2]
    dist = np.linalg.norm(d, axis=1)
    valid = np.isfinite(points).all(axis=(1, 2)) & np.isfinite(dist) & (dist >= min_xy)
    safe = np.where(valid, dist, 1.0)
    pose = _rotate_xy(points - origin[:, None, :], d[:, 0] / safe, d[:, 1] / safe)
    pose[~valid] = np.nan
    return pose, valid


def pair_feature_names(unit: str) -> list[str]:
    """Fixed order: fish-1 head, fish-1 tail, fish-2 head, fish-2 tail (x', y', z'), fish-1 pec x', z'."""
    a, b = PAIR_UNITS[unit]
    names = [f"{fish}_{part}_{ax}'" for fish in (a, b) for part in ("head", "tail") for ax in "xyz"]
    return names + [f"{a}_pec_x'", f"{a}_pec_z'"]


_PAIR_SLOTS = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (5, 0), (5, 1), (5, 2),
               (1, 0), (1, 2)]  # (point, axis) of each of the 14 features


def pair_features(pose: np.ndarray) -> np.ndarray:
    return np.stack([pose[:, p, a] for p, a in _PAIR_SLOTS], axis=1)


def pair_pose_from_features(feat: np.ndarray) -> np.ndarray:
    """14 features -> 6 points. Fish-2 pec is the mirror of fish-1 pec through the origin, y' = 0."""
    feat = np.atleast_2d(np.asarray(feat, dtype=np.float64))
    pose = np.zeros((len(feat), 6, 3))
    for j, (p, a) in enumerate(_PAIR_SLOTS):
        pose[:, p, a] = feat[:, j]
    pose[:, 4, 0] = -pose[:, 1, 0]
    pose[:, 4, 2] = -pose[:, 1, 2]
    return pose


def align_triple(nodes: np.ndarray, min_xy: float) -> tuple[np.ndarray, np.ndarray]:
    """Optional co-rotating frame for the triple (choice of this notebook, not from the paper).

    Origin = mean of the three pectoral points; Z' = Z; X' = horizontal direction from F's pec
    to the midpoint of the M1 and M2 pecs; Y' = Z' x X'.
    """
    pec = nodes[:, :, PEC]
    origin = pec.mean(axis=1)
    d = 0.5 * (pec[:, 1] + pec[:, 2]) - pec[:, 0]
    dist = np.linalg.norm(d[:, :2], axis=1)
    valid = np.isfinite(nodes).all(axis=(1, 2, 3)) & np.isfinite(dist) & (dist >= min_xy)
    safe = np.where(valid, dist, 1.0)
    flat = nodes.reshape(len(nodes), 9, 3) - origin[:, None, :]
    pose = _rotate_xy(flat, d[:, 0] / safe, d[:, 1] / safe).reshape(nodes.shape)
    pose[~valid] = np.nan
    return pose, valid


def triple_feature_names(corotate: bool) -> list[str]:
    tick = "'" if corotate else ""
    return [f"{fish}_{node}_{ax}{tick}" for fish in FISH for node in NODES for ax in "xyz"]


# ---------------------------------------------------------------- PCA
def fit_pca(x: np.ndarray, standardize: bool, fix_sign: bool) -> dict:
    """PCA through the covariance matrix. ``scores = ((x - mean) / scale) @ components``."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = np.ones(x.shape[1])
    if standardize:
        scale = x.std(axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    xc = (x - mean) / scale
    cov = (xc.T @ xc) / max(len(x) - 1, 1)
    eig, vec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]
    eig = np.clip(eig[order], 0.0, None)
    vec = vec[:, order]
    flipped = np.zeros(vec.shape[1], dtype=bool)
    if fix_sign:
        for j in range(vec.shape[1]):
            k = int(np.argmax(np.abs(vec[:, j])))
            if vec[k, j] < 0:
                vec[:, j] *= -1.0
                flipped[j] = True
    total = float(eig.sum())
    return {"mean": mean, "scale": scale, "components": vec, "eigvals": eig,
            "explained": eig / total if total > 0 else np.zeros_like(eig), "flipped": flipped}


def pca_scores(feat: np.ndarray, model: dict) -> np.ndarray:
    out = np.full((len(feat), model["components"].shape[1]), np.nan)
    ok = np.isfinite(feat).all(axis=1)
    out[ok] = ((feat[ok] - model["mean"]) / model["scale"]) @ model["components"]
    return out


# ---------------------------------------------------------------- autocorrelation
def acf_segments(z: np.ndarray, seg: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation of standardised series ``z (T, k)`` inside each segment.

    Per segment s of length n: c_s(l) = mean_t z_t z_{t+l} over its n - l pairs. The
    curves are averaged with weight n over the segments with n > l. ``z`` must already
    have mean 0 and variance 1 over all valid frames, so curve[0] is 1.
    Returns ``(curve (max_lag+1, k), weight (max_lag+1,))``.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    k = z.shape[1]
    acc = np.zeros((max_lag + 1, k))
    weight = np.zeros(max_lag + 1)
    starts, ends = _segment_bounds(seg)
    for a, b in zip(starts, ends):
        n = b - a
        if n < 2:
            continue
        x = z[a:b]
        m = min(max_lag, n - 1)
        nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
        spec = np.fft.rfft(x, nfft, axis=0)
        corr = np.fft.irfft(spec * np.conj(spec), nfft, axis=0)[:m + 1]
        pairs = (n - np.arange(m + 1))[:, None]
        acc[:m + 1] += n * corr / pairs
        weight[:m + 1] += n
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(weight[:, None] > 0, acc / weight[:, None], np.nan)
    return curve, weight


def _segment_bounds(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg = np.asarray(seg)
    change = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(seg)]])
    keep = seg[starts] >= 0
    return starts[keep], ends[keep]


def time_to_e(lag_s: np.ndarray, curve: np.ndarray) -> float:
    """First downward crossing of 1/e, linearly interpolated; NaN if it never crosses."""
    target = np.exp(-1)
    below = np.flatnonzero(np.isfinite(curve) & (curve <= target))
    below = below[below > 0]
    if len(below) == 0:
        return float("nan")
    i = int(below[0])
    y0, y1 = curve[i - 1], curve[i]
    if not np.isfinite(y0) or y0 == y1:
        return float(lag_s[i])
    return float(lag_s[i - 1] + (y0 - target) / (y0 - y1) * (lag_s[i] - lag_s[i - 1]))


# ---------------------------------------------------------------- binning and states
def equal_edges(values: np.ndarray, n_bins: int, percentile: tuple[float, float] | None = None) -> np.ndarray:
    use = values[np.isfinite(values)]
    if len(use) == 0:
        raise ValueError("no finite values to bin")
    if percentile is None:
        lo, hi = float(use.min()), float(use.max())
    else:
        lo, hi = (float(v) for v in np.percentile(use, percentile))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index 0..n-1; the top edge belongs to the last bin; outside or NaN -> -1."""
    out = np.full(len(values), -1, dtype=np.int64)
    inside = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    out[inside] = np.clip(np.searchsorted(edges, values[inside], side="right") - 1, 0, len(edges) - 2)
    return out


def joint_state(bins: np.ndarray, widths: list[int]) -> np.ndarray:
    """Mixed-radix code of the bin columns ``(T, n)``; first column varies fastest. -1 if any bin is -1."""
    bins = np.asarray(bins)
    state = np.zeros(len(bins), dtype=np.int64)
    scale = 1
    for j, w in enumerate(widths):
        state += np.where(bins[:, j] < 0, 0, bins[:, j]) * scale
        scale *= int(w)
    state[(bins < 0).any(axis=1)] = -1
    return state


def decode_state(state: np.ndarray, widths: list[int]) -> np.ndarray:
    state = np.asarray(state, dtype=np.int64)
    out = np.zeros((len(state), len(widths)), dtype=np.int64)
    rest = state.copy()
    for j, w in enumerate(widths):
        out[:, j] = rest % int(w)
        rest //= int(w)
    return out


def transition_counts(states: np.ndarray, seg: np.ndarray, tau: int, n_states: int) -> tuple[sparse.csr_matrix, int]:
    """Counts of (s_t, s_{t+tau}) with both states valid and both frames in the same segment."""
    if tau <= 0 or tau >= len(states):
        return sparse.csr_matrix((n_states, n_states)), 0
    a, b = states[:-tau], states[tau:]
    ok = (seg[:-tau] >= 0) & (seg[:-tau] == seg[tau:]) & (a >= 0) & (b >= 0)
    n = int(ok.sum())
    counts = sparse.coo_matrix((np.ones(n), (a[ok], b[ok])), shape=(n_states, n_states)).tocsr()
    counts.sum_duplicates()
    return counts, n


def row_normalize(counts: sparse.spmatrix) -> sparse.csr_matrix:
    c = sparse.csr_matrix(counts, dtype=np.float64)
    row = np.asarray(c.sum(axis=1)).ravel()
    inv = np.where(row > 0, 1.0 / np.where(row > 0, row, 1.0), 0.0)
    return (sparse.diags(inv) @ c).tocsr()


def sparse_to_arrays(prefix: str, m: sparse.spmatrix) -> dict:
    coo = sparse.coo_matrix(m)
    return {f"{prefix}_row": coo.row.astype(np.int64), f"{prefix}_col": coo.col.astype(np.int64),
            f"{prefix}_data": coo.data.astype(np.float64), f"{prefix}_shape": np.asarray(coo.shape, dtype=np.int64)}


def arrays_to_sparse(prefix: str, z: dict) -> sparse.csr_matrix:
    shape = tuple(int(v) for v in z[f"{prefix}_shape"])
    return sparse.coo_matrix((z[f"{prefix}_data"], (z[f"{prefix}_row"], z[f"{prefix}_col"])), shape=shape).tocsr()


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    from scipy.spatial.distance import jensenshannon

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


# ---------------------------------------------------------------- Infomap
def resolve_infomap_backend() -> dict:
    """python-igraph first, then the ``infomap`` package. Raises if neither imports."""
    try:
        import igraph

        return {"backend": "python-igraph", "version": igraph.__version__,
                "call": "Graph.community_infomap(edge_weights='weight', trials=N)",
                "seed_note": "community_infomap has no seed argument; igraph uses Python's random "
                             "module, which is seeded before every call"}
    except ImportError:
        pass
    try:
        import infomap

        return {"backend": "infomap", "version": getattr(infomap, "__version__", "unknown"),
                "call": "Infomap(directed=True, num_trials=N, seed=SEED, silent=True)", "seed_note": "seed argument"}
    except ImportError:
        pass
    raise RuntimeError("No Infomap backend. Install one of:\n"
                       "  .venv\\Scripts\\python.exe -m pip install python-igraph\n"
                       "  .venv\\Scripts\\python.exe -m pip install infomap")


def run_infomap(prob: sparse.csr_matrix, trials: int, seed: int, backend: str) -> tuple[np.ndarray, float]:
    """Module id per node of the weighted directed graph ``prob`` (all nodes are kept)."""
    coo = sparse.coo_matrix(prob)
    keep = coo.data > 0
    rows, cols, w = coo.row[keep], coo.col[keep], coo.data[keep]
    n = prob.shape[0]
    if backend == "python-igraph":
        import random

        import igraph

        random.seed(int(seed))
        g = igraph.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())), directed=True)
        g.es["weight"] = w.tolist()
        cl = g.community_infomap(edge_weights="weight", trials=int(trials))
        return _contiguous(np.asarray(cl.membership, dtype=np.int64)), float(getattr(cl, "codelength", np.nan))
    if backend == "infomap":
        import infomap

        im = infomap.Infomap(directed=True, num_trials=int(trials), seed=int(seed), silent=True)
        for i in range(n):
            im.add_node(i)
        for a, b, x in zip(rows, cols, w):
            im.add_link(int(a), int(b), float(x))
        im.run()
        labels = np.full(n, -1, dtype=np.int64)
        for node, module in im.get_modules().items():
            labels[int(node)] = int(module)
        missing = np.flatnonzero(labels < 0)
        labels[missing] = labels.max() + 1 + np.arange(len(missing))
        return _contiguous(labels), float(im.codelength)
    raise ValueError(f"unknown backend {backend}")


def _contiguous(labels: np.ndarray) -> np.ndarray:
    """Module ids renumbered 0..k-1 in order of first appearance of the sorted ids."""
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


def within_fraction(counts: sparse.spmatrix, labels: np.ndarray, drop_diagonal: bool = False) -> float:
    """Share of transition counts whose source and target are in the same cluster (labels >= 0 only)."""
    coo = sparse.coo_matrix(counts)
    keep = (labels[coo.row] >= 0) & (labels[coo.col] >= 0)
    if drop_diagonal:
        keep &= coo.row != coo.col
    total = coo.data[keep].sum()
    if total <= 0:
        return float("nan")
    same = labels[coo.row[keep]] == labels[coo.col[keep]]
    return float(coo.data[keep][same].sum() / total)


def per_cluster_within(counts: sparse.spmatrix, labels: np.ndarray, drop_diagonal: bool = False) -> pd.DataFrame:
    coo = sparse.coo_matrix(counts)
    keep = (labels[coo.row] >= 0) & (labels[coo.col] >= 0)
    if drop_diagonal:
        keep &= coo.row != coo.col
    src = labels[coo.row[keep]]
    same = src == labels[coo.col[keep]]
    data = coo.data[keep]
    n = int(labels.max()) + 1 if (labels >= 0).any() else 0
    out_mass = np.bincount(src, weights=data, minlength=n)
    in_mass = np.bincount(src[same], weights=data[same], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(out_mass > 0, in_mass / out_mass, np.nan)
    return pd.DataFrame({"cluster": np.arange(n), "outgoing_count": out_mass, "within_count": in_mass,
                         "within_fraction": frac})


def dwell_runs(labels: np.ndarray, seg: np.ndarray) -> pd.DataFrame:
    """Runs of one cluster label inside one segment: ``cluster, length``."""
    labels = np.asarray(labels)
    seg = np.asarray(seg)
    brk = np.ones(len(labels), dtype=bool)
    brk[1:] = (labels[1:] != labels[:-1]) | (seg[1:] != seg[:-1])
    starts = np.flatnonzero(brk)
    ends = np.concatenate([starts[1:], [len(labels)]])
    good = (labels[starts] >= 0) & (seg[starts] >= 0)
    return pd.DataFrame({"cluster": labels[starts][good], "length": (ends - starts)[good]})


# ---------------------------------------------------------------- compare images and index.html
def stack_images(paths: list[Path], titles: list[str], header: str, out: Path, max_width: int = 4000) -> Path | None:
    """Side by side, common height, a title above each panel and a header line."""
    from PIL import Image, ImageDraw, ImageFont

    ims = [Image.open(p).convert("RGB") for p in paths if Path(p).exists()]
    titles = [t for p, t in zip(paths, titles) if Path(p).exists()]
    if len(ims) < 2:
        return None
    h = min(im.height for im in ims)
    ims = [im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS) for im in ims]
    total = sum(im.width for im in ims)
    if total > max_width:
        f = max_width / total
        ims = [im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS) for im in ims]
    try:
        font = ImageFont.truetype("arial.ttf", 28)
        small = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = small = ImageFont.load_default()
    pad_top = 80
    canvas = Image.new("RGB", (sum(im.width for im in ims), max(im.height for im in ims) + pad_top), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), header, fill="black", font=font)
    x = 0
    for im, t in zip(ims, titles):
        canvas.paste(im, (x, pad_top))
        draw.text((x + 10, 44), t, fill="black", font=small)
        x += im.width
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    for im in ims:
        im.close()
    return out


def _order_key(value: str, order: tuple) -> tuple:
    return (order.index(value) if value in order else len(order), value)


def build_compares(layout: RunLayout, set_order=SET_ORDER, unit_order=UNIT_ORDER, tau_steps=("05", "05b"),
                   strip_for_unit=("corot",)) -> list[Path]:
    """Write ``compare/cmp_<type>__<step>__<fixed fields>.png`` from figures/ (missing panels skipped)."""
    idx = layout.read_index()
    idx = idx[idx["folder"] == "figures"].copy()
    for old in layout.compare.glob("cmp_*.png"):
        old.unlink()
    written = []

    def emit(kind, step, fixed, rows, labels):
        rows = list(rows)
        if len(rows) < 2:
            return
        name = f"cmp_{kind}__{step}__{fixed}.png"
        paths = [layout.figures / r["file"] for r in rows]
        p = stack_images(paths, labels, f"{kind}: {step} {fixed.replace('__', '  ')}", layout.compare / name)
        if p is not None:
            written.append(p)

    # by_set
    for (step, name, unit, params), g in idx.groupby(["step", "name", "unit", "params"]):
        g = g[g["set"].isin(set_order)]
        g = sorted(g.to_dict("records"), key=lambda r: _order_key(r["set"], set_order))
        emit("by_set", step, f"{name}__{unit}__{params}", g, [r["set"] for r in g])
    # by_unit (unit-specific tokens such as 'corot' do not block the match)
    idx["params_u"] = [join_label({k: v for k, v in parse_label(p).items() if k not in strip_for_unit})
                       for p in idx["params"]]
    for (step, name, set_name, params), g in idx.groupby(["step", "name", "set", "params_u"]):
        g = g[g["unit"].isin(unit_order[:-1])]
        g = sorted(g.to_dict("records"), key=lambda r: _order_key(r["unit"], unit_order))
        if len({r["unit"] for r in g}) == len(g):
            emit("by_unit", step, f"{name}__{set_name}__{params}", g, [r["unit"] for r in g])
    # by_tau and by_param
    for (step, name, unit, set_name), g in idx.groupby(["step", "name", "unit", "set"]):
        recs = g.to_dict("records")
        if len(recs) < 2:
            continue
        parsed = [parse_label(r["params"]) for r in recs]
        keys = sorted({k for p in parsed for k in p})
        for key in keys:
            groups: dict = {}
            for r, p in zip(recs, parsed):
                rest = join_label({k: v for k, v in p.items() if k != key})
                groups.setdefault(rest, []).append((p.get(key, None), r))
            for rest, items in groups.items():
                values = {v for v, _ in items}
                if len(items) < 2 or len(values) != len(items):
                    continue
                items.sort(key=lambda t: (t[0] is None, _num(t[0])))
                labels = [f"{key}={v}" if v not in (None, "") else (key if v == "" else f"no {key}") for v, _ in items]
                kind = "by_tau" if key == "tau" and step in tau_steps else f"by_param-{key}"
                emit(kind, step, f"{name}__{unit}__{rest}__{set_name}", [r for _, r in items], labels)
    return written


def _num(v):
    try:
        return (0, float(str(v).split("_")[0]))
    except (TypeError, ValueError):
        return (1, str(v))


_INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
header{padding:12px 18px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0;z-index:2}
h1{font-size:20px;margin:0 0 6px} h2{font-size:17px;margin:22px 18px 8px;border-bottom:1px solid #ccc}
.tabs button{margin-right:6px;padding:5px 12px;border:1px solid #bbb;background:#eee;cursor:pointer}
.tabs button.on{background:#333;color:#fff}
select{margin-right:12px} .grid{display:flex;flex-wrap:wrap;gap:12px;padding:0 18px}
.card{background:#fff;border:1px solid #ddd;padding:6px;width:410px}
.card img{width:400px;cursor:zoom-in;display:block}
.card .f{font-size:11px;word-break:break-all;color:#333}.card .p{font-size:11px;color:#777}
#ov{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9;overflow:auto;text-align:center}
#ov img{max-width:98%;margin:20px auto;background:#fff;cursor:zoom-out}
#ov .cap{color:#fff;font-size:13px}
details{margin:4px 0 0;font-size:12px} pre{white-space:pre-wrap;font-size:11px;max-height:260px;overflow:auto;background:#f4f4f4;padding:6px}
</style></head><body>
<header><h1>__TITLE__</h1>
<div style="font-size:12px">__SUMMARY__</div>
<details><summary>params.json</summary><pre id="pj"></pre></details>
<div class="tabs" style="margin-top:8px"><button id="t1" class="on" onclick="tab(1)">Figures</button><button id="t2" onclick="tab(2)">Comparisons</button>
<span id="filters" style="margin-left:14px">Step <select id="fs"></select> Set <select id="fa"></select> Unit <select id="fu"></select> <span id="cnt"></span></span></div>
</header>
<div id="p1"></div><div id="p2" style="display:none"></div>
<div id="ov" onclick="this.style.display='none'"><div class="cap" id="ovc"></div><img id="ovi"></div>
<script>
const FIGS=__FIGS__; const CMPS=__CMPS__; const PARAMS=__PARAMS__;
document.getElementById('pj').textContent=JSON.stringify(PARAMS,null,1);
function opts(id,vals){const s=document.getElementById(id);s.innerHTML='<option value="">all</option>'+vals.map(v=>`<option>${v}</option>`).join('');s.onchange=render;}
const uniq=k=>[...new Set(FIGS.map(f=>f[k]))].sort();
opts('fs',uniq('step'));opts('fa',uniq('set'));opts('fu',uniq('unit'));
function show(src,cap){document.getElementById('ovi').src=src;document.getElementById('ovc').textContent=cap;document.getElementById('ov').style.display='block';}
function card(thumb,full,file,sub){return `<div class="card"><img loading="lazy" src="${thumb}" onclick="show('${full}','${file}')"><div class="f">${file}</div><div class="p">${sub}</div></div>`;}
function render(){const s=fs.value,a=fa.value,u=fu.value;const sel=FIGS.filter(f=>(!s||f.step==s)&&(!a||f.set==a)&&(!u||f.unit==u));
cnt.textContent=sel.length+' figures';const by={};sel.forEach(f=>{(by[f.step]=by[f.step]||[]).push(f)});
p1.innerHTML=Object.keys(by).sort().map(k=>`<h2>Step ${k}</h2><div class="grid">`+by[k].map(f=>card(f.thumb,f.full,f.file,`params: ${f.params} | set: ${f.set} | unit: ${f.unit}<br>${f.title}`)).join('')+'</div>').join('');}
function tab(i){p1.style.display=i==1?'':'none';p2.style.display=i==2?'':'none';filters.style.display=i==1?'':'none';t1.className=i==1?'on':'';t2.className=i==2?'on':'';}
const byk={};CMPS.forEach(c=>{(byk[c.kind]=byk[c.kind]||[]).push(c)});
p2.innerHTML=Object.keys(byk).sort().map(k=>`<h2>${k}</h2><div class="grid">`+byk[k].map(c=>card(c.full,c.full,c.file,'')).join('')+'</div>').join('');
render();
</script></body></html>"""


def build_index_html(layout: RunLayout, title: str, summary_lines: list[str]) -> Path:
    idx = layout.read_index()
    figs = []
    for r in idx.to_dict("records"):
        folder = r["folder"]
        thumb = f"thumbs/{r['file']}" if (layout.thumbs / r["file"]).exists() else f"{folder}/{r['file']}"
        figs.append({"file": r["file"], "step": r["step"], "set": r["set"], "unit": r["unit"], "params": r["params"],
                     "title": html.escape(r.get("title", "")), "thumb": thumb, "full": f"{folder}/{r['file']}"})
    cmps = [{"file": p.name, "kind": p.name.split("__")[0].replace("cmp_", ""), "full": f"compare/{p.name}"}
            for p in sorted(layout.compare.glob("cmp_*.png"))]
    params = json.loads(layout.params_json.read_text(encoding="utf-8")) if layout.params_json.exists() else {}
    text = (_INDEX_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__SUMMARY__", "<br>".join(html.escape(s) for s in summary_lines))
            .replace("__FIGS__", json.dumps(figs)).replace("__CMPS__", json.dumps(cmps))
            .replace("__PARAMS__", json.dumps(params, default=_json_default)))
    out = layout.root / "index.html"
    out.write_text(text, encoding="utf-8")
    return out
