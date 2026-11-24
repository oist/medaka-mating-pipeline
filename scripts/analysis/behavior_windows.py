"""Standardised kinematics around BORIS bout starts, one row per behaviour.

Each value is standardised with a mean and a standard deviation taken from every usable
frame of the same recording (all male actors and targets pooled). Labelled bouts are not
removed from that reference and no frame counts twice. Skewed, non-negative features are put
on a log scale first. One (mean, sd) pair per experiment and feature is shared by every bout,
so bouts stay comparable and their mean curve is meaningful.

The identification window is read off the mean standardised curves: the stretch around the
BORIS start where the mean stays at least ``Z_ON`` standard deviations from the recording's
typical level.

    python -m scripts.analysis.behavior_windows [--out DIR] [--paths-config FILE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.analysis import behavior_features as F  # noqa: E402
from scripts.analysis.behavior_dataset import _read, resolve_roots  # noqa: E402
from scripts.analysis.behavior_timeline import attach_circle_columns  # noqa: E402
from scripts.review.behaviors import load_bouts  # noqa: E402

ORDER = ("attack", "attempted att", "incomplete qc", "quick circle", "wrap")
FEATS = ("dist", "speed", "heading", "orbit", "bearing", "sweep", "curve")
TITLES = {
    "dist": "Distance to target\n(log, standardised)",
    "speed": "Actor speed\n(log, standardised)",
    "heading": "Actor heading to target\n(standardised)",
    "orbit": "Actor angular speed\naround target (log, standardised)",
    "bearing": "Velocity direction\n(standardised)",
    "sweep": "Absolute angle swept around\ntarget in 1 s (log, standardised)",
    "curve": "Actor path curvature\n(log, standardised)",
}
# Added inside the log so zero is finite; chosen well below the smallest meaningful value.
LOG_OFFSET = {"dist": 1.0, "speed": 1.0, "orbit": 1.0, "sweep": 1.0, "curve": 0.01}
# Male actors only: no focus bout has the female as actor, and the traces carry no female heading.
PAIRS = tuple(p for p in F.DIRECTED if p[0] != "F")
DIST = {("F", "M1"): "dist_F_M1", ("M1", "F"): "dist_F_M1", ("F", "M2"): "dist_F_M2",
        ("M2", "F"): "dist_F_M2", ("M1", "M2"): "dist_M1_M2", ("M2", "M1"): "dist_M1_M2"}
FPS = 166.0
STEP = F.TRACE_BIN
DT = STEP / FPS
TIMES = np.round(np.arange(-10.0, 10.0 + 1e-9, DT), 4)
MIN_REFERENCE = 200       # frames needed before a feature gets a mean and sd
Z_ON = 1.0                # |mean z| at or above this counts as "on"
MIN_BOUTS = 5             # bouts with a value at a time before the mean is trusted
MIN_RUN_S = 0.15
TOUCH_S = 0.25            # a run must come within this of the BORIS start


def _column(traces: pd.DataFrame, name: str) -> np.ndarray:
    if name not in traces.columns:
        raise KeyError(f"trace column {name!r} is missing")
    return traces[name].to_numpy(float)


def series(traces: pd.DataFrame, actor: str, target: str) -> dict[str, np.ndarray]:
    """Raw features of one actor toward one target on the trace frames."""
    return {
        "dist": _column(traces, DIST[(actor, target)]),
        "speed": _column(traces, f"speed_{actor}"),
        "heading": _column(traces, f"head_{actor}_to_{target}"),
        "orbit": np.abs(_column(traces, f"orbit_{actor}_around_{target}")),
        "bearing": _column(traces, f"vel_angle_{actor}_to_{target}"),
        "sweep": np.abs(_column(traces, f"sweep_{actor}_around_{target}")),
        "curve": _column(traces, f"curve_{actor}"),
    }


def transform(feature: str, values: np.ndarray) -> np.ndarray:
    if feature in LOG_OFFSET:
        return np.log(np.asarray(values, float) + LOG_OFFSET[feature])
    return np.asarray(values, float)


def reference(traces: pd.DataFrame) -> dict[str, tuple[float, float, int]]:
    """``feature -> (mean, sd, n)`` over every usable trace frame of one recording."""
    pools: dict[str, list[np.ndarray]] = {f: [] for f in FEATS}
    for actor, target in PAIRS:
        raw = series(traces, actor, target)
        for feature in FEATS:
            pools[feature].append(transform(feature, raw[feature]))
    out = {}
    for feature, parts in pools.items():
        pool = np.concatenate(parts)
        pool = pool[np.isfinite(pool)]
        if len(pool) < MIN_REFERENCE or pool.std() == 0:
            raise ValueError(f"feature {feature!r}: only {len(pool)} usable frames for a reference")
        out[feature] = (float(pool.mean()), float(pool.std(ddof=0)), int(len(pool)))
    return out


def collect(folders: list[Path]) -> tuple[dict, list[dict]]:
    """Standardised curves per behaviour and the reference table."""
    curves: dict[str, dict[str, list[np.ndarray]]] = {b: {f: [] for f in FEATS} for b in ORDER}
    table: list[dict] = []
    for folder in folders:
        trial = _read(folder, folder.name)
        attach_circle_columns(trial)
        traces = trial.traces
        ref = reference(traces)
        for feature, (mu, sd, n) in ref.items():
            table.append({"experiment": folder.name, "feature": feature, "mean": mu, "sd": sd, "n_frames": n})
        frames = traces["frame"].to_numpy(int)
        bouts = F.normalise_behaviors(load_bouts(trial.provenance["boris"]))
        bouts = bouts[bouts["behavior"].isin(ORDER)]
        for r in bouts.itertuples(index=False):
            actor, target, directed = F.parse_who(r.who)
            if not directed or (actor, target) not in PAIRS:
                continue
            want = int(r.start) + np.round(TIMES * FPS).astype(int)
            idx = np.clip(np.searchsorted(frames, want), 0, len(frames) - 1)
            near = np.abs(frames[idx] - want) <= STEP
            raw = series(traces, actor, target)
            for feature in FEATS:
                mu, sd, _n = ref[feature]
                values = np.full(len(TIMES), np.nan)
                values[near] = transform(feature, raw[feature][idx[near]])
                if np.isfinite(values).sum() >= 10:
                    curves[r.behavior][feature].append((values - mu) / sd)
    return curves, table


def _runs(on: np.ndarray) -> list[tuple[int, int]]:
    found, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        found.append((i, j - 1))
        i = j
    return found


def feature_runs(lines: list[np.ndarray]) -> list[dict]:
    """Runs of the mean curve at or beyond ``Z_ON`` that touch the BORIS start."""
    if not lines:
        return []
    arr = np.vstack(lines)
    mean = np.nanmean(arr, axis=0)
    count = np.isfinite(arr).sum(axis=0)
    on = (count >= MIN_BOUTS) & (np.abs(mean) >= Z_ON)
    if len(on) >= 3:
        hole = ~on[1:-1] & on[:-2] & on[2:]       # bridge a single off sample
        on = on.copy()
        on[1:-1] |= hole
    found = []
    for a, b in _runs(on):
        lo, hi = TIMES[a], TIMES[b]
        length = hi - lo + DT
        if length < MIN_RUN_S or lo > TOUCH_S or hi < -TOUCH_S:
            continue
        found.append({"start_s": float(lo), "stop_s": float(hi), "length_s": float(length),
                      "peak_mean_z": float(mean[a:b + 1][np.argmax(np.abs(mean[a:b + 1]))])})
    return found


def envelopes(curves: dict) -> pd.DataFrame:
    rows = []
    for behavior in ORDER:
        lo, hi = [], []
        for feature in FEATS:
            runs = feature_runs(curves[behavior][feature])
            for r in runs:
                rows.append({"behavior": behavior, "feature": feature, "n_bouts": len(curves[behavior][feature]), **r})
                lo.append(r["start_s"])
                hi.append(r["stop_s"])
        rows.append({"behavior": behavior, "feature": "ALL", "n_bouts": len(curves[behavior]["dist"]),
                     "start_s": min(lo) if lo else np.nan, "stop_s": max(hi) if hi else np.nan,
                     "length_s": (max(hi) - min(lo) + DT) if lo else np.nan, "peak_mean_z": np.nan})
    return pd.DataFrame(rows)


def draw(curves: dict, env: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(len(ORDER), len(FEATS), figsize=(22.5, 11.2), sharex=True, sharey="col")
    limits = {}
    for feature in FEATS:
        pool = np.concatenate([v for b in ORDER for v in curves[b][feature]])
        pool = pool[np.isfinite(pool)]
        lo, hi = np.percentile(pool, [0.5, 99.5])
        limits[feature] = (min(lo, -2.5), max(hi, 2.5))
    union = env[env.feature == "ALL"].set_index("behavior")
    for i, behavior in enumerate(ORDER):
        for j, feature in enumerate(FEATS):
            ax = axes[i, j]
            lines = curves[behavior][feature]
            for r in env[(env.behavior == behavior) & (env.feature == feature)].itertuples():
                ax.axvspan(r.start_s, r.stop_s + DT, color="#fee08b", alpha=0.6, zorder=0)
            ax.axvline(0, color="black", lw=0.6, zorder=1)
            ax.axhline(0, color="#888888", lw=0.6, zorder=1)
            for level in (-Z_ON, Z_ON):
                ax.axhline(level, color="#bbbbbb", lw=0.6, ls="--", zorder=1)
            for values in lines:
                ax.plot(TIMES, values, color="#4c78a8", lw=0.4, alpha=0.2, zorder=2)
            if lines:
                ax.plot(TIMES, np.nanmean(np.vstack(lines), axis=0), color="black", lw=1.5, zorder=4)
            ax.set_ylim(*limits[feature])
            ax.set_xlim(-10, 10)
            if i == 0:
                ax.set_title(TITLES[feature], fontsize=8)
            if j == 0:
                u = union.loc[behavior]
                span = "no window" if np.isnan(u.start_s) else f"{u.start_s:+.2f} to {u.stop_s:+.2f} s"
                ax.set_ylabel(f"{behavior}\nn={len(curves[behavior]['dist'])}\n{span}", fontsize=8)
            if i == len(ORDER) - 1:
                ax.set_xlabel("seconds from start", fontsize=8)
    fig.legend(handles=[
        Line2D([], [], color="#4c78a8", lw=1.5, label="one bout"),
        Line2D([], [], color="black", lw=1.6, label="mean"),
        Line2D([], [], color="#fee08b", lw=8, alpha=0.8, label=f"mean at least {Z_ON:g} sd from typical, touching the start"),
        Line2D([], [], color="#bbbbbb", lw=0.8, ls="--", label="one standard deviation"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("z = (x - mean) / sd, with mean and sd from every usable frame of the same recording.",
                 fontsize=11, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--paths-config", type=Path)
    parser.add_argument("--name", default="behavior_windows_standardised.png")
    args = parser.parse_args(argv)
    _config, _review, root = resolve_roots(args.paths_config)
    root = args.out or root
    folders = sorted(p for p in root.iterdir() if p.is_dir() and (p / "traces.parquet").is_file())
    if not folders:
        raise SystemExit(f"no experiment folders with traces.parquet in {root}")
    curves, table = collect(folders)
    pd.DataFrame(table).to_csv(root / "behavior_windows_reference.csv", index=False)
    env = envelopes(curves)
    env.to_csv(root / "behavior_windows_envelope.csv", index=False)
    draw(curves, env, root / args.name)
    print(env.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("wrote", root / args.name)


if __name__ == "__main__":
    main()
