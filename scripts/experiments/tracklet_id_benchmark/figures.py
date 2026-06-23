from __future__ import annotations

"""Diagnostic figures for the identity benchmark. Matplotlib Agg (headless)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ID_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _color(fid: int) -> str:
    return _ID_COLORS[int(fid) % len(_ID_COLORS)] if fid >= 0 else "#999999"


def timeline(method_tables: dict, window: dict, events: list[dict], out_path: str,
             n_fish: int = 3) -> None:
    """One row per method; per frame, mark each slot coloured by id, grey where the
    method flags it unknown. Hard-core events drawn as light vertical spans."""
    s, e = int(window["start"]), int(window["end"])
    methods = list(method_tables)
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 1.4 * len(methods) + 1.5),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, m in zip(axes, methods):
        df = method_tables[m]
        df = df[(df["frame"] >= s) & (df["frame"] <= e)]
        for fid in range(n_fish):
            g = df[df["fish_id"] == fid]
            conf = g[g["flag"].fillna("") == ""]
            flg = g[g["flag"].fillna("") != ""]
            ax.scatter(conf["frame"], np.full(len(conf), fid), s=2, c=_color(fid), marker="|")
            ax.scatter(flg["frame"], np.full(len(flg), fid), s=8, c="#bbbbbb", marker="x", linewidths=0.4)
        ax.set_ylim(-0.5, n_fish - 0.5)
        ax.set_yticks(range(n_fish))
        ax.set_yticklabels([f"id{ i}" for i in range(n_fish)])
        ax.set_ylabel(m, rotation=0, ha="right", va="center", fontsize=8)
    for ev in events:
        if ev.get("end", -1) < s or ev.get("start", 1 << 30) > e:
            continue
        for ax in axes:
            ax.axvspan(max(ev["start"], s), min(ev["end"], e), color="#ffcc66", alpha=0.18, lw=0)
    axes[-1].set_xlabel("frame")
    axes[0].set_title(f"{window['name']} ({window['kind']})  frames {s}-{e}  "
                      f"grey x = flagged unknown; yellow = hard-core event", fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def traj3d(geom_table, window: dict, out_path: str, n_fish: int = 3) -> None:
    """Top-view (X-Y) and side (X-Z) trajectory of the geometry slots in-window."""
    s, e = int(window["start"]), int(window["end"])
    df = geom_table[(geom_table["frame"] >= s) & (geom_table["frame"] <= e)]
    if df[["X", "Y", "Z"]].notna().sum().sum() == 0:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for fid in range(n_fish):
        g = df[df["fish_id"] == fid].sort_values("frame")
        ax1.plot(g["X"], g["Y"], "-", lw=0.6, c=_color(fid), label=f"id{fid}")
        ax2.plot(g["X"], g["Z"], "-", lw=0.6, c=_color(fid))
    ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)"); ax1.set_title("top view (X-Y)")
    ax2.set_xlabel("X (mm)"); ax2.set_ylabel("Z (mm)"); ax2.set_title("side (X-Z)")
    ax1.legend(fontsize=8, loc="best")
    fig.suptitle(f"{window['name']} 3D trajectory (geometry slots)", fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
