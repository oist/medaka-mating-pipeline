from __future__ import annotations

"""Montage the Detectron2 appearance-ID crops actually used for training, laid out
so the (in)stability of per-fish appearance is visible at a glance.

Layout per view: rows = geometry slot (ID_1/ID_2/ID_3), columns = sample crops
GROUPED BY WINDOW (thin gaps between window groups). Reading along a row shows
whether one slot looks like a single consistent individual across the recording
(a real appearance identity, e.g. a gravid female) or whether it drifts with
imaging context window to window (the position/pose confound). Reading down a
column-group shows the three fish in the SAME imaging context.

Run::

    python -m sam2_id_experiment.scripts.42_visualize_segid_crops \
        --dataset <pooled_dir> --label K_1_1 --out-dir <viz_dir> \
        [--female-slot 1]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _load(dataset: Path):
    with h5py.File(dataset / "crops.h5", "r") as h:
        key = "crops" if "crops" in h else list(h.keys())[0]
        crops = h[key][...]
    meta = pd.read_parquet(dataset / "crop_metadata.parquet").reset_index(drop=True)
    return crops, meta


def _short_win(w) -> str:
    s = str(w).replace("win_", "")
    a, _, b = s.partition("_")
    try:
        return f"{int(a)//1000}k"
    except ValueError:
        return s


def montage_view(crops, meta, view, out_png, *, label, female_slot, per_window, seed):
    rng = np.random.default_rng(seed)
    m = meta[(meta.view == view) & (meta.qc_status == "clean_train") & (meta.global_id >= 0)]
    if m.empty:
        print(f"  [{view}] no clean crops, skip")
        return
    slots = sorted(m.global_id.unique())
    has_win = "window" in m.columns
    windows = sorted(m.window.unique()) if has_win else ["all"]

    n_rows = len(slots)
    n_cols = len(windows) * per_window
    fig = plt.figure(figsize=(max(8, n_cols * 0.95), n_rows * 1.05 + 0.8))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.06, hspace=0.06)

    for ri, slot in enumerate(slots):
        for wi, win in enumerate(windows):
            sub = m[(m.global_id == slot)]
            if has_win:
                sub = sub[sub.window == win]
            idx = sub.index.to_numpy()
            pick = rng.choice(idx, min(per_window, len(idx)), replace=False) if len(idx) else []
            for k in range(per_window):
                ax = fig.add_subplot(gs[ri, wi * per_window + k])
                ax.set_xticks([]); ax.set_yticks([])
                if k < len(pick):
                    ax.imshow(crops[pick[k]].astype(np.uint8))
                else:
                    ax.set_facecolor("0.9")
                for spine in ax.spines.values():
                    spine.set_visible(k == 0)  # left edge of each window group
                    spine.set_color("0.4"); spine.set_linewidth(1.2)
                if ri == 0 and k == per_window // 2:
                    ax.set_title(_short_win(win), fontsize=9)
                if wi == 0 and k == 0:
                    fem = "  (FEMALE)" if female_slot is not None and slot == female_slot else ""
                    ax.set_ylabel(f"ID_{int(slot)+1}{fem}", fontsize=10,
                                  color=("crimson" if fem else "black"))
    fig.suptitle(f"{label}  |  {view}  |  rows=fish slot, columns grouped by window "
                 f"(frame/1000)", fontsize=11)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{view}] -> {out_png}")


def build_parser():
    p = argparse.ArgumentParser(description="Montage seg-ID appearance crops by slot x window.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--views", nargs="+", default=None, help="default: all views in the dataset")
    p.add_argument("--female-slot", type=int, default=None, help="annotate this slot as female")
    p.add_argument("--per-window", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    crops, meta = _load(Path(args.dataset))
    views = args.views or sorted(meta.view.unique())
    print(f"{args.label}: {len(crops)} crops, views {views}")
    for view in views:
        montage_view(crops, meta, view, out / f"{args.label}_{view}_crops.png",
                     label=args.label, female_slot=args.female_slot,
                     per_window=args.per_window, seed=args.seed)


if __name__ == "__main__":
    main()
