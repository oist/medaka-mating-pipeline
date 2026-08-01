"""Phase-0 Sub-step 2 -- Detectron2 (segmentation) coverage audit on existing K_1_2 windows.

Question: per view, what fraction of frames have 3 clean separated masks vs merged/<3,
and how does that vary with scene difficulty (3D inter-fish distance)? This bounds how
much of the timeline Line B's appearance ID can even *see* a usable per-fish crop.

Data source: the seg-ID crop_metadata (post-tracklet-linking Detectron2 masks). One row
per kept mask; ``qc_status in {clean_train, reject_area_spike}``. Per (view, frame) we count
distinct clean tracklets present = clean separated masks that frame.

IMPORTANT caveats baked into the report:
  * This is the *seg-ID purity-filtered* coverage, NOT raw Detectron2. The seg-ID pipeline
    aggressively drops short/unlinked masks (``fp_no_tracklet``: ~65% of raw masks in
    win_176300) to keep ID-training crops pure. So "present-frame" coverage is a LOWER bound
    on Line B's true ceiling -- a tracking-oriented linker (Phase 2) would keep more.
  * Only win_176300_181300 has all 3 views; side1 spans 4 windows; side2/top only win_176300.
  * Scene difficulty = 3D min pairwise inter-fish distance (view-independent, robust to ID
    swaps since distance is symmetric). Small => fish near/contact.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SEG = Path("Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2")
THREED = Path("Z:/StephensU/medaka/reconstruction_3d/round5_260611/20251225_K_1_2_FCMC/round5_260611_3d_identified.csv")
# scene bins (mm), anchored to K_1_2 BORIS scale (wrap~8, touch~17, courtship~28, baseline~40)
DIST_BINS = [0, 8, 20, 40, np.inf]
DIST_LABELS = ["contact <8mm", "near 8-20", "mid 20-40", "separated >40"]


def load_meta() -> pd.DataFrame:
    sides = pd.read_parquet(SEG / "pooled_sides_full" / "crop_metadata.parquet")
    sides["tk"] = sides.view_tracklet.astype(str)
    top = pd.read_parquet(SEG / "top_cam_crop_metadata.parquet")
    top["view"] = "top_cam"
    top["window"] = top.get("window", "win_176300_181300")
    if "window" not in top.columns:
        top["window"] = "win_176300_181300"
    top["tk"] = top.tracklet_id.astype(str)
    cols = ["frame", "view", "window", "qc_status", "tk", "x", "y"]
    return pd.concat([sides[cols], top[cols]], ignore_index=True)


def per_frame_counts(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (view, window, frame): n_clean distinct tracklets, n_spike, 2D min-pair."""
    rows = []
    for (vw, wd, fr), g in meta.groupby(["view", "window", "frame"]):
        clean = g[g.qc_status == "clean_train"]
        n_clean = clean.tk.nunique()
        # one centroid per tracklet so a fragmented tracklet's duplicate rows don't fake a tiny min-pair
        pts = clean.groupby("tk")[["x", "y"]].mean().to_numpy()
        if len(pts) >= 2:
            min2d = min(np.linalg.norm(pts[i] - pts[j]) for i, j in combinations(range(len(pts)), 2))
        else:
            min2d = np.nan
        rows.append({"view": vw, "window": wd, "frame": int(fr), "n_clean": int(n_clean),
                     "n_spike": int((g.qc_status == "reject_area_spike").sum()), "min2d_px": min2d})
    return pd.DataFrame(rows)


def min3d_by_frame() -> pd.DataFrame:
    t = pd.read_csv(THREED, usecols=["frame", "fish_id", "X", "Y", "Z"])
    t = t.dropna(subset=["X", "Y", "Z"])
    rows = []
    for fr, g in t.groupby("frame"):
        pts = g[["X", "Y", "Z"]].to_numpy()
        if len(pts) >= 2:
            d = min(np.linalg.norm(pts[i] - pts[j]) for i, j in combinations(range(len(pts)), 2))
            rows.append((int(fr), d, len(pts)))
    return pd.DataFrame(rows, columns=["frame", "min3d_mm", "n3d"])


def main() -> None:
    meta = load_meta()
    pf = per_frame_counts(meta)
    d3 = min3d_by_frame()
    pf = pf.merge(d3, on="frame", how="left")
    pf["scene"] = pd.cut(pf.min3d_mm, bins=DIST_BINS, labels=DIST_LABELS)
    pf.to_csv(OUT / "s2_coverage_per_present_frame.csv", index=False)

    # ---- per (view,window) coverage over full span AND over present frames ---- #
    summ = []
    for (vw, wd), g in pf.groupby(["view", "window"]):
        span = int(g.frame.max() - g.frame.min() + 1)
        present = len(g)
        n3 = int((g.n_clean >= 3).sum())
        n2 = int((g.n_clean >= 2).sum())
        summ.append({
            "view": vw, "window": wd, "span_frames": span, "present_frames": present,
            "present_frac_of_span": round(present / span, 3),
            "frac3_of_present": round(n3 / present, 3),
            "frac2_of_present": round(n2 / present, 3),
            "frac3_of_span": round(n3 / span, 3),     # full-span (0-clean frames count as fail)
            "frac2_of_span": round(n2 / span, 3),
            "mean_clean_per_present": round(float(g.n_clean.mean()), 2),
            "frac_present_with_spike": round(float((g.n_spike > 0).mean()), 3),
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "s2_coverage_summary.csv", index=False)
    print(summ.to_string(index=False))

    # ---- coverage stratified by scene (3D distance) over present frames ---- #
    scene_rows = []
    for (vw, wd), g in pf.groupby(["view", "window"]):
        gg = g.dropna(subset=["scene"])
        for sc in DIST_LABELS:
            s = gg[gg.scene == sc]
            if len(s):
                scene_rows.append({"view": vw, "window": wd, "scene": sc, "n_present": len(s),
                                   "frac3": round(float((s.n_clean >= 3).mean()), 3),
                                   "frac2": round(float((s.n_clean >= 2).mean()), 3),
                                   "mean_clean": round(float(s.n_clean.mean()), 2)})
    scene = pd.DataFrame(scene_rows)
    scene.to_csv(OUT / "s2_coverage_by_scene.csv", index=False)
    print("\n--- coverage by 3D scene (present frames) ---")
    print(scene.to_string(index=False))

    # ---- multiview joint on win_176300 (all 3 views) ---- #
    W = "win_176300_181300"
    mv = None
    sub = pf[pf.window == W]
    if sub.view.nunique() == 3:
        piv = sub.pivot_table(index="frame", columns="view", values="n_clean", fill_value=0)
        for v in ["top_cam", "side1_cam", "side2_cam"]:
            if v not in piv:
                piv[v] = 0
        n_views_3clean = (piv[["top_cam", "side1_cam", "side2_cam"]] >= 3).sum(axis=1)
        d3w = d3.set_index("frame").min3d_mm.reindex(piv.index)
        mv = {
            "n_union_frames": int(len(piv)),
            "frac_>=1_view_3clean": round(float((n_views_3clean >= 1).mean()), 3),
            "frac_>=2_views_3clean": round(float((n_views_3clean >= 2).mean()), 3),
            "frac_all3_views_3clean": round(float((n_views_3clean == 3).mean()), 3),
            "frac_0_views_3clean (hard)": round(float((n_views_3clean == 0).mean()), 3),
            "among_contact<8mm_frames": None,
        }
        contact = d3w < 8
        if contact.any():
            mv["among_contact<8mm_frames"] = {
                "n": int(contact.sum()),
                "frac_>=1_view_3clean": round(float((n_views_3clean[contact.values] >= 1).mean()), 3),
                "frac_0_views_3clean": round(float((n_views_3clean[contact.values] == 0).mean()), 3),
            }
        print("\n--- multiview joint (win_176300, all 3 views) ---")
        print(json.dumps(mv, indent=2))
        piv["n_views_3clean"] = n_views_3clean
        piv["min3d_mm"] = d3w
        piv.reset_index().to_csv(OUT / "s2_multiview_joint_win176300.csv", index=False)

    # ------------------------------ figure ------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    lab = (summ.view + "\n" + summ.window.str.replace("win_", "").str.replace("_181300", "").str.replace("_150000", "").str.replace("_157000", "").str.replace("_290000", ""))
    x = np.arange(len(summ))
    ax.bar(x - 0.2, summ.frac3_of_present, 0.4, label="≥3 clean / present frame", color="#2c7fb8")
    ax.bar(x + 0.2, summ.frac3_of_span, 0.4, label="≥3 clean / full span", color="#d95f0e")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7); ax.set_ylim(0, 1)
    ax.set_title("3-clean-mask coverage per view/window"); ax.legend(fontsize=8); ax.set_ylabel("fraction")

    ax = axes[0, 1]
    ax.bar(x, summ.present_frac_of_span, color="#756bb1")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7); ax.set_ylim(0, 1)
    ax.set_title("Fraction of span with ANY clean mask\n(low = seg-ID FP-filter drops frames)")
    ax.set_ylabel("present / span")

    ax = axes[1, 0]
    for (vw, wd), g in scene.groupby(["view", "window"]):
        if wd == W:
            g = g.set_index("scene").reindex(DIST_LABELS)
            ax.plot(DIST_LABELS, g.frac3, marker="o", label=vw)
    ax.set_title(f"3-clean coverage vs scene ({W}, present frames)")
    ax.set_ylabel("frac frames ≥3 clean"); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=15); ax.legend(fontsize=8)

    ax = axes[1, 1]
    if mv is not None:
        keys = ["frac_>=1_view_3clean", "frac_>=2_views_3clean", "frac_all3_views_3clean", "frac_0_views_3clean (hard)"]
        vals = [mv[k] for k in keys]
        cols = ["#31a354", "#78c679", "#c2e699", "#e34a33"]
        ax.bar(range(len(keys)), vals, color=cols)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(["≥1 view\n3-clean", "≥2 views\n3-clean", "all 3 views\n3-clean", "0 views\n(hard)"], fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(f"Multiview joint Detectron2 coverage ({W})")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    fig.suptitle("Phase 0 / Sub-step 2 -- Detectron2 mask coverage (K_1_2, seg-ID purity-filtered)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "s2_detectron2_coverage_audit.png", dpi=130)
    print(f"\n[s2] wrote {OUT/'s2_detectron2_coverage_audit.png'}")

    (OUT / "s2_coverage_audit.json").write_text(json.dumps(
        {"summary": summ.to_dict("records"), "by_scene": scene.to_dict("records"),
         "multiview_win176300": mv}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
