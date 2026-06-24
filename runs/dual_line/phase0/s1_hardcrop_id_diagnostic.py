"""Phase-0 Sub-step 1 -- Hard-crop ID diagnostic.

Question: does the appearance signal SURVIVE on the hard crops we excluded from
training (qc_status == reject_area_spike merges, and non-co-visible/occluded frames),
or does it collapse to coin-flip + flicker? If it survives, Line B can carry crossings;
if not, Line B is bounded to clean frames (fusion only fills, never carries).

No ground truth is used. We measure three leakage-aware, self-consistent signals from
the frozen clean female model (model_female_clean) + self-supervised embedder
(model_resnet_full) on K_1_2 pooled side crops:

  1. P(female) decisiveness  -- margin = |2*P(female) - 1|  (1=decisive, 0=coin flip)
  2. within-tracklet sex agreement -- each tracklet is one fish; does each crop's
     sex call match the tracklet's reference call (modal over its CLEAN covisible crops)?
  3. embedding coherence -- cosine(crop embedding, tracklet clean centroid); does a hard
     crop still look like the same fish?

Leakage note: the female classifier trained ONLY on side1_cam clean+covisible crops from
windows {145000,152000,285000} (win_176300 dropped as degenerate; see correction_manifest).
=> side2_cam crops + all reject/non-covisible crops are genuinely held out. The side2
clean-vs-hard comparison is the gold (view-matched, leakage-free) read; side1 adds volume
but its clean baseline is optimistic (train leakage).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _segid_common import (
    EMBEDDER_MODEL,
    FEMALE_MODEL,
    SEGID,
    device,
    forward,
    load_classifier,
    load_crops,
    load_embedder,
)

OUT = Path(__file__).resolve().parent
DS = SEGID / "pooled_sides_full"
TRAIN_WINDOWS = {"win_145000_150000", "win_152000_157000", "win_285000_290000"}  # corrected-dataset windows
CONF_MARGIN = 0.5   # "confident" = P(female) <=0.25 or >=0.75
VCONF_MARGIN = 0.8  # "very confident"


def categorize(meta: pd.DataFrame) -> pd.Series:
    nuniq = meta.groupby(["view", "frame"]).global_id.transform("nunique")
    meta["covisible"] = nuniq >= 3
    cat = np.where(
        meta.qc_status == "reject_area_spike", "area_spike",
        np.where(meta.covisible, "clean_covisible", "non_covisible"),
    )
    return pd.Series(cat, index=meta.index)


def main() -> None:
    dev = device()
    meta = pd.read_parquet(DS / "crop_metadata.parquet").reset_index(drop=True)
    meta["row"] = np.arange(len(meta))
    meta["category"] = categorize(meta)
    meta["seen_by_classifier"] = (
        (meta.view == "side1_cam") & (meta.category == "clean_covisible") & meta.window.isin(TRAIN_WINDOWS)
    )

    rows = np.arange(len(meta))  # all crops, already sorted
    crops = load_crops(DS / "crops.h5", rows)
    print(f"[s1] loaded {len(crops)} crops {crops.shape[1:]} on {dev}")

    p_female = forward(load_classifier(FEMALE_MODEL, dev), crops, dev, head=True)
    emb = forward(load_embedder(EMBEDDER_MODEL, dev), crops, dev, head=False)  # (N,128) L2-normed
    meta["p_female"] = p_female
    meta["margin"] = np.abs(2 * p_female - 1)
    meta["call"] = np.where(p_female > 0.5, "F", "M")

    # ---- within-tracklet reference (from this tracklet's CLEAN covisible crops) ---- #
    meta["tk"] = meta.view + "|" + meta.window + "|" + meta.view_tracklet.astype(str)
    clean = meta[meta.category == "clean_covisible"]
    ref_sex = clean.groupby("tk").apply(lambda g: "F" if g.p_female.mean() > 0.5 else "M")
    meta["tk_ref_sex"] = meta.tk.map(ref_sex)
    meta["agree_ref"] = np.where(meta.tk_ref_sex.notna(), meta.call == meta.tk_ref_sex, np.nan)

    # ---- per-tracklet clean centroid -> coherence ---- #
    coh = np.full(len(meta), np.nan)
    for tk, g in meta.groupby("tk"):
        cg = g[g.category == "clean_covisible"]
        if len(cg) < 3:
            continue
        c = emb[cg.row.to_numpy()].mean(0)
        n = np.linalg.norm(c)
        if n < 1e-8:
            continue
        c = c / n
        coh[g.row.to_numpy()] = emb[g.row.to_numpy()] @ c
    meta["emb_coherence"] = coh

    # ----------------------------- aggregate ----------------------------- #
    def agg(df: pd.DataFrame) -> dict:
        a = df.agree_ref.dropna()
        return {
            "n": len(df),
            "median_margin": round(float(df.margin.median()), 4),
            "mean_margin": round(float(df.margin.mean()), 4),
            "frac_confident": round(float((df.margin >= CONF_MARGIN).mean()), 4),
            "frac_very_confident": round(float((df.margin >= VCONF_MARGIN).mean()), 4),
            "tracklet_agreement": round(float(a.mean()), 4) if len(a) else None,
            "n_with_ref": int(len(a)),
            "median_coherence": round(float(df.emb_coherence.median()), 4) if df.emb_coherence.notna().any() else None,
            "mean_p_female": round(float(df.p_female.mean()), 4),
        }

    summary_rows = []
    for view in ["side1_cam", "side2_cam"]:
        for cat in ["clean_covisible", "non_covisible", "area_spike"]:
            sub = meta[(meta.view == view) & (meta.category == cat)]
            if len(sub):
                summary_rows.append({"view": view, "category": cat, **agg(sub)})
    # pooled-over-view rows too
    for cat in ["clean_covisible", "non_covisible", "area_spike"]:
        sub = meta[meta.category == cat]
        summary_rows.append({"view": "BOTH", "category": cat, **agg(sub)})
    summary = pd.DataFrame(summary_rows)

    # --------------------------- write artifacts --------------------------- #
    keep_cols = ["crop_id", "row", "view", "window", "view_tracklet", "tk", "frame",
                 "qc_status", "category", "covisible", "seen_by_classifier", "global_id",
                 "p_female", "margin", "call", "tk_ref_sex", "agree_ref", "emb_coherence"]
    meta[keep_cols].to_csv(OUT / "s1_hardcrop_per_crop.csv", index=False)
    summary.to_csv(OUT / "s1_hardcrop_summary.csv", index=False)
    print(summary.to_string(index=False))

    # ------------------------------ figure ------------------------------ #
    cats = ["clean_covisible", "non_covisible", "area_spike"]
    colors = {"clean_covisible": "#2c7fb8", "non_covisible": "#7fcdbb", "area_spike": "#d95f0e"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for cat in cats:
        d = meta[(meta.view == "side1_cam") & (meta.category == cat)].margin
        ax.hist(d, bins=40, range=(0, 1), density=True, histtype="step", lw=2,
                color=colors[cat], label=f"{cat} (n={len(d)})")
    ax.set_title("side1_cam: P(female) decisiveness (margin)\n[clean baseline has TRAIN LEAKAGE]")
    ax.set_xlabel("margin = |2*P(female)-1|"); ax.set_ylabel("density"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for cat in cats:
        d = meta[(meta.view == "side2_cam") & (meta.category == cat)].margin
        if len(d):
            ax.hist(d, bins=40, range=(0, 1), density=True, histtype="step", lw=2,
                    color=colors[cat], label=f"{cat} (n={len(d)})")
    ax.set_title("side2_cam: margin  [HELD-OUT view = gold comparison]")
    ax.set_xlabel("margin"); ax.set_ylabel("density"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    x = np.arange(len(cats)); w = 0.35
    for i, view in enumerate(["side1_cam", "side2_cam"]):
        vals = [float((meta[(meta.view == view) & (meta.category == c)].margin >= CONF_MARGIN).mean())
                if len(meta[(meta.view == view) & (meta.category == c)]) else 0 for c in cats]
        ax.bar(x + (i - 0.5) * w, vals, w, label=view)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=15); ax.set_ylim(0, 1)
    ax.set_title("Fraction confident (margin >= 0.5)"); ax.set_ylabel("fraction"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    for i, view in enumerate(["side1_cam", "side2_cam"]):
        vals = []
        for c in cats:
            a = meta[(meta.view == view) & (meta.category == c)].agree_ref.dropna()
            vals.append(float(a.mean()) if len(a) else 0)
        ax.bar(x + (i - 0.5) * w, vals, w, label=view)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=15); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.set_title("Within-tracklet sex agreement\n(vs tracklet's clean reference)")
    ax.set_ylabel("agreement"); ax.legend(fontsize=8)

    fig.suptitle("Phase 0 / Sub-step 1 -- does appearance survive hard crops? (K_1_2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "s1_hardcrop_id_diagnostic.png", dpi=130)
    print(f"[s1] wrote {OUT/'s1_hardcrop_id_diagnostic.png'}")

    (OUT / "s1_hardcrop_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
