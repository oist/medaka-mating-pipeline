"""Phase-1b probe, stage 2 (run in main .venv).

Loads the linked, identity-labeled crops from p1b_detect.py and runs the clean female
classifier on BOTH crop styles built from the SAME masks:

  * ``prod`` -- mask-bbox-resized, unmasked (== training framing),
  * ``p1a``  -- raw 128px box, masked to black (the Phase-1a framing).

The decisive A/B: on the 20 untrained-window 3D-uncertain F-events, does the female model
DISCRIMINATE the female from the males when crops match the training distribution? The key
number is the female-vs-male within-frame P-spread, which Phase-1a reported as 0.029
(collapsed). If ``prod`` is large while ``p1a`` reproduces ~0.03, the Phase-1a NO-GO was a
crop-distribution artifact and the model DOES generalize across the timeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "phase0"))
sys.path.append(str(HERE.parents[2]))
from _segid_common import EMBEDDER_MODEL, FEMALE_MODEL, device, forward, load_classifier, load_embedder  # noqa: E402
from sam2_id_experiment.src.id_model.window_align import align_windows_by_appearance  # noqa: E402

VIEWS = ["side1_cam", "side2_cam"]
P_HI = 0.7
RESOLVE_SEP = 0.30   # per-event female-vs-male mean-P gap to call the female "resolved"


def load(out: Path, view: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with h5py.File(out / f"probe_crops_{view}.h5", "r") as h:
        df = pd.DataFrame({k: np.asarray(h[k]) for k in
                           ["frame", "event_id", "tracklet_id", "global_id", "cx", "cy", "area", "score"]})
        df["link_status"] = [s.decode() for s in np.asarray(h["link_status"])]
        prod = np.asarray(h["crops_prod"], np.uint8)
        p1a = np.asarray(h["crops_p1a"], np.uint8)
    df["view"] = view
    return df, prod, p1a


def within_frame_spread(df: pd.DataFrame, pcol: str) -> np.ndarray:
    """Per (view, event, frame) with >=2 distinct fish, max-min of per-fish mean P. The
    female-vs-male decisiveness Phase-1a measured at 0.029."""
    spreads = []
    for _, g in df.groupby(["view", "event_id", "frame"]):
        per_fish = g.groupby("global_id")[pcol].mean()
        if len(per_fish) >= 2:
            spreads.append(float(per_fish.max() - per_fish.min()))
    return np.array(spreads)


def per_event_separation(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    rows = []
    for eid, g in df.groupby("event_id"):
        per_fish = g.groupby("global_id")[pcol].agg(["mean", "size"]).sort_values("mean", ascending=False)
        if len(per_fish) < 2:
            rows.append({"event_id": int(eid), "n_gids": len(per_fish), "female_gid": int(per_fish.index[0]),
                         "sep": float("nan"), "resolved": False})
            continue
        sep = float(per_fish["mean"].iloc[0] - per_fish["mean"].iloc[1])
        rows.append({"event_id": int(eid), "n_gids": int(len(per_fish)),
                     "female_gid": int(per_fish.index[0]), "female_mean_p": round(float(per_fish["mean"].iloc[0]), 3),
                     "second_mean_p": round(float(per_fish["mean"].iloc[1]), 3),
                     "sep": round(sep, 3), "resolved": bool(sep >= RESOLVE_SEP)})
    return pd.DataFrame(rows)


def style_summary(df: pd.DataFrame, pcol: str) -> dict:
    sp = within_frame_spread(df, pcol)
    ev = per_event_separation(df, pcol)
    return {
        "n_crops": int(len(df)),
        "frac_crops_p_gt_0.7": round(float((df[pcol] > P_HI).mean()), 3),
        "median_withinframe_spread": round(float(np.median(sp)), 3) if len(sp) else None,
        "mean_withinframe_spread": round(float(np.mean(sp)), 3) if len(sp) else None,
        "n_multifish_frames": int(len(sp)),
        "frac_events_resolved": round(float(ev.resolved.mean()), 3) if len(ev) else None,
        "n_events_with_>=2_gids": int((ev.n_gids >= 2).sum()),
        "median_event_separation": round(float(ev.loc[ev.n_gids >= 2, "sep"].median()), 3)
        if (ev.n_gids >= 2).any() else None,
    }


def global_female_auc(df: pd.DataFrame, emb: np.ndarray, pcol: str) -> dict:
    """Strong secondary: name ONE physical female across the whole probe via the embedder
    (window-align by event), then global female-vs-male AUC on prod P(female)."""
    aligned, info = align_windows_by_appearance(
        emb, df.event_id.to_numpy(), df.global_id.to_numpy(), num_fish=3)
    d = df.copy()
    d["aligned"] = aligned
    d = d[d.aligned >= 0]
    if d.empty or d.aligned.nunique() < 2:
        return {"available": False, "reason": "alignment degenerate / <2 fish aligned",
                "n_events_aligned": 3 - len(info.get("dropped", []))}
    means = d.groupby("aligned")[pcol].mean()
    fem = int(means.idxmax())
    pos = d[d.aligned == fem][pcol].to_numpy()
    neg = d[d.aligned != fem][pcol].to_numpy()
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float); ranks[order] = np.arange(len(order))
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return {"available": True, "female_aligned_id": fem, "n_crops_aligned": int(len(d)),
            "n_events_dropped_by_align": len(info.get("dropped", [])),
            "global_female_vs_male_auc": round(float(auc), 3),
            "female_mean_p": round(float(pos.mean()), 3), "male_mean_p": round(float(neg.mean()), 3)}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    out = Path(ap.parse_args().out_dir)
    dev = device()
    parts = [load(out, v) for v in VIEWS if (out / f"probe_crops_{v}.h5").exists()]
    if not parts:
        raise SystemExit("no probe_crops_*.h5 found -- run p1b_detect.py first")
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    prod = np.concatenate([p[1] for p in parts], axis=0)
    p1a = np.concatenate([p[2] for p in parts], axis=0)

    clf = load_classifier(FEMALE_MODEL, dev)
    df["p_prod"] = forward(clf, prod, dev, head=True)
    df["p_p1a"] = forward(clf, p1a, dev, head=True)
    emb = forward(load_embedder(EMBEDDER_MODEL, dev), prod, dev, head=False)

    df.to_csv(out / "p1b_per_crop.csv", index=False)
    per_event_separation(df, "p_prod").to_csv(out / "p1b_per_event_prod.csv", index=False)

    summary = {
        "n_events": int(df.event_id.nunique()),
        "n_crops_total": int(len(df)),
        "link_status_counts": {k: int(v) for k, v in df.link_status.value_counts().items()},
        "global_id_counts": {int(k): int(v) for k, v in df.global_id.value_counts().items()},
        "PROD_style_trainingframed_unmasked": style_summary(df, "p_prod"),
        "P1A_style_raw128_masked": style_summary(df, "p_p1a"),
        "phase1a_reference": {"median_withinframe_spread": 0.029, "frac_crops_p_gt_0.7": 0.80,
                              "note": "Phase-1a used raw128/masked crops + NO tracklet FP-rejection"},
        "global_female_auc_PROD": global_female_auc(df, emb, "p_prod"),
        "resolve_sep_threshold": RESOLVE_SEP,
    }
    (out / "p1b_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    verdict(summary)
    make_figure(out, df)


def verdict(s: dict) -> None:
    prod = s["PROD_style_trainingframed_unmasked"]
    p1a = s["P1A_style_raw128_masked"]
    print("\n" + "=" * 78)
    print(f"WITHIN-FRAME female-vs-male P-spread:  PROD {prod['median_withinframe_spread']}  "
          f"vs  P1A {p1a['median_withinframe_spread']}  (Phase-1a was 0.029)")
    print(f"frac crops P(female)>0.7:              PROD {prod['frac_crops_p_gt_0.7']}  "
          f"vs  P1A {p1a['frac_crops_p_gt_0.7']}  (Phase-1a was 0.80; ideal ~0.33)")
    print(f"frac events resolved (sep>={s['resolve_sep_threshold']}):       PROD {prod['frac_events_resolved']}  "
          f"vs  P1A {p1a['frac_events_resolved']}")
    gf = s["global_female_auc_PROD"]
    if gf.get("available"):
        print(f"global female-vs-male AUC (PROD):      {gf['global_female_vs_male_auc']}")
    print("=" * 78)


def make_figure(out: Path, df: pd.DataFrame) -> None:
    sp_prod = within_frame_spread(df, "p_prod")
    sp_p1a = within_frame_spread(df, "p_p1a")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.hist(sp_p1a, bins=25, range=(0, 1), alpha=0.6, label="p1a (raw128/masked)", color="#e34a33")
    ax.hist(sp_prod, bins=25, range=(0, 1), alpha=0.6, label="prod (trainingframed)", color="#31a354")
    ax.axvline(0.029, color="k", ls="--", lw=1, label="Phase-1a median 0.029")
    ax.set_title("Within-frame female-vs-male P-spread\n(multi-fish frames)")
    ax.set_xlabel("max-min P(female) across fish in a frame"); ax.set_ylabel("# frames"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(df.p_p1a, bins=25, range=(0, 1), alpha=0.6, label="p1a", color="#e34a33")
    ax.hist(df.p_prod, bins=25, range=(0, 1), alpha=0.6, label="prod", color="#31a354")
    ax.axvline(0.7, color="k", ls=":", lw=1)
    ax.set_title("P(female) over all linked crops\n(1F:2M -> healthy mass should sit low)")
    ax.set_xlabel("P(female)"); ax.set_ylabel("# crops"); ax.legend(fontsize=8)
    ax = axes[2]
    ev_prod = per_event_separation(df, "p_prod")
    ev_p1a = per_event_separation(df, "p_p1a")
    m = ev_prod.n_gids >= 2
    ax.scatter(ev_p1a.loc[m, "event_id"], ev_p1a.loc[m, "sep"], label="p1a", color="#e34a33", s=40)
    ax.scatter(ev_prod.loc[m, "event_id"], ev_prod.loc[m, "sep"], label="prod", color="#31a354", s=40)
    ax.axhline(RESOLVE_SEP, color="k", ls="--", lw=1, label=f"resolve thr {RESOLVE_SEP}")
    ax.set_title("Per-event female-vs-male separation"); ax.set_xlabel("event_id")
    ax.set_ylabel("mean-P gap (top fish - 2nd)"); ax.legend(fontsize=8)
    fig.suptitle("Phase 1b -- crop-style A/B on 20 untrained-window 3D-uncertain F-events (K_1_2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "p1b_probe.png", dpi=130)
    print(f"[score] wrote {out / 'p1b_probe.png'}")


if __name__ == "__main__":
    main()
