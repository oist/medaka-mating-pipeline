"""Phase-1a probe, stage 2 (run in main .venv).

Loads the raw-Detectron2 crops from p1a_detect.py, runs the clean female classifier, and
fills the decisive 2x2 cell Phase 0 could not measure: at a 3D-UNCERTAIN (resolver `review`)
FEMALE-involved hard-core event, does appearance confidently + continuously identify the
female through the merge?  (All probe events are 3D-uncertain by construction, so this
populates the {3D-uncertain x appearance-confident} cell with REAL events.)

Tier-1 (headline, no calibration): per frame, count masks with P(female)>thr. The female is
"resolved" at a frame when exactly ONE mask is confidently female AND >=2 masks are present
(she is segmented apart from her close partner). Multiview = resolved if EITHER side view
resolves. appearance_confident(event) = resolved for >= min_frac of core frames.

Tier-2 (best-effort, needs calibration): reproject the 3D female (ID_2) into the view; in the
trusted PRE bracket confirm the appearance-female mask sits on the reprojected ID_2; in the
POST bracket check whether the appearance-tracked female lands on geometry-ID_2 (no swap) or a
different id (appearance CONTRADICTS the geometry -> a resolved swap).
"""
from __future__ import annotations

import argparse
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
REPO = HERE.parents[2]
sys.path.append(str(HERE.parent / "phase0"))
sys.path.append(str(REPO))
from _segid_common import FEMALE_MODEL, device, forward, load_classifier  # noqa: E402

THREED = "Z:/StephensU/medaka/reconstruction_3d/round5_260611/20251225_K_1_2_FCMC/round5_260611_3d_identified.csv"
CALIB = str(REPO / "calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models")
VIEWS = ["side1_cam", "side2_cam"]
FEM_THRESH = 0.7          # P(female) for a confident female mask
MIN_FRAC = 0.5            # frac of core frames resolved -> appearance-confident
REPROJ_GATE = 60.0        # px: mask centroid <-> reprojected fish point


def load_view(out: Path, view: str, dev: str) -> pd.DataFrame:
    p = out / f"probe_crops_{view}.h5"
    with h5py.File(p, "r") as h:
        crops = np.asarray(h["crops"], np.uint8)
        df = pd.DataFrame({k: np.asarray(h[k]) for k in ["frame", "event_id", "cx", "cy", "area", "score"]})
        pf = np.asarray(h["per_frame_counts"])  # (M,3): event_id, frame, n_masks
    df["view"] = view
    df["p_female"] = forward(load_classifier(FEMALE_MODEL, dev), crops, dev, head=True) if len(crops) else np.array([])
    counts = {(int(e), int(f)): int(n) for e, f, n in pf}
    df["n_masks_frame"] = [counts.get((int(e), int(f)), 0) for e, f in zip(df.event_id, df.frame)]
    return df


def reprojector():
    try:
        from scripts.reconstruction_3d.calibration import Calibration
        cal = Calibration(CALIB)
        t = pd.read_csv(THREED, usecols=["frame", "fish_id", "X", "Y", "Z"]).dropna(subset=["X", "Y", "Z"])
        return cal, t
    except Exception as e:  # noqa: BLE001
        print(f"[score] Tier-2 reproject unavailable ({e}); Tier-1 only", flush=True)
        return None, None


def reproj_points(cal, t3d, frames, view):
    """{frame: {fish_id: (x,y)}} for the female model's view, finite 3D only."""
    sub = t3d[t3d.frame.isin(frames)]
    out = {}
    for fr, g in sub.groupby("frame"):
        xy = cal.project_world_to_view(g[["X", "Y", "Z"]].to_numpy(), view)
        out[int(fr)] = {fid: (float(x), float(y)) for fid, (x, y) in zip(g.fish_id, xy)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    dev = device()
    events = {e["event_id"]: e for e in json.loads((out / "probe_events.json").read_text())}

    df = pd.concat([load_view(out, v, dev) for v in VIEWS if (out / f"probe_crops_{v}.h5").exists()],
                   ignore_index=True)
    df["is_female"] = df.p_female > FEM_THRESH
    df.to_csv(out / f"p1a_per_crop{args.tag}.csv", index=False)

    cal, t3d = reprojector()

    rows = []
    for eid, e in events.items():
        s, en = e["start"], e["end"]
        core = (df.frame >= s) & (df.frame <= en) & (df.event_id == eid)
        per_view = {}
        for v in VIEWS:
            d = df[core & (df.view == v)]
            if not len(d):
                continue
            g = d.groupby("frame")
            nfem = g.is_female.sum()
            nmask = g.frame.size()
            resolved = ((nfem == 1) & (nmask >= 2))
            per_view[v] = {"frac_resolved": float(resolved.mean()), "n_core_frames": int(len(nfem)),
                           "median_masks": float(nmask.median()), "frac_1female": float((nfem == 1).mean())}
        if not per_view:
            continue
        # multiview: resolved at a frame if EITHER view resolves it
        core_frames = sorted(df[core].frame.unique())
        res_any, soft_any = [], []
        for fr in core_frames:
            strict = soft = False
            for v in VIEWS:
                dd = df[(df.event_id == eid) & (df.frame == fr) & (df.view == v)]
                if len(dd) and (dd.is_female.sum() == 1):
                    soft = True                      # female uniquely called in >=1 view
                    if len(dd) >= 2:
                        strict = True                # ... AND segmented apart from a partner
            res_any.append(strict); soft_any.append(soft)
        mv_frac = float(np.mean(res_any)) if res_any else 0.0
        mv_frac_soft = float(np.mean(soft_any)) if soft_any else 0.0
        appearance_confident = mv_frac >= MIN_FRAC

        # Tier-2: swap adjudication (best-effort)
        t2 = None
        if cal is not None:
            t2 = adjudicate(df, eid, e, cal, t3d)
        rows.append({
            "event_id": eid, "pair": e["pair"], "start": s, "end": en, "n_frames": e["n_frames"],
            "min_mm": e["min_mm"], "reason": e["reason"],
            "side1_frac_resolved": round(per_view.get("side1_cam", {}).get("frac_resolved", float("nan")), 3),
            "side2_frac_resolved": round(per_view.get("side2_cam", {}).get("frac_resolved", float("nan")), 3),
            "multiview_frac_resolved": round(mv_frac, 3),
            "multiview_frac_1female_soft": round(mv_frac_soft, 3),
            "median_masks_s1": per_view.get("side1_cam", {}).get("median_masks"),
            "appearance_confident": appearance_confident,
            "tier2_swap_verdict": t2,
        })
    ev = pd.DataFrame(rows)
    ev.to_csv(out / f"p1a_per_event{args.tag}.csv", index=False)

    n = len(ev)
    n_conf = int(ev.appearance_confident.sum())
    summary = {
        "n_events_probed": n,
        "all_events_are_3D_uncertain": True,
        "n_appearance_confident": n_conf,
        "frac_appearance_confident": round(n_conf / n, 3) if n else None,
        "decisive_cell_3Duncertain_x_appearanceConfident": n_conf,
        "median_multiview_frac_resolved": round(float(ev.multiview_frac_resolved.median()), 3) if n else None,
        "fem_thresh": FEM_THRESH, "min_frac": MIN_FRAC,
    }
    if "tier2_swap_verdict" in ev and ev.tier2_swap_verdict.notna().any():
        summary["tier2_swap_verdicts"] = ev.tier2_swap_verdict.value_counts(dropna=True).to_dict()
    (out / f"p1a_summary{args.tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print("\nper-event:")
    print(ev[["event_id", "pair", "min_mm", "side1_frac_resolved", "side2_frac_resolved",
              "multiview_frac_resolved", "appearance_confident", "tier2_swap_verdict"]].to_string(index=False))

    make_figure(out, ev, df, events, args.tag)


def _nearest_id_votes(df, eid, frames, rp):
    """Count, over (view, frame), which reprojected fish_id the single appearance-female mask
    sits nearest to (within REPROJ_GATE). Returns Counter-like dict of fish_id -> votes."""
    votes = {}
    for v in VIEWS:
        for fr in frames:
            dd = df[(df.event_id == eid) & (df.frame == fr) & (df.view == v)]
            fem = dd[dd.is_female]
            if len(fem) != 1 or fr not in rp[v]:
                continue
            fx, fy = float(fem.cx.iloc[0]), float(fem.cy.iloc[0])
            pts = rp[v][fr]
            nearest = min(pts, key=lambda fid: (pts[fid][0] - fx) ** 2 + (pts[fid][1] - fy) ** 2)
            d = ((pts[nearest][0] - fx) ** 2 + (pts[nearest][1] - fy) ** 2) ** 0.5
            if d <= REPROJ_GATE:
                votes[nearest] = votes.get(nearest, 0) + 1
    return votes


def adjudicate(df, eid, e, cal, t3d):
    """Does the appearance-tracked female end on geometry-ID_2 (no swap) or another id (swap)?

    Self-validating: first confirm in the PRE bracket (fish separated, 3D trusted) that the
    appearance-female sits on reprojected ID_2 -- if not, the reproject is miscalibrated and
    we do NOT trust the post-bracket verdict.
    """
    try:
        s, en, br = e["start"], e["end"], 75
        pre = sorted(df[(df.event_id == eid) & (df.frame >= s - br) & (df.frame < s)].frame.unique())
        post = sorted(df[(df.event_id == eid) & (df.frame > en) & (df.frame <= en + br)].frame.unique())
        if not post or not pre:
            return "no_bracket"
        rp = {v: reproj_points(cal, t3d, pre + post, v) for v in VIEWS}
        pre_v = _nearest_id_votes(df, eid, pre, rp)
        if sum(pre_v.values()) == 0:
            return "inconclusive"
        if pre_v.get("ID_2", 0) < 0.6 * sum(pre_v.values()):
            return "reproj_uncalibrated"          # pre-bracket appearance-female != geometry ID_2
        post_v = _nearest_id_votes(df, eid, post, rp)
        if sum(post_v.values()) == 0:
            return "inconclusive"
        agree, disagree = post_v.get("ID_2", 0), sum(v for k, v in post_v.items() if k != "ID_2")
        return "agrees_3D" if agree >= disagree else "contradicts_3D(appearance_resolves)"
    except Exception:  # noqa: BLE001
        return "error"


def make_figure(out, ev, df, events, tag):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ax = axes[0]
    if len(ev):
        ax.hist(ev.multiview_frac_resolved, bins=20, range=(0, 1), color="#2c7fb8")
        ax.axvline(MIN_FRAC, color="red", ls="--", label=f"confident thr={MIN_FRAC}")
        ax.set_title(f"Multiview female-resolved fraction\nper 3D-uncertain F-event (n={len(ev)})")
        ax.set_xlabel("frac of core frames female resolved (>=1 view)"); ax.set_ylabel("# events"); ax.legend(fontsize=9)
    # the decisive gradient: resolution vs contact severity
    ax = axes[1]
    if len(ev):
        col = np.where(ev.appearance_confident, "#31a354", "#e34a33")
        ax.scatter(ev.min_mm, ev.multiview_frac_resolved, c=col, s=60, edgecolor="k", lw=0.5)
        ax.axhline(MIN_FRAC, color="red", ls="--", lw=1, label=f"confident thr={MIN_FRAC}")
        ax.set_xlabel("event min inter-fish 3D distance (mm)"); ax.set_ylabel("multiview frac resolved")
        ax.set_title("Resolution vs contact severity\n(green=appearance-confident)"); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9)
    # timeline for the most severe event
    ax = axes[2]
    if len(ev):
        worst = ev.sort_values("min_mm").iloc[0]
        eid = int(worst.event_id)
        for v, c in [("side1_cam", "#2c7fb8"), ("side2_cam", "#d95f0e")]:
            d = df[(df.event_id == eid) & (df.view == v)].sort_values("frame")
            if not len(d):
                continue
            g = d.groupby("frame")
            ax.plot(g.frame.first(), g.is_female.sum(), label=f"{v} #female", color=c)
            ax.plot(g.frame.first(), g.frame.size(), label=f"{v} #masks", color=c, ls=":", alpha=0.6)
        ax.axvspan(worst.start, worst.end, color="gray", alpha=0.2, label="event core")
        ax.axhline(1, color="green", ls="--", lw=0.8)
        ax.set_title(f"Most severe event {eid} ({worst.pair}, min {worst.min_mm}mm)")
        ax.set_xlabel("frame"); ax.set_ylabel("count"); ax.legend(fontsize=7)
    fig.suptitle("Phase 1a -- raw Detectron2 + female classifier at 3D-uncertain F-events (K_1_2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / f"p1a_probe{tag}.png", dpi=130)
    print(f"[score] wrote {out / f'p1a_probe{tag}.png'}")


if __name__ == "__main__":
    main()
