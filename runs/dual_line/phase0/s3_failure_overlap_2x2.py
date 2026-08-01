"""Phase-0 Sub-step 3 -- failure-mode overlap (the decisive metric).

At the 3D hard-core (924 events where two fish came within the linker's hard-core
threshold, min_mm small), is the APPEARANCE line confident (complementary -> fusion wins)
or also uncertain (coincident -> fusion can't help)? Output the 2x2:
{3D-confident, 3D-uncertain} x {appearance-confident, appearance-uncertain}.

3D axis: resolver verdict -- `clear` (velocity/3D resolved confidently) vs `review`
(ambiguous / suspect_swap / no_3d => 3D-UNCERTAIN). From hardcore_events_resolved.csv.

Appearance axis: the clean female model. K_1_2 female = ID_2 (global_id 1), margin 0.79
(female_anchor_report). The female classifier resolves F-vs-M, so it directly addresses
events that INVOLVE the female (ID_2); male-male (ID_1|ID_3) events need the male model
(deferred to Phase 2). Per Sub-step 1 the female read is confident even on hard crops.

Two complementary reads:
 (A) MEASURED 2x2 on the 17 events that fall inside the 4 crop-windows -- real appearance
     measurement, but small n and (by luck of which windows have crops) all 3 review events
     are M-M. Appearance confidence is measured in the BRACKET around the event (crops AT the
     merge are removed by the seg-ID purity filter, so Line B re-IDs the fragments on each
     side -- the operationally correct regime).
 (B) STRUCTURAL 2x2 over all 924 events: 3D verdict x female-involvement. Female-involved =
     the female model APPLIES and (Sub-step 1) is confident -> an upper bound on appearance
     complementarity via the female model alone. This is the robust complementary-mass number.

Reuses Sub-step 1's per-crop P(female) (no re-inference).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
EVENTS = Path("Z:/StephensU/medaka/runs/udmt_validation/tracklet_bind/hardcore_events_resolved.csv")
WINS = [(145000, 150000), (152000, 157000), (176300, 181300), (285000, 290000)]
FEMALE_FISH = "ID_2"     # K_1_2 female (3D track id); confirmed by female_anchor_report
# NOTE: the raw per-window 3D-bind `global_id` is PERMUTED across windows (the very problem
# window_align fixes), so the global_id carrying the female differs by window -- we derive it
# per window from the data (argmax mean P(female)) rather than hardcoding it.
BRACKET = 300            # frames each side of the event (~1.8s @166fps)
DECISIVE = 0.6           # mean P(female) for a confident female read


def window_of(start, end):
    for a, b in WINS:
        if not (end < a or start > b):
            return f"win_{a}_{b}"
    return None


def main() -> None:
    ev = pd.read_csv(EVENTS)
    ev["is_3d_uncertain"] = ev.verdict == "review"
    ev["pair_type"] = np.where(ev.pair.str.contains(FEMALE_FISH), "F-involved", "M-M (ID_1|ID_3)")
    ev["window"] = [window_of(s, e) for s, e in zip(ev.start, ev.end)]
    ev["inwin"] = ev.window.notna()

    # -------- (B) STRUCTURAL 2x2 over all 924: 3D verdict x female involvement -------- #
    structural = pd.crosstab(ev.verdict, ev.pair_type)
    n = len(ev)
    review = ev[ev.verdict == "review"]
    review_F = int((review.pair_type == "F-involved").sum())
    print("=== (B) STRUCTURAL over all 924 events ===")
    print(structural.to_string())
    print(f"\n3D-uncertain (review): {len(review)}/{n} = {len(review)/n:.1%}")
    print(f"  of which female-involved (female-model applies): {review_F}/{len(review)} = {review_F/len(review):.1%}")
    print(f"  => appearance-addressable complementary mass (upper bound via female model): "
          f"{review_F}/{n} = {review_F/n:.1%} of all hard-core events")

    # -------- per-crop P(female) (from Sub-step 1) -------- #
    crops = pd.read_csv(OUT / "s1_hardcrop_per_crop.csv",
                        usecols=["frame", "view", "window", "global_id", "p_female", "qc_status"])
    clean = crops[crops.qc_status == "clean_train"]
    # per-window female global_id (raw 3D-bind id is permuted across windows -> derive, don't hardcode)
    win_female_gid = (clean.groupby(["window", "global_id"]).p_female.mean()
                      .groupby(level=0).idxmax().apply(lambda t: int(t[1])).to_dict())
    print("per-window female global_id (argmax mean P(female)):", win_female_gid,
          "<- note the permutation: female is NOT the same raw gid in every window")

    def female_read(window, lo, hi):
        """mean P(female) per global_id over clean crops in [lo,hi); returns (best_gid, best_mean, present)."""
        sub = clean[(clean.window == window) & (clean.frame >= lo) & (clean.frame < hi)]
        if not len(sub):
            return None, np.nan, 0
        m = sub.groupby("global_id").p_female.mean()
        return int(m.idxmax()), float(m.max()), int(sub.global_id.nunique())

    # -------- (A) MEASURED on the 17 in-window events -------- #
    rows = []
    iw = ev[ev.inwin].copy()
    for _, e in iw.iterrows():
        W = e.window
        at = clean[(clean.window == W) & (clean.frame >= e.start) & (clean.frame <= e.end)]
        n_at = at.groupby("view").global_id.nunique().to_dict()
        pre_gid, pre_p, pre_n = female_read(W, e.start - BRACKET, e.start)
        post_gid, post_p, post_n = female_read(W, e.end + 1, e.end + 1 + BRACKET)
        f_involved = e.pair_type == "F-involved"
        win_fem = win_female_gid.get(W)
        female_bracket_clean = bool(pre_p >= DECISIVE and post_p >= DECISIVE)
        # the female-by-appearance keeps the SAME raw global_id before/after the event
        # (i.e. the 3D bind did not relabel her across the merge) -- per-window id, not hardcoded.
        female_consistent = bool(pre_gid is not None and pre_gid == post_gid)
        # for an F-involved event the female IS one of the two merging fish, so a clean both-sides
        # female read means appearance can re-identify her across the merge gap.
        appearance_confident = bool(f_involved and female_bracket_clean)
        rows.append({
            "pair": e.pair, "pair_type": e.pair_type, "start": int(e.start), "end": int(e.end),
            "min_mm": e.min_mm, "verdict": e.verdict, "reason": e.reason, "window": W,
            "window_female_gid": win_fem,
            "clean_crops_AT_event_byview": json.dumps(n_at),
            "pre_female_gid": pre_gid, "pre_female_p": round(pre_p, 3) if pre_p == pre_p else None,
            "post_female_gid": post_gid, "post_female_p": round(post_p, 3) if post_p == post_p else None,
            "female_bracket_clean": female_bracket_clean, "female_consistent": female_consistent,
            "appearance_confident": appearance_confident,
        })
    iwdf = pd.DataFrame(rows)
    iwdf.to_csv(OUT / "s3_inwindow_events.csv", index=False)

    measured = pd.crosstab(
        iwdf.verdict.map({"clear": "3D-confident", "review": "3D-uncertain"}),
        iwdf.appearance_confident.map({True: "appearance-confident", False: "appearance-uncertain"}),
    )
    # ensure full 2x2
    for r in ["3D-confident", "3D-uncertain"]:
        if r not in measured.index:
            measured.loc[r] = 0
    for c in ["appearance-confident", "appearance-uncertain"]:
        if c not in measured.columns:
            measured[c] = 0
    measured = measured.loc[["3D-confident", "3D-uncertain"], ["appearance-confident", "appearance-uncertain"]]
    print("\n=== (A) MEASURED 2x2 (n=17 in-window events; appearance = bracket female read) ===")
    print(measured.to_string())

    # crop availability AT merge (demonstrates the purity-filter gap)
    n_with_crops_at = int(iwdf.clean_crops_AT_event_byview.apply(lambda s: any(v >= 1 for v in json.loads(s).values())).sum())
    print(f"\nin-window events with ANY clean crop AT the merge frames: {n_with_crops_at}/{len(iwdf)} "
          f"(purity filter removes <20mm crops -> appearance must re-ID in the bracket)")

    # -------- deep dive: the review events + the suspect_swap -------- #
    print("\n=== deep dive: in-window 3D-uncertain (review) events ===")
    for _, r in iwdf[iwdf.verdict == "review"].iterrows():
        print(f"  {r.pair} f{r.start}-{r.end} min={r.min_mm}mm reason={r.reason} | "
              f"pair_type={r.pair_type} pre_fem(gid{r.pre_female_gid},p={r.pre_female_p}) "
              f"post_fem(gid{r.post_female_gid},p={r.post_female_p}) appearance_confident={r.appearance_confident}")

    # ------------------------------ figure ------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    def draw_2x2(ax, mat, title, rowlab, collab):
        a = mat.to_numpy().astype(float)
        ax.imshow(a, cmap="Blues", vmin=0, vmax=a.max() if a.max() > 0 else 1)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                frac = a[i, j] / a.sum() if a.sum() else 0
                ax.text(j, i, f"{int(a[i,j])}\n({frac:.0%})", ha="center", va="center",
                        fontsize=14, color="black")
        ax.set_xticks(range(len(collab))); ax.set_xticklabels(collab, fontsize=9)
        ax.set_yticks(range(len(rowlab))); ax.set_yticklabels(rowlab, fontsize=9)
        ax.set_title(title, fontsize=11)

    draw_2x2(axes[0], structural.loc[["clear", "review"]],
             "(B) STRUCTURAL over all 924 hard-core events\n3D verdict x female involvement\n(female-involved = female model applies & is confident)",
             ["3D clear", "3D review\n(uncertain)"], list(structural.columns))

    draw_2x2(axes[1], measured,
             f"(A) MEASURED 2x2 (n={len(iwdf)} in-window)\nbracket female read\n[all 3 review events here are M-M -> sampling-limited]",
             ["3D-confident", "3D-uncertain"], ["appear-\nconfident", "appear-\nuncertain"])

    ax = axes[2]
    cats = ["clear", "review"]
    Fc = [int(((ev.verdict == c) & (ev.pair_type == "F-involved")).sum()) for c in cats]
    Mc = [int(((ev.verdict == c) & (ev.pair_type != "F-involved")).sum()) for c in cats]
    x = np.arange(2)
    ax.bar(x, Fc, 0.5, label="F-involved (female model applies)", color="#2c7fb8")
    ax.bar(x, Mc, 0.5, bottom=Fc, label="M-M (needs male model, Phase 2)", color="#fdae6b")
    ax.set_xticks(x); ax.set_xticklabels(["3D clear\n(confident)", "3D review\n(UNCERTAIN)"])
    ax.set_title("Hard-core events by 3D verdict & pair type\n(decisive cell = review + F-involved)")
    ax.set_ylabel("# events"); ax.legend(fontsize=8)
    for i in range(2):
        ax.text(i, Fc[i] + Mc[i] + 8, f"{Fc[i]+Mc[i]}", ha="center", fontsize=10)

    fig.suptitle("Phase 0 / Sub-step 3 -- failure-mode overlap: appearance vs 3D at the hard-core (K_1_2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "s3_failure_overlap_2x2.png", dpi=130)
    print(f"\n[s3] wrote {OUT/'s3_failure_overlap_2x2.png'}")

    (OUT / "s3_failure_overlap.json").write_text(json.dumps({
        "n_events_total": n,
        "n_3d_uncertain_review": int(len(review)),
        "frac_3d_uncertain": round(len(review) / n, 4),
        "review_female_involved": review_F,
        "review_female_involved_frac_of_review": round(review_F / len(review), 4),
        "complementary_mass_upper_bound_frac_of_all": round(review_F / n, 4),
        "structural_2x2_verdict_x_pairtype": structural.to_dict(),
        "measured_2x2_inwindow": measured.to_dict(),
        "n_inwindow_events": int(len(iwdf)),
        "n_inwindow_with_crops_at_merge": n_with_crops_at,
        "pair_counts_all": ev.pair.value_counts().to_dict(),
    }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
