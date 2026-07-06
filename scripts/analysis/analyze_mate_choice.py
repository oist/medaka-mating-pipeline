"""Per-male mate-choice / courtship / aggression analysis on the dual-line 3-fish track.

This is what the dual-line male-ID work unlocks: with the female pinned per-frame (appearance anchor,
following her slot swaps) AND the two males distinguished (M1/M2), behaviour can be attributed to a
SPECIFIC male — which `compare_boris_trajectory` explicitly could not do ("males NOT distinguishable
-> validated as a dyad, not per individual").

Reads the interp 3-fish track (cols frame, id_final ∈ {F,M1,M2}, X, Y, Z) and reports, per male:
  * female ASSOCIATION (time within courtship range; preference index; sharpening to contact) — mate choice,
  * TEMPORAL DYNAMICS — does the female's preference BUILD over time, or SWITCH between males?,
  * COURTSHIP bouts (sustained female proximity) and a per-bout courtship-vs-aggression BREAKDOWN,
  * the SPAWN attribution (which male the female contacts at the spawn = the choice outcome),
  * male-male AGGRESSION bouts (close approach + speed spike; who initiates).
Optionally cross-checks the BORIS ethogram (per-male m1tf/m2tf courtship).

All metric logic lives in the shared, unit-tested `mate_choice_core` (also backs the
`compute_courtship_metrics.py` / `compute_aggression_metrics.py` CLIs).

Usage (K_1_2 defaults):
    .venv\\Scripts\\python.exe scripts\\analysis\\analyze_mate_choice.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis import mate_choice_core as mc

FPS = mc.FPS
DEFAULT_TRACK = r"runs\dual_line\lineB_fulltimeline\step2\round5_260611_3d_individual_id_interp.csv"
DEFAULT_BORIS = r"G:\.shortcut-targets-by-id\14OI1k7IaNOdC5yaC9GnYyCi2u-gfGLEs\OIST 1215-0122\boris_events\20251225_K_1_2_FCMC.csv"
DEFAULT_SPAWN = 135888   # 818.6 s * 166 fps (BORIS spawn)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-male mate-choice / courtship / aggression analysis.")
    ap.add_argument("--track", default=DEFAULT_TRACK)
    ap.add_argument("--boris", default=DEFAULT_BORIS)
    ap.add_argument("--spawn-frame", type=int, default=DEFAULT_SPAWN)
    ap.add_argument("--court-mm", type=float, default=mc.BehaviourParams.court_mm)
    ap.add_argument("--contact-mm", type=float, default=mc.BehaviourParams.contact_mm)
    ap.add_argument("--agg-mm", type=float, default=mc.BehaviourParams.agg_mm)
    ap.add_argument("--agg-speed", type=float, default=mc.BehaviourParams.agg_speed)
    ap.add_argument("--min-bout", type=int, default=mc.BehaviourParams.min_bout)
    ap.add_argument("--n-bins", type=int, default=mc.BehaviourParams.n_bins)
    ap.add_argument("--out-dir", default=r"runs\dual_line\lineB_fulltimeline\analysis")
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    p = mc.BehaviourParams(fps=FPS, court_mm=args.court_mm, contact_mm=args.contact_mm,
                           agg_mm=args.agg_mm, agg_speed=args.agg_speed,
                           min_bout=args.min_bout, n_bins=args.n_bins)
    pos, n = mc.load_individual_track(args.track)
    d = mc.compute_distances(pos, FPS)

    assoc = mc.association_metrics(d, p)
    cb = mc.courtship_bouts(d, p)
    temporal_tbl, trend = mc.temporal_preference(d, p)
    spawn = mc.spawn_attribution(d, p, args.spawn_frame)
    agg_bouts = mc.aggression_bouts(d, p)
    agg = mc.aggression_summary(agg_bouts)
    breakdown = mc.behaviour_breakdown(d, p, courtship=cb, aggression=agg_bouts)
    bouts_tbl = mc.bout_table(d, p, courtship=cb, aggression=agg_bouts)

    valid_f = assoc["valid_FM_frames"]
    lines = [f"# Per-male mate-choice analysis — {Path(args.track).name}", ""]
    lines.append(f"Frames {n} ({n/FPS:.0f}s @ {FPS:.0f}fps); valid F-M frames {valid_f} "
                 f"({valid_f/max(n,1):.1%}).\n")

    lines.append("## Female mate choice (association)")
    lines.append(f"- Female within {p.court_mm:.0f} mm of **M1 {assoc['time_near_M1_s']:.1f}s** vs "
                 f"**M2 {assoc['time_near_M2_s']:.1f}s** -> association index {assoc['association_index_M1']:.2f} "
                 f"({assoc['preferred_male']}-preferring).")
    tot_pref = max(assoc["pref_frames_M1"] + assoc["pref_frames_M2"], 1)
    lines.append(f"- Of frames with a male in courtship range, female nearer "
                 f"**M1 {assoc['pref_frames_M1']/tot_pref:.1%}** vs M2 {assoc['pref_frames_M2']/tot_pref:.1%}.")
    lines.append(f"- Female closer to M1 overall: {assoc['closer_to_M1_frac']:.1%} of valid frames.")
    sharp = assoc["sharpening_ratio_by_mm"]
    lines.append(f"- Preference SHARPENS toward contact ("
                 f"{' , '.join(f'<{k}mm {v:.2f}x' for k, v in sharp.items())} toward {assoc['preferred_male']}).")
    lines.append(f"- Mean distance: d(F,M1) {assoc['mean_d_F_M1_mm']:.1f} mm, d(F,M2) {assoc['mean_d_F_M2_mm']:.1f} mm, "
                 f"d(M1,M2) {assoc['mean_d_M1_M2_mm']:.1f} mm.\n")

    lines.append("## Temporal dynamics (does the preference build or switch?)")
    if "trend" in trend:
        lines.append(f"- Per-bin (n={trend['n_bins_with_data']}) preference index goes from early "
                     f"**{trend['early_pref_index_M1']:.2f}** to late **{trend['late_pref_index_M1']:.2f}** "
                     f"(slope {trend['slope_per_bin']:+.3f}/bin) -> **{trend['trend']}** toward "
                     f"{trend['overall_preferred_male']}.")
        lines.append(f"- Early preference {'PREDICTS' if trend['early_predicts_choice'] else 'does NOT predict'} "
                     f"the overall choice; {trend['n_switch_bins']} bin(s) favour the other male "
                     f"({'no clean switch' if trend['n_switch_bins'] == 0 else 'a switch is present'}).\n")
    else:
        lines.append("- Insufficient in-range time to characterise the temporal trend.\n")

    lines.append("## Courtship bouts (sustained female proximity)")
    lines.append(f"- M1: {cb['M1']['n_bouts']} bouts, {cb['M1']['total_s']:.1f}s total. "
                 f"M2: {cb['M2']['n_bouts']} bouts, {cb['M2']['total_s']:.1f}s total.\n")

    lines.append("## Per-bout courtship-vs-aggression breakdown")
    lines.append(f"- Courtship {breakdown['n_courtship_bouts']} bouts / {breakdown['courtship_total_s']:.1f}s "
                 f"(M1 {breakdown['courtship_M1_s']:.1f}s, M2 {breakdown['courtship_M2_s']:.1f}s) vs "
                 f"aggression {breakdown['n_aggression_bouts']} bouts / {breakdown['aggression_total_s']:.1f}s "
                 f"-> courtship:aggression time {breakdown['courtship_to_aggression_ratio']}.\n")

    if spawn is not None:
        lines.append("## Spawn (mate-choice outcome)")
        lines.append(f"- At frame {spawn['spawn_frame']} ({spawn['spawn_time_s']:.1f}s): median d(F,M1) "
                     f"{spawn['median_d_F_M1_mm']:.1f} mm, d(F,M2) {spawn['median_d_F_M2_mm']:.1f} mm -> "
                     f"**female spawned with {spawn['spawn_male']}** ({'contact' if spawn['contact'] else 'near'}).\n")

    lines.append("## Male-male aggression")
    lines.append(f"- {agg['n_bouts']} close+fast bouts (<{p.agg_mm:.0f} mm & >{p.agg_speed:.0f} mm/s), "
                 f"{agg['total_s']:.1f}s total; M1 the faster (initiator) in "
                 f"{agg['initiator_M1']}/{max(agg['n_bouts'],1)}.")

    boris_note = boris_cross_check(args.boris)
    if boris_note:
        lines += ["", "## BORIS ethogram cross-check"] + boris_note

    report = "\n".join(lines)
    (out / "mate_choice_report.md").write_text(report, encoding="utf-8")
    pd.DataFrame({"frame": np.arange(n), "d_F_M1": d.dFM1, "d_F_M2": d.dFM2, "d_M1_M2": d.dMM,
                  "court_M1": d.dFM1 < p.court_mm, "court_M2": d.dFM2 < p.court_mm}
                 ).to_csv(out / "mate_choice_per_frame.csv", index=False)
    temporal_tbl.to_csv(out / "mate_choice_temporal.csv", index=False)
    bouts_tbl.to_csv(out / "mate_choice_bouts.csv", index=False)
    make_figure(d, temporal_tbl, p, args.spawn_frame, out / "mate_choice.png")
    print(report)
    print(f"\nwrote {out/'mate_choice_report.md'}, mate_choice.png, mate_choice_temporal.csv, mate_choice_bouts.csv")


def make_figure(d: mc.Distances, temporal_tbl: pd.DataFrame, p: mc.BehaviourParams, spawn_frame, png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(figure skipped: {exc})"); return
    n = d.n; t = np.arange(n) / p.fps
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw=dict(height_ratios=[2, 1, 1], hspace=0.35))
    s = max(1, n // 4000)
    ax[0].plot(t[::s], d.dFM1[::s], lw=0.6, color="#1f77b4", label="d(F, M1)")
    ax[0].plot(t[::s], d.dFM2[::s], lw=0.6, color="#d62728", label="d(F, M2)")
    ax[0].axhline(p.court_mm, ls="--", color="gray", lw=0.8, label=f"courtship {p.court_mm:.0f} mm")
    if spawn_frame is not None and 0 <= spawn_frame < n:
        ax[0].axvline(spawn_frame / p.fps, color="green", lw=1.2, label="spawn")
    ax[0].set_ylabel("female-male distance (mm)"); ax[0].set_ylim(0, 200)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Female-male distance over time (lower = associating)")

    # `nan < court_mm` is already False, so the mask has no NaN -> plain cumsum (a missing
    # frame counts as 'not in range', i.e. 0, which is the intended cumulative-time semantics).
    cum1 = np.cumsum((d.dFM1 < p.court_mm).astype(float)) / p.fps
    cum2 = np.cumsum((d.dFM2 < p.court_mm).astype(float)) / p.fps
    ax[1].plot(t[::s], cum1[::s], color="#1f77b4", label="M1")
    ax[1].plot(t[::s], cum2[::s], color="#d62728", label="M2")
    if spawn_frame is not None and 0 <= spawn_frame < n:
        ax[1].axvline(spawn_frame / p.fps, color="green", lw=1.2)
    ax[1].set_ylabel("cumulative\ncourtship time (s)"); ax[1].legend(loc="upper left", fontsize=8)
    ax[1].set_title("Cumulative time within courtship range of the female")

    # temporal preference trajectory (build vs switch)
    tt = temporal_tbl.dropna(subset=["pref_index_M1"])
    if len(tt):
        bx = (tt["frame_lo"] + tt["frame_hi"]) / 2 / p.fps
        ax[2].plot(bx, tt["pref_index_M1"], "-o", color="#6a3d9a", ms=4)
        ax[2].axhline(0.5, ls="--", color="gray", lw=0.8)
        ax[2].fill_between(bx, 0.5, tt["pref_index_M1"], where=tt["pref_index_M1"] >= 0.5,
                           color="#1f77b4", alpha=0.25)
        ax[2].fill_between(bx, 0.5, tt["pref_index_M1"], where=tt["pref_index_M1"] < 0.5,
                           color="#d62728", alpha=0.25)
    if spawn_frame is not None and 0 <= spawn_frame < n:
        ax[2].axvline(spawn_frame / p.fps, color="green", lw=1.2)
    ax[2].set_ylim(0, 1); ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("pref index\n(M1 share)")
    ax[2].set_title("Temporal preference: >0.5 favours M1 (blue), <0.5 favours M2 (red)")
    fig.savefig(png, dpi=110, bbox_inches="tight"); plt.close(fig)


def boris_cross_check(boris_path):
    p = Path(boris_path)
    if not p.exists():
        return [f"_(BORIS file not found at {p}; skipped.)_"]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.analysis.compare_boris_trajectory import parse_boris
        ev = parse_boris(p)
    except Exception as exc:
        return [f"_(BORIS parse failed: {exc})_"]
    dur = ev.groupby("subject").dur_s.sum()
    m1tf, m2tf = float(dur.get("m1tf", 0.0)), float(dur.get("m2tf", 0.0))
    out = [f"- BORIS male->female courtship: m1tf {m1tf:.1f}s vs m2tf {m2tf:.1f}s "
           f"(coded asymmetry {max(m1tf,m2tf)/max(min(m1tf,m2tf),1e-6):.1f}x toward "
           f"{'m1' if m1tf>=m2tf else 'm2'}).",
           "  BORIS m1/m2 labels are the annotator's and not directly mappable to appearance M1/M2, but "
           "the per-male ASYMMETRY (one male courts/associates far more) should agree with the appearance "
           "association index above — the dual-line track now lets that be checked per individual."]
    return out


if __name__ == "__main__":
    main()
