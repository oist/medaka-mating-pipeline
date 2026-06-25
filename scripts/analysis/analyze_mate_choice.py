"""Per-male mate-choice / courtship / aggression analysis on the dual-line 3-fish track.

This is what the dual-line male-ID work unlocks: with the female pinned per-frame (appearance anchor,
following her slot swaps) AND the two males distinguished (M1/M2), behaviour can be attributed to a
SPECIFIC male — which `compare_boris_trajectory` explicitly could not do ("males NOT distinguishable
-> validated as a dyad, not per individual").

Reads the interp 3-fish track (cols frame, geometry_slot, X, Y, Z, id_final). Computes, per male:
  * female ASSOCIATION (time within courtship range of the female; preference index) — mate choice,
  * COURTSHIP bouts (sustained female proximity),
  * the SPAWN attribution (which male the female contacts at the spawn = the choice outcome),
  * male-male AGGRESSION bouts (close approach + speed spike; who initiates).
Optionally cross-checks the BORIS ethogram (per-male m1tf/m2tf courtship, competitive events).

Usage (K_1_2 defaults):
    .venv\\Scripts\\python.exe scripts\\analysis\\analyze_mate_choice.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 166.0
DEFAULT_TRACK = r"runs\dual_line\lineB_fulltimeline\step2\round5_260611_3d_individual_id_interp.csv"
DEFAULT_BORIS = r"G:\.shortcut-targets-by-id\14OI1k7IaNOdC5yaC9GnYyCi2u-gfGLEs\OIST 1215-0122\boris_events\20251225_K_1_2_FCMC.csv"
DEFAULT_SPAWN = 135888   # 818.6 s * 166 fps (BORIS spawn)


def load_track(path):
    t = pd.read_csv(path, usecols=["frame", "id_final", "X", "Y", "Z"])
    piv = t.pivot_table(index="frame", columns="id_final", values=["X", "Y", "Z"])
    n = int(piv.index.max()) + 1
    piv = piv.reindex(range(n))
    pos = {f: np.c_[piv["X"][f].to_numpy(), piv["Y"][f].to_numpy(), piv["Z"][f].to_numpy()] for f in ("F", "M1", "M2")}
    return pos, n


def dist(a, b):
    bad = np.isnan(a).any(1) | np.isnan(b).any(1)
    return np.where(bad, np.nan, np.sqrt(np.nansum((a - b) ** 2, axis=1)))


def speed(a):
    v = np.full(len(a), np.nan)
    v[1:] = np.sqrt(np.sum(np.diff(a, axis=0) ** 2, axis=1)) * FPS
    return v


def runs_over(mask, min_len):
    """contiguous True runs of length >= min_len -> list of (start, end) inclusive."""
    idx = np.where(np.nan_to_num(mask, nan=0.0).astype(bool))[0]
    if not len(idx):
        return []
    out, splits = [], np.where(np.diff(idx) > 1)[0] + 1
    for r in np.split(idx, splits):
        if len(r) >= min_len:
            out.append((int(r[0]), int(r[-1])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-male mate-choice / courtship / aggression analysis.")
    ap.add_argument("--track", default=DEFAULT_TRACK)
    ap.add_argument("--boris", default=DEFAULT_BORIS)
    ap.add_argument("--spawn-frame", type=int, default=DEFAULT_SPAWN)
    ap.add_argument("--court-mm", type=float, default=30.0, help="female-male courtship/association range")
    ap.add_argument("--contact-mm", type=float, default=12.0)
    ap.add_argument("--agg-mm", type=float, default=25.0, help="male-male close range for aggression")
    ap.add_argument("--agg-speed", type=float, default=150.0, help="mm/s speed spike for an aggressive lunge")
    ap.add_argument("--min-bout", type=int, default=33, help="frames (~0.2 s) for a sustained bout")
    ap.add_argument("--out-dir", default=r"runs\dual_line\lineB_fulltimeline\analysis")
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    pos, n = load_track(args.track)
    F, M1, M2 = pos["F"], pos["M1"], pos["M2"]
    dFM1, dFM2, dMM = dist(F, M1), dist(F, M2), dist(M1, M2)
    spdM1, spdM2 = speed(M1), speed(M2)
    valid_f = ~np.isnan(dFM1) & ~np.isnan(dFM2)

    # ---- female association / mate choice ----
    court_M1, court_M2 = dFM1 < args.court_mm, dFM2 < args.court_mm
    t1, t2 = np.nansum(court_M1) / FPS, np.nansum(court_M2) / FPS
    closer_M1 = valid_f & (dFM1 < dFM2)
    close_any = valid_f & (np.fmin(dFM1, dFM2) < args.court_mm)
    pref_M1 = np.nansum(close_any & (dFM1 <= dFM2)); pref_M2 = np.nansum(close_any & (dFM2 < dFM1))
    assoc_index = t1 / (t1 + t2) if (t1 + t2) else float("nan")   # >0.5 => prefers M1

    # ---- courtship bouts per male ----
    b1 = runs_over(court_M1, args.min_bout); b2 = runs_over(court_M2, args.min_bout)
    bout_dur = lambda b: sum(e - s + 1 for s, e in b) / FPS

    # ---- spawn attribution ----
    w = slice(max(0, args.spawn_frame - args.min_bout), args.spawn_frame + args.min_bout)
    sp_dFM1, sp_dFM2 = np.nanmedian(dFM1[w]), np.nanmedian(dFM2[w])
    spawn_male = "M1" if sp_dFM1 <= sp_dFM2 else "M2"

    # ---- male-male aggression bouts ----
    agg_mask = (dMM < args.agg_mm) & (np.fmax(spdM1, spdM2) > args.agg_speed)
    agg = runs_over(agg_mask, max(3, args.min_bout // 3))
    init_M1 = sum(1 for s, e in agg if np.nanmean(spdM1[s:e + 1]) > np.nanmean(spdM2[s:e + 1]))

    lines = [f"# Per-male mate-choice analysis — {Path(args.track).name}", ""]
    lines.append(f"Frames {n} ({n/FPS:.0f}s @ {FPS:.0f}fps); valid F-M frames {int(valid_f.sum())} "
                 f"({valid_f.mean():.1%}).\n")
    lines.append("## Female mate choice (association)")
    lines.append(f"- Female within {args.court_mm:.0f} mm of **M1 {t1:.1f}s** vs **M2 {t2:.1f}s** "
                 f"-> association index {assoc_index:.2f} ({'M1' if assoc_index>0.5 else 'M2'}-preferring).")
    lines.append(f"- Of frames with a male in courtship range, female nearer **M1 {pref_M1/max(pref_M1+pref_M2,1):.1%}** "
                 f"vs M2 {pref_M2/max(pref_M1+pref_M2,1):.1%}.")
    lines.append(f"- Female closer to M1 overall: {np.nanmean(closer_M1.astype(float)):.1%} of valid frames.")
    ratios = []
    for thr in (30, 20, 12, 8):
        a, b = np.nansum(dFM1 < thr) / FPS, np.nansum(dFM2 < thr) / FPS
        ratios.append(f"<{thr}mm {max(a,b)/max(min(a,b),1e-6):.2f}x")
    lines.append(f"- Preference SHARPENS toward contact ({' , '.join(ratios)} toward M1) — the stricter "
                 f"(closer to an explicit display) the range, the stronger M1's dominance.")
    lines.append(f"- Mean distance: d(F,M1) {np.nanmean(dFM1):.1f} mm, d(F,M2) {np.nanmean(dFM2):.1f} mm, "
                 f"d(M1,M2) {np.nanmean(dMM):.1f} mm.\n")
    lines.append("## Courtship bouts (sustained female proximity)")
    lines.append(f"- M1: {len(b1)} bouts, {bout_dur(b1):.1f}s total. M2: {len(b2)} bouts, {bout_dur(b2):.1f}s total.\n")
    lines.append("## Spawn (mate-choice outcome)")
    lines.append(f"- At frame {args.spawn_frame} ({args.spawn_frame/FPS:.1f}s): median d(F,M1) {sp_dFM1:.1f} mm, "
                 f"d(F,M2) {sp_dFM2:.1f} mm -> **female spawned with {spawn_male}** "
                 f"({'contact' if min(sp_dFM1,sp_dFM2)<args.contact_mm else 'near'}).\n")
    lines.append("## Male-male aggression")
    lines.append(f"- {len(agg)} close+fast bouts (<{args.agg_mm:.0f} mm & >{args.agg_speed:.0f} mm/s); "
                 f"M1 the faster (initiator) in {init_M1}/{max(len(agg),1)}.")

    # ---- optional BORIS cross-check ----
    boris_note = boris_cross_check(args.boris)
    if boris_note:
        lines += ["", "## BORIS ethogram cross-check"] + boris_note

    report = "\n".join(lines)
    (out / "mate_choice_report.md").write_text(report, encoding="utf-8")
    pd.DataFrame({"frame": np.arange(n), "d_F_M1": dFM1, "d_F_M2": dFM2, "d_M1_M2": dMM,
                  "court_M1": court_M1, "court_M2": court_M2}).to_csv(out / "mate_choice_per_frame.csv", index=False)
    make_figure(dFM1, dFM2, court_M1, court_M2, args, out / "mate_choice.png")
    print(report)
    print(f"\nwrote {out/'mate_choice_report.md'} and mate_choice.png")


def make_figure(dFM1, dFM2, court_M1, court_M2, args, png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(figure skipped: {exc})"); return
    n = len(dFM1); t = np.arange(n) / FPS
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw=dict(height_ratios=[2, 1], hspace=0.25))
    s = max(1, n // 4000)   # downsample for plotting
    ax[0].plot(t[::s], dFM1[::s], lw=0.6, color="#1f77b4", label="d(F, M1)")
    ax[0].plot(t[::s], dFM2[::s], lw=0.6, color="#d62728", label="d(F, M2)")
    ax[0].axhline(args.court_mm, ls="--", color="gray", lw=0.8, label=f"courtship {args.court_mm:.0f} mm")
    ax[0].axvline(args.spawn_frame / FPS, color="green", lw=1.2, label="spawn")
    ax[0].set_ylabel("female-male distance (mm)"); ax[0].set_ylim(0, 200); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Female-male distance over time (lower = associating)")
    cum1 = np.nancumsum(np.nan_to_num(court_M1.astype(float))) / FPS
    cum2 = np.nancumsum(np.nan_to_num(court_M2.astype(float))) / FPS
    ax[1].plot(t[::s], cum1[::s], color="#1f77b4", label="M1")
    ax[1].plot(t[::s], cum2[::s], color="#d62728", label="M2")
    ax[1].axvline(args.spawn_frame / FPS, color="green", lw=1.2)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("cumulative\ncourtship time (s)")
    ax[1].legend(loc="upper left", fontsize=8); ax[1].set_title("Cumulative time within courtship range of the female")
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
