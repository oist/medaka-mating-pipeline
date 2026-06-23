"""Compare manual BORIS behaviour coding against the reconstructed 3D trajectories.

BORIS exports a tabular event list whose ``Image index`` column is the top-camera
frame number (166 fps). Our ``*_3d_identified.csv`` is indexed by the same frame, so
the two align directly. This script:

  * parses BORIS START/STOP state events and POINT events into intervals,
  * loads the 3D tracks (female slot pinned by the appearance female-anchor),
  * derives inter-fish distances and speeds per frame,
  * checks that the trajectories corroborate the coded behaviours
    (female-directed events -> a male is close to the female; competitive events
    -> male-male approach), and
  * renders an ethogram+distance timeline, a spawn close-up, a behaviour/distance
    summary, and a per-event metrics CSV + markdown.

Female identity comes from the Detectron2 appearance female-anchor (K_1_2: ID_2).
The two males are NOT distinguishable by appearance, so BORIS ``m1``/``m2`` cannot be
mapped onto geometry slots ID_1/ID_3 — male-directed events are validated as a
*dyad*, not per individual.

Usage (K_1_2 defaults baked in):
    .venv\\Scripts\\python.exe scripts\\analysis\\compare_boris_trajectory.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

FPS = 166.0

# behaviour category -> colour
CAT_COLOR = {
    "Courtship": "#2E86C1",
    "Competitive": "#C0392B",
    "Mating": "#1E8E3E",
    "Other": "#7F8C8D",
}
# subject code -> (kind, who is involved)
#  f / m1 / m2 single; m1tf/m2tf male->female; m1tm2/m2tm1 male->male; m1m2 both males
FEMALE_SUBJECTS = {"f"}
MALE_FEMALE_SUBJECTS = {"m1tf", "m2tf"}
MALE_MALE_SUBJECTS = {"m1tm2", "m2tm1", "m1m2"}


def parse_boris(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    need = ["Subject", "Behavior", "Behavioral category", "Behavior type", "Time", "Image index"]
    raw = raw[need].copy()
    raw["Image index"] = pd.to_numeric(raw["Image index"], errors="coerce")
    raw["Time"] = pd.to_numeric(raw["Time"], errors="coerce")

    events = []
    open_state: dict[tuple, dict] = {}
    for _, r in raw.iterrows():
        subj, beh = r["Subject"], r["Behavior"]
        cat = r["Behavioral category"] if isinstance(r["Behavioral category"], str) and r["Behavioral category"] else "Other"
        typ = r["Behavior type"]
        frame = r["Image index"]
        t = r["Time"]
        if typ == "POINT":
            events.append(dict(subject=subj, behavior=beh, category=cat,
                               start_frame=frame, stop_frame=frame, start_s=t, stop_s=t,
                               dur_s=0.0, kind="point"))
        elif typ == "START":
            open_state[(subj, beh)] = dict(frame=frame, t=t)
        elif typ == "STOP":
            st = open_state.pop((subj, beh), None)
            if st is None:
                continue
            events.append(dict(subject=subj, behavior=beh, category=cat,
                               start_frame=st["frame"], stop_frame=frame,
                               start_s=st["t"], stop_s=t, dur_s=t - st["t"], kind="state"))
    ev = pd.DataFrame(events).sort_values("start_frame").reset_index(drop=True)
    return ev


def load_tracks(path: Path, female="ID_2", males=("ID_1", "ID_3")):
    df = pd.read_csv(path, usecols=["frame", "fish_id", "X", "Y", "Z"])
    piv = df.pivot_table(index="frame", columns="fish_id", values=["X", "Y", "Z"])
    nframes = int(piv.index.max()) + 1
    piv = piv.reindex(range(nframes))

    def xyz(fid):
        return np.c_[piv["X"][fid].to_numpy(), piv["Y"][fid].to_numpy(), piv["Z"][fid].to_numpy()]

    F = xyz(female)
    M1, M2 = xyz(males[0]), xyz(males[1])

    def d(a, b):
        return np.sqrt(np.nansum((a - b) ** 2, axis=1)) * np.where(
            np.isnan(a).any(1) | np.isnan(b).any(1), np.nan, 1.0)

    out = pd.DataFrame({
        "frame": np.arange(nframes),
        "d_F_M1": d(F, M1),
        "d_F_M2": d(F, M2),
        "d_M1_M2": d(M1, M2),
    })
    out["d_F_nearest"] = np.nanmin(out[["d_F_M1", "d_F_M2"]].to_numpy(), axis=1)
    out["nearest_male"] = np.where(out["d_F_M1"] <= out["d_F_M2"], "ID_1", "ID_3")

    def speed(a):
        v = np.full(len(a), np.nan)
        dd = np.sqrt(np.sum(np.diff(a, axis=0) ** 2, axis=1)) * FPS
        v[1:] = dd
        return v

    out["spd_F"] = speed(F)
    out["spd_M1"] = speed(M1)
    out["spd_M2"] = speed(M2)
    out["t_s"] = out["frame"] / FPS
    return out, dict(F=F, M1=M1, M2=M2)


def window_stat(series, f0, f1, fn=np.nanmean):
    f0, f1 = int(f0), int(max(f1, f0 + 1))
    seg = series[f0:f1 + 1]
    if np.all(np.isnan(seg)):
        return np.nan
    return float(fn(seg))


def per_event_metrics(ev: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    dFn = tr["d_F_nearest"].to_numpy()
    dMM = tr["d_M1_M2"].to_numpy()
    spdF = tr["spd_F"].to_numpy()
    near = tr["nearest_male"].to_numpy()
    rows = []
    for _, e in ev.iterrows():
        f0, f1 = e["start_frame"], e["stop_frame"]
        m = dict(e)
        m["d_F_nearest_mean"] = window_stat(dFn, f0, f1)
        m["d_F_nearest_min"] = window_stat(dFn, f0, f1, np.nanmin)
        m["d_M1_M2_mean"] = window_stat(dMM, f0, f1)
        m["spd_F_mean"] = window_stat(spdF, f0, f1)
        m["spd_F_max"] = window_stat(spdF, f0, f1, np.nanmax)
        seg_near = near[int(f0):int(f1) + 1]
        m["nearest_male_mode"] = pd.Series(seg_near).mode().iloc[0] if len(seg_near) else ""
        rows.append(m)
    return pd.DataFrame(rows)


def fig_timeline(ev, tr, out_png, title, t0=None, t1=None):
    if t0 is None:
        t0 = max(0.0, ev["start_s"].min() - 10)
    if t1 is None:
        t1 = ev["stop_s"].max() + 10
    fig, axes = plt.subplots(3, 1, figsize=(15, 8.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 2.2, 1.6], hspace=0.12))

    # --- ethogram ---
    ax = axes[0]
    cats = ["Courtship", "Competitive", "Mating"]
    yrow = {c: i for i, c in enumerate(cats)}
    for _, e in ev.iterrows():
        c = e["category"]
        if c not in yrow:
            continue
        y = yrow[c]
        if e["kind"] == "state":
            ax.add_patch(mpatches.Rectangle((e["start_s"], y - 0.34),
                         max(e["stop_s"] - e["start_s"], 0.4), 0.68,
                         color=CAT_COLOR[c], alpha=0.85))
        else:
            ax.plot(e["start_s"], y, "v", color=CAT_COLOR[c], ms=7)
    # spawn marker
    spawn = ev[ev["behavior"] == "Spawn"]
    for _, sp in spawn.iterrows():
        for a in axes:
            a.axvline(sp["start_s"], color="#8E44AD", lw=1.8, ls="--", zorder=5)
        ax.annotate("Spawn", (sp["start_s"], 2.55), color="#8E44AD", fontsize=10,
                    ha="center", fontweight="bold")
    ax.set_yticks(list(yrow.values()))
    ax.set_yticklabels(cats)
    ax.set_ylim(-0.7, 2.9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("BORIS\nethogram")

    # --- distances ---
    ax = axes[1]
    ax.plot(tr["t_s"], tr["d_F_M1"], lw=0.5, color="#2E86C1", alpha=0.8, label="female–male(ID_1)")
    ax.plot(tr["t_s"], tr["d_F_M2"], lw=0.5, color="#E67E22", alpha=0.8, label="female–male(ID_3)")
    ax.plot(tr["t_s"], tr["d_M1_M2"], lw=0.5, color="#7F8C8D", alpha=0.6, label="male–male")
    # shade mating (wrap) intervals
    for _, e in ev[ev["category"] == "Mating"].iterrows():
        if e["kind"] == "state":
            for a in (axes[1], axes[2]):
                a.axvspan(e["start_s"], e["stop_s"], color="#1E8E3E", alpha=0.12)
    ax.set_ylabel("inter-fish\ndistance (mm)")
    ax.set_ylim(0, 220)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(alpha=0.2)

    # --- speed ---
    ax = axes[2]
    sm = tr["spd_F"].rolling(25, min_periods=1, center=True).median()
    ax.plot(tr["t_s"], sm, lw=0.6, color="#C0392B", label="female speed (mm/s, 25-fr median)")
    ax.set_ylabel("female speed\n(mm/s)")
    ax.set_xlabel("time (s)")
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, np.nanpercentile(sm, 99.5))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_behavior_distance(ev_m, out_png):
    """Boxplot: female-nearest-male distance during each behaviour vs baseline."""
    groups = []
    labels = []
    order = [("Courtship", "Courtship"), ("Touch", "Courtship"),
             ("Wrap", "Mating"), ("Dash", "Competitive")]
    for beh, _cat in order:
        vals = ev_m[(ev_m["behavior"] == beh)]["d_F_nearest_mean"].dropna().to_numpy()
        if len(vals):
            groups.append(vals)
            labels.append(f"{beh}\n(n={len(vals)})")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, showfliers=False, widths=0.6)
    cols = [CAT_COLOR["Courtship"], CAT_COLOR["Courtship"], CAT_COLOR["Mating"], CAT_COLOR["Competitive"]]
    for patch, c in zip(bp["boxes"], cols[:len(groups)]):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for g, x in zip(groups, range(1, len(groups) + 1)):
        ax.scatter(np.random.normal(x, 0.04, len(g)), g, s=14, color="k", alpha=0.5, zorder=3)
    ax.set_ylabel("female–nearest-male distance during event (mm)")
    ax.set_title("Trajectory confirms behaviour: female–male distance by coded behaviour", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_spawn_closeup(ev, tr, out_png, half_window_s=30.0):
    spawn = ev[ev["behavior"] == "Spawn"]
    if spawn.empty:
        return None
    ts = float(spawn.iloc[0]["start_s"])
    t0, t1 = ts - half_window_s, ts + half_window_s
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1.4], hspace=0.12))
    ax = axes[0]
    ax.plot(tr["t_s"], tr["d_F_M1"], color="#2E86C1", lw=1.1, label="female–male(ID_1)")
    ax.plot(tr["t_s"], tr["d_F_M2"], color="#E67E22", lw=1.1, label="female–male(ID_3)")
    ax.plot(tr["t_s"], tr["d_M1_M2"], color="#7F8C8D", lw=0.9, alpha=0.7, label="male–male")
    for _, e in ev.iterrows():
        if e["start_s"] > t1 or e["stop_s"] < t0:
            continue
        if e["category"] == "Mating" and e["kind"] == "state":
            ax.axvspan(e["start_s"], e["stop_s"], color="#1E8E3E", alpha=0.15)
            ax.annotate(f"{e['subject']} {e['behavior']}", (e["start_s"], 200),
                        fontsize=8, rotation=90, va="top", color="#1E8E3E")
    ax.axvline(ts, color="#8E44AD", ls="--", lw=2)
    ax.annotate("Spawn", (ts, 210), color="#8E44AD", ha="center", fontweight="bold")
    ax.set_ylabel("distance (mm)"); ax.set_ylim(0, 220); ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Spawn close-up (t={ts:.1f} s, frame {int(spawn.iloc[0]['start_frame'])})",
                 fontsize=12, fontweight="bold")
    ax = axes[1]
    for col, c, lab in [("spd_F", "#C0392B", "female"), ("spd_M1", "#2E86C1", "male ID_1"), ("spd_M2", "#E67E22", "male ID_3")]:
        ax.plot(tr["t_s"], tr[col].rolling(15, min_periods=1, center=True).median(),
                color=c, lw=0.9, label=lab)
    ax.axvline(ts, color="#8E44AD", ls="--", lw=2)
    ax.set_ylabel("speed (mm/s)"); ax.set_xlabel("time (s)"); ax.set_xlim(t0, t1)
    ax.grid(alpha=0.2); ax.legend(loc="upper left", fontsize=9, ncol=3)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return ts


def main():
    ap = argparse.ArgumentParser()
    g = r"G:\.shortcut-targets-by-id\14OI1k7IaNOdC5yaC9GnYyCi2u-gfGLEs\OIST 1215-0122"
    ap.add_argument("--boris", default=fr"{g}\boris_events\20251225_K_1_2_FCMC.csv")
    ap.add_argument("--tracks", default=r"Z:\StephensU\medaka\reconstruction_3d\round5_260611\20251225_K_1_2_FCMC\round5_260611_3d_identified.csv")
    ap.add_argument("--female-slot", default="ID_2")
    ap.add_argument("--male-slots", nargs=2, default=["ID_1", "ID_3"])
    ap.add_argument("--out-dir", default=r"report\20260622\behavior_comparison")
    ap.add_argument("--label", default="20251225_K_1_2_FCMC")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("Parsing BORIS:", args.boris)
    ev = parse_boris(Path(args.boris))
    print(f"  {len(ev)} events; categories: {ev['category'].value_counts().to_dict()}")
    print("Loading tracks:", args.tracks)
    tr, _xyz = load_tracks(Path(args.tracks), args.female_slot, tuple(args.male_slots))
    print(f"  {len(tr)} frames; valid 3D dist frac F-nearest = {tr['d_F_nearest'].notna().mean():.2%}")

    ev_m = per_event_metrics(ev, tr)
    ev_m.to_csv(out / "events_with_trajectory_metrics.csv", index=False)

    # --- figures ---
    fig_timeline(ev, tr, out / "ethogram_distance_timeline.png",
                 f"{args.label}: BORIS ethogram vs 3D inter-fish distance & female speed")
    fig_behavior_distance(ev_m, out / "behavior_distance_boxplot.png")
    spawn_t = fig_spawn_closeup(ev, tr, out / "spawn_closeup.png")

    # --- corroboration summary ---
    def med(beh, col="d_F_nearest_mean"):
        v = ev_m[ev_m["behavior"] == beh][col].dropna()
        return (float(v.median()), len(v)) if len(v) else (np.nan, 0)

    baseline = float(np.nanmedian(tr["d_F_nearest"]))
    summary = {
        "baseline_median_d_F_nearest_mm": baseline,
        "courtship_median_d_F_nearest_mm": med("Courtship"),
        "wrap_median_d_F_nearest_mm": med("Wrap"),
        "touch_median_d_F_nearest_mm": med("Touch"),
        "dash_median_d_M1_M2_mm": med("Dash", "d_M1_M2_mean"),
        "spawn_time_s": spawn_t,
    }

    # wrap accept vs reject (doc: accepted ~27 s, rejected ~3 s)
    wraps = ev_m[ev_m["behavior"] == "Wrap"].copy()
    if len(wraps):
        wraps["accepted"] = wraps["dur_s"] >= 10.0
        acc = wraps[wraps["accepted"]]
        rej = wraps[~wraps["accepted"]]
        summary["wrap_accepted_n"] = int(len(acc))
        summary["wrap_rejected_n"] = int(len(rej))
        summary["wrap_accepted_med_dur_s"] = float(acc["dur_s"].median()) if len(acc) else np.nan
        summary["wrap_rejected_med_dur_s"] = float(rej["dur_s"].median()) if len(rej) else np.nan

    # which male was nearest at spawn (father-by-proximity proxy)
    if spawn_t is not None:
        sp_frame = int(ev[ev["behavior"] == "Spawn"].iloc[0]["start_frame"])
        win = tr.iloc[max(0, sp_frame - 166):sp_frame + 166]
        summary["spawn_nearest_male_mode"] = win["nearest_male"].mode().iloc[0]
        summary["spawn_d_F_M1_mm"] = float(np.nanmedian(win["d_F_M1"]))
        summary["spawn_d_F_M2_mm"] = float(np.nanmedian(win["d_F_M2"]))

    # behaviour counts table
    counts = (ev.groupby(["category", "behavior"])
                .agg(n=("behavior", "size"), total_s=("dur_s", "sum"))
                .reset_index().sort_values(["category", "n"], ascending=[True, False]))
    counts.to_csv(out / "behavior_counts.csv", index=False)

    # markdown
    md = [f"# BORIS behaviour vs 3D trajectory — {args.label}", ""]
    md.append("Frame alignment: BORIS `Image index` == 3D `frame` (top cam, 166 fps). "
              "Female = appearance-anchor slot **%s**; the two males (%s) are not "
              "appearance-distinguishable, so male-directed events are validated as a dyad.\n" %
              (args.female_slot, ", ".join(args.male_slots)))
    md.append("## Does the geometry corroborate the coded behaviour?\n")
    md.append("| measure | value |\n|---|---|")
    md.append(f"| baseline median female–nearest-male distance | {baseline:.1f} mm |")
    md.append(f"| during Courtship (n={med('Courtship')[1]}) | {med('Courtship')[0]:.1f} mm |")
    md.append(f"| during Touch (n={med('Touch')[1]}) | {med('Touch')[0]:.1f} mm |")
    md.append(f"| during Wrap (n={med('Wrap')[1]}) | {med('Wrap')[0]:.1f} mm |")
    md.append(f"| during Dash — male–male dist (n={med('Dash','d_M1_M2_mean')[1]}) | {med('Dash','d_M1_M2_mean')[0]:.1f} mm |")
    md.append("")
    if len(wraps):
        md.append("## Wrap acceptance (doc: accepted ~27 s vs rejected ~3 s)\n")
        md.append(f"- accepted (≥10 s): n={summary['wrap_accepted_n']}, median {summary.get('wrap_accepted_med_dur_s', float('nan')):.1f} s")
        md.append(f"- rejected (<10 s): n={summary['wrap_rejected_n']}, median {summary.get('wrap_rejected_med_dur_s', float('nan')):.1f} s\n")
    if spawn_t is not None:
        md.append("## Spawn event\n")
        md.append(f"- Spawn at t={spawn_t:.1f} s (frame {sp_frame}). Nearest male in ±1 s window: "
                  f"**{summary['spawn_nearest_male_mode']}** "
                  f"(median F–ID_1 {summary['spawn_d_F_M1_mm']:.1f} mm vs F–ID_3 {summary['spawn_d_F_M2_mm']:.1f} mm).")
        md.append("- The trajectory identifies *which slot* mated; mapping that slot to BORIS m1/m2 "
                  "needs the (open) male-ID problem solved.\n")
    md.append("## Behaviour inventory (BORIS)\n")
    md.append("| category | behavior | n | total (s) |\n|---|---|---|---|")
    for _, r in counts.iterrows():
        md.append(f"| {r['category']} | {r['behavior']} | {int(r['n'])} | {r['total_s']:.1f} |")
    (out / "behavior_trajectory_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nWrote outputs to", out)


if __name__ == "__main__":
    main()
