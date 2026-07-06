"""Generate poster figure library for the medaka female-mate-choice project.

Every figure is built from REAL pipeline output (CSV/JSON under runs/dual_line/...)
or from validated numbers quoted in the per-experiment FINDINGS / report files.
Schematics (the question, the rig, the dual-line method) are drawn.

Outputs -> report/20260630/figures/*.png  (300 dpi, poster quality)

Re-runnable:
    .venv\\Scripts\\python.exe report\\20260630\\build_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

# ----------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REPO = Path(r"C:\Users\machi\Documents\GitHub\medaka-mating-pipeline")
LINEB = REPO / "runs" / "dual_line" / "lineB_fulltimeline"
EXT = REPO / "runs" / "dual_line" / "extend_k11_k13"
BORIS = REPO / "report" / "20260622" / "behavior_comparison"

# Per-experiment analysis dirs + final tracks (all use the interp+debounced track).
ANALYSIS = {
    "K_1_1": EXT / "K_1_1" / "analysis",
    "K_1_2": LINEB / "analysis",  # K_1_2 debounced ≈ interp; analysis/ has temporal too
    "K_1_3": EXT / "K_1_3" / "analysis",
}
TRACK = {
    "K_1_1": EXT / "K_1_1" / "round5_260611_3d_individual_id_interp_debounced.csv",
    "K_1_2": LINEB / "step2" / "round5_260611_3d_individual_id_interp_debounced.csv",
    "K_1_3": EXT / "K_1_3" / "round5_260611_3d_individual_id_interp_debounced.csv",
}
EXPS = ["K_1_1", "K_1_2", "K_1_3"]
FPS = 166.0

# ----------------------------------------------------------------------------- palette
F_COL = "#D55E00"     # female (vermilion)
M1_COL = "#0072B2"    # preferred / chosen male (blue)
M2_COL = "#7F7F7F"    # other male (grey)
NAVY = "#1F3A5F"
ACCENT = "#2E86C1"
GREEN = "#1E8E3E"
RED = "#B03030"
AMBER = "#B76E00"
GREY = "#888888"
LIGHT = "#EEF2F7"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
})

# ----------------------------------------------------------------------------- N=3 facts
# Source: per-experiment mate_choice_report.md / *.json + FINDINGS_N3_mate_choice.md.
DATA = {
    "K_1_1": dict(
        assoc=0.80, near_M1=563.3, near_M2=138.4,
        court_M1=544.7, court_M2=127.3, court_bouts_M1=374, court_bouts_M2=158,
        court_s=672.0, agg_s=30.8, c2a=21.84,
        sharp={30: 4.07, 20: 5.34, 12: 6.79, 8: 10.82},
        spawn_frame=82288, spawn_t=495.7, spawn_dM1=5.4, spawn_dM2=110.5,
        female_hi=99.8, boris_other=9.5, boris_chosen=118.2, boris_ratio=12.4,
        female_balacc=0.862, female_posbase=0.257, male_balacc=None,
        pref_early=0.75, pref_late=0.88,
    ),
    "K_1_2": dict(
        assoc=0.61, near_M1=287.31, near_M2=180.7,
        court_M1=271.25, court_M2=162.21, court_bouts_M1=275, court_bouts_M2=210,
        court_s=433.46, agg_s=17.7, c2a=24.49,
        sharp={30: 1.59, 20: 1.822, 12: 1.806, 8: 1.891},
        spawn_frame=135888, spawn_t=818.6, spawn_dM1=7.44, spawn_dM2=11.54,
        female_hi=99.0, boris_other=41.8, boris_chosen=103.6, boris_ratio=2.5,
        female_balacc=None, female_posbase=None, male_balacc=0.936,  # K_1_2 male-male (reversal)
        pref_early=0.6534, pref_late=0.6601,
    ),
    "K_1_3": dict(
        assoc=0.71, near_M1=639.9, near_M2=263.2,
        court_M1=614.4, court_M2=241.5, court_bouts_M1=467, court_bouts_M2=251,
        court_s=855.9, agg_s=44.4, c2a=19.28,
        sharp={30: 2.43, 20: 2.54, 12: 2.51, 8: 3.68},
        spawn_frame=121602, spawn_t=732.5, spawn_dM1=4.9, spawn_dM2=124.1,
        female_hi=99.8, boris_other=6.7, boris_chosen=99.2, boris_ratio=14.7,
        female_balacc=0.956, female_posbase=0.437, male_balacc=0.838,
        pref_early=0.73, pref_late=0.65,
    ),
}

# ----------------------------------------------------------------------------- helpers
def save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gen ] {name}  ({out.stat().st_size/1e3:.0f} kB)")


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def barlabel(ax, bars, fmt="{:.0f}", dy=2, **kw):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", xytext=(0, dy),
                    textcoords="offset points", **kw)


_TRACK_CACHE = {}


def load_track(exp, cols):
    key = (exp, tuple(cols))
    if key not in _TRACK_CACHE:
        _TRACK_CACHE[key] = pd.read_csv(TRACK[exp], usecols=list(cols))
    return _TRACK_CACHE[key]


def load_per_frame(exp):
    df = pd.read_csv(ANALYSIS[exp] / "mate_choice_per_frame.csv")
    for c in ("court_M1", "court_M2"):
        df[c] = df[c].astype(str).str.strip().eq("True")
    df["t"] = df["frame"] / FPS
    return df


def load_temporal(exp):
    return pd.read_csv(ANALYSIS[exp] / "mate_choice_temporal.csv")


# =============================================================================
#  SECTION 1 — schematics: question, challenge, setup
# =============================================================================
def fig_question():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # tank
    tank = FancyBboxPatch((0.4, 0.6), 9.2, 4.4, boxstyle="round,pad=0.02,rounding_size=0.15",
                          fc="#EAF4FB", ec=ACCENT, lw=2)
    ax.add_patch(tank)
    ax.text(5.0, 5.55, "1 female  +  2 males  (same strain, visually identical)",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)

    def fish(x, y, col, label, scale=1.0, sym=""):
        ax.add_patch(plt.matplotlib.patches.Ellipse((x, y), 1.5*scale, 0.6*scale,
                     fc=col, ec="white", lw=1.5, zorder=3))
        ax.add_patch(plt.Polygon([[x-0.75*scale, y], [x-1.15*scale, y+0.28*scale],
                     [x-1.15*scale, y-0.28*scale]], closed=True, fc=col, ec="white", zorder=3))
        ax.text(x, y-0.62*scale, label, ha="center", va="top", fontsize=11.5,
                fontweight="bold", color=col)
        if sym:
            ax.text(x+0.95*scale, y+0.3*scale, sym, ha="center", fontsize=13, color=col)

    fish(3.0, 3.4, F_COL, "♀ female", 1.25)
    fish(6.6, 4.1, M1_COL, "♂ male 1", 1.0)
    fish(6.9, 2.2, M2_COL, "♂ male 2", 1.0)
    ax.add_patch(FancyArrowPatch((4.0, 3.5), (5.7, 4.0), arrowstyle="-|>",
                 mutation_scale=16, lw=2, color=M1_COL, ls="-"))
    ax.add_patch(FancyArrowPatch((4.0, 3.2), (5.9, 2.4), arrowstyle="-|>",
                 mutation_scale=16, lw=2, color=M2_COL, ls=(0, (4, 3))))
    ax.text(5.0, 1.15,
            "Which male does she choose?   Is the choice driven by courtship — or by male–male aggression?",
            ha="center", fontsize=11.5, style="italic", color="#333333")
    fig.tight_layout()
    save(fig, "s1_question.png")


def fig_challenge():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(5, 5.6, "The hard part: the two males are indistinguishable to the eye",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)
    # human/BORIS side
    box1 = FancyBboxPatch((0.4, 1.2), 4.3, 3.4, boxstyle="round,pad=0.02,rounding_size=0.12",
                          fc="#FBEEE6", ec=AMBER, lw=1.8); ax.add_patch(box1)
    ax.text(2.55, 4.2, "Manual scoring (BORIS)", ha="center", fontsize=12, fontweight="bold", color=AMBER)
    ax.text(2.55, 3.45, "can label the female,\nbut the two males collapse\ninto one  “male”  category",
            ha="center", fontsize=10.5, color="#5a4327")
    ax.text(2.55, 1.75, "→ only a  ♀–♂dyad,\nno per-male mate choice", ha="center",
            fontsize=10.5, fontweight="bold", color=RED)
    # ML side
    box2 = FancyBboxPatch((5.3, 1.2), 4.3, 3.4, boxstyle="round,pad=0.02,rounding_size=0.12",
                          fc="#E9F6EE", ec=GREEN, lw=1.8); ax.add_patch(box2)
    ax.text(7.45, 4.2, "This pipeline", ha="center", fontsize=12, fontweight="bold", color=GREEN)
    ax.text(7.45, 3.45, "recovers all three identities\nautomatically &  continuously\nin 3D (1 ♀ + 2 ♂)",
            ha="center", fontsize=10.5, color="#1c4b30")
    ax.text(7.45, 1.75, "→ per-male choice\nbecomes measurable", ha="center",
            fontsize=10.5, fontweight="bold", color=GREEN)
    ax.add_patch(FancyArrowPatch((4.8, 2.9), (5.25, 2.9), arrowstyle="-|>",
                 mutation_scale=20, lw=2.4, color=NAVY))
    fig.tight_layout()
    save(fig, "s1_challenge.png")


def fig_setup():
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.text(5, 6.05, "Experimental setup — 3 calibrated cameras around a 20 cm tank",
            ha="center", fontsize=12.5, fontweight="bold", color=NAVY)
    # tank cube (isometric-ish)
    cube = Rectangle((3.6, 2.0), 2.8, 2.4, fc="#EAF4FB", ec=ACCENT, lw=2); ax.add_patch(cube)
    # offset top face
    ax.add_patch(plt.Polygon([[3.6, 4.4], [4.3, 5.0], [7.1, 5.0], [6.4, 4.4]],
                 closed=True, fc="#D6EAF8", ec=ACCENT, lw=1.5))
    ax.add_patch(plt.Polygon([[6.4, 4.4], [7.1, 5.0], [7.1, 2.6], [6.4, 2.0]],
                 closed=True, fc="#CFE3F3", ec=ACCENT, lw=1.5))
    ax.text(5.0, 3.2, "20×20×20 cm\n1 ♀ + 2 ♂", ha="center", fontsize=11,
            fontweight="bold", color=NAVY)

    def cam(x, y, label, axis):
        ax.add_patch(FancyBboxPatch((x-0.55, y-0.35), 1.1, 0.7,
                     boxstyle="round,pad=0.02,rounding_size=0.08", fc=NAVY, ec="white", lw=1.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
        ax.text(x, y-0.55, axis, ha="center", va="top", fontsize=8.5, color=GREY)

    cam(5.0, 5.6, "top cam", "XY plane")
    cam(1.5, 3.2, "side cam 1", "XZ plane")
    cam(8.5, 3.2, "side cam 2", "YZ plane")
    for (x0, y0, x1, y1) in [(5.0, 5.25, 5.0, 4.7), (2.05, 3.2, 3.5, 3.2), (7.95, 3.2, 6.5, 3.2)]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=13, lw=1.6, color=ACCENT, ls=(0, (3, 2))))
    specs = ("166 fps   ·   ~35 min / trial (348,600 frames)   ·   N = 3 trials (K_1_1, K_1_2, K_1_3)\n"
             "learned 6D→3D calibration (absorbs water/glass refraction)   ·   0.90 mm mean accuracy")
    ax.text(5.0, 1.1, specs, ha="center", fontsize=10, color="#333333")
    fig.tight_layout()
    save(fig, "s1_setup.png")


# =============================================================================
#  SECTION 2 — calibration
# =============================================================================
def fig_calibration_summary():
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    metrics = ["mean", "RMSE", "max"]
    vals = [0.90, 1.00, 2.15]
    bars = ax.bar(metrics, vals, color=[ACCENT, NAVY, AMBER], width=0.6, edgecolor="white")
    barlabel(ax, bars, fmt="{:.2f} mm", fontsize=11, fontweight="bold", color=NAVY)
    ax.axhspan(0, 0, color="none")
    ax.set_ylim(0, 2.7)
    ax.set_ylabel("3D reconstruction error (mm)")
    ax.set_title("Calibration accuracy (228 held-out points)\nlearned polynomial 6D pixel → 3D world")
    ax.text(0.98, 0.94, "error ≈ 3–7 % of a 30 mm fish body\n→ trustworthy for centroids & inter-fish distance",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, style="italic", color=GREEN,
            bbox=dict(boxstyle="round", fc="#E9F6EE", ec=GREEN, alpha=0.9))
    despine(ax)
    fig.tight_layout()
    save(fig, "s2_calibration_summary.png")


# =============================================================================
#  SECTION 3 — ML tracking
# =============================================================================
def fig_dual_line():
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc="white", fs=9.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                     lw=1.6, fc=fc, ec=ec, zorder=2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold", zorder=3)

    def arrow(x0, y0, x1, y1, col):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=16, lw=1.9, color=col, zorder=1))

    ax.text(5.0, 5.78, "Two independent lines — each carries the frames the other fails on",
            ha="center", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.15, 5.5, "LINE A — geometry", color=ACCENT, fontsize=11.5, fontweight="bold")
    box(0.2, 4.35, 2.7, 0.95, "3D reconstruction\n(reproject via calibration)", "#EAF2FB", ACCENT, tc=NAVY)
    box(3.3, 4.35, 2.7, 0.95, "velocity/gap-aware\nHungarian ID-link", "#EAF2FB", ACCENT, tc=NAVY)
    arrow(2.9, 4.82, 3.3, 4.82, ACCENT)
    ax.text(1.55, 4.12, "strong in open water · smooth-swaps at crossings", color=ACCENT,
            fontsize=8.3, ha="center", style="italic")

    ax.text(0.15, 2.05, "LINE B — appearance", color=GREEN, fontsize=11.5, fontweight="bold")
    box(0.2, 0.85, 2.7, 0.95, "Detectron2 masks →\n33._square_crop", "#E9F6EE", GREEN, tc="#0B5128")
    box(3.3, 0.85, 2.7, 0.95, "per-frame F / M1 / M2\nclassifiers (+ top model)", "#E9F6EE", GREEN, tc="#0B5128")
    arrow(2.9, 1.32, 3.3, 1.32, GREEN)
    ax.text(1.55, 0.6, "strong at contact / crossings", color=GREEN, fontsize=8.3, ha="center", style="italic")

    box(6.5, 2.6, 1.9, 1.0, "FUSE\nby scene\nstrength", NAVY, NAVY, fs=9.8)
    box(8.55, 2.7, 1.3, 0.8, "interpolate\nresidual", "#3A5C82", NAVY, fs=8.8)
    arrow(6.0, 4.82, 7.0, 3.6, ACCENT)
    arrow(6.0, 1.32, 7.0, 2.7, GREEN)
    arrow(8.4, 3.1, 8.55, 3.1, NAVY)
    box(6.5, 0.55, 3.35, 1.0, "Gap-free 3-fish track\n♀ + M1 + M2 · 98.6% high-conf", "#0B5128", "#0B5128", fs=10)
    arrow(9.2, 2.7, 9.2, 1.55, NAVY)
    fig.tight_layout()
    save(fig, "s3_dual_line.png")


def fig_reversal_auc():
    groups = ["Male vs male\nidentity", "Female vs male"]
    before = [0.57, 0.79]
    after = [0.994, 1.00]
    nb = ["raw labels\n(ARI ≈ 0)", "pooled\nLOWO 0.83/0.75"]
    na = ["LOSO\nAUC 0.994", "LOWO 1.00\nAUC 0.982"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w/2, before, w, color=GREY, label="Before (cross-window label noise)", edgecolor="white")
    b2 = ax.bar(x + w/2, after, w, color=GREEN, label="After swap-correction (window_align)", edgecolor="white")
    ax.axhline(0.5, ls="--", lw=1.2, color=RED, alpha=0.8)
    ax.text(1.5, 0.515, "chance (0.5)", color=RED, fontsize=9, ha="right", va="bottom")
    ax.scatter([-w/2], [0.477], marker="_", s=600, color=AMBER, zorder=5)
    ax.text(-w/2, 0.43, "position-only\nbaseline 0.48", color=AMBER, fontsize=8, ha="center", va="top")
    for r, n in zip(b1, nb):
        ax.annotate(n, (r.get_x()+r.get_width()/2, r.get_height()), ha="center", va="bottom",
                    fontsize=8.5, color="#444", xytext=(0, 3), textcoords="offset points")
    for r, n in zip(b2, na):
        ax.annotate(n, (r.get_x()+r.get_width()/2, r.get_height()-0.02), ha="center", va="top",
                    fontsize=8.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("identity separability\n(AUC / balanced-acc · 0.5 = chance)")
    ax.set_title("It was label noise, not weak appearance:\nswap-correction unlocks BOTH male and female identity")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    despine(ax)
    fig.tight_layout()
    save(fig, "s3_reversal_auc.png")


def fig_separability_n3():
    exps = ["K_1_1", "K_1_3"]
    bal = [DATA[e]["female_balacc"] for e in exps]
    pos = [DATA[e]["female_posbase"] for e in exps]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(exps)); w = 0.36
    b1 = ax.bar(x - w/2, bal, w, color=GREEN, edgecolor="white", label="appearance classifier (balanced acc.)")
    b2 = ax.bar(x + w/2, pos, w, color=AMBER, edgecolor="white", label="position-only baseline")
    ax.axhline(0.5, ls="--", lw=1.2, color=RED, alpha=0.8)
    ax.text(len(exps)-0.5, 0.515, "chance", color=RED, fontsize=9, ha="right", va="bottom")
    barlabel(ax, b1, fmt="{:.2f}", fontsize=10, fontweight="bold", color=NAVY)
    barlabel(ax, b2, fmt="{:.2f}", fontsize=10, color="#7a5300")
    ax.set_xticks(x); ax.set_xticklabels(exps)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("female-vs-male separability")
    ax.set_title("Identity lives in the appearance, not the position\n(corrected per-experiment classifiers)")
    ax.legend(loc="upper center", fontsize=8.6, framealpha=0.95, ncol=1)
    ax.text(0.5, -0.20, "Male-vs-male is the harder task (correct structure): K_1_3 balanced acc. 0.84.   "
            "K_1_2 separability → see “the reversal”.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.4, style="italic", color=GREY)
    despine(ax)
    fig.tight_layout()
    save(fig, "s3_separability_n3.png")


def fig_track_quality(comp):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    hi = comp["hi_pct"]; interp = comp["interp_pct"]; amb = comp["amb_pct"]
    left = 0.0
    for label, val, col in [("high", hi, GREEN), ("interp", interp, ACCENT), ("amb", amb, RED)]:
        axL.barh(0, val, left=left, color=col, edgecolor="white", height=0.5)
        left += val
    axL.set_xlim(0, 100); axL.set_ylim(-0.6, 0.9); axL.set_yticks([])
    axL.set_xlabel("% of all fish-frames")
    axL.set_title(f"K_1_2 track-wide identity\n(348,600 frames × 3 fish)")
    axL.annotate(f"high-confidence  {hi:.1f}%", (hi/2, 0.32), ha="center", fontsize=10.5,
                 color="white", fontweight="bold")
    axL.annotate(f"interp {interp:.1f}%", (99.5, 0.78), ha="right", fontsize=8.5, color=ACCENT)
    axL.annotate(f"ambiguous {amb:.2f}%", (100, -0.4), ha="right", fontsize=8.5, color=RED)
    axL.text(0, -0.5, "exactly 1 ♀ + 1 M1 + 1 M2 every frame · gap-free", fontsize=8.5,
             color=GREY, style="italic")
    despine(axL, keep=("bottom",))

    fish = ["♀", "M1", "M2"]
    vals = [comp["F_hi"], comp["M1_hi"], comp["M2_hi"]]
    bars = axR.bar(fish, vals, color=[F_COL, M1_COL, M2_COL], edgecolor="white", width=0.62)
    barlabel(axR, bars, fmt="{:.1f}%", fontsize=10.5, fontweight="bold", color=NAVY)
    axR.set_ylim(70, 102); axR.set_ylabel("% frames high-confidence")
    axR.set_title("Per-fish high-confidence")
    despine(axR)
    fig.tight_layout()
    save(fig, "s3_track_quality.png")


def fig_id_composition(src):
    order = ["high", "bridged", "edge", "motion_debounced", "motion_localized", "ambiguous"]
    labels = ["high-conf (direct)", "bridged (certain)", "edge fill", "motion debounce",
              "motion-localized", "ambiguous"]
    cols = [GREEN, "#5FA86A", ACCENT, "#7FB2D6", AMBER, RED]
    total = sum(src.values())
    fig, ax = plt.subplots(figsize=(9.6, 2.7))
    left = 0.0
    for k, lab, c in zip(order, labels, cols):
        pct = 100 * src.get(k, 0) / total
        ax.barh(0, pct, left=left, color=c, edgecolor="white", height=0.6)
        if pct > 3.0:
            ax.text(left + pct/2, 0, f"{lab}\n{pct:.1f}%", ha="center", va="center",
                    fontsize=9, color="white" if c in (GREEN, ACCENT, RED) else "#222", fontweight="bold")
        left += pct
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.set_xlabel("% of all fish-frames (3 fish × 348,600)")
    ax.set_title("How each fish-frame's identity was decided  —  90.9 % direct, the rest recovered & flagged")
    # legend for small slices
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(handles, [f"{l} ({100*src.get(k,0)/total:.2f}%)" for k, l in zip(order, labels)],
              loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=8.2, frameon=False)
    despine(ax, keep=("bottom",))
    fig.tight_layout()
    save(fig, "s3_id_composition.png")


def fig_confidence_ladder():
    stages = ["side\nappearance", "+ top-view\nfusion", "+ interpolate\nresidual"]
    vals = [78.5, 88.2, 98.4]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    x = np.arange(3)
    bars = ax.bar(x, vals, color=[M2_COL, ACCENT, GREEN], width=0.6, edgecolor="white")
    barlabel(ax, bars, fmt="{:.1f}%", fontsize=11, fontweight="bold", color=NAVY)
    # step connectors
    for i in range(2):
        ax.annotate("", (x[i+1]-0.3, vals[i+1]), (x[i]+0.3, vals[i]),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
        ax.text((x[i]+x[i+1])/2, (vals[i]+vals[i+1])/2 + 1.0, f"+{vals[i+1]-vals[i]:.1f}",
                ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylim(60, 104); ax.set_ylabel("% frames high-confidence")
    ax.set_title("Recovering the male contact core (K_1_2)\nfemale already 97 → 99 %")
    despine(ax)
    fig.tight_layout()
    save(fig, "s3_confidence_ladder.png")


def fig_trajectory_topdown():
    exp = "K_1_2"
    df = load_track(exp, ["frame", "id_final", "X", "Y"])
    sf = DATA[exp]["spawn_frame"]
    lo, hi = sf - 2500, sf + 700  # ~19 s: female approach into the spawn contact
    win = df[(df.frame >= lo) & (df.frame <= hi)].dropna(subset=["X", "Y"])
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    for lab, col in [("M2", M2_COL), ("M1", M1_COL), ("F", F_COL)]:
        d = win[win.id_final == lab].sort_values("frame")
        lw = 2.1 if lab == "F" else 1.5
        ax.plot(d.X, d.Y, "-", color=col, lw=lw, alpha=0.8,
                label={"F": "♀ female", "M1": "M1 (chosen)", "M2": "M2"}[lab])
        if len(d):
            ax.scatter(d.X.iloc[0], d.Y.iloc[0], color=col, s=40, zorder=5,
                       edgecolor="white", linewidth=1.0, marker="o")        # start
            ax.scatter(d.X.iloc[-1], d.Y.iloc[-1], color=col, s=70, zorder=5,
                       edgecolor="white", linewidth=1.2, marker="s")        # end
    # spawn marker
    sp = win[win.frame.between(sf-30, sf+30) & (win.id_final == "F")]
    if len(sp):
        sx, sy = sp.X.mean(), sp.Y.mean()
        ax.scatter(sx, sy, marker="*", s=460, color="#FFD23F", edgecolor=NAVY, lw=1.3, zorder=6)
        ax.annotate("spawn", (sx, sy), xytext=(12, 10), textcoords="offset points",
                    fontsize=10.5, fontweight="bold", color=NAVY)
    # zoom to the data with a margin (the fish stay in one region of the tank)
    xlo, xhi = win.X.min(), win.X.max()
    ylo, yhi = win.Y.min(), win.Y.max()
    m = 12
    ax.set_xlim(xlo - m, xhi + m); ax.set_ylim(ylo - m, yhi + m)
    ax.add_patch(Rectangle((0, 0), 200, 200, fill=False, ec="#CCCCCC", lw=1.2, ls="--"))
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title("3D track held through the spawn approach\n(top-down zoom, ~19 s; ○ start, ■ end)")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    ax.text(0.99, 0.01, "dashed = 20 cm tank edge", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=GREY, style="italic")
    despine(ax)
    fig.tight_layout()
    save(fig, "s3_trajectory_topdown.png")


def fig_interfish_distance():
    exp = "K_1_2"
    df = load_per_frame(exp)
    st = 40  # downsample stride for plotting
    d = df.iloc[::st]
    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    ax.plot(d.t, d.d_F_M1, color=M1_COL, lw=0.6, alpha=0.8, label="♀–M1")
    ax.plot(d.t, d.d_F_M2, color=M2_COL, lw=0.6, alpha=0.7, label="♀–M2")
    ax.plot(d.t, d.d_M1_M2, color=AMBER, lw=0.5, alpha=0.5, label="M1–M2")
    ax.axhline(12, ls="--", lw=1, color=RED, alpha=0.7)
    ax.text(2090, 13, "contact ≤12 mm", color=RED, fontsize=8.5, ha="right")
    ax.axvline(DATA[exp]["spawn_t"], color="#FFD23F", lw=2, alpha=0.9)
    ax.text(DATA[exp]["spawn_t"]+8, 195, "spawn", color=NAVY, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 2100); ax.set_ylim(0, 210)
    ax.set_xlabel("time (s)"); ax.set_ylabel("inter-fish distance (mm)")
    ax.set_title("K_1_2 inter-fish distance over the full trial — the contact regime the tracker must survive")
    ax.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.95)
    despine(ax)
    fig.tight_layout()
    save(fig, "s3_interfish_distance.png")


# =============================================================================
#  SECTION 4 — biology: mate choice
# =============================================================================
def fig_matechoice_master(exp):
    df = load_per_frame(exp)
    tmp = load_temporal(exp)
    d = DATA[exp]
    win = int(FPS * 5)  # 5 s rolling median
    roll1 = df.d_F_M1.rolling(win, min_periods=int(win/3), center=True).median()
    roll2 = df.d_F_M2.rolling(win, min_periods=int(win/3), center=True).median()
    cum1 = (df.court_M1.cumsum() / FPS)
    cum2 = (df.court_M2.cumsum() / FPS)
    t = df.t.values
    spawn_t = d["spawn_t"]

    fig, axs = plt.subplots(3, 1, figsize=(7.6, 7.4), sharex=True,
                            gridspec_kw={"height_ratios": [1.25, 1, 0.85]})
    # (a) distance
    ax = axs[0]
    ax.plot(t, roll1, color=M1_COL, lw=1.8, label="♀–M1 (chosen)")
    ax.plot(t, roll2, color=M2_COL, lw=1.6, label="♀–M2")
    ax.axhline(30, ls=":", lw=1, color=GREY)
    ax.text(2090, 32, "courtship range 30 mm", color=GREY, fontsize=8, ha="right")
    ax.axvline(spawn_t, color="#FFD23F", lw=2)
    ax.set_ylabel("F–male dist (mm)\n5-s median")
    ax.set_ylim(0, 160)
    ax.set_title(f"{exp}  —  female mate choice   (association index {d['assoc']:.2f} toward M1)")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    despine(ax)
    # (b) cumulative courtship time
    ax = axs[1]
    ax.plot(t, cum1, color=M1_COL, lw=2.0, label=f"M1  {d['court_M1']:.0f} s")
    ax.plot(t, cum2, color=M2_COL, lw=1.8, label=f"M2  {d['court_M2']:.0f} s")
    ax.axvline(spawn_t, color="#FFD23F", lw=2)
    ax.fill_between(t, cum2, cum1, where=(cum1 >= cum2), color=M1_COL, alpha=0.10)
    ax.set_ylabel("cumulative time\nin courtship range (s)")
    ax.legend(loc="upper left", fontsize=8.5)
    despine(ax)
    # (c) preference index per bin
    ax = axs[2]
    tb = (tmp.t_start_s + tmp.t_end_s) / 2
    ax.plot(tb, tmp.pref_index_M1, "-o", color=NAVY, ms=4, lw=1.6)
    ax.axhline(0.5, ls="--", lw=1, color=RED, alpha=0.7)
    ax.fill_between(tb, 0.5, tmp.pref_index_M1, where=(tmp.pref_index_M1 >= 0.5),
                    color=M1_COL, alpha=0.18)
    ax.fill_between(tb, 0.5, tmp.pref_index_M1, where=(tmp.pref_index_M1 < 0.5),
                    color=M2_COL, alpha=0.18)
    ax.axvline(spawn_t, color="#FFD23F", lw=2)
    ax.set_ylim(0, 1); ax.set_ylabel("preference\nindex (M1)")
    ax.set_xlabel("time (s)")
    ax.text(2090, 0.92, "→ M1", color=M1_COL, fontsize=8.5, ha="right", fontweight="bold")
    ax.text(2090, 0.06, "→ M2", color=M2_COL, fontsize=8.5, ha="right", fontweight="bold")
    despine(ax)
    fig.tight_layout()
    save(fig, f"s4_matechoice_{exp}.png")


def fig_assoc_index_n3():
    vals = [DATA[e]["assoc"] for e in EXPS]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(EXPS, vals, color=M1_COL, edgecolor="white", width=0.6)
    ax.axhline(0.5, ls="--", lw=1.4, color=RED)
    ax.text(2.45, 0.515, "chance (no preference)", color=RED, fontsize=9, ha="right", va="bottom")
    barlabel(ax, bars, fmt="{:.2f}", dy=3, fontsize=12, fontweight="bold", color=NAVY)
    ax.set_ylim(0, 1.0); ax.set_ylabel("association index\n(share of close-range time with preferred male)")
    ax.set_title("Female prefers one male in every trial\n(N = 3, all above chance)")
    despine(ax)
    fig.tight_layout()
    save(fig, "s4_assoc_index_n3.png")


def fig_sharpening_n3():
    thr = [30, 20, 12, 8]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    cols = {"K_1_1": "#1B6CA8", "K_1_2": "#5B8FB0", "K_1_3": "#7E57C2"}
    for e in EXPS:
        y = [DATA[e]["sharp"][t] for t in thr]
        ax.plot(range(len(thr)), y, "-o", color=cols[e], lw=2, ms=6, label=f"{e} (assoc {DATA[e]['assoc']:.2f})")
        ax.annotate(f"{y[-1]:.1f}×", (len(thr)-1, y[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=10, fontweight="bold", color=cols[e], va="center")
    ax.axhline(1.0, ls="--", lw=1, color=GREY)
    ax.set_xticks(range(len(thr))); ax.set_xticklabels([f"≤{t} mm" for t in thr])
    ax.set_xlabel("distance threshold (tighter → closer to contact →)")
    ax.set_ylabel("preference ratio  (time near M1 ÷ M2)")
    ax.set_title("The preference sharpens toward contact\nstrongest exactly where mating happens")
    ax.legend(loc="upper left", fontsize=9)
    despine(ax)
    fig.tight_layout()
    save(fig, "s4_sharpening_n3.png")


def fig_courtship_M1M2_n3():
    m1 = [DATA[e]["court_M1"] for e in EXPS]
    m2 = [DATA[e]["court_M2"] for e in EXPS]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w/2, m1, w, color=M1_COL, edgecolor="white", label="M1 (preferred)")
    b2 = ax.bar(x + w/2, m2, w, color=M2_COL, edgecolor="white", label="M2")
    barlabel(ax, b1, fmt="{:.0f}s", fontsize=9.5, fontweight="bold", color=NAVY)
    barlabel(ax, b2, fmt="{:.0f}s", fontsize=9.5, color="#555")
    for i, e in enumerate(EXPS):
        r = DATA[e]["court_M1"] / DATA[e]["court_M2"]
        ax.text(i, max(m1[i], m2[i]) + 55, f"{r:.1f}×", ha="center", fontsize=10.5,
                fontweight="bold", color=M1_COL)
    ax.set_xticks(x); ax.set_xticklabels(EXPS)
    ax.set_ylim(0, max(m1) * 1.28)
    ax.set_ylabel("courtship-range time (s)")
    ax.set_title("The preferred male courts far more (N = 3)")
    ax.legend(loc="upper right", fontsize=9)
    despine(ax)
    fig.tight_layout()
    save(fig, "s4_courtship_M1M2_n3.png")


def fig_courtship_vs_aggression_n3():
    court = [DATA[e]["court_s"] for e in EXPS]
    agg = [DATA[e]["agg_s"] for e in EXPS]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w/2, court, w, color=GREEN, edgecolor="white", label="courtship")
    b2 = ax.bar(x + w/2, agg, w, color=RED, edgecolor="white", label="male–male aggression")
    barlabel(ax, b1, fmt="{:.0f}s", fontsize=9.5, fontweight="bold", color="#13501f")
    barlabel(ax, b2, fmt="{:.0f}s", fontsize=9.5, color=RED)
    for i, e in enumerate(EXPS):
        ax.text(i, court[i] + 50, f"{DATA[e]['c2a']:.0f}×", ha="center", fontsize=11,
                fontweight="bold", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(EXPS)
    ax.set_ylim(0, max(court) * 1.25)
    ax.set_ylabel("total time (s)")
    ax.set_title("Courtship — not aggression — tracks the choice\n(courtship : aggression ≈ 19–25× in every trial)")
    ax.legend(loc="upper left", fontsize=9)
    despine(ax)
    fig.tight_layout()
    save(fig, "s4_courtship_vs_aggression_n3.png")


def fig_temporal_stability_n3():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cols = {"K_1_1": "#1B6CA8", "K_1_2": "#5B8FB0", "K_1_3": "#7E57C2"}
    for e in EXPS:
        tmp = load_temporal(e)
        tb = (tmp.t_start_s + tmp.t_end_s) / 2
        ax.plot(tb, tmp.pref_index_M1, "-o", color=cols[e], ms=4, lw=1.7, label=e)
        ax.axvline(DATA[e]["spawn_t"], color=cols[e], lw=1.2, ls=":", alpha=0.7)
    ax.axhline(0.5, ls="--", lw=1.3, color=RED)
    ax.text(2090, 0.52, "chance", color=RED, fontsize=8.5, ha="right", va="bottom")
    ax.set_ylim(0, 1.05); ax.set_xlim(0, 2100)
    ax.set_xlabel("time (s)   ·   dotted line = spawn")
    ax.set_ylabel("preference index (toward preferred male)")
    ax.set_title("The female favours the eventual mate from the start\n(preference stable, not a late switch)")
    ax.legend(loc="lower right", fontsize=9, ncol=3)
    despine(ax)
    fig.tight_layout()
    save(fig, "s4_temporal_stability_n3.png")


def fig_spawn_n3():
    dM1 = [DATA[e]["spawn_dM1"] for e in EXPS]
    dM2 = [DATA[e]["spawn_dM2"] for e in EXPS]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w/2, dM1, w, color=M1_COL, edgecolor="white", label="♀–M1 (chosen)")
    b2 = ax.bar(x + w/2, dM2, w, color=M2_COL, edgecolor="white", label="♀–M2")
    barlabel(ax, b1, fmt="{:.1f}", fontsize=10, fontweight="bold", color=NAVY)
    barlabel(ax, b2, fmt="{:.0f}", fontsize=10, color="#555")
    ax.axhline(12, ls="--", lw=1.2, color=RED, alpha=0.8)
    ax.text(2.45, 14, "contact ≤12 mm", color=RED, fontsize=8.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(EXPS)
    ax.set_ylabel("female–male distance at spawn (mm)")
    ax.set_title("She spawns in contact with the preferred male\n(and far from the other) — N = 3")
    ax.legend(loc="upper center", fontsize=9)
    despine(ax)
    fig.tight_layout()
    save(fig, "s4_spawn_n3.png")


def fig_summary_table():
    cols = ["", "K_1_1", "K_1_2", "K_1_3"]
    rows = [
        ["Female high-conf (track)", "99.8%", "99.0%", "99.8%"],
        ["Association index (M1)", "0.80", "0.61", "0.71"],
        ["Courtship  M1 vs M2 (s)", "545 / 127", "271 / 162", "614 / 242"],
        ["Sharpening 30→8 mm", "4.1→10.8×", "1.6→1.9×", "2.4→3.7×"],
        ["Courtship : aggression", "22×", "24×", "19×"],
        ["Spawn — d(♀,M1) mm", "5.4", "7.4", "4.9"],
        ["BORIS chosen / other (s)", "118 / 10", "104 / 42", "99 / 7"],
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.axis("off")
    nrow, ncol = len(rows) + 1, len(cols)
    cw = [0.34, 0.22, 0.22, 0.22]
    x0 = np.concatenate([[0], np.cumsum(cw)])
    rh = 1.0 / nrow
    # header
    for j, c in enumerate(cols):
        ax.add_patch(Rectangle((x0[j], 1 - rh), cw[j], rh, fc=NAVY, ec="white"))
        ax.text(x0[j] + cw[j]/2, 1 - rh/2, c, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
    for i, row in enumerate(rows):
        y = 1 - (i + 2) * rh
        for j, val in enumerate(row):
            fc = LIGHT if i % 2 == 0 else "white"
            ax.add_patch(Rectangle((x0[j], y), cw[j], rh, fc=fc, ec="#DDDDDD"))
            ax.text(x0[j] + (0.02 if j == 0 else cw[j]/2), y + rh/2, val,
                    ha="left" if j == 0 else "center", va="center",
                    fontsize=10, fontweight="bold" if j == 0 else "normal",
                    color=NAVY if j == 0 else "#222")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Female mate choice is consistent across all three trials", fontsize=12.5, pad=10)
    fig.tight_layout()
    save(fig, "s4_summary_table.png")


# =============================================================================
#  SECTION 5 — BORIS ethogram cross-validation
# =============================================================================
def fig_behavior_distance():
    beh = ["baseline", "Courtship", "Touch", "Wrap"]
    dist = [40.4, 27.5, 16.8, 8.0]
    cols = ["#9AA7B4", "#7FB2D6", ACCENT, M1_COL]
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    bars = ax.bar(beh, dist, color=cols, edgecolor="white", width=0.62)
    barlabel(ax, bars, fmt="{:.0f} mm", fontsize=11, fontweight="bold", color=NAVY)
    ax.axhline(12, ls="--", lw=1.1, color=RED, alpha=0.7)
    ax.text(3.45, 13.2, "contact ≤12 mm", color=RED, fontsize=8.5, ha="right")
    ax.set_ylabel("female–nearest-male distance (mm)")
    ax.set_title("Geometry corroborates the manual ethogram (K_1_2)\nescalating behaviour → shrinking distance")
    despine(ax)
    fig.tight_layout()
    save(fig, "s5_behavior_distance.png")


def fig_boris_vs_appearance_n3():
    chosen = [DATA[e]["boris_chosen"] for e in EXPS]
    other = [DATA[e]["boris_other"] for e in EXPS]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w/2, chosen, w, color=M1_COL, edgecolor="white",
                label="toward appearance-preferred male")
    b2 = ax.bar(x + w/2, other, w, color=M2_COL, edgecolor="white", label="toward the other male")
    barlabel(ax, b1, fmt="{:.0f}s", fontsize=9.5, fontweight="bold", color=NAVY)
    barlabel(ax, b2, fmt="{:.0f}s", fontsize=9.5, color="#555")
    for i, e in enumerate(EXPS):
        ax.text(i, chosen[i] + 7, f"{DATA[e]['boris_ratio']:.1f}×", ha="center",
                fontsize=10.5, fontweight="bold", color=M1_COL)
    ax.set_xticks(x); ax.set_xticklabels(EXPS)
    ax.set_ylim(0, max(chosen) * 1.25)
    ax.set_ylabel("BORIS male→female courtship time (s)")
    ax.set_title("Independent manual coding courts the SAME male\nthe appearance track calls preferred (N = 3)")
    ax.legend(loc="upper left", fontsize=8.6)
    despine(ax)
    fig.tight_layout()
    save(fig, "s5_boris_vs_appearance_n3.png")


def fig_wrap_acceptance():
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(["accepted\nwrap", "rejected\nwrap"], [28.0, 3.6],
                  color=[GREEN, GREY], edgecolor="white", width=0.6)
    barlabel(ax, bars, fmt="{:.1f} s", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_ylabel("wrap duration (s)")
    ax.set_title("Wrap acceptance (K_1_2)\nmatches the manual write-up (~27 s vs ~3 s)")
    ax.text(0.5, -0.16, "accepted ≥10 s (n=2, median 28.0 s)   vs   rejected (n=1, 3.6 s)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6, style="italic", color=GREY)
    despine(ax)
    fig.tight_layout()
    save(fig, "s5_wrap_acceptance.png")


def fig_behavior_inventory():
    df = pd.read_csv(BORIS / "behavior_counts.csv")
    df = df[df.total_s > 0.05].sort_values("total_s", ascending=True)
    cat_col = {"Courtship": ACCENT, "Mating": M1_COL, "Competitive": RED, "Other": GREY}
    cols = [cat_col.get(c, GREY) for c in df.category]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(df.behavior, df.total_s, color=cols, edgecolor="white")
    for b, n in zip(bars, df.n):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2, f"{b.get_width():.0f}s · n={n}",
                va="center", fontsize=8.5, color="#333")
    ax.set_xlabel("total time coded (s)")
    ax.set_title("BORIS behaviour inventory (K_1_2)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_col.values()]
    ax.legend(handles, cat_col.keys(), loc="lower right", fontsize=8.5, frameon=False)
    ax.set_xlim(0, df.total_s.max() * 1.25)
    despine(ax)
    fig.tight_layout()
    save(fig, "s5_behavior_inventory.png")


# =============================================================================
def main():
    print("Schematics…")
    fig_question(); fig_challenge(); fig_setup()
    print("Calibration…")
    fig_calibration_summary()
    print("ML — method + reversal + separability…")
    fig_dual_line(); fig_reversal_auc(); fig_separability_n3(); fig_confidence_ladder()

    print("Reading K_1_2 final track (for composition + trajectory)…")
    tk = load_track("K_1_2", ["id_final", "id_source", "id_conf"])
    src = tk.id_source.value_counts().to_dict()
    conf = tk.id_conf.value_counts()
    total = len(tk)
    per_fish = (tk.groupby("id_final").id_conf.apply(lambda s: (s == "high").mean() * 100))
    comp = dict(
        hi_pct=100 * conf.get("high", 0) / total,
        interp_pct=100 * conf.get("interp", 0) / total,
        amb_pct=100 * conf.get("ambiguous", 0) / total,
        F_hi=per_fish.get("F", float("nan")),
        M1_hi=per_fish.get("M1", float("nan")),
        M2_hi=per_fish.get("M2", float("nan")),
    )
    fig_track_quality(comp)
    fig_id_composition(src)
    fig_trajectory_topdown()
    fig_interfish_distance()

    print("Biology — mate choice per trial…")
    for e in EXPS:
        fig_matechoice_master(e)
    print("Biology — N=3 summaries…")
    fig_assoc_index_n3(); fig_sharpening_n3(); fig_courtship_M1M2_n3()
    fig_courtship_vs_aggression_n3(); fig_temporal_stability_n3(); fig_spawn_n3()
    fig_summary_table()

    print("BORIS validation…")
    fig_behavior_distance(); fig_boris_vs_appearance_n3()
    fig_wrap_acceptance(); fig_behavior_inventory()

    print("\nComputed K_1_2 composition (sanity):")
    print(f"  high {comp['hi_pct']:.2f}% · interp {comp['interp_pct']:.2f}% · amb {comp['amb_pct']:.2f}%")
    print(f"  per-fish high: F {comp['F_hi']:.2f} · M1 {comp['M1_hi']:.2f} · M2 {comp['M2_hi']:.2f}")
    print(f"\nAll figures written to {FIG}")


if __name__ == "__main__":
    main()
