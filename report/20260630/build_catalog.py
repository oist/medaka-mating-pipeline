"""Assemble the poster deliverables for report/20260630.

1. Copies reused REAL pipeline figures into report/20260630/reused/.
2. Builds  medaka_poster_figures_20260630.pptx  — a figure CATALOG: one slide per
   candidate figure (large) with its title, caption and on-disk source, so you can
   browse and pick. Section dividers group Background / Methods / ML / Biology / BORIS.
3. Builds  draft_poster_A0_landscape_20260630.pptx  — one recommended A0 arrangement.

Run AFTER build_figures.py:
    .venv\\Scripts\\python.exe report\\20260630\\build_catalog.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
REUSED = HERE / "reused"
REUSED.mkdir(parents=True, exist_ok=True)
REPO = Path(r"C:\Users\machi\Documents\GitHub\medaka-mating-pipeline")
P22 = REPO / "report" / "20260622"
P25 = REPO / "report" / "20260625"
P29 = REPO / "report" / "20260629"
LINEB = REPO / "runs" / "dual_line" / "lineB_fulltimeline"
EXT = REPO / "runs" / "dual_line" / "extend_k11_k13"

# ---------------------------------------------------------------- palette
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
GREEN = RGBColor(0x1E, 0x8E, 0x3E)
AMBER = RGBColor(0x8A, 0x55, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)

SECTION_COLOR = {
    "1 · Background & question": NAVY,
    "2 · Pipeline & methods": ACCENT,
    "3 · Machine-learning tracking": GREEN,
    "4 · Biology — mate choice": RGBColor(0x6A, 0x3D, 0x9A),
    "5 · Ethogram cross-validation": AMBER,
}

# ---------------------------------------------------------------- reused-figure sources
REUSE_SRC = {
    "pipeline_3d_id_tracking.png": P29 / "pipeline_3d_id_tracking.png",
    "3d_multiview_composite_still.png": P25 / "images" / "3d_multiview_composite_still.png",
    "calibration_error_3d.png": P22 / "plots" / "calibration_error_3d.png",
    "calibration_reprojection_2d.png": P22 / "plots" / "calibration_reprojection_2d.png",
    "calibration_volume_error.png": P22 / "plots" / "calibration_volume_error.png",
    "detectron2_vs_sleap_top.png": P22 / "plots" / "detectron2_vs_sleap_top.png",
    "detectron2_vs_sleap_side1.png": P22 / "plots" / "detectron2_vs_sleap_side1.png",
    "sleap_round4_vs_round5_eval.png": P22 / "plots" / "sleap_round4_vs_round5_eval.png",
    "segid_per_view_pca.png": P22 / "plots" / "segid_per_view_pca.png",
    "female_prob_by_tracklet.png": P22 / "plots" / "female_prob_by_tracklet.png",
    "segid_fish_crops_top.png": P22 / "plots" / "segid_fish_crops_top.png",
    "tracklet_coverage_stifd.png": P22 / "plots" / "tracklet_coverage_stifd.png",
    "gt_montage.png": P25 / "plots" / "gt_montage.png",
    "ethogram_distance_timeline.png": P22 / "behavior_comparison" / "ethogram_distance_timeline.png",
    "behavior_distance_boxplot.png": P22 / "behavior_comparison" / "behavior_distance_boxplot.png",
    "spawn_closeup.png": P22 / "behavior_comparison" / "spawn_closeup.png",
    "mate_choice_K_1_1.png": EXT / "K_1_1" / "analysis" / "mate_choice.png",
    "mate_choice_K_1_2.png": LINEB / "analysis" / "mate_choice.png",
    "mate_choice_K_1_3.png": EXT / "K_1_3" / "analysis" / "mate_choice.png",
}


def copy_reused():
    for name, src in REUSE_SRC.items():
        dst = REUSED / name
        if dst.exists():
            continue
        if not src.exists():
            print(f"  [MISS] {name} <- {src}")
            continue
        shutil.copy2(src, dst)
        print(f"  [copy] {name}")


# ---------------------------------------------------------------- figure registry
# (num, where, filename, section, title, caption)
G, R = "figures", "reused"
SECT = ["1 · Background & question", "2 · Pipeline & methods",
        "3 · Machine-learning tracking", "4 · Biology — mate choice",
        "5 · Ethogram cross-validation"]

FIGURES = [
    ("1.1", G, "s1_question.png", SECT[0], "The biological question",
     "1 female + 2 visually identical males — which does she choose, and is it courtship or aggression that wins?"),
    ("1.2", G, "s1_challenge.png", SECT[0], "Why it is hard",
     "Manual scoring (BORIS) cannot tell the two males apart → only a dyad. The pipeline recovers all 3 IDs in 3D."),
    ("1.3", G, "s1_setup.png", SECT[0], "Experimental setup",
     "3 calibrated cameras (1 top + 2 side), 20 cm tank, 166 fps, ~35 min/trial, N=3 (K_1_1/K_1_2/K_1_3)."),
    ("3.0", R, "3d_multiview_composite_still.png", SECT[1], "Multi-view + 3D (context still)",
     "The three synchronized views and the reconstructed 3D scene used throughout the pipeline."),
    ("2.1", R, "pipeline_3d_id_tracking.png", SECT[1], "Pipeline overview (lead methods figure)",
     "6 stages: capture → SLEAP pose → Detectron2 masks → appearance ID → 3D reconstruction + ID-link → stable ID through crossings."),
    ("2.2a", G, "s2_calibration_summary.png", SECT[1], "Calibration accuracy (summary)",
     "Learned 6D→3D model: 0.90 mm mean / 1.00 RMSE / 2.15 max over 228 points ≈ 3–7 % of a fish body."),
    ("2.2b", R, "calibration_error_3d.png", SECT[1], "Calibration — 3D error",
     "Per-point 3D reconstruction error of the production calibration model."),
    ("2.2c", R, "calibration_reprojection_2d.png", SECT[1], "Calibration — 2D reprojection",
     "Reprojection residuals per camera."),
    ("2.2d", R, "calibration_volume_error.png", SECT[1], "Calibration — volume error",
     "Spatial distribution of error across the tank volume."),
    ("2.3a", R, "detectron2_vs_sleap_top.png", SECT[1], "Pose + segmentation (top)",
     "SLEAP 2-node skeleton (mouth→tail) + Detectron2 per-instance masks, top view."),
    ("2.3b", R, "detectron2_vs_sleap_side1.png", SECT[1], "Pose + segmentation (side)",
     "Same, side view — masks gate detections and feed appearance crops."),
    ("2.4", R, "sleap_round4_vs_round5_eval.png", SECT[1], "SLEAP retraining",
     "Round-5 model in production (centroid mOKS ≈ 0.99)."),
    ("3.1", G, "s3_dual_line.png", SECT[2], "The dual-line method",
     "Line A geometry ∥ Line B appearance, fused by scene then interpolated. Each carries the frames the other fails on."),
    ("3.2", G, "s3_reversal_auc.png", SECT[2], "The reversal (ML headline)",
     "'Males not separable' was label noise. Swap-correction: male AUC 0.994, female 1.00; position baseline 0.48."),
    ("3.3", G, "s3_separability_n3.png", SECT[2], "Identity ≫ position (N=3)",
     "Corrected per-experiment classifiers: appearance balanced-acc 0.86/0.96 vs position-only 0.26/0.44."),
    ("3.4", G, "s3_track_quality.png", SECT[2], "Track quality (K_1_2)",
     "348,600 frames, exactly 1 ♀+1 M1+1 M2 each; 98.6 % high-conf / 1.3 % interp / 0.06 % ambiguous."),
    ("3.5", G, "s3_confidence_ladder.png", SECT[2], "Confidence ladder",
     "Male core recovery: side 78.5 % → +top fusion 88.2 % → +interpolation 98.4 % high-confidence."),
    ("3.6", G, "s3_id_composition.png", SECT[2], "How each ID was decided",
     "90.9 % direct high-conf; the rest bridged / edge / motion-resolved / flagged ambiguous (0.06 %)."),
    ("3.7", G, "s3_trajectory_topdown.png", SECT[2], "3D trajectories (spawn approach)",
     "Top-down zoom: IDs held through close contact as the female converges on M1 at the spawn."),
    ("3.8", G, "s3_interfish_distance.png", SECT[2], "Inter-fish distance (full trial)",
     "How often the three fish sit in the hard contact regime the tracker must survive."),
    ("3.9", R, "gt_montage.png", SECT[2], "Female validation — gravid belly",
     "The called female is the deeper-bodied/gravid fish in every row AND across her geometry-slot swaps → biology, not a slot."),
    ("3.10a", R, "segid_per_view_pca.png", SECT[2], "Appearance embedding (PCA)",
     "Per-view PCA of appearance embeddings — the three fish separate."),
    ("3.10b", R, "female_prob_by_tracklet.png", SECT[2], "Female probability by tracklet",
     "Per-tracklet female-probability is clean and confident."),
    ("3.10c", R, "segid_fish_crops_top.png", SECT[2], "Segmentation crops",
     "Example per-fish mask crops feeding the appearance classifier."),
    ("3.10d", R, "tracklet_coverage_stifd.png", SECT[2], "Tracklet coverage",
     "Coverage / fragmentation of the top-view tracklets."),
    ("4.1a", G, "s4_matechoice_K_1_1.png", SECT[3], "Mate choice — K_1_1",
     "F–male distance · cumulative courtship time · preference index. Assoc 0.80; spawns with M1."),
    ("4.1b", G, "s4_matechoice_K_1_2.png", SECT[3], "Mate choice — K_1_2 (lead biology)",
     "Assoc 0.61; courtship 271 vs 162 s; spawns with M1 at 7.4 mm."),
    ("4.1c", G, "s4_matechoice_K_1_3.png", SECT[3], "Mate choice — K_1_3",
     "Assoc 0.71; courtship 614 vs 242 s; spawns with M1 at 4.9 mm."),
    ("4.1d1", R, "mate_choice_K_1_1.png", SECT[3], "Mate choice — K_1_1 (original)",
     "Original analysis figure (alt styling)."),
    ("4.1d2", R, "mate_choice_K_1_2.png", SECT[3], "Mate choice — K_1_2 (original)",
     "Original analysis figure (alt styling)."),
    ("4.1d3", R, "mate_choice_K_1_3.png", SECT[3], "Mate choice — K_1_3 (original)",
     "Original analysis figure (alt styling)."),
    ("4.2", G, "s4_assoc_index_n3.png", SECT[3], "Association index (N=3)",
     "0.61 / 0.71 / 0.80 — preference above chance (0.5) in every trial."),
    ("4.3", G, "s4_sharpening_n3.png", SECT[3], "Preference sharpens at contact",
     "Preference ratio rises as the threshold tightens 30→8 mm (up to 10.8×) — strongest where mating happens."),
    ("4.4", G, "s4_courtship_M1M2_n3.png", SECT[3], "Courtship M1 vs M2 (N=3)",
     "The preferred male courts 1.7–4.3× more in every trial."),
    ("4.5", G, "s4_courtship_vs_aggression_n3.png", SECT[3], "Courtship ≫ aggression",
     "Courtship : aggression time = 19–25× → courtship, not male–male aggression, tracks the choice."),
    ("4.6", G, "s4_temporal_stability_n3.png", SECT[3], "Preference is stable in time",
     "The female favours the eventual mate from the start — not a late switch."),
    ("4.7", G, "s4_spawn_n3.png", SECT[3], "Spawn outcome (N=3)",
     "At the spawn she is 4.9–7.4 mm from the preferred male and >100 mm from the other."),
    ("4.8", G, "s4_summary_table.png", SECT[3], "N=3 summary table",
     "All three trials agree: associate, courted, spawn — with the same male."),
    ("5.1a", G, "s5_behavior_distance.png", SECT[4], "Behaviour ↔ distance (clean)",
     "BORIS coding lines up with geometry: baseline 40 → courtship 28 → touch 17 → wrap 8 mm."),
    ("5.1b", R, "behavior_distance_boxplot.png", SECT[4], "Behaviour ↔ distance (boxplot)",
     "Distribution of female–male distance per coded behaviour."),
    ("5.2", R, "ethogram_distance_timeline.png", SECT[4], "Ethogram + distance timeline",
     "BORIS events overlaid on the female–male distance trace (same 166 fps clock)."),
    ("5.3", G, "s5_boris_vs_appearance_n3.png", SECT[4], "BORIS agrees on the chosen male (N=3)",
     "Hand-coded courtship is 2.5–14.7× asymmetric toward the SAME male the appearance track calls preferred."),
    ("5.4", G, "s5_wrap_acceptance.png", SECT[4], "Wrap acceptance",
     "Accepted wraps 28 s vs rejected 3.6 s — matches the manual write-up."),
    ("5.5", G, "s5_behavior_inventory.png", SECT[4], "Behaviour inventory",
     "BORIS counts/time by category (courtship / competitive / mating)."),
    ("5.6", R, "spawn_closeup.png", SECT[4], "Spawn close-up",
     "Trajectory zoom at the spawn frame."),
]


def path_of(where, fname):
    return (FIG if where == G else REUSED) / fname


# ---------------------------------------------------------------- pptx helpers
def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_image_fit(slide, path, left, top, max_w, max_h, frame=True):
    if not Path(path).exists():
        tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(max_w), Inches(0.5))
        tb.text_frame.text = f"[missing {Path(path).name}]"
        return None
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    if ar > max_w / max_h:
        w = max_w; h = max_w / ar
    else:
        h = max_h; w = max_h * ar
    l = left + (max_w - w) / 2
    t = top + (max_h - h) / 2
    pic = slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))
    if frame:
        pic.line.color.rgb = RGBColor(0xC4, 0xCC, 0xD6); pic.line.width = Pt(0.75)
    return pic


def textbox(slide, text, left, top, width, height, size=12, color=DARK, bold=False,
            align=PP_ALIGN.LEFT, italic=False, anchor=None):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    if anchor:
        tf.vertical_anchor = anchor
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.italic = italic
    return tb


def fill_rect(slide, left, top, width, height, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
                                Inches(width), Inches(height))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


# ---------------------------------------------------------------- catalog deck (16:9)
def build_catalog():
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)

    # title
    s = blank(prs)
    fill_rect(s, 0, 0, 13.333, 7.5, NAVY)
    fill_rect(s, 0, 4.55, 13.333, 0.07, ACCENT)
    textbox(s, "Medaka female mate choice — poster figure catalogue", 0.9, 1.7, 11.6, 1.6,
            size=34, color=WHITE, bold=True)
    textbox(s, "One figure per slide · pick the ones you want · generated + real-pipeline output",
            0.9, 3.2, 11.6, 0.7, size=17, color=RGBColor(0x9F, 0xC5, 0xE8))
    textbox(s, "Storyboard: STORYBOARD.md   ·   regenerate: build_figures.py + build_catalog.py   ·   2026-06-30",
            0.9, 5.0, 11.6, 0.6, size=13, color=RGBColor(0x86, 0xA8, 0xC8))

    cur_sect = None
    for (num, where, fname, section, title, caption) in FIGURES:
        if section != cur_sect:
            cur_sect = section
            d = blank(prs)
            col = SECTION_COLOR[section]
            fill_rect(d, 0, 0, 13.333, 7.5, col)
            textbox(d, section, 0.9, 3.0, 11.6, 1.4, size=40, color=WHITE, bold=True,
                    align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        s = blank(prs)
        col = SECTION_COLOR[section]
        fill_rect(s, 0, 0, 13.333, 1.0, col)
        textbox(s, f"{num}  {title}", 0.35, 0.12, 10.8, 0.8, size=22, color=WHITE, bold=True,
                anchor=MSO_ANCHOR.MIDDLE)
        textbox(s, section.split(" · ")[0], 11.6, 0.12, 1.5, 0.8, size=14, color=WHITE,
                align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        p = path_of(where, fname)
        add_image_fit(s, p, 0.5, 1.2, 12.33, 5.1)
        tag = "generated" if where == G else "real pipeline output"
        textbox(s, caption, 0.6, 6.45, 12.1, 0.55, size=13, color=DARK, italic=True,
                align=PP_ALIGN.CENTER)
        rel = (("figures/" if where == G else "reused/") + fname)
        textbox(s, f"{rel}   ·   [{tag}]", 0.6, 6.98, 12.1, 0.35, size=9.5, color=GREY,
                align=PP_ALIGN.CENTER)

    out = HERE / "medaka_poster_figures_20260630.pptx"
    prs.save(str(out))
    print(f"  catalogue: {out.name}  ({len(prs.slides._sldIdLst)} slides)")
    return out


# ---------------------------------------------------------------- draft poster (A0 landscape)
def build_draft_poster():
    prs = Presentation()
    prs.slide_width = Inches(46.8); prs.slide_height = Inches(33.1)  # A0 landscape
    s = blank(prs)

    # title band
    fill_rect(s, 0, 0, 46.8, 4.3, NAVY)
    fill_rect(s, 0, 4.3, 46.8, 0.12, ACCENT)
    textbox(s, "Which male does she choose?", 1.2, 0.45, 44, 1.9, size=66, color=WHITE, bold=True)
    textbox(s, "Resolving three identical fish in 3D to quantify female mate choice in medaka (Oryzias latipes)",
            1.2, 2.55, 40, 1.0, size=28, color=RGBColor(0x9F, 0xC5, 0xE8))
    textbox(s, "Multi-camera tracking pipeline · dual-line geometry ∥ appearance · N = 3 trials · BORIS-validated",
            1.2, 3.45, 44, 0.7, size=20, color=RGBColor(0x86, 0xA8, 0xC8))

    col_w = 11.0
    gap = 0.55
    x0 = 0.7
    cols_x = [x0 + i * (col_w + gap) for i in range(4)]
    top0 = 5.0
    bottom_band = 30.2

    headers = ["BACKGROUND & SETUP", "PIPELINE & METHOD", "TRACKING RESULTS", "MATE CHOICE (N=3)"]
    hcols = [NAVY, ACCENT, GREEN, RGBColor(0x6A, 0x3D, 0x9A)]
    layout = [
        # col 0
        [("figures/s1_question.png", 0.0), ("figures/s1_setup.png", 0.0)],
        # col 1
        [("reused/pipeline_3d_id_tracking.png", 0.0), ("figures/s3_dual_line.png", 0.0),
         ("figures/s3_reversal_auc.png", 0.0)],
        # col 2
        [("figures/s3_track_quality.png", 0.0), ("figures/s3_trajectory_topdown.png", 0.0),
         ("reused/gt_montage.png", 0.0)],
        # col 3
        [("figures/s4_matechoice_K_1_2.png", 0.0), ("figures/s4_assoc_index_n3.png", 0.0),
         ("figures/s4_courtship_vs_aggression_n3.png", 0.0)],
    ]
    for ci, items in enumerate(layout):
        x = cols_x[ci]
        # column header
        fill_rect(s, x, top0, col_w, 0.95, hcols[ci])
        textbox(s, headers[ci], x + 0.2, top0 + 0.08, col_w - 0.3, 0.8, size=22, color=WHITE,
                bold=True, anchor=MSO_ANCHOR.MIDDLE)
        y = top0 + 1.2
        for (rel, _) in items:
            p = HERE / rel
            if not p.exists():
                continue
            with Image.open(p) as im:
                iw, ih = im.size
            ar = iw / ih
            w = col_w
            h = w / ar
            avail = bottom_band - y - 0.3
            if h > avail:               # cap height, keep aspect
                h = avail; w = h * ar
            pic = s.shapes.add_picture(str(p), Inches(x + (col_w - w) / 2), Inches(y),
                                       Inches(w), Inches(h))
            pic.line.color.rgb = RGBColor(0xC4, 0xCC, 0xD6); pic.line.width = Pt(1.0)
            y += h + 0.55

    # conclusions band
    fill_rect(s, 0.7, bottom_band, 45.4, 2.5, LIGHT)
    fill_rect(s, 0.7, bottom_band, 0.18, 2.5, GREEN)
    concl = ("CONCLUSIONS   A dual-line tracker resolves three visually identical medaka into continuous "
             "individual-ID 3D tracks (98.6 % high-confidence, gap-free). In all three trials the female "
             "associates with, is courted by, and spawns with one specific male; the preference sharpens "
             "toward contact and is stable from the start. Courtship — not male–male aggression — tracks "
             "the choice (≈20–25×). Validated by gravid-belly morphology, an independent embedder, and manual BORIS ethograms.")
    textbox(s, concl, 1.1, bottom_band + 0.25, 44.6, 2.1, size=19, color=DARK)

    out = HERE / "draft_poster_A0_landscape_20260630.pptx"
    prs.save(str(out))
    print(f"  draft poster: {out.name}")
    return out


def main():
    print("Copying reused real-pipeline figures…")
    copy_reused()
    print("Building figure catalogue…")
    build_catalog()
    print("Building draft A0 poster…")
    build_draft_poster()
    print("\nDone.")


if __name__ == "__main__":
    main()
