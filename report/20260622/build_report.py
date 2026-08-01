"""Build the 2026-06-22 medaka-tracking progress report.

Curates plots/images/videos into report/20260622/{plots,images,videos,posters}
and generates medaka_progress_report_20260622.pptx.

Run with the project venv:
    .venv\\Scripts\\python.exe report\\20260622\\build_report.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
IMAGES = HERE / "images"
VIDEOS = HERE / "videos"
POSTERS = HERE / "posters"
for d in (PLOTS, IMAGES, VIDEOS, POSTERS):
    d.mkdir(parents=True, exist_ok=True)

REPO = Path(r"c:\Users\machi\Documents\GitHub\medaka-mating-pipeline")
CALQC = REPO / "calibration_results" / "cali_20251217_manual" / "merged_rule_reassigned_beads_only"

# ---------------------------------------------------------------- media sources
# (source, dest) — copied as report deliverables.
PLOT_SRC = {
    "calibration_error_3d.png": CALQC / "beads_only_qc" / "cali_20251217_merged_error_3d.png",
    "calibration_reprojection_2d.png": CALQC / "beads_only_qc" / "cali_20251217_merged_point_mapping_reprojection_2d_views.png",
    "calibration_volume_error.png": CALQC / "qc" / "cali_20251217_merged_point_mapping_volume_error.png",
    "sleap_round4_vs_round5_eval.png": Path(r"Z:\StephensU\medaka\sleap_models\eval_round4_vs_round5\round4_vs_round5_eval.png"),
    "detectron2_vs_sleap_top.png": Path(r"Z:\StephensU\medaka\runs\udmt_validation\seg_qc\K_1_2\top_cam_detectron2_vs_sleap.png"),
    "detectron2_vs_sleap_side1.png": Path(r"Z:\StephensU\medaka\runs\udmt_validation\seg_qc\K_1_2\side1_cam_detectron2_vs_sleap.png"),
    "segid_per_view_pca.png": Path(r"Z:\StephensU\medaka\runs\udmt_validation\segid\K_1_2\eval_all\per_view_pca.png"),
    "segid_fish_crops_top.png": Path(r"Z:\StephensU\medaka\runs\udmt_validation\segid\K_1_2\eval_all\fish_identity_crops_top.png"),
    "female_prob_by_tracklet.png": Path(r"Z:\StephensU\medaka\runs\udmt_validation\segid\K_1_2\female_anchor\female_prob_by_tracklet.png"),
    "tracklet_coverage_stifd.png": Path(r"Z:\StephensU\medaka\runs\udmt_validation\tracklet_bind_stifd\coverage_stifd_vs_const.png"),
}
IMAGE_SRC = {
    "3d_multiview_composite_still.png": Path(r"Z:\StephensU\medaka\runs\udmt_validation\full_cosmetic_still.png"),
    "calibration_cube_top.png": REPO / "calibration_results" / "cali_20251217_manual" / "way1" / "manual_marking" / "cali_251217_way1_cam0_top_frame100_manual_marking.png",
}
VIDEO_SRC = {
    "3d_multiview_composite_clip.mp4": Path(r"Z:\StephensU\medaka\runs\udmt_validation\3d_and_views_133798.mp4"),
    "K_1_1_crossing_montage.mp4": Path(r"Z:\StephensU\medaka\reconstruction_3d\round5_260611\20251225_K_1_1_FCMC\round5_260611_crossing_montage.mp4"),
    "K_1_2_crossing_montage.mp4": Path(r"Z:\StephensU\medaka\reconstruction_3d\round5_260611\20251225_K_1_2_FCMC\round5_260611_crossing_montage.mp4"),
    "K_1_3_crossing_montage.mp4": Path(r"Z:\StephensU\medaka\reconstruction_3d\round5_260611\20251225_K_1_3_FCMC\round5_260611_crossing_montage.mp4"),
    "tracklet_suspect_swaps18.mp4": Path(r"Z:\StephensU\medaka\runs\udmt_validation\tracklet_bind\suspect_swaps18.mp4"),
}


def copy_media() -> None:
    for tbl, dest_dir in ((PLOT_SRC, PLOTS), (IMAGE_SRC, IMAGES), (VIDEO_SRC, VIDEOS)):
        for name, src in tbl.items():
            dest = dest_dir / name
            if dest.exists():
                print(f"  [skip] {name} (exists)")
                continue
            if not src.exists():
                print(f"  [MISS] {name} <- {src}")
                continue
            shutil.copy2(src, dest)
            mb = dest.stat().st_size / 1e6
            print(f"  [ok ] {name} ({mb:.1f} MB)")


def grab_poster(video: Path, out: Path, frac: float = 0.35) -> bool:
    if out.exists():
        return True
    if not video.exists():
        return False
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, int(frac * n)) if n > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    cv2.imwrite(str(out), frame)
    return True


def make_posters() -> None:
    for name in VIDEO_SRC:
        src = VIDEOS / name
        out = POSTERS / (Path(name).stem + ".png")
        ok = grab_poster(src, out)
        print(f"  poster {out.name}: {'ok' if ok else 'FAIL'}")


# ----------------------------------------------------------------- deck styling
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
SUCCESS = RGBColor(0x1E, 0x8E, 0x3E)
WARN = RGBColor(0xB7, 0x6E, 0x00)
DARK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

EMU_IN = 914400
SW, SH = 13.333, 7.5


def _frame(shape, color=None):
    if color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = color


def title_bar(slide, prs, text, sub=None, color=NAVY):
    from pptx.enum.shapes import MSO_SHAPE
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.margin_left = Inches(0.35)
    tf.margin_top = Inches(0.08)
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = WHITE
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run(); r2.text = sub
        r2.font.size = Pt(13); r2.font.color.rgb = RGBColor(0xCF, 0xDF, 0xEF)
    # accent underline
    acc = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.05), prs.slide_width, Inches(0.06))
    acc.fill.solid(); acc.fill.fore_color.rgb = ACCENT; acc.line.fill.background()
    acc.shadow.inherit = False


def add_bullets(slide, items, left, top, width, height, size=15, gap=4):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        if isinstance(it, tuple):
            text, lvl, color, bold = (list(it) + [0, DARK, False])[:4]
        else:
            text, lvl, color, bold = it, 0, DARK, False
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(gap)
        bullet = ("• " if lvl == 0 else "– ") if not text.startswith(("✓", "✗", "▶", "→")) else ""
        r = p.add_run(); r.text = bullet + text
        r.font.size = Pt(size - 1 if lvl else size)
        r.font.color.rgb = color
        r.font.bold = bold
    return box


def add_image_fit(slide, path: Path, left, top, max_w, max_h, caption=None, frame=True):
    if not Path(path).exists():
        tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(max_w), Inches(0.5))
        tb.text_frame.text = f"[missing image: {Path(path).name}]"
        tb.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        tb.text_frame.paragraphs[0].runs[0].font.color.rgb = WARN
        return
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    box_ar = max_w / max_h
    if ar > box_ar:
        w = max_w; h = max_w / ar
    else:
        h = max_h; w = max_h * ar
    l = left + (max_w - w) / 2
    t = top + (max_h - h) / 2
    pic = slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))
    if frame:
        pic.line.color.rgb = RGBColor(0xBB, 0xC4, 0xCE); pic.line.width = Pt(0.75)
    if caption:
        cap = slide.shapes.add_textbox(Inches(left), Inches(top + max_h + 0.02), Inches(max_w), Inches(0.4))
        cp = cap.text_frame.paragraphs[0]; cp.alignment = PP_ALIGN.CENTER
        r = cp.add_run(); r.text = caption; r.font.size = Pt(10.5); r.font.italic = True; r.font.color.rgb = GREY
    return pic


def add_video(slide, video: Path, poster: Path, left, top, w, h, caption=None):
    if video.exists() and poster.exists():
        slide.shapes.add_movie(str(video), Inches(left), Inches(top), Inches(w), Inches(h),
                               poster_frame_image=str(poster), mime_type="video/mp4")
    elif poster.exists():
        add_image_fit(slide, poster, left, top, w, h)
    if caption:
        cap = slide.shapes.add_textbox(Inches(left), Inches(top + h + 0.02), Inches(w), Inches(0.4))
        cp = cap.text_frame.paragraphs[0]; cp.alignment = PP_ALIGN.CENTER
        r = cp.add_run(); r.text = caption; r.font.size = Pt(10.5); r.font.italic = True; r.font.color.rgb = GREY


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_table(slide, rows, left, top, width, col_w=None, header=True, fs=11):
    nrow = len(rows); ncol = len(rows[0])
    gtbl = slide.shapes.add_table(nrow, ncol, Inches(left), Inches(top), Inches(width), Inches(0.35 * nrow))
    tbl = gtbl.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(cw)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02); cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            r.font.size = Pt(fs)
            if header and ri == 0:
                r.font.bold = True; r.font.color.rgb = WHITE
                cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
            else:
                r.font.color.rgb = DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
            if ci == 0 and not (header and ri == 0):
                r.font.bold = True
    return tbl


# --------------------------------------------------------------------- build it
def build_deck() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    from pptx.enum.shapes import MSO_SHAPE
    BC = HERE / "behavior_comparison"

    # ---- 1. Title
    s = blank(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background(); band.shadow.inherit = False
    strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(4.7), prs.slide_width, Inches(0.08))
    strip.fill.solid(); strip.fill.fore_color.rgb = ACCENT; strip.line.fill.background(); strip.shadow.inherit = False
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.1), Inches(11.5), Inches(2.5))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run()
    r.text = "Medaka 3-Fish Mating-Behaviour Tracking"
    r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); r2 = p2.add_run()
    r2.text = "Multi-camera 3D pipeline — progress report"
    r2.font.size = Pt(22); r2.font.color.rgb = RGBColor(0x9F, 0xC5, 0xE8)
    tb2 = s.shapes.add_textbox(Inches(0.9), Inches(5.0), Inches(11.5), Inches(1.6))
    tf2 = tb2.text_frame; tf2.word_wrap = True
    for txt, sz, col in [
        ("Calibration  ·  SLEAP pose tracking  ·  3D reconstruction  ·  Detectron2 appearance-ID  ·  female anchor", 15, RGBColor(0xD7,0xE3,0xF0)),
        ("Dataset focus: 20251225 K group (K_1_1 / K_1_2 / K_1_3)   —   1 female + 2 males, 166 fps, 20 cm tank", 14, RGBColor(0xB9,0xCd,0xE0)),
        ("2026-06-22", 14, RGBColor(0x86,0xA8,0xC8)),
    ]:
        pp = tf2.add_paragraph() if txt != "Calibration  ·  SLEAP pose tracking  ·  3D reconstruction  ·  Detectron2 appearance-ID  ·  female anchor" else tf2.paragraphs[0]
        rr = pp.add_run(); rr.text = txt; rr.font.size = Pt(sz); rr.font.color.rgb = col

    # ---- 2. Pipeline overview / goal
    s = blank(prs)
    title_bar(s, prs, "Goal & pipeline at a glance")
    add_bullets(s, [
        ("Research question: quantify female mate choice from male aggression vs courtship in a 1F + 2M setup.", 0, DARK, True),
        "Stages: Acquisition → Calibration → SLEAP pose inference (HPC) → 2D/3D tracking → 3D reconstruction → behaviour analysis.",
        "3 synchronised cameras (top XY + two side views), 166 fps, 20 cm tank; per-stage outputs are CSV/JSON on disk (re-runnable).",
        ("Where we are now:", 0, ACCENT, True),
        ("✓ Calibration model validated (0.90 mm mean error).", 1, SUCCESS, False),
        ("✓ SLEAP at round-5; 3D geometry clean (0 speed-spikes over 348,600 frames).", 1, SUCCESS, False),
        ("✓ Female identifiable from appearance — anchor wired into the pipeline.", 1, SUCCESS, False),
        ("→ Residual problem: telling the two near-identical males apart through crossings (a geometry/continuity job).", 1, WARN, False),
    ], 0.5, 1.35, 6.6, 5.6, size=15)
    add_image_fit(s, IMAGES / "3d_multiview_composite_still.png", 7.4, 1.5, 5.5, 4.7,
                  caption="Live view: top + two side cameras (top) and the reconstructed 3D tracks of all 3 fish (bottom).")

    # ---- 3. Calibration
    s = blank(prs)
    title_bar(s, prs, "Calibration — learned 6D→3D mapping (refraction-free)")
    add_bullets(s, [
        ("Method: polynomial regression maps 6D pixel coords (3 cams) → 3D world — not a classical pinhole model.", 0, DARK, True),
        "Implicitly absorbs water/glass refraction → no explicit refraction model needed.",
        "Object: 190 mm cube, 49 colour-coded beads (7×7) + 8 corners, filmed in 4 orientations.",
        ("Production accuracy (228 held-out points):", 0, ACCENT, True),
        ("✓ 0.90 mm mean   ·   1.00 mm RMSE   ·   2.15 mm max", 1, SUCCESS, True),
        "Good for body centroids (25–35 mm fish) & inter-fish distances; not for <2 mm appendages / fine Z.",
        ("Extension built: rigid-object refinement (Kabsch pose + retrain) for a second filming, accepted only if held-out error does not regress.", 0, GREY, False),
    ], 0.5, 1.35, 6.5, 5.6, size=15)
    add_image_fit(s, PLOTS / "calibration_error_3d.png", 7.2, 1.4, 5.8, 3.0,
                  caption="3D error across the cube volume — uniformly sub-2 mm.")
    add_image_fit(s, PLOTS / "calibration_reprojection_2d.png", 7.2, 4.75, 5.8, 2.3,
                  caption="2D reprojection check across the three camera views.")

    # ---- 4. SLEAP tracking + improvement
    s = blank(prs)
    title_bar(s, prs, "SLEAP pose tracking — iterative retraining (round 3→4→5)")
    add_bullets(s, [
        ("2-node skeleton (mouth→tail); centroid + centered-instance models, top & side.", 0, DARK, True),
        "Round-5 (2026-06-11): top n=900 / side n=1800 labelled frames (~35% more than round-4).",
        ("Key fix: centered-instance crop 256→448 px — fish were being clipped.", 0, ACCENT, True),
        ("Round-5 vs round-4 (held-out):", 0, DARK, True),
        ("✓ centered-instance mOKS: top 0.66→0.74, side 0.58→0.69", 1, SUCCESS, False),
        ("✓ side PCK@5 0.78→0.90; side worst-case (p99) 75→60 px", 1, SUCCESS, False),
        ("✓ centroids near-perfect (mOKS 0.99–1.00)", 1, SUCCESS, False),
        ("✗ one regression: top p99 mouth/tail 45→56 px → more top hard-cases next round", 1, WARN, False),
        ("Full-length inference on the OIST saion HPC (A100): 348,600 frames/view × 3 experiments.", 0, GREY, False),
    ], 0.5, 1.35, 6.2, 5.6, size=14)
    add_image_fit(s, PLOTS / "sleap_round4_vs_round5_eval.png", 6.9, 1.4, 6.1, 5.4,
                  caption="Held-out eval: round-5 (blue) improves accuracy across the board; bottom-right p99 = mouth/tail-swap signal.")

    # ---- 5. SLEAP close-contact failure mode
    s = blank(prs)
    title_bar(s, prs, "SLEAP — the close-contact failure mode & fixes in flight")
    add_bullets(s, [
        ("The remaining errors are concentrated at crossings, not in open water.", 0, DARK, True),
        ("Mechanism (identified by review): at close contact two instances keep distinct mouths but share ONE tail —", 0, DARK, False),
        ("the loser's centroid is pulled ~half a body toward the neighbour → the two 3D positions converge → an ID swap.", 1, WARN, False),
        ("Shared-tail dedup (built + tested): null the shared tail, keep the mouth; ~6–7% of frames per view —", 0, DARK, False),
        ("concentrated at crossings (40/40 of the tightest crossings have a shared-tail frame in-window).", 1, DARK, False),
        ("Round-6 in progress: 420-frame close-contact annotation seed built; awaiting labels → warm-start train (the real fix).", 0, ACCENT, True),
        ("These swaps are SMOOTH — invisible to residual / speed-spike metrics → must be verified by eye at crossings.", 0, GREY, False),
    ], 0.5, 1.35, 7.4, 5.0, size=14.5)
    add_image_fit(s, PLOTS / "detectron2_vs_sleap_top.png", 8.1, 1.5, 4.9, 2.5,
                  caption="Top view: Detectron2 masks (filled) vs SLEAP skeletons (yellow).")
    add_image_fit(s, POSTERS / "K_1_2_crossing_montage.png", 8.1, 4.35, 4.9, 2.4,
                  caption="▶ videos/K_1_2_crossing_montage.mp4 — review at crossings.")

    # ---- 6. 3D reconstruction quality (K group)
    s = blank(prs)
    title_bar(s, prs, "Current 3D tracking quality — 1225 K group (full length, round-5)")
    rows = [
        ["Experiment", "Coverage", "Resid med/p90 (px²)", "frac>5000", "speed-spike", "ID-bind"],
        ["K_1_1", "87.4 %", "137 / 432", "0.00 %", "0.000", "96.4 %"],
        ["K_1_2", "87.2 %", "132 / 457", "0.00 %", "0.000", "96.1 %"],
        ["K_1_3", "85.1 %", "107 / 437", "0.00 %", "0.000", "95.0 %"],
    ]
    add_table(s, rows, 0.5, 1.4, 7.2, col_w=[1.4, 1.1, 1.7, 1.0, 1.1, 0.9], fs=11.5)
    add_bullets(s, [
        ("✓ Zero speed-spikes across all 348,600 frames → no detectable 3D ID swaps.", 0, SUCCESS, True),
        ("✓ Residual far cleaner than round-4 (K_1_2: p90 3016→457, frac>5000 6.7%→0%).", 0, SUCCESS, False),
        ("✓ 95–96% of frames bound to a stable top-camera 2D identity.", 0, SUCCESS, False),
        ("Coverage 85–87% (pre gap-fill); the missing fish are occluded/undetected, not bad geometry.", 0, DARK, False),
        ("Caveat: metrics can't see smooth swaps at crossings → eye-review remains the acceptance test.", 0, WARN, False),
    ], 0.5, 3.5, 7.2, 3.3, size=13.5)
    add_video(s, VIDEOS / "3d_multiview_composite_clip.mp4", POSTERS / "3d_multiview_composite_clip.png",
              8.0, 1.45, 5.0, 4.2,
              caption="▶ 3D + multiview composite (embedded). Full K-group montages in videos/.")

    # ---- 7. Segmentation (Detectron2)
    s = blank(prs)
    title_bar(s, prs, "Segmentation — Detectron2 instance masks (identity cue independent of geometry)")
    add_bullets(s, [
        ("Why: 3D geometry smooth-swaps IDs at crossings; we wanted an identity cue that does NOT come from position.", 0, DARK, True),
        "Mask R-CNN (R50-FPN) fine-tuned per experiment: top 1,001 imgs / 2,551 inst; side 870 / 2,454.",
        "Dense inference over several windows → 33,179 crops (31,535 clean) across 171 view-tracklets.",
        ("Mask→fish linking by point-in-mask:", 0, ACCENT, True),
        ("exactly-1 tracklet point inside → clean single-fish crop", 1, DARK, False),
        ("0 points → reflection/false-positive (dropped); ≥2 → merged mask (dropped); + area-spike QC", 1, DARK, False),
        "Crops feed the self-supervised appearance-ID model (next slide).",
    ], 0.5, 1.35, 6.5, 5.5, size=14.5)
    add_image_fit(s, PLOTS / "detectron2_vs_sleap_top.png", 7.2, 1.5, 5.8, 2.5,
                  caption="Top view — Detectron2 masks (filled) agree with SLEAP poses (yellow).")
    add_image_fit(s, PLOTS / "detectron2_vs_sleap_side1.png", 7.2, 4.4, 5.8, 2.4,
                  caption="Side view — clean per-fish instance masks at separation.")

    # ---- 8. Appearance-ID self-supervised loop
    s = blank(prs)
    title_bar(s, prs, "Appearance ID — self-supervised, no human labels")
    add_bullets(s, [
        ("Two free constraints from the existing pipeline:", 0, DARK, True),
        ("must-link — crops in the same concrete tracklet are the same fish", 1, DARK, False),
        ("cannot-link — two instances in the same frame are different fish", 1, DARK, False),
        ("Self-supervision is internally consistent: cannot_link_satisfied = 0.99 (zero ID labels).", 0, ACCENT, True),
        ("Key finding (PCA, all 3 views):", 0, DARK, True),
        ("✓ ONE fish (the gravid female) separates onto its own arc;", 1, SUCCESS, True),
        ("✗ the two males collapse together (silhouette 0.31–0.43 ≪ 0.91 needed for per-frame ID).", 1, WARN, True),
        ("This exactly matches the 1-female / 2-look-alike-males biology.", 0, GREY, False),
    ], 0.5, 1.35, 6.2, 5.6, size=14.5)
    add_image_fit(s, PLOTS / "segid_per_view_pca.png", 6.7, 1.5, 6.4, 2.6,
                  caption="Per-view embedding (PCA): female (orange) splits off; the two males (blue/green) overlap.")
    add_image_fit(s, PLOTS / "segid_fish_crops_top.png", 6.7, 4.55, 6.4, 2.0,
                  caption="Segmented top-view fish crops fed to the model.")

    # ---- 9. Female anchor (SUCCESS)
    s = blank(prs)
    title_bar(s, prs, "Female anchor — achieved & wired into the pipeline", color=RGBColor(0x14,0x6C,0x2E))
    rows = [
        ["Fish (geometry slot)", "mean P(female)", "call"],
        ["ID_2  (slot 1)", "0.87", "FEMALE"],
        ["ID_1  (slot 0)", "0.08", "male"],
        ["ID_3  (slot 2)", "0.02", "male"],
    ]
    add_table(s, rows, 0.5, 1.5, 5.4, col_w=[2.6, 1.6, 1.2], fs=13)
    add_bullets(s, [
        ("✓ Per-fish call is decisive: confidence margin 0.79 (not ambiguous).", 0, SUCCESS, True),
        ("✓ Per-tracklet 91% female / 97% male correctly sided (clean bimodal).", 0, SUCCESS, False),
        ("Per-crop is only modest (0.72) due to gravid-state drift — but aggregating", 0, DARK, False),
        ("hundreds of crops per fish makes the per-FISH sex call unambiguous.", 1, DARK, False),
        ("Wired as a production filter → <tag>_3d_identified_sexed.csv (sex, female_prob).", 0, ACCENT, True),
        ("Whole-length orchestrator (script 40) built; K_1_1 / K_1_3 sexing runs launched 2026-06-22.", 0, GREY, False),
        ("The female is the chooser in mate-choice → she can now be pulled out directly downstream.", 0, DARK, True),
    ], 0.5, 3.6, 6.4, 3.4, size=13.5)
    add_image_fit(s, PLOTS / "female_prob_by_tracklet.png", 6.5, 1.5, 6.5, 4.0,
                  caption="Per-tracklet P(female): female tracklets (red) separate cleanly to the right.")

    # ---- 10. Two males NOT separable (honest)
    s = blank(prs)
    title_bar(s, prs, "The two males are NOT separable by appearance (verified)", color=RGBColor(0x8A,0x55,0x00))
    add_bullets(s, [
        ("This was a two-trap question — both misleading readings were caught:", 0, DARK, True),
        ("Trap 1 — pooled male-vs-male ≈ chance (balacc 0.51–0.59, ARI≈0).", 1, DARK, False),
        ("Trap 2 — per-window male-vs-male looks HIGH (up to 1.0) — but that's a position confound.", 1, DARK, False),
        ("The tracklet gate removes close-contact frames → males are ALWAYS spatially apart (0% < 100 px),", 0, WARN, False),
        ("so a within-window model can cheat on location, not appearance.", 1, WARN, False),
        ("De-confounded test (background-masked crops + window holdout + position baseline):", 0, ACCENT, True),
        ("✗ CNN 0.235 vs position baseline 0.303 — both BELOW chance. The male signal vanished.", 1, RGBColor(0xB0,0x30,0x30), True),
        ("Conclusion: resolve the two males with 3D geometry + the motion resolver, NOT appearance.", 0, DARK, True),
        ("(Male appearance only worth revisiting with un-gated crossing crops + a small human label set.)", 0, GREY, False),
    ], 0.6, 1.4, 12.1, 5.5, size=15)

    # ---- 11. Tracklet 3D-bind + hard-core resolver
    s = blank(prs)
    title_bar(s, prs, "Carrying ID through crossings — tracklet↔3D binding + resolver")
    add_bullets(s, [
        ("Conservative (stIFD-gated) tracklets bind to a unique 3D slot: ~6 px residual, ~209 px margin, 99.9% bijection.", 0, DARK, True),
        "3D binding carries identity through ~89–96% of single-view-ambiguous frames (Z from the side cams).",
        ("924 hard-core events remain (pair < 12 mm in 3D, 8.2% of frames):", 0, ACCENT, True),
        ("857 short (<0.5 s) · 67 ≥0.5 s · 20 ≥1 s dominate the burden", 1, DARK, False),
        ("Motion-continuity resolver: 513 cleared (55.5%), 411 flagged —", 0, DARK, False),
        ("incl. 18 suspect-swaps = velocity contradicts the linker = likely REAL ID errors (high-priority review).", 1, WARN, True),
        ("Next lever: fuse the female anchor + 3D geometry only inside crossing windows (resolve-at-crossing).", 0, SUCCESS, True),
    ], 0.5, 1.35, 7.1, 5.6, size=14)
    add_image_fit(s, PLOTS / "tracklet_coverage_stifd.png", 7.8, 1.5, 5.2, 2.4,
                  caption="Adaptive stIFD gate lifts clean coverage ~30%→~92%.")
    add_image_fit(s, POSTERS / "tracklet_suspect_swaps18.png", 7.8, 4.35, 5.2, 2.3,
                  caption="▶ videos/tracklet_suspect_swaps18.mp4 — the 18 suspects to eyeball.")

    # ---- 12. Behaviour validation — BORIS vs trajectory
    s = blank(prs)
    title_bar(s, prs, "From tracking to biology — 3D tracks vs manual behaviour coding (BORIS)",
              color=RGBColor(0x14, 0x6C, 0x2E))
    add_bullets(s, [
        ("K_1_2: 89 hand-coded events (54 courtship, 26 competitive, 4 mating). BORIS frame index", 0, DARK, True),
        ("aligns 1:1 with our 3D frame (top cam, 166 fps) → direct overlay, no registration needed.", 1, DARK, False),
        ("Female = appearance anchor (ID_2); the two males validated as a dyad (not yet separable).", 0, GREY, False),
        ("✓ Geometry corroborates the coding — female–nearest-male distance tracks the escalation:", 0, SUCCESS, True),
        ("baseline 40 mm  →  Courtship 28 mm  →  Touch 17 mm  →  Wrap 8 mm (contact)", 1, SUCCESS, True),
        ("✓ Wrap acceptance: accepted 28.0 s vs rejected 3.6 s — matches the lab's independent", 0, SUCCESS, False),
        ("~27 s vs ~3 s finding (results_EN_260616), derived purely from event durations.", 1, SUCCESS, False),
        ("This closes the loop: automated 3D geometry reproduces the manual ethogram.", 0, ACCENT, True),
    ], 0.5, 1.4, 6.6, 5.5, size=14)
    add_image_fit(s, BC / "behavior_distance_boxplot.png", 7.2, 1.55, 5.9, 5.1,
                  caption="Female–nearest-male distance during each coded behaviour.")

    # ---- 13. Spawn close-up
    s = blank(prs)
    title_bar(s, prs, "Spawn event — the trajectory pinpoints the mating", color=RGBColor(0x14, 0x6C, 0x2E))
    add_image_fit(s, BC / "spawn_closeup.png", 0.4, 1.3, 8.4, 5.7,
                  caption="Spawn at 818.6 s: female–male distances collapse through the wrap, re-open after.")
    add_bullets(s, [
        ("At Spawn (t=818.6 s, frame 135,883):", 0, DARK, True),
        ("female held in sustained contact ~7–10 mm", 1, DARK, False),
        ("with male slot ID_3 (7.1 mm) vs ID_1 (13.1 mm)", 1, DARK, False),
        ("through the green wrap interval.", 1, DARK, False),
        ("Distances drop to contact during the wrap and", 0, DARK, False),
        ("re-open immediately after spawning.", 1, DARK, False),
        ("→ Trajectory tells us WHICH slot mated;", 0, WARN, True),
        ("mapping slot→m1/m2 (i.e. paternity) needs the", 1, WARN, False),
        ("open male-ID problem solved.", 1, WARN, False),
    ], 8.9, 1.5, 4.2, 5.3, size=13.5)

    # ---- 14. What trajectories unlock next
    s = blank(prs)
    title_bar(s, prs, "What the 3D trajectories unlock next (results_EN_260616)")
    add_image_fit(s, BC / "ethogram_distance_timeline.png", 0.4, 1.25, 8.7, 4.2,
                  caption="Full session: BORIS ethogram (top) over inter-fish distance (mid) and female speed (bottom).")
    add_bullets(s, [
        ("Lab's candidate parameters — now computable from our 3D tracks:", 0, NAVY, True),
        ("✓ male–female distance mean/variance (following stability) — demoed", 1, SUCCESS, False),
        ("✓ female velocity modulation around accept/reject — demoed", 1, SUCCESS, False),
        ("→ quick-circle radius & peak velocity during courtship", 1, ACCENT, False),
        ("→ male–female heading-alignment (synchrony) index", 1, ACCENT, False),
        ("→ female trajectory after each courtship = accept vs reject signal", 1, ACCENT, False),
    ], 0.5, 5.65, 12.4, 1.7, size=13)

    # ---- 15. Summary & next steps
    s = blank(prs)
    title_bar(s, prs, "Summary & next steps")
    add_bullets(s, [
        ("Where the pipeline stands:", 0, NAVY, True),
        ("✓ Calibration — production model validated (0.90 mm); rigid-refinement path ready.", 0, SUCCESS, False),
        ("✓ SLEAP round-5 in production — clean 3D, zero speed-spikes over 348k frames.", 0, SUCCESS, False),
        ("✓ 3D reconstruction — geometry solid; IDs bound 95–96% with stable top-cam anchor.", 0, SUCCESS, False),
        ("✓ Female anchor — female reliably identified F/M and wired into the sexed-CSV output.", 0, SUCCESS, False),
        ("✓ Behaviour validation — 3D tracks reproduce the manual BORIS ethogram (distance escalation + wrap accept/reject).", 0, SUCCESS, False),
        ("✗ Two males by appearance — confirmed not separable; it's a geometry/continuity problem.", 0, WARN, False),
        ("Next:", 0, NAVY, True),
        ("→ Resolve-at-crossing: fuse female anchor + 3D inside crossing windows; verify the 18 suspect-swaps by eye.", 1, ACCENT, False),
        ("→ Round-6 SLEAP train on close-contact labels (warm-start) to reduce shared-tail swaps at the source.", 1, ACCENT, False),
        ("→ Compute the lab's mate-choice metrics from 3D (following stability, heading synchrony, circle radius); extend to K_1_1 / K_1_3.", 1, ACCENT, False),
    ], 0.6, 1.45, 12.1, 5.6, size=14.5)

    out = HERE / "medaka_progress_report_20260622.pptx"
    prs.save(str(out))
    return out


def main() -> None:
    print("Copying media…")
    copy_media()
    print("Extracting video posters…")
    make_posters()
    print("Building deck…")
    out = build_deck()
    print(f"\nDeck written: {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
