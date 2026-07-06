"""Build the poster pipeline figure: "How we track fish IDs in 3D".

Serpentine 2-row x 3-col grid of real pipeline outputs:
  Row 1 (L->R):  1 multi-view capture -> 2 SLEAP pose -> 3 Detectron2 masks
                 then  v  (down)
  Row 2 (R->L):  4 appearance ID -> 5 3D reconstruction + ID link -> 6 stable ID through crossings

Re-runnable: reads existing PNGs from prior report folders, crops sub-panels,
lays them out with stage badges + captions + flow arrows. Exports vector PDF.

Run:
  .venv\\Scripts\\python.exe report\\20260629\\build_pipeline_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Source images and pixel crop boxes (left, upper, right, lower).
# ----------------------------------------------------------------------------
COMPOSITE = REPO / "report/20260625/images/3d_multiview_composite_still.png"
D2SLEAP = REPO / "report/20260622/plots/detectron2_vs_sleap_top.png"
CROPS = REPO / "report/20260622/plots/segid_fish_crops_top.png"
CROSSING = REPO / "report/20260625/posters/K_1_2_crossing_montage.png"


def load_crop(path: Path, box: tuple[int, int, int, int]):
    img = Image.open(path).convert("RGB")
    return img.crop(box)


# Panel definitions in *serpentine* visual order keyed by (row, col).
ACCENT = "#1f5fa6"
TEAL = "#0e7c7b"
PANELS = {
    (0, 0): dict(
        n=1,
        title="Multi-view capture",
        cap="3 calibrated cameras (1 top + 2 side),\n166 fps, 1F + 2M medaka in a 20 cm tank",
        img=lambda: load_crop(COMPOSITE, (0, 0, 1440, 495)),
    ),
    (0, 1): dict(
        n=2,
        title="SLEAP pose estimation",
        cap="2-node skeleton (mouth→tail) per fish,\nper view (yellow overlay)",
        img=lambda: load_crop(D2SLEAP, (60, 60, 410, 470)),
    ),
    (0, 2): dict(
        n=3,
        title="Detectron2 segmentation",
        cap="Per-instance masks gate detections and\nfeed appearance crops (filled colors)",
        img=lambda: load_crop(D2SLEAP, (505, 495, 825, 905)),
    ),
    (1, 2): dict(
        n=4,
        title="Appearance ID",
        cap="Per-fish mask crops → embedding;\nthe 3 fish separate (sil 0.3–0.4)",
        img=lambda: load_crop(CROPS, (0, 0, 512, 384)),
    ),
    (1, 1): dict(
        n=5,
        title="3D reconstruction + ID link",
        cap="Cross-view triangulation via calibration;\nNN + optic-flow + velocity/gap-aware\nHungarian linking unifies IDs in 3D",
        img=lambda: load_crop(COMPOSITE, (505, 565, 935, 1035)),
    ),
    (1, 0): dict(
        n=6,
        title="Stable ID through crossings",
        cap="IDs held through close contact\n(min inter-fish dist 2.3 mm)",
        img=lambda: load_crop(CROSSING, (0, 0, 1080, 363)),
    ),
}

# ----------------------------------------------------------------------------
# Geometry (figure coordinates 0..1).
# ----------------------------------------------------------------------------
COL_X = [0.045, 0.370, 0.695]   # left edge of each column
COL_W = 0.260
ROW_Y = [0.575, 0.130]          # bottom edge of image axes per row
IMG_H = 0.235
TITLE_DY = 0.050                # title sits this far above the image
CAP_DY = 0.038                  # caption sits this far below the image


def add_arrow(fig, x0, y0, x1, y1):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=26,
        lw=3.2,
        color=ACCENT,
        shrinkA=0,
        shrinkB=0,
        zorder=5,
    )
    fig.add_artist(arr)


def main():
    fig = plt.figure(figsize=(14.0, 10.2))
    fig.patch.set_facecolor("white")

    # Header
    fig.text(0.5, 0.965, "Tracking individual fish identities in 3D",
             ha="center", va="top", fontsize=24, fontweight="bold", color="#16263a")
    fig.text(0.5, 0.925,
             "Multi-camera medaka mating pipeline: pose + segmentation → geometry → identity-stable 3D tracks",
             ha="center", va="top", fontsize=13, color="#46586b")

    for (row, col), spec in PANELS.items():
        x = COL_X[col]
        y = ROW_Y[row]
        ax = fig.add_axes([x, y, COL_W, IMG_H])
        im = spec["img"]()
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#c4ccd6")
            s.set_linewidth(1.2)

        cx = x + COL_W / 2
        # Title
        fig.text(cx + 0.018, y + IMG_H + TITLE_DY, spec["title"],
                 ha="center", va="center", fontsize=14, fontweight="bold",
                 color="#16263a")
        # Numbered badge
        bx = x + 0.012
        by = y + IMG_H + TITLE_DY
        badge = Circle((bx, by), 0.018, transform=fig.transFigure,
                       facecolor=ACCENT, edgecolor="white", lw=1.5, zorder=6)
        fig.add_artist(badge)
        fig.text(bx, by, str(spec["n"]), ha="center", va="center",
                 fontsize=13, fontweight="bold", color="white", zorder=7)
        # Caption
        fig.text(cx, y - CAP_DY, spec["cap"], ha="center", va="top",
                 fontsize=10.5, color="#3c4a5a", linespacing=1.35)

    # --- Flow arrows (serpentine) ---
    yc1 = ROW_Y[0] + IMG_H / 2
    yc2 = ROW_Y[1] + IMG_H / 2
    gap = 0.012
    # Row 1: col0 -> col1 -> col2
    add_arrow(fig, COL_X[0] + COL_W + gap, yc1, COL_X[1] - gap, yc1)
    add_arrow(fig, COL_X[1] + COL_W + gap, yc1, COL_X[2] - gap, yc1)
    # Down: col2 row1 -> col2 row2
    dx = COL_X[2] + COL_W / 2
    add_arrow(fig, dx, ROW_Y[0] - CAP_DY - 0.055, dx, ROW_Y[1] + IMG_H + 0.040)
    # Row 2: col2 -> col1 -> col0  (leftward)
    add_arrow(fig, COL_X[2] - gap, yc2, COL_X[1] + COL_W + gap, yc2)
    add_arrow(fig, COL_X[1] - gap, yc2, COL_X[0] + COL_W + gap, yc2)

    pdf = OUT_DIR / "pipeline_3d_id_tracking.pdf"
    png = OUT_DIR / "pipeline_3d_id_tracking.png"
    fig.savefig(pdf)                       # vector
    fig.savefig(png, dpi=200)              # raster QC preview
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
