from __future__ import annotations

"""Build source-backed figures and diagnostic clips for geometry-contrastive ID.

The static panels use only locked samples, pair scores, and benchmark summaries.
Videos overlay tracklet-first dispositions on the original synchronized cameras.
All displayed identity labels are explicitly local geometry sources or fragment
components; the script never promotes them to biological fish names.
"""

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#25313B"
MUTED = "#65727D"
GRID = "#DDE3E8"
BLUE = "#2F6B8A"
BLUE_LIGHT = "#BFD8E6"
GOLD = "#D9A441"
GOLD_LIGHT = "#F2E2B8"
ORANGE = "#D66A3A"
ORANGE_LIGHT = "#F1C7B4"
PINK = "#A95783"
PINK_LIGHT = "#E6C2D5"
GREY = "#A7B0B7"
LIGHT_GREY = "#F2F4F6"
WHITE = "#FFFFFF"
LOCAL_COLORS = {0: BLUE, 1: GOLD, 2: PINK}
VIEW_ORDER = ("top_cam", "side1_cam", "side2_cam")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def _save(fig: plt.Figure, output_base: Path) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


class CropReader:
    def __init__(self, path: Path) -> None:
        self._h5 = h5py.File(path, "r")
        self._crops = self._h5["crops"]

    def get(self, crop_index: int) -> np.ndarray:
        return np.asarray(self._crops[int(crop_index)])

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "CropReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _display_crop(ax: plt.Axes, crop: np.ndarray, title: str = "") -> None:
    ax.imshow(crop)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=9, pad=4)


def _load_sources(run_root: Path) -> dict[str, Any]:
    experiment = "20251225_K_1_2_FCMC"
    experiment_root = run_root / experiment
    dataset = experiment_root / "dataset_v2"
    evaluation = experiment_root / "evaluations" / "frozen_d128_e3_s100"
    return {
        "experiment": experiment,
        "experiment_root": experiment_root,
        "dataset": dataset,
        "samples": pd.read_parquet(dataset / "samples.parquet"),
        "pairs": pd.read_csv(evaluation / "pair_scores.csv"),
        "evaluation": json.loads((evaluation / "evaluation_report.json").read_text(encoding="utf-8")),
        "models": pd.read_csv(run_root / "MODEL_COMPARISON_BY_EXPERIMENT.csv"),
        "strata": pd.read_csv(run_root / "STRATIFIED_METRICS.csv"),
        "global": pd.read_csv(run_root / "GLOBAL_STITCHING_COMPARISON.csv"),
        "crops_path": dataset / "crops.h5",
    }


def _pair_quality(pairs: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sample_id",
        "crop_index",
        "matched_mask_score",
        "matched_mask_area_px",
        "component_id_provenance",
    ]
    left = samples[columns].add_suffix("_a")
    right = samples[columns].add_suffix("_b")
    table = pairs.merge(left, left_on="sample_a_id", right_on="sample_id_a", how="left")
    table = table.merge(right, left_on="sample_b_id", right_on="sample_id_b", how="left")
    table["display_quality"] = table[["matched_mask_score_a", "matched_mask_score_b"]].min(axis=1)
    table["display_area"] = table[["matched_mask_area_px_a", "matched_mask_area_px_b"]].min(axis=1)
    return table


def _select_diverse(
    rows: pd.DataFrame,
    *,
    n: int,
    probability_order: str,
    used_samples: set[int],
) -> pd.DataFrame:
    ascending = probability_order == "low"
    ranked = rows.sort_values(
        ["same_probability", "display_quality", "display_area"],
        ascending=[ascending, False, False],
    )
    selected: list[pd.Series] = []
    seen_cameras: set[str] = set()
    for _index, row in ranked.iterrows():
        sample_ids = {int(row.sample_a_id), int(row.sample_b_id)}
        if sample_ids & used_samples:
            continue
        camera = str(row.camera_pair)
        if camera in seen_cameras and len(seen_cameras) < n:
            continue
        selected.append(row)
        used_samples.update(sample_ids)
        seen_cameras.add(camera)
        if len(selected) == n:
            break
    if len(selected) < n:
        for _index, row in ranked.iterrows():
            sample_ids = {int(row.sample_a_id), int(row.sample_b_id)}
            if sample_ids & used_samples:
                continue
            selected.append(row)
            used_samples.update(sample_ids)
            if len(selected) == n:
                break
    if len(selected) != n:
        raise ValueError(f"Could not select {n} display pairs from {len(rows)} candidates")
    return pd.DataFrame(selected)


def select_pair_examples(pairs: pd.DataFrame, samples: pd.DataFrame, n_each: int = 2) -> pd.DataFrame:
    table = _pair_quality(pairs.query("split == 'test'").copy(), samples)
    table = table[(table.display_quality >= 0.75) & (table.display_area >= 250)]
    prediction = table.predicted_same_frozen_threshold.astype(bool)
    same = table.label.eq(1)
    cross_camera = table.camera_a.astype(str).ne(table.camera_b.astype(str))
    groups = [
        ("Correct same", table[same & prediction & cross_camera], "high"),
        ("Missed same", table[same & ~prediction & cross_camera], "low"),
        ("Correct different", table[~same & ~prediction], "low"),
        ("False same", table[~same & prediction], "high"),
    ]
    used: set[int] = set()
    parts: list[pd.DataFrame] = []
    for outcome, candidates, order in groups:
        chosen = _select_diverse(candidates, n=n_each, probability_order=order, used_samples=used)
        chosen.insert(0, "outcome", outcome)
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True)


def select_complete_local_source_frame(samples: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    test = samples.query("split == 'test'").copy()
    unique = (
        test.sort_values("matched_mask_score", ascending=False)
        .drop_duplicates(["frame", "geometry_source_idx_local", "view"])
    )
    summary = unique.groupby("frame").agg(
        n_ids=("geometry_source_idx_local", "nunique"),
        n_views=("view", "nunique"),
        n_cells=("sample_id", "size"),
        mean_residual=("residual_px2", "mean"),
        max_residual=("residual_px2", "max"),
        min_mask=("matched_mask_score", "min"),
        mean_mask=("matched_mask_score", "mean"),
        mean_separation=("min_fish_separation_mm", "mean"),
    )
    eligible = summary.query(
        "n_ids == 3 and n_views == 3 and n_cells >= 9 and min_mask >= 0.85 and mean_separation >= 25"
    )
    if eligible.empty:
        eligible = summary.query("n_ids == 3 and n_views == 3 and n_cells >= 9")
    frame = int(
        eligible.sort_values(
            ["mean_residual", "max_residual", "min_mask"],
            ascending=[True, True, False],
        ).index[0]
    )
    chosen = unique[unique.frame.eq(frame)].copy()
    return frame, chosen


def _draw_overview(sources: dict[str, Any], output_dir: Path) -> list[str]:
    pairs = sources["pairs"].query("split == 'test'")
    report = sources["evaluation"]
    threshold = report["thresholds_frozen_on_calibration"]["decision_threshold_cosine"]
    models = sources["models"].query("status == 'complete'").sort_values("embedding_dim")
    best_strata = sources["strata"].query("model == 'frozen_d128_e3_s100'")
    camera = best_strata[
        best_strata.stratifier.eq("camera_pair")
        & best_strata.stratum.isin(VIEW_ORDER)
    ].copy()
    camera["label"] = camera.stratum.str.replace("_cam", "", regex=False)
    cross = best_strata[
        best_strata.stratifier.eq("camera_pair")
        & best_strata.stratum.astype(str).str.contains(r"\|")
    ].copy()
    cross["label"] = cross.stratum.str.replace("_cam", "", regex=False).str.replace("|", "↔", regex=False)

    fig = plt.figure(figsize=(16, 10.5))
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.1, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.065,
        right=0.98,
        bottom=0.075,
        top=0.855,
        hspace=0.36,
        wspace=0.22,
    )
    fig.suptitle(
        "Appearance contains signal—but not enough for reliable biological fish identity",
        fontsize=19,
        fontweight="bold",
        x=0.02,
        y=0.975,
        ha="left",
        color=INK,
    )
    fig.text(
        0.02,
        0.925,
        "Held-out K1_2 real data · geometry-supervised pair labels · frozen ResNet18 · biological names unavailable",
        fontsize=11,
        color=MUTED,
        ha="left",
    )

    ax_dist = fig.add_subplot(outer[0, 0])
    _panel_label(ax_dist, "A")
    bins = np.linspace(float(pairs.cosine_similarity.min()), 1.0, 42)
    same = pairs.loc[pairs.label.eq(1), "cosine_similarity"].to_numpy(float)
    different = pairs.loc[pairs.label.eq(0), "cosine_similarity"].to_numpy(float)
    ax_dist.hist(different, bins=bins, density=True, color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.4, label=f"Different source (n={len(different):,})")
    ax_dist.hist(same, bins=bins, density=True, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.4, alpha=0.78, label=f"Same constraint (n={len(same):,})")
    ax_dist.axvline(threshold, color=INK, linestyle="--", linewidth=1.6, label=f"Calibration threshold = {threshold:.3f}")
    ax_dist.set_title("Held-out similarity distributions overlap substantially", loc="left", fontweight="bold")
    ax_dist.set_xlabel("Cosine similarity (larger = model thinks ‘same’)")
    ax_dist.set_ylabel("Density (each class normalized)")
    ax_dist.grid(axis="y", color=GRID, linewidth=0.7)
    ax_dist.legend(frameon=False, fontsize=9, loc="upper left")
    ax_dist.text(
        0.02,
        0.03,
        "Overlap produces both false links and missed links.",
        transform=ax_dist.transAxes,
        color=INK,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": WHITE, "edgecolor": GRID},
    )

    capacity = outer[0, 1].subgridspec(1, 2, wspace=0.38)
    ax_auc = fig.add_subplot(capacity[0, 0])
    _panel_label(ax_auc, "B")
    dimensions = models.embedding_dim.astype(int).to_numpy()
    x = np.arange(len(models))
    auc = models.test_auroc.to_numpy(float)
    lower = auc - models.test_auroc_ci_lower.to_numpy(float)
    upper = models.test_auroc_ci_upper.to_numpy(float) - auc
    ax_auc.errorbar(x, auc, yerr=np.vstack([lower, upper]), fmt="o", color=BLUE, ecolor=BLUE, capsize=5, markersize=8, linewidth=2)
    ax_auc.axhline(0.5, color=GREY, linestyle="--", linewidth=1.2)
    ax_auc.set_xticks(x, [f"{value}-D" for value in dimensions])
    ax_auc.set_ylim(0.45, 0.86)
    ax_auc.set_ylabel("Held-out AUROC")
    ax_auc.set_title("Ranking improves with capacity", loc="left", fontweight="bold")
    ax_auc.grid(axis="y", color=GRID, linewidth=0.7)
    for xpos, value in zip(x, auc, strict=True):
        ax_auc.text(xpos, value + 0.025, f"{value:.3f}", ha="center", fontsize=9, color=INK)

    ax_eer = fig.add_subplot(capacity[0, 1])
    eer = models.test_eer.to_numpy(float)
    ax_eer.bar(x, eer, color=GOLD_LIGHT, edgecolor=GOLD, linewidth=1.3, width=0.62)
    ax_eer.set_xticks(x, [f"{value}-D" for value in dimensions])
    ax_eer.set_ylim(0, 0.45)
    ax_eer.set_ylabel("Equal-error rate (lower is better)")
    ax_eer.set_title("Error remains high", loc="left", fontweight="bold")
    ax_eer.grid(axis="y", color=GRID, linewidth=0.7)
    for xpos, value in zip(x, eer, strict=True):
        ax_eer.text(xpos, value + 0.015, f"{value:.1%}", ha="center", fontsize=9, color=INK)

    camera_grid = outer[1, 0].subgridspec(1, 2, wspace=0.34)
    ax_cam = fig.add_subplot(camera_grid[0, 0])
    _panel_label(ax_cam, "C")
    camera = camera.sort_values("label")
    bars = ax_cam.bar(camera.label, camera.auroc, color=[BLUE_LIGHT, GOLD_LIGHT, PINK_LIGHT], edgecolor=[BLUE, GOLD, PINK], linewidth=1.3)
    ax_cam.axhline(0.5, color=GREY, linestyle="--", linewidth=1.2)
    ax_cam.set_ylim(0.45, 0.9)
    ax_cam.set_ylabel("Within-camera AUROC")
    ax_cam.set_title("Top view is the weakest camera", loc="left", fontweight="bold")
    ax_cam.grid(axis="y", color=GRID, linewidth=0.7)
    for bar, value in zip(bars, camera.auroc, strict=True):
        ax_cam.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.018, f"{float(value):.3f}", ha="center", fontsize=9)

    ax_cross = fig.add_subplot(camera_grid[0, 1])
    cross = cross.sort_values("label")
    bars = ax_cross.barh(cross.label, cross.tar_at_frozen_threshold, color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.3)
    ax_cross.set_xlim(0, 1)
    ax_cross.set_xlabel("Cross-camera true-accept rate")
    ax_cross.set_title("Most true cross-view pairs are missed", loc="left", fontweight="bold")
    ax_cross.grid(axis="x", color=GRID, linewidth=0.7)
    for bar, value in zip(bars, cross.tar_at_frozen_threshold, strict=True):
        ax_cross.text(float(value) + 0.02, bar.get_y() + bar.get_height() / 2, f"{float(value):.1%}", va="center", fontsize=9)

    ax_take = fig.add_subplot(outer[1, 1])
    _panel_label(ax_take, "D")
    ax_take.axis("off")
    ax_take.set_title("System-level result", loc="left", fontweight="bold", pad=10)
    boxes = [
        (0.05, 0.62, 0.25, 0.2, "36 fragments", LIGHT_GREY, INK),
        (0.38, 0.62, 0.25, 0.2, "13 paths\ngeometry only", BLUE_LIGHT, BLUE),
        (0.71, 0.62, 0.25, 0.2, "13 paths\n+ appearance", GOLD_LIGHT, GOLD),
    ]
    for x0, y0, w, h, text, face, edge in boxes:
        patch = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02", facecolor=face, edgecolor=edge, linewidth=1.5, transform=ax_take.transAxes)
        ax_take.add_patch(patch)
        ax_take.text(x0 + w / 2, y0 + h / 2, text, transform=ax_take.transAxes, ha="center", va="center", fontsize=11, fontweight="bold")
    for x0, x1 in ((0.30, 0.38), (0.63, 0.71)):
        ax_take.add_patch(FancyArrowPatch((x0, 0.72), (x1, 0.72), transform=ax_take.transAxes, arrowstyle="-|>", mutation_scale=14, color=MUTED, linewidth=1.4))
    callouts = [
        ("1 edge changed", "Appearance altered one predecessor choice."),
        ("0 path gain", "Fragment reduction and coverage were unchanged."),
        ("3D unchanged", "XYZ matched exactly; speed spikes stayed 3 → 3."),
        ("No biological anchor", "The changed edge cannot be scored as correct or wrong."),
    ]
    y = 0.50
    for headline, body in callouts:
        ax_take.text(0.05, y, headline, transform=ax_take.transAxes, fontsize=10.5, fontweight="bold", color=INK)
        ax_take.text(0.36, y, body, transform=ax_take.transAxes, fontsize=9.2, color=MUTED)
        y -= 0.10
    bottom = FancyBboxPatch(
        (0.04, 0.015),
        0.92,
        0.10,
        boxstyle="round,pad=0.02",
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        linewidth=1.3,
        transform=ax_take.transAxes,
    )
    ax_take.add_patch(bottom)
    ax_take.text(
        0.50,
        0.065,
        "Bottom line: useful appearance evidence, but not yet reliable biological identity.",
        transform=ax_take.transAxes,
        fontsize=10.3,
        fontweight="bold",
        color=ORANGE,
        ha="center",
        va="center",
    )
    return _save(fig, output_dir / "FIGURE_1_PERFORMANCE_OVERVIEW")


def _draw_pair_examples(
    sources: dict[str, Any],
    examples: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    sample_map = sources["samples"].set_index("sample_id")
    outcome_order = ["Correct same", "Missed same", "Correct different", "False same"]
    outline = {
        "Correct same": BLUE,
        "Missed same": ORANGE,
        "Correct different": BLUE,
        "False same": ORANGE,
    }
    fig = plt.figure(figsize=(15.5, 13.2))
    grid = fig.add_gridspec(
        4,
        3,
        width_ratios=[0.25, 1.0, 1.0],
        left=0.035,
        right=0.98,
        bottom=0.065,
        top=0.88,
        hspace=0.34,
        wspace=0.14,
    )
    fig.suptitle(
        "What the appearance model gets right—and wrong on real held-out crops",
        fontsize=19,
        fontweight="bold",
        x=0.02,
        y=0.975,
        ha="left",
    )
    threshold = sources["evaluation"]["thresholds_frozen_on_calibration"]["decision_threshold_probability"]
    fig.text(
        0.02,
        0.925,
        f"Blue = correct decision · orange = error · calibrated ‘same’ threshold = {threshold:.1%} · pair labels come from geometry constraints",
        fontsize=10.5,
        color=MUTED,
    )
    outputs: list[dict[str, Any]] = []
    with CropReader(sources["crops_path"]) as crops:
        definitions = {
            "Correct same": "Same source\naccepted",
            "Missed same": "Same source\nrejected",
            "Correct different": "Different sources\nrejected",
            "False same": "Different sources\naccepted",
        }
        for row_index, outcome in enumerate(outcome_order):
            rows = examples[examples.outcome.eq(outcome)].reset_index(drop=True)
            ax_label = fig.add_subplot(grid[row_index, 0])
            ax_label.axis("off")
            ax_label.text(
                0.02,
                0.65,
                outcome,
                transform=ax_label.transAxes,
                fontsize=12.2,
                fontweight="bold",
                color=outline[outcome],
                va="center",
            )
            ax_label.text(
                0.02,
                0.35,
                definitions[outcome],
                transform=ax_label.transAxes,
                fontsize=10,
                color=MUTED,
                va="center",
                linespacing=1.35,
            )
            for example_index, row in rows.iterrows():
                cell = grid[row_index, example_index + 1].subgridspec(
                    2,
                    2,
                    height_ratios=[1.0, 0.31],
                    hspace=0.04,
                    wspace=0.04,
                )
                ax_a = fig.add_subplot(cell[0, 0])
                ax_b = fig.add_subplot(cell[0, 1])
                meta_a = sample_map.loc[int(row.sample_a_id)]
                meta_b = sample_map.loc[int(row.sample_b_id)]
                _display_crop(ax_a, crops.get(int(meta_a.crop_index)), f"{str(meta_a.view).replace('_cam','')} · f{int(meta_a.frame)}")
                _display_crop(ax_b, crops.get(int(meta_b.crop_index)), f"{str(meta_b.view).replace('_cam','')} · f{int(meta_b.frame)}")
                for ax in (ax_a, ax_b):
                    for spine in ax.spines.values():
                        spine.set_color(outline[outcome])
                        spine.set_linewidth(2.0)
                ax_text = fig.add_subplot(cell[1, :])
                ax_text.axis("off")
                truth = "same" if int(row.label) == 1 else "different"
                predicted = "same" if bool(row.predicted_same_frozen_threshold) else "different"
                ax_text.text(0.02, 0.72, f"Truth: {truth}  →  model: {predicted}", fontsize=10, fontweight="bold", color=INK, transform=ax_text.transAxes)
                ax_text.text(0.02, 0.39, f"P(same) = {float(row.same_probability):.1%} · cosine = {float(row.cosine_similarity):.3f}", fontsize=9.2, color=MUTED, transform=ax_text.transAxes)
                pair_kind = str(row.pair_kind).replace("_", " ")
                ax_text.text(0.02, 0.10, pair_kind, fontsize=8.8, color=MUTED, transform=ax_text.transAxes)
                outputs.append(
                    {
                        "outcome": outcome,
                        "sample_a_id": int(row.sample_a_id),
                        "sample_b_id": int(row.sample_b_id),
                        "camera_pair": str(row.camera_pair),
                        "frame_a": int(row.frame_a),
                        "frame_b": int(row.frame_b),
                        "label": int(row.label),
                        "same_probability": float(row.same_probability),
                        "cosine_similarity": float(row.cosine_similarity),
                    }
                )
    fig.text(
        0.02,
        0.015,
        "Interpretation: cross-camera changes can make the same geometry source look different, while two different fish can look deceptively similar. These are local constraints, not verified lifelong fish identities.",
        fontsize=10,
        color=INK,
    )
    paths = _save(fig, output_dir / "FIGURE_2_REAL_PAIR_DECISIONS")
    (output_dir / "PAIR_EXAMPLES.json").write_text(json.dumps(_json_safe(outputs), indent=2), encoding="utf-8")
    return paths


def _draw_local_source_montage(
    sources: dict[str, Any],
    frame: int,
    selected: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    fig, axes = plt.subplots(3, 3, figsize=(11.8, 10.8))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.075, top=0.86, hspace=0.30, wspace=0.12)
    fig.suptitle(
        "One synchronized frame: three local geometry sources across three cameras",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        y=0.975,
        ha="left",
    )
    fig.text(
        0.02,
        0.925,
        f"Real held-out frame {frame} · all 9 strict RGB mask crops present · rows are local source constraints, not biological fish names",
        fontsize=10.5,
        color=MUTED,
    )
    records: list[dict[str, Any]] = []
    with CropReader(sources["crops_path"]) as crops:
        for row_index, source_id in enumerate((0, 1, 2)):
            for column, view in enumerate(VIEW_ORDER):
                ax = axes[row_index, column]
                row = selected[
                    selected.geometry_source_idx_local.eq(source_id) & selected.view.eq(view)
                ].sort_values("matched_mask_score", ascending=False).iloc[0]
                _display_crop(
                    ax,
                    crops.get(int(row.crop_index)),
                    f"{view.replace('_cam','')}\nmask {float(row.matched_mask_score):.2f} · residual {float(row.residual_px2):.0f} px²",
                )
                for spine in ax.spines.values():
                    spine.set_color(LOCAL_COLORS[source_id])
                    spine.set_linewidth(2.5)
                if column == 0:
                    ax.text(
                        -0.18,
                        0.5,
                        f"Local\nsource {source_id}",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color=LOCAL_COLORS[source_id],
                    )
                records.append(
                    {
                        "frame": frame,
                        "local_source": source_id,
                        "view": view,
                        "sample_id": int(row.sample_id),
                        "crop_index": int(row.crop_index),
                        "mask_score": float(row.matched_mask_score),
                        "residual_px2": float(row.residual_px2),
                    }
                )
    fig.text(
        0.02,
        0.017,
        "Why this matters: pose, illumination, refraction, and camera angle change appearance strongly. The learning problem asks whether those views can still be linked without confusing another fish.",
        fontsize=10,
        color=INK,
    )
    paths = _save(fig, output_dir / "FIGURE_3_THREE_CAMERA_LOCAL_SOURCES")
    (output_dir / "LOCAL_SOURCE_MONTAGE.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return paths


def _component_thumbnails(
    ax: plt.Axes,
    samples: pd.DataFrame,
    component: str,
    crops: CropReader,
    *,
    title: str,
    edge_color: str,
) -> list[dict[str, Any]]:
    rows = samples[samples.component_id_provenance.astype(str).eq(component)].copy()
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=edge_color)
    if rows.empty:
        patch = FancyBboxPatch((0.05, 0.2), 0.9, 0.58, boxstyle="round,pad=0.02", facecolor=LIGHT_GREY, edgecolor=GREY, linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(0.5, 0.56, "No strict appearance crop", transform=ax.transAxes, ha="center", va="center", fontweight="bold", color=INK)
        ax.text(0.5, 0.38, "appearance state = unknown", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        return []
    chosen = (
        rows.sort_values("matched_mask_score", ascending=False)
        .drop_duplicates("view")
        .head(3)
    )
    records: list[dict[str, Any]] = []
    width = 0.29
    for index, row in enumerate(chosen.itertuples(index=False)):
        inset = ax.inset_axes([0.02 + index * 0.325, 0.18, width, 0.66])
        _display_crop(inset, crops.get(int(row.crop_index)), str(row.view).replace("_cam", ""))
        for spine in inset.spines.values():
            spine.set_color(edge_color)
            spine.set_linewidth(1.7)
        records.append({"component": component, "sample_id": int(row.sample_id), "frame": int(row.frame), "view": str(row.view), "crop_index": int(row.crop_index)})
    return records


def _draw_global_linker_case(sources: dict[str, Any], output_dir: Path) -> list[str]:
    source_supported = "test_78000_78999::side1_cam:172"
    source_baseline = "test_78000_78999::side1_cam:183"
    target = "test_78000_78999::side2_cam:265"
    app_edges = pd.read_csv(
        sources["experiment_root"] / "global_stitching_v2" / "geometry_plus_appearance" / "selected_global_edges.csv"
    )
    edge = app_edges[
        app_edges.source_id.eq(source_supported) & app_edges.target_id.eq(target)
    ].iloc[0]
    fig = plt.figure(figsize=(15.5, 9.8))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.78, 1.25],
        left=0.055,
        right=0.98,
        bottom=0.065,
        top=0.86,
        hspace=0.14,
        wspace=0.16,
    )
    fig.suptitle(
        "One global-link decision changed by appearance—without biological ground truth",
        fontsize=18.5,
        fontweight="bold",
        x=0.02,
        y=0.975,
        ha="left",
    )
    fig.text(
        0.02,
        0.925,
        "Real held-out fragments around frames 78,334–78,342 · geometry-only and appearance-assisted solutions compared",
        fontsize=10.5,
        color=MUTED,
    )
    records: list[dict[str, Any]] = []
    fig.text(0.02, 0.85, "A", fontsize=14, fontweight="bold", color=INK)
    with CropReader(sources["crops_path"]) as crops:
        records += _component_thumbnails(fig.add_subplot(grid[0, 0]), sources["samples"], source_supported, crops, title="Appearance-supported predecessor · C172", edge_color=GOLD)
        records += _component_thumbnails(fig.add_subplot(grid[0, 1]), sources["samples"], source_baseline, crops, title="Geometry-only predecessor · C183", edge_color=PINK)
        records += _component_thumbnails(fig.add_subplot(grid[0, 2]), sources["samples"], target, crops, title="Target fragment · C265", edge_color=BLUE)

    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    fig.text(0.02, 0.49, "B", fontsize=14, fontweight="bold", color=INK)
    positions = {
        "C172": (0.08, 0.60, GOLD_LIGHT, GOLD),
        "C183": (0.08, 0.23, PINK_LIGHT, PINK),
        "C265": (0.73, 0.42, BLUE_LIGHT, BLUE),
    }
    for label, (x0, y0, face, edge_color) in positions.items():
        patch = FancyBboxPatch((x0, y0), 0.16, 0.18, boxstyle="round,pad=0.025", facecolor=face, edgecolor=edge_color, linewidth=2, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x0 + 0.08, y0 + 0.09, label, ha="center", va="center", transform=ax.transAxes, fontweight="bold", fontsize=13)
    ax.add_patch(FancyArrowPatch((0.24, 0.69), (0.73, 0.51), transform=ax.transAxes, connectionstyle="arc3,rad=-0.08", arrowstyle="-|>", mutation_scale=18, color=GOLD, linewidth=3))
    ax.text(0.47, 0.67, f"appearance support\nP(same) = {float(edge.appearance_same_probability):.2%}", transform=ax.transAxes, ha="center", va="bottom", color=GOLD, fontweight="bold", fontsize=11)
    ax.add_patch(FancyArrowPatch((0.24, 0.32), (0.73, 0.48), transform=ax.transAxes, connectionstyle="arc3,rad=0.08", arrowstyle="-|>", mutation_scale=18, color=PINK, linewidth=2.2, linestyle="--"))
    ax.text(0.47, 0.26, "geometry-only choice", transform=ax.transAxes, ha="center", va="top", color=PINK, fontweight="bold", fontsize=11)
    ax.text(0.81, 0.24, "7-frame gap", transform=ax.transAxes, ha="center", color=MUTED, fontsize=10)

    callout = FancyBboxPatch((0.29, 0.01), 0.63, 0.15, boxstyle="round,pad=0.025", facecolor=LIGHT_GREY, edgecolor=GREY, linewidth=1.3, transform=ax.transAxes)
    ax.add_patch(callout)
    ax.text(
        0.605,
        0.085,
        "The solutions swap one predecessor edge, but both keep 13 global paths and 23 selected edges.\nNo trusted biological anchor exists here, so the appearance decision is plausible—not validated.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
    )
    paths = _save(fig, output_dir / "FIGURE_4_GLOBAL_LINKER_CASE")
    (output_dir / "GLOBAL_LINKER_CASE.json").write_text(
        json.dumps(
            {
                "appearance_supported_edge": [source_supported, target],
                "geometry_only_edge": [source_baseline, target],
                "appearance_same_probability": float(edge.appearance_same_probability),
                "appearance_cosine_distance": float(edge.appearance_cosine_distance),
                "gap_frames": int(edge.gap_frames),
                "endpoint_displacement_mm": float(edge.endpoint_displacement_mm),
                "motion_prediction_error_mm": float(edge.motion_prediction_error_mm),
                "strict_crop_examples": records,
                "biological_ground_truth_available": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def build_figures(run_root: Path, output_dir: Path) -> dict[str, Any]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _load_sources(run_root)
    examples = select_pair_examples(sources["pairs"], sources["samples"])
    frame, local_samples = select_complete_local_source_frame(sources["samples"])
    outputs: list[str] = []
    outputs += _draw_overview(sources, output_dir)
    outputs += _draw_pair_examples(sources, examples, output_dir)
    outputs += _draw_local_source_montage(sources, frame, local_samples, output_dir)
    outputs += _draw_global_linker_case(sources, output_dir)

    model_metrics = sources["models"].query("status == 'complete'").copy()
    model_metrics.insert(0, "metric_scope", "model")
    best_strata = sources["strata"].query("model == 'frozen_d128_e3_s100'").copy()
    best_strata.insert(0, "metric_scope", "stratum")
    pd.concat([model_metrics, best_strata], ignore_index=True, sort=False).to_csv(
        output_dir / "FIGURE_METRICS.csv", index=False
    )
    chart_map = """# Communication figure map

## Figure 1 — Performance overview

- Question: Does appearance reliably separate same-source from different-source fish observations?
- Forms: normalized similarity histogram; dot-and-interval AUROC comparison; EER bars; camera bars; structural flow.
- Takeaway: signal improves with embedding capacity, but pair overlap, top-camera weakness, and cross-camera misses remain substantial.

## Figure 2 — Real pair decisions

- Question: What do correct and incorrect appearance decisions look like?
- Form: deterministic held-out crop-pair small multiples.
- Takeaway: viewpoint changes create false negatives, while visually similar different fish create false positives.

## Figure 3 — Three-camera local sources

- Question: How different can the same local geometry source look across synchronized views?
- Form: 3 × 3 real crop montage at one complete held-out frame.
- Takeaway: the model must bridge major pose, illumination, and refraction changes; row labels are not biological identities.

## Figure 4 — Global linker case

- Question: What changed when appearance entered global stitching?
- Form: real crop evidence plus competing-edge diagram.
- Takeaway: one plausible edge changed at high appearance confidence, but biological correctness is unscorable without an anchor.

Palette: blue/gold/orange/pink roots plus neutrals; correct/error states also use text and outlines, not color alone.
"""
    (output_dir / "CHART_MAP.md").write_text(chart_map, encoding="utf-8")
    manifest = {
        "schema_version": "geometry_contrastive_communication_figures_v1",
        "experiment": sources["experiment"],
        "identity_semantics": "local geometry source and fragment constraints only; no biological fish names",
        "selected_complete_frame": frame,
        "outputs": outputs,
        "sources": {
            "samples": str(sources["dataset"] / "samples.parquet"),
            "crops": str(sources["crops_path"]),
            "pair_scores": str(sources["experiment_root"] / "evaluations" / "frozen_d128_e3_s100" / "pair_scores.csv"),
            "evaluation_report": str(sources["experiment_root"] / "evaluations" / "frozen_d128_e3_s100" / "evaluation_report.json"),
            "model_comparison": str(run_root / "MODEL_COMPARISON_BY_EXPERIMENT.csv"),
            "stratified_metrics": str(run_root / "STRATIFIED_METRICS.csv"),
            "global_comparison": str(run_root / "GLOBAL_STITCHING_COMPARISON.csv"),
        },
    }
    (output_dir / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    red, green, blue = (int(color[index:index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = _hex_to_bgr(INK),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def _open_videos(video_paths: dict[str, Path], start_frame: int) -> tuple[dict[str, cv2.VideoCapture], dict[str, Any]]:
    captures: dict[str, cv2.VideoCapture] = {}
    metadata: dict[str, Any] = {}
    for view in VIEW_ORDER:
        path = video_paths[view]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for {view}: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        captures[view] = capture
        metadata[view] = {
            "path": str(path),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    return captures, metadata


def _release(captures: Iterable[cv2.VideoCapture]) -> None:
    for capture in captures:
        capture.release()


def _disposition_lookup(dispositions: pd.DataFrame) -> dict[tuple[int, str], list[dict[str, Any]]]:
    table = dispositions.copy()
    table["frame"] = pd.to_numeric(table.frame, errors="coerce").astype("Int64")
    table["selected_source_idx"] = pd.to_numeric(table.selected_source_idx, errors="coerce")
    table = table[table.frame.notna() & table.selected_source_idx.between(0, 2, inclusive="both")]
    lookup: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for (frame, view), rows in table.groupby(["frame", "view"]):
        lookup[(int(frame), str(view))] = rows.to_dict("records")
    return lookup


def _compose_video_frame(
    frames: dict[str, np.ndarray],
    frame_number: int,
    lookup: dict[tuple[int, str], list[dict[str, Any]]],
    *,
    panel_size: int,
    title: str,
    footer_lines: list[str],
    component_map: dict[str, str] | None = None,
    component_colors: dict[str, str] | None = None,
) -> np.ndarray:
    header = 50
    footer = 155
    canvas = np.full((header + panel_size + footer, panel_size * 3, 3), 255, dtype=np.uint8)
    _draw_text(canvas, title, (18, 31), color=_hex_to_bgr(INK), scale=0.78, thickness=2)
    for column, view in enumerate(VIEW_ORDER):
        frame = frames[view]
        height, width = frame.shape[:2]
        panel = cv2.resize(frame, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
        x_offset = column * panel_size
        canvas[header:header + panel_size, x_offset:x_offset + panel_size] = panel
        _draw_text(canvas, view.replace("_cam", ""), (x_offset + 12, header + 25), color=(255, 255, 255), scale=0.62, thickness=2)
        cv2.rectangle(canvas, (x_offset, header), (x_offset + panel_size - 1, header + panel_size - 1), _hex_to_bgr(GRID), 1)
        scale_x = panel_size / max(1, width)
        scale_y = panel_size / max(1, height)
        for row in lookup.get((frame_number, view), []):
            x = row.get("x")
            y = row.get("y")
            if x is None or y is None or not np.isfinite(float(x)) or not np.isfinite(float(y)):
                continue
            source = int(float(row["selected_source_idx"]))
            node = str(row.get("tracklet_node", ""))
            component = component_map.get(node) if component_map else None
            if component_colors is not None:
                if component not in component_colors:
                    color = _hex_to_bgr(GREY)
                    radius = 4
                    thickness = 1
                    label = ""
                else:
                    color = _hex_to_bgr(component_colors[component])
                    radius = 10
                    thickness = 3
                    label = component_colors[component + "__label"] if component + "__label" in component_colors else ""
            else:
                color = _hex_to_bgr(LOCAL_COLORS[source])
                radius = 9
                thickness = 3
                label = f"S{source}"
            point = (x_offset + int(float(x) * scale_x), header + int(float(y) * scale_y))
            cv2.circle(canvas, point, radius, color, thickness, cv2.LINE_AA)
            if label:
                label_scale = 0.43
                (label_width, _label_height), _baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_DUPLEX,
                    label_scale,
                    2,
                )
                panel_right = x_offset + panel_size - 8
                label_x = point[0] + 8
                if label_x + label_width > panel_right:
                    label_x = max(x_offset + 8, point[0] - label_width - 8)
                _draw_text(
                    canvas,
                    label,
                    (label_x, point[1] - 8),
                    color=color,
                    scale=label_scale,
                    thickness=2,
                )
    footer_y = header + panel_size + 28
    _draw_text(canvas, f"source frame {frame_number}", (18, footer_y), color=_hex_to_bgr(MUTED), scale=0.55)
    y = footer_y + 28
    for line in footer_lines:
        _draw_text(canvas, line, (18, y), color=_hex_to_bgr(INK), scale=0.54, thickness=1)
        y += 26
    return canvas


def _render_clip(
    *,
    output: Path,
    poster: Path,
    video_paths: dict[str, Path],
    dispositions: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    output_fps: float | None,
    title: str,
    footer_factory: Any,
    component_map: dict[str, str] | None = None,
    component_colors: dict[str, str] | None = None,
) -> dict[str, Any]:
    captures, metadata = _open_videos(video_paths, start_frame)
    source_fps = float(np.median([metadata[view]["fps"] for view in VIEW_ORDER]))
    fps = float(output_fps if output_fps is not None else source_fps / frame_step)
    panel_size = 420
    frame_height = 50 + panel_size + 155
    frame_width = panel_size * 3
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    if not writer.isOpened():
        _release(captures.values())
        raise RuntimeError(f"Could not create video writer: {output}")
    lookup = _disposition_lookup(dispositions)
    output_frames = 0
    poster_written = False
    try:
        for frame_number in range(start_frame, end_frame + 1):
            frames: dict[str, np.ndarray] = {}
            for view, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read {view} frame {frame_number}")
                frames[view] = frame
            if (frame_number - start_frame) % frame_step:
                continue
            footer = footer_factory(frame_number, source_fps)
            canvas = _compose_video_frame(
                frames,
                frame_number,
                lookup,
                panel_size=panel_size,
                title=title,
                footer_lines=footer,
                component_map=component_map,
                component_colors=component_colors,
            )
            writer.write(canvas)
            output_frames += 1
            if not poster_written and frame_number >= (start_frame + end_frame) // 2:
                cv2.imwrite(str(poster), canvas)
                poster_written = True
    finally:
        writer.release()
        _release(captures.values())
    return {
        "output": str(output),
        "poster": str(poster),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_step": frame_step,
        "source_fps": source_fps,
        "output_fps": fps,
        "output_frames": output_frames,
        "playback_seconds": output_frames / fps,
        "video_sources": metadata,
    }


def build_videos(run_root: Path, output_dir: Path, video_paths: dict[str, Path]) -> dict[str, Any]:
    experiment_root = run_root / "20251225_K_1_2_FCMC"
    inputs = experiment_root / "global_inputs_test_v2"
    dispositions = pd.read_csv(inputs / "detection_dispositions.csv")
    assignments = pd.read_csv(inputs / "tracklet_assignments.csv")
    component_map = assignments.set_index("node_id").component_id.astype(str).to_dict()
    source_supported = "test_78000_78999::side1_cam:172"
    source_baseline = "test_78000_78999::side1_cam:183"
    target = "test_78000_78999::side2_cam:265"

    video_dir = output_dir / "videos"
    clip1 = _render_clip(
        output=video_dir / "CLIP_1_SYNCHRONIZED_LOCAL_SOURCES.mp4",
        poster=video_dir / "CLIP_1_POSTER.png",
        video_paths=video_paths,
        dispositions=dispositions,
        start_frame=78000,
        end_frame=78999,
        frame_step=5,
        output_fps=None,
        title="Held-out real sequence: synchronized local geometry sources",
        footer_factory=lambda frame, fps: [
            "Colored rings S0/S1/S2 are frame-local geometry sources, not biological fish names.",
            f"Approx. real-time playback - {(frame - 78000) / fps:4.1f} s into the 1,000-frame test window",
            "Watch how viewpoint, occlusion, and body pose change the apparent color/shape of each source.",
        ],
    )

    component_colors = {
        source_supported: GOLD,
        source_baseline: PINK,
        target: BLUE,
        source_supported + "__label": "C172 app. predecessor",
        source_baseline + "__label": "C183 geom. predecessor",
        target + "__label": "C265 target",
    }

    def linker_footer(frame: int, fps: float) -> list[str]:
        if frame <= 78334:
            phase = "Before gap: two predecessor hypotheses are visible."
        elif frame <= 78341:
            phase = "Seven-frame gap: predecessor fragments have ended; target has not started."
        else:
            phase = "After gap: target C265 begins."
        return [
            phase,
            "Geometry only: C183 -> C265   |   + appearance: C172 -> C265   |   P(same)=99.35%",
            f"Slow motion ({fps / (2 * 30):.2f}x slower than real time); no trusted biological anchor scores the choice.",
        ]

    clip2 = _render_clip(
        output=video_dir / "CLIP_2_APPEARANCE_LINKER_CASE.mp4",
        poster=video_dir / "CLIP_2_POSTER.png",
        video_paths=video_paths,
        dispositions=dispositions,
        start_frame=78270,
        end_frame=78470,
        frame_step=2,
        output_fps=30.0,
        title="Case study: appearance changes one global predecessor link",
        footer_factory=linker_footer,
        component_map=component_map,
        component_colors=component_colors,
    )
    manifest = {
        "schema_version": "geometry_contrastive_communication_videos_v1",
        "identity_semantics": "local geometry sources and fragment components only; no biological fish names",
        "clips": [clip1, clip2],
        "linker_case": {
            "geometry_only_edge": [source_baseline, target],
            "appearance_edge": [source_supported, target],
            "appearance_same_probability": 0.99347,
            "biological_ground_truth_available": False,
        },
    }
    (video_dir / "VIDEO_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("figures", "videos", "all"), default="all")
    parser.add_argument(
        "--view-video",
        nargs=2,
        action="append",
        metavar=("VIEW", "PATH"),
        help="Required for videos; repeat for top_cam, side1_cam, side2_cam.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result: dict[str, Any] = {}
    if args.mode in {"figures", "all"}:
        result["figures"] = build_figures(args.run_root, args.output_dir)
    if args.mode in {"videos", "all"}:
        videos = {str(view): Path(path) for view, path in (args.view_video or [])}
        missing = set(VIEW_ORDER) - set(videos)
        if missing:
            raise ValueError(f"Missing --view-video entries: {sorted(missing)}")
        result["videos"] = build_videos(args.run_root, args.output_dir, videos)
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
