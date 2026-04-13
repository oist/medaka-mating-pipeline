"""
Visualize calibration errors across the 3D water volume.

Creates 3D, XY, XZ, and YZ scatter plots where color shows the Euclidean
calibration error in millimeters. This helps reveal whether errors are evenly
distributed or localized in part of the tank.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_error_dataframe(mapping_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(mapping_csv)
    required = ["X", "Y", "Z", "pred_X", "pred_Y", "pred_Z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{mapping_csv} is missing required columns: {missing}")

    err = df[["pred_X", "pred_Y", "pred_Z"]].to_numpy(float) - df[["X", "Y", "Z"]].to_numpy(float)
    df = df.copy()
    df["err_X"] = err[:, 0]
    df["err_Y"] = err[:, 1]
    df["err_Z"] = err[:, 2]
    df["err_mm"] = np.linalg.norm(err, axis=1)
    return df


def summarize_errors(df: pd.DataFrame) -> Dict[str, float]:
    e = df["err_mm"].to_numpy(float)
    return {
        "n_points": int(len(df)),
        "mean_error_mm": float(np.mean(e)),
        "median_error_mm": float(np.median(e)),
        "rmse_mm": float(np.sqrt(np.mean(e ** 2))),
        "std_error_mm": float(np.std(e)),
        "max_error_mm": float(np.max(e)),
    }


def plot_volume_error(df: pd.DataFrame, output_path: str | Path, *, title: str | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    e = df["err_mm"].to_numpy(float)
    vmax = max(1.0, float(np.nanpercentile(e, 95)))
    scatter_kwargs = dict(c=e, cmap="viridis", vmin=0, vmax=vmax, s=45, alpha=0.9, edgecolors="k", linewidths=0.3)

    fig = plt.figure(figsize=(15, 12))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    sc = ax3d.scatter(df["X"], df["Y"], df["Z"], **scatter_kwargs)
    ax3d.set_title("3D volume")
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_xlim(0, 190)
    ax3d.set_ylim(0, 190)
    ax3d.set_zlim(0, 190)
    ax3d.invert_yaxis()
    try:
        ax3d.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.scatter(df["X"], df["Y"], **scatter_kwargs)
    ax_xy.set_title("XY projection")
    ax_xy.set_xlabel("X (mm)")
    ax_xy.set_ylabel("Y (mm)")
    ax_xy.set_xlim(0, 190)
    ax_xy.set_ylim(0, 190)
    ax_xy.invert_yaxis()
    ax_xy.set_aspect("equal", adjustable="box")

    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.scatter(df["X"], df["Z"], **scatter_kwargs)
    ax_xz.set_title("XZ projection")
    ax_xz.set_xlabel("X (mm)")
    ax_xz.set_ylabel("Z (mm)")
    ax_xz.set_xlim(0, 190)
    ax_xz.set_ylim(0, 190)
    ax_xz.set_aspect("equal", adjustable="box")

    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.scatter(df["Y"], df["Z"], **scatter_kwargs)
    ax_yz.set_title("YZ projection")
    ax_yz.set_xlabel("Y (mm)")
    ax_yz.set_ylabel("Z (mm)")
    ax_yz.set_xlim(0, 190)
    ax_yz.set_ylim(0, 190)
    ax_yz.set_aspect("equal", adjustable="box")

    summary = summarize_errors(df)
    fig.suptitle(
        title
        or f"Calibration volume error: mean={summary['mean_error_mm']:.2f} mm, "
        f"RMSE={summary['rmse_mm']:.2f} mm, max={summary['max_error_mm']:.2f} mm",
        fontsize=13,
    )
    fig.colorbar(sc, ax=[ax3d, ax_xy, ax_xz, ax_yz], shrink=0.75, label="3D error (mm)")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize calibration error across the 3D volume")
    parser.add_argument("--mapping_csv", required=True, help="Calibration point_mapping CSV")
    parser.add_argument("--output", default=None, help="Output PNG path. Default: <mapping_stem>_volume_error.png")
    parser.add_argument("--summary_json", default=None, help="Optional output JSON path for numeric summary")
    args = parser.parse_args()

    mapping_csv = Path(args.mapping_csv)
    df = load_error_dataframe(mapping_csv)
    output = Path(args.output) if args.output else mapping_csv.with_name(f"{mapping_csv.stem}_volume_error.png")
    summary_json = Path(args.summary_json) if args.summary_json else mapping_csv.with_name(f"{mapping_csv.stem}_volume_error_summary.json")

    plot_volume_error(df, output)
    summary = summarize_errors(df)
    if "way" in df.columns:
        summary["by_way"] = {
            str(way): summarize_errors(sub)
            for way, sub in df.groupby("way")
        }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved volume error plot: {output}")
    print(f"Saved volume error summary: {summary_json}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
