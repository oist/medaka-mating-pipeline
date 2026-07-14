from __future__ import annotations

"""Plot structural global-fragment benchmark summaries."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = args.labels or [path.parent.name for path in args.summaries]
    if len(labels) != len(args.summaries):
        raise ValueError("--labels must match the number of summaries")
    rows = []
    for label, path in zip(labels, args.summaries):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"window": label, **payload})
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "global_stitching_benchmark_comparison.csv", index=False)

    x = np.arange(len(table))
    blue, gold, charcoal = "#2F6B9A", "#C89B3C", "#2E3338"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    width = 0.34
    axes[0].bar(
        x - width / 2, table.input_components, width,
        color=blue, edgecolor=charcoal, label="Input components",
    )
    axes[0].bar(
        x + width / 2, table.output_path_components, width,
        color=gold, edgecolor=charcoal, label="Path-cover components",
    )
    axes[0].set_title("Fragment components before and after path cover")
    axes[0].set_ylabel("Component count")
    axes[0].set_xticks(x, table.window)
    axes[0].legend(frameon=False)

    axes[1].bar(
        x - width / 2, table.feasible_edges, width,
        color=blue, edgecolor=charcoal, label="Feasible candidate edges",
    )
    axes[1].bar(
        x + width / 2, table.selected_edges, width,
        color=gold, edgecolor=charcoal, label="Selected edges",
    )
    axes[1].set_title("Global edge candidates and selected links")
    axes[1].set_ylabel("Edge count")
    axes[1].set_xticks(x, table.window)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Global fragment stitching structural benchmark",
        fontsize=14,
        color=charcoal,
    )
    fig.text(
        0.5, 0.025,
        "Development and held-out 5,000-frame windows; no independent biological anchors. "
        "Component reduction is not identity accuracy.",
        ha="center", va="bottom", fontsize=9, color="#59636E",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.20, wspace=0.28)
    fig.savefig(args.output_dir / "global_stitching_structural_overview.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
