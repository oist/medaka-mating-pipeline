from __future__ import annotations

"""Extract an inspectable idtracker.ai learning curve and final session summary."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROW = re.compile(
    r"(?P<batch>\d+)\s+\|\s*(?P<batches_per_second>[0-9.]+)\s+"
    r"\|\s*(?P<silhouette>[0-9.]+)(?P<improved>!)?\s+\|\s*"
    r"(?P<positive_loss>[0-9.]+)%\s+\|\s*(?P<negative_loss>[0-9.]+)%"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, help="training log; defaults to session/idtrackerai.log")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-silhouette", type=float, default=0.91)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    log_path = args.log or (args.session_dir / "idtrackerai.log")
    session_path = args.session_dir / "session.json"
    if not log_path.exists() or not session_path.exists():
        raise FileNotFoundError("completed session must contain idtrackerai.log and session.json")
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROW.search(line)
        if not match:
            continue
        rows.append(
            {
                "batch": int(match.group("batch")),
                "batches_per_second": float(match.group("batches_per_second")),
                "silhouette": float(match.group("silhouette")),
                "new_best": bool(match.group("improved")),
                "positive_pairs_with_loss_fraction": float(match.group("positive_loss")) / 100,
                "negative_pairs_with_loss_fraction": float(match.group("negative_loss")) / 100,
            }
        )
    curve = pd.DataFrame(rows)
    if curve.empty:
        raise ValueError("no contrastive validation rows found in log")
    best_index = curve["silhouette"].idxmax()
    session = json.loads(session_path.read_text(encoding="utf-8"))
    original_best = float(curve.loc[best_index, "silhouette"])
    recovered_best = session.get("silhouette_score")
    recovered_best = float(recovered_best) if recovered_best is not None else float("nan")
    if recovered_best > original_best:
        best_score = recovered_best
        best_source = "checkpoint_recovery_kmeans_plus_plus"
        best_batch = None
    else:
        best_score = original_best
        best_source = "official_training_log"
        best_batch = int(curve.loc[best_index, "batch"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output_dir / "idtracker_contrastive_learning_curve.csv", index=False)
    summary = {
        "schema_version": "idtracker_ai_v6_run_summary_v1",
        "version": session.get("version"),
        "session_dir": str(args.session_dir.resolve()),
        "frames": session.get("number_of_frames"),
        "animals": session.get("number_of_animals"),
        "segmentation_error_frames": session.get("number_of_error_frames"),
        "best_validation_batch": best_batch,
        "best_silhouette": best_score,
        "best_silhouette_source": best_source,
        "best_silhouette_from_official_log": original_best,
        "target_silhouette": float(args.target_silhouette),
        "target_reached": bool(best_score >= args.target_silhouette),
        "final_session_silhouette": session.get("silhouette_score"),
        "estimated_internal_accuracy": session.get("estimated_accuracy"),
        "ratio_accumulated_images": session.get("ratio_accumulated_images"),
        "warning": "internal estimates are not independent biological identity accuracy",
    }
    (args.output_dir / "idtracker_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
