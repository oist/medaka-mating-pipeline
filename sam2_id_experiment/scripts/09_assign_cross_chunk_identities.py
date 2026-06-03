from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.id_model.assign import assign_tracklet_identities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign target chunk tracklets to reference pseudo-identities by embedding distance."
    )
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--reference-inference", required=True, help="Reference chunk id_inference directory.")
    parser.add_argument("--target-inference", default=None, help="Target chunk id_inference directory.")
    parser.add_argument("--max-distance", type=float, default=0.35)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--identity-prefix", default="pseudo_ID_")
    parser.add_argument(
        "--many-to-one",
        action="store_true",
        help="Allow multiple target fragments to map to the same reference identity.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.target_config)
    target_inference = Path(args.target_inference) if args.target_inference else ensure_output_dir(cfg) / "id_inference"
    reference_inference = Path(args.reference_inference)

    result = assign_tracklet_identities(
        reference_embeddings_path=reference_inference / "tracklet_embeddings.npy",
        reference_assignments_path=reference_inference / "tracklet_identity_assignments.parquet",
        target_embeddings_path=target_inference / "tracklet_embeddings.npy",
        target_assignments_path=target_inference / "tracklet_identity_assignments.parquet",
        output_assignments_path=target_inference / "tracklet_identity_assignments.parquet",
        distance_matrix_path=target_inference / "cross_chunk_distance_matrix.csv",
        summary_path=target_inference / "cross_chunk_assignment_summary.json",
        max_distance=float(args.max_distance),
        min_margin=float(args.min_margin),
        identity_prefix=str(args.identity_prefix),
        one_to_one=not bool(args.many_to_one),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
