from __future__ import annotations

"""Finish an interrupted idtracker.ai v6 session from its saved best checkpoint.

idtracker.ai checkpoints only the contrastive model, not the patience counters.
This recovery therefore performs a short, explicitly recorded k-means++ fallback
phase from the exact saved checkpoint, then runs the unmodified accumulation,
assignment, and trajectory-creation APIs.  It does not repeat segmentation or
discard the learned checkpoint.
"""

import argparse
import json
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--recovery-patience", type=int, default=5)
    parser.add_argument("--target-silhouette", type=float, default=0.91)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Session
    from idtrackerai.base.postprocess import trajectories_API
    from idtrackerai.base.tracker import tracker_API
    from idtrackerai.base.tracker.contrastive import ContrastiveLearning

    session_dir = args.session_dir.resolve()
    checkpoint = session_dir / "accumulation" / "contrastive_checkpoint.pt"
    params = session_dir / "accumulation" / "model_params.json"
    if not checkpoint.exists() or not params.exists():
        raise FileNotFoundError("interrupted session checkpoint/model_params not found")
    backup = session_dir / "checkpoint_recovery_input"
    backup.mkdir(exist_ok=True)
    shutil.copy2(checkpoint, backup / checkpoint.name)
    shutil.copy2(params, backup / params.name)
    if args.training_log and args.training_log.exists():
        shutil.copy2(args.training_log, session_dir / "idtrackerai_training_interrupted.log")

    session = Session.load(session_dir)
    list_of_blobs = ListOfBlobs.load(session.blobs_path)
    list_of_fragments = ListOfFragments.load(session.fragments_path)
    list_of_global_fragments = ListOfGlobalFragments.load(
        session.global_fragments_path, list_of_fragments.fragments
    )
    session.knowledge_transfer_folder = backup

    original_train = ContrastiveLearning.train

    def recovery_train(self, *unused_args, **unused_kwargs):
        # The interrupted official run had already exhausted most patience steps
        # with Global-Fragment KMeans initialization. Its documented fallback is
        # k-means++, represented by a null gfrag_loader.
        self.gfrag_loader = None
        return original_train(
            self,
            skipped_validations=0,
            patience=int(args.recovery_patience),
            target_silhouette_score=float(args.target_silhouette),
        )

    ContrastiveLearning.train = recovery_train
    try:
        with session.new_timer("Checkpoint recovery tracking"):
            identifier_model = tracker_API(
                session,
                list_of_blobs,
                list_of_fragments,
                list_of_global_fragments,
            )
        session.save()
        list_of_blobs.save(session.blobs_path)
        list_of_fragments.save(session.fragments_path)
        list_of_global_fragments.save(session.global_fragments_path)
        with session.new_timer("Checkpoint recovery trajectories"):
            trajectories_API(session, list_of_blobs, list_of_fragments, identifier_model)
    finally:
        ContrastiveLearning.train = original_train

    session.knowledge_transfer_folder = None
    session.save()
    list_of_blobs.save(session.blobs_path)
    list_of_fragments.save(session.fragments_path)
    list_of_global_fragments.save(session.global_fragments_path)
    trajectories = session_dir / "trajectories" / "trajectories_tidy.csv"
    report = {
        "schema_version": "idtracker_ai_v6_checkpoint_recovery_v1",
        "session_dir": str(session_dir),
        "input_checkpoint": str((backup / checkpoint.name).resolve()),
        "reason": "outer shell timeout after 11,400 official contrastive batches",
        "recovery_phase": "k-means++ initialization with short explicit patience",
        "recovery_patience": int(args.recovery_patience),
        "target_silhouette": float(args.target_silhouette),
        "recovered_session_silhouette": session.silhouette_score,
        "estimated_internal_accuracy": session.estimated_accuracy,
        "ratio_accumulated_images": getattr(session, "ratio_accumulated_images", None),
        "trajectories_tidy": str(trajectories.resolve()),
        "trajectories_exist": trajectories.exists(),
        "warning": "internal accuracy is not independent biological identity accuracy",
    }
    (session_dir / "checkpoint_recovery_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()

