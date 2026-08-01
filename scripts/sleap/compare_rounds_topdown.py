from __future__ import annotations

"""
Common-manifest cross-round top-down accuracy comparison.

Runs Round 7 and Round 8 on exactly the same Phase-1 manifest frames, loading
each row from its recorded source SLP, and reports the shared detection and
keypoint metrics overall, by experiment, and by difficulty. It supports cached
native SLP predictions and full, centroid-only, or GT-mouth-crop oracle
centered-instance inference.

Why split by experiment
------------------------
- round-4 / round-5 were trained on K_1_2 ONLY.
- round-7 added K_1_1 + K_1_3 (rounds-6 annotation) + difficult K_1_2 frames.

So on K_1_2 the comparison is a regression check (did adding other-line data hurt
the original line?), and on K_1_1 / K_1_3 it is a generalization test (round-4/5
never saw those experiments -> out-of-sample for them, in-sample for round-7).

There is NO separate round-6 model: round-6 was an annotation round whose frames
were merged into round-7.

Without ``--manifest`` this retains the historical deterministic Round-7
merged-label stress mode. That fallback is in-sample and is not a held-out
generalization benchmark.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths

# Reuse the eval harness's inference + metric machinery.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_topdown_vs_bottomup as E  # noqa: E402
from merge_cumulative_labels import (  # noqa: E402
    _experiment_of,
    _leaf_map,
    _norm_clip,
)
from split_pkg_by_view import _video_dataset_index  # noqa: E402

# Per round: (directory below sleap_models_dir, run-name tag).
ROUND_LAYOUT = {
    "round4": ("retrain_20260508", "round4_260520"),
    "round5": ("retrain_20260611", "round5_260611"),
    "round7": ("retrain_20260704", "round7_260704"),
    # Round 8's production evaluation pair deliberately uses the strict-mouth
    # run names. Keep this exact tag so discovery cannot silently select an
    # earlier exploratory round8 checkpoint.
    "round8": (
        "retrain_20260722_round8",
        "round8_260723_strictmouth",
    ),
}

METRIC_KEYS = [
    ("detect_rate", "detect"),
    ("miss_rate", "miss"),
    ("merge_rate", "merge"),
    ("swap_rate", "swap"),
    ("mouth_err_p95", "mouth_p95"),
    ("tail_err_p95", "tail_p95"),
    ("oks_mean", "OKS"),
]


def resolve_topdown_dirs(base: Path, view: str, tag: str) -> tuple[Path, Path]:
    """Glob the centroid + centered_instance run dirs for one round/view."""
    runs = base / view / "runs"
    cents = sorted(runs.glob(f"{tag}.centroid*"))
    cis = sorted(runs.glob(f"{tag}.centered_instance*"))
    if not cents or not cis:
        raise SystemExit(
            f"missing top-down runs for {tag} under {runs} "
            f"(centroid={[c.name for c in cents]}, centered={[c.name for c in cis]})"
        )
    for d in (cents[-1], cis[-1]):
        if not (d / "best.ckpt").exists():
            raise SystemExit(f"{d} has no best.ckpt")
    return cents[-1], cis[-1]


def resolve_round_specs(
    *,
    paths_config: Path,
    round_dir_overrides: list[str],
) -> dict[str, tuple[Path, str]]:
    """Resolve round roots from local config plus optional ``round=PATH`` values."""
    local = load_local_paths(paths_config)
    model_root = local.sleap_models_dir if local is not None else None
    specs: dict[str, tuple[Path, str]] = {}
    if model_root is not None:
        specs.update(
            {
                round_name: (model_root / relative_dir, tag)
                for round_name, (relative_dir, tag) in ROUND_LAYOUT.items()
            }
        )
    for value in round_dir_overrides:
        round_name, separator, raw_path = value.partition("=")
        if not separator or round_name not in ROUND_LAYOUT or not raw_path.strip():
            raise ValueError(
                "--round-dir must be ROUND=PATH where ROUND is one of "
                f"{sorted(ROUND_LAYOUT)}; got {value!r}"
            )
        specs[round_name] = (
            Path(raw_path.strip()),
            ROUND_LAYOUT[round_name][1],
        )
    return specs


def parse_model_specs(
    values: list[str], evaluation_mode: str
) -> dict[str, list[Path]]:
    """Parse repeatable ``NAME=PATH[,PATH]`` candidate model specifications."""
    expected_paths = 2 if evaluation_mode == "full_topdown" else 1
    parsed: dict[str, list[Path]] = {}
    for value in values:
        name, separator, raw_paths = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw_paths.strip():
            raise ValueError(
                "--model must be NAME=PATH for component evaluation or "
                "NAME=CENTROID_PATH,CENTERED_INSTANCE_PATH for full top-down; "
                f"got {value!r}"
            )
        if name in parsed:
            raise ValueError(f"duplicate --model name: {name!r}")
        paths = [
            Path(raw_path.strip())
            for raw_path in raw_paths.split(",")
            if raw_path.strip()
        ]
        if len(paths) != expected_paths:
            raise ValueError(
                f"--model {name!r} requires {expected_paths} path(s) for "
                f"{evaluation_mode}, got {len(paths)}"
            )
        for path in paths:
            if not path.is_dir():
                raise ValueError(f"model directory not found: {path}")
            if not (path / "best.ckpt").is_file():
                raise ValueError(f"model directory has no best.ckpt: {path}")
        parsed[name] = paths
    return parsed


def safe_cache_name(value: str) -> str:
    """Return a stable filesystem-safe candidate label."""
    cleaned = "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in value
    ).strip("_")
    if not cleaned:
        raise ValueError(f"candidate name has no filesystem-safe characters: {value!r}")
    return cleaned


def manifest_bool(value: object) -> bool:
    """Parse common CSV boolean spellings without treating non-empty text as true."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def offset_oracle_mouth_anchors(
    labels,
    *,
    mouth_idx: int,
    dx: float,
    dy: float,
):
    """Clone labels with GT mouth crop anchors shifted by a fixed offset.

    The returned labels are used only as centered-instance inference inputs.
    Metrics must still be computed against the original, unshifted labels.
    """
    import sleap_io as sio

    offset = np.asarray([dx, dy], dtype=float)
    if not np.isfinite(offset).all():
        raise ValueError(f"oracle anchor offset must be finite, got {offset}")

    labeled_frames = []
    for labeled_frame in labels.labeled_frames:
        instances = []
        for instance_index, instance in enumerate(
            labeled_frame.user_instances
        ):
            points = np.asarray(instance.numpy(), dtype=float)
            if (
                points.ndim != 2
                or mouth_idx < 0
                or mouth_idx >= len(points)
                or not np.isfinite(points[mouth_idx]).all()
            ):
                raise ValueError(
                    "oracle offset stress requires every mouth anchor to be "
                    "finite; failed at frame "
                    f"{int(labeled_frame.frame_idx)}, instance {instance_index}"
                )
            shifted = np.full((len(points), 3), np.nan, dtype=float)
            shifted[:, :2] = points
            shifted[:, 2] = np.isfinite(points).all(axis=1).astype(float)
            shifted[mouth_idx, :2] += offset
            instances.append(
                sio.Instance.from_numpy(
                    points_data=shifted,
                    skeleton=instance.skeleton,
                    track=getattr(instance, "track", None),
                )
            )
        labeled_frames.append(
            sio.LabeledFrame(
                video=labeled_frame.video,
                frame_idx=int(labeled_frame.frame_idx),
                instances=instances,
                is_negative=bool(
                    getattr(labeled_frame, "is_negative", False)
                ),
            )
        )

    provenance = dict(getattr(labels, "provenance", {}) or {})
    provenance["oracle_anchor_offset_px"] = [float(dx), float(dy)]
    return sio.Labels(
        labeled_frames=labeled_frames,
        videos=list(labels.videos),
        skeletons=list(labels.skeletons),
        tracks=list(getattr(labels, "tracks", [])),
        provenance=provenance,
    )


def experiment_tags(gt_path: Path, labels) -> dict[int, str]:
    """Map each labeled-frame POSITION -> experiment (k_1_1 / k_1_2 / k_1_3)."""
    leaf = _leaf_map(gt_path)
    out: dict[int, str] = {}
    for i, lf in enumerate(labels.labeled_frames):
        ds = _video_dataset_index(lf.video)
        out[i] = _experiment_of(_norm_clip(leaf.get(ds) if ds is not None else None))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True, choices=["top", "side"])
    ap.add_argument("--rounds", nargs="+", default=["round7", "round8"],
                    choices=list(ROUND_LAYOUT))
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH[,PATH]",
        help=(
            "Evaluate an explicitly named candidate instead of --rounds. "
            "Repeat for multiple candidates. Component modes require one model "
            "directory; full_topdown requires centroid,centered-instance paths."
        ),
    )
    ap.add_argument(
        "--paths-config",
        type=Path,
        default=DEFAULT_PATHS_CONFIG,
        help="TOML containing paths.sleap_models_dir.",
    )
    ap.add_argument(
        "--round-dir",
        action="append",
        default=[],
        metavar="ROUND=PATH",
        help=(
            "Override one round root; repeat for multiple rounds. Useful when "
            "the paths config mount is unavailable."
        ),
    )
    ap.add_argument("--difficult-only", action="store_true")
    ap.add_argument("--close-contact-px", type=float, default=80.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--match-gate-px", type=float, default=150.0)
    ap.add_argument("--max-instances", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--evaluation-mode",
        choices=[
            "full_topdown",
            "centroid_only",
            "oracle_centered_instance",
        ],
        default="full_topdown",
        help=(
            "Full two-stage inference, detector-only output, or the "
            "centered-instance model on GT mouth-centered crops."
        ),
    )
    ap.add_argument(
        "--oracle-anchor-offset-px",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("DX", "DY"),
        help=(
            "Shift every GT mouth crop anchor by fixed image-pixel offsets "
            "before oracle centered-instance inference. Metrics remain against "
            "the unshifted GT. Requires --evaluation-mode "
            "oracle_centered_instance."
        ),
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--manifest",
        type=Path,
        help="Phase-1 benchmark CSV. Uses its ordered source_labeled_frame_index rows.",
    )
    ap.add_argument(
        "--gt-slp",
        type=Path,
        help=(
            "Override the manifest source SLP (or legacy common GT). With a "
            "manifest, all filtered rows must name this same source."
        ),
    )
    ap.add_argument(
        "--cohort",
        nargs="+",
        help="Only evaluate manifest rows whose cohort is one of these values.",
    )
    ap.add_argument(
        "--split-role",
        nargs="+",
        help="Only evaluate manifest rows whose split_role is one of these values.",
    )
    ap.add_argument(
        "--strict-centroid-eligible-only",
        action="store_true",
        help=(
            "Keep only manifest frames where every GT mouth anchor is finite. "
            "Recommended for centroid-only and mouth-anchored oracle-crop modes."
        ),
    )
    ap.add_argument(
        "--predictions-dir",
        type=Path,
        help=(
            "Cache/reuse selection-hashed native prediction SLP artifacts here."
        ),
    )
    ap.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Load cached predictions instead of running inference.",
    )
    ap.add_argument(
        "--reuse-existing-predictions",
        action="store_true",
        help=(
            "Reuse each matching prediction cache when present and run "
            "inference only for missing candidate caches."
        ),
    )
    args = ap.parse_args(argv)
    if args.reuse_predictions and args.reuse_existing_predictions:
        ap.error(
            "--reuse-predictions and --reuse-existing-predictions are "
            "mutually exclusive"
        )
    oracle_anchor_offset = tuple(
        float(value) for value in args.oracle_anchor_offset_px
    )
    if not np.isfinite(oracle_anchor_offset).all():
        ap.error("--oracle-anchor-offset-px values must be finite")
    if (
        oracle_anchor_offset != (0.0, 0.0)
        and args.evaluation_mode != "oracle_centered_instance"
    ):
        ap.error(
            "--oracle-anchor-offset-px requires --evaluation-mode "
            "oracle_centered_instance"
        )

    import sleap_io as sio

    round_specs: dict[str, tuple[Path, str]] = {}
    explicit_models: dict[str, list[Path]] = {}
    try:
        if args.model:
            if args.round_dir:
                raise ValueError("--round-dir cannot be combined with --model")
            explicit_models = parse_model_specs(
                args.model, args.evaluation_mode
            )
        else:
            round_specs = resolve_round_specs(
                paths_config=args.paths_config,
                round_dir_overrides=args.round_dir,
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if explicit_models:
        evaluation_specs = explicit_models
    else:
        missing_round_roots = [
            name for name in args.rounds if name not in round_specs
        ]
        if missing_round_roots:
            raise SystemExit(
                f"no model root for {missing_round_roots}; define "
                f"sleap_models_dir in {args.paths_config} or pass "
                "--round-dir ROUND=PATH"
            )
        evaluation_specs = {}
        for round_name in args.rounds:
            base, tag = round_specs[round_name]
            centroid, centered_instance = resolve_topdown_dirs(
                base, args.view, tag
            )
            mode_models = {
                "full_topdown": [centroid, centered_instance],
                "centroid_only": [centroid],
                "oracle_centered_instance": [centered_instance],
            }
            evaluation_specs[round_name] = mode_models[args.evaluation_mode]

    view = args.view
    manifest_rows: list[dict[str, str]]
    if args.manifest:
        manifest_rows = E.read_benchmark_manifest(args.manifest, view=view)
        if args.cohort:
            wanted = set(args.cohort)
            manifest_rows = [
                row for row in manifest_rows if row.get("cohort") in wanted
            ]
        if args.split_role:
            wanted = set(args.split_role)
            manifest_rows = [
                row for row in manifest_rows if row.get("split_role") in wanted
            ]
        if args.strict_centroid_eligible_only:
            manifest_rows = [
                row
                for row in manifest_rows
                if manifest_bool(row.get("strict_centroid_eligible"))
            ]
        if args.difficult_only:
            manifest_rows = [
                row
                for row in manifest_rows
                if E.normalize_difficulty(row) == "close_contact"
            ]
        if args.limit is not None:
            manifest_rows = manifest_rows[:args.limit]
        if not manifest_rows:
            raise SystemExit(
                f"manifest has no rows after view/cohort/split filters: "
                f"{args.manifest}"
            )

        if args.gt_slp:
            def normalized_path(value: str | Path) -> str:
                return str(value).replace("/", "\\").rstrip("\\").casefold()

            sources = {
                normalized_path(row.get("source_slp", ""))
                for row in manifest_rows
                if row.get("source_slp")
            }
            wanted_source = normalized_path(args.gt_slp)
            if len(sources) != 1:
                raise SystemExit(
                    "--gt-slp requires the filtered manifest to contain exactly "
                    f"one source_slp; found {len(sources)}. Use --cohort or "
                    "--split-role, or omit --gt-slp for automatic multi-source loading."
                )
            if sources != {wanted_source}:
                raise SystemExit(
                    f"--gt-slp {args.gt_slp} does not match the filtered "
                    f"manifest source_slp: {next(iter(sources))}"
                )
            if not args.gt_slp.exists():
                raise SystemExit(f"GT not found: {args.gt_slp}")
            print(f"loading filtered manifest GT: {args.gt_slp}")
            labels = sio.load_slp(str(args.gt_slp))
            try:
                eval_positions, manifest_rows = E.manifest_positions(
                    labels, manifest_rows
                )
            except ValueError as exc:
                raise SystemExit(
                    f"invalid common benchmark manifest: {exc}"
                ) from exc
            gt_sources = [str(args.gt_slp)]
        else:
            print("loading GT from each manifest row's source_slp")
            try:
                labels = E.labels_from_manifest_sources(manifest_rows)
            except ValueError as exc:
                raise SystemExit(
                    f"invalid common benchmark manifest: {exc}"
                ) from exc
            eval_positions = list(range(len(manifest_rows)))
            gt_sources = sorted(
                {row["source_slp"] for row in manifest_rows}
            )
    else:
        gt_path = args.gt_slp or (
            round_specs["round7"][0] / view / "labels.pkg.slp"
        )
        if not gt_path.exists():
            raise SystemExit(f"GT not found: {gt_path}")
        print(f"loading common GT: {gt_path}")
        labels = sio.load_slp(str(gt_path))
        gt_sources = [str(gt_path)]
        exp_of = experiment_tags(gt_path, labels)
        vid_index = {id(v): i for i, v in enumerate(labels.videos)}

        node_names = [n.name for n in labels.skeletons[0].nodes]
        mouth_idx, tail_idx = node_names.index(E.MOUTH), node_names.index(E.TAIL)
        # Legacy stress-set mode: deterministic hash selection from round-7
        # labels. This is in-sample and must not be called a held-out benchmark.
        cand, keys = [], []
        for i, lf in enumerate(labels.labeled_frames):
            gt_pts = E._instances_as_arrays(lf.user_instances)
            if not gt_pts:
                continue
            if args.difficult_only and \
                    E.min_pairwise_centroid_distance(gt_pts) > args.close_contact_px:
                continue
            cand.append(i)
            keys.append((vid_index.get(id(lf.video), -1), int(lf.frame_idx)))
        sel = E.select_eval_indices(keys, args.limit)
        eval_positions = [cand[i] for i in sel]
        manifest_rows = []
        for pos in eval_positions:
            gt_pts = E._instances_as_arrays(labels.labeled_frames[pos].user_instances)
            is_partial = any(
                not (
                    np.all(np.isfinite(points[mouth_idx]))
                    and np.all(np.isfinite(points[tail_idx]))
                )
                for points in gt_pts
            )
            close = (
                E.min_pairwise_centroid_distance(gt_pts) <= args.close_contact_px
            )
            manifest_rows.append(
                {
                    "frame_key": f"label-position:{pos}",
                    "experiment": exp_of[pos],
                    "difficulty": (
                        "partial" if is_partial else
                        "close_contact" if close else "ordinary"
                    ),
                }
            )

    node_names = [n.name for n in labels.skeletons[0].nodes]
    mouth_idx, tail_idx = node_names.index(E.MOUTH), node_names.index(E.TAIL)
    row_experiment = [
        row.get("experiment") or (
            exp_of[pos] if not args.manifest else "unknown"
        )
        for row, pos in zip(manifest_rows, eval_positions)
    ]
    row_cohort = [
        row.get("cohort") or "unknown" for row in manifest_rows
    ]
    row_camera = [
        row.get("view") or row.get("camera") or "unknown"
        for row in manifest_rows
    ]
    exps = sorted(set(row_experiment))
    cohorts = sorted(set(row_cohort))
    cameras = sorted(set(row_camera))
    from collections import Counter
    exp_hist = Counter(row_experiment)
    candidate_count = len(manifest_rows) if args.manifest else len(cand)
    print(f"candidate={candidate_count} -> eval subset={len(eval_positions)}  "
          f"by experiment: {dict(exp_hist)}")

    gt_subset_all = E.build_subset_labels(labels, eval_positions)
    inference_subset = gt_subset_all
    if oracle_anchor_offset != (0.0, 0.0):
        inference_subset = offset_oracle_mouth_anchors(
            gt_subset_all,
            mouth_idx=mouth_idx,
            dx=oracle_anchor_offset[0],
            dy=oracle_anchor_offset[1],
        )
    frame_keys = [
        row.get("frame_key") or f"label-position:{pos}"
        for row, pos in zip(manifest_rows, eval_positions)
    ]
    difficulty = [E.normalize_difficulty(row) for row in manifest_rows]
    difficulty_names = sorted(set(difficulty))
    digest_material = "\n".join(frame_keys)
    if oracle_anchor_offset != (0.0, 0.0):
        digest_material += (
            "\noracle_anchor_offset_px:"
            f"{oracle_anchor_offset[0]:.17g},"
            f"{oracle_anchor_offset[1]:.17g}"
        )
    selection_digest = hashlib.sha1(
        digest_material.encode("utf-8")
    ).hexdigest()[:10]
    if args.cohort and len(args.cohort) == 1:
        cache_scope = args.cohort[0]
    elif args.split_role and len(args.split_role) == 1:
        cache_scope = args.split_role[0]
    elif args.manifest:
        cache_scope = args.manifest.stem
    else:
        cache_scope = "legacy_stress"
    cache_scope = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in cache_scope
    )

    results: dict[str, dict] = {}
    match_rows: list[dict] = []
    evaluation_names = list(evaluation_specs)
    for rnd, model_paths in evaluation_specs.items():
        print(
            f"\n===== {rnd} / {args.evaluation_mode}: "
            f"{' + '.join(path.name for path in model_paths)} ====="
        )
        cache_model_name = rnd if not args.model else safe_cache_name(rnd)
        prediction_filename = (
            f"{cache_model_name}_{view}_{args.evaluation_mode}_"
            f"{cache_scope}_{selection_digest}.predictions.slp"
        )
        pred_path = (
            args.predictions_dir / prediction_filename
            if args.predictions_dir
            else None
        )
        if args.reuse_predictions or (
            args.reuse_existing_predictions
            and pred_path is not None
            and pred_path.exists()
        ):
            if pred_path is None or not pred_path.exists():
                raise SystemExit(
                    f"--reuse-predictions requested but cache is missing: {pred_path}"
                )
            print(f"loading cached predictions: {pred_path}")
            pred = sio.load_slp(str(pred_path))
        else:
            pred = E.run_strategy(
                strategy=f"{args.evaluation_mode}::{rnd}",
                model_paths=model_paths,
                subset_labels=inference_subset,
                max_instances=args.max_instances,
                batch_size=args.batch_size,
                device=args.device,
                output_path=pred_path,
                prediction_mode=args.evaluation_mode,
            )
        if len(pred.labeled_frames) != len(eval_positions):
            raise SystemExit(
                f"{rnd} predictions contain {len(pred.labeled_frames)} frames, "
                f"expected {len(eval_positions)} ordered manifest frames"
            )
        overall = E.evaluate_strategy(
            strategy=rnd, gt_subset_labels=gt_subset_all, pred_labels=pred,
            mouth_idx=mouth_idx, tail_idx=tail_idx, gate=args.match_gate_px,
            frame_keys=frame_keys, include_records=True,
            prediction_mode=args.evaluation_mode)
        for record in overall.get("match_records", []):
            match_rows.append(
                {
                    "round": rnd,
                    "view": view,
                    "evaluation_mode": args.evaluation_mode,
                    **record,
                }
            )

        def grouped_summary(name: str, values: list[str], wanted: str) -> dict:
            local = [i for i, value in enumerate(values) if value == wanted]
            return E.evaluate_strategy(
                strategy=f"{rnd}:{name}:{wanted}",
                gt_subset_labels=E.build_subset_labels(gt_subset_all, local),
                pred_labels=E.build_subset_labels(pred, local),
                mouth_idx=mouth_idx,
                tail_idx=tail_idx,
                gate=args.match_gate_px,
                prediction_mode=args.evaluation_mode,
            )

        per_exp = {
            exp: grouped_summary("experiment", row_experiment, exp) for exp in exps
        }
        per_cohort = {
            cohort: grouped_summary("cohort", row_cohort, cohort)
            for cohort in cohorts
        }
        per_camera = {
            camera: grouped_summary("camera", row_camera, camera)
            for camera in cameras
        }
        per_difficulty = {
            name: grouped_summary("difficulty", difficulty, name)
            for name in difficulty_names
        }
        results[rnd] = {"runs": [path.name for path in model_paths],
                        "evaluation_mode": args.evaluation_mode,
                        "prediction_path": str(pred_path) if pred_path else None,
                        "overall": overall, "per_experiment": per_exp,
                        "per_cohort": per_cohort,
                        "per_camera": per_camera,
                        "per_difficulty": per_difficulty}

    # ---- report ----
    def fmt(d):
        return "  ".join(f"{short}={d[key]:.3f}" for key, short in METRIC_KEYS)

    print("\n" + "=" * 70)
    print(f"CROSS-ROUND TOP-DOWN  view={view}  "
          f"difficult_only={args.difficult_only}  limit={args.limit}")
    print("=" * 70)
    print("\n[OVERALL]")
    for rnd in evaluation_names:
        print(f"  {rnd:8s} {fmt(results[rnd]['overall'])}")
    for exp in exps:
        seen = "K_1_2 (all rounds saw it)" if "k_1_2" in exp else \
               f"{exp} (round-7 only saw it)"
        print(f"\n[{exp}]  {seen}")
        for rnd in evaluation_names:
            print(f"  {rnd:8s} {fmt(results[rnd]['per_experiment'][exp])}")

    out = {"view": view, "difficult_only": args.difficult_only,
           "evaluation_mode": args.evaluation_mode,
           "oracle_anchor_offset_px": list(oracle_anchor_offset),
           "limit": args.limit, "n_eval_frames": len(eval_positions),
           "manifest": str(args.manifest) if args.manifest else None,
           "gt_slp_sources": gt_sources,
           "cohort_filter": args.cohort,
           "split_role_filter": args.split_role,
           "strict_centroid_eligible_only": (
               args.strict_centroid_eligible_only
           ),
           "round_roots": {
               name: str(round_specs[name][0])
               for name in evaluation_names
               if name in round_specs
           },
           "model_specs": {
               name: [str(path) for path in paths]
               for name, paths in evaluation_specs.items()
           },
           "experiments": dict(exp_hist), "results": results,
           # Canonical long-form rows for the Phase-1 visual-QC loader. The
           # same records are also written as a sibling *_matches.csv.
           "matches": match_rows}
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out_json}")
        matches_path = args.out_json.with_name(
            f"{args.out_json.stem}_matches.csv"
        )
        if match_rows:
            with matches_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(match_rows[0]))
                writer.writeheader()
                writer.writerows(match_rows)
            print(f"wrote {matches_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
