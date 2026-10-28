"""Export the human-reviewed 3D tracks of one review folder.

Applies the ledger to ``review_data.npz``: first the geometry edits
(per-view fixes, flagged bodies, solved close intervals - recomputed from
``detections.npz`` with the calibration, so run it on the desktop), then
names, swaps, head/tail flips, flags, analysis start and watched coverage::

    python -m scripts.review.export_reviewed <review_root>/<exp>/<round_tag>

Writes into the folder (or ``--out-dir``):

* ``reviewed_3d.csv`` - one row per frame and corrected fish: ``label``
  (F / M1 / M2), ``source_track``, the fish position ``X/Y/Z`` (= the pectoral node; the 3D
  reconstruction's centroid is kept as ``centroid_X/Y/Z``), every body node in 3D
  (``<node>_X/Y/Z``, completed where fewer than two views saw it) with
  ``<node>_source`` (0 triangulated, 1 one view + body model, 2 body model
  only, 3 one-/two-frame spike replaced, 4 reviewer-flagged body re-estimated,
  -1 missing), and the flags ``is_interpolated``, ``head_flipped_by_human``, ``uncertain``,
  ``bad3d``, ``nodes_bad``, ``watched``, ``analysed``.
  The machine post-pass (:mod:`scripts.review.postprocess`) runs on every corrected fish after the
  reviewer's edits; ``--no-postprocess`` leaves it out.
* ``postprocess_report.csv`` - every run the post-pass replaced, and the ``jerk`` runs it left.
* ``versions.npz`` - initial / machine / manual / final nodes per fish for the comparison GUI
  (:mod:`scripts.review.versions`, :mod:`scripts.review.compare_gui`); ``--no-versions`` skips it.
* ``reviewed_summary.json`` - watched fraction of the analysed range,
  corrections vs. machine risk windows, and readiness checks (analysis start
  marked, every fish named, everything watched).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.review.geometry import GEOMETRY_KINDS, Detections, Geometry, apply_geometry, effective_intervals
from scripts.review.ledger import (
    COVERAGE_COLUMNS,
    EVENT_COLUMNS,
    analysis_start,
    apply_events,
    compare_with_machine,
    read_csv,
    static_labels,
)
from scripts.review.postprocess import NODE_BRIDGED, PostParams, postprocess_table
from scripts.review.versions import VERSIONS_FILE, initial_tracks, write_versions

EXPECTED_LABELS = ("F", "M1", "M2")


def export(folder: Path, ledger_dir: Path | None = None, out_dir: Path | None = None, log=None, *,
           postprocess: bool = True, versions: bool = True, post_params: PostParams | None = None) -> dict:
    folder = Path(folder)
    ledger_dir = Path(ledger_dir) if ledger_dir else folder
    out_dir = Path(out_dir) if out_dir else folder
    with np.load(folder / "review_data.npz") as z:
        data = {k: z[k] for k in z.files}
    manifest = json.loads((folder / "review_manifest.json").read_text(encoding="utf-8"))
    events = read_csv(ledger_dir / "human_events.csv", EVENT_COLUMNS)
    coverage = read_csv(ledger_dir / "coverage.csv", COVERAGE_COLUMNS)
    risks = pd.read_csv(folder / "machine_risk.csv")
    fish_ids = [str(f) for f in data["fish_ids"]]
    first, last = int(data["frames"][0]), int(data["frames"][-1])

    # Geometry edits (per-view fixes, flagged bodies, solved intervals) need the calibration and detections.
    machine_iv = folder / "machine_intervals.csv"
    intervals = effective_intervals(pd.read_csv(machine_iv) if machine_iv.exists() else None, events)
    geometry_note = "no geometry edits"
    geometry_ok = True
    recomputed: list = []
    machine_data = data
    cal, cal_error = None, None
    try:
        from scripts.reconstruction_3d.calibration import Calibration

        cal = Calibration(str(manifest["sources"]["calibration_dir"]), backend="auto")
    except Exception as exc:  # noqa: BLE001 - a laptop without the calibration still exports
        cal_error = exc
    if events["kind"].isin(list(GEOMETRY_KINDS) + ["interval", "confirm"]).any():
        geometry_ok = False
        try:
            if cal is None:
                raise cal_error
            geo = Geometry.from_manifest(data, Detections(folder), cal, manifest)
            data, recomputed = apply_geometry(geo, events, ledger_dir, intervals, log=log)
            geometry_note = f"applied to {len(recomputed)} range(s)"
            geometry_ok = True
        except Exception as exc:  # noqa: BLE001 - reported in the summary, export continues
            geometry_note = f"NOT applied ({exc!r}) - run the export on the desktop with the calibration"

    node_names = [str(n) for n in data["node_names"]]
    manual = apply_events(data, events, coverage)
    table, post_report = postprocess_table(manual, node_names, p=post_params, clean=postprocess)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "reviewed_3d.csv"
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    post_report.to_csv(out_dir / "postprocess_report.csv", index=False)

    labels = static_labels(events, fish_ids)
    start = analysis_start(events)
    comparison = compare_with_machine(events, risks, coverage, first, last)
    analysed = table[table["analysed"]]
    versions_note: object = "skipped"
    if versions:
        try:
            initial = initial_tracks(folder, cal, log=log) if cal is not None else None
            info = write_versions(out_dir / VERSIONS_FILE, machine_data, initial, manual, table, labels, start)
            versions_note = info["versions"] + ([] if initial is not None else ["(no calibration: no initial)"])
        except Exception as exc:  # noqa: BLE001 - the table is already written
            versions_note = f"NOT written ({exc!r})"
    in_range = post_report[post_report["frame_end"] >= (start or 0)]
    post_summary = {
        "enabled": postprocess,
        "params": asdict(post_params or PostParams()),
        "position": "pectoral node (centroid_X/Y/Z = 3D reconstruction centroid)",
        "bridged_fraction_of_analysed_rows": {
            name: round(float((analysed[f"{name}_source"] == NODE_BRIDGED).mean()), 6) if len(analysed) else None
            for name in node_names},
        "runs_after_analysis_start": {
            f"{node}/{kind}/{'replaced' if rep else 'left'}": int(n)
            for (node, kind, rep), n in in_range.groupby(["node", "kind", "repaired"]).size().items()},
    }
    checks = {
        "analysis_start_marked": start is not None,
        "all_fish_named": sorted(v for v in labels.values() if v in EXPECTED_LABELS) == sorted(EXPECTED_LABELS),
        "fully_watched": comparison["watched_fraction"] >= 0.999,
        "geometry_edits_applied": geometry_ok,
    }
    summary = {
        "experiment": manifest.get("experiment"), "round_tag": manifest.get("round_tag"),
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_version": manifest.get("data_version"),
        "analysis_start": start, "labels": labels,
        "ready_for_analysis": all(checks.values()), "checks": checks,
        "flagged_fraction_of_analysed_rows": {
            k: round(float(analysed[k].mean()), 5) if len(analysed) else None
            for k in ("is_interpolated", "uncertain", "bad3d", "nodes_bad")},
        "node_source_fraction_of_analysed_rows": {
            name: {kind: round(float((analysed[f"{name}_source"] == code).mean()), 6) if len(analysed) else None
                   for kind, code in (("triangulated", 0), ("one_view", 1), ("body_model", 2), ("despiked", 3),
                                     ("reestimated", 4), ("bridged", NODE_BRIDGED), ("missing", -1))}
            for name in (str(n) for n in data["node_names"])},
        "comparison_with_machine": comparison,
        "geometry_edits": geometry_note,
        "recomputed_ranges": [list(map(int, r)) for r in recomputed],
        "intervals": int(len(intervals)),
        "postprocess": post_summary,
        "versions": versions_note,
        "files": {"table": str(csv_path), "postprocess_report": str(out_dir / "postprocess_report.csv"),
                  "versions": str(out_dir / VERSIONS_FILE) if versions else None},
    }
    (out_dir / "reviewed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="<review_root>/<exp>/<round_tag>")
    p.add_argument("--ledger-dir", type=Path, default=None, help="Ledger location if not the folder.")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: the folder.")
    p.add_argument("--no-postprocess", action="store_true", help="Leave out the machine post-pass.")
    p.add_argument("--no-versions", action="store_true", help="Do not write versions.npz.")
    args = p.parse_args(argv)
    summary = export(args.folder, args.ledger_dir, args.out_dir, log=print, postprocess=not args.no_postprocess,
                     versions=not args.no_versions)
    print(json.dumps({k: summary[k] for k in ("ready_for_analysis", "checks", "labels", "analysis_start")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
