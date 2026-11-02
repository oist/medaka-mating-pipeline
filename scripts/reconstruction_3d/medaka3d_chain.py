"""Run medaka-3d's own full tracking chain, unchanged, on a recording of this repo.

``medaka3d_bridge`` swaps in only medaka-3d's per-frame correspondence and keeps this repo's linker and
top-camera bind. This module runs medaka-3d's whole production chain instead (its
``docs/TRACKING_PIPELINE.md``), so the two repositories can be compared end to end::

    c27  per-camera 2D tracking (sleap_nn Tracker, uncapped, serial)
    c28  correspondence as a Viterbi over the whole recording, at every lambda of the inherited grid
    c33  ray rejection at doubted frames
    c30  slot stitch (v4 evidence ladder)
    c34  out-and-back flip repair

Two things stand between those tools and this repo's recordings. Both are bridged here; no medaka-3d file
is edited.

* The chain reads detections from ``c01``'s cache, and ``c01`` builds it only from medaka-3d's own
  K-group ``s16`` chunk files. ``build_detection_cache`` writes the same cache from this repo's merged
  per-view ``.slp`` through ``c01.read_pair`` / ``c01.save``, so every frame is still checked against its
  video path. The model is the same (s14 bottom-up, 2026-08-10 reviewed labels); this repo infers with
  ``--max_instances 6`` where s16 was uncapped, and the inference command is recorded in the provenance.
* ``c02_tracking_probe.CALIBRATION`` is an absolute path on the machine the chain was written on.
  ``run_tool`` re-roots it onto this machine's bucket mount, checks that it is byte-identical to
  ``calibration_results/fixed_refractive_192x192x186`` (the calibration ``reconstruct_from_slp`` uses), and
  runs the tool through ``runpy``. Only c28 and c33 read it; c27 (a process pool, which ``runpy`` would
  break) and c30 / c34 run as plain scripts.

CLI::

    python -m scripts.reconstruction_3d.medaka3d_chain cache 20260112_P_1_1_FCMC --slp-dir <sleap_output/exp/round>
    python -m scripts.reconstruction_3d.medaka3d_chain chain 20260112_P_1_1_FCMC --log-dir <dir>
    python -m scripts.reconstruction_3d.medaka3d_chain run c28 -- --experiment 20260112_P_1_1_FCMC
    python -m scripts.reconstruction_3d.medaka3d_chain export 20260112_P_1_1_FCMC --slp-dir <dir> \\
        --out-dir <reconstruction_3d/exp/tag> --tag <tag>

``export`` turns the chain's final slot labelling into the files ``scripts.review.review_data`` reads, so
the result becomes a review package next to ``reconstruct_from_slp``'s and both are scored the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("top_cam", "side1_cam", "side2_cam")
#: c01's cache namespace. The bottom-up run is the one the production chain reads.
RUN = "bottomup"
#: The calibration this repo's 3D reconstruction uses; medaka-3d's copy must match it byte for byte.
CALIBRATION_DIR = REPO_ROOT / "calibration_results" / "fixed_refractive_192x192x186"
TOOLS = {
    "c27": "c27_uncapped_tracklets_any.py",
    "c28": "c28_lambda_grid_transfer.py",
    "c33": "c33_ray_rejection.py",
    "c30": "c30_slot_stitch.py",
    "c34": "c34_microflip.py",
}
#: Tools that read ``c02_tracking_probe.CALIBRATION`` and therefore run through ``run_tool``.
CALIBRATION_TOOLS = frozenset({"c28", "c33"})
#: Output tags of K_1_2's production chain, reused so the leaves read the same.
C30_TAG = "C33-on-C28-v4"
CHAIN_TAG = "chain-c28-c33-c30v4-c34"
#: c28's inherited lambda grid (mm). No lambda is chosen: a claim must hold at every one.
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.767, 2.0, 8.0)
MAX_GAP_FRAMES = 100      # as the original pipeline's --max-gap-frames: longer gaps are not bridged


def medaka3d_repo(path: str | Path | None = None) -> Path:
    """The medaka-3d checkout (default: a sibling of this repo), verified to hold the package and tools."""
    repo = Path(path).resolve() if path else (REPO_ROOT.parent / "medaka-3d").resolve()
    for need in ("src/medaka3d/calib.py", "tools/c01_s16_predictions.py", "configs/paths.local.toml"):
        if not (repo / need).is_file():
            raise FileNotFoundError(f"medaka-3d checkout {repo} has no {need}")
    return repo


def _on_path(m3d: Path) -> None:
    for entry in (m3d / "src", m3d / "tools"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))


def m3d_paths(m3d: Path):
    """medaka-3d's own existence-checked path config for this machine."""
    _on_path(m3d)
    from medaka3d import load_paths

    return load_paths(m3d / "configs" / "paths.local.toml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inference_command(slp: Path) -> str | None:
    """The ``# cmd:`` line ``batch_bottomup_track`` writes at the top of ``<view>.inference.log``."""
    log = slp.with_name(slp.stem + ".inference.log")
    if not log.is_file():
        return None
    with open(log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("# cmd:"):
                return line[len("# cmd:"):].strip()
    return None


def build_detection_cache(experiment: str, slp_dir: str | Path, *, m3d: str | Path | None = None,
                          views=VIEWS, force: bool = False, log=print) -> dict[str, Path]:
    """Write c01's cache for ``experiment`` from ``<slp_dir>/<view>.slp``; returns the cache file per view.

    A view whose cache is newer than its ``.slp`` is kept unless ``force``. ``c01.read_pair`` refuses
    a file that already carries tracks or whose frames resolve to another experiment or camera.
    """
    m3d = medaka3d_repo(m3d)
    paths = m3d_paths(m3d)
    import c01_s16_predictions as c01
    from medaka3d.provenance import provenance

    slp_dir = Path(slp_dir)
    out_dir = paths.work_dir / "c01_s16_predictions" / RUN
    cached = {view: c01.cache_path(paths.work_dir, experiment, view, RUN) for view in views}
    if not force and all(p.is_file() and p.stat().st_mtime > (slp_dir / f"{v}.slp").stat().st_mtime
                         for v, p in cached.items()):
        log(f"  all {len(cached)} views cached and newer than their .slp; kept (provenance untouched)")
        return cached
    written: dict[str, Path] = {}
    with provenance("s16_predictions", out_dir, name=f"c01_{RUN}__{experiment}") as record:
        record.param("experiment", experiment)
        record.param("run", RUN)
        record.param("source", "medaka-mating-pipeline merged per-view .slp, not s16 chunks")
        record.param("adapter", "medaka-mating-pipeline scripts/reconstruction_3d/medaka3d_chain.py")
        record.param("n_frames", c01.N_FRAMES)
        summary = {}
        for view in views:
            slp = slp_dir / f"{view}.slp"
            if not slp.is_file():
                raise FileNotFoundError(slp)
            path = c01.cache_path(paths.work_dir, experiment, view, RUN)
            record.input(view, slp)
            record.param(f"inference_command__{view}", _inference_command(slp))
            if path.is_file() and path.stat().st_mtime > slp.stat().st_mtime and not force:
                log(f"  {view:<10} cached, kept: {path}")
                written[view] = path
                record.output(f"{experiment}__{view}", path)
                continue
            started = time.time()
            detections = c01.read_pair([slp], experiment, view, c01.N_FRAMES)
            if detections.frame_idx.size and int(detections.frame_idx.max()) >= c01.N_FRAMES:
                raise ValueError(f"{slp}: frame {int(detections.frame_idx.max())} is past c01.N_FRAMES "
                                 f"{c01.N_FRAMES}; the chain's arrays would drop it")
            c01.save(detections, path)
            counts = detections.counts()
            summary[view] = {
                "instances": int(detections.points.shape[0]),
                "frames_empty": int((counts == 0).sum()),
                "eq3": int((counts == 3).sum()),
                "max_per_frame": int(counts.max()) if counts.size else 0,
            }
            record.output(f"{experiment}__{view}", path)
            written[view] = path
            log(f"  {view:<10} {detections.points.shape[0]:>9,} instances  empty {summary[view]['frames_empty']:>5}  "
                f"=3 {100 * (counts == 3).mean():5.1f}%  max {summary[view]['max_per_frame']}  "
                f"{time.time() - started:5.1f}s -> {path}")
        record.param("summary", summary)
    return written


def local_calibration(m3d: str | Path | None = None) -> Path:
    """``c02_tracking_probe.CALIBRATION`` on this machine, checked against ``CALIBRATION_DIR``.

    The stored path is ``<bucket medaka root>/calibration/...``; it is re-rooted onto the parent of
    medaka-3d's ``work_dir`` (the same bucket root here) when it does not exist as written.
    """
    m3d = medaka3d_repo(m3d)
    paths = m3d_paths(m3d)
    import c02_tracking_probe as c02

    original = Path(c02.CALIBRATION)
    local = original
    if not original.is_dir():
        parts = original.parts
        local = paths.work_dir.parent.joinpath(*parts[parts.index("calibration"):])
    if not local.is_dir():
        raise FileNotFoundError(f"medaka-3d calibration {original} is not on this machine (tried {local})")
    bundles = sorted(CALIBRATION_DIR.glob("cam*_bundle_refractive.json"))
    if len(bundles) != 3:
        raise FileNotFoundError(f"expected 3 refractive bundles in {CALIBRATION_DIR}, found {len(bundles)}")
    for ours in bundles:
        theirs = local / ours.name
        if not theirs.is_file() or _sha256(theirs) != _sha256(ours):
            raise ValueError(f"{theirs} is not byte-identical to {ours}; the two pipelines would not share "
                             f"a calibration")
    return local


def run_tool(tool: str, argv: list[str], m3d: str | Path | None = None) -> None:
    """Run a medaka-3d tool in this process, with ``CALIBRATION`` pointed at this machine's copy."""
    m3d = medaka3d_repo(m3d)
    _on_path(m3d)
    script = m3d / "tools" / TOOLS[tool]
    if tool in CALIBRATION_TOOLS:
        import c02_tracking_probe as c02

        c02.CALIBRATION = local_calibration(m3d)
        print(f"[medaka3d_chain] CALIBRATION -> {c02.CALIBRATION}", flush=True)
    os.chdir(m3d)
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


def tool_command(tool: str, argv: list[str], m3d: str | Path | None = None) -> tuple[list[str], dict, Path]:
    """``(command, env, cwd)`` that runs one tool as a separate process."""
    m3d = medaka3d_repo(m3d)
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([str(m3d / "src"), str(REPO_ROOT)]))
    if tool in CALIBRATION_TOOLS:
        cmd = [sys.executable, "-m", "scripts.reconstruction_3d.medaka3d_chain", "run", tool,
               "--m3d", str(m3d), "--", *argv]
    else:
        cmd = [sys.executable, str(m3d / "tools" / TOOLS[tool]), *argv]
    return cmd, env, m3d


def chain_steps(experiment: str, m3d: str | Path | None = None,
                lambdas: list[float] | None = None) -> list[tuple[str, list[str], Path]]:
    """The production chain as ``(tool, argv, output dir)``, in order."""
    m3d = medaka3d_repo(m3d)
    work = m3d_paths(m3d).work_dir
    lam = ["--lambdas", *(f"{x:g}" for x in lambdas)] if lambdas else []
    c28_dir = work / "c28_lambda_grid_transfer" / experiment
    c33_dir = work / "c33_ray_rejection" / f"{experiment}__C28-gated"
    c30_dir = work / "c30_slot_stitch" / f"{experiment}__{C30_TAG}"
    c34_dir = work / "c34_microflip" / f"{experiment}__{CHAIN_TAG}"
    exp = ["--experiment", experiment]
    return [
        ("c27", exp, work / "c27_uncapped_tracklets_any" / experiment / "mouth-pectoral_oks_serial"),
        ("c28", [*exp, *lam], c28_dir),
        ("c33", [*exp, "--arm-dir", str(c28_dir), "--arm-prefix", "C28-gated", *lam], c33_dir),
        ("c30", [*exp, "--arm-dir", str(c33_dir), "--arm-prefix", "C33-gated", "--tag", C30_TAG, *lam], c30_dir),
        ("c34", [*exp, "--arm-dir", str(c30_dir), "--tag", CHAIN_TAG, *lam], c34_dir),
    ]


def step_status(tool: str, out_dir: Path) -> str:
    """``ok`` / ``failed`` / ``running`` from the tool's provenance sidecar, ``missing`` without one."""
    sidecar = Path(out_dir) / f"{tool}.provenance.json"
    if not sidecar.is_file():
        return "missing"
    try:
        return str(json.loads(sidecar.read_text(encoding="utf-8")).get("status", "unknown"))
    except json.JSONDecodeError:
        return "unreadable"


def run_chain(experiment: str, *, m3d: str | Path | None = None, lambdas: list[float] | None = None,
              start: str | None = None, force: bool = False, log_dir: str | Path | None = None,
              log=print) -> None:
    """Run the chain step by step, each as its own process; steps already ``ok`` are skipped unless
    ``force``. ``start`` skips every step before it. Stops at the first failing step."""
    m3d = medaka3d_repo(m3d)
    paths = m3d_paths(m3d)
    import c01_s16_predictions as c01

    for view in VIEWS:
        if not c01.cache_path(paths.work_dir, experiment, view, RUN).is_file():
            raise FileNotFoundError(f"no c01 cache for {experiment} {view}; run the 'cache' step first")
    steps = chain_steps(experiment, m3d, lambdas)
    names = [s[0] for s in steps]
    if start is not None:
        steps = steps[names.index(start):]
    log_dir = Path(log_dir) if log_dir else paths.work_dir / "medaka3d_chain_logs" / experiment
    log_dir.mkdir(parents=True, exist_ok=True)
    for tool, argv, out_dir in steps:
        if not force and step_status(tool, out_dir) == "ok":
            log(f"[chain] {tool}: already ok in {out_dir}, skipped")
            continue
        cmd, env, cwd = tool_command(tool, argv, m3d)
        log_path = log_dir / f"{tool}.log"
        started = time.time()
        log(f"[chain] {tool}: {' '.join(cmd)}  (log {log_path})")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n# started {time.ctime()}\n# cmd: {' '.join(cmd)}\n")
            handle.flush()
            code = subprocess.call(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        status = step_status(tool, out_dir)
        log(f"[chain] {tool}: exit {code}, provenance {status}, {(time.time() - started) / 60:.1f} min")
        if code != 0 or status != "ok":
            raise RuntimeError(f"{tool} failed (exit {code}, provenance {status}); see {log_path}")
    log(f"[chain] {experiment}: done")


def chain_arm(experiment: str, lam: float = 0.767, kind: str = "label", m3d: str | Path | None = None) -> Path:
    """The chain's final ``C34-<kind>`` arm. ``label`` is the slot labelling (three permanent ids, what an
    analysis consumes); ``gated`` is the gated tracker that breaks instead of guessing."""
    work = m3d_paths(medaka3d_repo(m3d)).work_dir
    return work / "c34_microflip" / f"{experiment}__{CHAIN_TAG}" / f"C34-{kind}__{experiment}__lam{lam:g}.npz"


def detector_row_slots(slp: Path, n_frames: int) -> np.ndarray:
    """``(n_frames, most instances in a frame)`` the ``load_2d_slp`` slot of every detector row; -1 where it
    has none.

    A detector row is an instance's position in its labeled frame - what medaka-3d's ``source`` holds.
    ``load_2d_slp`` drops false positives and sorts by score, and ``_best.json`` and the review build
    address detections by that slot, so every row is translated here.
    """
    import sleap_io as sio

    from scripts.reconstruction_3d.load_2d_slp import _instances_in_slot_order

    labels = sio.load_slp(str(slp))
    node_names = [n.name for n in labels.skeletons[0].nodes]
    max_rows = max((len(lf.instances) for lf in labels.labeled_frames), default=0)
    out = np.full((n_frames, max_rows), -1, np.int8)
    for lf in labels.labeled_frames:
        frame = int(lf.frame_idx)
        if not 0 <= frame < n_frames:
            continue
        instances = list(lf.instances)
        slot ={id(inst): s for s, inst in enumerate(_instances_in_slot_order(instances, node_names=node_names))}
        for row, inst in enumerate(instances):
            out[frame, row] = slot.get(id(inst), -1)
    return out


def _git_rev(repo: Path) -> str | None:
    try:
        rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src", "tools"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def fill_track_csv(csv_3d: Path, max_gap_frames: int = MAX_GAP_FRAMES) -> int:
    """Interpolate the gaps of a ``<tag>_3d.csv`` in place, as ``reconstruct_from_slp`` does; returns the
    fish-frames filled.

    Observed rows are those not flagged ``is_interpolated``, so a second run changes nothing. Gaps over
    ``max_gap_frames`` and the frames before a fish's first or after its last position stay NaN.
    """
    import pandas as pd

    from scripts.reconstruction_3d.fill_missing_3d import ensure_all_frames_and_fill_spline

    csv_3d = Path(csv_3d)
    table = pd.read_csv(csv_3d)
    frames = sorted(int(f) for f in table["frame"].unique())
    seen = table[table[["X", "Y", "Z"]].notna().all(axis=1) & ~table["is_interpolated"].astype(bool)]
    tracks = {fish: seen[seen["fish_id"] == fish][["frame", "X", "Y", "Z"]] for fish in table["fish_id"].unique()}
    filled = ensure_all_frames_and_fill_spline(tracks, frames, max_gap_frames=max_gap_frames,
                                               ensure_non_negative=False)
    # A track with fewer than two positions comes back without the empty frames.
    out = pd.concat([d.set_index("frame").reindex(frames).rename_axis("frame").reset_index().assign(fish_id=fish)
                     for fish, d in filled.items()]).sort_values(["frame", "fish_id"])
    out["is_interpolated"] = out["is_interpolated"].fillna(False).astype(bool)
    tmp = csv_3d.with_suffix(".partial.csv")
    out[["frame", "fish_id", "X", "Y", "Z", "is_interpolated"]].to_csv(tmp, index=False)
    os.replace(tmp, csv_3d)
    return int(out["is_interpolated"].sum())


def export_review_inputs(experiment: str, slp_dir: str | Path, out_dir: str | Path, tag: str, *,
                         lam: float = 0.767, m3d: str | Path | None = None, log=print) -> dict:
    """Write ``<tag>_best.json``, ``<tag>_3d.csv`` and ``<tag>_reconstruction_provenance.json`` from the
    chain's final slot labelling, in the layout ``scripts.review.review_data.build_review_data`` reads.

    Identity is medaka-3d's own: slot column ``k`` of ``C34-label`` is ``ID_(k+1)`` for the whole
    recording, with no top-camera bind, so no ``_3d_identified.csv`` is written. X/Y/Z are the chain's
    pectoral anchor. The chain leaves whole frames empty, so ``<tag>_3d.csv`` is gap-filled and flagged like
    the original pipeline's (``fill_track_csv``); ``<tag>_best.json`` keeps the gaps. A detection the chain
    used but ``load_2d_slp`` drops cannot be addressed by slot; those are counted in the provenance.
    """
    from scripts.reconstruction_3d.reconstruct_from_slp import _save_best_tracked_json

    m3d = medaka3d_repo(m3d)
    arm_path = chain_arm(experiment, lam, m3d=m3d)
    with np.load(arm_path) as z:
        track, source = z["slot_track"], z["source"]
        xyz, n_used = z["xyz"].astype(float), z["n_views_used"]
    n_frames, n_slots, n_views = source.shape
    ids = [f"ID_{i + 1}" for i in range(n_slots)]
    present = (track >= 0) & np.isfinite(xyz).all(axis=-1)
    perm = np.full((n_frames, n_slots, n_views), -1, np.int16)
    unnamed: dict[str, int] = {}
    for v, view in enumerate(VIEWS):
        t0 = time.time()
        rows = detector_row_slots(Path(slp_dir) / f"{view}.slp", n_frames)
        f, k = np.nonzero(present & (source[:, :, v] >= 0))
        perm[f, k, v] = rows[f, source[f, k, v]]
        unnamed[view] = int((perm[f, k, v] < 0).sum())
        log(f"[export] {view}: {f.size:,} detections used, {unnamed[view]:,} without a load_2d_slp slot "
            f"({time.time() - t0:.0f} s)")

    # Identity index i of label column k at each frame (-1 where the slot is empty).
    id_of = np.where(present, track, -1)
    best_tracked: dict[int, dict] = {}
    nan3 = [float("nan")] * 3
    for t in range(n_frames):
        mapping = [-1] * n_slots
        for k in range(n_slots):
            if 0 <= id_of[t, k] < n_slots:
                mapping[int(id_of[t, k])] = k
        best_tracked[t] = {
            "id_mapping": mapping,
            **{f"perm_cam{v}": [int(p) for p in perm[t, :, v]] for v in range(n_views)},
            "recovered": [bool(k >= 0 and (perm[t, k] >= 0).sum() < n_views) for k in mapping],
            "missing_views": [],
            "groups": {ids[i]: ([float(c) for c in xyz[t, k]] if k >= 0 else list(nan3))
                       for i, k in enumerate(mapping)},
        }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "triangulation": {"backend": "external", "provenance": {"model_folder": str(CALIBRATION_DIR)}},
        "pipeline": "medaka-3d full chain (c27 -> c28 -> c33 -> c30 v4 -> c34)",
        "chain_tag": CHAIN_TAG,
        "arm": str(arm_path),
        "lambda_mm": lam,
        "medaka3d_repo": str(m3d),
        "medaka3d_rev": _git_rev(m3d),
        "pipeline_rev": _git_rev(REPO_ROOT),
        "identity": "C34-label slot columns as ID_1..ID_3; no top-camera bind",
        "xyz": f"pectoral anchor from the chain; gaps up to {MAX_GAP_FRAMES} frames interpolated in the CSV",
        "frames_resolved": int((n_used >= 2).sum()),
        "fish_frames_present": int(present.sum()),
        "detections_without_slot": unnamed,
    }
    (out_dir / f"{tag}_reconstruction_provenance.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                                     encoding="utf-8")
    best_json = out_dir / f"{tag}_best.json"
    _save_best_tracked_json(best_tracked, best_json, num_individuals=n_slots, calibration_metadata=meta)

    import pandas as pd

    t_idx = np.repeat(np.arange(n_frames), n_slots)
    i_idx = np.tile(np.arange(n_slots), n_frames)
    k_of = np.full((n_frames, n_slots), -1)
    for t, info in best_tracked.items():
        k_of[t] = info["id_mapping"]
    k_flat = k_of[t_idx, i_idx]
    pos = np.where((k_flat >= 0)[:, None], xyz[t_idx, np.clip(k_flat, 0, None)], np.nan)
    csv_3d = out_dir / f"{tag}_3d.csv"
    pd.DataFrame({"frame": t_idx, "fish_id": np.array(ids)[i_idx], "X": pos[:, 0], "Y": pos[:, 1],
                  "Z": pos[:, 2], "is_interpolated": False}).to_csv(csv_3d, index=False)
    filled = fill_track_csv(csv_3d)
    log(f"[export] wrote {best_json.name}, {csv_3d.name} ({filled:,} fish-frames interpolated) -> {out_dir}")
    return {"best_json": best_json, "csv_3d": csv_3d, "provenance": meta}


def lambda_switch_table(experiment: str, state, lambdas=LAMBDA_GRID, m3d: str | Path | None = None):
    """ID switches of the chain's slot labelling at every lambda, against a reviewed reference.

    ``state`` is a ``scripts.review.recon_compare.ReconState`` whose reference (version 0) is the review.
    Each arm's pectoral anchor is matched to the reference's position node frame by frame and named at
    ``state.link_frame``, exactly as ``switch_table`` does for a review package - without building one.
    """
    import pandas as pd

    from scripts.review.recon_compare import match_to_reference, switches

    ref = state.nodes[0, :, :, state.pos_node].astype(float)
    n = len(state.frames)
    t0 = max(0, (state.analysis_start or state.first) - state.first)
    tl = state.link_frame - state.first
    rows = []
    for lam in lambdas:
        path = chain_arm(experiment, lam, m3d=m3d)
        if not path.is_file():
            continue
        with np.load(path) as z:
            xyz = z["xyz"].astype(float)
        arm = np.full((n, xyz.shape[1], 3), np.nan)
        part = xyz[state.first:state.first + n]
        arm[:len(part)] = part
        perm, margin = match_to_reference(ref, arm)
        good = np.flatnonzero((perm[tl:] >= 0).all(axis=1) & (margin[tl:] >= state.params.ambiguous_mm))
        link = perm[tl + int(good[0])] if good.size else perm[tl]
        ev = switches(perm[t0:], margin[t0:], link, state.labels, state.first + t0, state.params)
        row = {"lambda_mm": lam, "switches": len(ev),
               "into wrong": int((~ev["correct_after"].astype(bool)).sum())}
        for name in state.labels:
            row[f"with {name}"] = int(ev["fish"].str.split("+").map(lambda f, nm=name: nm in f).sum())
        row["coverage %"] = round(100 * float(np.isfinite(arm[t0:]).all(axis=-1).mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cache", help="write c01's detection cache from this repo's per-view .slp")
    c.add_argument("experiment")
    c.add_argument("--slp-dir", type=Path, required=True)
    c.add_argument("--m3d", type=Path, default=None)
    c.add_argument("--force", action="store_true")
    r = sub.add_parser("run", help="run one medaka-3d tool here, with the calibration re-rooted; the tool's "
                                   "own arguments follow '--'")
    r.add_argument("tool", choices=sorted(TOOLS))
    r.add_argument("--m3d", type=Path, default=None)
    ch = sub.add_parser("chain", help="run c27 -> c28 -> c33 -> c30 -> c34, skipping steps already ok")
    ch.add_argument("experiment")
    ch.add_argument("--m3d", type=Path, default=None)
    ch.add_argument("--lambdas", nargs="+", type=float, default=None, help="default: each tool's full grid")
    ch.add_argument("--start", choices=sorted(TOOLS), default=None)
    ch.add_argument("--force", action="store_true")
    ch.add_argument("--log-dir", type=Path, default=None)
    e = sub.add_parser("export", help="write review-package inputs from the chain's final slot labelling")
    e.add_argument("experiment")
    e.add_argument("--slp-dir", type=Path, required=True)
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--tag", required=True)
    e.add_argument("--lam", type=float, default=0.767)
    e.add_argument("--m3d", type=Path, default=None)
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after the first '--' belongs to the tool; argparse would otherwise mix in this CLI's options.
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]
    args = p.parse_args(argv)
    if passthrough and args.command != "run":
        p.error("only 'run' takes tool arguments after '--'")
    if args.command == "cache":
        build_detection_cache(args.experiment, args.slp_dir, m3d=args.m3d, force=args.force)
    elif args.command == "run":
        run_tool(args.tool, passthrough, args.m3d)
    elif args.command == "export":
        export_review_inputs(args.experiment, args.slp_dir, args.out_dir, args.tag, lam=args.lam, m3d=args.m3d)
    else:
        run_chain(args.experiment, m3d=args.m3d, lambdas=args.lambdas, start=args.start, force=args.force,
                  log_dir=args.log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
