"""Load one reviewed trial and write its behaviour-feature tables.

A trial is an experiment folder under ``<data_root>/review_packages`` that holds exactly
one package with an ``id_pass`` ledger, plus a BORIS csv. The package directory is
discovered from the ledger, so P-1 (``s14_...``) and K-1 (``s16_...``) use the same
command. Nothing is written back into the review package: a review that is still open
can be read, and the tables say so.

BORIS files are matched by the experiment folder name first, then by the trial key with
the date dropped (``20260113_P_1_2_FCMC`` finds ``20260112_P_1_2_FCMC.csv``). Two files
with the same trial key and different dates is an error rather than a guess.
"""

from __future__ import annotations

import getpass
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import behavior_features as F
from scripts.common.local_paths import REPO_ROOT, load_local_paths
from scripts.review import id_segments as S
from scripts.review.behaviors import default_search, find_boris_csv, load_bouts
from scripts.review.ledger import (COVERAGE_COLUMNS, EVENT_COLUMNS, analysis_start, read_csv, static_labels,
                                   swap_list, watched_intervals)

PARAMS = {"behaviors": list(F.FOCUS_BEHAVIORS), "alias": F.BEHAVIOR_ALIAS,
          "align_s": [F.ALIGN_BEFORE_S, F.ALIGN_AFTER_S], "trace_columns": list(F.TRACE_COLUMNS), "pose": "ignored",
          "short_bout_s": F.SHORT_BOUT_S, "short_pad_s": F.SHORT_PAD_S,
          "after_s": F.COMPETITION_AFTER_S, "baseline_limit": F.BASELINE_LIMIT, "trace_bin": F.TRACE_BIN}


@dataclass
class Trial:
    experiment: str
    package: Path
    boris: Path
    fps: float
    provenance: dict
    bouts: pd.DataFrame
    separability: pd.DataFrame
    baseline: pd.DataFrame
    competition: pd.DataFrame
    traces: pd.DataFrame
    aligned: pd.DataFrame

    @property
    def review_open(self) -> bool:
        return bool(self.provenance.get("review_lock"))


def resolve_roots(config: Path | None = None) -> tuple[Path, Path, Path]:
    """``(paths config, review root, output root)``. Fails if the review root is absent."""
    if config is None:
        user = REPO_ROOT / "configs" / f"paths.{getpass.getuser()}.toml"
        config = user if user.is_file() else REPO_ROOT / "configs" / "paths.local.toml"
    paths = load_local_paths(config)
    if paths is None or paths.data_root is None:
        raise SystemExit(f"no data_root in {config}")
    review = paths.data_root / "review_packages"
    if not review.is_dir():
        raise SystemExit(f"review packages not found: {review}")
    out = (paths.runs_dir or REPO_ROOT / "runs") / "behavior_features"
    return config, review, out


def find_package(review_root: Path, experiment: str, package: str | None = None) -> Path:
    """The one package of ``experiment`` that has an identity-pass ledger."""
    exp = Path(review_root) / experiment
    if not exp.is_dir():
        raise SystemExit(f"no review folder for {experiment} under {review_root}")
    if package:
        folder = exp / package
        if not (folder / "id_pass" / "human_events.csv").is_file():
            raise SystemExit(f"{folder} has no id_pass/human_events.csv")
        return folder
    hits = sorted(p.parents[1] for p in exp.glob("*/id_pass/human_events.csv"))
    if len(hits) != 1:
        found = ", ".join(p.name for p in hits) or "none"
        raise SystemExit(f"{experiment}: expected one package with an id_pass ledger, found {found}")
    return hits[0]


def locate_boris(experiment: str, search: list[Path]) -> tuple[Path, str]:
    """``(csv, how)`` with ``how`` ``exact`` or ``trial_key``."""
    exact = find_boris_csv(experiment, search)
    if exact is not None:
        return exact, "exact"
    parts = experiment.split("_", 1)
    key = parts[1] if len(parts) == 2 and parts[0].isdigit() else ""
    hits: list[Path] = []
    for place in search:
        place = Path(place)
        if place.is_file() and place.name.endswith(f"_{key}.csv"):
            hits.append(place)
        elif place.is_dir() and key:
            hits.extend(p for p in place.rglob(f"*_{key}.csv") if p.is_file())
    unique = list({p.resolve(): p for p in hits}.values())
    names = {p.name for p in unique}
    if len(names) == 1:
        return max(unique, key=lambda p: p.stat().st_mtime), "trial_key"
    if not unique:
        raise SystemExit(f"no BORIS csv for {experiment}: neither {experiment}.csv nor *_{key}.csv in "
                         + ", ".join(str(p) for p in search))
    raise SystemExit(f"several BORIS files match trial {key}: " + ", ".join(str(p) for p in unique))


def experiments_of(review_root: Path, group: str) -> list[str]:
    """Experiment folders of one group, e.g. ``P_1`` matches ``20260112_P_1_1_FCMC``."""
    token = f"_{group}_"
    return sorted(p.name for p in Path(review_root).iterdir() if p.is_dir() and token in p.name)


def build_trial(package: Path, boris: Path, experiment: str = "") -> tuple[Trial, pd.DataFrame]:
    """Read the package and return the trial plus the per-frame table (not saved)."""
    from scripts.review.ledger import owner_series

    led = package / "id_pass"
    with np.load(package / "review_data.npz") as z:
        frames = np.asarray(z["frames"], int)
        fish_ids = [str(f) for f in z["fish_ids"]]
        node_names = [str(n) for n in z["node_names"]]
        key = "nodes3d_filled" if "nodes3d_filled" in z.files else "nodes3d"
        nodes = np.asarray(z[key], float)
        xyz = np.asarray(z["xyz"], float)
        present = np.asarray(z["view_present"])
        interp = np.asarray(z["is_interpolated"])
    missing = [n for n in ("mouth", "pectoral", "tail") if n not in node_names]
    if missing:
        raise SystemExit(f"{package} nodes are {node_names}; missing {missing}")
    mouth, pectoral, tail = (node_names.index(n) for n in ("mouth", "pectoral", "tail"))
    manifest_path = package.parent / "proxy" / "proxy_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no proxy manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fps = float(manifest["views"]["top_cam"]["source_fps"])

    events = read_csv(led / "human_events.csv", EVENT_COLUMNS)
    coverage = read_csv(led / "coverage.csv", COVERAGE_COLUMNS)
    labels = static_labels(events, fish_ids)
    unnamed = [name for name in F.NAMES if name not in labels.values()]
    if unnamed:
        raise SystemExit(f"{package} labels are {labels}; missing {unnamed}")
    start = analysis_start(events)
    if start is None:
        raise SystemExit(f"{package} has no analysis_start")
    swaps = swap_list(events, fish_ids)
    danger = S.danger_map(xyz, present, interp, fps)
    status = S.frame_status(int(frames[0]), len(frames), start, danger["danger"], danger["runs"],
                            watched_intervals(coverage), [f for f, _, _ in swaps], S.manual_ranges(events), fps)
    usable = np.isin(status, (S.WATCHED_SAFE, S.WATCHED_DANGER))
    analysed = status != S.NOT_ANALYSED
    owner = owner_series(len(fish_ids), swaps, frames)
    named, valid = F.named_nodes(nodes, owner, fish_ids, labels)
    named[~usable | ~valid] = np.nan

    table = F.kinematic_table(named, frames, fps, mouth, pectoral, tail)
    raw = F.normalise_behaviors(load_bouts(boris))
    summarised = F.bout_features(table, raw, fps, int(frames[0]))
    finite = np.isfinite(table[["dist_F_M1", "dist_F_M2", "dist_M1_M2"]]).any(axis=1).to_numpy()
    summary, baseline = F.separability(summarised, table, raw, finite)
    races = F.competition(table, summarised, fps)
    curves = F.aligned(table, summarised, fps)
    traces = F.decimate(table, F.TRACE_COLUMNS)
    reviewed = F.review_bins(frames, status)
    traces["review"] = reviewed["review"].to_numpy()
    n_analysed = max(1, int(analysed.sum()))
    review_pct = {"kept": round(100 * float(usable.sum()) / n_analysed, 2),
                  "discarded": round(100 * float((status == S.UNUSABLE).sum()) / n_analysed, 2),
                  "unwatched": round(100 * float((status == S.UNWATCHED).sum()) / n_analysed, 2)}
    play = manifest["views"]["top_cam"]
    provenance = {"experiment": experiment or package.parent.name, "package": str(package), "boris": str(boris),
                  "fps": fps, "analysis_start": int(start), "frames": int(len(frames)),
                  "usable_frames": int(usable.sum()),
                  "usable_pct": review_pct["kept"], "review_pct": review_pct,
                  "review_lock": (led / "review.lock").is_file(), "labels": labels, "params": PARAMS,
                  "node_source": key, "play_file": play["play_file"], "play_step": int(play["play_step"]),
                  "proxy_dir": str(package.parent / "proxy")}
    trial = Trial(provenance["experiment"], package, boris, fps, provenance, summarised, summary, baseline,
                  races, traces, curves)
    return trial, table


def _fresh(folder: Path, sources: list[Path]) -> bool:
    path = folder / "provenance.json"
    if not path.is_file():
        return False
    meta = json.loads(path.read_text(encoding="utf-8"))
    stamps = {p.name: p.stat().st_mtime_ns for p in sources}
    return meta.get("params") == PARAMS and meta.get("source_mtime_ns") == stamps


def _read(folder: Path, experiment: str) -> Trial:
    meta = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    return Trial(experiment, Path(meta["package"]), Path(meta["boris"]), float(meta["fps"]), meta,
                 pd.read_csv(folder / "bouts.csv"), pd.read_csv(folder / "separability.csv"),
                 pd.read_csv(folder / "baseline.csv"), pd.read_csv(folder / "competition.csv"),
                 pd.read_parquet(folder / "traces.parquet"), pd.read_parquet(folder / "aligned.parquet"))


def save_trial(trial: Trial, folder: Path, sources: list[Path]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(trial.provenance)
    meta["source_mtime_ns"] = {p.name: p.stat().st_mtime_ns for p in sources}
    (folder / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    trial.bouts.to_csv(folder / "bouts.csv", index=False)
    trial.separability.to_csv(folder / "separability.csv", index=False)
    trial.baseline.to_csv(folder / "baseline.csv", index=False)
    trial.competition.to_csv(folder / "competition.csv", index=False)
    trial.traces.to_parquet(folder / "traces.parquet", index=False)
    trial.aligned.to_parquet(folder / "aligned.parquet", index=False)


def ensure_trial(review_root: Path, out: Path, experiment: str, search: list[Path],
                 package: str | None = None, rebuild: bool = False) -> Trial:
    """Load the saved tables, building them when the sources are newer than the save."""
    folder = find_package(review_root, experiment, package)
    boris, how = locate_boris(experiment, search)
    sources = [folder / "id_pass" / "human_events.csv", folder / "id_pass" / "coverage.csv",
               folder / "review_data.npz", boris]
    dest = out / experiment
    if not rebuild and _fresh(dest, sources):
        return _read(dest, experiment)
    trial, _table = build_trial(folder, boris, experiment)
    trial.provenance["boris_match"] = how
    save_trial(trial, dest, sources)
    return _read(dest, experiment)


def pool(trials: list[Trial], group: str) -> Trial:
    """One trial-shaped bundle of several days, for the group figures. Not saved as a day."""
    bouts = pd.concat([t.bouts.assign(experiment=t.experiment) for t in trials], ignore_index=True)
    baseline = pd.concat([t.baseline.assign(experiment=t.experiment) for t in trials], ignore_index=True)
    competition = pd.concat([t.competition.assign(experiment=t.experiment) for t in trials], ignore_index=True)
    aligned = pd.concat([t.aligned.assign(experiment=t.experiment) for t in trials], ignore_index=True)
    summary = F.separability_from_bouts(bouts, baseline)
    open_days = [t.experiment for t in trials if t.review_open]
    provenance = {"experiment": group, "days": [t.experiment for t in trials], "review_open": open_days,
                  "fps": trials[0].fps, "params": PARAMS, "usable_pct": None,
                  "note": "days of one group are repeated measures of the same fish"}
    return Trial(group, Path("."), Path("."), trials[0].fps, provenance, bouts, summary, baseline,
                 competition, trials[0].traces.iloc[0:0], aligned)


def group_status(review_root: Path, group: str) -> tuple[list[str], list[str]]:
    """``(ready experiments, skipped with the reason)``."""
    ready, skipped = [], []
    for experiment in experiments_of(review_root, group):
        hits = list((Path(review_root) / experiment).glob("*/id_pass/human_events.csv"))
        if len(hits) == 1:
            ready.append(experiment)
        elif not hits:
            skipped.append(f"{experiment}: no id_pass ledger")
        else:
            skipped.append(f"{experiment}: {len(hits)} packages with an id_pass ledger")
    if not experiments_of(review_root, group):
        raise SystemExit(f"no experiment folder matching _{group}_ under {review_root}")
    return ready, skipped


def parse_group(text: str) -> str:
    """``P-1``, ``p1`` and ``P_1`` all name the group ``P_1``."""
    token = text.strip().upper().replace("-", "_")
    if re.fullmatch(r"[A-Z]+_\d+", token):
        return token
    collapsed = re.fullmatch(r"([A-Z]+)(\d+)", token)
    if collapsed:
        return f"{collapsed.group(1)}_{collapsed.group(2)}"
    raise SystemExit(f"group {text!r} is not a group key such as P_1 or K_1")
