"""Compare 3D reconstructions of one experiment (the compare GUI's reconstruction mode).

Every reconstruction is a review package folder - ``build_review_data`` output, e.g.
``<review_root>/<exp>/s14_bottomup_3node_reviewed`` (``reconstruct_from_slp``) and
``<review_root>/<exp>/s14_bottomup_3node_reviewed_medaka3d`` (medaka-3d per-frame correspondence through
the same linker, see ``scripts.reconstruction_3d.medaka3d_bridge``). The first one is the reference.
Per version either

* ``initial`` - plain triangulation of that reconstruction's detections with the build's head/tail
  orientation and node completion (``initial_tracks.npz``, needs the calibration), or
* ``machine`` - the review data (``nodes3d_filled``: + two-view re-triangulation and despiking),
* ``manual`` / ``final`` - a finished review (``versions.npz`` from ``export_reviewed``): the reviewer's
  identities throughout, + body fixes (and the post-pass for ``final``). As the reference this is the
  answer key: names are fixed, and every other version's disagreement with it is that version's error.
  Identity errors are judged without bias (the reviewer watched every frame). Geometry is not: the review
  corrected one reconstruction, so where the reviewer left it alone ``final`` equals that reconstruction.

``build`` writes ``recon_versions.npz`` into ``<reference>/recon_vs_<other>/``; the GUI writes the names
(``recon_labels.csv``) and the reviewer's verdicts on disagreements (``recon_verdicts.csv``) next to it.
Nothing else is touched, so the reference's own review ledger stays as it is.

Identity (unreviewed reference), as in the review GUI: a person names the reference's fish once (F / M1 / M2); every other
version takes the names of the tracks at the same place at that frame and then follows its own tracks,
so an identity swap in either version shows as a disagreement. "Same identity" instead matches the
versions frame by frame (6 permutations, position node) and leaves only geometry differences.
An experiment that has been reviewed already uses the names of its ledger.

Disagreements (per reference fish): ``swap`` (the names point at different fish), ``position``
(position node > 5 mm apart), ``nodes`` (head or tail > 5 mm apart, position agreeing), ``only <A>`` /
``only <B>`` (one version has no 3D for that fish).

CLI::

    python -m scripts.review.recon_compare <reference folder> <other folder> --kinds initial initial \
        --names old medaka3d
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

RECON_FILE = "recon_versions.npz"
LABELS_FILE = "recon_labels.csv"
VERDICTS_FILE = "recon_verdicts.csv"
KINDS = ("initial", "machine", "manual", "final")
REVIEWED_KINDS = ("manual", "final")
GROUPS = ("identity", "geometry", "coverage")
VERDICTS = {"a_right": "A right", "b_right": "B right", "both_wrong": "both wrong", "both_ok": "both OK"}
LABEL_COLUMNS = ["version", "fish_id", "ref_fish_id", "label", "frame", "created"]
VERDICT_COLUMNS = ["start", "end", "ref_fish", "kind", "other", "verdict", "created"]
CHANGE_COLUMNS = ["start", "end", "fish", "ref_fish", "group", "kind", "size_mm", "note", "other"]
POSITION_NODE = "pectoral"
_BIG = 1e4                      # matching cost of a missing fish
NODE_INTERPOLATED = 6           # node_source: filled by ``fill_gaps`` (linear in time along the version's track)


@dataclass(frozen=True)
class ReconParams:
    position_mm: float = 5.0          # position-node disagreement that is listed
    nodes_mm: float = 5.0             # head / tail disagreement (position agreeing)
    merge_frames: int = 17            # entries closer than this (~0.1 s) become one
    min_swap_frames: int = 8          # shorter identity disagreements are matching noise
    min_coverage_frames: int = 17     # shorter one-sided data is not listed
    ambiguous_mm: float = 2.0         # frames whose best and second-best matching differ less are skipped
    inherit_max_mm: float = 5.0       # inherited names: warn when the fish are further apart
    separation_mm: float = 15.0       # default naming frame: every fish at least this far from the others


# ---------------------------------------------------------------------------
# Build


def version_data(folder: Path, kind: str, cal=None, log=None) -> dict:
    """One reconstruction's per-track nodes ``(F, K, N, 3)`` plus the metadata of its review data."""
    folder = Path(folder)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if kind in REVIEWED_KINDS:
        from scripts.review.versions import VERSIONS, VERSIONS_FILE

        with np.load(folder / VERSIONS_FILE) as z:
            v = VERSIONS.index(kind)
            if kind not in [str(p) for p in z["present"]]:
                raise ValueError(f"{folder / VERSIONS_FILE} has no {kind!r} version")
            labels = [str(x) for x in z["labels"]]
            return {"frames": z["frames"], "fish_ids": labels, "machine_label": labels,
                    "node_names": [str(n) for n in z["node_names"]], "edges": z["edges"],
                    "head_node": int(z["head_node"]), "tail_node": int(z["tail_node"]),
                    "nodes": z["nodes"][v], "source": z["source"][v], "analysis_start": int(z["analysis_start"])}
    with np.load(folder / "review_data.npz") as z:
        out = {"frames": z["frames"], "fish_ids": [str(f) for f in z["fish_ids"]],
               "node_names": [str(n) for n in z["node_names"]], "edges": z["edges"],
               "head_node": int(z["head_node"]), "tail_node": int(z["tail_node"])}
        out["machine_label"] = ([str(x) for x in z["machine_label"]] if "machine_label" in z.files
                                else list(out["fish_ids"]))
        if kind == "machine":
            out["nodes"], out["source"] = z["nodes3d_filled"], z["node_source"]
    if kind == "initial":
        from scripts.review.versions import initial_tracks

        init = initial_tracks(folder, cal=cal, log=log)
        out["nodes"], out["source"] = init["nodes3d_filled"], init["node_source"]
    return out


def default_out_dir(folders: list[Path], names: list[str] | None = None) -> Path:
    """``<reference>/recon_vs_<other>`` for two folders, else ``<reference>/recon_<name>_<name>...``."""
    folders = [Path(f) for f in folders]
    if names and (len(folders) > 2 or len({f.resolve() for f in folders}) < len(folders)):
        return folders[0] / ("recon_" + "_".join(names))
    return folders[0] / ("recon_vs_" + "+".join(f.name for f in folders[1:]))


def build(folders: list[Path], kinds: list[str], out_dir: Path | None = None, names: list[str] | None = None,
          cal=None, log=print) -> Path:
    """Write ``recon_versions.npz`` for ``folders`` (the first is the reference) on one frame axis."""
    folders = [Path(f) for f in folders]
    if len(folders) < 2 or len(kinds) != len(folders):
        raise ValueError("need at least two folders and one kind per folder")
    names = list(names) if names else [chr(ord("A") + i) for i in range(len(folders))]
    if len(set(names)) != len(names) or len(names) != len(folders):
        raise ValueError(f"need one distinct name per folder, got {names}")
    t0 = time.time()
    data = [version_data(f, k, cal=cal, log=log) for f, k in zip(folders, kinds)]
    ref = data[0]
    for f, d in zip(folders[1:], data[1:]):
        if d["node_names"] != ref["node_names"] or len(d["fish_ids"]) != len(ref["fish_ids"]):
            raise ValueError(f"{f}: nodes {d['node_names']} / {len(d['fish_ids'])} fish differ from the reference")
    first = min(int(d["frames"][0]) for d in data)
    last = max(int(d["frames"][-1]) for d in data)
    n_frames, n_fish, n_nodes = last - first + 1, len(ref["fish_ids"]), len(ref["node_names"])
    nodes = np.full((len(data), n_frames, n_fish, n_nodes, 3), np.nan, np.float32)
    source = np.full((len(data), n_frames, n_fish, n_nodes), -1, np.int8)
    for v, d in enumerate(data):
        t = np.asarray(d["frames"]) - first
        nodes[v, t], source[v, t] = d["nodes"], d["source"]
    out_dir = Path(out_dir) if out_dir else default_out_dir(folders, names)
    out_dir.mkdir(parents=True, exist_ok=True)
    node_names = ref["node_names"]
    tmp = out_dir / "recon_versions.partial.npz"
    np.savez_compressed(
        tmp, nodes=nodes, source=source, versions=np.array(names), kinds=np.array(kinds),
        tags=np.array([f.name for f in folders]), folders=np.array([str(f) for f in folders]),
        frames=np.arange(first, last + 1), fish_ids=np.array([d["fish_ids"] for d in data]),
        machine_label=np.array([d["machine_label"] for d in data]), node_names=np.array(node_names),
        edges=ref["edges"], head_node=ref["head_node"], tail_node=ref["tail_node"],
        position_node=node_names.index(POSITION_NODE) if POSITION_NODE in node_names else 0,
        analysis_start=int(ref.get("analysis_start", -1)))
    tmp.replace(out_dir / RECON_FILE)
    (out_dir / "recon_manifest.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": names, "kinds": list(kinds),
        "folders": [str(f) for f in folders], "frames": [first, last]}, indent=2), encoding="utf-8")
    log(f"[recon_compare] {RECON_FILE}: {names} ({time.time() - t0:.0f} s) -> {out_dir}")
    return out_dir


def fill_gaps(recon_dir: Path, version: str, max_gap_frames: int = 166, log=print) -> dict:
    """Fill ``version``'s gaps in ``recon_dir/recon_versions.npz`` by linear interpolation in time.

    Per track (the version's own fish - medaka-3d's slots are permanent) and node, every run of missing
    frames with data on both sides and at most ``max_gap_frames`` long is interpolated; gaps at the start
    or end stay empty. Filled nodes get ``node_source`` ``NODE_INTERPOLATED``. The file before the first
    fill is kept as ``recon_versions.before_fill.npz``. Returns per track the frames filled.
    """
    import shutil

    recon_dir = Path(recon_dir)
    path = recon_dir / RECON_FILE
    backup = recon_dir / "recon_versions.before_fill.npz"
    if not backup.exists():
        shutil.copy2(path, backup)
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    versions = [str(v) for v in data["versions"]]
    if version not in versions:
        raise ValueError(f"{path}: no version {version!r} (has {versions})")
    v = versions.index(version)
    nodes, source = data["nodes"], data["source"]
    filled = {}
    frames = np.arange(nodes.shape[1])
    for k in range(nodes.shape[2]):
        n_filled = 0
        for n in range(nodes.shape[3]):
            ok = np.isfinite(nodes[v, :, k, n]).all(axis=-1)
            if ok.sum() < 2:
                continue
            first, last = np.flatnonzero(ok)[[0, -1]]
            for s, e in _runs(~ok):
                if s <= first or e >= last or e - s + 1 > max_gap_frames:
                    continue
                for c in range(3):
                    nodes[v, s:e + 1, k, n, c] = np.interp(frames[s:e + 1], frames[ok], nodes[v, ok, k, n, c])
                source[v, s:e + 1, k, n] = NODE_INTERPOLATED
                n_filled += e - s + 1
        filled[f"track {k}"] = n_filled
    tmp = recon_dir / "recon_versions.partial.npz"
    np.savez_compressed(tmp, **data)
    tmp.replace(path)
    manifest = recon_dir / "recon_manifest.json"
    meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    meta.setdefault("filled", {})[version] = {"method": "linear in time per track and node",
                                              "max_gap_frames": max_gap_frames, "node_frames_filled": filled,
                                              "node_source": NODE_INTERPOLATED,
                                              "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    manifest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[recon_compare] {version}: filled node-frames {filled} (gaps <= {max_gap_frames} frames) -> {path}")
    return filled


# ---------------------------------------------------------------------------
# Matching and disagreements


def match_to_reference(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, the permutation of ``other``'s tracks closest to ``ref``'s (summed position distance).

    ``ref``/``other`` are ``(F, K, 3)``. Returns ``perm (F, K)`` - the other track at each reference fish,
    -1 where one of the two has no position - and ``margin (F,)``, second-best minus best cost (mm).
    """
    n_fish = ref.shape[1]
    perms = np.array(list(permutations(range(n_fish))), dtype=np.intp)          # (P, K)
    d = np.linalg.norm(ref[:, :, None, :] - other[:, None, :, :], axis=-1)      # (F, K, K)
    d = np.where(np.isfinite(d), d, _BIG)
    cost = d[:, np.arange(n_fish)[None, :], perms].sum(axis=-1)                 # (F, P)
    order = np.argsort(cost, axis=1)
    rows = np.arange(len(cost))
    best = perms[order[:, 0]]
    margin = cost[rows, order[:, 1]] - cost[rows, order[:, 0]] if len(perms) > 1 else np.full(len(cost), np.inf)
    matched = d[rows[:, None], np.arange(n_fish)[None, :], best] < _BIG
    return np.where(matched, best, -1), margin


def _runs(mask: np.ndarray, merge: int = 0) -> list[tuple[int, int]]:
    """Inclusive index runs of ``mask``; runs separated by at most ``merge`` false entries are joined."""
    d = np.diff(np.r_[0, np.asarray(mask, bool).astype(np.int8), 0])
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))
    out: list[list[int]] = []
    for s, e in runs:
        if out and s - out[-1][1] - 1 <= merge:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(int(s), int(e)) for s, e in out]


def _forward_fill(state: np.ndarray) -> np.ndarray:
    """NaN entries take the last known value (0 before the first)."""
    known = np.isfinite(state)
    idx = np.where(known, np.arange(len(state)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = state[idx]
    return np.where(np.isfinite(filled), filled, 0.0)


def gather(nodes: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``nodes (F, K, ...)`` indexed per frame by ``idx (F, K)`` (or ``(K,)``); NaN where ``idx < 0``."""
    idx = np.broadcast_to(idx, nodes.shape[:2])
    out = nodes[np.arange(len(nodes))[:, None], np.clip(idx, 0, None)].astype(float)
    out[idx < 0] = np.nan
    return out


def disagreements(ref: np.ndarray, other: np.ndarray, perm: np.ndarray, margin: np.ndarray, link: np.ndarray,
                  *, pos: int, head: int, tail: int, first: int, ref_ids: list[str], ref_name: str,
                  other_name: str, params: ReconParams) -> pd.DataFrame:
    """Disagreements of ``other`` against ``ref`` (both ``(F, K, N, 3)``), one row per reference fish and run."""
    p = params
    aligned = gather(other, perm)
    named = gather(other, link)
    with np.errstate(invalid="ignore"):
        d_pos = np.linalg.norm(ref[:, :, pos] - aligned[:, :, pos], axis=-1)
        d_ht = np.fmax(np.linalg.norm(ref[:, :, head] - aligned[:, :, head], axis=-1),
                       np.linalg.norm(ref[:, :, tail] - aligned[:, :, tail], axis=-1))
        d_named = np.linalg.norm(ref[:, :, pos] - named[:, :, pos], axis=-1)
    ref_ok = np.isfinite(ref[:, :, pos]).all(axis=-1)
    aligned_ok = np.isfinite(aligned[:, :, pos]).all(axis=-1)
    named_ok = np.isfinite(named[:, :, pos]).all(axis=-1)
    rows = []

    def add(k, runs, group, kind, size, note=""):
        for s, e in runs:
            seg = size[s:e + 1]
            value = float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan
            rows.append((first + s, first + e, ref_ids[k], group, kind, value, note))

    for k in range(ref.shape[1]):
        far = np.nan_to_num(d_pos[:, k], nan=0.0) > p.position_mm
        add(k, _runs(far, p.merge_frames), "geometry", "position", d_pos[:, k])
        ht = (np.nan_to_num(d_pos[:, k], nan=np.inf) <= p.position_mm) & (np.nan_to_num(d_ht[:, k]) > p.nodes_mm)
        add(k, _runs(ht, p.merge_frames), "geometry", "nodes", d_ht[:, k], "head / tail")
        for mask, kind in ((ref_ok[:, k] & ~aligned_ok[:, k], f"only {ref_name}"),
                           (~ref_ok[:, k] & named_ok[:, k], f"only {other_name}")):
            runs = [r for r in _runs(mask, p.merge_frames) if r[1] - r[0] + 1 >= p.min_coverage_frames]
            add(k, runs, "coverage", kind, np.zeros(len(mask)))
        if link[k] >= 0:
            known = (perm[:, k] >= 0) & (margin >= p.ambiguous_mm)
            state = np.where(known, (perm[:, k] != link[k]).astype(float), np.nan)
            wrong = _forward_fill(state) > 0.5
            runs = [r for r in _runs(wrong, p.merge_frames) if r[1] - r[0] + 1 >= p.min_swap_frames]
            add(k, runs, "identity", "swap", d_named[:, k], f"{other_name}'s fish of this name is elsewhere")
    df = pd.DataFrame(rows, columns=["start", "end", "ref_fish", "group", "kind", "size_mm", "note"])
    df["other"] = other_name
    return df


# ---------------------------------------------------------------------------
# State: names, links, changes, verdicts (no Qt, no video)


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for c in columns:
            if c not in df:
                df[c] = ""
        return df[columns]
    return pd.DataFrame(columns=columns)


class ReconState:
    """The comparison of ``recon_dir/recon_versions.npz`` with the reference folder's names."""

    def __init__(self, folder: Path, recon_dir: Path, params: ReconParams | None = None, fps: float = 166.0):
        self.folder, self.recon_dir = Path(folder), Path(recon_dir)
        self.params = params or ReconParams()
        self.fps = float(fps)
        with np.load(self.recon_dir / RECON_FILE) as z:
            self.nodes = z["nodes"]
            self.source = z["source"]
            self.versions = [str(v) for v in z["versions"]]
            self.kinds = [str(k) for k in z["kinds"]]
            self.tags = [str(t) for t in z["tags"]]
            self.frames = z["frames"]
            self.fish_ids = [[str(f) for f in row] for row in z["fish_ids"]]
            self.machine_label = [[str(f) for f in row] for row in z["machine_label"]]
            self.node_names = [str(n) for n in z["node_names"]]
            self.edges = [tuple(int(a) for a in e) for e in z["edges"]]
            self.head, self.tail = int(z["head_node"]), int(z["tail_node"])
            self.pos_node = int(z["position_node"])
            start = int(z["analysis_start"]) if "analysis_start" in z.files else -1
        self.first = int(self.frames[0])
        self.reviewed_reference = self.kinds[0] in REVIEWED_KINDS
        self.ref_ids = self.fish_ids[0]
        pos = self.nodes[:, :, :, self.pos_node].astype(float)
        self.perm, self.margin = {}, {}
        for v in range(1, len(self.versions)):
            self.perm[v], self.margin[v] = match_to_reference(pos[0], pos[v])
        self.names: dict[str, str] = {f: "" for f in self.ref_ids}
        self.links: dict[int, np.ndarray] = {}
        self.link_frame = self.first
        self.analysis_start: int | None = start if start >= 0 else None
        self._load_naming()
        self.verdicts = _read(self.recon_dir / VERDICTS_FILE, VERDICT_COLUMNS)
        self.changes = self.compute_changes()

    # -- names -----------------------------------------------------------------
    @property
    def labels(self) -> list[str]:
        return [self.names.get(f) or f for f in self.ref_ids]

    def vindex(self, version: str) -> int:
        return self.versions.index(version)

    def _load_naming(self):
        from scripts.review.ledger import EVENT_COLUMNS, analysis_start, read_csv, static_labels

        events_path = self.folder / "human_events.csv"
        events = read_csv(events_path, EVENT_COLUMNS) if events_path.exists() else None
        if events is not None and len(events) and self.analysis_start is None:
            self.analysis_start = analysis_start(events)
        if self.reviewed_reference:
            # The answer key: its tracks are the reviewed fish, named for the whole recording.
            self.names = {f: f for f in self.ref_ids}
            self.link_frame = self.default_link_frame(self.analysis_start)
            self._relink(self.link_frame)
            return
        saved = _read(self.recon_dir / LABELS_FILE, LABEL_COLUMNS)
        if len(saved):
            ref = saved[saved["version"] == self.versions[0]]
            self.names.update({r.fish_id: r.label for r in ref.itertuples() if r.fish_id in self.names})
            self.link_frame = int(ref["frame"].iloc[0]) if len(ref) else self.first
            for v in range(1, len(self.versions)):
                link = np.full(len(self.ref_ids), -1, np.intp)
                for r in saved[saved["version"] == self.versions[v]].itertuples():
                    if r.ref_fish_id in self.ref_ids and r.fish_id in self.fish_ids[v]:
                        link[self.ref_ids.index(r.ref_fish_id)] = self.fish_ids[v].index(r.fish_id)
                self.links[v] = link
            return
        if events is not None and len(events):
            self.names.update(static_labels(events, self.ref_ids))
        self.link_frame = self.default_link_frame(self.analysis_start)
        self._relink(self.link_frame)

    def default_link_frame(self, start: int | None = None) -> int:
        """First frame from ``start`` where every version has every fish, well separated and unambiguous."""
        t0 = max(0, (start if start is not None else self.first) - self.first)
        p = self.params
        ref = self.nodes[0, :, :, self.pos_node].astype(float)
        sep = np.linalg.norm(ref[:, :, None] - ref[:, None, :], axis=-1)
        sep[:, np.arange(sep.shape[1]), np.arange(sep.shape[1])] = np.inf
        ok = np.isfinite(ref).all(axis=(1, 2)) & (np.nan_to_num(sep, nan=0.0).min(axis=(1, 2)) >= p.separation_mm)
        for v in self.perm:
            ok &= (self.perm[v] >= 0).all(axis=1) & (self.margin[v] >= p.ambiguous_mm)
        good = np.flatnonzero(ok[t0:])
        return self.first + t0 + (int(good[0]) if len(good) else 0)

    def _relink(self, frame: int) -> float:
        """Other versions take the reference's tracks at ``frame``; returns the largest distance (mm)."""
        t = int(np.clip(frame - self.first, 0, len(self.frames) - 1))
        worst = 0.0
        ref = self.nodes[0, t, :, self.pos_node].astype(float)
        for v, perm in self.perm.items():
            link = perm[t].copy()
            if (link < 0).any() and v in self.links:
                link = np.where(link < 0, self.links[v], link)
            self.links[v] = link
            other = gather(self.nodes[v, t:t + 1, :, self.pos_node], link)[0]
            with np.errstate(invalid="ignore"):
                d = np.linalg.norm(ref - other, axis=-1)
            worst = max(worst, float(np.nanmax(d)) if np.isfinite(d).any() else np.inf)
        self.link_frame = int(frame)
        return worst

    def set_name(self, k: int, label: str, frame: int) -> str:
        """Name reference fish ``k`` at ``frame``; every other version takes the names of its tracks there."""
        from scripts.review.ledger import assign_label

        if self.reviewed_reference:
            return (f"names are fixed: the reference '{self.versions[0]}' is the reviewed result "
                    f"(other versions took its names at frame {self.link_frame})")
        fish = self.ref_ids[k]
        self.names.update(assign_label(self.names, fish, label))
        worst = self._relink(frame)
        self.save_naming()
        self.changes = self.compute_changes()
        others = ", ".join(self.versions[1:])
        msg = f"{fish} = {label}; {others} took the names at frame {frame} (fish up to {worst:.1f} mm apart)"
        if not worst <= self.params.inherit_max_mm:
            msg += " - WARNING: the versions disagree here, name at another frame"
        return msg

    def save_naming(self):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = [(self.versions[0], f, f, self.names.get(f, ""), self.link_frame, now) for f in self.ref_ids]
        for v, link in self.links.items():
            for k, j in enumerate(link):
                if j >= 0:
                    rows.append((self.versions[v], self.fish_ids[v][j], self.ref_ids[k],
                                 self.names.get(self.ref_ids[k], ""), self.link_frame, now))
        pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(self.recon_dir / LABELS_FILE, index=False)

    # -- data --------------------------------------------------------------------
    def series(self, version: str, lo: int, hi: int, aligned: bool) -> np.ndarray:
        """Nodes ``(T, K, N, 3)`` of rows ``lo:hi`` in reference-fish order. ``aligned``: matched to the
        reference frame by frame (geometry only); otherwise each version's own named tracks."""
        v = self.vindex(version)
        nodes = self.nodes[v, lo:hi]
        if v == 0:
            return nodes.astype(float)
        return gather(nodes, self.perm[v][lo:hi] if aligned else self.links[v])

    def sources(self, version: str, t: int, aligned: bool) -> np.ndarray:
        v = self.vindex(version)
        if v == 0:
            return self.source[0, t]
        idx = self.perm[v][t] if aligned else self.links[v]
        out = self.source[v, t][np.clip(idx, 0, None)].copy()
        out[idx < 0] = -1
        return out

    def speed(self, nodes: np.ndarray, half: int = 2) -> np.ndarray:
        p = nodes[:, :, self.pos_node]
        out = np.full(p.shape[:2], np.nan)
        if len(p) > 2 * half:
            out[half:-half] = np.linalg.norm(p[2 * half:] - p[:-2 * half], axis=-1) / (2 * half) * self.fps
        return out

    # -- changes and verdicts --------------------------------------------------------
    def compute_changes(self) -> pd.DataFrame:
        parts = []
        kw = dict(pos=self.pos_node, head=self.head, tail=self.tail, first=self.first, ref_ids=self.ref_ids,
                  ref_name=self.versions[0], params=self.params)
        ref = self.nodes[0].astype(float)
        for v in self.perm:
            parts.append(disagreements(ref, self.nodes[v].astype(float), self.perm[v], self.margin[v],
                                       self.links[v], other_name=self.versions[v], **kw))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CHANGE_COLUMNS)
        df["fish"] = df["ref_fish"].map(lambda f: self.names.get(f) or f)
        speeds = np.full(len(df), np.nan)
        for i, r in enumerate(df.itertuples(index=False)):
            k = self.ref_ids.index(r.ref_fish)
            lo, hi = max(0, r.start - self.first - 2), min(len(self.frames), r.end - self.first + 3)
            seg = self.speed(self.nodes[0, lo:hi, k:k + 1].astype(float))
            if np.isfinite(seg).any():
                speeds[i] = np.nanmax(seg)
        df["speed_mm_s"] = speeds
        df = df[CHANGE_COLUMNS + ["speed_mm_s"]].sort_values(["start", "group"], kind="stable").reset_index(drop=True)
        given = {self._key(v.start, v.end, v.ref_fish, v.kind, v.other): str(v.verdict)
                 for v in self.verdicts.itertuples(index=False)}
        df["verdict"] = [given.get(self._key(r.start, r.end, r.ref_fish, r.kind, r.other), "")
                         for r in df.itertuples(index=False)]
        return df

    @staticmethod
    def _key(start, end, ref_fish, kind, other) -> tuple:
        return int(start), int(end), str(ref_fish), str(kind), str(other)

    def set_verdict(self, row: int, verdict: str):
        """Record (or with ``""`` clear) the reviewer's verdict on change ``row``."""
        if verdict and verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VERDICTS)}")
        r = self.changes.loc[row]
        key = self._key(r["start"], r["end"], r["ref_fish"], r["kind"], r["other"])
        keep = [self._key(v.start, v.end, v.ref_fish, v.kind, v.other) != key
                for v in self.verdicts.itertuples(index=False)]
        verdicts = self.verdicts[keep] if len(self.verdicts) else self.verdicts
        if verdict:
            new = pd.DataFrame([[*map(str, key[:2]), key[2], key[3], key[4], verdict,
                                 time.strftime("%Y-%m-%d %H:%M:%S")]], columns=VERDICT_COLUMNS)
            verdicts = pd.concat([verdicts, new], ignore_index=True)
        self.verdicts = verdicts.reset_index(drop=True)
        self.verdicts.to_csv(self.recon_dir / VERDICTS_FILE, index=False)
        self.changes.loc[row, "verdict"] = verdict

    def verdict_counts(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.verdicts["verdict"].value_counts().items()} if len(self.verdicts) else {}


def summary(state: ReconState) -> pd.DataFrame:
    """Disagreement count, frames and verdicts per kind."""
    ch = state.changes
    if not len(ch):
        return pd.DataFrame()
    out = ch.assign(frames=ch["end"] - ch["start"] + 1).groupby(["other", "group", "kind"]).agg(
        entries=("start", "size"), frames=("frames", "sum"), max_mm=("size_mm", "max"))
    for code in VERDICTS:
        out[VERDICTS[code]] = ch.assign(x=ch["verdict"] == code).groupby(["other", "group", "kind"])["x"].sum()
    return out.reset_index()


def error_table(state: ReconState) -> pd.DataFrame:
    """With a reviewed reference: per kind, the entries (runs; a swap lasts until undone) and the analysed
    fish-frames where each other version disagrees
    with the review, and where several do at once (rows overlap: ``swap`` frames are usually also
    ``position`` frames). Before the analysis start nothing counts."""
    ch = state.changes
    n_frames, n_fish = len(state.frames), len(state.ref_ids)
    start = max(0, (state.analysis_start or state.first) - state.first)
    analysed = (n_frames - start) * n_fish
    others = state.versions[1:]
    rows = []
    kinds = sorted(ch["kind"].unique()) if len(ch) else []
    for kind in kinds:
        masks, entries = {}, {}
        for other in others:
            m = np.zeros((n_frames, n_fish), bool)
            sel = ch[(ch["kind"] == kind) & (ch["other"] == other) & (ch["end"] >= state.first + start)]
            for r in sel.itertuples(index=False):
                m[r.start - state.first:r.end - state.first + 1, state.ref_ids.index(r.ref_fish)] = True
            masks[other] = m[start:]
            entries[other] = len(sel)
        row = {"kind": kind}
        for other in others:
            row[f"{other} entries"] = entries[other]
            row[f"{other} fish-frames"] = int(masks[other].sum())
            row[f"{other} %"] = round(100 * masks[other].sum() / max(1, analysed), 3)
        if len(others) == 2:
            a, b = (masks[o] for o in others)
            row[f"only {others[0]}"] = int((a & ~b).sum())
            row[f"only {others[1]}"] = int((b & ~a).sum())
            row["both"] = int((a & b).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _encode(perm: np.ndarray) -> np.ndarray:
    """``(..., K)`` track permutations -> one integer each (base ``K``)."""
    k = perm.shape[-1]
    return (perm * k ** np.arange(k - 1, -1, -1)).sum(axis=-1)


def switches(perm: np.ndarray, margin: np.ndarray, link: np.ndarray, labels: list[str], first_frame: int,
             params: ReconParams | None = None) -> pd.DataFrame:
    """ID switches of one version against a reference, from its per-frame matching.

    ``perm (F, K)`` / ``margin (F,)`` as ``match_to_reference`` returns them, ``link (K,)`` the naming at
    the start (the track holding each reference fish), ``labels`` the reference fish names and
    ``first_frame`` the frame number of row 0. Ambiguous frames (margin < ``ambiguous_mm``, or a fish
    missing) keep the previous matching, and runs shorter than ``min_swap_frames`` are absorbed into the
    run before them, as for ``swap`` entries. ``correct_after``: the matching after the switch is ``link``.
    """
    p = params or ReconParams()
    known = (perm >= 0).all(axis=1) & (margin >= p.ambiguous_mm)
    columns = ["frame", "fish", "correct_after"]
    if not known.any():
        return pd.DataFrame(columns=columns)
    code = _forward_fill(np.where(known, _encode(perm), np.nan).astype(float))
    code[:np.flatnonzero(known)[0]] = code[np.flatnonzero(known)[0]]
    starts = np.flatnonzero(np.r_[True, np.diff(code) != 0])
    ends = np.r_[starts[1:], len(code)]
    runs: list[list] = []
    for s, e in zip(starts, ends):
        if runs and (e - s < p.min_swap_frames or code[s] == runs[-1][2]):
            runs[-1][1] = e
        else:
            runs.append([s, e, code[s]])
    n = perm.shape[1]
    decode = lambda c: [(int(c) // n ** (n - 1 - k)) % n for k in range(n)]   # noqa: E731
    right = float(_encode(np.asarray(link)))
    rows = []
    for (_, _, before), (s, _, after) in zip(runs[:-1], runs[1:]):
        changed = [labels[k] for k, (a, b) in enumerate(zip(decode(before), decode(after))) if a != b]
        rows.append((first_frame + int(s), "+".join(sorted(changed)), after == right))
    return pd.DataFrame(rows, columns=columns)


def switch_events(state: ReconState, version: str) -> pd.DataFrame:
    """Every frame after the analysis start where ``version``'s matching to the reference changes.

    Each row is an ID switch - or the switch back (see ``switches``). Unlike swap fish-frames this does
    not grow with how long a wrong name lasts.
    """
    v = state.vindex(version)
    t0 = max(0, (state.analysis_start or state.first) - state.first)
    return switches(state.perm[v][t0:], state.margin[v][t0:], state.links[v], state.labels,
                    state.first + t0, state.params)


def switch_table(state: ReconState, near_s: float = 1.0) -> pd.DataFrame:
    """With a reviewed reference: per other version, the ID switches after the analysis start.

    ``only this`` counts switches with no switch of any other version within ``near_s`` seconds - the
    errors a method makes on its own rather than at an encounter every method finds hard.
    """
    others = state.versions[1:]
    events = {o: switch_events(state, o) for o in others}
    near = round(near_s * state.fps)
    t0 = max(0, (state.analysis_start or state.first) - state.first)
    rows = []
    for o in others:
        ev = events[o]
        frames = ev["frame"].to_numpy()
        rest = np.concatenate([events[x]["frame"].to_numpy() for x in others if x != o] or [np.array([])])
        alone = int(sum(not (np.abs(rest - f) <= near).any() for f in frames)) if len(rest) else len(frames)
        v = state.vindex(o)
        perm, margin = state.perm[v][t0:], state.margin[v][t0:]
        known = (perm >= 0).all(axis=1) & (margin >= state.params.ambiguous_mm)
        right = _encode(np.asarray(state.links[v])[None])[0]
        ok = _forward_fill(np.where(known, _encode(perm) == right, np.nan).astype(float)) > 0.5
        longest = max((e - s + 1 for s, e in _runs(ok)), default=0)
        row = {"version": o, "switches": len(ev), "into wrong": int((~ev["correct_after"].astype(bool)).sum()),
               "back to right": int(ev["correct_after"].astype(bool).sum()), f"only this (±{near_s:g} s)": alone}
        for name in state.labels:
            row[f"with {name}"] = int(ev["fish"].str.split("+").map(lambda f, n=name: n in f).sum())
        row["longest right run s"] = round(longest / state.fps, 1)
        row["ambiguous %"] = round(100 * float((~known).mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def geometry_table(state: ReconState, steps_mm: tuple[float, ...] = (10.0, 30.0)) -> pd.DataFrame:
    """Per version, reference-free geometry after the analysis start, each on its own tracks.

    Coverage of the position node, one-frame steps of the position node larger than ``steps_mm`` (a
    teleport or an unrepaired swap), and how the nodes were obtained (``node_source``: triangulated / one
    view / body model). The reference row includes the reviewer's edits, so it is context, not a rival.
    """
    t0 = max(0, (state.analysis_start or state.first) - state.first)
    rows = []
    for v, name in enumerate(state.versions):
        pos = state.nodes[v, t0:, :, state.pos_node].astype(float)
        ok = np.isfinite(pos).all(axis=-1)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
        both = ok[1:] & ok[:-1]
        src = state.source[v, t0:]
        present = src >= 0
        row = {"version": name, "kind": state.kinds[v], "coverage %": round(100 * float(ok.mean()), 2)}
        for mm in steps_mm:
            row[f"steps > {mm:g} mm"] = int((both & (np.nan_to_num(step) > mm)).sum())
        row["nodes triangulated %"] = round(100 * float((src[present] == 0).mean()), 2) if present.any() else 0.0
        row["nodes one view %"] = round(100 * float((src[present] == 1).mean()), 2) if present.any() else 0.0
        row["nodes body model %"] = round(100 * float((src[present] == 2).mean()), 2) if present.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", type=Path, help="Review package folders; the first is the reference.")
    p.add_argument("--kinds", nargs="+", default=None, choices=KINDS, help="Per folder (default: initial).")
    p.add_argument("--names", nargs="+", default=None, help="Short version names (default A, B, ...).")
    p.add_argument("--out", type=Path, default=None, help="Default: <reference>/recon_vs_<other>/.")
    args = p.parse_args(argv)
    kinds = args.kinds or ["initial"] * len(args.folders)
    out = build(args.folders, kinds, args.out, args.names)
    state = ReconState(args.folders[0], out)
    print(json.dumps({"params": asdict(state.params), "link_frame": state.link_frame,
                      "labels": dict(zip(state.ref_ids, state.labels))}, indent=2))
    print(summary(state).to_string(index=False))
    if state.reviewed_reference:
        print(error_table(state).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
