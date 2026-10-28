"""Human review ledger: events, playback coverage, sessions, apply and compare.

Pure numpy / pandas so it runs on a review laptop without the GPU stack.

Events never edit the pipeline output; :func:`apply_events` applies them.
``fish`` names a 3D track (geometry ``fish_id`` such as ``ID_2``).

Identity (a biological fish = a corrected track):

* ``swap``  - ``fish = "ID_a|ID_b"``: from ``frame`` on, the 3D tracker had
  exchanged these two tracks, so their data is exchanged back. Swaps compose
  in frame order, whatever order they were entered in.
* ``label`` - name of a corrected fish (``F`` / ``M1`` / ``M2``; ``-`` = not
  a fish), one name for the whole recording. A corrected fish is called by
  the track it starts on (no swap before the first frame). The most recently
  written label of a fish wins and a name is held by one fish only.

Per-track flags (they describe the data of a track, so they stay with the
track, not with the fish):

* ``flip``      - head/tail swapped back on ``frame``..``frame_end`` (usually
  one ``orientation_segment``); overlapping flips cancel (XOR).
* ``uncertain`` - ``1`` from ``frame`` on = reviewer is not sure.
* ``bad3d``     - ``1`` from ``frame`` on = 3D position not trustworthy.
* ``nodes_bad`` - ``1`` from ``frame`` on = the body is wrong here (tail too
  short, orientation off, centre shifted): mouth, tail and centre are
  re-estimated from the pectoral node and the body direction around the run
  (:mod:`scripts.review.geometry`).

Per-view geometry edits (``frame``..``frame_end``, ``value`` = view name, see
:mod:`scripts.review.geometry`; the 3D is recomputed with them):
``view_flip`` (head/tail of the track reversed in that view), ``view_drop``
(that view left out of the track's triangulation) and ``view_swap``
(``fish`` = ``ID_a|ID_b``: the two tracks' detections exchanged in that view).

Close co-moving intervals: ``interval`` (``value`` = interval id, ``fish`` =
member tracks, ``frame``..``frame_end``; creates or moves one),
``interval_del`` (``value`` = id) and ``confirm`` (reviewer keyframe:
identities and head/tail at ``frame`` are right; ``value`` holds the
displayed head and tail positions).

``analysis_start`` (``fish`` = ``*``): the most recently written one marks
the first usable frame (hands or a net in the tank before it). Frames before
it are left out of the watched fraction and flagged ``analysed = False``.

Events created by one key press share a ``group`` and are undone or removed
together (removed rows are appended to ``human_events_removed.csv``).
``frame_marked`` is the frame the reviewer was looking at; ``frame`` is where
the change applies (swaps snap to the closest approach of the two tracks
between the previously shown frame and the marked one).

Coverage rows record which source-frame intervals were shown (played at any
speed or stepped through; jumps are not recorded) and at which playback speed,
so "every frame watched" can be verified later.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["event_id", "group", "kind", "fish", "frame", "frame_end", "frame_marked", "value", "risk_id",
                 "session", "created", "note"]
COVERAGE_COLUMNS = ["session", "frame_start", "frame_end", "speed", "mode", "wall_start", "wall_end"]
SESSION_COLUMNS = ["session", "host", "started", "ended", "seconds", "events_added"]
FLAG_KINDS = ("uncertain", "bad3d", "nodes_bad")          # "state from frame on" per track
TRACK_KINDS = ("flip",) + FLAG_KINDS
VIEW_KINDS = ("view_flip", "view_drop", "view_swap")
INTERVAL_KINDS = ("interval", "interval_del", "confirm")
CORRECTION_KINDS = ("swap",) + TRACK_KINDS + VIEW_KINDS
KINDS = ("label", "swap") + TRACK_KINDS + VIEW_KINDS + INTERVAL_KINDS + ("analysis_start",)
ALL_FISH = "*"
PAIR_SEP = "|"
DEFAULTS = {"label": "", "swap": "", "flip": "0", "uncertain": "0", "bad3d": "0", "nodes_bad": "0",
            "analysis_start": "", "view_flip": "", "view_drop": "", "view_swap": "", "interval": "",
            "interval_del": "", "confirm": ""}
LEDGER_FILES = ("human_events.csv", "coverage.csv", "session_log.csv")
REMOVED_FILE = "human_events_removed.csv"
WATCHED_MAX_SPEED: float | None = None   # None: every shown frame counts, whatever the speed
SAVE_RETRIES = 20                        # network drive: a reader (indexer, antivirus) can hold the file briefly
SAVE_RETRY_SECONDS = 0.15


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write via a temporary file. On a network drive the replace can be refused for a moment while
    another process (search indexer, antivirus, a second reader) holds the target; retry for up to ~7 s."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    for attempt in range(SAVE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == SAVE_RETRIES - 1:
                raise
            time.sleep(SAVE_RETRY_SECONDS * (1 + attempt // 5))


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in columns:
        if c not in df:
            df[c] = ""
    return df[columns]


# ---------------------------------------------------------------------------
# State over time


def state_at(events: pd.DataFrame, kind: str, fish: str, frame: int) -> str:
    """Value of a "from frame on" ``kind`` for ``fish`` at ``frame`` (last event at or before it)."""
    e = events[(events["kind"] == kind) & (events["fish"] == fish)]
    if e.empty:
        return DEFAULTS[kind]
    f = e["frame"].astype(int).to_numpy()
    before = np.flatnonzero(f <= frame)
    if len(before) == 0:
        return DEFAULTS[kind]
    # Latest frame wins; among equal frames the later-written row wins.
    order = np.lexsort((before, f[before]))
    return str(e.iloc[before[order[-1]]]["value"])


def _written_order(e: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted by frame; equal frames keep the order they were written in."""
    e = e.copy()
    e["_f"] = e["frame"].astype(int)
    e["_i"] = np.arange(len(e))
    return e.sort_values(["_f", "_i"], kind="stable")


def state_series(events: pd.DataFrame, kind: str, fish: str, frames: np.ndarray) -> np.ndarray:
    """Vectorised :func:`state_at` over a sorted frame array (object array of strings)."""
    out = np.full(len(frames), DEFAULTS[kind], dtype=object)
    e = events[(events["kind"] == kind) & (events["fish"] == fish)]
    if e.empty:
        return out
    e = _written_order(e)
    starts = np.searchsorted(frames, e["_f"].to_numpy(), side="left")
    for s, value in zip(starts, e["value"].astype(str)):
        out[s:] = value
    return out


def analysis_start(events: pd.DataFrame) -> int | None:
    """First usable frame (the most recently written ``analysis_start`` event), or None if not marked."""
    e = events[events["kind"] == "analysis_start"]
    return int(e.iloc[-1]["frame"]) if len(e) else None


def static_labels(events: pd.DataFrame, fish_ids: list[str]) -> dict[str, str]:
    """Name of every corrected fish: its most recently written ``label``. A name stays with one fish
    (should a ledger give one name to two fish, the most recently written keeps it)."""
    out = {f: "" for f in fish_ids}
    e = events[events["kind"] == "label"]
    for fish, value in zip(e["fish"], e["value"].astype(str)):
        if fish not in out:
            continue
        if value not in ("", "-"):
            for other in out:
                if other != fish and out[other] == value:
                    out[other] = ""
        out[fish] = value
    return out


def assign_label(current: dict[str, str], fish: str, label: str) -> dict[str, str]:
    """New labels after giving ``label`` to ``fish``; a fish already holding it gets ``fish``'s old label.

    Returns only the fish whose label changes.
    """
    changes = {fish: label}
    if label not in ("", "-"):
        for other, other_label in current.items():
            if other != fish and other_label == label:
                changes[other] = current.get(fish, "")
    return {f: v for f, v in changes.items() if current.get(f, "") != v}


def swap_list(events: pd.DataFrame, fish_ids: list[str]) -> list[tuple[int, int, int]]:
    """``(frame, track_a, track_b)`` of every swap in application order (frame, then write order)."""
    e = events[events["kind"] == "swap"]
    if e.empty:
        return []
    index = {f: k for k, f in enumerate(fish_ids)}
    e = _written_order(e)
    out = []
    for f, pair in zip(e["_f"], e["fish"]):
        a, _, b = str(pair).partition(PAIR_SEP)
        if a in index and b in index and a != b:
            out.append((int(f), index[a], index[b]))
    return out


def owner_series(n_fish: int, swaps: list[tuple[int, int, int]], frames: np.ndarray) -> np.ndarray:
    """``(len(frames), n_fish)``: the corrected fish whose data sits on each track at each frame."""
    owner = np.tile(np.arange(n_fish), (len(frames), 1))
    for f, a, b in swaps:
        s = np.searchsorted(frames, f, side="left")
        owner[s:, [a, b]] = owner[s:, [b, a]]
    return owner


def owner_at(n_fish: int, swaps: list[tuple[int, int, int]], frame: int) -> np.ndarray:
    """:func:`owner_series` at one frame."""
    owner = np.arange(n_fish)
    for f, a, b in swaps:
        if f <= frame:
            owner[[a, b]] = owner[[b, a]]
    return owner


def flip_ranges(events: pd.DataFrame, fish: str, last_frame: int) -> list[tuple[int, int]]:
    """Inclusive frame ranges of the ``flip`` events of one track (no end = until ``last_frame``)."""
    e = events[(events["kind"] == "flip") & (events["fish"] == fish) & (events["value"].astype(str) == "1")]
    out = []
    for f, fe in zip(e["frame"], e["frame_end"]):
        end = str(fe).strip()
        out.append((int(f), int(end) if end not in ("", "nan") else int(last_frame)))
    return out


def flip_series(events: pd.DataFrame, fish: str, frames: np.ndarray) -> np.ndarray:
    """Human head/tail flip per frame of one track (overlapping flips cancel)."""
    out = np.zeros(len(frames), dtype=bool)
    for a, b in flip_ranges(events, fish, int(frames[-1]) if len(frames) else 0):
        out[np.searchsorted(frames, a, side="left"):np.searchsorted(frames, b, side="right")] ^= True
    return out


def snap_frame(frames: np.ndarray, distance: np.ndarray, lo: int, hi: int) -> int:
    """Frame of minimum ``distance`` in ``(lo, hi]``; ``hi`` if the window holds no finite value."""
    if hi <= lo:
        return hi
    a, b = np.searchsorted(frames, lo, side="right"), np.searchsorted(frames, hi, side="right")
    seg = distance[a:b]
    if b <= a or not np.isfinite(seg).any():
        return hi
    return int(frames[a + int(np.nanargmin(seg))])


# ---------------------------------------------------------------------------
# Coverage


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in intervals):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def watched_intervals(coverage: pd.DataFrame, max_speed: float | None = WATCHED_MAX_SPEED) -> list[tuple[int, int]]:
    """Merged source-frame intervals that were shown; ``max_speed`` (default: none) leaves out faster playback."""
    if coverage.empty:
        return []
    c = coverage if max_speed is None else coverage[coverage["speed"].astype(float) <= max_speed]
    return merge_intervals(list(zip(c["frame_start"].astype(int), c["frame_end"].astype(int))))


def covered_fraction(intervals: list[tuple[int, int]], first: int, last: int) -> float:
    total = last - first + 1
    inside = sum(max(0, min(b, last) - max(a, first) + 1) for a, b in intervals)
    return inside / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Ledger object (used by the GUI)


@dataclass
class Ledger:
    folder: Path
    backup_dir: Path | None = None
    session: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4])

    def __post_init__(self):
        self.folder = Path(self.folder)
        self.events = read_csv(self.folder / "human_events.csv", EVENT_COLUMNS)
        self.coverage = read_csv(self.folder / "coverage.csv", COVERAGE_COLUMNS)
        self.sessions = read_csv(self.folder / "session_log.csv", SESSION_COLUMNS)
        self.started = time.time()
        self.events_added = 0
        self.sessions = pd.concat([self.sessions, pd.DataFrame([{
            "session": self.session, "host": socket.gethostname(), "started": _now(), "ended": "",
            "seconds": "0", "events_added": "0"}])], ignore_index=True)
        self._save("session_log.csv", self.sessions)

    # -- persistence -------------------------------------------------------
    def _save(self, name: str, df: pd.DataFrame) -> None:
        atomic_write_csv(df, self.folder / name)
        if self.backup_dir is not None:
            try:
                self.backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.folder / name, self.backup_dir / name)
            except OSError:
                pass  # backup location offline (e.g. laptop away from the network)

    def save_events(self) -> None:
        self._save("human_events.csv", self.events)

    def close(self) -> None:
        i = self.sessions.index[self.sessions["session"] == self.session]
        self.sessions.loc[i, "ended"] = _now()
        self.sessions.loc[i, "seconds"] = str(round(time.time() - self.started))
        self.sessions.loc[i, "events_added"] = str(self.events_added)
        self._save("session_log.csv", self.sessions)

    # -- events --------------------------------------------------------------
    def add(self, kind: str, changes: dict[str, str], frame: int, frame_marked: int, risk_id: str = "",
            note: str = "", frame_end: int | None = None) -> str:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        group = uuid.uuid4().hex[:8]
        rows = [{"event_id": uuid.uuid4().hex[:10], "group": group, "kind": kind, "fish": fish,
                 "frame": str(int(frame)), "frame_end": "" if frame_end is None else str(int(frame_end)),
                 "frame_marked": str(int(frame_marked)), "value": str(value),
                 "risk_id": risk_id, "session": self.session, "created": _now(), "note": note}
                for fish, value in changes.items()]
        if not rows:
            return ""
        self.events = pd.concat([self.events, pd.DataFrame(rows, columns=EVENT_COLUMNS)], ignore_index=True)
        self.events_added += len(rows)
        self.save_events()
        return group

    def undo(self) -> pd.DataFrame:
        """Remove the most recent event group written in this session; returns the removed rows."""
        mine = self.events[self.events["session"] == self.session]
        if mine.empty:
            return mine
        return self.remove_group(mine.iloc[-1]["group"], log=False)

    def remove_group(self, group: str, log: bool = True) -> pd.DataFrame:
        """Remove every row of one event group (any session); logged to ``human_events_removed.csv``."""
        removed = self.events[self.events["group"] == group]
        if removed.empty:
            return removed
        self.events = self.events[self.events["group"] != group].reset_index(drop=True)
        self.events_added -= int((removed["session"] == self.session).sum())
        self.save_events()
        if log:
            gone = removed.copy()
            gone["note"] = (gone["note"].astype(str) + f" [removed {_now()} in {self.session}]").str.strip()
            old = read_csv(self.folder / REMOVED_FILE, EVENT_COLUMNS)
            self._save(REMOVED_FILE, pd.concat([old, gone[EVENT_COLUMNS]], ignore_index=True))
        return removed

    def add_coverage(self, start: int, end: int, speed: float, mode: str, wall_start: float, wall_end: float) -> None:
        row = {"session": self.session, "frame_start": str(min(start, end)), "frame_end": str(max(start, end)),
               "speed": f"{speed:g}", "mode": mode,
               "wall_start": time.strftime("%H:%M:%S", time.localtime(wall_start)),
               "wall_end": time.strftime("%H:%M:%S", time.localtime(wall_end))}
        self.coverage = pd.concat([self.coverage, pd.DataFrame([row], columns=COVERAGE_COLUMNS)], ignore_index=True)
        self._save("coverage.csv", self.coverage)


def pid_alive(pid: int) -> bool:
    """True if a process with this id runs on this computer (never signals it)."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ReviewLock:
    """``review.lock`` guarding against two computers editing one ledger."""

    def __init__(self, folder: Path):
        self.path = Path(folder) / "review.lock"

    def holder(self) -> dict | None:
        """Who else has the folder open; None if nobody, this process, or a crashed process on this computer."""
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if info.get("host") == socket.gethostname():
            pid = info.get("pid")
            if pid == os.getpid() or not isinstance(pid, int) or not pid_alive(pid):
                return None
        return info

    def acquire(self) -> None:
        self.path.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "since": _now()}),
                             encoding="utf-8")

    def release(self) -> None:
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            if info.get("host") == socket.gethostname() and info.get("pid") == os.getpid():
                self.path.unlink()
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Apply and compare


def apply_events(data: dict, events: pd.DataFrame, coverage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table, one row per frame and corrected fish, with the human corrections applied.

    ``data`` is the loaded ``review_data.npz``. ``fish_id`` is the corrected
    fish, ``source_track`` the 3D track its data comes from at that frame
    (they differ after a swap). Human head/tail flips swap the head and tail
    node coordinates.
    """
    frames = data["frames"]
    fish_ids = [str(f) for f in data["fish_ids"]]
    n_fish = len(fish_ids)
    node_names = [str(n) for n in data["node_names"]]
    head, tail = int(data["head_node"]), int(data["tail_node"])
    watched = watched_intervals(coverage) if coverage is not None else []
    watched_mask = np.zeros(len(frames), dtype=bool)
    for a, b in watched:
        watched_mask[np.searchsorted(frames, a):np.searchsorted(frames, b, side="right")] = True
    start = analysis_start(events)
    analysed = frames >= start if start is not None else np.ones(len(frames), dtype=bool)
    labels = static_labels(events, fish_ids)
    owner = owner_series(n_fish, swap_list(events, fish_ids), frames)     # (F, K) fish on each track
    source = np.argsort(owner, axis=1)                                    # (F, K) track of each fish
    rows = np.arange(len(frames))

    # Flags and flipped nodes per track, then gathered per corrected fish. Completed nodes (every node in
    # 3D, review data v4+) are used where present, with their source (0 triangulated, 1 one view + body
    # model, 2 body model only, -1 missing).
    nodes = np.asarray(data.get("nodes3d_filled", data["nodes3d"])).astype(float).copy()   # (F, K, N, 3)
    node_src = np.asarray(data["node_source"]).copy() if "node_source" in data else \
        np.where(np.isfinite(nodes).all(axis=-1), 0, -1).astype(np.int8)               # (F, K, N)
    flags = {kind: np.zeros((len(frames), n_fish), dtype=bool) for kind in TRACK_KINDS}
    for k, fid in enumerate(fish_ids):
        flip = flip_series(events, fid, frames)
        flags["flip"][:, k] = flip
        nodes[flip, k, head], nodes[flip, k, tail] = nodes[flip, k, tail], nodes[flip, k, head].copy()
        node_src[flip, k, head], node_src[flip, k, tail] = node_src[flip, k, tail], node_src[flip, k, head].copy()
        for kind in FLAG_KINDS:
            flags[kind][:, k] = state_series(events, kind, fid, frames) == "1"

    parts = []
    for c, fid in enumerate(fish_ids):
        src = source[:, c]
        xyz = np.asarray(data["xyz"])[rows, src]
        df = pd.DataFrame({
            "frame": frames, "fish_id": fid, "label": labels[fid],
            "source_track": np.array(fish_ids)[src],
            "X": xyz[:, 0], "Y": xyz[:, 1], "Z": xyz[:, 2],
            "is_interpolated": np.asarray(data["is_interpolated"])[rows, src],
            "head_flipped_by_human": flags["flip"][rows, src],
            "uncertain": flags["uncertain"][rows, src],
            "bad3d": flags["bad3d"][rows, src],
            "nodes_bad": flags["nodes_bad"][rows, src],
            "watched": watched_mask,
            "analysed": analysed,
        })
        fish_nodes = nodes[rows, src]                                     # (F, N, 3)
        fish_source = node_src[rows, src]                                 # (F, N)
        for n, name in enumerate(node_names):
            for a, axis in enumerate("XYZ"):
                df[f"{name}_{axis}"] = fish_nodes[:, n, a]
            df[f"{name}_source"] = fish_source[:, n]
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).sort_values(["frame", "fish_id"], kind="stable")
    return out.reset_index(drop=True)


def compare_with_machine(events: pd.DataFrame, risks: pd.DataFrame, coverage: pd.DataFrame,
                         first: int, last: int, tolerance: int = 0) -> dict:
    """Summary of human corrections vs. machine risk windows (from the analysis start, if marked).

    Corrections are swaps, flips and flags; naming the fish (``label``) is not.
    """
    start = analysis_start(events)
    if start is not None:
        first = max(first, start)
        risks = risks[risks["frame_end"].astype(int) >= first]
    ev = events[events["kind"].isin(CORRECTION_KINDS)].copy()
    ev["frame_i"] = ev["frame"].astype(int) if len(ev) else pd.Series(dtype=int)
    corr = ev[(ev["frame_i"] >= first) & (ev["frame_i"] <= last)]
    groups = corr.drop_duplicates("group")
    starts = risks["frame_start"].astype(int).to_numpy() - tolerance
    ends = risks["frame_end"].astype(int).to_numpy() + tolerance

    def in_risk(f: int) -> bool:
        return bool(((starts <= f) & (f <= ends)).any())

    hit = [in_risk(int(f)) for f in groups["frame_i"]]
    watched = watched_intervals(coverage)
    risk_hit = [bool(((groups["frame_i"] >= a) & (groups["frame_i"] <= b)).any()) for a, b in zip(starts, ends)]
    return {
        "frames": last - first + 1,
        "watched_fraction": round(covered_fraction(watched, first, last), 4),
        "risk_windows": int(len(risks)),
        "risk_windows_with_correction": int(sum(risk_hit)),
        "corrections": int(len(groups)),
        "corrections_by_kind": {k: int(v) for k, v in groups["kind"].value_counts().items()},
        "corrections_inside_risk": int(sum(hit)),
        "corrections_outside_risk": int(len(hit) - sum(hit)),
        "risk_recall": round(sum(hit) / len(hit), 4) if hit else None,
    }
