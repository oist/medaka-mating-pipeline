"""Judge two 3D reconstructions of one experiment frame by frame, fish by fish (data and grades; no Qt).

Reads a comparison folder written by :func:`scripts.review.recon_compare.build` with three versions:
the reference (a finished review, e.g. ``reviewed``), ``old`` and ``new`` (e.g. ``m3dfull``).

**Grades.** The reviewer watches the whole recording (or any part of it) and grades **each version
separately**, per fish, for its 3D IDs and its 3D body parts, on any frame range: ``ok`` (correct),
``minor`` (a small error - no effect on the analysis), ``major`` (a clear error), or ``unsure``. Which
version is better follows from the two grades (``compare``): old ok / new minor = old better, by a little;
old ok / new major = old better, clearly; the same grade = no difference. A grade is a segment - fish,
aspect (``id`` / ``body``), version (``old`` / ``new``), first and last frame - applied in the order given,
so a later one paints over an earlier one (an empty grade erases). Segments are appended to
``<recon_dir>/recon_judgements.csv`` (``judged_on = 3D``: the grades are on the 3D reconstruction, the
camera views only help to check it), with the event that was open, if any, for reference. Notes cover a
frame range.

**Machine hints** (``recon_judge_events.csv``, optional to follow) - every place where any of the three
disagree:

* ``identity`` - an ID switch of old or new against the reference (``recon_compare.switch_events``).
  Switches of either version within ``switch_merge_s`` of each other are one event (an encounter both
  methods find hard), shown ``switch_pad_s`` before and after.
* ``body`` - per reference fish, frames where any node (mouth / pectoral / tail) of old and new, or of
  either and the reference, are more than ``body_mm`` apart, or where only one of old / new has 3D.
  Old and new are matched to the reference frame by frame here, so a swap does not count as a body error.

The GUI is :mod:`scripts.review.recon_judge_gui`.

CLI (hints and grade summary)::

    python -m scripts.review.recon_judge <reference folder> --recon-dir <reference>/recon_reviewed_old_m3dfull
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.review.recon_compare import ReconState, _read, _runs, switch_events

EVENTS_FILE = "recon_judge_events.csv"
JUDGE_FILE = "recon_judgements.csv"
GRADES = {"ok": "correct", "minor": "small error", "major": "clear error", "unsure": "unsure"}
GRADE_CODES = list(GRADES)       # grade arrays: 0 = not graded, i + 1 = GRADE_CODES[i]
RANK = {"ok": 0, "minor": 1, "major": 2}
EVENT_COLUMNS = ["event", "kind", "start", "end", "fish", "detail", "old_new_mm", "old_ref_mm", "new_ref_mm",
                 "old_jump_mm", "new_jump_mm"]
JUDGE_COLUMNS = ["fish", "aspect", "version", "start", "end", "grade", "event", "note", "judged_on", "created"]
ASPECTS = ("id", "body")
JUDGED = ("old", "new")          # the two versions that are graded
COMPARE = ("same", "old better, a little", "old better, clearly", "new better, a little", "new better, clearly",
           "unsure")
JUDGED_ON = "3D"                 # verdicts are on the 3D reconstruction; the camera views are an aid


@dataclass(frozen=True)
class JudgeParams:
    switch_merge_s: float = 1.0      # identity: switches of old / new this close are one event
    switch_pad_s: float = 0.5        # identity: shown this long before the first and after the last switch
    body_mm: float = 5.0             # body: any node this far apart between two of old / new / reference
    merge_frames: int = 17           # body: runs closer than this (~0.1 s) are one event
    min_body_frames: int = 3         # body: shorter runs are matching noise


def _nanmax(a: np.ndarray) -> float:
    return float(np.nanmax(a)) if np.isfinite(a).any() else np.nan


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Largest node distance ``(F, K)`` of two ``(F, K, N, 3)`` arrays (NaN where a node is missing)."""
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(a - b, axis=-1)
    out = np.full(d.shape[:2], np.nan)
    ok = np.isfinite(d).any(axis=-1)
    out[ok] = np.nanmax(d[ok], axis=-1)
    return out


class JudgeState:
    """Events and verdicts of one comparison folder; ``old`` / ``new`` default to versions 1 and 2."""

    def __init__(self, folder: Path, recon_dir: Path, *, old: str | None = None, new: str | None = None,
                 params: JudgeParams | None = None, fps: float = 166.0, rebuild: bool = False,
                 recon: ReconState | None = None):
        self.recon = recon if recon is not None else ReconState(folder, recon_dir, fps=fps)
        s = self.recon
        if len(s.versions) < 3:
            raise ValueError(f"{recon_dir}: need a reference and two reconstructions, got {s.versions}")
        self.ref = s.versions[0]
        self.old, self.new = old or s.versions[1], new or s.versions[2]
        self.params = params or JudgeParams()
        self.recon_dir = Path(recon_dir)
        self.fps = s.fps
        path = self.recon_dir / EVENTS_FILE
        self.events = (pd.read_csv(path, keep_default_na=False, na_values=[""])
                       if path.exists() and not rebuild else self.build_events())
        if rebuild or not path.exists():
            self.events.to_csv(path, index=False)
        self.labels = list(s.labels)
        self.first, self.n_frames = s.first, len(s.frames)
        # (aspect, version, frame, fish) int8 grade codes, rebuilt from the segments.
        self.grades = np.zeros((len(ASPECTS), len(JUDGED), self.n_frames, len(self.labels)), np.int8)
        self.segments = self._load_segments()
        for r in self.segments.itertuples(index=False):
            self._apply(r)

    def _load_segments(self) -> pd.DataFrame:
        path = self.recon_dir / JUDGE_FILE
        if path.exists():
            head = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=0)
            if "version" not in head.columns:         # an earlier format (one verdict for both versions)
                n = 1
                while path.with_name(f"recon_judgements.earlier_format_{n}.csv").exists():
                    n += 1
                path.replace(path.with_name(f"recon_judgements.earlier_format_{n}.csv"))
        return _read(path, JUDGE_COLUMNS)

    def _apply(self, r):
        if r.aspect not in ASPECTS or r.fish not in self.labels:
            return
        lo, hi = int(r.start) - self.first, int(r.end) - self.first
        self.grades[ASPECTS.index(r.aspect), JUDGED.index(r.version), max(0, lo):max(0, hi + 1),
                    self.labels.index(r.fish)] = GRADE_CODES.index(r.grade) + 1 if r.grade else 0

    # -- events ----------------------------------------------------------------
    def build_events(self) -> pd.DataFrame:
        s, p = self.recon, self.params
        t0 = max(0, (s.analysis_start or s.first) - s.first)
        n = len(s.frames)
        ref = s.series(self.ref, 0, n, aligned=False)
        named = {v: s.series(v, 0, n, aligned=False) for v in (self.old, self.new)}
        aligned = {v: s.series(v, 0, n, aligned=True) for v in (self.old, self.new)}
        pos = s.pos_node
        steps = {}
        for v in (self.old, self.new):
            with np.errstate(invalid="ignore"):
                st = np.linalg.norm(np.diff(named[v][:, :, pos], axis=0), axis=-1)
            steps[v] = np.vstack([np.full((1, st.shape[1]), np.nan), st])           # (F, K): step into frame t

        def jump(v, lo, hi, ks):
            return _nanmax(steps[v][lo:hi + 1, ks])

        rows = []
        # Identity: the switch moments of either version.
        sw = []
        for v in (self.old, self.new):
            ev = switch_events(s, v)
            sw += [(int(r.frame), v, str(r.fish), bool(r.correct_after)) for r in ev.itertuples(index=False)]
        sw.sort()
        merge, pad = round(p.switch_merge_s * self.fps), round(p.switch_pad_s * self.fps)
        groups: list[list] = []
        for item in sw:
            if groups and item[0] - groups[-1][-1][0] <= merge:
                groups[-1].append(item)
            else:
                groups.append([item])
        for g in groups:
            lo = max(0, g[0][0] - s.first - pad)
            hi = min(n - 1, g[-1][0] - s.first + pad)
            fish = sorted({f for _, _, fs, _ in g for f in fs.split("+") if f})
            ks = [s.labels.index(f) for f in fish if f in s.labels]
            detail = "; ".join(f"{v} {fs} {'back to right' if ok else 'into wrong'} @{f}" for f, v, fs, ok in g)
            d_on = _dist(named[self.old][lo:hi + 1, ks], named[self.new][lo:hi + 1, ks])
            d_or = _dist(named[self.old][lo:hi + 1, ks], ref[lo:hi + 1, ks])
            d_nr = _dist(named[self.new][lo:hi + 1, ks], ref[lo:hi + 1, ks])
            rows.append(("identity", s.first + lo, s.first + hi, "+".join(fish), detail, _nanmax(d_on),
                         _nanmax(d_or), _nanmax(d_nr), jump(self.old, lo, hi, ks), jump(self.new, lo, hi, ks)))
        # Body: geometry of the same fish (matched frame by frame) disagreeing anywhere among the three.
        d_on = _dist(aligned[self.old], aligned[self.new])
        d_or = _dist(aligned[self.old], ref)
        d_nr = _dist(aligned[self.new], ref)
        has = {v: np.isfinite(aligned[v][:, :, pos]).all(axis=-1) for v in (self.old, self.new)}
        one_sided = has[self.old] ^ has[self.new]
        for k, name in enumerate(s.labels):
            far = np.zeros(n, bool)
            for d in (d_on, d_or, d_nr):
                far |= np.nan_to_num(d[:, k], nan=0.0) > p.body_mm
            mask = far | one_sided[:, k]
            mask[:t0] = False
            for lo, hi in _runs(mask, p.merge_frames):
                if hi - lo + 1 < p.min_body_frames:
                    continue
                parts = []
                gap = one_sided[lo:hi + 1, k]
                if gap.any():
                    who = self.old if has[self.old][lo:hi + 1, k][gap].sum() >= gap.sum() / 2 else self.new
                    parts.append(f"only {who} has 3D for {gap.sum()} frames")
                if far[lo:hi + 1].any():
                    parts.append("nodes apart")
                rows.append(("body", s.first + lo, s.first + hi, name, "; ".join(parts),
                             _nanmax(d_on[lo:hi + 1, k]), _nanmax(d_or[lo:hi + 1, k]), _nanmax(d_nr[lo:hi + 1, k]),
                             jump(self.old, lo, hi, [k]), jump(self.new, lo, hi, [k])))
        df = pd.DataFrame(rows, columns=EVENT_COLUMNS[1:])
        df.insert(0, "event", [f"{r.kind}:{r.start}:{r.fish}" for r in df.itertuples(index=False)])
        for c in EVENT_COLUMNS[6:]:
            df[c] = df[c].astype(float).round(1)
        return df.sort_values(["start", "kind"], kind="stable").reset_index(drop=True)

    # -- grades ----------------------------------------------------------------
    def event(self, event: str) -> pd.Series:
        row = self.events[self.events["event"] == event]
        if not len(row):
            raise KeyError(event)
        return row.iloc[0]

    def event_fish(self, event: str) -> list[str]:
        return [f for f in str(self.event(event)["fish"]).split("+") if f in self.labels]

    def clip(self, start: int, end: int) -> tuple[int, int]:
        last = self.first + self.n_frames - 1
        return max(self.first, min(start, end)), min(last, max(start, end))

    def paint(self, fish: list[str], start: int, end: int, aspect: str, version: str, grade: str,
              event: str = "") -> tuple[int, int]:
        """Grade ``version``'s ``aspect`` of ``fish`` on frames ``start..end`` (``""`` erases). Returns the
        frames written (clipped to the recording)."""
        if aspect not in ASPECTS or version not in JUDGED:
            raise ValueError(f"aspect must be one of {ASPECTS} and version one of {JUDGED}")
        if grade and grade not in GRADES:
            raise ValueError(f"grade must be one of {GRADE_CODES}, got {grade!r}")
        if not fish or any(f not in self.labels for f in fish):
            raise ValueError(f"fish must be some of {self.labels}, got {fish}")
        a, b = self.clip(start, end)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = pd.DataFrame([[f, aspect, version, str(a), str(b), grade, event, "", JUDGED_ON, now] for f in fish],
                            columns=JUDGE_COLUMNS)
        self._append(rows)
        for r in rows.itertuples(index=False):
            self._apply(r)
        return a, b

    def notes(self, start: int, end: int) -> list[tuple[int, int, str]]:
        """Notes overlapping frames ``start..end`` (the latest per range)."""
        n = self.segments[self.segments["aspect"] == "note"]
        out = {}
        for r in n.itertuples(index=False):
            if int(r.start) <= end and int(r.end) >= start:
                out[(int(r.start), int(r.end))] = r.note
        return [(a, b, t) for (a, b), t in out.items() if t]

    def set_note(self, start: int, end: int, text: str, event: str = ""):
        a, b = self.clip(start, end)
        self._append(pd.DataFrame([["", "note", "", str(a), str(b), "", event, " ".join(text.splitlines()),
                                    JUDGED_ON, time.strftime("%Y-%m-%d %H:%M:%S")]], columns=JUDGE_COLUMNS))

    def _append(self, rows: pd.DataFrame):
        path = self.recon_dir / JUDGE_FILE
        rows.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
        self.segments = pd.concat([self.segments, rows], ignore_index=True)

    def at(self, frame: int) -> dict[tuple[str, str], list[str]]:
        """``(aspect, version)`` -> each fish's grade at ``frame`` (``""`` not graded)."""
        t = frame - self.first
        if not 0 <= t < self.n_frames:
            return {(a, v): [""] * len(self.labels) for a in ASPECTS for v in JUDGED}
        return {(a, v): [GRADE_CODES[c - 1] if c else "" for c in self.grades[i, j, t]]
                for i, a in enumerate(ASPECTS) for j, v in enumerate(JUDGED)}

    @staticmethod
    def compare(old: str, new: str) -> str:
        """Which version is better, from its two grades (``""`` when either is not graded)."""
        if not old or not new:
            return ""
        if "unsure" in (old, new):
            return "unsure"
        d = RANK[new] - RANK[old]
        if d == 0:
            return "same"
        return f"{'old' if d > 0 else 'new'} better, {'a little' if abs(d) == 1 else 'clearly'}"

    def graded(self, aspect: str | None = None) -> np.ndarray:
        """``(frames, fish)`` True where both versions are graded (for ``aspect``, or for both aspects)."""
        sel = self.grades if aspect is None else self.grades[ASPECTS.index(aspect)][None]
        return (sel > 0).all(axis=(0, 1))

    def done(self, start: int, end: int, fish: list[str], aspect: str) -> float:
        """Share (0-1) of ``fish`` x frames ``start..end`` where both versions are graded for ``aspect``."""
        a, b = self.clip(start, end)
        ks = [self.labels.index(f) for f in fish] or list(range(len(self.labels)))
        g = self.grades[ASPECTS.index(aspect)][:, a - self.first:b - self.first + 1][:, :, ks]
        return float((g > 0).all(axis=0).mean()) if g.size else 0.0

    @staticmethod
    def own_aspect(kind: str) -> str:
        return "id" if kind == "identity" else "body"

    def table(self) -> pd.DataFrame:
        """Hint events with the share of their fish x frames graded per aspect and ``judged`` (own aspect
        complete)."""
        out = self.events.copy()
        ids, bodies = [], []
        any_graded = (self.grades > 0).any(axis=(0, 1, 3))
        for r in out.itertuples(index=False):
            if not any_graded[int(r.start) - self.first:int(r.end) - self.first + 1].any():
                ids.append(0), bodies.append(0)
                continue
            fish = [f for f in str(r.fish).split("+") if f in self.labels]
            ids.append(round(100 * self.done(int(r.start), int(r.end), fish, "id")))
            bodies.append(round(100 * self.done(int(r.start), int(r.end), fish, "body")))
        out["id %"], out["body %"] = ids, bodies
        out["judged"] = np.where(out["kind"] == "identity", out["id %"], out["body %"]) >= 100
        return out

    def summary(self, start: int | None = None, end: int | None = None) -> pd.DataFrame:
        """Per aspect and fish, over frames ``start..end`` (default: the analysis start to the end): frames
        graded, each version's grades, and the comparison."""
        a = self.first + 0 if start is None else start
        if start is None and self.recon.analysis_start is not None:
            a = int(self.recon.analysis_start)
        b = self.first + self.n_frames - 1 if end is None else end
        a, b = self.clip(a, b)
        sl = slice(a - self.first, b - self.first + 1)
        rows = []
        for i, asp in enumerate(ASPECTS):
            for k, f in enumerate(self.labels):
                go, gn = self.grades[i, 0, sl, k], self.grades[i, 1, sl, k]
                both = (go > 0) & (gn > 0)
                row = {"aspect": asp, "fish": f, "frames graded": int(both.sum())}
                for name, g in (("old", go), ("new", gn)):
                    for c, code in enumerate(GRADE_CODES):
                        row[f"{name} {code}"] = int((g == c + 1).sum())
                pairs = np.unique(np.stack([go[both], gn[both]]), axis=1) if both.any() else np.zeros((2, 0), int)
                counts = dict.fromkeys(COMPARE, 0)
                for co, cn in pairs.T:
                    n = int(((go == co) & (gn == cn)).sum())
                    counts[self.compare(GRADE_CODES[co - 1], GRADE_CODES[cn - 1])] += n
                row.update(counts)
                rows.append(row)
        return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="The reference review package folder.")
    p.add_argument("--recon-dir", type=Path, required=True, help="A recon_compare folder with three versions.")
    p.add_argument("--old", default=None, help="Default: the second version.")
    p.add_argument("--new", default=None, help="Default: the third version.")
    p.add_argument("--rebuild", action="store_true", help="Rebuild the event list (verdicts are kept).")
    args = p.parse_args(argv)
    js = JudgeState(args.folder, args.recon_dir, old=args.old, new=args.new, rebuild=args.rebuild)
    print("machine hints:", js.events.groupby("kind").size().to_dict())
    print(js.summary().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
