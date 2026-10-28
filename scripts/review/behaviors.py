"""Manual BORIS behaviour annotations for the review GUI's behaviour row.

BORIS was run on the raw top-camera video at the source frame rate, so its ``Image index`` is the
review data's frame number. START/STOP pairs become bouts, POINT events one-frame bouts. Subjects
follow the annotation vocabulary of :mod:`scripts.common.boris_events` (``m1tf`` = M1 towards the
female, ``m1tm2`` = M1 towards M2, ``m1m2`` = both males, ``f`` / ``m1`` / ``m2`` = one fish alone);
they are shown as ``M1->F`` etc. and laid out in three lanes: towards the female, between the males,
one fish alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from scripts.common.boris_events import load_boris_csv, state_intervals

LANES = ("to F", "M-M", "solo")
CATEGORIES = ("courtship", "aggressive", "mating", "other")
CATEGORY_ALIASES = {"courtship": "courtship", "aggressive": "aggressive", "competitive": "aggressive",
                    "mating": "mating"}
BOUT_COLUMNS = ["start", "end", "subject", "who", "behavior", "category", "lane", "point"]
_PAIR = re.compile(r"^(?P<a>f|m1|m2)t(?P<b>f|m1|m2)$")


def subject_label(subject: str) -> tuple[str, int]:
    """``m1tf`` -> (``M1->F``, lane 0); ``m1tm2`` -> (``M1->M2``, 1); ``m1m2`` -> (``M1<->M2``, 1);
    ``f`` -> (``F``, 2). Unknown subjects keep their text in the solo lane."""
    s = subject.strip().lower()
    m = _PAIR.match(s)
    if m:
        a, b = m["a"].upper(), m["b"].upper()
        return f"{a}->{b}", 0 if b == "F" else 1
    if s in ("m1m2", "m2m1"):
        return "M1<->M2", 1
    if s in ("f", "m1", "m2"):
        return s.upper(), 2
    return subject.strip() or "?", 2


def category_of(category: str) -> str:
    return CATEGORY_ALIASES.get(category.strip().lower(), "other")


def empty_bouts() -> pd.DataFrame:
    return pd.DataFrame(columns=BOUT_COLUMNS)


def load_bouts(path: str | Path) -> pd.DataFrame:
    """Bouts of one BORIS CSV, sorted by start frame (columns ``BOUT_COLUMNS``)."""
    events = load_boris_csv(path)
    rows = []
    for r in state_intervals(events).itertuples(index=False):
        if r.start_frame < 0 or r.stop_frame < 0:
            continue
        rows.append((int(r.start_frame), int(r.stop_frame), r.subject, r.behavior, r.category, False))
    points = events[(events["behavior_type"] == "POINT") & (events["frame_index"] >= 0)]
    for r in points.itertuples(index=False):
        rows.append((int(r.frame_index), int(r.frame_index), r.subject, r.behavior, r.category, True))
    out = []
    for start, end, subject, behavior, category, point in rows:
        who, lane = subject_label(str(subject))
        out.append({"start": min(start, end), "end": max(start, end), "subject": str(subject), "who": who,
                    "behavior": str(behavior), "category": category_of(str(category)), "lane": lane,
                    "point": point})
    df = pd.DataFrame(out, columns=BOUT_COLUMNS)
    return df.sort_values(["start", "lane"], kind="stable").reset_index(drop=True)


def bout_rows(bouts: pd.DataFrame, max_rows: int = 2) -> tuple[list[int], dict]:
    """Sub-rows for overlapping bouts: per lane the number of rows it needs (1..``max_rows``) and, per
    bout index, its row inside the lane (greedy: the first row free at the bout's start; rarer deeper
    overlaps share the last row)."""
    rows = [1] * len(LANES)
    sub: dict = {}
    for lane in range(len(LANES)):
        ends: list[int] = []
        for i, r in bouts[bouts["lane"] == lane].sort_values("start", kind="stable").iterrows():
            row = next((j for j, e in enumerate(ends) if e < r["start"]), len(ends))
            if row == len(ends):
                ends.append(r["end"])
            else:
                ends[row] = r["end"]
            sub[i] = min(row, max_rows - 1)
        rows[lane] = max(1, min(len(ends), max_rows))
    return rows, sub


def find_boris_csv(experiment: str, search: list[Path]) -> Path | None:
    """``<experiment>.csv`` given directly or found (recursively) in the search folders; the most
    recently modified wins if several copies exist."""
    hits = []
    for place in search:
        place = Path(place)
        if place.is_file() and place.name == f"{experiment}.csv":
            hits.append(place)
        elif place.is_dir():
            hits.extend(place.rglob(f"{experiment}.csv"))
    hits = [h for h in hits if h.is_file()]
    return max(hits, key=lambda h: h.stat().st_mtime) if hits else None


def default_search(extra: list[Path] | None = None) -> list[Path]:
    """Where the GUI looks for BORIS files: ``--boris``, the paths config's ``boris_events_dir``, and the
    repository's ``archive/boris``."""
    import getpass

    places = list(extra or [])
    try:
        from scripts.common.local_paths import REPO_ROOT, load_local_paths

        cfg = REPO_ROOT / "configs" / f"paths.{getpass.getuser()}.toml"
        paths = load_local_paths(cfg if cfg.exists() else REPO_ROOT / "configs" / "paths.local.toml")
        if paths is not None and paths.boris_events_dir is not None:
            places.append(Path(paths.boris_events_dir))
    except Exception:  # noqa: BLE001 - laptop without the paths config: repository copy only
        pass
    places.append(Path(__file__).resolve().parents[2] / "archive" / "boris")
    return places
