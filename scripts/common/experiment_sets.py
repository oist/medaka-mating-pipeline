"""Resolve experiment-set shorthand (e.g. ``P12``) to recording folder names.

Folder names follow ``<YYYYMMDD>_<group>_<trial>_<day>_FCMC`` with an optional
``_P2`` suffix for the second recording part of the same session. A set token
is ``<group><trials>``: ``P12`` = group P, trials 1 and 2, every day. Dates
differ within one set (e.g. ``20260113_P_1_2_FCMC``), so matching is by
group/trial only.

Accepted tokens:
  * a full folder name, e.g. ``20260112_P_1_1_FCMC``
  * a set token, e.g. ``P12``, ``K1``, ``M2``
  * ``ALL`` = every recording folder
"""

from __future__ import annotations

import re
from pathlib import Path

FOLDER_RE = re.compile(r"^(?P<date>\d{8})_(?P<group>[A-Z])_(?P<trial>\d+)_(?P<day>\d+)_FCMC(?P<part>_P2)?$")
SET_RE = re.compile(r"^(?P<group>[A-Z])(?P<trials>\d+)$")


def list_recordings(data_dir: Path, include_part2: bool = False) -> list[str]:
    names = []
    for d in sorted(Path(data_dir).iterdir()):
        m = FOLDER_RE.match(d.name)
        if d.is_dir() and m and (include_part2 or not m["part"]):
            names.append(d.name)
    return names


def resolve_experiments(tokens: list[str], data_dir: Path, include_part2: bool = False) -> list[str]:
    """Expand tokens to an ordered, de-duplicated list of existing folders."""
    available = list_recordings(data_dir, include_part2=True)
    selected: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.upper() == "ALL":
            matches = [n for n in available if include_part2 or not FOLDER_RE.match(n)["part"]]
        elif FOLDER_RE.match(token):
            if token not in available:
                raise FileNotFoundError(f"experiment folder not found under {data_dir}: {token}")
            matches = [token]
        elif m := SET_RE.match(token):
            trials = set(m["trials"])
            matches = [
                n for n in available
                if (fm := FOLDER_RE.match(n))["group"] == m["group"]
                and fm["trial"] in trials
                and (include_part2 or not fm["part"])
            ]
            if not matches:
                raise FileNotFoundError(f"set {token!r} matched no folders under {data_dir}")
        else:
            raise ValueError(f"unrecognised experiment token: {token!r}")
        selected.extend(n for n in matches if n not in selected)
    return selected
