"""Build fragments from Hungarian tracklets and mine same/diff training pairs."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fragment:
    """A gap-free run of rows sharing one `(track, fish_id)` label."""

    fragment_id: int
    fish_id: str
    track: int
    frame_start: int
    frame_end: int  # inclusive
    rows: list[int] = field(default_factory=list)  # indices into the crop index table

    def __len__(self) -> int:
        return len(self.rows)


def build_fragments(
    index_df: pd.DataFrame,
    max_gap: int = 1,
) -> list[Fragment]:
    """
    Group `index_df` rows into fragments.

    A **fragment** is a maximal run of rows such that:
      - the same `(track, fish_id)` label is carried throughout,
      - consecutive rows within the run differ in `frame` by at most `max_gap`.

    Rows with `fish_id == "unassigned"` or NaN are skipped entirely — those
    are ambiguous by construction and make poor training anchors.

    Args:
        index_df: DataFrame with columns `frame`, `track`, `fish_id`, `row`.
        max_gap: Maximum frame gap tolerated inside one fragment. 1 means
            strictly consecutive frames.

    Returns:
        A list of `Fragment` objects, ordered by (fish_id, frame_start).
    """
    required = {"frame", "track", "fish_id", "row"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"index_df missing columns: {sorted(missing)}")

    df = index_df.copy()
    df = df[df["fish_id"].notna() & (df["fish_id"].astype(str) != "unassigned")]
    df = df.sort_values(["fish_id", "track", "frame"]).reset_index(drop=True)

    fragments: list[Fragment] = []
    fid_counter = 0
    current: Fragment | None = None
    for _, row in df.iterrows():
        frame = int(row["frame"])
        track = int(row["track"]) if not pd.isna(row["track"]) else -1
        fish_id = str(row["fish_id"])
        row_idx = int(row["row"])

        if (
            current is not None
            and current.fish_id == fish_id
            and current.track == track
            and frame - current.frame_end <= max_gap
        ):
            current.rows.append(row_idx)
            current.frame_end = frame
        else:
            if current is not None:
                fragments.append(current)
            current = Fragment(
                fragment_id=fid_counter,
                fish_id=fish_id,
                track=track,
                frame_start=frame,
                frame_end=frame,
                rows=[row_idx],
            )
            fid_counter += 1
    if current is not None:
        fragments.append(current)
    return fragments


def _overlap(a: Fragment, b: Fragment) -> bool:
    """True iff two fragments share at least one frame."""
    return not (a.frame_end < b.frame_start or b.frame_end < a.frame_start)


def mine_pairs(
    fragments: list[Fragment],
    max_same: int = 2000,
    max_diff: int = 2000,
    rng_seed: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Sample `(row_i, row_j)` training pairs from the fragment structure.

    - **same** pairs: two rows drawn from the same fragment (same fish).
    - **diff** pairs: two rows from fragments that temporally coexist in at
      least one frame (guaranteed different fish — one-to-one per frame).

    Args:
        fragments: Output of `build_fragments`.
        max_same: Cap on same-fish pairs.
        max_diff: Cap on different-fish pairs.
        rng_seed: Seed for reproducible sampling.

    Returns:
        `(same_pairs, diff_pairs)`, each a list of `(row_i, row_j)` with
        `row_i < row_j`.
    """
    rng = random.Random(rng_seed)

    # Same-fish pairs: sample within fragments that have >= 2 rows. Weight
    # sampling by fragment length so a few long fragments do not starve the
    # shorter ones.
    eligible = [f for f in fragments if len(f) >= 2]
    same_pairs: list[tuple[int, int]] = []
    if eligible:
        weights = [len(f) for f in eligible]
        total = sum(weights)
        # Draw up to max_same pairs with replacement (cheap and fine for training).
        for _ in range(max_same):
            f = rng.choices(eligible, weights=weights, k=1)[0]
            i, j = rng.sample(f.rows, 2)
            if i > j:
                i, j = j, i
            same_pairs.append((i, j))

    # Different-fish pairs: iterate temporally overlapping fragment pairs.
    diff_pairs: list[tuple[int, int]] = []
    overlaps: list[tuple[Fragment, Fragment]] = []
    # Sort by frame_start so we can early-exit the inner loop.
    ordered = sorted(fragments, key=lambda f: f.frame_start)
    for a_idx, a in enumerate(ordered):
        for b in ordered[a_idx + 1 :]:
            if b.frame_start > a.frame_end:
                break
            if _overlap(a, b):
                overlaps.append((a, b))
    if overlaps:
        for _ in range(max_diff):
            a, b = rng.choice(overlaps)
            i = rng.choice(a.rows)
            j = rng.choice(b.rows)
            if i > j:
                i, j = j, i
            diff_pairs.append((i, j))

    return same_pairs, diff_pairs
