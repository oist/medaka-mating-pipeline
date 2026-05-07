"""
Recover a pre-link candidate JSON from an existing post-link `_best.json`.

Use this when you have a 3D reconstruction run that pre-dates the
``_best_pre_link.json`` artifact (i.e. the run was produced before
`save_best_pre_link_json_to` existed) and you want to re-run the
identity linker without redoing triangulation.

The recovery is exact for `groups`, `perm_cam1`, `perm_cam2`. It cannot
recover `errors` or `dropped_high_residual` (these are not preserved in
the post-link artifact); they are filled with `None` / `[]`.

Caveat: any per-fish slot that the original linker dropped to NaN (e.g.
high-residual filter, gap retirement) appears as NaN in the recovered
pre-link too, even though the pristine pre-link from `select_best`
would have had a valid 3D point. The forward+backward re-linker is
robust to this — it operates on whatever points are present.

CLI:
    python -m scripts.reconstruction_3d.recover_pre_link \\
        Z:/StephensU/medaka/reconstruction_3d/20251225_K_1_2_FCMC_best.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def _src_labels(n: int) -> List[str]:
    return [chr(ord("a") + i) for i in range(n)]


def _id_to_src_map(cam0_to_id: List, n: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for src_idx in range(n):
        if src_idx < len(cam0_to_id) and cam0_to_id[src_idx]:
            out[cam0_to_id[src_idx]] = src_idx
    return out


def _recover_groups(
    cam0_to_id: List, post_groups: Dict, n: int, src_labels: List[str]
) -> Dict[str, List[float]]:
    nan = float("nan")
    out: Dict[str, List[float]] = {}
    for src_idx, src_lbl in enumerate(src_labels):
        id_name = cam0_to_id[src_idx] if src_idx < len(cam0_to_id) else None
        if id_name and id_name in post_groups:
            pt = post_groups[id_name]
            if pt is not None and len(pt) == 3:
                out[src_lbl] = [float(v) for v in pt]
                continue
        out[src_lbl] = [nan, nan, nan]
    return out


def _recover_perm(
    cam_to_id: List, id_to_src: Dict[str, int], n: int
) -> List[int]:
    """
    Inverse of `cam_to_id` → `perm[src_idx] = det_idx`. Slots whose ID was
    NaN'd by the linker are filled with the remaining unused detection
    indices (preserves the permutation property; safe because the linker
    does not use perm_cam* values).
    """
    perm: List[int | None] = [None] * n
    used: set[int] = set()
    for det_idx in range(min(n, len(cam_to_id))):
        id_name = cam_to_id[det_idx]
        if not id_name:
            continue
        src_idx = id_to_src.get(id_name)
        if src_idx is None:
            continue
        perm[src_idx] = det_idx
        used.add(det_idx)
    free = [d for d in range(n) if d not in used]
    out: List[int] = []
    for s in range(n):
        if perm[s] is not None:
            out.append(int(perm[s]))
        elif free:
            out.append(int(free.pop(0)))
        else:
            out.append(-1)
    return out


def recover_pre_link(post_link_path: Path) -> Dict:
    """
    Read a post-link best.json and return the same payload shape that
    `save_best_pre_link_json_to` writes.
    """
    with Path(post_link_path).open("r", encoding="utf-8") as fh:
        post = json.load(fh)

    n = int(post["num_individuals"])
    src = _src_labels(n)

    frames_out: Dict[int, Dict] = {}
    for f_str, info in post["frames"].items():
        cam0_to_id = info.get("cam0_to_id", [None] * n)
        cam1_to_id = info.get("cam1_to_id", [None] * n)
        cam2_to_id = info.get("cam2_to_id", [None] * n)
        post_groups = info.get("groups", {}) or {}

        id_to_src = _id_to_src_map(cam0_to_id, n)
        groups = _recover_groups(cam0_to_id, post_groups, n, src)
        perm_cam1 = _recover_perm(cam1_to_id, id_to_src, n)
        perm_cam2 = _recover_perm(cam2_to_id, id_to_src, n)

        frames_out[int(f_str)] = {
            "groups": groups,
            "errors": {lb: None for lb in src},
            "perm_cam1": perm_cam1,
            "perm_cam2": perm_cam2,
            "dropped_high_residual": [],
        }

    return {
        "num_individuals": n,
        "src_labels": src,
        "frames": frames_out,
        "_recovered_from": str(post_link_path),
        "_note": (
            "errors and dropped_high_residual are not recoverable from a "
            "post-link best.json; they are filled with None / []."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a pre-link candidate JSON from an existing post-link "
            "best.json so it can feed alternate linkers."
        )
    )
    parser.add_argument(
        "best_json",
        help="Path to <tag>_best.json (post-link).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Defaults to <input_stem>_pre_link.json next to input.",
    )
    args = parser.parse_args(argv)

    src_path = Path(args.best_json).resolve()
    if not src_path.is_file():
        parser.error(f"Input not found: {src_path}")

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        # 20251225..._best.json -> 20251225..._best_pre_link.json
        if src_path.stem.endswith("_best"):
            out_name = src_path.stem[: -len("_best")] + "_best_pre_link.json"
        else:
            out_name = src_path.stem + "_pre_link.json"
        out_path = src_path.with_name(out_name)

    print(f"Reading  : {src_path}")
    payload = recover_pre_link(src_path)
    print(f"Frames   : {len(payload['frames'])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote    : {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
