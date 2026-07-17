"""Count mouth/tail visibility states in raw predicted SLEAP instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


VIEWS = ("top_cam", "side1_cam", "side2_cam")


def inspect(path: Path) -> dict:
    with h5py.File(path, "r") as source:
        metadata_raw = source["metadata"].attrs["json"]
        if isinstance(metadata_raw, bytes):
            metadata_raw = metadata_raw.decode("utf-8")
        metadata = json.loads(str(metadata_raw))
        global_nodes = [str(node["name"]) for node in metadata["nodes"]]
        skeleton_nodes = [
            global_nodes[int(node["id"])] for node in metadata["skeletons"][0]["nodes"]
        ]
        mouth_index = skeleton_nodes.index("mouth")
        tail_index = skeleton_nodes.index("tail")
        instances = source["instances"][:]
        instances = instances[instances["instance_type"] == 1]
        visible = source["pred_points"]["visible"][:]
        starts = instances["point_id_start"].astype(np.int64)
        ends = instances["point_id_end"].astype(np.int64)
        mouth_supported = starts + mouth_index < ends
        tail_supported = starts + tail_index < ends
        mouth = np.zeros(len(instances), dtype=bool)
        tail = np.zeros(len(instances), dtype=bool)
        mouth[mouth_supported] = visible[starts[mouth_supported] + mouth_index]
        tail[tail_supported] = visible[starts[tail_supported] + tail_index]
        states = {
            "mouth_and_tail": int(np.sum(mouth & tail)),
            "mouth_only": int(np.sum(mouth & ~tail)),
            "tail_only": int(np.sum(~mouth & tail)),
            "neither": int(np.sum(~mouth & ~tail)),
        }
        return {
            "path": str(path),
            "predicted_instance_count": int(len(instances)),
            "node_order": skeleton_nodes,
            "visibility_state_counts": states,
            "visibility_state_shares": {
                key: float(value / max(len(instances), 1)) for key, value in states.items()
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slp-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {view: inspect(args.slp_dir / f"{view}.slp") for view in VIEWS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
