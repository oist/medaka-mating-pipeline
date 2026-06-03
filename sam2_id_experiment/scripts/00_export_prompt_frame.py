from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.sam2_adapter.prompt_frame import (
    export_prompt_frame,
    resolve_prompt_frame_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one video frame for manual SAM2 point-prompt selection."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help=(
            "Absolute video frame to export. Defaults to config start_frame. "
            "Use --local-frame for a window-relative index."
        ),
    )
    parser.add_argument(
        "--local-frame",
        type=int,
        default=None,
        help="Frame offset relative to config start_frame, e.g. 0 means start_frame.",
    )
    parser.add_argument("--output", default=None, help="Optional output PNG path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    start_frame = int(cfg.raw.get("start_frame") or 0)
    frame_index = resolve_prompt_frame_index(
        start_frame=start_frame,
        frame=args.frame,
        local_frame=args.local_frame,
    )
    output_path = (
        Path(args.output)
        if args.output
        else out_dir / f"prompt_frame_abs{frame_index}_local{frame_index - start_frame}.png"
    )
    meta = export_prompt_frame(
        video_path=cfg.video_path,
        output_path=output_path,
        frame_index=frame_index,
        start_frame=start_frame,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
