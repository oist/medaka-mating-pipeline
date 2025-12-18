from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _sleap_cmd(exe: str, args: list[str]) -> list[str]:
    """Prefer `uv tool run` so SLEAP can be installed as a uv tool."""
    if shutil.which("uv") is not None:
        return ["uv", "tool", "run", exe, *args]
    return [exe, *args]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export SLEAP predictions (.slp) to analysis-friendly formats via `sleap-convert`. "
            "(SLEAP 1.5.x uses `sleap-convert`; `sleap-export` is legacy.)"
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input predictions .slp (or labels .slp).",
    )
    parser.add_argument(
        "--format",
        default="analysis",
        choices=["analysis", "analysis.csv",
                 "analysis.nix", "h5", "json", "slp"],
        help="Output format passed to `sleap-convert --format`.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional output file path. For multi-video SLPs, SLEAP may create additional "
            "outputs using its default naming convention if only one output is provided."
        ),
    )
    parser.add_argument(
        "--video",
        help="Optional video path for conversions that require a video reference.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    convert_args: list[str] = [str(input_path), "--format", args.format]
    if args.output:
        convert_args += ["--output", args.output]
    if args.video:
        convert_args += ["--video", args.video]

    _run(_sleap_cmd("sleap-convert", convert_args))


if __name__ == "__main__":
    main()
