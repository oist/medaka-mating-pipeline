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
        description="Quick QC for SLEAP .slp datasets using `sleap-inspect` (SLEAP 1.5.x)."
    )
    parser.add_argument(
        "--slp",
        required=True,
        help="Path to SLEAP dataset (.slp), e.g. <video>.predictions.slp",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to sleap-inspect.",
    )
    args = parser.parse_args()

    slp_path = Path(args.slp)
    if not slp_path.exists():
        raise SystemExit(f"Input not found: {slp_path}")

    inspect_args = [str(slp_path)]
    if args.verbose:
        inspect_args.insert(0, "--verbose")

    _run(_sleap_cmd("sleap-inspect", inspect_args))


if __name__ == "__main__":
    main()
