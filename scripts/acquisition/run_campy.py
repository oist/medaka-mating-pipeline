from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Thin wrapper around campy-acquire. "
            "This repo treats campy as the acquisition tool and standardizes config location."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/acquisition/campy_config.yaml",
        help="Path to campy YAML config (default: %(default)s)",
    )
    parser.add_argument(
        "--",
        dest="passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to campy-acquire after --",
    )
    args = parser.parse_args()

    exe = shutil.which("campy-acquire")
    if exe is None:
        raise SystemExit(
            "campy-acquire not found on PATH.\n"
            "Install campy (https://github.com/ksseverson57/campy) and ensure the env is active.\n"
            "Then run: campy-acquire <config.yaml>"
        )

    cmd = [exe, args.config]
    if args.passthrough:
        cmd.extend(args.passthrough)

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
