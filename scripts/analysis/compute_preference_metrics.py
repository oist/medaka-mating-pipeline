from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute preference metrics (placeholder).")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raise SystemExit(
        f"TODO: implement preference metrics for {args.tracks} using {args.config} -> {args.output}"
    )


if __name__ == "__main__":
    main()
