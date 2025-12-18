from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute courtship metrics (placeholder).")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raise SystemExit(f"TODO: implement courtship metrics for {args.tracks} -> {args.output}")


if __name__ == "__main__":
    main()
