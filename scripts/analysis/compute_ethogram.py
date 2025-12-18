from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ethogram from tracks (placeholder).")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raise SystemExit(f"TODO: implement ethogram for {args.tracks} -> {args.output}")


if __name__ == "__main__":
    main()
