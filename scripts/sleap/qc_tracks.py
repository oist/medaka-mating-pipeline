from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="QC SLEAP tracks (placeholder).")
    parser.add_argument("--tracks", required=True, help="Path to exported tracks")
    args = parser.parse_args()

    raise SystemExit(f"TODO: implement QC for {args.tracks}")


if __name__ == "__main__":
    main()
