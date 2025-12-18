from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tracks from SLEAP outputs (placeholder).")
    parser.add_argument("--input", required=True, help="Path to SLEAP output (e.g., .slp)")
    parser.add_argument("--output", required=True, help="Path to exported tracks")
    args = parser.parse_args()

    raise SystemExit(f"TODO: implement export from {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
