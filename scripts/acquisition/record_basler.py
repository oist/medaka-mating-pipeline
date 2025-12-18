from __future__ import annotations

import argparse
import subprocess


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Acquisition entrypoint (campy). "
			"This repo delegates camera acquisition to campy-acquire."
		)
	)
	parser.add_argument(
		"--config",
		default="configs/acquisition/campy_config.yaml",
		help="Path to campy YAML config (default: %(default)s)",
	)
	args = parser.parse_args()

	# Prefer calling the wrapper so we get a clear error if campy is missing.
	raise SystemExit(
		subprocess.call(
			[
				sys.executable,
				"scripts/acquisition/run_campy.py",
				"--config",
				args.config,
			]
		)
	)


if __name__ == "__main__":
	import sys

	main()
