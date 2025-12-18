from __future__ import annotations

import textwrap


def main() -> None:
		msg = """
		This repository uses campy for acquisition and triggering.

		Configure triggering in your campy YAML config:
			configs/acquisition/campy_config.yaml

		Campy supports Arduino/Teensy integration and external triggering.
		See: https://github.com/ksseverson57/campy
		"""
		print(textwrap.dedent(msg).strip())


if __name__ == "__main__":
		main()
