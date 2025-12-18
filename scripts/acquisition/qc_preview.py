from __future__ import annotations

import textwrap


def main() -> None:
		msg = """
		Quick acquisition QC lives here.

		With campy-based acquisition, a common workflow is:
			1) Acquire videos with campy
			2) Use a lightweight preview step to verify exposure/focus/framing
				 on a few frames from each camera.

		TODO: implement preview utilities for files written under:
			data/raw/<trial_id>/videos/
		"""
		print(textwrap.dedent(msg).strip())


if __name__ == "__main__":
		main()
