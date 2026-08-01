"""Finalize the full-length stage with the audited bounded finalizer contract."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "finalize_bounded_postfreeze_stage_v2.py"
CORE_SHA256 = "c9ea5d44a57470bda06583a3f5311a9c84708cb8040b0d0233b8156c5c7a7f27"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    assert sha256(CORE_PATH) == CORE_SHA256
    spec = importlib.util.spec_from_file_location(
        "round8_full_length_finalizer_core", CORE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(Path(__file__).resolve())
    module.ROOTS = {"full_length": HERE / "full_length_postfreeze"}
    module.main()


if __name__ == "__main__":
    main()
