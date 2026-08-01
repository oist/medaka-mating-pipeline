"""Authorize the frozen v4 runner for the critical descriptive stage.

This wrapper changes no scientific command, method, threshold, or input. It
adds the two post-heldout controls that did not exist when v4 was sealed:

1. an exact binding to the independent heldout quality PASS; and
2. an exact registration of the permanently nonconfirmatory critical-event
   reporting scope.

The wrapped v4 module sees this file as the runner so its ordinary
authentication snapshot and prepared/completed markers bind this wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "run_bounded_postfreeze_stage_v4.py"
QUALITY_SEAL = HERE / "quality_review" / "HELDOUT_QUALITY_REVIEW_COMPLETED.json"
HELDOUT_FINALIZED = HERE / "heldout_postfreeze" / "STAGE_FINALIZED.json"

CORE_SHA256 = "0aaeac2293db434f3794b02a992ab208f4dc061162b8da017f6c85d94a21e818"
QUALITY_SEAL_SHA256 = (
    "022fbca3f3d0c13d7388a83c42b3b2bad60ded00de2fe89fad428943e7f213cc"
)
HELDOUT_FINALIZED_SHA256 = (
    "3b58e073c9cf6c97348e1f0c8cdbb9ca5c15d689ee222b450df1c91dc3303152"
)
HELDOUT_COMPLETED_SHA256 = (
    "615b03f161db28982ee1d6b5648e79e1d4bef146b281cc911b13d4654d7da143"
)
FREEZE_SHA256 = (
    "09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d"
)

CRITICAL_REGISTRATION = {
    "role": "permanently_unblinded_descriptive_nonconfirmatory",
    "confirmatory": False,
    "may_change_frozen_settings_or_finalists": False,
    "selection_or_tuning_permitted": False,
    "full_length_launch_permitted": False,
    "inclusive_frame_intervals": True,
    "execution_window": [34000, 35100],
    "required_detail_window": [34540, 34565],
    "critical_event_assignment_rows": [34547, 34555],
    "required_reporting": {
        "all_methods": ["A1", "A2", "A3", "A4", "A5"],
        "detail_window_rows": [34540, 34565],
        "assignment_rows": [34547, 34555],
        "abstentions_and_unassigned_rows_preserved": True,
        "interpretation": (
            "descriptive stress evidence only; never tuning, selection, "
            "confirmatory evidence, or a basis for changing frozen finalists"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_core() -> Any:
    assert sha256(CORE_PATH) == CORE_SHA256
    spec = importlib.util.spec_from_file_location("round8_postfreeze_v4_core", CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Make v4's existing runner authentication bind this authorization wrapper.
    module.__file__ = str(Path(__file__).resolve())
    return module


def quality_binding(core: Any) -> dict[str, Any]:
    assert sha256(QUALITY_SEAL) == QUALITY_SEAL_SHA256
    quality = load_json(QUALITY_SEAL)
    assert quality["status"] == "heldout_independently_authenticated_scored_and_passed"
    assert quality["independent_quality_verdict"] == "PASS"
    assert quality["critical_opened_during_review"] is False
    assert quality["frozen_settings_changed"] is False
    assert quality["selection_or_tuning_performed"] is False
    assert quality["full_length_launched_during_review"] is False
    bindings = quality["bindings"]
    assert bindings["freeze"]["sha256"] == FREEZE_SHA256
    assert (
        bindings["heldout_finalized_marker"]["sha256"]
        == HELDOUT_FINALIZED_SHA256
    )
    assert (
        bindings["heldout_completion_marker"]["sha256"]
        == HELDOUT_COMPLETED_SHA256
    )
    assert sha256(HELDOUT_FINALIZED) == HELDOUT_FINALIZED_SHA256
    return {
        "path": core.relative(QUALITY_SEAL),
        "sha256": QUALITY_SEAL_SHA256,
        "status": quality["status"],
        "independent_quality_verdict": "PASS",
        "heldout_finalized_sha256": HELDOUT_FINALIZED_SHA256,
        "heldout_completed_sha256": HELDOUT_COMPLETED_SHA256,
        "freeze_sha256": FREEZE_SHA256,
    }


def prepare(core: Any) -> Path:
    binding = quality_binding(core)
    prepared_path = core.prepare("critical")
    stage_root = prepared_path.parent

    prelaunch_path = stage_root / "prelaunch_auth.json"
    prelaunch = load_json(prelaunch_path)
    prelaunch["heldout_quality_authorization"] = binding
    prelaunch["critical_descriptive_registration"] = CRITICAL_REGISTRATION
    core.write_json_atomic(prelaunch_path, prelaunch)

    prepared = load_json(prepared_path)
    assert (
        prepared["predecessor_finalized_sha256"]["heldout"]
        == HELDOUT_FINALIZED_SHA256
    )
    prepared["prelaunch_auth"]["sha256"] = sha256(prelaunch_path)
    prepared["heldout_quality_authorization"] = binding
    prepared["critical_descriptive_registration"] = CRITICAL_REGISTRATION
    prepared["scientific_commands_changed_from_v4"] = False
    prepared["authorization_wrapper_core"] = {
        "path": core.relative(CORE_PATH),
        "sha256": CORE_SHA256,
    }
    core.write_json_atomic(prepared_path, prepared)
    return prepared_path


def execute(core: Any) -> Path:
    binding = quality_binding(core)
    stage_root = core.REPORT / core.STAGES["critical"]["root"]
    prepared_path = stage_root / "STAGE_PREPARED.json"
    prepared = load_json(prepared_path)
    assert prepared["runner_sha256"] == sha256(Path(__file__))
    assert prepared["heldout_quality_authorization"] == binding
    assert prepared["critical_descriptive_registration"] == CRITICAL_REGISTRATION
    assert prepared["scientific_commands_changed_from_v4"] is False
    assert prepared["authorization_wrapper_core"]["sha256"] == CORE_SHA256
    assert (
        prepared["predecessor_finalized_sha256"]["heldout"]
        == HELDOUT_FINALIZED_SHA256
    )
    prelaunch_path = stage_root / "prelaunch_auth.json"
    assert prepared["prelaunch_auth"]["sha256"] == sha256(prelaunch_path)
    prelaunch = load_json(prelaunch_path)
    assert prelaunch["heldout_quality_authorization"] == binding
    assert prelaunch["critical_descriptive_registration"] == CRITICAL_REGISTRATION

    completion_path = core.execute("critical")
    completion = load_json(completion_path)
    completion["heldout_quality_authorization"] = binding
    completion["critical_descriptive_registration"] = CRITICAL_REGISTRATION
    completion["scientific_commands_changed_from_v4"] = False
    completion["authorization_wrapper_core"] = {
        "path": core.relative(CORE_PATH),
        "sha256": CORE_SHA256,
    }
    core.write_json_atomic(completion_path, completion)
    return completion_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("critical",))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-prepared", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core = load_core()
    result = prepare(core) if args.prepare_only else execute(core)
    print(
        json.dumps(
            {"path": core.relative(result), "sha256": sha256(result)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
