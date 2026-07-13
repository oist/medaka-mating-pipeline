from __future__ import annotations

from scripts.calibration.compare_round7_calibrations import compare_assignments


def _frame(triplets):
    return {
        "groups": {
            label: [float(slot), 0.0, 0.0]
            for slot, (label, _triplet) in enumerate(zip(("a", "b", "c"), triplets))
        },
        "perm_cam0": [triplet[0] for triplet in triplets],
        "perm_cam1": [triplet[1] for triplet in triplets],
        "perm_cam2": [triplet[2] for triplet in triplets],
    }


def test_assignment_comparison_is_slot_order_invariant() -> None:
    original = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    reordered = [original[2], original[0], original[1]]
    summary = compare_assignments({"10": _frame(original)}, {"10": _frame(reordered)})
    assert summary["exact_selected_triplet_set_rate"] == 1.0
    assert summary["selected_triplet_jaccard"] == 1.0


def test_assignment_comparison_records_changed_runs() -> None:
    original = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    changed = [(0, 0, 0), (1, 2, 1), (2, 1, 2)]
    frames_a = {str(frame): _frame(original) for frame in (10, 11, 13)}
    frames_b = {str(frame): _frame(changed) for frame in (10, 11, 13)}
    summary = compare_assignments(frames_a, frames_b)
    assert summary["changed_frame_count"] == 3
    assert summary["changed_frame_runs"] == [
        {"start": 10, "end": 11, "length": 2},
        {"start": 13, "end": 13, "length": 1},
    ]
