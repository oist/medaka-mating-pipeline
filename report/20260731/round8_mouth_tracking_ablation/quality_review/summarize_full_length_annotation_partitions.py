"""Partition sealed full-length female assignments by registered cohort role."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[4]
REPORT = REPO / "report/20260731/round8_mouth_tracking_ablation"
QUALITY = REPORT / "quality_review"
ASSIGNMENTS = QUALITY / "full_length_independent_female_assignments.csv"
PROTOCOL = REPORT / "cohort_protocol.json"
ASSIGNMENTS_SHA256 = (
    "2420ff3db22ffb26a788008c6eee5586e10d3e53a10282ee3f77fff3f363ed67"
)
PROTOCOL_SHA256 = (
    "8f81068ca50154ffb77eb8c32015ddbe2788e6dc395d7216c3b58e9d5956fb86"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intervals(protocol: dict, section: str) -> list[tuple[int, int]]:
    records = []
    for key in (
        "reviewed_contact_events",
        "matched_control_windows",
        "round8_risk_windows",
    ):
        records.extend(
            tuple(map(int, item["frames"]))
            for item in protocol[section].get(key, [])
        )
    return records


def summarize(group: pd.DataFrame) -> dict:
    mapped = group[group.consensus_method_id.fillna("").ne("")].copy()
    identities = mapped.consensus_method_id.astype(str).to_numpy()
    counts = Counter(identities)
    best_label, best_count = counts.most_common(1)[0] if counts else ("", 0)
    return {
        "eligible": int(len(group)),
        "mapped": int(len(mapped)),
        "best_fixed_identity": best_label,
        "best_fixed_correct": int(best_count),
        "full_eligible_best_fixed_accuracy": (
            float(best_count / len(group)) if len(group) else None
        ),
        "selective_mapped_best_fixed_accuracy": (
            float(best_count / len(mapped)) if len(mapped) else None
        ),
        "observed_identity_changes": int(
            np.sum(identities[1:] != identities[:-1])
            if len(identities) > 1
            else 0
        ),
        "identity_counts": dict(sorted(counts.items())),
    }


def main() -> None:
    if sha256(ASSIGNMENTS) != ASSIGNMENTS_SHA256:
        raise ValueError("sealed full-length assignment table changed")
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("registered cohort protocol changed")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    table = pd.read_csv(ASSIGNMENTS, low_memory=False)
    table = table.sort_values(["method", "frame"], kind="stable")

    registered = {
        "training": intervals(protocol, "training"),
        "validation": intervals(protocol, "validation"),
        "heldout": intervals(protocol, "held_out"),
        "critical_descriptive": [(34000, 35100)],
    }

    def partition(frame: int) -> str:
        matches = [
            name
            for name, ranges in registered.items()
            if any(left <= frame <= right for left, right in ranges)
        ]
        if len(matches) > 1:
            raise ValueError(f"overlapping cohort roles at frame {frame}: {matches}")
        return matches[0] if matches else "other_full_length"

    table["partition"] = table.frame.map(partition)
    summaries = {
        method: {
            role: summarize(rows.sort_values("frame", kind="stable"))
            for role, rows in method_rows.groupby("partition", sort=True)
        }
        for method, method_rows in table.groupby("method", sort=True)
    }
    comparison_rows = []
    for left, right in (("A1", "A2"), ("A1", "A3"), ("A2", "A3")):
        lhs = table[table.method.eq(left)][
            ["annotation_id", "frame", "partition", "consensus_method_id"]
        ].rename(columns={"consensus_method_id": "left_identity"})
        rhs = table[table.method.eq(right)][
            ["annotation_id", "frame", "partition", "consensus_method_id"]
        ].rename(columns={"consensus_method_id": "right_identity"})
        joined = lhs.merge(
            rhs,
            on=["annotation_id", "frame", "partition"],
            how="inner",
            validate="one_to_one",
        )
        joined["different"] = (
            joined.left_identity.fillna("").astype(str)
            != joined.right_identity.fillna("").astype(str)
        )
        for role, rows in joined.groupby("partition", sort=True):
            comparison_rows.append(
                {
                    "left": left,
                    "right": right,
                    "partition": role,
                    "annotations": int(len(rows)),
                    "different_assignments": int(rows.different.sum()),
                }
            )

    payload = {
        "schema_version": 1,
        "status": "full_length_annotation_partitions_independently_summarized",
        "bindings": {
            "assignments": {
                "path": str(ASSIGNMENTS),
                "sha256": ASSIGNMENTS_SHA256,
            },
            "cohort_protocol": {
                "path": str(PROTOCOL),
                "sha256": PROTOCOL_SHA256,
            },
        },
        "partition_definitions": {
            **{
                name: [list(item) for item in ranges]
                for name, ranges in registered.items()
            },
            "other_full_length": "all annotations outside registered bounded cohorts",
        },
        "methods": summaries,
        "pairwise_assignment_differences": comparison_rows,
        "interpretation": (
            "Bounded-window scores describe tracks initialized inside each "
            "window. These partition scores describe the same annotation "
            "frames after full-history linking from frame zero; they are not "
            "interchangeable. Female-only assignments cannot observe a "
            "male-to-male exchange."
        ),
    }
    json_path = QUALITY / "full_length_annotation_partition_summary.json"
    csv_path = QUALITY / "full_length_annotation_partition_summary.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for method, roles in summaries.items():
        for role, record in roles.items():
            flat_rows.append(
                {
                    "method": method,
                    "partition": role,
                    **{
                        key: value
                        for key, value in record.items()
                        if key != "identity_counts"
                    },
                    "identity_counts_json": json.dumps(
                        record["identity_counts"], sort_keys=True
                    ),
                }
            )
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "json_sha256": sha256(json_path),
                "csv": str(csv_path),
                "csv_sha256": sha256(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
