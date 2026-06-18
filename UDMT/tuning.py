from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from UDMT.io import read_csv
from UDMT.metrics import TrackQualityConfig, score_track_table


@dataclass(frozen=True)
class ParameterGrid:
    """UDMT-style parameter grid names used by the official GUI."""

    search_region_scales: tuple[float, ...] = (1.5, 2.0, 2.5)
    target_size_biases: tuple[float, ...] = (-0.1, 0.0, 0.1)

    def iter_parameters(self) -> Iterable[dict[str, float]]:
        for target_size_bias in self.target_size_biases:
            for search_region_scale in self.search_region_scales:
                yield {
                    "target_size_bias": float(target_size_bias),
                    "search_region_scale": float(search_region_scale),
                }


@dataclass(frozen=True)
class CandidateTrackRun:
    name: str
    tracks_csv: Path
    parameters: dict[str, float | int | str] = field(default_factory=dict)


def rank_candidate_track_runs(
    candidates: Iterable[CandidateTrackRun],
    *,
    config: TrackQualityConfig | None = None,
) -> list[dict]:
    """Rank precomputed candidate tracking CSVs by transparent QC metrics."""

    ranked: list[dict] = []
    for candidate in candidates:
        tracks = read_csv(candidate.tracks_csv)
        report = score_track_table(tracks, config=config)
        ranked.append(
            {
                "name": candidate.name,
                "tracks_csv": str(candidate.tracks_csv),
                "parameters": dict(candidate.parameters),
                "quality": report.to_dict(),
            }
        )
    ranked.sort(key=lambda item: item["quality"]["quality_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["selected"] = rank == 1
    return ranked


def candidates_from_csvs(paths: list[str | Path]) -> list[CandidateTrackRun]:
    return [
        CandidateTrackRun(name=Path(path).stem, tracks_csv=Path(path), parameters={})
        for path in paths
    ]


def ranking_to_frame(ranking: list[dict]) -> pd.DataFrame:
    rows = []
    for item in ranking:
        row = {
            "rank": item["rank"],
            "selected": item["selected"],
            "name": item["name"],
            "tracks_csv": item["tracks_csv"],
            **item["parameters"],
            **{f"quality_{k}": v for k, v in item["quality"].items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)
