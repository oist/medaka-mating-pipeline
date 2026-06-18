"""UDMT-inspired tracking adapters for the medaka pipeline.

This package does not vendor the official cabooster/UDMT GUI or ST-Net
implementation. It provides small, scriptable pieces that borrow UDMT's most
useful pipeline ideas for this repository: automatic quality scoring,
mask-centroid localization refinement, and bidirectional trajectory repair.
"""

from UDMT.bidirectional import (
    BidirectionalConfig,
    BidirectionalSummary,
    detect_motion_anomalies,
    repair_tracks_with_backward,
)
from UDMT.metrics import TrackQualityConfig, TrackQualityReport, score_track_table
from UDMT.refinement import RefinementConfig, RefinementSummary, refine_tracks_with_masks
from UDMT.tuning import CandidateTrackRun, ParameterGrid, rank_candidate_track_runs

__all__ = [
    "BidirectionalConfig",
    "BidirectionalSummary",
    "CandidateTrackRun",
    "ParameterGrid",
    "RefinementConfig",
    "RefinementSummary",
    "TrackQualityConfig",
    "TrackQualityReport",
    "detect_motion_anomalies",
    "rank_candidate_track_runs",
    "refine_tracks_with_masks",
    "repair_tracks_with_backward",
    "score_track_table",
]
