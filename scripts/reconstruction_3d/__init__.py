"""
三视角 2D 坐标 → 3D 重建、跨帧 ID 统一、轨迹可视化。

使用方式：
  from scripts.reconstruction_3d import run_pipeline, discover_sets

  sets = discover_sets(r"D:\\sleap_output\\processed")
  run_pipeline(
      processed_dir=r"D:\\sleap_output\\processed",
      set_key=sets[0],
      model_folder=r"D:\\calibration_model\\cali_251217_way1",
      output_dir=r"D:\\sleap_output\\reconstruction_3d",
  )
"""

from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.load_2d import load_processed_dfs, get_points_per_frame, run_load_2d
from scripts.reconstruction_3d.triangulate import compute_candidates
from scripts.reconstruction_3d.select_best import select_best_all_frames
from scripts.reconstruction_3d.cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    ensure_all_frames_and_fill_spline,
    dfs_to_single_csv,
)
from scripts.reconstruction_3d.visualize_3d import plot_3d_trajectories, run_plot
from scripts.reconstruction_3d.pipeline import run_pipeline, discover_sets

__all__ = [
    "Calibration",
    "load_processed_dfs",
    "get_points_per_frame",
    "run_load_2d",
    "compute_candidates",
    "select_best_all_frames",
    "apply_cross_frame_tracking",
    "best_by_frame_to_dfs",
    "ensure_all_frames_and_fill_spline",
    "dfs_to_single_csv",
    "plot_3d_trajectories",
    "run_plot",
    "run_pipeline",
    "discover_sets",
]
