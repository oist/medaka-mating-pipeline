# Legacy 2D-to-3D Reconstruction

This directory contains the older reconstruction workflow. It supports
automatic best-correspondence selection when the cross-view identity mapping is
uncertain.

## Steps

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `step1_get_points.py` | extract 2D points per frame from processed CSV files |
| 2 | `step2_compute_3d_candidates.py` | compute 3D candidates and registration costs |
| 3 | `step3_select_best_correspondence.py` | choose the best a/b/c correspondence from the candidates |
| 4 | `step4_cross_frame_tracking.py` | stabilize labels over time with Hungarian matching and motion cues |
| export | `export_3d_groups.py` | write per-group or long-format 3D CSV files |
| visualization | `visualize_3d.py` | render 3D trajectory videos or plots |

## Example usage

```bash
# Full pipeline
python -m scripts.reconstruction.pipeline ^
  --model_folder "H:\calibration_output\cali_way1" ^
  --processed_dir "H:\sleap_output\processed" ^
  --set_key "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422" ^
  --output_dir "H:\3d_trajectories" ^
  --matching_csv "H:\3d_trajectories\2Dto3D_matching.csv"

# Step-by-step
python -m scripts.reconstruction.step1_get_points --processed_dir ... --set_key ... --output points.json
python -m scripts.reconstruction.step2_compute_3d_candidates --model_folder ... --points points.json --output candidates.json
python -m scripts.reconstruction.step3_select_best_correspondence --candidates candidates.json --output best.json
python -m scripts.reconstruction.step4_cross_frame_tracking --best_json best.json --output best.json

# Visualization
python -m scripts.reconstruction.visualize_3d --output_dir ... --set_key ... --format mp4
```

## Outputs

- `{set_key}_a_3d.csv` and related files: one trajectory per fish/group
- `2Dto3D_matching_merged.csv`: merged correspondence table with chain fields
- chain format example:
  `PLAYER1-FISH_1-PLAYER2-FISH_0-PLAYER3-FISH_0`
