# Tracking Post-Processing

Use the tracking stage from the terminal instead of notebooks:

```powershell
python -m scripts.tracking.main --paths-config configs/paths.local.toml `
  "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422"
```

You can also start from a specific PLAYER1 CSV:

```powershell
python -m scripts.tracking.main --paths-config configs/paths.local.toml `
  "D:\sleap_output\top_mouth_180\top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv"
```

Useful flags:

- `--simple`: run the lightweight identity-stabilization path without outlier removal
- `--no-viz`: skip visualization PNGs
- `--skip-2d-interpolation`: keep NaN gaps for later 3D-stage handling
- `--flow-weight`: add motion-direction cost to the Hungarian assignment

What this stage does:

1. Groups the three matching camera CSV files.
2. Assigns per-view identities across frames.
3. Optionally removes outliers and mirror detections.
4. Optionally interpolates/smooths 2D gaps.
5. Writes processed CSVs for downstream 3D reconstruction.

Outputs:

- `{set_key}_player1_processed.csv`
- `{set_key}_player2_processed.csv`
- `{set_key}_player3_processed.csv`
- `{set_key}_after_id_assignment.png`
- `{set_key}_final.png`

See [ALGORITHM_DESCRIPTION.md](ALGORITHM_DESCRIPTION.md) for implementation notes.
