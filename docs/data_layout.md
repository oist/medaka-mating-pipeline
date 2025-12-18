# Data layout

Raw videos and large artifacts should not be committed to git.

## Recommended on-disk layout

```
data/
  README.md
  raw/
    <trial_id>/
      videos/
        top.*
        sideA.*
        sideB.*
      metadata.yaml
  derived/
    <trial_id>/
      calibration/
      tracking/
      analysis/
```

## Trial metadata (suggested fields)

- `trial_id`
- `date_time`
- `subjects`: `F`, `M1`, `M2`
- `camera_settings`
- `calibration_ref`
- `sleap_model_versions`
