# SAM2-assisted medaka identity pseudo-labeling

Prototype pipeline for turning high-confidence segmentation tracklets into
identity-training examples. SAM2 object ids are treated as local provisional
tracklet ids, not final biological identities.

Initial workflow:

```text
raw video -> masks -> raw tracklets -> QC-filtered tracklets -> ID crops/pairs
```

The first branch prioritizes clean pseudo-label extraction over recall. Crossing,
merged, unstable, low-confidence, or otherwise ambiguous regions should stay out
of the training set and be kept for inference or review only.

Dry run:

```bash
python sam2_id_experiment/scripts/00_dry_run.py \
  --config sam2_id_experiment/configs/sam2_medaka_test.yaml
```

Export a prompt frame for manual SAM2 point selection:

```bash
python sam2_id_experiment/scripts/00_export_prompt_frame.py \
  --config sam2_id_experiment/configs/sam2_medaka_test.yaml
```

The default exported frame is `start_frame` from the config. In SAM2 prompt
YAML, `frame: 0` means that exported start frame, not video frame zero.

Interactive prompt selection:

```bash
python sam2_id_experiment/scripts/00_select_prompt_points.py \
  --config sam2_id_experiment/configs/sam2_medaka_test.yaml \
  --sequence \
  --update-config
```

Controls: number keys choose object id, left click adds a positive point, right
click adds a negative point, right/left arrows move one frame, `n`/`p` step
forward/backward, `.`/`,` also move one frame, `u` undoes the last point on the
current frame, `s` saves, and `q` quits.
With `--update-config`, the script writes the selected prompts into the config
and switches `sam2.runner` to `sam2_video_predictor`.

Data-contract smoke tests use synthetic masks and do not require SAM2.
