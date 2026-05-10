# SLEAP-NN training templates (medaka mating pipeline)

Hand-tuned training configs for the top-down pipeline:

- `centroid.yaml` — finds one anchor point per fish.
- `centered_instance.yaml` — predicts `mouth -> tail` skeleton inside the centroid crop.

Both are committed templates; override per-run paths via Hydra overrides on the
CLI rather than editing the file in place.

## Layout assumptions

- **Cache**: local SSD `C:/sleap_cache/<run>_<centroid|centered_instance>/`.
  Don't cache to Z:\ — it's network storage and thrashes the LAN.
- **Outputs**: `Z:/StephensU/medaka/sleap_models/<round>/<view>/runs/<kind>/`.
  The `ckpt_dir` is the parent; `run_name` is the subfolder.
- **Labels**: `<round>/<view>/labels.pkg.slp` from
  `python -m scripts.sleap.prepare_retraining`.

## Train from scratch

```bash
sleap-nn train configs/sleap/training/centroid.yaml \
  data_config.train_labels_path=[Z:/StephensU/medaka/sleap_models/retrain_20260508/top/labels.pkg.slp] \
  data_config.cache_img_path=C:/sleap_cache/retrain_20260508_top_centroid \
  trainer_config.ckpt_dir=Z:/StephensU/medaka/sleap_models/retrain_20260508/top/runs \
  trainer_config.run_name=centroid

sleap-nn train configs/sleap/training/centered_instance.yaml \
  data_config.train_labels_path=[Z:/StephensU/medaka/sleap_models/retrain_20260508/top/labels.pkg.slp] \
  data_config.cache_img_path=C:/sleap_cache/retrain_20260508_top_centered_instance \
  trainer_config.ckpt_dir=Z:/StephensU/medaka/sleap_models/retrain_20260508/top/runs \
  trainer_config.run_name=centered_instance
```

`scripts/sleap/prepare_retraining.py` automates these invocations and prints
the exact commands for the current round.

## Fine-tune from a previous round

To warm-start the backbone from a prior run (random head init), add
`model_config.pretrained_backbone_weights=<path>/best.ckpt`:

```bash
sleap-nn train configs/sleap/training/centroid.yaml \
  data_config.train_labels_path=[<round_N+1>/top/labels.pkg.slp] \
  data_config.cache_img_path=C:/sleap_cache/<round_N+1>_top_centroid \
  trainer_config.ckpt_dir=<round_N+1>/top/runs \
  trainer_config.run_name=centroid \
  model_config.pretrained_backbone_weights=<round_N>/top/runs/centroid/best.ckpt
```

This is the active-learning pattern: round N+1 inherits backbone features from
round N but learns fresh heads on the corrected labels.

## Resume a crashed run

```bash
sleap-nn train ... trainer_config.resume_ckpt_path=<run>/last.ckpt
```

This restores full optimizer + scheduler state.

## Skeleton change

`centered_instance.yaml` declares `part_names: [mouth, tail]`. If you add or
rename nodes (e.g. add `body`), update that list to match the skeleton in your
`pkg.slp`. Mismatches make sleap-nn fail at config validation time.
