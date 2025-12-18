# Medaka Mating Pipeline (3-fish)

This repository contains the end-to-end pipeline for filming, calibration, SLEAP inference, and behavioral analysis for **three medaka** recorded during pre-mating interactions.

## Project description
This study investigates the interplay between female mate choice, male courtship behavior, and male-male aggression in medaka fish (*Oryzias latipes*).
Through video recording and quantitative behavioral analysis before mating events, we aim to dissect how multiple mating-related behaviors interact and identify which specific behavioral factors determine female mating decisions.

## Goals
1. **Female preference determinants:** quantify preference and test whether it is driven by male aggression, courtship intensity, or their interaction.
2. **Decision dynamics:** identify whether mate choice occurs early or requires observation of male-male aggression.
3. **Consistency:** assess stability of male strategies across time and female choice patterns across different male pairs.

## Pipeline overview
1. Acquisition (multi-camera, synchronized)
2. Calibration (intrinsics/extrinsics; refraction-aware if underwater)
3. SLEAP inference (2D keypoints per view; optional identity resolution)
4. 3D reconstruction (optional, if multi-view triangulation is used)
5. Behavioral feature extraction (courtship/aggression/preference metrics)
6. Statistics + reporting

## Data layout (expected)
See `docs/data_layout.md`.

## Quick start
- Install dependencies: see `environment/`
- Configure experiment: edit `configs/project.yaml`
- Run inference (HPC): `scripts/sleap/run_inference_slurm.sh`
- Compute metrics: `scripts/analysis/compute_preference_metrics.py`

## Reproducibility
- All parameters should live in `configs/`.
- Outputs should be written to `runs/<date>/<experiment_id>/...` (recommended).
