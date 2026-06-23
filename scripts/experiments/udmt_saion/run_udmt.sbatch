#!/bin/bash
#SBATCH --job-name=udmt_medaka
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --output=udmt_%A_%a.log
set -euo pipefail

# --- paths (edit) ---
# /bucket is READ-ONLY on saion compute nodes -> read inputs from /bucket, write to $HOME.
export UDMT_ROOT=/bucket/StephensU/medaka/udmt/UDMT          # read-only inputs (clone + checkpoints)
export BUNDLE=/bucket/StephensU/medaka/udmt/bundle          # this folder (rsync'd; read-only OK)
export RUNROOT=$HOME/udmt_run                                # WRITABLE outputs
export PROJECT=$RUNROOT/medaka_v100-2026-06-22                # UDMT project folder (writable)
export START=176300
export XMEM_CKPT=$UDMT_ROOT/udmt/gui/tabs/xmem/saves/XMem.pth
export TRDIMP_CKPT=$UDMT_ROOT/udmt/gui/pretrained/trdimp_net_ep.pth.tar
export CSV=$BUNDLE/round5_260611_3d_identified.csv
export PATHS_CONFIG=$BUNDLE/paths.local.toml

VIEWS=(top_cam side1_cam side2_cam)
V=${VIEWS[$SLURM_ARRAY_TASK_ID]}
VIDEO=$RUNROOT/videos/${V}_${START}_181300.mp4
STEM=${V}_${START}_181300

# --- env: native Miniforge conda env (set up under $HOME/miniforge3) ---
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate udmt
export CUDA_HOME=$CONDA_PREFIX                                              # nvcc + headers live in the env
RUN=""                                                                     # run with the env's python directly
export TORCH_CUDA_ARCH_LIST=7.0                                            # V100 = sm_70
export TORCH_EXTENSIONS_DIR=$HOME/udmt_torch_ext                              # pre-warmed PrRoIPool .so
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 DEBUG=

mkdir -p $RUNROOT/videos $PROJECT/{tmp,models,training-datasets,tracking-results}

# --- run native UDMT (XMem -> ST-Net), Detectron2-seeded ---
$RUN python $BUNDLE/run_udmt_headless.py \
  --udmt-root $UDMT_ROOT --project $PROJECT --video $VIDEO --stem $STEM \
  --init-mask  $BUNDLE/init/${V}_init_mask.png \
  --init-startpos $BUNDLE/init/${V}_start_pos.txt \
  --xmem-ckpt $XMEM_CKPT --trdimp-ckpt $TRDIMP_CKPT \
  --num-fish 3 --resize-factor 0.8 --downsample 1 --frame-rate 166 \
  --epochs 10 --batch-size 8

echo "FINAL .npy: $PROJECT/tracking-results/$STEM/${STEM}-whole-filter5.npy"
# Scoring vs the 3D identity needs the medaka repo + calibration model, which are NOT
# on saion. Copy the .npy back to the repo machine and run there:
#   python -m scripts.experiments.udmt_saion.compare_udmt_to_proxy \
#     --npy <stem>-whole-filter5.npy --view $V \
#     --identified-3d-csv <..._3d_identified.csv> --start-frame $START \
#     --resize-factor 0.8 --paths-config configs/paths.local.toml --out <report>.json
