#!/usr/bin/env bash
# Post-inference: link all side windows -> pool -> train full + male-only ResNet18
# -> measure female and male-male separability (grouped CV) vs the tiny-model baseline.
set -e
cd "c:/Users/machi/Documents/GitHub/medaka-mating-pipeline"
PY=.venv/Scripts/python.exe
SEGID="C:/udmt_validation/segid/K_1_2"
TB="C:/udmt_validation/tracklet_bind"
DATA="Z:/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC"

# window  start  end  mask-relpath (under outputs/20251225_K_1_2_FCMC/<view>...)
# 176300 uses the existing dense fulltest masks; others use segpilot masks.
WINS=(
  "176300 181300 _f176300_181300_sam2_fulltest_firstprompt/detectron2_dense_masks.h5"
  "132000 137000 _f132000_137000_segpilot/detectron2_masks.h5"
  "145000 150000 _f145000_150000_segpilot/detectron2_masks.h5"
  "152000 157000 _f152000_157000_segpilot/detectron2_masks.h5"
  "285000 290000 _f285000_290000_segpilot/detectron2_masks.h5"
  "300000 305000 _f300000_305000_segpilot/detectron2_masks.h5"
)
OUTB="outputs/20251225_K_1_2_FCMC"
declare -A VID=( [side1_cam]=cam1_side.mp4 [side2_cam]=cam2_side.mp4 )

echo "[male-chain] waiting for the 4 new windows' masks ..."
n=0
until [ -f "$OUTB/side1_cam_f300000_305000_segpilot/detectron2_masks.h5" ] \
   && [ -f "$OUTB/side2_cam_f300000_305000_segpilot/detectron2_masks.h5" ] \
   || [ $n -ge 150 ]; do n=$((n+1)); sleep 20; done
echo "[male-chain] proceeding after ${n}x20s"

POOL_DIRS=""
for entry in "${WINS[@]}"; do
  set -- $entry; s=$1; e=$2; rel=$3
  wdir="$SEGID/win_${s}_${e}_masked"; mkdir -p "$wdir"
  for v in side1_cam side2_cam; do
    mask="$OUTB/${v}${rel}"
    [ -f "$mask" ] || { echo "[skip] no mask $mask"; continue; }
    [ -f "$wdir/${v}_crops.h5" ] && { echo "[have] $wdir/${v}_crops.h5"; continue; }
    echo "[male-chain] link $v $s-$e"
    $PY -m sam2_id_experiment.scripts.33_build_segid_dataset \
      --view "$v" --masks "$mask" \
      --tracklet-bound-csv "$TB/${v}.tracklet_bound.csv" \
      --video "$DATA/${v}/${VID[$v]}" --window "$s" "$e" \
      --mask-background --out-dir "$wdir" 2>&1 | tail -2
  done
  POOL_DIRS="$POOL_DIRS $wdir"
done

echo "[male-chain] === assemble pooled sides (all windows) ==="
$PY -m sam2_id_experiment.scripts.34_assemble_segid_training \
  --segid-dir $POOL_DIRS --views side1_cam side2_cam \
  --tracklet-bound-dir "$TB" --out-dir "$SEGID/pooled_sides_masked" \
  --positive-pairs-per-tracklet 300 --max-negative-pairs 120000

echo "[male-chain] === train FULL resnet18 (female+male) ==="
$PY -c "
from sam2_id_experiment.src.id_model.train_resnet import train_resnet_embedding
print(train_resnet_embedding('$SEGID/pooled_sides_masked','$SEGID/model_resnet_full_masked',
      epochs=15,batch_size=128,lr=1e-4,embedding_dim=128,device='cuda'))"

echo "[male-chain] === build male-only dataset (drop female gid1) ==="
$PY -m sam2_id_experiment.scripts.36_build_male_dataset \
  --pooled-dir "$SEGID/pooled_sides_masked" --out-dir "$SEGID/male_sides_masked" --female-gid 1

echo "[male-chain] === train MALE-ONLY resnet18 ==="
$PY -c "
from sam2_id_experiment.src.id_model.train_resnet import train_resnet_embedding
print(train_resnet_embedding('$SEGID/male_sides_masked','$SEGID/model_resnet_male_masked',
      epochs=20,batch_size=128,lr=1e-4,embedding_dim=128,device='cuda'))"

echo "[male-chain] === DECISIVE eval (grouped CV) ==="
$PY sam2_id_experiment/scripts/eval_male_separability.py \
  --full-dataset "$SEGID/pooled_sides_masked" --full-model "$SEGID/model_resnet_full_masked/model.pt" \
  --male-dataset "$SEGID/male_sides_masked" --male-model "$SEGID/model_resnet_male_masked/model.pt" \
  --out "$SEGID/eval_male/report_masked.json"

echo "[male-chain] === weak-supervised male classifier (window holdout + position baseline) ==="
$PY -m sam2_id_experiment.scripts.39_train_identity_classifier \
  --dataset "$SEGID/pooled_sides_masked" --mode male --female-gid 1 \
  --split window --out-dir "$SEGID/model_male_identity_classifier_masked" \
  --epochs 30 --batch-size 128 --device cuda
echo "[male-chain] DONE"
