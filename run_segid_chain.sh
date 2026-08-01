#!/usr/bin/env bash
# Chained seg-ID pilot finish: wait for top masks lock to release, then
# top-linkage -> assemble 3 views -> train (20 epochs) -> eval. Logs inline.
set -e
cd "c:/Users/machi/Documents/GitHub/medaka-mating-pipeline"
PY=.venv/Scripts/python.exe
SEGID="Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2"
TOPH5="outputs/20251225_K_1_2_FCMC/top_cam_f176300_181300_segpilot/detectron2_masks.h5"

echo "[chain] waiting for top h5 lock to release ..."
n=0
until $PY -c "import h5py,sys;
try:
    h=h5py.File('$TOPH5','r'); n=len([k for k in h.keys() if k.startswith('frame_')]); h.close(); print('readable',n); sys.exit(0 if n>4000 else 1)
except Exception as e:
    sys.exit(1)" 2>/dev/null; do
  n=$((n+1)); [ $n -ge 120 ] && { echo "[chain] timeout waiting for h5"; exit 1; }
  sleep 10
done
echo "[chain] top h5 ready after ${n}x10s"

echo "[chain] === top linkage ==="
$PY -m sam2_id_experiment.scripts.33_build_segid_dataset \
  --view top_cam --masks "$TOPH5" \
  --tracklet-bound-csv "Z:/StephensU/medaka/runs/udmt_validation/tracklet_bind/top_cam.tracklet_bound.csv" \
  --video "Z:/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC/top_cam/cam0_top.mp4" \
  --window 176300 181300 --out-dir "$SEGID"

echo "[chain] === assemble 3 views ==="
$PY -m sam2_id_experiment.scripts.34_assemble_segid_training \
  --segid-dir "$SEGID" --views top_cam side1_cam side2_cam \
  --tracklet-bound-dir "Z:/StephensU/medaka/runs/udmt_validation/tracklet_bind" \
  --out-dir "$SEGID/pooled_all"

echo "[chain] === train (20 epochs) ==="
$PY -c "
from sam2_id_experiment.src.id_model.train import train_embedding_model
import json
r=train_embedding_model(dataset_dir='$SEGID/pooled_all', output_dir='$SEGID/model_all',
                        epochs=20, batch_size=256, device='cuda')
print(json.dumps(r,indent=2,default=str))"

echo "[chain] === eval ==="
$PY -m sam2_id_experiment.scripts.35_eval_segid_embedding \
  --dataset "$SEGID/pooled_all" --model "$SEGID/model_all/model.pt" \
  --out-dir "$SEGID/eval_all" --device cuda

echo "[chain] DONE"
