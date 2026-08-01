#!/usr/bin/env bash
# Retrain full + male-only ResNet18 with fixed (orientation-preserving) aug + cosine
# LR + more epochs, then re-measure. Female recovery = sanity that training is sound.
set -e
cd "c:/Users/machi/Documents/GitHub/medaka-mating-pipeline"
PY=.venv/Scripts/python.exe
SEGID="Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2"

echo "=== train FULL resnet18 (30 ep, fixed aug) ==="
$PY -c "
from sam2_id_experiment.src.id_model.train_resnet import train_resnet_embedding
print(train_resnet_embedding('$SEGID/pooled_sides_full','$SEGID/model_resnet_full_v2',
      epochs=30,batch_size=128,lr=1.5e-4,embedding_dim=128,device='cuda'))"

echo "=== train MALE-ONLY resnet18 (30 ep, fixed aug) ==="
$PY -c "
from sam2_id_experiment.src.id_model.train_resnet import train_resnet_embedding
print(train_resnet_embedding('$SEGID/male_sides_full','$SEGID/model_resnet_male_v2',
      epochs=30,batch_size=128,lr=1.5e-4,embedding_dim=128,device='cuda'))"

echo "=== DECISIVE eval v2 ==="
$PY sam2_id_experiment/scripts/eval_male_separability.py \
  --full-dataset "$SEGID/pooled_sides_full" --full-model "$SEGID/model_resnet_full_v2/model.pt" \
  --male-dataset "$SEGID/male_sides_full" --male-model "$SEGID/model_resnet_male_v2/model.pt" \
  --out "$SEGID/eval_male/report_v2.json"
echo "RERUN DONE"
