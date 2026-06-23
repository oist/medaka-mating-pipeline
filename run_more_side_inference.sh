#!/usr/bin/env bash
# Extra dense Detectron2 inference on more side windows for the male-ID model.
# Sequential (avoid GPU OOM). ~10 min/job.
set -e
cd "c:/Users/machi/Documents/GitHub/medaka-mating-pipeline"
D2=.venv-detectron2/Scripts/python.exe
SIDEM=outputs/20251225_K_1_2_FCMC/side_detectron2_model_firstprompt
CFGDIR=sam2_id_experiment/configs/segpilot_auto
mkdir -p "$CFGDIR"

WINDOWS="145000:150000 152000:157000 285000:290000 300000:305000"
for view in side1_cam side2_cam; do
  if [ "$view" = "side1_cam" ]; then vid="cam1_side.mp4"; else vid="cam2_side.mp4"; fi
  for w in $WINDOWS; do
    s=${w%:*}; e=${w#*:}
    cfg="$CFGDIR/${view}_f${s}_${e}.yaml"
    cat > "$cfg" <<EOF
experiment_name: 20251225_K_1_2_FCMC
view: $view
video_path: Z:/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC/$view/$vid
output_dir: outputs/{experiment_name}/{view}_f${s}_${e}_segpilot
start_frame: $s
end_frame: $e
n_animals_expected: 3
EOF
    out="outputs/20251225_K_1_2_FCMC/${view}_f${s}_${e}_segpilot/detectron2_masks.h5"
    if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
    echo "=== $view $s-$e ==="
    $D2 -m sam2_id_experiment.scripts.13_run_detectron2_segmentation \
      --config "$cfg" \
      --detectron-config "$SIDEM/detectron2_config.yaml" \
      --model "$SIDEM/model_final.pth" \
      --frame-stride 1 --score-thresh 0.5 --device cuda 2>&1 | tail -2
  done
done
echo "MORE-SIDE-INFERENCE DONE"
