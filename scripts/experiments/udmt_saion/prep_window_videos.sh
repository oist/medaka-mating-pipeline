#!/usr/bin/env bash
# Cut the 176300-181300 window of each view into its own mp4 so UDMT frame 0 == 176300.
# ffmpeg lives in the conda env (no ffmpeg module on saion); self-activate it.
set -euo pipefail
if ! command -v ffmpeg >/dev/null 2>&1; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate udmt
fi
DATA="${DATA:-/bucket/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC}"
OUT="${OUT:-$HOME/udmt_run/videos}"
START="${START:-176300}"; END="${END:-181299}"; FPS="${FPS:-166}"
mkdir -p "$OUT"
declare -A CAM=( [top_cam]=cam0_top.mp4 [side1_cam]=cam1_side.mp4 [side2_cam]=cam2_side.mp4 )
for V in top_cam side1_cam side2_cam; do
  SRC=$(ls "$DATA/$V/"*.mp4 | head -1)
  echo "cutting $V frames $START..$END from $SRC"
  ffmpeg -y -i "$SRC" \
    -vf "select='between(n,${START},${END})',setpts=N/FRAME_RATE/TB" -an -r "$FPS" \
    "$OUT/${V}_${START}_$((END+1)).mp4"
done
echo "done -> $OUT"
