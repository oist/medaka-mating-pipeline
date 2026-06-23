#!/usr/bin/env bash
# Headless patch to the UDMT clone: UDMT hardcodes `vis_flag=True` (module-level in
# tracker_multi.py AND local inside single_object_track.py::single_compensate). Headless
# that crashes on `gui_param['video_display_widget*'].width()` (NoneType). Flip vis_flag
# -> False ONLY. Do NOT touch save_flag: `if save_flag:` (tracker_multi.py:477) gates the
# build of result_save_file_list (the actual per-animal track output), so save_flag must
# stay True or the tracker writes nothing -> IndexError. save_flag writes only cv2 .avi
# debug videos (no Qt) -- headless-safe.
# Run ONCE after cloning UDMT (idempotent). Also: env must `pip install segment_anything==1.0`
# (the udmt.gui.tabs import chain hard-imports it even though SAM is unused here).
set -e
U=${1:-/bucket/StephensU/medaka/udmt/UDMT}
D=$U/udmt/gui/tabs/ST_Net/pytracking/evaluation
sed -i -E 's/^([[:space:]]*)vis_flag[[:space:]]*=[[:space:]]*True/\1vis_flag = False/' \
  "$D/tracker_multi.py" "$D/single_object_track.py"
echo "patched vis_flag -> False (save_flag left True):"
grep -nE '^[[:space:]]*(vis_flag|save_flag)[[:space:]]*=' \
  "$D/tracker_multi.py" "$D/single_object_track.py"

# Gate the `frame_num < 199` fast-abort on `search_period`. That guard exists to reject bad
# parameter combos quickly during the param search (status 1/2, search_period=True). On the
# FINAL pass (status 3, search_period=False) it otherwise truncates the whole-video track to
# ~100-130 frames whenever not_found_times_per_animal>20 early -- which fires on medaka even
# though the tracker holds the full window. Gating it lets status 3 produce the full track.
sed -i 's/if frame_num < 199:/if search_period and frame_num < 199:/' "$D/tracker_multi.py"
echo "gated frame_num<199 abort on search_period:"
grep -n 'frame_num < 199' "$D/tracker_multi.py"
