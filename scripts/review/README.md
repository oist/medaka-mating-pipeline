# Human identity / head-tail review

Every frame of every experiment is watched by a person (at any playback speed); the
person fixes fish identities (F / M1 / M2) and head/tail orientation. The
pipeline output is never edited: corrections are an event ledger that is
applied afterwards, and machine risk windows are kept unchanged so the two can
be compared.

Driver notebook: [archive/notebooks/experiment_workflows/review_pipeline.ipynb](../../archive/notebooks/experiment_workflows/review_pipeline.ipynb).

## 1. Build packages (desktop with the data and an NVIDIA GPU)

```powershell
# Continuous, idle priority, independent of the 3D watcher:
.venv\Scripts\python.exe -m scripts.review.build_packages P12 ALL --round-tag s14_bottomup_3node_reviewed --watch 10
```

| Step | Script | Needs | Time per experiment |
|---|---|---|---|
| Proxy videos (800 px, ~28 fps playback + full-rate) | `proxies.py` | raw videos only | ~40 min (NVENC) |
| Review data + machine risk windows | `review_data.py` | `batch_3d.done` | minutes |

Output (`<data_root>/review_packages/`, i.e. `Z:\medaka\review_packages`):

```
<exp>/proxy/{top_cam,side1_cam,side2_cam}.{play,full}.mp4 + proxy_manifest.json
<exp>/<round_tag>/review_data.npz, review_manifest.json, machine_risk.csv,
    detections.npz (every detection of every view), machine_intervals.csv
<exp>/<round_tag>/human_events.csv, coverage.csv, session_log.csv, interval_solutions/   (written by the GUI)
```

`review_data.py` (shared 2D-to-3D steps in `reconstruct.py`; reviewer edits recomputed by `geometry.py`) recovers mouth / pectoral / tail of the detections the 3D step
used (`<tag>_best.json`), fixes head/tail (cross-view consistency, a temporal
Viterbi over votes, continuity and swimming direction, then undoes head/tail
swaps shorter than 8 frames (~50 ms) - per view in 2D and in 3D - because a
fish's head never jumps onto its tail position), triangulates every node (a node whose three views disagree is re-triangulated from the two views that agree with its own track over time; one- and two-frame spikes are replaced),
orients every displayed view by the projected 3D head/tail, flags views where
two fish overlap (`view_overlap`, skeletons < 20 px apart), splits each track
into `orientation_segment`s (runs over which head and tail move continuously)
completes every node in 3D wherever the fish has a 3D centroid (`nodes3d_filled`; `node_source` 0 = triangulated, 1 = seen in one view - fitted to that view's 2D point at the fish's median node-to-pectoral distance, 2 = no view - pectoral offset interpolated in time; on 20260112_P_1_1_FCMC ~99% of fish-frames have all three nodes triangulated) and writes risk windows: `close`, `jump`, `gap`, `missing_view`, `residual`,
`unbound`, `orientation`, `head_tail_jump`, `fish_count`.
`--drop-overlapped-view` also leaves the most-overlapped view out of the node
triangulation; it is off by default because it barely helped on
20260112_P_1_1_FCMC (body-length outliers 0.33% -> 0.29%) while making the
rare large frame-to-frame node jumps larger.

## 2. Review (any Windows PC, no GPU)

1. Copy `<exp>/proxy/` and `<exp>/<round_tag>/` to the external SSD, keeping
   that folder structure.
2. Once per PC: clone the repo, then
   `py -3.11 -m venv .venv-review` and
   `.venv-review\Scripts\python.exe -m pip install -r environment\requirements-review.txt`.
3. `.venv-review\Scripts\python.exe -m scripts.review.review_gui <SSD>\...\<exp>\<round_tag>`
   (optional `--backup-dir <folder>` copies the ledger after every save).

Workflow: press S at the first usable frame (hands or a net in the tank
before it), name each fish once (select with a click or 1/2/3, then 4/5/6 =
F/M1/M2), then watch. Whenever a name ends up on the wrong fish, select the
fish that really is F (or M1 / M2) and press its key: the tracker swapped two
fish, and the two tracks are exchanged from there on. Keyboard only: 1 then 5
= "the fish shown as F is really M1".

Keys (F1 in the GUI): Space play/pause, J/K/L shuttle, Left/Right one shown
frame, `,`/`.` one source frame, click/Tab/1/2/3 select, S analysis start,
4/5/6 = F/M1/M2 (first time: names the fish; afterwards: swap),
Shift+4/5/6 rename only, 0 not a fish, H flip head/tail over the whole
orientation segment (Shift+H: from here to its end), X bad 3D and U uncertain
(from here on, press again to end), E edit list (double-click = go there,
Delete = remove), N/P risk windows, W next unwatched frame, Ctrl+Z undo.
Display: double-click a view (or V) to make it the large one, Z zooms every
view onto the fish, +/- resize marks and text (remembered between sessions).

Geometry fixes (the 3D of the edited frames is recomputed at once from
`detections.npz` with the calibration - on the desktop; on a laptop the edit
is saved and applied by the export). Each is pressed at the first and again
at the last wrong frame:

* B - the body is wrong (tail too short, orientation off, centre shifted):
  mouth and tail are rebuilt from the pectoral node, the fish's length and the
  body direction around the run; the centre follows.
* Q / R - in the large view only: the selected fish's head/tail is reversed /
  that view is left out of the fish's 3D (e.g. its nodes sit on another fish).
* T - in the large view only, the selected fish has another fish's
  detection: T, then 1/2/3 for the other fish, T again at the last frame.

Close intervals (fish stuck together, e.g. sinking side by side, where the
tracker swaps them often): the build lists them (`machine_intervals.csv`:
centroids < 12 mm apart for >= 0.5 s, overlapping pairs merged; 23 on
20260112_P_1_1_FCMC). Purple band on the timeline, I / Shift+I next /
previous, Home / End first / last frame, [ / ] move its start / end here
(outside one: start a new one), Backspace deletes it. At both ends check the
identities and head/tail (fix with 4/5/6, H), press C (keyframe), then Y:
inside the interval every view's detections are re-assigned to the fish by a
Viterbi over frames (triangulation residual, quadratic frame-to-frame motion,
a cost per change, keyframes forced; where two fish are one blob in a view,
that view cannot reorder them and the separating views decide). A fish not
triangulable keeps its offset to its group mate and moves with it. The
status bar reports how many frames moved and where the identity now differs
from before; check those, add a keyframe (fix + C) where it is wrong and press
Y again. On P_1_1's 32.7 s sinking interval (M015) it took 3 s and cut
frame-to-frame jumps > 4 mm from 854 to 282. A keyframe pins what was
confirmed on screen (each fish's position then), and a solution is
dropped as stale when its keyframes, the identity swaps over it or the view
swaps inside it change. Intervals longer than 120 s must be split.

Start-up: ranges edited in earlier sessions are recomputed (P_1_1 with ~118
edited ranges: ~24 s) and cached in `geometry_cache.npz` next to the ledger
when the window closes; the next start loads them in under a second unless
something they depend on changed.

3D window (D; a dock that can be moved or floated to a second screen): every fish's nodes at the current frame in the tank box, with a 1 s centroid trail; drag rotates, wheel zooms (and then follows the fish), right-click steps through oblique / top / front / side, double-click resets. Filled dots are triangulated nodes, hollow circles were fitted to one view, squares are body-model only.

Behaviour row (under the zoomed timeline, same span): the manual BORIS
annotation of the experiment - `<experiment>.csv` from `--boris <file or folder>`,
the paths config's `boris_events_dir` or the repository's `archive/boris`
(searched recursively, newest copy wins). BORIS ran on the raw top-camera
video at 166 fps, so its `Image index` is the review frame number. START/STOP
pairs are bouts, in three lanes: towards F (`m1tf`, `m2tf`), between the males
(`m1tm2`, `m2tm1`, `m1m2`) and one fish alone; colour = category (blue
courtship, orange aggressive, green mating, grey other) with `M1->F follow` as
text; overlapping bouts get a second sub-row. The whole-video timeline shows
them as a thin strip, the info panel lists the bouts at the current frame and
Ctrl+Right / Ctrl+Left jump to the next / previous bout - handy for checking that
the GUI's F / M1 / M2 are the annotator's.

What is drawn: the cross is the fish's 3D position projected into the view,
always (grey = interpolated, no detection in any view); skeletons are the
SLEAP detections the 3D step assigned to the track (head = big dot); a thin
line joins skeleton and cross when they are more than 40 px apart; a dashed
skeleton with `~` overlaps another fish in that view, so its nodes may sit on
the other fish; a dashed ring with `no 2D` = not detected in that view; thin dashes ending in small squares = body nodes this view did not detect, drawn at the projection of their 3D position. Under
the timelines: speed (mm/s) and distance to the nearest fish (mm, dashed line
15 mm) per named fish over the zoomed span - a swap placed at the right frame
leaves no speed spike.

Ledger model (see `ledger.py`): names are per corrected fish for the whole
recording, so a name can never sit on two fish; swaps compose in frame order
whatever order they were entered in; flips cover a frame range; flags are
"from here on" per track. A swap entered while or right after playing snaps
to the closest 3D approach of the two tracks in the last 2 s (the key comes
after the reaction time); after stepping, to the closest approach since the
previously shown frame; never back past an earlier swap touching either
track (swaps of three fish do not commute). One entered after a jump is
placed at that frame and reports the closest approach in the last 60 s. Removed
edits are kept in `human_events_removed.csv`. A `review.lock` file warns when
a folder is still open on another computer.

## 3. Export for analysis

```powershell
.venv\Scripts\python.exe -m scripts.review.export_reviewed <review_root>\<exp>\<round_tag>
```

writes `reviewed_3d.csv` (one row per frame and named fish: `label`,
`source_track`, `X/Y/Z`, every node in 3D with flips applied and its `<node>_source`, `is_interpolated`,
`uncertain`, `bad3d`, `nodes_bad`, `watched`, `analysed`) and
`reviewed_summary.json` (`ready_for_analysis` = start marked, all three fish
named, everything after the start watched; corrections vs. machine risk
windows).
