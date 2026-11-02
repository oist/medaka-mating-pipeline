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

Driver notebook: [archive/notebooks/experiment_workflows/machine_correction.ipynb](../../archive/notebooks/experiment_workflows/machine_correction.ipynb).

```powershell
.venv\Scripts\python.exe -m scripts.review.export_reviewed <review_root>\<exp>\<round_tag>
```

applies the ledger (and recomputes the geometry edits; needs the calibration), then runs the machine
post-pass and writes:

* `reviewed_3d.csv` - one row per frame and named fish: `label`, `source_track`, the fish position
  `X/Y/Z` (= the cleaned pectoral node; the 3D reconstruction's centroid is kept as
  `centroid_X/Y/Z`), every node in 3D with flips applied and its `<node>_source` (0 triangulated,
  1 one view, 2 body model, 3 despiked in the build, 4 re-estimated body (B), 5 bridged by the
  post-pass, -1 missing), `is_interpolated`, `uncertain`, `bad3d`, `nodes_bad`, `watched`, `analysed`;
* `postprocess_report.csv` - every run the post-pass replaced and every `jerk` it left;
* `versions.npz` - the four versions below, for the comparison;
* `reviewed_summary.json` - `ready_for_analysis` (start marked, all three fish named, everything
  after the start watched - at any playback speed - and the geometry edits applied), corrections vs.
  machine risk windows, node-source fractions and post-pass counts.

`--no-postprocess` leaves the post-pass out (the position is still the pectoral node);
`--no-versions` skips `versions.npz`. On 20260112_P_1_1_FCMC the export takes ~1 min (plus ~1 min
once for `initial_tracks.npz`, cached next to the review data).

### Machine post-pass (`postprocess.py`)

Runs per corrected fish, i.e. after the reviewer's swaps, flips and body fixes: about 40% of the
pectoral jumps left after review sit exactly on a reviewer swap (the two tracks were not at the same
place), which only shows on the corrected fish's series.

* Spikes: a run of <= 12 frames where a node leaves its path and returns. Candidates are frames with
  acceleration > 3 mm/frame^2; the shortest run whose cubic Hermite bridge (end points and velocities
  of the good frames either side) removes the acceleration is replaced, if the original deviates from
  the bridge by >= 1.5 mm and the motion before and after the run agrees (end mismatch <= half the
  deviation). Turns, tail beats and accelerations follow the bridge and stay untouched.
* Jumps: a pectoral step > 6 mm/frame (~1 m/s) is spread over a few frames each side; the body
  follows. Sustained runs of such steps (tracker interpolation across a gap) are reported, not fixed.
* Pectoral first, in world coordinates; mouth and tail as offsets from the cleaned pectoral.

The parameters are fixed for every experiment (`PostParams`), so group comparisons do not mix in a
processing difference. On 20260112_P_1_1_FCMC (after the analysis start) it replaced 0.3-0.7% of
node-frames; share of fish-frames off their 9-frame median by > 2 mm: pectoral 0.26 -> 0.08%,
mouth 0.32 -> 0.17%, tail 0.86 -> 0.58%; largest pectoral step 27 -> 8 mm. Corrections concentrate
where fish are close (11% of fish-frames < 10 mm apart vs 0.4% > 40 mm; there only 56% of
fish-frames have all three nodes triangulated), so run analyses once on measured frames only
(`<node>_source == 0`) as a sensitivity check. `postprocess_qc.py` makes these tables.

## 4. Compare the versions (`compare_gui.py`)

| Version | What |
|---|---|
| `initial` | plain triangulation, no machine fixes (`initial_tracks.npz`) |
| `machine` | the review data (two-view re-triangulation + despiking): what the reviewer watched |
| `manual` | + the reviewer's ledger |
| `final` | + the machine post-pass = `reviewed_3d.csv` |

```powershell
.venv\Scripts\python.exe -m scripts.review.compare_gui <review_root>\<exp>\<round_tag>
```

Read-only. Version A dashed, B solid in the three views, a 3D view and traces (speed, |A - B|,
acceleration, body length, X/Y/Z; wheel = span, click = go). Presets 1-4: machine step
(initial -> machine), review (machine -> manual), post-pass (manual -> final), all
(initial -> final). I = every version with the reviewed identity (geometry differences only; off,
the review's swaps show too). The change list holds every build fix, reviewer edit and post-pass
run, filtered by group / fish / kind / fast (> 200 mm/s) / size; N / P or double-click to go.
E exports the traced span as an mp4 (three views + speed strip, zoomed onto the fish) into
`compare_clips/`; without the GUI:

```powershell
.venv\Scripts\python.exe -m scripts.review.compare_gui <folder> --clip 105000 105400 --a machine --b final --speed 0.25
```

## 5. Compare two 3D reconstructions (`compare_gui --mode recon`)

The compare GUI has two modes; without `--mode` it asks when a folder has both. The second one
compares **reconstructions** of the same recording and needs no finished review, so it works on any
experiment with a review package.

1. Make the second reconstruction in the layout `reconstruct_from_slp` writes. For the medaka-3d
   per-frame correspondence, `scripts/reconstruction_3d/medaka3d_bridge.py` feeds its triples through
   this repo's own `apply_cross_frame_tracking` (the batch `RECONSTRUCT_ARGS`, parsed by
   `reconstruct_from_slp`'s parser), `_best.json` / `_3d.csv` writers and `bind_topcam_id`, so only the
   per-frame step differs. Driver notebook:
   `archive/notebooks/experiment_workflows/20260112_P_1_1_FCMC_medaka3d_refractive_3d.ipynb`.
2. Build its review package with the same `build_review_data` (same head/tail fixes, node completion,
   risk windows) into `<review_root>/<exp>/<tag>_medaka3d/`; the proxies are shared.
3. Build the comparison (the first folder is the reference; `initial` = plain triangulation + head/tail
   orientation for both, `machine` = the review data):

```powershell
.venv\Scripts\python.exe -m scripts.review.recon_compare <review_root>\<exp>\<tag> <review_root>\<exp>\<tag>_medaka3d `
  --kinds initial initial --names old medaka3d
.venv\Scripts\python.exe -m scripts.review.compare_gui <review_root>\<exp>\<tag> --mode recon `
  --recon-dir <review_root>\<exp>\<tag>\recon_vs_<tag>_medaka3d
```

A **reviewed experiment** gets three versions, the reviewed result as the answer key:

```powershell
.venv\Scripts\python.exe -m scripts.review.recon_compare <folder> <folder> <folder>_medaka3d `
  --kinds final initial initial --names reviewed old medaka3d      # -> <folder>\recon_reviewed_old_medaka3d
```

Its names are fixed (correct for the whole recording); `old` and `medaka3d` take them near the
analysis start and follow their own tracks, so every disagreement is that version's error, and
`recon_compare.error_table` counts the analysed fish-frames each version gets wrong (only one / both).
Identity errors are judged without bias; geometry is not fully: the review corrected `old`, so where
the reviewer left it alone `final` equals `old`. Keys 7/8/9 (or buttons) show reviewed vs old,
reviewed vs medaka3d, old vs medaka3d; going to an entry shows the version it is about.

An **unreviewed experiment** gets two versions; identity is assigned as in the review GUI: name the
reference's fish once (1/2/3 or click, then 4/5/6 = F/M1/M2); every other version takes the names of
its tracks at the same place in that frame and then follows its own tracks, so an identity swap in
either version is listed. I ("same identity") matches the versions frame by frame instead and leaves
only geometry differences.

Playback: Space or the Play button, Left/Right (Shift: 1 s) or the step buttons, Up/Down or the speed
box, G / "go to frame". All keys are application-wide shortcuts; the window (not an input box) has the
keyboard focus at start, and the "mm min" box hands it back on Enter. Drawing: A dashed, B solid in the
fish's colour; where both put a node on the same spot (< 4 display px) that part is drawn once in green.
Every entry of the list is one run of frames (gaps < 0.1 s merged), judged once; "after analysis
start" (default on) hides entries that end before it.

The disagreement list (per reference fish): `swap` (the same name on different fish, frames whose
best and second-best matching differ by < 2 mm skipped, >= 8 frames), `position` (position node > 5 mm
apart), `nodes` (head or tail > 5 mm, position agreeing), `only <name>` (one version has no 3D,
>= 0.1 s). Judge each with A / B (that version is right), W (both wrong), O (both OK); Backspace
clears; the GUI moves on to the next entry. Names go to `recon_labels.csv`, verdicts to
`recon_verdicts.csv`, clips to `compare_clips/` - all inside the `recon_vs_*` folder; the reference's
own ledger is never written. `recon_compare.summary(ReconState(...))` counts entries and verdicts per kind.

## 6. Grade old vs new frame by frame (`recon_judge_gui.py`)

For a three-version comparison folder (reviewed + old + new, e.g. `recon_reviewed_old_m3dfull`), a
separate GUI grades **old and new directly, over the whole recording** - it plays straight through; the
machine hints are only suggestions where to look, and the reviewed result is hidden by default (V).

```powershell
.venv\Scripts\python.exe -m scripts.review.recon_judge_gui <review_root>\<exp>\<tag> `
    --recon-dir <review_root>\<exp>\<tag>
econ_reviewed_old_m3dfull
```

What is graded is the **3D reconstruction**: the main panel is a 3D view (old dashed, new solid, green
where they agree, floor shadows, 0.3 s trails, old-new distance per node in mm; left-drag rotates,
right-drag moves, wheel zooms towards the mouse, double-click shows the whole tank, 7/8/9/0 top / front /
side / oblique, K trails, X follow the selected fish; M cycles both versions / only old / only new, in the
3D and the camera views). The camera views show the same 3D nodes projected
into the video, as an aid, together with every raw SLEAP instance of that frame, unfiltered - two when
SLEAP found two, six when it found six (magenta: SLEAP's skeleton edges as lines, triangle = mouth,
square = pectoral, x = tail; "SLEAP raw: n" = instances in that view; B hides them). They come from
`slp_raw.npz` in the package, built once from the package's `.slp` files with
`python -m scripts.review.slp_raw <package folder>` (unlike `detections.npz`, nothing is dropped - no
false-positive or duplicate filter). Every panel border is draggable.

**Grades.** Each version is graded on its own - correct / small error / clear error / unsure - for its
3D IDs and its 3D body parts; which one is better follows (old correct + new small error = "old better,
a little"; correct vs clear error = "clearly"; the same grade = "same"). Keys: IDs old Q W E, new R T Y;
body old A S D, new F G H (correct / small / clear); U / J IDs / body unsure; O everything correct;
Backspace erase. A grade goes to the selected fish (1 / 2 / 3 only that fish, 4 all; a click on a fish or its button
selects or deselects it) on the
selected frames: drag on the timeline, or press [ and grade later - the range runs to the current frame
and restarts there after each grade, so you can grade while watching; ] closes it, \ / Esc = current
frame only. A later grade paints over an earlier one. Enter jumps to the next frame not fully graded.

**Timeline** (bottom, resizable): the whole recording in three rows - ID hints (red), body hints (orange),
frames graded for every fish (light green) - with the shown span in white; click to jump, a time ruler, the hints in the span, and per fish an ID row and a
body row with old (upper half) over new (lower half). Wheel zooms (0.1 s to the whole recording), right-
drag or Shift+wheel moves, click seeks, drag selects.

Grades are appended to `recon_judgements.csv` (`fish, aspect, version, start, end, grade, event, note,
judged_on = 3D`); `python -m scripts.review.recon_judge <folder> --recon-dir <dir>` prints, per aspect and
fish, the frames of each grade and of each comparison (after the analysis start).

## 7. Audit: numbers for SLEAP and old vs new (`audit.py`, `audit_gui.py`)

Grading every frame is not feasible, and a plain random sample of frames is almost all correct. The
audit asks a person only where the answer carries information:

* **id** (old vs new, blind) - every place where old's and new's identities stop matching (one of them
  swapped). All of them are judged (P_1_1: 30).
* **geometry** (old vs new, blind) - old and new give the same 3D in ~98% of frames (same 2D points);
  those frames cannot tell them apart. A random sample (150) of the stretches where they differ, each judged
  at one key frame (a random frame where they differ; the frames around it are context).

  Old-vs-new items are judged **one version at a time**: first only A is shown and judged, then only B
  (id: does every colour stay on the same fish - right / wrong / wrong briefly = swaps and comes back;
  geometry: right / small error / clear
  error). Which version is better is worked out by `stats`.
* **sleap** - every frame and view is screened from `slp_raw.npz` and `review_data.npz` into suspicious
  groups: fewer / more than 3 instances, an instance missing a body part, an instance far longer than usual
  (two fish joined), head-tail against the other views, a view left out of 3D as inconsistent. 30 random
  frames per group are checked ("real error?"), giving each group's precision and so the estimated number
  of real errors in the whole recording.
* **clean** - 100 random unflagged frames: what the screening misses (0 errors -> < 3% upper bound).
* **lost** (appended later: `python -m scripts.review.audit add-lost <reference> --recon-dir <dir>`) - in
  close encounters (< 10 mm) a fish often keeps a detection in fewer than two views, and ID swaps cluster
  there. Each (fish, view, frame) where that happens is sorted by the raw SLEAP output: A fewer than 3
  instances, B1 a separate unassigned instance where the fish should be (3D projected), B2 one that
  overlaps another fish's detection, C nothing there. Up to 30 per group (>= 20 frames apart) are judged by
  eye, blind to the group: does every fish in the yellow ring have its own magenta instance (detected = the
  tracker did not use it / not detected = SLEAP missed it / hidden)? `stats` weights the groups by size.
  The earlier task list and summary are backed up as `*.before_lost_<time>.*`; answers are kept.

A and B are old / new in random order per item, hidden until `stats`. Everywhere 1 = right, 2 / 3 = wrong, 0 = unsure. The task list is built once with a
fixed seed (the GUI builds it on first start); answers are appended to `audit/audit_answers.csv`.

```powershell
.venv\Scripts\python.exe -m scripts.review.audit_gui <reference> --recon-dir <reference>\recon_reviewed_old_m3dfull
.venv\Scripts\python.exe -m scripts.review.audit stats <reference> --recon-dir <reference>\recon_reviewed_old_m3dfull
```

`stats` prints per SLEAP group the precision with its 95% Wilson interval and the estimated real errors,
the unflagged error rate, and the unblinded old / new counts with a sign test.

## 8. One identity pass (`id_pass_gui.py`, first version)

Watch a recording once, without pausing, and end with the frames whose identities can be used. Works on
any review package, written for the `*_m3dfull` ones:

```powershell
.venv\Scripts\python.exe -m scripts.review.id_pass_gui <review_root>\<exp>\<tag>
```

Without a folder it asks for the experiment (every `*_m3dfull` package whose experiment has proxies).
`--safe-mm`, `--high-risk-mm` and `--min-safe-s` change the machine's danger runs (`id_segments.py`,
defaults 15 mm, 8 mm, 1 s).

Press S at the first usable frame and name each fish once (1/2/3 or click, then 4/5/6). Space then plays
at 2x, and at 1x in danger runs (Up/Down, Shift+Up/Down change the two speeds). Left/Right move one frame, 10 at a time when held down, Shift+Left/Right 2 s; 1/2/3 or a click on the selected fish, or Esc, clears the selection. Nothing pressed = right.
When a name is on the wrong fish, select it and press its true name: the names change from there on and
stay until the next change. The change goes back to the closest approach of the two tracks in the latest
danger run of the last 5 s, and that run becomes unusable. X ... X marks any other range as unusable.

B ... B marks a wrong pose with the right identity (selected fish, else all): written as `pose_bad` in `human_events.csv` and shown on the `pose` row of the timelines, the frames stay usable for identity. X ... X is for frames where identity cannot be trusted; it shows on the `unusable` row of the timelines from the first press. On the status row a bright yellow line is an identity change. A range still open on closing ends at the current frame.
Every fish has its whole skeleton in every view: solid where the view detected it, thin dashes and squares where a node is projected from 3D.
Layout as in the review GUI: one large camera (double-click another to swap), two stacked beside it, the 3D view on the right, the widest panel (drag rotates, wheel zooms, click selects a fish, double-click resets).

Timelines (whole recording, then around the frame): status, danger zone, BORIS lanes. The grey dividers resize the panels; the wheel on the lower one changes its span (+-5 to +-240 s).

Frame status: watched safe, watched danger (nothing changed there), unusable, unwatched. `id_pass/` in
the package holds the ledger (`human_events.csv`, `coverage.csv`, `session_log.csv`), `id_pass_clock.json`
(playback and window-open time over all sessions) and, on closing or P, `id_segments.csv` (usable
segments with the names on the tracks), `id_pass_summary.json` and `id_timeline.png`. The package's own
ledger is not touched.

`id_timeline.png` shows the safe and unsafe intervals over time, in 5 minute rows, with their percentages.
It opens by itself on closing once 99% of the recording has been watched, and at any time with P.
`python -m scripts.review.id_figure <folder> --open` makes it for any package: from the identity pass if
one was watched, else from the package's own review, else the machine's danger runs alone; `--wait`
first waits until the package has been built.

