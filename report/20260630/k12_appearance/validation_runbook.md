# K_1_2 individual-ID VALIDATION HARNESS runbook (handoff aspect 4)

Read-only. Ready-to-run commands to validate ANY candidate per-frame individual-ID
track for `20251225_K_1_2_FCMC` before it is trusted for mate-choice analysis.
Do NOT render videos as part of standing this up (slow, ~25-30 min/chunk) — these
are the commands the next session runs once a candidate CSV exists.

3 fish (1 female + 2 males, identical size), 166 fps, ~348,600 frames.

---

## 0. The two CSV identity conventions (READ THIS FIRST)

A candidate track CSV must be in ONE of two conventions; the validators differ by which they read.

| Convention | id column | values | Validators that consume it |
|---|---|---|---|
| **Geometry-slot** | `fish_id` or `final_fish_id` | `ID_1`, `ID_2`, `ID_3` (female = **ID_2**) | `compare_boris_trajectory.py`, `render_crossing_montage.py` |
| **Sexed dual-line** | `id_final` | `F`, `M1`, `M2` | `analyze_mate_choice.py`, `render_qc_4panel_fast.py` |

- Required columns in all cases: `frame, <id-col>, X, Y, Z` (mm; cam0=top XY, cam1=side1 XZ, cam2=side2 YZ).
- `render_crossing_montage.py` auto-prefers `final_fish_id` over `fish_id`; it colours by **sorted id** (arbitrary ints/strings → distinct colours), so identify the female by eye once in a side view, then watch her colour.
- `compare_boris_trajectory.py` pins the female by `--female-slot` (default `ID_2`) and validates the two males only as a **dyad** (they are not appearance-separable in the geometry track).
- The existing recovered backbone `recon_nofill/.../round5_260611_recovered_3d.csv` is **geometry-slot** convention (`frame,fish_id,X,Y,Z,is_interpolated,source`). A new candidate may instead be sexed (`id_final` F/M1/M2) — pick the matching block below.

Reference candidate paths used throughout (substitute your candidate):
```
# geometry-slot backbone (geometrically correct, BORIS-valid, swaps at dense contact):
CAND_SLOT = Z:\StephensU\medaka\runs\udmt_validation\recon_nofill\20251225_K_1_2_FCMC\round5_260611_recovered_3d.csv
# sexed dual-line interp track (analyze_mate_choice native form):
CAND_SEXED = runs\dual_line\lineB_fulltimeline\step2\round5_260611_3d_individual_id_interp.csv
```

Shared real paths (verified to exist on this machine):
```
BORIS  = G:\.shortcut-targets-by-id\14OI1k7IaNOdC5yaC9GnYyCi2u-gfGLEs\OIST 1215-0122\boris_events\20251225_K_1_2_FCMC.csv
SLP    = Z:\StephensU\medaka\sleap_output\round5_260611\20251225_K_1_2_FCMC      (top_cam.slp/side1_cam.slp/side2_cam.slp = 2-node mouth+tail)
CALIB  = calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models
VTOP   = Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\top_cam\cam0_top.mp4
VSIDE1 = Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\side1_cam\cam1_side.mp4
VSIDE2 = Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\side2_cam\cam2_side.mp4
FFMPEG = C:\ffmpeg\ffmpeg-8.0.1-full_build-shared\bin\ffmpeg.exe
```

### K_1_2 BORIS facts (from report/20260630/boris_cohort/data, cohort summary)
- **Spawn**: `t = 818.572 s` → `Image index`/frame = **135883** (166 fps). `analyze_mate_choice.py` ships default `--spawn-frame 135888` (818.6 s rounded); the difference is 5 frames / 0.03 s — use **135883** for the strict spawn-proximity test, both are within the ±1 s window the scripts already use.
- **Display codes**: `m1tf` = male-1→female courtship (84.77 s? no — see below), `m2tf` = male-2→female courtship. For K_1_2: `m1tf_duration_s = 4.168 s` vs `m2tf_duration_s = 41.825 s` → BORIS coder saw **m2** courting the female far more (`boris_more_courtship_subject = m2tf`). Male→male (competitive) total `m1tm2+m2tm1+m1m2` is small. So the expected direction is: female associates more with the **m2** male; **courtship (84.8 s) >> competitive (9.2 s)** (ratio ~9.25).
- Wrap (mating): 3 events, accepted n=2 / rejected n=1, total 59.7 s; ready_time 65.0 s.
- These are the ground-truth directions the candidate track must reproduce (criterion iii).

---

## 1. BORIS ↔ trajectory agreement

### 1a. Geometry-slot candidate (female = ID_2, males as dyad)
```powershell
.venv\Scripts\python.exe scripts\analysis\compare_boris_trajectory.py `
  --boris  "G:\.shortcut-targets-by-id\14OI1k7IaNOdC5yaC9GnYyCi2u-gfGLEs\OIST 1215-0122\boris_events\20251225_K_1_2_FCMC.csv" `
  --tracks "Z:\StephensU\medaka\runs\udmt_validation\recon_nofill\20251225_K_1_2_FCMC\round5_260611_recovered_3d.csv" `
  --female-slot ID_2 --male-slots ID_1 ID_3 `
  --label 20251225_K_1_2_FCMC `
  --out-dir report\20260630\k12_appearance\boris_compare
```
Outputs: `events_with_trajectory_metrics.csv`, `behavior_trajectory_summary.md`,
`ethogram_distance_timeline.png`, `spawn_closeup.png`, `behavior_distance_boxplot.png`,
`behavior_counts.csv`. Read `behavior_trajectory_summary.md` for the pass numbers
(courtship/touch/wrap median female–male distance must DROP below baseline; spawn nearest-male).

> NOTE: `compare_boris_trajectory.py` reads `fish_id` (not `final_fish_id`). The recovered backbone uses `fish_id`/`ID_*` directly. If the candidate uses `final_fish_id`, add a one-line rename or pass a slot-named copy. If the candidate is **sexed** (`id_final` F/M1/M2), map F→ID_2, M1→ID_1, M2→ID_3 into a `fish_id` column first (the script has no `--id-col` flag).

### 1b. Sexed dual-line candidate (downstream payoff — the real prize)
`analyze_mate_choice.py` is the downstream consumer; it needs `id_final ∈ {F,M1,M2}`.
```powershell
.venv\Scripts\python.exe scripts\analysis\analyze_mate_choice.py `
  --track "runs\dual_line\lineB_fulltimeline\step2\round5_260611_3d_individual_id_interp.csv" `
  --boris "G:\.shortcut-targets-by-id\14OI1k7IaNOdC5yaC9GnYyCi2u-gfGLEs\OIST 1215-0122\boris_events\20251225_K_1_2_FCMC.csv" `
  --spawn-frame 135883 `
  --out-dir report\20260630\k12_appearance\mate_choice
```
This is the *payoff* test: it reports per-male female association index, temporal
trend, courtship bouts, courtship-vs-aggression breakdown, and spawn attribution,
and prints the BORIS m1tf/m2tf cross-check. A correct track makes the geometry agree
with the BORIS direction (more female time near the m2-side male; courtship >> aggression).

---

## 2. Eye-review of the highest-suspicion crossings (`--events-csv` driven)

A worklist is pre-built so the montage hits the riskiest moments instead of
auto-ranking by proximity. It pools, in suspicion order (lowest `rank` first):
1. handoff frames **8042, 8809, 8852** (mouth>centroid / any-unambiguous-anchor / one-view-gap cases from the K_1_2 handoff);
2. the BORIS **spawn** frame **135883** (must look correct here);
3. all 73 detector swaps from `Z:\StephensU\medaka\runs\udmt_validation\qc\_swaps.txt`, **longest span first** (8521, 8826, 12643, 41085, 45601, 58599 … included).

Worklist file (built this session, columns `rank,start,end,center,span_frames,pair,source`):
```
report\20260630\k12_appearance\k12_validation_worklist.csv
```
`render_crossing_montage.py --events-csv` reads `center` (or `start`+`end` midpoint);
`--events-sort rank` replays them highest-suspicion first; `--n-events` caps how many.

### 2a. Top-20 worklist montage (handoff + spawn + longest swaps)
```powershell
.venv\Scripts\python.exe -m scripts.analysis.render_crossing_montage `
  --csv-3d "Z:\StephensU\medaka\runs\udmt_validation\recon_nofill\20251225_K_1_2_FCMC\round5_260611_recovered_3d.csv" `
  --slp-dir "Z:\StephensU\medaka\sleap_output\round5_260611\20251225_K_1_2_FCMC" `
  --model-folder "calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models" `
  --video-top   "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\top_cam\cam0_top.mp4" `
  --video-side1 "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\side1_cam\cam1_side.mp4" `
  --video-side2 "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\side2_cam\cam2_side.mp4" `
  --events-csv  "report\20260630\k12_appearance\k12_validation_worklist.csv" `
  --events-sort rank --n-events 20 `
  --window-s 1.5 --zoom-crop 500 `
  --output "report\20260630\k12_appearance\k12_validation_montage_top20.mp4"
```
(`--csv-3d` reads `final_fish_id` else `fish_id`; no need to rename for the montage.
`--zoom-crop 500` fills the panel with the fish so body shape is readable.
Drop `--zoom-crop`/raise `--window-s 2.0` for wider context. If the top video is
unreadable, add `--views side1,side2`.)

### 2b. Spawn-only close look (must hold female colour through the wrap)
```powershell
.venv\Scripts\python.exe -m scripts.analysis.render_crossing_montage `
  --csv-3d "Z:\StephensU\medaka\runs\udmt_validation\recon_nofill\20251225_K_1_2_FCMC\round5_260611_recovered_3d.csv" `
  --slp-dir "Z:\StephensU\medaka\sleap_output\round5_260611\20251225_K_1_2_FCMC" `
  --model-folder "calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models" `
  --video-top   "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\top_cam\cam0_top.mp4" `
  --video-side1 "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\side1_cam\cam1_side.mp4" `
  --video-side2 "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC\side2_cam\cam2_side.mp4" `
  --events-csv  "report\20260630\k12_appearance\k12_validation_worklist.csv" `
  --events-filter "source=='boris_spawn_818.572s'" `
  --window-s 4.0 --zoom-crop 600 `
  --output "report\20260630\k12_appearance\k12_validation_spawn.mp4"
```

### 2c. (Optional) Full-length 2x2 QC for a SEXED candidate
`render_qc_4panel_fast.py` needs `id_final ∈ {F,M1,M2}` and is slow but
parallelisable (`--workers K`). Use only after 2a/2b look clean.
```powershell
.venv\Scripts\python.exe -m scripts.visualization.render_qc_4panel_fast 20251225_K_1_2_FCMC `
  --slp-dir   "Z:\StephensU\medaka\sleap_output\round5_260611\20251225_K_1_2_FCMC" `
  --track-csv "runs\dual_line\lineB_fulltimeline\step2\round5_260611_3d_individual_id_interp.csv" `
  --video-dir "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC" `
  --calib     "calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models" `
  --out "report\20260630\k12_appearance\k12_qc_4panel_full.mp4" `
  --workers 8 --fps 60
```
(For a quick spawn-window sanity instead of full length, add `--start-frame 135000 --end-frame 136800`.)

---

## 3. PASS criteria (a candidate must meet ALL FOUR)

| # | Criterion | How to check | Pass threshold |
|---|---|---|---|
| **(i)** | Exactly **1F + 1M1 + 1M2 per frame** | per-frame count of distinct ids on frames with any track; no frame with 2 ids on one slot, no >3 ids | ≥99% of tracked frames have exactly one of each id; 0 frames with a duplicated id. (Quick check: `df.groupby('frame')[idcol].nunique().value_counts()` should be dominated by 3.) |
| **(ii)** | **Female ≤ ~12 mm from a male at spawn** | `compare_boris_trajectory.py` spawn block (`spawn_d_F_M1_mm` / `spawn_d_F_M2_mm`, ±1 s median around frame 135883); or read the spawn close-up | min(female–M1, female–M2) median over ±1 s of frame 135883 **≤ 12 mm** (BORIS-validated wrap/spawn contact is 5–8 mm in N=3 prior work). |
| **(iii)** | **BORIS direction agreement** | `analyze_mate_choice.py` summary (association index, courtship-vs-aggression) and `compare_boris_trajectory.py` behaviour-distance table | (a) female associates **more with one male** (association index clearly ≠ 0.5, and the favoured male matches BORIS `m2tf`>`m1tf` direction); (b) **courtship time >> aggression** (geometry: female–male courtship-range time far exceeds male–male dash time; BORIS ratio ~9.25); (c) courtship/touch/wrap median female–male distance **drops below baseline**. |
| **(iv)** | **Female colour holds through top crossings** | eye-review montage 2a (top-20) + spawn 2b | At every handoff frame (8042/8809/8852), the spawn, and the longest detector swaps (58599, 226714, 45601, 41085, 8521 …), the female-coloured marker **stays on the same physical fish across the occlusion** (no colour jump to a male). A single sustained swap = FAIL. |

Decision: **ACCEPT** only if all four pass. If (iv) fails at specific frames, those
frames are the worklist for the appearance/female-anchor fix (handoff aspect: pure
geometry is exhausted on K_1_2; the residual is identity swaps at dense contact).

---

## 4. Artifacts produced by this task (no videos rendered)
- `report/20260630/k12_appearance/validation_runbook.md` (this file)
- `report/20260630/k12_appearance/k12_validation_worklist.csv` (77 events: 3 handoff + 1 spawn + 73 detector swaps, ranked highest-suspicion first; drives the montage `--events-csv`)
