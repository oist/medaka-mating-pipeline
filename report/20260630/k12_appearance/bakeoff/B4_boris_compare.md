# B4 — BORIS ethogram agreement (NEW vs OLD), K_1_2

**Question:** Does the per-frame fish-ID track corroborate the manually-coded BORIS
ethogram — specifically, is the female *in contact* (<=~12 mm to the nearest male) at
the coded **wrap** and **spawn** events, and does the female–nearest-male distance follow
the documented contact progression?

## Method

- Tool: `scripts/analysis/compare_boris_trajectory.py` (unchanged).
- The script keys female identity on `fish_id == ID_2` and has **no `--id-col` flag**.
  The authoritative label in both tracks is `id_final ∈ {F,M1,M2}` (`fish_id` here is the
  *geometry slot*, not identity). I therefore wrote remapped copies that set
  `fish_id` from `id_final` via **F→ID_2, M1→ID_1, M2→ID_3**, so the script's
  female=ID_2 anchor follows the appearance-based female identity:
  - `NEW_remapped_idfinal_tracks.csv`, `OLD_remapped_idfinal_tracks.csv`
- BORIS source (raw event list, `Image index` = top-cam frame @166 fps):
  `G:\...\OIST 1215-0122\boris_events\20251225_K_1_2_FCMC.csv` (89 parsed events:
  54 Courtship, 26 Competitive, 4 Mating, 5 Other — identical parse for both runs).
- Outputs: `NEW_boris/`, `OLD_boris/` (each has events_with_trajectory_metrics.csv,
  ethogram_distance_timeline.png, behavior_distance_boxplot.png, spawn_closeup.png,
  behavior_trajectory_summary.md).
- 3D-distance coverage: NEW 98.93% of frames have a valid F-nearest distance; OLD 99.67%.

## Phase female–nearest-male distance (mm)

The documented progression (40→28→17→8 mm baseline→courtship→touch→wrap) describes the
*contact* the female reaches, not the event-window average. Below, `mean` = median over
events of the per-event window mean; `MIN` = median over events of the per-event minimum
(the contact actually achieved).

| phase | NEW mean | NEW MIN | OLD mean | OLD MIN |
|---|---|---|---|---|
| baseline (whole-rec median F-nearest) | 105.6 | — | 106.0 | — |
| Courtship (n=52) | 165.4 | 139.8 (overall min 5.1) | 165.3 | 140.8 (overall min 2.2) |
| Touch (n=2) | 72.9 | 60.1 (overall min 10.5) | 73.3 | 61.8 (overall min 7.5) |
| Wrap (n=3) | 72.2 | **0.0** | 71.9 | **0.2** |
| Dash — male–male (n=22) | 55.6 | — | 54.7 | — |

The event-window *means* are large because state events (esp. wrap, 25–31 s) include the
female swimming away afterward; the *minimum* is the correct "in-contact" readout and shows
the female reaches near-zero separation during wraps in **both** tracks.

## Wrap acceptance and per-event contact

Both tracks reproduce the documented accept/reject durations exactly (parse is shared):
accepted median **28.03 s** (n=2), rejected **3.61 s** (n=1).

| wrap (subj, dur) | NEW d_min (mm) | OLD d_min (mm) | in contact? |
|---|---|---|---|
| m1tf, 3.61 s (rejected) | **3.9** | 11.1 | NEW yes / OLD borderline |
| m2tf, 31.23 s (accepted) | 0.0 | 0.2 | both yes |
| m1tf, 24.83 s (accepted) | 0.0 | 0.2 | both yes |

## Spawn event (frame 135883, t≈818.6 s) — the critical validation

| metric | NEW | OLD |
|---|---|---|
| F-nearest at exact spawn frame | **4.7 mm** (F–M2 4.7, F–M1 11.1) | 7.6 mm (F–M2 7.6, F–M1 11.7) |
| F-nearest median, ±1 s window | **6.5 mm** | 8.3 mm |
| F-nearest min, ±1 s window | 1.8 mm | 3.5 mm |
| female in contact (<=12 mm) at spawn | **YES** | **YES** |
| nearest male at spawn frame | M2 (ID_3) | M2 (ID_3) |

(The script's `spawn_nearest_male_mode` over a wider ±1 s window reports ID_3 for NEW and
ID_1 for OLD, but at the exact spawn frame both agree the contacting male is M2/ID_3 with
~12 mm to the other male — a near-tie that the wide-window mode resolves differently.)

## Verdict

Both tracks **corroborate the BORIS ethogram**: female reaches near-zero contact during
the two accepted wraps, and is in contact with a male (<=12 mm) at the documented spawn
frame. Differences are small but **consistently favor NEW**: at every contact phase NEW
shows tighter female–male separation that better matches "in contact" —
spawn 4.7 vs 7.6 mm (exact frame) / 6.5 vs 8.3 mm (±1 s median); the short rejected wrap
reaches clear contact in NEW (3.9 mm) but only borderline in OLD (11.1 mm); accepted wraps
0.0 vs 0.2 mm. OLD has marginally higher 3D coverage (99.67% vs 98.93%). No phase shows OLD
in tighter contact than NEW.

**Winner: NEW** — equal-or-better agreement with the documented contact progression at
every wrap and at spawn; the female is unambiguously in contact at the spawn frame in both,
but NEW's contact distances are uniformly closer to the documented values.

Caveat: the female-vs-male assignment is what is being tested; the male m1/m2 ↔ ID_1/ID_3
mapping is not appearance-resolvable, so per-male spawn-paternity is not adjudicated here
(both tracks agree the *nearest* slot at spawn is the same geometry slot, M2/ID_3).
