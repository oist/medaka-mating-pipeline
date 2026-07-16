# Motion-aware identity repair

The round7 K_1_2 speed spikes are primarily biological-label transitions, not
teleports in the anonymous 3D geometry slots. The three geometry trajectories
stay below the 8 mm/frame gate, while appearance-derived `id_final` can change
slot mapping on its stride-10 evaluation grid.

## Motion state and three conservative repair layers

The online 3D linker in
`scripts/reconstruction_3d/cross_frame_tracking.py` keeps one persistent state
object per geometry identity:

- last valid frame and XYZ position;
- velocity and the interval over which it was measured; and
- acceleration when enough observations are available.

Every frame is still solved simultaneously with one Hungarian assignment. The
complete cost matrix is computed from the pre-update states; fish states are
updated only after the one-to-one assignment is fixed. Position remains the
default cost. Velocity prediction, tangent discontinuity, and acceleration
discontinuity are opt-in, so the validated production command is unchanged.

The post-link biological-ID repair in
`sam2_id_experiment/src/id_model/motion_identity_optimizer.py` treats each
frame as one of the six possible slot-to-`F/M1/M2` permutations. Its production
candidate is deliberately conservative:

1. Remove a short `A -> B -> A` permutation run only when the run is at most
   `max_flicker_frames` and entering it causes a jump above the threshold.
2. For each remaining discontinuous transition, search a bounded local window
   and move the boundary only when the new boundary is both better and at or
   below the jump threshold.
3. Cluster the remaining jumps into bounded local windows and run a six-state
   minimum-edit dynamic program inside each window. The biological mapping at
   both window ends is fixed, every accepted transition must be at or below the
   jump threshold, and a window is changed only when a complete feasible path
   exists. Infeasible windows are reported and left unchanged.

This preserves the order of sustained identity regimes. The six-state global
Viterbi solver is retained as `--strategy viterbi` for experiments, but it is
not the default because smooth motion alone cannot resolve biological identity
over long ambiguous intervals.

## Run the conservative repair

```powershell
.venv\Scripts\python.exe -m sam2_id_experiment.scripts.52_optimize_identity_motion `
  <round7_3d_individual_id_interp_debounced.csv> `
  --output-csv <round7_3d_individual_id_motion_layered_local.csv> `
  --report-json <round7_motion_layered_local_qc.json>
```

Defaults are `--strategy layered`, `--jump-threshold-mm 8`,
`--max-flicker-frames 166`, `--search-radius-frames 350`, and
`--local-window-radius-frames 166`. The input is never overwritten. The output
retains the old labels in `id_final_pre_motion`, marks changed rows as
`id_source=motion_layered`, and writes a JSON report containing before/after
speed, permutation, one-each, changed-span, and per-layer diagnostics.

Use `--strategy relocate` to skip short-flicker removal. Use `--strategy
viterbi` only for an explicitly reviewed experiment.

## K_1_2 round7 validation

Full-track validation covers 348,600 frames / 1,045,800 rows:

| QC | Input | Flicker + relocation | Final local repair |
| --- | ---: | ---: | ---: |
| >8 mm identity-step rows | 649 | 197 | 111 |
| Unique frames with a >8 mm step | 318 | 93 | 53 |
| Permutation changes | 340 | 172 | 150 |
| Frames with exactly one F, M1, M2 | 100% | 100% | 100% |

The final repair edited 11,038 frames. Its bounded local stage accepted 22 jump
clusters and rejected 38 clusters for which no <=8 mm path connected the fixed
endpoint identities inside the +/-166-frame window. Those 38 clusters are left
visible for review instead of forcing a broad identity rewrite.

Against the independent round5 trajectory, final agreement within 20 mm is
98.58% for F, 97.25% for M1, and 97.59% for M2. All three improve over the
flicker + relocation result (98.53%, 97.04%, and 97.52%, respectively). The
remaining 53 jump frames need visual review or an independent appearance/
segmentation identity anchor before further automatic changes are justified.

The rejected global-Viterbi trial removed all >8 mm jumps but reduced female
agreement within 20 mm to 86.94%. That result demonstrates why trajectory
smoothness alone is not a sufficient identity metric.
