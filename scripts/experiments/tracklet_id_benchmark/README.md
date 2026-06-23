# Tracklet → 3-fish identity benchmark

A small, reproducible harness that **compares the methods we already have** for
assigning short 2D/3D tracklets to exactly three biological fish IDs and for
reconnecting identity through hard events (crossings, occlusions, false
positives). It is deliberately *not* a new tracker — every method it scores is an
adapter over an existing pipeline output.

## Why this exists

Across prior spikes (see the project memory) every method family converged to the
same wall:

- **geometry + 3D bind** carries identity through ~89–96% of single-view-ambiguous
  frames, but the per-frame Hungarian linker can *smooth-swap* through the hard
  core (pair < ~12 mm in 3D);
- **appearance (idtracker.ai v6)** trains a usable medaka embedding but silhouette
  is only ~0.73 (it needs ≥0.91 for <1% per-frame error) — moderate, not enough;
- **motion/dynamics resolvers (UDMT-style)** detect crossings at 100% recall but the
  motion/geometry cues separate the two fish by only ~0.13 (median rel. diff), so
  they work as a **review-ranker, not an auto-resolver**.

The decisive blocker is the **absence of ground truth**: with ~6 eye-checked events
and 0 `suspect_correct` labels we cannot say which method is *right* at a crossing.

So this harness does three things:

1. computes every metric that **does not need GT** (coverage, switch/spike rate,
   unknown fraction, view agreement, reconnection proxy) so methods can still be
   compared today;
2. computes **inter-method agreement** (Hungarian-aligned per-frame labels) — where
   methods disagree is exactly where annotation pays off;
3. emits a **prioritized annotation worklist** and auto-activates GT scoring the
   moment labels appear, so the experiment becomes decisive with minimal labeling.

## The assignment problem (graph view)

- **Nodes** = reliable per-view concrete tracklets (stIFD edge-extended gate) and
  their 3D-bound slots.
- **Edges** = candidate *same-fish* links across a time gap / hard event.
- **Edge features** (already produced by the existing tools, surfaced here per
  window): 3D geometry continuity (extrapolated-velocity fit, `no_swap_err` vs
  `swap_err` from the resolver), reprojection residual (`bind_resid`,
  `*_best_pre_link.json` `errors`), bind margin/consistency, time gap
  (`max-gap`), per-view agreement, appearance similarity (idtracker
  `id_probabilities`, optional), and the crossing/occlusion flag (`in_hardcore`).

A *method* is any rule that resolves these edges into ≤3 connected components +
an "unknown/mix" bucket. The harness scores the resulting per-frame assignment.

## Method families compared

| key | family | source on disk | needs |
|-----|--------|----------------|-------|
| `geometry_3d` | geometry-only (production bind) | `<recon>/..._3d_identified.csv` | always |
| `tracklet_bind` | geometry tracklet-linking | `<bind_dir>/*.tracklet_bound.csv` | bind run |
| `motion_resolver` | temporal/dynamics overlay | `<bind_dir>/hardcore_events_resolved.csv` | resolver run |
| `hybrid` | tracklet_bind + resolver overlay | the two above | both |
| `appearance_idtracker` | appearance classifier | idtracker `trajectories.csv` | `.venv-idtrackerai` (optional) |

`appearance_idtracker` is optional because idtracker.ai lives in a separate env;
when its `trajectories.csv` is absent the harness records it as *unavailable* with
reproducible instructions (the "closest reproducible comparison" rule), so the
table stays honest rather than fabricating a row.

## Pilot windows (taxonomy)

Selected by `windows.py` from `hardcore_events_resolved.csv` + GT notes. Every run
covers: `early_fp` (single→few fish + false positives), `easy_separated`
(upper-bound), `crossing_clear` (a velocity-consistent pass), `dwell_ambiguous`
(the unresolvable hard core), and one window per `suspect_swap` (likely real
errors). Use `--max-windows` / `--kinds` to shrink the pilot.

## Metric panel (`metrics.py`)

GT-free: `coverage`, `exactly_three`, `unknown_frac`, `speed_spike_rate`
(per-fish 3D jumps > thr), `reconnection_rate` (ID continuity across in-window
events), `view_agreement` (tracklet methods). Orchestrator-level:
`inter_method_agreement`. GT-gated (auto): `gt_swap_accuracy`,
`gt_event_coverage`.

## Outputs (per run, under `--out-dir`, outside git)

```
<out-dir>/<run_tag>/
  config.resolved.json        # exact params used
  windows.csv                 # the selected pilot windows
  run.log                     # full log
  trials/<window>__<method>.json   # one metric record per (window, method)
  summary.csv                 # all trials, one row each
  summary.json                # aggregate + inter-method agreement
  annotation_worklist.csv     # prioritized: where to label, what type, how many
  figures/<window>_timeline.png    # tracklets coloured by assigned fish ID
  figures/<window>_traj3d.png      # 3D trajectory coloured by assigned fish ID
```

## Running

```powershell
# dry-run: list windows x methods, write windows.csv, touch nothing expensive
.venv\Scripts\python.exe -m scripts.experiments.tracklet_id_benchmark.run_benchmark `
  --config config/experiments/tracklet_id_benchmark.example.toml --dry-run

# small pilot (2 windows), real metrics + figures
.venv\Scripts\python.exe -m scripts.experiments.tracklet_id_benchmark.run_benchmark `
  --config config/experiments/tracklet_id_benchmark.example.toml --max-windows 2

# full pilot
.venv\Scripts\python.exe -m scripts.experiments.tracklet_id_benchmark.run_benchmark `
  --config config/experiments/tracklet_id_benchmark.example.toml
```

Paths are resolved from the config + `configs/paths.local.toml`; no machine drive
letters are hardcoded in code. The run is resumable: an existing
`trials/<window>__<method>.json` is reused unless `--force`.

## What a conclusion looks like

The harness ranks methods by the GT-free panel and reports inter-method
disagreement. If (as expected) the panel cannot separate the top methods, the
`annotation_worklist.csv` names the exact event windows to label, the label type
(fish-ID at crossing / same-vs-different tracklet pair / crossing-vs-mix), the
count, and which method each label is expected to discriminate.
