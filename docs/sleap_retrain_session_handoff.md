# Handoff: SLEAP active-learning frame-mining tool

This file is a self-contained brief for a fresh Claude Code session that will
design and implement the next concrete deliverable in the medaka pipeline:
the SLEAP training-frame mining + annotation-seed tool. Paste the
**Prompt to start the new session** block (bottom of this file) into the new
session, optionally prefixed with `ultrathink` to request deeper reasoning.

---

## What this session is being asked to do

Implement `scripts/sleap/mine_training_frames.py`. Walk the 24 inferred
experiments under `Z:/StephensU/medaka/sleap_output/`, score every frame on
the active-learning signals listed below, diversify the picks, and produce:

1. `mined_frames.csv` — `experiment, view, frame, signals_fired, suspicion_score, time_bin, video_quota_bucket`
2. `annotation_seed.slp` — a single SLEAP labels project containing every
   picked frame, with `mouth` (existing prediction) + a freshly seeded `tail`
   node placed near each instance. The annotator opens this in SLEAP-label
   and corrects rather than annotating from scratch.

The annotation, training, and re-inference are *not* in scope for this
session. The deliverable is the mining + seed-project tool, validated on at
least one experiment end-to-end.

---

## Background — read first

The project's auto-memory will load these on session start:
- `project_sleap_failure_modes.md` — the three named failure modes and their
  structural causes, with file:line evidence.
- `project_sleap_retrain_plan.md` — decisions already made, signal list,
  skeleton-extension and tail-seeding plan.
- `feedback_iteration_loop.md` — how the user prefers to iterate.

Also read in the repo (in priority order):
- `scripts/reconstruction_3d/load_2d_slp.py` — current 2D loader; shows the
  existing top-N truncation and centroid extraction. The mining tool needs
  similar primitives but should consume *raw* per-instance points, not
  collapsed centroids.
- `scripts/reconstruction_3d/reconstruct_from_slp.py` — has
  `save_best_pre_link_json_to` / `load_best_pre_link_json` whose output
  contains per-fish reprojection error (signal 4 below). Use it where pre-link
  JSON exists for an experiment.
- `scripts/sleap/batch_track.py`, `scripts/sleap/batch_predict.py` — show how
  SLEAP is currently driven; do not modify.
- `scripts/hpc/sleap_chunk.slurm` — the inference is locked to
  `--max_instances 3` here; future re-inference will need this lifted, but
  not now.
- `configs/sleap/inference.yaml`, `configs/project.yaml` — hard-coded
  `max_instances: 3`; not changed in this session.

Existing artifacts to use:
- `Z:/StephensU/medaka/sleap_output/<experiment>/{top_cam,side1_cam,side2_cam}.slp`
  — merged per-view predictions for 24 experiments. Each `.slp` has a
  single-node skeleton (`mouth`) and per-instance score.
- `Z:/StephensU/medaka/reconstruction_3d/20251225_K_1_2_FCMC_best_pre_link.json`
  — pre-link JSON for the 30-min run on this one experiment. Useful for
  signal 4 (residuals) on this experiment only; other experiments don't have
  it yet (would require running the 3D reconstruction first).

---

## Hard constraints

- **Skeleton stays single-node `mouth` in the SLEAP `.slp` files we read
  from.** The new node (`tail`) is added in the *output* annotation seed,
  not back-edited into the existing inference outputs.
- **Frames before `5 * 60 * fps` of each video are skipped** — no animals
  yet. Default `fps = 166`. Make `fps` and skip-seconds configurable; default
  to those values.
- **Mine from the existing top-3 predictions only.** Re-inference with
  `max_instances=6` is deferred to after the next training round. We
  therefore cannot directly observe what the cap dropped; we infer it from
  cross-view count mismatch (signal 3) and stationary-instance evidence
  (signal 1).
- **Cross-platform Windows path handling.** Network drives `Z:` and `G:`
  appear as forward-slashed paths in shell calls and as `\\bucket.oist.jp\…`
  when resolved. Use `pathlib.Path` and `Path.resolve()`; don't hand-edit
  separators.
- **Memory-safety for the full corpus.** 24 experiments × ~350k frames ×
  3 views ≈ 25M (frame, view) records. Stream per-experiment, write
  per-experiment intermediate parquet/CSV, then aggregate. Don't build a
  single in-memory dataframe of everything.
- **Don't break tests** (`pytest -q`). Add at least one test if you modify
  shared helpers.

---

## Goals

1. Produce a candidate-frame list whose composition reflects the failure
   modes (reflection FPs, mouth↔tail flips, max-3 cap dropping a fish) and
   has good diversity across experiments, time, behavior, and tank position.
2. Produce a SLEAP labels project that loads cleanly in `sleap-label` and
   shows the user, for each picked frame:
   - The original predicted `mouth` instances (positions unchanged).
   - A `tail` node seeded near each `mouth` so the annotator drags it to
     the correct location.
   - A note (frame-level metadata or filename hint) about which signals
     fired, so the annotator knows what to look for.
3. Be re-runnable cheaply: cached per-experiment intermediates so adding a
   new experiment or tweaking thresholds doesn't re-scan everything.

---

## Done conditions

- `scripts/sleap/mine_training_frames.py` exists with a clean argparse CLI.
- Running it end-to-end on one experiment (`20251225_K_1_2_FCMC` is the
  reference) produces both output files without errors.
- Each signal is implemented and verifiable from the output CSV (the
  `signals_fired` field per frame is human-inspectable).
- The tail-seeding logic uses per-view 2D velocity from the existing
  predictions when available, falls back to a documented default offset
  otherwise.
- The `annotation_seed.slp` is loadable by `sleap-io` and contains the
  expected number of frames with two-node instances each.
- Smoke test: a tiny pytest checking that, given a synthetic 100-frame
  per-view stub, the mining function returns the expected schema and
  diversification picks roughly evenly across time bins.
- Documentation: a short section in `scripts/sleap/README.md` describing
  the tool and its outputs.

---

## Open questions to resolve with the user upfront

Ask before writing the bulk of the implementation:

1. **Body length in pixels per view.** Needed for tail-seed offset (signal-2
   threshold and seed placement). Options: (a) user gives a measured value
   per view (top, side1, side2); (b) script estimates from the 95th-percentile
   per-instance frame-to-frame step length over a multi-minute window
   (rationale: a near-tail-flip appears as a body-length-scale jump, and the
   long tail of step-length distribution is dominated by these). Default to
   (b) with a printed estimate the user can override.
2. **Target candidate count.** Affects diversification aggressiveness.
   Defaults to consider: 500 (focused first round), 1500 (mid), 5000 (broad).
   Active-learning literature suggests 500–1000 for a first iteration; if the
   user wants 5000 we can do it but the diversification matters more.
3. **Annotator preferences for the tail seed default direction.** When
   velocity is unavailable (stationary or first-frame instances), where
   should `tail` default to? Proposed default: `mouth + (0, +BL_px)` — i.e.
   one body length downward in image space. Confirm direction.
4. **Are there any experiments whose data should be excluded from mining?**
   E.g. recordings with known bad lighting or partial-tank captures. Default
   to including all 24.

---

## Implementation plan to think through

(The new session has more context window — design carefully before coding.)

### Architecture

- Single CLI; sub-stages can be `--stage features|select|seed|all`.
- Streaming per experiment per view: build a generator over labeled frames,
  emit per-frame feature row, write to a parquet file under
  `Z:/StephensU/medaka/training_frame_mining/<experiment>/<view>.parquet`.
  Re-runnable: skip if parquet exists and is newer than the `.slp`.
- Aggregation step reads all parquet files, joins on `(experiment, frame)`
  to compute cross-view signals, and writes `mined_frames.csv`.
- Seed-project step reads the candidate CSV and the per-view `.slp` files
  and assembles a single output `.slp` with `sleap-io`.

### Signals — implementation notes

Implement each as a pure function over a per-view feature dataframe (or, for
cross-view ones, a joined dataframe). Each frame's `signals_fired` is a
sorted list of signal IDs that triggered.

1. **Stationary instances (S1)** — per view, NN-link existing predictions
   frame-to-frame with a generous gate (e.g. 50 px). Compute per-track
   rolling-window position std over the last 100 frames. Flag any frame
   where any instance has `std < 0.5 mm-equivalent` over a window > 100
   frames. Use this also to retire stationary tracks from cross-view
   matching.
2. **Adjacent-frame jumps (S2)** — within a track, flag frames where
   `|Δposition|` is in `[0.6 × BL_px, 1.5 × BL_px]` and the track is
   otherwise active before and after (so this isn't a re-acquisition jump).
3. **Cross-view count mismatch (S3)** — at the joined-frame level, flag
   `(max(n_inst) - min(n_inst)) >= 1`. Severity scales with the gap.
4. **Per-fish reprojection residual (S4)** — only when pre-link JSON exists
   for the experiment. From the JSON's `errors` field, flag frames where
   any `errors[a/b/c] > 5000` (current cutoff) but the frame survived
   triplet selection.
5. **Score dispersion (S5)** — flag frames where
   `max(score) - min(score) > 0.3` AND `min(score) < 0.6` AND `n_inst >= 2`.
6. **Per-experiment quota (D1, diversification)** — implemented in the
   selection step, not as a per-frame signal.
7. **Time-bin diversification (D2)** — bin each video into N (default 30)
   equal time buckets after the 5-min skip; cap candidates per bucket.
8. **Behavioral state (D3)** — proxy: when 3D pre-link is available for an
   experiment, compute pairwise inter-fish distance; tag each frame with
   `rest|approach|clinch` using percentile thresholds. Sample roughly evenly
   per state. Skip this signal silently for experiments without 3D output.
9. **Spatial coverage (D4)** — bin tank XYZ into 4×4×4 voxels; cap per
   voxel. Same gating as D3.
10. **Negative-example frames (D5)** — explicit quota: at least 10–20% of
    selected frames should be ones where (n_inst == 3) AND at least one
    instance is suspected reflection (S1 fired). Tagged so the annotator
    knows to delete the spurious instance.

### Selection algorithm

After computing per-frame `suspicion_score = sum(weights[s] for s in
signals_fired)` (signals 1–5 only), run a greedy submodular pick:

```
picks = []
while len(picks) < target_count:
    candidate = argmax over remaining frames of:
        suspicion_score(f)
        + λ_exp * (1 if f's experiment under-represented in picks else 0)
        + λ_time * (1 if f's time bucket under-represented in picks else 0)
        + λ_state * (...)
        + λ_voxel * (...)
        - μ * 1[f has a neighbor (same exp, same view) in picks within W frames]
    picks.append(candidate)
```

`λ_*` and `μ` are tunable; sane defaults plus log them. Make `target_count`,
`min_inter_pick_frames` (W), and the per-bucket caps CLI flags.

### SLEAP labels-project assembly

Use `sleap-io`. Steps:
1. Build a Skeleton with nodes `[mouth, tail]` and edge `mouth → tail`.
2. For each picked `(experiment, view, frame)`:
   - Open the source `.slp`'s LabeledFrame at that index.
   - For each predicted instance there, create a new Instance under the new
     two-node skeleton:
     - `mouth` point = original predicted `mouth` (x, y, score).
     - `tail` point = seeded position (see below).
3. Set the LabeledFrame's `video` to point at the canonical mp4 path
   (`/bucket/.../<experiment>/<view>/<file>.mp4`) so SLEAP-label can show
   the frame.
4. Save with `sio.save_slp(labels, out_path)`.

Tail seed for an instance:
- If the per-view NN tracker (built for S1/S2) has a track containing this
  instance with valid velocity at `t-3..t-1`, use `unit_v = v / |v|`. Tail =
  `mouth - BL_px * unit_v`.
- Else: tail = `mouth + (0, +BL_px)` (default direction; confirm with user).

### Output formats

`mined_frames.csv` columns:
```
experiment, view, frame, suspicion_score, signals_fired (semicolon-joined),
n_inst, scores (semicolon-joined), time_bin, behavior_bin, voxel_id, picked
```
`picked` is the boolean for whether this row made the final candidate list.
The non-picked rows are useful for diagnostics; keep them in the CSV (or in
a separate parquet for size reasons — your call).

`annotation_seed.slp`:
- One Skeleton: `[mouth, tail]`.
- One Video object per (experiment, view) used.
- LabeledFrames for each picked frame, each with N instances (N = original
  prediction count, may be 1–3).

### Validation

- Smoke test on `20251225_K_1_2_FCMC` only; report:
  - signal frequency (how many frames fired each signal)
  - diversification achievement (per-experiment, per-time-bucket counts)
  - tail seed direction sanity (histogram of seed offset angles)
- Pytest with synthetic 100-frame stub.

---

## Suggested first actions for the new session

1. Read the three project memory files plus the four code files listed in
   "Background — read first".
2. Ask the user the four open questions, blocking on at least Q1 and Q2
   (body length, candidate count) — the others can default.
3. Sketch the per-view feature schema and the greedy-pick loop in pseudo-code
   in the response, get user agreement before coding the full thing.
4. Implement, then run the smoke test on one experiment.
5. Report distributions; iterate with the user on weights.

---

## Prompt to start the new session

Copy from here, optionally prepend `ultrathink`:

```
I'm continuing work on the medaka pipeline. The next deliverable is
implementing scripts/sleap/mine_training_frames.py — an active-learning
frame mining + annotation-seed tool for the next SLEAP retraining round.

Read docs/sleap_retrain_session_handoff.md in this repo for the full brief
(goals, constraints, signal list, implementation plan, done conditions). The
auto-memory's project_sleap_*.md and feedback_iteration_loop.md files have
the prior-session context.

Before coding, ask me the four open questions in the brief (especially body
length per view and target candidate count) and sketch the per-view feature
schema + greedy selection loop in pseudo-code so I can confirm the design.
Then implement with the streaming/parquet architecture described, validate
with the smoke test on 20251225_K_1_2_FCMC, and report signal-frequency and
diversification distributions.
```
