# New-session prompt: SLEAP top-down optimization, Phase 1

You are working in:

`C:\Users\machi\Documents\GitHub\medaka-mating-pipeline`

Read and follow the repository `AGENTS.md` before acting. Large videos, SLEAP
packages, checkpoints, and experiment outputs live on the OIST share; resolve
paths through `configs/paths.local.toml` and existing provenance rather than
hardcoding drive letters in reusable code.

## Objective

Outline the complete parameter-optimization plan for the SLEAP top-down
pipeline, then **execute Phase 1 only**: establish a trustworthy, reproducible,
common benchmark for Round 7 and Round 8. Do not launch new training or tune
parameters in this session.

Top-down has already been selected over bottom-up for this dataset. Round 7's
resolution-fair comparison showed that bottom-up still had roughly 3–4 times
more misses/merges during close contact even when keypoint precision became
similar. The optimization target is therefore the top-down pipeline:

1. mouth-anchored centroid detector;
2. centered-instance mouth/tail predictor;
3. full top-down inference;
4. downstream multiview/3D behavior.

Treat top and side views separately. The side view is the current priority.

## Current evidence to verify

Inspect source artifacts rather than trusting this summary blindly:

- Round 7 root:
  `\\bucket.oist.jp\bucket\StephensU\medaka\sleap_models\retrain_20260704`
- Round 8 root:
  `\\bucket.oist.jp\bucket\StephensU\medaka\sleap_models\retrain_20260722_round8`
- Round 7 summary:
  `retrain_20260704\EVAL_SUMMARY.md`
- Round 8 finalization report:
  `retrain_20260722_round8\finalize_report.json`
- Round 8 strict-mouth runs use names matching
  `round8_260723_strictmouth.*`.

Previously observed Round 8 validation results:

- top centroid: precision 1.000, recall 0.967, F1 0.983;
- side centroid: precision 0.955, recall 0.883, F1 0.918;
- top and side centered-instance PCK@5 are approximately 0.949;
- Round 8 strict centroid filtering retained 19/20 top validation images and
  only 29/40 side validation images;
- side centroid best validation loss occurred at epoch 8, followed by 30
  non-improving epochs;
- side centered-instance trained to epoch 199, with its best checkpoint around
  epoch 35.

Known evaluation issue:

- side1 frame 97219 contains a real tail-association error of about 45.2 px;
- side2 frames 97340 and 97630 produced reported errors of about 109 px and
  51 px, but inspection showed predictions close to their correct partial
  annotations. Those two large values came from instance matching on partially
  labeled fish, not genuine model misses.

The existing illustration for frame 97219 is:

`report/20260724/round8_qc/side1_frame_97219_annotation_vs_prediction.png`

## Full optimization-plan outline

Briefly restate this staged plan in the Phase 1 report so later sessions have
the whole roadmap:

1. common benchmark and visual-QC tooling;
2. centroid-versus-instance bottleneck decomposition using full inference and
   oracle-mouth crops;
3. side centroid parameter ablations;
4. centered-instance crop, augmentation, loss, and hard-keypoint ablations;
5. inference-threshold and model-pair optimization;
6. fixed-clip 3D reconstruction and identity-QC promotion gate.

Only item 1 is in scope for implementation now.

## Phase 1 requirements

### 1. Audit available evaluation data and leakage

Inventory the actual labeled packages and their provenance before choosing
frames. Clearly classify every evaluation cohort as one of:

- genuinely held out from both models;
- held out from Round 8 but seen by Round 7;
- in-sample stress/regression data;
- unknown provenance.

Do not describe Round 7's merged-label evaluation as held-out: its existing
top-down-versus-bottom-up and cross-round stress subsets were largely training
data. Do not silently use training frames as a locked test set.

Check whether corrected labels from the existing Round 8 expansion-240 package
are available. If they are not available, do not fabricate a larger locked
test. Instead:

- use the valid existing event-level pilot holdout as the true held-out cohort;
- keep a clearly labeled multi-experiment in-sample stress cohort for
  K_1_1/K_1_2/K_1_3 regression checks;
- write a manifest of additional frames requiring manual review/annotation for
  a future locked test.

All splits must be event/window aware. Nearby frames from one behavioral event
must not cross train/development/test boundaries.

### 2. Create a deterministic benchmark manifest

Create a versioned CSV or Parquet manifest containing at least:

- stable frame key;
- experiment;
- view/camera;
- video/source path and frame index;
- source SLP and provenance;
- cohort and split role;
- event/window identifier;
- ordinary versus close-contact/crossing versus partial/occluded category;
- node-visibility status;
- whether all three mouths are finite;
- whether the frame is eligible for strict centroid evaluation;
- whether the frame needs manual review.

Selection must be deterministic and reproducible. Record the seed or stable
hash rule. Preserve source packages unchanged.

### 3. Make Round 7 and Round 8 comparable

Inspect and reuse:

- `scripts/sleap/eval_topdown_vs_bottomup.py`
- `scripts/sleap/compare_rounds_topdown.py`
- `scripts/sleap/eval_compare_rounds.py`

Extend or refactor the smallest appropriate evaluator so Round 7 and Round 8
best checkpoints run on exactly the same manifest frames. Avoid duplicating
metric logic. Round 8 run discovery must support the
`round8_260723_strictmouth` naming.

Support at least:

- top and side evaluation;
- overall and per-experiment results;
- ordinary, close-contact, and partial subsets;
- centroid-only detection metrics;
- centered-instance evaluation with oracle mouth/crop when possible;
- complete top-down inference;
- saved predictions so results can be inspected without rerunning inference.

### 4. Correct partial-label-aware matching

Investigate the existing Hungarian association and implement an explicit,
tested matching policy for partial instances. Requirements:

- never use a missing node as a numeric coordinate;
- preserve global one-to-one matching;
- prefer the semantic mouth anchor when it exists on both sides;
- use only jointly visible nodes for partial cases;
- record the match basis, such as `mouth`, `tail`, `both`, or `unmatchable`;
- report partial cases separately from fully labeled cases;
- do not allow a matching artifact to masquerade as a keypoint error.

Add regression tests based on the structures observed at:

- side1 frame 97219: retain the genuine tail error;
- side2 frame 97340: eliminate the artificial approximately 109 px error;
- side2 frame 97630: eliminate the artificial approximately 51 px error.

If exact source frames make tests too heavy, construct small synthetic
instances reproducing their visibility/matching geometry.

### 5. Report metrics that match the scientific objective

Write machine-readable JSON/CSV plus a concise Markdown report. Report counts
and denominators, not only rates.

Centroid/detection metrics:

- precision, recall, F1;
- missed and extra fish;
- all-three-fish detection coverage;
- close-contact merge rate;
- mouth error median/p90/p95/p99.

Centered-instance metrics:

- mouth and tail error separately;
- median/p90/p95/p99;
- PCK@5 and PCK@10;
- mouth-tail swap rate;
- shared/duplicated-tail rate where measurable;
- metrics split into fully labeled and partial instances.

Full top-down metrics:

- detection/miss/extra/merge;
- mouth/tail error;
- swap/shared-tail failures;
- results by experiment, view, and difficulty.

Do not use OKS or validation loss as the sole ranking criterion. Prioritize
miss/merge behavior and high-percentile mouth/tail errors.

Use bootstrap confidence intervals where the sample size supports them. If the
cohort is too small, state that explicitly.

### 6. Build a manual visual-QC GUI

The user will perform visual decisions manually. Build a local GUI for all
cases requiring visual review. Reuse patterns or components from existing
repository GUIs where practical, especially:

- `scripts/sleap/multiview_annotation_gui.py`
- `scripts/analysis/contact_episode_annotator.py`

The GUI must:

- load the benchmark manifest plus cached Round 7/Round 8 predictions;
- show the raw frame with annotation and prediction overlays;
- support Round 7, Round 8, and side-by-side or toggle comparison;
- distinguish mouth, tail, skeleton, GT, and each model with a clear legend;
- show experiment, view, frame, cohort, difficulty, visibility, and numerical
  errors;
- filter/sort to misses, extras, merges, swaps, shared tails, partial labels,
  largest mouth errors, and largest tail errors;
- navigate next/previous and jump to a frame key;
- support zoom and pan;
- include keyboard shortcuts;
- allow review labels:
  `model_error`, `annotation_error`, `matching_error`, `ambiguous`,
  `exclude`, and `looks_correct`;
- allow free-text notes;
- autosave and resume;
- write review decisions to a separate CSV/JSON file without modifying source
  SLP packages;
- export the current overlaid frame as PNG;
- make unreviewed/reviewed counts visible.

Include a short GUI runbook. If practical, provide a screenshot generated from
an existing case such as frame 97219 and visually inspect it before reporting
completion.

### 7. Tests and validation

Add focused tests for:

- deterministic manifest selection;
- event-level split leakage prevention;
- partial-label matching;
- metric denominators and subset splits;
- GUI review-file serialization/resume logic without requiring display access.

Run focused tests first, then the repository's exact CI command in the Python
3.11 environment. The current CI has known pre-existing issues; distinguish
new failures from pre-existing failures and do not claim full success from
targeted tests alone.

Do not train models, overwrite checkpoints, alter source annotations, or launch
full-length inference in Phase 1. Short benchmark inference is allowed when
needed and should save reusable outputs with provenance.

## Expected deliverables

Place code in the appropriate `scripts/sleap/` modules and reports under a new
dated report directory. At completion provide:

1. the full staged optimization-plan outline;
2. data/provenance audit;
3. deterministic benchmark manifest;
4. corrected common Round 7 versus Round 8 metrics;
5. reusable prediction artifacts or exact commands to generate them;
6. visual-QC GUI and runbook;
7. manual-review CSV/JSON location;
8. focused test results and exact CI status;
9. remaining data limitations and the precise decision needed from the user,
   if additional annotation is required.

Work autonomously through safe, read-only inspection and local implementation.
Ask for user input only when a missing manual scientific decision would
materially change the benchmark. Do not proceed to Phase 2.
