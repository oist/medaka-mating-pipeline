# Prompt: build a female-sex ground-truth annotation GUI

You are working in the `medaka-mating-pipeline` repository. Read the repository
`AGENTS.md` before making changes. Implement and test a focused GUI for recording
frame-level ground truth that identifies the female fish (`F`) from the two side
cameras. The human can distinguish the female from side appearance in
`side1_cam` and `side2_cam`.

This is an annotation tool, not a candidate-selection algorithm and not a
tracking-repair tool. Candidate frames may be produced later by any method. The
GUI must accept those candidates through a small, stable input contract and
write all human decisions into one authoritative, resumable annotation ledger.

## Goal

For an experiment such as `20251225_K_1_2_FCMC`, let an annotator:

1. open a queue of candidate source-frame numbers;
2. inspect synchronized full-resolution frames from `side1_cam` and `side2_cam`;
3. scrub to any nearby source frame, or jump to an arbitrary frame that was not
   in the candidate queue;
4. click the fish that is female in either or both side views;
5. save a conservative GT decision with enough raw evidence and provenance for
   later appearance training, identity linking, and tracking evaluation; and
6. close and resume without losing work or duplicating annotations.

The canonical annotation is the human-selected side-view observation. A current
geometry/tracking ID may be associated with the click, but it is only provenance:
never use the current tracker to decide which fish is female, never silently
propagate `F` to another frame, and never force the two cameras to agree. A
cross-view disagreement is valuable swap evidence and must remain representable.

## Implementation shape

Create:

- `scripts/analysis/female_sex_gt_annotator.py` — PyQt5 GUI and CLI entry point;
- a small reusable persistence/candidate-normalization module under `scripts/`
  if keeping it separate materially improves testing;
- `scripts/analysis/FEMALE_SEX_GT_ANNOTATOR.md` — user documentation and the
  exact K_1_2 launch command; and
- focused pytest coverage under `tests/`.

Reuse repository code where practical instead of creating another incompatible
viewer:

- `scripts.analysis.geometry_swap_review.IndexedSlpPredictions` for indexed,
  on-demand `.slp` reads;
- the synchronized video and navigation patterns in
  `scripts/analysis/geometry_swap_review.py` and
  `scripts/analysis/crossing_review_gui.py`;
- `scripts.reconstruction_3d.calibration.Calibration` through its public API
  only; and
- the atomic CSV write plus append-only JSONL audit pattern in
  `scripts/analysis/geometry_swap_review.py`.

Do not modify source videos, `.slp` files, reconstruction CSVs, candidate files,
or calibration assets. Do not hardcode `Z:` or machine-specific roots in reusable
code. Resolve defaults through `configs/paths.local.toml` and accept explicit CLI
overrides.

## CLI contract

Support at least:

```text
python -m scripts.analysis.female_sex_gt_annotator EXPERIMENT
  --paths-config PATH
  --candidate-csv PATH                 # repeatable; optional if using manual jump
  --video-dir PATH
  --slp-dir PATH
  --geometry-csv PATH                  # optional but recommended
  --calibration-model-dir PATH         # needed only for geometry reprojection
  --calibration-backend auto|refractive|polynomial
  --annotations-csv PATH               # authoritative unified ledger
  --audit-jsonl PATH                    # defaults beside annotations CSV
  --annotator NAME
  --source-fps 166
  --context-frames 166
  --match-gate-px 80
  --smoke-test
```

Multiple `--candidate-csv` inputs must merge into one queue. The GUI must also
launch with no candidates and allow direct frame entry.

For K_1_2, document an example using:

- experiment `20251225_K_1_2_FCMC`;
- round `round7_260704` side-view `.slp` files;
- `calibration_results/fixed_refractive_192x192x186` with the refractive backend;
- the round7 refractive geometry CSV when available; and
- a small experiment-level GT path such as
  `annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv`.

The example may show resolved local paths, but reusable code must use the local
path loader and CLI arguments.

## Candidate input contract

Candidate selection is deliberately a separate step. Accept CSV inputs with one
row per proposed frame.

- Required: one frame column named `frame`, `center_frame`, `center`, or
  `jump_frame`.
- Optional standard columns: `experiment`, `candidate_id`, `source`, `priority`,
  `score`, `reason`, and `view_hint`.
- Preserve every unrecognized column in `candidate_metadata_json` in the saved
  annotation record.
- If `experiment` is present, reject rows for a different experiment rather than
  mixing datasets silently.
- Validate integral, non-negative, in-range source frames.
- Deduplicate the display queue by `(experiment, frame)`, while retaining all
  contributing candidate IDs, sources, and metadata.
- Do not rewrite or augment the input candidate files.

The queue determines what to review, not where annotations are stored. Importing
a new candidate file later must not replace, fork, or duplicate the existing GT
ledger.

## GUI behavior

Make `side1_cam` and `side2_cam` the two large primary panels. Both must always
show the same exact source-frame number. A small top-camera or 3D context panel
may be optional, but must not reduce the side panels below a useful inspection
size.

Required interaction:

- Previous/next candidate, candidate dropdown, and clear progress counts.
- Exact frame spinbox/text entry so any valid source frame can be annotated.
- A context slider around the candidate plus play/pause.
- Step by 1, 10, or 100 source frames; always show frame and seconds.
- Independent zoom and pan for each side panel, with fit/reset controls.
- Raw-frame toggle that hides every overlay without changing selections.
- On-demand indexed SLEAP overlay. Draw all raw mouth-tail poses neutrally so
  current tracking labels do not bias the sex judgment.
- Click a fish in either panel to select it as `F` for that view. Hit-test the
  nearest raw SLEAP pose using distance to its visible mouth-tail segment and
  endpoints, subject to a configurable pixel gate. Always store the actual click
  even when no SLEAP pose passes the gate.
- Highlight the selected observation clearly. Clicking again replaces that
  view's selection; provide clear/undo.
- Allow one-view confirmation when the female is identifiable in only one side
  camera. Do not require both views.
- If both views are selected, store both decisions independently.
- Provide explicit decisions: `confirmed`, `ambiguous`, `not_visible`, and
  `bad_frame`. Only `confirmed` rows are positive female GT.
- Notes field and required annotator name.
- Autosave every decision and save before candidate/frame navigation and exit.
- Resume existing work, restore selections, and support editing a prior decision.
- Filter or jump to unreviewed, confirmed, ambiguous, and not-visible rows.

Suggested shortcuts:

```text
Left/Right        previous/next source frame
Shift+Left/Right  step 10 frames
Ctrl+Left/Right   step 100 frames
P / N             previous/next candidate
Space             play/pause
J                 jump to exact frame
H                 hide/show overlays
U                 clear the active view selection
1                 save confirmed
2                 save ambiguous
3                 save not_visible
4                 save bad_frame
Ctrl+Z            undo the last annotation edit in this session
Ctrl+S            save now
```

## Relating clicks to detections and geometry

Raw SLEAP instance order is frame-local and must never be treated as a persistent
fish identity. For each selected view, save:

- a stable frame-local detection key such as
  `EXPERIMENT|VIEW|FRAME|INSTANCE_INDEX`;
- the frame-local SLEAP instance index;
- click `x,y` in original source pixels;
- visible mouth and tail coordinates in original source pixels;
- the SLEAP file path and node names; and
- whether the selection was linked to SLEAP or is a free click.

When `--geometry-csv` and calibration are supplied, reproject all finite geometry
slots with the selected public calibration backend and match them one-to-one to
raw SLEAP instances using the production visible-node-centroid convention and
the configured gate. Record the matched geometry ID separately for each view.

Derive, but do not require, these fields:

- `side1_geometry_id` and `side2_geometry_id`;
- `geometry_id_consistency = agree|disagree|one_view|unmapped`; and
- `consensus_geometry_id`, populated only when both available mappings agree or
  only one view supplies a mapping.

If side1 and side2 map to different geometry IDs, show a prominent warning but
allow the confirmed human annotation to be saved. Never rewrite either view to
make the tracker appear consistent. This disagreement is useful GT for finding
an ID swap.

If a geometry CSV is absent, the per-view SLEAP detection/click annotations must
remain fully usable.

## Authoritative annotation schema

Use one canonical CSV row per `(experiment, frame, annotator)`. Upsert that key
atomically when an annotator edits a decision; do not append duplicate current
rows. Keep edits in the audit JSONL.

The CSV must have a fixed, documented schema including at least:

```text
schema_version
annotation_id
experiment
frame
time_seconds
annotator
gt_status
sex_label                         # F only for confirmed rows
side1_selection_status            # sleap_linked|free_click|not_selected
side1_detection_key
side1_instance_index
side1_click_x
side1_click_y
side1_mouth_x
side1_mouth_y
side1_tail_x
side1_tail_y
side1_geometry_id
side2_selection_status
side2_detection_key
side2_instance_index
side2_click_x
side2_click_y
side2_mouth_x
side2_mouth_y
side2_tail_x
side2_tail_y
side2_geometry_id
geometry_id_consistency
consensus_geometry_id
candidate_frame
candidate_ids
candidate_sources
candidate_metadata_json
video_side1_path
video_side2_path
slp_side1_path
slp_side2_path
geometry_csv_path
calibration_backend
calibration_model_files_json
created_at_utc
updated_at_utc
notes
```

Rules:

- `frame` is the exact source frame being labeled; `candidate_frame` is only the
  queue frame that led the annotator there and may differ.
- `annotation_id` is deterministic for `(experiment, frame, annotator)`.
- `sex_label` is `F` only when `gt_status=confirmed`; otherwise it is empty.
- A confirmed row requires at least one side-view click/selection, but it does
  not require a SLEAP or geometry match.
- Store paths in normalized form and record resolved calibration backend/model
  provenance when geometry mapping is used.
- Sort the canonical CSV deterministically by experiment, frame, and annotator.
- Write through a temporary file and atomic replace.
- The append-only audit JSONL records add/edit/undo actions with timestamp, old
  row, new row, candidate provenance, and software/schema version.
- Validate the existing schema before writing. Fail loudly on incompatible
  columns or schema versions; never truncate a ledger that cannot be parsed.

## Bias and safety requirements

- Appearance judgment must be possible with all tracking/geometry labels hidden.
- Neutral raw SLEAP poses may be displayed, but geometry IDs should be behind an
  explicit toggle and off by default.
- Do not preselect a female from body size, current ID, historical labels, or an
  appearance model.
- Do not propagate one GT call across neighboring frames.
- Never overwrite source data or silently discard an older annotation edit.
- If videos, SLP data, frame counts, or experiment names disagree, block saving
  and show a useful error.

## Tests

Add fast tests that do not require the network share, a GPU, or real videos:

1. normalize each accepted candidate frame alias;
2. merge multiple candidate files and retain all source metadata;
3. reject another experiment, fractional/negative frames, and out-of-range frames;
4. deterministic annotation IDs and CSV sort order;
5. atomic upsert without duplicate `(experiment, frame, annotator)` rows;
6. edit and undo audit records preserve old/new values;
7. SLEAP pose hit-testing, including overlapping instances and free-click fallback;
8. one-view, cross-view-agree, cross-view-disagree, and unmapped geometry cases;
9. confirmed decisions require at least one side-view selection;
10. resume restores a saved annotation and queue status;
11. manual frame jump saves with `candidate_source=manual_jump` or equivalent;
12. an offscreen Qt `--smoke-test` opens, renders synthetic side frames, navigates,
    saves to a temporary ledger, and exits.

Use synthetic fixtures for image arrays, SLEAP nodes, geometry IDs, and candidate
tables. Run focused tests, then the relevant existing analysis/annotation tests.

## Acceptance criteria

The work is complete when:

- a candidate CSV from any selection method can drive the queue without code
  changes;
- the annotator can choose a nearby or completely arbitrary source frame;
- a female can be confirmed from side1, side2, or both;
- the exact clicks, raw SLEAP evidence, optional per-view geometry association,
  and candidate provenance are saved;
- a tracker disagreement between views is retained rather than corrected away;
- reopening the same experiment resumes from the same unified ledger;
- adding a new candidate manifest does not fork or overwrite GT;
- all saves are atomic and all edits are auditable;
- K_1_2 has a documented refractive-backend launch command; and
- tests and the synthetic GUI smoke test pass.

Keep this implementation focused. Candidate ranking, active learning, propagation
of `F` through time, model training, and automatic tracking correction are
explicitly out of scope for this change.
