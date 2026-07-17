# Female-sex ground-truth annotator

`female_sex_gt_annotator.py` records conservative, frame-level human evidence
that identifies the female (`F`) from `side1_cam`, `side2_cam`, or both. The
human click is the ground truth. SLEAP detections and optional geometry IDs are
saved only as provenance: they never preselect a fish, propagate a label to a
neighboring frame, or force the two cameras to agree.

Install the repository GUI dependencies once with `uv sync --extra gui`.

## K_1_2 first launch: build and annotate

From the repository root in PowerShell, replace `YOUR_NAME` with the annotator's
name:

```powershell
.venv\Scripts\python.exe -m scripts.analysis.female_sex_gt_annotator `
  20251225_K_1_2_FCMC `
  --paths-config configs\paths.local.toml `
  --video-dir "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC" `
  --slp-dir "Z:\StephensU\medaka\sleap_output\20251225_K_1_2_FCMC\round7_260704" `
  --geometry-csv "Z:\StephensU\medaka\reconstruction_3d\20251225_K_1_2_FCMC\round7_260704_refractive\round7_260704_refractive_3d.csv" `
  --reconstruction-provenance-json "Z:\StephensU\medaka\reconstruction_3d\20251225_K_1_2_FCMC\round7_260704_refractive\round7_260704_refractive_reconstruction_provenance.json" `
  --calibration-model-dir calibration_results\fixed_refractive_192x192x186 `
  --calibration-backend refractive `
  --build-candidates `
  --candidate-target-count 300 `
  --candidate-min-frame-spacing 166 `
  --candidate-min-view-separation-px 40 `
  --candidate-border-margin-px 20 `
  --candidate-max-local-jump-mm 8 `
  --candidate-seed 0 `
  --annotations-csv annotations\fish_sex_gt\20251225_K_1_2_FCMC\female_sex_gt.csv `
  --annotator "YOUR_NAME" `
  --source-fps 166 `
  --context-frames 166 `
  --match-gate-px 80
```

`--build-candidates` runs the bound three-view selector before Qt opens and
shows a frame progress bar. It writes
`annotations\fish_sex_gt\20251225_K_1_2_FCMC\clean_three_view_candidates.csv`
and its summary JSON, then loads that queue immediately. On later sessions,
omit `--build-candidates`; the annotator automatically reuses that default
candidate CSV and resumes the existing GT ledger. Pass `--build-candidates`
again only when the geometry, SLP inputs, or selection thresholds changed.

The geometry file is required for the bound selector and tracklet GT mapping.
It remains optional for `--manual-only` or an external candidate queue; in that
case clicks and frame-local SLEAP evidence remain usable without geometry.
Geometry is reprojected
through the public `Calibration.project_world` API. The resolved backend and
model-file provenance are written to every row that uses geometry.

The audit path defaults to `female_sex_gt.audit.jsonl` beside the canonical CSV.
With no explicit `--video-dir`, `--slp-dir`, or calibration directory, the tool
uses `configs/paths.local.toml` where the directory layout is unambiguous. A
round-tagged SLP directory should normally be given explicitly as above.

## Candidate queue integration

The built-in selector is the production default. It verifies that the geometry
and active calibration use the same refractive model, constructs conservative
finite/non-interpolated/no-jump geometry segments, and requires clean one-to-one
mouth-tail matches in all three source cameras. The rendered 2x2 video is useful
for external QC but is not used as selection data.

Existing or additional candidate manifests can still be loaded by repeating:

```powershell
  --candidate-csv runs\candidate_method_a.csv `
  --candidate-csv runs\candidate_method_b.csv
```

Each CSV needs one source-frame column named `frame`, `center_frame`, `center`,
or `jump_frame`. Standard optional columns are `experiment`, `candidate_id`,
`source`, `priority`, `score`, `reason`, and `view_hint`. Extra columns are
retained in `candidate_metadata_json`. Rows for another experiment, fractional
or negative frames, and frames beyond the synchronized videos are rejected.

The display queue is deduplicated by experiment and frame while retaining every
contributing candidate ID, source, input row, standard field, and extra field.
Candidate files are opened read-only and never rewritten. Starting later with
another candidate file changes only the queue; the same authoritative annotation
CSV is resumed and upserted.

To intentionally open without any queue, pass `--manual-only`; exact-frame entry
and `J` then create `manual_jump` provenance records.

## Annotation workflow

Both large panels always display the same source frame. Click the female in one
or both panels, then choose one explicit decision:

- `confirmed`: positive `F` ground truth; requires at least one side click.
- `ambiguous`: no positive sex label.
- `not_visible`: no positive sex label.
- `bad_frame`: unusable source image; no positive sex label.

The actual click is always stored in original video pixels. If a raw mouth-tail
pose is within the pixel gate, the nearest pose segment/endpoints supply a
frame-local SLEAP link, node coordinates, and detection key. Otherwise the
selection is a valid `free_click`. Raw SLEAP instance indices are never treated
as identities across frames.

When geometry is enabled, finite 3D slots are projected to each camera and
matched one-to-one to visible-node SLEAP centroids. The two selected views keep
their geometry IDs independently. Different IDs produce a prominent warning and
are saved as `geometry_id_consistency=disagree`; neither view is rewritten.
Geometry labels are off by default. The raw-frame toggle hides all overlays and
selection marks without changing the pending selection.

Selector candidates carry an explicit `geometry_tracklet_map_json`. After a
human clicks the female, the selected geometry ID is mapped to the provisional
refractive segment and saved directly as `side1_tracklet_id`,
`side2_tracklet_id`, and `consensus_tracklet_id`. A disagreement is preserved;
the application never extends the female label to another frame or across a
segment boundary. These explicit fields are the GT join keys for learning
reliable tracklet connections.

Decisions save immediately. Edits upsert the unique
`(experiment, frame, annotator)` row through a temporary file and atomic replace.
Navigation and exit save dirty edits to an already-reviewed row. Reopening the
same ledger restores prior decisions and selections. The session undo action
restores the previous canonical row and appends its own audit record.

## Navigation and shortcuts

Each new source frame automatically fits the three geometry-associated fish in
each side view. Extra SLEAP/reflection detections do not enlarge that region.
`Fish` restores this crop and `Full` fits the complete source frame. Independent
scroll areas support panning, 100%, and zoom buttons. Ctrl+wheel zoom keeps the
source pixel under the mouse pointer fixed; a manual zoom remains in place until
navigation changes the source frame.

| Shortcut | Action |
|---|---|
| Left / Right | Previous / next source frame |
| Shift+Left / Shift+Right | Step 10 frames |
| Ctrl+Left / Ctrl+Right | Step 100 frames |
| P / N | Previous / next filtered candidate |
| Space | Play / pause within the context range |
| J | Focus exact-frame entry |
| H | Toggle raw frame / all overlays |
| U | Clear the active side-view selection |
| 1 / 2 / 3 / 4 | Save confirmed / ambiguous / not-visible / bad-frame |
| Ctrl+Z | Undo the last annotation edit in this session |
| Ctrl+S | Save the current reviewed row now |

The candidate filter supports all, unreviewed, confirmed, ambiguous,
not-visible, and bad-frame queue entries. The saved-row dropdown can jump back
to any existing annotation, including manual frames outside the candidate queue.

## Canonical schema and audit

Schema version `1.1` adds explicit provisional-tracklet GT fields. Relevant new
columns are:

```text
side1_tracklet_id, side2_tracklet_id,
tracklet_id_consistency, consensus_tracklet_id,
candidate_geometry_tracklet_map_json, candidate_geometry_source_mode
```

All version 1.0 columns remain in their documented order around these additions.
Version 1.0 ledgers are upgraded in memory when opened; existing rows and
annotation IDs are retained, and the expanded 1.1 schema is written on the next
save.

`annotation_id` is deterministic for the canonical key. `sex_label` is `F`
only for a confirmed row. `candidate_frame` is the queue/manual origin and may
differ from the exact annotated `frame`. Paths are normalized absolute paths.
Rows are deterministically sorted by experiment, numeric frame, and annotator.

The existing ledger is fully parsed and schema-validated before any write. An
incompatible schema/version, duplicate canonical key, malformed provenance
JSON, or invalid row stops the launch/save without truncating the file.

The append-only JSONL audit stores `add`, `edit`, and `undo` actions with UTC
timestamp, software/schema version, full old row, full new row, and candidate
provenance. The audit does not replace the canonical CSV; it preserves edit
history.

## Input safety checks

The tool blocks launch or saving when required source files cannot be opened,
the two video frame counts disagree, an SLP source frame exceeds the video, the
geometry/candidate data exceed the video, or a requested geometry backend cannot
project. It refuses to use a source/candidate path as either output. Videos,
SLPs, geometry, calibration, and candidate manifests are never modified.

## Synthetic smoke test and focused tests

The smoke test opens an offscreen Qt window, renders synchronized in-memory side
frames, navigates candidates, makes a SLEAP-linked click, writes one canonical
row plus audit, and exits:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe -m scripts.analysis.female_sex_gt_annotator synthetic_experiment --annotator smoke --smoke-test
```

Run the focused suite with:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.venv\Scripts\python.exe -m pytest -q tests\test_female_sex_gt_annotator.py
```
