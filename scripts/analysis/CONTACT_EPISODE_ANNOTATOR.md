# Contact episode detector and annotator

`scripts.analysis.contact_episode_annotator` creates human ground truth for the
question: *does a current 3D tracking method preserve physical fish identity
through close contact or a crossing?*

The annotation unit is one episode, not one detector frame. Select the physical
female and one physical male at two clean temporal anchors around the detector
cue. Raw video clicks are authoritative. SLEAP detections and current geometry
IDs are saved only as frame-local provenance, so the same clicks can later be
reassociated with `jump6`, `jump8`, or a future tracking method.

## Scientific and detector contract

For the three stable ID slots, the core calculates the pairs in this order:

| Pair | Plot color |
|---|---|
| `ID_1-ID_2` | orange (`#f28e2b`) |
| `ID_1-ID_3` | purple (`#9467bd`) |
| `ID_2-ID_3` | cyan (`#17becf`) |

`minimum_IFD` is the minimum of the three distances. It is invariant to a
permutation of current labels and therefore remains a useful contact signal
even when the colored pair traces exchange meaning during an ID swap.

Only finite, non-interpolated XYZ rows can contribute to direct 3D IFD and
contact detection. Filled/interpolated distances remain available as dotted,
secondary plot data. A required missing row creates a gap. While a detector
interval is active, a frame with no finite direct pair closes the raw interval;
it is never treated as evidence that contact continued. The detector then
optionally merges raw intervals separated by at most `--merge-gap-frames`, and
records `merged_raw_interval_count` and the missing-frame burden.

Defaults are:

- enter when minimum direct 3D IFD is `< 12 mm`;
- exit when it is `> 16 mm`;
- merge gaps of at most 5 source frames;
- add 166 source frames of review context on both sides.

The GUI exposes the enter and exit thresholds. Updating them regenerates only
the candidate CSV and detector summary; tracking CSVs, videos, and SLPs remain
read-only.

## Generic CLI

Named tracking inputs use repeatable `NAME=PATH` arguments. The first method is
the detector source unless `--detector-method` selects another one.

```powershell
.venv\Scripts\python.exe -m scripts.analysis.contact_episode_annotator `
  20251225_K_1_2_FCMC `
  --paths-config configs\paths.local.toml `
  --tracking-csv jump6=path\to\jump6.csv `
  --tracking-csv jump8=path\to\jump8.csv `
  --detector-method jump8 `
  --sleap-subdir round7_260704 `
  --anchor-mode mouth `
  --annotator YOUR_NAME
```

Normal video, SLEAP, and calibration roots are resolved from
`configs/paths.local.toml`. Explicit `--video-dir`, `--slp-dir`, and
`--calibration-model-dir` override them. Reusable Python code contains no
machine-specific `Z:` or UNC paths.

Use `--preflight` to load and validate every real source, build/reuse compact
IFD caches, run the detector, and write the candidate artifacts without opening
Qt:

```powershell
... --annotator YOUR_NAME --preflight
```

Use `--anchor-mode visible-node-centroid` for a centroid experiment. `mouth` is
the default and production choice for the strict-mouth K_1_2 inputs.

## GUI workflow

The central vertical splitter contains:

1. synchronized `top_cam`, `side1_cam`, and `side2_cam` raw-video panels;
2. an aligned 3D IFD plot plus top, side1, and side2 2D plots.

The raw frame is the default. SLEAP poses, current method IDs, raw click
markers, and short 3D trajectory tails are independent toggles. Full-frame and
fish-focused modes retain original source-pixel click coordinates. Playback,
preview mouse-wheel stepping (Shift+wheel for 10 frames), exact frame entry,
and plot clicks/drags move every view to the same
source frame. A background raw-frame cache follows the current cursor and
reserves capacity on both sides, making forward/backward playback and mouse
scrubbing much smoother after the scene warms. The status beside the overlay
controls shows cached frames, RAM, hits, misses, and warming state. The default
total budget is 768 MB across all views; tune it with `--scene-cache-mb` and
`--scene-cache-radius`.

The plots share one x-axis. Use Matplotlib's toolbar for synchronized zoom and
pan. Each plot can be collapsed. Candidate ranges are faint red; the active
episode is stronger. Missing data remains a gap. The three 2D panels default to
per-current-ID XY displacement from the previous source frame (`px/frame`),
which is label-sensitive and exposes instance/identity jumps. Use the **2D
metric** selector to switch all three panels to pairwise 2D IFD (`px`) when
separation is more useful. Both metrics are calculated from the active method
and raw SLEAP only for the current bounded review context, with one-to-one
projected-ID/instance matching, and are retained together in a bounded window
cache. Non-consecutive, unmatched, or FP-excluded observations remain gaps in
the displacement traces.

For each episode, the fastest workflow is:

1. Open an episode. The GUI automatically suggests **Anchor A** and **Anchor B**:
   the nearest frames on opposite sides of the detector interval where all
   three fish are directly observed and separated beyond the exit threshold.
   These are navigation suggestions, not claims about the ground-truth contact
   point. Move either anchor if the suggested frame is visually ambiguous.
2. At Anchor A, click the female. The GUI moves to Anchor B and asks for the
   same female. This two-click female evidence is required so a
   female-involved swap can be scored.
3. The GUI then returns to Anchor A and asks for one chosen physical male.
   After that click it moves to Anchor B and asks for the same male.
4. Confirm that the summary shows F and M complete at both anchors. Choose
   contact validity and tracker assessment, then save. Add a brief note only
   when ambiguity changes the physical Anchor A/B identity interpretation.

The default left-click workflow follows the active Anchor/Target selectors.
Right-clicking a fish exposes direct actions for any Anchor A/B identity. The
automatic cue can be disabled if evidence needs to be collected from several
views before moving to the other anchor.

A click close to a pose is stored as `sleap_linked`; a free click is stored as
`free_click` and remains valid human evidence. No click is copied to another
frame or view. The automatic mapping suggestion is displayed and saved beside,
but never replaces, the human tracker assessment.

If a SLEAP pose is a false-positive prediction, right-click it and choose
**Mark SLEAP prediction ... as false positive**. The exclusion is local to the
exact frame and view. That instance is drawn in red and is disabled for current
ID assignment and both 2D metrics. Any identity click previously linked to it is
kept at its original source pixel as `free_click`; it is not deleted. Use the
same menu action to restore an instance marked by mistake.

If a real fish is visible but SLEAP supplied no instance, right-click the raw
image at the fish and choose **Mark missing SLEAP fish (FN) here**. FN evidence
is drawn as a distinct magenta marker and is local to the exact frame and view.
Choosing **Remove missing SLEAP fish (FN) marker** near it toggles it off. An FN
click is candidate-mining evidence only; it is not an identity assignment and
is never associated automatically with a fish ID.

The GUI does not force the annotator to classify other transient details such
as shared tails or model-specific pose defects. Those can change with a new
SLEAP model or tracking strategy. Raw clicks, FP/FN evidence, detection keys,
source-instance indices, and per-method geometry links are captured explicitly
and can be recomputed. Older fine-grained issue flags remain schema-compatible
and are preserved silently when editing a row.

The episode queue supports previous/next, dropdown selection, unreviewed and
decision filters, progress counts, and ranking by minimum IFD, duration,
missing burden, or frame. Enter `frame` or `start-end` to add a detector miss as
a session manual episode. The GUI resumes at the most recently updated row for
the annotator. Existing rows can be edited and the last edit in the current
session can be undone.

## Artifacts and persistence

Default outputs are below
`annotations/contact_episode_gt/<experiment>/`:

- `contact_episode_gt.csv`: canonical, versioned annotation table;
- `contact_episode_gt.audit.jsonl`: append-only add/edit/undo audit;
- `contact_candidates_<method>.csv`: inspectable detector episodes;
- `contact_candidates_<method>_summary.json`: detector parameters, policy,
  schema, and source provenance;
- `ifd_cache/<method>.ifd.npz`: compact XYZ/direct/secondary IFD cache validated
  against source path, byte size, and nanosecond modification time.

Canonical keys are `(experiment, episode_id, annotator)`. `annotation_id` is a
deterministic UUID derived from that key. Canonical writes use a same-directory
temporary file and atomic replacement; duplicate keys are rejected. Audit rows
contain complete old and new canonical rows.

The annotation schema version is `1.2`. Version `1.0` and `1.1` ledgers are
upgraded in memory on read. For version `1.0`, `pre_frame`/`post_frame` become
neutral `anchor_a_frame`/`anchor_b_frame` and old click-phase keys are migrated;
for version `1.1`, `false_negative_detections_json` is initialized to an empty
list. The source file is not rewritten until an annotation is saved.
`clicks_json` has the explicit nesting:

```text
phase (anchor_a/anchor_b)
  -> target (female/anchor_male)
    -> view (top_cam/side1_cam/side2_cam)
      -> raw click and optional frame-local provenance
```

Each populated click contains `selection_status`, exact `view` and `frame`,
`click_x/y`, optional SLEAP `detection_key`, `instance_index` and anchor, node
names, anchor mode, and `geometry_by_method`. Schema validation round-trips this
JSON before every save. `false_positive_detections_json` separately records the
frame/view/instance exclusions so they can be audited and recomputed with a new
SLEAP model. `false_negative_detections_json` records independent
`view`, `frame`, `click_x`, `click_y`, and `marked_at_utc` values and deliberately
contains no instance or identity association.

`evidence_quality` follows the existing annotator distinction:

- `usable`: the four physical identity anchors are usable for tracker scoring;
- `human_only`: physical identity is usable but automatic associations should
  be ignored;
- `unusable`: the same physical female/male cannot be identified across the
  episode;
- `unreviewed`: no final evidence classification yet.

## K_1_2 launcher

The machine-specific production paths are confined to:

`report/20260719/contact_episode_gt/run_k12_contact_episode_annotator.ps1`

Run its non-GUI validation first:

```powershell
report\20260719\contact_episode_gt\run_k12_contact_episode_annotator.ps1 `
  -Annotator YOUR_NAME -Preflight
```

Then omit `-Preflight` to open the interactive GUI.

## Validation

Focused tests are:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.venv\Scripts\python.exe -m pytest -q `
  tests\test_contact_episode_gt.py `
  tests\test_contact_episode_annotator.py `
  --basetemp .pytest_tmp_contact
```

The Qt smoke test uses synthetic frame-locked videos, SLEAP poses, and tracks.
It opens offscreen, changes frame through the plot-navigation entry point, uses
the sequential Anchor A/B identity path for F/M, marks one SLEAP instance as a
false positive, saves atomically, and reopens the ledger for verification. A
separate fixture verifies that the raw scene cache prefetches frames on both
sides and serves exact cached frames.
