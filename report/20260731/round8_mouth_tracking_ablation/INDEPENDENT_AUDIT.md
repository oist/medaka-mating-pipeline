# Independent audit: controlled Round8 mouth-tracking ablation

## Audit status

This audit is independent of the implementation/execution work. I did not edit
the executor's code, configuration, caches, or result artifacts. At the time of
this interim record, the input and cohort audit is complete; method-result and
full-recording claims remain pending and must not be inferred from this file.

Interim verdict:

- **PASS** — exact Round8 input selection, source hashes, calibration identity,
  unchanged A0 provenance, and direct reproduction of the critical raw event.
- **PASS after correction** — the canonical six-slot cache is strict finite
  mouth and disables the legacy tail/body false-positive filter. The first
  cache is correctly rejected as diagnostic-only.
- **CONDITIONAL** — the temporal cohort split is event-disjoint, but the
  controls are length-matched temporal controls, not behavior- or risk-matched
  controls. Ground-truth denominators are very small.
- **PASS** — training-only A1–A5 selection, the corrected v3 threshold freeze,
  and the prelaunch no-output gate. A1–A3 are frozen as the full-recording
  finalists; A4/A5 are frozen as bounded nonfinalists.
- **NOT YET AUDITABLE** — post-freeze validation/held-out/critical results,
  full-recording finalist outputs, and the production recommendation.

## Exact-input and provenance checks

The experiment is `20251225_K_1_2_FCMC` and the exact completed Round8 tag is
`round8_260729_snn031_m6`. I verified the merged SLP files directly:

| View | Bytes | SHA-256 |
|---|---:|---|
| top | 124,339,365 | `22727b64c3ef91267e86b4125b2ac3bcd326b84bd61d699718020b5deaf7f36f` |
| side1 | 125,380,235 | `357f2930fe2d6ae28154288a4bf1f1004bdd32c164367c058bd1454f440783a2` |
| side2 | 125,558,086 | `10bbb2135da9a8559b2644216bcd2b32851c58644fe309846c874aa24115faa6` |

The source inference manifest records `sleap-nn/0.3.1`, native Torch tracking,
batch size 12, maximum six instances, four chunks, and every-frame processing.
Direct SLP decoding found 348,442/348,591/348,593 labeled frames for
top/side1/side2, spanning frame 0 through 348,599, without duplicate labeled
frame indices.

The fixed-interface calibration is the checked-in refractive model. Its score
is mean squared point-to-water-ray distance in `mm^2`. Direct hashes are:

- manifest:
  `9bb7dda927c01c018c411d3b56da58f3e44379f4d4ce1491c5ab67d33e5bf105`;
- cam0:
  `f6756885f0b4b2c86033f1818f3681a85084ff56f40dd478ccdc66720063106c`;
- cam1:
  `393cbe50dc8d434d97e2a6ec41005abdb92e08d1691370f42a4de978c31af7f0`;
- cam2:
  `87af873ff80710f51c622c9305b20a5219b3aaede3837cdd92c7964ae0cd06f8`.

The independently verified cam2 value is the 64-character SHA-256 shown above.

The existing A0 reference was not recomputed. Its stage-3 command and
provenance use visible-node centroids, six candidates, three fish, bounds
`-20..220`, no positive shift, an 8 mm jump gate, 80 mm hard cap, 100-frame
gap, top-ID binding, and the refractive backend. Direct output hashes are:

- geometry CSV:
  `6f42483363dd3be4c1aa7ec758521a4f1fde72877be18b5e4dafc8eee14ffb03`;
- identified CSV:
  `b2100d24cec8a660cff5888ceee1ade26c9990d900f1e67508e566aaab2870b6`;
- best JSON:
  `ae4b727bb6679a825e3a9dac143a85fa04c41cfe620c4be875e3b0187a28f2b2`;
- pre-link JSON:
  `93f322114e10b2fae897f28375237c719a86964b232e3fd686ccd2561fba9068`.

## Strict-mouth cache contract

The first cache under `candidate_cache/` used `filter_fp=true`. That filter
reads tail/body span and mouth–tail overlap, so it is not eligible for A1–A5
mouth-only conclusions. The executor now labels it rejected diagnostic
provenance.

The next cache under `candidate_cache_strict_mouth_unfiltered/` correctly
removed that filter and has bit-valid coordinates. It is nevertheless
superseded: the installed `sleap_io` exposes node confidence through a
structured points array, which the first cache implementation did not read, so
all cached `mouth_score` values were missing. It is not eligible for A4/A5
confidence gating.

The final canonical cache is
`candidate_cache_strict_mouth_unfiltered_nodescores/`. It is complete, float64,
six slots, 348,600 frames, `anchor_node=mouth`, `require_anchor_node=1`, and
`filter_fp=0`. Its hashes are:

| View | Cache SHA-256 | Bound source SLP SHA-256 |
|---|---|---|
| top | `6d2f86e8822525e83f18c921bb2b75d0e2821b929d2dcced994dea29b2737d74` | `22727b64c3ef91267e86b4125b2ac3bcd326b84bd61d699718020b5deaf7f36f` |
| side1 | `59d9a73895f0b0c507e5dd1c57a8503ae8ca932cdad7e2dd0a16685b7fb042be` | `357f2930fe2d6ae28154288a4bf1f1004bdd32c164367c058bd1454f440783a2` |
| side2 | `1f2d639606e77cc4c1b3188e8991cdf02c2895556d225996a957f91af7832699` | `10bbb2135da9a8559b2644216bcd2b32851c58644fe309846c874aa24115faa6` |

I independently inspected the source SLP HDF5 frame tables. The maximum raw
instances per frame is exactly six in every view; frames with six raw instances
number 49, 481, and 426 for top, side1, and side2. Maximum finite-mouth
instances are also six. Thus pose-score slot ordering cannot discard a seventh
finite mouth in this dataset. Raw-minus-finite-mouth counts are 678, 5,331,
and 3,642, matching the reason strict-mouth cache totals are below raw totals.

I compared the final cache against both the superseded coordinate cache and
the source SLP arrays. `frame_present`, mouth and tail coordinates,
pose-level score, visibility, and source-instance index are bit-exact between
cache versions. Cached mouth coordinates and mouth-node scores are also
bit-exact to the source SLP for every retained row. Finite-mouth score coverage
is 1,024,875/1,024,875 (top), 1,029,722/1,029,722 (side1), and
1,033,042/1,033,042 (side2), with no node score attached to a nonfinite mouth.

The conservative tracklet implementation was corrected before method runs to
gate admission on `mouth_score`, not pose-level `instance_score`. Any A4/A5
artifact using either earlier cache, or code predating that correction, remains
ineligible.

## Direct critical-event reproduction

I decoded the merged Round8 SLPs at frames 34,547 and 34,548. The relevant top
instance has:

| Frame | Mouth `(x,y)` px | Tail `(x,y)` px | Visible-node centroid `(x,y)` px |
|---:|---|---|---|
| 34,547 | `(952.250183, 1016.494995)` | `(844.389709, 986.474548)` | `(898.319946, 1001.484772)` |
| 34,548 | `(952.454468, 1016.613281)` | missing | `(952.454468, 1016.613281)` |

The mouth moves only `0.236059 px`; the visible-node centroid moves
`56.208702 px`. At both frames each side view contains only two total poses:
one distant fish plus one pose for the close pair.

The A0 pre-link 3D candidates for the close pair are:

| Frame | Candidate b `(X,Y,Z)` mm | Candidate c `(X,Y,Z)` mm |
|---:|---|---|
| 34,547 | `(13.403691, 182.953662, 11.067043)` | `(20.097191, 185.839006, 10.892424)` |
| 34,548 | `(16.055115, 183.776815, 11.704868)` | `(8.949584, 186.188230, 10.811007)` |

The previous-to-current distance matrix in millimetres is:

| | b at 34,548 | c at 34,548 |
|---|---:|---:|
| b at 34,547 | 2.848587 | 5.510631 |
| c at 34,547 | 4.609890 | 11.153373 |

The straight pairing is infeasible under the 8 mm gate because
`c→c = 11.153373 mm`; both crossed links are feasible and the linked IDs
cross. This verifies the mechanism in the request but does not establish the
biologically correct male-to-male assignment.

## Cohort and freeze audit

The declared training, validation, held-out, risk, and critical intervals are
temporally disjoint. Inclusive length matching is exact, and the
33,547–35,555 critical exclusion is absent from tuning and validation. The
cohort file predates method scoring of the critical smoke window.

This is not enough to call the controls “matched” without qualification:
they were inherited from a Round7 random temporal-control protocol and then
length-truncated. They are not manually verified non-contact windows and were
not matched on Round8 close-contact difficulty. Using the first three
score-ordered finite mouths as a transparent diagnostic, the fraction of
frames with minimum pair separation below 0.65 local median body lengths was:

| Cohort | top | side1 | side2 |
|---|---:|---:|---:|
| validation event 2 | 30.8% | 18.3% | 24.1% |
| validation control 2 | 4.9% | 7.0% | 7.0% |

Difficulty relationships vary for the other pairs as well. These windows are
valid **length-matched temporal controls**, but event-versus-control
differences are not causal treatment estimates and the controls must not be
described as behavior- or risk-matched.

The female ledger contributes very sparse evidence:

| Cohort | confirmed/ledger rows in interval |
|---|---:|
| training event 1 / control 1 | 2 / 1 |
| training event 3 / control 3 | 6 / 6 |
| validation event 2 / control 2 | 2 / 1 |
| held-out event 4 / control 4 | 6 / 3 |
| required critical detail | 0 |

The ledger identifies only the female and cannot detect male-to-male swaps.
Its selected SLEAP instances and geometry IDs refer to Round7. Round8 scoring
must remap the original manual video clicks to current Round8 mouth detections
with a documented distance gate and ambiguity audit; reusing Round7 instance
indices, geometry IDs, or tracklet IDs would be invalid.

Before critical scoring, `frozen_thresholds.json` must:

1. tune parameters on training only, with a deterministic objective and
   tie-break order;
2. treat validation as one-shot confirmation, or explicitly relabel it tuning
   if it influences selection and reserve held-out as the only confirmatory
   cohort;
3. estimate clean-frame body length/motion distributions from training
   controls only if validation is claimed independent;
4. hash the cohort, code, inputs, selected settings, and outputs, with a
   creation timestamp preceding the critical run; and
5. freeze the reproducible selection rule and candidate universe for the
   data-driven risk windows, or label them a challenge/stress set rather than
   an unbiased held-out sample.

After this preregistration, the quality-review work inspected all 54 tracklet
cutoff combinations on frames 34,000–35,100 before a threshold-freeze artifact
existed. If the final choice is generated strictly by an immutable,
deterministic training-only objective, the critical values do not enter its
calculation. Nevertheless, the critical cohort is no longer blinded or
untouched; its result must be labeled descriptive stress evidence rather than
confirmatory. The held-out event/control cohort remains the appropriate
confirmatory set.

The same diagnostic found reciprocal side-view bridges through the transition.
Those bridges follow the single fused close-pair pose present in each side
view; they do not establish continuity of either biological fish and are not,
by themselves, high-confidence A5 identity constraints.

The later A4 implementation adds a `reject_two_view_count_mismatch` safety
rule. At the critical transition the top view has three finite mouths while
each side has only two, so this rule intentionally rejects the recovered
two-view close-pair groups. Its critical outcome is therefore abstention, not
recovery or proof of a biological identity. This rule was developed after the
critical interval was unblinded by the safety review. It must be disclosed as
critical-informed safety-method development; only post-freeze held-out
evidence can test whether it generalizes.

### Corrected threshold-freeze gate

I rejected two freeze artifacts before any post-freeze execution:

- v1, SHA-256
  `412e365b8c8fc0dc09061dc6c3ac235897d89f213a003f940c615c398ccb206e`,
  omitted the `cross_frame_tracking.py`, `fill_missing_3d.py`, and
  `calibration.py` implementation hashes and the complete selected A4
  group/path/block replay configuration. It is preserved under
  `quarantine_invalid_freeze_20260731_0414/`.
- v2, SHA-256
  `4d74438fad0d244c383ac63acd7f23575ab3869f752b2090cb2eafe9ec9b92ca`,
  omitted the direct confirmed-female annotation-ledger path and SHA-256 even
  though that ledger drove A2/A4 selection. It is preserved under
  `quarantine_invalid_freeze_20260731_0416/`.

The authoritative v3 `frozen_thresholds.json` passes the independent launch
gate:

- freeze SHA-256:
  `09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d`;
- frozen timestamp:
  `2026-07-31T04:18:09.058553+09:00`;
- writer SHA-256:
  `8622edf4fd6af2d47456efa1c4484a64b1293a950967486c3f705fe38d885b75`.

Before authorizing execution, I independently rehashed all ten implementation
files, 17 training quality/completion artifacts, eight strict-mouth geometry
artifacts, three candidate caches, four calibration files, the frozen body
scales, the cohort/provenance/deviation files, the annotation ledger, the
writer, and both quarantined freezes. The direct ledger hash
`ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f`
also matches the transitive A2 and A4 selection-audit bindings.

Semantic replay checks passed for the selected A2 velocity/history setting and
A3 acceleration/deferred setting, all four selected A4 configurations
(tracklet, cross-view grouping, three-path solver, block sizes, calibration,
caches, body scales, and code), and the structured A5 hint thresholds,
A3 comparator, and linker. The 12 typed inclusive training, validation,
held-out/risk, and critical-exclusion intervals are valid and mutually
disjoint. No JSON object/array was serialized as an opaque string. All bound
selection inputs predate the freeze, and the live forbidden-root scan found no
validation, held-out, post-freeze, critical-post-freeze, or full-length output
root. I therefore authorized launch only against the v3 hash above.

One nonblocking provenance distinction is explicit: the four training A1
summaries record `max_motion_gap_frames=20`, whereas the frozen A1
position-only baseline uses `null`. In the frozen implementation this field
only gates the availability of motion diagnostics/cost terms; every A1 motion
weight is zero. It therefore cannot alter A1 assignment costs or identities.
The frozen `null` value is authoritative for post-freeze and full-length A1
runs.

### Post-freeze runner launch audit

I blocked the first validation preparation, runner SHA-256
`1b31ecda4f91865fc85d870556e88eca9100ccf25805da4a6fb9602a5bdea4b1`,
before any scientific child process. Its A4 verifier compared the emitted
`frozen_body_length_px` field by indexing the nonexistent same-named freeze
field; the authoritative freeze calls that value `body_length_px`. This would
have raised a `KeyError` only after the first expensive A4 run. The two-file
invalid preparation (prepared-manifest SHA-256
`387df1afd126370b1b731fb97cf63e85c3b2253ead390f80592b55040aefdf19`)
is preserved under
`quarantine_invalid_validation_prepare_runner1b31_20260731_0437/`; it
contains no scientific directory or output.

The corrected bounded-stage runner passes the validation prelaunch review:

- runner SHA-256:
  `54ec5698f84c6c65aa2e0aff226dd208445ab1db51c35a60bed8e396f3abab16`;
- validation prepared-manifest SHA-256:
  `3fb4e79994df34c5d0d140a1ed5f24d5358970114230950d7cd9addb74eabae6`;
- validation prelaunch-authentication SHA-256:
  `abdfe584b902c3e904712231f5b4ba1b701c89b5d7c7cac8af0ed7870eb2c514`.

The prepared root contains only those two files before launch. I checked the
exact 18 planned commands for inclusive validation frames 96,612–97,802 and
151,050–152,240. A1 omits the motion-horizon argument and therefore records
the frozen `null`; A2/A3 explicitly pass 20. All controlled gates and the
complete selected A3 deferred settings are exact. A4 passes the selected
tracklet cutoffs and block sizes, binds the frozen code/caches/calibration/body
scales, and verifies the complete emitted tracklet/group/path configuration
against the freeze, including the corrected body-scale field mapping. A5
passes the exact frozen hint and deferred-linker settings.

The manifest explicitly records A0 as unchanged reference-only, A1–A3 as
production finalists, and A4/A5 as bounded exploratory nonfinalists.
Validation is report-only, with selection, tuning, and full-length launch
disabled. Each child has an exact command record, stdout/stderr hashes,
wall time, and 0.1-second process-tree RSS sampling. Prelaunch and
postcompletion authentication bind the freeze, ten implementation files,
three caches, four calibration files, frozen scales, and annotation ledger.
The atomic completion marker inventories and hashes every stage artifact.
Geometry and A4 are executed a second time against their completed blocks;
the gate requires every block to be reused and all advertised scientific
output hashes to remain identical while excluding expected summary/log
runtime drift. I authorized validation child launch only after these checks.

The first foreground orchestration shell was accidentally given a one-second
timeout and exited with code 124 at 04:33:46 JST. I independently confirmed
that it left only the first geometry directory and two zero-byte stdout/stderr
files: no block, summary, parquet, method result, process record, or newly
started Python process remained. The incident record has SHA-256
`9047506507769e875f52af369351ef424ed3957514e536900c57ec89ae454558`.
The runner and prepared-manifest hashes remained unchanged. I authorized
resumption of that same prepared stage through a detached hidden launcher,
with the incident and outer command/PID/stdout/stderr provenance retained in
the final stage inventory. This interruption has no scientific or selection
effect.

The validation runner subsequently completed 18/18 child processes and wrote
completion marker SHA-256
`7b11328f9304c9ad2a472d93125c0e108ea3e810e4a06e616b9da09128f32fa8`.
All two-window A1–A5 statuses are completed, the 14 summary hashes and 18
process-record hashes match, every child returned zero with empty stderr, the
pre/post authentication bindings are identical, and all four geometry/A4
resume checks pass. The marker reports 120.800 seconds stage wall time and
308,023,296 bytes maximum sampled process-tree RSS.

I nevertheless held held-out preparation because the in-process artifact
inventory was finalized just before the runner printed its own 194-byte
completion line. As a result, the marker records detached stdout as empty,
while its stable final SHA-256 is
`60e9d1458ce9de2d1f6837628ca5c0c6014c93c04f0a4f95d635193ebffc9b24`.
This is the sole mismatch among 181 inventoried files; the other 180 rehash
exactly and the detached stderr remains empty. The scientific marker must
remain immutable. A separately authenticated post-launch finalization sidecar
is required to bind this expected append and the stable current inventory
before the held-out stage may start.

That finalization gate now passes. `STAGE_FINALIZED.json` has SHA-256
`297249d8cc9cdff2e0a5c850884f795e238b6911b42605e440d70397bc6fb706`
and was produced by preserved finalizer SHA-256
`379e94ccfb3979f36e418cdcf304f442c4cc935640a1c72b27f95e496c50dee7`.
I independently rehashed all 182 stable current stage files and every binding
to the original marker, freeze, runner, prepared manifest, and pre/post
authentication. The original marker's other 180 entries remain exact; the
sole stdout append and empty stderr match the sidecar. The sidecar records no
scientific change, no selection/tuning, and no held-out/later-stage opening.
I therefore released held-out preparation, while retaining a separate launch
gate for its runner and prepared manifest.

The held-out prelaunch gate passes:

- runner v2 SHA-256:
  `1f737fb484d0b81fb52bcd1dc2551aea35ec1137ff3010bb8bc82d487aef4a7d`;
- finalizer v2 SHA-256:
  `c9ea5d44a57470bda06583a3f5311a9c84708cb8040b0d0233b8156c5c7a7f27`;
- prepared-manifest SHA-256:
  `d1defd0543b74318a1573b92a64e026179969310571b587a1b80b3601efddf97`;
- prelaunch-authentication SHA-256:
  `eecebccc56e682cac5265d81ac3b08e3b1e0fd1214d2d9f8d79d18d98411d42b`.

Before launch, the held-out root again contains only its two preparation
files and no scientific directory. It binds the authenticated validation
sidecar above, not merely the earlier completion marker. I checked all 45
commands for the exact inclusive held-out event/control and three registered
risk windows: 273,617–278,182; 57,363–61,928; 123,124–123,274;
136,451–136,601; and 167,108–167,258. The frozen A1–A5 settings, method
roles, implementation/cache/calibration/scales bindings, and no-later-root
gate match the validation contract.

Runner v2 intentionally excludes only the still-open detached-launcher
stdout/stderr files from its in-process marker. Finalizer v2 requires zero
mismatch in every marker-listed artifact, then authenticates the closed outer
logs and stable full-stage inventory. This removes the known validation
post-marker-append issue without weakening scientific authentication.

The independent validation scorer was also run before held-out launch. Its
current self-authenticated artifacts at this gate are metrics SHA-256
`b219bf5df25f81270b130a2e54bb968f1571affe7f0aa4c6f3c6743fa55601d9`,
scorer SHA-256
`b9eaa1cc906a07ca024667f661dd8fc1ff1d4c7e2d0c9a6c47645db82c7b8a2e`,
and assignment-ledger SHA-256
`a3a6f82058dfd179757a2f397499d24447dda76d559dd7452e25266b20aece7b`.
The artifact labels validation report-only and prohibits changing the frozen
settings or finalists. With that independent quality gate and the runner
checks above, I authorized held-out child launch.

The first held-out attempt then stopped safely at the A5-hint step for risk
window 136,451–136,601. A4 had produced an authenticated scientific
abstention (`insufficient_three_path_solution`), and its advertised selected
parquet (SHA-256 beginning `27f5e0c1`) is exactly zero rows by zero columns.
The frozen A5 builder validates its 16 required A4 column names before its
empty-data path, so it raised a missing-column `ValueError`. No critical or
full-length stage was opened.

This is an operational schema-compatibility bug, not evidence against the A4
abstention. I confirmed that the window's frozen A2 output has only
`identity_segment=0`. Therefore a separate zero-row table carrying the normal
29-column A4 selected-output schema will yield zero A5 hints and zero forced
resets; with the frozen A5 linker, its core assignment/XYZ/segment result must
then equal A3 (method-specific status labels may differ).

I did not authorize a partial in-place resume. The complete failed held-out
root must be preserved unchanged under quarantine with a self-authenticated
failure inventory. A fresh held-out root must chain the old prepared manifest,
launcher/final streams/exit, failed process record/logs, and schema-adapter
provenance, then rerun all five windows. The adapter may act only after
authenticating the untouched original A4 selected hash, its 0×0 shape, and the
scientific abstention status. The fresh completion must require exact hashes
for every overlapping scientific artifact from the interrupted attempt
(excluding expected runtime summaries/logs), zero hints/resets in the
abstention case, and A5-versus-A3 core-data equality. This technical repair
must be recorded as held-out-opened operational development with no threshold,
method-selection, or frozen implementation change.

The failed-attempt quarantine now passes that independent authentication
gate. `FAILED_ATTEMPT_FINALIZED.json` has SHA-256
`75a506ce112b8df00b49024f039ddb6ef554637620840bf5f9a78c73204a85d5`.
I independently rehashed all 403 inventory entries after excluding only the
finalization sidecar itself: the path set, byte counts, and SHA-256 values
match exactly. All seven bound failure/prelaunch artifacts match their
advertised hashes, including the prior inventory manifest
`d4ccda6fcd1995c359f97b3a1e99c8153a85c362cc35259a805130ce691ba644`;
the finalizer self-hash also matches
`6a29b77c76bc2a3a9dbdde07e3a340347cdd095883cfca25fbf6e07344a5d1f5`.
The authenticated state explicitly records outer exit 1, no scientific
completion marker, no critical opening, and no selection/tuning impact.

The fresh whole-heldout replacement has now passed my final prelaunch gate.
The immutable runner is
`0aaeac2293db434f3794b02a992ab208f4dc061162b8da017f6c85d94a21e818`,
the prepared manifest is
`72557e96b34c2582c58419bbaa1e71ae6379b89f7e8e3624a42fbc1e5c216037`,
and prelaunch authentication is
`871420dd5b55ed4d3b5b69beb91dc12035644e1c30c919199e590d0737c1f61b`.
The root contains exactly those two files and no scientific directory. It
binds the validation finalization sidecar `297249d8...`, the unchanged freeze,
the repaired adapter `eea6877...`, repair protocol `d42a5095...`, and failed
sentinel `75a506ce...`; a fresh authentication snapshot exactly equals all
prelaunch bindings. The runner itself independently replays the exact
403-entry failed-quarantine inventory before deriving its baseline.

The manifest has the exact five preregistered held-out windows and 50 commands.
Against failed runner `1f737fb4...`, all 45 pre-existing commands are
identical except that the five A5-hint inputs point at a separate adapter copy;
the only five added commands are the schema-adapter calls. Thus no scientific
setting changed. The replacement baseline has 26 completed scientific steps
and now additionally requires exact hashes for all 22 completed geometry block
parquets plus equality of all scientific status fields for their 22 status
JSONs, excluding only `wall_seconds`. The adapter smoke sentinel
`8d4e645d...` independently reproduces the failed A4 source hash `27f5e0c1...`
unchanged, writes a zero-row 29-column copy, and the frozen builder emits zero
hints and an empty forced-reset list. Critical and full-length roots remain
absent, and the manifest prohibits tuning, selection, and full-length launch.
I therefore authorized only the fresh held-out v4 child launch.

The replacement held-out run and its closed-launch finalization now pass the
independent completion audit. `STAGE_COMPLETED.json` has SHA-256
`615b03f161db28982ee1d6b5648e79e1d4bef146b281cc911b13d4654d7da143`;
`STAGE_FINALIZED.json` has SHA-256
`3b58e073c9cf6c97348e1f0c8cdbb9ca5c15d689ee222b450df1c91dc3303152`.
I independently rehashed the final sidecar's complete 516-file inventory
(excluding only the sidecar itself): its path set, byte counts, and hashes
match exactly. The six principal bindings and finalizer self-hash
`c9ea5d44...` match. All 512 completion-marker artifacts remain unchanged,
the detached launcher exited zero with empty stderr, all 50 process records
have the exact prepared commands and empty stderr, and all 40 summary hashes
match. Prelaunch and postcompletion authentication bindings are identical.

All five geometry outputs have exactly three finite hypotheses per frame.
Every direct and filled A1–A5 table has the exact inclusive frame × three-ID
grid, no duplicate keys, and no direct row marked interpolated. All 26
overlapping scientific-step comparisons match the quarantined attempt
exactly, including the 22 geometry block parquets and canonical scientific
block-status fields. Geometry and A4 resume checks pass for every window.
A4 completes in three windows and scientifically abstains in two. For both
abstentions (136,451–136,601 and 167,108–167,258), the original A4 selected
table remains 0×0, the adapter copy is 0×29, hints and forced resets are both
empty, all A4 path rows stay explicitly unassigned, and A5 equals A3 exactly
on the eight frozen core columns for all 453 rows.

This authorizes scoring the finalized held-out evidence. It does **not** yet
authorize opening the critical descriptive interval: the held-out quality
artifact must first be completed and independently checked, with no tuning or
selection response.

The held-out quality gate now passes. Its sealed record has SHA-256
`022fbca3f3d0c13d7388a83c42b3b2bad60ded00de2fe89fad428943e7f213cc`;
all six bindings independently match, including scorer `91727ce7...`,
metrics `0aed2e14...`, assignments `f6e8190b...`, and held-out finalization
`3b58e073...`. The metrics file's two code hashes also match, and the
assignment CSV has the expected 72 rows (six methods × three validation plus
nine held-out female annotations).

On held-out evidence, A1, A2, and A3 are scientifically identical in every
window: each maps 9/9 female annotations with no observed female-ID change,
has `0.999583` all-three-direct coverage, `0.000139` unavailable rows, one
jump at least 7.5 mm, and none over 8 mm. A0 also maps 9/9 with no observed
change but has 11 jumps at least 7.5 mm; that smoothness difference is not
biological identity proof. A4 maps 0/9 and leaves 88.98% unavailable. A5 maps
8/9, is correct on only 3/9 under fixed-ID consistency, has five observed
female-ID changes, and leaves 6.02% unavailable. These results retain A1–A3
as tied finalists, reject A4/A5 as full-length finalists, and make A1 the
parsimonious provisional candidate. Female-only labels still cannot detect a
male-to-male exchange.

The quality seal records no setting change, tuning, selection, critical
opening, or full-length launch. I therefore authorize preparation of only the
predeclared critical descriptive window. Its results must remain permanently
nonconfirmatory and cannot change the freeze or finalist set.

The resulting critical prelaunch manifest passes. Authorization wrapper v5
has SHA-256
`2eb4012e428bf8951609a8c345d55fc75c74cd934b955ef5400bb2c56d72c38d`,
prepared manifest SHA-256
`8900912581348c4df936767c639b429100d17c3d5736d528c108679824862d6b`,
and prelaunch-authentication SHA-256
`0cfe3a3f000d964838d6c1cd0ba075a2cc6ac5c5c086644dd02441aa4100cd4c`.
It binds unchanged v4 core `0aaeac22...`, validation `297249d8...`,
held-out finalization `3b58e073...`, quality seal `022fbca3...`, and freeze
`09579754...`; every non-runner authentication binding independently equals a
fresh core snapshot.

The manifest contains exactly ten unchanged scientific commands for inclusive
frames 34,000–35,100, all equal to the frozen v4 critical plan. It separately
registers required detail rows 34,540–34,565 and assignment-audit rows
34,547–34,555. Its role is permanently unblinded, descriptive, and
nonconfirmatory; tuning, selection, freeze/finalist changes, and full-length
launch are false. The root contains only its two manifests and no scientific
directory. I authorized only this critical descriptive child launch.

The critical descriptive execution and closed-launch finalization now pass my
independent completion audit. `STAGE_COMPLETED.json` has SHA-256
`e493fdcc449034c4f9dd22e3f124283fe8d3cd06d9bd50b0c46002e440367706`;
`STAGE_FINALIZED.json` has SHA-256
`fe432101a7580f4e540d3c4537ae8b1c774a2b5231d8cc0900a790e250d3149c`.
I independently rehashed all 96 completion-inventoried artifacts and the exact
100-file finalized inventory (excluding only the finalization sidecar).
Path sets, byte counts, and SHA-256 values match with zero discrepancy.
Prepared manifest `89009125...`, prelaunch authentication `0cfe3a3f...`,
held-out finalization `3b58e073...`, held-out quality seal `022fbca3...`,
freeze `09579754...`, wrapper `2eb4012e...`, unchanged v4 core
`0aaeac22...`, and finalizer `c9ea5d44...` all rehash exactly. The detached
launcher exited zero, its authenticated stderr is empty, and finalization
changed no scientific result.

All ten process-record names exactly equal the ten prepared-command names.
Every recorded command is byte-for-byte equal as an argument vector to its
prepared command, returned zero, and has empty authenticated stderr; all eight
advertised summary hashes match. Geometry and A4 resume replays both preserve
their complete advertised output-hash maps exactly. The three geometry blocks
are contiguous over inclusive frames 34,000–35,100, all report `completed`,
and the aggregate contains exactly 3,303 finite rows: three hypotheses at
each of 1,101 frames.

Every A1–A5 direct and filled output has the exact 3,303-row frame × three-ID
grid with no duplicate keys, and no direct row is interpolated. A1–A3 each
have 3,292 finite rows and 11 explicit missing/unassigned rows; their frozen
core identity/XYZ assignments are exactly identical. A4 remains sparse by
design (92 finite direct rows, two additional within-segment filled rows, and
3,209 explicitly unassigned rows). Its nonempty 92×29 selected table is
preserved byte-identically by the schema adapter, which adds no identity
information or scientific change. It yields 33 accepted A5 hint rows and one
mapped segment. A5 is exactly equal to A3 on the eight frozen core columns
throughout this interval.

The completed marker and final sidecar continue to record this stage as
permanently unblinded, descriptive, and nonconfirmatory, with no setting
change, selection/tuning, later-stage opening, or full-length launch. This
passes the technical gate for preparing an A1–A3-only full-length run.
Full-length execution remains blocked until the independent critical
descriptive review seals the required 34,540–34,565 detail and
34,547–34,555 assignment-audit evidence. That review cannot alter the frozen
settings or finalist set.

That critical descriptive quality gate now passes. The seal has SHA-256
`55a479e1b97a1054b7f195530d1505202d4d87935a1dd2f36b06f51e665fe83e`.
I independently rehashed all 11 bound artifacts, including scorer
`5d7e4e8f...`, combined metrics `d9a4d449...`, exact-event rows
`e69e7442...`, required event audit `c6417f73...`, critical finalization
`fe432101...`, completion `e493fdcc...`, raw trigger `dabbb35b...`, A0
geometry audit `59320ae8...`, and freeze `09579754...`; all match.

The exact-event CSV has the complete unique six-method × nine-frame ×
three-ID grid (162 rows). I matched every A1–A5 row back to its authenticated
direct output and every A0 XYZ row to the bound A0 geometry audit, then
independently recomputed finite-row and consecutive-jump counts. A1, A2, A3,
and A5 each have 27/27 finite direct rows, no source-index switch, and
identical normalized scientific rows; their maximum event step is
`0.618649 mm`. A3 and A5 have nine resolved assignment-audit rows, zero
ambiguous or constraint-conflict rows, and margins from `15.024493` to
`17.714434`. A4 preserves all 27 rows as explicitly unassigned and nonfinite.
A5 has zero hint rows inside the exact assignment interval; its equality to
A3 there is fallback behavior, not evidence that tracklet hints solved the
event.

A0's centroid geometry exchange at 34,547→34,548 is mechanically reproduced:
straight continuation is infeasible because the c→c step is `11.153373 mm`
under the 8 mm gate, whereas both crossed edges are feasible. The strict-mouth
methods preserve stable source slots through those frames. Because the close
fish are male and there is no male-specific ground truth, this is evidence of
internal mouth-trajectory continuity and avoidance of the known centroid
failure mechanism, not proof of biological identity. The seal is permanently
descriptive and nonconfirmatory and records no tuning, setting/finalist
change, or full-length launch.

The full-length prelaunch gate subsequently passes. The exact prepared
manifest is
`d9c14fcb973f0fa29877443802eeba643fd5b288245988414b13c53d9f1f4aa1`,
prelaunch authentication is
`bd1ef6119ff770121ad65c1ce87d772190c3d637c479940045ee84a5f81f8b7b`,
and runner is
`32a8452da10f0b53ec555cef1212aa4dabbda13ba7e146f7d508deca5c0fb87d`.
The root contains exactly the two preparation files and no scientific child.
All repository/cache/calibration/code/freeze bindings and validation,
held-out, and critical predecessor sidecars rehash exactly. I also rehashed
the seven unchanged A0 reference artifacts directly on the experiment share;
all byte counts and frozen hashes match.

The eight commands cover exactly inclusive frames 0–348,599: one fresh and
one resume run of shared strict-mouth mouth-only geometry, followed by fresh
and repeat relinks for only A1, A2, and A3. Their arguments exactly implement
the frozen settings. A4 and A5 are forbidden; A0 is reference-only. Geometry
requires block-contract equivalence and every finalist requires exact
scientific-output hash equality on repeat. Each child records wall time,
0.1-second process-tree peak RSS, exit code, and authenticated stdout/stderr.
The execution-time gate must authenticate the now-sealed critical PASS and
record it separately. I therefore authorize this exact full-length detached
child and no other full-length method or setting.

The full-length execution and closed-launch finalization now pass my
independent completion audit. `STAGE_COMPLETED.json` has SHA-256
`7e9e826cc48965d2b0ada1569b63d7a77b3b2972a8e20649eff15bc2de00dca0`;
`STAGE_FINALIZED.json` has SHA-256
`a9a8ae25cca6ef0c94dc7996cf39680017b9bb58e20ab60473bced96284e07b0`.
I independently rehashed all 1,438 completion-inventoried artifacts
(885,752,260 bytes) and the exact 1,442-file pre-sidecar finalized inventory
(886,952,876 bytes). Both path sets, byte counts, and hashes match with zero
missing, extra, or changed file. Freeze `09579754...`, prepared manifest
`d9c14fcb...`, prelaunch authentication `bd1ef611...`, execution
authorization `13073080...`, runner `32a8452d...`, finalizer
`1a223d5c...`, completion marker, and all local path/hash bindings rehash
exactly. The detached launcher exited zero, its authenticated stderr is empty,
and finalization records no scientific change, tuning, setting change, or
later-stage opening.

The exact process-record set equals the eight prepared commands. All eight
argument vectors match the prepared and completed manifests byte-for-byte,
returned zero, report `completed`, have empty authenticated stderr, and bind
their current stdout/stderr hashes. Fresh/repeat scientific hashes are exact
for A1 (two files), A2 (two files), and A3 (four files). The shared-geometry
fresh and resume scientific contracts are identical, including all block and
aggregate hashes; resume reused 698/698 blocks. The stage took 5,314.741 s.
The largest sampled process-tree RSS was 3,418,509,312 bytes (A2 repeat).
A1 fresh/repeat took 618.120/614.291 s, A2 691.229/691.152 s, and A3
773.145/773.915 s. The geometry fresh pass took 1,132.583 s and its exact
resume 9.512 s.

The geometry tree contains exactly 698 parquet/status pairs covering
contiguous inclusive blocks 0–348,599; the last block is 348,500–348,599.
Every status is completed, names/ranges/row counts/output hashes match, and
the 1,038,963 block rows equal the aggregate row count. The aggregate SHA-256
is `c3b01c460dddb2f0dfb06e7e3a1e354098d6464fed508c3e03c64c9324af4fd6`.
All 348,600 frames have one to three hypotheses: 345,062 have three, 239
have two, and 3,299 have one. All XYZ and residuals are finite, XYZ is within
the frozen -20 to 220 mm bounds, residuals are nonnegative `mm^2`, hypothesis
IDs are unique, and active camera slots are mutually exclusive within frame.
The configuration is strict finite mouth, six candidates, three fish,
no false-positive filter, no visible-node centroid, no tail-coordinate
consumption, and fixed-interface refractive calibration. I freshly rehashed
the three caches, four calibration files, and five recorded implementation
files with zero mismatch.

Each A1–A3 direct and filled table has exactly 1,045,800 unique
frame × `ID_1`–`ID_3` rows over 0–348,599, with coordinate finiteness in XYZ
triplets and no direct or filled interpolation. Every finite direct row maps
back to the same-frame aggregate hypothesis, frame-local source rank, XYZ,
and residual (CSV round-trip error at most `2.28e-13`); finite sources are
mutually exclusive within frame. Every nonfinite row is explicitly
unresolved with source index -1. I independently reproduced all recorded
coverage, unresolved, residual, adjacent-jump, speed-spike, and identity-
segment metrics. A1 and A2 each have 1,036,191 finite rows, 342,861
all-three-direct frames (`0.983537`), 9,609 unresolved rows (`0.009188`),
ten identity segments, and no adjacent jump over 8 mm. A3 has 1,029,780
finite rows, 341,975 all-three-direct frames (`0.980995`), 16,020 unresolved
rows (`0.015318`), 117 identity segments, and no jump over 8 mm. Its
348,600-row assignment audit is complete and unique, and its 98 unresolved-
interval records are valid, sorted, and nonoverlapping.

The full-length outputs are not biologically interchangeable despite their
held-out and critical-window tie. A1 and A2 use the same within-frame
candidate set in 348,598/348,600 frames, but assign a different source to at
least one fixed ID in 114,297 frames (228,593 rows); 113,779 of those frames
have all three candidates in both methods. A3 differs from A1 in candidate
retention on 5,091 frames and in fixed-ID source assignment on 320,237
frames. This is not a completion defect: it is the intended distinction among
the frozen temporal linkers. It is a required warning for quality review:
small residual/jump differences cannot select the biologically correct male
identity path without applicable truth.

I also freshly rehashed the seven unchanged A0 artifacts on the experiment
share after full-length finalization; all byte counts and frozen hashes still
match (`ae4b727b...`, `93f32211...`, `6f424833...`, `b2100d24...`,
`757f9808...`, `e9a895b0...`, and `36aa6fe0...`). No A4/A5 command or
full-length directory exists, and A0 was neither copied nor rerun. This PASS
authorizes independent scoring and report generation from the sealed
full-length evidence. It does not authorize threshold changes, reruns, or
promotion based on smoothness/residual alone.

The full-length independent quality seal now also passes final audit. Its
authoritative SHA-256 is
`7e9896716a693ce81287320af1aacd6ab686f909ad4f5cc8de43af54dc03cc92`.
I recursively rehashed all 24 unique bindings with zero mismatch, including
the final full-length sidecar/completion/preparation chain, freeze, held-out
and critical quality gates, scorer `f2df63f2...`, sealed base scorer
`5d7e4e8f...`, metric helper `11e7f14d...`, female ledger `ab0a173a...`,
cohort protocol, four calibration files, all four primary quality outputs,
annotation-partition scorer/outputs, and the closed scorer incident record.
The initial draft seal `4454b20a...` is superseded: it briefly became stale
when the incident wording was corrected before final issue and is not an
authoritative artifact.

Pre-seal review caught and corrected two quality-provenance/reporting defects.
The first scorer did not hash-assert or bind the materially imported metric
helper and female ledger; the corrected scorer now also binds the exact
refractive calibration manifest and bundles. The generic bounded-stage
resource helper initially parsed `A1.fresh` as method `fresh`; the final
scorer parses the full-stage keys as method/purpose and reports exact A1,
A2, A3, and shared-geometry rollups. These corrections did not modify a
scientific stage file, metric definition, or CSV result. The authenticated
incident record, SHA-256 `6ecf08b3...`, accurately records the original
shell-wrapper timeout, a stopped duplicate that wrote no output, deterministic
clean reruns, provenance enhancements, pairwise diagnostic additions, and
resource-metadata correction.

I independently reconstructed all 1,200 rows of the female-assignment ledger
(four methods × 300 confirmed annotations) from the annotation coordinates,
the exact A0/A1/A2/A3 tracks, and the bound refractive projection. Every
annotation ID/frame, projected side-view distance, chosen identity, mapping
status, consensus, and strict-gate flag matches; each method has exactly 900
track rows at the 300 annotated frames. The segment-aware summaries also
match exactly: A0 maps 299/300 and concentrates 221/300 on one fixed identity
with 15 changes; A1 maps 297/300, 180/300, and nine changes; A2 maps 297/300,
192/300, and nine changes; A3 maps 297/300, 94/300, and 19 changes. I
separately recomputed A0's exact 1,045,800-row three-ID grid, `0.989544`
all-three-direct coverage, 597 jumps at least 7.5 mm, no jump over 8 mm, and
the recorded jump quantiles. The A1–A3 geometry, coverage, residual, jump,
segment, and assignment-audit metrics were already independently reproduced
in the completion audit above.

The cohort partition artifact is correctly bound to the 1,200-row assignment
ledger and the preregistered protocol. It exposes a crucial distinction:
bounded held-out tracks initialized inside each window scored 9/9, whereas
the frame-zero full-history A1, A2, and A3 tracks place only 6/9 of those same
annotations on one fixed segment-aware identity and show one change. A0 is
9/9 with no change there. A1 and A2 differ on 40/300 annotations, but 39 are
in the post-hoc remainder, one is validation, and none is held-out, training,
or critical-descriptive. Their descriptive 0.60 versus 0.64 full-ledger
consistency therefore cannot promote A2.

The full critical-event ledger has the exact 108-row four-method × nine-frame
× three-ID grid and matches the bound direct sources. A1 and A3 use
`ID_1=0, ID_2=2, ID_3=1`; A2 uses
`ID_1=2, ID_2=0, ID_3=1`. Each has 27/27 finite rows, zero source-slot switch
inside the event, and the same local jump/residual distributions. This is a
full-history global-label difference, not a local geometry difference. The
two male paths cannot be adjudicated by the female-only ledger.

Final resource rollups match the raw process records: A1 fresh+repeat
`1232.410 s` and 2,643,316,736-byte peak RSS; A2 `1382.381 s` and
3,418,509,312 bytes; A3 `1547.060 s` and 3,396,063,232 bytes; shared
geometry fresh+resume `1142.095 s` and 2,546,966,528 bytes. The sealed
quality report has SHA-256
`5bd61f1c26edf42e11eea6b3b9d443401baa481cc0b25fb5b27b8a84bd0ec67f`
and faithfully distinguishes geometry, temporal smoothness, and biological
identity. I concur with its result: execution/QC **PASS**, but replacing A0
is **not established**. Retain A0 as the current reference, carry A1 only as
the parsimonious experimental mouth-only candidate, investigate the
114,297-frame A1/A2 divergence with male-resolving truth, and reject A3/A4/A5
as production finalists under the present evidence.

## Training-only geometry checkpoint audit

I audited the four completed training geometry outputs without reading any
validation, held-out, or post-freeze method output. Their intervals exactly
match the preregistered inclusive training windows:

| Window | Frames | Rows | All-three frames | Aggregate SHA-256 |
|---|---:|---:|---:|---|
| control for event 3, 21,840–28,039 | 6,200 | 18,600 | 6,200 | `7be2e1584ff74ef7fa6fa4fa970349787e585f604a1f158e0183b9eb8604dd99` |
| control for event 1, 32,264–32,920 | 657 | 1,971 | 657 | `31e831080aa0fdd089d9d1c4f0a163b5e00dddfc3c05fae816848d5a6e756f79` |
| reviewed event 1, 72,371–73,027 | 657 | 1,971 | 657 | `8db09b6411e9e0d7528c9256c86c73f32e3f160d7324f55263c526ba6cb9fd94` |
| reviewed event 3, 174,316–180,515 | 6,200 | 18,600 | 6,200 | `9519a363adc3f1a7b0bae50a9747e12914c14fcc0fb931cb7be83d8f824df13d` |

All 30 block status files say `completed`; every recorded block hash matches
its parquet, block intervals are contiguous and exact, and each aggregate
parquet is byte-scientifically identical as a dataframe to the sorted
concatenation of its blocks. Every expected frame has exactly three finite
XYZ/residual rows, all coordinates lie inside `-20..220`, view-mask/slot
declarations agree, and raw candidate slots are mutually exclusive within
each frame. There are no outside-interval rows or duplicate
frame/hypothesis pairs. No block was reused in this first pass
(`reused_blocks=0`), so checkpoint integrity passes but actual resume
equivalence was not exercised here.

All four summaries have the same normalized settings: strict required mouth,
six candidates, three fish, no centroid, no legacy false-positive filter, no
tail consumption, refractive calibration in `mm^2`, 500-frame blocks, and
128-frame evaluation batches. Recorded cache and calibration hashes match the
files directly. The current geometry-relevant code hashes also match, including
`mouth_ablation.py` at
`80b760c0d2d7e7ae8af9f3809026357b68d33021ad4c2ac036357d3340d0dddf`
and corrected `windowed_tracklet_core.py` at
`4e762136bc2131a57024c96e7558ad6361f5585a32bc5feaaf128c51aefdcd26`.
The geometry summaries bind the canonical cache hashes; the cache-to-SLP
SHA-256 binding remains layered through `input_provenance.json`.

Coverage does not imply uniformly good geometry. Pooled residual tails were
especially large in control 1 (`p99=268.29`, maximum `354.07 mm^2`); maxima
were `169.20`, `90.92`, and `155.08 mm^2` in event 1, event 3, and control 3.
No refractive residual gate was part of the fixed settings, so these rows are
expected inputs to temporal-method comparison and must remain visible in QC.

The external resource summaries for these training jobs sampled only a
roughly 4.6 MB wrapper process and missed the worker; they are invalid peak
memory measurements. Internal process-tree samples were about 176–216 MB but
remain advisory. Finalist runs need the corrected external descendant monitor
before runtime/memory claims are accepted.

## Metric-definition requirements

- Report full-eligible female accuracy with unresolved/missing labels counted
  as failures, alongside selective mapped accuracy and unresolved fraction.
  Otherwise a conservative method can improve apparent accuracy by abstaining.
- “False merge” and “false break” are proxies without three-fish ground truth.
  Name and define the exact proxy; do not report it as biological truth.
  In the current conservative-tracklet implementation, `false_merge_proxy`
  counts accepted edges whose directional error exceeds the base (not
  gap-expanded) prediction gate, while `false_break_proxy` is the count of
  forward-versus-backward proposal disagreements. Neither is a labeled error
  rate, and the merge proxy is structurally constrained by the admission gate.
- Compute direct geometry coverage before spline filling, and separate it from
  named-identity resolution coverage.
- Define speed-spike denominators on direct observations, including how
  nonconsecutive frames are normalized or excluded. Do not mix filled points
  into the identity-link diagnostic.
- Report assignment margins/entropy on the same feasible hypothesis set and
  cost scale, or state that cross-method values are not comparable.
- Female accuracy, refractive residual, and smoothness are necessary
  diagnostics but cannot prove the requested male-to-male identity at
  34,547→34,548.

## Implementation defects caught before scoring

The first cached-geometry runner inherited a validity error in
`enumerate_refractive_hypotheses`: it accepted a slot tuple whenever at least
two pixel coordinates were finite, even when the tuple's declared
`view_mask=0b111` claimed all three views. On frames 34,547–34,548, side1 and
side2 each have only candidate slots 0 and 1 finite, yet the initial smoke
selection produced alleged three-view hypotheses referencing side slots 2 and
3. This fabricated three-view support and nonexistent raw-detection resources.

This is a blocking defect. The corrected enumerator must require every slot
declared present by the view mask to be finite; a genuine two-view hypothesis
must have the missing slot set to `-1`, `view_count=2`, and explicit
missing/synthetic-view provenance. All geometry or A1–A5 outputs created
before that correction are ineligible.

The executor corrected this before scored method runs and added a regression.
I then compared the corrected cached geometry on the training-control smoke
interval 32,264–32,320 against
`triangulate_extra_candidates.compute_best_assignments_per_frame`, fed the
same canonical cache arrays with refractive calibration, six candidates,
three fish, bounds `-20..220`, and missing-view recovery. Across 57 frames and
171 selected rows, selected slot tuples were identical, XYZ values were
bit-exact (maximum absolute difference `0.0 mm`), and residuals were bit-exact
(maximum absolute difference `0.0 mm^2`). There were no cardinality or
resource-key mismatches.

The first deferred A3 implementation also allowed a long run of missing
geometry to retire all states and then silently reseed identities under the
same identity-segment label. That could claim continuity unsupported by any
observation. Before scoring, the executor changed incomplete three-fish
geometry to explicit `unresolved_insufficient_geometry`, records the interval,
and starts a new identity segment when the 100-frame state gap is exceeded.
Targeted tests cover both all-missing and partial-geometry long gaps. Final
method artifacts still need to bind the post-correction code hash.

The new A4 solver is structurally distinct from the retired connected-component
workflow. In code review it uses median plus MAD refractive residual, mouth
confidence, missing-view penalties, a bounded saturating support bonus,
per-view tracklet-resource exclusion, explicit end-to-start links (including
an `all_views_break` case), a bounded three-path beam, and optional unassignment.
It does not sum residual by duration. This is a structural implementation
check, not a result verdict; cutoff selection, regression results, runtime,
coverage, held-out behavior, determinism, and full-recording feasibility
remain to be audited.

During scale review I found that its first robust-velocity implementation
materialized every pairwise slope, making one fragment quadratic in length.
The executor replaced this with deterministic lags 1, 2, 4, 8, 16, 32, and 64
(linear work). A second review found that the initial beam copied growing full
paths/resource histories and omitted `path_segment` from selected observations,
which could both impair full-scale execution and falsely imply continuity
across an arbitrary restart. The then-reviewed hash
`9ad89c56b0388c98bdf3a37e27c7a3bda2e63812395d85ddf69708b574e3406d`
uses compact last-state/backpointers, prunes resource ownership after its final
future use, propagates `path_segment`, and marks links beyond 100 frames
`segment_restart` with `continuity_claimed=false`. The independent reviewer
reported 32/32 focused tests passing. A repeated-resource benchmark at the
default 256-state beam took 13.09 s and +21.2 MB RSS for 300 groups, and
62.87 s and +22.8 MB RSS for 1,200 groups. Actual group counts and an externally
monitored end-to-end run are still required before A4/A5 are full finalists.

That structural hash was superseded by evidence propagation needed for A5.
The freeze-candidate hashes are
`conservative_cross_view_paths.py`
`4bed90ac17656c7792baf62105ea2acb8253b034846123c215e378d8d3757c8a`
and `mouth_tracklet_ablation.py`
`a2b1dff909716a0f59f6c4c422b6cd1caa59eb84a3324d3f18e90b648a143e2e`.
The expanded focused suite passes 47/47 at the reported freeze candidate.
Final training-regeneration provenance and actual full-scale group counts still
need audit; an old source hash cannot inherit that verdict.

The current A4 runner now writes a complete three-path-by-frame direct grid,
uses explicit unresolved statuses, and writes a segment-bounded filled output
without nonnegative shifting. This resolves the earlier sparse-output
denominator defect. Its summary still needs explicit direct/filled QC metrics,
including resolution and interpolation denominators, unless those are
generated and bound by the final report assembly.

The integrated A5 workflow now exports strong A4 hint rows and the union of A4
and A2 segment-reset frames, then invokes the frozen deferred tracker and
writes direct, filled, assignment-audit, unresolved-interval, hint-application,
QC, and hash-bound summary artifacts. Its hint filter requires eligible,
count-consistent, three-view groups with minimum support, robust residual and
mouth-confidence evidence, and favorable group assignment cost. It maps each
A4 segment by high-purity vote to the composite frozen-A2 identity segment,
validates exact `frame+hypothesis_id` plus XYZ, rejects overlapping segment
maps to the same identity, and enforces a one-to-one source/identity mapping
per frame. A hint/motion conflict defers instead of bypassing the hard motion
gate. An independent actual-runner smoke test initially caught an `a/b/c`
versus `group1/2/3` source-label crash; after correction, the same test passes
with 9/9 hints honored, complete output grids, matching hashes, and the fixed
refractive calibration contract.

## Training relink sweep audit

The completed v3 training relink sweep passes an independent 4-window by
6-setting grid audit (24/24 summaries). The grid is exactly A1 plus five A2
configurations:

- velocity weights/history lengths `0.25/5`, `0.5/5`, `1.0/5`, `0.5/9`,
  and `1.0/9`;
- all A2 runs have tangent weight `0`, acceleration weight `0`, uncertainty
  growth `0.25`, and `max_motion_gap_frames=20`;
- every run uses jump `8 mm`, hard cap `80 mm`, gap `100`, no coordinate
  shifting, no nonnegative fill, and no deferred-A3 configuration.

All summaries bind the expected one of the four independently audited training
geometry hashes. Their recorded code hashes match the current files:
`cross_frame_tracking.py`
`db29b8251717ea32b13b54a2c2b1c077111a276a247b88ed73ba17aa41f53496`,
`deferred_identity_tracking.py`
`c08ac9071b7ab218d338a2e1a2859811a2c322bf1d93f5740e1202c88a663add`,
`fill_missing_3d.py`
`7a2f7fe1e0e9ab94c9541fd62e2737bc033eb902658bbac59c91f7ea5d174818`,
and `mouth_ablation.py`
`80b760c0d2d7e7ae8af9f3809026357b68d33021ad4c2ac036357d3340d0dddf`.
Every direct and filled hash matches its file. Direct outputs contain exactly
three unique `ID_1`–`ID_3` rows for every inclusive frame, have no duplicate
frame/ID keys, and mark no direct row as interpolated. Recorded direct
all-three coverage is `0.9756468798` for control event 1 and at least
`0.9998387097` for the other windows; all 24 summaries report zero unresolved
row fraction and zero adjacent steps over 8 mm.

All 24 stderr logs and the v3 wrapper stderr are empty; every stdout log parses
as JSON and matches its summary configuration hash. The active output tree
contains no validation, critical, or held-out filename/path and no stale
external-resource summary. The v3 jobs all started after 02:51:52 JST, so the
previous partial A1 output was overwritten by the controlled sweep. Failed v1
and v2 wrapper logs remain only at the report root and explicitly record their
monitor failures; they are superseded diagnostics, not scored results. Internal
process-tree peaks span about 179–223 MB and wall times about 0.37–5.94 s.
They are valid within-run measurements, but independent external monitoring
is still required for finalist runtime/memory claims.

The independent training scorer found every A2 direct output byte-identical
within each window, and all A1/A2 aggregate diagnostics are consequently tied:
15/15 female annotations mapped to one fixed identity within their window,
zero observed female-label changes, direct all-three coverage
`0.9973749453`, 38 unavailable direct rows, seven adjacent direct jumps at
least 7.5 mm, and none over 8 mm. Selecting A2 weight `0.25` with history `5`
is therefore a predeclared lower-complexity tie-break, not an empirical
performance advantage. The female denominator is only 15 (window counts
2/1/6/6), is insufficient to estimate generalization, and cannot detect the
male-to-male error central to the critical event.

After the training sweep, I found a full-recording continuity defect not
exercised by those short windows. `apply_cross_frame_tracking` retires an A1/A2
identity after a gap over 100 frames and then reseeds its label from an
unmatched observation. The old `run_relink_window` stamped every A1/A2 output
row `identity_segment=0`, so a retired label could silently resume as an
arbitrary fish while claiming global continuity. The quality reviewer
independently reproduced this using actual relink output at frames 0 and 102.
The freeze-candidate code now exposes per-identity segment increments, marks
missing geometry unresolved, has long-gap regressions, and makes A5 vote on
the composite A2 identity segment with forced resets at A2 segment starts.
All earlier training relink outputs are consequently provenance-superseded.

## Rejected and superseded run registry

- Training relink v1 is rejected: its resource-monitor `Wait-Process` failed
  after the first partial run.
- Training relink v2 is rejected: the wrapper detected a missing resource
  summary. Its reviewed-event-1 A1 output was subsequently overwritten.
- Training relink v3 was internally complete and passed the 24-grid audit
  described above, but is now provenance-superseded by the A1/A2 explicit
  segment/unresolved correction. It must not be used for final selection.
- Training A3 v1 completed 12 runs, but its assignment-audit CSVs were not
  included in the summary output hash manifest and its code hash was
  superseded.
- Training A3 v2 is rejected as a mixed generation. It was killed during
  `reviewed_event_3/A3_a005` after writing new direct, filled, assignment-audit,
  and unresolved files at about 03:12:12 JST, but before replacing the old
  03:08:28 summary or writing stdout. Only 4/12 summaries bind the
  freeze-candidate code. No artifact from this directory is eligible for
  selection.

Regeneration must use fresh directories, or delete every expected output before
starting, so an old summary can never authenticate a partially replaced output
set. Final audit requires a single generation in which all geometry, direct,
filled, assignment-audit, unresolved, stdout, stderr, and summary timestamps
and hashes agree. The reported code-freeze candidate is:

- `mouth_ablation.py`
  `d856585da1515a06067909f0b7aeead1eab2af81973a2cf097e17de33a7192e0`;
- `mouth_tracklet_ablation.py`
  `1e75e10654b5d6c233ec3ea5ff2f8c969dac3dd41027031cd332d3c88096d4e0`;
- `deferred_identity_tracking.py`
  `c08ac9071b7ab218d338a2e1a2859811a2c322bf1d93f5740e1202c88a663add`;
- `conservative_cross_view_paths.py`
  `4bed90ac17656c7792baf62105ea2acb8253b034846123c215e378d8d3757c8a`;
- `conservative_mouth_tracklets.py`
  `a454ae0b6e59cfd51bdea4b60de29438fc8c407c861e05befd80d3e1d9da2631`.

The independent post-fix checks report 47/47 focused tests passing. The actual
A5 runner end-to-end smoke passes at the freeze-candidate hash with 9/9 hints
honored, a complete direct grid, matching output hashes, and the refractive
calibration contract. The A1/A2 long-gap relink audit also passes after the
segment fix.

Fresh training geometry was regenerated at 03:14–03:15 JST after quarantining
the pre-freeze work tree. All four summaries bind final `mouth_ablation.py`
hash `d856585d...`, the current cache/calibration/core hashes, and the fixed
settings. All 30 blocks are completed, contiguous, and hash-authenticated.
The four aggregate parquet hashes are byte-identical to the previously audited
scientific outputs, so the earlier dataframe-to-block, bounds, finiteness, and
resource-exclusion checks transfer exactly. No validation, critical, or
held-out path appears. The fresh directories do not contain external resource
summaries; internal process-tree peaks remain recorded, so this is sufficient
to launch training relinks but not to support final external memory claims.

Fresh final-hash relinks were then regenerated in clean directories. The A1/A2
grid passes 24/24: exact 4×6 configurations, final code and fresh-geometry
hashes, fixed motion isolation, authenticated direct/filled outputs, parseable
matching stdout, empty stderr, exact frame×ID keys, and no nonfinite row
misreported as resolved. All five A2 outputs remain byte-identical within each
window. The old zero unresolved fraction was corrected: control event 1 has
16 explicitly unresolved rows (`0.812%`), while no training window triggers a
segment reset. The independent long-gap test covers that unobserved boundary.
The old training-motion selection file was superseded and was regenerated
before freeze.

The fresh A3 grid also passes 12/12. Every run binds the final code and geometry,
the exact fixed deferred settings, and all four direct/filled/assignment-audit/
unresolved output hashes. Assignment audit has exactly one row per frame;
stdout/stderr and key grids pass. Acceleration settings are `0`, `0.05`, and
`0.1` with frozen velocity weight `0.25`; all other deferred parameters are
identical. Control event 1 again has 16 row-level
`unresolved_unassigned_identity` rows. Its unresolved-interval JSON is empty
because the ambiguity episodes resolved even though individual assignments
remained unavailable, so final reporting must retain the row-level denominator
and not equate an empty interval list with complete resolution.

The regenerated final-hash training selections are authenticated. The A2
selection file SHA-256 is
`4e83c1e84c1775595315dec43579b135eb4d5998aaaa9b780e87ece6921a4038`
and its full audit is
`92c99a84beef714f0433c5844616f00a2ecc83679ddcdcf68d94f824cadcfa4d`;
all recorded direct hashes match the fresh summaries. It selects velocity
weight `0.25`, history `5`, solely by the lower-complexity tie-break because
all A2 outputs and metrics are tied. The A3 audit SHA-256 is
`432358a6595dec4b430a14d2d1d70dacbb880367489d4f515259884c2054b059`;
all summary, direct, and summary-authenticated assignment-audit hashes match.
It correctly excludes zero acceleration as a negative-control deferred-
velocity ablation and selects the lower eligible positive weight `0.05`.
Both artifacts state training-only scope and remain subject to the sparse
female/male-male limitations above.

The first launched A4 cutoff sweep is rejected. Its script used the same
already-exited-PID `Wait-Process` pattern: only the two 657-frame windows of
the first configuration completed, while the two long-window logs remained
empty and their workers disappeared. That output root must be quarantined and
must not be resumed. The corrected sweep must use process-object waiting and a
fresh root.

Two later A4 launch generations are also rejected and quarantined. The second
generation progressed to 14/72 summaries before review found that
`a4_summary.json` authenticated only the nine final tables, not the per-view
tracklet, disposition, bridge, and summary files. The third generation was
stopped before scoring because its first manifest correction still omitted
the individual hypothesis-block parquet and status files. Neither generation
may be resumed or contribute to training selection; the defect was provenance
completeness, and no threshold was selected from either run.

The final A4 provenance candidate passes the clean relaunch gate. The runner
hash is
`1e75e10654b5d6c233ec3ea5ff2f8c969dac3dd41027031cd332d3c88096d4e0`.
Body-length scales were regenerated after that code change; their artifact
hash is
`63bf6de26bb85e76da97976ffa0158c8d4e27d7ea9e3208b627fe8ade2eb9209`.
The three values are unchanged (`109.58292697843919`,
`91.11906436144642`, and `86.47002765799765` pixels for top, side1, and
side2), and the artifact binds the final runner and the three canonical cache
hashes. The final manifest smoke summary
`smoke_a4_manifest_final/a4_summary.json` has SHA-256
`952da39e372efc3ceb4db51d04f7e2339f863c54481d15696a7cb6bbc1556155`.
It authenticates 23 artifacts for its one-block interval: nine final
scientific tables, 12 per-view files, and the hypothesis-block parquet and
status JSON. Independent recomputation found zero missing or mismatched
artifact hashes. Its runner and scale bindings are exact.

The final cutoff launcher has SHA-256
`0d87de5894af9335721c1f1b672ca1921160a7128694c2d526a2562da3a89b4f`.
It binds the runner and scale hashes above, refuses an existing output root,
uses process-object `WaitForExit()`/`Refresh()`, launches the exact
4-window × 18-configuration grid, and recomputes every artifact hash advertised
by each summary. The fresh production sweep root was absent when this gate was
audited. Runner, scale, and launcher hashes must remain unchanged throughout
the sweep.

The earlier 03:17 `score_training_tracklet_cutoff_sweep.py` and its outputs are
per-view diagnostics only and are explicitly ineligible for A4 selection. That
script independently rebuilds reciprocal tracklets but does not authenticate
or score the 72 integrated A4 summaries, three-path solutions, or direct
abstention/coverage. It also uses names different from the production launcher.
A separate method-level scorer must authenticate the fresh runner/scale/config
and all advertised outputs, then include path status, direct resolution,
unassignment, and coverage before applying the purity-first training rule.

During the live final sweep, three early summaries reported
`insufficient_three_path_solution`. These are valid completed scientific
abstentions rather than execution failures: `control_for_event_1` at
`s050_m005_g2` and `s050_m010_g2`, and `reviewed_event_1` at
`s035_m010_g5`. Each has exactly 1,971 direct and filled rows (657 inclusive
frames × three path IDs), 1,971 unique frame/ID keys, an empty path table,
and all rows explicitly marked `unassigned_tracklet_path`,
`identity_segment=-1`, non-interpolated, with unavailable geometry. Every
one of the 27 advertised artifacts exists and independently rehashes.
Selection must retain these configurations with zero direct coverage and 100%
unresolved rows; it must not silently exclude abstaining configurations as
failed runs.

The final training sweep completed 72/72 runs at 03:45:30 JST. A read-only
full-tree revalidation passed: the directory is the exact four-window ×
18-configuration design; status counts are 69 `completed` and three
`insufficient_three_path_solution`; all 3,528 advertised artifacts
independently rehash; all 1,008 hypothesis-block parquet/status pairs are
exactly named, contiguous, completed, data-hash authenticated, and bound to a
recomputed block configuration hash. Every direct and filled table has exactly
the inclusive frame × `PATH_1`–`PATH_3` key grid, and every direct row is
non-interpolated. Code, controlled settings, frozen scales, stdout, and empty
stderr checks pass. The sum of per-run internal wall times is 1,857.262 s and
the largest internally sampled process-tree RSS is 735,158,272 bytes.

The launcher source was extended with canonical completion-manifest fields
while its already-running PowerShell process was executing. The start-gate
launcher hash was `0d87de58...`; the later replay-script hash is
`bc534144...`. This metadata-only drift did not change the frozen Python
runner, scales, configurations, or outputs, but both hashes and their roles
must be recorded. The original completion sentinel (SHA-256 `1e5765ff...`)
contains the correct 72 count and status split but lacks the promised schema,
runner hash, all-artifacts flag, and 72-summary hash manifest. It is therefore
insufficient for selection until replaced by a post-validation sentinel whose
contents independently match the revalidation above.

The corrected canonical sentinel passes. Its SHA-256 is
`30a6841efdd5215b96202c567a08478f20530fca5ca40248d7b1a3ab9a4dae2f`;
the postvalidator hash is
`3092cbdd00993ebd9b7c4ca44cb2135a41de1424c2f43e8d5fb488a72ae16e24`.
It contains the exact 72 unique, path-sorted summary set. Every recorded
summary hash, scientific status, and configuration hash matches; the 3,528
artifact count and 69/3 status split match the independent full-tree audit.
The recorded final runner, scale, replay-launcher, and postvalidator hashes all
match the current files.

The definitive method-level A4 selection audit has SHA-256
`81ba93ef249188824ad1263ee550e51b0890e19f85231e82a37be80ce71ef697`.
It self-authenticates scorer hash
`5144b65a2835a6295b8cc186faade9387a5a47ca68c5462e2ed3507894126cc8`,
base-scorer hash
`f98e7c11391913455e8cbe48472edd58a85d27225575b972f0f5c167776e7b9a`,
annotation hash
`ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f`,
and the canonical sentinel above. Independent recomputation exactly matches
its full ranking and the 270-row (18 settings × 15 annotations) assignment
ledger. It selects `s035_m005_g0`: separation `0.35` body lengths, ambiguity
margin `0.05` body lengths, and maximum missing gap `0`.

This A4 result is high-purity but extremely sparse. It maps three of 15 female
annotations, all three to one fixed identity (selective accuracy 3/3, full-
eligible accuracy 0.20), has zero observed female-label changes and zero
false-merge/false-break proxy counts, but has zero all-three-direct frame
coverage, 91.157% unavailable direct rows, and selects only 27.419% of eligible
cross-view groups. It is not evidence for a standalone production trajectory.
It is eligible only as sparse high-confidence temporal evidence in the
training-only A5 hybrid audit before freeze.

The training-only A5 hybrid audit passes its provenance and semantic checks
but fails the performance gate for a production finalist. All four runs bind
the selected A4 `s035_m005_g0` selected-observation hash, frozen A2
velocity `0.25`/history `5` direct hash, and fresh strict-mouth geometry hash.
They use the fixed A3 acceleration `0.05` deferred settings and the declared
high-confidence hint rules: three views, mouth confidence at least `0.5`,
group support at least three, group median residual at most `1.0 mm^2`,
residual MAD at most `0.25 mm^2`, mapping votes at least three, purity at
least `0.9`, and vote margin at least two. Code, calibration, all input/output
hashes, exact frame × `ID_1`–`ID_3` grids, one audit row per frame, accepted
vote criteria, bijectivity, and the union of A4/A2 segment-reset frames pass.

Across training there are 3,263 hint rows and 1,351 are honored. Every one of
the 1,912 unhonored hints occurs on an assignment-audit frame explicitly
marked constraint-conflicted and ambiguous, not resolved on that frame; 1,092
are later resolved after lookahead and 820 remain unresolved. Thus the hybrid
does implement the declared defer/retain behavior rather than silently
overriding a conflicting hard motion gate.

The definitive quality artifact is
`871570d10ed4333a7aa8192966a31a674cd240f613bf2144db023373c73855b1`
under scorer
`0d2346cbf30b9ccabccb05664fe08472633c999bdbc21f78e7080636dacf6f93`;
the four-run completion artifact is
`b9cef67a3b88e3200c10305422cd2abe6ba8644cf1fdd49c04bad9215bf26749`.
A5 attains only 6/15 full-eligible female accuracy with eight observed
identity changes, all-three-direct coverage `0.937582`, and 6.077%
unavailable rows. Selected A3 has 15/15, zero changes, `0.997375`, and
0.092%, respectively. Both have seven jumps at least `7.5 mm` and none over
`8 mm`. A5 is therefore frozen as an evaluated exploratory configuration,
not a full-length or production finalist.
