# Item 3 final independent verifier report: explicit fragment links

Date: 2026-07-17

## Verdict

**Implementation/safety verdict: PASS for the final v2 links-on artifact.** The
revised code now implements the frozen endpoint estimator, native-mm^2 mouth
objective, gap and motion gates, neutral missing-motion behavior, separate
3.0 birth/death costs, deterministic edge ordering, and restart segments rather
than fabricating three recording-wide biological identities.

**Scientific acceptance verdict: FAIL / defer.** The v2 artifact proves that
378 feasible links compress 491 fragments into 113 restart segments without a
gate violation or speed spike, but there is no matched final v2 links-off run on
the exact same fragment table. The current `fragment_link_mode=none` branch also
uses an implicit per-frame Hungarian tracker rather than leaving every fragment
break as a birth/death, so it is not the frozen Item 3 control. No reviewed-event
or identity truth result establishes that the 378 links are biologically right.

I did not author the implementation and changed no reusable code.

## Reproduced tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv\Scripts\python.exe -m pytest -q tests\test_windowed_tracklet_stitching.py `
  --basetemp report\20260717\windowed_tracklet_stitching\pytest_tmp\foundation_audit_item34_final
```

Result: **16 passed, 1 pytest-cache permission warning, 1.64 s**. The warning
concerned the repository `.pytest_cache`; the actual basetemp was unique and
workspace-local. Third-party plugin autoload was disabled because the installed
Napari pytest plugin otherwise tries to read a denied user-settings path before
collection.

An additional read-only probe with both endpoint velocities missing returned
`motion_prediction_error_mm=NaN`, `velocity_sources_available=0`, and
`prediction_cost=0.0`. Reversing the two input fragment rows produced an equal
canonical edge table.

## Final artifact inspected

`bench_item3_mouth_v2_32264_33263` covers frames 32264--33263 of
`20251225_K_1_2_FCMC`, SLEAP `round7_260704`, single-window mode, XYZ bounds
0--200 mm, fixed refractive calibration, and seed 20260717.

| Measure | Verified value |
|---|---:|
| Direct observations | 2,819 |
| Input fragments | 491 |
| Feasible directed edges | 38,592 |
| Selected edges | 378 |
| Restart segments/paths | 113 |
| Objective | 1,677.260984288198 |
| Selected edge-cost sum | 999.260984288198 |
| Birth/death contribution | 113 x (3 + 3) = 678 |
| Unresolved intervals | 0 |
| Resource conflicts / frames over 3 active | 0 / 0 |
| Wall time / peak RSS | 16.213 s / 295.87 MiB |

The objective reproduces exactly as
`999.260984288198 + 113 * 6 = 1677.260984288198`. Also,
`491 - 378 = 113`, so every selected edge accounts for exactly one reduction in
path count; there is no hidden merge or recording-wide three-ID relabel.

## Claim-by-claim verification

| Claim | Result | Evidence |
|---|---|---|
| Fixed refractive/native units | **PASS** | Resolved backend is `refractive`; all 38,592 edge rows say `mm^2`; calibration provenance defines the score as mean squared point-to-water-ray distance. |
| Gap/time direction | **PASS** | Candidate gaps span 0--100 only; the focused test rejects gap 101. |
| 8 mm/frame and 80 mm hard caps | **PASS** | Zero violations across all edges; maximum edge displacement 79.9572 mm and maximum edge rate 7.99650 mm/frame. |
| Robust endpoint velocity | **PASS** | Code uses up to 15 samples inside 30 endpoint frames, requires at least three spanning two frames, and takes all pairwise median slopes. |
| Bidirectional/missing motion | **PASS** | Available forward/backward errors are median-combined; 28,367 edges had neither velocity and receive a zero motion term. The direct missing-motion probe passed. |
| Frozen mouth cost | **PASS** | Position, prediction, gap, overlap, residual, and confidence weights/scales match the primary method; velocity-angle weight is zero. Maximum serialized term-sum error was `8.88e-16`, below `1e-12`. |
| Residual is soft, not a gate | **PASS** | Native residuals reached 2,076.41 mm^2 yet remained inspectable, capped-cost candidates. |
| Restart path-cover objective | **PASS** | 378 selected edges, 113 distinct `fish_id` segment labels, exact birth/death accounting, and no spline-filled continuation in restart mode. |
| Selected-link jump safety | **PASS** | Maximum selected gap 37, displacement 36.8987 mm, and rate 7.96565 mm/frame. No selected rate exceeded 8 and none exceeded 80 mm displacement. |
| Direct trajectory jump audit | **PASS** | Across 2,706 within-segment transitions: rate p95 1.27211, p99 5.42100, max 7.96565 mm/frame; zero rates above 8 and zero distances above 80 mm. |
| Deterministic canonical edge ordering | **PASS locally** | Reversed-row probe was equal; the builder explicitly sorts returned edges. No multiworker/checkpoint rerun artifact was supplied. |
| Links-on versus frozen links-off attribution | **FAIL** | No final v2 links-off artifact shares the 491-fragment table. Existing older baselines have different fragment counts/hashes. |
| Biological identity improvement | **NOT ESTABLISHED** | Fragment reduction and physical smoothness are necessary but do not prove correct fish association; no matched truth result was provided. |

## Remaining specification mismatches

1. **The links-off control is still wrong in reusable code.** When
   `fragment_link_mode=none`, the pipeline calls `track_observations_hungarian`
   with gap and motion gates. That performs implicit continuation association.
   The frozen Item 3 control requires no inter-fragment continuation edges and
   one birth/death per frozen fragment. A final control should reuse the exact
   v2 `fragment_observations.parquet` and `fragments.parquet`, assign one segment
   per fragment, and serialize no solver links.

2. **Overlap consistency is not the specified pairwise edge evidence.** The
   reconciler totals accepted alias support per canonical fragment; the edge
   builder then uses the minimum agree and maximum disagree counts of its two
   endpoints. The primary method defines agreement directly for that proposed
   continuation from duplicated context resource sets. This v2 single-window
   run had zero non-null overlap values, so the discrepancy was neutral but not
   validated.

3. **Rejected candidates are not serialized.** The artifact contains feasible
   edges only, without the required rejected-edge rows/reason codes for gap,
   displacement, or resource preflight failures.

4. **Boundary determinism is untested here.** This is a 1,000-frame single
   window with zero alias candidates and zero seams. It cannot verify equality
   across a 10,000-frame boundary, worker completion permutations, or checkpoint
   resume.

## Scientific acceptance

The final Item 3 implementation is suitable for continued controlled testing:
its internal arithmetic, units, hard safeguards, and segment semantics pass.
It should **not yet be retained as a scientifically supported improvement**.
Acceptance requires the exact same v2 fragments evaluated with all links off,
the four frozen reviewed windows and synthetic truths, boundary/resume
determinism, and identity evidence showing that reduced fragmentation does not
come from wrong joins. The available sex annotation identifies only the female,
so any such metric must be labeled female-only rather than full three-fish
identity accuracy.

## Addendum — exact links-off analytical control (2026-07-17)

After the verifier report was written,
`item3_exact_links_off_control.json` supplied the previously missing exact
same-fragment links-off control. I independently checked it against
`bench_item3_mouth_v2_32264_33263/run_summary.json`:

- graph hash `793c547c698f3a906975adaf32321dc2d9dbcfe9d353aed0f2542e47d636d4a7`
  matches the immutable Item 3 graph;
- both identify 491 fragments and 2,819 direct rows;
- links-off has 0 selected links, 491 restart segments, and objective
  `491 * (3 + 3) = 2946.0` exactly; and
- links-on has 378 links, 113 segments, and objective
  `1677.260984288198`, an exact change of `-1268.739015711802`.

This **supersedes the earlier “matched control absent” status**: same-graph
fragment-association attribution now passes algebraically. The control is
untimed and does not run the reusable `fragment_link_mode=none` code path, so it
does not establish a runtime comparison. Most importantly, reducing 491
segments to 113 proves link attribution but not biological correctness; the
scientific acceptance remains deferred, and female-only ground truth cannot
detect male-to-male swaps.
