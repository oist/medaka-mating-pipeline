# Shared benchmark HITL v2

The first human-reviewed export revealed that two selected K_1_2 source
frames were acquisition setup rather than three-fish behavior:

- frame 1207 contains only one fish in all three views;
- frame 4626 is fish-introduction footage (the top camera is obstructed).

These are selection errors, not false-negative annotations. HITL v2 replaces
them with the next deterministic, event-disjoint eligible events:

- frame 19323 (`contact_19256_19378_a730f6fd79`);
- frame 27474 (`orphan_27471_27498_f196b26f18`).

Both replacements were visually checked in all three cameras and contain all
three biological fish. The 166-frame event guard and the original 14/13/13
experiment quotas are preserved. The v1 manifest and annotation export remain
unchanged.

## Replacement review

The portable bundle is installed as `prediction_replacement2` beside the
original annotator bundle.

1. Run `run_replacement_review_2.bat` to review the 16 Round-8 candidate
   instances one at a time.
2. Save and close the instance reviewer.
3. Run `run_replacement_add_fn.bat`. In this two-frame view, ensure every
   camera has exactly three real fish. Use `I`, then click mouth and tail, to
   add a missed fish. Do not label reflections.
4. Save `output/replacement_review_2_complete.pkg.slp`.

Partial or hidden mouth/tail nodes are valid when their true locations cannot
be observed. The evaluator is partial-label aware and reports full and partial
subsets separately.

## Freeze after replacement annotation

Merge the original 40-frame HITL export with the two reviewed replacements:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.finalize_hitl_benchmark `
  --base-slp "<prediction_review40>\output\prediction_review_40_labeled.pkg.slp" `
  --replacement-slp "<prediction_replacement2>\output\replacement_review_2_complete.pkg.slp" `
  --manifest report\20260726\sleap_hitl_benchmark_v2\shared_benchmark_40_hitl_v2_manifest.csv `
  --output-dir "<benchmark-output>" `
  --prefix shared_benchmark_40_hitl_v2
```

The finalizer never modifies either input. It:

- retains only the 40 source frames in the v2 manifest;
- gives exact replacement-frame keys precedence over the v1 export;
- requires 120 view frames and exactly three user instances per view;
- preserves partial/hidden nodes while flagging them in the ledger;
- normalizes the duplicate skeleton and removes review-only tracks;
- writes an embedded locked SLP, a 360-row instance ledger, a compatible
  120-row evaluation manifest, and SHA-256 provenance.

The original reviewed export audited on 2026-07-26 was 159,728,825 bytes with
SHA-256
`b4f6253ba8a30106a6448cc26234081b31553e043b1661f950e13b3a702ee6da`.
