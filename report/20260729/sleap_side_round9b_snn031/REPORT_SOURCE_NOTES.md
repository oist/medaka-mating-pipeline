# SLEAP side-model 0.3.1 validation report source notes

`report_snapshot.sql` is an exact SQL materialization of the six bounded
datasets embedded in `artifact.json`. It exists to satisfy the portable-report
widget provenance contract. The upstream computational evidence remains the
JSON artifacts listed below; no metric is recomputed or altered by the SQL
materialization.

## Reporting job

- Question: Does SLEAP-NN 0.3.1 improve the current side-view top-down models, which weights should be used in production, and what should be trained next?
- Audience: technical.
- Decision baseline: frozen Round8 centroid plus Round8 centered-instance model.
- Success criteria: retain detection while improving localization on the locked shared benchmark and under the production legacy `track` path.
- Primary cohort: 80 side-view images from 40 synchronized moments, 240 GT fish, spanning K_1_1, K_1_2, and K_1_3.

## Required-structure mapping

- Title: title markdown block.
- Technical summary: answer-first summary block.
- Key findings with visual evidence: production per-GT OKS, per-experiment OKS, and oracle offset-stress charts.
- Scope and definitions: visible benchmark/metric definitions block.
- Methodology and validation: visible three-stage validation block plus checkpoint-selection table.
- Limitations and robustness: visible uncertainty/caveat block and paired cluster bootstrap table.
- Recommended next steps: visible production and training recommendation block.
- Further questions: visible final block.
- Dedicated experimental-design detail: merged into the methodology block because all three requested stages are one linked validation design.

## Chart map

1. `production_per_gt_oks`
   - Question: Which centered-instance model gives the best production-path pose quality when misses are penalized?
   - Takeaway: Round8 is best; Round9b and fresh are materially lower.
   - Family/type: comparison and ranking / horizontal bar.
   - Fields: candidate, per-GT OKS with misses zero, matched-only OKS, F1, matched/missed/extra counts.
   - Palette: single blue root plus neutral comparison shades; candidate labels provide non-color distinction.
   - Source: `stage3_fresh_legacy_track_m6.json` plus the paired-bootstrap transform.

2. `production_per_experiment_oks`
   - Question: Is the Round8 advantage confined to K_1_2?
   - Takeaway: Round8 leads in all three experiments, with the largest fresh regression on K_1_2.
   - Family/type: comparison / grouped bar.
   - Fields: experiment, candidate, matched-only OKS, image count.
   - Palette: three restrained roots; direct experiment and candidate labels.
   - Source: `stage3_fresh_legacy_track_m6.json`.

3. `oracle_offset_robustness`
   - Question: Did crop-position jitter improve centered-instance robustness to anchor error?
   - Takeaway: yes; both jitter-trained models are much more robust at fixed diagonal +/-8 px offsets, but Round8 is substantially more accurate at the correct crop and still wins end-to-end.
   - Family/type: comparison / grouped bar.
   - Fields: offset, candidate, matched-only OKS, mouth p95, tail p95.
   - Palette: same candidate mapping as chart 2.
   - Source: the three `stage3_oracle_ci_offset*.json` artifacts, distilled in `final_summary.json`.

## Statistical transform

- The evaluator's standard OKS is matched-only.
- The paired sensitivity metric assigns OKS 0 to each unmatched GT fish, averages the six side-view GT fish within each `selection_frame_key`, and treats the resulting 40 synchronized moments as the independent resampling units.
- Two-sided percentile 95% intervals use 200,000 paired cluster-bootstrap resamples with seed 20260729.
- This is descriptive uncertainty for this fixed benchmark, not a population-randomized causal interval.

## Material caveats

- Only 40 synchronized moments are independent; camera images are paired within moment.
- The locked benchmark was selected independently of the models and excludes Round8 fine-tune members, but coordinates were human-reviewed from Round8-assisted suggestions. Subpixel localization may therefore favor Round8.
- Round9b epoch 32 was retained after e32/e55/e57 comparisons on this benchmark, so its benchmark result is test-selected and optimistic.
- Internal callback mOKS, built-in post-fit validation mOKS, external matched-only OKS, and per-GT missing-zero OKS use different matching/denominator rules and must not be compared as if identical.
- Automatic post-fit train-set inference hit a sparse packaged-frame index and produced no train metrics. Validation inference, checkpoint selection, and both external benchmarks completed successfully.

## Visual QA contract

- Quantitative charts use native artifact charts, zero-based OKS axes, explicit candidate labels, and bounded reviewed snapshot rows.
- Tables provide exact lookup for detection/pose tradeoffs, checkpoint selection, parity, and bootstrap intervals.
- Every chart has an adjacent narrative block stating the takeaway and interpretation.
- Final QA is delegated to the packaged portable-report validator and browser verifier; if Chromium is unavailable, the semantic chart tables remain the fallback and the structural-only result must be disclosed.
