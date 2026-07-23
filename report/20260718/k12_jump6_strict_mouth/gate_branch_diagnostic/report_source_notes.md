# Report source notes

## Reporting job

- Question: where does the 6 mm gate create a new identity branch that the
  paired 8 mm linker would not create?
- Audience: technical; the reader will visually validate biological female ID.
- Scope: full 348,600-frame K_1_2 Round-7 strict-mouth reconstruction.
- Baseline: paired 8 mm rerun with identical unlabeled direct 3D observations.
- Success criterion: exact trigger frames, reproducible gate evidence, practical
  review windows, and an annotator-compatible queue.

## Technical-report structure mapping

- Title: `Where the 6 mm gate creates new identity branches`.
- Technical summary: direct answer, review priority, and causal boundary.
- Key findings with visual evidence: trigger timeline, window table, transition
  table.
- Scope, data, and definitions: full-frame cohort, direct-frame filter,
  permutation and effective-gate definitions.
- Methodology: exact coordinate matching and gate verification.
- Limitations and robustness: biological identity remains unverified; male-male
  swaps and interpolation are outside the evidence.
- Recommended next steps: prepared GUI launcher and isolated review ledger.
- Further questions: biological validity, top-camera correction, and improved
  motion/image gates.

## Chart map and contract

- Segment: branch timing.
- Analytical question: when do verified 6-only gate exclusions occur, how deep
  are they inside the 6-to-8 mm interval, and where is female evidence missing?
- Takeaway: 26 transitions cluster into 12 windows; four windows have no
  confirmed female annotation.
- Family/type: relationship, native scatter.
- Grain and sufficiency: 26 branch transitions across the full source timeline.
- Fields: trigger time in minutes (x), normalized gate-interval fraction (y),
  annotation-context status (color), branch/frame/distance/state-span tooltips.
- Palette: hard two-root cap, blue and orange; color is reinforced by legend
  text and exact tables.
- Delivery: canonical portable HTML report, full-width chart.
- QA: packaged reader plus semantic chart table; desktop/narrow verifier when
  compatible Chromium is available.

## Sources and transformations

- `find_gate_branches.py` is the reproducible transformation.
- `gate_branch_transitions.csv` contains one row per verified permutation
  change.
- `gate_branch_trigger_details.csv` contains one row per affected ID.
- `gate_branch_review_windows.csv` groups adjacent trigger frames using a
  1,000-frame maximum inter-trigger gap and adds +/-166-frame context.
- `female_annotator_candidates.csv` is validated through the production
  candidate normalizer.

The report omits a second chart because exact frame lookup is the primary task;
the two tables are more honest than another aggregate visual for the 12 windows
and 26 transitions.
