# Step 1b — male-vs-male de-risk (K_1_2): GO

Mirror of the female Phase-0/1b path, for the two MALES (the actual mate-choice / aggression
question). Decides whether male identity is per-frame appearance-resolvable before any
whole-timeline run. **Result: GO (qualified — males are weaker than the female, but no collapse
and stable per-frame separation).**

## 1a — train swap-corrected male classifier (`39 --mode male`)
- Dataset: `pooled_sides_corrected` (window_align swap-corrected; female=aligned id 1, so the two
  males = aligned ids **0 and 2**, globally consistent across the 3 kept windows). All `clean_train`,
  purity 1.0, 3 windows × ~2500/class.
- **Window-holdout LOSO** (holdout `win_285000_290000`): crop balacc **0.936**, tracklet balacc
  **1.0**, **AUC 0.994** (P(gid_2): gid_0→0.21, gid_2→0.84). **Position-only baseline 0.477 (chance)**
  → the CNN uses appearance, not spatial position.
- vs the OLD raw-geometry-label male eval (`eval_male`): balacc 0.52–0.62, ARI ~0.0 ("near chance →
  female anchor only"). **The swap-correction fixed male-male exactly as it fixed the female** — the
  old weakness was cross-window label noise, not weak male appearance.
- Deployment model (all 3 windows, tracklet split, best val crop balacc 0.96):
  `C:/udmt_validation/segid/K_1_2/model_male_clean/model.pt` (classes [gid_0, gid_2]; head →
  P(gid_2)=P(M_β)).

## 1b — per-frame probe on UNTRAINED windows (`step1b_male_probe.py`)
Female slot known per frame (step2 `per_frame_female` CSV) → the 2 non-female slots are the males →
run the male model on them. Probe windows 64k / 194k / 325k (none in the training set {145k,152k,285k}).

| window | n_frames | median spread | frac>0.3 | AUC-vs-raw-geom | beta_slot flips |
|--------|----------|---------------|----------|-----------------|-----------------|
| 64000–69000 (courtship bout) | 376 | 0.286 | 0.46 | 0.751 | 5 / 173 conf |
| 194000–199000 | 484 | 0.534 | 0.76 | **0.033** | **0 / 369** |
| 325000–330000 | 483 | 0.542 | 0.82 | 0.555 | **0 / 396** |
| **overall** | **1343** | **0.520** | **0.70** | — | — |

**Reading:**
- **No collapse.** Within-frame male P-spread **0.520** (female ref 0.911; Phase-1a collapse 0.025).
  Males are appearance-separable per frame, ~70% of frames confidently (>0.3), weaker than the female.
- **AUC-vs-raw-geometry is a depressed/inverted metric** (raw `tracklet_bound` male labels are the
  swap-prone Line-A labels). The **0.033** at win 194k = appearance *confidently and stably*
  disagreeing with the geometry male labels = appearance catching a **window-wide geometry male-swap**
  (the same cross-window permutation the female showed). Not a failure — the dual-line value.
- **Temporal stability confirms self-consistency:** win 194k **0 flips / 369 frames**, win 325k
  **0 flips / 396 frames** — the per-frame male call is rock-stable within a window (so the
  cross-window AUC scramble is geometry swaps, not appearance flicker). Win 64k (heavy contact) has
  5 flips / 173 — the hard contact core, expected; still mostly stable (max run 111).

## Verdict
**GO** for the whole-timeline male run. Males are per-frame appearance-resolvable and stable, just
weaker than the female (more uncertainty in contact). Open risk to resolve during the full run:
**global α↔physical-male anchoring** — within each window the call is stable, and LOSO AUC 0.994
shows the boundary generalizes to untrained windows under swap-aligned labels, but the full run must
verify the SAME physical male is M_β across ALL windows (via cross-window embedder linking /
within-tracklet consistency), since M_α/M_β are defined only by the reference window (152k), not a
biological anchor like the female.

Artifacts: `model_male_clean/` (model + reports), `step1b_male_probe.py`,
`step1b_male/{male_probe_per_frame.csv, male_probe_summary.csv}`.
