# Step 4 — production fusion (Goal 3): Line B wired into the pipeline

The session's per-frame resolution (steps 2/2c/3) is the production form of Line B. Goal 3 extracts it
into reusable, tested code and wires it into the production pipeline + the hard-core resolver. No new
modelling, all local; the validated K_1_2 artifacts are reproduced byte-identically.

## What was built
1. **`sam2_id_experiment/src/id_model/per_frame_resolve.py`** — the single source of truth for the
   resolve/fuse logic (NumPy/pandas, torch/cv2/h5py lazy): `forward_fill_slot`, `confident_anchors`,
   `female_slot_from_probs`, `beta_from_probs`, `resolve_female_track`, `resolve_male_track`,
   `top_male_anchors`, `fuse_side_top` (side-primary layered), `assemble_individual_labels`,
   `appearance_swap_across_event`, + the mask+video inference loop `per_frame_class_prob` and
   `load_classifier`/`forward_head`.
2. **`tests/test_per_frame_resolve.py`** — 15 unit tests (synthetic) over the pure functions. All pass;
   full suite 252 pass (the lone collection error is the pre-existing torch/c10.dll quirk in the
   unrelated `test_prroi_pool_pytorch.py`).
3. **Refactor** of `runs/dual_line/lineB_fulltimeline/{step2_resolve, step2_resolve_male, step3_fuse}.py`
   onto the module — re-running reproduces `round5_260611_3d_identified_sexed_resolved.csv`,
   `round5_260611_3d_individual_id.csv`, `_fused.csv`, `male_reassignments.csv`, `appearance_swaps.csv`
   **byte-identical** (md5 unchanged). Proves the extraction is faithful.
4. **`scripts/tracking/resolve_hardcore_events.py --appearance-id-csv`** — independent per-event
   appearance adjudication: does the Line-B individual_id of the event's slot pair stay put (no_swap)
   or exchange (swap) across the event, vs the 3D-motion verdict. On the 924 K_1_2 events:
   appearance **resolves 318** motion-ambiguous events, **agrees on 415**, **CONTRADICTS 45**
   (motion "clear" but appearance "swap" — the highest-priority review set), 146 na.
5. **`sam2_id_experiment/scripts/45_per_frame_resolve.py`** + a gated **stage 5 `--per-frame-resolve`**
   in `40_female_anchor_pipeline.py`. Stage 5 runs the per-frame female + M1/M2 (+ optional `--top-model`
   fusion) over full-length masks and writes `<tag>_3d_individual_id.csv`, superseding the window-anchor
   sexing (which left ~46% low). Smoke on K_1_2 frames 194000–195000 reproduces the validated track at
   **1.0 agreement** (3003/3003 rows). Defaults wired to the K_1_2 clean models (`DEFAULT_MALE_MODEL`).

## Production usage
```
python -m sam2_id_experiment.scripts.40_female_anchor_pipeline <exp> --paths-config configs/paths.local.toml \
    --per-frame-resolve --masks-dir <full-length Detectron2 masks> [--top-model .../model_top_male/model.pt]
```
Requires the saion Detectron2 fan-out (full-length `<view>.masks_<S>-<E>.h5`). Stage 5 is opt-in; without
it `40` behaves exactly as before (window-anchor sexing).

## Note / future optimisation
`45` does a separate full video+mask scan per model (female, male, top) — correct but ~2-3× the I/O of a
single pass. A future `per_frame_class_prob` variant that runs multiple heads on one crop-pass would cut
the full-timeline stage-5 cost. Left as-is (matches the validated session path; correctness over speed).
