# Line B step 1 at full-timeline scale (K_1_2)

Scaled the production Line-B identity build across the **whole K_1_2 timeline** from the saion
Detectron2 masks: one ~3000-frame window sampled from each of the 8 mask chunks (centered),
`33`→`34`→`44` (`window_align` → single global female id). Driver: `run_step1.py`.

## Throughput notes (two bottlenecks found + fixed)
- Video over `Z:` (network) decodes ~4 fps → **copy videos to local disk** (~100 fps). *Not* the
  main cost, though.
- The real cost: `33` gzip-decompresses the WHOLE 43,575-frame mask chunk per window. **Pre-slice**
  each chunk to the window's ~3000 frames with an h5py object-copy (compressed, no decompress) →
  `33` dropped **772 s → ~95 s/view** (crops byte-identical). (Worth porting into `33`/`mask_io`
  as a `frame_range` filter so future per-chunk runs don't over-read.)

## Result
- 8/8 windows built → pooled **137,366 crops (125,154 clean), global_id purity 1.0**.
- **Global female call (`44`): aligned id 0, P(female)=0.93** (vs 0.049 / 0.067), **margin 0.864,
  not ambiguous, consistent across all 6 kept windows.** Reference window win_238162.
- `window_align` **dropped 2 windows** (degenerate=false overall): win_20287 (early/inactive start)
  and **win_63862** — the courtship-bout window where the female's geometry slot swaps (gid0↔gid2,
  see `../lineB_w64000/`), so its per-fish centroids don't separate. Correct, expected behavior.
- **Per-window female raw 3D-slot (the swap map): `{107k:0, 151k:1, 194k:2, 238k:0, 281k:2, 325k:2}`**
  — the female occupies ALL THREE geometry slots across the timeline = the 3D linker's cross-window
  permutation, now mapped from appearance.
- **Whole-track sexed CSV** (`female_anchor/round5_260611_3d_identified_sexed.csv`): 348,600 frames
  × 3 fish, **exactly 1 female/frame (0 violations)**; sex_confidence high 54k / med 505k / low 487k.

## Caveat / next (step 2)
The female *call* is confident (margin 0.864), but propagating it across the whole track from 8
sparse window anchors leaves **~46% of fish-frames `low`-confidence** — the spans *between* windows
where the female's slot disagrees (a 3D-bind swap happened somewhere in the gap). **Step 2** resolves
these: run the female model **per-frame** on the masks (not just window anchors) to pin the exact
swap frames and convert `low` → resolved sex — the appearance line directly correcting Line A's
swaps. The within-window per-frame separation is strong (0.911-0.912, see Phase-1b / lineB_w64000),
so the signal to do this exists.

Artifacts: `runs/dual_line/lineB_fulltimeline/{run_step1.py, win_*/, pooled/, female_anchor/}`.
Masks `/bucket/.../detectron2_output/20251225_K_1_2_FCMC/` (= `Z:\...\detectron2_output\...`).
