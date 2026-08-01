# Self-supervised appearance-ID pilot — findings

**Question:** can we auto-train an appearance ID model from Detectron2 segmentations
+ SLEAP concrete tracklets, with no manual labels?

**Answer: yes, the loop works — and it tells us something biologically specific.**

## Pipeline (all reproducible, no manual labels)

`13_run_detectron2_segmentation` (dense, window 176300–181300, 3 views)
→ `33_build_segid_dataset` (link masks→concrete tracklets by point-in-mask; drop
reflections=0-points and merged masks=≥2-points; area-spike QC)
→ `34_assemble_segid_training` (pool views; must-link = same view-tracklet,
cannot-link = same-frame same-view; 3D `global_id` kept only as a HELD-OUT label)
→ `train_embedding_model` (20 epochs) → `35_eval_segid_embedding`.

Data: 15,339 clean single-fish crops over 66 view-tracklets; 20,861 positive /
14,798 negative pairs. **`cannot_link_satisfied = 0.99`** — the self-supervision is
internally consistent with zero identity labels.

## Result (honest, per-view — pooling across views adds a *view* confound)

| view | silhouette | crop ARI | crop acc (remap) | tracklet acc (remap) |
|------|-----------|----------|------------------|----------------------|
| top  | **0.43**  | 0.57     | 0.79             | **0.81** |
| side1| 0.32      | 0.48     | 0.71             | 0.83 |
| side2| 0.31      | 0.50     | 0.67             | 0.67 |

Comparable to the prior pseudo-label model (~0.72 crop / ~0.77 tracklet) — but here
with cleaner self-supervision and a tiny 3-layer CNN.

## The key finding (PCA `eval_all/per_view_pca.png`)

In **all three views** the embedding cleanly separates **one fish (global_id 1)** onto
its own arc while the **other two (0 and 2) collapse together**. The crop montage
(`eval_all/fish_identity_crops_top.png`) shows fish 1 as subtly fuller-bodied — the
signature of a **gravid female**. So:

- appearance **can** isolate the one distinct fish (likely the female — the
  scientifically critical individual for mate choice),
- appearance **cannot** separate the two look-alike males from each other
  (silhouette 0.31–0.43 ≪ the 0.91 needed for reliable per-frame ID).

This refines the earlier flat "appearance is moderate (0.73)": it is **good for 1 of 3,
poor for the other 2**, which is exactly the 1-female-2-similar-males biology.

## What this means for identity assignment

- **Don't trash appearance** — but use it for what it can do: a per-tracklet
  **female anchor**. That anchor is precisely the global "which cluster is which fish"
  signal geometry lacks. Female-vs-not is the easy, reliable call.
- **Geometry/3D must still handle male–male crossings** — appearance won't.
- The natural design is therefore **hybrid**: appearance pins the female globally,
  3D geometry + the motion resolver carry the two males through crossings.

## Why it didn't break the ceiling, and the levers that might

The strict same-frame negatives reproduced (did not exceed) the prior accuracy, so the
ceiling is the **appearance signal + model capacity**, not negative quality. Levers,
in expected-payoff order:
1. **Bigger backbone** — the current net is a 244 KB 3-layer CNN; idtracker used
   ResNet18. Most likely single improvement.
2. **3D cross-view positives** — add same-`global_id` cross-view pairs so the
   embedding is globally consistent (kills the pooled view-confound; enables one
   global female anchor across all views).
3. **More data** — more dense Detectron2 inference over more windows (cheap; the
   loop is built and runs unattended).
4. **A small human GT set** — label the female at ~20–40 crossings to *measure*
   female-anchor precision directly, instead of leaning on the 3D `global_id` proxy.

Artifacts: `Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/` (crops, pooled_all, model_all, eval_all).
Pilot runner: `run_segid_chain.sh`.

---

## Male investigation — ResNet18 + 5× data (2026-06-21)

Adopted an ImageNet-pretrained **ResNet18** (11.7M params, `resnet_embedder.py`) and
ran **4 more dense side windows** (`run_more_side_inference.sh`) → 33k crops / 171
tracklets / 21k male crops over 6 side windows. Two-stage: female filter → male-only
model. Measured with **tracklet-grouped CV** (no crop leakage). The result was a
two-trap minefield; both traps caught:

**Trap 1 — pooled male–male ≈ chance (`balacc 0.51–0.59`, KMeans `ARI ≈ 0`).**
Would read as "appearance can't separate males." **False.**

**Trap 2 — per-window male–male is HIGH** (`balacc` 1.0 / 1.0 / 0.95 / 0.43 across the
4 windows). Would read as "appearance separates males locally." **Also not trustworthy.**

**Why both are wrong — the position confound.** The concrete-tracklet gate removes
close-contact frames, so male crops are *always* spatially separated (inter-male
distance min 117px, median 386px, **0% < 100px**). A within-window classifier can
therefore cheat on *location* (male-left vs male-right) instead of fish appearance —
and we have **no close-contact male crops to control for it**. The relevant regime
(males close, at crossings) is exactly the regime the data excludes — and exactly
where male ID must be resolved.

**The clean contrast that IS trustworthy:** the **female** transfers *across* windows
(`balacc 0.83`; ~1.0 within-window) — i.e. a position-robust, time-robust appearance
signal. The **males** do **not** transfer across windows (chance) and have no
position-controlled test. So:

- **Female appearance = a real, usable anchor.** (Local ~1.0; global 0.83 with
  gravid-state drift → apply locally and re-anchor over time.) Directly serves the
  science — the female is the chooser.
- **Male appearance = unvalidated, and unvalidatable with tracklet-gated data.** Not
  demonstrably better than position/geometry. Do not productionise it.

**Recommendation (avoids both wrong directions):**
1. Ship the **female anchor** as a global "which fish is the female" filter.
2. Resolve the **two males with 3D geometry + the motion resolver** (continuity
   through breaks), not appearance.
3. Only pursue male appearance if male1-vs-male2 is scientifically required, and only
   via the experiment that actually tests the right regime: **un-gated crops at
   close-contact / crossing frames** (Detectron2 already segments them — just skip the
   separation gate) **plus a small human male-ID label set**. idtracker sidesteps this
   exact trap with a dedicated crossing-frame path; our gate threw those frames away.

New code: `resnet_embedder.py`, `train_resnet.py`, `36_build_male_dataset.py`,
`eval_male_separability.py`; runners `run_more_side_inference.sh`,
`run_male_pipeline_chain.sh`, `rerun_male_train.sh`. Reports in
`Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/eval_male/`.

---

## Female anchor — wired as a production filter (2026-06-21)

The validated capability is now a pipeline filter that tags each 3D fish identity
**F/M** (males not distinguished — that's geometry+motion's job).

**Modules/scripts:** `id_model/female_anchor.py` (pure aggregation + exactly-one-female
assignment with confidence margin / ambiguity flag; tested), `37_train_female_classifier.py`
(ResNet18 + 2-class head, female-vs-male, tracklet-grouped val), `38_apply_female_anchor.py`
(per-crop P(female) → aggregate per 3D slot → label argmax female → write sexed CSV).

**Per-experiment flow:**
```
Detectron2 side inference (13) → link masks→tracklets→crops (33) → pool (34)
→ train female classifier (37, once) → apply (38) → <tag>_3d_identified_sexed.csv
```

**K_1_2 result:** held-out classifier `crop balacc 0.72 / tracklet 0.76` (per-crop is
modest due to gravid-state drift), but **per-fish aggregation is decisive**: female
slot mean P(female) **0.87** vs males **0.08 / 0.02**, confidence margin **0.79**,
not ambiguous. Female = `ID_2` (`final_fish_id 15`). Per-tracklet 91% female / 97%
male correctly sided (`female_prob_by_tracklet.png` shows clean bimodal separation).
Output `<...>_3d_identified_sexed.csv` adds `sex` + `female_prob` columns keyed by
geometry `fish_id`, so mate-choice analysis can pull out the female (the chooser).

**Caveats / next:** the classifier is K_1_2-trained; for other experiments retrain
(37) — or identify the female slot self-supervised as the cleanly-separable cluster
(no labels needed). Per-tracklet tagging is ~0.76 held-out; trust the per-FISH call.
Artifacts in `Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/female_anchor/`.

---

## Method upgrade after classifier pilot (2026-06-21)

The next classifier run should change the data and training protocol before asking
for more GPU time:

1. Build crops with `33_build_segid_dataset --mask-background` so the model sees the
   fish body rather than tank/background context. This directly attacks the male
   position confound.
2. Re-assemble with `34_assemble_segid_training`; it now writes `global_id_purity`,
   `global_id_modal_n`, and `global_id_n` for each crop. Supervised/weak-supervised
   runs should keep only high-purity modal 3D labels (default 0.9).
3. Use `39_train_identity_classifier.py` for classifier-style experiments. It trains
   a ResNet18 identity classifier with class/tracklet-balanced sampling, optional
   supervised-contrastive loss, whole-window holdout, per-tracklet aggregation, and a
   position-only baseline.
4. Treat a male-male result as real only if it transfers to a held-out window and
   clearly beats the position-only baseline. Otherwise it is still a layout cue, not
   a biological identity cue.

This still does not replace the need for hard-event labels: the missing male data are
close-contact/crossing frames, not more separated crops.
