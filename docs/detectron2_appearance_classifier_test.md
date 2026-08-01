# Detectron2-segmentation appearance ID classifier — test summary

Experiment: `20251225_K_1_2_FCMC` (round5). Status: **female anchor achieved and wired
into the pipeline; two-male discrimination not achievable with the current data.**
All artifacts in `Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/`; code in `sam2_id_experiment/`.

---

## Background

The recording is a 3-fish mate-choice setup — **1 female + 2 males** — and the two
males are visually near-identical. The 3D geometry pipeline tracks all three fish but
*smooth-swaps* identity at crossings (and the speed-spike QC is blind to it), and prior
single-cue trackers (idtracker.ai, UDMT) degraded to review-rankers on this footage.
What was missing was an **identity cue independent of geometry**.

The hypothesis tested here (proposed by the lab): train an **appearance** ID model
**without any human labels**, using two self-supervised constraints that the existing
pipeline supplies for free:

- **must-link** — two crops in the same "concrete" tracklet (the conservative
  stIFD-gated tracklets that break at any ambiguity) are the same fish;
- **cannot-link** — two segmented instances in the *same frame* are different fish.

Detectron2 instance segmentation provides the per-fish crops; the concrete tracklets
provide the constraints. The goal: recover the 3 identities (or at least the female)
from appearance, to anchor/repair the geometric IDs.

---

## Methods

**1. Segmentation (Detectron2 Mask R-CNN, R50-FPN).** Models already fine-tuned on this
experiment: top-cam 1,001 images / 2,551 instances; side-cams 870 images / 2,454
instances (`max_iter=1000`). Dense (stride-1) inference was run on **6 side-camera
windows** spanning the recording (frames 132k, 145k, 152k, 176k, 285k, 300k) to build
a large crop set; additional inference can be added at will (it only costs time).

**2. Mask → tracklet linking + QC** (`33_build_segid_dataset.py`). Each mask is matched
to the concrete-tracklet detections by point-in-mask: **exactly one** tracklet point
inside → clean single-fish crop; **zero** → reflection / false positive (dropped);
**two or more** → merged mask (dropped). Per-tracklet area-spike QC removes frames whose
mask area jumps. Yield: **33,179 crops** (31,535 clean) over **171 view-tracklets**
(57 per fish-slot). New runs should pass `--mask-background` so the crop contains the
fish body with the surrounding tank/background blacked out; otherwise a male classifier
can learn stable camera-position texture instead of appearance.

**3. Self-supervised contrastive training.** Pairs built with must-link (same tracklet)
and **strict same-frame cannot-link** (`sample_constrained_pairs`, an improvement over
the prior span-overlap negatives, which can create false negatives). Two backbones were
compared: a small 3-layer CNN and an **ImageNet-pretrained ResNet18** (11.7M params).
Evaluation always used **tracklet-grouped cross-validation** (crops from one tracklet
never split across folds — otherwise scores inflate), scored against the held-out 3D
`global_id` as a proxy label, plus permutation-invariant silhouette / ARI.

**3b. Weak-supervised classifier training.** For classifier-style experiments, use
`39_train_identity_classifier.py` instead of only pairwise contrastive training. It
filters weak labels by `global_id_purity` from `34_assemble_segid_training.py`,
balances classes and tracklets during sampling, trains a ResNet18 classifier with an
optional supervised-contrastive term, and reports per-crop, per-tracklet, per-window,
and **position-only baseline** metrics. A male result is only credible if it beats that
position baseline and transfers to a held-out window.

**4. Two-stage design.** (a) Female-vs-male; (b) a **male-only** sub-model (drop the
female so all capacity targets the subtle male axis), plus a **constraint-propagation
2-colouring** of the male cannot-link graph (with only 2 males, co-occurrence is
transitive).

**5. Female-anchor production filter** (`37_train_female_classifier.py`,
`38_apply_female_anchor.py`, `id_model/female_anchor.py`). A ResNet18 + 2-class head,
aggregated per fish to a single F/M call with a confidence margin, writing a sexed
3D-identity CSV.

---

## Results

### Achieved — the female anchor

- **The female is appearance-separable.** Tracklet-grouped CV: female-vs-rest
  **0.997–1.000** within a window, **0.83** across the full 17-minute span (the residual
  drop is **gravid-state drift**, not a methodology artifact — it persisted after fixing
  augmentation). The self-supervision is internally consistent (`cannot_link_satisfied =
  0.99`).
- **Production filter is decisive.** The standalone classifier is only modest per crop
  (held-out balacc 0.72 crop / 0.76 tracklet, again the drift), but **aggregating
  hundreds of crops per fish** makes the per-fish call unambiguous:

  | fish (geometry slot) | mean P(female) | sex |
  |---|---|---|
  | ID_2 (slot 1) | **0.87** | **F** |
  | ID_1 (slot 0) | 0.08 | M |
  | ID_3 (slot 2) | 0.02 | M |

  Confidence margin **0.79** (not ambiguous). Per-tracklet, 91% of female / 97% of male
  tracklets are correctly sided (clean bimodal separation). Output:
  `round5_260611_3d_identified_sexed.csv` (adds `sex`, `female_prob`). The female — the
  *chooser* in mate-choice — can now be pulled out directly downstream.

### Not achieved — distinguishing the two males

The male question was a two-trap minefield; **both misleading readings were caught**:

- **Pooled male-vs-male ≈ chance** (balacc 0.51–0.59, KMeans **ARI ≈ 0**). Read alone →
  "appearance can't separate males."
- **Per-window male-vs-male is high** (1.0 / 1.0 / 0.95 / 0.43 across windows). Read alone
  → "appearance separates males locally."

Both are untrustworthy because of a **position confound**: the concrete-tracklet gate
removes close-contact frames, so male crops are *always* spatially separated (inter-male
distance min 117 px, median 386 px, **0 % closer than 100 px**). A within-window
classifier can therefore exploit *location* (male-left vs male-right) rather than fish
appearance — and there are **no close-contact male crops to control for it**. The regime
that actually needs resolving (males close together, at crossings) is exactly the regime
the data excludes, and the local signal **does not transfer across windows** (pooled
chance). A bigger backbone (ResNet18) + 5× data did **not** change this.

The clean contrast confirms the asymmetry: the **female transfers across windows**
(position- and time-robust) while the **males do not**. Relatedly, the male cannot-link
graph **fragments into isolated co-occurrence pairs** (largest component = 2) because
tracklets break synchronously at crossings — so constraint propagation alone cannot
globally 2-colour the males either. The two-male identity reduces to *resolving each
crossing*, identical to the geometry hard core.

**De-confounded confirmation (background-masked crops + window holdout + position
baseline).** Re-running with `--mask-background` (fish body only, tank blacked out),
holding out a whole window (`win_300000_305000`), and a position-only baseline
(`39_train_identity_classifier.py`) removed every spatial cue and settled it:

| male1-vs-male2 (held-out window) | crop balacc | tracklet balacc |
|---|---|---|
| CNN on masked crops | **0.235** | 0.241 |
| position-only baseline | 0.303 | 0.310 |

The CNN does **not** beat the position baseline, and **both are below chance (0.5)** —
masking removed the layout cue and the apparent male signal vanished with it. The
below-chance score also fingerprints the geometry `global_id` males as *flipped*
between the training and holdout windows (geometry itself swapped them in between), so
the males are unresolvable by appearance *and* their geometry labels are globally
inconsistent. Masked embedding eval agrees: male-male balacc 0.39–0.46, ARI ≈ 0.

**Conclusion:** appearance gives a real, usable **female anchor** but — now confirmed
under masking + window holdout + a position baseline — provides **no transferable
signal to distinguish the two males**. (Masking also drops the female per-crop score
0.83 → 0.70, i.e. some female score was context; but she still transfers far above
the males, and the per-fish anchor aggregates to a decisive call regardless.)

---

## Next Steps

1. **Use the female anchor in production** (done): apply per fish (decisive), re-anchor
   over time for drift, and feed `sex` into mate-choice analysis. For a new experiment,
   retrain the classifier (`37`) or identify the female slot label-free as the cleanly
   separable cluster.
2. **Resolve the two males with 3D geometry + the motion resolver**, not appearance — the
   crossing-resolution problem, not a classification problem.
3. **Only if male1-vs-male2 is scientifically required**, test it in the *right* regime:
   generate **un-gated crops at close-contact / crossing frames** (Detectron2 already
   segments these — simply skip the tracklet separation gate) **plus a small set of
   human male-ID labels** at crossings. This is the only way to separate real appearance
   from a position artifact. (idtracker.ai sidesteps this trap with a dedicated
   crossing-frame CNN path; our gate discarded those frames.)
4. **Cross-experiment validation** of the female classifier (different fish / gravid
   states) before reusing it beyond `K_1_2`.

### Updated training recipe

```bash
python -m sam2_id_experiment.scripts.33_build_segid_dataset \
  --view side1_cam --masks <detectron2_masks.h5> \
  --tracklet-bound-csv <side1_cam.tracklet_bound.csv> \
  --video <cam1_side.mp4> --window <start> <end> \
  --mask-background --out-dir <segid_window_dir>

python -m sam2_id_experiment.scripts.34_assemble_segid_training \
  --segid-dir <segid_window_dir> ... \
  --views side1_cam side2_cam \
  --tracklet-bound-dir <tracklet_bound_dir> \
  --out-dir <pooled_dataset>

python -m sam2_id_experiment.scripts.39_train_identity_classifier \
  --dataset <pooled_dataset> --mode male --female-gid 1 \
  --split window --out-dir <male_classifier_dir>
```

Read `identity_classifier_report.json` first. If the CNN score is close to the
`position_only_baseline`, the model is still using layout cues rather than fish identity.

### Lessons
- Aggregating a weak per-crop signal (0.72) over a whole fish yields a strong per-fish
  call (0.87, margin 0.79) — match the evaluation grain to the use.
- Tracklet-grouped CV + a per-window breakdown + a position control each caught a
  different way to reach a wrong conclusion. The honest answer was only visible after all
  three.
