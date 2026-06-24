# Dual-Line Tracking — Phase 0 Findings (K_1_2, existing artifacts only)

Cheap go/no-go gate for **Line B** (Detectron2 masks → 2D tracklets → appearance ID),
run entirely on artifacts already on disk. No new Detectron2 inference, no saion.
Plan: [docs/dual_line_tracking_plan.md](../../../docs/dual_line_tracking_plan.md).

Each sub-step appends a section below (CSV + PNG live alongside this file). Models:
clean female classifier `model_female_clean/model.pt` (val balacc 0.994) + self-supervised
embedder `model_resnet_full/model.pt`. Data: `segid/K_1_2/pooled_sides_full` (33,179 side
crops, K_1_2, 4 windows). The female classifier trained ONLY on side1 clean+covisible crops
from windows {145000,152000,285000} ⇒ **side2_cam + win_176300 + all rejected/non-covisible
crops are genuinely held out**.

---

## Sub-step 1 — Hard-crop ID diagnostic ✅ (appearance survives)

**Question:** does the appearance signal survive on the crops we *excluded* from training
(`qc_status==reject_area_spike` merges + non-co-visible/occluded frames), or collapse to
coin-flip + flicker? Survives ⇒ Line B can carry crossings; collapses ⇒ Line B only fills
clean frames.

**Method (no GT):** frozen clean female model + embedder over all 33,179 crops. Three
self-consistent signals: (1) **margin** = |2·P(female)−1| (decisiveness), (2) **within-tracklet
sex agreement** — each tracklet is one fish, does each crop's sex call match the tracklet's
reference call (modal over its *clean* covisible crops), (3) **embedding coherence** =
cosine(crop, tracklet clean centroid). Artifacts: `s1_hardcrop_per_crop.csv`,
`s1_hardcrop_summary.csv`, `s1_hardcrop_id_diagnostic.png`.

**Result — hard crops stay confident AND self-consistent:**

| view | category | n | median margin | frac confident (≥0.5) | tracklet sex agreement |
|---|---|---:|---:|---:|---:|
| **side2 (held-out, gold)** | clean_covisible | 4383 | 0.913 | 0.999 | 0.996 |
| **side2 (held-out, gold)** | non_covisible | 108 | 0.913 | 1.000 | 1.000 |
| **side2 (held-out, gold)** | **area_spike** | 141 | 0.913 | **1.000** | **0.922** |
| side1 | clean_covisible (leakage) | 26813 | 0.913 | 0.999 | 0.996 |
| side1 | area_spike | 1503 | 0.913 | 0.986 | 0.957 |
| BOTH | area_spike | 1644 | 0.913 | 0.987 | **0.954** |

- On the **held-out hard crops** (side2 area_spike), the model is still **100% confident**
  and **92% self-consistent** with the fish's clean sex. Pooled area-spike agreement 95.4%.
- The residual **4.6% of area-spike crops are *confidently wrong*** (median margin 0.89 among
  the disagreers) — i.e. on a merge/area anomaly the model can confidently misread sex. This
  is the real per-crop failure mode, but it is **absorbed by fragment-vote**: within-tracklet
  agreement is 95–100%, so a tracklet-level majority (what `window_align` already does) is
  correct. Per-frame argmax would flicker; per-tracklet vote does not.
- Disagreement is **spread across windows**, not concentrated in the degenerate
  win_176300 (its area_spike agreement 0.95, clean 0.992). Confirms the old "bad apple" was a
  *cross-window embedder-alignment* problem, **not** a weak classifier — the per-crop sex call
  is healthy even inside win_176300.
- Embedding coherence **saturates at 1.000** across all categories (embedder contrastive loss
  ~0.001 ⇒ within-tracklet crops are near-identical in embedding even when hard). Uninformative
  as a discriminator here; the discriminating signal is the sex-call agreement above.

**Caveat (bounds what this proves):** the "hard" crops here are *area-spike single crops* and
*frames where a fish was missing* that nonetheless produced a clean single-instance crop. The
catastrophic case — two fish fully merged into one mask, or all views occluded — produces **no
crop at all**, so it is not measurable in this dataset. Whether a usable crop even *exists* in
the dense/crossing scenes is exactly what Sub-step 2 (coverage) and Sub-step 3 (overlap at the
3D hard-core) probe.

**Sub-step 1 verdict:** ✅ PASS. Where a crop exists, appearance is decisive (margin ~0.91)
and ~95–100% self-consistent even on excluded hard crops. Line B's identity signal does **not**
collapse on the hard regime it can see — its ceiling is set by *mask availability*, not
appearance confidence. → coverage is the next question.

---

## Sub-step 2 — Detectron2 mask coverage audit ⚠️ (good in separated scenes; contact scenes unmeasurable)

**Question:** per view, fraction of frames with 3 clean separated masks vs merged/<3,
stratified by scene (3D inter-fish distance). Bounds how much timeline Line B can *see*.

**Source:** seg-ID crop_metadata (post-tracklet-linking Detectron2 masks). Per (view,frame),
distinct clean tracklets present = clean separated masks. Scene = 3D min pairwise distance
(view-independent, swap-robust). Artifacts: `s2_coverage_summary.csv`, `s2_coverage_by_scene.csv`,
`s2_multiview_joint_win176300.csv`, `s2_detectron2_coverage_audit.png`.

**Results:**

| view / window | present/span | ≥3 clean / present | ≥3 clean / span |
|---|---:|---:|---:|
| top_cam / 176300 | 0.38 | **0.975** | 0.371 |
| side1 / 145000 | 0.56 | 0.898 | 0.500 |
| side1 / 152000 | 0.50 | 0.898 | 0.451 |
| side1 / 176300 | 0.45 | 0.757 | 0.340 |
| side1 / 285000 | 0.52 | 0.876 | 0.451 |
| side2 / 176300 | 0.34 | 0.891 | 0.300 |

**Multiview joint** (win_176300, all 3 views, 2263 union frames): **≥1 view 3-clean 97.8%**,
≥2 views 74.6%, all-3-views 30.3%, **0 views (hard) only 2.2%** — matches the prior
`multiview_clean_coverage` ~4% hard core. So **when masks exist, ≥1 view separates all 3 fish
~98% of the time** → strong multiview redundancy, the basis for Line-B multiview fusion.

**Two critical caveats (both point at Phase 1):**
1. **Purity-filter denominator.** This is seg-ID *purity-filtered* coverage, not raw Detectron2.
   The pipeline drops ~65% of raw masks (`fp_no_tracklet`) to keep ID-training crops pure ⇒
   "present/span" is only 0.34–0.56. This is a **lower bound** on Line B's ceiling — a
   tracking-oriented linker (Phase 2) keeps far more. The high "≥3 clean / present" (0.76–0.98)
   is the operative signal: *when the pipeline keeps masks, they're cleanly separated.*
2. **Contact scenes are EMPTY.** Every present (clean-mask) frame has 3D distance **>20 mm** —
   there are zero clean-mask frames below 20 mm (the area-spike rejection removes near-contact
   crops by design). So **this dataset cannot measure whether 2D masks separate at the 3D
   hard-core (<12 mm)** — exactly the make-or-break question. That measurement requires raw
   full-length Detectron2 (Phase 1).

**Sub-step 2 verdict:** ⚠️ PARTIAL. Coverage is excellent in separated scenes (≥1 view 3-clean
97.8%, hard core 2.2%), confirming multiview redundancy. But the decisive contact-scene
separability is **unmeasurable from purity-filtered data** — it is the primary thing Phase 1 must
quantify. Not a fail; an explicit hand-off.

---

## Sub-step 3 — Failure-mode overlap (the decisive metric) ✅ structurally / ⚠️ direct-measure deferred

**Question:** at 3D hard-core events (924 events, `hardcore_events_resolved.csv`), is the
appearance line confident (complementary) or also uncertain (coincident)? Output the 2×2.
**3D axis** = resolver verdict `clear` (confident) vs `review` (ambiguous/suspect_swap/no_3d =
uncertain). **Appearance axis** = clean female model (K_1_2 female = **ID_2 / global_id 1**,
margin 0.79). Artifacts: `s3_inwindow_events.csv`, `s3_failure_overlap.json`, `s3_failure_overlap_2x2.png`.

**(B) STRUCTURAL 2×2 over ALL 924 events — the robust complementary-mass number:**

| | F-involved (female model applies) | M-M (ID_1\|ID_3, needs male model) |
|---|---:|---:|
| **3D clear** (513) | 356 | 157 |
| **3D review** (411, UNCERTAIN) | **249** | 162 |

- **44.5%** of hard-core events (411/924) are **3D-uncertain**.
- **60.6%** of the uncertain events (249) **involve the female** → the confident female model
  applies → **26.9% of ALL hard-core events are directly appearance-addressable** by the female
  model alone (upper bound; Sub-step 1 says the read is confident there). This is the
  complementary mass: meaningful and large.
- The remaining 162 uncertain events (17.5% of all) are **male-male** → need the male-vs-male
  model (exists, LOWO 0.90; weaker than female 0.99) → Phase 2.

**(A) MEASURED 2×2 on the 17 in-window events** (crops only exist in 4 windows):

| | appearance-confident | appearance-uncertain |
|---|---:|---:|
| 3D-confident (14) | 4 | 10 |
| 3D-uncertain (3) | **0** | 3 |

- The decisive cell (3D-uncertain × appearance-confident) is **0 — purely a sampling artifact**:
  all 3 in-window review events are **M-M** (female model can't separate two males), and crops
  exist in only 4 windows. The 4 appearance-confident cells are exactly the 4 in-window
  F-involved events (all 3D-clear). This 17-event sample **cannot observe** the 249 F-involved
  review events that the structural table shows exist (they fall outside the crop-windows).
- **0/17 events have any clean crop AT the merge frames** (purity filter, <20 mm) ⇒ Line B
  re-IDs in the **bracket** around the event. Bracket female reads are clean and **consistent
  across the gap** (same raw global_id before and after, all 17): e.g. event f152817 (M-M,
  min 6.7 mm) — the female reads **p=0.955 on both sides**, i.e. appearance confidently
  confirms the female was the *bystander* and did not swap. Same at the extreme **suspect_swap
  f179245 (min 1.95 mm)**: female read stable p=0.955 post-event → the suspected swap is
  between the two **males**, which the female model correctly cannot adjudicate but does
  *bound* (rules the female out).
- **Live confirmation of the cross-window permutation:** the raw 3D-bind global_id carrying the
  female is **gid 1 in win_145000/152000/176300 but gid 0 in win_285000** — exactly the
  permutation `window_align` exists to correct. The script derives the female gid per window
  (does not hardcode it); the female *appearance* read (p≈0.955) is window-independent.

**Sub-step 3 verdict:** ✅ structurally PASS (26.9% of hard-core events are female-addressable
where 3D is uncertain, + 17.5% male-model-addressable), ⚠️ the *direct* measurement of the
decisive cell needs full-length crops (Phase 1) — the 4-window sample is too small and
M-M-biased to observe it.

---

## GATE — proceed to Phase 1? ✅ YES (qualified)

**Decision: YES, proceed to Phase 1** (full-length Detectron2 × 3 views on saion), with Phase 1
re-scoped so its first deliverable is the *direct* measurement Phase 0 could not make.

**The numbers behind it:**

| Gate criterion | Result | Verdict |
|---|---|---|
| (1) Appearance survives hard crops | held-out area-spike crops: margin 0.91, 100% confident, 92% within-tracklet self-consistent; pooled 95.4% | ✅ strong |
| (3) Meaningful complementary mass | 44.5% of hard-core events 3D-uncertain; **26.9% of all events 3D-uncertain AND female-addressable**; +17.5% male-model-addressable | ✅ (structural) |
| (2) Detectron2 coverage | ≥1 view 3-clean 97.8% in separated scenes, hard core 2.2%; contact-scene separability unmeasurable (purity filter) | ⚠️ partial |

**Why YES:** the two make-or-break unknowns Phase 0 could test are favorable — appearance does
**not** collapse on hard crops (Sub-step 1), and 3D uncertainty is large (44.5%) and substantially
appearance-addressable (≥26.9% via the female model alone, where Sub-step 1 says it is confident).
The lines are structurally **complementary**, not coincident.

**The qualification (what Phase 0 could NOT settle, now Phase 1's job):** both Sub-step 2 and the
measured Sub-step 3 hit the same wall — the seg-ID **purity filter removes all <20 mm crops**, and
crops exist in only 4 windows (one of them the degenerate win_176300). So the *direct* contact-scene
measurements (does Detectron2 separate masks at the 3D hard-core? does appearance resolve a 3D-review
**F-involved** event?) are unobservable here. They require raw, full-length Detectron2 with a
tracking-oriented (not purity) filter.

**Phase 1 re-scope (recommendation):** before committing full 3-view × 348k-frame inference, run a
**targeted Phase-1a probe** — raw Detectron2 (no purity filter) on the frames of the ~50 largest
3D-review **F-involved** events (a few thousand frames), build bracket crops, and directly fill the
decisive 2×2 cell. If appearance resolves those, green-light the full Phase 1 fan-out. This keeps
the next compute step cheap and makes the full run conditional on the one number Phase 0 couldn't
produce. **This is the saion / new-inference boundary — paused here for the user's call.**

