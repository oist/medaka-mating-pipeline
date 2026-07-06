# Step 6 (Goal 5) — gravid-belly GT spot-check

The dual-line female call was validated only by appearance-internal consistency (per-frame vs per-window
0.99); its biological grounding came solely from the K_1_2 training anchor. This is the external check:
render the called F / M1 / M2 at well-separated high-confidence frames across the timeline and look for
the gravid-belly morphology by eye. Script `step6_gt_montage.py` (tight mask-bbox side1 crops);
montage `step2/gt_montage.png` (8 frames × F|M1|M2, + spawn frame 135888).

## Observation
- **The called F consistently shows the deeper-bodied / rounder-abdomen (gravid) morphology** across the
  timeline (f67799, f142060, f171107, f201144, f237164, f277889), distinct from the slender, elongated
  M1/M2. The female occupies DIFFERENT geometry slots across these rows (the cross-window swap map), yet
  the called F is the rounder fish every time → the appearance line tracks the **biological gravid
  female**, not a geometry slot. At the spawn (f135888, BORIS 818.6 s) the called female is present while
  the males are occluded (expected at that contact moment).
- **M1 vs M2 are NOT visually distinguishable** here — both are same-sized slender males. Expected: male-
  male is the harder problem (LOSO AUC 0.91 vs female 0.98), relying on subtle appearance cues, not size.
  Visual GT can corroborate the FEMALE anchor but cannot easily confirm M1≠M2 by eye (hence the
  model + the embedder global-anchor check carry that claim).

## Conclusion
External corroboration that appearance-"female" = the deeper-bodied gravid fish, consistent across the
whole timeline and across her slot swaps. This grounds the 0.99 internal validation in biology for the
FEMALE. The two males remain distinguished by the (validated, globally-consistent) appearance model
rather than by eye. Final biological confirmation of the gravid-belly call is the user's to make from the
montage; this is a qualitative spot-check, not a quantitative GT label set.
