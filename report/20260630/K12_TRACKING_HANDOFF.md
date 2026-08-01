# K_1_2 Identity-Tracking — New-Session Handoff Prompt

> Paste the section below into a fresh session. It is written so the session can
> decompose the work with the **agent team / Workflow** system (many independent
> aspects). Everything above the line is context; the "MISSION" block is the prompt.

---

## MISSION (paste me)

We need a **clean, full-length per-fish 3D identity track for experiment `20251225_K_1_2_FCMC`**
(348,600 frames @166 fps; 1 female + 2 males, all the same size). The end goal is mate-choice
analysis: *which male is the female near during courtship/spawning*. A previous session proved
that **pure-geometry identity correction has hit a hard ceiling** — use an **agent team** to
attack the remaining aspects in parallel, because they're largely independent.

**Read first:** `MEMORY.md`, the memories `project_mate_choice_k12`, `project_female_anchor_pipeline`,
`project_dual_line_phase0`, `reference_concrete_tracklets_3d_bind`, `reference_id_quality_check`,
and this whole file. Then plan the agent team before touching code.

### What is trustworthy (the BACKBONE)
- `Z:/StephensU/medaka/runs/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d.csv`
  — 3 continuous 3D slots (`fish_id` = ID_1/ID_2/ID_3), **full coverage** (~344–348k frames each),
  **0 speed spikes**, smooth. This is the reconstruction + 3D Hungarian linker output.
- It is **geometrically correct** (positions good) and **BORIS-validated at the spawn**
  (frame 135888 / 818.6 s: female in contact with a male). The ONLY problem is that **slot↔physical-fish
  identity can swap at close-approach crossings.**

### The core finding (do NOT re-derive — build on it)
Three identical fish in **dense contact for seconds** cannot be told apart by any *geometric*
signal (3D position, mouth position, heading, cross-view continuity). Quantified this session:
- 872 close-approach events (<18 mm). An independent multi-view mouth tracker, seeded clean
  from the backbone at each event, classified them: **263 no-swap, 73 swap, 536 UNCERTAIN (61%)**.
- "Uncertain" = the tracker *lost a fish* mid-window — i.e. genuinely unresolvable, concentrated
  on the long dense-contact events.
- Applying only the 73 confident swaps is **unsafe**: swap-composition needs every crossing caught,
  so partial correction makes each slot flip identity ~50× over the timeline (chaotic). The spawn
  is a 3-fish bunch (all pairs 6.7–9.1 mm) so it can't disambiguate either.

**Conclusion: geometry is exhausted. The remaining identity must come from APPEARANCE.**

### The bet that should work (appearance path)
- **Detectron2 female-anchor** (already validated in your notes: **female-vs-male AUC 0.982**,
  `33._square_crop` crops — NOT `crop_from_mask`). Pin the *female* identity through crossings.
  The female is the single identity that matters most for mate choice; once she's pinned, "which
  male is she near" is answerable even if M1/M2 remain swapped.
- **Round-6 SLEAP** (close-contact annotation seed already built) → better detections in exactly
  the dense scenes where geometry fails.
- Existing infra: `scripts/hpc/.../40_female_anchor_pipeline.py`, `reproject_slot_points.py`,
  the K_1_2 female model; `resolve_hardcore_events.py`, `bind_tracklets_3d.py`.

### Suggested agent-team decomposition (independent aspects → parallel agents)
1. **Backbone-swap audit** — for each of the 872 events, characterize the geometry (antiparallel?
   true net-crossing vs milling? min-distance, duration, which views overlap). Produce a ranked,
   labeled event table. (Read-only analysis.)
2. **Female-anchor integration** — wire the Detectron2 female model onto K_1_2: reproject the 3
   backbone slots into each view, crop with `33._square_crop`, score female-vs-male, and produce a
   per-frame **female slot assignment** with confidence. This is the highest-value deliverable.
3. **Male–male resolution** — only where the female is pinned and two males remain: decide if
   round-6 SLEAP / appearance can separate them, or mark as unresolvable.
4. **Validation harness** — BORIS ethogram agreement (`scripts/analysis/compare_boris_trajectory.py`),
   spawn-contact check, and eye-review renders at crossings (`render_qc_4panel_fast.py`,
   `render_crossing_montage.py`). Every proposed fix must pass these.
5. **Synthesis** — merge female-anchor + backbone + any confident male resolution into the final
   `*_identified.csv`; honestly flag the residual-uncertain spans rather than silently filling them.

### What was already TRIED this session (don't repeat the dead ends)
| Approach | File | Result |
|---|---|---|
| 2D FP ingestion filters (collapsed/dup/flip) | `load_2d_slp.py` | ✅ WORKED, **committed 3026930** |
| Entry-phantom fix (2D FP + gap-capped fill) | committed | ✅ committed 3026930 |
| Multi-view 3D **mouth** tracker, 1-confident-view anchor | `scratchpad/track_mv_mouth_fast.py` | ✅ fixes 8042/8809/8852 **in a clean window**; ❌ unstable full-length |
| — v1 tight jump-gate | | ❌ 95% "coast-freeze" (gate blocks re-acquire) |
| — v2 alive-prediction, no frozen output | | ❌ fish die after entry (mutual-ambiguity cascade) |
| Windowed mouth tracker (seed clean mid-timeline) | `scratchpad/track_mv_mouth_win.py` | ✅ reliable ONLY when seeded clean with separated fish |
| Backbone + crossing corrector (872 events) | `scratchpad/backbone_corrector.py` | ⚠️ 263 no-swap / 73 swap / **536 uncertain**; partial correction unsafe → geometry ceiling |

**Lessons that hold:** track by **mouth not centroid** (antiparallel fish have separated mouths →
fixes 8042); **whichever view is unambiguous should anchor identity**, not just "top" (fixes 8809);
a fish gapped in one view stays alive via the others (fixes 8852); but **from-scratch 3-fish mouth
tracking diverges under mutual ambiguity** — it only works seeded-clean over short windows.

### Key paths / facts
- Calibration: `calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models`;
  cam0=top(XY), cam1=side1(XZ), cam2=side2(YZ). Top view less trustworthy (water ~1 cm low) → prefer
  side-driven triangulation (`complete_triplet`). Key fns: `compute_XYZ_from_imageCoord_triplet`,
  `complete_triplet`, `compute_imCoords_from_XYZ` (batched reproject, one call/frame).
- Staged `.slp` (2-node mouth+tail, round4_260520 skeleton): `Z:/StephensU/medaka/runs/udmt_validation/qc/_stage/20251225_K_1_2_FCMC/{top_cam,side1_cam,side2_cam}.slp`
- Cached mouths (fast load, no slp reload): `Z:/StephensU/medaka/runs/udmt_validation/qc/k12_mouths.npy` (cols: view,frame,x,y)
- Corrector event log + parsed swaps: `Z:/StephensU/medaka/runs/udmt_validation/qc/_corrector.log`, `_swaps.txt`
- Use the project venv: `.venv\Scripts\python.exe`. ffmpeg at `C:/ffmpeg/ffmpeg-8.0.1-full_build-shared/bin/ffmpeg.exe`.
- 3 fish are **same size** → no geometric auto-ID; verify swaps **by eye at crossings**; `final_fish_id`
  are arbitrary ints (color by sorted index).

### First moves (cheap, do before any big run)
1. Emit the **event worklist CSV** (aspect 1) — ranked by duration × suspicion — so annotation/appearance
   effort is aimed, not blind.
2. Render the **BACKBONE** (not the chaotic corrected track) through frames 8042 / 8809 / 8852 and a
   few long events so a human can confirm where it actually swaps.
3. Stand up **female-anchor on K_1_2** (aspect 2) and measure female-pin coverage/confidence over the
   872 events. If the female pins reliably through dense contact, that's the win — pivot all effort there.
