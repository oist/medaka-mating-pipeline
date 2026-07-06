# report/20260630 — Medaka mate-choice POSTER kit

Storyboard + figure library + PowerPoint catalogue for a conference poster covering the
whole project: background, goal, experimental procedure, the ML tracking method, and the
biology results (N = 3), with BORIS ethogram cross-validation.

## What to open first

| File | What it is |
|------|------------|
| **`STORYBOARD.md`** | The poster narrative — 5 sections, every panel mapped to a figure with its key message. Start here. |
| **`medaka_poster_figures_20260630.pptx`** | **Figure catalogue** — one slide per candidate figure (45), large, with caption + source. Browse and pick. |
| **`draft_poster_A0_landscape_20260630.pptx`** | One recommended A0-landscape arrangement of the lead figures. A starting point to rearrange. |
| **`boris_cohort/medaka_boris_cohort_figures_20260630.pptx`** | Supplemental BORIS cohort figure deck from the 23 provided manual ethogram CSVs. |
| `boris_cohort/data/` | Parsed BORIS event table, experiment summary, behavior summary, source manifest, and figure manifest. |
| `slide_previews/` | PNG of every catalogue slide (quick browsing without opening PowerPoint). |
| `poster_preview/` | PNG of the draft poster. |

## Figures

- `figures/` — **26 generated** plots/schematics, built from the real per-frame CSVs / track CSVs / metrics JSON.
- `reused/` — **19 real pipeline outputs** copied from earlier analysis (calibration QC, pose/seg, embeddings,
  gravid-belly montage, BORIS plots, original mate-choice panels).
- `boris_cohort/` — **9-slide editable PowerPoint supplement** plus normalized CSV/JSON tables from all
  23 user-provided BORIS files (926 parsed state/point events; 15 spawn-coded observations).

Naming: `s1_*` background · `s2_*` methods · `s3_*` ML tracking · `s4_*` mate choice · `s5_*` BORIS.

## Regenerate (all re-runnable)

```powershell
.venv\Scripts\python.exe report\20260630\build_figures.py      # 26 figures -> figures/
.venv\Scripts\python.exe report\20260630\build_catalog.py      # copy reused + build both .pptx
.venv\Scripts\python.exe report\20260630\boris_cohort\build_boris_cohort_data.py
node report\20260630\boris_cohort\build_boris_cohort_deck.mjs
powershell -ExecutionPolicy Bypass -File report\20260630\render_previews.ps1   # PNG previews (PowerPoint COM)
```

## Data sources (all real, on disk)

- K_1_2 track + analysis: `runs/dual_line/lineB_fulltimeline/{step2,analysis}/`
- K_1_1 / K_1_3 track + analysis: `runs/dual_line/extend_k11_k13/{K_1_1,K_1_3}/...`
- BORIS comparison: `report/20260622/behavior_comparison/`
- BORIS cohort supplement: the 23 CSVs in the Drive-backed `boris_events` folder listed in the request.
- N = 3 summary: `runs/dual_line/extend_k11_k13/FINDINGS_N3_mate_choice.md`

## Notes / honesty

- M1 = the preferred male in every trial (appearance label, arbitrary per experiment, globally consistent within a trial).
- N = 3 ML validation leads with **spawn-landing** (female ≤ a few mm from a male at the BORIS spawn) — the strongest
  ground truth — plus the K_1_2 swap-correction reversal. The earlier C0 male-male AUCs (0.948 / 0.990) are *not* shown
  as separability: they were the tell of a female-mislabel bug that was caught and fixed per experiment.
- Assumed poster format: A0 landscape, 4 columns. Easy to switch to portrait / different size.
