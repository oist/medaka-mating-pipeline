# UDMT-on-saion bundle (Detectron2-seeded, native XMem→ST-Net)

Run the **real** UDMT tracker on saion (V100), with Detectron2 replacing the SEEM
first-frame init only — the native XMem propagation + ST-Net/TrDiMP tracker stay
intact. Tests UDMT's **segmentation-based tracking**, not appearance classification.
Conceptual background + verified code refs: [`docs/udmt_saion_runbook.md`](../../../docs/udmt_saion_runbook.md).

## ✅ Environment is ALREADY SET UP on saion (2026-06-22)

Built + validated by Claude over SSH (login `saion`). Paths:
- conda env: `~/miniforge3/envs/udmt` (py3.8, **torch 2.1.1+cu121**, nvcc 12.1 + CUDA dev libs, ffmpeg)
  — activate with `source ~/miniforge3/etc/profile.d/conda.sh && conda activate udmt`.
- UDMT clone + checkpoints (trdimp, XMem.pth): `/bucket/StephensU/medaka/udmt/UDMT`
- this bundle (scripts + `init/`): `/bucket/StephensU/medaka/udmt/bundle`
- **PrRoIPooling `.so` pre-warmed + cached**: `~/udmt_torch_ext` (FWD+BWD verified on V100)
- writable run root: `$HOME/udmt_run` (**`/bucket` is read-only on compute nodes**)

Validated on a V100: `torch.cuda` → `(7,0)`; `PrRoIPooling FWD+BWD OK grad True`; and the
**driver's XMem path smoke-passed** (`run_udmt_headless.py --xmem-only` on a 200-frame top
clip → 200 masks written, init injected + merged). NOT yet exercised on saion: the ST-Net
steps 4–6 (need >2000 frames to pass the training-dataset gate) — validate with a >2000-frame
clip before the full 3-view run.

Remaining for a full run: `prep_window_videos.sh` (cut the 3 windows), then `sbatch run_udmt.sbatch`.
Scoring (`compare_udmt_to_proxy.py`) runs back on the repo machine (needs calibration).

## What's here
- `make_xmem_init.py` — Detectron2 masks + 3D positions → UDMT first-frame init
  (`<view>_init_mask.png` binary fg + `<view>_start_pos.txt` 3 rows + `_init_meta.json`).
  Reflections dropped by 3D match; a fish Detectron2 missed on frame 0 gets a synthetic
  disk seed at its 3D position (recorded). **Already produced for the 3 views** — see
  `init/` (rsync it).
- `run_udmt_headless.py` — drives native UDMT for ONE view without the GUI (project →
  inject init → XMem propagate → ST-Net dataset → train → analyze/track → filtered `.npy`).
- `compare_udmt_to_proxy.py` — scores the `.npy` vs the 3D identity with the SAME panel
  as the motion-proxy (`scripts/experiments/udmt_individual_tracking.py`) → directly comparable.
- `prep_window_videos.sh` — ffmpeg-cut each view's 176300–181300 window (frame 0 = 176300).
- `run_udmt.sbatch` — Slurm array (one view per task): run + score.

## Steps on saion
1. **Get UDMT + checkpoints** (login node, internet): clone UDMT to `$UDMT_ROOT`; download
   `XMem.pth` → `udmt/gui/tabs/xmem/saves/` and `trdimp_net_ep.pth.tar` → `udmt/pretrained/`.
   **Skip SAM/SEEM/`requirements_custom.txt`** (not used).
2. **Env** — recommended **Singularity** (`udmt.def` in `docs/udmt_saion_runbook.md` §1A;
   the CUDA-devel base ships `nvcc` for the PrRoIPooling JIT). Native fallback:
   `module load uv/0.11.0 cuda/12.2 gcc/11.2.1 ninja/1.12.0.git` + `uv venv --python 3.8`.
3. **Pre-warm PrRoIPooling** once on a GPU node into a shared `TORCH_EXTENSIONS_DIR`
   (`docs/udmt_saion_runbook.md` §1b) so batch jobs need no `nvcc`.
4. **Data** — `bash prep_window_videos.sh` → `$SCRATCH/udmt_run/videos/<view>_176300_181300.mp4`.
5. **rsync this bundle** (incl. `init/` + the 3D `*_3d_identified.csv` + `paths.local.toml`)
   to `$BUNDLE` on bucket. Edit the paths at the top of `run_udmt.sbatch`.
6. **Smoke gate (do this first, ~minutes):** one view, XMem-only + a tiny ST-Net:
   ```bash
   python run_udmt_headless.py --udmt-root $UDMT_ROOT --project $PROJECT \
     --video .../top_cam_176300_181300.mp4 --stem top_cam_176300_181300 \
     --init-mask init/top_cam_init_mask.png --init-startpos init/top_cam_start_pos.txt \
     --xmem-ckpt $XMEM_CKPT --trdimp-ckpt $TRDIMP_CKPT --xmem-only
   ```
   Pass = `tmp/<stem>/masks/%07d.png` fill in with a moving foreground, no Qt/X11 crash.
   Then drop `--xmem-only` with `--epochs 1 --train-frames 300 --tune-frames 300` and
   confirm a `<stem>-whole-filter5.npy` of shape `(3, ~N, 2)` appears.
7. **Full run:** `sbatch run_udmt.sbatch` (array 0–2). Each task writes
   `<stem>-whole-filter5.npy` + `<stem>_udmt_vs_3d.json`.
8. **Compare** the three `*_udmt_vs_3d.json` to the motion-proxy table
   (`docs/udmt_individual_tracking_test.md`): does native DiMP **fragment less / hold the
   female on side2** better than the motion proxy?

## ⚠️ Validation note
`run_udmt_headless.py` is transcribed from a verified code mapping of the UDMT clone but
was **not executed on Windows** (no UDMT env). The two parts most likely to need a
line-level tweak to your exact UDMT commit: the **XMem propagation loop** (mirrors
`gui.py:659-690`) and the **ST-Net step-4/5/6 control flow** (`create_training_dataset.py:287-314`
/ `analyze_videos.py:540-595`). The smoke gate (step 6) catches both cheaply.

## Honest scope
This measures DiMP/ST-Net **continuity/fragmentation** on the female + male fragments. It
will **not** separate the two appearance-identical males (UDMT carries identity by track
continuity, no appearance-ID) — they can swap at male crossings like the geometry methods.
The female on side1 is seeded synthetically (Detectron2 missed her on frame 0). Treat
results as evidence about **tracklet robustness**, not about resolving the two males.
