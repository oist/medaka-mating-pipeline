# Runbook — original UDMT (ST-Net/TrDiMP) on saion (V100), Detectron2-seeded

Goal: run the **real** UDMT tracker on OIST saion (Slurm, **Tesla V100-SXM2-16GB, sm_70**),
seeding its initial + per-frame masks from our **Detectron2** instance masks instead of
UDMT's SAM/XMem/SEEM, on window **176300–181300** of `20251225_K_1_2_FCMC`, and compare to
the motion-proxy result (`docs/udmt_individual_tracking_test.md`).

Source-verified against the clone at `C:/udmt_analysis/UDMT` (2026-06-22): function names,
line numbers, and on-disk contracts below were checked in code, not assumed.

## 0. Verdict + the key simplification

- **V100 sm_70 runs UDMT's pinned `torch 2.1.1/cu121` out of the box** (cu121 wheels ship
  sm_70 kernels). The Blackwell blocker that killed the local-RTX-5070-Ti option is gone.
  **Do NOT apply the README RTX-4090 `cpp_extension` arch patch** (that is for sm_89).
- **Seeding from Detectron2 masks avoids SAM + XMem + SEEM entirely** — the only display/
  VRAM-heavy parts. You then need only **one custom CUDA op (PreciseRoIPooling, JIT-compiled)**
  and **one checkpoint (`trdimp_net_ep.pth.tar`)**. Skip `detectron2-xyz`, MSDA/Deformable-DETR,
  `XMem.pth`, `sam_vit_*`, SEEM. This removes most of the porting risk.
- UDMT is a PySide6 GUI app; on a headless compute node you drive it by calling the free
  functions directly with two module-level display flags turned off (§4).
- **Honest scope (read §8):** this measures DiMP's temporal **continuity/fragmentation** on the
  female + male fragments. It will **NOT** separate the two appearance-identical males — UDMT
  carries identity purely by track continuity (no appearance-ID), so it can swap at a male–male
  crossing exactly like the geometry methods. Expect a *robustness* delta, not an identity fix.

---

## 1. Environment on saion

### Option A (recommended) — Singularity container

The stack is fragile *and* PrRoIPooling JIT-compiles at **runtime** (needs `nvcc` on the node).
A CUDA **devel** container ships `nvcc` inside, so the run needs no `cuda` module and is fully
reproducible. `singularity/4.1.4` is available.

`udmt.def`:
```singularity
Bootstrap: docker
From: nvidia/cuda:12.1.1-devel-ubuntu22.04
%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y python3.8 python3.8-venv python3.8-dev \
        python3-pip git ffmpeg libgl1 libglib2.0-0 ninja-build && rm -rf /var/lib/apt/lists/*
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
        --index-url https://download.pytorch.org/whl/cu121
    git clone <your UDMT remote> /opt/UDMT
    cd /opt/UDMT && python3.8 -m pip install -r requirements.txt   # SKIP requirements_custom.txt
%environment
    export TORCH_CUDA_ARCH_LIST=7.0
    export QT_QPA_PLATFORM=offscreen
    export MPLBACKEND=Agg
```
Build where you have Docker+internet (or `singularity build --fakeroot udmt.sif udmt.def` on
saion if fakeroot is permitted), then run on the V100 with `singularity exec --nv udmt.sif ...`.

### Option B — native env with saion modules (uv, no py3.8 module)

```bash
module load uv/0.11.0 cuda/12.2 gcc/11.2.1 ninja/1.12.0.git cmake/3.31.7
cd /bucket/StephensU/medaka/udmt/UDMT
uv venv --python 3.8 .venv-udmt && source .venv-udmt/bin/activate
uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt           # NOT requirements_custom.txt (SEEM-only)
export CUDA_HOME=$CUDA_ROOT                   # from cuda/12.2 module
export TORCH_CUDA_ARCH_LIST=7.0
```
Note: `cuda/12.2` toolkit builds cu121-torch extensions fine (same major; PyTorch warns on the
minor mismatch but compiles). `pillow==9.4.0`, `pyside6<6.3.2`, `kornia==0.7.0`, `timm==0.4.12`,
`numba`, `tables` can fight on a fresh py3.8 Linux — **resolve all of this on the login node**
(internet) before any offline batch job.

### 1a. Login-node downloads (internet; compute nodes are offline)
```bash
mkdir -p udmt/gui/pretrained
wget -O udmt/gui/pretrained/trdimp_net_ep.pth.tar \
  "https://zenodo.org/records/14671891/files/trdimp_net_ep.pth.tar?download=1"   # verify record id vs README
# DO NOT fetch XMem.pth / sam_vit_* / SEEM — unused when seeding from Detectron2.
```
> **Do NOT call `launch_script.py` / `adjust_external_folder()` on Linux** — `external/` already
> holds the Linux PrRoIPooling; the swap could install the Windows `.pyd`. Ignore `external-win/`.

### 1b. Pre-warm PrRoIPooling on a GPU node (so offline jobs reuse the `.so`)
```bash
srun -p <gpu_partition> --gres=gpu:v100:1 -t 1:00:00 --mem 32G --pty bash
# (container: singularity exec --nv udmt.sif bash ; native: activate env + module load cuda/12.2)
export TORCH_EXTENSIONS_DIR=/bucket/StephensU/medaka/udmt/torch_ext_cache   # SHARED path
export TORCH_CUDA_ARCH_LIST=7.0 FORCE_CUDA=1
python - <<'PY'
import sys; sys.path.insert(0,".")
import torch
from udmt.gui.tabs.ST_Net.ltr.external.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
x=torch.rand(1,4,16,16,device='cuda',requires_grad=True)
r=torch.tensor([[0,0,0,8,8]],dtype=torch.float32,device='cuda')
y=prroi_pool2d(x,r,4,4,1.0); y.sum().backward(); print("PrRoIPooling OK", y.shape)
PY
```

---

## 2. Data — window 176300–181300 onto saion

UDMT trains on the **first 2000 frames** post-downsample (`create_training_dataset.py:229/295`)
and tunes on **≤4000** (`analyze_videos.py:556`). Pre-cut the slice so **frame 176300 = UDMT
frame 0** and run `downsample=1` so mask index `%07d` == raw index.
```bash
export SCRATCH=/flash/StephensU/$USER/udmt_run; mkdir -p $SCRATCH/videos   # work on scratch, not masters
for V in top_cam side1_cam side2_cam; do
  ffmpeg -y -i /bucket/.../<exp>/$V/cam*.mp4 \
    -vf "select='between(n,176300,181299)',setpts=N/FRAME_RATE/TB" -an -r 166 \
    $SCRATCH/videos/${V}_176300_181300.mp4
done
```
> **Data-loss guard (verified `new.py:157` `os.remove(video)`):** `create_new_project` deletes the
> source video if `VideoReader` can't open it. Use `copy_videos=True` and point at the **scratch**
> copies, never an irreplaceable `/bucket` master.

---

## 3. Detectron2-mask injection — on-disk contract (write a small adapter)

Replace UDMT's SAM/XMem selection **and** XMem propagation by writing 3 artifacts per view
under `tmp/<stem>/` — **no tracker code is patched**. New code (`inject_detectron2_masks.py`,
the `.h5`→png/centroid adapter) is the main integration risk.

| Artifact | Path | Format (verified) |
|---|---|---|
| **Seed (the init)** | `tmp/<stem>/extracted-images/start_pos_array.txt` | ASCII CSV, **one row per fish**, `x,y` floats `%.2f`, comma-delim. Coords = extracted-image px **÷ resize_scale** (re-multiplied at `run_tracker.py:64`). 3 fish → 3 rows. **Row order = UDMT track index** (identity lives ONLY here). Ref `auto_initial_prediction.py:299-310`. |
| **First-frame mask** | `tmp/<stem>/masks/0000000.png` | single-channel **uint8 'L', fg=1, bg=0**, `(H,W)` at resized working res. Read `cv2.imread(...,0)` then `>0`. Ref `set_ini_value.py:49`. |
| **Per-frame masks** | `tmp/<stem>/masks/%07d.png`, **every** frame 0000000…0004999 | same fg PNG; binarized `>0→255`. Indexed `"%07d" % (n/down_sample_fg)` → run `downsample=1`. Consumed `missing_object_detect.py:35/185/269`. |

Critical (verified):
- Centroid `(x,y)`: `x=col, y=row`; `set_ini_value` reads `mask[y,x]` and **silently drops** any
  fish whose seed lands on background (label 0). Put each centroid on its own blob.
- The PNG **collapses instances to foreground**; UDMT separates fish by `connectedComponentsWithStats`,
  not label value. **Keep Detectron2 instances as separate connected components — do NOT dilate
  into one blob** (the close-contact regime is exactly what you're comparing).
- The `resize_scale` you divide centroids by **must equal** the `resize_factor` you run with
  (default 0.8), else boxes land off-fish.
> Run UDMT's frame extraction first (writes `extracted-images/*.jpg` + `tmp/` at `resize_factor=0.8`),
> then **overwrite** `masks/` + `start_pos_array.txt` with the Detectron2 versions before tracking.

---

## 4. Headless driver — `run_udmt_headless.py`

One module-level constant forces a live Qt display on the per-frame hot path
(`ST_Net/pytracking/evaluation/tracker_multi.py`): **`vis_flag=True` (line 49)** —
set it `False`. Keep **`save_flag=True`**: UDMT uses it to create the per-animal
`*_new.txt` outputs that `post_process_results()` consumes. `run_tracking` calls
`argparse.parse_args()` (`run_tracker.py:107`) → neutralize `sys.argv`. Call the free functions:
- `udmt.gui.create_new_project(...)` (Qt-free, `__init__.py:23`)
- `udmt.gui.tabs.ST_Net.pytracking.run_tracker.run_tracking(params)` (`:94`)
- `udmt.gui.tabs.ST_Net.ltr.run_training.run_training_process(params)` (`:44`)
- `...pytracking.utils.filter_and_save.post_process_results(...)`

```python
import os, sys, glob, logging
os.environ.update(QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="0", DEBUG="")
UDMT_ROOT="/opt/UDMT"; ST="%s/udmt/gui/tabs/ST_Net"%UDMT_ROOT
sys.path[:0]=[UDMT_ROOT, ST]            # code mixes `udmt.gui...` AND bare `ltr...`/`pytracking...`
sys.argv=[sys.argv[0]]                  # run_tracking reads argv
import udmt.gui.tabs.ST_Net.pytracking.evaluation.tracker_multi as tm
tm.vis_flag=False; tm.save_flag=True    # no Qt display; keep text outputs
from udmt.gui import create_new_project
from udmt.gui.tabs.ST_Net.pytracking.run_tracker import run_tracking
from udmt.gui.tabs.ST_Net.ltr.run_training import run_training_process
from udmt.gui.tabs.ST_Net.pytracking.utils.filter_and_save import post_process_results
# (1) create_new_project(copy_videos=True) (2) extract frames (3) inject_detectron2_masks()
# (4) run_tracking(status_flag=1, widgets=None)  -> training dataset / pseudo-labels
# (5) run_training_process(epoch_num=5-10, batch_size=8)
# (6) run_tracking(status_flag=2, frame_num=4000) tune -> pick best (loaded_results_list[-1], NOT argmin)
#     run_tracking(status_flag=3, all frames) final
# (7) post_process_results(stem, 5, raw_dir, out_dir) -> <stem>-whole-filter5.npy  shape (3, n, 2)
```
Output `<stem>-whole-filter5.npy` = filtered **center XY in resized px** (× 1/0.8 for original).
IDs are arbitrary 0..2 — align to the motion-proxy by first-frame proximity before scoring.
> **Uncertain (flagged):** `create_training_dataset.py:249-314` passes live Qt widgets into
> `run_tracking`; the driver must pass `video_display_widget1/2=None` and confirm `run_tracking`
> tolerates `None` with `vis_flag=False`. `pick_best_param` must mirror `loaded_results_list[-1]`.

---

## 5. sbatch (V100)
```bash
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:v100:1 --cpus-per-task=8 --mem=48G --time=36:00:00 --array=0-2
export TORCH_EXTENSIONS_DIR=/bucket/StephensU/medaka/udmt/torch_ext_cache  # reuse pre-warmed .so
export TORCH_CUDA_ARCH_LIST=7.0 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0
# container:  singularity exec --nv udmt.sif python run_udmt_headless.py $SLURM_ARRAY_TASK_ID
# native:     module load cuda/12.2 gcc/11.2.1; source .venv-udmt/bin/activate; python run_udmt_headless.py $SLURM_ARRAY_TASK_ID
```
Batch 8 / 352×352 / ResNet-50+transformer+DiMP is well under 16 GB; drop batch if OOM. One view per array task.

## 6. Go / no-go gates (stop on first failure)
1. `torch.cuda.get_device_capability(0)` → `(7,0)` on `Tesla V100`.
2. PrRoIPooling §1b prints `PrRoIPooling OK` (else node lacks nvcc/CUDA_HOME).
3. Injection sanity: `start_pos_array.txt` has 3 rows; each centroid lands on a nonzero pixel of
   `masks/0000000.png`; `masks/` has 5000 single-channel uint8 PNGs.
4. **200-frame smoke** (`frame_num=200, epoch_num=1`, top_cam only): a `*-whole-filter5.npy` of
   shape `(3,~200,2)`, finite, IDs stable, no Qt/X11/`None.width()` crash. Validates the whole
   headless path for ~free before the long job.
5. Full run only after 1–4.

## 7. Effort + top risks
**~2–4 days** + **~14–28 h/view** training (est., not measured on V100; use `epoch_num=5–10` for the
pilot — the guide says 20 is overkill). Risks: (1) **PrRoIPooling JIT at runtime** — pre-warm into a
shared `TORCH_EXTENSIONS_DIR` (gate 2); (2) **Detectron2 adapter correctness** — wrong centroid
scale silently drops fish; merging contacting fish breaks ID-correction (gate 3 + eyeball frame 0);
(3) **headless GUI couplings** — `vis_flag`, `sys.argv`, `None` widgets (gate 4);
keep `save_flag=True` so track text files are written.

## 8. Honest note — what this can/can't show
Measures **DiMP/ST-Net per-target continuity**: does a self-trained appearance+motion tracker hold
each fish's box through false-negatives/occlusions/crossings better than the motion-proxy, and how
much does it fragment — a legitimate comparison on the female + male fragments. It will **NOT**
separate the two appearance-identical males: UDMT carries identity by track continuity seeded from
`start_pos_array.txt` row order, with no appearance-ID, so it swaps at male–male crossings like the
geometry methods (already `label_agreement 1.0`). If UDMT agrees with the motion-proxy except at
crossings, that is the expected outcome; only ground-truth labels at crossings can adjudicate.
Treat results as evidence about **tracklet robustness**, not about resolving the two males.
