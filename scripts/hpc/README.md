# HPC batch SLEAP inference (OIST saion)

These scripts run SLEAP predictions for medaka recordings on the OIST **saion**
cluster. They run from the **login node**, submit Slurm array jobs, and copy
results back to `/bucket` when each experiment finishes.

## Layout

```
scripts/hpc/
  sleap_chunk.slurm      # Array worker. Runs one chunk of one video.
  submit_experiment.sh   # Submits 3 views × N chunks for ONE experiment.
  submit_batch.sh        # Iterates many experiments, calling submit_experiment.sh.
  merge_chunks.sh        # Copies per-view chunks from /work → /bucket (no merging;
                         # chunks stay separate and are merged at analysis import).
```

## Parallelism model

For one experiment (say `20251220_D_1_1_FCMC`) this submits three Slurm
**array jobs** in parallel, one per view:

```
top_cam   → sbatch --array=0-(N-1)   # N chunks of cam0_top.mp4
side1_cam → sbatch --array=0-(N-1)   # N chunks of cam1_side.mp4
side2_cam → sbatch --array=0-(N-1)   # N chunks of cam2_side.mp4
```

If a single video fits in the 12 h wall, set `NUM_CHUNKS=1`. If a video is
long, set `NUM_CHUNKS=4` and each view runs as 4 parallel tasks — so a
4-chunk × 3-view experiment consumes up to 12 GPUs simultaneously.

The account has 8 largegpu and 8 gpu nodes max. Slurm queues the rest — you
don't have to throttle manually.

## Expected filesystem layout on saion

```
/bucket/StephensU/medaka/medaka_202512/
    20251220_D_1_1_FCMC/
        top_cam/cam0_top.mp4
        side1_cam/cam1_side.mp4
        side2_cam/cam2_side.mp4
    ...

/bucket/StephensU/medaka/medaka_202512/sleap_models/
    sleap_top_mouth/models/260313_152145.centroid.n=330/
    sleap_top_mouth/models/260313_175737.centered_instance.n=330/
    sleap_side_mouth/models/260111_170857.centroid.n=593/
    sleap_side_mouth/models/260112_115936.centered_instance.n=593/

/work/StephensU/<user>/medaka_sleap/<experiment>/
    top_cam/{cam0_top.mp4, top_cam.frames_*.slp, chunks.txt, slurm_*.out}
    ...

/bucket/StephensU/medaka/sleap_output/<experiment>/
    top_cam.frames_*.slp      (4 chunks by default)
    side1_cam.frames_*.slp
    side2_cam.frames_*.slp
    logs/                     (slurm per-task .out/.err)
```

Chunks are **not merged on the HPC side** — they stay as parallel per-frame-range
`.slp` files and are combined into a single `.slp` or `.h5` the first time the
analysis code loads them (in a venv that has sleap-io available).

Models must live on `/bucket` so compute nodes can read them (they can't
read `/home` or Google Drive). If your models are currently on Google Drive,
copy them over once:

```bash
# from login node
rsync -av "/path/to/local/sleap_models/" /bucket/StephensU/medaka/medaka_202512/sleap_models/
```

## Quick start (one experiment)

```bash
ssh saion.oist.jp
cd ~/medaka-mating-pipeline

# Preview — prints sbatch commands, submits nothing
DRY_RUN=1 bash scripts/hpc/submit_experiment.sh 20251220_D_1_1_FCMC

# Actual submission (uses default largegpu, 4 chunks per view)
bash scripts/hpc/submit_experiment.sh 20251220_D_1_1_FCMC

# Monitor
squeue -u $USER
tail -f /work/StephensU/$USER/medaka_sleap/20251220_D_1_1_FCMC/merge_and_copy.log
```

## Batch submission (many experiments)

```bash
# List what would be submitted (default regex: FCMC recordings only)
bash scripts/hpc/submit_batch.sh --list

# Dry-run every matching experiment
bash scripts/hpc/submit_batch.sh --dry-run

# Submit all; skip experiments that already have all three merged view .slps
bash scripts/hpc/submit_batch.sh --skip-existing

# Custom filter
bash scripts/hpc/submit_batch.sh --experiments '^20260109_'
```

## Configuration (environment variables)

All of these are honoured by both `submit_experiment.sh` and `submit_batch.sh`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEDAKA_BUCKET_ROOT` | `/bucket/StephensU/medaka/medaka_202512` | Input recordings root |
| `MEDAKA_WORK_ROOT` | `/work/StephensU/makoto/medaka_sleap` | Scratch for staged videos + chunk .slps |
| `MEDAKA_BUCKET_OUT` | `/bucket/StephensU/medaka/sleap_output` | Where final .slps go |
| `TOP_CENTROID` | `.../sleap_top_mouth/models/260313_152145.centroid.n=330` | Top centroid model |
| `TOP_INSTANCE` | `.../sleap_top_mouth/models/260313_175737.centered_instance.n=330` | Top instance model |
| `SIDE_CENTROID` | `.../sleap_side_mouth/models/260111_170857.centroid.n=593` | Side centroid model |
| `SIDE_INSTANCE` | `.../sleap_side_mouth/models/260112_115936.centered_instance.n=593` | Side instance model |
| `NUM_CHUNKS` | `4` | Chunks per view; 1 disables splitting |
| `MAX_INSTANCES` | `3` | 1 female + 2 males |
| `BATCH_SIZE` | `16` | sleap-nn batch size |
| `PARTITION` | `largegpu` | `largegpu` / `gpu` / `test-gpu` |
| `TIME_LIMIT` | `12:00:00` | Per-task wall (Slurm `--time`) |
| `SLEAP_MODULE` | `sleap-nn/0.1.2` | Module to `module load` on compute nodes (also: `sleap-nn/0.1.0`, `sleap/1.5.2`, `sleap/1.6.2`) |
| `MODULEPATH_EXTRA` | `/apps/unit/ReiterU/.modulefiles` | Prepended via `module use`; holds the sleap/sleap-nn modulefiles |
| `SKIP_VIEWS` | _(empty)_ | space-separated views to skip, e.g. `side2_cam` |

Example — use the `gpu` partition (V100, lower queue pressure) and no
chunking for a short clip:

```bash
PARTITION=gpu NUM_CHUNKS=1 TIME_LIMIT=04:00:00 \
    bash scripts/hpc/submit_experiment.sh 20251220_D_1_1_FCMC
```

## What happens end-to-end

1. `submit_experiment.sh` probes `cam0_top.mp4` etc. with `ffprobe`, writes a
   `chunks.txt` listing `START END` frame pairs for each view, and submits
   one `sbatch` array per view (jobs queue on `largegpu` by default).
2. Each array task (in `sleap_chunk.slurm`) stages the video to `/work`
   (task 0 of each array copies; the others wait on a sentinel file), loads
   the SLEAP module, and writes `<view>.frames_<S>-<E>.slp` to the
   per-view `/work` dir.
3. The submitter also starts a `nohup` watcher on the login node that polls
   `squeue` for the three array IDs, then runs `merge_chunks.sh` when they
   all leave the queue.
4. `merge_chunks.sh` rsyncs the per-chunk `.slp` files plus slurm logs to
   `/bucket/.../sleap_output/<experiment>/`. No merging — chunks stay
   separate and are combined by the analysis code on first load.

## Troubleshooting
- **`module load sleap-nn` fails**: your unit may not expose that module.
  Set `MODULEPATH_EXTRA=/apps/unit/<YourUnit>/.modulefiles` or use an
  installed name like `SLEAP_MODULE=sleap/1.5.2`.
- **Videos missing**: medaka recordings sometimes lack `side2_cam/`; pass
  `SKIP_VIEWS="side2_cam"` to submit only the two available views.
- **Running out of queue time**: raise `NUM_CHUNKS` (more parallelism per
  video) or lower it if the 12 h wall is generous. With 4 chunks × 3 views
  one experiment takes 12 GPU-slots simultaneously.
- **Frame count**: `submit_experiment.sh` reads `totalFrames` directly from
  each camera folder's `metadata.csv` (written by campy when recording
  closed, so it's exact). No `ffprobe`/`ffmpeg` module needed. If a
  `metadata.csv` is missing, it falls back to `frametimes.npy` via
  `python3 + numpy`.

## Related Python entry points

The HPC pipeline writes `.slp` files in the layout the Python batch script
expects. After chunks are copied back, you can run:

```bash
python scripts/sleap/export_tracks.py --input <BUCKET_EXP_OUT>/top_cam.slp --format analysis.csv
python -m scripts.tracking.main --paths-config configs/paths.local.toml <set_key>
```

and the downstream tracking / 3D reconstruction stages don't need any
HPC-specific code changes.
