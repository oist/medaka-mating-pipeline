"""Scale Line-B step 1 across the full K_1_2 timeline.

Samples one window from each of the 8 saion Detectron2 mask chunks (spanning 0-348599),
runs the production crop build (33) per view per window, pools (34), and calls the female
globally via window_align (44) -> per-window female 3D-slot + a whole-track sexed CSV.

Local: masks copied from Z: (detectron2_output) chunk-by-chunk; video read from Z:;
tracklet-bound CSVs + models from Z:/StephensU/medaka/runs/udmt_validation. Run with the repo .venv python.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SCRATCH = Path(r"C:\Users\machi\AppData\Local\Temp\claude\c--Users-machi-Documents-GitHub-medaka-mating-pipeline\b07af347-293e-4fed-a208-e52ffcd113b6\scratchpad\masks")
MASKS_Z = Path(r"Z:\StephensU\medaka\detectron2_output\20251225_K_1_2_FCMC")
# videos copied to local scratch -- network (Z:) frame-by-frame decode is ~4 fps vs ~100 fps local
VIDEO_Z = Path(r"C:\Users\machi\AppData\Local\Temp\claude\c--Users-machi-Documents-GitHub-medaka-mating-pipeline\b07af347-293e-4fed-a208-e52ffcd113b6\scratchpad\videos")
TB = Path(r"Z:\StephensU\medaka\runs\udmt_validation\tracklet_bind_extend_smooth")
FEMALE_MODEL = r"Z:\StephensU\medaka\runs\udmt_validation\segid\K_1_2\model_female_clean\model.pt"
EMBEDDER = r"Z:\StephensU\medaka\runs\udmt_validation\segid\K_1_2\model_resnet_full\model.pt"
THREED = r"Z:\StephensU\medaka\reconstruction_3d\round5_260611\20251225_K_1_2_FCMC\round5_260611_3d_identified.csv"

CHUNKS = [(0, 43574), (43575, 87149), (87150, 130724), (130725, 174299),
          (174300, 217874), (217875, 261449), (261450, 305024), (305025, 348599)]
VIEWS = [("side1_cam", "cam1_side"), ("side2_cam", "cam2_side")]
WIN_HALF = 1500   # 3000-frame window centered in each chunk
PY = sys.executable


def run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(str(c) for c in cmd[-6:]), flush=True)
    subprocess.run(cmd, check=True, **kw)


def slice_masks(src: Path, dst: Path, ws: int, we: int) -> int:
    """Copy only the frame groups in [ws,we] from a chunk masks .h5 to a small one.
    Uses h5py object-copy (compressed, NO decompress) so 33 then iterates ~3000 frames
    instead of the whole 43,575-frame chunk -- that full-chunk gzip decompress, not the
    video, was the real bottleneck (~12 min/view)."""
    n = 0
    with h5py.File(src, "r") as s, h5py.File(dst, "w") as d:
        for k, v in s.attrs.items():
            d.attrs[k] = v
        for fk in s.keys():
            if fk.startswith("frame_") and ws <= int(fk.split("_")[1]) <= we:
                s.copy(fk, d)
                n += 1
    return n


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = []
    for cs, ce in CHUNKS:
        mid = (cs + ce) // 2
        ws, we = mid - WIN_HALF, mid + WIN_HALF - 1
        win_dir = OUT / f"win_{ws}_{we}"
        candidates.append(win_dir)
        if all((win_dir / f"{v}_crops.h5").exists() for v, _ in VIEWS):
            print(f"[chunk {cs}-{ce}] window {ws}-{we} already built, skip", flush=True)
            continue
        print(f"\n=== chunk {cs}-{ce} -> window {ws}-{we} ===", flush=True)
        try:
            for v, cam in VIEWS:
                slice_h5 = SCRATCH / f"{v}.win_{ws}_{we}.h5"
                t = time.time()
                nfr = slice_masks(MASKS_Z / f"{v}.masks_{cs}-{ce}.h5", slice_h5, ws, we)
                print(f"  sliced {v}: {nfr} frames in {time.time()-t:.0f}s", flush=True)
                t = time.time()
                run([PY, "-m", "sam2_id_experiment.scripts.33_build_segid_dataset",
                     "--view", v, "--masks", str(slice_h5),
                     "--tracklet-bound-csv", str(TB / f"{v}.tracklet_bound.csv"),
                     "--video", str(VIDEO_Z / v / f"{cam}.mp4"),
                     "--window", str(ws), str(we), "--out-dir", str(win_dir)], cwd=str(REPO))
                print(f"  33 {v} done in {time.time()-t:.0f}s", flush=True)
                slice_h5.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001 -- one bad window must not kill the sweep
            print(f"  !! window {ws}-{we} FAILED ({e}); skipping", flush=True)
        finally:
            for v, _ in VIEWS:
                (SCRATCH / f"{v}.win_{ws}_{we}.h5").unlink(missing_ok=True)

    # only pool windows that produced both views' crops
    win_dirs = [d for d in candidates if all((d / f"{v}_crops.h5").exists() for v, _ in VIEWS)]
    print(f"\nwindows usable for pooling: {len(win_dirs)}/{len(candidates)}", flush=True)

    # ---- pool all windows ----
    pooled = OUT / "pooled"
    print("\n=== 34: pool all windows ===", flush=True)
    run([PY, "-m", "sam2_id_experiment.scripts.34_assemble_segid_training",
         "--segid-dir", *[str(d) for d in win_dirs],
         "--views", "side1_cam", "side2_cam",
         "--tracklet-bound-dir", str(TB), "--out-dir", str(pooled)], cwd=str(REPO))

    # ---- global female call + whole-track sexed CSV ----
    fa = OUT / "female_anchor"
    print("\n=== 44: window_align + global female call ===", flush=True)
    cmd = [PY, "-m", "sam2_id_experiment.scripts.44_apply_clean_female_anchor",
           "--target-dataset", str(pooled), "--model", FEMALE_MODEL, "--embedder", EMBEDDER,
           "--out-dir", str(fa)]
    if Path(THREED).exists():
        cmd += ["--identified-3d-csv", THREED]
    else:
        print("  (3D track missing -> female call only, no whole-track sexed CSV)", flush=True)
    run(cmd, cwd=str(REPO))
    print("\nDONE -> see", fa / "female_anchor_clean_report.json", flush=True)


if __name__ == "__main__":
    main()
