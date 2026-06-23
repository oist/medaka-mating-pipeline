#!/usr/bin/env python
from __future__ import annotations

"""Headless driver for native UDMT (XMem -> ST-Net/TrDiMP), Detectron2-seeded.

Runs UDMT's real pipeline on ONE view without the PySide6 GUI: create project ->
extract frames + XMem init from a Detectron2 seed (replaces SEEM) -> XMem
propagation -> ST-Net training-dataset -> train -> analyze/track -> filtered .npy.

Transcribed from the UDMT clone (function names / arg dicts / line refs verified);
the XMem-loop and the steps-4-6 control flow mirror gui.py:659-690 and
create_training_dataset.py:287-314 / analyze_videos.py:540-595. VALIDATE on saion
with the 200-frame smoke gate (README) before the full run -- this driver is written
from a code mapping, not executed on Windows.

This script runs ONLY inside the UDMT env (python3.8 + torch 2.1.1/cu121 on a V100).
Inputs (init_mask.png / start_pos.txt) come from make_xmem_init.py.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


def _headless_env(udmt_root: str):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("DEBUG", "")
    st_net = os.path.join(udmt_root, "udmt/gui/tabs/ST_Net")
    sys.path[:0] = [udmt_root, st_net]      # code mixes `udmt.gui...` and bare `ltr...`/`pytracking...`
    sys.argv = [sys.argv[0]]                # run_tracking/run_training parse argv internally
    # if detectron2-xyz (SEEM) is installed, the xmem import chain pulls in detectron2,
    # whose transforms reference Image.LINEAR/CUBIC removed in newer Pillow -> shim them.
    from PIL import Image
    for a, b in [("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC")]:
        if not hasattr(Image, a):
            setattr(Image, a, getattr(Image, b))


# --------------------------- small local helpers --------------------------- #
# Reimplemented (instead of importing the Qt tab modules) to reduce coupling.

def _extract_frames(video: str, out_dir: str, scale: float, downsample: int, cap: int | None):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    capv = cv2.VideoCapture(video)
    ok, first = capv.read()
    if not ok:
        raise SystemExit(f"cannot read {video}")
    w = int(first.shape[1] * scale); h = int(first.shape[0] * scale)
    capv.set(cv2.CAP_PROP_POS_FRAMES, 0)
    saved = fc = 0
    step = max(1, int(round(1 / downsample)))
    while True:
        ok, frame = capv.read()
        if not ok or (cap is not None and saved >= cap):
            break
        if fc % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{saved:05d}.jpg"), cv2.resize(frame, (w, h)))
            saved += 1
        fc += 1
    capv.release()
    return saved


def _newest_dir(parent: str):
    subs = [d for d in glob.glob(os.path.join(parent, "*")) if os.path.isdir(d)]
    return max(subs, key=os.path.getmtime) if subs else None


def _subfolder_with_keyword(parent: str, keyword: str):
    for d in glob.glob(os.path.join(parent, f"*{keyword}*")):
        if os.path.isdir(d):
            return os.path.basename(d)
    raise SystemExit(f"no train_set_results subfolder matching {keyword} in {parent}")


def _select_model(stem: str, models_folder: str):
    cands = sorted(glob.glob(os.path.join(models_folder, "**", "*.pth.tar"), recursive=True),
                   key=os.path.getmtime)
    if not cands:
        raise SystemExit(f"no trained .pth.tar under {models_folder}")
    return cands[-1]


# ------------------------------ XMem propagation --------------------------- #

def _xmem_config(video: str, workspace: str, xmem_ckpt: str, num_objects: int, size: int) -> dict:
    # mirrors interactive_demo.py:49-87 argparse defaults + the 2 manual keys.
    # `size` MUST make ResourceManager.resize_scale == ST-Net resize_factor, else the
    # masks (used as `bg` in tracker_multi.py:330 -> set_ini_value for target_sz / area /
    # kernel) land at a different resolution than the 960px extracted-images and the DiMP
    # template comes out half-size. Caller passes round(resize_factor * min(H, W)).
    return {
        "model": xmem_ckpt, "s2m_model": None, "fbrs_model": None,
        "images": None, "video": video, "workspace": workspace,
        "buffer_size": 100, "num_objects": num_objects,
        "max_mid_term_frames": 100, "min_mid_term_frames": 90,
        "max_long_term_elements": 10000, "num_prototypes": 128,
        "top_k": 30, "mem_every": 1000, "deep_update_every": -1,
        "no_amp": False, "size": size,
        "enable_long_term": True, "enable_long_term_count_usage": True,
    }


def run_xmem_propagation(rm, config: dict, xmem_ckpt: str, init_fg: np.ndarray, num_objects: int):
    """Seed object channel(s) from `init_fg` (uint8 labels, fg=1) on frame 0 and
    propagate to every frame, writing masks/%07d.png via rm.save_mask. Mirrors
    gui.py:659-690 with the Qt calls removed."""
    import torch
    from udmt.gui.tabs.xmem.model.network import XMem
    from udmt.gui.tabs.xmem.inference.inference_core import InferenceCore
    from udmt.gui.tabs.xmem.inference.interact.interactive_utils import (
        image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask,
    )
    torch.set_grad_enabled(False)
    net = XMem(config, xmem_ckpt).cuda().eval()
    processor = InferenceCore(net, config)
    processor.set_all_labels(list(range(1, num_objects + 1)))

    with torch.cuda.amp.autocast(enabled=not config["no_amp"]):
        # frame 0 seed
        img0 = rm.get_image(0)
        img0_t, _ = image_to_torch(img0)                                   # normalized 3*H*W
        cur_prob = index_numpy_to_one_hot_torch(init_fg, num_objects + 1).cuda()
        cur_prob = processor.step(img0_t, cur_prob[1:])                     # seed: mask w/o bg channel
        rm.save_mask(0, torch_prob_to_numpy_mask(cur_prob))
        # propagate
        for ti in range(1, rm.length):
            img_t, _ = image_to_torch(rm.get_image(ti))
            cur_prob = processor.step(img_t)                               # image only
            rm.save_mask(ti, torch_prob_to_numpy_mask(cur_prob))
            if ti % 500 == 0:
                print(f"  xmem propagated {ti}/{rm.length}", flush=True)


# ------------------------------- ST-Net steps ------------------------------ #

def run_stnet(project: str, stem: str, video: str, *, resize_factor, downsample, frame_rate,
              trdimp_ckpt, epochs, batch_size, num_workers, train_frames, tune_frames,
              quick=False):
    import numpy as np
    from udmt.gui.tabs.ST_Net.pytracking.run_tracker import run_tracking
    from udmt.gui.tabs.ST_Net.ltr.run_training import run_training_process
    from udmt.gui.tabs.ST_Net.pytracking.utils.convert_to_train_set import create_train_label
    from udmt.gui.tabs.ST_Net.pytracking.utils.filter_and_save import post_process_results

    # --quick: single param point (skip the 3x3 search) for a fast steps-4-6 validation
    scales = [2.0] if quick else np.arange(1.5, 3, 0.5)
    biases = [0.0] if quick else [-0.1, 0, 0.1]

    # Step 4: training-dataset frames + param-search (status_flag=1) + label conversion
    img_dir = f"{project}/training-datasets/{stem}/img"
    _extract_frames(video, img_dir, resize_factor, downsample, cap=train_frames)
    p1 = {"video_display_widget1": None, "video_display_widget2": None,
          "project_folder": project, "video_name": stem, "model_path": trdimp_ckpt,
          "resize_factor": resize_factor, "downsample_factor": downsample,
          "frame_rate": frame_rate, "frame_num": train_frames,
          "search_scale_range": scales, "target_sz_bias_range": biases,
          "status_flag": 1, "evaluation_metric": [], "is_concave": False}
    run_tracking(p1)
    best = json.loads(Path(f"{project}/tmp/{stem}/evaluation_metric_for_train.json").read_text())[-1]
    keyword = best["target_sz"] + "_" + best["search_scale"]
    res = f"{project}/tmp/{stem}/train_set_results"
    create_train_label(stem, f"{res}/{_subfolder_with_keyword(res, keyword)}",
                       f"{project}/training-datasets/{stem}/label")

    # Step 5: train ST-Net
    import logging
    run_training_process({"project_folder": project, "video_name": stem,
                          "num_workers": num_workers, "epoch_num": epochs,
                          "batch_size": batch_size, "max_save_snapshots": 3,
                          "logger": logging.getLogger("GUI")})

    # Step 6: whole-video extract -> test-tune (status 2) -> final (status 3) -> filter
    ex_dir = f"{project}/tmp/{stem}/extracted-images"
    n_ex = _extract_frames(video, ex_dir, resize_factor, downsample, cap=None)
    model = _select_model(stem, f"{project}/models")
    p2 = {**p1, "model_path": model, "frame_num": tune_frames, "status_flag": 2,
          "search_scale_range": scales, "target_sz_bias_range": biases}
    run_tracking(p2)
    best2 = json.loads(Path(f"{project}/tmp/{stem}/evaluation_metric_for_test.json").read_text())[-1]
    p3 = {**p2, "status_flag": 3, "frame_num": n_ex - 1,
          "search_scale_range": [float(best2["search_scale"])],
          "target_sz_bias_range": [best2["target_sz_bias"]], "evaluation_metric": []}
    run_tracking(p3)

    result_dir = f"{project}/tracking-results/{stem}"
    raw = os.path.join(result_dir, os.path.basename(_newest_dir(result_dir)))
    dataset_name, save_path = post_process_results(stem, 5, raw, result_dir)
    print(f"FINAL tracks -> {save_path}")
    return save_path


def run_final_only(project: str, stem: str, video: str, *, resize_factor, downsample,
                   frame_rate, model_path, tune_frames, quick=False):
    """Diagnostic: skip XMem/dataset/train; reuse an existing project's extracted-images
    + a chosen model, run only status-2 (tune) -> status-3 (final) -> filter. Lets us
    swap trained vs pretrained model for the SAME window without a 1 h retrain."""
    import numpy as np
    from udmt.gui.tabs.ST_Net.pytracking.run_tracker import run_tracking
    from udmt.gui.tabs.ST_Net.pytracking.utils.filter_and_save import post_process_results
    scales = [2.0] if quick else np.arange(1.5, 3, 0.5)
    biases = [0.0] if quick else [-0.1, 0, 0.1]
    ex_dir = f"{project}/tmp/{stem}/extracted-images"
    n_ex = len([f for f in os.listdir(ex_dir) if f.endswith(".jpg")]) if os.path.isdir(ex_dir) else 0
    if n_ex == 0:
        n_ex = _extract_frames(video, ex_dir, resize_factor, downsample, cap=None)
    print(f"[{stem}] resume-final: {n_ex} extracted frames, model={os.path.basename(model_path)}", flush=True)
    p2 = {"video_display_widget1": None, "video_display_widget2": None,
          "project_folder": project, "video_name": stem, "model_path": model_path,
          "resize_factor": resize_factor, "downsample_factor": downsample,
          "frame_rate": frame_rate, "frame_num": tune_frames,
          "search_scale_range": scales, "target_sz_bias_range": biases,
          "status_flag": 2, "evaluation_metric": [], "is_concave": False}
    run_tracking(p2)
    best2 = json.loads(Path(f"{project}/tmp/{stem}/evaluation_metric_for_test.json").read_text())[-1]
    p3 = {**p2, "status_flag": 3, "frame_num": n_ex - 1,
          "search_scale_range": [float(best2["search_scale"])],
          "target_sz_bias_range": [best2["target_sz_bias"]], "evaluation_metric": []}
    run_tracking(p3)
    result_dir = f"{project}/tracking-results/{stem}"
    raw = os.path.join(result_dir, os.path.basename(_newest_dir(result_dir)))
    _, save_path = post_process_results(stem, 5, raw, result_dir)
    print(f"FINAL tracks -> {save_path}")
    return save_path


def build_parser():
    p = argparse.ArgumentParser(description="Headless native UDMT, Detectron2-seeded, one view.")
    p.add_argument("--udmt-root", required=True)
    p.add_argument("--project", required=True, help="UDMT project folder (…-YYYY-MM-DD)")
    p.add_argument("--video", required=True, help="window-cut video for this view")
    p.add_argument("--stem", required=True, help="video filename stem (== UDMT video_name)")
    p.add_argument("--init-mask", required=True, help="<view>_init_mask.png (orig-res 0/255)")
    p.add_argument("--init-startpos", required=True, help="<view>_start_pos.txt (orig-px x,y)")
    p.add_argument("--xmem-ckpt", required=True)
    p.add_argument("--trdimp-ckpt", required=True)
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--resize-factor", type=float, default=0.8)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--frame-rate", type=float, default=166.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--train-frames", type=int, default=2000)
    p.add_argument("--tune-frames", type=int, default=4000)
    p.add_argument("--xmem-only", action="store_true", help="stop after XMem propagation (smoke)")
    p.add_argument("--quick", action="store_true",
                   help="single param point (scale 2.0, bias 0) -> fast steps-4-6 validation")
    p.add_argument("--resume-final", action="store_true",
                   help="diagnostic: skip XMem/dataset/train, reuse extracted-images + a model, "
                        "run only tune+final (use --final-model to swap trained vs pretrained)")
    p.add_argument("--final-model", default=None,
                   help="model_path for --resume-final (default: newest trained DiMPnet in project)")
    return p


def main():
    args = build_parser().parse_args()
    _headless_env(args.udmt_root)
    import cv2
    # vis off so the per-frame loop never touches a Qt widget (tracker_multi.py:49)
    import udmt.gui.tabs.ST_Net.pytracking.evaluation.tracker_multi as tm
    tm.vis_flag = False

    if args.resume_final:
        model = args.final_model or _select_model(args.stem, f"{args.project}/models")
        save_path = run_final_only(args.project, args.stem, args.video,
                                   resize_factor=args.resize_factor, downsample=args.downsample,
                                   frame_rate=args.frame_rate, model_path=model,
                                   tune_frames=args.tune_frames, quick=args.quick)
        print(json.dumps({"stem": args.stem, "final_npy": save_path, "model": model}))
        return

    from udmt.gui.tabs.xmem.inference.interact.resource_manager import ResourceManager

    workspace = f"{args.project}/tmp/{args.stem}"
    Path(workspace + "/extracted-images").mkdir(parents=True, exist_ok=True)
    # XMem mask resolution MUST equal the ST-Net extracted-images resolution
    # (resize_factor x original), else set_ini_value reads masks + frames at different
    # scales and the DiMP template comes out the wrong size. ResourceManager.resize_scale
    # = size / min(H, W), so pick size = resize_factor * min(H, W).
    _cap = cv2.VideoCapture(args.video); _ok, _fr = _cap.read(); _cap.release()
    if not _ok:
        raise SystemExit(f"cannot read {args.video}")
    xmem_size = max(1, round(args.resize_factor * min(_fr.shape[0], _fr.shape[1])))
    print(f"[{args.stem}] xmem size={xmem_size} (resize_factor {args.resize_factor} "
          f"x min{_fr.shape[:2]}) -> masks match extracted-images", flush=True)
    config = _xmem_config(args.video, workspace, args.xmem_ckpt, args.num_fish, xmem_size)
    rm = ResourceManager(config, args.downsample)        # extracts images/%07d.jpg, sets resize_scale/h/w
    print(f"[{args.stem}] frames={rm.length} size={rm.h}x{rm.w} resize_scale={rm.resize_scale:.4f}", flush=True)

    # inject Detectron2 init: resize orig-res PER-INSTANCE label mask (0,1,2,3) to (h,w),
    # preserving labels (NEAREST, no binarisation) so XMem seeds one object per fish.
    init = cv2.imread(args.init_mask, cv2.IMREAD_GRAYSCALE)
    init_fg = cv2.resize(init, (rm.w, rm.h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    from PIL import Image
    Image.fromarray(init_fg, mode="L").save(f"{rm.mask_dir}/0000000.png")
    startpos = np.loadtxt(args.init_startpos, delimiter=",").reshape(-1, 2)
    np.savetxt(f"{workspace}/extracted-images/start_pos_array.txt", startpos, fmt="%.2f", delimiter=",")
    print(f"[{args.stem}] injected init: {startpos.shape[0]} start positions", flush=True)

    run_xmem_propagation(rm, config, args.xmem_ckpt, init_fg, args.num_fish)
    print(f"[{args.stem}] XMem propagation done -> {rm.mask_dir}", flush=True)
    if args.xmem_only:
        return

    save_path = run_stnet(args.project, args.stem, args.video,
                          resize_factor=args.resize_factor, downsample=args.downsample,
                          frame_rate=args.frame_rate, trdimp_ckpt=args.trdimp_ckpt,
                          epochs=args.epochs, batch_size=args.batch_size,
                          num_workers=args.num_workers, train_frames=args.train_frames,
                          tune_frames=args.tune_frames, quick=args.quick)
    print(json.dumps({"stem": args.stem, "final_npy": save_path}))


if __name__ == "__main__":
    main()
