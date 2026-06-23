#!/usr/bin/env python
"""Run UDMT's NATIVE auto initial prediction (SEEM / X-Decoder panoptic model) on the
first frame of a video, to generate the first-frame seed exactly as native UDMT would --
so we can confirm whether the Detectron2-seeded result differs from the real SEEM seed.

Writes <mask_dir>/0000000.png (per-instance) + <out>/extracted-images/start_pos_array.txt.
Runs on a GPU node. Usage (saion)::

    python run_seem_seed.py --udmt-root /bucket/.../UDMT --video <mp4> \
        --out /home/.../seem_seed --resize-factor 1.0
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--udmt-root", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="work dir; images/ + masks/ written here")
    p.add_argument("--frame", type=int, default=0, help="frame index in the (cut) video")
    p.add_argument("--resize-factor", type=float, default=1.0)
    p.add_argument("--config", default=None)
    p.add_argument("--model-path", default=None)
    args = p.parse_args()

    seem_dir = os.path.join(args.udmt_root, "udmt/gui/tabs/initial_masks_inference")
    xmem_dir = os.path.join(args.udmt_root, "udmt/gui/tabs/xmem")
    sys.path.insert(0, seem_dir)     # panoptic_inference + modeling/ + utils/
    sys.path.insert(0, xmem_dir)     # auto_initial_prediction

    # newer Pillow dropped Image.LINEAR/CUBIC that detectron2-xyz references at import
    from PIL import Image
    for a, b in [("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC")]:
        if not hasattr(Image, a):
            setattr(Image, a, getattr(Image, b))

    import cv2
    import numpy as np
    from auto_initial_prediction import run_auto_initial_prediction

    img_dir = os.path.join(args.out, "images")
    mask_dir = os.path.join(args.out, "masks")
    ex_dir = os.path.join(args.out, "extracted-images")
    for d in (img_dir, mask_dir, ex_dir):
        os.makedirs(d, exist_ok=True)

    # extract the requested frame at resize_factor and write as 0000000.jpg
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {args.frame} of {args.video}")
    h0, w0 = fr.shape[:2]
    w, h = int(round(w0 * args.resize_factor)), int(round(h0 * args.resize_factor))
    fr_r = cv2.resize(fr, (w, h))
    cv2.imwrite(os.path.join(img_dir, "0000000.jpg"), fr_r)
    print(f"frame {args.frame}: orig {w0}x{h0} -> seed image {w}x{h}", flush=True)

    config = args.config or os.path.join(
        seem_dir, "configs/seem/focall_unicl_lang_demo_low_threshold.yaml")
    model_path = args.model_path or os.path.join(
        args.udmt_root, "udmt/gui/pretrained/model_state_dict.pt")

    ok, n = run_auto_initial_prediction(
        image_dir=img_dir, mask_dir=mask_dir, config_file=config, model_path=model_path,
        target_size=(h, w), start_point_save_path=ex_dir, resize_scale=1.0)
    print(f"SEEM auto prediction: success={ok} num_instances={n}", flush=True)

    mpath = os.path.join(mask_dir, "0000000.png")
    if os.path.exists(mpath):
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        print(f"seed mask {mpath} shape={m.shape} unique={np.unique(m).tolist()}", flush=True)
    sp = os.path.join(ex_dir, "start_pos_array.txt")
    if os.path.exists(sp):
        print("start_pos_array.txt:\n" + open(sp).read(), flush=True)
    print("SEEM_SEED_DONE")


if __name__ == "__main__":
    main()
