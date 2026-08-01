"""Phase-1b scale/mask sensitivity pre-test (run in main .venv, no Detectron2).

Corroborates the crop-distribution hypothesis using ONLY in-distribution training crops
(``pooled_sides_corrected``, which the clean female model scores at ~0.91 margin). We feed
the SAME model the SAME crops, but rendered toward the Phase-1a crop style -- the fish
shrunk onto a black canvas (small-fish-on-black, like ``crop_from_mask``) instead of the
mask-bbox-zoomed, unmasked training framing. If female-vs-male separation collapses under
that transform, the Phase-1a "non-generalization" is (at least partly) a crop-style artifact,
not a timeline-generalization failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "phase0"))
from _segid_common import FEMALE_MODEL, device, forward, load_classifier  # noqa: E402

DATASET = Path("Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/pooled_sides_corrected")
FEMALE_ALIGNED_ID = 1   # from correction_manifest.json
N_SAMPLE = 6000


def shrink_onto_black(crops: np.ndarray, s: int) -> np.ndarray:
    out = np.zeros_like(crops)
    off = (crops.shape[1] - s) // 2
    for i, c in enumerate(crops):
        r = cv2.resize(c, (s, s), interpolation=cv2.INTER_AREA)
        out[i, off:off + s, off:off + s] = r
    return out


def sep_stats(p_female: np.ndarray, gid: np.ndarray) -> dict:
    fem = p_female[gid == FEMALE_ALIGNED_ID]
    male = p_female[gid != FEMALE_ALIGNED_ID]
    # AUC = P(random female crop scored > random male crop): clean separability measure
    order = np.argsort(np.concatenate([fem, male]))
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(len(order))
    auc = (ranks[:len(fem)].sum() - len(fem) * (len(fem) - 1) / 2) / (len(fem) * len(male))
    frac_all_high = float((p_female > 0.7).mean())
    return {"mean_p_female_FEMALE": round(float(fem.mean()), 3),
            "mean_p_female_MALE": round(float(male.mean()), 3),
            "female_minus_male": round(float(fem.mean() - male.mean()), 3),
            "auc_female_vs_male": round(float(auc), 3),
            "frac_crops_p>0.7": round(frac_all_high, 3)}


def main() -> None:
    dev = device()
    meta = pd.read_parquet(DATASET / "crop_metadata.parquet").reset_index(drop=True)
    with h5py.File(DATASET / "crops.h5", "r") as h:
        key = "crops" if "crops" in h else list(h.keys())[0]
        n = h[key].shape[0]
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=min(N_SAMPLE, n), replace=False))
        crops = np.asarray(h[key][idx], np.uint8)
    gid = meta.global_id.to_numpy()[idx]

    model = load_classifier(FEMALE_MODEL, dev)
    results = {}
    # baseline: training-style crops (what the model was trained + Phase-0-validated on)
    results["baseline_training_style"] = sep_stats(forward(model, crops, dev, head=True), gid)
    # shrink the fish onto black at decreasing sizes -> toward Phase-1a small-fish-on-black look
    for s in [64, 48, 36]:
        sc = shrink_onto_black(crops, s)
        results[f"shrunk_onto_black_{s}px"] = sep_stats(forward(model, sc, dev, head=True), gid)

    print(json.dumps(results, indent=2))
    (HERE / "p1b_scale_pretest.json").write_text(json.dumps(results, indent=2))
    base = results["baseline_training_style"]["auc_female_vs_male"]
    worst = min(r["auc_female_vs_male"] for k, r in results.items() if k != "baseline_training_style")
    print(f"\n[pretest] female-vs-male AUC: baseline {base} -> shrunk-onto-black {worst}", flush=True)
    print("[pretest] big drop => the model is crop-distribution sensitive (scale/mask), "
          "so Phase-1a's collapse is confounded by crop style, not only timeline window.")


if __name__ == "__main__":
    main()
