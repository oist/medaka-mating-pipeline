from __future__ import annotations

"""Self-supervised female slot-pick: identify the female WITHOUT reusing the
K_1_2-trained classifier, by training a per-experiment appearance embedder on the
experiment's OWN crops and asking which of the three geometry slots is the single
appearance-distinct fish.

Motivation: the reused K_1_2 female classifier does not transfer to other
experiments (per-window female slot flips, pooled P(female) ~ uniform). The seg-ID
pilot found that appearance reliably isolates exactly ONE fish -- the gravid female
-- onto its own cluster while the two males collapse together. So instead of a
transferred supervised label, we:

  1. train a contrastive ResNet18 embedder on this experiment's pooled crops
     (must-link = same slot/window, cannot-link = same-frame different slot),
  2. score each slot's ONE-VS-REST separability in that embedding,
  3. the female = the most-isolated slot, IF it is clearly separable AND the two
     remaining (male) slots are NOT separable from each other (the biological
     signature), AND the choice is consistent across windows (not a per-window
     artifact).

The per-window consistency check is the direct antidote to the reused-classifier
failure mode. We report every number and a conservative verdict; a sexed CSV is
written only when ``--identified-3d-csv`` is given AND the call is confident
(unless ``--force-write``).

Run::

    python -m sam2_id_experiment.scripts.41_self_supervised_slot_pick \
        --dataset <pooled_dir> --out-dir <slot_pick_dir> \
        --identified-3d-csv <..._3d_identified.csv>
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sam2_id_experiment.src.id_model.resnet_embedder import ResNet18Embedder
from sam2_id_experiment.src.id_model.train_resnet import train_resnet_embedding
from sklearn.metrics import silhouette_score


def _embed(dataset_dir: Path, model_path: Path, device: str, batch_size: int = 256) -> np.ndarray:
    ckpt = torch.load(model_path, map_location=device)
    model = ResNet18Embedder(embedding_dim=int(ckpt.get("embedding_dim", 128)), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    with h5py.File(dataset_dir / "crops.h5", "r") as h:
        crops = np.asarray(h["crops"][...], dtype=np.float32) / 255.0
    out = []
    with torch.no_grad():
        for s in range(0, len(crops), batch_size):
            b = torch.from_numpy(crops[s:s + batch_size]).permute(0, 3, 1, 2).to(device)
            out.append(model(b).cpu().numpy())
    return np.concatenate(out, axis=0)


def _l2(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def _subsample(idx: np.ndarray, labels: np.ndarray, per_class: int, rng) -> np.ndarray:
    """Stratified subsample so silhouette stays O(n^2)-tractable and slot-balanced."""
    keep = []
    for c in np.unique(labels):
        ci = idx[labels == c]
        keep.append(ci if len(ci) <= per_class else rng.choice(ci, per_class, replace=False))
    return np.concatenate(keep)


def _ovr_silhouettes(X: np.ndarray, slot: np.ndarray) -> dict[int, float]:
    """One-vs-rest silhouette per slot: how isolated is slot s from the other two."""
    out = {}
    for s in np.unique(slot):
        binary = (slot == s).astype(int)
        if len(np.unique(binary)) < 2:
            out[int(s)] = float("nan")
        else:
            out[int(s)] = float(silhouette_score(X, binary))
    return out


def analyse(emb: np.ndarray, meta_clean: pd.DataFrame, *, sample_per_slot: int,
            seed: int) -> dict:
    rng = np.random.default_rng(seed)
    X_all = _l2(emb[meta_clean.index.to_numpy()])
    slot_all = meta_clean.global_id.to_numpy()
    window_all = meta_clean.window.to_numpy() if "window" in meta_clean else np.array(["all"] * len(slot_all))

    pos = np.arange(len(slot_all))
    samp = _subsample(pos, slot_all, sample_per_slot, rng)
    X, slot = X_all[samp], slot_all[samp]

    overall_sil = float(silhouette_score(X, slot)) if len(np.unique(slot)) >= 2 else float("nan")
    ovr = _ovr_silhouettes(X, slot)
    female_slot = int(max(ovr, key=lambda k: (ovr[k] if np.isfinite(ovr[k]) else -1)))
    males = [s for s in ovr if s != female_slot]

    # male-vs-male separability (should be LOW if the biology holds)
    mm_mask = np.isin(slot, males)
    male_male_sil = (float(silhouette_score(X[mm_mask], slot[mm_mask]))
                     if mm_mask.sum() > 2 and len(np.unique(slot[mm_mask])) == 2 else float("nan"))

    # per-window argmax slot (consistency vs the reused-classifier flip)
    per_window = {}
    for w in np.unique(window_all):
        wpos = pos[window_all == w]
        if len(wpos) < 30:
            continue
        wsamp = _subsample(wpos, slot_all[wpos], min(sample_per_slot, 800), rng)
        Xw, sw = X_all[wsamp], slot_all[wsamp]
        if len(np.unique(sw)) < 3:
            continue
        wovr = _ovr_silhouettes(Xw, sw)
        per_window[str(w)] = {int(k): round(v, 4) for k, v in wovr.items()}
    window_argmax = [max(v, key=v.get) for v in per_window.values()]
    consistency = (window_argmax.count(female_slot) / len(window_argmax)) if window_argmax else float("nan")

    female_ovr = ovr[female_slot]
    male_ovr_max = max((ovr[s] for s in males), default=float("nan"))
    gap = female_ovr - male_ovr_max
    # Conservative verdict: female clearly isolated, males not separable from each
    # other, and the pick agrees across most windows.
    separable = bool(
        np.isfinite(female_ovr) and female_ovr >= 0.15 and gap >= 0.10
        and (not np.isfinite(male_male_sil) or male_male_sil < female_ovr)
        and (not np.isfinite(consistency) or consistency >= 0.6)
    )
    return {
        "female_slot": female_slot,
        "female_fish_id": f"ID_{female_slot + 1}",
        "separable": separable,
        "overall_silhouette_3slot": round(overall_sil, 4),
        "one_vs_rest_silhouette": {int(k): round(v, 4) for k, v in ovr.items()},
        "female_ovr": round(female_ovr, 4),
        "male_ovr_max": round(male_ovr_max, 4),
        "female_minus_male_gap": round(gap, 4),
        "male_male_silhouette": round(male_male_sil, 4) if np.isfinite(male_male_sil) else None,
        "per_window_argmax_consistency": round(consistency, 3) if np.isfinite(consistency) else None,
        "per_window_one_vs_rest": per_window,
        "n_crops_scored": int(len(samp)),
        "slot_crop_counts": {int(k): int(v) for k, v in
                             pd.Series(slot_all).value_counts().sort_index().items()},
        "verdict_rule": "separable if female_ovr>=0.15 AND gap>=0.10 AND males collapse AND window-consistency>=0.6",
    }


def _pca_plot(emb, meta_clean, female_slot, out_png, seed):
    try:
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(seed)
        X = _l2(emb[meta_clean.index.to_numpy()])
        slot = meta_clean.global_id.to_numpy()
        samp = _subsample(np.arange(len(slot)), slot, 2000, rng)
        P = PCA(n_components=2).fit_transform(X[samp])
        fig, ax = plt.subplots(figsize=(6, 5))
        for s in sorted(np.unique(slot[samp])):
            mk = slot[samp] == s
            lbl = f"ID_{s+1}" + (" (female?)" if s == female_slot else " (male?)")
            ax.scatter(P[mk, 0], P[mk, 1], s=5, alpha=0.4, label=lbl)
        ax.legend(); ax.set_title("self-supervised appearance embedding (PCA) by geometry slot")
        fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    except Exception as e:
        print("pca plot skipped:", e)


def build_parser():
    p = argparse.ArgumentParser(description="Self-supervised female slot-pick (no reused classifier).")
    p.add_argument("--dataset", required=True, help="pooled crops dir (crops.h5 + crop_metadata.parquet + pairs.parquet)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--identified-3d-csv", default=None, help="if given (and confident), write a sexed CSV")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sample-per-slot", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reuse-embedder", action="store_true", help="skip training if model.pt exists")
    p.add_argument("--force-write", action="store_true", help="write sexed CSV even if not confident")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    emb_dir = out / "embedder"
    model_path = emb_dir / "model.pt"

    if not (args.reuse_embedder and model_path.exists()):
        print(f"[slot-pick] training embedder ({args.epochs} ep) on {dataset.name} ...", flush=True)
        train_resnet_embedding(str(dataset), str(emb_dir), epochs=args.epochs,
                               batch_size=args.batch_size, lr=args.lr,
                               embedding_dim=args.embedding_dim, device=args.device)
    else:
        print(f"[slot-pick] reusing embedder {model_path}")

    emb = _embed(dataset, model_path, args.device)
    meta = pd.read_parquet(dataset / "crop_metadata.parquet").reset_index(drop=True)
    assert len(meta) == len(emb), f"meta/emb mismatch {len(meta)} vs {len(emb)}"
    clean = meta[(meta.qc_status == "clean_train") & (meta.global_id >= 0)].copy()

    report = analyse(emb, clean, sample_per_slot=args.sample_per_slot, seed=args.seed)
    _pca_plot(emb, clean, report["female_slot"], out / "slot_pick_pca.png", args.seed)
    (out / "slot_pick_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\n[slot-pick] female = {report['female_fish_id']} "
          f"(ovr_sil {report['female_ovr']}, gap {report['female_minus_male_gap']}, "
          f"window-consistency {report['per_window_argmax_consistency']}) "
          f"-> {'CONFIDENT' if report['separable'] else 'NOT confident (appearance does not cleanly isolate a female)'}")

    if args.identified_3d_csv and Path(args.identified_3d_csv).exists():
        if report["separable"] or args.force_write:
            df = pd.read_csv(args.identified_3d_csv)
            fslot = report["female_slot"]
            df["sex"] = np.where(df["fish_id"] == f"ID_{fslot+1}", "F", "M")
            df["female_slot_method"] = "self_supervised_slot_pick"
            dst = Path(args.identified_3d_csv)
            outcsv = out / (dst.stem + "_sexed_selfsup.csv")
            df.to_csv(outcsv, index=False)
            report["sexed_csv"] = str(outcsv)
            report["sexed_csv_confident"] = bool(report["separable"])
            (out / "slot_pick_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[slot-pick] wrote sexed CSV -> {outcsv} (confident={report['separable']})")
        else:
            print("[slot-pick] NOT writing sexed CSV: female not confidently separable "
                  "(pass --force-write to override).")


if __name__ == "__main__":
    main()
