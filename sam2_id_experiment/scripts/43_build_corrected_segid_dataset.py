from __future__ import annotations

"""Stage B-fix: turn a multi-window seg-ID pool into a SWAP-CORRECTED, co-visible
training dataset for the per-video identity / female-anchor classifier.

The raw 3D-bind ``global_id`` is reliable within a window but permuted between the
scattered windows (cross-window RAW ~chance vs Hungarian-REMAP ~1.0). Pooling under
the raw label mixes individuals and was the dominant cause of the weak female model
(0.72 within-video, ~chance on transfer). This script:

  1. keeps only co-visible frames (all ``num_fish`` fish present at once),
  2. embeds the crops with a frozen self-supervised embedder (never trained on identity
     labels), and aligns every window's per-fish centroids to the most-separated
     reference window (Hungarian bijection) -> a cross-window-consistent ``aligned`` id,
  3. drops bad-apple windows whose fish do not separate,
  4. writes a drop-in dataset (``crops.h5`` + ``crop_metadata.parquet`` with the
     corrected ``global_id`` = aligned id) that ``37_train_female_classifier`` /
     ``35_eval_segid_embedding`` consume unchanged, plus a ``correction_manifest.json``.

The female designation (one bit) is anchored by ``--anchor-female-gid`` (the female's
raw global_id in the un-swapped majority of windows); the resulting female *aligned*
id is reported in the manifest and is what you pass to ``37 --female-gid``.

Run::

    python -m sam2_id_experiment.scripts.43_build_corrected_segid_dataset \
        --dataset Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/pooled_sides_full \
        --embedder Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/model_resnet_full/model.pt \
        --out-dir  Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/pooled_sides_corrected \
        --anchor-female-gid 1
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

from sam2_id_experiment.src.id_model.resnet_embedder import ResNet18Embedder
from sam2_id_experiment.src.id_model.window_align import (
    align_windows_by_appearance,
    select_covisible_mask,
)


def embed(crops_uint8: np.ndarray, model_path: str, device: str, batch_size: int = 512) -> np.ndarray:
    ck = torch.load(model_path, map_location=device)
    model = ResNet18Embedder(embedding_dim=int(ck.get("embedding_dim", 128)), pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(crops_uint8), batch_size):
            x = torch.from_numpy(crops_uint8[s:s + batch_size].astype(np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2).to(device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, 128))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a swap-corrected, co-visible seg-ID training dataset.")
    p.add_argument("--dataset", required=True, help="pooled dataset dir (crops.h5 + crop_metadata.parquet)")
    p.add_argument("--embedder", required=True, help="frozen self-supervised embedder model.pt (alignment only)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--drop-sep-frac", type=float, default=0.5,
                   help="drop a window if its fish separate by < this fraction of the reference window")
    p.add_argument("--anchor-female-gid", type=int, default=1,
                   help="the female's raw global_id in the un-swapped majority of windows")
    p.add_argument("--no-require-covisible", action="store_true",
                   help="keep all clean_train crops, not only co-visible (all fish present) frames")
    p.add_argument("--device", default="cuda")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ds = Path(args.dataset)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    meta = pd.read_parquet(ds / "crop_metadata.parquet").reset_index(drop=True)
    meta["row"] = np.arange(len(meta))
    with h5py.File(ds / "crops.h5", "r") as h:
        key = "crops" if "crops" in h else list(h.keys())[0]
        crops = np.asarray(h[key][...], dtype=np.uint8)
    assert len(crops) == len(meta), "crops/meta length mismatch"

    valid = meta[(meta.qc_status == "clean_train") & (meta.global_id.isin(range(args.num_fish)))].copy()
    if not args.no_require_covisible:
        mask = select_covisible_mask(valid.view, valid.frame, valid.global_id, num_fish=args.num_fish)
        valid = valid[mask]
    valid = valid.reset_index(drop=True)
    print(f"clean co-visible crops: {len(valid)}  windows: {sorted(valid.window.unique())}", flush=True)

    emb = embed(crops[valid.row.to_numpy()], args.embedder, device)
    aligned, info = align_windows_by_appearance(
        emb, valid.window.to_numpy(), valid.global_id.to_numpy(),
        num_fish=args.num_fish, drop_sep_frac=args.drop_sep_frac)
    valid["aligned"] = aligned
    if info.get("degenerate") or not (valid.aligned >= 0).any():
        raise SystemExit("alignment degenerate: the embedder does not separate the fish in any "
                         "window (every window dropped). Check the embedder / co-visible crops.")

    # female aligned id = the aligned id the anchor-female-gid maps to in the kept (un-dropped) windows
    anchor = valid[(valid.global_id == args.anchor_female_gid) & (valid.aligned >= 0)]
    female_votes = anchor.aligned.value_counts()
    female_aligned = int(female_votes.index[0]) if len(female_votes) else -1

    kept = valid[valid.aligned >= 0].reset_index(drop=True)
    out_meta = kept.copy()
    out_meta["raw_global_id"] = out_meta["global_id"]
    out_meta["global_id"] = out_meta["aligned"].astype(int)   # corrected, cross-window-consistent
    out_meta["global_id_purity"] = 1.0
    out_meta["crop_id"] = np.arange(len(out_meta))

    with h5py.File(out / "crops.h5", "w") as h:
        h.create_dataset("crops", data=crops[kept.row.to_numpy()], compression="gzip")
    out_meta.drop(columns=["row", "aligned"]).to_parquet(out / "crop_metadata.parquet", index=False)

    manifest = {
        "source_dataset": str(ds),
        "embedder": args.embedder,
        "num_fish": args.num_fish,
        "require_covisible": not args.no_require_covisible,
        "reference_window": info["ref"],
        "window_separations": {k: round(v, 3) for k, v in info["seps"].items()},
        "dropped_windows": info["dropped"],
        "drop_threshold": round(info["thr"], 3),
        "anchor_female_gid": args.anchor_female_gid,
        "female_aligned_id": female_aligned,
        "female_anchor_vote_counts": {int(k): int(v) for k, v in female_votes.items()},
        "n_windows": info.get("n_windows"),
        "n_crops_out": int(len(out_meta)),
        "per_aligned_id_counts": {int(k): int(v) for k, v in out_meta.global_id.value_counts().items()},
        "note": "train with: 37_train_female_classifier --dataset <out> --female-gid <female_aligned_id> "
                "--min-global-id-purity 0",
    }
    (out / "correction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote corrected dataset -> {out}\n  -> 37 --female-gid {female_aligned}", flush=True)


if __name__ == "__main__":
    main()
