from __future__ import annotations

"""Stage D-fix: apply the swap-corrected female model to a target recording and call
the female robustly, handling the target's own cross-window label swaps.

For a new recording the 3D-bind ``global_id`` is again permuted between windows, so
naively averaging P(female) per raw ``global_id`` across windows mixes fish (that is
what made the old transfer land at chance). This script:

  1. selects co-visible crops, embeds them with the frozen self-supervised embedder,
     and aligns the target's windows to a reference (Hungarian bijection) -> a
     cross-window-consistent ``aligned`` id;
  2. applies the clean female classifier -> per-crop P(female);
  3. aggregates P(female) per aligned id and calls exactly one female
     (``female_anchor.assign_single_female``), reporting the confidence margin and
     per-segment consistency;
  4. writes, per kept window, which RAW ``global_id`` is the female -- the mapping a
     swap-prone 3D track needs to be sexed correctly window-by-window.

Run::

    python -m sam2_id_experiment.scripts.44_apply_clean_female_anchor \
        --target-dataset runs/female_anchor/<exp>/round5_260611/pooled \
        --model     C:/udmt_validation/segid/K_1_2/model_female_clean/model.pt \
        --embedder  C:/udmt_validation/segid/K_1_2/model_resnet_full/model.pt \
        --out-dir   runs/female_anchor/<exp>/round5_260611/female_anchor_clean
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

from sam2_id_experiment.src.id_model.female_anchor import (
    aggregate_female_prob,
    assign_single_female,
    sex_labels,
)
from sam2_id_experiment.src.id_model.resnet_embedder import EmbeddingClassifier, ResNet18Embedder
import re

from sam2_id_experiment.src.id_model.reproject_slot_points import slot_from_fish_id
from sam2_id_experiment.src.id_model.window_align import (
    align_windows_by_appearance,
    select_covisible_mask,
    sex_track_from_window_anchors,
)

_WIN_RE = re.compile(r"win_(\d+)_(\d+)")


def _load_embedder(path: str, device: str) -> ResNet18Embedder:
    ck = torch.load(path, map_location=device)
    m = ResNet18Embedder(embedding_dim=int(ck.get("embedding_dim", 128)), pretrained=False).to(device)
    m.load_state_dict(ck["model_state"])
    return m.eval()


def _load_classifier(path: str, device: str) -> EmbeddingClassifier:
    ck = torch.load(path, map_location=device)
    m = EmbeddingClassifier(ResNet18Embedder(embedding_dim=int(ck.get("embedding_dim", 128)), pretrained=False), 2)
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval()


def _forward(model, crops_uint8, device, head: bool, batch_size: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(crops_uint8), batch_size):
            x = torch.from_numpy(crops_uint8[s:s + batch_size].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
            z = torch.softmax(model(x), 1)[:, 1] if head else model(x)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply the clean female model to a target recording.")
    p.add_argument("--target-dataset", required=True, help="pooled crops dir (crops.h5 + crop_metadata.parquet)")
    p.add_argument("--model", required=True, help="clean female classifier model.pt (from 37 on corrected data)")
    p.add_argument("--embedder", required=True, help="frozen self-supervised embedder model.pt (alignment)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--drop-sep-frac", type=float, default=0.5)
    p.add_argument("--subsample-per-group", type=int, default=800, help="crops/(window,slot) for the call (0=all)")
    p.add_argument("--margin-threshold", type=float, default=0.15)
    p.add_argument("--identified-3d-csv", default=None,
                   help="if given, write a whole-track sexed CSV by following the female's slot via the window anchors")
    p.add_argument("--sexed-csv-out", default=None, help="default: <out-dir>/<identified stem>_sexed.csv")
    p.add_argument("--device", default="cuda")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    ds = Path(args.target_dataset)

    meta = pd.read_parquet(ds / "crop_metadata.parquet").reset_index(drop=True)
    meta["row"] = np.arange(len(meta))
    valid = meta[(meta.qc_status == "clean_train") & (meta.global_id.isin(range(args.num_fish)))].copy()
    mask = select_covisible_mask(valid.view, valid.frame, valid.global_id, num_fish=args.num_fish)
    valid = valid[mask]
    if args.subsample_per_group > 0:
        valid = pd.concat([d.sample(min(args.subsample_per_group, len(d)), random_state=0)
                           for _, d in valid.groupby(["window", "global_id"])])
    valid = valid.reset_index(drop=True)
    rows = np.unique(valid.row.to_numpy())            # sorted + unique (h5py fancy-index requirement)
    rowpos = {r: i for i, r in enumerate(rows)}
    with h5py.File(ds / "crops.h5", "r") as h:
        key = "crops" if "crops" in h else list(h.keys())[0]
        crops = np.asarray(h[key][rows], dtype=np.uint8)
    li = valid.row.map(rowpos).to_numpy()

    emb = _forward(_load_embedder(args.embedder, device), crops[li], device, head=False)
    aligned, info = align_windows_by_appearance(
        emb, valid.window.to_numpy(), valid.global_id.to_numpy(),
        num_fish=args.num_fish, drop_sep_frac=args.drop_sep_frac)
    valid["aligned"] = aligned
    valid["female_prob"] = _forward(_load_classifier(args.model, device), crops[li], device, head=True)

    kept = valid[valid.aligned >= 0].copy()
    agg = aggregate_female_prob(kept, group_col="aligned")
    per_fish = dict(zip(agg.aligned, agg.mean_female_prob))
    n_crops = dict(zip(agg.aligned, agg.n_crops))
    call = assign_single_female(per_fish, n_crops, margin_threshold=args.margin_threshold)
    sexes = sex_labels(per_fish, call.female_fish)

    # per-window female RAW global_id (which 3D slot is the female in each window)
    per_window_female_raw = {}
    for w, g in kept.groupby("window"):
        sub = g.groupby("global_id").aligned.agg(lambda s: s.mode().iloc[0])
        fem_raw = [int(raw) for raw, al in sub.items() if al == call.female_fish]
        per_window_female_raw[str(w)] = fem_raw[0] if fem_raw else None
    winners = kept.groupby(["window", "aligned"]).female_prob.mean().reset_index()
    seg_winner = winners.sort_values("female_prob").groupby("window").tail(1).set_index("window").aligned.to_dict()

    # ------- whole-track sexed CSV: follow the female's geometry slot via the anchors ------- #
    sexed_csv = None
    sex_conf_counts = None
    if args.identified_3d_csv and call.female_fish is not None:
        track = pd.read_csv(args.identified_3d_csv)
        track_slot = track["fish_id"].map(slot_from_fish_id).to_numpy()
        spans = []
        for w, fem_raw in per_window_female_raw.items():
            m = _WIN_RE.search(str(w))
            if m and fem_raw is not None:
                spans.append((int(m.group(1)), int(m.group(2)), int(fem_raw)))
        global_fem = int(pd.Series([s[2] for s in spans]).mode().iloc[0]) if spans else None
        sex, conf = sex_track_from_window_anchors(track["frame"].to_numpy(), track_slot, spans, global_fem)
        track["geometry_slot"] = track_slot
        track["sex"] = sex
        track["sex_confidence"] = conf
        track["female_prob"] = round(float(per_fish.get(call.female_fish, float("nan"))), 4)
        sexed_csv = str(Path(args.sexed_csv_out) if args.sexed_csv_out
                        else out / (Path(args.identified_3d_csv).stem + "_sexed.csv"))
        track.to_csv(sexed_csv, index=False)
        sex_conf_counts = {k: int(v) for k, v in pd.Series(conf).value_counts().items()}

    report = {
        "target_dataset": str(ds),
        "model": args.model,
        "reference_window": info["ref"],
        "dropped_windows": info["dropped"],
        "degenerate": info.get("degenerate", False),
        "n_windows": info.get("n_windows"),
        "female_aligned_id": int(call.female_fish) if call.female_fish is not None else None,
        "female_prob_by_aligned_id": {int(k): round(float(v), 4) for k, v in per_fish.items()},
        "confidence_margin": call.confidence_margin,
        "ambiguous": call.ambiguous,
        "sex_by_aligned_id": {int(k): v for k, v in sexes.items()},
        "per_segment_female_winner": {str(k): int(v) for k, v in seg_winner.items()},
        "female_call_consistent": len(set(seg_winner.values())) == 1,
        "per_window_female_raw_global_id": per_window_female_raw,
        "n_crops_used": int(len(kept)),
        "sexed_csv": sexed_csv,
        "sex_confidence_counts": sex_conf_counts,
    }
    (out / "female_anchor_clean_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    kept.drop(columns=["row"]).to_parquet(out / "female_prob_by_crop.parquet", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
