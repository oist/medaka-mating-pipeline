"""Are the same 3 individuals used across K_1_1 / K_1_2 / K_1_3? Embed each experiment's 3 fish
(window_align aligned-ids) with the frozen self-supervised embedder, get 9 fish-centroids, and test
whether they link cross-trial by INDIVIDUAL (3 groups of 3, one per trial = same individuals), by
EXPERIMENT (domain shift, inconclusive), or just by SEX (female-shape vs male-shape)."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sam2_id_experiment.src.id_model.resnet_embedder import ResNet18Embedder  # noqa: E402

EMBEDDER = r"Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/model_resnet_full/model.pt"
# female aligned-id per experiment (K_1_2 from the original work; K_1_1/K_1_3 from the corrected fix)
FEMALE_ALIGNED = {"K_1_1": 0, "K_1_2": 1, "K_1_3": 0}
DS = {e: rf"Z:/StephensU/medaka/runs/udmt_validation/segid/{e}/pooled_sides_corrected" for e in FEMALE_ALIGNED}


def embed(crops, model, dev, bs=512):
    out = []
    with torch.no_grad():
        for s in range(0, len(crops), bs):
            x = torch.from_numpy(crops[s:s + bs].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
            out.append(torch.nn.functional.normalize(model(x), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 128))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(EMBEDDER, map_location=dev)
    model = ResNet18Embedder(embedding_dim=int(ck.get("embedding_dim", 128)), pretrained=False).to(dev)
    model.load_state_dict(ck["model_state"]); model.eval()

    labels, cents = [], []
    rng = np.random.RandomState(0)
    for e, ds in DS.items():
        meta = pd.read_parquet(Path(ds) / "crop_metadata.parquet")
        with h5py.File(Path(ds) / "crops.h5", "r") as h:
            crops = np.asarray(h["crops"][...], np.uint8)
        cid = meta["crop_id"].to_numpy() if "crop_id" in meta else np.arange(len(meta))
        for g in sorted(meta.global_id.unique()):
            idx = meta.index[meta.global_id == g].to_numpy()
            if len(idx) > 3000:
                idx = rng.choice(idx, 3000, replace=False)
            emb = embed(crops[cid[idx].astype(int)], model, dev)
            role = "F" if g == FEMALE_ALIGNED[e] else "M"
            labels.append(f"{e}:{role}{g}")
            cents.append(emb.mean(0))
    C = np.array(cents)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    S = C @ C.T  # cosine similarity (9x9)

    print("=== 9 fish-centroid cosine similarity ===")
    print("            " + "  ".join(f"{l:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>11} " + "  ".join(f"{S[i,j]:9.2f}" for j in range(len(labels))))

    print("\n=== each fish's nearest fish in EACH OTHER experiment ===")
    exps = list(DS)
    for i, li in enumerate(labels):
        ei = li.split(":")[0]
        line = [li]
        for ej in exps:
            if ej == ei:
                continue
            cand = [j for j, lj in enumerate(labels) if lj.startswith(ej + ":")]
            j = max(cand, key=lambda j: S[i, j])
            line.append(f"{ej}->{labels[j].split(':')[1]} ({S[i,j]:.2f})")
        print("  " + "  |  ".join(line))

    # female-vs-male: does each female's nearest cross-trial fish = the other female?
    fem = [i for i, l in enumerate(labels) if ":F" in l]
    print("\n=== female cross-trial linkage ===")
    for i in fem:
        ei = labels[i].split(":")[0]
        others = [j for j in range(len(labels)) if not labels[j].startswith(ei + ":")]
        j = max(others, key=lambda j: S[i, j])
        print(f"  {labels[i]} nearest cross-trial = {labels[j]} (sim {S[i,j]:.2f})  "
              f"{'-> FEMALE (consistent w/ same female)' if ':F' in labels[j] else '-> a MALE'}")
    print("\nReading: if each female's nearest cross-trial fish is the OTHER female AND the 9 split into")
    print("3 individual-groups (not 3 experiment-groups), the same 3 fish were reused across trials.")


if __name__ == "__main__":
    main()
