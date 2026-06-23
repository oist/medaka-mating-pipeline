from __future__ import annotations

"""Female-anchor production filter: turn per-crop female probabilities into a single
sex label per 3D fish identity, robustly and conservatively.

Only the VALIDATED capability is used -- female vs male (the two males are never
distinguished here). Aggregating many crops per fish makes the per-fish call robust
to the ~0.83 per-crop cross-window accuracy (gravid-state drift): hundreds of crops
vote, so the per-fish margin is large. Exactly one fish is labelled female.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FemaleCall:
    female_fish: object              # the id labelled female
    female_prob: dict                # fish_id -> mean female probability
    n_crops: dict                    # fish_id -> crops voted
    confidence_margin: float         # top female_prob - 2nd
    ambiguous: bool                  # margin below threshold -> needs review


def aggregate_female_prob(crops: pd.DataFrame, group_col: str,
                          prob_col: str = "female_prob") -> pd.DataFrame:
    """Mean female probability per group (fish/tracklet/slot)."""
    g = crops.groupby(group_col)[prob_col]
    return pd.DataFrame({"mean_female_prob": g.mean(), "n_crops": g.size()}).reset_index()


def assign_single_female(per_fish_prob: dict, n_crops: dict | None = None,
                         margin_threshold: float = 0.15) -> FemaleCall:
    """Pick exactly one female = argmax mean female probability. Flags the call
    ambiguous when the top-vs-second margin is below ``margin_threshold`` (then a
    human / geometry tie-break is warranted)."""
    if not per_fish_prob:
        return FemaleCall(None, {}, {}, float("nan"), True)
    items = sorted(per_fish_prob.items(), key=lambda kv: kv[1], reverse=True)
    female_fish = items[0][0]
    margin = float(items[0][1] - items[1][1]) if len(items) > 1 else float(items[0][1])
    return FemaleCall(
        female_fish=female_fish,
        female_prob={k: float(v) for k, v in per_fish_prob.items()},
        n_crops={k: int((n_crops or {}).get(k, 0)) for k in per_fish_prob},
        confidence_margin=round(margin, 4),
        ambiguous=bool(margin < margin_threshold),
    )


def sex_labels(per_fish_prob: dict, female_fish) -> dict:
    """fish_id -> 'F' for the female, 'M' otherwise."""
    return {k: ("F" if k == female_fish else "M") for k in per_fish_prob}


def predict_female_prob(model, crops_uint8: np.ndarray, device: str = "cuda",
                        batch_size: int = 256) -> np.ndarray:
    """Per-crop P(female) from a 2-logit EmbeddingClassifier. crops: (N,H,W,3) uint8."""
    import torch
    model.eval()
    probs = []
    x = crops_uint8.astype(np.float32) / 255.0
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            b = torch.from_numpy(x[s:s + batch_size]).permute(0, 3, 1, 2).to(device)
            logits = model(b)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])
