"""Refine fish_id labels on one df using a trained identity embedder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .crops import extract_crops_from_video
from .fragments import Fragment, build_fragments, mine_pairs
from .model import IdentityEmbedder
from .train import TrainConfig, embed_all, train_embedder


@dataclass
class RefineResult:
    df: pd.DataFrame
    cluster_to_fish: dict[int, str]  # e.g. {0: "a", 1: "b", 2: "c"}
    confidence: float  # mean overlap with Hungarian labels (0..1)


def _cluster_fragments(
    fragment_embeddings: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Return per-fragment cluster label from k-means in embedding space."""
    if len(fragment_embeddings) < n_clusters:
        return np.zeros(len(fragment_embeddings), dtype=int)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    return km.fit_predict(fragment_embeddings)


def _majority_cluster_to_fish(
    fragments: list[Fragment],
    fragment_clusters: np.ndarray,
    fish_ids: list[str],
) -> tuple[dict[int, str], float]:
    """
    For each cluster, assign the fish_id that the Hungarian tracker used most
    often inside fragments of that cluster. Returns the mapping plus the
    mean fragment-weighted agreement.
    """
    # Count cluster/fish_id co-occurrences weighted by fragment length.
    counts: dict[tuple[int, str], int] = {}
    cluster_total: dict[int, int] = {}
    for frag, cluster in zip(fragments, fragment_clusters):
        key = (int(cluster), frag.fish_id)
        counts[key] = counts.get(key, 0) + len(frag)
        cluster_total[int(cluster)] = cluster_total.get(int(cluster), 0) + len(frag)

    mapping: dict[int, str] = {}
    used_fish: set[str] = set()
    # Greedy assignment: strongest cluster-fish pair first; ensures 1:1 when
    # possible, which matches N=3 animals in this setup.
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    for (cluster, fish), _ in ranked:
        if cluster in mapping or fish in used_fish:
            continue
        mapping[cluster] = fish
        used_fish.add(fish)
    # Any unmapped cluster gets whatever fish_id remains (shouldn't happen
    # when N_clusters == N_fish, but be defensive).
    leftover = [f for f in fish_ids if f not in used_fish]
    for cluster in range(len(cluster_total)):
        if cluster not in mapping and leftover:
            mapping[cluster] = leftover.pop(0)

    # Agreement = fraction of fragment-weighted rows that already match the
    # mapping (i.e. Hungarian was probably right on those).
    total_rows = sum(len(f) for f in fragments) or 1
    agreeing = sum(
        len(f) for f, c in zip(fragments, fragment_clusters)
        if mapping.get(int(c)) == f.fish_id
    )
    return mapping, agreeing / total_rows


def refine_fish_ids(
    tracked_df: pd.DataFrame,
    video_path: str | Path,
    *,
    n_fish: int = 3,
    crop_size: int = 96,
    frame_offset: int = 0,
    train_config: TrainConfig | None = None,
    rng_seed: int = 0,
    min_confidence: float = 0.6,
    save_model_path: str | Path | None = None,
) -> RefineResult:
    """
    Train a per-clip identity embedder and rewrite `fish_id` labels.

    Strategy:
      1. Extract one crop per row from `video_path`.
      2. Build fragments from the Hungarian `fish_id` labels.
      3. Mine same/diff pairs; train the embedder.
      4. Embed all crops, average per fragment.
      5. k-means(k=`n_fish`) on fragment means, map clusters to fish_ids by
         majority vote against the Hungarian labels.
      6. Overwrite `fish_id` with the cluster-derived label; keep Hungarian
         labels if cluster agreement is below `min_confidence`.

    Returns a `RefineResult` whose `.df` is a copy of `tracked_df` with an
    added `fish_id_cnn` column and a possibly-overwritten `fish_id`.
    """
    train_config = train_config or TrainConfig()

    # (1) crops
    crop_set = extract_crops_from_video(
        video_path, tracked_df, size=crop_size, frame_offset=frame_offset
    )
    if len(crop_set.images) == 0:
        out = tracked_df.copy()
        out["fish_id_cnn"] = out.get("fish_id", "unassigned")
        return RefineResult(df=out, cluster_to_fish={}, confidence=0.0)

    # (2) fragments
    fragments = build_fragments(crop_set.index, max_gap=1)
    if len(fragments) < n_fish:
        # Not enough structure to cluster; short-circuit.
        out = tracked_df.copy()
        out["fish_id_cnn"] = out.get("fish_id", "unassigned")
        return RefineResult(df=out, cluster_to_fish={}, confidence=0.0)

    # (3) mine pairs + train
    same, diff = mine_pairs(fragments, rng_seed=rng_seed)
    model, history = train_embedder(crop_set.images, same, diff, config=train_config)
    if save_model_path is not None:
        Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(model.state_dict(), str(save_model_path))

    # (4) embed + per-fragment mean
    embeddings = embed_all(model, crop_set.images, batch_size=128)
    frag_means = np.stack(
        [embeddings[f.rows].mean(axis=0) for f in fragments], axis=0
    )

    # (5) cluster + map
    clusters = _cluster_fragments(frag_means, n_clusters=n_fish)
    fish_ids = sorted({f.fish_id for f in fragments})
    mapping, agreement = _majority_cluster_to_fish(fragments, clusters, fish_ids)

    # (6) write back
    row_to_fish_cnn: dict[int, str] = {}
    for frag, cluster in zip(fragments, clusters):
        fish_cnn = mapping.get(int(cluster), frag.fish_id)
        for row_idx in frag.rows:
            row_to_fish_cnn[row_idx] = fish_cnn

    out = tracked_df.copy().reset_index(drop=True)
    crop_index = crop_set.index.set_index("row")
    if "fish_id" in out.columns:
        cnn_col = out["fish_id"].astype(str).copy()
    else:
        cnn_col = pd.Series(["unassigned"] * len(out), dtype=str)
    for row_idx, fish in row_to_fish_cnn.items():
        orig = int(crop_index.loc[row_idx, "orig_row"])
        cnn_col.iloc[orig] = fish
    out["fish_id_cnn"] = cnn_col

    if agreement >= min_confidence:
        out["fish_id"] = cnn_col
        print(
            f"  [id_network] refined fish_id (agreement={agreement:.3f} >= "
            f"{min_confidence:.2f}, final pair_acc={history['pair_accuracy'][-1]:.3f})"
        )
    else:
        print(
            f"  [id_network] keeping Hungarian fish_id (agreement={agreement:.3f} < "
            f"{min_confidence:.2f})"
        )

    return RefineResult(df=out, cluster_to_fish=mapping, confidence=agreement)
