from __future__ import annotations

"""Geometry-supervised appearance embeddings without global identity labels.

The input contract is deliberately different from the historical classifier data:

``crops.h5``
    HDF5 dataset ``crops`` containing black-background RGB uint8 crops.
``samples.parquet`` (or ``samples.csv``)
    One row per crop. ``sample_id`` (or ``crop_id``) identifies a crop and
    ``crop_index`` gives its row in the HDF5 array (the latter is optional when
    ``crop_id`` is the row index).
``positive_pairs.parquet`` / ``negative_pairs.parquet``
    Explicit must-link and cannot-link pairs. Pair tables use
    ``sample_a_id,sample_b_id`` or the legacy ``crop_a_id,crop_b_id`` spelling.

No biological/global class name is consumed. Positives are clean same-fragment or
synchronised cross-camera observations; negatives are different fish in the same
frame. Splits are assigned to whole blocks before pairs are partitioned.
"""

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset


SPLITS = ("train", "calibration", "test")


@dataclass(frozen=True)
class GeometryContrastiveConfig:
    embedding_dim: int = 128
    train_backbone: bool = False
    pretrained: bool = True
    normalize_embeddings: bool = True
    epochs: int = 12
    batch_size: int = 96
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    negative_margin: float = 1.0
    min_positive_frame_gap: int = 15
    train_fraction: float = 0.70
    calibration_fraction: float = 0.15
    test_fraction: float = 0.15
    temporal_block_frames: int = 3000
    seed: int = 13
    augment: bool = True
    num_workers: int = 0
    max_steps_per_epoch: int | None = None
    smoke: bool = False

    def validate(self) -> None:
        if self.embedding_dim not in {8, 32, 128}:
            raise ValueError("embedding_dim must be one of 8, 32, or 128")
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be >=1 and batch_size must be >=2")
        if self.negative_margin <= 0.0:
            raise ValueError("negative_margin must be positive")
        if self.normalize_embeddings and self.negative_margin > 2.0:
            raise ValueError("negative_margin cannot exceed 2 for normalized embeddings")
        fractions = (self.train_fraction, self.calibration_fraction, self.test_fraction)
        if any(x < 0 for x in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-8):
            raise ValueError("train/calibration/test fractions must be nonnegative and sum to 1")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def _find_table(dataset_dir: Path, stem: str) -> Path:
    for suffix in (".parquet", ".csv"):
        path = dataset_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Expected {stem}.parquet or {stem}.csv under {dataset_dir}")


def load_samples(dataset_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Load and canonicalise the crop table, retaining only row-index semantics.

    ``samples`` is preferred. ``crop_metadata`` is accepted to make migration from
    the old crop writer mechanical, but explicit positive/negative pair tables are
    still mandatory.
    """
    root = Path(dataset_dir)
    samples_path: Path | None = None
    for stem in ("samples", "crop_metadata"):
        try:
            samples_path = _find_table(root, stem)
            break
        except FileNotFoundError:
            pass
    if samples_path is None:
        raise FileNotFoundError(f"Expected samples.parquet/csv under {root}")
    h5_path = root / "crops.h5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    samples = _read_table(samples_path).copy()
    if "sample_id" not in samples:
        if "crop_id" not in samples:
            raise ValueError("samples table needs sample_id or crop_id")
        samples["sample_id"] = samples["crop_id"]
    if samples["sample_id"].isna().any() or samples["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-null and unique")
    if "crop_index" not in samples:
        if "crop_id" not in samples or not pd.api.types.is_numeric_dtype(samples["crop_id"]):
            raise ValueError("samples needs numeric crop_index when sample_id is not a numeric crop row")
        samples["crop_index"] = samples["crop_id"]
    samples["crop_index"] = pd.to_numeric(samples["crop_index"], errors="raise").astype(np.int64)
    if (samples["crop_index"] < 0).any() or samples["crop_index"].duplicated().any():
        raise ValueError("crop_index must be unique and nonnegative")
    with h5py.File(h5_path, "r") as h5:
        if "crops" not in h5:
            raise ValueError(f"{h5_path} has no 'crops' dataset")
        crop_dataset = h5["crops"]
        n_crops = int(crop_dataset.shape[0])
        shape = tuple(int(x) for x in crop_dataset.shape)
        dtype = str(crop_dataset.dtype)
        declared = crop_dataset.attrs.get("color_order", h5.attrs.get("color_order"))
        if isinstance(declared, bytes):
            declared = declared.decode("utf-8")
        if declared is None:
            raise ValueError(
                "crops.h5 must explicitly declare color_order=RGB; missing color metadata is unsafe"
            )
        if str(declared).upper() != "RGB":
            raise ValueError(
                f"crops.h5 declares color_order={declared!r}; this trainer requires RGB and "
                "will not silently reinterpret BGR crops"
            )
    if len(samples) != n_crops:
        raise ValueError(f"samples/crops length mismatch: {len(samples)} != {n_crops}")
    if int(samples["crop_index"].max()) >= n_crops:
        raise ValueError("crop_index exceeds crops.h5 bounds")
    samples.attrs["crop_shape"] = shape
    samples.attrs["crop_dtype"] = dtype
    samples.attrs["color_order"] = "RGB"
    samples.attrs["color_order_status"] = "declared_rgb"
    return samples, {"crops": h5_path, "samples": samples_path}


def _stable_block_order(blocks: Sequence[str], seed: int) -> list[str]:
    return sorted(
        blocks,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )


def _split_block_counts(n_blocks: int, fractions: Sequence[float]) -> list[int]:
    raw = np.asarray(fractions, dtype=float) * n_blocks
    counts = np.floor(raw).astype(int)
    for idx in np.argsort(-(raw - counts))[: n_blocks - int(counts.sum())]:
        counts[int(idx)] += 1
    nonzero = [i for i, fraction in enumerate(fractions) if fraction > 0]
    if n_blocks >= len(nonzero):
        for empty_idx in [i for i in nonzero if counts[i] == 0]:
            donors = [i for i in nonzero if counts[i] > 1]
            if donors:
                donor = max(donors, key=lambda i: counts[i])
                counts[donor] -= 1
                counts[empty_idx] += 1
    return counts.tolist()


def assign_whole_block_splits(
    samples: pd.DataFrame,
    *,
    seed: int = 13,
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    block_column: str | None = None,
    split_column: str = "split",
    temporal_block_frames: int = 3000,
) -> tuple[pd.DataFrame, str]:
    """Assign deterministic train/calibration/test splits to indivisible blocks.

    Existing split assignments are honoured only after verifying that each block
    occurs in one split. If no block column exists, temporal blocks are derived from
    ``frame`` and prefixed by experiment/session when available so views of the same
    time interval stay together.
    """
    out = samples.copy()
    if block_column is None:
        block_column = next(
            (c for c in ("split_block_id", "block_id", "geometry_block_id", "temporal_block_id", "window") if c in out),
            None,
        )
    if block_column is None:
        if "frame" not in out:
            raise ValueError("Need a block column or frame to make leakage-safe whole-block splits")
        if temporal_block_frames < 1:
            raise ValueError("temporal_block_frames must be positive")
        prefix_cols = [c for c in ("experiment", "session", "recording") if c in out]
        prefix = out[prefix_cols].astype(str).agg("|".join, axis=1) if prefix_cols else pd.Series("recording", index=out.index)
        frame = pd.to_numeric(out["frame"], errors="raise").astype(np.int64)
        block_column = "_derived_split_block_id"
        out[block_column] = prefix + ":" + (frame // temporal_block_frames).astype(str)
    if out[block_column].isna().any():
        raise ValueError(f"{block_column} contains null values")
    out[block_column] = out[block_column].astype(str)

    if split_column in out and out[split_column].notna().all():
        out[split_column] = out[split_column].astype(str).str.lower()
        # Normalise the common ML spelling while keeping one public vocabulary in
        # manifests, metrics, and checkpoints.
        out[split_column] = out[split_column].replace({"validation": "calibration", "val": "calibration"})
        unknown = sorted(set(out[split_column]) - set(SPLITS))
        if unknown:
            raise ValueError(f"Unknown split values in {split_column}: {unknown}")
        leakage = out.groupby(block_column)[split_column].nunique()
        if (leakage > 1).any():
            bad = leakage[leakage > 1].index[:5].tolist()
            raise ValueError(f"Blocks span multiple splits: {bad}")
        return out, block_column

    blocks = _stable_block_order(out[block_column].unique().tolist(), seed)
    counts = _split_block_counts(len(blocks), fractions)
    assignment: dict[str, str] = {}
    cursor = 0
    for split, count in zip(SPLITS, counts, strict=True):
        assignment.update({block: split for block in blocks[cursor : cursor + count]})
        cursor += count
    out[split_column] = out[block_column].map(assignment)
    return out, block_column


def _canonicalise_pair_ids(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = table.copy()
    aliases = (
        ("sample_a_id", "sample_b_id"),
        ("crop_a_id", "crop_b_id"),
        ("sample_id_a", "sample_id_b"),
        ("crop_id_a", "crop_id_b"),
    )
    found = next(((a, b) for a, b in aliases if a in out and b in out), None)
    if found is None:
        raise ValueError(f"{path} needs an explicit pair-id column pair")
    out.rename(columns={found[0]: "sample_a_id", found[1]: "sample_b_id"}, inplace=True)
    return out


def _pair_subtype(row: pd.Series, *, positive: bool, different_view: bool) -> str:
    if not positive:
        return "simultaneous_frame"
    descriptive = " ".join(
        str(row.get(column, ""))
        for column in ("pair_type", "positive_type", "reason", "source")
    ).lower()
    if different_view or "cross" in descriptive or "synchron" in descriptive:
        return "cross_camera"
    return "same_fragment"


def load_explicit_pairs(
    dataset_dir: str | Path,
    samples: pd.DataFrame,
    *,
    min_positive_frame_gap: int = 15,
    strict_simultaneous_negatives: bool = True,
    split_column: str = "split",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Load, validate, filter, and split explicit pair constraints."""
    root = Path(dataset_dir)
    pos_path = _find_table(root, "positive_pairs")
    neg_path = _find_table(root, "negative_pairs")
    pos = _canonicalise_pair_ids(_read_table(pos_path), pos_path)
    neg = _canonicalise_pair_ids(_read_table(neg_path), neg_path)
    if pos.empty or neg.empty:
        raise ValueError("Both positive_pairs and negative_pairs must be non-empty")
    pos["label"] = np.float32(1.0)
    neg["label"] = np.float32(0.0)
    pos["constraint_source"] = "positive_pairs"
    neg["constraint_source"] = "negative_pairs"
    pairs = pd.concat([pos, neg], ignore_index=True, sort=False)
    if pairs[["sample_a_id", "sample_b_id"]].isna().any().any():
        raise ValueError("Pair IDs cannot be null")
    if (pairs["sample_a_id"] == pairs["sample_b_id"]).any():
        raise ValueError("Self-pairs are not valid training constraints")

    meta_cols = ["sample_id", "crop_index", split_column]
    for optional in ("frame", "view"):
        if optional in samples:
            meta_cols.append(optional)
    left = samples[meta_cols].rename(columns={c: f"{c}_a" for c in meta_cols if c != "sample_id"})
    right = samples[meta_cols].rename(columns={c: f"{c}_b" for c in meta_cols if c != "sample_id"})
    pairs = pairs.merge(left, left_on="sample_a_id", right_on="sample_id", how="left").drop(columns="sample_id")
    pairs = pairs.merge(right, left_on="sample_b_id", right_on="sample_id", how="left").drop(columns="sample_id")
    if pairs[["crop_index_a", "crop_index_b"]].isna().any().any():
        missing = pairs[pairs["crop_index_a"].isna() | pairs["crop_index_b"].isna()][["sample_a_id", "sample_b_id"]].head()
        raise ValueError(f"Pairs reference unknown samples:\n{missing.to_string(index=False)}")
    pairs["crop_index_a"] = pairs["crop_index_a"].astype(np.int64)
    pairs["crop_index_b"] = pairs["crop_index_b"].astype(np.int64)
    different_view = (
        pairs["view_a"].astype(str) != pairs["view_b"].astype(str)
        if "view_a" in pairs
        else pd.Series(False, index=pairs.index)
    )
    pairs["pair_subtype"] = [
        _pair_subtype(row, positive=bool(row["label"]), different_view=bool(different_view.loc[idx]))
        for idx, row in pairs.iterrows()
    ]

    audit: dict[str, Any] = {
        "positive_input": int(len(pos)),
        "negative_input": int(len(neg)),
        "positive_too_close_removed": 0,
        "cross_split_removed": 0,
        "duplicates_removed": 0,
    }
    if "frame_a" in pairs:
        frame_a = pd.to_numeric(pairs["frame_a"], errors="coerce")
        frame_b = pd.to_numeric(pairs["frame_b"], errors="coerce")
        bad_negative = (pairs["label"] == 0) & (frame_a != frame_b)
        if strict_simultaneous_negatives and bad_negative.any():
            raise ValueError(
                f"{int(bad_negative.sum())} negative pairs are not simultaneous; "
                "cannot-link supervision must come from the same frame"
            )
        same_fragment = (pairs["label"] == 1) & (pairs["pair_subtype"] == "same_fragment")
        too_close = same_fragment & ((frame_a - frame_b).abs() < int(min_positive_frame_gap))
        audit["positive_too_close_removed"] = int(too_close.sum())
        pairs = pairs[~too_close].copy()
    elif min_positive_frame_gap > 0:
        raise ValueError("samples needs frame to enforce min_positive_frame_gap")

    # Exact duplicate constraints are unhelpful and conflicting constraints are fatal.
    pairs["_lo"] = pairs[["crop_index_a", "crop_index_b"]].min(axis=1)
    pairs["_hi"] = pairs[["crop_index_a", "crop_index_b"]].max(axis=1)
    conflicts = pairs.groupby(["_lo", "_hi"])["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"{int((conflicts > 1).sum())} crop pairs have conflicting labels")
    before = len(pairs)
    pairs = pairs.drop_duplicates(["_lo", "_hi", "label"]).drop(columns=["_lo", "_hi"])
    audit["duplicates_removed"] = int(before - len(pairs))

    cross_split = pairs[f"{split_column}_a"] != pairs[f"{split_column}_b"]
    audit["cross_split_removed"] = int(cross_split.sum())
    if cross_split.any():
        raise ValueError(
            f"{int(cross_split.sum())} pair constraints cross train/calibration/test splits"
        )
    pairs["split"] = pairs[f"{split_column}_a"].astype(str)
    pairs.reset_index(drop=True, inplace=True)
    audit["retained"] = int(len(pairs))
    audit["by_split"] = {str(k): int(v) for k, v in pairs["split"].value_counts().items()}
    audit["by_subtype"] = {str(k): int(v) for k, v in pairs["pair_subtype"].value_counts().items()}
    return pairs, audit, {"positive_pairs": pos_path, "negative_pairs": neg_path}


class GeometryContrastiveEmbedder(nn.Module):
    """ResNet18 feature extractor plus an 8/32/128-D normalized projection."""

    def __init__(
        self,
        embedding_dim: int = 128,
        *,
        pretrained: bool = True,
        train_backbone: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim not in {8, 32, 128}:
            raise ValueError("embedding_dim must be 8, 32, or 128")
        # Lazy import keeps schema/split tools usable on machines without torchvision.
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        network = resnet18(weights=weights)
        self.features = nn.Sequential(*list(network.children())[:-1])
        self.projection = nn.Linear(512, embedding_dim)
        self.embedding_dim = int(embedding_dim)
        self.train_backbone = bool(train_backbone)
        self.normalize = bool(normalize)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for parameter in self.features.parameters():
            parameter.requires_grad = self.train_backbone
        if not self.train_backbone:
            self.features.eval()

    def train(self, mode: bool = True) -> "GeometryContrastiveEmbedder":
        super().train(mode)
        if not self.train_backbone:
            # Freeze BatchNorm running statistics as well as feature weights.
            self.features.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = (rgb - self.rgb_mean) / self.rgb_std
        x = self.features(x).flatten(1)
        embedding = self.projection(x)
        return F.normalize(embedding, dim=1) if self.normalize else embedding


def margin_contrastive_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    label: torch.Tensor,
    *,
    negative_margin: float = 1.0,
) -> torch.Tensor:
    """Pair loss where label 1 is must-link and label 0 is cannot-link."""
    if embedding_a.shape != embedding_b.shape:
        raise ValueError("Embedding tensor shapes must match")
    distance = torch.linalg.vector_norm(embedding_a - embedding_b, dim=1)
    label = label.to(distance.dtype).reshape(-1)
    positive = label * distance.square()
    negative = (1.0 - label) * F.relu(float(negative_margin) - distance).square()
    return (positive + negative).mean()


class GeometryPairDataset(Dataset):
    """Lazy HDF5 pair reader for black-background RGB crops."""

    def __init__(self, crops_path: str | Path, pairs: pd.DataFrame, *, augment: bool = False) -> None:
        self.crops_path = str(crops_path)
        self.pairs = pairs.reset_index(drop=True).copy()
        self.augment = bool(augment)
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(self, index: int) -> torch.Tensor:
        if self._h5 is None:
            self._h5 = h5py.File(self.crops_path, "r")
        array = np.asarray(self._h5["crops"][int(index)])
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB crop, got {array.shape}")
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        else:
            array = array.astype(np.float32)
            if array.max(initial=0.0) > 1.5:
                array /= 255.0
        if self.augment:
            if np.random.random() < 0.5:
                array = array[:, ::-1]
            if np.random.random() < 0.5:
                gain = np.float32(np.random.uniform(0.90, 1.10))
                foreground = np.any(array > 0, axis=2, keepdims=True)
                # Preserve the exact black background during photometric jitter.
                array = np.where(foreground, np.clip(array * gain, 0.0, 1.0), 0.0)
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.pairs.iloc[int(index)]
        return (
            self._crop(int(row["crop_index_a"])),
            self._crop(int(row["crop_index_b"])),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()


class StratifiedPairBatchSampler(BatchSampler):
    """Balanced batches retaining positive subtypes and simultaneous negatives.

    Half of each batch is sampled from cannot-links. The other half is divided
    across available positive subtypes (same-fragment and cross-camera). Pools are
    reshuffled and cycled, so a small but important cross-camera pool is not erased
    by a much larger set of within-fragment pairs.
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        batch_size: int,
        *,
        seed: int = 13,
        max_steps: int | None = None,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[str, np.ndarray] = {}
        neg = pairs.index[pairs["label"] == 0].to_numpy(dtype=np.int64)
        if len(neg):
            self.groups["negative:simultaneous_frame"] = neg
        positives = pairs[pairs["label"] == 1]
        for subtype, rows in positives.groupby("pair_subtype"):
            self.groups[f"positive:{subtype}"] = rows.index.to_numpy(dtype=np.int64)
        if not any(name.startswith("positive:") for name in self.groups) or not len(neg):
            raise ValueError("Training needs both positive and negative constraints")
        natural_steps = max(1, math.ceil(len(pairs) / self.batch_size))
        self.steps = min(natural_steps, int(max_steps)) if max_steps is not None else natural_steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    @staticmethod
    def _draw(pool: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
        if count <= 0:
            return []
        return rng.choice(pool, size=count, replace=count > len(pool)).astype(int).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        negative_name = "negative:simultaneous_frame"
        positive_names = sorted(name for name in self.groups if name.startswith("positive:"))
        n_negative = self.batch_size // 2
        n_positive = self.batch_size - n_negative
        positive_counts = [n_positive // len(positive_names)] * len(positive_names)
        for idx in range(n_positive % len(positive_names)):
            positive_counts[idx] += 1
        for _ in range(self.steps):
            batch = self._draw(self.groups[negative_name], n_negative, rng)
            for name, count in zip(positive_names, positive_counts, strict=True):
                batch.extend(self._draw(self.groups[name], count, rng))
            rng.shuffle(batch)
            yield batch


class _CropDataset(Dataset):
    def __init__(self, crops_path: Path, rows: pd.DataFrame) -> None:
        self.reader = GeometryPairDataset(crops_path, pd.DataFrame(), augment=False)
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[int(index)]
        return self.reader._crop(int(row["crop_index"])), int(index)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _pair_loss_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for crop_a, crop_b, label in loader:
        crop_a = crop_a.to(device, non_blocking=True)
        crop_b = crop_b.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = margin_contrastive_loss(model(crop_a), model(crop_b), label, negative_margin=margin)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu()) * len(label)
        count += len(label)
    return total / max(count, 1)


@torch.inference_mode()
def embed_samples(
    model: nn.Module,
    crops_path: Path,
    samples: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> np.ndarray:
    dataset = _CropDataset(crops_path, samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    output: np.ndarray | None = None
    for crops, row_indices in loader:
        embedded = model(crops.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
        if output is None:
            output = np.empty((len(samples), embedded.shape[1]), dtype=np.float32)
        output[row_indices.numpy()] = embedded
    if output is None:
        raise ValueError("Cannot embed an empty sample table")
    return output


def pair_distances(pairs: pd.DataFrame, samples: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    by_crop_index = {int(crop): idx for idx, crop in enumerate(samples["crop_index"].astype(int))}
    a = np.asarray([by_crop_index[int(x)] for x in pairs["crop_index_a"]], dtype=np.int64)
    b = np.asarray([by_crop_index[int(x)] for x in pairs["crop_index_b"]], dtype=np.int64)
    return np.linalg.norm(embeddings[a] - embeddings[b], axis=1)


def select_distance_threshold(labels: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    """Choose the same-fish distance threshold by calibration balanced accuracy."""
    labels = np.asarray(labels, dtype=np.int8)
    distances = np.asarray(distances, dtype=float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"threshold": float("nan"), "balanced_accuracy": float("nan")}
    unique = np.unique(distances)
    candidates = np.r_[unique[0] - 1e-9, (unique[:-1] + unique[1:]) / 2, unique[-1] + 1e-9]
    best = (-1.0, float(candidates[0]))
    for threshold in candidates:
        predicted = distances <= threshold
        tpr = float(predicted[labels == 1].mean())
        tnr = float((~predicted[labels == 0]).mean())
        score = (tpr + tnr) / 2
        if score > best[0]:
            best = (score, float(threshold))
    return {"threshold": best[1], "balanced_accuracy": best[0]}


def pair_metrics(labels: np.ndarray, distances: np.ndarray, *, threshold: float | None = None) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    distances = np.asarray(distances, dtype=float)
    result: dict[str, Any] = {
        "n_pairs": int(len(labels)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
    }
    if len(labels) == 0:
        return result
    for label, name in ((1, "positive"), (0, "negative")):
        values = distances[labels == label]
        result[f"{name}_distance_mean"] = float(values.mean()) if len(values) else float("nan")
        result[f"{name}_distance_median"] = float(np.median(values)) if len(values) else float("nan")
    if len(np.unique(labels)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        similarity = -distances
        result["roc_auc"] = float(roc_auc_score(labels, similarity))
        result["average_precision"] = float(average_precision_score(labels, similarity))
    if threshold is not None and np.isfinite(threshold):
        predicted = distances <= threshold
        result["accuracy_at_calibration_threshold"] = float((predicted == labels.astype(bool)).mean())
        if (labels == 1).any() and (labels == 0).any():
            result["balanced_accuracy_at_calibration_threshold"] = float(
                (predicted[labels == 1].mean() + (~predicted[labels == 0]).mean()) / 2
            )
    return result


def _file_provenance(path: Path, *, hash_file: bool) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_file:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def write_split_manifest(
    samples: pd.DataFrame,
    output_path: str | Path,
    *,
    block_column: str,
    split_column: str = "split",
) -> tuple[Path, str]:
    """Persist and hash the exact sample-to-block-to-split assignment."""
    columns = ["sample_id", "crop_index", block_column, split_column]
    for optional in ("experiment", "session", "recording", "frame", "view"):
        if optional in samples and optional not in columns:
            columns.append(optional)
    manifest = samples[columns].copy()
    manifest.sort_values([block_column, "crop_index"], inplace=True)
    path = Path(output_path)
    manifest.to_parquet(path, index=False)
    # Hash a stable JSON-lines representation rather than parquet bytes (whose
    # metadata can vary across pyarrow versions).
    canonical = manifest.fillna("<NA>").astype(str).to_dict("records")
    payload = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in canonical)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return path, digest


def write_fit_membership(
    samples: pd.DataFrame,
    train_pairs: pd.DataFrame,
    output_path: str | Path,
    *,
    block_column: str,
    split_column: str = "split",
) -> tuple[Path, str, pd.DataFrame]:
    """Persist the exact union of crops eligible for gradient-bearing batches."""
    fit_crop_indices = pd.unique(
        train_pairs[["crop_index_a", "crop_index_b"]].to_numpy().reshape(-1)
    )
    columns = ["sample_id", "crop_index", block_column, split_column]
    for optional in ("experiment", "session", "recording", "frame", "view", "fragment_id"):
        if optional in samples and optional not in columns:
            columns.append(optional)
    membership = samples[samples["crop_index"].isin(fit_crop_indices)][columns].copy()
    membership.sort_values("crop_index", inplace=True)
    if len(membership) != len(fit_crop_indices):
        raise ValueError("Could not resolve every train-pair crop into fit membership")
    if set(membership[split_column].astype(str)) != {"train"}:
        raise ValueError("Fit membership contains a calibration/test sample")
    path = Path(output_path)
    membership.to_parquet(path, index=False)
    canonical = membership.fillna("<NA>").astype(str).to_dict("records")
    payload = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in canonical)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return path, digest, membership


def inspect_black_background(crops_path: Path, crop_indices: Iterable[int], *, max_samples: int = 64) -> dict[str, float]:
    """Report (but do not invent) evidence that crop corners are black."""
    indices = list(dict.fromkeys(int(x) for x in crop_indices))[:max_samples]
    corner_values: list[np.ndarray] = []
    with h5py.File(crops_path, "r") as h5:
        crops = h5["crops"]
        for index in indices:
            image = np.asarray(crops[index])
            corner_values.append(np.concatenate([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]))
    values = np.concatenate(corner_values) if corner_values else np.asarray([], dtype=np.uint8)
    return {
        "n_crops_checked": int(len(indices)),
        "corner_channel_zero_fraction": float((values == 0).mean()) if len(values) else float("nan"),
    }


def prepare_run_directory(output_root: str | Path, run_name: str | None, *, resume: bool) -> Path:
    root = Path(output_root)
    if run_name is None:
        run_name = "geometry_contrastive_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / run_name
    if resume:
        if not run_dir.is_dir() or not (run_dir / "checkpoint_last.pt").exists():
            raise FileNotFoundError(f"Cannot resume: checkpoint_last.pt not found in {run_dir}")
        if (run_dir / "report.json").exists():
            raise FileExistsError(f"Run is already complete: {run_dir}")
    else:
        # exist_ok=False prevents accidental checkpoint/report replacement.
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_geometry_contrastive(
    *,
    dataset_dir: str | Path,
    run_dir: str | Path,
    config: GeometryContrastiveConfig,
    device: str = "auto",
    block_column: str | None = None,
    split_column: str = "split",
    strict_simultaneous_negatives: bool = True,
    resume: bool = False,
    allow_missing_eval_splits: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a pair-only appearance embedding."""
    run_started_perf = time.perf_counter()
    config.validate()
    _seed_everything(config.seed)
    root = Path(dataset_dir)
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples, source_paths = load_samples(root)
    color_order_status = samples.attrs.get("color_order_status", "legacy_missing_assumed_rgb")
    samples, resolved_block_column = assign_whole_block_splits(
        samples,
        seed=config.seed,
        fractions=(config.train_fraction, config.calibration_fraction, config.test_fraction),
        block_column=block_column,
        split_column=split_column,
        temporal_block_frames=config.temporal_block_frames,
    )
    split_manifest_path, split_manifest_sha256 = write_split_manifest(
        samples,
        output / "split_manifest.parquet",
        block_column=resolved_block_column,
        split_column=split_column,
    )
    pairs, pair_audit, pair_paths = load_explicit_pairs(
        root,
        samples,
        min_positive_frame_gap=config.min_positive_frame_gap,
        strict_simultaneous_negatives=strict_simultaneous_negatives,
        split_column=split_column,
    )
    source_paths.update(pair_paths)
    pair_splits = {split: pairs[pairs["split"] == split].reset_index(drop=True) for split in SPLITS}
    if pair_splits["train"].empty:
        raise ValueError("No training pairs remain after whole-block splitting and filtering")
    missing_eval = [split for split in ("calibration", "test") if pair_splits[split].empty]
    if missing_eval and not allow_missing_eval_splits:
        raise ValueError(f"No pairs in required evaluation splits: {missing_eval}")
    fit_membership_path, fit_membership_sha256, fit_membership = write_fit_membership(
        samples,
        pair_splits["train"],
        output / "fit_membership.parquet",
        block_column=resolved_block_column,
        split_column=split_column,
    )

    actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    torch_device = torch.device(actual_device)
    effective_pretrained = config.pretrained and not config.smoke
    model = GeometryContrastiveEmbedder(
        config.embedding_dim,
        pretrained=effective_pretrained,
        train_backbone=config.train_backbone,
        normalize=config.normalize_embeddings,
    ).to(torch_device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_dataset = GeometryPairDataset(source_paths["crops"], pair_splits["train"], augment=config.augment)
    batch_sampler = StratifiedPairBatchSampler(
        pair_splits["train"],
        config.batch_size,
        seed=config.seed,
        max_steps=(2 if config.smoke else config.max_steps_per_epoch),
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        pin_memory=torch_device.type == "cuda",
        worker_init_fn=_worker_seed,
        generator=generator,
    )
    eval_loaders: dict[str, DataLoader] = {}
    for split in ("calibration", "test"):
        split_pairs = pair_splits[split]
        if config.smoke and len(split_pairs) > 128:
            split_pairs = split_pairs.sample(n=128, random_state=config.seed).reset_index(drop=True)
        dataset = GeometryPairDataset(source_paths["crops"], split_pairs, augment=False)
        eval_loaders[split] = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    config_payload = {
        **asdict(config),
        "dataset_dir": str(root.resolve()),
        "device": actual_device,
        "block_column": resolved_block_column,
        "split_column": split_column,
        "strict_simultaneous_negatives": strict_simultaneous_negatives,
        "effective_pretrained": effective_pretrained,
        "split_manifest_sha256": split_manifest_sha256,
        "fit_membership_sha256": fit_membership_sha256,
    }
    config_path = output / "config.json"
    if resume:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        keys = (
            "embedding_dim",
            "train_backbone",
            "normalize_embeddings",
            "block_column",
            "split_manifest_sha256",
            "fit_membership_sha256",
        )
        mismatch = [key for key in keys if previous.get(key) != config_payload.get(key)]
        if mismatch:
            raise ValueError(f"Resume configuration mismatch: {mismatch}")
    else:
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    history: list[dict[str, float | int]] = []
    start_epoch = 0
    best_loss = float("inf")
    if resume:
        checkpoint = torch.load(output / "checkpoint_last.pt", map_location=torch_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_calibration_loss", float("inf")))
        history = list(checkpoint.get("history", []))

    for epoch in range(start_epoch, config.epochs):
        batch_sampler.set_epoch(epoch)
        train_loss = _pair_loss_epoch(
            model, train_loader, torch_device, config.negative_margin, optimizer=optimizer
        )
        calibration_loss = (
            _pair_loss_epoch(model, eval_loaders["calibration"], torch_device, config.negative_margin)
            if len(eval_loaders["calibration"].dataset)
            else float("nan")
        )
        scheduler.step()
        selection_loss = calibration_loss if np.isfinite(calibration_loss) else train_loss
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "calibration_loss": calibration_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_calibration_loss": min(best_loss, selection_loss),
            "history": history,
            "architecture": "resnet18_pair_embedding",
            "model_kind_version": "geometry_contrastive_resnet18_v1",
            "embedding_dim": config.embedding_dim,
            "train_backbone": config.train_backbone,
            "normalize_embeddings": config.normalize_embeddings,
            "rgb_input": True,
            "split_manifest_sha256": split_manifest_sha256,
            "fit_membership_sha256": fit_membership_sha256,
        }
        _atomic_torch_save(checkpoint_payload, output / "checkpoint_last.pt")
        if selection_loss < best_loss:
            best_loss = selection_loss
            _atomic_torch_save(checkpoint_payload, output / "checkpoint_best.pt")
        pd.DataFrame(history).to_csv(output / "training_log.csv", index=False)
        print(
            f"epoch {epoch + 1}/{config.epochs}: train={train_loss:.5f} "
            f"calibration={calibration_loss:.5f}",
            flush=True,
        )

    best = torch.load(output / "checkpoint_best.pt", map_location=torch_device, weights_only=False)
    model.load_state_dict(best["model_state"])
    prediction_samples = samples
    if config.smoke:
        used_crop_indices = pd.unique(
            pairs[["crop_index_a", "crop_index_b"]].to_numpy().reshape(-1)
        )[:256]
        prediction_samples = samples[samples["crop_index"].isin(used_crop_indices)].copy()
    embeddings = embed_samples(
        model,
        source_paths["crops"],
        prediction_samples,
        device=torch_device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    predictions = prediction_samples[[c for c in ("sample_id", "crop_index", "frame", "view", resolved_block_column, split_column) if c in prediction_samples]].copy()
    for dimension in range(config.embedding_dim):
        predictions[f"embedding_{dimension:03d}"] = embeddings[:, dimension]
    predictions["checkpoint"] = "checkpoint_best.pt"
    predictions.to_parquet(output / "embeddings.parquet", index=False)

    evaluation: dict[str, Any] = {}
    distance_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        split_pairs = pair_splits[split]
        if config.smoke and len(split_pairs) > 128:
            split_pairs = split_pairs.sample(n=128, random_state=config.seed).reset_index(drop=True)
        available = split_pairs[
            split_pairs["crop_index_a"].isin(prediction_samples["crop_index"])
            & split_pairs["crop_index_b"].isin(prediction_samples["crop_index"])
        ]
        distance_by_split[split] = pair_distances(available, prediction_samples, embeddings) if len(available) else np.asarray([])
        pair_splits[split] = available
    calibration_selection = select_distance_threshold(
        pair_splits["calibration"]["label"].to_numpy(), distance_by_split["calibration"]
    )
    threshold = calibration_selection["threshold"]
    pair_prediction_parts: list[pd.DataFrame] = []
    for split in SPLITS:
        evaluation[split] = pair_metrics(
            pair_splits[split]["label"].to_numpy(), distance_by_split[split], threshold=threshold
        )
        scored = pair_splits[split][
            [
                column
                for column in (
                    "sample_a_id",
                    "sample_b_id",
                    "crop_index_a",
                    "crop_index_b",
                    "label",
                    "pair_subtype",
                    "constraint_source",
                    "split",
                )
                if column in pair_splits[split]
            ]
        ].copy()
        scored["embedding_distance"] = distance_by_split[split]
        scored["calibration_threshold"] = threshold
        scored["predicted_same"] = (
            scored["embedding_distance"] <= threshold if np.isfinite(threshold) else pd.NA
        )
        scored["checkpoint"] = "checkpoint_best.pt"
        pair_prediction_parts.append(scored)
    evaluation["calibration_threshold"] = calibration_selection
    pd.concat(pair_prediction_parts, ignore_index=True).to_parquet(
        output / "pair_predictions.parquet", index=False
    )

    provenance = {
        name: _file_provenance(path, hash_file=name != "crops") for name, path in source_paths.items()
    }
    provenance["crops"]["note"] = "Large HDF5 source recorded by resolved path, size, and mtime; not fully hashed."
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        runtime_telemetry = {
            "device": str(torch_device),
            "device_name": torch.cuda.get_device_name(torch_device),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(torch_device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(torch_device)),
            "peak_memory_allocated_mib": float(
                torch.cuda.max_memory_allocated(torch_device) / (1024**2)
            ),
            "peak_memory_reserved_mib": float(
                torch.cuda.max_memory_reserved(torch_device) / (1024**2)
            ),
            "elapsed_seconds": float(time.perf_counter() - run_started_perf),
        }
    else:
        runtime_telemetry = {
            "device": str(torch_device),
            "device_name": "cpu",
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
            "peak_memory_allocated_mib": None,
            "peak_memory_reserved_mib": None,
            "elapsed_seconds": float(time.perf_counter() - run_started_perf),
        }
    report = {
        "status": "smoke_complete" if config.smoke else "complete",
        "objective": "pair-only appearance embedding; no clustering and no global class labels",
        "architecture": {
            "model_kind_version": "geometry_contrastive_resnet18_v1",
            "backbone": "ResNet18",
            "backbone_mode": "fine_tuned" if config.train_backbone else "frozen",
            "embedding_dim": config.embedding_dim,
            "l2_normalized": config.normalize_embeddings,
            "input": "black-background RGB crops",
            "color_order_status": color_order_status,
            "loss": "margin contrastive pair loss",
        },
        "pair_audit": pair_audit,
        "split_manifest": {
            "path": str(split_manifest_path.resolve()),
            "sha256": split_manifest_sha256,
        },
        "fit_membership": {
            "path": str(fit_membership_path.resolve()),
            "sha256": fit_membership_sha256,
            "n_samples": int(len(fit_membership)),
            "split_values": sorted(fit_membership[split_column].astype(str).unique().tolist()),
        },
        "sample_counts_by_split": {str(k): int(v) for k, v in samples[split_column].value_counts().items()},
        "block_counts_by_split": {
            str(k): int(v) for k, v in samples.groupby(split_column)[resolved_block_column].nunique().items()
        },
        "black_background_audit": inspect_black_background(
            source_paths["crops"], samples["crop_index"], max_samples=64
        ),
        "evaluation": evaluation,
        "best_epoch": int(best["epoch"]),
        "best_calibration_loss": float(best_loss),
        "outputs": {
            "checkpoint_best": str((output / "checkpoint_best.pt").resolve()),
            "checkpoint_last": str((output / "checkpoint_last.pt").resolve()),
            "embeddings": str((output / "embeddings.parquet").resolve()),
            "pair_predictions": str((output / "pair_predictions.parquet").resolve()),
            "fit_membership": str(fit_membership_path.resolve()),
            "training_log": str((output / "training_log.csv").resolve()),
        },
        "provenance": provenance,
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runtime_telemetry": runtime_telemetry,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    return report
