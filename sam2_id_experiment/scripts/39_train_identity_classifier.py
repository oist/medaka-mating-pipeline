from __future__ import annotations

"""Train an identity classifier on segmentation-instance crops.

This is the stronger, idtracker-style training path for the Detectron2/SAM2 crop
datasets written by ``34_assemble_segid_training`` or ``36_build_male_dataset``.
It deliberately treats ``global_id`` labels as weak labels: high-purity geometry
tracklets are useful supervision, mixed tracklets are label noise.

Compared with the self-supervised pair trainer, this script adds:

* class- and tracklet-balanced sampling so long fragments do not dominate;
* optional supervised-contrastive loss on the ResNet embedding;
* whole-window validation when multiple windows are present;
* per-crop, per-tracklet, and per-window metrics;
* a position-only baseline so a high CNN score can be checked against spatial
  confounds.

Run::

    python -m sam2_id_experiment.scripts.39_train_identity_classifier \
        --dataset C:/udmt_validation/segid/K_1_2/pooled_sides_full \
        --mode all3 --out-dir C:/udmt_validation/segid/K_1_2/id_classifier_all3

For the male-only question::

    python -m sam2_id_experiment.scripts.39_train_identity_classifier \
        --dataset C:/udmt_validation/segid/K_1_2/pooled_sides_full \
        --mode male --female-gid 1 \
        --split window --out-dir C:/udmt_validation/segid/K_1_2/id_classifier_males
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

torch = None
nn = None
F = None
DataLoader = None
WeightedRandomSampler = None
EmbeddingClassifier = None
ResNet18Embedder = None


def ensure_training_deps() -> None:
    """Import Torch/Torchvision only when training is actually requested."""
    global torch, nn, F, DataLoader, WeightedRandomSampler, EmbeddingClassifier, ResNet18Embedder
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data import WeightedRandomSampler as _WeightedRandomSampler

    from sam2_id_experiment.src.id_model.resnet_embedder import (
        EmbeddingClassifier as _EmbeddingClassifier,
        ResNet18Embedder as _ResNet18Embedder,
    )

    torch = _torch
    nn = _nn
    F = _F
    DataLoader = _DataLoader
    WeightedRandomSampler = _WeightedRandomSampler
    EmbeddingClassifier = _EmbeddingClassifier
    ResNet18Embedder = _ResNet18Embedder


class CropClassDataset:
    def __init__(self, crops: np.ndarray, crop_ids: np.ndarray, labels: np.ndarray, *, augment: bool) -> None:
        self.crops = crops
        self.crop_ids = crop_ids.astype(int)
        self.labels = labels.astype(int)
        self.augment = augment

    def __len__(self) -> int:
        return int(len(self.labels))

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if not self.augment:
            return image
        if np.random.rand() < 0.5:
            image = image[:, ::-1]
        if np.random.rand() < 0.5:
            gain = np.float32(np.random.uniform(0.85, 1.15))
            bias = np.float32(np.random.uniform(-0.05, 0.05))
            image = np.clip(image * gain + bias, 0.0, 1.0)
        return np.ascontiguousarray(image)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        crop = self.crops[int(self.crop_ids[idx])].astype(np.float32) / 255.0
        crop = self._augment(crop)
        return torch.from_numpy(crop).permute(2, 0, 1), torch.tensor(int(self.labels[idx]))


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss over one batch of L2-normalized embeddings."""
    if features.shape[0] < 2:
        return features.sum() * 0.0
    labels = labels.view(-1, 1)
    same_label = torch.eq(labels, labels.T).float()
    self_mask = torch.eye(features.shape[0], device=features.device, dtype=torch.float32)
    positives = same_label * (1.0 - self_mask)
    positive_counts = positives.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return features.sum() * 0.0

    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_mask = 1.0 - self_mask
    exp_logits = torch.exp(logits) * denominator_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    per_anchor = -(positives * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
    return per_anchor[valid].mean()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a supervised identity classifier on seg-ID crops.")
    parser.add_argument("--dataset", required=True, help="directory with crops.h5 + crop_metadata.parquet")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["all3", "female", "male"], default="all3")
    parser.add_argument("--label-col", default="global_id")
    parser.add_argument("--female-gid", type=int, default=1)
    parser.add_argument("--classes", nargs="*", type=int, default=None,
                        help="optional label values to keep for --mode all3")
    parser.add_argument("--min-global-id-purity", type=float, default=0.9)
    parser.add_argument("--min-tracklet-crops", type=int, default=5)
    parser.add_argument("--split", choices=["auto", "tracklet", "window"], default="auto")
    parser.add_argument("--holdout-window", default=None)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--supcon-weight", type=float, default=0.15)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_training_deps()

    dataset_dir = Path(args.dataset)
    with h5py.File(dataset_dir / "crops.h5", "r") as h5:
        key = "crops" if "crops" in h5 else "images"
        crops = np.asarray(h5[key][...])
    meta = pd.read_parquet(dataset_dir / "crop_metadata.parquet").reset_index(drop=True)
    use, class_values, class_names = prepare_labeled_metadata(meta, args)
    train_rows, val_rows, split_info = make_split(use, args)
    if train_rows.empty or val_rows.empty:
        raise SystemExit("empty train/validation split; reduce filters or choose a different split")

    device = resolve_device(args.device)
    model = EmbeddingClassifier(
        ResNet18Embedder(
            embedding_dim=int(args.embedding_dim),
            pretrained=not args.no_pretrained,
        ),
        n_classes=len(class_values),
    ).to(device)

    train_log, best_state, best_epoch = train_model(model, crops, train_rows, val_rows, args, device)
    model.load_state_dict(best_state)
    val_probs = predict_probs(model, crops, val_rows["crop_id"].to_numpy(), device, batch_size=args.batch_size)

    crop_metrics, tracklet_table, window_metrics = evaluate_predictions(
        val_rows,
        val_rows["label_idx"].to_numpy(),
        val_probs,
        class_values=class_values,
        class_names=class_names,
    )
    position_report = position_only_baseline(train_rows, val_rows, class_values, class_names)

    tracklet_table.to_csv(out_dir / "val_tracklet_predictions.csv", index=False)
    val_crop_out = val_rows.copy()
    for i, name in enumerate(class_names):
        val_crop_out[f"prob_{name}"] = val_probs[:, i]
    val_crop_out["pred_label_idx"] = val_probs.argmax(axis=1)
    val_crop_out["pred_label"] = [class_values[int(i)] for i in val_crop_out["pred_label_idx"]]
    val_crop_out.to_parquet(out_dir / "val_crop_predictions.parquet", index=False)
    pd.DataFrame(train_log).to_csv(out_dir / "training_log.csv", index=False)

    report = {
        "dataset": str(dataset_dir),
        "mode": args.mode,
        "label_col": args.label_col,
        "class_values": [int(v) for v in class_values],
        "class_names": class_names,
        "n_train_crops": int(len(train_rows)),
        "n_val_crops": int(len(val_rows)),
        "n_train_tracklets": int(train_rows["view_tracklet"].nunique()),
        "n_val_tracklets": int(val_rows["view_tracklet"].nunique()),
        "split": split_info,
        "filters": {
            "min_global_id_purity": float(args.min_global_id_purity),
            "min_tracklet_crops": int(args.min_tracklet_crops),
        },
        "best_epoch": int(best_epoch),
        "cnn": crop_metrics | {"per_window": window_metrics},
        "position_only_baseline": position_report,
        "confusion_matrix_labels": class_names,
    }
    (out_dir / "identity_classifier_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": "resnet18_identity_classifier",
            "embedding_dim": int(args.embedding_dim),
            "class_values": [int(v) for v in class_values],
            "class_names": class_names,
            "mode": args.mode,
            "label_col": args.label_col,
            "female_gid": int(args.female_gid),
        },
        out_dir / "model.pt",
    )

    print(json.dumps(report, indent=2))


def prepare_labeled_metadata(meta: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[int], list[str]]:
    required = {"crop_id", "qc_status", "view_tracklet", "tracklet_id", "frame", args.label_col}
    missing = required - set(meta.columns)
    if missing:
        raise SystemExit(f"metadata missing required columns: {sorted(missing)}")

    purity = meta.get("global_id_purity", pd.Series(1.0, index=meta.index)).astype(float)
    use = meta[
        (meta["qc_status"] == "clean_train")
        & (pd.to_numeric(meta[args.label_col], errors="coerce") >= 0)
        & (purity >= float(args.min_global_id_purity))
    ].copy()

    track_counts = use.groupby("view_tracklet")["crop_id"].transform("size")
    use = use[track_counts >= int(args.min_tracklet_crops)].copy()
    labels = pd.to_numeric(use[args.label_col], errors="coerce").astype(int)

    if args.mode == "female":
        use["label_value"] = (labels == int(args.female_gid)).astype(int)
        class_values = [0, 1]
        class_names = ["male", "female"]
    elif args.mode == "male":
        use = use[labels != int(args.female_gid)].copy()
        labels = pd.to_numeric(use[args.label_col], errors="coerce").astype(int)
        class_values = sorted(int(v) for v in labels.unique())
        class_names = [f"gid_{v}" for v in class_values]
        use["label_value"] = labels
    else:
        class_values = sorted(int(v) for v in (args.classes if args.classes else labels.unique()))
        use = use[labels.isin(class_values)].copy()
        labels = pd.to_numeric(use[args.label_col], errors="coerce").astype(int)
        class_names = [f"gid_{v}" for v in class_values]
        use["label_value"] = labels

    if len(class_values) < 2:
        raise SystemExit(f"need at least two classes after filtering, got {class_values}")
    class_to_idx = {int(value): i for i, value in enumerate(class_values)}
    use["label_idx"] = use["label_value"].map(class_to_idx).astype(int)
    return use.reset_index(drop=True), class_values, class_names


def make_split(meta: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    split = args.split
    if split == "auto":
        split = "window" if "window" in meta.columns and meta["window"].nunique() > 1 else "tracklet"

    if split == "window" and ("window" not in meta.columns or meta["window"].nunique() <= 1):
        split = "tracklet"

    if split == "window":
        window_order = (
            meta.groupby("window")["frame"].min().sort_values().index.astype(str).tolist()
        )
        holdout = str(args.holdout_window) if args.holdout_window else window_order[-1]
        val_mask = meta["window"].astype(str) == holdout
        train = meta[~val_mask].copy()
        val = meta[val_mask].copy()
        info = {"kind": "window", "holdout_window": holdout, "windows": window_order}
    else:
        rng = np.random.default_rng(13)
        val_groups: set[int] = set()
        group_labels = meta.groupby("view_tracklet")["label_idx"].agg(lambda s: int(s.mode().iloc[0]))
        for _label, groups in group_labels.groupby(group_labels):
            group_ids = groups.index.to_numpy().copy()
            rng.shuffle(group_ids)
            n_val = int(round(len(group_ids) * float(args.val_frac)))
            if len(group_ids) > 1:
                n_val = min(max(n_val, 1), len(group_ids) - 1)
            else:
                n_val = 0
            val_groups.update(int(g) for g in group_ids[:n_val])
        val_mask = meta["view_tracklet"].astype(int).isin(val_groups)
        train = meta[~val_mask].copy()
        val = meta[val_mask].copy()
        info = {"kind": "tracklet", "val_frac": float(args.val_frac)}

    check_split_classes(train, val)
    return train.reset_index(drop=True), val.reset_index(drop=True), info


def check_split_classes(train: pd.DataFrame, val: pd.DataFrame) -> None:
    train_classes = set(int(v) for v in train["label_idx"].unique())
    val_classes = set(int(v) for v in val["label_idx"].unique())
    missing_train = sorted(val_classes - train_classes)
    missing_val = sorted(train_classes - val_classes)
    if missing_train or missing_val:
        raise SystemExit(
            "split does not contain every class in train and validation; "
            f"missing_train={missing_train}, missing_val={missing_val}"
        )


def train_model(
    model: EmbeddingClassifier,
    crops: np.ndarray,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict, int]:
    train_ds = CropClassDataset(
        crops,
        train_rows["crop_id"].to_numpy(),
        train_rows["label_idx"].to_numpy(),
        augment=not args.no_augment,
    )
    weights = row_sampling_weights(train_rows)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        sampler=sampler,
        drop_last=True,
        num_workers=int(args.num_workers),
    )
    class_counts = train_rows["label_idx"].value_counts().sort_index()
    class_weights = torch.tensor(
        [1.0 / max(int(class_counts.get(i, 0)), 1) for i in range(len(class_counts))],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.mean().clamp_min(1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))

    best_metric = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    log_rows: list[dict] = []
    for epoch in range(int(args.epochs)):
        set_backbone_frozen(model, frozen=epoch < int(args.freeze_backbone_epochs))
        model.train()
        total_loss = total_ce = total_sup = total_n = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model.embedder(x)
            logits = model.head(z)
            ce = criterion(logits, y)
            sup = supervised_contrastive_loss(
                z,
                y,
                temperature=float(args.supcon_temperature),
            )
            loss = ce + float(args.supcon_weight) * sup
            loss.backward()
            optimizer.step()
            n = int(y.numel())
            total_loss += float(loss.detach().cpu()) * n
            total_ce += float(ce.detach().cpu()) * n
            total_sup += float(sup.detach().cpu()) * n
            total_n += n
        scheduler.step()

        val_probs = predict_probs(model, crops, val_rows["crop_id"].to_numpy(), device, batch_size=args.batch_size)
        metrics, _tracklet_table, _window = evaluate_predictions(
            val_rows,
            val_rows["label_idx"].to_numpy(),
            val_probs,
            class_values=list(range(val_probs.shape[1])),
            class_names=[str(i) for i in range(val_probs.shape[1])],
        )
        metric = float(metrics["tracklet_balacc"])
        log = {
            "epoch": int(epoch),
            "loss": total_loss / max(total_n, 1),
            "ce_loss": total_ce / max(total_n, 1),
            "supcon_loss": total_sup / max(total_n, 1),
            "val_crop_balacc": float(metrics["crop_balacc"]),
            "val_tracklet_balacc": metric,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        log_rows.append(log)
        print(
            f"epoch {epoch}: loss={log['loss']:.4f} "
            f"val_crop_balacc={log['val_crop_balacc']:.3f} "
            f"val_tracklet_balacc={metric:.3f}",
            flush=True,
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience) and epoch >= int(args.freeze_backbone_epochs):
                break
    return log_rows, best_state, best_epoch


def row_sampling_weights(rows: pd.DataFrame) -> torch.DoubleTensor:
    label_counts = rows["label_idx"].value_counts().to_dict()
    tracklet_counts = rows["view_tracklet"].value_counts().to_dict()
    weights = []
    for row in rows.to_dict("records"):
        label_weight = 1.0 / max(int(label_counts[int(row["label_idx"])]), 1)
        track_weight = 1.0 / max(int(tracklet_counts[int(row["view_tracklet"])]), 1)
        weights.append(label_weight * track_weight)
    return torch.as_tensor(weights, dtype=torch.double)


def set_backbone_frozen(model: EmbeddingClassifier, *, frozen: bool) -> None:
    for name, param in model.embedder.net.named_parameters():
        param.requires_grad = (not frozen) or name.startswith("fc.")


def predict_probs(
    model: EmbeddingClassifier,
    crops: np.ndarray,
    crop_ids: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(crop_ids), int(batch_size)):
            ids = crop_ids[start:start + int(batch_size)].astype(int)
            batch = crops[ids].astype(np.float32) / 255.0
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32)


def evaluate_predictions(
    rows: pd.DataFrame,
    y_true_idx: np.ndarray,
    probs: np.ndarray,
    *,
    class_values: list[int],
    class_names: list[str],
) -> tuple[dict, pd.DataFrame, dict]:
    pred_idx = probs.argmax(axis=1)
    crop_balacc = balanced_accuracy(y_true_idx, pred_idx)
    tracklet_table = aggregate_tracklet_predictions(rows, y_true_idx, probs, class_values, class_names)
    tracklet_balacc = balanced_accuracy(
        tracklet_table["true_label_idx"].to_numpy(),
        tracklet_table["pred_label_idx"].to_numpy(),
    )
    cm = confusion_matrix(y_true_idx, pred_idx, labels=list(range(len(class_values)))).astype(int)
    metrics = {
        "crop_balacc": round(float(crop_balacc), 4),
        "tracklet_balacc": round(float(tracklet_balacc), 4),
        "crop_confusion_matrix": cm.tolist(),
    }
    per_window = {}
    if "window" in rows.columns:
        tmp = rows[["window"]].copy()
        tmp["true"] = y_true_idx
        tmp["pred"] = pred_idx
        for window, group in tmp.groupby("window"):
            per_window[str(window)] = {
                "n_crops": int(len(group)),
                "crop_balacc": round(float(balanced_accuracy(group["true"], group["pred"])), 4),
            }
    return metrics, tracklet_table, per_window


def aggregate_tracklet_predictions(
    rows: pd.DataFrame,
    y_true_idx: np.ndarray,
    probs: np.ndarray,
    class_values: list[int],
    class_names: list[str],
) -> pd.DataFrame:
    work = rows.copy().reset_index(drop=True)
    work["true_label_idx"] = y_true_idx.astype(int)
    for i, name in enumerate(class_names):
        work[f"prob_{name}"] = probs[:, i]
    out_rows = []
    group_cols = ["view_tracklet"]
    for tracklet, group in work.groupby(group_cols[0]):
        mean_probs = np.asarray([group[f"prob_{name}"].mean() for name in class_names], dtype=float)
        pred_idx = int(np.argmax(mean_probs))
        true_idx = int(group["true_label_idx"].mode().iloc[0])
        row = {
            "view_tracklet": int(tracklet),
            "tracklet_id": int(group["tracklet_id"].iloc[0]),
            "view": str(group["view"].iloc[0]) if "view" in group.columns else "",
            "window": str(group["window"].iloc[0]) if "window" in group.columns else "",
            "frame_min": int(group["frame"].min()),
            "frame_max": int(group["frame"].max()),
            "n_crops": int(len(group)),
            "true_label_idx": true_idx,
            "true_label": int(class_values[true_idx]),
            "pred_label_idx": pred_idx,
            "pred_label": int(class_values[pred_idx]),
            "correct": bool(pred_idx == true_idx),
        }
        for i, name in enumerate(class_names):
            row[f"mean_prob_{name}"] = float(mean_probs[i])
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def position_only_baseline(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    class_values: list[int],
    class_names: list[str],
) -> dict:
    if len(class_values) < 2:
        return {"available": False, "reason": "need at least two classes"}
    x_train = metadata_features(train_rows)
    x_val = metadata_features(val_rows).reindex(columns=x_train.columns, fill_value=0.0)
    scaler = StandardScaler()
    y_train = train_rows["label_idx"].to_numpy()
    y_val = val_rows["label_idx"].to_numpy()
    try:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(scaler.fit_transform(x_train), y_train)
        pred = clf.predict(scaler.transform(x_val))
        crop_balacc = balanced_accuracy(y_val, pred)
        probs = clf.predict_proba(scaler.transform(x_val))
        class_order = [int(c) for c in clf.classes_]
        full_probs = np.zeros((len(val_rows), len(class_values)), dtype=float)
        for source_col, class_idx in enumerate(class_order):
            full_probs[:, class_idx] = probs[:, source_col]
        tracklets = aggregate_tracklet_predictions(val_rows, y_val, full_probs, class_values, class_names)
        return {
            "available": True,
            "features": list(x_train.columns),
            "crop_balacc": round(float(crop_balacc), 4),
            "tracklet_balacc": round(
                float(balanced_accuracy(tracklets["true_label_idx"], tracklets["pred_label_idx"])),
                4,
            ),
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {"available": False, "reason": str(exc)}


def metadata_features(rows: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=rows.index)
    for col in ["x", "y", "area"]:
        if col in rows.columns:
            features[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0).astype(float)
    if "view" in rows.columns:
        features = pd.concat(
            [features, pd.get_dummies(rows["view"].astype(str), prefix="view", dtype=float)],
            axis=1,
        )
    if features.empty:
        features["constant"] = 1.0
    return features


def balanced_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for label in sorted(set(y_true.tolist())):
        mask = y_true == label
        if mask.any():
            recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()
