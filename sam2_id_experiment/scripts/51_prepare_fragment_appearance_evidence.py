from __future__ import annotations

"""Pool learned crop embeddings into label-free global-fragment evidence.

This is the bridge between geometry-contrastive model inference and
``scripts.tracking.benchmark_global_fragment_stitching``.  It groups crops by a
graph-fragment provenance key, balances camera views, and writes a compact NPZ.
It does not cluster embeddings and does not consume or emit biological IDs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.tracking.appearance_prototypes import PooledEmbeddings, pool_embeddings


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported table format: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_fragment_embeddings(
    embedding_table: pd.DataFrame,
    sample_table: pd.DataFrame,
    *,
    fragment_column: str,
    sample_id_column: str = "sample_id",
    view_column: str = "view",
    splits: tuple[str, ...] = (),
    min_samples: int = 3,
) -> tuple[PooledEmbeddings, pd.DataFrame]:
    embedding_columns = sorted(
        (column for column in embedding_table if column.startswith("embedding_")),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    if not embedding_columns:
        raise ValueError("embedding table has no embedding_000... columns")
    if sample_id_column not in embedding_table.columns:
        raise ValueError(f"embedding table missing join key {sample_id_column!r}")
    required = {sample_id_column, fragment_column, view_column}
    missing = required - set(sample_table.columns)
    if missing:
        raise ValueError("sample table missing columns: " + ", ".join(sorted(missing)))
    if sample_table[sample_id_column].duplicated().any():
        raise ValueError(f"{sample_id_column} must be unique in the sample table")
    if embedding_table[sample_id_column].duplicated().any():
        raise ValueError(f"{sample_id_column} must be unique in the embedding table")

    metadata_columns = [sample_id_column, fragment_column, view_column]
    if "split" in sample_table:
        metadata_columns.append("split")
    merged = embedding_table[[sample_id_column, *embedding_columns]].merge(
        sample_table[metadata_columns],
        on=sample_id_column,
        how="left",
        validate="one_to_one",
    )
    if merged[[fragment_column, view_column]].isna().any().any():
        raise ValueError("embedding rows are missing fragment/view provenance")
    blank_fragment = merged[fragment_column].astype(str).str.strip().eq("")
    blank_view = merged[view_column].astype(str).str.strip().eq("")
    if blank_fragment.any() or blank_view.any():
        raise ValueError(
            "embedding rows contain blank fragment/view provenance: "
            f"blank_fragments={int(blank_fragment.sum())}, blank_views={int(blank_view.sum())}"
        )
    if splits:
        if "split" not in merged:
            raise ValueError("--split was requested but samples have no split column")
        merged = merged[merged["split"].astype(str).isin(splits)].copy()
    if merged.empty:
        raise ValueError("no embedding rows remain after filtering")

    vectors = merged[embedding_columns].to_numpy(dtype=np.float32)
    pooled = pool_embeddings(
        vectors,
        merged[fragment_column].astype(str).tolist(),
        views=merged[view_column].astype(str).tolist(),
        min_samples=min_samples,
        balance_views=True,
    )
    if not pooled.keys:
        raise ValueError("no fragment has enough samples for appearance evidence")
    count_by_fragment = pooled.count_mapping()
    provenance = (
        merged.groupby(fragment_column, as_index=False)
        .agg(
            available_samples=(sample_id_column, "nunique"),
            available_views=(view_column, "nunique"),
        )
    )
    provenance[fragment_column] = provenance[fragment_column].astype(str)
    provenance = provenance[provenance[fragment_column].isin(pooled.keys)].copy()
    provenance["pooled_samples"] = provenance[fragment_column].map(count_by_fragment)
    provenance["camera_balanced_pooling"] = True
    provenance.sort_values(fragment_column, inplace=True)
    return pooled, provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fragment-column", default="component_id_provenance")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--view-column", default="view")
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Optional split to include; repeat for multiple splits.",
    )
    parser.add_argument("--min-samples", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = _read_table(args.embeddings)
    samples = _read_table(args.samples)
    pooled, provenance = prepare_fragment_embeddings(
        embeddings,
        samples,
        fragment_column=args.fragment_column,
        sample_id_column=args.sample_id_column,
        view_column=args.view_column,
        splits=tuple(map(str, args.split)),
        min_samples=args.min_samples,
    )
    output_npz = args.output_dir / "fragment_embeddings.npz"
    np.savez_compressed(
        output_npz,
        fragment_ids=np.asarray(pooled.keys),
        embeddings=pooled.embeddings,
        sample_counts=np.asarray(pooled.sample_counts, dtype=np.int64),
    )
    provenance_path = args.output_dir / "fragment_embedding_provenance.csv"
    provenance.to_csv(provenance_path, index=False)
    summary = {
        "schema_version": "fragment_appearance_evidence_v1",
        "objective": "label-free appearance edge evidence; no clustering or identity labels",
        "n_fragments": len(pooled.keys),
        "embedding_dim": int(pooled.embeddings.shape[1]),
        "min_samples": int(args.min_samples),
        "camera_balanced_pooling": True,
        "fragment_column": str(args.fragment_column),
        "included_splits": list(map(str, args.split)),
        "inputs": {
            "embeddings": str(args.embeddings.resolve()),
            "embeddings_sha256": _sha256(args.embeddings),
            "samples": str(args.samples.resolve()),
            "samples_sha256": _sha256(args.samples),
        },
        "outputs": {
            "fragment_embeddings": str(output_npz.resolve()),
            "provenance": str(provenance_path.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
