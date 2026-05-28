from __future__ import annotations

from pathlib import Path

import pandas as pd

from sam2_id_experiment.src.id_dataset.pair_sampler import sample_positive_negative_pairs


def write_pairs(crop_metadata_path: str | Path, output_dir: str | Path, config: dict) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_metadata = pd.read_parquet(crop_metadata_path)
    pairs = sample_positive_negative_pairs(crop_metadata, config)
    out_path = out_dir / "pairs.parquet"
    pairs.to_parquet(out_path, index=False)
    return out_path

