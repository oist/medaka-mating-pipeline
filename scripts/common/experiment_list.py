from __future__ import annotations

"""
Loader for the lab's "Experiment List" Google Sheet, cached locally as CSVs.

Source of truth:
    Google Sheet "Experiment List"
    File ID: 1Ae9OXJpCsjcB6z77K2oIKPGTIGS6Rl9Bt1enHOL8-uY
    Owner:   zou.jingyi.t1@dc.tohoku.ac.jp

Re-sync procedure (manual for now — Drive MCP runs in the assistant, not Python):
    1. Ask Claude to "refresh the experiment list" — it will download the sheet
       via the Google Drive MCP and overwrite the CSVs under experiment_list_dir.
    2. Re-run `python -m scripts.common.build_catalog --paths-config ...` to
       pick up the new values.

This module exposes only the two sheets that meaningfully enrich the catalog:
    lay_eggs_table — per-experiment outcomes (eggs laid, which male accepted,
                     dash / courtship / wrap / sneak counts)
    group_list     — per-group fish description (female / male1 / male2 size
                     and colour identifiers)

Other sheets in the workbook (boris_events summary, by-date / by-pair / by-group
aggregates, the secondary 22x5 "lay eggs table") are derivable from these two
plus the per-frame BORIS CSVs, so we don't bake them into the catalog.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LAY_EGGS_FILENAME = "lay_eggs_table.csv"
GROUP_LIST_FILENAME = "group_list.csv"

LAY_EGGS_COUNT_COLUMNS = (
    "male12_dash",
    "male21_dash",
    "male1_courtship",
    "male2_courtship",
    "male1_wrap",
    "male2_wrap",
    "male1_sneak",
    "male2_sneak",
)


@dataclass(frozen=True)
class ExperimentListEntry:
    """One row of lay_eggs_table after normalisation."""

    folder_name: str
    eggs: str | None             # yes / no / "yes,<note>" / NO / NaN
    eggs_after: str | None       # yes / no / NaN
    female_accept: str | None    # m1, m2, m1m2, m2m1, ?, no, NaN
    male_contest: str | None     # yes / no / NaN
    sneak: str | None            # yes / no / NaN
    counts: dict[str, int | None]


def _read_csv_clean(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _strip_or_none(value: str | float | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_lay_eggs_table(experiment_list_dir: str | Path) -> pd.DataFrame:
    """
    Return the normalised lay_eggs_table indexed by folder_name.

    - Drops the spreadsheet's header-description row.
    - Strips empty strings to NaN.
    - Coerces count columns (`male12_dash`, ...) to nullable Int64.
    """
    path = Path(experiment_list_dir) / LAY_EGGS_FILENAME
    df = _read_csv_clean(path)
    df = df[df["file_names"].str.match(r"^\d{8}_[A-Za-z]+_\d+_\d+_\w+$", na=False)].copy()
    df = df.rename(columns={"file_names": "folder_name"})

    text_cols = ["eggs", "eggs_after", "female_accept", "male_contest", "sneak"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].map(_strip_or_none)

    for col in LAY_EGGS_COUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_to_int).astype("Int64")

    return df.reset_index(drop=True)


def load_group_list(experiment_list_dir: str | Path) -> pd.DataFrame:
    """
    Return the per-group fish description table indexed by group code.

    Columns are renamed to snake_case (`set_date`, `group`, `female_desc`,
    `male1_desc`, `male2_desc`, `detail`).
    """
    path = Path(experiment_list_dir) / GROUP_LIST_FILENAME
    df = _read_csv_clean(path)
    rename = {
        "SET DATE": "set_date",
        "GROUP NAME": "group",
        "FEMALE": "female_desc",
        "MALE1": "male1_desc",
        "MALE2": "male2_desc",
        "DETAIL": "detail",
    }
    df = df.rename(columns=rename)
    for col in ("set_date", "group", "female_desc", "male1_desc", "male2_desc", "detail"):
        if col in df.columns:
            df[col] = df[col].map(_strip_or_none)
    df = df[df["group"].notna()].reset_index(drop=True)
    return df


def lookup_lay_eggs(
    df: pd.DataFrame, folder_name: str
) -> dict | None:
    """Return the lay_eggs_table row for `folder_name` as a JSON-friendly dict."""
    hit = df[df["folder_name"] == folder_name]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()
    out: dict[str, object] = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def lookup_group(df: pd.DataFrame, group: str) -> dict | None:
    """Return the group_list row for `group` as a JSON-friendly dict."""
    hit = df[df["group"] == group]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}
