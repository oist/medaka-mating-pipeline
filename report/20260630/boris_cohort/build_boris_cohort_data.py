"""Build a BORIS cohort summary for the medaka poster supplement.

Inputs are the Drive-backed BORIS CSV files listed by the user. Outputs are
small CSV/JSON tables under report/20260630/boris_cohort/data, intended for a
PowerPoint figure bank and for manual auditing.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)

BORIS_DIR = Path(
    r"G:\.shortcut-targets-by-id\14OI1k7IaNOdC5yaC9GnYyCi2u-gfGLEs"
    r"\OIST 1215-0122\boris_events"
)

BORIS_FILES = [
    "20251220_E_1_1_FCMC.csv",
    "20251220_E_1_2_FCMC.csv",
    "20251220_E_1_3_FCMC.csv",
    "20251223_G_1_1_FCMC.csv",
    "20251223_G_1_2_FCMC.csv",
    "20251223_G_1_3_FCMC.csv",
    "20251224_I_1_1_FCMC.csv",
    "20251224_I_1_2_FCMC.csv",
    "20251224_I_1_3_FCMC.csv",
    "20251225_K_1_1_FCMC.csv",
    "20251225_K_1_2_FCMC.csv",
    "20251225_K_1_3_FCMC.csv",
    "20251225_L_1_1_FCMC.csv",
    "20251225_L_1_2_FCMC.csv",
    "20251225_L_1_3_FCMC.csv",
    "20260106_H_1_1_FCMC.csv",
    "20260106_H_1_2_FCMC.csv",
    "20260109_O_1_1_FCMC.csv",
    "20260109_O_1_2_FCMC.csv",
    "20260109_O_1_3_FCMC.csv",
    "20260112_I_2_1_FCMC.csv",
    "20260112_I_2_2_FCMC.csv",
    "20260112_I_2_3_FCMC.csv",
]

FPS_DEFAULT = 166.0
CATEGORY_ORDER = ["Courtship", "Competitive", "Mating", "Other"]
MALE_TO_FEMALE = {"m1tf": "m1", "m2tf": "m2"}
MALE_MALE = {"m1tm2", "m2tm1", "m1m2"}


def _num(value, default=np.nan):
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(out) else float(out)


def _first_pipe_float(value: str, default: float = FPS_DEFAULT) -> float:
    if not isinstance(value, str) or not value.strip():
        return default
    return _num(value.split("|")[0], default)


def _experiment_parts(experiment: str) -> dict[str, str]:
    parts = experiment.split("_")
    return {
        "date": parts[0] if len(parts) > 0 else "",
        "group": parts[1] if len(parts) > 1 else "",
        "block": parts[2] if len(parts) > 2 else "",
        "replicate": parts[3] if len(parts) > 3 else "",
        "condition": parts[4] if len(parts) > 4 else "",
    }


def _subject_role(subject: str) -> str:
    if subject == "f":
        return "female"
    if subject in MALE_TO_FEMALE:
        return "male_to_female"
    if subject in MALE_MALE:
        return "male_male"
    if subject in {"m1", "m2"}:
        return "single_male"
    return "other"


def parse_boris_file(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    experiment = path.stem
    parts = _experiment_parts(experiment)

    if raw.empty:
        return pd.DataFrame(), {
            "experiment": experiment,
            **parts,
            "source_path": str(path),
            "raw_rows": 0,
            "fps": FPS_DEFAULT,
            "observation_duration_s": np.nan,
            "unpaired_stop_rows": 0,
            "unclosed_start_rows": 0,
        }

    fps = _first_pipe_float(raw.get("FPS", pd.Series([""])).iloc[0])
    obs_duration = _num(raw.get("Observation duration", pd.Series([""])).iloc[0])

    rows = raw.copy()
    rows["Time"] = pd.to_numeric(rows.get("Time"), errors="coerce")
    rows["Image index"] = pd.to_numeric(rows.get("Image index"), errors="coerce")
    rows = rows.sort_values(["Time", "Image index"], na_position="last")

    open_edges: dict[tuple[str, str], list[dict]] = defaultdict(list)
    events: list[dict] = []
    unpaired_stop = 0

    for source_row, r in rows.iterrows():
        subject = str(r.get("Subject", "")).strip()
        behavior = str(r.get("Behavior", "")).strip()
        category = str(r.get("Behavioral category", "")).strip() or "Other"
        behavior_type = str(r.get("Behavior type", "")).strip()
        time_s = _num(r.get("Time", np.nan))
        frame = _num(r.get("Image index", np.nan))
        frame = np.nan if frame < 0 else frame
        key = (subject, behavior)

        edge = {
            "source_row": int(source_row),
            "subject": subject,
            "subject_role": _subject_role(subject),
            "behavior": behavior,
            "category": category,
            "time_s": time_s,
            "frame": frame,
        }

        if behavior_type == "START":
            open_edges[key].append(edge)
            continue

        if behavior_type == "STOP":
            if not open_edges[key]:
                unpaired_stop += 1
                continue
            start = open_edges[key].pop(0)
            start_time = start["time_s"]
            stop_time = time_s
            duration = max(0.0, stop_time - start_time) if np.isfinite(stop_time - start_time) else np.nan
            start_frame = start["frame"]
            stop_frame = frame
            frame_duration = (
                max(0, int(stop_frame) - int(start_frame) + 1)
                if np.isfinite(start_frame) and np.isfinite(stop_frame)
                else np.nan
            )
            events.append({
                "experiment": experiment,
                **parts,
                "event_kind": "state",
                "subject": subject,
                "subject_role": _subject_role(subject),
                "behavior": behavior,
                "category": category,
                "start_time_s": start_time,
                "stop_time_s": stop_time,
                "duration_s": duration,
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                "duration_frames": frame_duration,
                "source_start_row": int(start["source_row"]),
                "source_stop_row": int(source_row),
                "source_path": str(path),
            })
            continue

        if behavior_type == "POINT":
            events.append({
                "experiment": experiment,
                **parts,
                "event_kind": "point",
                "subject": subject,
                "subject_role": _subject_role(subject),
                "behavior": behavior,
                "category": category,
                "start_time_s": time_s,
                "stop_time_s": time_s,
                "duration_s": 0.0,
                "start_frame": frame,
                "stop_frame": frame,
                "duration_frames": 1 if np.isfinite(frame) else np.nan,
                "source_start_row": int(source_row),
                "source_stop_row": int(source_row),
                "source_path": str(path),
            })

    unclosed_start = sum(len(v) for v in open_edges.values())
    meta = {
        "experiment": experiment,
        **parts,
        "source_path": str(path),
        "raw_rows": int(len(raw)),
        "fps": fps,
        "observation_duration_s": obs_duration,
        "unpaired_stop_rows": int(unpaired_stop),
        "unclosed_start_rows": int(unclosed_start),
    }
    out = pd.DataFrame(events)
    if not out.empty:
        out = out.sort_values(["experiment", "start_time_s", "source_start_row"]).reset_index(drop=True)
    return out, meta


def _sum_duration(events: pd.DataFrame, mask) -> float:
    if events.empty:
        return 0.0
    return float(events.loc[mask, "duration_s"].fillna(0).sum())


def build_experiment_summary(events: pd.DataFrame, metas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, meta in metas.iterrows():
        exp = meta["experiment"]
        ev = events[events["experiment"] == exp].copy()
        row = meta.to_dict()
        row.update({
            "parsed_events": int(len(ev)),
            "state_events": int((ev["event_kind"] == "state").sum()) if not ev.empty else 0,
            "point_events": int((ev["event_kind"] == "point").sum()) if not ev.empty else 0,
            "first_event_s": float(ev["start_time_s"].min()) if not ev.empty else np.nan,
            "last_event_s": float(ev["stop_time_s"].max()) if not ev.empty else np.nan,
            "unique_behaviors": int(ev["behavior"].nunique()) if not ev.empty else 0,
            "unique_subjects": int(ev["subject"].nunique()) if not ev.empty else 0,
        })
        for cat in CATEGORY_ORDER:
            row[f"{cat.lower()}_events"] = int((ev["category"] == cat).sum()) if not ev.empty else 0
            row[f"{cat.lower()}_duration_s"] = _sum_duration(ev, ev["category"] == cat)

        m1tf = _sum_duration(ev, ev["subject"] == "m1tf")
        m2tf = _sum_duration(ev, ev["subject"] == "m2tf")
        row["m1tf_duration_s"] = m1tf
        row["m2tf_duration_s"] = m2tf
        row["male_to_female_duration_s"] = m1tf + m2tf
        row["male_male_duration_s"] = _sum_duration(ev, ev["subject"].isin(MALE_MALE))
        row["courtship_to_competitive_ratio"] = (
            row["courtship_duration_s"] / row["competitive_duration_s"]
            if row["competitive_duration_s"] > 0 else np.nan
        )
        smaller = min(m1tf, m2tf)
        larger = max(m1tf, m2tf)
        row["boris_male_courtship_ratio"] = larger / smaller if smaller > 0 else np.nan
        row["boris_more_courtship_subject"] = "m1tf" if m1tf >= m2tf else "m2tf"

        spawn = ev[ev["behavior"].str.lower() == "spawn"] if not ev.empty else pd.DataFrame()
        ready = ev[ev["behavior"].str.lower() == "ready"] if not ev.empty else pd.DataFrame()
        wrap = ev[ev["behavior"].str.lower() == "wrap"] if not ev.empty else pd.DataFrame()
        row["has_spawn"] = bool(len(spawn))
        row["spawn_time_s"] = float(spawn["start_time_s"].iloc[0]) if len(spawn) else np.nan
        row["spawn_frame"] = int(spawn["start_frame"].iloc[0]) if len(spawn) and np.isfinite(spawn["start_frame"].iloc[0]) else np.nan
        row["ready_time_s"] = float(ready["start_time_s"].iloc[0]) if len(ready) else np.nan
        row["wrap_events"] = int(len(wrap))
        row["wrap_total_s"] = float(wrap["duration_s"].sum()) if len(wrap) else 0.0
        row["wrap_accepted_n"] = int((wrap["duration_s"] >= 10.0).sum()) if len(wrap) else 0
        row["wrap_rejected_n"] = int((wrap["duration_s"] < 10.0).sum()) if len(wrap) else 0
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["date", "group", "block", "replicate"]).reset_index(drop=True)


def build_behavior_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    return (
        events.groupby(["category", "behavior"], dropna=False)
        .agg(
            events=("behavior", "size"),
            state_events=("event_kind", lambda s: int((s == "state").sum())),
            point_events=("event_kind", lambda s: int((s == "point").sum())),
            total_s=("duration_s", "sum"),
            median_s=("duration_s", "median"),
            experiments=("experiment", "nunique"),
        )
        .reset_index()
        .sort_values(["total_s", "events"], ascending=[False, False])
    )


def build_subject_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    return (
        events.groupby(["subject_role", "subject"], dropna=False)
        .agg(
            events=("subject", "size"),
            total_s=("duration_s", "sum"),
            median_s=("duration_s", "median"),
            experiments=("experiment", "nunique"),
        )
        .reset_index()
        .sort_values(["subject_role", "total_s"], ascending=[True, False])
    )


def write_json(path: Path, payload: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, float):
            return None if not np.isfinite(value) else value
        return value

    path.write_text(json.dumps(clean(payload), indent=2), encoding="utf-8")


def main() -> None:
    all_events = []
    metas = []
    missing = []
    for name in BORIS_FILES:
        path = BORIS_DIR / name
        if not path.exists():
            missing.append(str(path))
            continue
        events, meta = parse_boris_file(path)
        if not events.empty:
            all_events.append(events)
        metas.append(meta)

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    metas_df = pd.DataFrame(metas)
    exp_summary = build_experiment_summary(events, metas_df)
    behavior_summary = build_behavior_summary(events)
    subject_summary = build_subject_summary(events)

    category_summary = (
        events.groupby("category")
        .agg(events=("category", "size"), total_s=("duration_s", "sum"), experiments=("experiment", "nunique"))
        .reset_index()
        .sort_values("total_s", ascending=False)
        if not events.empty else pd.DataFrame()
    )

    top_behaviors = behavior_summary.head(15).copy()
    spawn_exps = exp_summary[exp_summary["has_spawn"]]
    wrap_events = events[events["behavior"].str.lower() == "wrap"].copy() if not events.empty else pd.DataFrame()
    accepted_wraps = wrap_events[wrap_events["duration_s"] >= 10.0] if not wrap_events.empty else pd.DataFrame()
    rejected_wraps = wrap_events[wrap_events["duration_s"] < 10.0] if not wrap_events.empty else pd.DataFrame()

    key_metrics = {
        "input_file_count": len(BORIS_FILES),
        "parsed_file_count": int(len(metas_df)),
        "missing_files": missing,
        "raw_rows": int(metas_df["raw_rows"].sum()) if not metas_df.empty else 0,
        "parsed_events": int(len(events)),
        "state_events": int((events["event_kind"] == "state").sum()) if not events.empty else 0,
        "point_events": int((events["event_kind"] == "point").sum()) if not events.empty else 0,
        "date_range": [
            str(exp_summary["date"].min()) if not exp_summary.empty else "",
            str(exp_summary["date"].max()) if not exp_summary.empty else "",
        ],
        "experiments_with_spawn": int(exp_summary["has_spawn"].sum()) if not exp_summary.empty else 0,
        "total_observation_min": float(exp_summary["observation_duration_s"].sum() / 60.0) if not exp_summary.empty else 0.0,
        "total_courtship_min": float(exp_summary["courtship_duration_s"].sum() / 60.0) if not exp_summary.empty else 0.0,
        "total_competitive_min": float(exp_summary["competitive_duration_s"].sum() / 60.0) if not exp_summary.empty else 0.0,
        "total_mating_min": float(exp_summary["mating_duration_s"].sum() / 60.0) if not exp_summary.empty else 0.0,
        "top_behaviors_by_duration": top_behaviors[["category", "behavior", "total_s", "events", "experiments"]].to_dict("records"),
        "wrap_accepted_n": int(len(accepted_wraps)),
        "wrap_rejected_n": int(len(rejected_wraps)),
        "wrap_accepted_median_s": float(accepted_wraps["duration_s"].median()) if len(accepted_wraps) else np.nan,
        "wrap_rejected_median_s": float(rejected_wraps["duration_s"].median()) if len(rejected_wraps) else np.nan,
        "spawn_experiments": spawn_exps[["experiment", "spawn_time_s", "spawn_frame"]].to_dict("records"),
    }

    exp_chart_cols = [
        "experiment", "date", "group", "replicate", "parsed_events",
        "courtship_duration_s", "competitive_duration_s", "mating_duration_s",
        "other_duration_s", "m1tf_duration_s", "m2tf_duration_s",
        "male_male_duration_s", "has_spawn", "spawn_time_s", "ready_time_s",
        "wrap_events", "wrap_accepted_n", "wrap_rejected_n",
        "boris_male_courtship_ratio", "boris_more_courtship_subject",
    ]
    behavior_chart_cols = [
        "category", "behavior", "events", "total_s", "median_s", "experiments",
    ]
    deck_data = {
        "metrics": key_metrics,
        "experimentSummary": exp_summary[exp_chart_cols].to_dict("records"),
        "behaviorSummary": behavior_summary[behavior_chart_cols].head(15).to_dict("records"),
        "categorySummary": category_summary.to_dict("records"),
        "subjectSummary": subject_summary.to_dict("records"),
        "figureManifest": [
            {
                "id": "B1",
                "title": "BORIS cohort scope",
                "message": "Manual ethogram coverage across the 23 user-provided observation CSVs.",
                "data": "boris_cohort_key_metrics.json",
            },
            {
                "id": "B2",
                "title": "Ethogram time by experiment",
                "message": "Courtship, competition and mating intensity vary strongly by trial.",
                "data": "boris_cohort_experiment_summary.csv",
            },
            {
                "id": "B3",
                "title": "Behavior repertoire",
                "message": "The most time-consuming manually coded states are the strongest candidates for poster biology panels.",
                "data": "boris_cohort_behavior_summary.csv",
            },
            {
                "id": "B4",
                "title": "Male-to-female asymmetry",
                "message": "BORIS m1tf/m2tf labels provide an independent, manual asymmetry check for the individual-ID tracking results.",
                "data": "boris_cohort_experiment_summary.csv",
            },
            {
                "id": "B5",
                "title": "Courtship versus male-male competition",
                "message": "Manual ethograms separate mate-choice-relevant female-directed behavior from competitive male-male behavior.",
                "data": "boris_cohort_experiment_summary.csv",
            },
            {
                "id": "B6",
                "title": "Mating endpoints and wrap duration",
                "message": "Spawn and wrap annotations identify the behavioral endpoint and separate accepted from rejected wrap bouts.",
                "data": "boris_cohort_experiment_summary.csv; boris_cohort_events.csv",
            },
            {
                "id": "B7",
                "title": "Timing of readiness and spawning",
                "message": "Ready and Spawn point events show where each manual observation places the decision window.",
                "data": "boris_cohort_experiment_summary.csv",
            },
        ],
    }

    events.to_csv(DATA / "boris_cohort_events.csv", index=False)
    metas_df.to_csv(DATA / "boris_cohort_input_files.csv", index=False)
    exp_summary.to_csv(DATA / "boris_cohort_experiment_summary.csv", index=False)
    behavior_summary.to_csv(DATA / "boris_cohort_behavior_summary.csv", index=False)
    subject_summary.to_csv(DATA / "boris_cohort_subject_summary.csv", index=False)
    category_summary.to_csv(DATA / "boris_cohort_category_summary.csv", index=False)
    write_json(DATA / "boris_cohort_key_metrics.json", key_metrics)
    write_json(DATA / "boris_cohort_deck_data.json", deck_data)
    pd.DataFrame(deck_data["figureManifest"]).to_csv(DATA / "boris_cohort_figure_manifest.csv", index=False)

    print(f"parsed {len(metas_df)} files, {len(events)} events")
    print(f"wrote {DATA}")


if __name__ == "__main__":
    main()
