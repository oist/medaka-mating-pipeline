"""Step 2 male (Part B): resolve M1/M2 per frame from appearance and emit the full 3-fish
individual-ID track (F + M1 + M2).

Now a thin wrapper over the shared production core
``sam2_id_experiment.src.id_model.per_frame_resolve`` (the resolution logic lives there and is unit-
tested + reused by the production stage 5 of 40_female_anchor_pipeline). This script keeps the K_1_2
research paths + the validation prints.

Naming: M2 = M_beta = the reference-window (152k) aligned-gid_2 male; M1 = M_alpha = aligned-gid_0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from sam2_id_experiment.src.id_model import per_frame_resolve as pfr  # noqa: E402

NFRAMES = 348600
MARGIN_THR = 0.30
HIGH_DIST = 25

male = pd.read_csv(HERE / "step2" / f"per_frame_male_0_{NFRAMES-1}.csv")
res = pd.read_csv(HERE / "step2" / "round5_260611_3d_identified_sexed_resolved.csv")

female_ff = pfr.female_dense_from_track(res, NFRAMES)
out = pfr.resolve_male_track(male, female_ff, NFRAMES, MARGIN_THR, HIGH_DIST)
beta_final, male_conf = out["beta_final"], out["male_conf"]
af, abeta = out["anchor_frames"], out["anchor_betas"]
print(f"confident male anchors: {len(af)}/{len(male)} sampled frames "
      f"({len(af)/len(male):.1%})")

chg = np.where(abeta[1:] != abeta[:-1])[0]
runs = np.diff(np.r_[0, chg + 1, len(abeta)]) if len(abeta) else np.array([0])
print(f"appearance-pinned male reassignments: {len(chg)}  "
      f"(anchor run lengths: median {int(np.median(runs))}, max {int(runs.max())})")

frames = res.frame.to_numpy()
slots = res.geometry_slot.to_numpy().astype(int)
res["individual_id"] = pfr.assemble_individual_labels(frames, slots, female_ff, beta_final, NFRAMES)
res["app_male_beta_slot"] = beta_final[frames]
res["male_conf"] = np.asarray(male_conf)[frames]
res["indiv_conf"] = np.where(res.individual_id == "F", res.app_conf, res.male_conf)

counts = res.groupby("frame").individual_id.value_counts().unstack(fill_value=0)
ok = int(((counts.get("F", 0) == 1) & (counts.get("M1", 0) == 1) & (counts.get("M2", 0) == 1)).sum())
print(f"frames with exactly 1 F + 1 M1 + 1 M2: {ok}/{len(counts)}")

hi_male = res[(res.individual_id != "F") & (res.male_conf == "high")].drop_duplicates("frame")
agree_geo = float((hi_male.app_male_beta_slot == 2).mean())
print(f"\nM_beta == geometry slot 2 (raw-geometry agreement, confident frames): {agree_geo:.3f} "
      f"(uninterpretable: appearance was trained on window_align-ALIGNED gid_2, not raw slot 2)")
print("\nindiv_conf (per fish-frame):", {k: int(v) for k, v in res.indiv_conf.value_counts().items()})
print("male_conf  (per fish-frame):", {k: int(v) for k, v in res.male_conf.value_counts().items()})

out_csv = HERE / "step2" / "round5_260611_3d_individual_id.csv"
res.to_csv(out_csv, index=False)
pd.DataFrame({"anchor_before": af[chg], "anchor_after": af[chg + 1],
              "beta_slot_before": abeta[chg], "beta_slot_after": abeta[chg + 1]}).to_csv(
    HERE / "step2" / "male_reassignments.csv", index=False)
print(f"\nwrote {out_csv}")
