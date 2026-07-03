"""B2 head-to-head on close-approach event worklist."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("C:/Users/machi/Documents/GitHub/medaka-mating-pipeline")
OUT = ROOT / "report/20260630/k12_appearance/bakeoff"
WL = ROOT / "report/20260630/k12_appearance/event_worklist.csv"
NEW_P = "C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d_individual_id_interp_debounced.csv"
OLD_P = ROOT / "runs/dual_line/lineB_fulltimeline/step2/round5_260611_3d_individual_id_interp_debounced.csv"

PAD = 30

def load(p):
    df = pd.read_csv(p, usecols=["frame", "geometry_slot", "id_final"])
    df = df.dropna(subset=["id_final", "geometry_slot"])
    df["geometry_slot"] = df["geometry_slot"].astype(int)
    return df

print("loading...")
new = load(NEW_P)
old = load(OLD_P)
wl = pd.read_csv(WL)

# Build per-frame slot-of-label maps for fast lookup.
# slot_of[label] -> Series indexed by frame giving the slot holding that label.
def build_label_slot(df):
    out = {}
    for lab in ["F", "M1", "M2"]:
        sub = df[df.id_final == lab][["frame", "geometry_slot"]]
        # if duplicate label in a frame (shouldn't), keep first
        sub = sub.drop_duplicates("frame", keep="first").set_index("frame")["geometry_slot"]
        out[lab] = sub
    return out

new_ls = build_label_slot(new)
old_ls = build_label_slot(old)


def event_metrics(ls, s, e):
    """ls: dict label->Series(frame->slot). window [s,e]. Returns metrics."""
    frames = np.arange(s, e + 1)
    n = len(frames)
    # F slot sequence
    fser = ls["F"].reindex(frames)
    m1ser = ls["M1"].reindex(frames)
    m2ser = ls["M2"].reindex(frames)

    # n_female_flips: # times F's slot changes (consecutive non-nan distinct values)
    def count_slot_changes(ser):
        v = ser.dropna()
        if len(v) < 2:
            return 0
        a = v.values
        return int((a[1:] != a[:-1]).sum())

    n_female_flips = count_slot_changes(fser)

    # m1m2 flips: # times the (M1slot,M2slot) ordered pair swaps, i.e. M1's slot
    # becomes what M2's was. Count changes in M1's slot that coincide with exchange.
    # Simpler & robust: count frames-transitions where M1 and M2 swap slots.
    n_m1m2_flips = 0
    prev = None
    for f in frames:
        a = m1ser.get(f, np.nan)
        b = m2ser.get(f, np.nan)
        if np.isnan(a) or np.isnan(b):
            continue
        cur = (int(a), int(b))
        if prev is not None and cur != prev and cur == (prev[1], prev[0]):
            n_m1m2_flips += 1
        if prev is None or (not np.isnan(a) and not np.isnan(b)):
            prev = cur

    # female_runlen_frac: longest single-slot F run / n
    fv = fser.values
    best = 0
    cur_len = 0
    cur_val = None
    for x in fv:
        if np.isnan(x):
            cur_len = 0
            cur_val = None
            continue
        if x == cur_val:
            cur_len += 1
        else:
            cur_val = x
            cur_len = 1
        if cur_len > best:
            best = cur_len
    female_runlen_frac = best / n if n else 0.0

    # female slot at midpoint
    mid = (s + e) // 2
    # nearest non-nan F slot to mid within window
    fmid = np.nan
    if not fser.dropna().empty:
        idx = fser.dropna().index.values
        j = np.argmin(np.abs(idx - mid))
        fmid = int(fser.dropna().iloc[j])

    return dict(n_female_flips=n_female_flips, n_m1m2_flips=n_m1m2_flips,
                female_runlen_frac=round(female_runlen_frac, 4),
                female_slot_mid=fmid, n_frames=n)


# Select events: duration_s>=0.5 AND n_overlap_views==3, UNION top-50 by suspicion.
sel_prime = wl[(wl.duration_s >= 0.5) & (wl.n_overlap_views == 3)]
sel_top50 = wl.sort_values("suspicion", ascending=False).head(50)
sel = pd.concat([sel_prime, sel_top50]).drop_duplicates("rank").copy()
print(f"prime={len(sel_prime)} top50={len(sel_top50)} union={len(sel)}")

rows = []
for _, ev in sel.iterrows():
    s = int(ev["start_frame"]) - PAD
    e = int(ev["end_frame"]) + PAD
    nm = event_metrics(new_ls, s, e)
    om = event_metrics(old_ls, s, e)
    new_flips = nm["n_female_flips"] + nm["n_m1m2_flips"]
    old_flips = om["n_female_flips"] + om["n_m1m2_flips"]
    if new_flips < old_flips:
        winner = "NEW"
    elif old_flips < new_flips:
        winner = "OLD"
    else:
        winner = "TIE"
    fem_disagree = (not np.isnan(nm["female_slot_mid"]) and not np.isnan(om["female_slot_mid"])
                    and nm["female_slot_mid"] != om["female_slot_mid"])
    rows.append(dict(
        rank=int(ev["rank"]), start_frame=int(ev["start_frame"]), end_frame=int(ev["end_frame"]),
        duration_s=ev["duration_s"], pair=ev["pair"], min_distance_mm=ev["min_distance_mm"],
        antiparallel=ev["antiparallel"], net_crossing=ev["net_crossing"],
        n_overlap_views=int(ev["n_overlap_views"]), suspicion=ev["suspicion"],
        in_prime=bool((ev["duration_s"] >= 0.5) and (ev["n_overlap_views"] == 3)),
        NEW_female_flips=nm["n_female_flips"], NEW_m1m2_flips=nm["n_m1m2_flips"],
        NEW_total_flips=new_flips, NEW_female_runlen_frac=nm["female_runlen_frac"],
        NEW_female_slot_mid=nm["female_slot_mid"],
        OLD_female_flips=om["n_female_flips"], OLD_m1m2_flips=om["n_m1m2_flips"],
        OLD_total_flips=old_flips, OLD_female_runlen_frac=om["female_runlen_frac"],
        OLD_female_slot_mid=om["female_slot_mid"],
        winner=winner, female_disagree=fem_disagree,
    ))

res = pd.DataFrame(rows).sort_values("rank")
res.to_csv(OUT / "B2_worklist_compare.csv", index=False)

# Summary counts
new_better = int((res.winner == "NEW").sum())
old_better = int((res.winner == "OLD").sum())
ties = int((res.winner == "TIE").sum())

# Disagreements: top 15 by suspicion among female_disagree
dis = res[res.female_disagree].sort_values("suspicion", ascending=False).copy()
dis_out = dis[["rank", "start_frame", "end_frame", "duration_s", "pair", "min_distance_mm",
               "suspicion", "NEW_female_slot_mid", "OLD_female_slot_mid",
               "NEW_total_flips", "OLD_total_flips", "winner"]].head(15)
dis_out.to_csv(OUT / "B2_disagreements.csv", index=False)

# Avg runlen
print("=== SUMMARY ===")
print(f"events analyzed: {len(res)}")
print(f"NEW fewer flips: {new_better}")
print(f"OLD fewer flips: {old_better}")
print(f"TIE: {ties}")
print(f"NEW mean female_runlen_frac: {res.NEW_female_runlen_frac.mean():.4f}")
print(f"OLD mean female_runlen_frac: {res.OLD_female_runlen_frac.mean():.4f}")
print(f"NEW total flips sum: {res.NEW_total_flips.sum()}  OLD: {res.OLD_total_flips.sum()}")
print(f"NEW female_flips sum: {res.NEW_female_flips.sum()}  OLD: {res.OLD_female_flips.sum()}")
print(f"NEW m1m2_flips sum: {res.NEW_m1m2_flips.sum()}  OLD: {res.OLD_m1m2_flips.sum()}")
print(f"female disagreements (total): {int(res.female_disagree.sum())}")
print("--- top 15 disagreements ---")
print(dis_out.to_string(index=False))
# events with zero flips both
print(f"NEW zero-flip events: {int((res.NEW_total_flips==0).sum())}  OLD: {int((res.OLD_total_flips==0).sum())}")
