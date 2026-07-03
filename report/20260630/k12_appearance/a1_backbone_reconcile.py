import numpy as np
import pandas as pd
from itertools import permutations

NEW = r"C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d.csv"
OLD = r"Z:/StephensU/medaka/reconstruction_3d/round5_260611/20251225_K_1_2_FCMC/round5_260611_3d_identified.csv"
OUT = r"report/20260630/k12_appearance/backbone_divergence.csv"

new = pd.read_csv(NEW, usecols=["frame","fish_id","X","Y","Z","is_interpolated","source"])
old = pd.read_csv(OLD, usecols=["frame","fish_id","X","Y","Z","final_fish_id"])

def cov(df, name):
    fr = df["frame"].unique()
    present = df.dropna(subset=["X","Y","Z"])
    pc = present.groupby("frame")["fish_id"].nunique()
    all3 = (pc==3).sum()
    print(f"[{name}] frames {df['frame'].min()}-{df['frame'].max()} nframes={len(fr)} rows={len(df)} present_rows={len(present)} frames_all3={all3}")
    return present

print("=== COVERAGE ===")
new_p = cov(new, "NEW")
old_p = cov(old, "OLD")
print("NEW interpolated rows:", int(new["is_interpolated"].sum()), "of", len(new))
print("NEW source counts:", new["source"].value_counts().to_dict())

# pivot to frame -> {ID: (x,y,z)} for present-only
def pivot(df):
    d = {}
    for fr, g in df.groupby("frame"):
        arr = g[["X","Y","Z"]].to_numpy(float)
        d[fr] = arr  # rows in some order; we match by position so order irrelevant for the unmatched set
    return d

# We need per-slot labels to assess permutation. Build frame -> dict label->xyz for frames with all 3 in BOTH
def pivot_labeled(df):
    out = {}
    for fr, g in df.groupby("frame"):
        g = g.dropna(subset=["X","Y","Z"])
        if g["fish_id"].nunique()==3 and len(g)==3:
            out[fr] = {row.fish_id: np.array([row.X,row.Y,row.Z]) for row in g.itertuples()}
    return out

new_l = pivot_labeled(new_p)
old_l = pivot_labeled(old_p)
common = sorted(set(new_l) & set(old_l))
print("\n=== ALIGNMENT (frames all-3 in both) ===")
print("n common all3 frames:", len(common))

labels = ["ID_1","ID_2","ID_3"]
perms = list(permutations(range(3)))
records = []
for fr in common:
    O = np.array([old_l[fr][l] for l in labels])  # 3x3
    N = np.array([new_l[fr][l] for l in labels])
    # find perm p mapping new labels to old labels minimizing dist: assign N[p] to O
    best = None
    for p in perms:
        d = np.linalg.norm(O - N[list(p)], axis=1)
        c = d.sum()
        if best is None or c < best[0]:
            best = (c, p, d)
    cost, p, d = best
    rmse = np.sqrt((d**2).mean())
    is_identity = (p == (0,1,2))
    # min interfish dist in NEW
    pts = N
    dd = [np.linalg.norm(pts[i]-pts[j]) for i in range(3) for j in range(i+1,3)]
    records.append((fr, rmse, is_identity, p, min(dd)))

rec = pd.DataFrame(records, columns=["frame","rmse_mm","identity_perm","perm","min_interfish_mm"])
n_id = rec["identity_perm"].sum()
print(f"identity-perm frames: {n_id}/{len(rec)} = {100*n_id/len(rec):.2f}%")
print(f"disagreement (non-identity) frames: {len(rec)-n_id} = {100*(len(rec)-n_id)/len(rec):.2f}%")
print(f"matched-slot RMSE mm: mean={rec['rmse_mm'].mean():.3f} median={rec['rmse_mm'].median():.3f} p95={rec['rmse_mm'].quantile(.95):.3f} max={rec['rmse_mm'].max():.3f}")
print("perm type counts (after best-match):", {str(k):int(v) for k,v in rec['perm'].astype(str).value_counts().items()})

# Re-bind ambiguity: among ALL common frames where new has present slots, if we re-bind OLD appearance to NEW by nearest slot,
# how often is the nearest assignment ambiguous? Use margin = 2nd-nearest - nearest for each old slot vs new slots.
print("\n=== RE-BIND AMBIGUITY (nearest-slot OLD->NEW) ===")
amb_count = 0; tot = 0; margins = []
for fr in common:
    O = np.array([old_l[fr][l] for l in labels])
    N = np.array([new_l[fr][l] for l in labels])
    for i in range(3):
        dists = np.sort(np.linalg.norm(N - O[i], axis=1))
        margins.append(dists[1]-dists[0])
        tot += 1
        if (dists[1]-dists[0]) < 5.0:  # < 5mm margin = ambiguous
            amb_count += 1
margins = np.array(margins)
print(f"nearest-slot assignments: {tot}, ambiguous(<5mm margin): {amb_count} = {100*amb_count/tot:.2f}%")
print(f"margin mm: median={np.median(margins):.2f} p10={np.percentile(margins,10):.2f} p5={np.percentile(margins,5):.2f}")

# Timeline of disagreement: group disagreement frames into ranges
print("\n=== DISAGREEMENT TIMELINE ===")
dis = rec[~rec["identity_perm"]].sort_values("frame")
print("n disagreement frames:", len(dis))
if len(dis):
    fr = dis["frame"].to_numpy()
    # cluster into contiguous-ish ranges (gap > 50 frames starts new range)
    ranges = []
    start = prev = fr[0]; buf=[dis.iloc[0]]
    for k in range(1,len(fr)):
        if fr[k]-prev > 50:
            ranges.append((start, prev, buf))
            start = fr[k]; buf=[]
        buf.append(dis.iloc[k]); prev = fr[k]
    ranges.append((start, prev, buf))
    rrec = []
    for s,e,buf in ranges:
        sub = pd.DataFrame(buf)
        rrec.append((int(s), int(e), len(sub), float(sub["rmse_mm"].mean()),
                     "non_identity", float(sub["min_interfish_mm"].min())))
    rdf = pd.DataFrame(rrec, columns=["frame_range_start","end","n_frames","rmse_mm","perm_type","min_interfish_mm"])
    rdf = rdf.sort_values("n_frames", ascending=False)
    print("n disagreement ranges:", len(rdf))
    print(rdf.head(15).to_string())
    rdf.to_csv(OUT, index=False)
    print("wrote", OUT, "rows:", len(rdf))
    # close-contact characterization of top 10 largest ranges
    top = rdf.head(10)
    print("\nTop-10 largest divergence ranges, min interfish mm:")
    print(top[["frame_range_start","end","n_frames","min_interfish_mm","rmse_mm"]].to_string())
    print(f"\nfrac of disagreement ranges with min_interfish < 8mm (close contact): "
          f"{(rdf['min_interfish_mm']<8).mean():.2f}")
else:
    pd.DataFrame(columns=["frame_range_start","end","n_frames","rmse_mm","perm_type","min_interfish_mm"]).to_csv(OUT, index=False)
    print("No disagreement; wrote empty", OUT)
