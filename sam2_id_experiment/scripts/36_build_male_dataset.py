from __future__ import annotations

"""Stage 2 (males): female-filter a pooled seg-ID dataset, build a male-only
contrastive dataset, and 2-colour male tracklets by cannot-link propagation.

Rationale (see SEGID_PILOT_FINDINGS): the female is ~perfectly separable, so we
drop her and give a fresh model's full capacity to the subtle 2-male axis. With
only 2 males the same-frame cannot-link is TRANSITIVE -- co-occurrence chains
2-colour most male tracklets with no appearance at all; appearance is only needed
to bridge breaks. This script:

  1. selects male crops (``global_id`` != female id; female id = the cleanly
     separable one, default 1),
  2. writes a male-only dataset (``crops.h5``/``pairs.parquet``/metadata) for
     ``train_resnet`` -- must-link = same (view,tracklet), cannot-link = same
     frame+view, different tracklet,
  3. builds the per-view male-tracklet cannot-link graph (edge = two male
     tracklets share >=1 frame in a view), 2-colours each connected component,
     and reports coverage (how many tracklets are pinned by propagation alone),
     bipartite conflicts (odd cycles = impossible with 2 males -> impure
     tracklet/reflection), and agreement with the geometry ``global_id``.

The propagation 2-colouring is the geometry-free male baseline; the trained male
embedding only has to beat it at the breaks. Run::

    python -m sam2_id_experiment.scripts.36_build_male_dataset \
        --pooled-dir <pooled_sides> --out-dir <male_dataset> --female-gid 1
"""

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.id_dataset.pair_sampler import sample_constrained_pairs


def cooccurrence_graph(meta: pd.DataFrame):
    """Edges between male view_tracklets that share a frame within the same view."""
    edges = set()
    for (_v, _f), g in meta.groupby(["view", "frame"]):
        vts = sorted(set(int(t) for t in g["view_tracklet"]))
        for i in range(len(vts)):
            for j in range(i + 1, len(vts)):
                edges.add((vts[i], vts[j]))
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    return adj


def two_color(adj, nodes):
    """2-colour the cannot-link graph by BFS. Returns (color{node:0/1},
    conflicts[list of components with odd cycle], components[list[set]])."""
    color, conflicts, comps = {}, [], []
    for start in nodes:
        if start in color:
            continue
        color[start] = 0
        comp = {start}
        q = deque([start])
        ok = True
        while q:
            u = q.popleft()
            for w in adj.get(u, ()):
                if w not in color:
                    color[w] = 1 - color[u]; comp.add(w); q.append(w)
                elif color[w] == color[u]:
                    ok = False
        comps.append(comp)
        if not ok:
            conflicts.append(comp)
    return color, conflicts, comps


def build_parser():
    p = argparse.ArgumentParser(description="Build male-only dataset + cannot-link 2-colouring.")
    p.add_argument("--pooled-dir", required=True, help="output of 34_assemble_segid_training")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--female-gid", type=int, default=1)
    p.add_argument("--positive-pairs-per-tracklet", type=int, default=600)
    p.add_argument("--negative-pairs-per-frame", type=int, default=8)
    p.add_argument("--max-negative-pairs", type=int, default=80000)
    p.add_argument("--min-global-id-purity", type=float, default=0.9,
                   help="drop weak labels whose modal 3D global_id purity is below this threshold")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    pooled = Path(args.pooled_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(pooled / "crop_metadata.parquet").reset_index(drop=True)
    with h5py.File(pooled / "crops.h5", "r") as h:
        crops = h["crops"][...]

    purity = meta.get("global_id_purity", pd.Series(1.0, index=meta.index))
    male = meta[(meta.qc_status == "clean_train") & (meta.global_id >= 0)
                & (purity.astype(float) >= float(args.min_global_id_purity))
                & (meta.global_id != args.female_gid)].copy()
    print(f"male crops {len(male)} over {male.view_tracklet.nunique()} view-tracklets "
          f"(dropped female gid={args.female_gid})", flush=True)

    # reindex crops -> contiguous male crop ids
    male = male.sort_values("crop_id").reset_index(drop=True)
    old_ids = male["crop_id"].to_numpy()
    male["crop_id"] = np.arange(len(male))
    male_crops = crops[old_ids]

    # build male-only pairs (cannot-link = same frame+view via view_tracklet grouping)
    pm = male.copy(); pm["tracklet_id"] = pm["view_tracklet"]
    pairs = sample_constrained_pairs(pm, {
        "positive_pairs_per_tracklet": args.positive_pairs_per_tracklet,
        "negative_pairs_per_frame": args.negative_pairs_per_frame,
        "max_negative_pairs": args.max_negative_pairs, "random_seed": 21})

    with h5py.File(out / "crops.h5", "w") as h:
        h.create_dataset("crops", data=male_crops, compression="gzip")
    male.to_parquet(out / "crop_metadata.parquet", index=False)
    pairs.to_parquet(out / "pairs.parquet", index=False)

    # cannot-link 2-colouring per view (co-occurrence is within-view)
    prop = {}
    report_views = {}
    for v, gv in male.groupby("view"):
        adj = cooccurrence_graph(gv)
        nodes = sorted(set(int(t) for t in gv["view_tracklet"]))
        color, conflicts, comps = two_color(adj, nodes)
        for t, c in color.items():
            prop[int(t)] = int(c)
        pinned = sum(len(c) for c in comps if len(c) >= 2)
        # agreement with geometry global_id within each component (best of 2 alignments)
        agree = tot = 0
        tgid = gv.groupby("view_tracklet")["global_id"].agg(lambda s: int(s.value_counts().idxmax()))
        for comp in comps:
            if len(comp) < 2:
                continue
            sub = [(t, prop[int(t)], int(tgid[t])) for t in comp if t in tgid.index]
            gids = sorted(set(g for _, _, g in sub))
            if len(gids) < 2:
                continue
            a0 = sum(1 for _, c, g in sub if (c == 0) == (g == gids[0]))
            agree += max(a0, len(sub) - a0); tot += len(sub)
        report_views[v] = {
            "n_male_tracklets": len(nodes), "n_components": len(comps),
            "largest_component": max((len(c) for c in comps), default=0),
            "tracklets_pinned_by_cooccurrence": pinned,
            "frac_pinned": round(pinned / max(len(nodes), 1), 3),
            "bipartite_conflicts": len(conflicts),
            "propagation_vs_geometry_agreement": round(agree / tot, 3) if tot else None,
        }

    male["male_prop_color"] = male["view_tracklet"].map(prop).fillna(-1).astype(int)
    male.to_parquet(out / "crop_metadata.parquet", index=False)
    summary = {"n_male_crops": int(len(male)), "n_male_tracklets": int(male.view_tracklet.nunique()),
               "pairs_pos": int((pairs.label == 1).sum()), "pairs_neg": int((pairs.label == 0).sum()),
               "female_gid": args.female_gid,
               "min_global_id_purity": float(args.min_global_id_purity),
               "per_view_propagation": report_views}
    (out / "male_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
