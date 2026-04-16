"""
Learned identity tracker for cam0 (top view).

Re-implements the idtrackerai-v6 identification core in our own repo:
ResNet18 + 8-D contrastive embedding, trained per-clip (self-supervised) from
same/diff pairs mined out of our Hungarian tracklets. Inference clusters
fragment-level embeddings into N=3 identities and rewrites the `fish_id`
column before outlier detection.

Public entry points:
    crops.extract_crops_from_video
    fragments.build_fragments, fragments.mine_pairs
    model.IdentityEmbedder
    train.train_embedder
    infer.refine_fish_ids
"""
