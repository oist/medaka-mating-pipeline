"""Tests for the Detectron2 -> UDMT XMem first-frame init adapter (pure logic)."""

import numpy as np

from scripts.experiments.udmt_saion.make_xmem_init import (
    build_binary_union, match_fish_to_masks,
)


def test_match_drops_reflection():
    # 3 fish; 4 mask centroids (#3 is a far reflection)
    fish = {0: (100.0, 100.0), 12: (300.0, 100.0), 15: (200.0, 300.0)}
    centroids = [(102.0, 98.0), (298.0, 103.0), (205.0, 297.0), (900.0, 900.0)]
    m = match_fish_to_masks(fish, centroids, gate_px=50.0)
    assert set(m) == {0, 12, 15}
    assert 3 not in m.values()          # reflection unmatched
    assert m[0] == 0 and m[12] == 1 and m[15] == 2


def test_match_unique_assignment():
    # two fish near the same mask -> each gets a distinct mask (no double-assign)
    fish = {0: (100.0, 100.0), 1: (110.0, 100.0)}
    centroids = [(101.0, 100.0), (130.0, 100.0)]
    m = match_fish_to_masks(fish, centroids, gate_px=60.0)
    assert len(set(m.values())) == 2


def test_match_gate_excludes_far():
    fish = {0: (100.0, 100.0)}
    centroids = [(400.0, 400.0)]
    assert match_fish_to_masks(fish, centroids, gate_px=50.0) == {}


def test_build_binary_union():
    a = np.zeros((10, 10), bool); a[2:4, 2:4] = True
    b = np.zeros((10, 10), bool); b[6:8, 6:8] = True
    u = build_binary_union([a, b], (10, 10))
    assert u.dtype == np.uint8
    assert set(np.unique(u)) == {0, 1}
    assert u.sum() == 8
    assert u[2, 2] == 1 and u[6, 6] == 1 and u[0, 0] == 0
