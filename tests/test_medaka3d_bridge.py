"""The medaka-3d bridge: per-frame triples -> this repo's linker, best.json, 3D CSV and provenance."""

import inspect
import json

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.medaka3d_bridge import (
    batch_cli_args,
    best_by_frame,
    link_and_write,
    linking_kwargs,
    slot_arrays,
    slot_ids,
    visible_centroid,
)


def test_linking_kwargs_are_the_batch_pipeline_arguments():
    from scripts.reconstruction_3d.cross_frame_tracking import apply_cross_frame_tracking

    kw = linking_kwargs(batch_cli_args())
    assert kw["max_jump_mm"] == 8.0 and kw["max_jump_hard_cap_mm"] == 80.0 and kw["max_gap_frames"] == 100
    assert kw["num_individuals"] == 3 and kw["velocity_prediction_weight"] == 0.0
    assert kw["source_id_hints_by_frame"] is None and kw["source_hints_override_motion_gate"] is False
    assert set(kw) <= set(inspect.signature(apply_cross_frame_tracking).parameters)


def test_slot_arrays_keep_the_first_three_slots_and_count_all():
    pts = lambda v: np.full((3, 2), float(v), np.float32)   # noqa: E731
    by_frame = {0: [pts(1), pts(2), pts(3), pts(4)], 2: [pts(5), pts(6)]}
    points, counts = slot_arrays(by_frame, n_frames=3, n_nodes=3)
    assert points.shape == (3, 3, 3, 2)
    assert counts.tolist() == [4, 0, 2]
    assert points[0, :, 0, 0].tolist() == [1, 2, 3]            # slot 4 dropped
    assert np.isnan(points[1]).all() and np.isnan(points[2, 2]).all()


def test_load_slot_ordered_sorts_by_score_and_refuses_another_camera(tmp_path):
    import pytest
    import sleap_io as sio

    from scripts.reconstruction_3d.medaka3d_bridge import load_slot_ordered

    skel = sio.Skeleton(["mouth", "pectoral", "tail"])
    video = sio.Video(filename="/data/exp/side1_cam/cam1_side.mp4", open_backend=False)
    pts = np.array([[10.0, 10.0], [30.0, 10.0], [50.0, 10.0]])
    low = sio.PredictedInstance.from_numpy(pts, skeleton=skel, score=0.5)
    high = sio.PredictedInstance.from_numpy(pts + 100, skeleton=skel, score=0.9)
    sio.Labels([sio.LabeledFrame(video=video, frame_idx=3, instances=[low, high])]).save(str(tmp_path / "s.slp"))

    by_frame, nodes, n_frames = load_slot_ordered(tmp_path / "s.slp", experiment="exp", view="side1_cam")
    assert nodes == ["mouth", "pectoral", "tail"] and n_frames == 4
    assert [float(p[0, 0]) for p in by_frame[3]] == [110.0, 10.0]          # higher score first
    with pytest.raises(ValueError, match="mix cameras"):
        load_slot_ordered(tmp_path / "s.slp", experiment="exp", view="side2_cam")


def test_visible_centroid_ignores_missing_nodes():
    p = np.array([[[0.0, 0.0], [np.nan, 5.0], [4.0, 2.0]]])
    assert np.allclose(visible_centroid(p), [[2.0, 1.0]])
    assert np.isnan(visible_centroid(np.full((1, 3, 2), np.nan))).all()


def test_best_by_frame_matches_the_pre_link_format():
    xyz = np.zeros((3, 3, 3))
    xyz[:, :, 0] = [10.0, 50.0, 300.0]               # slot 2 outside the -20..220 bounds
    xyz[2] = np.nan                                   # frame 2 unresolved
    residual = np.full((3, 3), 0.5)
    source = np.tile(np.array([[2, 0, 1]]).T[None, :, :], (3, 1, 3)).astype(np.int8)   # (F, K, V)
    source[0, 1, 2] = -1                              # slot 1 seen by two views at frame 0
    best = best_by_frame(np.arange(3), xyz, residual, source, xyz_bounds=(-20.0, 220.0))
    f0 = best[0]
    assert f0["groups"]["a"] == [10.0, 0.0, 0.0]
    assert f0["perm_cam0"] == [2, 0, -1] and f0["perm_cam2"] == [2, -1, -1]
    assert f0["recovered"] == [False, True, False] and f0["missing_views"] == [-1, 2, -1]
    assert np.isclose(f0["errors"]["a"], 0.25)        # mm^2, the unit of the refractive scorer
    assert np.isnan(f0["groups"]["c"]).all() and f0["errors"]["c"] == float("inf")
    assert all(np.isnan(best[2]["groups"][k]).all() for k in "abc")


def test_link_and_write_gives_continuous_ids_and_review_compatible_json(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    frames = np.arange(n)
    truth = np.zeros((n, 3, 3))
    truth[:, :, 0] = np.array([20.0, 80.0, 140.0]) + 0.3 * frames[:, None]
    truth[:, :, 1] = 100.0
    perm = np.array([rng.permutation(3) for _ in range(n)])          # slot order shuffled every frame
    xyz = truth[frames[:, None], perm]
    source = np.repeat(perm[:, :, None], 3, axis=2).astype(np.int8)  # slot k used detection perm[k] in every view
    best = best_by_frame(frames, xyz, np.full((n, 3), 0.4), source)
    out = link_and_write(best, out_dir=tmp_path, tag="t", top_tracks_csv=None, calibration_dir=tmp_path,
                         all_frames=frames, log=lambda *_: None)

    table = pd.read_csv(out["csv_3d"])
    for _, g in table.groupby("fish_id"):
        assert np.abs(np.diff(g.sort_values("frame")["X"])).max() < 1.0      # no identity jumps
    payload = json.loads((tmp_path / "t_best.json").read_text())
    assert payload["id_labels"] == ["ID_1", "ID_2", "ID_3"]
    ids = slot_ids(out["best_tracked"], frames, 3)
    for f in (0, 30, 59):
        info = payload["frames"][str(f)]
        for k in range(3):
            assert info["cam0_to_id"][source[f, k, 0]] == f"ID_{ids[f, k] + 1}"
    prov = json.loads((tmp_path / "t_reconstruction_provenance.json").read_text())
    assert prov["triangulation"]["provenance"]["model_folder"] == str(tmp_path)
    assert out["csv_identified"] is None                              # no top tracks given: no bind
