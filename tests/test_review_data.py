import numpy as np

from scripts.review.review_data import (
    OrientationParams,
    cross_view_flips,
    drop_overlapped_views,
    flip_nodes,
    end_jumps,
    merge_risks,
    orient_by_projection,
    orientation_segments,
    overlap_distance,
    runs,
    segment_distance,
    short_flip_mask,
    viterbi_orientation,
)


def _swimming_fish(n=120, speed=0.5, half_length=12.5):
    centre = np.stack([np.arange(n) * speed, np.zeros(n), np.zeros(n)], axis=1)
    head = centre + [half_length, 0, 0]
    tail = centre - [half_length, 0, 0]
    velocity = np.tile([speed, 0.0, 0.0], (n, 1))
    return head, tail, velocity


def test_viterbi_keeps_orientation_through_short_all_view_flip():
    head, tail, vel = _swimming_fish()
    votes = np.full(len(head), 3)
    votes[40:46] = 0  # every view swapped mouth/tail for 6 frames
    state, margin = viterbi_orientation(head, tail, votes, np.full(len(head), 3), vel, OrientationParams())
    assert (state == 0).all()
    assert (margin[40:46] < 0).all()  # emission alone preferred the flip; continuity overruled it


def test_viterbi_flips_consistently_reversed_labels():
    head, tail, vel = _swimming_fish()
    # SLEAP calls the tail "mouth" everywhere and the fish swims towards the real head.
    state, _ = viterbi_orientation(tail, head, np.full(len(head), 3), np.full(len(head), 3), vel,
                                   OrientationParams())
    assert (state == 0).all()  # votes win over the weak motion prior when all views agree
    state, _ = viterbi_orientation(tail, head, np.zeros(len(head)), np.full(len(head), 3), vel,
                                   OrientationParams())
    assert (state == 1).all()


def test_viterbi_marks_frames_without_both_ends_invalid():
    head, tail, vel = _swimming_fish(n=10)
    head[3] = np.nan
    state, _ = viterbi_orientation(head, tail, np.full(10, 3), np.full(10, 3), vel, OrientationParams())
    assert state[3] == -1 and (np.delete(state, 3) == 0).all()


class _MeanCal:
    """Fake calibration: 3D = mean of finite views, score = mean squared spread."""

    def evaluate_hypotheses(self, obs):
        obs = np.asarray(obs, float)
        with np.errstate(all="ignore"):
            mean = np.nanmean(obs, axis=-2)
            spread = np.nanmean(np.sum((obs - mean[..., None, :]) ** 2, axis=-1), axis=-1)
        n = np.isfinite(obs).all(-1).sum(-1)
        spread = np.where(n >= 2, spread, np.inf)
        xyz = np.concatenate([mean, np.zeros(mean.shape[:-1] + (1,))], axis=-1)
        return np.where(n[..., None] >= 2, xyz, np.nan), spread


def test_cross_view_flips_detects_swapped_side_view():
    kp = np.zeros((2, 1, 3, 3, 2))
    kp[..., 0, :] = [10.0, 0.0]   # mouth
    kp[..., 1, :] = [0.0, 0.0]    # pectoral
    kp[..., 2, :] = [-10.0, 0.0]  # tail
    kp[1, 0, 2] = flip_nodes(kp[1, 0, 2], np.array(True), 0, 2)  # side2 swapped in frame 1
    flips, resid = cross_view_flips(_MeanCal(), kp, 0, 2)
    assert flips[0, 0].tolist() == [False, False, False]
    assert flips[1, 0].tolist() == [False, False, True]
    assert np.allclose(resid, 0.0)


def test_runs_and_merge_risks():
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    assert runs(mask) == [(1, 2), (4, 4), (7, 9)]
    assert runs(mask, 2) == [(1, 2), (7, 9)]
    rows = [{"type": "jump", "start": 100, "end": 100, "peak": 100, "fish": "ID_1", "score": 9.0, "details": "a"},
            {"type": "close", "start": 105, "end": 110, "peak": 107, "fish": "ID_1|ID_2", "score": 3.0, "details": "b"},
            {"type": "gap", "start": 400, "end": 420, "peak": 400, "fish": "ID_3", "score": 21, "details": "c"}]
    out = merge_risks(rows, pad_frames=10, first=0, last=1000)
    assert list(out["frame_start"]) == [90, 390]
    assert list(out["frame_end"]) == [120, 430]
    assert out.iloc[0]["types"] == "close;jump" and out.iloc[0]["frame_peak"] == 107


def test_segment_distance_parallel_crossing_and_missing():
    a, b = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    assert np.isclose(segment_distance(a, b, np.array([0.0, 3.0]), np.array([10.0, 3.0])), 3.0)
    assert segment_distance(a, b, np.array([5.0, -5.0]), np.array([5.0, 5.0])) == 0.0
    assert np.isclose(segment_distance(a, b, np.array([13.0, 4.0]), np.array([20.0, 4.0])), 5.0)
    assert np.isnan(segment_distance(a, b, np.array([np.nan, 0.0]), np.array([1.0, 1.0])))


def _three_fish_side_overlap():
    """Fish 0 and 1 overlap in side1 only (crossing skeletons); fish 2 is far away everywhere."""
    kp = np.zeros((1, 3, 3, 3, 2))
    for k, y in enumerate([0.0, 100.0, 300.0]):
        kp[0, k, :, :, 0] = [0.0, 50.0, 100.0]
        kp[0, k, :, :, 1] = y
    kp[0, 1, 1] = [[50.0, -40.0], [50.0, 0.0], [50.0, 40.0]]   # side1: fish 1 crosses fish 0
    return kp


def test_overlap_distance_flags_only_the_crossing_view():
    d = overlap_distance(_three_fish_side_overlap(), [(0, 1), (1, 2)])
    assert d[0, 0, 1] == 0.0 and d[0, 1, 1] == 0.0
    assert d[0, 0, 0] == 100.0 and d[0, 2, 1] > 100.0


def test_drop_overlapped_views_keeps_two_views_per_node():
    kp = _three_fish_side_overlap()
    present = np.ones((1, 3, 3), bool)
    dist = overlap_distance(kp, [(0, 1), (1, 2)])
    out, dropped = drop_overlapped_views(kp, present, dist, overlap_px=20.0)
    assert dropped[0, :, 1].tolist() == [True, True, False]
    assert not dropped[0, :, [0, 2]].any()
    assert np.isnan(out[0, 0, 1]).all() and np.isfinite(out[0, 2, 1]).all()
    # Fish 0's tail is missing in top: dropping side1 would leave it in one view, so side1 keeps the tail only.
    kp[0, 0, 0, 2] = np.nan
    out, dropped = drop_overlapped_views(kp, present, dist, overlap_px=20.0)
    assert dropped[0, 0, 1]
    assert np.isnan(out[0, 0, 1, :2]).all() and np.isfinite(out[0, 0, 1, 2]).all()
    # A fish seen in only two views never loses one.
    present[0, 0, 2] = False
    _, dropped = drop_overlapped_views(kp, present, dist, overlap_px=20.0)
    assert not dropped[0, 0].any()


def test_orient_by_projection():
    kp = np.array([[[10.0, 0.0], [0.0, 0.0], [-10.0, 0.0]]])
    ends = np.array([[[9.0, 1.0], [-9.0, 0.0]]])           # projected head, tail
    assert not orient_by_projection(kp, ends, 0, 2)[0]
    assert orient_by_projection(kp, ends[:, ::-1], 0, 2)[0]
    assert not orient_by_projection(kp, np.full_like(ends, np.nan), 0, 2)[0]
    # One detected end: labelled by the nearer projected end.
    one = kp.copy()
    one[0, 2] = np.nan
    assert orient_by_projection(one, ends[:, ::-1], 0, 2)[0]
    assert not orient_by_projection(np.full_like(kp, np.nan), ends, 0, 2)[0]


def test_short_head_tail_swaps_are_undone_and_long_ones_split_segments():
    head, tail, _ = _swimming_fish(100)
    h, t = head.copy(), tail.copy()
    h[40:42], t[40:42] = tail[40:42], head[40:42]     # 2 frames: labelling error
    h[70:], t[70:] = tail[70:], head[70:]             # 30 frames to the end: kept, becomes a segment
    h[10] = np.nan
    assert list(np.flatnonzero(short_flip_mask(h, t, 8, 30))) == [40, 41]
    assert list(np.flatnonzero(end_jumps(h, t, 30)[2])) == [40, 42, 70]
    seg = orientation_segments(h, t, 30)
    assert seg[10] == -1 and seg[9] == seg[11] == 0 and seg[69] == 2 and seg[70] == 3
    # Short swaps at the start of a stretch take the parity of the first long segment.
    h2, t2 = head.copy(), tail.copy()
    h2[:3], t2[:3] = tail[:3], head[:3]
    assert list(np.flatnonzero(short_flip_mask(h2, t2, 8, 30))) == [0, 1, 2]
    # A stretch without any long segment is left alone.
    assert not short_flip_mask(h2[:6], t2[:6], 8, 30).any()


def _ortho_project(points):
    """Three orthographic cameras (top: x,y; side 1: x,z; side 2: y,z), 2 px per mm."""
    pts = np.asarray(points, float)
    return np.stack([pts[..., [0, 1]], pts[..., [0, 2]], pts[..., [1, 2]]], axis=-2) * 2.0


def test_fill_nodes_completes_every_node_and_records_its_source():
    from scripts.review.review_data import (
        NODE_BODY_MODEL,
        NODE_ONE_VIEW,
        NODE_TRIANGULATED,
        fill_nodes,
    )

    n = 60
    t = np.arange(n)
    pect = np.stack([50 + 0.5 * t, 80 + 5 * np.sin(t / 10), 30 + 2 * np.cos(t / 15)], axis=1)
    axis = np.stack([np.cos(t / 20), np.sin(t / 20), np.zeros(n)], axis=1)
    truth = np.stack([pect + 9 * axis, pect, pect - 16 * axis], axis=1)[:, None]      # (F, 1, N=3, 3)
    xyz = (truth[:, :, 0] + truth[:, :, 2]) / 2
    kp = _ortho_project(truth)                                                         # (F, 1, 3, V, 2)
    kp = np.moveaxis(kp, 3, 2).astype(np.float32)                                      # (F, 1, V, N, 2)
    nodes = truth.copy()
    # Mouth seen only in side 1 on frames 20-24; tail seen in no view on frames 40-42.
    nodes[20:25, 0, 0] = np.nan
    kp[20:25, 0, [0, 2], 0] = np.nan
    nodes[40:43, 0, 2] = np.nan
    kp[40:43, 0, :, 2] = np.nan
    filled, source = fill_nodes(nodes, kp, xyz, 0, 2, _ortho_project)
    assert np.isfinite(filled).all()
    assert (source[20:25, 0, 0] == NODE_ONE_VIEW).all() and (source[40:43, 0, 2] == NODE_BODY_MODEL).all()
    assert (source[:20] == NODE_TRIANGULATED).all()
    err_one = np.linalg.norm(filled[20:25, 0, 0] - truth[20:25, 0, 0], axis=1)
    err_model = np.linalg.norm(filled[40:43, 0, 2] - truth[40:43, 0, 2], axis=1)
    assert err_one.max() < 1.0, err_one          # one view + body length pins the point down
    assert err_model.max() < 2.0, err_model      # short gap: interpolated body offset
