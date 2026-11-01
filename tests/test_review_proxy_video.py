"""ProxyVideo: frame-exact reads after seeks, forwards and backwards."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _video(path, n, fps):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 96))
    if not w.isOpened():
        pytest.skip("no mp4 writer")
    for i in range(n):
        img = np.zeros((96, 96, 3), np.uint8)
        for bit in range(9):                        # the frame index as 3x3 black / white blocks
            if i >> bit & 1:
                r, c = divmod(bit, 3)
                img[32 * r:32 * r + 32, 32 * c:32 * c + 32] = 255
        w.write(img)
    w.release()


def _index(img):
    g = img.mean(axis=2)
    return sum(1 << bit for bit in range(9) if g[32 * (bit // 3) + 8:32 * (bit // 3) + 24,
                                                 32 * (bit % 3) + 8:32 * (bit % 3) + 24].mean() > 128)


@pytest.mark.parametrize("fps", [166 / 6, 30.0])
def test_seeks_land_on_the_frame_asked_for(tmp_path, fps):
    from scripts.review.review_gui import ProxyVideo, proxy_fps

    path = tmp_path / "v.mp4"
    _video(path, 400, fps)
    pv = ProxyVideo(path, cache=8, fps=fps)
    for k in (0, 1, 150, 151, 399, 200, 37, 380, 2):         # far jumps ahead and back
        assert _index(pv.get(k)) == k
    pv.close()
    assert proxy_fps({"views": {"top_cam": {"source_fps": 166.0, "play_step": 6}}}, "top_cam", True) == 166 / 6
