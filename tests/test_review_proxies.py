from pathlib import Path

from scripts.review.proxies import build_ffmpeg_cmd, playback_step, proxy_size


def test_playback_step_and_size():
    assert playback_step(166.0, 30.0) == 6
    assert playback_step(20.0, 30.0) == 1
    assert proxy_size(1200, 1200, 800) == (800, 800)
    assert proxy_size(1920, 1200, 800) == (800, 500)


def test_ffmpeg_cmd_selects_exact_frames_and_limits_input():
    cmd = build_ffmpeg_cmd("ffmpeg", Path("in.mp4"), Path("full.mp4"), Path("play.mp4"), width=800, height=800,
                           step=6, source_fps=166.0, gpu=True, limit_seconds=3)
    assert cmd.index("-t") < cmd.index("-i")  # input option: limits both outputs
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "scale_cuda=800:800" in graph and "not(mod(n\\,6))" in graph and "83/3" in graph
    assert cmd.count("-fps_mode") == 2 and cmd[-1] == "play.mp4"
    cpu = build_ffmpeg_cmd("ffmpeg", Path("in.mp4"), Path("f.mp4"), Path("p.mp4"), width=800, height=800,
                           step=6, source_fps=166.0, gpu=False)
    assert "libx264" in cpu and "-hwaccel" not in cpu and "-t" not in cpu
