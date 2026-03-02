from __future__ import annotations

"""
三视角 + BORIS 行为时间轴合成视频
================================

功能概述
--------
- 输入：仅提供 top 视角视频路径，以及对应的 BORIS 事件 CSV。
- 自动：根据 top 视频所在文件夹，自动推断 side1 / side2 视角视频。
- 输出：一行是三视角原始视频（top, side1, side2 并排），
        第二行是一条覆盖整段视频的行为时间轴（来自 BORIS CSV）：
        - 不同行为用不同颜色表示；
        - 图例展示「颜色 → 行为 + 个体」；
        - 带有实时移动的时间光标，时间格式为 mm:ss:ff（分:秒:帧）。

依赖：opencv-python, numpy, pandas
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # pragma: no cover - 运行时才会触发
    cv2 = None


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------


@dataclass
class BehaviorEvent:
    start_s: float
    end_s: float
    behavior: str
    subject: str

    @property
    def label(self) -> str:
        """行为 + 个体，用于图例显示。"""
        b = self.behavior or "UnknownBehavior"
        s = self.subject or "UnknownSubject"
        return f"{s}: {b}"


# -----------------------------------------------------------------------------
# 三视角视频路径推断
# -----------------------------------------------------------------------------


def infer_three_view_set_from_top(
    video_top: Path,
    video_side1: Optional[Path] = None,
    video_side2: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    根据 top 视角视频自动推断 side1 / side2。

    假设目录结构类似：
        D:/videos_raw/20251225_K_1_2_FCMC/
            top_cam/cam0_top.mp4
            side1_cam/cam1_side.mp4
            side2_cam/cam2_side.mp4

    若显式提供 video_side1 / video_side2，则优先生效。
    """
    video_top = Path(video_top).resolve()
    if not video_top.exists():
        raise FileNotFoundError(f"top 视角视频不存在: {video_top}")

    root = video_top.parents[1] if len(video_top.parents) >= 2 else video_top.parent

    out: Dict[str, Path] = {"top": video_top}

    if video_side1 is not None and Path(video_side1).exists():
        out["side1"] = Path(video_side1).resolve()
    if video_side2 is not None and Path(video_side2).exists():
        out["side2"] = Path(video_side2).resolve()
    if len(out) == 3:
        return out

    # 1）优先尝试固定命名：side1_cam/cam1_side.mp4, side2_cam/cam2_side.mp4
    direct_candidates = {
        "side1": root / "side1_cam" / "cam1_side.mp4",
        "side2": root / "side2_cam" / "cam2_side.mp4",
    }
    for view, cand in direct_candidates.items():
        if view in out:
            continue
        if cand.exists():
            out[view] = cand.resolve()

    if len(out) == 3:
        return out

    # 2）回退：在 root 下递归搜索包含 side1/side2 关键字的 mp4
    all_mp4 = list(root.rglob("*.mp4"))

    def pick_one(candidates: List[Path]) -> Optional[Path]:
        if not candidates:
            return None
        # 按路径字符串排序，保证稳定结果
        return sorted(candidates, key=lambda p: str(p))[-1].resolve()

    if "side1" not in out:
        side1_cands = [
            p
            for p in all_mp4
            if re.search(r"(side1|cam1_side)", p.stem, flags=re.IGNORECASE)
        ]
        p1 = pick_one(side1_cands)
        if p1 is not None:
            out["side1"] = p1

    if "side2" not in out:
        side2_cands = [
            p
            for p in all_mp4
            if re.search(r"(side2|cam2_side)", p.stem, flags=re.IGNORECASE)
        ]
        p2 = pick_one(side2_cands)
        if p2 is not None:
            out["side2"] = p2

    missing = [v for v in ["side1", "side2"] if v not in out]
    if missing:
        raise FileNotFoundError(
            f"未能在 {root} 下推断出以下视角的视频: {missing}\n"
            f"top: {video_top}\n"
            f"请显式提供 --video-side1 / --video-side2。"
        )

    return out


# -----------------------------------------------------------------------------
# BORIS 事件解析
# -----------------------------------------------------------------------------


def _find_first_col(cols: List[str], *keywords: str) -> Optional[str]:
    """在列名中按关键字模糊查找，返回第一个匹配的原始列名。"""
    lower_map = {c.lower(): c for c in cols}
    for lc, orig in lower_map.items():
        if all(k in lc for k in keywords):
            return orig
    return None


def load_boris_events(csv_path: Path, fps: float | None = None) -> List[BehaviorEvent]:
    """
    加载 BORIS 事件 CSV，返回统一的 BehaviorEvent 列表。

    预期兼容几类常见导出格式：
    1）区间事件：Start (s), Stop (s) / Duration (s)
    2）时间点事件：Time (s)（例如从 0 开始，每行一个事件，未出现时间段视为背景）
    3）帧索引事件：Frame（配合 fps，把 frame / fps 当成秒）
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"BORIS 事件 CSV 不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    cols = list(df.columns)

    # 时间列（优先 Start，其次 Time，再次 Frame）
    start_col = (
        _find_first_col(cols, "start", "(s")
        or _find_first_col(cols, "start")
        or ("Start (s)" if "Start (s)" in df.columns else None)
    )
    stop_col = (
        _find_first_col(cols, "stop", "(s")
        or _find_first_col(cols, "stop")
        or None
    )
    duration_col = (
        _find_first_col(cols, "duration", "(s")
        or _find_first_col(cols, "duration")
        or None
    )

    time_col = None
    frame_col = None
    if start_col is None or start_col not in df.columns:
        # 没有 Start，用 Time (s) / Time 之类
        time_col = (
            _find_first_col(cols, "time", "(s")
            or _find_first_col(cols, "time")
        )
        if time_col is None or time_col not in df.columns:
            # 再退一步：用 Frame + fps
            frame_col = _find_first_col(cols, "frame")
            if frame_col is None or frame_col not in df.columns:
                raise ValueError(
                    "BORIS CSV 中既没有 Start (s) / Time (s) / Time，也没有 Frame 列，"
                    f"无法推断事件时间轴。当前列: {cols}"
                )

    # 行为列
    behavior_col = (
        _find_first_col(cols, "behavior")
        or _find_first_col(cols, "behav")
        or "Behavior"
        if "Behavior" in df.columns
        else None
    )
    if behavior_col is None or behavior_col not in df.columns:
        raise ValueError(
            f"BORIS CSV 中未找到行为列（如 'Behavior'）。当前列: {cols}"
        )

    # 个体列
    subject_col = (
        _find_first_col(cols, "subject")
        or _find_first_col(cols, "individual")
        or _find_first_col(cols, "id")
    )
    # subject_col 允许为 None（缺省个体）

    events: List[BehaviorEvent] = []
    for _, row in df.iterrows():
        # 统一计算 start_s
        start_s: Optional[float]
        if start_col and start_col in df.columns:
            try:
                start_s = float(row[start_col])
            except Exception:
                start_s = None
        elif time_col and time_col in df.columns:
            try:
                start_s = float(row[time_col])
            except Exception:
                start_s = None
        elif frame_col and frame_col in df.columns and fps and fps > 0:
            try:
                start_s = float(row[frame_col]) / float(fps)
            except Exception:
                start_s = None
        else:
            start_s = None

        if start_s is None:
            continue

        if np.isnan(start_s):
            continue

        end_s: Optional[float] = None
        if stop_col and stop_col in df.columns:
            try:
                end_s = float(row[stop_col])
            except Exception:
                end_s = None
        if (end_s is None or not np.isfinite(end_s)) and duration_col and duration_col in df.columns:
            try:
                dur = float(row[duration_col])
                if np.isfinite(dur):
                    end_s = start_s + dur
            except Exception:
                end_s = None

        if end_s is None or not np.isfinite(end_s):
            # 没有 Stop / Duration 时：
            # - 若有 fps，则按“一帧”的时长画成极窄短条；
            # - 否则用一个很短的默认时长（0.1s）。
            if fps and fps > 0:
                end_s = start_s + 1.0 / float(fps)
            else:
                end_s = start_s + 0.1

        behavior = str(row[behavior_col]) if not pd.isna(row[behavior_col]) else ""
        if subject_col and subject_col in df.columns:
            subject_val = row[subject_col]
            subject = str(subject_val) if not pd.isna(subject_val) else ""
        else:
            subject = ""

        events.append(
            BehaviorEvent(
                start_s=float(start_s),
                end_s=float(end_s),
                behavior=behavior,
                subject=subject,
            )
        )

    # 过滤非法与零长度
    clean_events = [
        e for e in events if np.isfinite(e.start_s) and np.isfinite(e.end_s) and e.end_s > e.start_s
    ]
    return clean_events


# -----------------------------------------------------------------------------
# 时间轴画布与光标
# -----------------------------------------------------------------------------


def _build_color_map(labels: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """为每个 label 分配一个 BGR 颜色。"""
    unique = sorted(set(labels))
    if not unique:
        return {}

    palette = [
        (0, 255, 0),      # 绿
        (0, 165, 255),    # 橙
        (255, 0, 0),      # 蓝
        (255, 0, 255),    # 品红
        (0, 255, 255),    # 黄
        (255, 255, 0),    # 青
        (128, 0, 255),    # 紫
        (0, 128, 255),    # 淡蓝
        (0, 255, 128),    # 黄绿
        (255, 128, 0),    # 橙红
    ]
    color_map: Dict[str, Tuple[int, int, int]] = {}
    for i, label in enumerate(unique):
        color_map[label] = palette[i % len(palette)]
    return color_map


def build_timeline_base_image(
    events: List[BehaviorEvent],
    total_duration_s: float,
    width: int,
    *,
    bar_height: int = 60,
    legend_height: int = 80,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, int]], Tuple[int, int, int, int]]:
    """
    构建不含光标的时间轴底图（长条 + 图例），返回：
    - base_img: ndarray[h, w, 3]
    - color_map: {label -> BGR}
    - bar_box: (x0, y0, x1, y1) 表示时间条区域
    """
    h = bar_height + legend_height + 20
    img = np.zeros((h, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # 深灰背景

    if total_duration_s <= 0 or not events:
        # 即便没有事件，也画一个空条
        x0, x1 = 40, width - 20
        y0, y1 = 20, 20 + bar_height
        cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1)
        return img, {}, (x0, y0, x1, y1)

    x0, x1 = 40, width - 20
    y0, y1 = 20, 20 + bar_height
    bar_width = max(1, x1 - x0)

    labels = [e.label for e in events]
    color_map = _build_color_map(labels)

    # 先画外框
    cv2.rectangle(img, (x0, y0), (x1, y1), (100, 100, 100), 1)

    # 再填充每个事件段
    for e in events:
        start_s = max(0.0, min(e.start_s, total_duration_s))
        end_s = max(0.0, min(e.end_s, total_duration_s))
        if end_s <= start_s:
            continue

        xs = x0 + int(round(start_s / total_duration_s * bar_width))
        xe = x0 + int(round(end_s / total_duration_s * bar_width))
        xe = max(xs + 1, min(xe, x1))

        color = color_map.get(e.label, (200, 200, 200))
        cv2.rectangle(img, (xs, y0 + 2), (xe, y1 - 2), color, thickness=-1)

    # 图例
    legend_y = y1 + 10
    line_height = 20
    x_color = 40
    x_text = 65
    for i, label in enumerate(sorted(color_map.keys())):
        yy = legend_y + i * line_height
        if yy + line_height >= h:
            break
        color = color_map[label]
        cv2.rectangle(img, (x_color, yy), (x_color + 18, yy + 18), color, thickness=-1)
        cv2.putText(
            img,
            label,
            (x_text, yy + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return img, color_map, (x0, y0, x1, y1)


def format_time_mm_ss_ff(frame_idx: int, fps: float) -> str:
    if fps <= 0:
        return "00:00:00"
    t = frame_idx / fps
    minutes = int(t // 60)
    seconds = int(t % 60)
    frac = t - int(t)
    frames = int(round(frac * fps))
    return f"{minutes:02d}:{seconds:02d}:{frames:02d}"


def draw_cursor_on_timeline(
    base_img: np.ndarray,
    frame_idx: int,
    fps: float,
    total_duration_s: float,
    bar_box: Tuple[int, int, int, int],
) -> np.ndarray:
    """在时间轴底图上叠加当前帧对应的光标和时间文本。"""
    x0, y0, x1, y1 = bar_box
    bar_width = max(1, x1 - x0)
    out = base_img.copy()

    if fps <= 0 or total_duration_s <= 0:
        return out

    t = frame_idx / fps
    t = max(0.0, min(t, total_duration_s))
    x = x0 + int(round(t / total_duration_s * bar_width))
    x = max(x0, min(x, x1))

    # 竖直光标
    cv2.line(out, (x, y0 - 2), (x, y1 + 2), (255, 255, 255), 2, cv2.LINE_AA)

    # 时间文本
    time_str = format_time_mm_ss_ff(frame_idx, fps)
    cv2.putText(
        out,
        time_str,
        (x0, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


# -----------------------------------------------------------------------------
# 主合成逻辑
# -----------------------------------------------------------------------------


def render_three_view_with_behavior_timeline(
    video_top: Path | str,
    boris_csv: Path | str,
    output_path: Path | str,
    *,
    video_side1: Path | str | None = None,
    video_side2: Path | str | None = None,
    max_frames: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> Path:
    """
    三视角原始视频 + BORIS 行为时间轴合成。

    参数
    ----
    video_top: top 视角 mp4 路径（必需）
    boris_csv: BORIS 事件 CSV 路径（必需）
    output_path: 输出 mp4 路径
    video_side1 / video_side2: 可选，若不提供则自动在与 top 同一 set 下推断。
    """
    if cv2 is None:
        raise SystemExit("需要 opencv-python：pip install opencv-python")

    video_top = Path(video_top)
    boris_csv = Path(boris_csv)
    output_path = Path(output_path)

    views = infer_three_view_set_from_top(video_top, video_side1, video_side2)

    # 打开三路视频
    order = ["top", "side1", "side2"]
    caps: Dict[str, "cv2.VideoCapture"] = {}
    for v in order:
        p = views.get(v)
        if p is None or not p.exists():
            raise FileNotFoundError(f"缺少视角视频 {v}: {p}")
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频 {v}: {p}")
        caps[v] = cap

    # 使用 top 的 fps 与总帧数
    cap_top = caps["top"]
    fps = float(cap_top.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 0:
        fps = 30.0

    # 总帧数（全长），用于时间轴长度
    total_frames = int(min(caps[v].get(cv2.CAP_PROP_FRAME_COUNT) for v in order))
    if total_frames <= 0:
        raise RuntimeError("视频帧数为 0，无法合成。")

    # 选择要写出的帧区间 [sf, ef]
    sf = 0 if start_frame is None else max(0, int(start_frame))
    ef = total_frames - 1 if end_frame is None else min(total_frames - 1, int(end_frame))
    if max_frames is not None and max_frames > 0:
        ef = min(ef, sf + int(max_frames) - 1)
    if ef < sf:
        raise RuntimeError(f"无效的帧区间: start_frame={start_frame}, end_frame={end_frame}, max_frames={max_frames}")

    total_duration_s = total_frames / fps

    # 计算每路尺寸（统一高度，保持宽高比）
    target_h = 360
    scales: List[Tuple[int, int]] = []  # (w, h)
    for v in order:
        cap = caps[v]
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or target_h
        scale = target_h / h
        new_w = int(w * scale)
        scales.append((new_w, target_h))

    out_w = sum(s[0] for s in scales)
    timeline_h = 160
    out_h = target_h + timeline_h

    # 读取 BORIS 事件，构建时间轴底图
    events = load_boris_events(boris_csv, fps=fps)
    base_timeline, _, bar_box = build_timeline_base_image(
        events,
        total_duration_s=total_duration_s,
        width=out_w,
        bar_height=60,
        legend_height=80,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover - 仅在无 tqdm 时触发
        tqdm = lambda x, **kw: x

    for idx in tqdm(range(sf, ef + 1), desc="写入行为可视化视频", unit="frame", ncols=80):
        # 上排三视角
        row_video: List[np.ndarray] = []
        for i, v in enumerate(order):
            cap = caps[v]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((target_h, scales[i][0], 3), dtype=np.uint8)
            elif len(frame.shape) == 2:  # 灰度转 BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame = cv2.resize(frame, (scales[i][0], scales[i][1]))
            cv2.putText(
                frame,
                v,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            row_video.append(frame)

        top_row = np.hstack(row_video)

        # 下排行为时间轴 + 光标
        timeline = draw_cursor_on_timeline(
            base_timeline,
            frame_idx=idx,
            fps=fps,
            total_duration_s=total_duration_s,
            bar_box=bar_box,
        )

        # 若时间轴高度不足，补齐至 timeline_h
        if timeline.shape[0] < timeline_h:
            pad_h = timeline_h - timeline.shape[0]
            pad = np.zeros((pad_h, out_w, 3), dtype=np.uint8)
            pad[:] = (30, 30, 30)
            timeline = np.vstack([timeline, pad])
        elif timeline.shape[0] > timeline_h:
            timeline = timeline[:timeline_h, :, :]

        combined = np.vstack([top_row, timeline])
        writer.write(combined)

    for cap in caps.values():
        cap.release()
    writer.release()

    print(f"已写出行为可视化视频: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "根据 top 视角视频自动推断三视角，并叠加 BORIS 行为时间轴，"
            "生成两行布局（上: top/side1/side2，下: 行为时间条 + 光标）的 mp4。"
        )
    )
    p.add_argument("--video-top", type=Path, required=True, help="top 视角视频路径，例如 D:/videos_raw/.../top_cam/cam0_top.mp4")
    p.add_argument("--boris-csv", type=Path, required=True, help="BORIS 事件 CSV，例如 D:/boris_events/20251225_K_1_2_FCMC.csv")
    p.add_argument("--output", type=Path, required=True, help="输出 mp4 路径")
    p.add_argument("--video-side1", type=Path, default=None, help="可选，side1 视角视频路径（若不提供则自动推断）")
    p.add_argument("--video-side2", type=Path, default=None, help="可选，side2 视角视频路径（若不提供则自动推断）")
    p.add_argument("--start-frame", type=int, default=None, help="仅导出从此帧开始的片段（包含该帧），默认从 0 开始")
    p.add_argument("--end-frame", type=int, default=None, help="仅导出到此帧结束的片段（包含该帧），默认到视频最后一帧")
    p.add_argument("--max-frames", type=int, default=None, help="最多写出多少帧，可与 start/end 组合使用")
    args = p.parse_args()

    render_three_view_with_behavior_timeline(
        video_top=args.video_top,
        boris_csv=args.boris_csv,
        output_path=args.output,
        video_side1=args.video_side1,
        video_side2=args.video_side2,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

