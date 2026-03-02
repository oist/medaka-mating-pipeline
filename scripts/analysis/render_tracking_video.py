"""
把「配好 ID + 补全后的轨迹」叠到 clip 视频上，生成带标注的合成视频
========================================================================

流程与 analyze_tracking_patterns 一致：先按 SLEAP CSV 做 ID 分配与帧级补全，
再把个体 0/1/2 的 (x,y) 画到对应视角的每一帧上，输出三视角并排的 mp4。

依赖：opencv-python, numpy, pandas

用法示例：

  # 指定实验 ID、SLEAP CSV 目录、视频目录、fps 与输出路径
  python scripts/analysis/render_tracking_video.py ^
    --input-dir data/sleap_output ^
    --experiment-id "20251220_E_1_2_FCMC__male1_Courtship_1464.337-1472.705" ^
    --video-dir "D:/clips" ^
    --fps 166 ^
    --output out_tracking_id.mp4

  # 也可分别指定三个视角的视频路径
  python scripts/analysis/render_tracking_video.py ^
    --input-dir data/sleap_output ^
    --experiment-id "20251220_E_1_2_FCMC__male1_Courtship_1464.337-1472.705" ^
    --video-top "D:/clips/top.mp4" ^
    --video-side1 "D:/clips/side1.mp4" ^
    --video-side2 "D:/clips/side2.mp4" ^
    --fps 166 ^
    --output out_tracking_id.mp4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

try:
    import cv2
except ImportError:
    cv2 = None


# -----------------------------------------------------------------------------
# 与 analyze_tracking_patterns 一致的辅助函数（逻辑保持同步）
# -----------------------------------------------------------------------------

def normalise_individual_to_int(val) -> int:
    """把 individual 列统一成整数：支持 0/1/2、'a'/'b'/'c'、'fish_0'/'fish_1'、'0'/'1' 等。"""
    if pd.isna(val):
        return -1
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return int(val)
    s = str(val).strip().lower()
    if s in ("a", "b", "c"):
        return ord(s) - ord("a")
    m = re.match(r"fish_(\d+)", s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return -1


def extract_clip_time_range(path: str | Path) -> Tuple[float, float] | None:
    """从文件名解析 clip 时间范围（秒），例如 ..._1464.337-1472.705... -> (1464.337, 1472.705)。"""
    s = str(path)
    m = re.search(r"(\d+\.\d+)-(\d+\.\d+)", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def load_tracking_data(csv_path: Path) -> pd.DataFrame:
    """加载 SLEAP 导出 CSV，需含 frame, x, y。"""
    df = pd.read_csv(csv_path)
    for c in ["frame", "x", "y"]:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少列: {c}")
    return df


def extract_centroid_per_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    按帧提取质心并分配临时个体 ID（0,1,2）。
    首帧按 score 取前 3 个；后续帧用上一帧位置做最近邻匹配。
    """
    frames = sorted(df["frame"].unique())
    required = ["frame", "x", "y"]
    for c in required:
        if c not in df.columns:
            return pd.DataFrame()

    out: List[dict] = []
    prev: Dict[int, np.ndarray] = {}

    for fi in frames:
        sub = df[df["frame"] == fi]
        if sub.empty:
            continue
        pts = sub[["x", "y"]].values.astype(float)
        if "score" in sub.columns:
            sub = sub.sort_values("score", ascending=False)

        if not prev:
            n = min(3, len(sub))
            for k in range(n):
                r = sub.iloc[k]
                prev[k] = np.array([float(r["x"]), float(r["y"])])
                out.append({"frame": fi, "individual": k, "x": prev[k][0], "y": prev[k][1]})
            continue

        prev_pts = np.array([prev[i] for i in sorted(prev)])
        dist = cdist(pts, prev_pts)
        used_j = set()
        for _ in range(min(len(pts), len(prev_pts))):
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            d = dist[i, j]
            if d > 100:
                break
            dist[i, :] = np.inf
            dist[:, j] = np.inf
            ind_id = sorted(prev.keys())[j]
            x, y = float(sub.iloc[i]["x"]), float(sub.iloc[i]["y"])
            prev[ind_id] = np.array([x, y])
            out.append({"frame": fi, "individual": ind_id, "x": x, "y": y})
            used_j.add(j)
    return pd.DataFrame(out)


def complete_tracks_with_interpolation(
    individuals_df: pd.DataFrame,
    fps: float,
    clip_range: Tuple[float, float],
) -> pd.DataFrame:
    """按 clip 时长补全帧，缺失位置线性插值；新增 is_interpolated 列。"""
    if individuals_df.empty or fps <= 0:
        return individuals_df
    t0, t1 = clip_range
    dur = max(t1 - t0, 0.0)
    n = int(round(dur * fps))
    if n <= 0:
        return individuals_df

    f0 = int(individuals_df["frame"].min())
    full_frames = np.arange(f0, f0 + n, dtype=int)
    parts = []
    for ind_id in sorted(individuals_df["individual"].unique()):
        sub = individuals_df[individuals_df["individual"] == ind_id].set_index("frame").reindex(full_frames)
        sub["individual"] = ind_id
        sub["is_interpolated"] = sub["x"].isna() | sub["y"].isna()
        sub["x"] = sub["x"].interpolate(method="linear", limit_direction="both")
        sub["y"] = sub["y"].interpolate(method="linear", limit_direction="both")
        parts.append(sub.reset_index())
    return pd.concat(parts, ignore_index=True)


def identify_view_from_path(csv_path: Path) -> str | None:
    """PLAYER1 -> top, PLAYER2 -> side1, PLAYER3 -> side2."""
    n = csv_path.name.upper()
    if "PLAYER1" in n:
        return "top"
    if "PLAYER2" in n:
        return "side1"
    if "PLAYER3" in n:
        return "side2"
    return None


def extract_experiment_id_from_filename(name: str) -> str:
    """
    去掉扩展名、视角/PLAYER 等后缀，得到可对比的实验 ID。
    支持两种命名：
    - top_mouth_180_..._PLAYER1_...csv / side_mouth_593_..._PLAYER2/3_...csv
    - 20251220_E_1_2_FCMC_male1_Courtship_2082.090-2090.072_player1_id_assigned.csv
    """
    s = name.replace(".csv", "").replace(".predictions.csv", "")
    s = re.sub(r"_player[123]_(?:id_assigned|processed)$", "", s, flags=re.I)
    s = re.sub(r"PLAYER[123]", "", s, flags=re.I)
    s = re.sub(r"^(top|side)_[^_]+_\d+_", "", s, flags=re.I)
    return s.strip("_")


def _normalize_exp_id(s: str) -> str:
    """实验 ID 标准化：连续下划线视为同一分隔，便于兼容 FCMC_male1 与 FCMC__male1。"""
    return re.sub(r"_+", "_", s).strip("_")


def find_view_csvs(input_dir: Path, experiment_id: str) -> Dict[str, Path]:
    """在 input_dir 下递归找 CSV，按实验 ID 和 PLAYER 匹配，返回 {top|side1|side2: path}。
    注意：这里只会使用 *_processed.csv 文件；若目标实验没有对应的 processed 文件，
    会报错提醒，而不会回退到 `_id_assigned` 或原始 SLEAP 导出。"""
    by_view: Dict[str, List[Path]] = {}
    seen_eids: set[str] = set()
    want_norm = _normalize_exp_id(experiment_id)
    for p in input_dir.rglob("*.csv"):
        view = identify_view_from_path(p)
        if view is None:
            continue
        eid = extract_experiment_id_from_filename(p.name)
        seen_eids.add(eid)
        # 仅匹配实验 ID 一致且文件名中包含 "processed" 的 CSV
        if (eid == experiment_id or _normalize_exp_id(eid) == want_norm) and "processed" in p.name.lower():
            by_view.setdefault(view, []).append(p)
    if by_view:
        # 多个 processed 文件时，按文件名排序后取最后一个（通常为最新版本）
        return {v: sorted(fs, key=lambda x: x.name)[-1] for v, fs in by_view.items() if fs}
    # 未匹配时给出提示
    sample = sorted(seen_eids)[:8]
    raise FileNotFoundError(
        f"未在 {input_dir} 下找到 processed CSV 对应实验 ID: {experiment_id!r}\n"
        f"提示：2D 轨迹视频现在强制只使用 *_processed.csv 文件，"
        f"请确认已通过 scripts.tracking.process_tracking 生成处理结果并放在该目录。\n"
        f"当前目录下检测到的实验 ID 示例（不一定都有 processed 文件，请仅作参考）：\n  " + "\n  ".join(sample)
    )


# -----------------------------------------------------------------------------
# 视频叠加与写出
# -----------------------------------------------------------------------------

def draw_one_frame(
    img: np.ndarray,
    individuals_df: pd.DataFrame,
    frame_idx: int,
    colors: List[Tuple[int, int, int]],
    simple_dots: bool = False,
) -> np.ndarray:
    """在 img 上按 individuals_df 中该帧的 (x,y) 画圆；simple_dots 时仅纯色圆、不画 ID 文字。"""
    out = img.copy()
    h, w = out.shape[:2]
    sub = individuals_df[individuals_df["frame"] == frame_idx].dropna(subset=["x", "y"])
    dot_color = (255, 255, 255) if simple_dots else None
    for _, row in sub.iterrows():
        k = int(row["individual"])
        x, y = int(round(float(row["x"]))), int(round(float(row["y"])))
        if not (0 <= x < w and 0 <= y < h):
            continue
        color = dot_color if simple_dots else colors[k % len(colors)]
        cv2.circle(out, (x, y), 8, color, 2)
        if not simple_dots:
            label = f"Ind {k}"
            cv2.putText(
                out, label, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
            )
    return out


def draw_trail_panel(
    individuals_df: pd.DataFrame,
    frame_idx: int,
    trail_frames: int,
    width: int,
    height: int,
    colors: List[Tuple[int, int, int]],
    ref_wh: Tuple[int, int] | None,
    simple_dots: bool = False,
) -> np.ndarray:
    """
    绘制纯轨迹面板：空白画布 + 过去 trail_frames 帧的轨迹线 + 当前点；simple_dots 时仅纯色、不画 ID。
    """
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[:] = (40, 40, 40)  # 深灰底
    if individuals_df.empty or trail_frames <= 0:
        return out
    ref_w, ref_h = (ref_wh if ref_wh else (width, height))
    scale_x = width / max(ref_w, 1)
    scale_y = height / max(ref_h, 1)
    line_color = (200, 200, 200) if simple_dots else None

    start_frame = max(0, frame_idx - trail_frames)
    for ind_id in sorted(individuals_df["individual"].unique()):
        seg = individuals_df[
            (individuals_df["individual"] == ind_id)
            & (individuals_df["frame"] >= start_frame)
            & (individuals_df["frame"] <= frame_idx)
        ].sort_values("frame")
        seg = seg.dropna(subset=["x", "y"])
        c = line_color if simple_dots else colors[ind_id % len(colors)]
        if len(seg) < 2:
            pt = seg[["x", "y"]].values
            if len(pt) == 1:
                x, y = int(round(pt[0, 0] * scale_x)), int(round(pt[0, 1] * scale_y))
                cv2.circle(out, (x, y), 6, c, 2)
                if not simple_dots:
                    cv2.putText(out, f"Ind {ind_id}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
            continue
        pts = seg[["x", "y"]].values
        pts_scaled = (pts * np.array([scale_x, scale_y])).astype(np.int32)
        pts_scaled = np.clip(pts_scaled, (0, 0), (width - 1, height - 1))
        cv2.polylines(out, [pts_scaled], False, c, 2, cv2.LINE_AA)
        x, y = int(pts_scaled[-1, 0]), int(pts_scaled[-1, 1])
        cv2.circle(out, (x, y), 6, c, 2)
        if not simple_dots:
            cv2.putText(out, f"Ind {ind_id}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    return out


def render_three_view_video(
    view_data: Dict[str, pd.DataFrame],
    video_paths: Dict[str, Path],
    output_path: Path,
    fps: float,
    trail_duration_s: float = 1.0,
    show_trail_row: bool = True,
    simple_dots: bool = False,
) -> None:
    """
    上方：三视角视频 + 点叠加；可选下方：三视角纯轨迹。
    simple_dots=True 时仅画纯色圆点、不画颜色区分与 Ind_0 等文字。
    """
    if cv2 is None:
        raise SystemExit("需要 opencv-python：pip install opencv-python")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    order = ["top", "side1", "side2"]
    caps = {}
    for v in order:
        if v not in video_paths or not video_paths[v].exists():
            raise FileNotFoundError(f"缺少视角视频: {v} -> {video_paths.get(v)}")
        cap = cv2.VideoCapture(str(video_paths[v]))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_paths[v]}")
        caps[v] = cap

    n_frames = int(min(caps[v].get(cv2.CAP_PROP_FRAME_COUNT) for v in order))
    target_h = 360
    scales = []
    ref_wh_per_view = []
    for v in order:
        w = int(caps[v].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(caps[v].get(cv2.CAP_PROP_FRAME_HEIGHT)) or target_h
        scale = target_h / h
        new_w = int(w * scale)
        scales.append((new_w, target_h))
        ref_wh_per_view.append((w, h))
    out_w = sum(s[0] for s in scales)
    out_h_row = target_h
    trail_frames = max(1, int(round(trail_duration_s * fps)))
    extra_margin_h = 50
    n_rows = 2 if show_trail_row else 1
    total_h = out_h_row * n_rows + extra_margin_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]  # BGR

    for idx in tqdm(range(n_frames), desc="写入视频", unit="frame", ncols=80):
        row_video = []
        for i, v in enumerate(order):
            cap = caps[v]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((out_h_row, scales[i][0], 3), dtype=np.uint8)
            elif len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if v in view_data and not view_data[v].empty:
                frame = draw_one_frame(frame, view_data[v], idx, colors, simple_dots=simple_dots)
            frame = cv2.resize(frame, (scales[i][0], scales[i][1]))
            cv2.putText(frame, v, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            row_video.append(frame)

        top_row = np.hstack(row_video)
        if show_trail_row:
            row_trail = []
            for i, v in enumerate(order):
                rw, rh = ref_wh_per_view[i]
                trail_img = draw_trail_panel(
                    view_data.get(v, pd.DataFrame()),
                    idx,
                    trail_frames,
                    scales[i][0],
                    scales[i][1],
                    colors,
                    (rw, rh),
                    simple_dots=simple_dots,
                )
                cv2.putText(trail_img, f"{v} (trail)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
                row_trail.append(trail_img)
            bottom_row = np.hstack(row_trail)
            body = np.vstack([top_row, bottom_row])
            frame_label_y = out_h_row * 2 + int(extra_margin_h * 0.7)
        else:
            body = top_row
            frame_label_y = out_h_row + int(extra_margin_h * 0.7)
        margin_band = np.zeros((extra_margin_h, out_w, 3), dtype=np.uint8)
        combined = np.vstack([body, margin_band])
        cv2.putText(
            combined, f"Frame: {idx}",
            (10, frame_label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(combined)

    for cap in caps.values():
        cap.release()
    writer.release()
    print(f"已写出: {output_path}")


def resolve_video_paths(
    view_csvs: Dict[str, Path],
    video_dir: Path | None,
    video_top: Path | None,
    video_side1: Path | None,
    video_side2: Path | None,
) -> Dict[str, Path]:
    """确定每个视角使用的视频路径。多种目录结构都尝试匹配。"""
    explicit = {"top": video_top, "side1": video_side1, "side2": video_side2}
    out: Dict[str, Path] = {}
    for v, csv_path in view_csvs.items():
        if explicit.get(v) is not None and Path(explicit[v]).exists():
            out[v] = Path(explicit[v])
            continue
        if video_dir is None or not video_dir.exists():
            raise FileNotFoundError(
                f"未找到视角 {v} 的视频。请指定 --video-dir 或 --video-{v}。"
            )
        stem = csv_path.stem
        parent_name = csv_path.parent.name
        candidates = [
            video_dir / (stem + ".mp4"),                          # 扁平方格：video_dir/同名.mp4
            video_dir / parent_name / (stem + ".mp4"),            # 与 CSV 同结构：video_dir/top_mouth_180/xxx.mp4
            video_dir / parent_name / (csv_path.name.replace(".csv", ".mp4")),
        ]
        for cand in candidates:
            if cand.exists():
                out[v] = cand
                break
        if v in out:
            continue
        # 递归按 stem 匹配（含子目录）
        for f in video_dir.rglob("*.mp4"):
            if stem in f.stem or f.stem in stem:
                out[v] = f
                break
        if v in out:
            continue
        raise FileNotFoundError(
            f"未找到视角 {v} 的视频。已尝试在 {video_dir} 下匹配 stem={stem!r}、目录 {parent_name}。"
            f"可指定 --video-{v} <路径> 或把该视角 mp4 放到上述目录中。"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="把配好 ID 且已处理后的 2D 轨迹叠到 clip 视频上，输出三视角合成 mp4（仅使用 *_processed.csv）",
    )
    ap.add_argument("--input-dir", type=Path, required=True, help="处理后 CSV 所在目录（会递归查找 *_processed.csv，仅使用这些文件）")
    ap.add_argument("--processed-dir", type=Path, default=None, help="可选。另一个 processed 目录；若指定且存在匹配文件，则优先生效于 --input-dir")
    ap.add_argument("--experiment-id", type=str, required=True, help="实验 ID，与 analyze_tracking_patterns 中一致")
    ap.add_argument("--video-dir", type=Path, default=None, help="视频根目录（下含与 CSV 同名的 mp4 或同结构子目录）")
    ap.add_argument("--video-top", type=Path, default=None, help="top 视角视频路径")
    ap.add_argument("--video-side1", type=Path, default=None, help="side1 视角视频路径")
    ap.add_argument("--video-side2", type=Path, default=None, help="side2 视角视频路径")
    ap.add_argument("--fps", type=float, required=True, help="视频帧率，用于补全帧数")
    ap.add_argument("--output", type=Path, default=None, help="输出 mp4 路径（默认在 --output-dir 下）")
    ap.add_argument("--output-dir", type=Path, default=Path("D:/sleap_output"), help="输出目录，默认 D:\\sleap_output")
    ap.add_argument("--trail-duration-s", type=float, default=1.0, help="下方纯轨迹保留时长（秒），默认 1")
    ap.add_argument("--no-trail-row", action="store_true", help="不显示下方纯轨迹行，仅保留上方三视角视频+点")
    ap.add_argument("--simple-dots", action="store_true", help="仅画纯色圆点，不画颜色区分与 Ind_0 等文字")
    args = ap.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"输入目录不存在: {input_dir}")

    try:
        view_csvs = find_view_csvs(input_dir, args.experiment_id)
    except FileNotFoundError as e:
        raise SystemExit(e) from e

    # 若指定了 --processed-dir，优先用该目录下的 *_processed.csv 覆盖同视角
    if args.processed_dir and args.processed_dir.exists():
        try:
            from_processed = find_view_csvs(args.processed_dir, args.experiment_id)
            for v, path in from_processed.items():
                view_csvs[v] = path
        except FileNotFoundError:
            pass

    video_paths = resolve_video_paths(
        view_csvs,
        args.video_dir,
        args.video_top,
        args.video_side1,
        args.video_side2,
    )

    view_data: Dict[str, pd.DataFrame] = {}
    for v, csv_path in view_csvs.items():
        df = load_tracking_data(csv_path)
        if "individual" in df.columns:
            ind = df[["frame", "individual", "x", "y"]].copy()
        elif "fish_id" in df.columns:
            ind = df[["frame", "fish_id", "x", "y"]].copy()
            ind = ind.rename(columns={"fish_id": "individual"})
        else:
            ind = extract_centroid_per_frame(df)
        # _processed 等可能用 "a"/"b"/"c" 或 "fish_0"/"fish_1" 等字符串，统一为整数
        ind["individual"] = ind["individual"].apply(normalise_individual_to_int)
        ind = ind[ind["individual"] >= 0]
        if ind.empty:
            print(f"  警告: {v} 未得到有效轨迹，跳过该视角")
            continue
        clip_range = extract_clip_time_range(csv_path)
        if clip_range and args.fps > 0:
            ind = complete_tracks_with_interpolation(ind, args.fps, clip_range)
        view_data[v] = ind
        n = csv_path.name.lower()
        src = "processed" if "processed" in n else ("id_assigned" if "id_assigned" in n else "raw")
        print(f"  {v}: {len(ind)} 行, {ind['frame'].nunique()} 帧 ({src})")

    if not view_data:
        raise SystemExit("没有任何视角得到有效轨迹，无法生成视频")

    vp = {v: video_paths[v] for v in view_data if v in video_paths}
    if len(vp) != len(view_data):
        missing = set(view_data) - set(vp)
        raise SystemExit(f"缺少视角视频: {missing}")

    out_dir = args.output_dir
    out_path = args.output if args.output is not None else (out_dir / "tracking_id_video.mp4")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_three_view_video(
        view_data, vp, out_path, args.fps,
        trail_duration_s=args.trail_duration_s,
        show_trail_row=not args.no_trail_row,
        simple_dots=args.simple_dots,
    )


if __name__ == "__main__":
    main()
