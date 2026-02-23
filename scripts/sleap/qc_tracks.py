from __future__ import annotations

"""
多视角 SLEAP 跟踪结果质量评估（缺失 vs 遮挡）
================================================

目标：
- 同时读取 3 个视角（例如 top / side1 / side2）的 .slp 跟踪结果
- 在逐帧尺度上区分：
  - true_missing: 三个视角都少于期望个体数（真数据缺失）
  - *_occlusion: 某个视角少于期望个体数，但至少一个其他视角是完整的（视角遮挡）

输出：
- multiview_missing_occlusion.csv
  - frame, top_count, side1_count, side2_count
  - top_is_missing, side1_is_missing, side2_is_missing
  - true_missing
  - top_occlusion, side1_occlusion, side2_occlusion
  - event_name, events

- multiview_missing_occlusion_overview.png
- multiview_summary.csv

用法示例：

批量模式（推荐，自动匹配 side 文件）：
    python scripts/sleap/qc_tracks.py ^
      --top-slp D:\sleap_output\trial1_top.slp ^
      --top-slp D:\sleap_output\trial2_top.slp ^
      --expected-count 3 ^
      --output-dir D:\sleap_qc_multiview

单文件模式（向后兼容）：
    python scripts/sleap/qc_tracks.py ^
      --top D:\sleap_output\trial_top.slp ^
      --side1 D:\sleap_output\trial_side1.slp ^
      --side2 D:\sleap_output\trial_side2.slp ^
      --expected-count 3 ^
      --output-dir D:\sleap_qc_multiview
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import sleap  # type: ignore[import]
except ImportError as e:  # noqa: BLE001
    raise SystemExit(
        "无法导入 sleap 包。请先在当前环境中安装 SLEAP，\n"
        "例如：pip install 'sleap[nn]' 或在 sleap 环境下运行此脚本。"
    ) from e


def load_slp_data(slp_path: Path) -> pd.DataFrame:
    """从 .slp 文件加载实例级数据，并转换为 DataFrame。"""
    print(f"Loading SLEAP file: {slp_path}")

    labels = None

    # 方法1: sleap.io.dataset.Labels.load_file (SLEAP 1.5+)
    try:
        from sleap.io.dataset import Labels  # type: ignore[import]

        labels = Labels.load_file(str(slp_path))  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass

    # 方法2: sleap.load_file (旧版本)
    if labels is None:
        try:
            labels = sleap.load_file(str(slp_path))  # type: ignore[attr-defined]
        except AttributeError:
            pass

    # 方法3: sleap.Labels.load_file (另一种导入方式)
    if labels is None:
        try:
            labels = sleap.Labels.load_file(str(slp_path))  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

    # 方法4: 使用 sleap_io (如果安装了)
    if labels is None:
        try:
            import sleap_io as sio  # type: ignore[import]

            labels = sio.load_slp(str(slp_path))  # type: ignore[attr-defined]
        except ImportError:
            pass

    if labels is None:
        raise SystemExit(
            "Failed to load SLEAP file. Please check:\n"
            "1. SLEAP is correctly installed\n"
            "2. File path is correct\n"
            "3. File format is valid"
        )

    rows: list[dict] = []

    # 遍历所有带实例的帧
    for lf in labels.labeled_frames:  # type: ignore[attr-defined]
        frame_idx = lf.frame_idx

        for inst in lf.instances:
            # 获取 track 名称
            track_name = None
            if getattr(inst, "track", None) is not None:
                track_name = getattr(inst.track, "name", None) or getattr(
                    inst.track, "track_id", None
                )
            if track_name is None:
                track_name = f"untracked_{id(inst)}"

            # 收集该实例的所有关键点（这里只需要计数即可）
            node_count = 0
            points_data = inst.points

            # 尝试多种方式统计节点数
            if isinstance(points_data, dict) or (
                hasattr(points_data, "items")
                and callable(getattr(points_data, "items"))
            ):
                try:
                    node_count = len(points_data)
                except Exception:  # noqa: BLE001
                    node_count = 0
            else:
                try:
                    import numpy as _np

                    if hasattr(inst, "numpy"):
                        arr = inst.numpy()  # type: ignore[attr-defined]
                    else:
                        arr = _np.array(points_data)
                    if hasattr(arr, "shape") and arr.size > 0:
                        if arr.ndim >= 2:
                            node_count = arr.shape[0]
                        else:
                            node_count = 1
                except Exception:  # noqa: BLE001
                    node_count = 0

            rows.append(
                dict(
                    frame=int(frame_idx),
                    track=str(track_name),
                    node_count=int(node_count),
                )
            )

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        raise SystemExit("Error: No tracking data found in .slp file.")

    return df


def summarize_instance_counts(df: pd.DataFrame, expected_count: int) -> pd.DataFrame:
    """把单视角的 df 汇总成按帧统计的表：frame, instance_count, is_missing。"""
    frame_counts_raw = df.groupby("frame")["track"].nunique()
    frame_min = int(df["frame"].min())
    frame_max = int(df["frame"].max())
    full_index = range(frame_min, frame_max + 1)
    frame_counts = frame_counts_raw.reindex(full_index, fill_value=0)

    out = pd.DataFrame(
        {
            "frame": list(full_index),
            "instance_count": frame_counts.values.astype(int),
        }
    )
    out["missing_degree"] = np.clip(expected_count - out["instance_count"], 0, None)
    out["is_missing"] = out["missing_degree"] > 0
    return out


def _normalize_event_label(label: str) -> str:
    """把事件名称转成适合作为列名的形式（只包含字母/数字/下划线）。"""
    label = str(label)
    # 把空白和连字符替换为下划线
    label = re.sub(r"\s+|-+", "_", label)
    # 其他非常规字符也替换成下划线
    label = re.sub(r"[^\w]", "_", label)
    # 避免空字符串
    return label or "event"


def load_events_csv(
    events_path: Path,
    time_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """加载行为事件CSV，返回标准化后的区间表：start_s, end_s, label。

    支持三种常见格式：
    1）区间事件：起止时间列（BORIS 默认：Start (s), Stop (s)）
       - 使用 `start_col`, `end_col` 指定列名（默认尝试常见名字）
    2）点事件：单一时间列（例如某些导出只有一个 Time 列）
       - 使用 `time_col` 指定列名（默认尝试常见名字）
       - start_s = end_s = 该时间点
    3）BORIS START/STOP 事件：
       - 只有一个时间列（如 Time），再配合一个“类型”列（值为 START/STOP/POINT）
       - 脚本会自动把同一行为（同一 label）的 START/STOP 配对成区间 [start_s, end_s]
       - POINT 事件会作为点事件处理（start_s = end_s）
    """
    df = pd.read_csv(events_path)

    # 先确定 label 所在的列
    label_source_col: str | None = None
    if label_col is not None and label_col in df.columns:
        label_source_col = label_col
    else:
        for c in ["Behavior", "behavior", "event", "label"]:
            if c in df.columns:
                label_source_col = c
                break

    # 优先尝试显式的区间事件格式（Start/Stop 两列）
    candidate_start = [c for c in [start_col, "Start (s)", "Start(s)", "start_s", "start", "Start"] if c is not None]
    candidate_end = [c for c in [end_col, "Stop (s)", "Stop(s)", "end_s", "end", "Stop"] if c is not None]

    start_used = next((c for c in candidate_start if c in df.columns), None)
    end_used = next((c for c in candidate_end if c in df.columns), None)

    if start_used is not None and end_used is not None:
        events = pd.DataFrame(
            {
                "start_s": df[start_used].astype(float),
                "end_s": df[end_used].astype(float),
            }
        )
        # 显式区间格式下，直接从原表复制 label
        if label_source_col is not None:
            events["label"] = df[label_source_col].astype(str).values
        else:
            events["label"] = "event"
    else:
        # 没有显式的 Start/Stop 列：尝试 Time + START/STOP/POINT 这种格式
        candidate_time = [
            c
            for c in [
                time_col,
                "Time (s)",
                "Time(s)",
                "Time",
                "time_s",
                "time",
                "Image index",
                "Image index (frames)",
                "Frame",
            ]
            if c is not None
        ]
        time_used = next((c for c in candidate_time if c in df.columns), None)
        if time_used is None:
            raise SystemExit(
                "无法在行为事件 CSV 中找到时间列。\n"
                "请通过 --events-time-col 或 --events-start-col/--events-end-col "
                "指定与 BORIS 导出格式对应的列名。"
            )

        # 自动检测“事件类型”列（值为 START/STOP/POINT）
        type_col: str | None = None
        for c in df.columns:
            vals = df[c].astype(str).str.upper().str.strip()
            if vals.isin({"START", "STOP", "POINT"}).mean() > 0.5:
                type_col = c
                break

        rows: list[dict] = []

        if type_col is not None:
            # BORIS START/STOP 形式：按 label 分组，内部按时间排序配对
            if label_source_col is not None and label_source_col in df.columns:
                groups = df.groupby(label_source_col, sort=False)
            else:
                # 没有 label 列，就把所有事件当成同一类别
                groups = [(None, df)]

            for label_value, g in groups:
                g_sorted = g.sort_values(time_used)
                current_start: float | None = None
                for _, row in g_sorted.iterrows():
                    t = float(row[time_used])
                    etype = str(row[type_col]).upper().strip()
                    label_str = (
                        str(label_value)
                        if label_value is not None
                        else "event"
                    )
                    if etype == "START":
                        current_start = t
                    elif etype == "STOP" and current_start is not None:
                        rows.append(
                            {
                                "start_s": current_start,
                                "end_s": t,
                                "label": label_str,
                            }
                        )
                        current_start = None
                    elif etype == "POINT":
                        rows.append(
                            {
                                "start_s": t,
                                "end_s": t,
                                "label": label_str,
                            }
                        )
                # 如果最后还有未闭合的 START，可以选择忽略或当点事件；
                # 这里选择忽略，避免产生过长区间。
        else:
            # 只有时间列，没有 START/STOP：退化为点事件
            label_values: pd.Series | None = None
            if label_source_col is not None and label_source_col in df.columns:
                label_values = df[label_source_col].astype(str)

            for _, row in df.iterrows():
                t = float(row[time_used])
                label_str = (
                    str(row[label_source_col])
                    if label_source_col is not None and label_source_col in row
                    else "event"
                )
                rows.append({"start_s": t, "end_s": t, "label": label_str})

        events = pd.DataFrame(rows)
        if events.empty:
            raise SystemExit(
                "在行为事件 CSV 中未能构造任何有效事件。\n"
                "请检查是否包含 Time + START/STOP/POINT 或 Start/Stop 列。"
            )

    # 去除无效行
    events = events.dropna(subset=["start_s", "end_s"])
    events = events[events["end_s"] >= events["start_s"]]
    events = events.sort_values("start_s").reset_index(drop=True)
    return events


def extract_clip_start_time(filename: str | Path) -> float | None:
    """
    从文件名中提取视频片段的起始时间（秒）。
    
    支持的文件名格式：
    - ..._496.253-500.434_top.slp 或 ..._496.253-500.434.mp4
    - 提取第一个时间戳（496.253）作为起始时间
    
    返回：
    - 如果找到时间戳，返回起始时间（秒）
    - 否则返回 None
    """
    filename_str = str(filename)
    
    # 尝试匹配格式：..._start-end_... 或 ..._start-end.slp
    # 匹配模式：数字.数字-数字.数字（例如 496.253-500.434）
    pattern = r'(\d+\.\d+)-(\d+\.\d+)'
    match = re.search(pattern, filename_str)
    
    if match:
        start_time = float(match.group(1))
        return start_time
    
    return None


def find_side_slp_files(top_slp_path: Path) -> tuple[Path | None, Path | None]:
    """
    根据 top .slp 文件路径查找对应的 side .slp 文件（side1 和 side2）。
    
    命名规则（按优先级）：
    1. 如果 top 文件名同时包含 `PLAYER1` 和 `_top`，则：
       - 先替换 `PLAYER1` 为 `PLAYER2`/`PLAYER3`
       - 再替换 `_top` 为 `_side1`/`_side2`
    2. 如果 top 文件名只包含 `_top.slp`，则直接替换为 `_side1.slp` 和 `_side2.slp`
    3. 如果 top 文件名只包含 `PLAYER1`，则替换为 `PLAYER2` 和 `PLAYER3`
    4. 在同目录下查找可能的 side 文件
    
    返回：
    - (side1_path, side2_path) 如果找到，否则为 (None, None)
    """
    top_str = str(top_slp_path)
    top_path = Path(top_slp_path)
    
    # 方法1: 如果同时包含 PLAYER1 和 _top，先替换 PLAYER1，再替换 _top
    if "PLAYER1" in top_str and "_top" in top_str:
        side1_str = top_str.replace("PLAYER1", "PLAYER2").replace("_top", "_side1")
        side2_str = top_str.replace("PLAYER1", "PLAYER3").replace("_top", "_side2")
        side1_path = Path(side1_str)
        side2_path = Path(side2_str)
        
        if side1_path.exists() and side2_path.exists():
            return side1_path, side2_path
    
    # 方法2: 如果只包含 _top.slp，直接替换为 _side1.slp 和 _side2.slp
    if "_top.slp" in top_str:
        side1_str = top_str.replace("_top.slp", "_side1.slp")
        side2_str = top_str.replace("_top.slp", "_side2.slp")
        side1_path = Path(side1_str)
        side2_path = Path(side2_str)
        
        if side1_path.exists() and side2_path.exists():
            return side1_path, side2_path
    
    # 方法3: 如果只包含 PLAYER1，替换为 PLAYER2 和 PLAYER3
    if "PLAYER1" in top_str:
        side1_str = top_str.replace("PLAYER1", "PLAYER2")
        side2_str = top_str.replace("PLAYER1", "PLAYER3")
        
        side1_path = Path(side1_str)
        side2_path = Path(side2_str)
        
        side1_exists = side1_path.exists()
        side2_exists = side2_path.exists()
        
        if side1_exists and side2_exists:
            return side1_path, side2_path
        elif side1_exists or side2_exists:
            # 至少找到一个，返回找到的
            return (
                side1_path if side1_exists else None,
                side2_path if side2_exists else None,
            )
    
    # 方法4: 在同目录下查找可能的 side 文件
    # 尝试查找包含相同基础名称但带有 side1/side2 的文件
    parent_dir = top_path.parent
    top_stem = top_path.stem
    
    # 移除可能的 _top 后缀
    base_stem = top_stem.replace("_top", "")
    
    # 尝试多种可能的命名模式
    candidates = [
        (parent_dir / f"{base_stem}_side1.slp", parent_dir / f"{base_stem}_side2.slp"),
        (parent_dir / f"{top_stem}_side1.slp", parent_dir / f"{top_stem}_side2.slp"),
    ]
    
    for side1_candidate, side2_candidate in candidates:
        if side1_candidate.exists() and side2_candidate.exists():
            return side1_candidate, side2_candidate
    
    return None, None


def classify_multiview_missing(
    slp_top: Path,
    slp_side1: Path,
    slp_side2: Path,
    expected_count: int = 3,
) -> pd.DataFrame:
    """
    多视角缺失/遮挡分类逻辑：

    对每一帧：
      - 如果 top/side1/side2 三个视角的 instance_count 都 < expected_count
        → 这帧是"真缺失"（true_missing=True）
      - 如果某个视角 < expected_count，但至少有一个其他视角 >= expected_count
        → 该视角视为"遮挡"（*_occlusion=True）

    返回一个按帧的 DataFrame，其中每行对应一帧。
    """
    # 逐视角加载
    df_top = load_slp_data(slp_top)
    df_a = load_slp_data(slp_side1)
    df_b = load_slp_data(slp_side2)

    # 汇总成按帧统计
    fr_top = summarize_instance_counts(df_top, expected_count)
    fr_a = summarize_instance_counts(df_a, expected_count)
    fr_b = summarize_instance_counts(df_b, expected_count)

    # 对齐帧号（取三者共同覆盖的帧区间）
    frame_min = max(fr_top["frame"].min(), fr_a["frame"].min(), fr_b["frame"].min())
    frame_max = min(fr_top["frame"].max(), fr_a["frame"].max(), fr_b["frame"].max())

    def _clip_to_range(fr: pd.DataFrame) -> pd.DataFrame:
        return fr[(fr["frame"] >= frame_min) & (fr["frame"] <= frame_max)]

    fr_top = _clip_to_range(fr_top).set_index("frame")
    fr_a = _clip_to_range(fr_a).set_index("frame")
    fr_b = _clip_to_range(fr_b).set_index("frame")

    # 合并成一个表
    merged = pd.DataFrame(index=range(frame_min, frame_max + 1))
    merged["top_count"] = fr_top["instance_count"]
    merged["side1_count"] = fr_a["instance_count"]
    merged["side2_count"] = fr_b["instance_count"]

    # 视角级缺失标记：instance_count < expected_count
    merged["top_is_missing"] = merged["top_count"] < expected_count
    merged["side1_is_missing"] = merged["side1_count"] < expected_count
    merged["side2_is_missing"] = merged["side2_count"] < expected_count

    # 帧级"真缺失"：三个视角都缺
    merged["true_missing"] = (
        merged["top_is_missing"]
        & merged["side1_is_missing"]
        & merged["side2_is_missing"]
    )

    # 视角级"遮挡"：该视角缺，但至少一个其他视角是满的
    def view_occlusion(this_missing, other1_count, other2_count):
        other_ok = (other1_count >= expected_count) | (other2_count >= expected_count)
        return this_missing & other_ok

    merged["top_occlusion"] = view_occlusion(
        merged["top_is_missing"], merged["side1_count"], merged["side2_count"]
    )
    merged["side1_occlusion"] = view_occlusion(
        merged["side1_is_missing"], merged["top_count"], merged["side2_count"]
    )
    merged["side2_occlusion"] = view_occlusion(
        merged["side2_is_missing"], merged["top_count"], merged["side1_count"]
    )

    # 为了方便后续分析，把索引还原成列
    merged = merged.reset_index().rename(columns={"index": "frame"})
    return merged


def generate_multiview_visualizations(
    df: pd.DataFrame,
    expected_count: int,
    output_dir: Path,
    fps: float | None = None,
    events: pd.DataFrame | None = None,
    events_margin_s: float = 0.0,
    output_filename: str | None = None,
    time_offset_s: float = 0.0,
    title: str | None = None,
) -> None:
    """为三视角 QC 生成可视化图像（单张热图）。

    - 图：三视角缺失/遮挡分类的时间轴热图
    如果提供了 events（BORIS 行为事件），会在图中用深红/浅红高亮：
    - 深红 = 核心事件区间 [start_s, end_s]
    - 浅红 = 缓冲区 [start_s-margin, start_s) 和 (end_s, end_s+margin]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统一时间轴
    frames = df["frame"].to_numpy()
    true_missing = df["true_missing"].to_numpy()

    # -----------------------------
    # 缺失 / 遮挡 分类的时间轴可视化
    # -----------------------------
    # 为每一帧、每一视角分配一个类别编码：
    # 0: 灰色 = 缺失/无数据
    # 1: 橙色 = 只在第四行（all_missing），表示三个视角都缺失
    # 2: 蓝色 = 有数据/正常
    #
    # 主图中包含 4 行：
    #   top, side1, side2, all_missing
    # 前三行：蓝色 = 该视角有数据，灰色 = 该视角缺失
    # 第四行：橙色 = 三个视角都缺失，灰色 = 不是全部缺失
    views = ["top", "side1", "side2", "all_missing"]

    # 构造 (n_views, n_frames) 的矩阵
    n_frames = len(df)
    n_views = len(views)
    category = np.zeros((n_views, n_frames), dtype=int)

    for v_idx, view in enumerate(views):
        if view == "all_missing":
            # 第四行颜色逻辑：
            # 橙色 = 全缺失（三个视角都缺三个，即每个视角 count == 0）
            # 浅蓝 = 非全缺失（只要有缺失就算，无论缺失一个还是两个还是三个）
            # 蓝色 = 三个视角都是三个（即每个视角 count >= expected_count）
            top_count = df["top_count"].to_numpy()
            side1_count = df["side1_count"].to_numpy()
            side2_count = df["side2_count"].to_numpy()
            
            # 橙色：三个视角都完全缺失（每个视角 count == 0）
            all_fully_missing = (top_count == 0) & (side1_count == 0) & (side2_count == 0)
            
            # 蓝色：三个视角都有完整数据（每个视角 count >= expected_count）
            all_have_data = (top_count >= expected_count) & (side1_count >= expected_count) & (side2_count >= expected_count)
            
            # 浅蓝：非全缺失（至少有一个视角缺失，但不是全缺失）
            has_missing = (top_count < expected_count) | (side1_count < expected_count) | (side2_count < expected_count)
            partial_missing = has_missing & (~all_fully_missing) & (~all_have_data)
            
            cat = np.zeros(n_frames, dtype=int)
            cat[all_fully_missing] = 1  # 橙色
            cat[all_have_data] = 4  # 绿色（三个视角都有完整数据）
            cat[partial_missing] = 3  # 浅蓝
        else:
            # 前三行：蓝色表示有数据，不同深浅灰色表示缺失程度
            count_col = f"{view}_count"
            counts = df[count_col].to_numpy()
            has_data = counts >= expected_count
            missing_count = expected_count - counts
            missing_count = np.clip(missing_count, 0, expected_count)
            
            cat = np.zeros(n_frames, dtype=int)
            cat[has_data] = 2  # 蓝色 = 有数据
            # 缺失程度：用不同深浅灰色表示
            # 0 = 正常（蓝色，已设置）
            # 1 = 缺失1个个体（浅灰）
            # 2 = 缺失2个个体（中灰）
            # 3 = 缺失3个或更多个体（深灰）
            # 但为了简化，我们用 0 表示所有缺失，然后用不同的灰度值来区分
            # 这里先用 0 标记所有缺失，后面在颜色映射时用不同灰度
            
        category[v_idx, :] = cat

    # 颜色映射
    # 0: 白色（不应该出现）
    # 1: 橙色 = 全缺失（第四行：三个视角都完全缺失）
    # 2: 蓝色 = 有数据/正常（前三行：单个视角有完整数据）
    # 3: 浅蓝 = 非全缺失（第四行：有缺失但不是全缺失）
    # 4: 绿色 = 所有视角都有数据（第四行：三个视角都有完整数据）
    cmap = {
        0: "#FFFFFF",  # 白色（不应该出现）
        1: "#F39C12",  # 橙色
        2: "#1F77B4",  # 蓝色
        3: "#87CEEB",  # 浅蓝（SkyBlue）
        4: "#2E7D32",  # 绿色（深绿）
    }
    
    # 为前三行创建缺失程度的灰度映射
    # 缺失1个：浅灰 #E8E8E8
    # 缺失2个：中灰 #B0B0B0
    # 缺失3个或更多：深灰 #707070
    missing_gray_map = {
        1: "#E8E8E8",  # 浅灰
        2: "#B0B0B0",  # 中灰
        3: "#707070",  # 深灰（3个或更多）
    }

    # 构造 RGB 图像（4 行：top/side1/side2/all_missing）
    color_array = np.ones((n_views, n_frames, 3), dtype=float)  # 初始化为白色
    
    # 先处理前三行的缺失程度（用不同深浅灰色）
    for v_idx, view in enumerate(views):
        if view != "all_missing":
            count_col = f"{view}_count"
            counts = df[count_col].to_numpy()
            missing_count = np.clip(expected_count - counts, 0, expected_count)
            has_data = counts >= expected_count
            
            for c in range(3):
                # 有数据的帧：蓝色
                color_array[v_idx, :, c][has_data] = int(cmap[2][1 + c * 2:3 + c * 2], 16) / 255.0
                
                # 缺失的帧：根据缺失程度用不同灰度
                for miss_level in [1, 2, 3]:
                    if miss_level == 3:
                        # 缺失3个或更多，都用最深灰
                        mask_miss = (missing_count >= 3) & (~has_data)
                    else:
                        mask_miss = (missing_count == miss_level) & (~has_data)
                    if mask_miss.any():
                        gray_hex = missing_gray_map[miss_level]
                        color_array[v_idx, :, c][mask_miss] = int(gray_hex[1 + c * 2:3 + c * 2], 16) / 255.0
    
    # 处理第四行（all_missing）：橙色=全缺失，浅蓝=非全缺失，蓝色=都有数据
    all_missing_idx = views.index("all_missing")
    for code, hex_color in cmap.items():
        if code == 0:
            continue  # 白色不应该出现
        rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        mask = category[all_missing_idx, :] == code
        for c in range(3):
            color_array[all_missing_idx, :, c][mask] = rgb[c]

    # ---------- 画布与子图布局 ----------
    # 主图：top/side1/side2/all_missing 的缺失/遮挡状态
    # 事件图：每个行为单独一行，不覆盖主图
    n_event_labels = 0
    event_labels: list[str] = []

    # 只有在事件与当前片段时间轴存在重叠时，才保留该行为用于绘图
    if (
        events is not None
        and len(events) > 0
        and fps is not None
        and fps > 0
    ):
        frame0_tmp = float(frames.min())
        times_tmp = (frames - frame0_tmp) / float(fps) + time_offset_s
        t_min = float(times_tmp.min())
        t_max = float(times_tmp.max())

        labels_all = sorted(str(x) for x in events.get("label", ["event"]).unique())
        kept_labels: list[str] = []
        for lb in labels_all:
            g = events[events["label"].astype(str) == lb]
            has_overlap = False
            for _, row in g.iterrows():
                start_true = float(row["start_s"])
                end_true = float(row["end_s"])
                margin = float(events_margin_s)
                start_ext = start_true - margin
                end_ext = end_true + margin
                # 区间是否与片段时间 [t_min, t_max] 相交
                if (end_ext >= t_min) and (start_ext <= t_max):
                    has_overlap = True
                    break
            if has_overlap:
                kept_labels.append(lb)

        event_labels = kept_labels
        n_event_labels = len(event_labels)

    if n_event_labels > 0:
        fig, (ax_state, ax_evt) = plt.subplots(
            2,
            1,
            figsize=(18, 5.0),
            sharex=False,  # 不共享X轴，因为前四行用frame，事件图用时间
            gridspec_kw={"height_ratios": [4, max(1, n_event_labels)]},
        )
    else:
        fig, ax_state = plt.subplots(1, 1, figsize=(18, 3.5))
        ax_evt = None

    im = ax_state.imshow(color_array, aspect="auto", interpolation="nearest")
    ax_state.set_yticks(list(range(n_views)))
    ax_state.set_yticklabels(views, fontsize=10)

    # 前四行（状态图）的 X 轴：直接用 DataFrame 中的 frame 列的值（原始帧号）
    # frames 是 df["frame"] 的 numpy 数组，直接使用这些值
    idx_ticks = np.linspace(0, n_frames - 1, num=min(6, n_frames), dtype=int)
    ax_state.set_xticks(idx_ticks)
    # 直接使用 frames 数组中对应索引的值（这是CSV文件中的frame列的值）
    frame_labels = [f"{int(frames[i])}" for i in idx_ticks]
    ax_state.set_xticklabels(frame_labels, fontsize=9)
    ax_state.set_xlabel("Frame", fontsize=12, fontweight="bold")
    # 确保 X 轴刻度值可见，不被遮挡
    ax_state.tick_params(axis='x', labelsize=9, labelrotation=0, bottom=True, labelbottom=True)
    plt.setp(ax_state.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # 设置标题
    if title is not None:
        ax_state.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax_state.set_title(
            "Per-view Missing / Occlusion Classification", fontsize=14, fontweight="bold"
        )

    # 如有事件，则在下方单独的事件轴上叠加事件区间，每个行为单独一行，避免互相遮挡
    if (
        ax_evt is not None
        and fps is not None
        and fps > 0
        and n_event_labels > 0
    ):
        frame0 = float(frames.min())
        times = (frames - frame0) / float(fps) + time_offset_s
        margin = float(events_margin_s)
        
        # 计算视频片段的时间范围（不因margin而扩大）
        clip_start_time = float(times.min())  # 视频片段开始时间
        clip_end_time = float(times.max())    # 视频片段结束时间
        
        # 调试信息：打印 margin 值
        if margin > 0:
            print(f"  Event margin: {margin:.2f} seconds")
            print(f"  Clip time range: {clip_start_time:.2f} - {clip_end_time:.2f} seconds")

        n_evt_rows = len(event_labels)
        for row_idx, label in enumerate(event_labels):
            y0 = row_idx / max(1, n_evt_rows)
            y1 = (row_idx + 1) / max(1, n_evt_rows)

            if label == "__ALL_MISSING__":
                # 使用 true_missing 生成连续区间
                mask_all = df["true_missing"].to_numpy()
                if not mask_all.any():
                    continue
                idx_all = np.where(mask_all)[0]
                # 把连续索引段分成多个区间
                splits = np.split(
                    idx_all,
                    np.where(np.diff(idx_all) != 1)[0] + 1,
                )
                for seg in splits:
                    if len(seg) == 0:
                        continue
                    start_idx = seg[0]
                    end_idx = seg[-1]
                    start_true = times[start_idx]
                    end_true = times[end_idx]
                    start_ext = start_true - margin
                    end_ext = end_true + margin
                    
                    # 裁剪到视频片段的时间范围内
                    start_true_clipped = max(start_true, clip_start_time)
                    end_true_clipped = min(end_true, clip_end_time)
                    start_ext_clipped = max(start_ext, clip_start_time)
                    end_ext_clipped = min(end_ext, clip_end_time)

                    mask_ext = (times >= start_ext_clipped) & (times <= end_ext_clipped)
                    mask_true = (times >= start_true_clipped) & (times <= end_true_clipped)

                    if mask_ext.any():
                        idx_ext = np.where(mask_ext)[0]
                        x0_ext = idx_ext[0] - 0.5
                        x1_ext = idx_ext[-1] + 0.5
                        ax_evt.axvspan(
                            x0_ext,
                            x1_ext,
                            color="#B0BEC5",
                            alpha=0.4,
                            ymin=y0,
                            ymax=y1,
                        )
                    if mask_true.any():
                        idx_true = np.where(mask_true)[0]
                        x0_true = idx_true[0] - 0.5
                        x1_true = idx_true[-1] + 0.5
                        ax_evt.axvspan(
                            x0_true,
                            x1_true,
                            color="#455A64",
                            alpha=0.8,
                            ymin=y0,
                            ymax=y1,
                        )
            else:
                if events is None or len(events) == 0:
                    continue
                g = events[events["label"].astype(str) == label]
                if g.empty:
                    continue
                
                # 先收集所有该label的事件区间，然后合并重叠的区间，避免叠加导致颜色变浅
                event_intervals = []
                for _, row in g.iterrows():
                    start_true = float(row["start_s"])
                    end_true = float(row["end_s"])
                    start_ext = start_true - margin
                    end_ext = end_true + margin
                    
                    # 将事件时间裁剪到视频片段的时间范围内
                    start_true_clipped = max(start_true, clip_start_time)
                    end_true_clipped = min(end_true, clip_end_time)
                    start_ext_clipped = max(start_ext, clip_start_time)
                    end_ext_clipped = min(end_ext, clip_end_time)
                    
                    if start_true_clipped <= end_true_clipped:
                        event_intervals.append({
                            'core': (start_true_clipped, end_true_clipped),
                            'margin': (start_ext_clipped, end_ext_clipped)
                        })
                
                # 合并重叠的区间
                if event_intervals:
                    # 合并core区间
                    core_intervals = sorted([iv['core'] for iv in event_intervals])
                    merged_cores = []
                    for start, end in core_intervals:
                        if merged_cores and start <= merged_cores[-1][1]:
                            merged_cores[-1] = (merged_cores[-1][0], max(merged_cores[-1][1], end))
                        else:
                            merged_cores.append((start, end))
                    
                    # 合并margin区间
                    margin_intervals = sorted([iv['margin'] for iv in event_intervals])
                    merged_margins = []
                    for start, end in margin_intervals:
                        if merged_margins and start <= merged_margins[-1][1]:
                            merged_margins[-1] = (merged_margins[-1][0], max(merged_margins[-1][1], end))
                        else:
                            merged_margins.append((start, end))
                    
                    # 绘制合并后的区间
                    for start_ext, end_ext in merged_margins:
                        mask_ext = (times >= start_ext) & (times <= end_ext)
                        if mask_ext.any():
                            idx_ext = np.where(mask_ext)[0]
                            x0_ext = idx_ext[0] - 0.5
                            x1_ext = idx_ext[-1] + 0.5
                            ax_evt.axvspan(
                                x0_ext,
                                x1_ext,
                                color="#F5B7A6",
                                alpha=0.5,
                                ymin=y0,
                                ymax=y1,
                                edgecolor='none',
                                linewidth=0,
                            )
                    
                    # 绘制core（深红），会覆盖在margin上面
                    for start_true, end_true in merged_cores:
                        mask_true = (times >= start_true) & (times <= end_true)
                        if mask_true.any():
                            idx_true = np.where(mask_true)[0]
                            x0_true = idx_true[0] - 0.5
                            x1_true = idx_true[-1] + 0.5
                            ax_evt.axvspan(
                                x0_true,
                                x1_true,
                                color="#C73E1D",
                                alpha=0.5,  # 统一 alpha 为 0.5，与图例一致
                                ymin=y0,
                                ymax=y1,
                                edgecolor='none',
                                linewidth=0,
                            )

        # 事件图的 X 轴：用全局时间（秒），范围限制在视频片段内
        if fps is not None and fps > 0:
            frame0_evt = float(frames.min())
            times_evt = (frames - frame0_evt) / float(fps) + time_offset_s
            idx_ticks_evt = np.linspace(0, n_frames - 1, num=min(6, n_frames), dtype=int)
            ax_evt.set_xticks(idx_ticks_evt)
            ax_evt.set_xticklabels([f"{times_evt[i]:.1f}" for i in idx_ticks_evt], fontsize=9)
            ax_evt.set_xlabel("Time (s, global)", fontsize=12, fontweight="bold")
            # 设置X轴范围，限制在视频片段的时间范围内（不因margin而扩大）
            ax_evt.set_xlim(-0.5, n_frames - 0.5)
        else:
            ax_evt.set_xlabel("Frame index (global)", fontsize=12, fontweight="bold")
            ax_evt.set_xlim(-0.5, n_frames - 0.5)

        # 把 "__ALL_MISSING__" 的标签替换成更友好的显示文字
        yticks = [(i + 0.5) / max(1, n_evt_rows) for i in range(n_evt_rows)]
        ylabels = [lb for lb in event_labels]
        ax_evt.set_yticks(yticks)
        ax_evt.set_yticklabels(ylabels, fontsize=9)
        ax_evt.set_ylabel("Events", fontsize=11, fontweight="bold")
        ax_evt.set_ylim(0, 1)

    # 构造图例
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(color=cmap[2], label="Has all individuals (top/side1/side2 rows: this view has all individuals)"),
        Patch(color=missing_gray_map[1], label="Missing 1 (top/side1/side2 rows: 1 individual missing)"),
        Patch(color=missing_gray_map[2], label="Missing 2 (top/side1/side2 rows: 2 individuals missing)"),
        Patch(color=missing_gray_map[3], label="Missing 3+ (top/side1/side2 rows: 3+ individuals missing)"),
        Patch(color=cmap[1], label="All fully missing (4th row: all 3 views count=0)"),
        Patch(color=cmap[3], label="Partial missing (4th row: has missing but not all count=0)"),
        Patch(color=cmap[4], label="All views have all individuals (4th row: all 3 views count>=expected)"),
    ]
    if events is not None and fps is not None and fps > 0 and len(events) > 0:
        legend_patches.extend(
            [
                Patch(color="#C73E1D", alpha=0.5, label="Event (core)"),
                Patch(color="#F5B7A6", alpha=0.5, label="Event margin"),
            ]
        )

    ax_state.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=3,
        fontsize=8,
    )

    # 调整布局，确保 X 轴标签不被遮挡
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    if output_filename is None:
        output_filename = "multiview_missing_occlusion_overview.png"
    fig.savefig(
        output_dir / output_filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _compute_in_event_mask(
    df: pd.DataFrame,
    events: pd.DataFrame | None,
    fps: float | None,
    time_offset_s: float = 0.0,
) -> np.ndarray:
    """根据事件表计算每一帧是否处于"核心事件区间"（返回布尔数组）。"""
    frames = df["frame"].to_numpy()
    in_event = np.zeros_like(frames, dtype=bool)

    if events is None or len(events) == 0:
        return in_event

    # 如果提供 fps，则假设事件时间是以秒为单位；否则假设事件时间已经是帧
    if fps is not None and fps > 0:
        frame0 = float(frames.min())
        times = (frames - frame0) / float(fps) + time_offset_s
        for _, row in events.iterrows():
            start_true = float(row["start_s"])
            end_true = float(row["end_s"])
            in_event |= (times >= start_true) & (times <= end_true)
    else:
        for _, row in events.iterrows():
            start_true = float(row["start_s"])
            end_true = float(row["end_s"])
            in_event |= (frames >= start_true) & (frames <= end_true)

    return in_event


def annotate_events_on_frames(
    df: pd.DataFrame,
    events: pd.DataFrame | None,
    fps: float | None,
    time_offset_s: float = 0.0,
) -> pd.DataFrame:
    """在逐帧表上增加事件相关列：events (布尔) 和 event_name (字符串)。"""
    df = df.copy()
    frames = df["frame"].to_numpy()

    # 默认：没有事件
    df["events"] = False
    df["event_name"] = ""

    if events is None or len(events) == 0:
        return df

    # 帧是否在任一事件核心区间内
    in_event = _compute_in_event_mask(df, events, fps, time_offset_s)
    df["events"] = in_event

    # 逐事件写入名称；若一帧落在多个事件中，则名称用 ';' 连接
    names = np.array([""] * len(df), dtype=object)

    # 为每个事件类型单独建立一列（布尔），列名形如 event_<label_norm>
    label_to_mask: dict[str, np.ndarray] = {}

    # 根据 fps 决定用帧还是秒对齐
    if fps is not None and fps > 0:
        frame0 = float(frames.min())
        coord = (frames - frame0) / float(fps) + time_offset_s
    else:
        coord = frames

    for _, row in events.iterrows():
        start_true = float(row["start_s"])
        end_true = float(row["end_s"])
        raw_label = str(row.get("label", "event"))
        label = raw_label
        col_key = _normalize_event_label(raw_label)

        mask = (coord >= start_true) & (coord <= end_true)
        idxs = np.where(mask)[0]
        for idx in idxs:
            if names[idx]:
                # 已有名称，追加
                names[idx] = f"{names[idx]};{label}"
            else:
                names[idx] = label

        # 更新该行为对应的布尔列
        if col_key not in label_to_mask:
            label_to_mask[col_key] = np.zeros(len(df), dtype=bool)
        label_to_mask[col_key] |= mask

    df["event_name"] = names

    # 把每个事件类型的掩码写入 DataFrame，列名为 event_<label_norm>
    for col_key, mask in label_to_mask.items():
        col_name = f"event_{col_key}"
        df[col_name] = mask

    return df


def generate_multiview_summary_table(
    df: pd.DataFrame,
    events: pd.DataFrame | None,
    fps: float | None,
    output_dir: Path,
    output_filename: str | None = None,
    time_offset_s: float = 0.0,
) -> None:
    """生成一个 CSV 表格，总结整体及事件内的缺失情况。"""
    rows: list[dict] = []

    total_frames = len(df)
    rows.append({"Metric": "total_frames", "Value": total_frames})

    # 整体 true_missing 帧数
    total_true_missing = int(df["true_missing"].sum())
    rows.append({"Metric": "total_true_missing_frames", "Value": total_true_missing})

    # 各视角整体缺失帧数
    for view in ["top", "side1", "side2"]:
        miss = int(df[f"{view}_is_missing"].sum())
        rows.append({"Metric": f"{view}_missing_frames", "Value": miss})

    # 如有事件，则统计事件区间内的情况（只用核心区间）
    if events is not None and len(events) > 0:
        in_event = _compute_in_event_mask(df, events, fps, time_offset_s)
        rows.append({"Metric": "event_frames", "Value": int(in_event.sum())})

        # 事件内：全部缺失（三视角都缺）
        event_all_missing = int((df["true_missing"].to_numpy() & in_event).sum())
        rows.append({"Metric": "event_all_views_missing_frames", "Value": event_all_missing})

        # 事件内：恰好单视角缺失
        miss_counts = (
            df[["top_is_missing", "side1_is_missing", "side2_is_missing"]]
            .to_numpy(dtype=int)
            .sum(axis=1)
        )
        event_single_view_missing = int(((miss_counts == 1) & in_event).sum())
        rows.append(
            {"Metric": "event_single_view_missing_frames", "Value": event_single_view_missing}
        )

    summary_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        output_filename = "multiview_summary.csv"
    summary_df.to_csv(output_dir / output_filename, index=False)


def process_single_trial(
    slp_top: Path,
    slp_side1: Path,
    slp_side2: Path,
    expected_count: int,
    output_dir: Path | None,
    events_df: pd.DataFrame | None,
    fps: float | None,
    events_margin_s: float,
    trial_prefix: str | None = None,
) -> None:
    """处理单个 trial 的 QC 分析。"""
    print(f"\n{'='*60}")
    print(f"处理: {slp_top.name}")
    print(f"{'='*60}")

    if not slp_top.exists():
        raise SystemExit(f"Input file not found: {slp_top}")
    if not slp_side1.exists():
        raise SystemExit(f"Input file not found: {slp_side1}")
    if not slp_side2.exists():
        raise SystemExit(f"Input file not found: {slp_side2}")

    if output_dir is None:
        output_dir = slp_top.parent / f"{slp_top.stem}_multiview_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从文件名提取视频片段的起始时间（用于时间对齐）
    time_offset_s = extract_clip_start_time(slp_top) or 0.0
    if time_offset_s > 0:
        print(f"检测到视频片段起始时间: {time_offset_s:.3f} 秒（从文件名提取）")

    print("Classifying multi-view missing vs occlusion...")
    result = classify_multiview_missing(
        slp_top=slp_top,
        slp_side1=slp_side1,
        slp_side2=slp_side2,
        expected_count=expected_count,
    )

    # 计算相对于原始长视频的绝对时间（秒）
    # 公式：time_s = (frame - frame0) / fps + time_offset_s
    # 其中：
    #   - frame 来自当前片段（对齐后的多视角帧索引）
    #   - frame0 是当前片段中最小的帧号（通常是 0 或 1）
    #   - fps 是原始视频的帧率
    #   - time_offset_s 是从文件名中解析出的片段起始时间（单位：秒）
    if fps is not None and fps > 0:
        frame0 = float(result["frame"].min())
        result["time_s"] = (result["frame"].to_numpy() - frame0) / float(fps) + time_offset_s
    else:
        # 如果没有提供 fps，则无法换算为秒，这里填充为 NaN
        result["time_s"] = np.nan

    # 如提供行为事件 CSV，则加载，并在逐帧结果中加入 events / event_name 列
    if events_df is not None:
        # 在逐帧表中标记 events 和 event_name
        result = annotate_events_on_frames(
            df=result,
            events=events_df,
            fps=fps,
            time_offset_s=time_offset_s,
        )
    else:
        # 确保列存在
        result["events"] = False
        result["event_name"] = ""

    # 写出带有 events / event_name 的逐帧结果
    if trial_prefix:
        out_csv = output_dir / f"{trial_prefix}_multiview_missing_occlusion.csv"
    else:
        out_csv = output_dir / "multiview_missing_occlusion.csv"
    result.to_csv(out_csv, index=False)

    # 简单打印一些汇总信息
    n_frames = len(result)
    n_true_missing = int(result["true_missing"].sum())
    print(f"Total frames (after alignment): {n_frames}")
    print(f"Frames with true_missing (all views missing): {n_true_missing}")

    for view in ["top", "side1", "side2"]:
        miss = int(result[f"{view}_is_missing"].sum())
        occ = int(result[f"{view}_occlusion"].sum())
        print(
            f"{view}: missing frames={miss}, occlusion-like frames={occ} "
            f"(missing but at least one other view OK)"
        )

    # 从 slp_top 文件名提取标题
    # 例如：20251225_K_1_2_FCMC_PLAYER1_f_Spawn_816.572-820.572_top.slp
    # 提取为：20251225_K_1_2_FCMC_Spawn_816.572-820.572
    title_base = slp_top.stem  # 获取不带扩展名的文件名
    if title_base.endswith("_top"):
        title_base = title_base[:-4]  # 去掉 "_top" 后缀
    # 去掉 PLAYER1/PLAYER2/PLAYER3 等标识
    title_base = re.sub(r'_PLAYER[123]_', '_', title_base)
    # 去掉行为类型前缀（如 m1tf, m2tf, f 等，保留行为名如 Spawn）
    # 匹配模式：_m1tf_, _m2tf_, _f_, _m1_ 等
    title_base = re.sub(r'_[a-z0-9]+_(?=[A-Z])', '_', title_base)
    title = title_base

    # 生成可视化图像（带真实时间轴和可选事件高亮）
    if trial_prefix:
        vis_filename = f"{trial_prefix}_multiview_missing_occlusion_overview.png"
    else:
        vis_filename = None
    
    generate_multiview_visualizations(
        df=result,
        expected_count=expected_count,
        output_dir=output_dir,
        fps=fps,
        events=events_df,
        events_margin_s=events_margin_s,
        output_filename=vis_filename,
        time_offset_s=time_offset_s,
        title=title,
    )

    # 生成汇总表（整体缺失 / 各视角缺失 / 事件内单视角 vs 全部缺失）
    if trial_prefix:
        summary_filename = f"{trial_prefix}_multiview_summary.csv"
    else:
        summary_filename = None
    
    generate_multiview_summary_table(
        df=result,
        events=events_df,
        fps=fps,
        output_dir=output_dir,
        output_filename=summary_filename,
        time_offset_s=time_offset_s,
    )

    print(f"\n✓ Multi-view QC complete! Results saved to: {out_csv}")
    if summary_filename:
        print(f"  - Summary table: {output_dir / summary_filename}")
    else:
        print(f"  - Summary table: {output_dir / 'multiview_summary.csv'}")
    if vis_filename:
        print(f"  - Missing / occlusion overview: {output_dir / vis_filename}")
    else:
        print(
            f"  - Missing / occlusion overview: "
            f"{output_dir / 'multiview_missing_occlusion_overview.png'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "多视角 SLEAP 跟踪质量评估（缺失 vs 遮挡）。\n"
            "支持两种模式：\n"
            "1. 批量模式：使用 --top-slp 指定多个 top .slp 文件，自动查找对应的 side1 和 side2 文件\n"
            "2. 单文件模式：使用 --top/--side1/--side2 指定单个 trial 的三个文件"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 批量模式：可以指定多个 top .slp 文件
    parser.add_argument(
        "--top-slp",
        action="append",
        default=None,
        help="top 视角的 .slp 文件路径（可以指定多个，每个文件用 --top-slp 指定一次）。"
        "脚本会自动查找对应的 side1 和 side2 文件（通过文件名匹配）。",
    )
    # 单文件模式：向后兼容
    parser.add_argument(
        "--top",
        default=None,
        help="top 视角的 .slp 文件路径（单文件模式，与 --side1 和 --side2 一起使用）",
    )
    parser.add_argument(
        "--side1",
        default=None,
        help="side1 视角的 .slp 文件路径（单文件模式）",
    )
    parser.add_argument(
        "--side2",
        default=None,
        help="side2 视角的 .slp 文件路径（单文件模式）",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=3,
        help="每帧期望的个体数量（默认：3）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认：使用 top .slp 同目录下的 <stem>_multiview_qc 文件夹）",
    )
    # 可选：行为事件，用于在三视角图上高亮事件区间
    parser.add_argument(
        "--events-csv",
        default=None,
        help=(
            "可选：BORIS 导出的行为事件 CSV 文件路径。"
            "如果提供，将在可视化图中高亮事件区间和缓冲区。"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="视频帧率（Hz）。若提供，将把帧号换算为真实时间（秒），并用于事件对齐。",
    )
    parser.add_argument(
        "--events-time-col",
        default=None,
        help="可选：行为CSV中点事件时间列名（例如 'Time (s)' 或 'Image index'）。",
    )
    parser.add_argument(
        "--events-start-col",
        default=None,
        help="可选：行为CSV中区间事件起始时间列名（例如 'Start (s)'）。",
    )
    parser.add_argument(
        "--events-end-col",
        default=None,
        help="可选：行为CSV中区间事件结束时间列名（例如 'Stop (s)'）。",
    )
    parser.add_argument(
        "--events-label-col",
        default=None,
        help="可选：行为CSV中行为名称/标签列名（例如 'Behavior'）。主要用于在报告中列出事件类型。",
    )
    parser.add_argument(
        "--events-margin-s",
        type=float,
        default=0.0,
        help=(
            "可选：行为事件前后扩展的缓冲时间（秒）。例如设为 1.0 表示把每个事件区间扩展为 "
            "[start_s-1, end_s+1]，用于考虑 BORIS 标注时间前后约 1 秒的误差。"
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="--events-margin-s 的简短别名。例如 --margin 1.0 等同于 --events-margin-s 1.0",
    )
    parser.add_argument(
        "--skip-missing-side",
        action="store_true",
        help="如果 side 文件不存在，跳过而不报错（仅批量模式）",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "批量模式下的扁平化输出（仅批量模式）。"
            "如果启用，所有结果文件将保存在同一个目录下，文件名包含 trial 标识以避免冲突。"
            "默认行为是为每个 trial 创建独立的子文件夹。"
        ),
    )
    args = parser.parse_args()

    # 确定使用哪种模式
    use_batch_mode = args.top_slp is not None and len(args.top_slp) > 0
    use_single_mode = args.top is not None

    if not use_batch_mode and not use_single_mode:
        raise SystemExit(
            "请指定处理模式：\n"
            "  - 批量模式：使用 --top-slp 指定一个或多个 top .slp 文件\n"
            "  - 单文件模式：使用 --top/--side1/--side2 指定单个 trial 的三个文件"
        )

    if use_batch_mode and use_single_mode:
        raise SystemExit(
            "不能同时使用批量模式（--top-slp）和单文件模式（--top/--side1/--side2）。"
            "请选择其中一种模式。"
        )

    # 加载行为事件（如果提供）
    events_df: pd.DataFrame | None = None
    if args.events_csv:
        events_path = Path(args.events_csv)
        if not events_path.exists():
            raise SystemExit(f"Events CSV not found: {events_path}")
        if args.fps is None or args.fps <= 0:
            raise SystemExit("当使用 --events-csv 时，必须通过 --fps 指定视频帧率（正数）。")

        print(f"Loading behavioural events from: {events_path}")
        events_df = load_events_csv(
            events_path,
            time_col=args.events_time_col,
            start_col=args.events_start_col,
            end_col=args.events_end_col,
            label_col=args.events_label_col,
        )
        print(f"Loaded {len(events_df)} events.")

    fps = float(args.fps) if args.fps is not None and args.fps > 0 else None
    
    # 处理 --margin 别名
    if args.margin is not None:
        events_margin_s = float(args.margin)
    else:
        events_margin_s = float(args.events_margin_s or 0.0)

    # 批量模式
    if use_batch_mode:
        top_slp_files = [Path(t).resolve() for t in args.top_slp]
        
        print(f"\n开始批量 QC 处理...")
        print(f"  Top .slp 文件数量: {len(top_slp_files)}")
        if args.flat_output:
            if args.output_dir:
                print(f"  输出模式: 扁平化（所有文件在同一目录，文件名包含 trial 标识）")
                print(f"  输出目录: {args.output_dir}")
            else:
                print(f"  输出模式: 扁平化（所有文件在同一目录，文件名包含 trial 标识）")
                print(f"  输出目录: {top_slp_files[0].parent / 'multiview_qc_batch'}")
        else:
            print(f"  输出模式: 每个 trial 独立子文件夹")
            if args.output_dir:
                print(f"  基础输出目录: {args.output_dir}")
            else:
                print(f"  输出目录: 每个 trial 的 .slp 文件同目录")
        print(f"  期望个体数: {args.expected_count}")
        if events_df is not None:
            print(f"  行为事件: {len(events_df)} 个事件")
        if fps is not None:
            print(f"  视频帧率: {fps} Hz")

        # 处理每个 top 文件
        for i, top_slp in enumerate(top_slp_files, 1):
            print(f"\n{'='*60}")
            print(f"处理 Top .slp {i}/{len(top_slp_files)}: {top_slp.name}")
            print(f"{'='*60}")

            if not top_slp.exists():
                raise SystemExit(f"Input file not found: {top_slp}")

            # 自动查找对应的 side 文件
            side1_slp, side2_slp = find_side_slp_files(top_slp)

            if side1_slp is None or side2_slp is None:
                error_msg = (
                    f"未找到对应的 side 文件：\n"
                    f"  Top: {top_slp}\n"
                )
                if side1_slp is None:
                    error_msg += f"  Side1: 未找到（预期: {str(top_slp).replace('PLAYER1', 'PLAYER2').replace('_top', '_side1')}）\n"
                if side2_slp is None:
                    error_msg += f"  Side2: 未找到（预期: {str(top_slp).replace('PLAYER1', 'PLAYER3').replace('_top', '_side2')}）\n"
                
                if args.skip_missing_side:
                    print(f"⚠ 跳过（side 文件未找到）")
                    continue
                else:
                    raise SystemExit(error_msg)

            # 确定输出目录和 trial 前缀
            if args.flat_output:
                # 扁平化输出：所有文件在同一目录，文件名包含 trial 标识
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                else:
                    # 如果没有指定输出目录，使用第一个 top 文件的目录，创建 qc 子目录
                    output_dir = top_slp_files[0].parent / "multiview_qc_batch"
                output_dir.mkdir(parents=True, exist_ok=True)
                # 使用 top 文件的 stem 作为前缀（去除可能的 _top 后缀）
                trial_prefix = top_slp.stem.replace("_top", "")
            else:
                # 默认行为：为每个 trial 创建独立的子文件夹
                if args.output_dir:
                    output_dir = Path(args.output_dir) / top_slp.stem
                else:
                    output_dir = top_slp.parent / f"{top_slp.stem}_multiview_qc"
                trial_prefix = None

            # 处理单个 trial
            process_single_trial(
                slp_top=top_slp,
                slp_side1=side1_slp,
                slp_side2=side2_slp,
                expected_count=args.expected_count,
                output_dir=output_dir,
                events_df=events_df,
                fps=fps,
                events_margin_s=events_margin_s,
                trial_prefix=trial_prefix,
            )

        print(f"\n{'='*60}")
        print(f"✓ 所有文件处理完成！")
        if args.flat_output:
            if args.output_dir:
                print(f"  输出目录: {args.output_dir} (扁平化模式)")
            else:
                print(f"  输出目录: {top_slp_files[0].parent / 'multiview_qc_batch'} (扁平化模式)")
        else:
            if args.output_dir:
                print(f"  基础输出目录: {args.output_dir} (每个 trial 有独立子文件夹)")
            else:
                print(f"  输出目录: 每个 trial 的 .slp 文件同目录")
        print(f"{'='*60}")

    # 单文件模式（向后兼容）
    else:
        slp_top = Path(args.top).resolve()
        slp_a = Path(args.side1).resolve() if args.side1 else None
        slp_b = Path(args.side2).resolve() if args.side2 else None

        if slp_a is None or slp_b is None:
            raise SystemExit("单文件模式需要同时指定 --top、--side1 和 --side2")

        # 确定输出目录
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = slp_top.parent / f"{slp_top.stem}_multiview_qc"

        # 处理单个 trial
        process_single_trial(
            slp_top=slp_top,
            slp_side1=slp_a,
            slp_side2=slp_b,
            expected_count=args.expected_count,
            output_dir=output_dir,
            events_df=events_df,
            fps=fps,
            events_margin_s=float(args.events_margin_s or 0.0),
        )


if __name__ == "__main__":
    main()
