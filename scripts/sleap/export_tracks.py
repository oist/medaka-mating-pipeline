from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    import sleap_io as sio  # type: ignore[import]
except ImportError:
    sio = None  # type: ignore[assignment]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _sleap_cmd(exe: str, args: list[str]) -> list[str]:
    """Resolve a CLI from PATH or next to the active Python interpreter."""
    resolved = shutil.which(exe)
    if resolved is not None:
        return [resolved, *args]
    adjacent_exe = Path(sys.executable).with_name(f"{exe}.exe")
    if adjacent_exe.exists():
        return [str(adjacent_exe), *args]
    return [exe, *args]


def _load_slp(slp_path: Path):
    """Load an SLP file with the native SLEAP-NN I/O package."""
    if sio is None:
        raise SystemExit(
            "Cannot import sleap-io. Install the project environment first, e.g.:\n"
            "  uv sync --extra sleap"
        )

    try:
        return sio.load_slp(slp_path)
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"Failed to load SLEAP file: {slp_path}\n"
            f"Error: {type(error).__name__}: {error}\n"
            "Please check that the path exists and the SLP file is valid."
        ) from error


def extract_video_and_model_info(slp_path: Path):
    """
    Extract the source video name and inferred model name from a SLEAP file.

    Returns:
    - `video_name`: source video filename without extension
    - `model_name`: inferred model name, usually from the parent directory
    """
    labels = _load_slp(slp_path)

    video_name = None
    if hasattr(labels, "videos") and len(labels.videos) > 0:
        video_path = Path(labels.videos[0].filename)
        video_name = video_path.stem

    model_name = None
    slp_parent = slp_path.parent
    if slp_parent.name:
        model_name = slp_parent.name

    return video_name, model_name


def load_slp_data_all_tracks(slp_path: Path):
    """Export all tracks from a `.slp` file using stable track identifiers."""
    print(f"Loading SLEAP file: {slp_path}")
    labels = _load_slp(slp_path)

    label_tracks = getattr(labels, "tracks", None) or []
    try:
        label_tracks = list(label_tracks)
    except Exception:
        label_tracks = []

    def _instance_points_with_score(inst, lf, score_thresh: float = 0.01) -> dict[str, tuple[float, float, float]]:
        """
        Return `node_name -> (x, y, score)` while keeping only points whose
        score is at least `score_thresh`.
        """
        pts = getattr(inst, "points", None)
        if pts is None:
            return {}

        out: dict[str, tuple[float, float, float]] = {}

        if hasattr(pts, "items"):
            for node, p in pts.items():
                score = float(getattr(p, "score", 1.0))
                if score < score_thresh:
                    continue
                x = float(getattr(p, "x", p[0] if hasattr(p, "__getitem__") else 0.0))
                y = float(getattr(p, "y", p[1] if hasattr(p, "__getitem__") else 0.0))
                out[str(node)] = (x, y, score)
            return out

        try:
            import numpy as np
        except ImportError as e:  # noqa: BLE001
            raise SystemExit(
                "numpy is required to export all tracks. "
                "Please install it, e.g.:\n  pip install numpy"
            ) from e
        sk = getattr(labels, "skeleton", None)
        if sk is None and hasattr(lf, "video"):
            sk = getattr(lf.video, "skeleton", None)

        node_names: list[str] = []
        if sk is not None:
            node_names = getattr(sk, "node_names", None) or getattr(sk, "nodes", [])
            node_names = [str(n) for n in node_names]

        points_array = getattr(inst, "numpy", lambda: None)() if hasattr(inst, "numpy") else pts
        if isinstance(points_array, np.ndarray) and len(points_array.shape) >= 2:
            for i in range(points_array.shape[0]):
                name = node_names[i] if i < len(node_names) else f"node_{i}"
                x = float(points_array[i, 0])
                y = float(points_array[i, 1])
                score = float(points_array[i, 2]) if points_array.shape[1] > 2 else 1.0
                if score < score_thresh:
                    continue
                out[name] = (x, y, score)

        return out

    rows: list[dict] = []

    for lf in labels.labeled_frames:  # type: ignore[attr-defined]
        frame_idx = int(getattr(lf, "frame_idx", 0))

        for inst_idx, inst in enumerate(getattr(lf, "instances", [])):
            track_name = None
            if getattr(inst, "track", None) is not None:
                track_name = getattr(inst.track, "name", None) or getattr(
                    inst.track, "track_id", None
                )
                if track_name is None and label_tracks:
                    try:
                        idx = label_tracks.index(inst.track)
                        track_name = f"Track_{idx}"
                    except (ValueError, TypeError):
                        pass
            if track_name is None:
                track_name = f"untracked_f{frame_idx}_{inst_idx}"

            pts = _instance_points_with_score(inst, lf)
            if not pts:
                continue

            for node_name, (x, y, score) in pts.items():
                rows.append(
                    {
                        "frame": frame_idx,
                        "track": str(track_name),
                        "node": str(node_name),
                        "x": float(x),
                        "y": float(y),
                        "score": float(score),
                    }
                )

    if not rows:
        raise SystemExit("Error: No tracking data found in .slp file.")

    df = pd.DataFrame.from_records(rows)
    print(f"Loaded {len(df)} data points, {df['track'].nunique()} tracks")
    return df


def load_analysis_h5_all_tracks(h5_path: Path) -> pd.DataFrame:
    """
    Load all tracks from an analysis HDF5 exported by `sleap-io export` and
    return them in long-table form with columns:
    `frame, track, node, x, y, score`.
    """
    try:
        import h5py  # type: ignore[import]
        import numpy as np  # type: ignore[import]
    except ImportError as e:  # noqa: BLE001
        raise SystemExit(
            "h5py and numpy are required to read analysis HDF5.\n"
            "Please install them, e.g.:\n"
            "  pip install h5py numpy"
        ) from e

    if not h5_path.exists():
        raise SystemExit(f"Analysis HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "tracks" not in f:
            raise SystemExit(f"'tracks' dataset not found in analysis HDF5: {h5_path}")
        tracks = f["tracks"][()]

        track_names_ds = f.get("track_names")
        if track_names_ds is not None:
            raw = track_names_ds[()]
            track_names = [
                (name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name))
                for name in raw
            ]
        else:
            track_names = []

        node_names_ds = f.get("node_names")
        if node_names_ds is not None:
            raw_nodes = node_names_ds[()]
            node_names = [
                (name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name))
                for name in raw_nodes
            ]
        else:
            node_names = []

        point_scores_ds = f.get("point_scores")
        point_scores = point_scores_ds[()] if point_scores_ds is not None else None

    if tracks.ndim != 4:
        raise SystemExit(f"Unsupported tracks ndim={tracks.ndim} in analysis HDF5 (expected 4).")

    use_matlab_order = False
    if tracks.shape[1] == 2:
        use_matlab_order = True
    elif tracks.shape[-1] == 2:
        use_matlab_order = False
    else:
        use_matlab_order = True

    rows: list[dict] = []

    if use_matlab_order:
        n_tracks, _, n_nodes, n_frames = tracks.shape
        if not track_names:
            track_names = [f"Track_{i}" for i in range(n_tracks)]
        if not node_names:
            node_names = [f"node_{i}" for i in range(n_nodes)]

        if point_scores is not None and point_scores.shape != (n_tracks, n_nodes, n_frames):
            point_scores = None

        for t in range(n_tracks):
            tname = track_names[t] if t < len(track_names) else f"Track_{t}"
            for n in range(n_nodes):
                nname = node_names[n] if n < len(node_names) else f"node_{n}"
                for fi in range(n_frames):
                    x = float(tracks[t, 0, n, fi])
                    y = float(tracks[t, 1, n, fi])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    score = 1.0
                    if point_scores is not None:
                        score = float(point_scores[t, n, fi])
                    rows.append(
                        {
                            "frame": int(fi),
                            "track": str(tname),
                            "node": str(nname),
                            "x": x,
                            "y": y,
                            "score": score,
                        }
                    )
    else:
        n_frames, n_tracks, n_nodes, _ = tracks.shape
        if not track_names:
            track_names = [f"Track_{i}" for i in range(n_tracks)]
        if not node_names:
            node_names = [f"node_{i}" for i in range(n_nodes)]

        if point_scores is not None and point_scores.shape != (n_frames, n_tracks, n_nodes):
            point_scores = None

        for fi in range(n_frames):
            for t in range(n_tracks):
                tname = track_names[t] if t < len(track_names) else f"Track_{t}"
                for n in range(n_nodes):
                    nname = node_names[n] if n < len(node_names) else f"node_{n}"
                    x = float(tracks[fi, t, n, 0])
                    y = float(tracks[fi, t, n, 1])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    score = 1.0
                    if point_scores is not None:
                        score = float(point_scores[fi, t, n])
                    rows.append(
                        {
                            "frame": int(fi),
                            "track": str(tname),
                            "node": str(nname),
                            "x": x,
                            "y": y,
                            "score": score,
                        }
                    )

    if not rows:
        raise SystemExit(f"No valid tracking data found in analysis HDF5: {h5_path}")

    df = pd.DataFrame.from_records(rows)
    print(
        f"Loaded {len(df)} data points from analysis HDF5, "
        f"{df['track'].nunique()} tracks, {df['node'].nunique()} nodes"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export SLEAP predictions (.slp) with the native `sleap-io` CLI.\n"
            "For CSV with stable track identity matching the overlay video, use --all-tracks."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input predictions .slp (or labels .slp).",
    )
    parser.add_argument(
        "--format",
        default="analysis",
        choices=["analysis", "analysis.csv", "analysis.nix", "h5", "json", "slp"],
        help=(
            "Output format. analysis/analysis.csv/h5 use `sleap-io export`; "
            "slp uses `sleap-io convert`. Legacy JSON/NIX exports are unsupported."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. If omitted and --all-tracks is used with CSV export, "
             "the script auto-generates {model_name}_{video_name}.csv",
    )
    parser.add_argument(
        "--video",
        help="Optional video path for conversions that require a video reference.",
    )
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="For CSV export, write all tracks instead of only the first one using the direct SLEAP API.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if args.format == "analysis.csv" and args.all_tracks:
        print("Exporting all tracks to CSV using direct loading from .slp ...")
        df = load_slp_data_all_tracks(input_path)

        video_name, model_name = extract_video_and_model_info(input_path)

        if video_name is None:
            raise SystemExit(
                "Could not extract the source video name from the SLEAP file. "
                "Please provide --output explicitly."
            )

        if model_name is None:
            model_name = "unknown_model"
            print(f"Warning: could not infer model name from the path; using: {model_name}")

        output_filename = f"{model_name}_{video_name}.csv"

        if args.output is None:
            output_path = input_path.parent / output_filename
            print(f"Auto-generated output filename: {output_filename}")
        else:
            output_path = Path(args.output)
            if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
                output_path = output_path / output_filename
                print(f"Creating the auto-named file inside the specified directory: {output_filename}")
            else:
                print(f"Using the explicitly requested output path: {output_path}")
                print(
                    f"Tip: if you want the auto-generated name ({output_filename}), "
                    "pass a directory instead of a full file path."
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully exported to: {output_path}")
        print(f"  - Total data points: {len(df)}")
        print(f"  - Number of tracks: {df['track'].nunique()}")
        print(f"  - Number of nodes: {df['node'].nunique()}")
        print(f"  - Frame range: {df['frame'].min()} - {df['frame'].max()}")
        return

    if args.output is None:
        raise SystemExit("--output is required unless you use CSV export with --all-tracks")

    if args.format in {"json", "analysis.nix"}:
        raise SystemExit(
            f"Format {args.format!r} depended on the removed legacy `sleap-convert` "
            "command. Use analysis.csv, analysis/h5, or slp instead."
        )

    if args.format in {"analysis", "h5", "analysis.csv"}:
        output_format = "csv" if args.format == "analysis.csv" else "h5"
        export_args = [
            "export",
            str(input_path),
            "--output",
            args.output,
            "--format",
            output_format,
        ]
        if args.video:
            export_args += ["--video", args.video]
        _run(_sleap_cmd("sleap-io", export_args))
        return

    convert_args = [
        "convert",
        str(input_path),
        "--output",
        args.output,
        "--to",
        "slp",
    ]
    _run(_sleap_cmd("sleap-io", convert_args))


if __name__ == "__main__":
    main()
