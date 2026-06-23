from __future__ import annotations

"""Local runtime helpers for UDMT's decoupled ST-Net / TrDiMP tracker.

This module intentionally does not vendor PyTracking/LTR. It makes an existing UDMT
clone usable from this repo by patching the custom PrRoIPool op, installing tiny Qt
stubs for headless imports, preparing UDMT's on-disk project contract, and calling
the patched ST-Net runner.
"""

import argparse
import csv
import importlib
import json
import os
import re
import shutil
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


QT_MODULES = ("PySide6", "PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtCore")


@dataclass
class ClonePatchReport:
    stnet_root: str
    pure_prroi: str
    functional_py: str
    functional_backup: str
    copied_pure_prroi: bool
    wrote_functional: bool
    rewritten_files: int
    remaining_gui_import_refs: int
    remaining_base_dir_imports: int


@dataclass
class ProjectPrepReport:
    project: str
    stem: str
    frames_dir: str
    masks_dir: str
    start_positions: str
    n_frames: int
    n_masks: int


@dataclass
class TrackingReport:
    project: str
    stem: str
    checkpoint: str
    tracking_result_dir: str
    npy: str
    dataset_name: str
    n_frames: int
    tuned_search_scale: float | None
    tuned_target_size_bias: float | None


def find_stnet_root(udmt_root: Path | str | None = None,
                    stnet_root: Path | str | None = None) -> Path:
    if stnet_root is not None:
        root = Path(stnet_root)
    elif udmt_root is not None:
        root = Path(udmt_root) / "udmt" / "gui" / "tabs" / "ST_Net"
    else:
        raise ValueError("Provide udmt_root or stnet_root.")
    if not root.exists():
        raise FileNotFoundError(f"ST_Net root not found: {root}")
    return root.resolve()


def default_checkpoint(udmt_root: Path | str) -> Path:
    root = Path(udmt_root)
    candidates = [
        root / "udmt" / "gui" / "pretrained" / "trdimp_net_ep.pth.tar",
        root / "udmt" / "pretrained" / "trdimp_net_ep.pth.tar",
        root / "pretrained" / "trdimp_net_ep.pth.tar",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find trdimp_net_ep.pth.tar. Pass --checkpoint explicitly."
    )


def install_qt_stubs(force: bool = False) -> bool:
    if not force:
        try:
            importlib.import_module("PySide6")
            return False
        except Exception:
            pass

    parent = sys.modules.setdefault("PySide6", types.ModuleType("PySide6"))
    for name in QT_MODULES[1:]:
        module = sys.modules.setdefault(name, types.ModuleType(name))
        setattr(parent, name.split(".")[-1], module)
    return True


def configure_runtime(stnet_root: Path | str, *,
                      udmt_root: Path | str | None = None,
                      base_dir: Path | str | None = None,
                      qt_stubs: bool = True,
                      cuda_visible_devices: str | None = None) -> None:
    root = Path(stnet_root).resolve()
    paths = [str(root)]
    if udmt_root is not None:
        paths.insert(0, str(Path(udmt_root).resolve()))
    for entry in reversed(paths):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("DEBUG", "")
    if base_dir is not None:
        os.environ["UDMT_BASE_DIR"] = str(Path(base_dir).resolve())
    elif udmt_root is not None:
        os.environ.setdefault("UDMT_BASE_DIR", str((Path(udmt_root) / "udmt" / "gui").resolve()))
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    sys.argv = [sys.argv[0]]
    if qt_stubs:
        install_qt_stubs()
    install_legacy_dependency_stubs()


def install_legacy_dependency_stubs() -> list[str]:
    installed: list[str] = []
    if _missing_module("jpeg4py"):
        jpeg4py = types.ModuleType("jpeg4py")

        class JPEG:
            def __init__(self, path):
                self.path = path

            def decode(self):
                import cv2

                image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(self.path)
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        jpeg4py.JPEG = JPEG
        sys.modules["jpeg4py"] = jpeg4py
        installed.append("jpeg4py")

    if _missing_module("similaritymeasures"):
        pkg = types.ModuleType("similaritymeasures")
        sub = types.ModuleType("similaritymeasures.similaritymeasures")

        def dtw(p, q):
            import numpy as np

            p_arr = np.asarray(p, dtype=float)
            q_arr = np.asarray(q, dtype=float)
            dp = np.full((len(p_arr) + 1, len(q_arr) + 1), np.inf)
            dp[0, 0] = 0.0
            for i in range(1, len(p_arr) + 1):
                for j in range(1, len(q_arr) + 1):
                    cost = float(np.linalg.norm(p_arr[i - 1] - q_arr[j - 1]))
                    dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            return float(dp[-1, -1]), []

        sub.dtw = dtw
        pkg.similaritymeasures = sub
        sys.modules["similaritymeasures"] = pkg
        sys.modules["similaritymeasures.similaritymeasures"] = sub
        installed.append("similaritymeasures")

    if _missing_module("visdom"):
        visdom = types.ModuleType("visdom")
        server = types.ModuleType("visdom.server")

        class Visdom:
            def __init__(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                def _noop(*args, **kwargs):
                    return None

                return _noop

        visdom.Visdom = Visdom
        visdom.server = server
        sys.modules["visdom"] = visdom
        sys.modules["visdom.server"] = server
        installed.append("visdom")

    return installed


def patch_clone(udmt_root: Path | str, pure_prroi: Path | str | None = None) -> ClonePatchReport:
    stnet = find_stnet_root(udmt_root=udmt_root)
    pure = Path(pure_prroi) if pure_prroi is not None else Path(__file__).with_name("prroi_pool.py")
    pure = pure.resolve()
    pool_dir = (
        stnet / "ltr" / "external" / "PreciseRoIPooling" / "pytorch" / "prroi_pool"
    )
    if not pool_dir.exists():
        raise FileNotFoundError(f"PreciseRoIPooling package not found: {pool_dir}")

    pure_dst = pool_dir / "pure_prroi.py"
    functional = pool_dir / "functional.py"
    backup = pool_dir / "functional.py.cuda-orig"
    copied = _copy_if_changed(pure, pure_dst)
    if functional.exists() and not backup.exists():
        shutil.copy2(functional, backup)
    functional_body = (
        "# Patched: use the pure-PyTorch Precise RoI Pooling from this repo.\n"
        "from .pure_prroi import prroi_pool2d\n"
        "__all__ = ['prroi_pool2d']\n"
    )
    wrote_functional = _write_text_if_changed(functional, functional_body)
    rewritten = decouple_imports(stnet)
    remaining_gui = _count_files_containing(stnet, "udmt.gui.tabs.ST_Net")
    remaining_base = _count_files_containing(stnet, "from udmt.gui import BASE_DIR")
    return ClonePatchReport(
        stnet_root=str(stnet),
        pure_prroi=str(pure_dst),
        functional_py=str(functional),
        functional_backup=str(backup),
        copied_pure_prroi=copied,
        wrote_functional=wrote_functional,
        rewritten_files=rewritten,
        remaining_gui_import_refs=remaining_gui,
        remaining_base_dir_imports=remaining_base,
    )


def decouple_imports(stnet_root: Path | str) -> int:
    root = Path(stnet_root)
    changed = 0
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new = text
        new = new.replace("udmt.gui.tabs.ST_Net.ltr", "ltr")
        new = new.replace("udmt.gui.tabs.ST_Net.pytracking", "pytracking")
        new = re.sub(r"from \.\.+(ltr|pytracking)", r"from \1", new)
        new = new.replace(
            "from udmt.gui import BASE_DIR",
            'import os as _os; BASE_DIR = _os.environ.get("UDMT_BASE_DIR", '
            '_os.path.dirname(_os.path.abspath(__file__)))',
        )
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    return changed


def verify_runtime(udmt_root: Path | str, *, run_iou_forward: bool = False,
                   device: str = "cpu") -> dict:
    stnet = find_stnet_root(udmt_root=udmt_root)
    configure_runtime(stnet, udmt_root=udmt_root)
    import torch

    imports = {}
    for name in (
        "pytracking.run_tracker",
        "pytracking.evaluation.tracker_multi",
        "ltr.models.bbreg.atom_iou_net",
    ):
        imports[name] = importlib.import_module(name).__file__

    tm = importlib.import_module("pytracking.evaluation.tracker_multi")
    getattr(tm, "Tracker")
    tm.vis_flag = False
    tm.save_flag = True

    report = {
        "stnet_root": str(stnet),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "imports": imports,
    }
    if run_iou_forward:
        report["iou_forward"] = _run_iou_forward(device)
    return report


def prepare_project(video: Path | str, project: Path | str, stem: str,
                    start_positions: Path | str, *,
                    resize_factor: float = 0.8,
                    downsample: int = 1,
                    start_frame: int = 0,
                    frame_limit: int | None = None,
                    mask_dir: Path | str | None = None,
                    init_mask: Path | str | None = None,
                    repeat_init_mask: bool = False) -> ProjectPrepReport:
    project_path = Path(project)
    frames_dir = project_path / "tmp" / stem / "extracted-images"
    masks_out = project_path / "tmp" / stem / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    n_frames = extract_frames(
        video,
        frames_dir,
        resize_factor,
        downsample,
        frame_limit,
        start_frame=start_frame,
    )
    xy = read_start_positions(start_positions)
    write_start_positions(xy, frames_dir / "start_pos_array.txt")
    n_masks = prepare_masks(
        masks_out,
        n_frames=n_frames,
        mask_dir=mask_dir,
        init_mask=init_mask,
        repeat_init_mask=repeat_init_mask,
    )
    return ProjectPrepReport(
        project=str(project_path),
        stem=stem,
        frames_dir=str(frames_dir),
        masks_dir=str(masks_out),
        start_positions=str(frames_dir / "start_pos_array.txt"),
        n_frames=n_frames,
        n_masks=n_masks,
    )


def extract_frames(video: Path | str, frames_dir: Path | str, resize_factor: float,
                   downsample: int, frame_limit: int | None = None, *,
                   start_frame: int = 0) -> int:
    import cv2

    frames = Path(frames_dir)
    frames.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read video: {video}")
    width = max(1, int(first.shape[1] * resize_factor))
    height = max(1, int(first.shape[0] * resize_factor))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
    step = max(1, int(round(1 / downsample)))
    saved = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (frame_limit is not None and saved >= frame_limit):
            break
        if frame_idx % step == 0:
            out = frames / f"{saved:05d}.jpg"
            cv2.imwrite(str(out), cv2.resize(frame, (width, height)))
            saved += 1
        frame_idx += 1
    cap.release()
    if saved == 0:
        raise RuntimeError(f"No frames extracted from video: {video}")
    return saved


def read_start_positions(path: Path | str) -> list[tuple[float, float]]:
    rows = list(csv.reader(Path(path).read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"Empty start-position file: {path}")
    header = [c.strip().lower() for c in rows[0]]
    has_header = not _is_float(header[0]) if header else False
    positions: list[tuple[float, float]] = []
    if has_header:
        x_idx = _first_existing(header, ("x", "centroid_x", "start_x"))
        y_idx = _first_existing(header, ("y", "centroid_y", "start_y"))
        data_rows = rows[1:]
    else:
        x_idx, y_idx = 0, 1
        data_rows = rows
    for row in data_rows:
        if len(row) <= max(x_idx, y_idx) or not row:
            continue
        positions.append((float(row[x_idx]), float(row[y_idx])))
    if not positions:
        raise ValueError(f"No x/y start positions found in: {path}")
    return positions


def write_start_positions(positions: Iterable[tuple[float, float]], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for x, y in positions:
            writer.writerow((f"{x:.2f}", f"{y:.2f}"))


def prepare_masks(masks_out: Path | str, *, n_frames: int,
                  mask_dir: Path | str | None = None,
                  init_mask: Path | str | None = None,
                  repeat_init_mask: bool = False) -> int:
    out = Path(masks_out)
    out.mkdir(parents=True, exist_ok=True)
    if mask_dir is not None:
        src_dir = Path(mask_dir)
        pngs = sorted(src_dir.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"No PNG masks found in: {src_dir}")
        copied = 0
        for src in pngs[:n_frames]:
            _copy_if_changed(src, out / src.name)
            copied += 1
        if not (out / "0000000.png").exists():
            first = copied and sorted(out.glob("*.png"))[0]
            if first:
                shutil.copy2(first, out / "0000000.png")
        return copied

    if init_mask is not None and repeat_init_mask:
        src = Path(init_mask)
        if not src.exists():
            raise FileNotFoundError(f"Initial mask not found: {src}")
        for i in range(n_frames):
            _copy_if_changed(src, out / f"{i:07d}.png")
        return n_frames

    existing = list(out.glob("*.png"))
    if existing:
        return len(existing)
    raise ValueError("Provide --mask-dir, or --init-mask with --repeat-init-mask.")


def run_tracking(udmt_root: Path | str, project: Path | str, stem: str,
                 checkpoint: Path | str, *, frame_rate: float = 166.0,
                 resize_factor: float = 0.8, downsample: int = 1,
                 tune_frames: int = 4000, quick: bool = False,
                 skip_tune: bool = False, search_scale: float = 2.0,
                 target_size_bias: float = 0.0) -> TrackingReport:
    stnet = find_stnet_root(udmt_root=udmt_root)
    configure_runtime(stnet, udmt_root=udmt_root)
    allow_legacy_torch_checkpoint_load()
    from pytracking.run_tracker import run_tracking as udmt_run_tracking
    from pytracking.utils.filter_and_save import post_process_results
    import pytracking.evaluation.tracker_multi as tm

    tm.vis_flag = False
    tm.save_flag = True

    project_s = str(Path(project))
    frames_dir = Path(project) / "tmp" / stem / "extracted-images"
    n_frames = len(list(frames_dir.glob("*.jpg")))
    if n_frames == 0:
        raise RuntimeError(f"No extracted frames found: {frames_dir}")

    scales = [float(search_scale)] if quick or skip_tune else [1.5, 2.0, 2.5]
    biases = [float(target_size_bias)] if quick or skip_tune else [-0.1, 0.0, 0.1]
    tuned_scale: float | None = float(search_scale) if skip_tune else None
    tuned_bias: float | None = float(target_size_bias) if skip_tune else None

    common = {
        "video_display_widget1": None,
        "video_display_widget2": None,
        "project_folder": project_s,
        "video_name": stem,
        "model_path": str(checkpoint),
        "resize_factor": resize_factor,
        "downsample_factor": downsample,
        "frame_rate": frame_rate,
        "evaluation_metric": [],
        "is_concave": False,
    }

    if not skip_tune:
        params = {
            **common,
            "frame_num": min(int(tune_frames), n_frames),
            "status_flag": 2,
            "search_scale_range": scales,
            "target_sz_bias_range": biases,
        }
        udmt_run_tracking(params)
        best = _read_last_metric(Path(project) / "tmp" / stem / "evaluation_metric_for_test.json")
        tuned_scale = float(best["search_scale"])
        tuned_bias = float(best["target_sz_bias"])

    final_params = {
        **common,
        "frame_num": n_frames - 1,
        "status_flag": 3,
        "search_scale_range": [tuned_scale],
        "target_sz_bias_range": [tuned_bias],
        "evaluation_metric": [],
    }
    udmt_run_tracking(final_params)

    result_dir = Path(project) / "tracking-results" / stem
    raw = _newest_dir(result_dir)
    dataset_name, npy = post_process_results(stem, 5, str(raw), str(result_dir))
    return TrackingReport(
        project=project_s,
        stem=stem,
        checkpoint=str(checkpoint),
        tracking_result_dir=str(result_dir),
        npy=str(npy),
        dataset_name=dataset_name,
        n_frames=n_frames,
        tuned_search_scale=tuned_scale,
        tuned_target_size_bias=tuned_bias,
    )


def allow_legacy_torch_checkpoint_load() -> None:
    """Make trusted UDMT checkpoints load on torch>=2.6.

    UDMT's TrDiMP checkpoint pickles its NetConstructor. PyTorch 2.6 changed
    ``torch.load`` to default to ``weights_only=True``, which blocks that legacy
    object. The local runner only calls this for the explicit user-provided
    checkpoint path, so use the old behavior there.
    """
    import torch

    if getattr(torch.load, "_udmt_legacy_patch", False):
        return
    original_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    _load._udmt_legacy_patch = True
    torch.load = _load


def run_video(udmt_root: Path | str, video: Path | str, project: Path | str,
              stem: str, start_positions: Path | str, *,
              checkpoint: Path | str | None = None,
              mask_dir: Path | str | None = None,
              init_mask: Path | str | None = None,
              repeat_init_mask: bool = False,
              resize_factor: float = 0.8,
              downsample: int = 1,
              start_frame: int = 0,
              frame_limit: int | None = None,
              frame_rate: float = 166.0,
              tune_frames: int = 4000,
              quick: bool = False,
              skip_tune: bool = False,
              search_scale: float = 2.0,
              target_size_bias: float = 0.0) -> dict:
    ckpt = Path(checkpoint).resolve() if checkpoint is not None else default_checkpoint(udmt_root)
    prep = prepare_project(
        video,
        project,
        stem,
        start_positions,
        resize_factor=resize_factor,
        downsample=downsample,
        start_frame=start_frame,
        frame_limit=frame_limit,
        mask_dir=mask_dir,
        init_mask=init_mask,
        repeat_init_mask=repeat_init_mask,
    )
    track = run_tracking(
        udmt_root,
        project,
        stem,
        ckpt,
        frame_rate=frame_rate,
        resize_factor=resize_factor,
        downsample=downsample,
        tune_frames=tune_frames,
        quick=quick,
        skip_tune=skip_tune,
        search_scale=search_scale,
        target_size_bias=target_size_bias,
    )
    return {"prepare": asdict(prep), "tracking": asdict(track)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run patched UDMT ST-Net / TrDiMP locally.")
    sub = p.add_subparsers(dest="command", required=True)

    patch = sub.add_parser("patch-clone", help="Patch a UDMT clone for pure PrRoIPool + bare imports.")
    patch.add_argument("--udmt-root", required=True, type=Path)
    patch.add_argument("--pure-prroi", type=Path)

    verify = sub.add_parser("verify", help="Verify patched ST-Net imports and optionally IoU-Net forward.")
    verify.add_argument("--udmt-root", required=True, type=Path)
    verify.add_argument("--run-iou-forward", action="store_true")
    verify.add_argument("--device", default="cpu")

    run = sub.add_parser("run-video", help="Prepare a project and run ST-Net tracking.")
    run.add_argument("--udmt-root", required=True, type=Path)
    run.add_argument("--video", required=True, type=Path)
    run.add_argument("--project", required=True, type=Path)
    run.add_argument("--stem", required=True)
    run.add_argument("--start-positions", required=True, type=Path)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--mask-dir", type=Path)
    run.add_argument("--init-mask", type=Path)
    run.add_argument("--repeat-init-mask", action="store_true")
    run.add_argument("--resize-factor", type=float, default=0.8)
    run.add_argument("--downsample", type=int, default=1)
    run.add_argument("--start-frame", type=int, default=0)
    run.add_argument("--frame-limit", type=int)
    run.add_argument("--frame-rate", type=float, default=166.0)
    run.add_argument("--tune-frames", type=int, default=4000)
    run.add_argument("--quick", action="store_true")
    run.add_argument("--skip-tune", action="store_true")
    run.add_argument("--search-scale", type=float, default=2.0)
    run.add_argument("--target-size-bias", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "patch-clone":
        report = patch_clone(args.udmt_root, args.pure_prroi)
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return
    if args.command == "verify":
        report = verify_runtime(
            args.udmt_root,
            run_iou_forward=args.run_iou_forward,
            device=args.device,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-video":
        report = run_video(
            args.udmt_root,
            args.video,
            args.project,
            args.stem,
            args.start_positions,
            checkpoint=args.checkpoint,
            mask_dir=args.mask_dir,
            init_mask=args.init_mask,
            repeat_init_mask=args.repeat_init_mask,
            resize_factor=args.resize_factor,
            downsample=args.downsample,
            start_frame=args.start_frame,
            frame_limit=args.frame_limit,
            frame_rate=args.frame_rate,
            tune_frames=args.tune_frames,
            quick=args.quick,
            skip_tune=args.skip_tune,
            search_scale=args.search_scale,
            target_size_bias=args.target_size_bias,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    raise SystemExit(f"Unknown command: {args.command}")


def _run_iou_forward(device: str) -> dict:
    import torch
    from ltr.models.bbreg.atom_iou_net import AtomIoUNet

    dev = torch.device(device)
    with torch.no_grad():
        net = AtomIoUNet().to(dev).eval()
        c3 = torch.randn(1, 128, 36, 36, device=dev)
        c4 = torch.randn(1, 256, 18, 18, device=dev)
        bb1 = torch.tensor([[[80.0, 80.0, 60.0, 60.0]]], device=dev)
        proposals = torch.tensor(
            [[[[80.0, 80.0, 60.0, 60.0],
               [70.0, 90.0, 50.0, 55.0],
               [100.0, 100.0, 40.0, 40.0]]]],
            device=dev,
        )
        out = net([c3, c4], [c3, c4], bb1, proposals)
    return {"shape": list(out.shape), "device": str(dev), "values": out.flatten().tolist()}


def _copy_if_changed(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and src.read_bytes() == dst.read_bytes():
        return False
    shutil.copy2(src, dst)
    return True


def _write_text_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _count_files_containing(root: Path, needle: str) -> int:
    n = 0
    for path in root.rglob("*.py"):
        if needle in path.read_text(encoding="utf-8", errors="ignore"):
            n += 1
    return n


def _newest_dir(parent: Path) -> Path:
    dirs = [p for p in parent.iterdir() if p.is_dir()]
    if not dirs:
        raise RuntimeError(f"No result directories under: {parent}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _read_last_metric(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise RuntimeError(f"No metrics in: {path}")
    return data[-1]


def _first_existing(header: list[str], names: tuple[str, ...]) -> int:
    for name in names:
        if name in header:
            return header.index(name)
    raise ValueError(f"Header must contain one of {names}: {header}")


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _missing_module(name: str) -> bool:
    if name in sys.modules:
        return False
    try:
        importlib.import_module(name)
        return False
    except Exception:
        return True


if __name__ == "__main__":
    main()
