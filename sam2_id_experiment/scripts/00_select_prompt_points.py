from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.sam2_adapter.prompt_frame import (
    export_prompt_frame,
    resolve_prompt_frame_index,
)


WINDOW_NAME = "SAM2 prompt selector"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive point-prompt picker for SAM2 video propagation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame", type=int, default=None, help="Absolute video frame.")
    parser.add_argument("--local-frame", type=int, default=None, help="Frame offset from config start_frame.")
    parser.add_argument("--image", default=None, help="Use an existing prompt-frame PNG.")
    parser.add_argument("--output", default=None, help="Prompt YAML output path.")
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Open a video sequence viewer instead of a single exported PNG.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Frame step for sequence navigation with n/p or arrow keys.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Write prompts back into the config and switch sam2.runner to sam2_video_predictor.",
    )
    parser.add_argument("--display-scale", type=float, default=0.75)
    return parser


def _draw_prompt_overlay(
    image,
    prompts: list[dict],
    *,
    current_obj_id: int,
    scale: float,
    local_frame: int,
    absolute_frame: int | None = None,
):
    import cv2
    import numpy as np

    canvas = image.copy()
    colors = [
        (40, 220, 70),
        (40, 140, 255),
        (230, 80, 230),
        (255, 200, 60),
        (80, 220, 220),
        (220, 120, 60),
    ]
    frame_prompts = [p for p in prompts if int(p.get("frame", -1)) == int(local_frame)]
    for prompt in frame_prompts:
        obj_id = int(prompt["obj_id"])
        color = colors[obj_id % len(colors)]
        for point, label in zip(prompt["points"], prompt["labels"]):
            x, y = int(point[0]), int(point[1])
            if int(label) == 1:
                cv2.circle(canvas, (x, y), 9, color, 2, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)
            else:
                cv2.drawMarker(
                    canvas,
                    (x, y),
                    color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=18,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                f"{obj_id}",
                (x + 10, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    header = (
        f"local {local_frame}"
        + (f" abs {absolute_frame}" if absolute_frame is not None else "")
        + f" | obj {current_obj_id} | left=+ right=- | "
        "0-9 obj | n/p step | ,/. frame | u undo | s save | q quit"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        header,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if scale != 1.0:
        w = max(1, int(round(canvas.shape[1] * scale)))
        h = max(1, int(round(canvas.shape[0] * scale)))
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(canvas)


def _add_point(prompts: list[dict], *, obj_id: int, frame: int, x: int, y: int, label: int) -> None:
    for prompt in prompts:
        if int(prompt["obj_id"]) == int(obj_id) and int(prompt["frame"]) == int(frame):
            prompt["points"].append([int(x), int(y)])
            prompt["labels"].append(int(label))
            return
    prompts.append({
        "obj_id": int(obj_id),
        "frame": int(frame),
        "points": [[int(x), int(y)]],
        "labels": [int(label)],
    })


def _pop_last_point(prompts: list[dict], *, frame: int | None = None) -> None:
    for prompt in reversed(prompts):
        if frame is not None and int(prompt.get("frame", -1)) != int(frame):
            continue
        if prompt["points"]:
            prompt["points"].pop()
            prompt["labels"].pop()
            if not prompt["points"]:
                prompts.remove(prompt)
            return


def _write_prompts(
    *,
    prompts: list[dict],
    output_path: Path,
    config_path: Path,
    update_config: bool,
) -> None:
    payload = {"sam2": {"runner": "sam2_video_predictor", "prompt_mode": "manual_initial_frame", "prompts": prompts}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    if update_config:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg_data = yaml.safe_load(fh) or {}
        sam2 = dict(cfg_data.get("sam2") or {})
        sam2["runner"] = "sam2_video_predictor"
        sam2["prompt_mode"] = "manual_initial_frame"
        sam2["prompts"] = prompts
        cfg_data["sam2"] = sam2
        config_path.write_text(yaml.safe_dump(cfg_data, sort_keys=False), encoding="utf-8")


def _load_existing_prompts(config_path: Path) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg_data = yaml.safe_load(fh) or {}
    prompts = list((cfg_data.get("sam2") or {}).get("prompts") or [])
    out: list[dict] = []
    for prompt in prompts:
        points = [[int(p[0]), int(p[1])] for p in prompt.get("points", [])]
        labels = [int(v) for v in prompt.get("labels", [1] * len(points))]
        if not points:
            continue
        out.append({
            "obj_id": int(prompt.get("obj_id", 0)),
            "frame": int(prompt.get("frame", 0)),
            "points": points,
            "labels": labels,
        })
    return out


def run_prompt_selector(
    *,
    image_path: Path,
    local_frame: int,
    output_path: Path,
    config_path: Path,
    update_config: bool,
    display_scale: float,
) -> list[dict]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read prompt image: {image_path}")

    prompts: list[dict] = _load_existing_prompts(config_path)
    state = {"obj_id": 0, "dirty": True}
    scale = float(display_scale)
    if scale <= 0:
        scale = 1.0

    def on_mouse(event, x, y, _flags, _param):
        if event not in {cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN}:
            return
        img_x = int(round(x / scale))
        img_y = int(round(y / scale))
        img_x = max(0, min(image.shape[1] - 1, img_x))
        img_y = max(0, min(image.shape[0] - 1, img_y))
        label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        _add_point(
            prompts,
            obj_id=int(state["obj_id"]),
            frame=local_frame,
            x=img_x,
            y=img_y,
            label=label,
        )
        state["dirty"] = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            if state["dirty"]:
                canvas = _draw_prompt_overlay(
                    image,
                    prompts,
                    current_obj_id=int(state["obj_id"]),
                    scale=scale,
                    local_frame=local_frame,
                )
                cv2.imshow(WINDOW_NAME, canvas)
                state["dirty"] = False
            key = cv2.waitKey(50) & 0xFF
            if key == 255:
                continue
            if ord("0") <= key <= ord("9"):
                state["obj_id"] = key - ord("0")
                state["dirty"] = True
            elif key in {ord("u"), 8}:
                _pop_last_point(prompts, frame=local_frame)
                state["dirty"] = True
            elif key == ord("s"):
                _write_prompts(
                    prompts=prompts,
                    output_path=output_path,
                    config_path=config_path,
                    update_config=update_config,
                )
                print(f"Wrote {output_path}")
                if update_config:
                    print(f"Updated {config_path}")
            elif key in {ord("q"), 27}:
                break
    finally:
        cv2.destroyWindow(WINDOW_NAME)

    return prompts


def _read_video_frame(cap, frame_index: int):
    cap.set(1, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def run_sequence_prompt_selector(
    *,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    initial_local_frame: int,
    output_path: Path,
    config_path: Path,
    update_config: bool,
    display_scale: float,
    step: int,
) -> list[dict]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    prompts: list[dict] = _load_existing_prompts(config_path)
    scale = float(display_scale)
    if scale <= 0:
        scale = 1.0
    max_local_frame = int(end_frame - start_frame)
    local_frame = max(0, min(max_local_frame, int(initial_local_frame)))
    state = {"obj_id": 0, "dirty": True, "frame": local_frame}

    def current_image():
        return _read_video_frame(cap, start_frame + int(state["frame"]))

    def on_mouse(event, x, y, _flags, _param):
        if event not in {cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN}:
            return
        image_shape = state.get("image_shape")
        if image_shape is None:
            return
        height, width = image_shape
        img_x = int(round(x / scale))
        img_y = int(round(y / scale))
        img_x = max(0, min(width - 1, img_x))
        img_y = max(0, min(height - 1, img_y))
        label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        _add_point(
            prompts,
            obj_id=int(state["obj_id"]),
            frame=int(state["frame"]),
            x=img_x,
            y=img_y,
            label=label,
        )
        state["dirty"] = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        image = current_image()
        state["image_shape"] = image.shape[:2]
        while True:
            if state["dirty"]:
                image = current_image()
                state["image_shape"] = image.shape[:2]
                canvas = _draw_prompt_overlay(
                    image,
                    prompts,
                    current_obj_id=int(state["obj_id"]),
                    scale=scale,
                    local_frame=int(state["frame"]),
                    absolute_frame=start_frame + int(state["frame"]),
                )
                cv2.imshow(WINDOW_NAME, canvas)
                state["dirty"] = False

            key = cv2.waitKeyEx(50)
            if key < 0:
                continue
            key_code = key & 0xFF
            if ord("0") <= key_code <= ord("9"):
                state["obj_id"] = key_code - ord("0")
                state["dirty"] = True
            elif key_code == ord("n"):
                state["frame"] = min(max_local_frame, int(state["frame"]) + max(1, step))
                state["dirty"] = True
            elif key_code == ord("p"):
                state["frame"] = max(0, int(state["frame"]) - max(1, step))
                state["dirty"] = True
            elif key_code == ord(".") or key in {83, 2555904}:
                state["frame"] = min(max_local_frame, int(state["frame"]) + 1)
                state["dirty"] = True
            elif key_code == ord(",") or key in {81, 2424832}:
                state["frame"] = max(0, int(state["frame"]) - 1)
                state["dirty"] = True
            elif key_code in {ord("u"), 8}:
                _pop_last_point(prompts, frame=int(state["frame"]))
                state["dirty"] = True
            elif key_code == ord("s"):
                _write_prompts(
                    prompts=prompts,
                    output_path=output_path,
                    config_path=config_path,
                    update_config=update_config,
                )
                print(f"Wrote {output_path}")
                if update_config:
                    print(f"Updated {config_path}")
            elif key_code in {ord("q"), 27}:
                break
    finally:
        cap.release()
        cv2.destroyWindow(WINDOW_NAME)

    return prompts


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    start_frame = int(cfg.raw.get("start_frame") or 0)
    end_frame = int(cfg.raw.get("end_frame") or start_frame)
    frame_index = resolve_prompt_frame_index(
        start_frame=start_frame,
        frame=args.frame,
        local_frame=args.local_frame,
    )
    local_frame = frame_index - start_frame

    output_path = Path(args.output) if args.output else out_dir / f"sam2_prompts.yaml"
    if args.sequence:
        prompts = run_sequence_prompt_selector(
            video_path=cfg.video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            initial_local_frame=local_frame,
            output_path=output_path,
            config_path=Path(args.config),
            update_config=bool(args.update_config),
            display_scale=float(args.display_scale),
            step=int(args.step),
        )
        print(json.dumps({"n_prompt_objects": len(prompts), "prompt_yaml": str(output_path)}, indent=2))
        return

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = out_dir / f"prompt_frame_abs{frame_index}_local{local_frame}.png"
        if not image_path.exists():
            export_prompt_frame(
                video_path=cfg.video_path,
                output_path=image_path,
                frame_index=frame_index,
                start_frame=start_frame,
            )

    prompts = run_prompt_selector(
        image_path=image_path,
        local_frame=local_frame,
        output_path=output_path,
        config_path=Path(args.config),
        update_config=bool(args.update_config),
        display_scale=float(args.display_scale),
    )
    print(json.dumps({"n_prompt_objects": len(prompts), "prompt_yaml": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
