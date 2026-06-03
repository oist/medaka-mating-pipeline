from __future__ import annotations

from pathlib import Path


def train_detectron2_medaka(
    *,
    train_json: str | Path,
    train_images: str | Path,
    output_dir: str | Path,
    val_json: str | Path | None = None,
    val_images: str | Path | None = None,
    max_iter: int = 1000,
    base_lr: float = 0.00025,
    ims_per_batch: int = 2,
    num_workers: int = 2,
    score_thresh: float = 0.5,
) -> dict:
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed in this environment. Install Detectron2 "
            "for your CUDA/PyTorch platform, then rerun this script."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_coco_instances("medaka_sam2_train", {}, str(train_json), str(train_images))
    datasets = ("medaka_sam2_train",)
    test_datasets: tuple[str, ...] = ()
    if val_json is not None and val_images is not None:
        register_coco_instances("medaka_sam2_val", {}, str(val_json), str(val_images))
        test_datasets = ("medaka_sam2_val",)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = datasets
    cfg.DATASETS.TEST = test_datasets
    cfg.DATALOADER.NUM_WORKERS = int(num_workers)
    cfg.SOLVER.IMS_PER_BATCH = int(ims_per_batch)
    cfg.SOLVER.BASE_LR = float(base_lr)
    cfg.SOLVER.MAX_ITER = int(max_iter)
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.OUTPUT_DIR = str(output_dir)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    config_path = output_dir / "detectron2_config.yaml"
    config_path.write_text(cfg.dump(), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "config": str(config_path),
        "model": str(output_dir / "model_final.pth"),
    }
