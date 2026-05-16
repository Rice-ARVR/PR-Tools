"""
Fine-tune YOLOv26 on an OWL-SUITS Auto Drive dataset.

Classes: Boulder, Crater, Lunar Home Base, Lunar Terrain Vehicle

Outputs
-------
  runs/<run_name>/stage1/weights/best.pt   <- stage-1 checkpoint
  runs/<run_name>/stage2/weights/best.pt   <- best checkpoint (use this for inference)
  runs/<run_name>/stage2/weights/last.pt   <- final epoch checkpoint
  runs/<run_name>/stage2/                  <- training curves, confusion matrix, PR curve
  runs/<run_name>/val/                     <- validation-split metrics
  runs/<run_name>/test/                    <- test-split metrics

Usage
-----
  # Direct invocation trains a single default config:
  uv run python train_yolo.py

  # Or import and call train() with custom dataset/model:
  from train_yolo import train
  train(dataset_dir=..., base_model="yolo26n.pt", run_name="yolo26n_v1")
"""

from pathlib import Path
from typing import Union

import torch
import yaml
from ultralytics import YOLO

HERE = Path(__file__).parent


def _check_device() -> None:
    if torch.cuda.is_available():
        print(f"CUDA available — training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available — training on CPU (this will be slow)")


def _build_data_yaml(dataset_dir: Path) -> Path:
    """Roboflow's data.yaml uses ../train/images paths that only resolve when
    the yaml sits one level above the splits. Patch it to absolute paths."""
    src = dataset_dir / "data.yaml"
    with open(src) as f:
        cfg = yaml.safe_load(f)

    cfg["path"]  = str(dataset_dir.resolve())
    cfg["train"] = "train/images"
    cfg["val"]   = "valid/images"
    cfg["test"]  = "test/images"

    out = dataset_dir / "data_fixed.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def train(
    dataset_dir: Union[str, Path],
    base_model: str,
    run_name: str,
    runs_root: Union[str, Path, None] = None,
    stage1_epochs: int = 20,
    stage2_epochs: int = 80,
    freeze: int = 10,
    img_size: int = 512,
    batch: int = 8,
    workers: int = 2,
    device: Union[int, str] = 0,
) -> Path:
    """Two-stage fine-tune: freeze backbone then unfreeze all.

    All outputs are isolated under ``{runs_root}/{run_name}/``, so multiple
    runs with different datasets or base models never collide.

    Returns the path to the best stage-2 checkpoint.
    """
    dataset_dir = Path(dataset_dir)
    runs_root   = Path(runs_root) if runs_root is not None else HERE / "runs"
    run_root    = runs_root / run_name

    _check_device()
    data_yaml = _build_data_yaml(dataset_dir)
    print(f"Dataset : {dataset_dir}")
    print(f"Config  : {data_yaml}")
    print(f"Model   : {base_model}")
    print(f"Run dir : {run_root}\n")

    # ── Stage 1: freeze backbone, train head and neck ─────────────────────────
    print("=" * 60)
    print(f"Stage 1: frozen backbone, {stage1_epochs} epochs")
    print("=" * 60)
    model = YOLO(base_model)
    model.train(
        data     = str(data_yaml),
        epochs   = stage1_epochs,
        freeze   = freeze,
        imgsz    = img_size,
        batch    = batch,
        workers  = workers,
        device   = device,
        project  = str(run_root),
        name     = "stage1",
        exist_ok = True,
        plots    = True,
        save     = True,
    )

    # ── Stage 2: unfreeze all, fine-tune with lower lr ────────────────────────
    stage1_best = run_root / "stage1" / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"Stage 2: all layers unfrozen, {stage2_epochs} epochs, lr0=0.001")
    print("=" * 60)
    model = YOLO(str(stage1_best))
    model.train(
        data     = str(data_yaml),
        epochs   = stage2_epochs,
        lr0      = 0.001,
        imgsz    = img_size,
        batch    = batch,
        workers  = workers,
        device   = device,
        project  = str(run_root),
        name     = "stage2",
        exist_ok = True,
        plots    = True,
        save     = True,
    )

    best_pt = run_root / "stage2" / "weights" / "best.pt"
    sep = "=" * 60
    print(f"\n{sep}")
    print("Training complete.")
    print(f"Best weights : {best_pt}")
    print(f"Results dir  : {run_root / 'stage2'}")

    val_results = model.val(
        data    = str(data_yaml),
        device  = device,
        project = str(run_root),
        name    = "val",
        exist_ok= True,
    )
    metrics = val_results.results_dict
    map50   = metrics.get("metrics/mAP50(B)",    "n/a")
    map5095 = metrics.get("metrics/mAP50-95(B)", "n/a")
    if isinstance(map50,   float): map50   = f"{map50:.4f}"
    if isinstance(map5095, float): map5095 = f"{map5095:.4f}"
    print(f"Val mAP@50       : {map50}")
    print(f"Val mAP@50-95    : {map5095}")

    # ── Evaluate on held-out test split ───────────────────────────────────────
    print(f"\nEvaluating on test split …")
    test_results = model.val(
        data    = str(data_yaml),
        split   = "test",
        device  = device,
        plots   = True,
        project = str(run_root),
        name    = "test",
        exist_ok= True,
    )

    tm    = test_results.results_dict
    t50   = tm.get("metrics/mAP50(B)",    "n/a")
    t5095 = tm.get("metrics/mAP50-95(B)", "n/a")
    if isinstance(t50,   float): t50   = f"{t50:.4f}"
    if isinstance(t5095, float): t5095 = f"{t5095:.4f}"
    print(f"Test mAP@50      : {t50}")
    print(f"Test mAP@50-95   : {t5095}")
    print(sep)
    print(f"\nTo run inference with the trained model:")
    print(f"  model = YOLO('{best_pt}')")
    print(f"  results = model('your_image.jpg')\n")
    return best_pt


if __name__ == "__main__":
    # Default single-run config: yolo26n on dataset version_1.
    # For batch sweeps over multiple datasets/models, use train_all.py.
    train(
        dataset_dir = HERE / "data" / "version_1",
        base_model  = "yolo26n.pt",
        run_name    = "yolo26n_v1",
    )
