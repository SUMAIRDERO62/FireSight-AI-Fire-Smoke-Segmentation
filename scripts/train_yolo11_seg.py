"""Command-line entry point for FireSight YOLO11-seg training."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firesight.training.yolo_trainer import YOLOSegTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train FireSight YOLO11 fire/smoke segmentation model."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/yolo11_seg.yaml"),
        help="Training YAML configuration.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu or CUDA device.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    trainer = YOLOSegTrainer(args.config)
    results = trainer.train(
        epochs=args.epochs,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
    )

    print("\nTraining completed.")
    print(f"Results: {results}")
    return results


if __name__ == "__main__":
    main()
