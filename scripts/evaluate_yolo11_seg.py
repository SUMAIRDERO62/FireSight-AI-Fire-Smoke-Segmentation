"""Evaluate a trained YOLO11 segmentation checkpoint."""

import argparse
import json

from firesight.evaluation.yolo_evaluator import YOLOSegEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        default="data/processed/yolo11_seg_split_safe/data.yaml",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--output",
        default="runs/evaluation/yolo11_seg",
    )
    args = parser.parse_args()

    evaluator = YOLOSegEvaluator(
        checkpoint=args.checkpoint,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
    )
    report = evaluator.evaluate(args.output)

    print("\n========== YOLO SEGMENTATION EVALUATION ==========")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
