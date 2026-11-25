"""Evaluate U-Net and YOLO11-seg on the identical original COCO test split.

Pixel metrics are computed independently for fire and smoke. YOLO instance masks
are unioned within each class, never across classes, so class overlap is retained.
YOLO native mask mAP is additionally obtained from Ultralytics validation.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from firesight.models.unet import UNet


def load_gt(base, stem, size):
    fire = np.asarray(Image.open(base / "masks/fire" / f"{stem}.png").convert("L")) > 0
    smoke = np.asarray(Image.open(base / "masks/smoke" / f"{stem}.png").convert("L")) > 0
    h, w = size
    fire = np.asarray(Image.fromarray(fire.astype("uint8") * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
    smoke = np.asarray(Image.fromarray(smoke.astype("uint8") * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
    return np.stack((fire, smoke), axis=0)


def metrics_for_counts(counts):
    rows = []
    for class_name, (tp, fp, fn) in zip(("fire", "smoke"), counts):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        d = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        i = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        rows.append({"class": class_name, "tp": int(tp), "fp": int(fp), "fn": int(fn),
                     "precision": p, "recall": r, "dice": d, "iou": i})
    return {
        "per_class": rows,
        "macro_precision": float(np.mean([x["precision"] for x in rows])),
        "macro_recall": float(np.mean([x["recall"] for x in rows])),
        "macro_dice": float(np.mean([x["dice"] for x in rows])),
        "macro_iou": float(np.mean([x["iou"] for x in rows])),
    }


def add_counts(counts, pred, target):
    for c in range(2):
        p, t = pred[c].astype(bool), target[c].astype(bool)
        counts[c][0] += int(np.logical_and(p, t).sum())
        counts[c][1] += int(np.logical_and(p, ~t).sum())
        counts[c][2] += int(np.logical_and(~p, t).sum())


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet", required=True)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--dataset", default="data/processed/phase05_fair")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="reports/results/phase05_comparison")
    args = parser.parse_args()

    from ultralytics import YOLO
    device = torch.device(args.device)
    dataset = Path(args.dataset).resolve()
    test_images = sorted((dataset / "unet/test/images").glob("*"))
    test_images = [p for p in test_images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(test_images) != 20:
        raise RuntimeError(f"Expected exactly 20 original COCO test images; found {len(test_images)}")

    unet = UNet(num_classes=2, in_channels=3, base_channels=16).to(device)
    state = torch.load(args.unet, map_location=device, weights_only=True)
    unet.load_state_dict(state.get("model_state_dict", state))
    unet.eval()
    yolo = YOLO(args.yolo)

    counts_unet = [[0, 0, 0] for _ in range(2)]
    counts_yolo = [[0, 0, 0] for _ in range(2)]
    times = {"unet_seconds_per_image": [], "yolo_seconds_per_image": []}
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for image_path in test_images:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
        gt = load_gt(dataset / "unet/test", image_path.stem, (args.imgsz, args.imgsz))

        arr = np.asarray(rgb.resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        x = (x - mean) / std
        sync(device); start = time.perf_counter()
        with torch.no_grad():
            logits = unet(x)
            pred_u = torch.sigmoid(logits)[0].cpu().numpy() >= args.threshold
        sync(device)
        times["unet_seconds_per_image"].append(time.perf_counter() - start)
        add_counts(counts_unet, pred_u, gt)

        sync(device); start = time.perf_counter()
        result = yolo.predict(source=str(image_path), imgsz=args.yolo_imgsz,
                              device=args.device, conf=0.25, verbose=False)[0]
        sync(device)
        times["yolo_seconds_per_image"].append(time.perf_counter() - start)
        pred_y = np.zeros((2, args.imgsz, args.imgsz), dtype=bool)
        if result.masks is not None and result.boxes is not None:
            masks = F.interpolate(result.masks.data.float().unsqueeze(1),
                                  size=(args.imgsz, args.imgsz), mode="bilinear",
                                  align_corners=False)[:, 0] > 0.5
            class_ids = result.boxes.cls.int().cpu().tolist()
            for mask, cls in zip(masks.cpu().numpy(), class_ids):
                if cls in (0, 1):
                    pred_y[cls] |= mask
        add_counts(counts_yolo, pred_y, gt)

    # Native mask AP on the same test split; Ultralytics reads the same fair YAML.
    yaml_path = dataset / "yolo/data.yaml"
    native = yolo.val(data=str(yaml_path), split="test", imgsz=args.yolo_imgsz,
                      device=args.device, plots=True, verbose=False)
    native_dict = native.results_dict
    native_mask = {k: (float(native_dict[k]) if native_dict.get(k) is not None else None)
                   for k in ("metrics/precision(M)", "metrics/recall(M)",
                             "metrics/mAP50(M)", "metrics/mAP50-95(M)")}

    reports = {
        "protocol": {
            "dataset": "Roboflow fire/smoke COCO polygon source",
            "test_images": len(test_images),
            "test_split": "original COCO test",
            "classes": ["fire", "smoke"],
            "overlap_policy": "independent binary channels; overlapping fire/smoke pixels retained",
            "threshold": args.threshold,
            "unet_input_size": args.imgsz,
            "yolo_input_size": args.yolo_imgsz,
        },
        "unet": {
            "pixel_metrics": metrics_for_counts(counts_unet),
            "mean_inference_seconds_per_image": float(np.mean(times["unet_seconds_per_image"])),
            "checkpoint_bytes": Path(args.unet).stat().st_size,
        },
        "yolo11_seg": {
            "pixel_metrics": metrics_for_counts(counts_yolo),
            "native_mask_metrics": native_mask,
            "mean_inference_seconds_per_image": float(np.mean(times["yolo_seconds_per_image"])),
            "checkpoint_bytes": Path(args.yolo).stat().st_size,
        },
        "timing_note": "Inference timings are measured in this runtime and are hardware-dependent.",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    lines = [
        "# FireSight AI — Phase 05 Model Comparison", "",
        f"Test images: **{len(test_images)}** (same original COCO test split)", "",
        "| Metric | U-Net | YOLO11-seg |",
        "|---|---:|---:|",
    ]
    for key, label in (("macro_dice", "Macro Dice"), ("macro_iou", "Macro IoU"),
                       ("macro_precision", "Macro Precision"), ("macro_recall", "Macro Recall")):
        lines.append(f"| {label} | {reports['unet']['pixel_metrics'][key]:.5f} | {reports['yolo11_seg']['pixel_metrics'][key]:.5f} |")
    lines += [
        f"| Mean inference (s/image) | {reports['unet']['mean_inference_seconds_per_image']:.5f} | {reports['yolo11_seg']['mean_inference_seconds_per_image']:.5f} |",
        f"| Checkpoint size (MB) | {reports['unet']['checkpoint_bytes']/1e6:.3f} | {reports['yolo11_seg']['checkpoint_bytes']/1e6:.3f} |",
        "", "## YOLO native instance-mask metrics", "",
        "```json", json.dumps(native_mask, indent=2), "```", "",
        "## Per-class pixel metrics", "",
        "### U-Net", "```json", json.dumps(reports["unet"]["pixel_metrics"]["per_class"], indent=2), "```",
        "### YOLO11-seg", "```json", json.dumps(reports["yolo11_seg"]["pixel_metrics"]["per_class"], indent=2), "```",
        "", "Results are valid only for the checkpoints and runtime recorded above. Do not tune thresholds on this test set.",
    ]
    (output / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"\nSaved: {output / 'comparison.json'}")
    print(f"Saved: {output / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
