# Phase 03 — U-Net Model Development

## Implemented Components

- U-Net semantic segmentation architecture
- Segmentation dataset loader
- Paired image and mask transformations
- Dice and Cross-Entropy + Dice losses
- Training and validation engine
- Best/latest checkpoint management
- Training history in JSON and CSV
- Early stopping and learning-rate scheduler support
- Confusion matrix and segmentation metrics
- Accuracy, Dice, IoU, Precision and Recall
- Model evaluation script
- Single-image prediction script
- Grayscale prediction masks
- Colored prediction overlays
- Training-loss curve visualization

## Commands

### Run training

```powershell
python scripts/train_unet.py --help
```

### Evaluate a trained model

```powershell
python scripts/evaluate_unet.py --help
```

### Predict one image

```powershell
python scripts/predict_unet.py --help
```

### Plot training history

```powershell
python scripts/plot_training_history.py --help
```

## Important Dataset Requirements

Real semantic segmentation requires pixel-level ground-truth masks.

Detection bounding boxes and Pascal VOC bounding-box annotations are not semantic segmentation masks.

Before training:

1. Obtain genuine segmentation masks.
2. Verify the class-ID mapping.
3. Configure num_classes in configs/unet.yaml.
4. Configure train, validation and test image/mask paths.
5. Train and evaluate using the appropriate dataset splits.

No fabricated masks or performance metrics should be used.

## Hardware

CPU is supported for development and testing.

GPU training is recommended for larger image sizes and datasets.

## Current Status

Software implementation and automated tests must be verified before marking this phase complete.

Real model training and performance evaluation remain dependent on verified segmentation annotations.
