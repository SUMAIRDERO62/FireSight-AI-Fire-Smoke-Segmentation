"""Generate a training-loss curve from training_history.json."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot FireSight AI training history.")
    parser.add_argument("--input", default="reports/training/training_history.json")
    parser.add_argument("--output", default="reports/training/training_curve.png")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Training history not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not epochs or len(epochs) != len(train_loss) or len(epochs) != len(val_loss):
        raise ValueError("Training history has missing or inconsistent values.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FireSight AI - U-Net Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Training curve saved: {output_path}")


if __name__ == "__main__":
    main()
