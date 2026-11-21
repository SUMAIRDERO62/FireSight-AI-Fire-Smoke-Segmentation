from pathlib import Path

from firesight.training.pipeline import summarize_dataset


if __name__ == "__main__":
    summary = summarize_dataset(Path("data/processed"))
    print(
        "Training is not started in Phase 00: "
        f"{summary['image_count']} processed images detected."
    )

