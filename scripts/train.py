from pathlib import Path

from firesight.training.pipeline import summarize_dataset


if __name__ == "__main__":
    summary = summarize_dataset(Path("data/processed"))
    print(f"Training placeholder: {summary['image_count']} processed images.")

