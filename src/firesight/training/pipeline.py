from pathlib import Path

from firesight.data.dataset import count_images


def summarize_dataset(data_dir: str | Path) -> dict[str, int]:
    """Return a minimal dataset summary for training scripts."""
    return {"image_count": count_images(data_dir)}

