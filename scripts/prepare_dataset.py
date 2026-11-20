from pathlib import Path

from firesight.data.dataset import count_images


if __name__ == "__main__":
    data_dir = Path("data/raw")
    print(f"Found {count_images(data_dir)} images in {data_dir}.")

