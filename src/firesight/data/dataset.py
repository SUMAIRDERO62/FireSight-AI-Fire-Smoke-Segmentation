from pathlib import Path
from typing import Iterator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: str | Path) -> Iterator[Path]:
    """Yield image files below root in stable order."""
    root = Path(root)
    if not root.exists():
        return
    yield from sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def count_images(root: str | Path) -> int:
    return sum(1 for _ in iter_images(root))
