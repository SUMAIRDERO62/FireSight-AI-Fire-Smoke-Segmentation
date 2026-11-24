from pathlib import Path
from PIL import Image, ImageDraw
import math

ROOT = Path("data/processed/yolo11_seg")
OUTPUT = Path("reports/figures/phase04_visual_audit/contact_sheets")
OUTPUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    0: (255, 60, 30, 100),
    1: (30, 150, 255, 100),
}
NAMES = {0: "Fire", 1: "Smoke"}

CELL_W, CELL_H = 360, 280
THUMB_W, THUMB_H = 340, 225
COLS, ROWS = 3, 4
PER_PAGE = COLS * ROWS


def make_preview(image_path, label_path):
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    counts = {"Fire": 0, "Smoke": 0}

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue

        class_id = int(parts[0])
        if class_id not in COLORS:
            continue

        coords = [float(v) for v in parts[1:]]
        points = [
            (coords[i] * image.width, coords[i + 1] * image.height)
            for i in range(0, len(coords), 2)
        ]

        draw.polygon(
            points,
            fill=COLORS[class_id],
            outline=COLORS[class_id][:3] + (255,),
            width=3,
        )
        counts[NAMES[class_id]] += 1

    return Image.alpha_composite(image, overlay).convert("RGB"), counts


for split in ["train", "val", "test"]:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    records = []

    for image_path in sorted(image_dir.glob("*")):
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        if not label_path.read_text(encoding="utf-8").strip():
            continue

        preview, counts = make_preview(image_path, label_path)
        records.append((image_path.name, preview, counts))

    total_pages = math.ceil(len(records) / PER_PAGE)

    for page in range(total_pages):
        page_records = records[page * PER_PAGE : (page + 1) * PER_PAGE]

        sheet = Image.new(
            "RGB",
            (COLS * CELL_W, ROWS * CELL_H),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for index, (filename, preview, counts) in enumerate(page_records):
            col = index % COLS
            row = index // COLS

            x = col * CELL_W
            y = row * CELL_H

            preview.thumbnail((THUMB_W, THUMB_H))
            px = x + (CELL_W - preview.width) // 2
            py = y + 5

            sheet.paste(preview, (px, py))

            text = f"{filename[:39]}\nFire: {counts['Fire']} | Smoke: {counts['Smoke']}"
            draw.text((x + 8, y + 235), text, fill="black")

        output_file = OUTPUT / f"{split}_page_{page + 1:02d}.jpg"
        sheet.save(output_file, quality=92)

    print(
        f"{split.upper()}: {len(records)} annotated images, "
        f"{total_pages} contact sheets"
    )

print("\nCONTACT SHEETS GENERATED")
print("Location:", OUTPUT.resolve())
