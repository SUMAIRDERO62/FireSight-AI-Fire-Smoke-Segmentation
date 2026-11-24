from pathlib import Path
from PIL import Image, ImageDraw
import json

ROOT = Path("data/processed/yolo11_seg")
OUTPUT = Path("reports/figures/phase04_visual_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "fire", 1: "smoke"}
CLASS_COLORS = {
    0: (255, 60, 30),
    1: (30, 150, 255),
}

def visualize(image_path, label_path, output_path):
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    polygon_count = 0
    class_counts = {"fire": 0, "smoke": 0}

    if label_path.exists():
        lines = label_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            class_id = int(parts[0])
            if class_id not in CLASS_NAMES:
                continue

            values = [float(x) for x in parts[1:]]
            points = [
                (values[i] * image.width, values[i + 1] * image.height)
                for i in range(0, len(values), 2)
            ]

            color = CLASS_COLORS[class_id]
            draw.polygon(points, fill=(*color, 65), outline=(*color, 255), width=3)

            polygon_count += 1
            class_counts[CLASS_NAMES[class_id]] += 1

    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(output_path, quality=95)

    return polygon_count, class_counts


summary = []

for split in ["train", "val", "test"]:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    candidates = sorted(image_dir.glob("*"))
    selected = None

    # Prefer an image that has annotations.
    for image_path in candidates:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            selected = image_path
            break

    if selected is None:
        print(f"{split}: No annotated image found")
        continue

    label_path = label_dir / f"{selected.stem}.txt"
    output_path = OUTPUT / f"{split}_annotation_preview.jpg"

    count, classes = visualize(selected, label_path, output_path)

    summary.append({
        "split": split,
        "image": selected.name,
        "polygons": count,
        "classes": classes,
        "output": str(output_path),
    })

    print(f"\n{split.upper()} PREVIEW")
    print("Image:", selected.name)
    print("Polygons:", count)
    print("Class counts:", classes)
    print("Saved:", output_path)

# Also generate a preview of the intentionally empty test label.
empty_label = ROOT / "labels" / "test" / "flare_0089_jpg.rf.4b52d9620ee0c678eda0fae322f2f8d6.txt"
empty_image = ROOT / "images" / "test" / "flare_0089_jpg.rf.4b52d9620ee0c678eda0fae322f2f8d6.jpg"

if empty_image.exists() and empty_label.exists():
    output_path = OUTPUT / "test_empty_annotation_preview.jpg"
    count, classes = visualize(empty_image, empty_label, output_path)
    print("\nEMPTY TEST IMAGE")
    print("Image:", empty_image.name)
    print("Polygons:", count)
    print("Saved:", output_path)

(OUTPUT / "visual_audit_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nVISUAL AUDIT PREVIEWS GENERATED.")
print("Output folder:", OUTPUT.resolve())
