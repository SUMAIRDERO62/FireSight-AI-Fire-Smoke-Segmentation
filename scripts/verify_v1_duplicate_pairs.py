import csv
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import statistics

csv_path = Path("reports/results/phase04_near_duplicate_pairs.csv")
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Select only within-v1 train-vs-validation/test pairs.
pairs = [
    r for r in rows
    if r["dataset_a"] == "v1"
    and r["dataset_b"] == "v1"
    and r["split_a"] == "train"
    and r["split_b"] in ("valid", "test")
]

if len(pairs) != 10:
    raise SystemExit(f"Expected 10 pairs, found {len(pairs)}. Stopping safely.")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def image_metrics(path_a, path_b):
    with Image.open(path_a) as a, Image.open(path_b) as b:
        size_a, size_b = a.size, b.size
        a = ImageOps.fit(a.convert("RGB"), (256, 256))
        b = ImageOps.fit(b.convert("RGB"), (256, 256))

        import PIL.ImageChops
        diff = PIL.ImageChops.difference(a, b)
        mean_diff = statistics.mean(diff.convert("L").getdata())

    return size_a, size_b, round(mean_diff, 3)

W = 1120
PANEL_W = 550
IMAGE_H = 300
ROW_H = 370
HEADER_H = 65

sheet = Image.new("RGB", (W, HEADER_H + ROW_H * len(pairs)), "white")
draw = ImageDraw.Draw(sheet)
report = []

for i, pair in enumerate(pairs, 1):
    y = HEADER_H + (i - 1) * ROW_H
    path_a = Path(pair["path_a"])
    path_b = Path(pair["path_b"])

    print(f"Checking pair {i}/10...")

    exists_a, exists_b = path_a.is_file(), path_b.is_file()
    if not exists_a or not exists_b:
        raise FileNotFoundError(
            f"Pair {i}: image missing: A={exists_a}, B={exists_b}"
        )

    hash_a, hash_b = sha256(path_a), sha256(path_b)
    size_a, size_b, pixel_diff = image_metrics(path_a, path_b)

    draw.text((10, y + 5),
              f"PAIR {i} | pHash distance: {pair['distance']} | "
              f"{pair['split_a']} vs {pair['split_b']}",
              fill="black")

    for x, path, title in [
        (10, path_a, "A - TRAIN"),
        (570, path_b, f"B - {pair['split_b'].upper()}")
    ]:
        draw.text((x, y + 27), title, fill="black")
        with Image.open(path) as im:
            im = ImageOps.contain(im.convert("RGB"), (PANEL_W - 20, IMAGE_H))
            px = x + (PANEL_W - im.width) // 2
            py = y + 50 + (IMAGE_H - im.height) // 2
            sheet.paste(im, (px, py))

    draw.text((10, y + 355), path_a.name[:85], fill="black")
    draw.text((570, y + 355), path_b.name[:85], fill="black")

    report.append({
        "pair": i,
        "scope": pair["scope"],
        "phash_distance": pair["distance"],
        "file_a": path_a.name,
        "file_b": path_b.name,
        "sha256_identical": hash_a == hash_b,
        "dimensions_a": str(size_a),
        "dimensions_b": str(size_b),
        "mean_pixel_difference_256": pixel_diff,
        "status": "VISUAL_REVIEW_REQUIRED"
    })

sheet_path = output_dir / "v1_cross_split_duplicate_review.jpg"
sheet.save(sheet_path, quality=95)

report_path = Path("reports/results/v1_cross_split_verification.csv")
with report_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=report[0].keys())
    writer.writeheader()
    writer.writerows(report)

print("\n10-PAIR COMPARISON COMPLETE")
print("Contact sheet:", sheet_path.resolve())
print("Metrics report:", report_path.resolve())
print("\nPAIR | PHASH | SHA IDENTICAL | PIXEL DIFFERENCE")
for r in report:
    print(
        f"{r['pair']:>2} | {r['phash_distance']:>2} | "
        f"{str(r['sha256_identical']):>13} | "
        f"{r['mean_pixel_difference_256']}"
    )
