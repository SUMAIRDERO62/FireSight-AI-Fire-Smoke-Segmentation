import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

ROOTS = {
    "v1": Path("data/raw/roboflow_fire_smoke_seg_v1"),
    "v11": Path("data/raw/Fire and Smoke Segmentation.v11-rfdetr-nano.coco-segmentation"),
}

THRESHOLD = 6
OUTPUT_CSV = Path("reports/results/phase04_near_duplicate_pairs.csv")
OUTPUT_DIR = Path("reports/figures/near_duplicate_review")

N = 32
x = np.arange(N)
dct = np.cos(np.pi * (2*x[:, None] + 1) * np.arange(N)[None, :] / (2*N))
dct[:, 0] *= 1 / np.sqrt(2)
dct *= np.sqrt(2/N)

def phash(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((N, N)), dtype=np.float32)
    transformed = dct.T @ arr @ dct
    low = transformed[:8, :8]
    median = np.median(low.flatten()[1:])
    value = 0
    for bit in (low > median).flatten():
        value = (value << 1) | int(bit)
    return value

def resolve_image(root, jf, filename, index):
    candidates = [jf.parent / filename, root / filename]
    for p in candidates:
        if p.is_file():
            return p
    matches = index.get(Path(filename).name.lower(), [])
    return matches[0] if len(matches) == 1 else None

def load_records(dataset, root):
    index = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            index[p.name.lower()].append(p)

    records = []
    for jf in root.rglob("_annotations.coco.json"):
        split = jf.parent.name.lower()
        data = json.loads(jf.read_text(encoding="utf-8"))

        for im in data.get("images", []):
            path = resolve_image(root, jf, im.get("file_name", ""), index)
            if path is None:
                continue
            try:
                records.append({
                    "dataset": dataset,
                    "split": split,
                    "filename": path.name,
                    "path": path,
                    "hash": phash(path),
                })
            except Exception as e:
                print("IMAGE ERROR:", path, str(e))

    return records

def compare(a_records, b_records, scope, same_collection=False):
    results = []
    for a in a_records:
        for b in b_records:
            if same_collection and a["path"] == b["path"]:
                continue
            dist = (a["hash"] ^ b["hash"]).bit_count()
            if dist <= THRESHOLD:
                results.append({
                    "scope": scope,
                    "dataset_a": a["dataset"],
                    "split_a": a["split"],
                    "file_a": a["filename"],
                    "path_a": str(a["path"]),
                    "dataset_b": b["dataset"],
                    "split_b": b["split"],
                    "file_b": b["filename"],
                    "path_b": str(b["path"]),
                    "distance": dist,
                })
    return results

def make_contact_sheets(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = sorted(rows, key=lambda r: r["distance"])[:40]

    if not selected:
        print("No candidate pairs for contact sheets.")
        return

    cell_w, cell_h = 590, 190
    thumb_w, thumb_h = 275, 145

    for page_start in range(0, len(selected), 6):
        page = selected[page_start:page_start + 6]
        sheet = Image.new("RGB", (cell_w * 2, cell_h * 3), "white")
        draw = ImageDraw.Draw(sheet)

        for i, row in enumerate(page):
            col, line = i % 2, i // 2
            ox, oy = col * cell_w, line * cell_h

            draw.text(
                (ox + 8, oy + 5),
                f'Distance {row["distance"]} | {row["scope"]}',
                fill="black"
            )

            for j, key in enumerate(("path_a", "path_b")):
                try:
                    with Image.open(row[key]) as im:
                        im = im.convert("RGB")
                        im.thumbnail((thumb_w, thumb_h))
                        px = ox + 8 + j * 285
                        py = oy + 28
                        sheet.paste(im, (px, py))
                        draw.text(
                            (px, py + thumb_h + 2),
                            row["file_a" if j == 0 else "file_b"][:38],
                            fill="black"
                        )
                except Exception as e:
                    print("CONTACT SHEET ERROR:", row[key], str(e))

        page_no = page_start // 6 + 1
        out = OUTPUT_DIR / f"near_duplicates_page_{page_no}.jpg"
        sheet.save(out, quality=90)
        print("CONTACT SHEET:", out)

def main():
    all_records = {}
    for name, root in ROOTS.items():
        print("\nLoading:", name)
        all_records[name] = load_records(name, root)
        print("Images hashed:", len(all_records[name]))

    results = []

    for name in ("v1", "v11"):
        records = all_records[name]
        train = [r for r in records if r["split"] == "train"]

        for split in ("valid", "test"):
            heldout = [r for r in records if r["split"] == split]
            pairs = compare(train, heldout, f"{name}_train_vs_{split}")
            results.extend(pairs)

            matched = len({p["path_b"] for p in pairs})
            print(
                f"{name} train vs {split}: "
                f"{len(pairs)} pairs; {matched} distinct held-out images matched"
            )

    cross = compare(
        all_records["v1"],
        all_records["v11"],
        "v1_vs_v11_all_splits"
    )
    results.extend(cross)
    print("\nv1 vs v11 cross-dataset pairs:", len(cross))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope", "dataset_a", "split_a", "file_a", "path_a",
        "dataset_b", "split_b", "file_b", "path_b", "distance"
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["distance"]))

    print("\nTOTAL CANDIDATE PAIRS:", len(results))
    print("DISTANCE <= 2:", sum(r["distance"] <= 2 for r in results))
    print("DISTANCE <= 4:", sum(r["distance"] <= 4 for r in results))
    print("DISTANCE <= 6:", sum(r["distance"] <= 6 for r in results))
    print("CSV REPORT:", OUTPUT_CSV)

    make_contact_sheets(results)
    print("\nIMPROVED PERCEPTUAL AUDIT COMPLETE")

if __name__ == "__main__":
    main()
