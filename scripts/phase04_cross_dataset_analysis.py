import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

source = Path("reports/results/phase04_near_duplicate_pairs.csv")
output = Path("reports/results/phase04_cross_dataset_analysis.txt")
output_csv = Path("reports/results/phase04_cross_dataset_split_matrix.csv")

with source.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

cross = [r for r in rows if r["dataset_a"] != r["dataset_b"]]

split_counts = Counter()
distance_counts = Counter()
source_matches = defaultdict(set)

for r in cross:
    split_pair = (r["split_a"], r["split_b"])
    split_counts[split_pair] += 1
    distance_counts[(r["split_a"], r["split_b"], int(r["distance"]))] += 1

    # Extract original-looking source identifier from Roboflow filename
    a = re.sub(r"_jpg\.rf\..*$", "", Path(r["file_a"]).stem)
    b = re.sub(r"_jpg\.rf\..*$", "", Path(r["file_b"]).stem)

    if a == b:
        source_matches[(a, r["split_a"], r["split_b"])].add(
            (r["file_a"], r["file_b"])
        )

lines = []
lines.append("FIRESIGHT AI - CROSS DATASET NEAR DUPLICATE ANALYSIS")
lines.append("=" * 60)
lines.append(f"Total CSV pairs: {len(rows)}")
lines.append(f"Cross-dataset candidate pairs: {len(cross)}")
lines.append("")

lines.append("CROSS-DATASET SPLIT MATRIX")
lines.append("-" * 60)

for (a, b), count in sorted(split_counts.items()):
    lines.append(f"v1 {a:5s} -> v11 {b:5s}: {count}")

lines.append("")
lines.append("DISTANCE BREAKDOWN BY SPLIT")
lines.append("-" * 60)

for (a, b, dist), count in sorted(distance_counts.items()):
    lines.append(f"v1 {a:5s} -> v11 {b:5s} | distance {dist}: {count}")

lines.append("")
lines.append("SAME SOURCE-ID CANDIDATES")
lines.append("-" * 60)
lines.append(f"Matching source-ID/split combinations: {len(source_matches)}")

for (source_id, a, b), pairs in sorted(source_matches.items()):
    lines.append(
        f"{source_id}: v1/{a} <-> v11/{b} | candidate pairs: {len(pairs)}"
    )

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(lines), encoding="utf-8")

with output_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["v1_split", "v11_split", "distance", "candidate_pairs"])
    for (a, b, dist), count in sorted(distance_counts.items()):
        writer.writerow([a, b, dist, count])

print("\n".join(lines))
print(f"\nTEXT REPORT: {output}")
print(f"CSV REPORT:  {output_csv}")
