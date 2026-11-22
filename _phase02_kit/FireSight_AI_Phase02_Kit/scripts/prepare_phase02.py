from __future__ import annotations
import argparse, csv, hashlib, json, shutil
from collections import defaultdict
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--dataset",choices=["d-fire"],default="d-fire")
    ap.add_argument("--val-ratio",type=float,default=0.15)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    if not 0.05 <= args.val_ratio <= 0.30: raise SystemExit("--val-ratio must be between 0.05 and 0.30")
    project=Path(args.project_root).resolve()
    raw=project/"data/raw/D-Fire"
    out=project/"data/processed/phase02/d-fire-detection"
    if not raw.exists(): raise SystemExit(f"Missing dataset folder: {raw}")

    pairs=[]
    for split in ("train","test"):
        images=raw/split/"images"; labels=raw/split/"labels"
        if not images.exists() or not labels.exists():
            raise SystemExit(f"Expected D-Fire layout missing: {images} or {labels}")
        for img in sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"}):
            lab=labels/(img.stem+".txt")
            if not lab.exists(): continue
            pairs.append({"image":img,"label":lab,"source_split":split,"hash":digest(img)})
    if not pairs: raise SystemExit("No image/label pairs found in expected D-Fire layout.")

    hash_splits=defaultdict(set)
    for p in pairs: hash_splits[p["hash"]].add(p["source_split"])
    cross_boundary={h for h,s in hash_splits.items() if len(s)>1}
    usable=[p for p in pairs if not (p["source_split"]=="train" and p["hash"] in cross_boundary)]
    train_pool=[p for p in usable if p["source_split"]=="train"]
    test_pool=[p for p in usable if p["source_split"]=="test"]

    # Stable hash-based split, independent of filesystem enumeration order.
    val, train=[],[]
    threshold=int(args.val_ratio*10000)
    for p in train_pool:
        token=f"{args.seed}:{p['hash']}".encode()
        bucket=int(hashlib.sha256(token).hexdigest()[:8],16)%10000
        (val if bucket < threshold else train).append(p)
    assignments=[("train",p) for p in train]+[("val",p) for p in val]+[("test",p) for p in test_pool]

    manifest=[]
    if not args.dry_run:
        for split in ("train","val","test"):
            (out/"images"/split).mkdir(parents=True,exist_ok=True)
            (out/"labels"/split).mkdir(parents=True,exist_ok=True)
    for split,p in assignments:
        dst_img=out/"images"/split/p["image"].name
        dst_lab=out/"labels"/split/p["label"].name
        if not args.dry_run:
            shutil.copy2(p["image"],dst_img)
            shutil.copy2(p["label"],dst_lab)
        manifest.append({"split":split,"source_split":p["source_split"],"image":str(p["image"]),
                         "label":str(p["label"]),"sha256":p["hash"],
                         "prepared_image":str(dst_img),"prepared_label":str(dst_lab)})
    excluded=[]
    for p in pairs:
        if p["source_split"]=="train" and p["hash"] in cross_boundary:
            excluded.append({"image":str(p["image"]),"reason":"exact_duplicate_hash_also_present_in_original_test"})
    if not args.dry_run:
        with (out/"split_manifest.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
        report={"dataset":"D-Fire","task":"object_detection_only","seed":args.seed,"val_ratio":args.val_ratio,
                "counts":{"train":len(train),"val":len(val),"test":len(test_pool)},
                "excluded_count":len(excluded),"excluded":excluded,
                "warning":"YOLO detection labels only; not compatible with YOLO11-seg or U-Net masks."}
        (out/"preparation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Prepared split counts: train={len(train)}, val={len(val)}, test={len(test_pool)}")
    print(f"Excluded train copies due to exact train/test hash overlap: {len(excluded)}")
    print("Mode: dry-run" if args.dry_run else f"Output: {out}")
    print("Task compatibility: YOLO object detection only; not segmentation.")

if __name__=="__main__": main()
