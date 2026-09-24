"""Convert the PV-Multi-Defect VOC dataset to YOLO format and split train/val.

Paper: Li et al., GBH-YOLOv5 (Electronics 2023, 12, 561).
Dataset: 5 defect classes, 600x600 EL images, ~1108 images / 1107 annotations.
Paper split: 886 train / 222 val. We do a deterministic 80/20 split (seed=0),
which lands on ~886/221 over the images that have a matching annotation.

Run from the repo root:
    uv run python prepare_pv_data.py --voc-root ../PV-Multi-Defect-main/PV-Multi-Defect-main
Output layout (what data/pv.yaml points at):
    datasets/pv/images/{train,val}/*.jpg
    datasets/pv/labels/{train,val}/*.txt
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copyfile

# Paper order (Table 5) — index is the YOLO class id, so keep it fixed.
CLASSES = ["broken", "hot_spot", "black_border", "scratch", "no_electricity"]


def voc_to_yolo(size, box):
    """(w,h), (xmin,xmax,ymin,ymax) -> normalized (xc,yc,w,h)."""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    xc = (box[0] + box[1]) / 2.0 * dw
    yc = (box[2] + box[3]) / 2.0 * dh
    w = (box[1] - box[0]) * dw
    h = (box[3] - box[2]) * dh
    return xc, yc, w, h


def convert_annotation(xml_path: Path) -> str:
    """Return YOLO label file text for one VOC xml (empty string if no objects)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        difficult = obj.find("difficult")
        if cls not in CLASSES or (difficult is not None and difficult.text == "1"):
            continue
        b = obj.find("bndbox")
        box = (float(b.find("xmin").text), float(b.find("xmax").text),
               float(b.find("ymin").text), float(b.find("ymax").text))
        xc, yc, bw, bh = voc_to_yolo((w, h), box)
        lines.append(f"{CLASSES.index(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True,
                    help="folder containing Annotations/ and JPEGImages/")
    ap.add_argument("--out", default="datasets/pv", help="output dataset root")
    ap.add_argument("--val-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    voc = Path(args.voc_root)
    ann_dir, img_dir = voc / "Annotations", voc / "JPEGImages"
    assert ann_dir.is_dir() and img_dir.is_dir(), f"bad --voc-root: {voc}"

    # Keep only images that have a matching annotation.
    pairs = []
    for xml in sorted(ann_dir.glob("*.xml")):
        img = img_dir / f"{xml.stem}.jpg"
        if img.exists():
            pairs.append((img, xml))
    assert pairs, "no image/annotation pairs found"

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = round(len(pairs) * args.val_ratio)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    out = Path(args.out)
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, xml in items:
            copyfile(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{xml.stem}.txt").write_text(
                convert_annotation(xml))

    print(f"pairs={len(pairs)}  train={len(splits['train'])}  val={len(splits['val'])}")
    print(f"classes={CLASSES}")
    print(f"output={out.resolve()}")


if __name__ == "__main__":
    main()
