"""把 LabelImg YOLO txt 转成 PascalVOC xml（阶段 C/D 兼容）。"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2

CLASSES = ["person", "car"]


def yolo_line_to_box(parts: list[str], w: int, h: int) -> tuple[int, int, int, int, int]:
    cls_id = int(parts[0])
    xc, yc, bw, bh = map(float, parts[1:5])
    xmin = max(0, int(round((xc - bw / 2) * w)))
    ymin = max(0, int(round((yc - bh / 2) * h)))
    xmax = min(w, int(round((xc + bw / 2) * w)))
    ymax = min(h, int(round((yc + bh / 2) * h)))
    return cls_id, xmin, ymin, xmax, ymax


def convert_one(txt_path: Path, images_dir: Path, xml_dir: Path, classes: list[str]) -> Path:
    stem = txt_path.stem
    img_path = images_dir / f"{stem}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"image not found for {txt_path.name}: {img_path}")

    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f"cannot read image: {img_path}")
    h, w = im.shape[:2]

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = img_path.name
    ET.SubElement(annotation, "path").text = str(img_path.resolve())
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id, xmin, ymin, xmax, ymax = yolo_line_to_box(parts, w, h)
        if cls_id < 0 or cls_id >= len(classes):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = classes[cls_id]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(xmin)
        ET.SubElement(bnd, "ymin").text = str(ymin)
        ET.SubElement(bnd, "xmax").text = str(xmax)
        ET.SubElement(bnd, "ymax").text = str(ymax)

    xml_dir.mkdir(parents=True, exist_ok=True)
    out = xml_dir / f"{stem}.xml"
    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="\t")
    tree.write(out, encoding="utf-8", xml_declaration=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--txt-dir", type=Path, default=Path("xml"), help="dir containing YOLO .txt")
    p.add_argument("--images", type=Path, default=Path("images"))
    p.add_argument("--xml-dir", type=Path, default=Path("xml"))
    p.add_argument("--move-txt-to", type=Path, default=Path("labels"), help="move YOLO txt here after convert")
    args = p.parse_args()

    txts = sorted(args.txt_dir.glob("london_*.txt"))
    if not txts:
        raise SystemExit(f"No london_*.txt in {args.txt_dir.resolve()}")

    args.move_txt_to.mkdir(parents=True, exist_ok=True)
    n = 0
    for txt in txts:
        out = convert_one(txt, args.images, args.xml_dir, CLASSES)
        dest = args.move_txt_to / txt.name
        txt.replace(dest)  # move into labels/
        print(f"{txt.name} -> {out.name} + labels/{dest.name}")
        n += 1
    print(f"converted {n} files; classes={CLASSES}")


if __name__ == "__main__":
    main()
