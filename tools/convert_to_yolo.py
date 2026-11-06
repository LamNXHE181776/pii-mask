"""
Convert label JSON files (polygon + category) to YOLO detection format.

One .txt per image, one line per field:

    class x_center y_center width height      # all values normalised to 0..1

Polygons are rotated quads, so the box is the min/max over all four points.
All 74,740 fields in this dataset produce a valid box (nothing degenerate, nothing
outside its frame), so boxes are written exactly as the labels describe them.

Usage:
    python tools/convert_to_yolo.py --out yolo_pii
    python tools/convert_to_yolo.py --out yolo_pii --val-fraction 0.2 --split-by doc_type
    python tools/convert_to_yolo.py --out yolo_pii --classes person_name,id_number,face
"""

from __future__ import annotations

from dataclasses import dataclass

import argparse
import json
import os
import random
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

DECIMALS = 6

# (x0, y0, x1, y1) in source-image pixels
Rect = tuple[float, float, float, float]

# (x1, y1, x2, y2, x3, y3, x4, y4)
Rect_OBB = tuple[float, float, float, float, float, float, float, float]


class RowOBB(NamedTuple):
    cls: int
    coords: tuple[float, float, float, float, float, float, float, float]

    def to_line(self) -> str:
        return f"{self.cls} " + " ".join(f"{v:.6f}" for v in self.coords) + "\n"


class Row(NamedTuple):
    """One YOLO label line, already normalised."""

    cls: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        return (
            f"{self.cls} {self.x_center:.{DECIMALS}f} {self.y_center:.{DECIMALS}f} "
            f"{self.width:.{DECIMALS}f} {self.height:.{DECIMALS}f}\n"
        )


# --- conversion -------------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bounding_box(polygon: list[list[float]]) -> Rect:
    """Axis-aligned extent of a (possibly rotated) polygon."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def convert_to_yolo(data, class_dict: dict[str, int], img_width: int, img_height: int) -> list[Row]:
    """Normalize every field of one document to a YOLO row."""
    rows: list[Row] = []
    for field in data["fields"]:
        x0, y0, x1, y1 = bounding_box(field["polygon"])

        # Normalize: YOLO expects fractions of the image size, not pixels.
        rows.append(
            Row(
                cls=class_dict[field["category"]],
                x_center=round(((x0 + x1) / 2) / img_width, DECIMALS),
                y_center=round(((y0 + y1) / 2) / img_height, DECIMALS),
                width=round((x1 - x0) / img_width, DECIMALS),
                height=round((y1 - y0) / img_height, DECIMALS),
            )
        )
    return rows


def order_corners(points: list[list[float]]) -> list[list[float]]:
    """Return 4 corners ordered top-left, top-right, bottom-right, bottom-left.

    Input and output are plain list[list[float]] ([[x, y], ...]); any input order works.
    Polygons that are not exactly 4 points are returned unchanged.

    Method: sort the points by angle around their centre (image y points down,
    so increasing angle runs clockwise on screen), then start the list at the
    corner with the smallest x + y (the top-left).
    """
    pts = [[float(x), float(y)] for x, y in points]
    if len(pts) != 4:
        return pts

    cx = sum(p[0] for p in pts) / 4
    cy = sum(p[1] for p in pts) / 4
    clockwise = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    start = min(range(4), key=lambda i: clockwise[i][0] + clockwise[i][1])
    return clockwise[start:] + clockwise[:start]


def convert_to_yolo_obb(data, class_dict: dict[str, int], img_width: int, img_height: int) -> list[RowOBB]:
    """Normalize every field of one document to a YOLO-OBB row."""

    rows: list[RowOBB] = []
    for field in data["fields"]:
        polygon = order_corners(field["polygon"])
        if len(polygon) != 4:
            continue

        coords = []
        for x, y in polygon:
            coords.append(min(1.00, max(0, x / img_width)))
            coords.append(min(1.00, max(0, y / img_height)))

        rows.append(
            RowOBB(
                cls=class_dict[field["category"]],
                coords=tuple(coords),
            )
        )

    return rows


def is_invalid(row: Row) -> bool:
    """A box Ultralytics would reject: empty, or hanging outside the frame."""
    return (
            row.width <= 0
            or row.height <= 0
            or row.x_center - row.width / 2 < 0
            or row.x_center + row.width / 2 > 1
            or row.y_center - row.height / 2 < 0
            or row.y_center + row.height / 2 > 1
    )

# --- filesystem -------------------------------------------------------------

def write_rows(path: Path, rows: list[Row | RowOBB]) -> None:
    """Write one label file atomically (temp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.writelines(row.to_line() for row in rows)
        os.replace(temp_path, path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise


def link_or_copy(source: Path, destination: Path) -> None:
    """Hardlink when possible (costs no disk), copy otherwise."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


# --- dataset layout ---------------------------------------------------------

def pick_val_images(docs: list[dict], fraction: float, group_by: str) -> set[str]:
    """Choose whole groups (doc_type / source) for val, never single images.

    Keeping a document type entirely on one side is what makes the validation
    score say something about layouts the models has not seen.
    """
    if fraction <= 0:
        return set()

    groups: dict[str, list[str]] = {}
    for doc in docs:
        groups.setdefault(str(doc.get(group_by, "unknown")), []).append(doc["image"])

    names = sorted(groups)
    random.Random(42).shuffle(names)  # deterministic split, same every run

    target = int(round(fraction * len(docs)))
    chosen: set[str] = set()
    total = 0
    for name in names:
        if total >= target:
            break
        chosen.add(name)
        total += len(groups[name])
    return chosen


def build_data_yaml(root: Path, names: list[str], split: bool) -> str:
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
    train_path = "images/train" if split else "images"
    val_path = "images/val" if split else "images"
    return f"path: {root}\ntrain: {train_path}\nval: {val_path}\nnames:\n{names_block}\n"


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", default="candidate/labels", help="directory of label .json files")
    parser.add_argument("--images", default="candidate/images", help="directory of images")
    parser.add_argument("--out", default="yolo_pii", help="output dataset directory")
    parser.add_argument(
        "--classes", default="", help="comma separated class list, in order; default: all categories"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.0, help="fraction of images for val (0 = flat layout)"
    )
    parser.add_argument(
        "--split-by", default="doc_type", choices=["doc_type", "source", "random"],
        help="group kept together on one side of the split (default: doc_type)",
    )
    parser.add_argument(
        "--no-link-images", dest="link_images", action="store_false",
        help="write labels only; data.yaml then points at --images",
    )
    parser.add_argument(
        "--data-type", "-d", default="seg", help="format of txt files (default: seg)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    label_dir = Path(args.labels)
    image_dir = Path(args.images)
    out_dir = Path(args.out)

    docs = [load_json(path) for path in sorted(label_dir.glob("*.json"))]
    if not docs:
        print(f"no label json found in {label_dir}", file=sys.stderr)
        return 2

    if args.classes:
        names = [name.strip() for name in args.classes.split(",") if name.strip()]
        wanted = set(names)
        # convert_to_yolo indexes class_dict directly, so drop other fields first.
        for doc in docs:
            doc["fields"] = [field for field in doc["fields"] if field["category"] in wanted]
    else:
        names = sorted({field["category"] for doc in docs for field in doc["fields"]})
    class_dict = {name: index for index, name in enumerate(names)}

    split = args.val_fraction > 0
    if split and args.split_by == "random":
        random.Random(42).shuffle(docs)
        val_count = int(round(args.val_fraction * len(docs)))
        val_images = {doc["image"] for doc in docs[:val_count]}
    else:
        val_groups = pick_val_images(docs, args.val_fraction, args.split_by)
        val_images = {
            doc["image"] for doc in docs if str(doc.get(args.split_by, "unknown")) in val_groups
        }

    written = missing_images = invalid_boxes = 0
    for doc in docs:
        if args.data_type == "seg":
            rows = convert_to_yolo(doc, class_dict, doc["width"], doc["height"])
            invalid_boxes += sum(1 for row in rows if is_invalid(row))

        elif args.data_type == "obb":
            rows = convert_to_yolo_obb(doc, class_dict, doc["width"], doc["height"])

        subset = ("val" if doc["image"] in val_images else "train") + "/" if split else ""
        write_rows(out_dir / "labels" / subset / f"{Path(doc['image']).stem}.txt", rows)
        written += 1

        if args.link_images:
            source = image_dir / doc["image"]
            if source.exists():
                link_or_copy(source, out_dir / "images" / subset / doc["image"])
            else:
                missing_images += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.yaml").write_text(build_data_yaml(out_dir.resolve(), names, split), encoding="utf-8")

    print(f"labels written: {written}  val images: {len(val_images)}  missing images: {missing_images}")
    if invalid_boxes:
        print(f"WARNING: {invalid_boxes} boxes are degenerate or outside the frame", file=sys.stderr)
    print(f"classes ({len(names)}): {', '.join(names)}")
    print(f"dataset: {out_dir.resolve() / 'data.yaml'}")
    if split:
        train_count = len(list((out_dir / "labels" / "train").glob("*.txt")))
        val_count = len(list((out_dir / "labels" / "val").glob("*.txt")))
        print(f"  train={train_count} val={val_count} (grouped by {args.split_by})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
