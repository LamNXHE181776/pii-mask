#!/usr/bin/env python3
"""Shrink loose `signature` boxes down to the actual ink.

Template / pseudo boxes (and some field boxes shipped with the public datasets)
cover the whole signature strip: blank card plus the printed caption underneath.
That is over-masking for the PII service, and for training it teaches the models
to predict a strip instead of a signature.

How the crop is found:

1. Otsu threshold on the crop isolates dark pen strokes from the lighter
   guilloche, hologram and caption text.
2. Connected components ("blobs") are filtered by size, shape and brightness.
3. The "core" is the tall blobs -- a signature stroke is much taller than the
   caption letters that sit inside the same strip.
4. Thin tail / underline strokes often form their own blobs, so anything touching
   the core box is absorbed; skipping this step clips the ink.
5. The crop box is mapped back through the quad's perspective, so tilted
   photo / scan captures stay aligned.

Usage:
    python tools/tighten_signature_ink.py --dump-viz viz_out --limit 12   # preview
    python tools/tighten_signature_ink.py --write                         # apply
    python tools/tighten_signature_ink.py --revert --write                # undo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# A quad is four (x, y) points; a Rect is (x0, y0, x1, y1); a Blob is
# (x, y, w, h) as returned by cv2.connectedComponentsWithStats.
Quad = list[list[float]]
Rect = tuple[int, int, int, int]
Blob = tuple[int, int, int, int]

# --- tuning -----------------------------------------------------------------
MIN_CROP_WIDTH = 10        # px; smaller input boxes are left untouched
MIN_CROP_HEIGHT = 8
INK_MIN_CROP = 12          # px; below this the ink detector has nothing to work with
MIN_BLOB_AREA = 8          # px; signature tails are only a few pixels thick
INK_MAX_MEAN_GRAY = 160    # blobs lighter than this are card texture, not ink
BACKGROUND_BAND = 0.92     # a blob wider/taller than this ratio is a border
CORE_MIN_HEIGHT = 0.10     # core blob must be at least this tall (ratio of crop)
CORE_VS_TALLEST = 0.45     # ... and at least this tall vs the tallest blob
CORE_MIN_AREA = 40         # px
CORE_MIN_AREA_RATIO = 0.0008
GROW_GAP_PX = 3            # absorb blobs this close to the core box
PAD_RATIO = 0.08           # breathing room so a stroke tip is never clipped
PAD_MIN_PX = 5
MIN_AREA_RATIO = 0.01      # result smaller than this is noise, keep the original
MAX_AREA_RATIO = 0.92      # result this big means the box was already tight


# --- geometry helpers -------------------------------------------------------

def quad_rect(quad: Quad) -> Rect:
    """Axis-aligned extent of a (possibly rotated) quad, as floats."""
    points = np.asarray(quad, dtype=float)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def blob_rect(blob: Blob) -> Rect:
    x, y, width, height = blob
    return x, y, x + width, y + height


def union_rect(rects: list[Rect]) -> Rect:
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def quad_from_crop(quad: Quad, crop_width: int, crop_height: int) -> np.ndarray:
    """Perspective transform from crop pixel space to the quad in image space."""
    crop_corners = np.array(
        [[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]],
        dtype=np.float32,
    )
    quad_corners = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(crop_corners, quad_corners)


def crop_rect_to_quad(rect: Rect, transform: np.ndarray) -> Quad:
    """Map a crop-space rectangle back onto the original (tilted) quad."""
    x0, y0, x1, y1 = rect
    corners = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.float32)
    points = cv2.perspectiveTransform(corners, transform).reshape(4, 2)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in points]


# --- ink detection ----------------------------------------------------------

def ink_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of dark strokes. Otsu beats adaptive thresholding here.

    An adaptive threshold latches onto the fine security pattern of document
    cards, which turns the whole crop into "ink".
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def find_blobs(mask: np.ndarray, gray: np.ndarray) -> list[Blob]:
    height, width = mask.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs: list[Blob] = []
    for index in range(1, count):
        x, y, blob_width, blob_height, area = stats[index]
        if area < MIN_BLOB_AREA:
            continue
        # A band spanning the crop is a card border or the background itself.
        if blob_width > BACKGROUND_BAND * width or blob_height > BACKGROUND_BAND * height:
            continue
        if gray[labels == index].mean() > INK_MAX_MEAN_GRAY:
            continue
        blobs.append((x, y, blob_width, blob_height))
    return blobs


def select_core(blobs: list[Blob], crop_width: int, crop_height: int) -> list[Blob]:
    """Keep only blobs tall enough to be signature strokes (drops caption text)."""
    tallest = max(blob[3] for blob in blobs)
    min_height = max(CORE_MIN_HEIGHT * crop_height, CORE_VS_TALLEST * tallest)
    min_area = max(CORE_MIN_AREA, CORE_MIN_AREA_RATIO * crop_width * crop_height)
    return [b for b in blobs if b[3] >= min_height and b[2] * b[3] >= min_area]


def blobs_touch(blob: Blob, rect: Rect, slack: int = GROW_GAP_PX) -> bool:
    """True when the blob overlaps or nearly touches the rectangle."""
    bx0, by0, bx1, by1 = blob_rect(blob)
    gap_x = max(0, rect[0] - bx1, bx0 - rect[2])
    gap_y = max(0, rect[1] - by1, by0 - rect[3])
    return max(gap_x, gap_y) <= slack and min(gap_x, gap_y) == 0


def grow_attached(core: list[Blob], blobs: list[Blob]) -> list[Blob]:
    """Absorb blobs touching the core box, until nothing new attaches.

    A signature is often one tall squash plus separate thin tail / underline
    strokes. Leaving those out would clip the ink.
    """
    core = list(core)
    grew = True
    while grew:
        grew = False
        core_rect = union_rect([blob_rect(b) for b in core])
        for blob in blobs:
            if blob in core:
                continue
            if blobs_touch(blob, core_rect):
                core.append(blob)
                grew = True
    return core


def ink_rect(crop_bgr: np.ndarray) -> Rect | None:
    """Tight rectangle around the signature ink, or None when unsure."""
    height, width = crop_bgr.shape[:2]
    if height < INK_MIN_CROP or width < INK_MIN_CROP:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blobs = find_blobs(ink_mask(crop_bgr), gray)
    if not blobs:
        return None

    core = select_core(blobs, width, height)
    if not core:
        return None

    ink = union_rect([blob_rect(b) for b in grow_attached(core, blobs)])

    # A few pixels of breathing room: never clip a stroke tip (that would leak).
    pad = max(PAD_MIN_PX, int(PAD_RATIO * max(ink[2] - ink[0], ink[3] - ink[1])))
    return (
        max(0, ink[0] - pad),
        max(0, ink[1] - pad),
        min(width, ink[2] + pad),
        min(height, ink[3] + pad),
    )


# --- per-field / per-document work -----------------------------------------

def tighten(quad: Quad, image: np.ndarray) -> tuple[Quad, str]:
    """Return the tightened polygon, plus the method or the reason it was kept."""
    image_height, image_width = image.shape[:2]
    x0, y0, x1, y1 = quad_rect(quad)
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(image_width, int(np.ceil(x1))), min(image_height, int(np.ceil(y1)))

    crop_width, crop_height = x1 - x0, y1 - y0
    if crop_width < MIN_CROP_WIDTH or crop_height < MIN_CROP_HEIGHT:
        return quad, "kept:box_too_small"

    crop = image[y0:y1, x0:x1]
    ink = ink_rect(crop)
    if ink is None:
        return quad, "kept:no_ink"

    old_area = crop_width * crop_height
    new_area = (ink[2] - ink[0]) * (ink[3] - ink[1])
    ratio = new_area / old_area
    if ratio < MIN_AREA_RATIO:
        return quad, "kept:ink_too_small"
    if ratio > MAX_AREA_RATIO:
        return quad, "kept:already_tight"

    transform = quad_from_crop(quad, crop_width, crop_height)
    return crop_rect_to_quad(ink, transform), "ink_bbox"


def sign_method(method: str) -> str:
    """Record that a field was tightened, without duplicating the marker."""
    parts = [m for m in method.split("+") if m]
    if "ink_bbox" not in parts:
        parts.append("ink_bbox")
    return "+".join(parts)


def unsign_method(method: str) -> str:
    return "+".join(m for m in method.split("+") if m and m != "ink_bbox")


def tighten_document(doc: dict, image: np.ndarray, viz: np.ndarray | None) -> tuple[bool, dict[str, int]]:
    """Tighten every signature field of one document. Returns (changed, stats)."""
    stats: dict[str, int] = {}
    changed = False

    for field in doc["fields"]:
        if field["category"] != "signature":
            continue

        old_polygon = field["polygon"]
        new_polygon, method = tighten(old_polygon, image)
        stats[method] = stats.get(method, 0) + 1

        if viz is not None:
            draw_quad(viz, old_polygon, color=(0, 200, 0))
            if method == "ink_bbox":
                draw_quad(viz, new_polygon, color=(0, 0, 255))

        if method != "ink_bbox":
            continue

        field.setdefault("polygon_orig", old_polygon)  # keeps --revert possible
        field["polygon"] = new_polygon
        field["method"] = sign_method(field.get("method", ""))
        changed = True

    return changed, stats


def revert_document(doc: dict) -> bool:
    """Restore the polygon saved before tightening. Returns True if anything moved."""
    changed = False
    for field in doc["fields"]:
        if "polygon_orig" not in field:
            continue
        field["polygon"] = field.pop("polygon_orig")
        field["method"] = unsign_method(field.get("method", ""))
        changed = True
    return changed


def draw_quad(image: np.ndarray, quad: Quad, color: tuple[int, int, int], thickness: int = 2) -> None:
    points = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)


def save_document(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="candidate/labels", help="directory of label .json files")
    parser.add_argument("--images", default="candidate/images", help="directory of images")
    parser.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    parser.add_argument("--revert", action="store_true", help="restore polygons saved in polygon_orig")
    parser.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    parser.add_argument("--sources", default="", help="comma separated, e.g. idnet,vn_synth")
    parser.add_argument("--dump-viz", default="", help="write before/after overlays to this directory")
    return parser.parse_args()


def select_files(labels: Path, sources: set[str], limit: int) -> list[Path]:
    files = sorted(labels.glob("*.json"))
    if sources:
        files = [p for p in files if json.loads(p.read_text(encoding="utf-8-sig"))["source"] in sources]
    return files[:limit] if limit else files


def main() -> int:
    args = parse_args()

    labels = Path(args.labels)
    images = Path(args.images)
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    files = select_files(labels, sources, args.limit)

    viz_dir = Path(args.dump_viz) if args.dump_viz else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))

        if args.revert:
            changed = revert_document(doc)
            key = "reverted" if changed else "nothing_to_revert"
            stats[key] = stats.get(key, 0) + 1
            if changed and args.write:
                save_document(path, doc)
            continue

        image = cv2.imread(str(images / doc["image"]))
        if image is None:
            print(f"  missing/unreadable image: {images / doc['image']}", file=sys.stderr)
            continue

        viz = image.copy() if viz_dir else None
        changed, doc_stats = tighten_document(doc, image, viz)
        for key, value in doc_stats.items():
            stats[key] = stats.get(key, 0) + value

        if changed and args.write:
            save_document(path, doc)
        if viz is not None:
            cv2.imwrite(str(viz_dir / doc["image"]), viz)

    print(f"files={len(files)}")
    for method, count in sorted(stats.items(), key=lambda item: -item[1]):
        print(f"  {method:20s} {count}")
    if not args.write:
        print("dry-run: nothing written. Re-run with --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
