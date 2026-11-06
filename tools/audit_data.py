#!/usr/bin/env python3
"""Audit the candidate PII dataset: integrity, schema, polygon geometry, label
consistency, distribution and duplicate content.

    python tools/audit_data.py --data-dir candidate --report reports/data_audit.json

Prints a ranked summary of problems; the JSON report keeps per-file detail.
"""

from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PII_TRUE = {
    "person_name", "date_of_birth", "place_of_birth", "id_number", "phone_number",
    "email", "address", "bank_account", "face", "signature", "mrz",
}
KNOWN = PII_TRUE | {"non_pii", "ambiguous"}
TEXT_OPTIONAL = {"face", "signature"}  # text must be null for these
OOB_TOL = 1.0          # px tolerance for polygon bounds
MIN_AREA = 8.0         # px^2, below this a polygon cannot be a usable box
MAX_PAGE_FRACTION = 0.60  # polygon larger than this share of the page is suspicious
CONFLICT_IOU = 0.30    # PII vs non_pii overlap that looks like a labeling conflict


DATE_FORMATS = (
    "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
    "%m/%d/%Y", "%d %m %Y", "%d %B %Y", "%B %d, %Y",
)
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def is_mojibake(text: str) -> bool:
    """True when text looks like UTF-8 bytes decoded as Latin-1.

    'XÃ HỘI' is legitimate Vietnamese, so marker characters alone are useless;
    a byte-level round trip is the reliable test.
    """
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    return fixed != text and "\ufffd" not in fixed


DATE_RE = re.compile(
    r"(\d{1,4})\s*[./\-\s]\s*(\d{1,2})\s*[./\-\s]\s*(\d{2,4})"
)
VN_WORDS = re.compile(r"ng[àa]y|th[áa]ng|n[ăa]m|,|\.", re.IGNORECASE)


def split_date_text(text: str):
    """Return (date_found, leftover_text) for a date_of_birth style string.

    Locale noise ('ngày 12 tháng 02 năm 1993', '26.09.1990.', '05 05 1993') is
    tolerated; leftover alphanumerics mean the box spans more than the date.
    """
    m = DATE_RE.search(text)
    if not m:
        return False, text
    leftover = VN_WORDS.sub(" ", text[: m.start()] + " " + text[m.end():])
    leftover = re.sub(r"[\s.,:;()\-]+", " ", leftover).strip()
    return True, leftover


def dhash(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def segments_cross(poly: np.ndarray) -> bool:
    n = len(poly)
    if n < 4:
        return False

    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    def intersects(p1, p2, p3, p4):
        o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
        o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
        return o1 != o2 and o3 != o4

    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if intersects(a, b, c, d):
                return True
    return False


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """True polygon IoU. Bounding boxes would report false conflicts on rotated text."""
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    union = cv2.contourArea(a32) + cv2.contourArea(b32) - inter
    return float(inter / union) if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="candidate")
    ap.add_argument("--report", default="reports/data_audit.json")
    ap.add_argument("--no-images", action="store_true", help="skip pixel decoding checks")
    ap.add_argument("--examples", type=int, default=5, help="examples to show per issue")
    args = ap.parse_args()

    root = Path(args.data_dir)
    img_dir, lbl_dir = root / "images", root / "labels"

    issues: dict[str, list[dict]] = collections.defaultdict(list)
    counts: collections.Counter = collections.Counter()
    per_source = collections.defaultdict(collections.Counter)
    doc_types = collections.Counter()
    img_hash: dict[str, str] = {}
    name_cat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    dhash_by_type: dict[str, list] = collections.defaultdict(list)

    images = {p.name: p for p in img_dir.iterdir() if p.is_file()}
    labels = {p.stem: p for p in lbl_dir.glob("*.json")}

    for name in sorted(set(images) - {k + ".jpg" for k in labels}):
        issues["image_without_label"].append({"image": name})
    for stem in sorted(set(labels) - {Path(n).stem for n in images}):
        issues["label_without_image"].append({"label": stem})

    for stem in sorted(labels):
        path = labels[stem]
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            issues["label_unreadable"].append({"label": stem, "error": str(exc)})
            continue

        counts["labels"] += 1
        fname = doc.get("image")
        if fname != stem + ".jpg":
            issues["image_field_mismatch"].append({"label": stem, "image_field": fname})

        W, H = doc.get("width"), doc.get("height")
        if not isinstance(W, (int, float)) or not isinstance(H, (int, float)) or W <= 0 or H <= 0:
            issues["bad_dimensions"].append({"label": stem, "width": W, "height": H})
            W = H = None

        src = doc.get("source", "?")
        per_source[src]["images"] += 1
        doc_types[str(doc.get("doc_type"))] += 1

        # ---- image pixel checks -------------------------------------------
        img_path = images.get(fname or (stem + ".jpg"))
        if img_path and not args.no_images:
            counts["images_decoded"] += 1
            data = img_path.read_bytes()
            digest = hashlib.md5(data).hexdigest()
            if digest in img_hash:
                issues["duplicate_image"].append({"image": fname, "same_as": img_hash[digest]})
            else:
                img_hash[digest] = fname
            buf = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                issues["image_unreadable"].append({"image": fname})
            else:
                ih, iw = img.shape[:2]
                per_source[src]["pixels_ok"] += 1
                dhash_by_type[str(doc.get("doc_type"))].append((fname, dhash(img)))
                try:
                    with Image.open(img_path) as im:
                        orient = im.getexif().get(274, 1)
                    if orient != 1:
                        issues["exif_orientation"].append({"image": fname, "orientation": orient})
                except Exception as exc:  # noqa: BLE001
                    issues["exif_unreadable"].append({"image": fname, "error": str(exc)})
                if W and H and (abs(iw - W) > 0 or abs(ih - H) > 0):
                    issues["size_mismatch"].append(
                        {"image": fname, "label": [W, H], "actual": [iw, ih]}
                    )
                    W, H = iw, ih

        fields = doc.get("fields")
        if not isinstance(fields, list):
            issues["fields_not_list"].append({"label": stem})
            continue
        if not fields:
            issues["no_fields"].append({"label": stem})

        n_pii = 0
        rects_pii, rects_non = [], []
        for idx, f in enumerate(fields):
            cat = f.get("category")
            counts[f"cat:{cat}"] += 1
            per_source[src][f"cat:{cat}"] += 1
            name_cat[str(f.get("name"))][str(cat)] += 1
            ref = {"label": stem, "field": idx, "name": f.get("name"), "category": cat}

            if cat not in KNOWN:
                issues["unknown_category"].append(ref)
            is_pii = f.get("is_pii")
            if not isinstance(is_pii, bool):
                issues["is_pii_not_bool"].append(ref)
            elif cat in PII_TRUE and is_pii is not True:
                issues["is_pii_false_but_pii_category"].append(ref)
            elif cat == "non_pii" and is_pii is not False:
                issues["non_pii_marked_true"].append(ref)
            if is_pii is True:
                n_pii += 1

            text = f.get("text")
            if cat in TEXT_OPTIONAL:
                if text is not None:
                    issues["text_on_face_or_signature"].append(ref)
            else:
                if text is None:
                    issues["missing_text"].append(ref)
                elif not isinstance(text, str):
                    issues["text_not_str"].append(ref)
                elif not text.strip():
                    issues["empty_text"].append(ref)
                else:
                    if text != text.strip():
                        issues["text_surrounding_space"].append({**ref, "text": text})
                    if "\n" in text:
                        issues["text_multiline"].append({**ref, "text": text})
                    if unicodedata.normalize("NFC", text) != text:
                        issues["text_not_nfc"].append({**ref, "text": text})
                    if is_mojibake(text):
                        issues["text_mojibake"].append({**ref, "text": text})
                    if cat == "date_of_birth":
                        found, leftover = split_date_text(text)
                        if not found:
                            issues["date_no_date_found"].append({**ref, "text": text})
                        elif len(leftover) >= 2:
                            issues["date_field_has_extra_text"].append(
                                {**ref, "text": text, "extra": leftover}
                            )
                    if cat == "email" and not EMAIL_RE.match(text.strip()):
                        issues["email_malformed"].append({**ref, "text": text})
                    if cat == "phone_number" and sum(c.isdigit() for c in text) < 7:
                        issues["phone_too_short"].append({**ref, "text": text})
                    if cat == "id_number" and sum(c.isalnum() for c in text) < 4:
                        issues["id_number_too_short"].append({**ref, "text": text})
                    if cat in ("person_name", "place_of_birth") and any(c.isdigit() for c in text):
                        issues["digit_in_name_field"].append({**ref, "text": text})

            poly = f.get("polygon")
            if not isinstance(poly, list) or len(poly) != 4:
                issues["polygon_not_quad"].append(
                    {**ref, "n_points": len(poly) if isinstance(poly, list) else None}
                )
                continue
            try:
                arr = np.asarray(poly, dtype=np.float64)
                if arr.shape != (4, 2) or not np.isfinite(arr).all():
                    raise ValueError("bad shape or non-finite")
            except (ValueError, TypeError):
                issues["polygon_bad_numbers"].append({**ref, "polygon": poly})
                continue

            if W and H:
                if (arr[:, 0] < -OOB_TOL).any() or (arr[:, 1] < -OOB_TOL).any() \
                        or (arr[:, 0] > W + OOB_TOL).any() or (arr[:, 1] > H + OOB_TOL).any():
                    issues["polygon_out_of_bounds"].append(
                        {**ref, "polygon": arr.round(1).tolist(), "size": [W, H]}
                    )
            area = shoelace(arr)
            if area < MIN_AREA:
                issues["polygon_degenerate"].append({**ref, "area": round(float(area), 2)})
            elif segments_cross(arr):
                issues["polygon_self_intersecting"].append({**ref, "area": round(float(area), 2)})
            if W and H and area > MAX_PAGE_FRACTION * W * H:
                issues["polygon_suspiciously_large"].append(
                    {**ref, "page_fraction": round(float(area) / (W * H), 3)}
                )

            if cat == "non_pii":
                rects_non.append((idx, arr))
            elif cat in PII_TRUE:
                rects_pii.append((idx, arr))

        for i, a in rects_pii:
            for j, b in rects_non:
                if quad_iou(a, b) > CONFLICT_IOU:
                    issues["pii_non_pii_overlap"].append(
                        {"label": stem, "pii_field": i, "non_pii_field": j,
                         "iou": round(quad_iou(a, b), 3)}
                    )

        if W and H and counts["labels"] % 500 == 0:
            print(f"  ... {counts['labels']} labels", file=sys.stderr)

    # ---- index.csv cross-check -----------------------------------------
    index_path = root / "index.csv"
    index_rows = {}
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                index_rows[row["image"]] = row
        counts["index_rows"] = len(index_rows)
        for name in sorted(set(index_rows) - {k + ".jpg" for k in labels}):
            issues["index_without_label"].append({"image": name})
        for stem in sorted(set(labels) - {Path(n).stem for n in index_rows}):
            issues["label_without_index_row"].append({"label": stem})
        for stem in sorted(labels):
            row = index_rows.get(stem + ".jpg")
            if not row:
                continue
            doc = json.loads(labels[stem].read_text(encoding="utf-8-sig"))
            fields = doc.get("fields", [])
            want_f, want_p = row.get("num_fields"), row.get("num_pii_fields")
            got_f = len(fields)
            got_p = sum(1 for f in fields if f.get("is_pii") is True)
            if str(got_f) != str(want_f) or str(got_p) != str(want_p):
                issues["index_count_mismatch"].append(
                    {"label": stem, "index": [want_f, want_p], "label_file": [got_f, got_p]}
                )
            for key in ("source", "doc_type", "capture"):
                if row.get(key) != str(doc.get(key)):
                    issues["index_metadata_mismatch"].append(
                        {"label": stem, "key": key, "index": row.get(key), "label_file": doc.get(key)}
                    )

    # ---- field-name vs category consistency ------------------------------
    for fname, cats in sorted(name_cat.items()):
        if len(cats) > 1:
            issues["name_with_multiple_categories"].append(
                {"name": fname, "categories": dict(cats.most_common())}
            )

    # ---- template clustering within a doc_type ----------------------------
    # Synthetic/templated documents repeat the same background, so pairwise
    # "duplicates" are meaningless; cluster them and report group sizes, which
    # is what actually matters for leak-free train/val splitting.
    near_dup_summary = {}
    for dtype, entries in dhash_by_type.items():
        n = len(entries)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if bin(entries[i][1] ^ entries[j][1]).count("1") <= 2:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        sizes = collections.Counter(find(i) for i in range(n))
        ordered = sorted(sizes.values(), reverse=True)
        near_dup_summary[dtype] = {
            "images": n,
            "clusters": len(sizes),
            "largest_cluster": ordered[0] if ordered else 0,
            "top5": ordered[:5],
        }
        if ordered and ordered[0] > 1:
            issues["template_cluster"].append(
                {"doc_type": dtype, "clusters": len(sizes), "largest": ordered[0]}
            )

    # ---- report ---------------------------------------------------------
    report = {
        "counts": dict(counts),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "doc_types": dict(doc_types),
        "template_clusters": near_dup_summary,
        "issue_counts": {k: len(v) for k, v in sorted(issues.items(), key=lambda kv: -len(kv[1]))},
        "issues": {k: v[:200] for k, v in issues.items()},
        "clean_files": counts["labels"] - len({i.get("label") for v in issues.values() for i in v if i.get("label")}),
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nfiles: {counts['labels']} labels, {len(images)} images, {counts.get('index_rows', 0)} index rows")
    print(f"images decoded: {counts['images_decoded']}")
    print(f"labels with no issue: {report['clean_files']}")
    print(f"report: {out}\n")
    for key, n in report["issue_counts"].items():
        print(f"{n:6d}  {key}")
        for ex in issues[key][: args.examples]:
            print(f"        e.g. {json.dumps(ex, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
