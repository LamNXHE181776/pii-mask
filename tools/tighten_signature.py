#!/usr/bin/env python3
"""Replace wide template signature boxes with tight PaddleOCR-detected boxes.

The pseudo boxes in reports/signature_pseudo.json are deliberately generous
(unions over signature variants). PP-OCR's text detector fires on cursive
signatures and returns an ink-tight polygon, so where it finds one we swap it in;
where it finds nothing we keep the template box rather than dropping the mask.

Usage:
    python tools/tighten_signature.py --limit 12            # dry-run sample
    python tools/tighten_signature.py --write               # all 1273, in place
    python tools/tighten_signature.py --write --limit 50    # partial
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def poly_area_box(poly) -> tuple[float, float, float, float]:
    a = np.asarray(poly, dtype=float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def center_inside(poly, box) -> bool:
    a = np.asarray(poly, dtype=float).mean(axis=0)
    x0, y0, x1, y1 = box
    return x0 <= a[0] <= x1 and y0 <= a[1] <= y1


def area(poly) -> float:
    x0, y0, x1, y1 = poly_area_box(poly)
    return (x1 - x0) * (y1 - y0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="candidate/labels")
    ap.add_argument("--pseudo", default="reports/signature_pseudo.json")
    ap.add_argument("--images", default="candidate/images")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    ap.add_argument("--min-area-ratio", type=float, default=0.25,
                    help="reject OCR boxes smaller than this fraction of the template box")
    ap.add_argument("--max-area-ratio", type=float, default=1.0)
    args = ap.parse_args()

    from paddleocr import PaddleOCR

    pseudo = json.loads(Path(args.pseudo).read_text(encoding="utf-8"))
    stems = sorted(pseudo)
    if args.limit:
        stems = stems[: args.limit]

    ocr = PaddleOCR(
        lang="vi",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,  # paddle 3.3 + onednn hits a PIR attribute bug otherwise
    )

    replaced, kept, failed = 0, 0, 0
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        path = Path(args.labels) / f"{stem}.json"
        img_path = Path(args.images) / f"{stem}.jpg"
        doc = json.loads(path.read_text(encoding="utf-8"))
        tmpl = next((f for f in doc["fields"] if f.get("pseudo")), None)
        if tmpl is None:
            kept += 1
            continue
        tbox = poly_area_box(tmpl["polygon"])
        tarea = area(tmpl["polygon"])
        try:
            res = ocr.predict(str(img_path))[0]
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  OCR failed on {stem}: {exc}", file=sys.stderr)
            continue

        best, best_area = None, 0.0
        for poly, text, score in zip(res["dt_polys"], res["rec_texts"], res["rec_scores"]):
            if not center_inside(poly, tbox):
                continue
            a = area(poly)
            if not (args.min_area_ratio * tarea <= a <= args.max_area_ratio * tarea):
                continue
            if a > best_area:
                best, best_area = poly, a

        if best is None:
            kept += 1
        else:
            tmpl["polygon"] = [[float(x), float(y)] for x, y in np.asarray(best, float)]
            tmpl["method"] = "ocr_det"
            replaced += 1
            if args.write:
                path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if i % 10 == 0:
            print(f"  {i}/{len(stems)}  replaced={replaced} kept={kept} "
                  f"({(time.time()-t0)/i:.1f}s/img)", file=sys.stderr)

    print(f"\nprocessed={len(stems)} replaced={replaced} kept_template={kept} failed={failed}")
    if not args.write:
        print("dry-run: nothing written. Re-run with --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
