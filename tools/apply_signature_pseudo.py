#!/usr/bin/env python3
"""Add template-derived `signature` regions to labels that lack them.

Some sources (idnet, idnet_us) ship without signature labels even though the
document has a visible signature. Their canvases are pixel-identical per
document type, so a fixed box per doc_type -- derived from the midv2020 labels
that share the same canvas -- is accurate.

Pseudo-labels are marked so they can be excluded later:

    "pseudo": true, "method": "template_box_from_midv2020_labels"

Usage:
    python tools/apply_signature_pseudo.py --dry-run     # report only (default)
    python tools/apply_signature_pseudo.py --write       # apply
    python tools/apply_signature_pseudo.py --strip       # remove them again
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PSEUDO_KEY = "template_box_from_midv2020_labels"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="candidate/labels")
    ap.add_argument("--pseudo", default="reports/signature_pseudo.json")
    ap.add_argument("--write", action="store_true", help="write changes (default is dry-run)")
    ap.add_argument("--strip", action="store_true", help="remove pseudo signature fields")
    args = ap.parse_args()

    labels = Path(args.labels)
    pseudo = json.loads(Path(args.pseudo).read_text(encoding="utf-8"))

    added = removed = skipped = 0
    for stem, entry in sorted(pseudo.items()):
        path = labels / f"{stem}.json"
        if not path.exists():
            skipped += 1
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        fields = doc.get("fields", [])
        existing = [i for i, f in enumerate(fields) if f.get("method") == PSEUDO_KEY]

        if args.strip:
            if not existing:
                continue
            doc["fields"] = [f for f in fields if f.get("method") != PSEUDO_KEY]
            removed += 1
            if args.write:
                path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        if any(f.get("category") == "signature" for f in fields):
            continue  # real label already present, never overwrite

        doc["fields"] = fields + [{
            "name": "signature",
            "category": "signature",
            "is_pii": True,
            "text": None,
            "polygon": entry["polygon"],
            "pseudo": True,
            "method": entry["method"],
        }]
        added += 1
        if args.write:
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mode = "write" if args.write else "dry-run"
    if args.strip:
        print(f"[{mode}] removed {removed} pseudo signature labels, skipped {skipped}")
    else:
        print(f"[{mode}] added {added} pseudo signature labels, skipped {skipped} files")
    if not args.write:
        print("re-run with --write to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
