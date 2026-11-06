#!/usr/bin/env python3
"""Pretty-print JSON files without exploding coordinate arrays.

Standard pretty-printers put every number of every polygon on its own line,
which turns a small OCR/annotation file into thousands of lines. This one keeps
any array/list that fits within --width on a single line, so polygons and short
values stay readable while the structure stays indented.

Usage:
    python tools/beautify_json.py FILE_OR_DIR [...]            # write in place
    python tools/beautify_json.py --dry-run candidate/labels   # preview stats
    python tools/beautify_json.py --width 120 --indent 4 ...   # tune layout
    python tools/beautify_json.py --check candidate/labels     # CI-style check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SCALARS = (str, int, float, bool, type(None))


def _compact(obj) -> str:
    """One-line JSON rendering used to test whether something fits."""
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def render(obj, indent: int = 2, width: int = 100, level: int = 0) -> str:
    """Serialize *obj* with indentation, inlining anything that fits in *width*."""
    pad = " " * (indent * level)
    inner_pad = " " * (indent * (level + 1))

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        inline = _compact(obj)
        if "\n" not in inline and len(inline) + len(pad) <= width:
            return inline
        parts = [
            f"{inner_pad}{json.dumps(k, ensure_ascii=False)}: "
            f"{render(v, indent, width, level + 1)}"
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"

    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        inline = _compact(obj)
        if "\n" not in inline and len(inline) + len(pad) <= width:
            return inline
        parts = [inner_pad + render(v, indent, width, level + 1) for v in obj]
        return "[\n" + ",\n".join(parts) + "\n" + pad + "]"

    return json.dumps(obj, ensure_ascii=False)


def load_json(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def collect(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"skip (not found): {target}", file=sys.stderr)
    return files


def write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="JSON files or directories (searched recursively)")
    ap.add_argument("--indent", type=int, default=2, help="spaces per level (default: 2)")
    ap.add_argument("--width", type=int, default=100, help="max line length for inline arrays (default: 100)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if any file is unformatted")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args()

    files = collect(args.targets)
    if not files:
        print("no JSON files found", file=sys.stderr)
        return 2

    changed, unchanged, failed, unformatted = 0, 0, 0, 0
    for path in files:
        try:
            data = load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            failed += 1
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            continue

        rendered = render(data, args.indent, args.width) + "\n"
        original = path.read_text(encoding="utf-8-sig")
        if rendered == original:
            unchanged += 1
            continue

        unformatted += 1
        if args.check:
            print(f"needs formatting: {path}")
            continue
        if args.dry_run:
            changed += 1
            continue

        try:
            write_atomic(path, rendered)
            changed += 1
        except OSError as exc:
            failed += 1
            print(f"ERROR writing {path}: {exc}", file=sys.stderr)
            continue
        if not args.quiet and changed <= 3:
            print(f"formatted: {path}")

    verb = "would change" if args.dry_run else "changed"
    print(
        f"total={len(files)} {verb}={changed} already-formatted={unchanged} failed={failed}",
        file=sys.stderr,
    )
    if args.check:
        return 1 if unformatted else 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
