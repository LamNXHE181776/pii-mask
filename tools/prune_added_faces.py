"""Remove face boxes that are not in the base annotation set.

The pseudo-label merge appended predicated face boxes to midv2020 label files.
This tool keeps only the face boxes that exist in the original label JSON and
drops everything else, so a source matches its base annotations again.

Library use::

    from prune_added_faces import AddedFacePruner

    pruner = AddedFacePruner(
        base_dir="/path/to/candidate/labels",
        label_dir="/path/to/yolo_pii_obb_2/labels",
    )
    print(pruner.prune(apply=True))

Command line::

    python prune_added_faces.py                       # dry run
    python prune_added_faces.py --apply               # write, with a backup
    python prune_added_faces.py --prefixes midv2020
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# --- defaults ---------------------------------------------------------------

DEFAULT_BASE_DIR = Path(
    "/home/lam/Desktop/Archive/NEW_PROJECT/ID_image_train/candidate/labels"
)
DEFAULT_LABEL_DIR = Path(
    "/home/lam/Desktop/Archive/NEW_PROJECT/ID_image_train/yolo_pii_obb_2/labels"
)
DEFAULT_PREFIXES: tuple[str, ...] = ("midv2020",)
DEFAULT_CLASS = "5"  # face
DEFAULT_MATCH_IOU = 0.9  # a kept box must match a base box at least this well

FIELDS = 9  # cls x1 y1 x2 y2 x3 y3 x4 y4


# --- geometry ---------------------------------------------------------------


def bounding_box(values: list[float]) -> tuple[float, float, float, float]:
    """Axis-aligned extent of a rotated quad: (x0, y0, x1, y1)."""
    xs = values[0::2]
    ys = values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


# --- result -----------------------------------------------------------------


@dataclass(frozen=True)
class PruneReport:
    label_files: int = 0
    files_changed: int = 0
    boxes_removed: int = 0
    backup_dir: Path | None = None
    applied: bool = False

    def __str__(self) -> str:
        lines = [
            f"label files scanned : {self.label_files}",
            f"files changed       : {self.files_changed}",
            f"boxes removed       : {self.boxes_removed}",
        ]
        if self.backup_dir is not None:
            lines.append(f"backup              : {self.backup_dir}")
        if not self.applied:
            lines.append("dry run only - rerun with apply=True to write")
        return "\n".join(lines)


# --- pruner -----------------------------------------------------------------


class AddedFacePruner:
    """Keep only face boxes that exist in the base label JSON."""

    def __init__(
        self,
        base_dir: str | Path = DEFAULT_BASE_DIR,
        label_dir: str | Path = DEFAULT_LABEL_DIR,
        cls: str = DEFAULT_CLASS,
        prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
        match_iou: float = DEFAULT_MATCH_IOU,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.label_dir = Path(label_dir)
        self.cls = cls
        self.prefixes = prefixes
        self.match_iou = match_iou

        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"base label dir not found: {self.base_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"label dir not found: {self.label_dir}")

    # -- public API ----------------------------------------------------------

    def base_boxes(self, stem: str) -> list[tuple[float, float, float, float]]:
        """Normalised face boxes from the original label JSON."""
        path = self.base_dir / f"{stem}.json"
        if not path.exists():
            return []

        document = json.loads(path.read_text(encoding="utf-8-sig"))
        width = document.get("width") or 1
        height = document.get("height") or 1

        boxes = []
        for field in document.get("fields", []):
            if field.get("category") != "face":
                continue
            values: list[float] = []
            for x, y in field["polygon"]:
                values.extend((x / width, y / height))
            boxes.append(bounding_box(values))
        return boxes

    def strip(self, stem: str, lines: list[list[str]]) -> tuple[list[list[str]], int]:
        """Drop face boxes that do not match a base box. Returns (kept, removed)."""
        allowed = self.base_boxes(stem)
        kept: list[list[str]] = []
        removed = 0

        for line in lines:
            if line[0] != self.cls:
                kept.append(line)
                continue
            box = bounding_box([float(v) for v in line[1:]])
            if any(iou(box, base) >= self.match_iou for base in allowed):
                kept.append(line)
            else:
                removed += 1

        return kept, removed

    def prune(self, apply: bool = False, backup: bool = True) -> PruneReport:
        """Run over every matching label file. ``apply=False`` is a dry run."""
        label_files = sorted(
            path
            for path in self.label_dir.glob("*.txt")
            if path.stem.startswith(self.prefixes)
        )
        backup_dir = self._backup() if apply and backup else None

        changed = removed_total = 0
        for label_file in label_files:
            lines = self._read(label_file)
            kept, removed = self.strip(label_file.stem, lines)
            if not removed:
                continue
            changed += 1
            removed_total += removed
            if apply:
                self._write(label_file, kept)

        return PruneReport(
            label_files=len(label_files),
            files_changed=changed,
            boxes_removed=removed_total,
            backup_dir=backup_dir,
            applied=apply,
        )

    # -- internals -----------------------------------------------------------

    def _backup(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.label_dir.with_name(f"{self.label_dir.name}_backup_{stamp}")
        shutil.copytree(self.label_dir, backup_dir)
        return backup_dir

    @staticmethod
    def _read(path: Path) -> list[list[str]]:
        return [
            tokens
            for line in path.read_text().splitlines()
            if len(tokens := line.split()) == FIELDS
        ]

    @staticmethod
    def _write(path: Path, lines: list[list[str]]) -> None:
        path.write_text("".join(" ".join(line) + "\n" for line in lines))


# --- command line -----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--class-id", default=DEFAULT_CLASS)
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--no-backup", action="store_true", help="skip the backup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pruner = AddedFacePruner(
        base_dir=args.base_dir,
        label_dir=args.label_dir,
        cls=args.class_id,
        prefixes=tuple(p.strip() for p in args.prefixes.split(",") if p.strip()),
        match_iou=args.match_iou,
    )
    print(pruner.prune(apply=args.apply, backup=not args.no_backup))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
