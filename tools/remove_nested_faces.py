"""Remove face boxes that sit completely inside a larger face box.

midv2020 labels often carry two face boxes for one face: a loose one and a
tight one nested inside it. This keeps the larger box and deletes the smaller,
which is what a detector should learn from.

Scope is controlled by ``prefixes`` so only the affected sources are rewritten.

Library use::

    from remove_nested_faces import NestedFaceCleaner

    cleaner = NestedFaceCleaner(label_dir="/path/to/labels")
    report = cleaner.clean(apply=True)
    print(report)

Command line::

    python remove_nested_faces.py                     # dry run
    python remove_nested_faces.py --apply             # write, with a backup
    python remove_nested_faces.py --apply --prefixes midv2020,idnet
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# --- defaults ---------------------------------------------------------------

DEFAULT_LABEL_DIR = Path(
    "/home/lam/Desktop/Archive/NEW_PROJECT/ID_image_train/yolo_pii_obb_2/labels"
)
DEFAULT_PREFIXES: tuple[str, ...] = ("midv2020",)
DEFAULT_CLASS = "5"  # face
DEFAULT_CONTAINMENT = 0.95  # share of the small box that must be inside the big one

FIELDS = 9  # cls x1 y1 x2 y2 x3 y3 x4 y4


# --- geometry ---------------------------------------------------------------


def bounding_box(values: list[float]) -> tuple[float, float, float, float]:
    """Axis-aligned extent of a rotated quad: (x0, y0, x1, y1)."""
    xs = values[0::2]
    ys = values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


# --- result -----------------------------------------------------------------


@dataclass(frozen=True)
class CleanReport:
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


# --- cleaner ----------------------------------------------------------------


class NestedFaceCleaner:
    """Delete face boxes contained inside a larger face box."""

    def __init__(
        self,
        label_dir: str | Path = DEFAULT_LABEL_DIR,
        cls: str = DEFAULT_CLASS,
        containment: float = DEFAULT_CONTAINMENT,
        prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
    ) -> None:
        self.label_dir = Path(label_dir)
        self.cls = cls
        self.containment = containment
        self.prefixes = prefixes

        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"label dir not found: {self.label_dir}")

    # -- public API ----------------------------------------------------------

    def strip(self, lines: list[list[str]]) -> tuple[list[list[str]], int]:
        """Drop nested boxes from one label file. Returns (kept, removed)."""
        boxes = [
            (index, bounding_box([float(v) for v in line[1:]]))
            for index, line in enumerate(lines)
            if line[0] == self.cls
        ]

        doomed: set[int] = set()
        for index, inner in boxes:
            inner_area = box_area(inner)
            if inner_area <= 0.0:
                continue
            for other_index, outer in boxes:
                if other_index == index or other_index in doomed:
                    continue
                outer_area = box_area(outer)
                # only a strictly larger box, or a later equal one, may win
                if outer_area < inner_area:
                    continue
                if outer_area == inner_area and other_index < index:
                    continue
                if intersection_area(inner, outer) / inner_area >= self.containment:
                    doomed.add(index)
                    break

        kept = [line for index, line in enumerate(lines) if index not in doomed]
        return kept, len(doomed)

    def clean(self, apply: bool = False, backup: bool = True) -> CleanReport:
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
            kept, removed = self.strip(lines)
            if not removed:
                continue
            changed += 1
            removed_total += removed
            if apply:
                self._write(label_file, kept)

        return CleanReport(
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
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--class-id", default=DEFAULT_CLASS, help="class id to clean (default 5, face)")
    parser.add_argument("--containment", type=float, default=DEFAULT_CONTAINMENT)
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    parser.add_argument("--no-backup", action="store_true", help="skip the backup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cleaner = NestedFaceCleaner(
        label_dir=args.label_dir,
        cls=args.class_id,
        containment=args.containment,
        prefixes=tuple(p.strip() for p in args.prefixes.split(",") if p.strip()),
    )
    print(cleaner.clean(apply=args.apply, backup=not args.no_backup))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
