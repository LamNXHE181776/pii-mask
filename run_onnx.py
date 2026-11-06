"""Run the OBB signature/PII detector from ONNX with no torch or ultralytics.

The exported graph already contains the rotated (probabilistic-IoU) NMS, so its
output is final: one row per detection in letterboxed coordinates,

    [cx, cy, w, h, conf, cls, angle]

``angle`` is in radians. This module maps every row back to the original image,
so the quads it returns can be drawn or masked directly.

Usage::

    python run_onnx.py --source path/to/image.jpg
    python run_onnx.py --source path/to/dir --conf 0.3 --save out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from utils.utils import letterbox

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Class ids, in the order the models was trained with.
NAMES = {
    0: "address",
    1: "ambiguous",
    2: "bank_account",
    3: "date_of_birth",
    4: "email",
    5: "face",
    6: "id_number",
    7: "mrz",
    8: "non_pii",
    9: "person_name",
    10: "phone_number",
    11: "place_of_birth",
    12: "signature",
}




class OnnxObbDetector:
    """Load the ONNX graph once and reuse it across images."""

    def __init__(self, model: str | Path, imgsz: int = 640, conf: float = 0.25) -> None:
        self.imgsz = imgsz
        self.conf = conf
        self.session = ort.InferenceSession(
            str(model), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_bgr: np.ndarray) -> list[dict]:
        """Return a list of {category, conf, polygon, box} in original pixel coords."""
        canvas, ratio, pad_x, pad_y = letterbox(image_bgr, self.imgsz)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]  # 1, 3, H, W

        raw = self.session.run(None, {self.input_name: blob})[0][0]  # (300, 7)

        results: list[dict] = []
        for row in raw:
            *box, conf, cls = row[:6]
            angle = row[6]
            if conf < self.conf:
                continue
            # Undo the letterbox: shift then scale.
            x, y, w, h = box
            x = (x - pad_x) / ratio
            y = (y - pad_y) / ratio
            w, h = w / ratio, h / ratio
            corners = self.xywhr_to_corners(np.array([x, y, w, h, angle], np.float32))
            results.append(
                {
                    "category": NAMES.get(int(cls), str(int(cls))),
                    "conf": float(conf),
                    "polygon": corners.astype(np.int32).tolist(),
                }
            )
        return results

    @staticmethod
    def xywhr_to_corners(box: np.ndarray) -> np.ndarray:
        """(cx, cy, w, h, angle_rad) -> 4x2 corner points, same convention as Ultralytics."""
        cx, cy, w, h, angle = box
        cos, sin = np.cos(angle), np.sin(angle)
        dx, dy = w / 2, h / 2
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], np.float32)
        rotate = np.array([[cos, -sin], [sin, cos]], np.float32)
        return corners @ rotate.T + np.array([cx, cy], np.float32)


def draw(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        pts = det["polygon"].astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        x, y = pts[0]
        cv2.putText(
            out,
            f"{det['category']} {det['conf']:.2f}",
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="yolo26s_obb_signature.onnx")
    parser.add_argument("--source", required=True, help="image file or directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    detector = OnnxObbDetector(args.model, imgsz=args.imgsz, conf=args.conf)
    source = Path(args.source)

    images = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS
    )

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"skip (unreadable): {path}")
            continue
        detections = detector.predict(image)
        print(f"\n{path.name}: {image.shape[1]}x{image.shape[0]}, {len(detections)} detections")
        for det in sorted(detections, key=lambda d: -d["conf"]):
            print(f"  {det['category']:14s} {det['conf']:.3f}")

        if args.save is not None:
            args.save.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save / path.name), draw(image, detections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
