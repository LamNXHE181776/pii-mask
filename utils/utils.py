from pathlib import Path
import numpy as np
import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PAD_COLOR = 114


def make_image_paths(path: Path) -> list[Path]:
    if path is not Path:
        path = Path(path)
    images = [path] if path.is_file() else sorted(
        p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    return images


def letterbox(image: np.ndarray, size: int = 640):
    """Resize+pad to ``size`` square, Ultralytics style.

    Returns ``(canvas, ratio, pad_x, pad_y)``. The canvas is padded with the
    rounded offset (that is where the pixels actually go), while the returned
    pad is the exact float offset Ultralytics uses to map boxes back, so the
    quads line up with the ``.pt`` models's own predictions.
    """
    orig_h, orig_w = image.shape[:2]

    # 1. One scale for both axes: the longer side becomes `size`.
    ratio = min(size / float(orig_h), size / float(orig_w))
    new_w, new_h = round(orig_w * ratio), round(orig_h * ratio)

    # 2. Resize
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 3. Padding. The canvas is placed at the rounded offset; the float pad is
    #    what Ultralytics uses to map boxes back, so quads match the .pt models.
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    right, bottom = size - new_w - left, size - new_h - top

    if min(left, top, right, bottom) < 0:
        raise ValueError(f"letterbox padding went negative for {orig_w}x{orig_h}")

    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_COLOR,) * 3
    )

    return canvas, ratio, pad_x, pad_y
