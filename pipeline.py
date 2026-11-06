import base64
import time

import cv2
import torch

from run_onnx import OnnxObbDetector
from pathlib import Path
from utils.utils import make_image_paths
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import numpy as np

CENSOR_TEXT_CAT ={'id_number', 'place_of_birth', 'date_of_birth',
             'address', 'person_name', 'email', 'mrz', 'bank_account', 'phone_number'}
CENSOR_OTHER_CAT ={'face', 'signature'}


"""
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠉⠀⢻⣿⣿⣿⣿⣿⣿⡿⠁⠀⠙⢿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠈⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠿⠛⠛⠛⠛⠿⣿⠀⠀⠀⠀⢀⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣀⣤⡀⠀⠀⠀⠀⣀⡀⠀⠀⠀⢀⠀⢸⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣴⣿⣿⠇⠀⠀⢠⣾⣿⣿⣆⠀⠀⠀⠀⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢡⣿⣿⣿⠀⠀⢀⣿⣿⣿⣿⣿⠀⠀⠀⢸⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⢸⣿⡿⠃⠀⠀⢸⣿⣿⣿⣿⠏⠀⠀⠀⡘⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠈⠀⠈⠭⠀⠈⠛⠿⠛⠋⠀⠀⠀⠀⠠⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡑⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠈⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠈⢃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿

WA!
"""

MODELS = Path(__file__).resolve().parent / "models"

class OnnxOCRPipeline:
    def __init__(self):
        self.detections = {}
        self.images = None
        self.onnx_detector = OnnxObbDetector(model = MODELS / "yolo26s_obb_signature.onnx")

        cfg = Cfg.load_config_from_file(str(MODELS / "vietocr_vgg_transformer.yml"))
        cfg["cnn"]["pretrained"] = False  # full weights come from the checkpoint
        cfg["weights"] = str(MODELS / "vgg_transformer.pth")
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["predictor"]["beamsearch"] = False

        self.ocr = Predictor(cfg)

    def enter_images(self, filepath):
        self.images = make_image_paths(Path(filepath))
        print("All images in folder loaded")
        return self.images

    @staticmethod
    def _key(file) -> str:
        return str(Path(file))

    @staticmethod
    def _read_image(file) -> np.ndarray:
        image = cv2.imread(str(file))
        if image is None:
            raise FileNotFoundError(f"cannot read image: {file}")
        return image

    def onnx_detect(self, file, image=None):
        image = self._read_image(file) if image is None else image
        results = self.onnx_detector.predict(image)
        self.detections[self._key(file)] = results
        return results

    def _ensure_detections(self, file, image=None):
        key = self._key(file)
        if key not in self.detections:
            self.onnx_detect(file, image)
        return self.detections[key]

    def ocr_detect(self, crop: np.ndarray, min_prob: float = 0.5):
        """Recognise a crop, retrying 180° only when the first pass is weak."""
        text, prob = self._recognise(crop)
        if prob >= min_prob:
            return text, prob
        alt_text, alt_prob = self._recognise(cv2.rotate(crop, cv2.ROTATE_180))
        return (alt_text, alt_prob) if alt_prob > prob else (text, prob)

    def _recognise(self, crop: np.ndarray):
        return self.ocr.predict(Image.fromarray(crop[:, :, ::-1]), return_prob=True)

    @staticmethod
    def _crop(image: np.ndarray, polygon) -> np.ndarray:
        """Perspective-warp one detection quad into a flat crop."""
        polygon = np.asarray(polygon, dtype=np.float32)
        width = max(1, int(round(np.linalg.norm(polygon[1] - polygon[0]))))
        height = max(1, int(round(np.linalg.norm(polygon[2] - polygon[1]))))
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(polygon, dst)
        return cv2.warpPerspective(image, matrix, (width, height))

    @staticmethod
    def _polygon_list(polygon) -> list[list[float]]:
        return np.round(np.asarray(polygon, dtype=np.float32), 2).tolist()
    
    def _regions(self, image: np.ndarray, dets: list[dict]) -> list[dict]:
        """One region per censored detection, with its recognised text.

        Text categories carry the OCR result; ``face``/``signature`` carry
        ``None``. Polygons are plain lists, so the result is JSON-serialisable
        and can be returned directly by the API.
        """
        result = []

        for det in dets:
            category = det["category"]

            if category in CENSOR_TEXT_CAT:
                crop = self._crop(image, det["polygon"])
                text, _ = self.ocr_detect(crop)
                result.append({
                    "category": category,
                    "polygon": self._polygon_list(det["polygon"]),
                    "text": text.upper() or None,
                })

            elif category in CENSOR_OTHER_CAT:
                result.append({
                    "category": category,
                    "polygon": self._polygon_list(det["polygon"]),
                    "text": None,
                })

        return result

    def _paint(self, image: np.ndarray, dets: list[dict]) -> np.ndarray:
        """Return a copy of the image with every censored region filled solid."""
        new_img = image.copy()

        for det in dets:
            if det["category"] not in CENSOR_TEXT_CAT and det["category"] not in CENSOR_OTHER_CAT:
                continue
            # fillPoly needs integer points
            polygon = np.asarray(det["polygon"], dtype=np.int32)
            cv2.fillPoly(new_img, [polygon], (255, 255, 255))

        return new_img

    def _payload(self, image: np.ndarray, dets: list[dict], started: float) -> dict:
        height, width = image.shape[:2]
        regions = self._regions(image, dets)
        masked = self._paint(image, dets)

        ok, buffer = cv2.imencode(".jpg", masked, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("failed to encode masked image as JPEG")

        return {
            "width": width,
            "height": height,
            "regions": regions,
            "masked_image": base64.b64encode(buffer.tobytes()).decode("ascii"),
            "processing_time_ms": int((time.perf_counter() - started) * 1000),
        }

    def extract_text(self, file: Path | str, image=None) -> list[dict]:
        image = self._read_image(file) if image is None else image
        return self._regions(image, self._ensure_detections(file, image))

    def censored(self, file: Path | str, save_path=None, image=None):
        image = self._read_image(file) if image is None else image
        new_img = self._paint(image, self._ensure_detections(file, image))

        if save_path:
            cv2.imwrite(str(save_path), new_img)
        return new_img

    def run_image(self, image: np.ndarray) -> dict:
        """Same payload as :meth:`run`, for an already-decoded image."""
        started = time.perf_counter()
        dets = self.onnx_detector.predict(image)
        return self._payload(image, dets, started)

    def run(self, file):
        """Detect, read, censor and encode one image; return the API payload."""
        started = time.perf_counter()
        image = self._read_image(file)
        dets = self._ensure_detections(file, image)
        return self._payload(image, dets, started)