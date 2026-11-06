# PII masking service

A FastAPI service that finds personal information on document photos (ID cards,
passports, medical and insurance paperwork), draws over it, and returns the
redacted image together with the list of masked regions.

One image in, one JSON out:

```json
{
  "width": 929,
  "height": 580,
  "regions": [
    {"category": "person_name", "polygon": [[340.0,122.0],[400.0,122.0],[400.0,153.0],[340.0,153.0]], "text": "TUUL"},
    {"category": "face", "polygon": [[32.0,92.0],[323.0,92.0],[323.0,377.0],[32.0,377.0]], "text": null}
  ],
  "masked_image": "/9j/4AAQSkZJRg...",
  "processing_time_ms": 761
}
```

---

## Requirements

| Component | Version |
|---|---|
| NVIDIA driver | ≥ 580 (CUDA 13.x) |
| Docker | ≥ 24 |
| [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | current |
| GPU VRAM | ≥ 4 GB (measured peak well under 2 GB) |
| Python (local dev only) | 3.12 |
| [uv](https://docs.astral.sh/uv/) (local dev only) | current |

Set up the GPU runtime once on the host:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick start (Docker)

```bash
docker build -t pii-mask:1 .
docker run --rm --gpus all -p 8000:8000 pii-mask:1
```

The service is then at:

- **http://127.0.0.1:8000/docs** — Swagger UI
- **http://127.0.0.1:8000/health** — liveness

```bash
curl -s -X POST http://127.0.0.1:8000/v1/mask \
     -F "file=@candidate/images/idnet_00003.jpg" | head -c 400
```

The image is fully self-contained and runs **without network access**:

```bash
docker run --rm --gpus all --network none pii-mask:1
```

## Local development

```bash
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

`models/` must contain the three artifacts listed below. `pipeline.py` resolves
them relative to the file itself, so the working directory doesn't matter.

---

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /v1/mask`

`multipart/form-data` with a single field **`file`** (JPEG or PNG).

| Field | Type | Notes |
|---|---|---|
| `width`, `height` | int | dimensions of the submitted image |
| `regions[]` | list | one entry per masked region |
| `regions[].category` | str | one of the 12 PII categories |
| `regions[].polygon` | 4×2 floats | pixel coordinates of the submitted image |
| `regions[].text` | str \| null | recognised text; `null` for `face` / `signature` |
| `masked_image` | str | base64-encoded JPEG, same dimensions as the input |
| `processing_time_ms` | int | server-side processing time |

Masked categories: `person_name`, `date_of_birth`, `place_of_birth`,
`id_number`, `phone_number`, `email`, `address`, `bank_account`, `face`,
`signature`, `mrz`, `ambiguous`.

Errors: `400` for an empty or non-decodable file; `500` is never produced for
malformed input (the handler catches and reports).

---

## How it works

```
upload bytes
   │
   ├─ cv2.imdecode ──────────────► 400 if not an image
   │
   ├─ YOLO26s-OBB (ONNX)  ───────► 13-class rotated boxes, NMS baked into the graph
   │     letterbox 640 → run → scale back to original pixels
   │
   ├─ per detected text box:
   │     perspective-warp the quad ──► VietOCR ──► text
   │     (retried at 180° only when confidence < 0.5)
   │
   ├─ face / signature: detector quad, text = null
   │
   ├─ paint every censored quad solid white
   │
   └─ JPEG q=92 → base64 → JSON
```

### Models

| File | Size | Purpose |
|---|---|---|
| `models/yolo26s_obb_signature.onnx` | 38 MB | 13-class OBB detector, exported with `nms=True` |
| `models/vgg_transformer.pth` | 145 MB | VietOCR recognition weights |
| `models/vietocr_vgg_transformer.yml` | 2 KB | merged VietOCR config, so startup never hits `vocr.vn` |

The ONNX graph was exported from `../signature_training/yolo26s_obb_signature.pt`:

```bash
cd ../signature_training
.venv/bin/python -c "
from ultralytics import YOLO
YOLO('yolo26s_obb_signature.pt').export(
    format='onnx', imgsz=640, nms=True, conf=0.05, iou=0.7,
    max_det=300, opset=17, simplify=True, dynamic=False)
"
cp yolo26s_obb_signature.onnx ../ID_image_train/models/
```

Output layout is `(1, 300, 7)` = `[cx, cy, w, h, conf, cls, angle]` in letterbox
coordinates. Because NMS is inside the graph, inference needs only `onnxruntime`
— no torch, no ultralytics.

VietOCR weights are the public `vgg_transformer` checkpoint; the config is
generated once with `Cfg.load_config_from_name("vgg_transformer").save(...)` and
then loaded from disk with `Cfg.load_config_from_file`, with
`cfg["cnn"]["pretrained"] = False` so torchvision never downloads the VGG19
backbone at runtime.

### Offline guarantees

`docker run --network none` starts clean and serves `/health`, `/docs`,
`/static/*` and `/v1/mask`. Everything that could reach the network is pinned
locally:

- the ONNX detector is a file,
- VietOCR weights are a file (`cfg["weights"]` is a path, not an URL),
- VietOCR config is a file,
- torchvision's pretrained backbone is disabled,
- Swagger UI assets are vendored under `static/`.

To confirm, clear any cached downloads and start without network:

```bash
docker run -d --name pii --gpus all --network none pii-mask:1
docker exec pii python -c "
import urllib.request
for p in ('/health','/docs','/static/swagger-ui.css','/openapi.json'):
    print(p, urllib.request.urlopen('http://127.0.0.1:8000'+p).status)
"
docker rm -f pii
```

---

## Performance

Measured on an RTX 4050 Laptop (6 GB), inside the container, offline:

| Image | Time |
|---|---|
| Estonian ID card (9 regions) | 282–761 ms |
| Vietnamese medical form (10 regions) | 679 ms |
| 9-image sample across sources | mean 1.37 s, max 2.35 s |

Well inside the 3 s/image average budget. The first request after startup is
slower (CUDA context creation); subsequent calls reflect steady state.

## Project layout

```
ID_image_train/
├── app.py               FastAPI app (routes, lifespan, validation)
├── pipeline.py          detect → OCR → mask → payload
├── run_onnx.py          ONNX detector wrapper (letterbox, OBB decode)
├── models/              detector + VietOCR weights/config
├── static/              vendored Swagger UI assets (offline /docs)
├── utils/               image listing helpers
├── docs/                design notes
├── Dockerfile
├── pyproject.toml / uv.lock
└── candidate/           labelled evaluation set (not in the image)
```

## Known limitations

- **Unseen layouts.** The detector was trained on the provided 13 document
  types; a layout it has never seen can be missed. Text is only read from
  detected boxes, so a missed box means a missed field.
- **Non-Latin / non-Vietnamese diacritics.** VietOCR's vocabulary is Vietnamese;
  Estonian `õ`, Finnish `ä`, Spanish `ñ` may be dropped or transliterated.
- **One worker.** VRAM headroom on a 6 GB card is enough for exactly one model
  instance; run uvicorn with a single worker.
- **Image size.** The image is ~20 GB because PyTorch's CUDA wheels are bundled.
