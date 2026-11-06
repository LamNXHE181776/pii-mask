# Pipeline: PII masking on document images

Design doc for the "Che thong tin ca nhan tren anh giay to" assignment.
Deadline: 09:00 Sunday 2026-11-08.

Goal: `POST /v1/mask` with a JPEG/PNG, return the same-size image with every PII
region filled solid, plus the list of masked regions (`category`, `polygon`,
`text`). Offline container, 1 GPU, <= 3 s/image average, unseen document types
must still work.

---

## 1. Why not a single detector

A layout-agnostic detector trained on these 2,923 images will memorise the
canvases of the 13 known `doc_type`s. The grader explicitly tests document types
that are not in the dataset, so the model must not depend on remembering where
the name field sits on an Estonian ID card.

But it also cannot be purely OCR + rules: rules alone cannot tell a person name
("Tuul Maria") from a printed hospital name ("BENH VIEN DA KHOA THU DUC") on a
layout it has never seen.

So the pipeline is two branches and a fusion step:

- **recall branch** - YOLO segmentation over all PII categories. Fires even when
  OCR fails (blur, glare, crop). Layout-sensitive but only ever *adds* masks.
- **precision branch** - OCR every text line, then classify each line with
  regex + multilingual label anchors + a text classifier trained on the 74k
  labelled field strings. Layout-independent, and it is the branch that decides
  *not* to mask printed labels.
- **fusion** - take the union, then let the precision branch veto anything that
  is clearly a printed label. Output quads, fill solid.

```
            ┌─────────────────────────────── input bytes ───────────────────────────────┐
            ▼                                                                           │
   ┌─────────────────┐                                                                  │
   │ 0. validate     │  magic bytes + PIL.verify + EXIF transpose + decode              │
   │    decode       │  400 on anything not a decodable image                            │
   └────────┬────────┘                                                                  │
            ▼                                                                           │
   ┌──────────────────────────────┐        ┌──────────────────────────────┐             │
   │ 1a. OCR  (PP-OCRv5)          │        │ 1b. PII seg (YOLO26n-seg)    │             │
   │  det + rec, lang=vi (Latin)  │        │  12 PII classes + non_pii     │             │
   │  -> quads + text + score     │        │  -> instance masks/quads      │             │
   └──────────┬───────────────────┘        └──────────────┬───────────────┘             │
              ▼                                           │                             │
   ┌──────────────────────────────────┐                 │                             │
   │ 2. classify each text line       │                 │                             │
   │    a) normalise (NFC, case)      │                 │                             │
   │    b) regex: date / phone /      │                 │                             │
   │       email / IBAN / ID / MRZ    │                 │                             │
   │    c) anchor dict -> label spans │                 │                             │
   │    d) text classifier (ONNX)     │                 │                             │
   └──────────┬───────────────────────┘                 │                             │
              ▼                                         │                             │
   ┌──────────────────────────────────┐                 │                             │
   │ 3. anchor - value association    │                 │                             │
   │    value = same line right of    │                 │                             │
   │    label, else block below       │                 │                             │
   └──────────┬───────────────────────┘                 │                             │
              ▼                                         ▼                             │
   ┌────────────────────────────────────────────────────────────┐                      │
   │ 4. fuse: union of quads, drop label-only regions,           │                      │
   │    IoU-dedupe, expand 2 px, clip to image                   │                      │
   └──────────┬─────────────────────────────────────────────────┘                      │
              ▼                                                                         │
   ┌──────────────────────────────┐                                                     │
   │ 5. mask: fillPoly solid      │  JPEG q=92 -> base64                                │
   │    + build regions[]         │  processing_time_ms                                 │
   └──────────┬───────────────────┘                                                     │
              ▼                                                                         │
        JSON response ────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage details

### 0. Input handling

- `PIL.Image.open` inside `try/except`, then `img.verify()`, then re-open to
  actually load (verify() leaves the object unusable). Any failure -> HTTP 400,
  never a 500.
- `ImageOps.exif_transpose` so EXIF-rotated phone photos are upright and the
  returned polygons match the returned pixels. Return `width`/`height` of the
  transposed image.
- Coerce to RGB, reject `DecompressionBomb` (catch `Image.DecompressionBombError`,
  keep PIL's default limit).
- Keep the original resolution for the output. Run inference on a resized copy
  (long side 1600) and scale quads back by `1/scale`.

### 1a. OCR branch

`PaddleOCR(lang="vi", use_doc_orientation_classify=True, use_doc_unwarping=True,
use_textline_orientation=True)`.

- `lang="vi"` gives Latin + full Vietnamese diacritics; the other sources
  (Estonian, Finnish, Spanish, Indonesian, English) are covered by the same
  Latin model.
- Turn on textline orientation: KTP photos are often 90 deg off.
- Keep `rec_polys` (quads) not `dt_polys` boxes - the spec wants a 4-point
  polygon and the quad already hugs the tilted text.
- Discard lines with `rec_score < 0.5` from *classification* but keep their
  geometry in a "unknown text" bucket - it may still overlap a YOLO PII region.
- Your `output/*_res.json` already has this exact shape (`rec_texts`,
  `rec_polys`, `rec_scores`), so stage 1a is done; only the caching remains.

### 1b. PII segmentation branch

Train `yolo26n-seg` on the 2,923 images with **all 13 categories** as classes
(the dataset is a segmentation dataset already - polygons, not boxes).

- Convert each label JSON to YOLO-seg format: `class x1 y1 x2 y2 x3 y3 x4 y4`
  normalised by `width`/`height`, clipped to the image.
- Keep `non_pii` as a real class (class 12). It teaches the model the difference
  between a printed label and a filled value, which is exactly the failure mode
  that causes over-masking.
- `ambiguous` (573 regions) -> train as PII; at inference decide per the fusion
  rule in stage 4.
- Split **by `doc_type`, stratified by `source` and `capture`** so validation
  measures generalisation, not memorisation. Hold out whole doc types: e.g.
  `est_id`, `alb_id`, `idn_ktp`, `us_az_dl`, `giay_ra_vien`, `don_yeu_cau`
  never appear in train.
- ~2,900 images is plenty for a 3M-parameter nano model. `imgsz=640`,
  `epochs=100`, `patience=20`, `mosaic=1.0` then `close_mosaic=10`.

Note the class imbalance: `non_pii` 48,025 vs `email` 164. Do **not** resample
per class; instead rely on `cls` loss with default weights, and evaluate
per-category recall so you can see the tail.

### 2. Text-line classification

Four signals, applied in this order, first confident hit wins but all are
recorded for the audit log:

**(a) Normalisation.** NFC, strip, collapse whitespace. Keep the original string
for the `text` field in the response; classify the normalised one.

**(b) Regex.** High precision, cheap, language agnostic:

| category | pattern (after normalisation) |
|---|---|
| `email` | `[\w.+-]+@[\w-]+\.[\w.]+` |
| `phone_number` | `(\+?\d[\d\s().-]{7,}\d)` and 9-12 digits after stripping separators |
| `date_of_birth` | `\d{1,2}[./-]\d{1,2}[./-]\d{2,4}`, `\d{4}[./-]\d{1,2}[./-]\d{1,2}`, `\d{1,2}\s+(jan|feb|...)\w*\s+\d{4}` |
| `bank_account` | `\d{8,20}` with separators, or IBAN `[A-Z]{2}\d{2}[A-Z0-9]{11,30}` |
| `mrz` | `^[A-Z0-9<]{28,44}$` and ratio of `<` > 0.1 |
| `id_number` | per-country patterns (EE `\d{11}`, FI `\d{6}[-+A]\d{3}[0-9A-Y]`, ID `\d{16}`, ES DNI `\d{8}[A-Z]`, passport `[A-Z]{1,2}\d{6,9}`) |

Dates need one extra rule: a date is `date_of_birth` only if the line (or the
anchor next to it) says birth/sinh/syntymä/sünni/geboren/fecha de nacimiento.
Otherwise it is `non_pii` (issue/expiry dates must stay visible per the spec).

**(c) Label anchors.** A dictionary from printed label -> category, in all six
languages present, then used geometrically:

```
person_name    ho va ten | ho ten | ten | name | nimi | eesnimi | perekonnanimi |
               nombre | apellido | nama | patient | benh nhan
date_of_birth  ngay sinh | sinh ngay | date of birth | sünnikuupäev | syntymäaika |
               fecha de nacimiento | tanggal lahir
place_of_birth noi sinh | place of birth | sünnikoht | synnyinpaikka | lugar de nacimiento
id_number      so cccd | so cmnd | so ho chieu | passport no | isikukood | henkilötunnus |
               dni | nie | nik | so the | ma the bhyt | ma so ca nhan
phone_number   dien thoai | sdt | tel | puhelin | telepon | telefono
email          email | e-mail | thu dien tu | sähköposti
address        dia chi | address | aadress | osoite | dirección | alamat
bank_account   so tai khoan | stk | account no | iban | rekening
```

Association: for each anchor span, look for a value on the same line to its
right (gap < 1.5x line height); if none, look at the nearest line below within
the same column (x-overlap > 50%, vertical gap < 2.5x line height). Once
associated, everything in the value's x-range is that category, and the anchor
itself is forced `non_pii`.

This is what prevents masking "Họ và tên:" itself - the anchor words are
excluded from the value span.

**(d) Text classifier.** Fine-tune a small multilingual encoder on the 74k
labelled strings: `distilbert-base-multilingual-cased` or
`microsoft/mdeberta-v3-xsmall`, 13-way softmax, then export to ONNX (int8) for
serving. Input: the normalised line text, plus a 3-token prefix of the nearest
left anchor category if one exists (cheap context without a full layout model).

Training data: every field in every label with `text != null` -> `(text,
category)`. Dedupe identical (text, category) pairs. Add OCR-noise augmentation
on a copy of the training set: drop diacritics, `0/O`, `1/l/I`, `5/S`,
`rn/m`, random char dropout at 5%. That is what the model will actually see at
runtime, and it costs nothing to generate.

Do **not** train on `face` / `signature` / `mrz` text (they are `null` or
pattern-only); mask those three categories off the geometry, not the text.

### 3. Fusion rules

```
regions = yolo_quads(score >= 0.35) U ocr_quads(classified as PII)

# precision veto: an OCR quad classified non_pii removes only OCR-derived
# regions; it never removes a YOLO face/signature/MRZ detection
for q in ocr_quads:
    if q.category == "non_pii" and q overlaps a region and q covers > 60% of it:
        drop that region

# ambiguous: mask only when adjacent to a PII region (<= 1 line height),
# otherwise leave visible - the spec penalises blanket masking
# label guard: drop any region whose text matches an anchor string exactly
# dedupe: greedy IoU >= 0.7 keep the higher-confidence quad
# expand: 2 px outward, clip to image bounds
```

Two failure modes this ordering fixes, both visible in your data:

1. `idnet`/`ktp`/`midv2020` labels only cover data fields, not printed text.
   If you trust OCR labels alone you will mask nothing on those documents. The
   YOLO branch supplies the geometry, the OCR branch supplies the "is this a
   label" veto.
2. `vn_synth` labels cover *every* text span including `non_pii`. There the OCR
   branch is authoritative and the YOLO branch is only a safety net.

### 4. Masking and response

- Solid fill, no blur (spec): `cv2.fillPoly(img, [quad], (0, 0, 0))`.
  Optionally fill with the median colour of a 3 px ring around the quad, which
  keeps the document readable; black is the safe default if you are unsure
  what the grader values.
- Encode JPEG q=92, base64, put in `masked_image`.
- `regions[]`: `category`, `polygon` (4 points, floats, pixel coords of the
  returned image), `text` (the OCR string, or `null` for `face`/`signature`/
  unreadable).
- `processing_time_ms` measured from decoded-request to encoded-response.

### 5. Evaluation harness (build this before the model)

Three metrics, on a held-out split of whole doc types:

| metric | definition | target |
|---|---|---|
| **PII leak** | % of GT PII quads whose area is < 90% covered by predicted masks | < 2% |
| **category F1** | matched predictions (IoU >= 0.5) with correct category | > 0.90 |
| **over-mask** | % of `non_pii` quad area covered by a predicted mask | < 5% |

Leak is the safety metric and the one the grader will feel; over-mask is what
"không được che tràn lan" means. Report them per category and per source, and
run latency p50/p95.

Also keep 20-30 hand-picked hard cases: tilted KTP photo, glare on a passport
MRZ, a doc_type you held out, a screenshot with a tiny signature. Those are the
grading scenarios.

---

## 3. Repository layout

```
service/
  app.py                 # FastAPI: /health, /v1/mask, /docs
  pipeline.py            # stages 0-5, one pure function mask(image_bytes) -> dict
  ocr.py                 # PP-OCR wrapper + cache
  seg.py                 # YOLO/ONNX PII segmenter
  classify.py            # regex + anchors + classifier, returns per-quad category
  anchors.py             # multilingual label dictionary
  render.py              # fillPoly + base64
  models/                # yolo26n-seg-pii.onnx, text-clf.onnx, ocr det/rec models
train/
  labels_to_yolo.py      # candidate/labels -> yolo-seg dataset
  build_text_dataset.py  # candidate/labels -> (text, category) csv + noise aug
  train_seg.py
  train_text_clf.py      # transformers Trainer, then export onnx
  eval.py                # the three metrics above
Dockerfile
README.md
docs/pipeline.md         # this file
```

Keep `pipeline.py` free of FastAPI so `eval.py` imports it directly - same code
path in evaluation and in production, or the numbers are fiction.

---

## 4. Serving and packaging

- FastAPI + uvicorn, `--host 0.0.0.0 --port 8000`, one worker.
- Load models at startup (FastAPI `lifespan`), **before** `/health` returns ok.
  The spec says `/health` -> 200 only when ready, so a warmup inference on a
  dummy image in the lifespan is required.
- Inference is not thread-safe on one GPU: wrap the pipeline in a single
  `asyncio.Lock` (or a 1-slot queue) and run it in a threadpool
  (`run_in_threadpool`). Fine for a grader that calls sequentially; a 3-slot
  semaphore if you want throughput.
- `/docs` is free with FastAPI; add `response_model` + `examples=` so the
  Swagger UI shows sample requests and the exact 200/400 schemas.
- Docker: `nvidia/cuda:13.0-cudnn-runtime` base or `python:3.12-slim` + paddle
  wheel from the cu130 index (your `pyproject.toml` already pins
  `paddlepaddle-gpu==3.3.1`). Copy `models/` into the image; no runtime download.
  `ENTRYPOINT ["uvicorn", "service.app:app", "--port", "8000"]`.

**ONNX for the YOLO branch.** Two CUDA stacks (paddle cu130 + torch cu13x) in one
image is ~10 GB and fragile. Export the trained YOLO26n-seg to ONNX and serve it
with `onnxruntime-gpu`, so the container ships paddle + onnxruntime only. Weights
are baked in, which also satisfies "download during build is allowed".

---

## 5. Latency budget (RTX 4050 laptop, 1240x1600 input)

| stage | est. |
|---|---|
| decode + EXIF + resize | 40 ms |
| PP-OCR det | 150 ms |
| PP-OCR rec (~40 lines) | 60 ms |
| YOLO-seg (ONNX, 640) | 15 ms |
| text classifier (~40 lines, batched) | 10 ms |
| fusion + fill + JPEG encode | 45 ms |
| **total** | **~0.32 s** |

Comfortably under 3 s. The 4050 has 6 GB, so stay at `imgsz=640` for seg and
1600 long side for OCR; do not upscale.

---

## 6. Plan (4.5 days to deadline)

| day | work |
|---|---|
| Tue 03 | `labels_to_yolo.py` + `build_text_dataset.py`; freeze held-out doc types; OCR cache over all 2,923 images |
| Wed 04 | train YOLO-seg (13 classes); train text classifier; `eval.py` on the held-out split |
| Thu 05 | anchors + regex + fusion; tune the three metrics on hard cases; per-category error review |
| Fri 06 | FastAPI service, `/docs`, error paths, warmup; ONNX export; Docker build |
| Sat 07 | end-to-end test with the grader's scenarios (unseen doc types, corrupt files, big/small images); latency profiling; write the flow doc |
| Sun 08 | 09:00 submit - freeze Friday evening if possible |

## 7. Risks and mitigations

| risk | mitigation |
|---|---|
| Detector memorises the 13 canvases | hold out whole doc types; the OCR branch carries unseen layouts |
| OCR fails on blur/glare/rotation | YOLO recall net; orientation classifier; report unknown-text regions rather than dropping them |
| Over-masking printed labels | anchor dictionary + `non_pii` class + label veto in fusion |
| `ambiguous` (initials, country codes) | mask only when adjacent to a PII region |
| Python/venv split (3.12 paddle vs 3.14 torch) | train anywhere, serve one 3.12 image with ONNX |
| Grader sends PDF/HEIC/1x1 px | magic-byte check + PIL verify -> 400; never 500 |
| Label noise in `candidate/` ("Nhãn có thể có lỗi") | review the worst 50 errors by hand before trusting eval numbers |
