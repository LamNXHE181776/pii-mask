# Async / concurrent serving — future project

Status: design note only, not implemented.
Scope: turning the current synchronous PII-masking pipeline into a service that
handles concurrent requests on one GPU without blowing the 3 s/image budget.

---

## 1. Why this is a separate project

The v1 pipeline is deliberately synchronous:

```
decode → ONNX detect (CPU) → for each box: warp crop → VietOCR (GPU) → mask → JPEG
```

Everything runs on the request thread, in sequence. That is fine for a single
caller and for evaluation, but the API contract is `POST /v1/mask`, and a grader
or a front end will not call it one request at a time.

Current per-request work, roughly:

| Stage | Time | Resource |
|---|---|---|
| decode + EXIF | ~5 ms | CPU |
| ONNX detect (640, 300 dets) | ~25–40 ms | CPU |
| N × perspective warp | ~1 ms each | CPU |
| N × VietOCR (N ≈ 10–15) | ~10–20 ms each (GPU), ~100–250 ms (CPU) | GPU |
| mask + JPEG encode | ~10 ms | CPU |

So a single image is 0.3–0.6 s on GPU, dominated by the serial OCR loop. Ten
concurrent requests do not take 10× longer only because the GPU has headroom —
they take 10× longer *and* the OCR calls interleave badly.

## 2. Goals / non-goals

Goals
- p95 ≤ 1.5 s for a single document on one GPU, with ≥ 8 concurrent in-flight.
- No image data leaves the host.
- No request can starve another or exhaust VRAM.
- Same response schema as v1 (`/v1/mask` unchanged).

Non-goals
- Multi-GPU / multi-node scheduling.
- Auth, rate limiting per tenant, persistence.
- Changing the model or the masking rules.

## 3. Bottleneck analysis (fill in before designing further)

Measure first, with a load test at concurrency 1 / 4 / 8 / 16:

1. GPU utilisation (should plateau near 100% if OCR is the bound).
2. Queue wait vs compute time per request.
3. OCR calls per image distribution (p50, p99) — the tail matters more than the mean.
4. Detector time with varying `ORT_NUM_THREADS`.

If GPU utilisation is already ~100% at concurrency 4, batching is the only win.
If it is low and latency still grows, the problem is thread contention, not GPU.

## 4. Options

### A. `async def` + bounded ThreadPoolExecutor (smallest change)

- Endpoints become `async def`, the pipeline call is wrapped in
  `await loop.run_in_executor(pool, fn)`.
- Pool size 1–2, so GPU work is serialised but the event loop stays responsive.
- Model objects stay process-global; one lock per model.

Pros: one day of work, keeps the code shape, removes head-of-line blocking of
`/health`. Cons: throughput equals 1/(serial time); no batching; queue depth is
the thread pool's.

### B. Scheduler + dynamic batching (recommended)

- FastAPI enqueues jobs into an `asyncio.Queue`; a fixed number of worker tasks
  (1 GPU worker) pull batches.
- The GPU worker collects up to `MAX_BATCH` requests or waits `BATCH_WINDOW_MS`
  (20–40 ms), then runs the detector once for the batch and VietOCR once for all
  crops of all images.
- Results are delivered through `asyncio.Future` per request.

Pros: near-linear throughput scaling up to GPU saturation; bounded VRAM; single
place to add timeouts/cancellation. Cons: added latency = batching window (20 ms,
acceptable against 3 s); more complex failure handling.

### C. Separate inference process / Triton / ONNX Runtime Server

- FastAPI becomes a thin client; model server does batching itself.

Pros: standard tooling, health/metrics for free. Cons: two runtimes in the image
(again), Triton config for an OBB model with baked NMS is awkward, and the
offline/no-internet Docker constraint gets harder. Only worth it at a much larger
scale.

### D. Async job API (202 + poll)

- Not a replacement for A/B, an addition: bulk uploads return a job id.

## 5. Recommended shape

Adopt **B**, implemented in three phases.

### Phase 1 — stop blocking (½–1 day)

- `async def mask(...)`, `await run_in_executor(gpu_pool, pipeline.run, ...)`.
- `gpu_pool = ThreadPoolExecutor(max_workers=1)`; everything touching a torch
  model goes through it.
- cv2 and ONNX (CPU) may run on a separate CPU pool.
- Keep `/health` trivial and always fast.

### Phase 2 — scheduler + batching (3–5 days)

- `Request → asyncio.Queue(maxsize=64)`.
- One GPU worker task; loops: pop one job, wait up to `BATCH_WINDOW_MS` for more,
  run batched detect, gather all crops, run batched OCR, fan results back.
- Detector needs a **dynamic-batch ONNX export**
  (`dynamic={"batch": "batch"}`, keep `nms=True`) — the current graph is fixed at
  batch 1.
- VietOCR batch predict: pad crops to a common max height/width, keep a boolean
  mask of valid pixels, then unpad per item. Verify with a batch of 16 crops that
  texts match single-image results.
- Cancellation: if the client disconnects, mark the future cancelled; the worker
  skips it before inference.

### Phase 3 — isolation (optional)

- Split detector (CPU) and OCR (GPU) into separate worker tasks with their own
  queues, so a slow detector batch cannot delay OCR (and vice versa).
- Consider warming both models and pinning `ORT_NUM_THREADS` to physical cores.

## 6. Thread-safety rules (do not violate)

| Object | Concurrent forward? | Rule |
|---|---|---|
| `onnxruntime.InferenceSession` | yes | share freely; it releases the GIL |
| `torch.nn.Module` (VietOCR) | **no** | one worker, or a lock |
| `Predictor` (VietOCR wrapper) | **no** | one instance per worker |
| `cv2` operations | yes | they release the GIL; keep them off the event loop |
| `PIL.Image` | per-item only | do not share |

Never call model inference directly from an `async def` body — that blocks the
loop and serialises everything, including `/health`.

## 7. Backpressure and failure

- Queue cap (64). When full, return `503` with `Retry-After` — never block
  indefinitely.
- Per-request wall-clock timeout (e.g. 10 s); on timeout cancel the future and
  return `504` (the spec only requires no crash).
- Batch failure isolation: one bad image must not fail the batch — catch per
  item, return a 400-equivalent for that item only.
- VRAM guard: refuse a batch whose total crop count exceeds `MAX_CROPS`
  (e.g. 256); split into sub-batches.
- Graceful shutdown: drain the queue with a deadline, then cancel workers.

## 8. API evolution

- `POST /v1/mask` keeps its exact schema; only latency/throughput change.
- Optional `POST /v1/mask/batch` (multipart multiple files) → list of results,
  no new semantics.
- Optional job API for very large inputs: `202 {job_id}` + `GET /v1/jobs/{id}`.

## 9. Observability

- Structured log per request: queue wait, detect ms, ocr ms, mask ms, batch size.
- Prometheus: `queue_depth`, `batch_size`, `gpu_util`, `p95_latency`,
  `ocr_calls_per_image`.
- The existing `processing_time_ms` response field becomes the sum of stages;
  keep it accurate.

## 10. Testing plan

- Load test: 50 documents, concurrency 1/2/4/8/16, report throughput and
  p50/p95/p99.
- Determinism: same image, concurrency 1 vs 8 → identical `regions` (order
  normalised by polygon position).
- Soak: 30 min at concurrency 8, watch RSS and VRAM for growth (torch caching
  allocator + PIL leaks are the usual suspects).
- Cancellation: abort 20% of requests mid-flight; service must stay healthy.

## 11. Risks

- **Batching hurts tail latency** if the window is too large; keep ≤ 40 ms.
- **VietOCR padding changes output** for short crops — verify per-item parity
  before enabling batching.
- **Two CUDA consumers** (if Paddle ever returns) will reintroduce the driver
  problem this project deliberately avoided.
- **Thread pool + torch** can silently oversubscribe CPU; pin thread counts
  (`OMP_NUM_THREADS`, `torch.set_num_threads`).

## 12. Milestones

| Phase | Deliverable | Effort |
|---|---|---|
| 0 | Load test + baseline numbers | ½ day |
| 1 | async endpoints + executor, `/health` unblocked | 1 day |
| 2 | queue + worker + batched detect/OCR | 4 days |
| 3 | metrics, backpressure, cancellation | 2 days |
| 4 | soak + load validation, docs | 1 day |

Phase 1 is worth doing even if 2–4 are deferred: it is small, it removes the
worst failure mode (one slow image blocking every request), and it does not
change the API. The current `OnnxOCRPipeline` is a good starting point — it just
needs splitting into `detect_batch()` and `recognize_batch()` with no mutable
per-request state on `self`.
