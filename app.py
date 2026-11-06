"""FastAPI service for PII masking.

Run it with uvicorn:

    uvicorn app:app --host 0.0.0.0 --port 8000

Then:
    GET  http://127.0.0.1:8000/health
    POST http://127.0.0.1:8000/v1/mask     (multipart/form-data, field `file`)
    GET  http://127.0.0.1:8000/docs        (Swagger UI, generated automatically)

The models is loaded once at startup and kept on ``app.state.pipeline``; loading
per request would take seconds and is the most common FastAPI mistake here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

from pipeline import OnnxOCRPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX detector and VietOCR once, when the server starts."""
    app.state.pipeline = OnnxOCRPipeline()
    yield
    app.state.pipeline = None


def create_app(pipeline=None) -> FastAPI:
    """Build the app. ``pipeline`` is only passed in tests to skip models loading."""
    app = FastAPI(
        title="PII masking",
        version="1.0.0",
        description="Detect, read and mask personal data on document images.",
        lifespan=lifespan,
        docs_url=None,
    )
    app.mount("/static", StaticFiles(directory="static"), name="static")

    if pipeline is not None:
        app.state.pipeline = pipeline

    @app.get("/health", summary="Liveness check")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/docs", include_in_schema=False, summary="Swagger UI")
    def swagger_ui_docs():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title,
            swagger_js_url="static/swagger-ui-bundle.js",
            swagger_css_url="static/swagger-ui.css",
        )

    @app.post("/v1/mask", summary="Mask all PII in one image")
    def mask(file: UploadFile = File(...)) -> dict:
        data = file.file.read()
        image = decode_image(data)

        try:
            return app.state.pipeline.run_image(image)
        except HTTPException:
            raise
        except Exception as exc:  # never let a bad image take the service down
            raise HTTPException(status_code=500, detail=f"masking failed: {exc}") from exc

    return app


def decode_image(data: bytes) -> np.ndarray:
    """Validate and decode upload bytes into a BGR image, or raise 400."""
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="file is not a valid image")
    return image


app = create_app()
