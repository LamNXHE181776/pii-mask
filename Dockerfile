FROM python:3.12-slim
LABEL authors="lam"

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY uv.lock ./

RUN pip install --no-cache uv && uv sync --frozen --no-dev
COPY app.py run_onnx.py pipeline.py ./
COPY models ./models/
COPY static ./static/
COPY utils ./utils/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]