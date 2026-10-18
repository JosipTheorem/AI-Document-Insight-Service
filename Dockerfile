FROM python:3.13-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app

# OCR uses headless OpenCV; only the OpenMP runtime is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install -r requirements.txt \
    && python -m pip check \
    && python -c "from rapidocr import RapidOCR; RapidOCR(params={'Global.log_level': 'warning'})"

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/runtime /app/app_logging/logs \
    && chown -R app:app /app/data/runtime /app/app_logging/logs

# Notebook entry point; dependencies are shared with the application.
FROM base AS evaluation
RUN python -m ipykernel install --name abysalto --display-name "Python (abysalto)"
COPY --chown=app:app . .
USER app
EXPOSE 8888
CMD ["python", "-m", "jupyterlab", "evaluation/evaluation.ipynb", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

FROM base AS runtime
COPY --chown=app:app . .
USER app
EXPOSE 8000 8501 8502
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
