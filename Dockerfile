FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python run_data_pipeline.py \
    && python run_model_training.py \
    && python run_knowledge_indexing.py \
    && mkdir -p database \
    && useradd --system --create-home --uid 10001 app \
    && chown -R app:app /app/database

USER app

EXPOSE 8080 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; [urllib.request.urlopen(url, timeout=3).close() for url in ('http://127.0.0.1:8080/docs', 'http://127.0.0.1:8501/_stcore/health')]" || exit 1

CMD ["python", "run_application.py"]
