FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEDRAG_DATA_DIR=/data \
    HOME=/home/app

WORKDIR /app

COPY requirements.txt .
COPY constraints.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x ./entrypoint.sh && useradd --create-home --uid 10001 app \
    && mkdir -p /data /home/app/.cache && chown -R app:app /data /home/app/.cache
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

ENTRYPOINT ["./entrypoint.sh"]
