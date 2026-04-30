FROM python:3.11-slim AS gateway

WORKDIR /srv

COPY pyproject.toml ./
RUN pip install --no-cache-dir . 2>/dev/null; \
    pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
