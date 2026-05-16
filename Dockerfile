FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "voyage_api.app:app"]
