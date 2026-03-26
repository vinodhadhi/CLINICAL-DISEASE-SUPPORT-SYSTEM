FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["/bin/sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
