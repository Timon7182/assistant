FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install cython numpy && pip install -r requirements.txt
COPY server.py entrypoint.sh ./
RUN chmod +x entrypoint.sh
EXPOSE 8060
ENTRYPOINT ["./entrypoint.sh"]
