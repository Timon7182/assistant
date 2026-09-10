FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 curl ca-certificates nodejs npm git \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code && npm cache clean --force
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install cython numpy && pip install -r requirements.txt
COPY server.py llm_cli.py dashboard.html entrypoint.sh ./
RUN chmod +x entrypoint.sh
EXPOSE 8060
ENTRYPOINT ["./entrypoint.sh"]
