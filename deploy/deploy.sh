#!/bin/bash
# Автодеплой: подтягивает main из GitHub и пересобирает контейнер, если появились новые коммиты.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=deploy.log
git fetch -q origin main
LOCAL=$(git rev-parse HEAD); REMOTE=$(git rev-parse origin/main)
if [ "$LOCAL" = "$REMOTE" ] && docker ps --format '{{.Names}}' | grep -q '^door-assistant$'; then
  exit 0
fi
echo "$(date -Is) deploying $LOCAL -> $REMOTE" >> $LOG
git reset -q --hard origin/main
if docker compose up -d --build >> $LOG 2>&1; then
  echo "$(date -Is) ok $(git rev-parse --short HEAD)" >> $LOG
  docker image prune -f > /dev/null 2>&1 || true
else
  echo "$(date -Is) FAILED" >> $LOG
fi
