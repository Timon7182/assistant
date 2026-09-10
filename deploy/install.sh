#!/bin/bash
# Первичная установка на Ubuntu: sudo bash deploy/install.sh  (репозиторий уже склонирован в /opt/assistant)
set -euo pipefail
cd /opt/assistant
[ -f .env ] || { cp .env.example .env; echo "заполните /opt/assistant/.env"; }
chmod +x deploy/deploy.sh
cp deploy/assistant-deploy.service deploy/assistant-deploy.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now assistant-deploy.timer
docker compose up -d --build
echo "готово: docker logs -f door-assistant"
