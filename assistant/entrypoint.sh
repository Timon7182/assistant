#!/bin/sh
set -e
mkdir -p /data/piper /data/models /data/insightface
if [ ! -f "/data/piper/${PIPER_VOICE}.onnx" ]; then
  echo "downloading piper voice ${PIPER_VOICE}"
  python -m piper.download_voices --data-dir /data/piper "${PIPER_VOICE}"
fi
exec uvicorn server:app --host 0.0.0.0 --port 8060 --ws-max-size 16777216
