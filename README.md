# Door assistant (локальный ассистент у входа)

Камера + микрофон -> распознавание лица (InsightFace) -> транскрибация (faster-whisper) -> LLM (vLLM на .49) -> озвучка (Piper).
Всё на CPU, кроме LLM. Память о людях в SQLite (/data в volume).

## Сервер (192.168.88.49, /opt/assistant)
    docker compose up -d --build
    docker logs -f door-assistant
    curl http://192.168.88.49:8060/health
    curl http://192.168.88.49:8060/persons              # кто запомнен, факты
    curl -X DELETE http://192.168.88.49:8060/persons/1  # забыть человека

OpenAI-совместимые STT/TTS: http://192.168.88.49:8060/v1 (audio/transcriptions, audio/speech). Можно указать в OpenWebUI.

## API моделей (для своего кода)
| Модель | Метод | Что даёт |
|---|---|---|
| faster-whisper small (STT) | POST /v1/audio/transcriptions, multipart file=<wav/mp3/ogg>, language=ru | {"text": ...} |
| Piper ru_RU-irina (TTS) | POST /v1/audio/speech, JSON {"input": "текст"} | wav 22050 Hz mono |
| InsightFace buffalo_l | POST /face/embed, multipart file=<jpg> | bbox, score, age, gender, embedding[512] на каждое лицо |
| то же + встроенная база | POST /face/enroll (name, file), POST /face/identify (file), GET/DELETE /persons | |
| LLM Qwen3-27B (vLLM, картинки) | http://192.168.88.49:8050/v1/chat/completions, model current-LLM | OpenAI API |

Эмбеддинги нормированы: похожесть = скалярное произведение, один человек при > 0.45.

## Клиент (машина с камерой, Windows/Linux)
    cd edge && pip install -r requirements.txt
    python client.py --server ws://192.168.88.49:8060/ws/edge --camera 0 --show
    python -m sounddevice        # список устройств, если нужен --mic / --speaker
В консоли клиента можно печатать текст вместо речи (отладка).

## Сценарий
1. Незнакомое лицо 3 кадра подряд -> "Привет! Я тебя ещё не знаю. Как тебя зовут?" -> имя вытаскивает LLM -> лицо+имя в БД.
2. Знакомое лицо -> персональное приветствие с учётом фактов -> диалог.
3. После каждой реплики LLM извлекает факты о человеке (должность, интересы, просьбы) в persons.facts.
4. Фразы "что ты видишь / посмотри / в кадре" -> кадр уходит в LLM (Qwen3 мультимодальный).
5. 40 с без лица -> сессия сбрасывается.

## Настройки (docker-compose.yml)
WHISPER_MODEL small|medium|large-v3-turbo, PIPER_VOICE ru_RU-irina|dmitri|ruslan|denis-medium, FACE_THRESHOLD 0.45.
