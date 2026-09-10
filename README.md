# Door assistant (ассистент у входа)

Камера + микрофон -> лицо (InsightFace) -> речь в текст (faster-whisper) -> LLM-агент с инструментами -> озвучка (Piper).
Модели на CPU. LLM: Claude через `claude` CLI (токен) или любой OpenAI-совместимый сервер (vLLM). Память, визиты, поручения в SQLite (/data в volume).

## Что умеет
- Незнакомое лицо -> "Как тебя зовут?" -> запоминает лицо и имя.
- Знакомое лицо -> приветствие по имени, передаёт накопленные для него сообщения, помнит факты.
- Инструменты LLM: `leave_message` ("передай Данияру ..."), `remember` ("запиши ..."), `who_visited` ("кто заходил?"), `list_people`, `end_session` ("выключись").
- Каждый визит: кадр, транскрипт, вызовы инструментов, краткое резюме. Дашборд на `/` (Basic auth).
- Кадры и транскрипты старше RETENTION_DAYS удаляются.

## Развёртывание (Ubuntu + docker)
    sudo git clone https://github.com/Timon7182/assistant.git /opt/assistant
    cd /opt/assistant && sudo cp .env.example .env && sudo nano .env     # API_KEY, DASH_PASSWORD, CLAUDE_CODE_OAUTH_TOKEN
    sudo bash deploy/install.sh
`install.sh` ставит systemd-таймер `assistant-deploy.timer`: раз в минуту проверяет GitHub и при новом коммите в main пересобирает контейнер (лог в `deploy.log`).

Токен Claude: на своей машине `claude setup-token`, полученный `sk-ant-oat...` вставить в `CLAUDE_CODE_OAUTH_TOKEN`. Без токена и без LLM_URL ассистент работает по шаблонам (визиты и лица пишутся, инструменты нет).

## API (заголовок `X-API-Key: <API_KEY>` или Basic auth)
| Что | Вызов | Ответ |
|---|---|---|
| STT faster-whisper | `POST /v1/audio/transcriptions` multipart `file`, `language=ru` | `{"text": ...}` |
| TTS Piper | `POST /v1/audio/speech` JSON `{"input": "текст"}` | wav 22050 Hz |
| Лица (чистая модель) | `POST /face/embed` multipart `file` | bbox, score, age, gender, embedding[512] |
| Лица с базой | `POST /face/enroll` (name, file), `POST /face/identify` (file) | |
| Данные дашборда | `GET /api/visits`, `/api/visits/{id}`, `/api/people`, `/api/tasks`, `/api/notes`, `/api/actions` | JSON |
| Edge | `WS /ws/edge?key=API_KEY` | см. docstring в server.py |

Эмбеддинги нормированы: похожесть = скалярное произведение, один человек при > 0.45.

## Клиент у двери (Windows/Linux, камера + микрофон + колонка)
    cd edge && pip install -r requirements.txt
    python client.py --server ws://HOST:8060/ws/edge --key API_KEY --camera 0 --show
В консоли клиента можно печатать текст вместо речи (отладка).

## Настройки (.env)
`WHISPER_MODEL` tiny|base|small|medium, `PIPER_VOICE` ru_RU-irina|dmitri|ruslan|denis-medium, `FACE_THRESHOLD`, `CLAUDE_MODEL` sonnet|opus, `MEM_LIMIT`, `CPU_LIMIT`.
