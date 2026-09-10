"""Door assistant: camera (face id) + mic (STT) -> LLM agent with tools -> TTS. Models run on CPU.

Edge client connects to /ws/edge?key=API_KEY and sends:
  text frame  {"type":"frame","jpeg":"<base64>"}   ~2 fps
  text        {"type":"text","text":"..."}          debug input without a mic
  binary      int16 PCM mono 16 kHz, 480 samples (30 ms) per message
Server sends: {"type":"say","text","wav"}, {"type":"heard","text"}, {"type":"status",...}, {"type":"end"}

HTTP API (X-API-Key header, Bearer token, or Basic auth with DASH_USER/DASH_PASSWORD):
  /v1/audio/transcriptions, /v1/audio/speech   OpenAI-compatible STT/TTS
  /face/embed, /face/identify, /face/enroll     InsightFace buffalo_l
  /api/visits, /api/people, /api/tasks, /api/notes, /api/actions   dashboard data
  /                                             dashboard (Basic auth)
"""
import asyncio, base64, hmac, io, json, logging, os, re, secrets, sqlite3, threading, time, wave
from collections import deque
from datetime import datetime, timedelta

import cv2, httpx, numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

log = logging.getLogger("assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ----------------------------------------------------------------- config
API_KEY = os.getenv("API_KEY", "")
DASH_USER = os.getenv("DASH_USER", "admin")
DASH_PASSWORD = os.getenv("DASH_PASSWORD", API_KEY)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
CLAUDE_OAUTH = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("claude-cli" if CLAUDE_OAUTH else "anthropic" if ANTHROPIC_API_KEY else "openai")
LLM_URL = os.getenv("LLM_URL", "http://192.168.88.49:8050/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "current-LLM")
LLM_KEY = os.getenv("LLM_API_KEY", "none")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
PIPER_VOICE = os.getenv("PIPER_VOICE", "ru_RU-irina-medium")
FACE_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.45"))
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Ассистент")
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "90"))
DATA = os.getenv("DATA_DIR", "/data")
SESSION_TIMEOUT = 40      # seconds without a face -> visit closed
SR = 16000
if not API_KEY:
    log.warning("API_KEY is empty: API and websocket are open to everyone")

# ----------------------------------------------------------------- models
log.info("loading whisper %s", WHISPER_MODEL)
from faster_whisper import WhisperModel
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", download_root=f"{DATA}/models")
whisper_lock = threading.Lock()

log.info("loading piper %s", PIPER_VOICE)
from piper import PiperVoice
voice = PiperVoice.load(f"{DATA}/piper/{PIPER_VOICE}.onnx")
piper_lock = threading.Lock()

log.info("loading insightface buffalo_l")
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(name="buffalo_l", root=f"{DATA}/insightface", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))
face_lock = threading.Lock()

import webrtcvad
log.info("models ready, llm provider=%s", LLM_PROVIDER)

# ----------------------------------------------------------------- storage
DB_PATH = f"{DATA}/assistant.sqlite"
SNAP_DIR = f"{DATA}/snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)
db_lock = threading.Lock()


def db():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS persons(id INTEGER PRIMARY KEY, name TEXT, facts TEXT DEFAULT '[]', role TEXT DEFAULT 'guest',
        created TEXT, last_seen TEXT, visits INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS embeddings(id INTEGER PRIMARY KEY, person_id INTEGER, vec BLOB);
    CREATE TABLE IF NOT EXISTS visits(id INTEGER PRIMARY KEY, person_id INTEGER, started TEXT, ended TEXT, summary TEXT, snapshot TEXT);
    CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, person_id INTEGER, visit_id INTEGER, role TEXT, content TEXT, ts TEXT);
    CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, from_person INTEGER, from_name TEXT, to_person INTEGER, to_name TEXT,
        text TEXT, status TEXT DEFAULT 'pending', created TEXT, delivered_at TEXT);
    CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, person_id INTEGER, person_name TEXT, text TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS actions(id INTEGER PRIMARY KEY, visit_id INTEGER, person_id INTEGER, tool TEXT, args TEXT, result TEXT, ts TEXT);
    """)
    cols = [r["name"] for r in c.execute("PRAGMA table_info(messages)")]
    if "visit_id" not in cols:
        c.execute("ALTER TABLE messages ADD COLUMN visit_id INTEGER")


def now():
    return datetime.now().isoformat(timespec="seconds")


def q(sql, *args):
    with db() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def ex(sql, *args):
    with db_lock, db() as c:
        return c.execute(sql, args).lastrowid


class FaceIndex:
    def __init__(self):
        self.reload()

    def reload(self):
        rows = q("SELECT person_id, vec FROM embeddings")
        self.ids = np.array([r["person_id"] for r in rows], dtype=np.int64)
        self.vecs = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 512), np.float32)

    def match(self, emb):
        if len(self.ids) == 0:
            return None, 0.0
        sims = self.vecs @ emb
        i = int(np.argmax(sims))
        return (int(self.ids[i]) if sims[i] >= FACE_THRESHOLD else None), float(sims[i])

    def add(self, person_id, emb):
        ex("INSERT INTO embeddings(person_id, vec) VALUES(?,?)", person_id, emb.astype(np.float32).tobytes())
        self.reload()


index = FaceIndex()


def get_person(pid):
    r = q("SELECT * FROM persons WHERE id=?", pid)
    return r[0] if r else None


def find_person_by_name(name):
    name = (name or "").strip().lower()
    if not name:
        return None
    rows = q("SELECT * FROM persons")
    for r in rows:
        if r["name"].lower() == name:
            return r
    for r in rows:
        if r["name"].lower().startswith(name[:4]):
            return r
    return None


def create_person(name, emb):
    pid = ex("INSERT INTO persons(name, created, last_seen, visits) VALUES(?,?,?,1)", name, now(), now())
    index.add(pid, emb)
    return pid


def cleanup_old():
    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).isoformat()
    for v in q("SELECT id, snapshot FROM visits WHERE started < ?", cutoff):
        if v["snapshot"] and os.path.exists(v["snapshot"]):
            os.remove(v["snapshot"])
    ex("DELETE FROM messages WHERE ts < ?", cutoff)
    ex("DELETE FROM visits WHERE started < ?", cutoff)


cleanup_old()

# ----------------------------------------------------------------- model helpers (blocking, run in threads)
def stt(audio_f32: np.ndarray) -> str:
    with whisper_lock:
        segs, _ = whisper.transcribe(audio_f32, language="ru", beam_size=1, vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()


def stt_file(data: bytes, language=None) -> str:
    with whisper_lock:
        segs, _ = whisper.transcribe(io.BytesIO(data), language=language or None, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()


def tts(text: str) -> bytes:
    buf = io.BytesIO()
    with piper_lock, wave.open(buf, "wb") as w:
        (getattr(voice, "synthesize_wav", None) or voice.synthesize)(text, w)
    return buf.getvalue()


def detect_faces(jpeg: bytes):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    with face_lock:
        faces = face_app.get(img)
    out = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        out.append({"area": int((x2 - x1) * (y2 - y1)), "score": float(f.det_score), "emb": f.normed_embedding.astype(np.float32),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)], "age": int(getattr(f, "age", 0) or 0),
                    "gender": int(f.gender) if getattr(f, "gender", None) is not None else -1})
    out.sort(key=lambda d: -d["area"])
    return out


def strip_think(s):
    return re.sub(r"<think>.*?</think>", "", s or "", flags=re.S).strip()


# ----------------------------------------------------------------- tools the LLM can call
TOOLS = [
    {"name": "leave_message", "description": "Оставить сообщение или поручение другому человеку. Ассистент передаст его, когда этот человек придёт.",
     "parameters": {"type": "object", "properties": {"to_name": {"type": "string", "description": "Имя адресата"},
                                                     "text": {"type": "string", "description": "Что передать, от третьего лица"}},
                    "required": ["to_name", "text"], "additionalProperties": False}},
    {"name": "remember", "description": "Записать заметку или факт, который попросили запомнить (о текущем человеке или вообще).",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}},
    {"name": "who_visited", "description": "Кто заходил за последние N часов и что говорил (кратко).",
     "parameters": {"type": "object", "properties": {"hours": {"type": "integer", "description": "по умолчанию 24"}}, "additionalProperties": False}},
    {"name": "list_people", "description": "Список всех знакомых людей.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "end_session", "description": "Завершить разговор и выключиться: когда человек прощается, говорит 'выключись', 'спасибо, всё' и т.п.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def tools_openai():
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in TOOLS]


def tools_anthropic():
    return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOLS]


# ----------------------------------------------------------------- LLM providers -> normalized {"text", "tool_calls":[{id,name,args}], "assistant_msg"}
http = httpx.AsyncClient(timeout=120)


async def llm_openai(system, messages, tools=None, max_tokens=300, temperature=0.6):
    body = {"model": LLM_MODEL, "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens, "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools_openai()
        body["tool_choice"] = "auto"
    r = await http.post(f"{LLM_URL}/chat/completions", json=body, headers={"Authorization": f"Bearer {LLM_KEY}"})
    r.raise_for_status()
    m = r.json()["choices"][0]["message"]
    calls = []
    for tc in m.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        calls.append({"id": tc["id"], "name": tc["function"]["name"], "args": args})
    assistant_msg = {"role": "assistant", "content": m.get("content") or ""}
    if calls:
        assistant_msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}} for c in calls]
    return {"text": strip_think(m.get("content")), "tool_calls": calls, "assistant_msg": assistant_msg}


def to_anthropic_messages(messages):
    """OpenAI-style history -> Anthropic blocks. Assistant turns keep their original blocks in _anthropic."""
    out = []
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, list):
                blocks = []
                for p in content:
                    if p.get("type") == "text":
                        blocks.append({"type": "text", "text": p["text"]})
                    elif p.get("type") == "image_url":
                        b64 = p["image_url"]["url"].split(",", 1)[1]
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
                content = blocks
            out.append({"role": "user", "content": content})
        elif m["role"] == "assistant":
            if m.get("_anthropic"):
                out.append({"role": "assistant", "content": m["_anthropic"]})
            elif m.get("content"):
                out.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


_anthropic_client = None


async def llm_anthropic(system, messages, tools=None, max_tokens=300, temperature=0.6):
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    kwargs = dict(model=ANTHROPIC_MODEL, max_tokens=max(max_tokens, 1024), system=system,
                  messages=to_anthropic_messages(messages), output_config={"effort": "low"})
    if tools:
        kwargs["tools"] = tools_anthropic()
    resp = await _anthropic_client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        return {"text": "Извини, на это я ответить не могу.", "tool_calls": [], "assistant_msg": {"role": "assistant", "content": "Извини, на это я ответить не могу."}}
    text = " ".join(b.text for b in resp.content if b.type == "text").strip()
    calls = [{"id": b.id, "name": b.name, "args": dict(b.input)} for b in resp.content if b.type == "tool_use"]
    assistant_msg = {"role": "assistant", "content": text, "_anthropic": [b.model_dump(exclude_none=True) for b in resp.content]}
    return {"text": text, "tool_calls": calls, "assistant_msg": assistant_msg}


async def llm(system, messages, tools=None, max_tokens=300, temperature=0.6):
    if LLM_PROVIDER == "claude-cli":
        import llm_cli
        return await asyncio.to_thread(llm_cli.chat, system, messages, TOOLS if tools else None, max_tokens, temperature)
    fn = llm_anthropic if LLM_PROVIDER == "anthropic" else llm_openai
    return await fn(system, messages, tools, max_tokens, temperature)


async def llm_text(system, user, max_tokens=200, temperature=0.3):
    r = await llm(system, [{"role": "user", "content": user}], None, max_tokens, temperature)
    return r["text"]


def system_prompt(person, pending=None):
    p = (f"Ты {ASSISTANT_NAME}, голосовой помощник у входа в офис UCO. Сейчас {datetime.now():%d.%m.%Y %H:%M}. "
         "Отвечай по-русски, коротко (1-3 предложения), дружелюбно, без markdown и без списков: текст будет озвучен. "
         "Если просят что-то передать другому человеку - вызови leave_message. Если просят запомнить/записать - remember. "
         "Если человек прощается или говорит выключись - end_session. Не выдумывай факты о людях.")
    if person:
        facts = json.loads(person.get("facts") or "[]")
        p += f"\nПеред тобой {person['name']} (визитов: {person['visits']}, последний раз: {(person['last_seen'] or '')[:16]})."
        if facts:
            p += "\nЧто ты знаешь об этом человеке:\n- " + "\n- ".join(facts)
    else:
        p += "\nЧеловек перед тобой пока незнакомый."
    if pending:
        p += "\nДля этого человека есть непереданные сообщения, озвучь их:\n- " + "\n- ".join(f"от {t['from_name'] or 'неизвестно'}: {t['text']}" for t in pending)
    return p


# ----------------------------------------------------------------- one edge connection = one dialog session
class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.vad = webrtcvad.Vad(2)
        self.ring = deque(maxlen=10)
        self.prebuf = deque(maxlen=10)
        self.speech = []
        self.in_speech = False
        self.silence = 0
        self.person = None
        self.visit_id = None
        self.pending_emb = None
        self.pending_area = 0
        self.pending_frame = None
        self.state = "idle"                    # idle | awaiting_name | ended
        self.history = []
        self.last_face_ts = 0.0
        self.unknown_hits = 0
        self.known_hits = {}
        self.last_jpeg = None
        self.busy = asyncio.Lock()
        self.name_attempts = 0

    async def send(self, obj):
        await self.ws.send_text(json.dumps(obj, ensure_ascii=False))

    def log_msg(self, role, content):
        ex("INSERT INTO messages(person_id, visit_id, role, content, ts) VALUES(?,?,?,?,?)",
           self.person["id"] if self.person else None, self.visit_id, role, content, now())

    async def say(self, text):
        text = strip_think(text)
        if not text:
            return
        log.info("SAY: %s", text)
        wav = await asyncio.to_thread(tts, text)
        await self.send({"type": "say", "text": text, "wav": base64.b64encode(wav).decode()})
        if self.visit_id:
            self.log_msg("assistant", text)

    # ---------------- visits
    def start_visit(self, jpeg):
        path = None
        if jpeg:
            path = f"{SNAP_DIR}/{int(time.time())}_{secrets.token_hex(3)}.jpg"
            with open(path, "wb") as f:
                f.write(jpeg)
        self.visit_id = ex("INSERT INTO visits(person_id, started, snapshot) VALUES(?,?,?)", self.person["id"] if self.person else None, now(), path)

    async def finish_visit(self, why):
        vid, self.visit_id = self.visit_id, None
        if not vid:
            return
        rows = q("SELECT role, content FROM messages WHERE visit_id=? ORDER BY id", vid)
        summary = ""
        if rows:
            transcript = "\n".join(f"{'Человек' if r['role'] == 'user' else 'Ассистент'}: {r['content']}" for r in rows)
            try:
                summary = await llm_text("Кратко, в 1-2 предложениях по-русски, что было в этом визите: кто, зачем приходил, о чём просил. Только текст.", transcript, 120)
            except Exception as e:
                log.warning("summary failed: %s", e)
                summary = rows[0]["content"][:120]
        ex("UPDATE visits SET ended=?, summary=? WHERE id=?", now(), summary, vid)
        log.info("visit %s closed (%s): %s", vid, why, summary)

    # ---------------- vision
    async def on_frame(self, jpeg: bytes):
        self.last_jpeg = jpeg
        faces = await asyncio.to_thread(detect_faces, jpeg)
        if not faces:
            if (self.person or self.state != "idle") and time.time() - self.last_face_ts > SESSION_TIMEOUT:
                await self.reset("человек ушёл")
            return
        self.last_face_ts = time.time()
        if self.state == "ended":
            return                                   # ждём, пока человек уйдёт из кадра
        f = faces[0]
        pid, sim = index.match(f["emb"])
        if pid is not None:
            self.unknown_hits = 0
            self.known_hits[pid] = self.known_hits.get(pid, 0) + 1
            if (self.person is None or self.person["id"] != pid) and self.known_hits[pid] >= 2:
                await self.on_known(pid, sim, jpeg)
        else:
            self.unknown_hits += 1
            if f["score"] > 0.7 and (self.pending_emb is None or f["area"] > self.pending_area):
                self.pending_emb, self.pending_area, self.pending_frame = f["emb"], f["area"], jpeg
            if self.person is None and self.state == "idle" and self.unknown_hits >= 3:
                await self.on_unknown()

    async def on_known(self, pid, sim, jpeg):
        if self.visit_id:
            await self.finish_visit("другой человек")
        ex("UPDATE persons SET last_seen=?, visits=visits+1 WHERE id=?", now(), pid)
        self.person = get_person(pid)
        self.state = "idle"
        self.history = []
        self.known_hits = {}
        self.start_visit(jpeg)
        pending = q("SELECT * FROM tasks WHERE status='pending' AND (to_person=? OR lower(to_name)=lower(?))", pid, self.person["name"])
        log.info("known person %s (%s) sim=%.2f pending=%d", pid, self.person["name"], sim, len(pending))
        await self.send({"type": "status", "person": self.person["name"], "sim": round(sim, 2), "visit": self.visit_id})
        async with self.busy:
            try:
                r = await llm(system_prompt(self.person, pending),
                              [{"role": "user", "content": "Человек только что вошёл. Поздоровайся по имени одной короткой фразой, передай сообщения, если они есть, и спроси чем помочь."}],
                              None, 200)
                greet = r["text"] or f"С возвращением, {self.person['name']}!"
            except Exception as e:
                log.warning("llm greet failed: %s", e)
                greet = f"С возвращением, {self.person['name']}! " + (" ".join(f"Вам сообщение от {t['from_name'] or 'неизвестно'}: {t['text']}." for t in pending)) + " Чем помочь?"
            self.history.append({"role": "assistant", "content": greet})
            await self.say(greet)
            for t in pending:
                ex("UPDATE tasks SET status='delivered', delivered_at=? WHERE id=?", now(), t["id"])

    async def on_unknown(self):
        self.state = "awaiting_name"
        self.name_attempts = 0
        self.start_visit(self.pending_frame or self.last_jpeg)
        await self.send({"type": "status", "person": None, "visit": self.visit_id})
        async with self.busy:
            await self.say("Привет! Я тебя ещё не знаю. Как тебя зовут?")

    async def reset(self, why):
        log.info("session reset: %s", why)
        await self.finish_visit(why)
        self.person = None
        self.pending_emb = None
        self.pending_frame = None
        self.state = "idle"
        self.history = []
        self.unknown_hits = 0
        self.known_hits = {}
        await self.send({"type": "status", "person": None, "reset": why})

    # ---------------- audio
    async def on_audio(self, pcm: bytes):
        if len(pcm) != 960 or self.state == "ended":
            return
        if time.time() - self.last_face_ts > 15:
            return
        voiced = self.vad.is_speech(pcm, SR)
        self.ring.append(voiced)
        if not self.in_speech:
            self.prebuf.append(pcm)
            if sum(self.ring) >= 6:
                self.in_speech = True
                self.speech = list(self.prebuf)
                self.silence = 0
            return
        self.speech.append(pcm)
        self.silence = 0 if voiced else self.silence + 1
        if self.silence >= 25 or len(self.speech) > 500:
            chunks, self.speech, self.in_speech = self.speech, [], False
            self.ring.clear()
            self.prebuf.clear()
            if len(chunks) < 15:
                return
            audio = np.frombuffer(b"".join(chunks), np.int16).astype(np.float32) / 32768.0
            asyncio.create_task(self.on_utterance(audio))

    async def on_utterance(self, audio):
        if self.busy.locked():
            return
        async with self.busy:
            text = await asyncio.to_thread(stt, audio)
            log.info("HEARD (%.1fs): %s", len(audio) / SR, text)
            if len(text) < 2:
                return
            await self.handle_text(text)

    async def on_text(self, text):
        async with self.busy:
            await self.handle_text(text)

    async def handle_text(self, text):
        await self.send({"type": "heard", "text": text})
        if self.state == "ended":
            return
        if self.state == "awaiting_name":
            await self.handle_name(text)
        elif self.state == "idle" and not self.person and not self.visit_id:
            self.start_visit(self.last_jpeg)      # текст без лица (отладка / wake word)
            await self.chat(text)
        else:
            await self.chat(text)

    async def handle_name(self, text):
        self.log_msg("user", text)
        try:
            name = await llm_text("Извлеки имя человека из его реплики. Ответь только именем в именительном падеже с большой буквы. Если имени нет, ответь NONE.", text, 20, 0)
        except Exception as e:
            log.warning("llm name failed: %s", e)
            name = text.strip().split()[-1].capitalize()
        name = name.strip().strip(".!")
        if not name or name.upper() == "NONE" or len(name) > 30:
            self.name_attempts += 1
            if self.name_attempts < 3:
                await self.say("Не расслышал имя. Скажи, пожалуйста, ещё раз, как тебя зовут?")
                return
            name = f"Гость {datetime.now():%d.%m %H:%M}"
        if self.pending_emb is None:
            await self.say("Посмотри, пожалуйста, в камеру, я не вижу лица.")
            return
        pid = create_person(name, self.pending_emb)
        self.person = get_person(pid)
        self.state = "idle"
        self.pending_emb = None
        self.history = []
        if self.visit_id:
            ex("UPDATE visits SET person_id=? WHERE id=?", pid, self.visit_id)
        log.info("enrolled %s as %s", pid, name)
        await self.send({"type": "status", "person": name, "enrolled": True})
        await self.say(f"Приятно познакомиться, {name}! Я тебя запомнил. Чем могу помочь?")

    # ---------------- agent loop
    async def chat(self, text):
        self.log_msg("user", text)
        wants_vision = any(k in text.lower() for k in ("видишь", "в кадре", "на камере", "что на мне", "посмотри", "как я выгляжу"))
        user = {"role": "user", "content": text}
        if wants_vision and self.last_jpeg:
            user = {"role": "user", "content": [{"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(self.last_jpeg).decode()}}]}
        self.history.append(user)
        end = False
        reply = ""
        try:
            for _ in range(4):
                r = await llm(system_prompt(self.person), self.history[-14:], tools=True)
                self.history.append(r["assistant_msg"])
                if not r["tool_calls"]:
                    reply = r["text"]
                    break
                for call in r["tool_calls"]:
                    result, stop = await self.run_tool(call["name"], call["args"])
                    end = end or stop
                    self.history.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})
                if r["text"] and end:
                    reply = r["text"]
                    break
            else:
                reply = reply or "Готово."
        except Exception as e:
            log.error("llm failed: %s", e)
            reply = "Извини, у меня проблемы со связью с моделью."
        if end and not reply:
            reply = "Хорошо, до встречи!"
        await self.say(reply)
        if end:
            await self.end_session()
        elif self.person:
            asyncio.create_task(self.extract_facts(text, reply))

    async def run_tool(self, name, args):
        log.info("TOOL %s %s", name, args)
        me = self.person
        result, stop = "", False
        try:
            if name == "leave_message":
                to = find_person_by_name(args.get("to_name", ""))
                ex("INSERT INTO tasks(from_person, from_name, to_person, to_name, text, created) VALUES(?,?,?,?,?,?)",
                   me["id"] if me else None, me["name"] if me else None, to["id"] if to else None,
                   (to["name"] if to else args.get("to_name", "")).strip(), args.get("text", ""), now())
                result = f"Сообщение сохранено, передам {to['name'] if to else args.get('to_name')} при встрече." + ("" if to else " Такого человека я пока не знаю, передам, когда познакомимся.")
            elif name == "remember":
                ex("INSERT INTO notes(person_id, person_name, text, created) VALUES(?,?,?,?)", me["id"] if me else None, me["name"] if me else None, args.get("text", ""), now())
                if me:
                    facts = json.loads(me.get("facts") or "[]")
                    facts = (facts + [args.get("text", "")])[-60:]
                    ex("UPDATE persons SET facts=? WHERE id=?", json.dumps(facts, ensure_ascii=False), me["id"])
                    self.person = get_person(me["id"])
                result = "Записал."
            elif name == "who_visited":
                since = (datetime.now() - timedelta(hours=int(args.get("hours") or 24))).isoformat()
                rows = q("SELECT v.started, p.name, v.summary FROM visits v LEFT JOIN persons p ON p.id=v.person_id WHERE v.started>? ORDER BY v.started DESC LIMIT 20", since)
                result = "\n".join(f"{r['started'][11:16]} {r['name'] or 'незнакомец'}: {r['summary'] or ''}" for r in rows) or "Никто не заходил."
            elif name == "list_people":
                result = ", ".join(r["name"] for r in q("SELECT name FROM persons ORDER BY name")) or "Пока никого не знаю."
            elif name == "end_session":
                result, stop = "Разговор завершён.", True
            else:
                result = f"Неизвестный инструмент {name}"
        except Exception as e:
            result = f"Ошибка: {e}"
        ex("INSERT INTO actions(visit_id, person_id, tool, args, result, ts) VALUES(?,?,?,?,?,?)",
           self.visit_id, me["id"] if me else None, name, json.dumps(args, ensure_ascii=False), result, now())
        return result, stop

    async def end_session(self):
        self.state = "ended"
        await self.finish_visit("завершено по просьбе")
        await self.send({"type": "end"})
        self.person = None
        self.history = []
        self.known_hits = {}
        self.unknown_hits = 0
        asyncio.get_event_loop().call_later(20, lambda: setattr(self, "state", "idle") if self.state == "ended" else None)

    async def extract_facts(self, user_text, reply):
        try:
            p = get_person(self.person["id"])
            facts = json.loads(p.get("facts") or "[]")
            out = await llm_text("Ты извлекаешь долговременные факты о человеке из диалога: имя близких, должность, интересы, предпочтения, планы. "
                                 "Верни JSON {\"facts\": [\"...\"]} с новыми фактами, коротко, от третьего лица. Если фактов нет - {\"facts\": []}. Только JSON.",
                                 f"Уже известно: {facts}\n\nЧеловек сказал: {user_text}\nАссистент ответил: {reply}", 200, 0)
            m = re.search(r"\{.*\}", out, re.S)
            new = json.loads(m.group(0)).get("facts", []) if m else []
            new = [f for f in new if isinstance(f, str) and f and f not in facts]
            if new:
                facts = (facts + new)[-60:]
                ex("UPDATE persons SET facts=? WHERE id=?", json.dumps(facts, ensure_ascii=False), self.person["id"])
                self.person = get_person(self.person["id"])
                log.info("new facts for %s: %s", self.person["name"], new)
        except Exception as e:
            log.warning("fact extraction failed: %s", e)


# ----------------------------------------------------------------- auth
def _basic_ok(header):
    try:
        user, pw = base64.b64decode(header.split(" ", 1)[1]).decode().split(":", 1)
        return hmac.compare_digest(user, DASH_USER) and hmac.compare_digest(pw, DASH_PASSWORD)
    except Exception:
        return False


def key_ok(key):
    return bool(API_KEY) and hmac.compare_digest(key or "", API_KEY)


def auth(request: Request):
    if not API_KEY:
        return True
    a = request.headers.get("authorization", "")
    if key_ok(request.headers.get("x-api-key")) or (a.lower().startswith("bearer ") and key_ok(a[7:])) or (a.lower().startswith("basic ") and _basic_ok(a)):
        return True
    if key_ok(request.query_params.get("key")):
        return True
    raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Basic realm=assistant"})


# ----------------------------------------------------------------- API
app = FastAPI(title="door-assistant")


@app.websocket("/ws/edge")
async def ws_edge(ws: WebSocket):
    key = ws.query_params.get("key") or ws.headers.get("x-api-key") or ""
    if API_KEY and not key_ok(key):
        await ws.close(code=4401)
        return
    await ws.accept()
    s = Session(ws)
    log.info("edge connected %s", ws.client)
    try:
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                await s.on_audio(m["bytes"])
            elif m.get("text"):
                d = json.loads(m["text"])
                if d.get("type") == "frame":
                    await s.on_frame(base64.b64decode(d["jpeg"]))
                elif d.get("type") == "text":
                    asyncio.create_task(s.on_text(d["text"]))
    except WebSocketDisconnect:
        pass
    await s.finish_visit("edge disconnected")
    log.info("edge disconnected")


@app.get("/health")
def health():
    return {"ok": True, "persons": q("SELECT count(*) n FROM persons")[0]["n"], "whisper": WHISPER_MODEL, "voice": PIPER_VOICE,
            "llm": f"{LLM_PROVIDER}: " + (ANTHROPIC_MODEL if LLM_PROVIDER == 'anthropic' else os.getenv('CLAUDE_MODEL', 'sonnet') if LLM_PROVIDER == 'claude-cli' else LLM_URL + ' ' + LLM_MODEL), "auth": bool(API_KEY)}


@app.get("/api/visits", dependencies=[Depends(auth)])
def api_visits(limit: int = 100):
    return q("SELECT v.*, p.name FROM visits v LEFT JOIN persons p ON p.id=v.person_id ORDER BY v.id DESC LIMIT ?", limit)


@app.get("/api/visits/{vid}", dependencies=[Depends(auth)])
def api_visit(vid: int):
    v = q("SELECT v.*, p.name FROM visits v LEFT JOIN persons p ON p.id=v.person_id WHERE v.id=?", vid)
    if not v:
        raise HTTPException(404)
    return v[0] | {"messages": q("SELECT role, content, ts FROM messages WHERE visit_id=? ORDER BY id", vid),
                   "actions": q("SELECT tool, args, result, ts FROM actions WHERE visit_id=? ORDER BY id", vid)}


@app.get("/api/visits/{vid}/snapshot", dependencies=[Depends(auth)])
def api_snapshot(vid: int):
    v = q("SELECT snapshot FROM visits WHERE id=?", vid)
    if not v or not v[0]["snapshot"] or not os.path.exists(v[0]["snapshot"]):
        raise HTTPException(404)
    return FileResponse(v[0]["snapshot"], media_type="image/jpeg")


@app.get("/api/people", dependencies=[Depends(auth)])
def api_people():
    return [r | {"facts": json.loads(r["facts"] or "[]")} for r in q("SELECT * FROM persons ORDER BY last_seen DESC")]


@app.delete("/api/people/{pid}", dependencies=[Depends(auth)])
def api_delete_person(pid: int):
    ex("DELETE FROM persons WHERE id=?", pid)
    ex("DELETE FROM embeddings WHERE person_id=?", pid)
    index.reload()
    return {"deleted": pid}


@app.get("/api/tasks", dependencies=[Depends(auth)])
def api_tasks():
    return q("SELECT * FROM tasks ORDER BY id DESC LIMIT 200")


@app.post("/api/tasks/{tid}/status", dependencies=[Depends(auth)])
async def api_task_status(tid: int, req: Request):
    status = (await req.json()).get("status", "done")
    ex("UPDATE tasks SET status=? WHERE id=?", status, tid)
    return {"ok": True}


@app.get("/api/notes", dependencies=[Depends(auth)])
def api_notes():
    return q("SELECT * FROM notes ORDER BY id DESC LIMIT 200")


@app.get("/api/actions", dependencies=[Depends(auth)])
def api_actions():
    return q("SELECT a.*, p.name FROM actions a LEFT JOIN persons p ON p.id=a.person_id ORDER BY a.id DESC LIMIT 200")


@app.get("/persons", dependencies=[Depends(auth)])
def persons():
    return api_people()


@app.delete("/persons/{pid}", dependencies=[Depends(auth)])
def delete_person(pid: int):
    return api_delete_person(pid)


@app.post("/face/embed", dependencies=[Depends(auth)])
async def face_embed(file: UploadFile = File(...)):
    faces = await asyncio.to_thread(detect_faces, await file.read())
    return {"faces": [{"bbox": f["bbox"], "score": round(f["score"], 3), "age": f["age"], "gender": f["gender"], "embedding": f["emb"].tolist()} for f in faces]}


@app.post("/face/identify", dependencies=[Depends(auth)])
async def face_identify(file: UploadFile = File(...)):
    faces = await asyncio.to_thread(detect_faces, await file.read())
    res = []
    for f in faces:
        pid, sim = index.match(f["emb"])
        p = get_person(pid) if pid else None
        res.append({"name": p["name"] if p else None, "sim": round(sim, 3), "score": round(f["score"], 2), "bbox": f["bbox"]})
    return {"faces": res}


@app.post("/face/enroll", dependencies=[Depends(auth)])
async def face_enroll(name: str = Form(...), file: UploadFile = File(...)):
    faces = await asyncio.to_thread(detect_faces, await file.read())
    if not faces:
        return JSONResponse({"error": "no face"}, 400)
    return {"id": create_person(name, faces[0]["emb"]), "name": name}


@app.post("/v1/audio/transcriptions", dependencies=[Depends(auth)])
async def transcriptions(file: UploadFile = File(...), model: str = Form(None), language: str = Form(None)):
    return {"text": await asyncio.to_thread(stt_file, await file.read(), language)}


@app.post("/v1/audio/speech", dependencies=[Depends(auth)])
async def speech(req: Request):
    body = await req.json()
    return Response(await asyncio.to_thread(tts, body.get("input", "")), media_type="audio/wav")


@app.get("/v1/models", dependencies=[Depends(auth)])
def models():
    return {"data": [{"id": f"whisper-{WHISPER_MODEL}", "object": "model"}, {"id": PIPER_VOICE, "object": "model"}]}


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
def dashboard():
    with open(os.path.join(os.path.dirname(__file__), "dashboard.html"), encoding="utf-8") as f:
        return f.read()
