"""Door assistant: camera (face id) + mic (STT) -> LLM -> TTS. All local, CPU only.

Edge client connects to /ws/edge and sends:
  text frame  {"type":"frame","jpeg":"<base64>"}   ~2 fps
  binary      int16 PCM mono 16 kHz, 480 samples (30 ms) per message
Server sends: {"type":"say","text":..., "wav":"<base64 wav>"}, {"type":"heard",...}, {"type":"status", ...}
Also exposes OpenAI-compatible /v1/audio/transcriptions and /v1/audio/speech (usable from OpenWebUI).
"""
import asyncio, base64, io, json, logging, os, sqlite3, threading, time, wave, re
from collections import deque
from datetime import datetime

import cv2, httpx, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.responses import Response, JSONResponse

log = logging.getLogger("assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LLM_URL = os.getenv("LLM_URL", "http://192.168.88.49:8050/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "current-LLM")
LLM_KEY = os.getenv("LLM_API_KEY", "none")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
PIPER_VOICE = os.getenv("PIPER_VOICE", "ru_RU-irina-medium")
FACE_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.45"))
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Ассистент")
DATA = "/data"
SESSION_TIMEOUT = 40      # сек без лица -> сессия закрыта
SR = 16000

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
log.info("models ready")

# ----------------------------------------------------------------- storage
DB_PATH = f"{DATA}/assistant.sqlite"
db_lock = threading.Lock()


def db():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS persons(id INTEGER PRIMARY KEY, name TEXT, facts TEXT DEFAULT '[]',
        created TEXT, last_seen TEXT, visits INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS embeddings(id INTEGER PRIMARY KEY, person_id INTEGER, vec BLOB);
    CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, person_id INTEGER, role TEXT, content TEXT, ts TEXT);
    """)


def now():
    return datetime.now().isoformat(timespec="seconds")


class FaceIndex:
    """all embeddings in memory: (N,512) normalized + person ids"""

    def __init__(self):
        self.reload()

    def reload(self):
        with db() as c:
            rows = c.execute("SELECT person_id, vec FROM embeddings").fetchall()
        self.ids = np.array([r["person_id"] for r in rows], dtype=np.int64)
        self.vecs = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 512), np.float32)

    def match(self, emb):
        if len(self.ids) == 0:
            return None, 0.0
        sims = self.vecs @ emb
        i = int(np.argmax(sims))
        return (int(self.ids[i]) if sims[i] >= FACE_THRESHOLD else None), float(sims[i])

    def add(self, person_id, emb):
        with db_lock, db() as c:
            c.execute("INSERT INTO embeddings(person_id, vec) VALUES(?,?)", (person_id, emb.astype(np.float32).tobytes()))
        self.reload()


index = FaceIndex()


def get_person(pid):
    with db() as c:
        r = c.execute("SELECT * FROM persons WHERE id=?", (pid,)).fetchone()
    return dict(r) if r else None


def create_person(name, emb):
    with db_lock, db() as c:
        cur = c.execute("INSERT INTO persons(name, created, last_seen, visits) VALUES(?,?,?,1)", (name, now(), now()))
        pid = cur.lastrowid
    index.add(pid, emb)
    return pid


def touch_person(pid):
    with db_lock, db() as c:
        c.execute("UPDATE persons SET last_seen=?, visits=visits+1 WHERE id=?", (now(), pid))


def save_facts(pid, facts):
    with db_lock, db() as c:
        c.execute("UPDATE persons SET facts=? WHERE id=?", (json.dumps(facts, ensure_ascii=False), pid))


def log_msg(pid, role, content):
    with db_lock, db() as c:
        c.execute("INSERT INTO messages(person_id, role, content, ts) VALUES(?,?,?,?)", (pid, role, content, now()))


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
                    "bbox": [int(x1), int(y1), int(x2), int(y2)], "age": int(getattr(f, "age", 0) or 0), "gender": int(getattr(f, "gender", -1) if getattr(f, "gender", None) is not None else -1)})
    out.sort(key=lambda d: -d["area"])
    return out


http = httpx.AsyncClient(timeout=120)


async def llm(messages, max_tokens=300, temperature=0.6):
    body = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = await http.post(f"{LLM_URL}/chat/completions", json=body, headers={"Authorization": f"Bearer {LLM_KEY}"})
    r.raise_for_status()
    return strip_think(r.json()["choices"][0]["message"]["content"] or "")


def strip_think(s):
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


def system_prompt(person):
    p = (f"Ты {ASSISTANT_NAME}, голосовой помощник у входа в офис UCO. Сейчас {datetime.now():%d.%m.%Y %H:%M}. "
         "Отвечай по-русски, коротко (1-3 предложения), дружелюбно, без markdown и без списков: текст будет озвучен.")
    if person:
        facts = json.loads(person.get("facts") or "[]")
        p += f"\nПеред тобой {person['name']} (визитов: {person['visits']}, последний раз: {person['last_seen'][:16]})."
        if facts:
            p += "\nЧто ты знаешь об этом человеке:\n- " + "\n- ".join(facts)
    else:
        p += "\nЧеловек перед тобой пока незнакомый."
    return p


# ----------------------------------------------------------------- one edge connection = one dialog session
class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.vad = webrtcvad.Vad(2)
        self.ring = deque(maxlen=10)          # последние решения VAD
        self.prebuf = deque(maxlen=10)        # 300 мс до начала речи
        self.speech = []                       # bytes chunks текущей фразы
        self.in_speech = False
        self.silence = 0
        self.person = None
        self.pending_emb = None
        self.pending_area = 0
        self.state = "idle"                    # idle | awaiting_name
        self.history = []
        self.last_face_ts = 0.0
        self.unknown_hits = 0
        self.known_hits = {}
        self.last_jpeg = None
        self.busy = asyncio.Lock()
        self.name_attempts = 0

    async def send(self, obj):
        await self.ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def say(self, text):
        text = strip_think(text)
        if not text:
            return
        log.info("SAY: %s", text)
        wav = await asyncio.to_thread(tts, text)
        await self.send({"type": "say", "text": text, "wav": base64.b64encode(wav).decode()})
        if self.person:
            log_msg(self.person["id"], "assistant", text)

    # ---------------- vision
    async def on_frame(self, jpeg: bytes):
        self.last_jpeg = jpeg
        faces = await asyncio.to_thread(detect_faces, jpeg)
        if not faces:
            if (self.person or self.state != "idle") and time.time() - self.last_face_ts > SESSION_TIMEOUT:
                await self.reset("человек ушёл")
            return
        self.last_face_ts = time.time()
        f = faces[0]
        pid, sim = index.match(f["emb"])
        if pid is not None:
            self.unknown_hits = 0
            self.known_hits[pid] = self.known_hits.get(pid, 0) + 1
            if (self.person is None or self.person["id"] != pid) and self.known_hits[pid] >= 2:
                await self.on_known(pid, sim)
        else:
            self.unknown_hits += 1
            if f["score"] > 0.7 and (self.pending_emb is None or f["area"] > self.pending_area):
                self.pending_emb, self.pending_area = f["emb"], f["area"]
            if self.person is None and self.state == "idle" and self.unknown_hits >= 3:
                await self.on_unknown()

    async def on_known(self, pid, sim):
        touch_person(pid)
        self.person = get_person(pid)
        self.state = "idle"
        self.history = []
        self.known_hits = {}
        log.info("known person %s (%s) sim=%.2f", pid, self.person["name"], sim)
        await self.send({"type": "status", "person": self.person["name"], "sim": round(sim, 2)})
        async with self.busy:
            try:
                greet = await llm([{"role": "system", "content": system_prompt(self.person)},
                                   {"role": "user", "content": "Человек только что вошёл. Поздоровайся с ним по имени одной короткой фразой, "
                                                               "можешь упомянуть что-то из того, что ты о нём знаешь, и спроси чем помочь."}], max_tokens=120)
            except Exception as e:
                log.warning("llm greet failed: %s", e)
                greet = f"С возвращением, {self.person['name']}! Чем помочь?"
            self.history.append({"role": "assistant", "content": greet})
            await self.say(greet)

    async def on_unknown(self):
        self.state = "awaiting_name"
        self.name_attempts = 0
        await self.send({"type": "status", "person": None})
        async with self.busy:
            await self.say("Привет! Я тебя ещё не знаю. Как тебя зовут?")

    async def reset(self, why):
        log.info("session reset: %s", why)
        self.person = None
        self.pending_emb = None
        self.state = "idle"
        self.history = []
        self.unknown_hits = 0
        self.known_hits = {}
        await self.send({"type": "status", "person": None, "reset": why})

    # ---------------- audio
    async def on_audio(self, pcm: bytes):
        if len(pcm) != 960:                              # ждём ровно 30 мс int16 @16k
            return
        if time.time() - self.last_face_ts > 15:         # никого нет перед камерой - не слушаем
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
        if self.silence >= 25 or len(self.speech) > 500:   # 750 мс тишины или 15 с речи
            chunks, self.speech, self.in_speech = self.speech, [], False
            self.ring.clear()
            self.prebuf.clear()
            if len(chunks) < 15:                             # < 0.45 c - шум
                return
            audio = np.frombuffer(b"".join(chunks), np.int16).astype(np.float32) / 32768.0
            asyncio.create_task(self.on_utterance(audio))

    async def on_utterance(self, audio):
        if self.busy.locked():                               # ассистент говорит/думает - игнорируем
            return
        async with self.busy:
            text = await asyncio.to_thread(stt, audio)
            log.info("HEARD (%.1fs): %s", len(audio) / SR, text)
            if len(text) < 2:
                return
            await self.send({"type": "heard", "text": text})
            if self.state == "awaiting_name":
                await self.handle_name(text)
            else:
                await self.chat(text)

    async def on_text(self, text):
        """текстовый ввод для отладки без микрофона"""
        async with self.busy:
            await self.send({"type": "heard", "text": text})
            if self.state == "awaiting_name":
                await self.handle_name(text)
            else:
                await self.chat(text)

    async def handle_name(self, text):
        try:
            name = await llm([{"role": "system", "content": "Извлеки имя человека из его реплики. Ответь только именем в именительном падеже с большой буквы. Если имени нет, ответь NONE."},
                              {"role": "user", "content": text}], max_tokens=20, temperature=0)
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
        log.info("enrolled %s as %s", pid, name)
        await self.send({"type": "status", "person": name, "enrolled": True})
        await self.say(f"Приятно познакомиться, {name}! Я тебя запомнил. Чем могу помочь?")

    async def chat(self, text):
        if self.person:
            log_msg(self.person["id"], "user", text)
        wants_vision = any(k in text.lower() for k in ("видишь", "в кадре", "на камере", "что на мне", "посмотри", "как я выгляжу"))
        user = {"role": "user", "content": text}
        if wants_vision and self.last_jpeg:
            user = {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(self.last_jpeg).decode()}}]}
        self.history.append(user)
        msgs = [{"role": "system", "content": system_prompt(self.person)}] + self.history[-12:]
        try:
            reply = await llm(msgs)
        except Exception as e:
            log.error("llm failed: %s", e)
            reply = "Извини, у меня проблемы со связью с моделью."
        self.history.append({"role": "assistant", "content": reply})
        await self.say(reply)
        if self.person:
            asyncio.create_task(self.extract_facts(text, reply))

    async def extract_facts(self, user_text, reply):
        try:
            p = get_person(self.person["id"])
            facts = json.loads(p.get("facts") or "[]")
            out = await llm([{"role": "system", "content": "Ты извлекаешь долговременные факты о человеке из диалога: имя близких, должность, интересы, предпочтения, планы, просьбы запомнить. "
                              "Верни JSON {\"facts\": [\"...\"]} с новыми фактами, коротко, от третьего лица. Если фактов нет - {\"facts\": []}. Только JSON."},
                             {"role": "user", "content": f"Уже известно: {facts}\n\nЧеловек сказал: {user_text}\nАссистент ответил: {reply}"}], max_tokens=200, temperature=0)
            m = re.search(r"\{.*\}", out, re.S)
            new = json.loads(m.group(0)).get("facts", []) if m else []
            new = [f for f in new if isinstance(f, str) and f and f not in facts]
            if new:
                facts = (facts + new)[-60:]
                save_facts(self.person["id"], facts)
                self.person["facts"] = json.dumps(facts, ensure_ascii=False)
                log.info("new facts for %s: %s", self.person["name"], new)
        except Exception as e:
            log.warning("fact extraction failed: %s", e)


# ----------------------------------------------------------------- API
app = FastAPI(title="door-assistant")


@app.websocket("/ws/edge")
async def ws_edge(ws: WebSocket):
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
    log.info("edge disconnected")


@app.get("/health")
def health():
    with db() as c:
        n = c.execute("SELECT count(*) FROM persons").fetchone()[0]
    return {"ok": True, "persons": n, "whisper": WHISPER_MODEL, "voice": PIPER_VOICE, "llm": f"{LLM_URL} {LLM_MODEL}"}


@app.get("/persons")
def persons():
    with db() as c:
        return [dict(r) | {"facts": json.loads(r["facts"] or "[]")} for r in c.execute("SELECT * FROM persons ORDER BY last_seen DESC")]


@app.delete("/persons/{pid}")
def delete_person(pid: int):
    with db_lock, db() as c:
        c.execute("DELETE FROM persons WHERE id=?", (pid,))
        c.execute("DELETE FROM embeddings WHERE person_id=?", (pid,))
    index.reload()
    return {"deleted": pid}


@app.post("/face/identify")
async def face_identify(file: UploadFile = File(...)):
    faces = await asyncio.to_thread(detect_faces, await file.read())
    res = []
    for f in faces:
        pid, sim = index.match(f["emb"])
        p = get_person(pid) if pid else None
        res.append({"name": p["name"] if p else None, "sim": round(sim, 3), "score": round(f["score"], 2)})
    return {"faces": res}


@app.post("/face/embed")
async def face_embed(file: UploadFile = File(...)):
    """Чистая модель без базы: детекция + 512-мерный нормированный эмбеддинг (ArcFace) для каждого лица.
    Сравнение: cosine = dot(a, b); один человек при > 0.45."""
    faces = await asyncio.to_thread(detect_faces, await file.read())
    return {"faces": [{"bbox": f["bbox"], "score": round(f["score"], 3), "age": f["age"], "gender": f["gender"],
                       "embedding": f["emb"].tolist()} for f in faces]}


@app.post("/face/enroll")
async def face_enroll(name: str = Form(...), file: UploadFile = File(...)):
    faces = await asyncio.to_thread(detect_faces, await file.read())
    if not faces:
        return JSONResponse({"error": "no face"}, 400)
    return {"id": create_person(name, faces[0]["emb"]), "name": name}


# OpenAI-compatible endpoints (можно указать в OpenWebUI как STT/TTS: http://192.168.88.49:8060/v1)
@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = Form(None), language: str = Form(None)):
    return {"text": await asyncio.to_thread(stt_file, await file.read(), language)}


@app.post("/v1/audio/speech")
async def speech(req: Request):
    body = await req.json()
    return Response(await asyncio.to_thread(tts, body.get("input", "")), media_type="audio/wav")


@app.get("/v1/models")
def models():
    return {"data": [{"id": f"whisper-{WHISPER_MODEL}", "object": "model"}, {"id": PIPER_VOICE, "object": "model"}]}
