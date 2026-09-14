"""Edge client: webcam + mic + speaker -> door-assistant server.
   pip install -r requirements.txt
   python client.py --server ws://192.168.88.49:8060/ws/edge --key API_KEY --camera 0

Summoning the assistant without standing in front of the camera:
   --hotkey ctrl+alt+a        global hot key (default; "" to disable)
   --wake "ассистент"         wake word/phrase recognised locally by Vosk (offline, light on CPU);
                              the small Russian model is downloaded on first run (~45 MB)
"""
import argparse, base64, io, json, os, sys, threading, time, wave, queue

import cv2, numpy as np, sounddevice as sd, websocket

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="ws://192.168.88.49:8060/ws/edge")
ap.add_argument("--key", default="", help="API_KEY сервера")
ap.add_argument("--camera", type=int, default=0, help="номер камеры; -1 = без камеры (только голос/кнопка)")
ap.add_argument("--fps", type=float, default=2.0, help="кадров в секунду на сервер")
ap.add_argument("--mic", default=None, help="индекс/имя микрофона (см. python -m sounddevice)")
ap.add_argument("--speaker", default=None)
ap.add_argument("--show", action="store_true", help="показывать окно с камерой")
ap.add_argument("--hotkey", default="ctrl+alt+a", help="горячая клавиша вызова; '' = выключить")
ap.add_argument("--wake", default="", help="слово-активатор, например 'ассистент' (распознаётся локально)")
ap.add_argument("--wake-model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "vosk-model-small-ru-0.22"))
ap.add_argument("--no-beep", action="store_true", help="не подавать звуковые сигналы (услышал / записал)")
args = ap.parse_args()
if args.key:
    args.server += ("&" if "?" in args.server else "?") + "key=" + args.key
args.mic = int(args.mic) if args.mic and args.mic.lstrip("-").isdigit() else args.mic
args.speaker = int(args.speaker) if args.speaker and args.speaker.lstrip("-").isdigit() else args.speaker

SR = 16000
CHUNK = 480                                  # 30 мс - то, что ждёт webrtcvad на сервере
playing = threading.Event()
send_lock = threading.Lock()
ws = None
wake_q = queue.Queue(maxsize=200)
last_wake = 0.0


def connected():
    return ws is not None and ws.sock is not None and ws.sock.connected


def send_text(obj):
    with send_lock:
        ws.send(json.dumps(obj, ensure_ascii=False))


def send_bin(b):
    with send_lock:
        ws.send(b, websocket.ABNF.OPCODE_BINARY)   # WebSocketApp has no send_binary()


sent_frames = 0
level_warned = False


def on_message(_, msg):
    d = json.loads(msg)
    t = d.get("type")
    if t == "say":
        print(f"[assistant] {d['text']}")
        threading.Thread(target=play_wav, args=(base64.b64decode(d["wav"]),), daemon=True).start()
    elif t == "heard":
        print(f"[you] {d['text']}")
    elif t == "status":
        print(f"[status] {d}")
    elif t == "vad":
        st = d.get("state")
        if st == "start":
            print("[listening...]")
        elif st == "stop":
            print("[processing]")
            if time.time() - last_wake < 180 and not args.no_beep:
                threading.Thread(target=beep, args=("done",), daemon=True).start()
    elif t == "end":
        print("[session ended]")


def play_wav(data):
    playing.set()
    try:
        with wave.open(io.BytesIO(data)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            sd.play(pcm, w.getframerate(), device=args.speaker)
            sd.wait()
        time.sleep(0.4)                       # чтобы хвост эха не попал в микрофон
    finally:
        playing.clear()


def tone(freq, dur, vol=6000):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, np.minimum(t, dur - t) * 100)          # 10 ms fade in/out, no click
    return (np.sin(2 * np.pi * freq * t) * vol * env).astype(np.int16)


def beep(kind="wake"):
    """wake: one short rising blip = услышал, говори. done: two blips = фразу записал, обрабатываю."""
    if kind == "wake":
        pcm = np.concatenate([tone(660, 0.07), tone(990, 0.09)])
        sd.play(pcm, SR, device=args.speaker)     # mic is NOT muted: the person may already be talking
        sd.wait()
        return
    pcm = np.concatenate([tone(880, 0.07), np.zeros(int(SR * 0.05), np.int16), tone(880, 0.07)])
    playing.set()
    try:
        sd.play(pcm, SR, device=args.speaker)
        sd.wait()
    finally:
        playing.clear()


def wake(source):
    """сообщить серверу, что ассистента позвали (без лица в кадре)"""
    global last_wake
    if time.time() - last_wake < 3:
        return
    last_wake = time.time()
    print(f"[wake] {source}")
    if not args.no_beep:
        threading.Thread(target=beep, args=("wake",), daemon=True).start()
    if connected():
        try:
            send_text({"type": "wake", "source": source, "word": args.wake})
        except Exception as e:
            print("wake send failed:", e)
    else:
        print("not connected to server")


def audio_cb(indata, frames, t, status):
    if playing.is_set():
        return
    b = bytes(indata)
    if args.wake:
        try:
            wake_q.put_nowait(b)
        except queue.Full:
            pass
    if not connected():
        return
    global sent_frames, level_warned
    try:
        send_bin(b)
        sent_frames += 1
    except Exception as e:
        if sent_frames % 100 == 0:
            print("audio send failed:", e)
    if sent_frames == 200 and not level_warned:                       # ~6 s after connect: is the mic alive?
        level_warned = True
        rms = float(np.sqrt(np.mean(np.frombuffer(b, np.int16).astype(np.float32) ** 2)))
        if rms < 5:
            print(f"[mic] уровень сигнала ~0 ({rms:.0f}) - проверьте микрофон: python -m sounddevice, затем --mic N")


# ---------------- wake word (Vosk, offline)
def ensure_vosk_model(path):
    if os.path.isdir(path):
        return path
    import urllib.request, zipfile
    name = os.path.basename(path.rstrip("/\\"))
    url = f"https://alphacephei.com/vosk/models/{name}.zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    zpath = path + ".zip"
    print(f"downloading wake-word model {url} ...")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(os.path.dirname(path))
    os.remove(zpath)
    return path


def wake_loop():
    """Vosk in full-vocabulary mode: a limited grammar gives too many false hits on similar words.
    Partial results are checked so the phrase is caught ~0.7 s after it is spoken, even mid-sentence."""
    import re
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    model = Model(ensure_vosk_model(args.wake_model))
    phrase = args.wake.lower().strip()
    rx = re.compile(r"(^|\s)" + re.escape(phrase) + r"(\s|$)")
    rec = KaldiRecognizer(model, SR)
    print(f"wake word ready: '{phrase}'")
    buf, silent, active = b"", 0, False
    while True:
        chunk = wake_q.get()
        rms = np.sqrt(np.mean(np.frombuffer(chunk, np.int16).astype(np.float32) ** 2))
        silent = silent + 1 if rms < 250 else 0
        if silent > 33:                         # ~1 s of silence: nothing to decode, utterance boundary
            if active:
                rec.Reset()
                active = False
            buf = b""
            continue
        active = True
        buf += chunk
        if len(buf) < CHUNK * 2 * 4:            # decode 120 ms at a time
            continue
        data, buf = buf, b""
        try:
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "")
            else:
                text = json.loads(rec.PartialResult()).get("partial", "")
        except Exception as e:
            print("vosk error:", e)
            continue
        if rx.search(text):
            rec.Reset()
            active = False
            wake("voice")


def hotkey_loop():
    try:
        import keyboard
    except ImportError:
        print("pip install keyboard  -  чтобы работала горячая клавиша")
        return
    try:
        keyboard.add_hotkey(args.hotkey, lambda: wake("hotkey"))
        print(f"hot key ready: {args.hotkey}")
    except Exception as e:
        print("hotkey failed:", e)


def camera_loop():
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    period = 1.0 / args.fps
    last = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.5)
            continue
        if args.show:
            cv2.imshow("edge", frame)
            cv2.waitKey(1)
        if time.time() - last >= period and connected():
            last = time.time()
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            try:
                send_text({"type": "frame", "jpeg": base64.b64encode(jpg.tobytes()).decode()})
            except Exception:
                pass


def stdin_loop():
    """ввод текста с клавиатуры для отладки без микрофона; пустая строка = вызов"""
    while True:
        try:
            line = input()
        except EOFError:
            return
        if not connected():
            continue
        if not line.strip():
            wake("stdin")
        else:
            send_text({"type": "text", "text": line.strip()})


if args.camera >= 0:
    threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=stdin_loop, daemon=True).start()
if args.wake:
    threading.Thread(target=wake_loop, daemon=True).start()
if args.hotkey:
    hotkey_loop()
stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=CHUNK, device=args.mic, callback=audio_cb)
stream.start()

while True:
    try:
        ws = websocket.WebSocketApp(args.server, on_message=on_message,
                                    on_open=lambda w: print("connected to", args.server),
                                    on_error=lambda w, e: print("ws error:", e))
        ws.run_forever(ping_interval=20)
    except KeyboardInterrupt:
        break
    print("reconnecting in 3s...")
    time.sleep(3)
