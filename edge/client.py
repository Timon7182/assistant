"""Edge client: webcam + mic + speaker -> door-assistant server.
   pip install -r requirements.txt
   python client.py --server ws://192.168.88.49:8060/ws/edge --camera 0
"""
import argparse, base64, io, json, threading, time, wave

import cv2, numpy as np, sounddevice as sd, websocket

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="ws://192.168.88.49:8060/ws/edge")
ap.add_argument("--key", default="", help="API_KEY сервера")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--fps", type=float, default=2.0, help="кадров в секунду на сервер")
ap.add_argument("--mic", default=None, help="индекс/имя микрофона (см. python -m sounddevice)")
ap.add_argument("--speaker", default=None)
ap.add_argument("--show", action="store_true", help="показывать окно с камерой")
args = ap.parse_args()
if args.key:
    args.server += ("&" if "?" in args.server else "?") + "key=" + args.key

SR = 16000
CHUNK = 480                                  # 30 мс - то, что ждёт webrtcvad на сервере
playing = threading.Event()
send_lock = threading.Lock()
ws = None


def send_text(obj):
    with send_lock:
        ws.send(json.dumps(obj))


def send_bin(b):
    with send_lock:
        ws.send_binary(b)


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


def audio_cb(indata, frames, t, status):
    if playing.is_set() or ws is None or not ws.sock or not ws.sock.connected:
        return
    try:
        send_bin(bytes(indata))
    except Exception:
        pass


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
        if time.time() - last >= period and ws and ws.sock and ws.sock.connected:
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
    """ввод текста с клавиатуры для отладки без микрофона"""
    while True:
        try:
            line = input()
        except EOFError:
            return
        if line.strip() and ws and ws.sock and ws.sock.connected:
            send_text({"type": "text", "text": line.strip()})


threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=stdin_loop, daemon=True).start()
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
