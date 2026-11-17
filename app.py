"""
모자이크 덧대기 생성기 (로컬 웹 화면)
 - 영상 업로드(또는 경로 입력) -> Lada v4 fast로 원본 모자이크 위치 탐지
 - 영상 전체 위치를 시간순으로 다듬기 (점멸/크기 흔들림 방지)
 - 주변 피부색으로 채운 뒤 부드러운 블러로 덮기 -> 소리 그대로 붙여서 mp4 저장
"""
import os, sys, json, time, uuid, threading, subprocess, traceback, urllib.request
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, Response

import cover

BASE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(BASE, "jobs")
MODEL_PATH = os.path.join(BASE, "models", "lada_mosaic_detection_model_v4_fast.pt")
MODEL_URL = "https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt"
FAKE = os.environ.get("MOSAIC_FAKE") == "1"   # 개발용 테스트 모드
os.makedirs(JOBS, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = None
state = {}            # job_id -> dict
run_lock = threading.Lock()
_model = None


# ---------------- 도구 준비 ----------------
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def ensure_model(job):
    if FAKE or os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    tmp = MODEL_PATH + ".part"
    job["msg"] = "Lada 탐지 모델 내려받는 중 (처음 한 번만)"

    def hook(n, bs, total):
        if total > 0:
            job["msg"] = f"Lada 탐지 모델 내려받는 중 {min(100, n * bs * 100 // total)}% (처음 한 번만)"
    urllib.request.urlretrieve(MODEL_URL, tmp, hook)
    os.replace(tmp, MODEL_PATH)


def get_model(job=None):
    global _model
    if _model is None:
        if job is not None:
            job["msg"] = "탐지 모델 불러오는 중 (처음 한 번은 30초~1분 걸려요)"
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model


# ---------------- 1단계: 위치 탐지 ----------------
def detect_fake(frames):
    out = []
    for f in frames:
        m = (f.astype(np.int32).sum(2) < 120).astype(np.uint8)
        ys, xs = np.nonzero(m)
        out.append(None if len(xs) < 30 else [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return out


def detect_lada(frames):
    import torch
    model = get_model()
    dev = 0 if torch.cuda.is_available() else "cpu"
    res = model.predict(frames, imgsz=640, conf=0.25, verbose=False, device=dev, half=(dev == 0))
    out = []
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            out.append(None)
            continue
        b = r.boxes.xyxy.cpu().numpy()
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        out.append([float(v) for v in b[int(area.argmax())]])   # 가장 큰 모자이크 하나
    return out


def detect_all(src, job, n):
    cap = cv2.VideoCapture(src)
    detect = detect_fake if FAKE else detect_lada
    if not FAKE:
        get_model(job)
    try:
        import torch
        how = "그래픽카드" if torch.cuda.is_available() else "CPU (느림: 그래픽카드를 못 쓰는 상태)"
    except Exception:
        how = "CPU"
    boxes, batch, i = {}, [], 0
    while True:
        ok, f = cap.read()
        if ok:
            batch.append(f)
        if batch and (len(batch) == 8 or not ok):
            for k, b in enumerate(detect(batch)):
                boxes[str(i + k)] = b
            i += len(batch)
            batch = []
            job["pct"] = int(i * 50 / max(n, 1))
            job["msg"] = f"원본 모자이크 위치 찾는 중 {i}/{n} 프레임 · {how}"
        if not ok:
            break
    cap.release()
    return boxes, i


# ---------------- 2단계: 덮기 + 저장 ----------------
def render(src, boxes_json, dst, job, n):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = cover.load_boxes(boxes_json, n)
    tr = cover.smooth_track(boxes)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
           "-i", src, "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-shortest", dst]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    col, hs, bad, i = None, None, [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if tr is not None and i < len(tr[4]) and tr[4][i]:
            f, col, d, hs = cover.cover_frame(f, tr[0][i], tr[1][i], tr[2][i], tr[3][i], col, hs)
            if d > 0.005:
                bad.append(i)
        else:
            col = None
            hs = None
        p.stdin.write(f.tobytes())
        i += 1
        if i % 15 == 0:
            job["pct"] = 50 + int(i * 50 / max(n, 1))
            job["msg"] = f"모자이크 덮는 중 {i}/{n} 프레임"
    cap.release()
    p.stdin.close()
    err = p.stderr.read().decode("utf-8", "ignore")
    if p.wait() != 0:
        raise RuntimeError("영상 저장 실패: " + err[-500:])
    miss = int(np.isnan(boxes[:, 0]).sum())
    covered = int(tr[4].sum()) if tr is not None else 0
    return {"frames": i, "missing": miss, "covered": covered, "leak": len(bad),
            "leak_times": [round(x / fps, 2) for x in bad[:30]], "fps": fps}


def run_job(jid, src, params):
    job = state[jid]
    if run_lock.locked():
        job["msg"] = "앞 작업이 끝나길 기다리는 중"
    with run_lock:
        job["msg"] = "시작하는 중"
        try:
            cover.SCALE = float(params.get("scale", cover.SCALE))
            cover.BLUR_FRAC = float(params.get("blur", cover.BLUR_FRAC))
            cover.FEATHER = float(params.get("feather", cover.FEATHER))
            cover.HAIR = params.get("hair", "1") == "1"
            cover.HAIR_REACH = float(params.get("reach", cover.HAIR_REACH))
            ensure_model(job)
            n = int(cv2.VideoCapture(src).get(cv2.CAP_PROP_FRAME_COUNT))
            t0 = time.time()
            boxes, n2 = detect_all(src, job, n)
            n = max(n, n2)
            bj = os.path.join(JOBS, jid, "boxes.json")
            json.dump(boxes, open(bj, "w"))
            out = os.path.join(JOBS, jid, os.path.splitext(os.path.basename(src))[0] + "_cover.mp4")
            rep = render(src, bj, out, job, n)
            rep["sec"] = round(time.time() - t0, 1)
            job.update(pct=100, msg="완료", done=True, out=out, report=rep)
        except Exception as e:
            traceback.print_exc()
            job.update(error=str(e), done=True, msg="오류")


# ---------------- 화면 ----------------
@app.get("/quit")
def quit_server():
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return "bye"


@app.get("/")
def index():
    return Response(open(os.path.join(BASE, "index.html"), encoding="utf-8").read(), mimetype="text/html")


@app.post("/start")
def start():
    jid = uuid.uuid4().hex[:8]
    d = os.path.join(JOBS, jid)
    os.makedirs(d)
    path = (request.form.get("path") or "").strip().strip('"')
    if path:
        if not os.path.exists(path):
            return jsonify(error="그 경로에 파일이 없어요: " + path), 400
        src = path
    else:
        f = request.files.get("file")
        if not f:
            return jsonify(error="영상 파일을 선택해 주세요"), 400
        src = os.path.join(d, os.path.basename(f.filename) or "input.mp4")
        f.save(src)
    state[jid] = {"pct": 0, "msg": "대기 중", "done": False}
    threading.Thread(target=run_job, args=(jid, src, request.form), daemon=True).start()
    return jsonify(job=jid)


@app.get("/status/<jid>")
def status(jid):
    j = state.get(jid)
    if not j:
        return jsonify(error="없는 작업"), 404
    return jsonify({k: v for k, v in j.items() if k != "out"})


@app.get("/download/<jid>")
def download(jid):
    j = state.get(jid)
    if not j or not j.get("out"):
        return "아직 없음", 404
    return send_file(j["out"], as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    try:   # 예전에 켜 둔 생성기가 남아 있으면 끄기
        urllib.request.urlopen(f"http://127.0.0.1:{port}/quit", timeout=2).read()
        time.sleep(1.5)
    except Exception:
        pass
    print(f"\n  모자이크 생성기:  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
