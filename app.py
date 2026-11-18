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
import queue, platform


def frames(src, depth=16):
    """영상 프레임을 다른 일과 동시에 미리 읽어 둠 (읽는 시간만큼 빨라짐)"""
    q = queue.Queue(maxsize=depth)

    def reader():
        cap = cv2.VideoCapture(src)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            q.put(f)
        cap.release()
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()
    while True:
        f = q.get()
        if f is None:
            return
        yield f


def fmt_sec(s):
    s = int(max(0, s))
    return f"{s // 60}분 {s % 60}초" if s >= 60 else f"{s}초"


_enc = None


_enc_note = ""


def encoder_args():
    """그래픽카드 인코더(NVENC)가 되면 그걸로, 안 되면 CPU(x264)로 저장. 실패 이유는 _enc_note에 남김"""
    global _enc, _enc_note
    if _enc is None:
        test = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=1920x1080:d=0.3",
                "-pix_fmt", "yuv420p", "-c:v", "h264_nvenc", "-preset", "p4", "-f", "null", "-"]
        try:
            r = subprocess.run(test, capture_output=True, timeout=60)
            ok = r.returncode == 0
            if not ok:
                _enc_note = r.stderr.decode("utf-8", "ignore").strip()[-300:]
        except Exception as e:
            ok = False
            _enc_note = str(e)[:300]
        _enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "17", "-b:v", "0"], "그래픽카드(NVENC)") if ok \
            else (["-c:v", "libx264", "-crf", "16", "-preset", "veryfast"], "CPU(x264)")
    return _enc


def sysinfo():
    info = {"os": platform.platform(), "cpu": platform.processor() or platform.machine(), "cores": os.cpu_count()}
    try:   # Windows: CPU 제품명
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 1e9, 1)
        else:
            info["gpu"] = "없음 (CPU로 탐지 — 느림)"
    except Exception:
        info["gpu"] = "확인 불가"
    info["encoder"] = encoder_args()[1]
    return info
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


def predict_safe(model, frames, dev):
    """그래픽카드 메모리가 모자라면 한 번에 넣는 장 수를 줄여 다시 시도, 그래도 안 되면 그 묶음만 CPU로"""
    import torch
    step = len(frames)
    while True:
        try:
            out = []
            for s in range(0, len(frames), step):
                out += model.predict(frames[s:s + step], imgsz=640, conf=0.25, verbose=False, device=dev, half=(dev == 0))
            return out
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or dev == "cpu":
                raise
            torch.cuda.empty_cache()
            if step > 1:
                step = max(1, step // 2)
            else:
                dev = "cpu"


def detect_lada(frames):
    """반환: 프레임마다 (가장 큰 모자이크 상자, 찾은 모자이크 전부 [x1,y1,x2,y2,확신도,윤곽])"""
    import torch
    model = get_model()
    dev = 0 if torch.cuda.is_available() else "cpu"
    res = predict_safe(model, frames, dev)
    out = []
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            out.append((None, []))
            continue
        b = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        polys = []
        try:                                   # 모자이크의 실제 윤곽 (있으면)
            for p in r.masks.xy:
                p = cv2.approxPolyDP(np.asarray(p, np.float32).reshape(-1, 1, 2), 2.0, True)
                polys.append(p.reshape(-1, 2).round().astype(int).tolist())
        except Exception:
            polys = []
        allb = []
        for k in range(len(b)):
            allb.append([round(float(v), 1) for v in b[k]] + [round(float(conf[k]), 3), polys[k] if k < len(polys) else None])
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        out.append(([float(v) for v in b[int(area.argmax())]], allb))   # 가장 큰 모자이크 + 전부
    return out


def detect_fake_all(frames):
    return [(b, [b + [1.0, None]] if b else []) for b in detect_fake(frames)]


def detect_all(src, job, n):
    detect = detect_fake_all if FAKE else detect_lada
    if not FAKE:
        get_model(job)
    try:
        import torch
        how = "그래픽카드" if torch.cuda.is_available() else "CPU (느림: 그래픽카드를 못 쓰는 상태)"
    except Exception:
        how = "CPU"
    boxes, all_boxes, exps, batch, i = {}, {}, {}, [], 0
    nude = None
    if not FAKE:
        try:                                             # 노출 탐지도 여기서 같이 (매 프레임, 그래픽카드가 있으면 그래픽카드로)
            import cover2
            nude = cover2.NudeBatch()
            job["nude_how"] = nude.how
        except Exception:
            traceback.print_exc()
            nude = None
    t0 = time.time()
    it = frames(src)
    while True:
        f = next(it, None)
        if f is not None:
            batch.append(f)
        if batch and (len(batch) == 8 or f is None):
            for k, (b, allb) in enumerate(detect(batch)):
                boxes[str(i + k)] = b
                all_boxes[str(i + k)] = allb
            if nude is not None:
                try:
                    for k, e in enumerate(nude.detect(batch)):
                        exps[str(i + k)] = e
                except Exception:
                    traceback.print_exc()
                    nude = None
                    exps = {}
            i += len(batch)
            batch = []
            job["pct"] = int(i * 50 / max(n, 1))
            el = time.time() - t0
            left = el / max(i, 1) * (n - i)
            job["msg"] = f"원본 모자이크 위치·노출 찾는 중 {i}/{n} 프레임 · {how} · 이 단계 남은 시간 약 {fmt_sec(left)}"
        if f is None:
            break
    return boxes, all_boxes, exps if (nude is not None and len(exps) == i) else None, i


# ---------------- 2단계: 덮기 + 저장 ----------------
WARMUP = 30   # 조각 앞부분에서 미리 돌려 두는 프레임 수 (앞 조각과 블러가 자연스럽게 이어지게)


def _dec(src, w0, total, W, H, fps):
    return subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-ss", f"{w0 / fps:.6f}", "-i", src,
                             "-frames:v", str(total), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 4)


def _scan_worker(src, start, count, W, H, fps, entries, exps, params, outp, counter, q):
    """1단계(훑기): 이 조각의 프레임마다 '반드시 덮을 곳'을 모아 파일로 저장"""
    try:
        import cover2, pickle
        for k, v in params.items():
            setattr(cover2, k, v)
        w0 = max(0, start - WARMUP)
        total = start + count - w0
        dec = _dec(src, w0, total, W, H, fps)
        A = cover2.Analyzer()
        res, fsz = [], W * H * 3
        for j in range(total):
            buf = dec.stdout.read(fsz)
            if len(buf) < fsz:
                break
            f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            i = w0 + j
            r = A.analyze(f, i, entries[j] if j < len(entries) else [], exps[j] if exps is not None and j < len(exps) else None)
            if i >= start:
                res.append(r)
                with counter.get_lock():
                    counter.value += 1
        dec.stdout.close(); dec.wait()
        pickle.dump(res, open(outp, "wb"))
        q.put((start, len(res), None))
    except Exception:
        q.put((start, 0, traceback.format_exc()[-800:]))


def _paint_worker(src, start, count, W, H, fps, ells, polys, params, enc, seg, threads, counter, q):
    """3단계(그리기): 계획된 타원대로만 그려서 조각 파일로 저장"""
    try:
        import cover2
        for k, v in params.items():
            setattr(cover2, k, v)
        w0 = max(0, start - WARMUP)
        total = start + count - w0
        dec = _dec(src, w0, total, W, H, fps)
        encx = list(enc) + (["-threads", str(threads)] if "libx264" in enc else [])
        out = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-", *encx, "-pix_fmt", "yuv420p", seg],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        P = cover2.Painter()
        done, fsz = 0, W * H * 3
        for j in range(total):
            buf = dec.stdout.read(fsz)
            if len(buf) < fsz:
                break
            f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            i = w0 + j
            f, _ = P.paint(f, ells[j] if j < len(ells) else [], polys[j] if j < len(polys) else [])
            if i >= start:
                out.stdin.write(f.tobytes())
                done += 1
                with counter.get_lock():
                    counter.value += 1
        out.stdin.close()
        err = out.stderr.read().decode("utf-8", "ignore")
        dec.stdout.close(); dec.wait()
        if out.wait() != 0:
            raise RuntimeError("조각 저장 실패: " + err[-300:])
        q.put((start, done, None))
    except Exception:
        q.put((start, 0, traceback.format_exc()[-800:]))


def _run_parallel(job, K, n, make_args, target, label, p0=50, p1=99):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    q = ctx.Queue()
    procs = []
    for k in range(K):
        p = ctx.Process(target=target, args=make_args(k) + (counter, q), daemon=True)
        p.start()
        procs.append(p)
    t0, results = time.time(), []
    while len(results) < K:
        try:
            results.append(q.get(timeout=1))
        except Exception:
            pass
        i = counter.value
        el = time.time() - t0
        left = el / i * (n - i) if i else 0
        job["msg"] = f"{label} {i}/{n} 프레임 · CPU {K}개 동시 처리 · 이 단계 남은 시간 약 {fmt_sec(left)}"
        job["pct"] = p0 + int(i * (p1 - p0) / max(n, 1))
        if not any(p.is_alive() for p in procs) and q.empty() and len(results) < K:
            raise RuntimeError(label + " 작업 일부가 비정상 종료됐어요")
    for p in procs:
        p.join()
    errs = [r[-1] for r in results if r[-1]]
    if errs:
        raise RuntimeError(errs[0])
    return results


def render(src, boxes_json, dst, job, n):
    import cover2, pickle
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    allp = os.path.join(os.path.dirname(boxes_json), "boxes_all.json")
    if os.path.exists(allp):
        all_boxes = json.load(open(allp))
    else:
        one = json.load(open(boxes_json))
        all_boxes = {k: ([v + [1.0, None]] if v else []) for k, v in one.items()}
    per = cover2.build_tracks(all_boxes, n, W, H)
    ep = os.path.join(os.path.dirname(boxes_json), "exposures.json")
    ex = None
    if os.path.exists(ep):
        try:
            ed = json.load(open(ep))
            ex = [ed.get(str(i), {"must": [], "faces": []}) for i in range(n)]
        except Exception:
            ex = None
    enc, enc_name = encoder_args()
    cores = os.cpu_count() or 2
    K = max(1, min(cores, 8, n // 60))
    threads = max(1, cores // K)
    params = {k: getattr(cover2, k) for k in ("SCALE", "BLUR_FRAC", "FEATHER_W", "HAIR", "HAIR_REACH")}
    d = os.path.dirname(dst)
    bounds = [round(n * k / K) for k in range(K + 1)]
    wb = lambda k: max(0, bounds[k] - WARMUP)

    # 1단계: 훑기 (덮을 곳 모으기)
    scans = [os.path.join(d, f"scan{k:02d}.pkl") for k in range(K)]
    _run_parallel(job, K, n, lambda k: (src, bounds[k], bounds[k + 1] - bounds[k], W, H, fps, per[wb(k):bounds[k + 1]], (ex[wb(k):bounds[k + 1]] if ex else None), params, scans[k]),
                  _scan_worker, "1/3 덮을 곳 훑는 중", 50, 75)
    info = []
    for sp in scans:
        info += pickle.load(open(sp, "rb"))
        os.remove(sp)
    info += [None] * (n - len(info))

    # 2단계: 계획 (키프레임)
    job["msg"] = "2/3 장면별 블러 크기·위치 정하는 중"
    ells, leak = cover2.plan(per, info, n, W, H)
    polys = [[e[2] for e in per[i]] for i in range(n)]
    try:                                               # 블러 계획 기록 (구간·위치 확인용, 나중에 편집 프로그램으로 넘길 때도 씀)
        json.dump(ells, open(os.path.join(d, "blur_plan.json"), "w"))
    except Exception:
        pass

    # 3단계: 그리기 + 저장
    KP = min(K, 5) if "nvenc" in enc[1] else K
    bp = [round(n * k / KP) for k in range(KP + 1)]
    segs = [os.path.join(d, f"part{k:02d}.mp4") for k in range(KP)]
    wp = lambda k: max(0, bp[k] - WARMUP)
    tp = max(1, cores // KP)
    results = _run_parallel(job, KP, n, lambda k: (src, bp[k], bp[k + 1] - bp[k], W, H, fps, ells[wp(k):bp[k + 1]], polys[wp(k):bp[k + 1]],
                                                      params, enc, segs[k], tp), _paint_worker, f"3/3 블러 그리고 저장 중 ({enc_name})", 77, 99)
    job["msg"] = "조각 이어 붙이는 중"
    lst = os.path.join(d, "parts.txt")
    with open(lst, "w", encoding="utf-8") as fp:
        for s in segs:
            fp.write("file '" + s.replace("\\", "/").replace("'", "'\\''") + "'\n")
    r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", src,
                        "-map", "0:v", "-map", "1:a?", "-c", "copy", "-shortest", dst], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("영상 저장 실패: " + r.stderr.decode("utf-8", "ignore")[-500:])
    for s in segs + [lst]:
        try:
            os.remove(s)
        except Exception:
            pass
    frames_done = sum(r[1] for r in results)
    miss = sum(1 for k in range(n) if not all_boxes.get(str(k)))
    covered = sum(1 for e in ells if e)
    return {"frames": frames_done, "missing": miss, "covered": covered, "leak": len(leak),
            "leak_times": [round(x / fps, 2) for x in leak[:30]], "fps": fps, "workers": KP, "encoder_note": _enc_note}


def run_job(jid, src, params):
    job = state[jid]
    if run_lock.locked():
        job["msg"] = "앞 작업이 끝나길 기다리는 중"
    with run_lock:
        job["msg"] = "시작하는 중"
        try:
            import cover2
            cover2.SIZE_K = float(params.get("scale", cover2.SIZE_K))
            cover2.BLUR_FRAC = float(params.get("blur", cover2.BLUR_FRAC))
            cover2.FEATHER_W = float(params.get("feather", cover2.FEATHER_W))
            cover2.HAIR = params.get("hair", "1") == "1"
            cover2.HAIR_REACH = float(params.get("reach", cover2.HAIR_REACH))
            ensure_model(job)
            n = int(cv2.VideoCapture(src).get(cv2.CAP_PROP_FRAME_COUNT))
            t0 = time.time()
            boxes, all_boxes, exps, n2 = detect_all(src, job, n)
            t_detect = time.time() - t0
            n = max(n, n2)
            bj = os.path.join(JOBS, jid, "boxes.json")
            json.dump(boxes, open(bj, "w"))
            try:   # 찾은 원본 모자이크 전부 (여러 개·윤곽까지) — 다음 개선용
                json.dump(all_boxes, open(os.path.join(JOBS, jid, "boxes_all.json"), "w"), separators=(",", ":"))
            except Exception:
                traceback.print_exc()
            ep = os.path.join(JOBS, jid, "exposures.json")
            if exps is not None:
                json.dump(exps, open(ep, "w"), separators=(",", ":"))
            out = os.path.join(JOBS, jid, os.path.splitext(os.path.basename(src))[0] + "_cover.mp4")
            t1 = time.time()
            rep = render(src, bj, out, job, n)
            rep["t_detect"] = round(t_detect, 1)
            rep["t_cover"] = round(time.time() - t1, 1)
            rep["sec"] = round(time.time() - t0, 1)
            rep["video_sec"] = round(rep["frames"] / rep["fps"], 1)
            rep["encoder"] = encoder_args()[1]
            job.update(pct=100, msg="완료", done=True, out=out, report=rep)
        except Exception as e:
            traceback.print_exc()
            info = ""
            try:   # 그래픽카드 상태를 오류와 함께 보여줌 (원인 확인용)
                import torch
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    info = (f" [그래픽카드 {torch.cuda.get_device_name(0)}: 전체 {total / 1e9:.1f}GB 중 남은 {free / 1e9:.1f}GB, "
                            f"이 프로그램이 쓰는 중 {torch.cuda.memory_reserved() / 1e9:.1f}GB · 진행 {job.get('pct', 0)}%]")
            except Exception:
                pass
            job.update(error=str(e)[:300] + info, done=True, msg="오류")


# ---------------- 화면 ----------------
@app.get("/sysinfo")
def sysinfo_api():
    return jsonify(sysinfo())


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
