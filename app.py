"""
모자이크 덧대기 생성기 (로컬 웹 화면)
 - 영상 업로드(또는 경로 입력) -> Lada v4 fast로 원본 모자이크 위치 탐지
 - 영상 전체 위치를 시간순으로 다듬기 (점멸/크기 흔들림 방지)
 - 주변 피부색으로 채운 뒤 부드러운 블러로 덮기 -> 원본과 같은 형식(압축 방식·화질·소리 트랙)으로 저장
"""
import os, sys, json, time, uuid, threading, subprocess, traceback, urllib.request
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, Response

import cover
import media

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
from fractions import Fraction


def frames(src, depth=16, table=None):
    """영상 프레임을 다른 일과 동시에 미리 읽어 둠 (읽는 시간만큼 빨라짐).
    그리기 단계와 똑같이 ffmpeg로 풀고, 프레임마다 시간을 table에 기록 (다른 단계에서 프레임 번호를 정확히 맞추는 데 씀)"""
    info = media.probe(src)
    W, H = info["W"], info["H"]
    q = queue.Queue(maxsize=depth)

    def reader():
        d = Dec(src, 0, 10 ** 9, W, H, info["fps"], seek=False, threads=max(2, min(8, (os.cpu_count() or 4) // 2)))   # 혼자 읽으니까 CPU를 넉넉히
        while True:
            got = d.read()
            if got is None:
                break
            if table is not None:
                table.append(got[2])
            q.put(np.frombuffer(got[1], np.uint8).reshape(H, W, 3).copy())
        d.close()
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


class ProxyWriter:
    """위치 찾기 단계에서 읽는 프레임으로 검수용 미리보기 영상도 같이 만듦 (영상을 한 번 더 읽지 않게)"""

    def __init__(self, path, W, H, fps_str):
        self.path, self.h = path, 540
        self.w = max(2, int(round(W * 540 / H / 2)) * 2)
        self.q = queue.Queue(maxsize=32)
        self.p = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{self.w}x{self.h}", "-r", fps_str, "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", "26", "-g", "15", "-pix_fmt", "yuv420p", "-threads", "2", "-an", path + ".part.mp4"],
                                  stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.ok = True
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            if not self.ok:
                continue
            try:
                self.p.stdin.write(cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_AREA).tobytes())
            except Exception:
                self.ok = False

    def put(self, f):
        self.q.put(f)

    def close(self):
        self.q.put(None)
        self.t.join()
        try:
            self.p.stdin.close()
        except Exception:
            pass
        if self.p.wait() == 0 and self.ok:
            os.replace(self.path + ".part.mp4", self.path)
            return True
        return False


NUDE_EVERY = 3   # 얼굴·노출 찾기는 3프레임마다 (모자이크 옆 노출·얼굴 위치만 쓰므로 충분)


def detect_all(src, job, n, proxy=None, table=None):
    detect = detect_fake_all if FAKE else detect_lada
    if not FAKE:
        get_model(job)
    try:
        import torch
        how = "그래픽카드" if torch.cuda.is_available() else "CPU (느림: 그래픽카드를 못 쓰는 상태)"
    except Exception:
        how = "CPU"
    job["det_how"] = how
    boxes, all_boxes, exps, batch, i = {}, {}, {}, [], 0
    nude = None
    if not FAKE:
        try:                                             # 얼굴·노출 찾기도 여기서 같이 (그래픽카드가 있으면 그래픽카드로)
            import cover2
            nude = cover2.NudeBatch()
            job["nude_how"] = nude.how
        except Exception:
            traceback.print_exc()
            nude = None
    last_e = {"must": [], "faces": []}
    t0 = time.time()
    it = frames(src, table=table)
    while True:
        f = next(it, None)
        if f is not None:
            batch.append(f)
            if proxy is not None:
                proxy.put(f)
        if batch and (len(batch) == 8 or f is None):
            for k, (b, allb) in enumerate(detect(batch)):
                boxes[str(i + k)] = b
                all_boxes[str(i + k)] = allb
            if nude is not None:
                try:
                    sel = [k for k in range(len(batch)) if (i + k) % NUDE_EVERY == 0]
                    res = dict(zip(sel, nude.detect([batch[k] for k in sel]))) if sel else {}
                    for k in range(len(batch)):
                        if k in res:
                            last_e = res[k]
                        exps[str(i + k)] = last_e
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


class Dec:
    """영상의 start번째 프레임부터 count장을 정확히 풀어 읽음.
    시간으로 건너뛰면 영상 형식에 따라 1~2프레임 어긋날 수 있어서, 1초 앞에서부터 풀고 각 프레임의 시간을 보고 번호를 맞춤.
    table(위치 찾기 단계에서 기록한 프레임별 시간)이 있으면 그걸로 번호를 찾음 (시간이 고르지 않은 영상도 정확)"""

    def __init__(self, src, start, count, W, H, fps, pix_fmt="bgr24", fsz=None, t0=0.0, table=None, seek=True, threads=2):
        self.src, self.start, self.end, self.fps, self.t0 = src, start, start + count, fps, t0
        self.pix_fmt, self.fsz = pix_fmt, fsz or W * H * 3
        self.table, self.threads = table, threads
        self.lut = {round(t * 1000, 1): k for k, t in enumerate(table)} if table else None
        self.order = 0
        self.last = None
        if not seek or start <= 0:
            ss = 0.0
        elif table and start < len(table):
            ss = max(0.0, table[start] - table[0] - 1.0)
        else:
            ss = max(0.0, start / fps - 1.0)
        self._open(ss)

    def _open(self, ss):
        self.ss = ss
        self.p = subprocess.Popen([ffmpeg_exe(), "-hide_banner", "-nostats", "-loglevel", "info", "-threads", str(self.threads),
                                   "-filter_threads", str(max(1, self.threads)), "-copyts",
                                   *(["-ss", f"{ss:.6f}"] if ss > 0 else []), "-i", self.src, "-map", "0:v:0",
                                   "-vf", "showinfo=checksum=0", "-fps_mode", "passthrough",   # 프레임을 복제·생략하지 않고 풀린 그대로
                                   "-f", "rawvideo", "-pix_fmt", self.pix_fmt, "-"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.fsz * 4)
        self.tq = queue.Queue()

        def rd(p=self.p, tq=self.tq):
            for line in iter(p.stderr.readline, b""):
                k = line.find(b"pts_time:")
                if k >= 0:
                    try:
                        tq.put(float(line[k + 9:].split()[0]))
                    except Exception:
                        tq.put(None)
            tq.put("end")
        threading.Thread(target=rd, daemon=True).start()
        self.first = True

    def _index(self, t):
        if self.ss == 0 and not self.lut:                 # 처음부터 순서대로 읽는 중이면 순서가 곧 번호
            return self.order
        if self.lut:
            k = self.lut.get(round(t * 1000, 1))
            if k is not None:
                return k
            if self.last is not None:                     # 시간표에 없는 시간이면 바로 앞 프레임 다음 번호로
                return self.last + 1
            return int(round((t - self.table[0]) * self.fps))
        return int(round((t - self.t0) * self.fps))

    def read(self):
        """→ (프레임 번호, 바이트, 프레임 시간) 또는 None(끝)"""
        while True:
            buf = self.p.stdout.read(self.fsz)
            if len(buf) < self.fsz:
                return None
            t = self.tq.get(timeout=120)
            if t == "end":
                return None
            idx = self._index(t if t is not None else 0.0)
            self.order += 1
            self.last = idx
            if self.first:
                self.first = False
                if idx > self.start and self.ss > 0:       # 건너뛰기가 너무 멀리 갔으면 처음부터 다시 (느리지만 정확)
                    self.close()
                    self.order, self.last = 0, None
                    self._open(0.0)
                    continue
            if idx < self.start:
                continue
            if idx >= self.end:
                return None
            return idx, buf, t

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass
        try:
            self.p.stdout.close()
        except Exception:
            pass


def _low_priority():
    """생성 작업은 낮은 우선순위로 (작업 중에도 컴퓨터를 쓸 수 있게). ffmpeg 같은 하위 작업도 그대로 물려받음"""
    try:
        import psutil
        pr = psutil.Process()
        if os.name == "nt":
            pr.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            pr.nice(10)
    except Exception:
        pass


def _cpu_workers(n):
    """동시에 쓸 CPU 수: 작업 중에도 컴퓨터를 쓸 수 있게 둘은 항상 비워 둠 (최대 8개)"""
    cores = os.cpu_count() or 2
    return max(1, min(cores - 2, 8, n // 60))


def _scan_worker(src, start, count, W, H, fps, t0, table, entries, exps, params, outp, counter, q):
    """1단계(훑기): 이 조각의 프레임마다 '반드시 덮을 곳'을 모아 파일로 저장"""
    _low_priority()
    try:
        import cover2, pickle
        for k, v in params.items():
            setattr(cover2, k, v)
        w0 = max(0, start - WARMUP)
        total = start + count - w0
        dec = Dec(src, w0, total, W, H, fps, t0=t0, table=table)
        A = cover2.Analyzer()
        res = [None] * count
        while True:
            got = dec.read()
            if got is None:
                break
            i, buf, _ = got
            j = i - w0
            f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            r = A.analyze(f, i, entries[j] if j < len(entries) else [], exps[j] if exps is not None and j < len(exps) else None)
            if i >= start:
                res[i - start] = r
                with counter.get_lock():
                    counter.value += 1
        dec.close()
        pickle.dump(res, open(outp, "wb"))
        q.put((start, len(res), None))
    except Exception:
        q.put((start, 0, traceback.format_exc()[-800:]))


def _paint_worker(src, start, count, W, H, fps, ells, polys, params, mi, seg, counter, q):
    """3단계(그리기): 원본을 원본 픽셀 형식(YUV) 그대로 풀고, 계획된 블러만 섞어서, 원본과 같은 압축 방식으로 조각 저장.
    블러가 없는 곳의 픽셀은 원본 그대로"""
    _low_priority()
    try:
        import cover2
        for k, v in params.items():
            setattr(cover2, k, v)
        fmt = mi["compose"]
        fsz = media.frame_bytes(fmt, W, H)
        w0 = max(0, start - WARMUP)
        total = start + count - w0
        dec = Dec(src, w0, total, W, H, fps, fmt, fsz, t0=mi["t0"], table=mi.get("table"))
        out = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", fmt,
                                "-s", f"{W}x{H}", "-r", mi["fps_str"], *mi["inp"], "-i", "-", *mi["enc"], "-threads", "2", "-f", "nut", seg],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        P = cover2.Painter()
        expect, data = start, None                       # 조각마다 정확히 count장을 순서대로 씀 (혹시 빠지면 앞 프레임으로 채움)

        def put(b_):
            try:
                out.stdin.write(b_)
            except (BrokenPipeError, OSError):
                raise RuntimeError("조각 저장 실패: " + out.stderr.read().decode("utf-8", "ignore")[-400:])
            with counter.get_lock():
                counter.value += 1
        while True:
            got = dec.read()
            if got is None:
                break
            i, buf, _ = got
            j = i - w0
            if i >= start and i < expect:                 # 같은 프레임이 또 나오면 건너뜀
                continue
            yf = media.YuvFrame(buf, fmt, W, H, mi["matrix"], mi["range"])
            changed = P.paint_yuv(yf, ells[j] if j < len(ells) else [], polys[j] if j < len(polys) else [])
            if i >= start:
                cur_ = yf.tobytes() if changed else buf
                while expect < i:
                    put(data if data is not None else cur_); expect += 1
                put(cur_); expect = i + 1
                data = cur_
        while data is not None and expect < start + count:
            put(data); expect += 1
        done = expect - start
        dec.close()
        out.stdin.close()
        err = out.stderr.read().decode("utf-8", "ignore")
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


def render(src, boxes_json, dst, job, n, times=None):
    import cover2, pickle
    times = times if times is not None else {}
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
    K = _cpu_workers(n)
    params = {k: getattr(cover2, k) for k in ("SCALE", "BLUR_FRAC", "FEATHER_W", "HAIR", "HAIR_REACH")}
    d = os.path.dirname(dst)
    bounds = [round(n * k / K) for k in range(K + 1)]
    wb = lambda k: max(0, bounds[k] - WARMUP)

    # 1단계: 훑기 (덮을 곳 모으기)
    t0 = time.time()
    scans = [os.path.join(d, f"scan{k:02d}.pkl") for k in range(K)]
    t0v = media.probe(src).get("t0", 0.0)
    table = _load_table(os.path.dirname(boxes_json))
    _run_parallel(job, K, n, lambda k: (src, bounds[k], bounds[k + 1] - bounds[k], W, H, fps, t0v, table, per[wb(k):bounds[k + 1]], (ex[wb(k):bounds[k + 1]] if ex else None), params, scans[k]),
                  _scan_worker, "1/3 덮을 곳 훑는 중", 50, 75)
    info = []
    for sp in scans:
        info += pickle.load(open(sp, "rb"))
        os.remove(sp)
    info += [None] * (n - len(info))
    times["scan"] = round(time.time() - t0, 1)

    # 2단계: 계획 (키프레임)
    job["msg"] = "2/3 장면별 블러 크기·위치 정하는 중"
    t0 = time.time()
    ells, leak, segtab = cover2.plan(per, info, n, W, H)
    times["plan"] = round(time.time() - t0, 1)
    polys = [[e[2] for e in per[i]] for i in range(n)]
    try:                                               # 블러 계획 기록 (구간·위치 확인용, 나중에 편집 프로그램으로 넘길 때도 씀)
        json.dump(ells, open(os.path.join(d, "blur_plan.json"), "w"))
    except Exception:
        pass

    # 검수 화면에서 다시 쓸 수 있게 저장
    params_save = {k: getattr(cover2, k) for k in ("SCALE", "BLUR_FRAC", "FEATHER_W", "HAIR", "HAIR_REACH", "FEATHER_MAX", "FILL_KEEP")}
    try:
        json.dump(segtab, open(os.path.join(d, "segments.json"), "w"))
        json.dump(polys, open(os.path.join(d, "polys.json"), "w"), separators=(",", ":"))
        json.dump({"src": src, "W": W, "H": H, "fps": fps, "n": n, "out": os.path.basename(dst), "params": params_save},
                  open(os.path.join(d, "review.json"), "w"), ensure_ascii=False)
    except Exception:
        traceback.print_exc()

    # 3단계: 그리기 + 저장
    results, KP = paint_save(src, ells, polys, dst, job, n, W, H, fps, params, times=times)
    frames_done = sum(r[1] for r in results)
    miss = sum(1 for k in range(n) if not all_boxes.get(str(k)))
    covered = sum(1 for e in ells if e)
    return {"frames": frames_done, "missing": miss, "covered": covered, "leak": len(leak),
            "leak_times": [round(x / fps, 2) for x in leak[:30]], "fps": fps, "workers": KP, "times": times,
            "out_fmt": times.get("out_fmt", ""), "out_name": os.path.basename(times.get("out_path", dst))}


def _load_table(d):
    """위치 찾기 단계에서 기록한 프레임별 시간표 (없으면 None: 예전 작업)"""
    try:
        return json.load(open(os.path.join(d, "pts.json")))
    except Exception:
        return None


def paint_save(src, ells, polys, dst, job, n, W, H, fps, params, p0=77, p1=99, times=None):
    """계획된 블러대로 그려서 원본과 같은 형식으로 저장 (CPU 여러 개로 나눠서) → 원본 소리 트랙 그대로 붙여 하나로.
    실제로 저장한 파일 경로를 times["out_path"]에 남김"""
    times = times if times is not None else {}
    t0 = time.time()
    info = media.probe(src)
    fmt = media.compose_fmt(info)
    enc, desc = media.encode_args(info, fmt)
    mi = {"compose": fmt, "enc": enc, "inp": media.input_args(info), "fps_str": info["fps_str"], "matrix": info["matrix"],
          "range": info["range"], "t0": info.get("t0", 0.0), "table": _load_table(os.path.dirname(dst))}
    times["out_fmt"] = desc
    K = _cpu_workers(n)
    KP = K
    d = os.path.dirname(dst)
    bp = [round(n * k / KP) for k in range(KP + 1)]
    segs = [os.path.join(d, f"part{k:02d}.nut") for k in range(KP)]
    wp = lambda k: max(0, bp[k] - WARMUP)
    results = _run_parallel(job, KP, n, lambda k: (src, bp[k], bp[k + 1] - bp[k], W, H, fps, ells[wp(k):bp[k + 1]], polys[wp(k):bp[k + 1]],
                                                      params, mi, segs[k]), _paint_worker, f"블러 그리고 저장 중 (원본과 같은 형식: {desc})", p0, p1)
    times["paint"] = round(time.time() - t0, 1)
    t0 = time.time()
    job["msg"] = "조각 이어 붙이고 원본 소리 트랙 붙이는 중"
    lst = os.path.join(d, "parts.txt")
    done_of = {r_[0]: r_[1] for r_ in results}
    fr = Fraction(info["fps_str"])
    with open(lst, "w", encoding="utf-8") as fp:      # 조각마다 길이를 적어 줌 → 이어 붙인 곳에서 시간이 1프레임씩 밀리지 않음
        for k, s_ in enumerate(segs):
            fp.write("file '" + s_.replace("\\", "/").replace("'", "'\\''") + "'\n")
            fp.write(f"duration {float(done_of.get(bp[k], bp[k + 1] - bp[k]) / fr):.9f}\n")
    base = os.path.splitext(dst)[0]
    tmpv = base + "_v.mov"
    tries = [(base + media.out_ext(info), False), (base + media.out_ext(info), True), (base + ".mov", False), (base + ".mkv", False)]
    err = ""
    for out, via in tries:
        if via:   # 바로 안 되면 영상만 먼저 이어 붙인 뒤 원본 형식으로 옮겨 담기 (H.264 MXF 등)
            r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                                "-map", "0:v", "-c", "copy", tmpv], capture_output=True)
            vin = ["-i", tmpv]
        else:
            vin = ["-f", "concat", "-safe", "0", "-i", lst]
        r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *vin, "-i", src,
                            "-map", "0:v", "-map", "1:a?", "-c", "copy", *media.mux_args(info, out), out], capture_output=True)
        if r.returncode == 0:
            break
        err = r.stderr.decode("utf-8", "ignore")[-400:]
        try:
            os.remove(out)
        except Exception:
            pass
    else:
        raise RuntimeError("영상 저장 실패: " + err)
    try:
        os.remove(tmpv)
    except Exception:
        pass
    if out != tries[0][0]:
        times["out_fmt"] = desc + f" (원본 파일 형식으로는 저장이 안 돼서 {os.path.splitext(out)[1]}로 저장: {err[-150:]})"
    times["out_path"] = out
    times["mux"] = round(time.time() - t0, 1)
    for s_ in segs + [lst]:
        try:
            os.remove(s_)
        except Exception:
            pass
    return results, KP


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
            px = None
            if os.path.splitext(src)[1].lower() not in PLAYABLE:   # 검수용 미리보기를 위치 찾기와 같이 만듦
                try:
                    mi0 = media.probe(src)
                    c_ = cv2.VideoCapture(src)
                    px = ProxyWriter(os.path.join(JOBS, jid, "proxy.mp4"), int(c_.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                     int(c_.get(cv2.CAP_PROP_FRAME_HEIGHT)), mi0["fps_str"])
                    c_.release()
                except Exception:
                    traceback.print_exc()
                    px = None
            table = []
            boxes, all_boxes, exps, n2 = detect_all(src, job, n, px, table)
            try:
                json.dump(table, open(os.path.join(JOBS, jid, "pts.json"), "w"))
            except Exception:
                traceback.print_exc()
            if px is not None and px.close():               # 미리보기에 소리 붙이기는 뒤에서 (작업과 동시에)
                audio_state[jid] = 0
                threading.Thread(target=_audio_job, args=(jid, src, os.path.join(JOBS, jid, "proxy.mp4"),
                                                          os.path.join(JOBS, jid, "proxy_av.mp4")), daemon=True).start()
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
            out = os.path.join(JOBS, jid, os.path.splitext(os.path.basename(src))[0] + "_cover" + os.path.splitext(src)[1])
            t1 = time.time()
            times = {"detect": round(t_detect, 1)}
            rep = render(src, bj, out, job, n, times)
            rep["t_detect"] = round(t_detect, 1)
            rep["t_cover"] = round(time.time() - t1, 1)
            rep["sec"] = round(time.time() - t0, 1)
            rep["video_sec"] = round(rep["frames"] / rep["fps"], 1)
            rep["det_how"] = job.get("det_how", "")
            rep["nude_how"] = job.get("nude_how", "")
            try:
                json.dump(rep, open(os.path.join(JOBS, jid, "report.json"), "w"), ensure_ascii=False)
            except Exception:
                pass
            job.update(pct=100, msg="완료", done=True, out=times.get("out_path", out), report=rep)
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



# ---------------- 검수·수정 화면 ----------------
def _jdir(jid):
    d = os.path.join(JOBS, os.path.basename(jid))
    if not os.path.exists(os.path.join(d, "review.json")):
        return None
    return d


@app.get("/jobs")
def jobs_list():
    out = []
    for jid in os.listdir(JOBS):
        d = os.path.join(JOBS, jid)
        rp = os.path.join(d, "review.json")
        if os.path.exists(rp):
            try:
                r = json.load(open(rp, encoding="utf-8"))
                out.append({"jid": jid, "name": os.path.basename(r["src"]), "t": os.path.getmtime(rp),
                            "sec": round(r["n"] / r["fps"], 1)})
            except Exception:
                pass
    out.sort(key=lambda x: -x["t"])
    return jsonify(out[:30])


@app.get("/review/<jid>")
def review_page(jid):
    if not _jdir(jid):
        return "검수할 수 있는 작업이 아니에요 (이번 버전 이후에 만든 작업만 돼요)", 404
    return Response(open(os.path.join(BASE, "review.html"), encoding="utf-8").read(), mimetype="text/html")


@app.get("/review/<jid>/data")
def review_data(jid):
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
    ep = os.path.join(d, "edits.json")
    return jsonify(meta={k: r[k] for k in ("W", "H", "fps", "n")} | {"params": r.get("params", {}), "name": os.path.basename(r["src"])},
                   segments=json.load(open(os.path.join(d, "segments.json"))),
                   plan=json.load(open(os.path.join(d, "blur_plan.json"))),
                   edits=json.load(open(ep)) if os.path.exists(ep) else {"segs": {}, "added": []})


PLAYABLE = (".mp4", ".m4v", ".webm")
proxy_state = {}


def _audio_args(src, idx):
    """원본의 소리를 검수용 미리보기에 넣는 ffmpeg 인자 (소리가 없으면 빈 목록).
    방송용 MXF는 왼쪽·오른쪽이 따로 된 모노 트랙이 많아서, 첫 두 트랙이 모노면 합쳐서 스테레오로 만듦"""
    try:
        err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", src], capture_output=True, text=True,
                             encoding="utf-8", errors="ignore").stderr
    except Exception:
        return []
    auds = [l for l in err.splitlines() if "Stream #" in l and "Audio:" in l]
    if not auds:
        return []
    mono = lambda l: "mono" in l or "1 channels" in l
    tail = ["-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000"]
    if len(auds) >= 2 and mono(auds[0]) and mono(auds[1]):
        return ["-filter_complex", f"[{idx}:a:0][{idx}:a:1]amerge=inputs=2[aud]", "-map", "[aud]"] + tail
    return ["-map", f"{idx}:a:0"] + tail


def _proxy_job(jid, src, dst, n):
    """브라우저가 바로 못 트는 영상(MXF 등)은 검수용 가벼운 미리보기 영상(소리 포함)을 만듦 (프레임 수·속도는 그대로)"""
    tmp = dst + ".part.mp4"
    try:
        p = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-progress", "pipe:1", "-i", src, "-map", "0:v:0",
                              "-vf", "scale=-2:540", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "15",
                              "-pix_fmt", "yuv420p"] + _audio_args(src, 0) + [tmp],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in p.stdout:
            if line.startswith("frame="):
                try:
                    proxy_state[jid] = min(99, int(line.split("=")[1]) * 100 // max(n, 1))
                except Exception:
                    pass
        if p.wait() == 0:
            os.replace(tmp, dst)
            proxy_state[jid] = 100
        else:
            proxy_state[jid] = -1
    except Exception:
        traceback.print_exc()
        proxy_state[jid] = -1


audio_state = {}


def _audio_job(jid, src, px, dst):
    """예전에 소리 없이 만든 미리보기에 원본 소리만 붙임 (영상은 다시 안 만들고 그대로 복사해서 빠름)"""
    tmp = dst + ".part.mp4"
    try:
        aa = _audio_args(src, 1)
        if not aa:
            audio_state[jid] = -1                 # 원본에 소리가 없음
            return
        r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", px, "-i", src, "-map", "0:v:0", "-c:v", "copy"]
                           + aa + [tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            os.replace(tmp, dst)
            audio_state[jid] = 100
        else:
            audio_state[jid] = -1
    except Exception:
        traceback.print_exc()
        audio_state[jid] = -1


@app.get("/review/<jid>/video-status")
def review_video_status(jid):
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
    if os.path.splitext(r["src"])[1].lower() in PLAYABLE:
        return jsonify(ready=True, audio="ok")
    pav, px = os.path.join(d, "proxy_av.mp4"), os.path.join(d, "proxy.mp4")
    if os.path.exists(pav):
        return jsonify(ready=True, audio="ok")
    if os.path.exists(px):                      # 소리 없는 예전 미리보기 → 소리만 붙이는 중
        if jid not in audio_state:
            audio_state[jid] = 0
            threading.Thread(target=_audio_job, args=(jid, r["src"], px, pav), daemon=True).start()
        return jsonify(ready=True, audio="none" if audio_state.get(jid) == -1 else "pending")
    if jid not in proxy_state:
        proxy_state[jid] = 0
        threading.Thread(target=_proxy_job, args=(jid, r["src"], pav, r["n"]), daemon=True).start()
    return jsonify(ready=False, pct=proxy_state.get(jid, 0))


@app.get("/review/<jid>/video")
def review_video(jid):
    d = _jdir(jid)
    if not d:
        return "없음", 404
    r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
    if os.path.splitext(r["src"])[1].lower() not in PLAYABLE:
        for nm in ("proxy_av.mp4", "proxy.mp4"):
            px = os.path.join(d, nm)
            if os.path.exists(px):
                return send_file(px, conditional=True, mimetype="video/mp4")
    return send_file(r["src"], conditional=True)


_rv_cache = {}


@app.post("/review/<jid>/frame/<int:f>")
def review_frame(jid, f):
    """멈춘 화면용: 수정본과 똑같은 방식으로 이 프레임에 블러를 그려서 돌려줌 (화면 = 결과물)"""
    d = _jdir(jid)
    if not d:
        return "없음", 404
    import cover2
    c = _rv_cache.get(jid)
    if c is None:
        r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
        c = {"r": r, "polys": json.load(open(os.path.join(d, "polys.json"))), "cap": None, "pos": -1, "lock": threading.Lock()}
        _rv_cache.clear()
        _rv_cache[jid] = c
    r = c["r"]
    ells = (request.get_json(silent=True) or {}).get("ells", [])
    with c["lock"]:
        for k, v in r.get("params", {}).items():
            if k in ("BLUR_FRAC", "FEATHER_W", "FEATHER_MAX", "FILL_KEEP"):
                setattr(cover2, k, v)
        W, H = r["W"], r["H"]
        fsz = W * H * 3
        if c["cap"] is None or f != c["pos"] + 1:   # 바로 다음 프레임이 아니면 그 자리부터 다시 풂 (그리기 단계와 같은 방식)
            if c["cap"] is not None:
                c["cap"].close()
            if "t0" not in c:
                c["t0"] = media.probe(r["src"]).get("t0", 0.0)
                c["table"] = _load_table(d)
            c["cap"] = Dec(r["src"], f, max(1, r["n"] - f), W, H, r["fps"], t0=c["t0"], table=c["table"])
        got = c["cap"].read()
        ok = got is not None and got[0] == f
        buf = got[1] if ok else b""
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy() if ok else None
        c["pos"] = f if ok else -1
        if not ok:
            c["cap"] = None
    if not ok:
        return "프레임을 못 읽음", 500
    P = cover2.Painter()
    out, _ = P.paint(frame, [tuple(e) for e in ells], c["polys"][f] if f < len(c["polys"]) else [])
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/review/<jid>/edits")
def review_save(jid):
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    json.dump(request.get_json(force=True), open(os.path.join(d, "edits.json"), "w"))
    return jsonify(ok=True)


def run_fix(fid, d):
    job = state[fid]
    with run_lock:
        try:
            import cover2
            r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
            for k, v in r.get("params", {}).items():
                setattr(cover2, k, v)
            n, fps = r["n"], r["fps"]
            ells = json.load(open(os.path.join(d, "blur_plan.json")))
            segtab = {int(k): v for k, v in json.load(open(os.path.join(d, "segments.json"))).items()}
            ep = os.path.join(d, "edits.json")
            edits = json.load(open(ep)) if os.path.exists(ep) else {}
            ells = cover2.apply_edits(ells, segtab, edits, n)
            polys = json.load(open(os.path.join(d, "polys.json")))
            out = os.path.join(d, os.path.splitext(os.path.basename(r["src"]))[0] + "_fix" + os.path.splitext(r["src"])[1])
            t0 = time.time()
            times = {}
            res, KP = paint_save(r["src"], ells, polys, out, job, n, r["W"], r["H"], fps, r.get("params", {}), 0, 99, times=times)
            frames = sum(x[1] for x in res)
            job.update(pct=100, msg="완료", done=True, out=times.get("out_path", out),
                       report={"frames": frames, "fps": fps, "sec": round(time.time() - t0, 1), "fix": True,
                               "out_fmt": times.get("out_fmt", ""), "out_name": os.path.basename(times.get("out_path", out)), "times": times})
        except Exception as e:
            traceback.print_exc()
            job.update(error=str(e)[:300], done=True, msg="오류")


@app.post("/review/<jid>/render")
def review_render(jid):
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    data = request.get_json(silent=True)
    if data is not None:
        json.dump(data, open(os.path.join(d, "edits.json"), "w"))
    fid = os.path.basename(jid) + "_fix_" + uuid.uuid4().hex[:4]
    state[fid] = {"pct": 0, "msg": "수정본 만들 준비 중", "done": False}
    threading.Thread(target=run_fix, args=(fid, d), daemon=True).start()
    return jsonify(job=fid)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    try:   # 예전에 켜 둔 생성기가 남아 있으면 끄기
        urllib.request.urlopen(f"http://127.0.0.1:{port}/quit", timeout=2).read()
        time.sleep(1.5)
    except Exception:
        pass
    print(f"\n  모자이크 생성기:  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
