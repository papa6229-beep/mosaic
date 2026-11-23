"""
모자이크 덧대기 생성기 (로컬 웹 화면)
 - 영상 업로드(또는 경로 입력) -> Lada v4 fast로 원본 모자이크 위치 탐지
 - 영상 전체 위치를 시간순으로 다듬기 (점멸/크기 흔들림 방지)
 - 주변 피부색으로 채운 뒤 부드러운 블러로 덮기 -> 원본과 같은 형식(압축 방식·화질·소리 트랙)으로 저장
"""
import os, sys, json, time, uuid, threading, subprocess, traceback, urllib.request, hashlib, shutil
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, Response

BASE = os.path.dirname(os.path.abspath(__file__))
# 모델·캐시는 전부 생성기 폴더(D드라이브 등) 안에 — C드라이브 사용자 폴더(.cache)에 쌓이지 않게. 받아쓰기 프로그램에도 그대로 물려줌
CACHE = os.path.join(BASE, "cache")
for _k, _v in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"), ("MODELSCOPE_CACHE", "modelscope"), ("XDG_CACHE_HOME", "")):
    os.environ[_k] = os.path.join(CACHE, _v) if _v else CACHE

import cover
import media

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


def det_size(W, H):
    """위치 찾기용 크기: 탐지 모델은 어차피 640(노출은 320)으로 줄여서 보므로, 4K 같은 큰 영상은 절반으로 줄여 풀어서 넘김 (탐지 결과는 같고 훨씬 빠름)"""
    if W <= 2200:
        return W, H
    return W // 4 * 2, H // 4 * 2


def frames(src, depth=16, table=None, size=None):
    """영상 프레임을 다른 일과 동시에 미리 읽어 둠 (읽는 시간만큼 빨라짐).
    그리기 단계와 똑같이 ffmpeg로 풀고, 프레임마다 시간을 table에 기록 (다른 단계에서 프레임 번호를 정확히 맞추는 데 씀)"""
    info = media.probe(src)
    W, H = size or (info["W"], info["H"])
    vf = f"scale={W}:{H}:flags=area" if (W, H) != (info["W"], info["H"]) else None
    q = queue.Queue(maxsize=depth)

    def reader():
        d = Dec(src, 0, 10 ** 9, W, H, info["fps"], seek=False, threads=max(2, min(8, (os.cpu_count() or 4) // 2)), mi=info, vf=vf)   # 혼자 읽으니까 CPU를 넉넉히
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
    inf_ = media.probe(src)
    Wd, Hd = det_size(inf_["W"], inf_["H"])
    k_ = inf_["W"] / Wd                                  # 줄여서 찾은 위치를 원래 크기로 되돌리는 배율
    sb = (lambda b: [v * k_ for v in b[:4]] + list(b[4:])) if k_ != 1 else (lambda b: b)
    sp = (lambda p: [[round(x * k_), round(y * k_)] for x, y in p] if p else p) if k_ != 1 else (lambda p: p)
    it = frames(src, table=table, size=(Wd, Hd))
    while True:
        f = next(it, None)
        if f is not None:
            batch.append(f)
            if proxy is not None:
                proxy.put(f)
        if batch and (len(batch) == 8 or f is None):
            for k, (b, allb) in enumerate(detect(batch)):
                boxes[str(i + k)] = sb(b) if b else b
                all_boxes[str(i + k)] = [sb(a[:5]) + [sp(a[5]) if len(a) > 5 else None] for a in allb] if k_ != 1 else allb
            if nude is not None:
                try:
                    sel = [k for k in range(len(batch)) if (i + k) % NUDE_EVERY == 0]
                    res = dict(zip(sel, nude.detect([batch[k] for k in sel]))) if sel else {}
                    if k_ != 1:
                        res = {kk: {"must": [sb(b) for b in e["must"]], "faces": [(c, sb(b)) for c, b in e["faces"]]} for kk, e in res.items()}
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
PART = 600    # 저장 조각 하나의 프레임 수 (약 20초). 수정본은 바뀐 조각만 다시 만듦
WARMUP = 30   # 조각 앞부분에서 미리 돌려 두는 프레임 수 (앞 조각과 블러가 자연스럽게 이어지게)


def _bt_fix():
    """BT.601로 바꾼 BGR → BT.709로 바꾼 BGR (같은 YUV를 두 공식으로 바꾼 결과 사이의 관계, BGR 순서)"""
    def T(kr, kb):
        kg = 1 - kr - kb
        return np.array([[kr, kg, kb], [-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), 0.5], [0.5, -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))]])
    C = np.linalg.inv(T(0.2126, 0.0722)) @ T(0.299, 0.114)
    return np.ascontiguousarray(C[::-1, ::-1]).astype(np.float32)


_BT709_FIX = _bt_fix()


class Dec:
    """영상의 start번째 프레임부터 count장을 정확히 풀어 읽음.
    시간으로 건너뛰면 영상 형식에 따라 1~2프레임 어긋날 수 있어서, 1초 앞에서부터 풀고 각 프레임의 시간을 보고 번호를 맞춤.
    table(위치 찾기 단계에서 기록한 프레임별 시간)이 있으면 그걸로 번호를 찾음 (시간이 고르지 않은 영상도 정확)"""

    def __init__(self, src, start, count, W, H, fps, pix_fmt="bgr24", fsz=None, t0=0.0, table=None, seek=True, threads=2, mi=None, vf=None):
        self.src, self.start, self.end, self.fps, self.t0 = src, start, start + count, fps, t0
        self.vf = vf
        self.pix_fmt, self.fsz = pix_fmt, fsz or W * H * 3
        self.W, self.H, self.conv = W, H, None
        # BGR이 필요할 때: ffmpeg에서 BGR로 바꿔 넘기면 한 장이 6MB라 윈도우에서 넘기는 통로가 막힘 →
        # 절반 크기(YUV 4:2:0)로 넘겨받고 여기서 바꿈 (ffmpeg가 바꾸던 것과 같은 색 공식)
        if pix_fmt == "bgr24" and mi and mi.get("range") != "pc" and W % 2 == 0 and H % 2 == 0 \
                and str(mi.get("pix_fmt", "")).startswith("yuv"):
            self.pix_fmt, self.fsz = "yuv420p", W * H * 3 // 2
            self.conv = _BT709_FIX if (mi.get("matrix_tagged") and mi.get("matrix") == "bt709") else False
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
                                   *(["-hwaccel", "cuda"] if _hw_for(self.src) else []),          # 그래픽카드로 풀기 (CPU로 푼 것과 같은 걸 확인한 영상만)
                                   *(["-ss", f"{ss:.6f}"] if ss > 0 else []), "-i", self.src, "-map", "0:v:0",
                                   "-vf", (self.vf + "," if self.vf else "") + "showinfo=checksum=0", "-fps_mode", "passthrough",   # 프레임을 복제·생략하지 않고 풀린 그대로
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
        """→ (프레임 번호, 바이트(또는 BGR 배열), 프레임 시간) 또는 None(끝)"""
        got = self._read()
        if got is None or self.conv is None:
            return got
        idx, buf, t = got
        b = cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(self.H * 3 // 2, self.W), cv2.COLOR_YUV2BGR_I420)
        if self.conv is not False:
            b = cv2.transform(b, self.conv)
        return idx, b, t

    def _read(self):
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


def _hw_for(src):
    return os.environ.get("MOSAIC_HWDEC_ON") == "1" and os.environ.get("MOSAIC_HWDEC_SRC") == src


def _set_hw(src, job=None):
    """이 영상을 그래픽카드로 풀어도 되는지 한 번 확인해서 일꾼들에게 알려 둠 (환경 변수로 물려줌)"""
    if job is not None:
        job["msg"] = "그래픽카드로 영상을 풀 수 있는지 확인하는 중"
    try:
        ok = media.hw_decode_ok(src, media.probe(src))
    except Exception:
        ok = False
    os.environ["MOSAIC_HWDEC_SRC"] = src
    os.environ["MOSAIC_HWDEC_ON"] = "1" if ok else "0"
    return ok


def _low_priority():
    """생성 작업은 낮은 우선순위로 (작업 중에도 컴퓨터를 쓸 수 있게). ffmpeg 같은 하위 작업도 그대로 물려받음.
    또 일꾼 하나는 CPU 하나만 쓰게 함: 그림 계산 도구(cv2)는 원래 혼자서 CPU를 전부 쓰려고 해서,
    8개가 동시에 그러면 서로 자리를 뺏느라 8배가 아니라 1.7배밖에 안 빨라짐 (속도 점검으로 확인)"""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    if os.environ.get("MOSAIC_FULL") == "1":          # 밤샘 모드: 보통 우선순위로 CPU를 다 씀
        return
    try:
        import psutil
        pr = psutil.Process()
        if os.name == "nt":
            pr.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            pr.nice(10)
    except Exception:
        pass


def _mem_free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 99


def _is_mem_err(e):
    t = str(e)
    return "MemoryError" in t or "Unable to allocate" in t or "out of memory" in t.lower()


def _mem_workers(K, W, H):
    """남은 메모리(RAM)에 맞춰 동시에 할 수를 줄임 (4K는 일꾼 하나가 프레임 몇 장씩 들고 계산해서 메모리를 많이 씀)"""
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        return K
    per = W * H * 3 * 40 + 300e6                         # 일꾼 하나에 필요한 대략의 메모리 (4K ≈ 1.3GB, FHD ≈ 0.55GB)
    return max(1, min(K, int(avail * 0.7 / per)))


def _cpu_workers(n):
    """동시에 쓸 CPU 수: 작업 중에도 컴퓨터를 쓸 수 있게 둘은 항상 비워 둠 (최대 8개)"""
    cores = os.cpu_count() or 2
    if os.environ.get("MOSAIC_FULL") == "1":          # 밤샘 모드: 비워 두는 CPU 없이 (최대 16개)
        return max(1, min(cores, 16, n // 60))
    return max(1, min(cores - 2, 8, n // 60))


def _scan_worker(src, start, count, W, H, fps, mi, table, entries, exps, params, outp, counter, q):
    """1단계(훑기): 이 조각의 프레임마다 '반드시 덮을 곳'을 모아 파일로 저장"""
    _low_priority()
    try:
        import cover2, pickle
        for k, v in params.items():
            setattr(cover2, k, v)
        w0 = max(0, start - WARMUP)
        total = start + count - w0
        dec = Dec(src, w0, total, W, H, fps, t0=mi.get("t0", 0.0), table=table, mi=mi)
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


def _paint_worker(src, runs, count, W, H, fps, params, mi, subs, counter, q):
    """3단계(그리기): 원본을 원본 픽셀 형식(YUV) 그대로 풀고, 계획된 블러만 섞어서, 원본과 같은 압축 방식으로 조각 저장.
    runs: [(시작, 프레임 수, 블러, 윤곽, [(조각번호, 조각시작, 조각길이, 파일)])] — 이어진 구간은 한 번에 풀고 조각마다 따로 압축
    (조각은 남겨 둬서, 검수 뒤 수정본을 만들 때 안 바뀐 조각은 다시 만들지 않음). 블러가 없는 곳의 픽셀은 원본 그대로"""
    _low_priority()
    try:
        import cover2
        for k, v in params.items():
            setattr(cover2, k, v)
        fmt = mi["compose"]
        fsz = media.frame_bytes(fmt, W, H)
        ov = None
        if subs and subs.get("cues"):                   # 자막 입히기
            import subrender
            ov = subrender.Overlay(subs["cues"], W, H, fps, subs.get("style"))
        made = {}
        for start, cnt, ells, polys, parts in runs:
            w0 = max(0, start - WARMUP)
            dec = Dec(src, w0, start + cnt - w0, W, H, fps, fmt, fsz, t0=mi["t0"], table=mi.get("table"))
            P = cover2.Painter()
            st = {"pi": 0, "out": None, "expect": start}

            def finish():
                o = st["out"]
                o.stdin.close()
                err = o.stderr.read().decode("utf-8", "ignore")
                if o.wait() != 0:
                    raise RuntimeError("조각 저장 실패: " + err[-300:])
                idx, ps, pc, path = parts[st["pi"]]
                os.replace(path + ".tmp", path)
                made[idx] = pc
                st["out"] = None
                st["pi"] += 1

            def put(b_):
                if st["out"] is None:
                    path = parts[st["pi"]][3]
                    st["out"] = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", fmt,
                                                  "-s", f"{W}x{H}", "-r", mi["fps_str"], *mi["inp"], "-i", "-", *mi["enc"], "-threads", "2",
                                                  "-f", "nut", path + ".tmp"], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    st["out"].stdin.write(b_)
                except (BrokenPipeError, OSError):
                    raise RuntimeError("조각 저장 실패: " + st["out"].stderr.read().decode("utf-8", "ignore")[-400:])
                st["expect"] += 1
                with counter.get_lock():
                    counter.value += 1
                _, ps, pc, _ = parts[st["pi"]]
                if st["expect"] >= ps + pc:
                    finish()
            data = None                                  # 조각마다 정확히 정해진 장수를 순서대로 씀 (혹시 빠지면 앞 프레임으로 채움)
            while st["expect"] < start + cnt:
                got = dec.read()
                if got is None:
                    break
                i, buf, _ = got
                j = i - w0
                if i >= start and i < st["expect"]:           # 같은 프레임이 또 나오면 건너뜀
                    continue
                if i >= start + cnt:
                    break
                yf = media.YuvFrame(buf, fmt, W, H, mi["matrix"], mi["range"])
                changed = P.paint_yuv(yf, ells[j] if j < len(ells) else [], polys[j] if j < len(polys) else [])
                if ov is not None and i >= start and ov.apply(yf, i):
                    changed = True
                if i >= start:
                    cur_ = yf.tobytes() if changed else buf
                    while st["expect"] < i:
                        put(data if data is not None else cur_)
                    put(cur_)
                    data = cur_
            dec.close()
            if data is None:
                raise RuntimeError(f"{start}번 프레임부터 영상을 읽지 못했어요")
            while st["expect"] < start + cnt:
                put(data)
        q.put((0, sum(made.values()), made, None))
    except Exception:
        q.put((0, 0, {}, traceback.format_exc()[-800:]))


def _run_parallel(job, K, n, make_args, target, label, p0=50, p1=99):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    for k_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k_] = "1"                                # 일꾼(새 프로그램)들이 계산 도구를 CPU 하나로만 쓰게
    counters = [ctx.Value("i", 0) for _ in range(K)]      # 일꾼마다 따로 셈 (남은 시간을 일꾼별 속도로 계산하려고)
    q = ctx.Queue()
    procs, sizes = [], []
    for k in range(K):
        args = make_args(k)
        sizes.append(args[2])                               # 이 일꾼이 맡은 프레임 수 (src, start, count, ...)
        p = ctx.Process(target=target, args=args + (counters[k], q), daemon=True)
        p.start()
        procs.append(p)
    t0, results = time.time(), []
    first = [None] * K                                     # 일꾼마다 실제로 프레임을 내기 시작한 때 (준비 시간은 빼고 속도를 잼)
    while len(results) < K:
        try:
            results.append(q.get(timeout=1))
        except Exception:
            pass
        dn = [c.value for c in counters]
        i = sum(dn)
        now = time.time()
        el = now - t0
        for k, d in enumerate(dn):
            if d and first[k] is None:
                first[k] = (now, d)
        # 가장 늦게 끝날 일꾼 기준 (블러가 많은 구간을 맡은 일꾼은 느려서, 전체 평균으로 재면 실제보다 짧게 나옴)
        lefts = []
        for k, (sz, d) in enumerate(zip(sizes, dn)):
            if d >= sz:
                continue
            if first[k] is None or d - first[k][1] < 10 or now - first[k][0] < 5:
                lefts = None                                # 아직 속도를 잴 만큼 안 됨
                break
            lefts.append((sz - d) * (now - first[k][0]) / (d - first[k][1]))
        left = max(lefts) if lefts else (0 if i >= n or lefts == [] else None)
        job["msg"] = f"{label} {i}/{n} 프레임 · CPU {K}개 동시 처리 · " + \
            ("이 단계 남은 시간 계산 중" if left is None else f"이 단계 남은 시간 약 {fmt_sec(left)}")
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
    K = _mem_workers(_cpu_workers(n), W, H)
    params = {k: getattr(cover2, k) for k in ("SCALE", "BLUR_FRAC", "FEATHER_W", "HAIR", "HAIR_REACH")}
    d = os.path.dirname(dst)

    # 1단계: 훑기 (덮을 곳 모으기)
    t0 = time.time()
    t0v = media.probe(src)
    table = _load_table(os.path.dirname(boxes_json))
    while True:
        bounds = [round(n * k / K) for k in range(K + 1)]
        wb = lambda k: max(0, bounds[k] - WARMUP)
        scans = [os.path.join(d, f"scan{k:02d}.pkl") for k in range(K)]
        try:
            _run_parallel(job, K, n, lambda k: (src, bounds[k], bounds[k + 1] - bounds[k], W, H, fps, t0v, table, per[wb(k):bounds[k + 1]], (ex[wb(k):bounds[k + 1]] if ex else None), params, scans[k]),
                          _scan_worker, "1/3 덮을 곳 훑는 중", 50, 75)
            break
        except RuntimeError as e:
            if not _is_mem_err(e) or K <= 1:
                raise
            K = max(1, K // 2)                             # 메모리가 모자라면 동시에 하는 수를 반으로 줄여 이 단계만 다시
            job["msg"] = f"메모리가 모자라서 CPU {K}개로 줄여 다시 훑는 중"
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


def paint_save(src, ells, polys, dst, job, n, W, H, fps, params, p0=77, p1=99, times=None, subs=None):
    """계획된 블러대로 그려서 원본과 같은 형식으로 저장 (CPU 여러 개로 나눠서) → 원본 소리 트랙 그대로 붙여 하나로.
    실제로 저장한 파일 경로를 times["out_path"]에 남김"""
    times = times if times is not None else {}
    t0 = time.time()
    info = media.probe(src)
    fmt = media.compose_fmt(info)
    enc, desc = media.encode_args(info, fmt)
    genc = media.gpu_encode_args(info, fmt)             # 원본이 H.264/H.265면 그래픽카드로 압축 (같은 비트레이트)
    if genc:
        enc, desc = genc, desc + " · 그래픽카드 압축"
    elif media.GPU_ENC_WHY:
        desc += " · CPU 압축 (" + media.GPU_ENC_WHY + ")"
    mi = {"compose": fmt, "enc": enc, "inp": media.input_args(info), "fps_str": info["fps_str"], "matrix": info["matrix"],
          "range": info["range"], "t0": info.get("t0", 0.0), "table": _load_table(os.path.dirname(dst))}
    times["out_fmt"] = desc
    d = os.path.dirname(dst)
    pd = os.path.join(d, "parts")
    os.makedirs(pd, exist_ok=True)
    bounds = [(s_, min(PART, n - s_)) for s_ in range(0, n, PART)]
    paths = [os.path.join(pd, f"part{k:05d}.nut") for k in range(len(bounds))]
    pk = {k: params.get(k) for k in ("SCALE", "BLUR_FRAC", "FEATHER_W", "HAIR", "HAIR_REACH")}   # 그리기에 쓰는 설정 (처음 만들 때와 수정본에서 같은 것만)
    sig = json.dumps([desc, enc, pk, W, H, n, PART, WARMUP], default=str)

    def _j(o):
        return o.tolist() if hasattr(o, "tolist") else str(o)

    def phash(k):                                       # 이 조각에 들어가는 블러(앞 준비 구간 포함)가 같으면 같은 값
        s_, c_ = bounds[k]
        w0 = max(0, s_ - WARMUP)
        cs = [c for c in (subs or {}).get("cues", []) if c["end"] * fps >= s_ - 1 and c["start"] * fps <= s_ + c_ + 1]   # 이 조각에 걸친 자막
        return hashlib.md5(json.dumps([sig, ells[w0:s_ + c_], polys[w0:s_ + c_], cs, (subs or {}).get("style") if cs else None],
                                      default=_j).encode()).hexdigest()
    man_p = os.path.join(pd, "parts.json")
    try:
        old = json.load(open(man_p))
    except Exception:
        old = {}
    hs = [phash(k) for k in range(len(bounds))]
    dirty = [k for k in range(len(bounds)) if old.get(str(k)) != hs[k] or not os.path.exists(paths[k])]
    try:
        os.remove(man_p)                                 # 만드는 도중에 멈추면 다음에 다 다시 확인하게
    except Exception:
        pass
    for k in dirty:                                    # 다시 만들 조각의 예전 파일은 지움 (중간에 멈췄다 다시 할 때 헷갈리지 않게)
        try:
            os.remove(paths[k])
        except FileNotFoundError:
            pass
    times["parts"] = len(bounds)
    times["reused"] = len(bounds) - len(dirty)
    total = sum(bounds[k][1] for k in dirty)
    K = max(1, min(_mem_workers(_cpu_workers(total), W, H), len(dirty), 6 if genc else 99))   # 그래픽카드 압축은 동시에 쓸 수 있는 개수가 정해져 있어서 6개까지
    KP = K
    groups, acc, w = [[] for _ in range(K)], 0, 0      # 다시 만들 조각을 일꾼마다 비슷한 양으로 (순서대로 이어지게)
    for k in dirty:
        if acc >= total * (w + 1) / K and w < K - 1:
            w += 1
        groups[w].append(k)
        acc += bounds[k][1]

    def runs_of(g):
        rs = []
        for k in g:
            if rs and rs[-1][-1] == k - 1:
                rs[-1].append(k)
            else:
                rs.append([k])
        out = []
        for r_ in rs:
            s_ = bounds[r_[0]][0]
            c_ = sum(bounds[k][1] for k in r_)
            w0 = max(0, s_ - WARMUP)
            out.append((s_, c_, ells[w0:s_ + c_], polys[w0:s_ + c_], [(k, *bounds[k], paths[k]) for k in r_]))
        return out
    if dirty:
        label = "블러 그리고 저장 중" + (f" (고친 곳만 다시: {len(dirty)}/{len(bounds)}조각)" if times["reused"] else f" (원본과 같은 형식: {desc})")
        while dirty:
            K = max(1, min(K, len(dirty)))
            groups, acc, w = [[] for _ in range(K)], 0, 0
            for k in dirty:
                if acc >= total * (w + 1) / K and w < K - 1:
                    w += 1
                groups[w].append(k)
                acc += bounds[k][1]
            try:
                _run_parallel(job, K, total, lambda k: (src, runs_of(groups[k]), sum(bounds[i][1] for i in groups[k]), W, H, fps, params, mi, subs),
                              _paint_worker, label, p0, p1)
                break
            except RuntimeError as e:
                if not _is_mem_err(e) or K <= 1:
                    raise
                K = max(1, K // 2)                         # 메모리가 모자라면 반으로 줄여 다시 (다 만든 조각은 그대로 씀)
                done_ = [k for k in dirty if os.path.exists(paths[k])]
                dirty = [k for k in dirty if k not in done_]
                total = sum(bounds[k][1] for k in dirty)
                KP = K
    json.dump({str(k): h for k, h in enumerate(hs)}, open(man_p, "w"))
    times["paint"] = round(time.time() - t0, 1)
    t0 = time.time()
    job["msg"] = "조각 이어 붙이고 원본 소리 트랙 붙이는 중"
    lst = os.path.join(d, "parts.txt")
    fr = Fraction(info["fps_str"])
    with open(lst, "w", encoding="utf-8") as fp:      # 조각마다 길이를 적어 줌 → 이어 붙인 곳에서 시간이 1프레임씩 밀리지 않음
        for k, s_ in enumerate(paths):
            fp.write("file '" + s_.replace("\\", "/").replace("'", "'\\''") + "'\n")
            fp.write(f"duration {float(bounds[k][1] / fr):.9f}\n")
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
                            "-map", "0:v", "-map", "1:a?", "-c", "copy",
                            *(["-r", info["fps_str"]] if out.lower().endswith(".mxf") else []),   # 이어 붙인 영상의 초당 프레임을 MXF에 정확히 적음 (안 적으면 2배로 적혀 길이가 반이 됨)
                            *media.mux_args(info, out), out], capture_output=True)
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
    try:
        os.remove(lst)
    except Exception:
        pass
    return [(0, n)], KP


def run_job(jid, src, params):
    job = state[jid]
    if run_lock.locked():
        job["msg"] = "앞 작업이 끝나길 기다리는 중"
    with run_lock:
        job["msg"] = "시작하는 중"
        os.environ["MOSAIC_FULL"] = "1" if night["on"] else "0"
        try:
            import cover2
            cover2.SIZE_K = float(params.get("scale", cover2.SIZE_K))
            cover2.BLUR_FRAC = float(params.get("blur", cover2.BLUR_FRAC))
            cover2.FEATHER_W = float(params.get("feather", cover2.FEATHER_W))
            cover2.HAIR = params.get("hair", "1") == "1"
            cover2.HAIR_REACH = float(params.get("reach", cover2.HAIR_REACH))
            ensure_model(job)
            job["hwdec"] = _set_hw(src, job)
            n = int(cv2.VideoCapture(src).get(cv2.CAP_PROP_FRAME_COUNT))
            t0 = time.time()
            px = None
            if os.path.splitext(src)[1].lower() not in PLAYABLE:   # 검수용 미리보기를 위치 찾기와 같이 만듦
                try:
                    mi0 = media.probe(src)
                    c_ = cv2.VideoCapture(src)
                    px = ProxyWriter(os.path.join(JOBS, jid, "proxy.mp4"), *det_size(int(c_.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                     int(c_.get(cv2.CAP_PROP_FRAME_HEIGHT))), mi0["fps_str"])
                    c_.release()
                except Exception:
                    traceback.print_exc()
                    px = None
            table = []
            boxes, all_boxes, exps, n2 = detect_all(src, job, n, px, table)
            subs_later = False
            if params.get("subs") == "1":                   # 자막은 그래픽카드가 비는 지금부터 뒤에서 같이 만듦
                if _mem_free_gb() >= 12:
                    start_subs(jid, src, params.get("profile"))
                else:                                       # 메모리가 빠듯하면 블러가 끝난 뒤에 시작
                    subs_later = True
                    sub_state[jid] = {"sub_msg": "블러가 끝나면 시작해요 (메모리가 빠듯해서 차례로)", "sub_pct": 0, "sub_done": False, "kind": "make"}
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
            rep["det_how"] = job.get("det_how", "") + (" · 영상 풀기 그래픽카드" if job.get("hwdec") else "")
            rep["nude_how"] = job.get("nude_how", "")
            try:
                json.dump(rep, open(os.path.join(JOBS, jid, "report.json"), "w"), ensure_ascii=False)
            except Exception:
                pass
            if subs_later:
                sub_state.pop(jid, None)
                start_subs(jid, src, params.get("profile"))
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
            es = str(e).strip()
            if "Traceback" in es:                   # 일꾼이 보낸 긴 기록이면 실제 오류가 적힌 끝부분을 보여 줌
                ls = [l for l in es.splitlines() if l.strip()]
                es = " / ".join(ls[-3:])
            job.update(error=es[-400:] + info, done=True, msg="오류")


# ---------------- 화면 ----------------
@app.get("/sysinfo")
def sysinfo_api():
    return jsonify(sysinfo())


@app.get("/quit")
def quit_server():
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return "bye"


@app.get("/fonts/<name>")
def fonts(name):
    """자막 글꼴 (본고딕 Noto Sans KR Medium — 방송본부 자막 샘플과 같은 모양, 무료 글꼴 OFL)"""
    f = os.path.join(BASE, "fonts", os.path.basename(name))
    if not os.path.exists(f):
        return "없음", 404
    return send_file(f, mimetype="font/otf", max_age=86400)


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
    if request.form.get("mode") == "subs":              # 자막만: 블러 없이 자막 작업만 (검수 화면 대신 자막 화면으로)
        try:
            info = media.probe(src)
            json.dump({"src": src, "W": info["W"], "H": info["H"], "fps": info["fps"],
                       "n": int(round(info.get("duration", 0) * info["fps"])) or int(cv2.VideoCapture(src).get(cv2.CAP_PROP_FRAME_COUNT)),
                       "out": "", "params": {}, "mode": "subs"}, open(os.path.join(d, "review.json"), "w", encoding="utf-8"), ensure_ascii=False)
        except Exception as e:
            return jsonify(error="영상을 읽지 못했어요: " + str(e)[:200]), 400
        state[jid] = {"pct": 100, "msg": "자막만 만들기", "done": True, "name": os.path.basename(src), "added": time.time(),
                      "main": True, "mode": "subs"}
        start_subs(jid, src, request.form.get("profile"))
        return jsonify(job=jid)
    state[jid] = {"pct": 0, "msg": "앞 작업이 끝나길 기다리는 중", "done": False, "name": os.path.basename(src),
                  "added": time.time(), "main": True, "subs_on": request.form.get("subs") == "1"}
    threading.Thread(target=run_job, args=(jid, src, request.form), daemon=True).start()
    return jsonify(job=jid)


night = {"on": False}     # 밤샘 모드: 켜 두면 다음 단계부터 CPU를 전부, 보통 우선순위로 씀 (컴퓨터를 안 쓸 때)


@app.route("/night", methods=["GET", "POST"])
def night_mode():
    if request.method == "POST":
        night["on"] = bool((request.get_json(silent=True) or {}).get("on"))
    return jsonify(on=night["on"])


@app.get("/queue")
def queue_list():
    """이번에 켠 뒤로 넣은 작업들 (줄 선 순서대로)"""
    out = []
    for jid, j in state.items():
        if j.get("main"):
            out.append({"jid": jid, "name": j.get("name"), "pct": j.get("pct", 0), "msg": j.get("msg", ""), "done": j.get("done"),
                        "error": j.get("error"), "added": j.get("added"), "report": j.get("report"), "subs": sub_state.get(jid),
                        "mode": j.get("mode", "blur")})
    out.sort(key=lambda x: x["added"])
    return jsonify(out)


@app.get("/status/<jid>")
def status(jid):
    j = state.get(jid)
    if not j:
        return jsonify(error="없는 작업"), 404
    return jsonify({k: v for k, v in j.items() if k != "out"} | {"subs": sub_state.get(jid)})


# ---------------- 자막 ----------------
sub_state = {}
sub_lock = threading.Lock()


def _free_gpu():
    global _model
    _model = None
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# 예전에 C드라이브 사용자 폴더(.cache)에 받아 둔 것 중 생성기가 쓰는 모델 → 생성기 폴더로 옮김 / 이제 안 쓰는 시험용 모델은 목록만
USED_HF = ("models--Qwen--Qwen3-ASR-1.7B", "models--jaykwok--Qwen3-ASR-1.7B-JA-Anime-Galgame", "models--Qwen--Qwen3-ForcedAligner-0.6B",
           "models--litagin--anime-whisper")
OLD_HF_HINTS = ("faster-whisper", "kotoba-whisper", "anime-whisper-ct2", "translategemma", "anime-whisper-faster")
cache_state = {"moving": "", "old": [], "moved": [], "checked": False}


def _home_hub():
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _dir_gb(d):
    t = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except Exception:
                pass
    return round(t / 1e9, 1)


def _move_caches():
    src = _home_hub()
    dst = os.path.join(CACHE, "huggingface", "hub")
    if not os.path.isdir(src) or os.path.abspath(src) == os.path.abspath(dst):
        return
    with sub_lock:                               # 옮기는 동안 자막 작업은 기다림
        for name in USED_HF:
            a, b = os.path.join(src, name), os.path.join(dst, name)
            if os.path.isdir(a) and not os.path.exists(b):
                try:
                    cache_state["moving"] = name.split("--")[-1]
                    gb = _dir_gb(a)
                    os.makedirs(dst, exist_ok=True)
                    shutil.move(a, b)
                    cache_state["moved"].append({"name": name.split("--")[-1], "gb": gb})
                    print(f"  {name} 를 C드라이브에서 생성기 폴더로 옮겼어요")
                except Exception as e:
                    print("  모델 옮기기 실패:", e)
        cache_state["moving"] = ""
        cache_state["checked"] = True
    try:
        cache_state["old"] = [{"name": n, "gb": _dir_gb(os.path.join(src, n))} for n in sorted(os.listdir(src))
                              if any(h in n.lower() for h in OLD_HF_HINTS)]
    except Exception:
        cache_state["old"] = []


@app.route("/cache-old", methods=["GET", "POST"])
def cache_old():
    """C드라이브에 남은 예전 시험용 모델(지금은 안 씀): 목록 보기 / 누르면 지우기"""
    if request.method == "POST":
        src = _home_hub()
        for o in list(cache_state["old"]):
            p = os.path.join(src, o["name"])
            if os.path.isdir(p) and o["name"] not in USED_HF:
                shutil.rmtree(p, ignore_errors=True)
        cache_state["old"] = [o for o in cache_state["old"] if os.path.isdir(os.path.join(src, o["name"]))]
    return jsonify(cache_state)


def _kill_leftovers():
    """생성기 창을 닫아도 남아서 메모리를 잡고 있는 자막 프로그램(받아쓰기·번역 모델)을 끔"""
    try:
        import psutil
    except Exception:
        return 0
    n = 0
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            name = (p.info.get("name") or "").lower()
            if ("whisperjav.main" in cmd or "whisperjav" in cmd.lower() and "pass_worker" in cmd) or \
                    (name.startswith("llama-server") and "8771" in cmd):
                for c in p.children(recursive=True):
                    c.kill()
                p.kill()
                n += 1
        except Exception:
            pass
    return n


# ---------- 자막 화면 소리 그림: 말소리 대역(300~4000Hz) 크기를 0.05초마다 + 자막 없는데 소리 있는 곳 ----------
@app.get("/review/<jid>/wave")
def review_wave(jid):
    import numpy as np, wave as wv
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    cp = os.path.join(d, "wave.json")
    if os.path.exists(cp):
        return Response(open(cp, encoding="utf-8").read(), mimetype="application/json")
    wav = os.path.join(d, "audio16k.wav")
    if not os.path.exists(wav):
        import subs
        r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
        subs.extract_audio(r["src"], wav)
    with wv.open(wav) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    hop = sr // 20                                             # 0.05초
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop) * np.hanning(hop)
    sp = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(hop, 1 / sr)
    e = 10 * np.log10(sp[:, (f >= 300) & (f <= 4000)].sum(1) + 1e-10)
    floor = float(np.percentile(e, 20))
    v = np.clip((e - floor) / 40, 0, 1)                        # 바닥 소음 0, +40dB 1
    out = {"hop": 0.05, "v": [int(round(t * 100)) for t in v], "th": 0.3}   # th: 이 이상이면 '소리 있음' (바닥보다 12dB)
    json.dump(out, open(cp, "w"))
    return jsonify(out)


# ---------- 한 줄만 번역 (자막 화면에서 '다른 받아쓰기로 바꾸기') — 번역 모델을 띄워 두고 5분 쉬면 내림 ----------
line_tr = {"tr": None, "last": 0.0}


def _close_line_tr():
    t = line_tr.get("tr")
    line_tr["tr"] = None
    if t is not None:
        try:
            t.close()
        except Exception:
            pass


def _line_tr_idle():
    while True:
        time.sleep(30)
        if line_tr["tr"] is not None and time.time() - line_tr["last"] > 300 and sub_lock.acquire(blocking=False):
            try:
                _close_line_tr()
            finally:
                sub_lock.release()


threading.Thread(target=_line_tr_idle, daemon=True).start()


@app.post("/review/<jid>/subs/line-tr")
def review_subs_line_tr(jid):
    import subs, profiles
    b = request.get_json(silent=True) or {}
    src = (b.get("src") or "").strip()
    if not src:
        return jsonify(ko="")
    if not sub_lock.acquire(timeout=1):
        return jsonify(error="자막을 만드는 중이라 끝난 뒤에 바꿀 수 있어요"), 409
    try:
        if line_tr["tr"] is None:
            _free_gpu()
            line_tr["tr"] = subs.Translator(None)
        line_tr["last"] = time.time()
        prof = profiles.get(b.get("profile"))
        ctx = [(c[0], c[1]) for c in (b.get("ctx") or [])][-subs.CTX:]
        import tm
        ko = line_tr["tr"].translate([subs.clean_ja(src)], ctx, "ja", profiles.prompt_part(prof) + tm.prompt_for([src]))[0]
        parts = [x.strip() for x in subs.clean_ko(profiles.apply_replace(ko, prof)).split("||") if x.strip()]
        ko = ("- " + parts[0] + "\n- " + parts[1]) if len(parts) == 2 else " ".join(parts)
        ko = ko.rstrip(".")
        if "\n" not in ko and len(ko) > subs.MAXL:
            ko = "\n".join(subs._wrap2(ko))
        line_tr["last"] = time.time()
        return jsonify(ko=ko)
    except Exception as e:
        _close_line_tr()
        return jsonify(error="번역 실패: " + str(e)[:200]), 500
    finally:
        sub_lock.release()


def start_subs(jid, src, profile=None, retranslate=False, fresh=False):
    """자막 만들기를 뒤에서 시작 (한 번에 하나씩). 진행 상황은 sub_state[jid]"""
    st = sub_state.get(jid)
    if st and not st.get("sub_done"):
        return st
    st = sub_state[jid] = {"sub_msg": "대기 중 (다른 자막 작업이 끝나면 시작)", "sub_pct": 0, "sub_done": False,
                           "kind": "make"}

    def run():
        with sub_lock:
            try:
                import subs
                _close_line_tr()                # 한 줄 번역용으로 띄워 둔 번역 모델은 내림 (같은 자리 씀)
                if not any(not v.get("sub_done") for k, v in sub_state.items() if k != jid):
                    _kill_leftovers()           # 앞에서 멈춘 자막 작업이 남긴 프로그램이 메모리를 잡고 있으면 정리
                _free_gpu()                     # 탐지 모델이 차지한 그래픽카드 메모리를 비워서 받아쓰기 모델이 들어갈 자리 확보
                subs.make_subtitles(src, os.path.join(JOBS, jid, "subs.json"), st, profile, retranslate, fresh)
            except Exception as e:
                st.update(sub_done=True, sub_error=st.get("sub_error") or str(e)[:300])
    threading.Thread(target=run, daemon=True).start()
    return st


def _busy(jid):
    j = state.get(jid)
    sj = sub_state.get(jid)
    return (j and not j.get("done")) or (sj and not sj.get("sub_done"))


@app.post("/jobs/<jid>/delete")
def job_delete(jid):
    """최근 작업 지우기: 이 작업 폴더(결과 영상·검수 기록·자막)만 지움. 경로로 넣은 원본 영상은 건드리지 않음"""
    d = os.path.join(JOBS, os.path.basename(jid))
    if not os.path.isdir(d):
        return jsonify(error="없는 작업"), 404
    if _busy(os.path.basename(jid)):
        return jsonify(error="아직 만드는 중인 작업은 지울 수 없어요"), 400
    shutil.rmtree(d, ignore_errors=True)
    return jsonify(ok=not os.path.exists(d))


@app.post("/review/<jid>/reset")
def review_reset(jid):
    """처음 상태로 되돌리기: 검수에서 고친 블러와 자막을 지우고 갓 만든 상태로"""
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    what = (request.get_json(silent=True) or {}).get("what", "all")
    if what in ("all", "blur"):
        try:
            os.remove(os.path.join(d, "edits.json"))
        except FileNotFoundError:
            pass
    p = os.path.join(d, "subs.json")
    if what in ("all", "subs") and os.path.exists(p):
        s_ = json.load(open(p, encoding="utf-8"))
        if s_.get("lines_orig"):
            s_["lines"] = s_.pop("lines_orig")
        s_.pop("edited", None)
        json.dump(s_, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return jsonify(ok=True)


@app.get("/review/<jid>/subs")
def review_subs(jid):
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    p = os.path.join(d, "subs.json")
    data = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    if data is not None:
        import subs
        data["current"] = data.get("asr") == subs.asr_id()     # 지금 받아쓰기 방식으로 만든 자막인지
        data["tr_current"] = data.get("tr") == subs.tr_id() and data.get("post", subs.POST_VER) == subs.POST_VER   # 지금 번역·정리 방식인지
        data.pop("asr_lines", None)
        data.pop("tr_lines", None)
    return jsonify(subs=data, state=sub_state.get(jid))


@app.post("/review/<jid>/subs")
def review_subs_save(jid):
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    body = request.get_json(silent=True) or {}
    lines = body.get("lines")
    if not isinstance(lines, list):
        return jsonify(error="자막 내용이 없어요"), 400
    p = os.path.join(d, "subs.json")
    old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    if "lines_orig" not in old and old.get("lines"):       # 처음 손대기 전 자막을 남겨 둠 (처음 상태로 되돌리기용)
        old["lines_orig"] = old["lines"]
    try:                                                   # 고친 줄 모으기 (번역 기억 · 받아쓰기 훈련 자료)
        import tm
        tm.record(os.path.basename(jid), d, old.get("lines_orig"), lines)
    except Exception:
        traceback.print_exc()
    tmp = p + ".tmp"
    json.dump({**old, "lines": lines, "edited": True}, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, p)
    return jsonify(ok=True)


@app.post("/review/<jid>/subs/make")
def review_subs_make(jid):
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
    b = request.get_json(silent=True) or {}
    return jsonify(state=start_subs(jid, r["src"], b.get("profile"), bool(b.get("retranslate")), bool(b.get("fresh"))))


# ---------------- 처음 준비 (설치 후 한 번: 쓸 것만 미리 받아 두기) ----------------
SETUP_P = os.path.join(BASE, "settings", "setup.json")
setup_state = {"running": False, "msg": "", "pct": 0, "error": ""}


def _setup_done():
    try:
        return json.load(open(SETUP_P, encoding="utf-8"))
    except Exception:
        return {}


def _prefetch_asr_models(job):
    """받아쓰기 모델들을 미리 받음 (WhisperJAV가 첫 작업 때 받는 것과 같은 저장 위치)"""
    import subs
    py = subs.ensure_whisperjav(job)
    code = ("from huggingface_hub import snapshot_download as s\n"
            "for m in ('jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame','Qwen/Qwen3-ForcedAligner-0.6B','litagin/anime-whisper'):\n"
            "    print('받는 중', m, flush=True); s(m)\n")
    env = dict(os.environ, HF_HUB_DISABLE_XET="1")
    p = subprocess.Popen([py, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="ignore")
    for line in p.stdout:
        if line.startswith("받는 중"):
            subs._log(job, "받아쓰기 모델 " + line.split()[-1] + " 받는 중 (처음 한 번만, 수 GB)")
    if p.wait() != 0:
        raise RuntimeError("받아쓰기 모델을 받지 못했어요 (인터넷 연결을 확인해 주세요)")


def run_setup(what):
    st = setup_state
    st.update(running=True, error="", pct=0)
    done = _setup_done()
    try:
        if what in ("blur", "both"):
            st["msg"] = "블러 모델 받는 중"
            job = {}
            t = threading.Thread(target=ensure_model, args=(job,), daemon=True)
            t.start()
            while t.is_alive():
                st["msg"] = job.get("msg") or st["msg"]
                time.sleep(0.5)
            if not FAKE and not os.path.exists(MODEL_PATH):
                raise RuntimeError("블러 모델을 받지 못했어요")
            done["blur"] = True
            st["pct"] = 30 if what == "both" else 100
        if what in ("subs", "both"):
            import subs
            job = {}
            def watch():
                while st["running"]:
                    if job.get("sub_msg"):
                        st["msg"] = job["sub_msg"]
                    time.sleep(0.5)
            threading.Thread(target=watch, daemon=True).start()
            st["msg"] = "받아쓰기 프로그램 설치 중 (10~30분)"
            _prefetch_asr_models(job)
            st["pct"] = 70
            st["msg"] = "번역 모델 받는 중"
            subs.ensure_translator(job)
            done["subs"] = True
        st.update(pct=100, msg="준비 끝 — 바로 작업할 수 있어요")
    except Exception as e:
        traceback.print_exc()
        st["error"] = str(e)[:300]
    finally:
        done["asked"] = True
        os.makedirs(os.path.dirname(SETUP_P), exist_ok=True)
        json.dump(done, open(SETUP_P, "w", encoding="utf-8"))
        st["running"] = False


@app.route("/setup", methods=["GET", "POST"])
def setup():
    if request.method == "POST":
        what = (request.get_json(silent=True) or {}).get("what")
        if what == "later":
            d = _setup_done(); d["asked"] = True
            os.makedirs(os.path.dirname(SETUP_P), exist_ok=True)
            json.dump(d, open(SETUP_P, "w", encoding="utf-8"))
        elif what in ("blur", "subs", "both") and not setup_state["running"]:
            threading.Thread(target=run_setup, args=(what,), daemon=True).start()
    return jsonify(done=_setup_done(), state=setup_state)


# ---------------- 자막 컨셉 (팀 기준: 단어 사전·바꿔 쓰기·말투 메모·받아쓰기 힌트·SRT 예시) ----------------
@app.get("/subtitle-style")
def page_profiles():
    return Response(open(os.path.join(BASE, "profiles.html"), encoding="utf-8").read(), mimetype="text/html")


@app.route("/models", methods=["GET", "POST"])
def models_cfg():
    """자막 모델 고르기 (번역·받아쓰기·목소리 걸러내기) + 모은 훈련 자료 수"""
    import models, tm, subs
    if request.method == "POST":
        models.save(request.get_json(silent=True) or {})
    trs = [dict(m, have=bool(m.get("file") and os.path.exists(os.path.join(subs.TR_DIR, m["file"])))) for m in models.TR_MODELS]
    return jsonify(cfg=models.load(), tr=trs, asr=models.ASR_MODELS, p2=models.PASS2, enh=models.ENHANCERS, train=tm.stats())


@app.get("/profiles")
def profiles_list():
    import profiles
    return jsonify(profiles.list_profiles())


@app.route("/profiles/<pid>", methods=["GET", "POST"])
def profile_one(pid):
    import profiles
    if request.method == "POST":
        return jsonify(profiles.save(pid, request.get_json(silent=True) or {}))
    return jsonify(profiles.get(pid))


@app.post("/profiles-new")
def profile_new():
    import profiles
    b = request.get_json(silent=True) or {}
    return jsonify(profiles.create((b.get("name") or "새 컨셉").strip()[:40], b.get("copy_from")))


@app.post("/profiles/<pid>/delete")
def profile_delete(pid):
    import profiles
    return jsonify(ok=profiles.delete(pid))


@app.post("/profiles/<pid>/learn")
def profile_learn(pid):
    import profiles
    fs = [f.read() for f in request.files.getlist("files")]
    try:
        return jsonify(profiles.learn(pid, fs))
    except Exception as e:
        return jsonify(error=str(e)[:200]), 400


@app.get("/review/<jid>/subs.srt")
def review_subs_srt(jid):
    d = _jdir(jid)
    if not d:
        return "없음", 404
    import subs
    p = os.path.join(d, "subs.json")
    if not os.path.exists(p):
        return "자막이 아직 없어요", 404
    lines = json.load(open(p, encoding="utf-8")).get("lines", [])
    field = "src" if request.args.get("field") == "src" else "ko"
    r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
    name = os.path.splitext(os.path.basename(r["src"]))[0] + ("_원문" if field == "src" else "_한국어") + ".srt"
    out = os.path.join(d, name)
    open(out, "w", encoding="utf-8-sig").write(subs.to_srt(lines, field))     # 윈도우 프로그램이 한글을 잘 읽게 BOM 붙임
    return send_file(out, as_attachment=True, download_name=name, mimetype="application/x-subrip")


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
                sp = os.path.join(d, "subs.json")
                out.append({"jid": jid, "name": os.path.basename(r["src"]), "t": os.path.getmtime(rp),
                            "sec": round(r["n"] / r["fps"], 1), "edited": os.path.exists(os.path.join(d, "edits.json")),
                            "subs": os.path.exists(sp), "mode": r.get("mode", "blur")})
            except Exception:
                pass
    out.sort(key=lambda x: -x["t"])
    return jsonify(out[:30])


@app.get("/review/<jid>")
def review_page(jid):
    if not _jdir(jid):
        return "검수할 수 있는 작업이 아니에요 (이번 버전 이후에 만든 작업만 돼요)", 404
    return Response(open(os.path.join(BASE, "review.html"), encoding="utf-8").read(), mimetype="text/html")


@app.get("/review/<jid>/subtitles")
def subs_page(jid):
    if not _jdir(jid):
        return "없는 작업이에요", 404
    return Response(open(os.path.join(BASE, "subs.html"), encoding="utf-8").read(), mimetype="text/html")


@app.get("/review/<jid>/data")
def review_data(jid):
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
    ep = os.path.join(d, "edits.json")
    sg, bp = os.path.join(d, "segments.json"), os.path.join(d, "blur_plan.json")
    return jsonify(meta={k: r[k] for k in ("W", "H", "fps", "n")} | {"params": r.get("params", {}), "name": os.path.basename(r["src"]),
                                                                    "mode": r.get("mode", "blur")},
                   segments=json.load(open(sg)) if os.path.exists(sg) else {},
                   plan=json.load(open(bp)) if os.path.exists(bp) else [],
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


def run_fix(fid, d, burn=False):
    job = state[fid]
    with run_lock:
        os.environ["MOSAIC_FULL"] = "1" if night["on"] else "0"
        try:
            import cover2
            r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
            for k, v in r.get("params", {}).items():
                setattr(cover2, k, v)
            n, fps = r["n"], r["fps"]
            _set_hw(r["src"], job)
            blur = os.path.exists(os.path.join(d, "blur_plan.json"))
            if blur:
                ells = json.load(open(os.path.join(d, "blur_plan.json")))
                segtab = {int(k): v for k, v in json.load(open(os.path.join(d, "segments.json"))).items()}
                ep = os.path.join(d, "edits.json")
                edits = json.load(open(ep)) if os.path.exists(ep) else {}
                ells = cover2.apply_edits(ells, segtab, edits, n)
                polys = json.load(open(os.path.join(d, "polys.json")))
            else:                                          # 자막만 작업: 블러 없이 원본에 자막만
                ells, polys = [[] for _ in range(n)], [[] for _ in range(n)]
            subs_ = None
            if burn:
                import subrender
                cues = subrender.cues_of(os.path.join(d, "subs.json"))
                if not cues:
                    raise RuntimeError("입힐 한국어 자막이 없어요")
                subs_ = {"cues": cues, "style": subrender.load_style()}
            out = os.path.join(d, os.path.splitext(os.path.basename(r["src"]))[0] + ("_fix" if blur else "") + ("_sub" if burn else "")
                               + os.path.splitext(r["src"])[1])
            t0 = time.time()
            times = {}
            res, KP = paint_save(r["src"], ells, polys, out, job, n, r["W"], r["H"], fps, r.get("params", {}), 0, 99, times=times, subs=subs_)
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
    burn = request.args.get("burn") == "1"
    fid = os.path.basename(jid) + "_fix_" + uuid.uuid4().hex[:4]
    state[fid] = {"pct": 0, "msg": "수정본 만들 준비 중", "done": False}
    threading.Thread(target=run_fix, args=(fid, d, burn), daemon=True).start()
    return jsonify(job=fid)


@app.post("/review/<jid>/subs/burn")
def review_subs_burn(jid):
    """자막 화면에서: 자막을 영상에 입혀 저장 (블러 작업이면 검수한 블러도 같이 — 한 번에 저장해서 화질 손해 없음)"""
    d = _jdir(jid)
    if not d:
        return jsonify(error="없는 작업"), 404
    fid = os.path.basename(jid) + "_sub_" + uuid.uuid4().hex[:4]
    state[fid] = {"pct": 0, "msg": "자막 입힐 준비 중", "done": False}
    threading.Thread(target=run_fix, args=(fid, d, True), daemon=True).start()
    return jsonify(job=fid)


@app.route("/sub-style", methods=["GET", "POST"])
def sub_style():
    import subrender
    if request.method == "POST":
        return jsonify(style=subrender.save_style(request.get_json(silent=True) or {}))
    return jsonify(style=subrender.load_style(), fonts=subrender.font_list(), default=subrender.DEFAULT_STYLE)


@app.get("/fontfile/<path:name>")
def font_file(name):
    import subrender
    name = os.path.basename(name)
    for dd in subrender.font_dirs():
        p = os.path.join(dd, name)
        if os.path.exists(p):
            return send_file(p, conditional=True)
    return "없음", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    try:   # 예전에 켜 둔 생성기가 남아 있으면 끄기
        urllib.request.urlopen(f"http://127.0.0.1:{port}/quit", timeout=2).read()
        time.sleep(1.5)
    except Exception:
        pass
    threading.Thread(target=_move_caches, daemon=True).start()
    try:   # 창을 닫아도 남아 있던 자막 프로그램(받아쓰기·번역)이 메모리를 잡고 있으면 끄기
        k = _kill_leftovers()
        if k:
            print(f"  남아 있던 자막 프로그램 {k}개를 정리했어요")
    except Exception:
        pass
    print(f"\n  모자이크 생성기:  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
