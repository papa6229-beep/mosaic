"""자막 만들기: 영상의 말(일본어)을 받아 적고 → 한국어로 번역해서 → 자막 줄(시작·끝 시간 + 글자)로 만듦

- 받아쓰기: WhisperJAV (성인 영상 전용 자막 도구, MIT) 를 뒤에서 불러 씀. 설정은 하나로 고정:
  1차 Qwen3-ASR + 2차 anime-whisper, 둘 다 aggressive(작은 말소리까지), 1차 기준 + 빈 곳은 2차로 채움
  (2026-11-21 같은 테스트 영상으로 여러 프로그램을 비교해서 대사를 가장 덜 놓친 조합)
  2차가 다르게 들은 줄은 'alt' 로 같이 넘겨서 번역할 때 흐름에 맞는 쪽을 고르게 함
- 번역: Qwen3.5 9B (검열 해제판) 를 llama.cpp(Vulkan 판)로 이 PC 안에서 돌림
- 결과: 작업 폴더의 subs.json  [{"start": 초, "end": 초, "src": 원문, "alt": 2차 원문(다를 때만), "ko": 한국어, "lang": "ja"}]
"""
import os, re, json, time, subprocess, threading, urllib.request, zipfile, shutil, traceback

# 모델 받는 새 방식(hf-xet)이 일부 인터넷 환경에서 멈춘 채 안 끝나는 문제 → 일반 방식으로 받음
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
# ---- 받아쓰기 프로그램 (WhisperJAV) — 이 판으로 고정 (같은 결과가 나오게)
WJ_VER = "1.10.2"
WJ_URL = f"https://github.com/meizhong986/WhisperJAV/releases/download/v{WJ_VER}/WhisperJAV-{WJ_VER}-Windows-x86_64.exe"
WJ_DIR = os.path.join(ROOT, "tools", "whisperjav")       # 우리 폴더 안에 조용히 설치
WJ_ARGS = ["--ensemble", "--language", "japanese",
           "--pass1-pipeline", "qwen", "--pass1-sensitivity", "aggressive",
           "--pass1-scene-detector", "semantic", "--pass1-speech-segmenter", "firered-vad",
           "--pass2-pipeline", "qwen", "--pass2-qwen-params", '{"generator_backend": "anime-whisper"}',
           "--pass2-sensitivity", "aggressive",
           "--pass2-scene-detector", "semantic", "--pass2-speech-segmenter", "ten",
           "--merge-strategy", "pass1_primary"]
TR_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"   # 번역: 검열 해제된 Qwen3.5 9B
TR_FILE = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
TR_MB = 5300
ASR_VER = "wj1.10.2/1"   # 받아쓰기 방식이 바뀌면 올림 → 예전 방식으로 만든 자막은 '다시 만들기' 버튼이 뜸


def asr_id():
    return ASR_VER


OLD_TR_FILES = ["translategemma-4b-it-Q8_0.gguf"]   # 예전 번역 모델 (새 모델 받으면 지움)
TR_DIR = os.path.join(MODELS, "translate")
LLAMA_DIR = os.path.join(ROOT, "tools", "llama")
LLAMA_PORT = 8771

MAX_SEC = 6.0        # 자막 한 줄이 화면에 머무는 최대 시간
MAX_CHARS = 32       # 원문 한 줄 최대 글자 수 (일본어 기준, 영어는 2.5배)
MIN_SEC = 0.6        # 너무 짧게 깜빡이지 않게
PAD = 0.15           # 말 끝난 뒤 조금 더 보여 줌 (다음 줄과 겹치지 않는 만큼만)


def _log(job, msg, pct=None):
    if job is not None:
        job["sub_msg"] = msg
        if pct is not None:
            job["sub_pct"] = pct


def _dir_mb(d):
    t = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return t / 1e6


class _DlWatch:
    """모델 받는 동안 폴더 크기를 2초마다 재서 '몇 MB / 몇 MB · 초당 몇 MB' 를 보여 줌"""
    def __init__(self, job, d, label, total_mb):
        self.stop = threading.Event()
        base, t0 = _dir_mb(d), time.time()

        def run():
            last, lt = base, t0
            while not self.stop.wait(2):
                mb = _dir_mb(d) - base
                now = time.time()
                spd = max(0.0, (mb + base - last) / max(now - lt, 1e-3))
                last, lt = mb + base, now
                if mb > 1:
                    tot = f" / 약 {total_mb / 1000:.1f}GB" if total_mb else ""
                    _log(job, f"{label} 받는 중 (처음 한 번만) {mb / 1000:.2f}GB{tot} · 초당 {spd:.0f}MB")
        threading.Thread(target=run, daemon=True).start()

    def close(self):
        self.stop.set()


# ---------------- 소리 꺼내기 ----------------
LEVEL = "anull"     # 소리는 손대지 않음 (WhisperJAV: 소리를 가공하면 오히려 정확도가 떨어질 수 있음)
def extract_audio(src, wav, job=None, ss=None, t=None):
    """원본에서 말소리를 16kHz 모노로 꺼냄 (방송용 MXF처럼 소리 트랙이 여러 개면 앞의 두 트랙을 합침)"""
    import media
    info = media.probe(src)
    ff = media.ffmpeg_exe()
    err = subprocess.run([ff, "-hide_banner", "-i", src], capture_output=True, text=True, encoding="utf-8", errors="ignore").stderr
    auds = [l for l in err.splitlines() if "Stream #" in l and "Audio:" in l]
    if not auds:
        raise RuntimeError("이 영상에는 소리가 없어요")
    mono = lambda l: "mono" in l or "1 channels" in l
    if len(auds) >= 2 and mono(auds[0]) and mono(auds[1]):
        a = ["-filter_complex", "[0:a:0][0:a:1]amerge=inputs=2,pan=mono|c0=0.5*c0+0.5*c1,aresample=async=1:first_pts=0," + LEVEL + "[a]",
             "-map", "[a]"]
    else:   # first_pts=0: 소리가 화면보다 늦게 시작하는 영상이면 앞을 무음으로 채워서 시간을 화면과 맞춤
        a = ["-map", "0:a:0", "-ac", "1", "-af", "aresample=async=1:first_pts=0," + LEVEL]
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0
    # 원본 파일을 처음부터 끝까지 읽어야 해서(방송용 MXF는 수십 GB) 시간이 걸림 → 어디까지 읽었는지 보여 줌
    cut = []
    if ss is not None:                       # 비교용: 이 구간만
        cut = ["-ss", f"{ss:.3f}", "-t", f"{t:.3f}"]
        dur = t
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1", *cut, "-i", src, *a,
                          "-ar", "16000", "-c:a", "pcm_s16le", wav],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    errs = []
    threading.Thread(target=lambda: errs.append(p.stderr.read()), daemon=True).start()
    t0 = time.time()
    for line in p.stdout:
        if line.startswith("out_time_us=") and dur:
            try:
                t = int(line.split("=")[1]) / 1e6
            except ValueError:
                continue
            if t > 1:
                left = (time.time() - t0) / t * max(dur - t, 0)
                _log(job, f"말소리 꺼내는 중 {int(t // 60)}분 / {int(dur // 60)}분 · 남은 시간 약 {int(left // 60)}분 {int(left % 60)}초",
                     1 + int(min(t / dur, 1) * 9))
    p.wait()
    if p.returncode != 0:
        time.sleep(0.2)
        raise RuntimeError("소리 꺼내기 실패: " + ("".join(errs))[-300:])
    return info


# ---------------- 받아쓰기 (WhisperJAV) ----------------
def _wj_python():
    """설치된 WhisperJAV 의 python.exe (우리 폴더 → 사용자가 따로 깐 기본 위치 순)"""
    c = [os.environ.get("WJ_PYTHON", ""), os.path.join(WJ_DIR, "python.exe"),
         os.path.join(os.environ.get("LOCALAPPDATA", ""), "WhisperJAV", "python.exe")]
    return next((x for x in c if x and os.path.exists(x)), None)


def _driver_ok():
    """NVIDIA 드라이버 580 이상인지 (WhisperJAV 1.10 조건). 확인을 못 하면 그냥 진행"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        v = float(out.splitlines()[0].split(".")[0])
        return v >= 580, out.splitlines()[0]
    except Exception:
        return True, ""


def ensure_whisperjav(job=None):
    py = _wj_python()
    if py:
        return py
    if os.name != "nt":
        raise RuntimeError("받아쓰기 프로그램(WhisperJAV)은 윈도우에서만 자동 설치돼요")
    ok, ver = _driver_ok()
    if not ok:
        raise RuntimeError(f"그래픽카드 드라이버가 오래됐어요({ver}). nvidia.com/drivers 에서 최신 드라이버(580 이상)를 "
                           "설치하고 컴퓨터를 다시 켠 뒤 다시 눌러 주세요")
    os.makedirs(os.path.dirname(WJ_DIR), exist_ok=True)
    exe = os.path.join(os.path.dirname(WJ_DIR), f"WhisperJAV-{WJ_VER}-setup.exe")
    if not os.path.exists(exe):
        _download(WJ_URL, exe, job, "받아쓰기 프로그램(WhisperJAV) 설치 파일")
    _log(job, "받아쓰기 프로그램(WhisperJAV) 설치 중 — 처음 한 번만, 10~30분 걸려요", 1)
    t0 = time.time()
    p = subprocess.Popen([exe, "/S", "/D=" + WJ_DIR])         # /D 는 맨 끝, 따옴표 없이 (설치 프로그램 규칙)
    while p.poll() is None:
        time.sleep(5)
        m = int((time.time() - t0) // 60)
        _log(job, f"받아쓰기 프로그램(WhisperJAV) 설치 중 — 처음 한 번만, 10~30분 걸려요 (지금 {m}분째)", 1)
    py = _wj_python()
    if not py:
        raise RuntimeError("받아쓰기 프로그램(WhisperJAV) 설치가 끝나지 않았어요. 다시 눌러 주세요")
    try:
        os.remove(exe)
    except OSError:
        pass
    return py


def _read_srt(path):
    lines = []
    if not path or not os.path.exists(path):
        return lines
    txt = open(path, encoding="utf-8-sig", errors="ignore").read().replace("\r", "")
    for blk in re.split(r"\n\s*\n", txt.strip()):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", blk)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(l.strip() for l in blk[m.end():].strip().splitlines() if l.strip())
        if text:
            lines.append({"start": round(a, 3), "end": round(b, 3), "src": text})
    return lines


def _norm(t):
    return re.sub(r"[\s、。，,.!?！？…・ー〜~「」『』()（）\-]", "", t)


def _attach_alt(main, other):
    """2차가 같은 시간에 다르게 들은 글자를 줄마다 'alt' 로 붙임 (번역할 때 흐름에 맞는 쪽을 고르게)"""
    for l in main:
        hit = [o["src"] for o in other
               if min(l["end"], o["end"]) - max(l["start"], o["start"]) > 0.3 * min(l["end"] - l["start"], o["end"] - o["start"])]
        alt = " ".join(hit)
        if alt and _norm(alt) != _norm(l["src"]):
            l["alt"] = alt
    return main


def transcribe(wav, dur, job=None):
    """WhisperJAV 로 받아 적기 → 자막 줄 목록 [{start, end, src, alt?, lang}]"""
    py = ensure_whisperjav(job)
    out = os.path.join(os.path.dirname(wav), "wj")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    # 생성기는 임시 폴더를 자기 폴더(tmp)로 바꿔 두는데, 그 경로에 한글·띄어쓰기가 있으면 WhisperJAV 가 실패함
    # → WhisperJAV 에는 윈도우 기본 임시 폴더(영문 경로)를 쓰게 하고, CPU 를 1개만 쓰게 하는 설정도 빼서 넘김
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    if os.name == "nt":                     # 생성기가 있는 드라이브(C 말고)의 영문 이름 폴더 — C 드라이브 공간을 안 씀
        wtmp = os.path.splitdrive(ROOT)[0] + "\\mosaic_tmp"
        try:
            os.makedirs(wtmp, exist_ok=True)
            env["TEMP"] = env["TMP"] = wtmp
        except OSError:
            pass
    log = os.path.join(out, "whisperjav_log.txt")
    cmd = [py, "-m", "whisperjav.main", wav, *WJ_ARGS, "--output-dir", out, "--log-file", log]
    _log(job, "말 받아 적기 준비 중 (처음엔 받아쓰기 모델을 받느라 오래 걸려요)", 2)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore",
                         env={**env, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    pas, tail, t0 = 1, [], time.time()
    for line in p.stdout:
        line = line.rstrip()
        if not line or "max_new_tokens" in line or "huggingface.co/docs" in line:
            continue
        tail = (tail + [line])[-20:]
        if re.search(r"Pass 1 completed|Starting Pass 2|Pass 2:", line):
            pas = 2
        m = re.search(r"(\d+) of (\d+) VAD groups transcribed", line)
        if m:
            k, n = int(m.group(1)), max(int(m.group(2)), 1)
            pct = int(((pas - 1) + k / n) / 2 * 57) + 2
            el = time.time() - t0
            _log(job, f"말 받아 적는 중 ({pas}/2차) {k}/{n} 조각 · 지금까지 {int(el // 60)}분", pct)
        elif "Fetching" in line or "Downloading" in line:
            _log(job, f"받아쓰기 모델 받는 중 ({pas}/2차, 처음 한 번만)")
    p.wait()
    if p.returncode == 3:
        raise RuntimeError("그래픽카드를 쓸 수 없어요. nvidia.com/drivers 에서 최신 드라이버(580 이상)를 설치하고 다시 켜 주세요")
    if p.returncode != 0:
        why = ""
        try:                                       # WhisperJAV 가 남긴 실행 기록에서 실패 이유를 꺼내 화면에 보여 줌
            man = json.load(open(os.path.join(out, "whisperjav_run.json"), encoding="utf-8"))
            why = " / ".join(str(f.get("error") or f.get("detail") or "") for f in man.get("files", []))
        except Exception:
            pass
        errs = [t for t in tail if "ERROR" in t or "Error" in t or "error:" in t]
        raise RuntimeError("받아쓰기 실패: " + (why or " / ".join((errs or tail)[-3:]))[-500:])
    def find(suffix):
        return next((os.path.join(out, f) for f in os.listdir(out) if f.endswith(suffix)), None)
    main = _read_srt(find(".merged.whisperjav.srt")) or _read_srt(find(".pass1.srt"))
    other = _read_srt(find(".pass2.srt"))
    if not main:
        raise RuntimeError("받아쓰기 결과가 비어 있어요 (말소리를 못 찾았거나 결과 파일이 없음): " + " / ".join(tail[-3:])[-300:])
    for l in main:
        l["lang"] = "ja"
    return _attach_alt(main, other)


# ---------------- 번역 ----------------
_NAMES = {"ja": "Japanese", "en": "English", "ko": "Korean"}


SYSTEM = ("You are a professional subtitle translator for Japanese adult videos (AV). Translate the {s} dialogue into "
          "natural spoken Korean, the way real Korean subtitles read. Translate sexual words, dirty talk, moans and "
          "profanity faithfully - never censor, soften, skip or refuse, and add no notes. The text comes from automatic "
          "speech recognition and may contain mis-heard words; if a line makes no sense, use the surrounding lines and "
          "the scene to pick the most likely meaning. Some lines also show [alt: ...], what a second recognizer heard "
          "at the same moment; use whichever version (or mix) fits the context. When one numbered line holds different "
          "speakers talking in turn (for example the man asks and the woman answers), put ' || ' between the speakers' "
          "parts in your Korean line; do not use || otherwise. Write only Korean (no kanji, no [alt]). Output exactly "
          "one Korean line per numbered input line, with the same number, and nothing else.")


def _messages(batch, ctx, src):
    s = _NAMES.get(src, "Japanese")
    u = ""
    if ctx:
        u += "Earlier lines (context only, do not translate):\n" + "\n".join(f"- {a} → {b}" for a, b in ctx) + "\n\n"
    u += "Translate:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))
    # t 는 '원문' 또는 '원문 [alt: 2차 원문]' 
    return [{"role": "system", "content": SYSTEM.format(s=s)}, {"role": "user", "content": u}]


def _parse(out, n):
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    got = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s*[.)．:：]\s*(.*)", line)
        if m and 1 <= int(m.group(1)) <= n:
            t = re.sub(r"\[alt:[^\]]*\]?", "", m.group(2))            # 2차 받아쓰기 표시가 따라 나오면 지움
            t = re.sub(r"[\u3040-\u30ff\u3400-\u9fff]+", "", t)      # 한자·가나가 섞여 나오면 지움 (예: 正真的한)
            got[int(m.group(1)) - 1] = re.sub(r"\s{2,}", " ", t).strip()
    return got


def _download(url, dst, job, label):
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "mosaic-generator"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fp:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fp.write(b)
            got += len(b)
            if total:
                _log(job, f"{label} 받는 중 (처음 한 번만) {got >> 20}MB / {total >> 20}MB")
    os.replace(tmp, dst)


def _llama_urls(want):
    """llama.cpp 윈도우용(Vulkan) 압축 파일 주소 후보들. 판은 매일 '미리 보기(Pre-release)'로 올라옴
    1) 깃허브 목록(API)에서 파일 이름으로 찾고  2) 그게 막히면(횟수 제한 등) 새 판 소식(releases.atom)의 판 번호로 주소를 만듦"""
    ua = {"User-Agent": "mosaic-generator"}
    urls = []
    try:
        rels = json.load(urllib.request.urlopen(urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10",
            headers={**ua, "Accept": "application/vnd.github+json"}), timeout=60))
        if isinstance(rels, list):
            urls += [a["browser_download_url"] for r in rels if not r.get("draft")
                     for a in r.get("assets", []) if a["name"].endswith(want)]
    except Exception as e:
        print("llama 목록(API) 실패:", e)
    if not urls:
        try:
            atom = urllib.request.urlopen(urllib.request.Request(
                "https://github.com/ggml-org/llama.cpp/releases.atom", headers=ua), timeout=60).read().decode("utf-8", "ignore")
            for tag in re.findall(r"/releases/tag/(b\d+)", atom):
                u = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-{want}"
                if u not in urls:
                    urls.append(u)
        except Exception as e:
            print("llama 목록(atom) 실패:", e)
    return urls


def ensure_translator(job=None):
    """번역 모델과 실행 도구가 없으면 받아 둠 (한 번만). → (llama-server 경로, 모델 경로)"""
    os.makedirs(TR_DIR, exist_ok=True)
    gguf = os.path.join(TR_DIR, TR_FILE)
    if not os.path.exists(gguf):
        try:
            from huggingface_hub import hf_hub_download
            _log(job, f"번역 모델 받는 중 (처음 한 번만 약 {TR_MB / 1000:.1f}GB)")
            watch = _DlWatch(job, TR_DIR, "번역 모델", TR_MB)
            try:
                p = hf_hub_download(TR_REPO, TR_FILE, local_dir=TR_DIR)
            finally:
                watch.close()
            if os.path.abspath(p) != os.path.abspath(gguf):
                shutil.move(p, gguf)
            for old in OLD_TR_FILES:              # 안 쓰는 예전 번역 모델 지워서 자리 확보
                try:
                    os.remove(os.path.join(TR_DIR, old))
                except OSError:
                    pass
        except Exception:
            _download(f"https://huggingface.co/{TR_REPO}/resolve/main/{TR_FILE}", gguf, job, "번역 모델")
    exe = os.path.join(LLAMA_DIR, "llama-server.exe" if os.name == "nt" else "llama-server")
    if not os.path.exists(exe):
        os.makedirs(LLAMA_DIR, exist_ok=True)
        _log(job, "번역 실행 도구 받는 중 (처음 한 번만)")
        want = "bin-win-vulkan-x64.zip" if os.name == "nt" else "bin-ubuntu-vulkan-x64.tar.gz"
        urls = _llama_urls(want)
        if not urls:
            raise RuntimeError("번역 실행 도구(llama.cpp) 받을 곳을 찾지 못했어요 (인터넷 확인)")
        z = os.path.join(LLAMA_DIR, "llama.zip")
        err = None
        for u in urls[:3]:                    # 갓 올라온 판은 파일이 아직 없을 수 있어서 몇 개 시도
            try:
                _download(u, z, job, "번역 실행 도구")
                err = None
                break
            except Exception as e:
                err = e
        if err:
            raise RuntimeError(f"번역 실행 도구를 받지 못했어요: {err}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(LLAMA_DIR)
        os.remove(z)
        found = None
        for dp, _, fs in os.walk(LLAMA_DIR):                # 압축 안에 폴더가 하나 더 있을 수 있음
            if os.path.basename(exe) in fs:
                found = os.path.join(dp, os.path.basename(exe))
                break
        if not found:
            raise RuntimeError("번역 실행 도구 압축을 풀었는데 llama-server가 없어요")
        exe = found
        if os.name != "nt":
            os.chmod(exe, 0o755)
    else:
        pass
    return exe, gguf


def _find_server():
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    for dp, _, fs in os.walk(LLAMA_DIR):
        if name in fs:
            return os.path.join(dp, name)
    return None


class Translator:
    """llama-server를 띄워 두고 한 줄씩 번역"""

    def __init__(self, job=None, exe=None, gguf=None):
        if exe is None:
            exe, gguf = ensure_translator(job)
            exe = _find_server() or exe
        self.url = f"http://127.0.0.1:{LLAMA_PORT}"
        _log(job, "번역 모델 띄우는 중")
        flags = 0x08000000 if os.name == "nt" else 0          # 윈도우에서 검은 창이 따로 안 뜨게
        self.log = open(os.path.join(TR_DIR, "server.log"), "w", encoding="utf-8", errors="ignore") \
            if os.path.isdir(TR_DIR) else subprocess.DEVNULL
        self.p = subprocess.Popen([exe, "-m", gguf, "--port", str(LLAMA_PORT), "--host", "127.0.0.1", "-ngl", "999",
                                   "-c", "8192"], stdout=self.log, stderr=self.log, creationflags=flags)
        t0 = time.time()
        while time.time() - t0 < 300:
            if self.p.poll() is not None:
                tail = ""
                try:
                    self.log.close()
                    tail = open(os.path.join(TR_DIR, "server.log"), encoding="utf-8", errors="ignore").read()[-300:]
                except Exception:
                    pass
                raise RuntimeError("번역 모델을 못 띄웠어요: " + tail.replace("\n", " / "))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError("번역 모델이 5분 안에 준비되지 않았어요")

    def chat(self, messages, max_tokens):
        body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0.3, "top_p": 0.8, "top_k": 20,
                           "min_p": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"].get("content") or ""

    def translate(self, batch, ctx, src="ja"):
        """여러 줄을 한 번에 (앞 대사를 같이 보여 줘서 흐름에 맞게) → 줄마다 한국어. 번호가 빠진 줄은 한 줄씩 다시"""
        got = _parse(self.chat(_messages(batch, ctx, src), 80 * len(batch) + 64), len(batch))
        res = []
        for i, t in enumerate(batch):
            if not got.get(i):
                one = _parse(self.chat(_messages([t], (ctx + list(zip(batch[:i], res)))[-CTX:], src), 120), 1)
                got[i] = one.get(0, "")
            res.append(got[i])
        return res

    def close(self):
        try:
            self.p.terminate()
            self.p.wait(10)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


BATCH = 8       # 한 번에 번역하는 줄 수
CTX = 4         # 같이 보여 주는 앞 대사 수


def translate_lines(lines, job=None, tr=None):
    own = tr is None
    tr = tr or Translator(job)
    try:
        t0 = time.time()
        bad = 0
        done = []
        for k in range(0, len(lines), BATCH):
            part = lines[k:k + BATCH]
            try:
                kos = tr.translate([l["src"] + (f" [alt: {l['alt']}]" if l.get("alt") else "") for l in part],
                                   done[-CTX:], part[0].get("lang", "ja"))
                bad = 0
            except Exception as e:
                kos = [""] * len(part)
                bad += 1
                if bad >= 3:                  # 연달아 안 되면 번역기 자체 문제 → 빈 자막으로 '완성'이라 하지 않고 멈춤
                    raise RuntimeError(f"번역이 계속 실패해요: {e}")
            for l, ko in zip(part, kos):
                l["ko"] = ko
                done.append((l["src"], ko))
            n = min(k + BATCH, len(lines))
            left = (time.time() - t0) / n * (len(lines) - n)
            _log(job, f"한국어로 번역 중 {n}/{len(lines)}줄 · 남은 시간 약 {int(left // 60)}분 {int(left % 60)}초",
                 60 + int(n * 39 / max(len(lines), 1)))
    finally:
        if own:
            tr.close()
    return lines


TR_VER = "2"     # 번역 방식이 바뀌면 올림 → '번역만 다시 하기' 버튼이 뜸 (받아쓰기는 그대로)


def split_speakers(lines):
    """번역기가 ' || ' 로 표시한 화자 바뀜 처리 (방송본부 자막 형식):
    두 사람이면 한 화면에 두 줄로 '- 가\n- 나', 세 번 이상 주고받으면 시간을 글자 수만큼 나눠 두 사람씩 끊어 보여 줌"""
    out = []
    for l in lines:
        parts = [p.strip(" -") for p in (l.get("ko") or "").split("||") if p.strip(" -")]
        if len(parts) <= 1:
            l["ko"] = parts[0] if parts else (l.get("ko") or "").replace("||", " ").strip()
            out.append(l)
            continue
        groups = [parts[i:i + 2] for i in range(0, len(parts), 2)]
        size = [sum(len(x) for x in g) for g in groups]
        t, dur = l["start"], l["end"] - l["start"]
        for k, g in enumerate(groups):
            e = l["end"] if k == len(groups) - 1 else round(t + dur * size[k] / sum(size), 3)
            n = dict(l, start=round(t, 3), end=e, ko=("- " + g[0] + "\n- " + g[1]) if len(g) == 2 else g[0])
            if k:
                n.pop("alt", None)
            out.append(n)
            t = e
    return out


# ---------------- 전체 ----------------
def make_subtitles(src, out_json, job=None):
    """영상 → subs.json. job 에 sub_msg / sub_pct / sub_done / sub_error 를 기록"""
    d = os.path.dirname(out_json)
    wav = os.path.join(d, "audio16k.wav")
    try:
        old = None
        if os.path.exists(out_json):
            try:
                old = json.load(open(out_json, encoding="utf-8"))
            except Exception:
                old = None
        if (old and old.get("asr") == asr_id() and (old.get("asr_lines") or old.get("lines"))
                and (not old.get("translated") or old.get("tr") != TR_VER)):
            lines = [dict(l) for l in (old.get("asr_lines") or old["lines"])]   # 받아쓰기는 그대로 → 번역만 다시
            for l in lines:
                l.pop("ko", None)
            _log(job, f"받아 적은 {len(lines)}줄로 번역만 다시 해요", 60)
        else:
            _log(job, "말소리 꺼내는 중", 1)
            extract_audio(src, wav, job)
            import wave
            with wave.open(wav) as w:
                dur = w.getnframes() / float(w.getframerate())
            lines = transcribe(wav, dur, job)
            json.dump({"lines": lines, "translated": False, "asr": asr_id()}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
        raw = [dict(l) for l in lines]                    # 받아쓰기 원본 (나중에 번역만 다시 할 때 씀)
        translate_lines(lines, job)
        lines = split_speakers(lines)
        json.dump({"lines": lines, "asr_lines": raw, "translated": True, "asr": asr_id(), "tr": TR_VER}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
        _log(job, f"자막 완성 ({len(lines)}줄)", 100)
        if job is not None:
            job["sub_done"] = True
        return lines
    except Exception as e:
        traceback.print_exc()
        if job is not None:
            job["sub_error"] = str(e)[:300]
            job["sub_done"] = True
        raise
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass


# ---------------- 자막 파일 ----------------
def _ts(t, sep=","):
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)) % 1000:03d}"


def to_srt(lines, field="ko"):
    out = []
    k = 0
    for l in lines:
        text = (l.get(field) or "").strip()
        if not text or l.get("del"):
            continue
        k += 1
        out.append(f"{k}\n{_ts(l['start'])} --> {_ts(l['end'])}\n{text}\n")
    return "\n".join(out)
