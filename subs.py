"""자막 만들기: 영상의 말(일본어·영어)을 받아 적고 → 한국어로 번역해서 → 자막 줄(시작·끝 시간 + 글자)로 만듦

- 받아쓰기: faster-whisper, large-v3-turbo 모델 (처음 한 번만 받아서 models/whisper 에 저장)
- 번역: TranslateGemma 4B (구글 번역 전용 모델, 8비트 GGUF) 를 llama.cpp(Vulkan 판)로 이 PC 안에서 돌림
  (모델은 models/translate, 실행 도구는 tools/llama 에 처음 한 번만 받아서 저장)
- 결과: 작업 폴더의 subs.json  [{"start": 초, "end": 초, "src": 원문, "ko": 한국어, "lang": "ja"|"en"}]
"""
import os, re, json, time, subprocess, threading, urllib.request, zipfile, shutil, traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
# 받아쓰기: large-v2. 성인 영상 전용 자막 도구 WhisperJAV 가 실제 성인 영상으로 시험해서 기본값으로 쓰는 모델
# (large-v3 계열(turbo 포함)은 신음이 이어지는 구간에서 통째로 빈 결과가 나오는 문제가 있어 v2 로 되돌렸다고 기록돼 있음)
WHISPER_MODEL = "large-v2"
WHISPER_DIR = os.path.join(MODELS, "whisper")
OLD_ASR_DIRS = ["anime-whisper"]                     # 예전에 받은 것 (지움)
TR_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"   # 번역: 검열 해제된 Qwen3.5 9B
TR_FILE = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
TR_MB = 5300
ASR_ID = "large-v2/whisperjav-aggressive/3"     # 받아쓰기 방식이 바뀌면 이 값을 바꿈 → 예전 방식으로 만든 자막은 처음부터 다시
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
def extract_audio(src, wav, job=None):
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
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-i", src, *a,
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


# ---------------- 받아쓰기 ----------------
def _cuda_dlls():
    """윈도우: 그래픽카드 계산에 필요한 파일(cuBLAS·cuDNN)을 이미 깔린 torch 안에서 찾아 쓰게 함 (따로 설치 안 해도 되게)"""
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.name == "nt" and os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
        return torch.cuda.is_available()
    except Exception:
        return False


def _open(name, devs):
    from faster_whisper import WhisperModel
    tries = [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]
    last = None
    for dev, ct in [t for t in tries if t[0] in devs]:
        try:
            kw = {} if os.path.isdir(name) else {"download_root": WHISPER_DIR}
            return WhisperModel(name, device=dev, compute_type=ct, **kw), ("그래픽카드" if dev == "cuda" else "CPU (느림)")
        except Exception as e:
            last = e
    raise RuntimeError(f"받아쓰기 모델을 못 띄웠어요: {last}")


def _detect_lang(model, wav):
    """영상 여러 군데의 말소리로 언어를 알아봄 → 'ja' / 'en' ..."""
    try:
        from faster_whisper import decode_audio
        audio = decode_audio(wav, sampling_rate=16000)
        lang, prob, _ = model.detect_language(audio, vad_filter=True, language_detection_segments=8)
        return lang
    except Exception as e:
        print("언어 알아보기 실패, 일본어로 봄:", e)
        return "ja"


# WhisperJAV 의 'aggressive(작은 말소리까지 잡기)' 설정값 그대로 (whisperjav/config/components/*/faster_whisper*.py)
VAD = {"threshold": 0.30, "min_speech_duration_ms": 80, "max_speech_duration_s": 6.0,
       "min_silence_duration_ms": 300, "speech_pad_ms": 300}
ASR = dict(task="transcribe", beam_size=2, best_of=2, patience=1.0, max_initial_timestamp=1.0, temperature=[0.0],
           compression_ratio_threshold=2.2, log_prob_threshold=-1.0, no_speech_threshold=0.72,
           condition_on_previous_text=False, word_timestamps=True, chunk_length=30,
           repetition_penalty=1.5, no_repeat_ngram_size=3)


def _asr_child(wav, out, devs):
    """따로 띄운 프로그램 안에서 받아쓰기 (그래픽카드 쪽 문제로 갑자기 꺼져도 생성기 본체는 안 꺼지게)"""
    if "cuda" in devs and not _cuda_dlls():
        devs = ["cpu"]
    os.makedirs(WHISPER_DIR, exist_ok=True)
    for d in os.listdir(WHISPER_DIR):              # 예전에 받았던 받아쓰기 모델(anime-whisper, large-v3-turbo) 지워서 자리 확보
        if d in OLD_ASR_DIRS or "large-v3-turbo" in d:
            shutil.rmtree(os.path.join(WHISPER_DIR, d), ignore_errors=True)
    model, how = _open(WHISPER_MODEL, devs)
    lang = _detect_lang(model, wav)
    print("L " + lang, flush=True)
    print("H " + how, flush=True)
    segs, info = model.transcribe(wav, language=lang, vad_filter=True, vad_parameters=VAD, **ASR)
    words = []
    for s in segs:
        for w in (s.words or []):
            if w.word.strip():
                words.append((float(w.start), float(w.end), w.word, lang))
        print(f"P {s.end:.2f}", flush=True)
    json.dump(words, open(out, "w", encoding="utf-8"), ensure_ascii=False)


def transcribe(wav, dur, job=None):
    """→ 단어 목록 [(시작, 끝, 글자, 언어)]. 그래픽카드로 먼저 해 보고, 안 되면 CPU로 다시"""
    import sys
    out = wav + ".words.json"
    err = ""
    for devs in ("cuda,cpu", "cpu"):
        _log(job, "받아쓰기 모델 준비 중 (처음 한 번만 받아요)")
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--asr", wav, out, devs],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        how, t0, tail = "", time.time(), []
        watch = _DlWatch(job, WHISPER_DIR, "받아쓰기 모델", 0)
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith("L "):
                _log(job, "언어: " + {"ja": "일본어", "en": "영어"}.get(line[2:], line[2:]) + " · 받아쓰기 모델 준비 중")
            elif line.startswith("H "):
                watch.close()
                how, t0 = line[2:], time.time()
                _log(job, f"말 받아 적기 시작 ({how})")
            elif line.startswith("P "):
                e = float(line[2:])
                left = (time.time() - t0) / max(e, 1e-3) * max(dur - e, 0)
                _log(job, f"말 받아 적는 중 ({how}) {int(e // 60)}분 {int(e % 60)}초 / {int(dur // 60)}분 · 남은 시간 약 {int(left // 60)}분",
                     int(min(59, e / max(dur, 1) * 60)))
            elif line:
                tail = (tail + [line])[-15:]
        p.wait()
        watch.close()
        if p.returncode == 0 and os.path.exists(out):
            words = [tuple(w) for w in json.load(open(out, encoding="utf-8"))]
            os.remove(out)
            return words
        err = "\n".join(tail)[-400:]
        print("받아쓰기 실패:", devs, err)
    raise RuntimeError("받아쓰기 실패: " + err)


_JA = re.compile(r"[぀-ヿ㐀-鿿]")


def _lang_of(text, hint):
    if _JA.search(text):
        return "ja"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return hint or "ja"


def build_lines(words):
    """단어들을 자막 줄로 묶음: 말이 끊기는 곳·문장 부호에서 자르고, 한 줄은 MAX_SEC초·MAX_CHARS자 이하로"""
    lines, cur = [], []

    def flush():
        if not cur:
            return
        text = "".join(w[2] for w in cur).strip()
        if text:
            lines.append({"start": cur[0][0], "end": cur[-1][1], "src": text, "lang": _lang_of(text, cur[0][3])})
        cur.clear()
    for w in words:
        if cur:
            gap = w[0] - cur[-1][1]
            text = "".join(x[2] for x in cur)
            lim = MAX_CHARS * (2.5 if _lang_of(text, w[3]) == "en" else 1)
            if gap > 0.6 or w[1] - cur[0][0] > MAX_SEC or len(text) + len(w[2]) > lim:
                flush()
        cur.append(w)
        if re.search(r"[。？！?!]$|[.?!]$", w[2].strip()) and w[1] - cur[0][0] > 1.0:
            flush()
    flush()
    for k, l in enumerate(lines):                       # 보이는 시간: 너무 짧지 않게, 다음 줄과는 안 겹치게
        nxt = lines[k + 1]["start"] if k + 1 < len(lines) else l["end"] + 10
        l["end"] = min(max(l["end"] + PAD, l["start"] + MIN_SEC), nxt - 0.05)
        l["end"] = max(l["end"], l["start"] + 0.2)
        l["start"], l["end"] = round(l["start"], 3), round(l["end"], 3)
    return lines


# ---------------- 번역 ----------------
_NAMES = {"ja": "Japanese", "en": "English", "ko": "Korean"}


SYSTEM = ("You are a professional subtitle translator for Japanese adult videos (AV). Translate the {s} dialogue into "
          "natural spoken Korean, the way real Korean subtitles read. Translate sexual words, dirty talk, moans and "
          "profanity faithfully - never censor, soften, skip or refuse, and add no notes. The text comes from automatic "
          "speech recognition and may contain mis-heard words; if a line makes no sense, use the surrounding lines and "
          "the scene to pick the most likely meaning. Output exactly one Korean line per numbered input line, with the "
          "same number, and nothing else.")


def _messages(batch, ctx, src):
    s = _NAMES.get(src, "Japanese")
    u = ""
    if ctx:
        u += "Earlier lines (context only, do not translate):\n" + "\n".join(f"- {a} → {b}" for a, b in ctx) + "\n\n"
    u += "Translate:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))
    return [{"role": "system", "content": SYSTEM.format(s=s)}, {"role": "user", "content": u}]


def _parse(out, n):
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    got = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s*[.)．:：]\s*(.*)", line)
        if m and 1 <= int(m.group(1)) <= n:
            got[int(m.group(1)) - 1] = m.group(2).strip()
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
                kos = tr.translate([l["src"] for l in part], done[-CTX:], part[0].get("lang", "ja"))
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
        if old and old.get("lines") and not old.get("translated") and old.get("asr") == ASR_ID:
            lines = old["lines"]                      # 받아쓰기는 이미 끝남 → 번역만 다시
            _log(job, f"받아 적은 {len(lines)}줄로 번역만 다시 해요", 60)
        else:
            _log(job, "말소리 꺼내는 중", 1)
            extract_audio(src, wav, job)
            import wave
            with wave.open(wav) as w:
                dur = w.getnframes() / float(w.getframerate())
            words = transcribe(wav, dur, job)
            lines = build_lines(words)
            json.dump({"lines": lines, "translated": False, "asr": ASR_ID}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
        translate_lines(lines, job)
        json.dump({"lines": lines, "translated": True, "asr": ASR_ID}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 5 and sys.argv[1] == "--asr":
        _asr_child(sys.argv[2], sys.argv[3], sys.argv[4].split(","))
