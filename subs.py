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
# 받아쓰기 힌트: 이 말들 쪽으로 더 잘 알아듣게 (Qwen3-ASR context). AV에 자주 나오는 말 + 전에 잘못 들었던 말
JA_CONTEXT = ("アダルトビデオの会話。よく出る言葉: 気持ちいい、気持ちよすぎる、イク、イッちゃう、イキそう、見ちゃう、見ちゃおうかな、"
              "見てもいいですか、恥ずかしい、我慢できない、ダメ、すごい、すごく、大丈夫、触って、舐めて、入れて、動いて、"
              "もっと、奥、中に出して、おっぱい、乳首、おちんちん、まんこ、クリ、お尻、濡れてる、硬い、大きい、気持ちいいところ、"
              "禁欲、男優さん、感触、生、ベッド、パンツ、キス、フェラ、潮吹き、敏感、くすぐったい、エッチ、可愛い、綺麗")
PASS1_MODEL = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"
# 말소리 찾기: WhisperSeg (일본어 ASMR 약 500시간으로 훈련 — 속삭이는 목소리용, WhisperJAV 제공). 두 받아쓰기 모두
VAD_TH = 0.25      # 말소리로 인정하는 기준 (낮을수록 속삭임까지 잡음). WhisperJAV 게시판 #425 JAV 시험 값
WJ_ARGS = ["--ensemble", "--language", "japanese",
           # 1차: 애니·미연시 목소리로 다시 훈련한 Qwen3-ASR (숨소리·속삭임 섞인 목소리를 일반 모델보다 잘 듣게) + 작은 소리도 말로 인정 (기본보다 낮은 기준)
           "--pass1-pipeline", "qwen", "--pass1-qwen-params", json.dumps({"model_id": PASS1_MODEL, "vad_threshold": VAD_TH}),
           "--pass1-sensitivity", "aggressive",
           "--pass1-scene-detector", "semantic", "--pass1-speech-segmenter", "whisperseg",
           "--pass2-pipeline", "qwen", "--pass2-qwen-params", json.dumps({"generator_backend": "anime-whisper", "vad_threshold": VAD_TH}),
           "--pass2-sensitivity", "aggressive",
           "--pass2-scene-detector", "semantic", "--pass2-speech-segmenter", "whisperseg",
           "--merge-strategy", "pass1_primary"]
TR_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"   # 번역: 검열 해제된 Qwen3.5 9B
TR_FILE = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
TR_MB = 5300
ASR_VER = "wj1.10.2/8"   # 8: Galgame(기본으로 보임)+anime-whisper 둘 다 듣고 번역 AI가 고름, 다르게 들은 줄 표시, 놓친 곳만 소리 키워 다시 듣기
ASR_VER_PREV = "wj1.10.2/6"  # 6은 같은 두 모델로 받아쓴 결과 → 합치는 방식만 바꿔 다시 씀 (다시 받아쓰지 않음)


def asr_id(lang="ja"):
    """받아쓰기 방식 + 고른 받아쓰기 모델·목소리 걸러내기 (바뀌면 '다시 만들기')"""
    import models
    if lang == "en":                       # 영어: Whisper(기본 모델) 하나로
        return ASR_VER + "|en"
    k = models.asr_key()
    return ASR_VER if k == PASS1_MODEL + "|none" else ASR_VER + "|" + k


def job_lang(d):
    """작업 폴더에 적어 둔 자막 원문 언어: ja(기본) / en"""
    try:
        v = open(os.path.join(d, "sub_lang.txt"), encoding="utf-8").read().strip()
        return v if v in ("ja", "en") else "ja"
    except OSError:
        return "ja"



def tr_id():
    """번역 방식 + 고른 번역 모델 (바뀌면 '번역 다시')"""
    import models
    k = models.tr_key()
    return TR_VER if k == "q9" else TR_VER + "|" + k


OLD_TR_FILES = ["translategemma-4b-it-Q8_0.gguf"]   # 예전 번역 모델 (새 모델 받으면 지움)
TR_DIR = os.path.join(MODELS, "translate")
LLAMA_DIR = os.path.join(ROOT, "tools", "llama")
LLAMA_CUDA_DIR = os.path.join(ROOT, "tools", "llama-cuda")      # 엔비디아용 (있으면 이걸 먼저 씀)
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


def _wj_args(words=None, lang="ja"):
    """받아쓰기 명령. 일본어: Galgame + anime-whisper 둘 다(기본). 영어: Whisper 하나(영어는 Whisper가 가장 잘 들음)"""
    import models
    if lang == "en":
        return ["--mode", "balanced", "--language", "english", "--sensitivity", "aggressive"]
    if models.pass2() == "none":
        a = ["--mode", "qwen", "--language", "japanese", "--qwen-model-id", models.asr_model_id(),
             "--qwen-segmenter", "whisperseg", "--qwen-sensitivity", "aggressive"]
        if models.enhancer() != "none":
            a += ["--qwen-enhancer", models.enhancer()]
        return a
    a = list(WJ_ARGS)
    i = a.index("--pass1-qwen-params") + 1
    a[i] = json.dumps({"model_id": models.asr_model_id(), "vad_threshold": VAD_TH})
    if models.pass2() == "v2":                    # 2차를 Whisper large-v2 로 (배경음 큰 장면에 강함)
        i0, i1 = a.index("--pass2-pipeline"), a.index("--merge-strategy")
        a[i0:i1] = ["--pass2-pipeline", "balanced", "--pass2-model", "large-v2", "--pass2-sensitivity", "aggressive"]
    if models.enhancer() != "none":
        a += ["--pass1-speech-enhancer", models.enhancer(), "--pass2-speech-enhancer", models.enhancer()]
    return a


def transcribe(wav, dur, job=None, words=None, lang="ja"):
    """WhisperJAV 로 받아 적기 → 자막 줄 목록 [{start, end, src, alt?, lang}]"""
    out = os.path.join(os.path.dirname(wav), "wj")
    _run_wj(wav, _wj_args(words, lang), out, job, 2, 57)
    main = lines_from_wj(out, lang=lang)
    if not main:
        raise RuntimeError("받아쓰기 결과가 비어 있어요 (말소리를 못 찾았거나 결과 파일이 없음)")
    return main


def _run_wj(wav, args, out, job, p0, p1, label="말 받아 적는 중"):
    """WhisperJAV 실행 (진행률 p0~p1%)"""
    py = ensure_whisperjav(job)
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
    cmd = [py, "-m", "whisperjav.main", wav, *args, "--output-dir", out, "--log-file", log]
    two = "--ensemble" in args
    _log(job, "말 받아 적기 준비 중 (처음엔 받아쓰기 모델을 받느라 오래 걸려요)", p0)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore",
                         env={**env, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    pas, tail, t0 = 1, [], time.time()
    for line in p.stdout:
        line = line.rstrip()
        if not line or "max_new_tokens" in line or "huggingface.co/docs" in line:
            continue
        tail = (tail + [line])[-20:]
        if two and re.search(r"Pass 1 completed|Starting Pass 2|Pass 2:", line):
            pas = 2
        m = re.search(r"(\d+) of (\d+) VAD groups transcribed", line)
        if m:
            k, n = int(m.group(1)), max(int(m.group(2)), 1)
            f = ((pas - 1) + k / n) / 2 if two else k / n
            el = time.time() - t0
            _log(job, f"{label}{f' ({pas}/2차)' if two else ''} {k}/{n} 조각 · 지금까지 {int(el // 60)}분", p0 + int(f * (p1 - p0)))
        elif "Fetching" in line or "Downloading" in line:
            _log(job, "받아쓰기 모델 받는 중 (처음 한 번만)")
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


RELISTEN_VER = "1"


def _is_moan(s):
    """신음·숨소리·입 소리로만 적힌 줄 (예: あっ、はぁ / んちゅっ、ふにぃー). 가나만 짧게 있고 작은 글자·ん·ー 가 섞인 것"""
    s = clean_ja(s or "")
    if not s or _MOAN.fullmatch(s):
        return True
    k = re.sub(r"[\s、。,.!?！？…~〜]", "", s)
    return (bool(re.fullmatch(r"[\u3041-\u309f\u30a0-\u30ffー]+", k)) and len(k) <= 8
            and bool(re.search(r"[んっッーぁぃぅぇぉゃゅょァィゥェォャュョ]", k)))


def _regions(wav, lines):
    """자막이 없는 곳(또는 신음·입 소리로만 적힌 줄) 중 실제로 소리가 있는 구간 → [(시작, 끝)] 초"""
    import wave
    import numpy as np
    with wave.open(wav) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    hop = int(sr * 0.05)
    n = len(x) // hop
    if n < 10:
        return [], x, sr
    db = 10 * np.log10((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)
    floor = np.percentile(db, 20)
    act = db > floor + 10                      # 바닥 소음보다 10dB 넘게 큰 곳만 (조용한 곳은 안 들음 → 시간 절약)
    for l in lines:                            # 이미 말로 받아 적은 곳은 다시 안 들음
        if not _is_moan(l["src"]):
            act[max(0, int((l["start"] - 0.15) / 0.05)):int((l["end"] + 0.15) / 0.05) + 1] = False
    out, i = [], 0
    while i < n:
        if not act[i]:
            i += 1
            continue
        j = i
        while j < n and (act[j] or (j + 10 < n and act[j:j + 10].any())):   # 0.5초 안 끊김은 한 덩어리
            j += 1
        a, b_ = i * 0.05, j * 0.05
        if b_ - a >= 0.4:
            out.append((max(0.0, a - 0.3), min(n * 0.05, b_ + 0.3)))
        i = j
    big = []                                   # 너무 긴 덩어리는 20초씩
    for a, b_ in out:
        while b_ - a > 20:
            big.append((a, a + 20))
            a += 20
        big.append((a, b_))
    return big, x, sr


def relisten(wav, lines, job=None):
    """첫 받아쓰기가 놓친 곳(소리는 있는데 자막 없음 · 신음으로만 적힘)만 소리를 키워 같은 모델로 한 번 더 → 찾은 말을 채워 넣음.
    속삭임은 소리가 작아서 말소리 찾기에서 빠지는 일이 많음. 전체가 아니라 그 구간만 들어서 몇 분"""
    import wave
    import numpy as np
    regs, x, sr = _regions(wav, lines)
    if not regs:
        return lines
    gap = np.zeros(int(sr * 1.0), np.float32)
    parts, where, pos = [], [], 0.0
    for a, b_ in regs:
        c = x[int(a * sr):int(b_ * sr)]
        pk = float(np.abs(c).max()) if len(c) else 0
        g = min(20.0, 0.5 / pk) if pk > 0 else 1.0   # 가장 큰 소리가 절반쯤 되게 키움 (최대 20배)
        parts += [np.clip(c * g, -1, 1), gap]
        where.append((pos, pos + len(c) / sr, a))
        pos += len(c) / sr + 1.0
    d = os.path.dirname(wav)
    rw = os.path.join(d, "relisten.wav")
    with wave.open(rw, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.concatenate(parts) * 32767).astype(np.int16).tobytes())
    tot = sum(b_ - a for a, b_ in regs)
    import models
    args = ["--mode", "qwen", "--language", "japanese", "--qwen-model-id", models.asr_model_id(),
            "--qwen-segmenter", "whisperseg", "--qwen-sensitivity", "aggressive"]
    out = os.path.join(d, "wj_re")
    try:
        _run_wj(rw, args, out, job, 57, 60, f"놓친 곳 다시 듣는 중 ({int(tot // 60)}분 {int(tot % 60)}초 분량)")
    finally:
        try:
            os.remove(rw)
        except OSError:
            pass
    srt = next((os.path.join(out, f) for f in os.listdir(out) if f.endswith(".whisperjav.srt")), None) if os.path.isdir(out) else None
    new = []
    for l in _read_srt(srt):
        s = clean_ja(l["src"])
        if not s or _MOAN.fullmatch(s):        # 다시 들어도 신음뿐이면 버림 (짧은 말 'もっと' 같은 건 살림)
            continue
        mid = (l["start"] + l["end"]) / 2
        for s0, s1, a in where:
            if s0 - 0.05 <= mid <= s1 + 0.05:
                st, en = max(l["start"], s0), min(l["end"], s1)
                new.append({"start": round(a + st - s0, 3), "end": round(a + max(en, st + 0.3) - s0, 3),
                            "src": l["src"], "lang": "ja", "re": True})
                break
    if new:
        keep = [l for l in lines if not (_is_moan(l["src"]) and any(_overlaps(l, n) for n in new))]
        lines = sorted(keep + new, key=lambda l: l["start"])
    _log(job, f"놓친 곳 다시 듣기: {len(regs)}곳 중 {len(new)}줄 찾음", 60)
    print("다시 듣기:", len(regs), "곳,", round(tot), "초 →", len(new), "줄")
    return lines


def _overlaps(a, b):
    return min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0.3 * min(a["end"] - a["start"], b["end"] - b["start"])


def lines_from_wj(out, only1=False, lang="ja"):
    """받아쓰기 결과 파일 두 개(1차 Qwen3-ASR, 2차 anime-whisper)로 자막 줄 만들기.
    1차(Galgame)를 기본으로 보임(2026-11-23 실제 영상: 대부분 1차가 맞음). 1차가 비운 곳은 2차로 채우고(one=한 모델만 들음),
    같은 곳을 2차가 다르게 들었으면 'alt'로 붙임 → 번역 AI가 뜻이 통하는 쪽을 골라 번역, 화면엔 '확인 필요'"""
    if not os.path.isdir(out):
        return []
    def find(suffix):
        return next((os.path.join(out, f) for f in os.listdir(out) if f.endswith(suffix)), None)
    if only1 or (not find(".pass1.srt") and find(".whisperjav.srt")):     # 한 번만 받아쓴 결과 (기본)
        one = _read_srt(find(".pass1.srt")) if only1 else _read_srt(find(".whisperjav.srt"))
        for l in one:
            l["lang"] = lang
        return one
    p1 = _read_srt(find(".pass1.srt")) or _read_srt(find(".merged.whisperjav.srt"))
    p2 = _read_srt(find(".pass2.srt"))
    main = [dict(l) for l in p1] + [dict(l, one=True) for l in p2 if not any(_overlaps(l, m) for m in p1)]
    main.sort(key=lambda l: l["start"])
    for l in main:
        l["lang"] = "ja"
        l.pop("alt", None)
    return _attach_alt(main, p2)


# ---------------- 번역 ----------------
_NAMES = {"ja": "Japanese", "en": "English", "ko": "Korean"}


SYSTEM = ("You are a professional subtitle translator for Japanese adult videos (AV). Translate the {s} dialogue into "
          "natural spoken Korean, the way real Korean subtitles read. Translate sexual words, dirty talk, moans and "
          "profanity faithfully - never censor, soften, skip or refuse, and add no notes. The text comes from automatic "
          "speech recognition and may contain mis-heard words; if a line makes no sense, use the surrounding lines and "
          "the scene to pick the most likely meaning. Some lines also show [alt: ...], what a second recognizer heard "
          "at the same moment. One of the two may be mis-heard: translate the reading that makes sense with the surrounding "
          "lines and the scene (they may also be combined to fix a word), but translate the line only once - never add the "
          "other reading as extra text. Some lines contain ‖, which marks where "
          "the speaker changes (found from the voices, e.g. man then woman). Put ' || ' in your Korean line at exactly "
          "those places, keeping the same number of parts in the same order; never add || anywhere else. Keep the speech level of the original: Japanese polite "
          "forms (です, ます, ございます, ください and the like) become Korean polite speech (-요 / -습니다), and only plain "
          "Japanese (だ, る-endings, ね/よ without です/ます) becomes 반말; keep it consistent for the same pair of "
          "speakers in the earlier lines. Keep lines short like real subtitles. "
          "Translate the MEANING of every word; never write Japanese sounds in Hangul (e.g. 奥に is '안쪽에/깊이', not '오쿠니'; "
          "やばい is '큰일이야/미치겠어', not '야바이'), except people's names. Words split by pauses (…) still form one sentence: "
          "translate the whole sentence naturally. "
          "A pure moan or breathing sound (あっ, んっ, はぁ, ああん) gets an empty translation. Inside a line, leave out "
          "sound effects, slurping, kissing and laughing sounds (ちゅ, くちゅくちゅ, じゅるっ, れろ, ぴくぴく, ふふっ, ひふっ) - "
          "translate only the words, never write such sounds in Hangul. Never repeat a phrase, and never add words, "
          "feelings or meaning that are not in the original line. When a TEAM DICTIONARY, "
          "TEAM STYLE NOTES or EXAMPLES are given below, they are this team's rules: follow them. "
          "Write only Korean (no kanji, no [alt]). Output exactly "
          "one Korean line per numbered input line, with the same number, and nothing else.")


def _messages(batch, ctx, src, extra=""):
    s = _NAMES.get(src, "Japanese")
    u = ""
    if ctx:
        u += "Earlier lines (context only, do not translate):\n" + "\n".join(f"- {a} → {b}" for a, b in ctx) + "\n\n"
    u += "Translate:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))
    # t 는 '원문' 또는 '원문 [alt: 2차 원문]' 
    if src == "en":
        extra = ("\nThe source here is ENGLISH (a Western adult video): the notes above about Japanese do not apply. "
                 "Choose 존댓말 or 반말 from the relationship and situation, and keep it consistent for the same pair of speakers.") + extra
    return [{"role": "system", "content": SYSTEM.format(s=s) + extra}, {"role": "user", "content": u}]


def clean_ja(t):
    """번역에 넘기기 전 일본어 정리: 2차 받아쓰기가 붙이는 말줄임표(…)·문장부호만 있는 조각 정리"""
    t = re.sub(r"[.．・]{2,}|…+", "…", t or "")
    t = re.sub(r"^[\s…、。,，.!！?？]+", "", t)
    t = re.sub(r"(…\s*[、。])+", "…", t)
    t = re.sub(r"[\s…、]+$", "", t)
    return t.strip()


_KO_SND = set("아으하응앗읏흐후헉흑음읍앙힝히흣야이우푸츄쪽쭈")


def _ko_sound(w):
    """하하·흐흐·하하야·으으이·히푸 같은 소리 낱말 (같은 글자가 되풀이되거나 푸·츄·쭈가 섞임). '아이'·'아니야' 같은 말은 아님"""
    c = re.sub(r"[,.!?~…]", "", w)
    if c in ("츄", "쭈", "푸", "쪽"):
        return True
    return (len(c) >= 2 and all(ch in _KO_SND for ch in c)
            and (len(set(c)) < len(c) or any(ch in "푸츄쭈" for ch in c)
                 or c in ("하아", "흐응", "으응", "아앙", "하앙", "흐아", "히잉", "아응", "하읏", "흐읏", "후우")))


def clean_ko(t):
    """번역 결과 정리: 일본어 문장부호·쓸데없는 기호 지우기, 줄 앞뒤 말줄임표 지우기 (방송본부 자막처럼). 화자 표시 ' || ' 는 남김"""
    t = (t or "").replace("‖", "||")
    t = re.sub(r"[\u3000-\u303f\uff61-\uff65]", " ", t)                      # 、。「」 같은 일본어 부호
    t = re.sub(r"(?<!\|)\|(?!\|)", " ", t)                                    # 혼자 있는 | 는 지움
    t = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\s.,?!~'\"‘’“”()…\-|]", "", t)        # 화살표·악센트 글자 같은 이상한 기호는 지움
    parts = []
    for p in t.split("||"):
        p = re.sub(r"\.{2,}|…+", "…", p)
        p = re.sub(r"^[\s…,.!?~-]+", "", p)
        p = re.sub(r"[\s…,~]+$", "", p).strip()
        p = re.sub(r"\s*…\s*", " ", p).strip()                                 # 가운데 말줄임표는 띄어쓰기로
        p = re.sub(r"\s{2,}", " ", p)
        p = " ".join(w for w in p.split(" ") if not _ko_sound(w)).strip(" ,")   # 하하·흐흐·하하야 같은 소리 낱말은 뺌
        p = re.sub(r"\s*,\s*,", ",", p).strip(" ,")
        if re.search(r"[가-힣A-Za-z0-9]", p):
            parts.append(p)
    return " || ".join(parts)


def _parse(out, n):
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    got = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s*[.)．:：]\s*(.*)", line)
        if m and 1 <= int(m.group(1)) <= n:
            t = re.sub(r"\[alt:[^\]]*\]?", "", m.group(2))            # 2차 받아쓰기 표시가 따라 나오면 지움
            t = re.sub(r"[\u3040-\u30ff\u3400-\u9fff]+", "", t)      # 한자·가나가 섞여 나오면 지움 (예: 正真的한)
            got[int(m.group(1)) - 1] = clean_ko(t)
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
    """llama.cpp 압축 파일 주소 후보들 → [(실행 도구, CUDA 실행 파일 묶음)]. 판은 매일 '미리 보기(Pre-release)'로 올라옴
    1) 깃허브 목록(API)에서 파일 이름으로 찾고  2) 그게 막히면(횟수 제한 등) 새 판 소식(releases.atom)의 판 번호로 주소를 만듦"""
    ua = {"User-Agent": "mosaic-generator"}
    urls = []
    try:
        rels = json.load(urllib.request.urlopen(urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10",
            headers={**ua, "Accept": "application/vnd.github+json"}), timeout=60))
        if isinstance(rels, list):
            for r in rels:
                if r.get("draft"):
                    continue
                al = {a["name"]: a["browser_download_url"] for a in r.get("assets", [])}
                main = [u for n, u in al.items() if n.startswith("llama-") and n.endswith(want)]
                if main:
                    urls.append((main[0], al.get("cudart-llama-" + want)))
    except Exception as e:
        print("llama 목록(API) 실패:", e)
    if not urls:
        try:
            atom = urllib.request.urlopen(urllib.request.Request(
                "https://github.com/ggml-org/llama.cpp/releases.atom", headers=ua), timeout=60).read().decode("utf-8", "ignore")
            for tag in re.findall(r"/releases/tag/(b\d+)", atom):
                b = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/"
                u = (b + f"llama-{tag}-{want}", b + f"cudart-llama-{want}")
                if u not in urls:
                    urls.append(u)
        except Exception as e:
            print("llama 목록(atom) 실패:", e)
    return urls


def ensure_translator(job=None):
    """번역 모델과 실행 도구가 없으면 받아 둠 (한 번만). → (llama-server 경로, 모델 경로)"""
    import models
    os.makedirs(TR_DIR, exist_ok=True)
    m = models.tr_model()
    TR_REPO_, TR_FILE_, TR_MB_ = m.get("repo", TR_REPO), m.get("file", TR_FILE), m.get("mb", TR_MB)
    gguf = m.get("path") or os.path.join(TR_DIR, TR_FILE_)
    if not os.path.exists(gguf):
        try:
            from huggingface_hub import hf_hub_download
            _log(job, f"번역 모델 받는 중 (처음 한 번만 약 {TR_MB_ / 1000:.1f}GB)")
            watch = _DlWatch(job, TR_DIR, "번역 모델", TR_MB_)
            try:
                p = hf_hub_download(TR_REPO_, TR_FILE_, local_dir=TR_DIR)
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
            _download(f"https://huggingface.co/{TR_REPO_}/resolve/main/{TR_FILE_}", gguf, job, "번역 모델")
    exe = _find_server()
    if os.name == "nt" and _has_nvidia() and not _find_server(LLAMA_CUDA_DIR):
        for cv in ("13.1", "12.4"):            # 엔비디아 그래픽카드면 CUDA판 (Vulkan판은 지포스에서 CPU로 돌아 초당 몇 글자밖에 안 나옴)
            try:                               # 13.1 먼저: RTX 50(5060 등) 지원. 드라이버 580 이상 필요(받아쓰기도 580 이상 필요)
                exe = _get_llama(LLAMA_CUDA_DIR, f"bin-win-cuda-{cv}-x64.zip", job, cudart=True)
                break
            except Exception as e:
                print(f"CUDA {cv}판 번역 도구 실패:", e)
    if not exe:
        exe = _get_llama(LLAMA_DIR, "bin-win-vulkan-x64.zip" if os.name == "nt" else "bin-ubuntu-vulkan-x64.tar.gz", job)
    return exe, gguf


def _has_nvidia():
    if shutil.which("nvidia-smi"):
        return True
    w = os.environ.get("WINDIR", r"C:\Windows")
    return os.path.exists(os.path.join(w, "System32", "nvidia-smi.exe"))


def _get_llama(dest, want, job, cudart=False):
    """llama.cpp 실행 도구를 dest 폴더에 받아 풀기. cudart=True면 같은 판의 CUDA 실행 파일 묶음도 함께"""
    os.makedirs(dest, exist_ok=True)
    _log(job, "번역 실행 도구 받는 중 (처음 한 번만)")
    pairs = _llama_urls(want)
    if not pairs:
        raise RuntimeError("번역 실행 도구(llama.cpp) 받을 곳을 찾지 못했어요 (인터넷 확인)")
    z = os.path.join(dest, "llama.zip")
    err = None
    for main, extra in pairs[:3]:              # 갓 올라온 판은 파일이 아직 없을 수 있어서 몇 개 시도
        try:
            _download(main, z, job, "번역 실행 도구")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            os.remove(z)
            if cudart:
                _download(extra, z, job, "그래픽카드 실행 파일")
                found = _find_server(dest)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(os.path.dirname(found) if found else dest)
                os.remove(z)
            err = None
            break
        except Exception as e:
            err = e
    if err:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"번역 실행 도구를 받지 못했어요: {err}")
    found = _find_server(dest)
    if not found:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError("번역 실행 도구 압축을 풀었는데 llama-server가 없어요")
    if os.name != "nt":
        os.chmod(found, 0o755)
    return found


def _find_server(*dirs):
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    for d in dirs or (LLAMA_CUDA_DIR, LLAMA_DIR):
        for dp, _, fs in os.walk(d):
            if name in fs:
                return os.path.join(dp, name)
    return None


def _kill_orphans():
    """예전에 켰다가 검은 창을 닫으며 남겨진 번역 프로그램(llama-server)을 정리 — 주인(파이썬)이 없는데 그래픽카드 메모리만 붙잡고 있는 것들"""
    if os.name != "nt":
        return
    ps = ("Get-CimInstance Win32_Process -Filter \"name='llama-server.exe'\" | "
          "Where-Object { -not (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue) } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $_.ProcessId }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.split()
        if out:
            print(f"남아 있던 번역 프로그램 {len(out)}개를 정리했어요 (그래픽카드 메모리 비움)")
            time.sleep(2)
    except Exception as e:
        print("남은 번역 프로그램 정리 실패:", e)


class Translator:
    """llama-server를 띄워 두고 한 줄씩 번역"""

    def __init__(self, job=None, exe=None, gguf=None):
        if exe is None:
            exe, gguf = ensure_translator(job)
        self.url = f"http://127.0.0.1:{LLAMA_PORT}"
        _log(job, "번역 모델 띄우는 중")
        try:
            self._start(exe, gguf)
        except RuntimeError:
            vk = _find_server(LLAMA_DIR)
            if not exe.startswith(LLAMA_CUDA_DIR) or not vk:
                raise
            print("CUDA판으로 못 띄움 → Vulkan판으로")          # 드라이버가 너무 옛날이면
            self._start(vk, gguf)
        self.where = self._where()
        print("번역 모델 위치:", self.where)

    def _start(self, exe, gguf):
        flags = 0x08000000 if os.name == "nt" else 0          # 윈도우에서 검은 창이 따로 안 뜨게
        _kill_orphans()
        logp = os.path.join(TR_DIR, "server.log")
        self.log = open(logp, "w", encoding="utf-8", errors="ignore") if os.path.isdir(TR_DIR) else subprocess.DEVNULL
        import models
        gpu = ["-ngl", "999"] if models.tr_model().get("gpu_all", False) else []   # 큰 모델: llama.cpp가 그래픽카드·메모리에 알아서 나눠 올림 (--fit)
        self.p = subprocess.Popen([exe, "-m", gguf, "--port", str(LLAMA_PORT), "--host", "127.0.0.1", *gpu,
                                   "-c", "8192", "--reasoning-budget", "0"], stdout=self.log, stderr=self.log, creationflags=flags)
        t0 = time.time()
        while time.time() - t0 < 300:
            if self.p.poll() is not None:
                tail = ""
                try:
                    self.log.close()
                    tail = open(logp, encoding="utf-8", errors="ignore").read()[-300:]
                except Exception:
                    pass
                raise RuntimeError("번역 모델을 못 띄웠어요: " + tail.replace("\n", " / "))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        self.close()
        raise RuntimeError("번역 모델이 5분 안에 준비되지 않았어요")

    def _where(self):
        """server.log에서 모델이 그래픽카드에 몇 층 올라갔는지 → '그래픽카드 33/33층 (CUDA)' 같은 글"""
        try:
            s = open(os.path.join(TR_DIR, "server.log"), encoding="utf-8", errors="ignore").read()
        except Exception:
            return ""
        m = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", s)
        dev = "CUDA" if "CUDA" in s else ("Vulkan" if "Vulkan" in s else "")
        if m:
            a, b = m[-1]
            return f"그래픽카드 {a}/{b}층" + (f" ({dev})" if dev else "")
        return "CPU" if not dev else dev

    def chat(self, messages, max_tokens):
        body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0.3, "top_p": 0.8, "top_k": 20,
                           "min_p": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r:      # 느린 PC에서도 묶음이 시간 초과로 통째 버려지지 않게
            d = json.load(r)
        self.calls = getattr(self, "calls", 0) + 1
        self.toks = getattr(self, "toks", 0) + int((d.get("usage") or {}).get("completion_tokens") or 0)
        self.secs = getattr(self, "secs", 0.0) + (time.time() - t0)
        msg = d["choices"][0]["message"]
        if (msg.get("reasoning_content") or "").strip() or "<think>" in (msg.get("content") or ""):
            self.thought = getattr(self, "thought", 0) + 1       # 생각하기를 껐는데도 생각을 했는지 기록
        return d["choices"][0]["message"].get("content") or ""

    def speed(self):
        """지금까지 번역 모델 속도 (글자 조각/초, 부른 횟수) — 느릴 때 원인 보려고"""
        return (getattr(self, "toks", 0) / max(0.1, getattr(self, "secs", 0.0)), getattr(self, "calls", 0), getattr(self, "thought", 0))

    def translate(self, batch, ctx, src="ja", extra=""):
        """여러 줄을 한 번에 (앞 대사를 같이 보여 줘서 흐름에 맞게) → 줄마다 한국어. 번호가 빠진 줄은 한 줄씩 다시"""
        got = _parse(self.chat(_messages(batch, ctx, src, extra), 80 * len(batch) + 64), len(batch))
        res = []
        for i, t in enumerate(batch):
            if i not in got:                  # 번호 자체가 빠진 줄만 다시 (일부러 비운 줄 — 신음 등 — 은 그대로)
                one = _parse(self.chat(_messages([t], (ctx + list(zip(batch[:i], res)))[-CTX:], src, extra), 120), 1)
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


BATCH = 16      # 한 번에 번역하는 줄 수 (한 번 부를 때 많이 묶을수록 빠름)
CTX = 4         # 같이 보여 주는 앞 대사 수


FIX_SYSTEM = ("You proofread Japanese speech-recognition output from a Japanese adult video, before it is translated. "
              "Each numbered line is what the main recognizer heard; some also show [alt: ...], what a second recognizer heard "
              "at the same moment. Write the Japanese that was most likely actually said: fix mis-heard words using the alt "
              "text, the earlier lines and common sense (e.g. じゃあみちゃんかな → じゃあ見ちゃうかな, 検疫 → 禁欲 when the scene is "
              "about abstaining, 大友さん → 男優さん). If a line already makes sense, copy it unchanged. Never translate, never "
              "add words that were not said, never merge or drop lines, and keep moans as they are. Output exactly one "
              "Japanese line per numbered input line, with the same number, and nothing else.")


def _parse_ja(out, n):
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    got = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s*[.)．:：]\s*(.*)", line)
        if m and 1 <= int(m.group(1)) <= n:
            got[int(m.group(1)) - 1] = re.sub(r"\[alt:[^\]]*\]?", "", m.group(2)).strip()
    return got


def fix_japanese(lines, tr, job=None):
    """번역 전에 받아 적은 일본어를 앞뒤 대사와 2차 받아쓰기로 바로잡음. 처음 받아 적은 말은 src_asr 에 남김"""
    t0 = time.time()
    done = []
    for k in range(0, len(lines), BATCH):
        part = lines[k:k + BATCH]
        u = ""
        if done:
            u += "Earlier lines (already corrected, context only):\n" + "\n".join("- " + x for x in done[-CTX:]) + "\n\n"
        u += "Correct:\n" + "\n".join(f"{i + 1}. {l['src']}" + (f" [alt: {l['alt']}]" if l.get("alt") else "") for i, l in enumerate(part))
        try:
            got = _parse_ja(tr.chat([{"role": "system", "content": FIX_SYSTEM}, {"role": "user", "content": u}], 80 * len(part) + 64), len(part))
        except Exception:
            got = {}
        for i, l in enumerate(part):
            new = got.get(i, "")
            if new and new != l["src"] and len(new) <= len(l["src"]) * 2 + 10:      # 엉뚱하게 길어지면 안 씀
                l["src_asr"] = l["src"]
                l["src"] = new
            done.append(l["src"])
        n = min(k + BATCH, len(lines))
        left = (time.time() - t0) / n * (len(lines) - n)
        _log(job, f"받아 적은 일본어 다듬는 중 {n}/{len(lines)}줄 · 남은 시간 약 {int(left // 60)}분 {int(left % 60)}초",
             60 + int(n * 15 / max(len(lines), 1)))
    return lines


def translate_lines(lines, job=None, tr=None, f0=None, prof=None, skip_fix=False):
    """f0: 목소리 높이 기록 (voice.pitch_track). 있으면 줄 안에서 남/여가 바뀌는 곳을 ‖ 로 표시해서 번역기에 넘김"""
    import voice
    import profiles
    extra = profiles.prompt_part(prof) if prof else ""
    own = tr is None
    tr = tr or Translator(job)
    try:
        if lines and lines[0].get("lang", "ja") == "ja" and not skip_fix and FIX_SEPARATE:
            fix_japanese(lines, tr, job)              # (꺼 둠) 따로 다듬는 단계 — 번역할 때 같이 바로잡게 바꿈
        for l in lines:
            src = clean_ja(l["src"])
            tn = voice.turns(src, l["start"], l["end"], f0) if f0 is not None else [(None, src)]
            tn = [x for x in tn if re.search(r"[\u3040-\u30ff\u3400-\u9fffA-Za-z0-9]", x[1])] or [(None, src)]   # 부호만 남은 조각은 버림
            l["turns"] = [t[0] for t in tn]
            l["src_t"] = " ‖ ".join(t[1] for t in tn)
        t0 = time.time()
        bad = 0
        done = []
        for k in range(0, len(lines), BATCH):
            part = lines[k:k + BATCH]
            import tm

            def go(ps, ctx):
                """한 묶음 번역. 실패(시간 초과 등)하면 반으로 나눠 다시 — 묶음 통째로 빈 자막이 되지 않게"""
                try:
                    return tr.translate([l["src_t"] + (f" [alt: {l['alt']}]" if l.get("alt") else "") for l in ps],   # 다른 받아쓰기도 같이 줘서 번역하면서 바로잡게
                                        ctx, ps[0].get("lang", "ja"), extra + tm.prompt_for([l["src"] for l in ps]))
                except Exception as e:
                    print("번역 묶음 실패, 나눠서 다시:", len(ps), "줄", e)
                    if len(ps) == 1:
                        fail.append(e)
                        return [""]
                    h = len(ps) // 2
                    a = go(ps[:h], ctx)
                    return a + go(ps[h:], (ctx + [(l["src"], k) for l, k in zip(ps[:h], a)])[-CTX:])
            fail = []
            kos = go(part, done[-CTX:])
            bad = bad + 1 if len(fail) == len(part) else 0
            if bad >= 3:                      # 연달아 통째로 안 되면 번역기 자체 문제 → 빈 자막으로 '완성'이라 하지 않고 멈춤
                raise RuntimeError(f"번역이 계속 실패해요: {fail[-1]}")
            for l, ko in zip(part, kos):
                l["ko"] = profiles.apply_replace(ko, prof) if prof else ko
                l.pop("src_t", None)
                done.append((l["src"], ko))
            n = min(k + BATCH, len(lines))
            left = (time.time() - t0) / n * (len(lines) - n)
            sp_, nc_, th_ = tr.speed()
            _log(job, f"한국어로 번역 중 {n}/{len(lines)}줄 · 남은 시간 약 {int(left // 60)}분 {int(left % 60)}초 · 번역 모델 초당 {sp_:.0f}토큰, {nc_}번 호출"
                 + (f", 생각함 {th_}번" if th_ else ", 생각 안 함") + (f" · {tr.where}" if getattr(tr, "where", "") else ""),
                 75 + int(n * 24 / max(len(lines), 1)))
        alts = [i for i, l in enumerate(lines) if l.get("alt")]     # 다르게 들은 말도 미리 번역해 둠 → 화면의 '이 말로 바꾸기'가 바로 바뀜
        for k in range(0, len(alts), BATCH):
            ids = alts[k:k + BATCH]
            _log(job, f"다르게 들은 말도 미리 번역 중 {min(k + BATCH, len(alts))}/{len(alts)}줄 (검수 때 바로 바꿔 쓰게)", 99)
            try:
                ctx = [(lines[j]["src"], lines[j].get("ko", "")) for j in range(max(0, ids[0] - CTX), ids[0])]
                kos = tr.translate([clean_ja(lines[i]["alt"]) for i in ids], ctx, lines[ids[0]].get("lang", "ja"), extra)
                for i, ko in zip(ids, kos):
                    ko = ko.replace("||", " ").strip()
                    lines[i]["alt_ko"] = profiles.apply_replace(ko, prof) if prof else ko
            except Exception as e:
                print("다르게 들은 말 번역 실패 (검수 때 버튼으로 번역):", e)
    finally:
        if own:
            tr.close()
    return lines


TR_VER = "12"    # 12: 다르게 들은 말도 미리 번역(alt_ko)
POST_VER = "2"   # 번역 뒤 정리(기호 지우기·나누기 등)만 바뀌면 올림 → 번역 모델을 다시 돌리지 않고 몇 초 만에 다시 정리
FIX_VER = "1"
FIX_SEPARATE = False   # 일본어 다듬기를 따로 한 번 더 돌리지 않음 (번역 모델을 두 번 부르느라 2배 걸렸음)    # 일본어 다듬기 방식이 바뀌면 올림 (그대로면 번역 다시 할 때 다듬은 일본어를 다시 씀)     # 번역 방식이 바뀌면 올림 → '번역만 다시 하기' 버튼이 뜸 (받아쓰기는 그대로)


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


MAXL = 16      # 한 줄 최대 글자 수 (방송본부 자막: 보통 8자, 길어도 14자 안팎)
_KO_MOAN = re.compile(r"[\s,.!?~…\-]*(?:(?:[아으하응앗읏흐후헉흑음읍앙힝히흣]|하아|으응)+[\s,.!?~…\-]*)+")   # 한국어가 감탄·신음 소리뿐
_MOAN = re.compile(r"[\sあぁいぃうぅえぇおぉんっッーゝ〜~はハぁふフひヒ…、。！？!?]+")


def _wrap2(t):
    """한 줄이 길면 가운데에 가까운 띄어쓰기에서 두 줄로"""
    if len(t) <= MAXL or " " not in t:
        return [t]
    sp = [i for i, ch in enumerate(t) if ch == " "]
    cm = [i for i in sp if t[i - 1] in ",?" and max(i, len(t) - i) <= MAXL + 4]   # 쉼표·물음표 뒤에서 끊을 수 있으면 거기서
    i = min(cm or sp, key=lambda x: abs(x - len(t) / 2))
    return [t[:i].strip(), t[i + 1:].strip()]


def _chunks(t):
    """두 줄에 안 들어가는 긴 말을 자막 여러 개로: 쉼표·띄어쓰기에서 끊어 한 자막에 두 줄(약 MAXL*2자)씩"""
    words, out, cur = t.split(" "), [], ""
    for j, w in enumerate(words):
        if cur and cur[-1] in ",?" and len(cur) >= MAXL:          # 쉼표·물음표에서 자막을 나눌 수 있으면 나눔
            out.append(cur)
            cur = ""
        if cur and len(cur) + 1 + len(w) > MAXL * 2 - 2:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


def resegment(lines):
    """방송본부 자막처럼: 순수한 신음만 있는 줄은 빼고, 한 줄은 짧게(MAXL자), 한 자막은 두 줄까지 — 넘치면 시간을 글자 수만큼 나눠 여러 자막으로.
    두 사람 대사(- 가 / - 나)는 그대로"""
    out = []
    for l in lines:
        ko = (l.get("ko") or "").strip().rstrip(".")
        if (not ko and _MOAN.fullmatch(l.get("src") or "")) or (ko and _KO_MOAN.fullmatch(ko.replace("\n", " "))):
            continue                                    # 신음·숨소리만 → 자막 안 씀 (방송본부 방식)
        if "\n" in ko or not ko:
            out.append(dict(l, ko=ko))
            continue
        parts = _chunks(ko) if len(ko) > MAXL * 2 else [ko]
        if len(parts) == 1:
            out.append(dict(l, ko="\n".join(_wrap2(ko))))
            continue
        t, dur, tot = l["start"], l["end"] - l["start"], sum(len(p) for p in parts)
        for k, p in enumerate(parts):
            e = l["end"] if k == len(parts) - 1 else round(t + dur * len(p) / tot, 3)
            n = dict(l, start=round(t, 3), end=e, ko="\n".join(_wrap2(p)))
            if k:
                n.pop("alt", None)
            out.append(n)
            t = e
    return out


# ---------------- 전체 ----------------
def make_subtitles(src, out_json, job=None, profile=None, retranslate=False, fresh=False):
    """영상 → subs.json. job 에 sub_msg / sub_pct / sub_done / sub_error 를 기록.
    profile: 자막 컨셉 이름(없으면 기본). retranslate: 받아쓰기는 그대로 두고 번역만 다시 (다른 컨셉으로 바꿀 때)"""
    import profiles
    prof = profiles.get(profile)
    d = os.path.dirname(out_json)
    wav = os.path.join(d, "audio16k.wav")
    lang = job_lang(d)
    aid = asr_id(lang)
    try:
        old = None
        if fresh:                                  # 처음부터: 저장된 받아쓰기·번역 결과를 하나도 안 씀
            for x in ("wj", "wj_re"):
                shutil.rmtree(os.path.join(d, x), ignore_errors=True)
            if os.path.exists(out_json):           # 예전 자막은 화면에서 치우고 한 벌만 남겨 둠
                os.replace(out_json, out_json[:-5] + ".prev.json")
        if os.path.exists(out_json) and not fresh:
            try:
                old = json.load(open(out_json, encoding="utf-8"))
            except Exception:
                old = None
        if (old and not retranslate and old.get("asr") == aid and old.get("tr") == tr_id()
                and old.get("post") != POST_VER and old.get("tr_lines")):
            lines = [dict(l) for l in old["tr_lines"]]          # 번역은 그대로, 정리만 다시 (몇 초)
            for l in lines:
                l["ko"] = clean_ko(profiles.apply_replace(l.get("ko") or "", prof))
            lines = resegment(split_speakers(lines))
            old.update(lines=lines, post=POST_VER)
            old.pop("lines_orig", None)
            json.dump(old, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
            _log(job, f"자막 다시 정리 끝 ({len(lines)}줄)", 100)
            if job is not None:
                job["sub_done"] = True
            return lines
        skip_fix = False
        times = {}
        rebuilt = None
        import models
        wjd = os.path.join(d, "wj")
        if (old and old.get("asr") == ASR_VER_PREV and old.get("asr") != aid and models.pass2() == "anime" and lang == "ja"
                and os.path.isdir(wjd) and any(f.endswith(".pass2.srt") for f in os.listdir(wjd))):
            rebuilt = lines_from_wj(wjd)          # 예전 결과 = 지금과 같은 두 모델·말소리 찾기 → 다시 받아쓰지 않음
        if rebuilt:
            lines = rebuilt
            _log(job, f"받아쓴 결과({len(lines)}줄)를 다시 써요 (처음부터 받아쓰지 않음)", 50)
            if not os.path.exists(wav):
                _log(job, "놓친 곳 다시 듣기용 소리 꺼내는 중", 50)
                extract_audio(src, wav, job)
            t_re = time.time()
            lines = relisten(wav, lines, job)
            times["다시 듣기"] = round(time.time() - t_re)
            json.dump({"lines": lines, "translated": False, "asr": aid}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
        elif (old and old.get("asr") == aid and (old.get("asr_lines") or old.get("lines"))
                and (retranslate or not old.get("translated") or old.get("tr") != tr_id())):
            lines = [dict(l) for l in (old.get("asr_lines") or old["lines"])]   # 받아쓰기는 그대로 → 번역만 다시
            if old.get("tr_lines") and old.get("fix") == FIX_VER and len(old["tr_lines"]) == len(lines):
                for l, t in zip(lines, old["tr_lines"]):       # 다듬어 둔 일본어 다시 씀 (다듬기 단계 건너뜀)
                    l["src"] = t.get("src", l["src"])
                    if t.get("src_asr"):
                        l["src_asr"] = t["src_asr"]
                skip_fix = True
            for l in lines:
                l.pop("ko", None)
            _log(job, f"받아 적은 {len(lines)}줄로 번역만 다시 해요 (컨셉: {prof.get('name', '')})", 60)
        else:
            _log(job, "말소리 꺼내는 중", 1)
            extract_audio(src, wav, job)
            import wave
            with wave.open(wav) as w:
                dur = w.getnframes() / float(w.getframerate())
            t_asr = time.time()
            lines = transcribe(wav, dur, job, prof.get("asr_words"), lang)
            times["받아쓰기"] = round(time.time() - t_asr)
            if lang == "ja":                       # 놓친 곳 다시 듣기는 일본어 모델로만
                t_re = time.time()
                lines = relisten(wav, lines, job)
                times["다시 듣기"] = round(time.time() - t_re)
            json.dump({"lines": lines, "translated": False, "asr": aid}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
        raw = [dict(l) for l in lines]                    # 받아쓰기 원본 (나중에 번역만 다시 할 때 씀)
        f0 = None
        try:                                               # 목소리 높이로 남/여 구분 (하이픈 자막용) — 기록은 작업 폴더에 보관
            import voice
            f0c = os.path.join(d, "voice_f0.npy")
            if not os.path.exists(f0c) and not os.path.exists(wav):
                _log(job, "목소리 구분용 소리 꺼내는 중", 60)
                extract_audio(src, wav, job)
            _log(job, "남자·여자 목소리 구분 중", 60)
            f0 = voice.pitch_track(wav, f0c)
        except Exception:
            traceback.print_exc()
        t_tr = time.time()
        translate_lines(lines, job, f0=f0, prof=prof, skip_fix=skip_fix)
        times["번역"] = round(time.time() - t_tr)
        tr_lines = [{k: v for k, v in l.items() if k not in ("src_t",)} for l in lines]
        lines = resegment(split_speakers([dict(l) for l in lines]))
        json.dump({"lines": lines, "asr_lines": raw, "tr_lines": tr_lines, "translated": True, "asr": aid, "tr": tr_id(),
                   "post": POST_VER, "fix": FIX_VER, "times": times,
                   "profile": prof.get("id"), "profile_name": prof.get("name"), "pver": profiles.version(prof)},
                  open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
        _log(job, f"자막 완성 ({len(lines)}줄) · " + " · ".join(f"{k} {v // 60}분 {v % 60}초" for k, v in times.items()), 100)
        if job is not None:
            job["sub_done"] = True
        return lines
    except Exception as e:
        traceback.print_exc()
        if job is not None:
            job["sub_error"] = str(e)[:300]
            job["sub_done"] = True
        raise
    # 소리 파일(16kHz, 20분에 약 40MB)은 남겨 둠: 검수자가 일본어를 고치면 그 구간 소리를 받아쓰기 공부 자료로 바로 잘라 씀
    # (지우면 고칠 때마다 수십 GB 원본에서 다시 꺼내느라 저장이 몇 분씩 멈춤)


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
