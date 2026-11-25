"""번역 모델(llama.cpp + Qwen3.5 9B 무검열) 받기·띄우기 — 모자이크 생성기의 번역 부분을 이미지 번역기용으로 따로 둔 것"""
import os, re, json, time, shutil, zipfile, subprocess, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
TR_DIR = os.path.join(ROOT, "models", "translate")
LLAMA_DIR = os.path.join(ROOT, "tools", "llama")
LLAMA_CUDA_DIR = os.path.join(ROOT, "tools", "llama-cuda")
LLAMA_PORT = 8772                       # 모자이크 생성기(8771)와 같은 PC에서 같이 켜도 안 부딪치게
TR_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
TR_FILE = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
TR_MB = 5300


def _log(job, msg, pct=None):
    if job is not None:
        job["msg"] = msg
    print(msg)


def ensure_translator(job=None):
    """번역 모델과 실행 도구가 없으면 받아 둠 (한 번만). → (llama-server 경로, 모델 경로)"""
    os.makedirs(TR_DIR, exist_ok=True)
    gguf = os.path.join(TR_DIR, TR_FILE)
    if not os.path.exists(gguf):
        _download(f"https://huggingface.co/{TR_REPO}/resolve/main/{TR_FILE}", gguf, job, f"번역 모델(약 {TR_MB / 1000:.1f}GB)")
    exe = _find_server()
    if os.name == "nt" and _has_nvidia() and not _find_server(LLAMA_CUDA_DIR):
        for cv in ("13.1", "12.4"):            # 엔비디아면 CUDA판 (13.1: RTX 50 지원, 드라이버 580 이상)
            try:
                exe = _get_llama(LLAMA_CUDA_DIR, f"bin-win-cuda-{cv}-x64.zip", job, cudart=True)
                break
            except Exception as e:
                print(f"CUDA {cv}판 번역 도구 실패:", e)
    if not exe:
        exe = _get_llama(LLAMA_DIR, "bin-win-vulkan-x64.zip" if os.name == "nt" else "bin-ubuntu-vulkan-x64.tar.gz", job)
    return exe, gguf


def _download(url, dst, job, label):
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "image-translator"})
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
    ua = {"User-Agent": "image-translator"}
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
        gpu = ["-ngl", "999"]                                  # 9B 모델은 그래픽카드에 다 올림
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

    def close(self):
        try:
            self.p.terminate()
            self.p.wait(10)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass
