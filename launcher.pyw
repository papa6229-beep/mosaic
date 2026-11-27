"""모자이크 생성기 실행기 — 검은 창 없이 켬.
바탕화면 아이콘이 이 파일을 python\\pythonw.exe 로 실행함:
  1) 조용히 업데이트 확인  2) 생성기 본체를 창 없이 켬  3) 브라우저 열기
  4) 시계 옆(알림 영역)에 작은 아이콘 → 열기 / 기록 보기 / 끄기
본체가 남기는 글은 logs\\server.log 에 쌓임 (문제가 생기면 그 파일을 보냄)."""
import os, sys, time, threading, subprocess, webbrowser, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.dirname(sys.executable)
PY = os.path.join(PYDIR, "python.exe") if os.path.exists(os.path.join(PYDIR, "python.exe")) else sys.executable
URL = "http://127.0.0.1:7860"
LOGD = os.path.join(ROOT, "logs")
NOWIN = 0x08000000 if os.name == "nt" else 0          # 검은 창 안 띄움
os.makedirs(LOGD, exist_ok=True)
TMP = os.path.join(ROOT, "tmp")
os.makedirs(TMP, exist_ok=True)
ENV = dict(os.environ, PYTHONUTF8="1", PIP_NO_CACHE_DIR="1", TEMP=TMP, TMP=TMP, PYTHONIOENCODING="utf-8")
SLOG = os.path.join(LOGD, "server.log")
ULOG = os.path.join(LOGD, "update.log")

state = {"msg": "켜는 중…", "srv": None, "quit": False, "icon": None}


def alive():
    try:
        urllib.request.urlopen(URL + "/settings", timeout=1.5).read()
        return True
    except Exception:
        return False


def single():
    """이미 켜져 있으면 브라우저만 열고 끝"""
    if os.name != "nt":
        return True
    import ctypes
    ctypes.windll.kernel32.CreateMutexW(None, False, "MosaicGeneratorLauncher")
    return ctypes.windll.kernel32.GetLastError() != 183


def say(m):
    state["msg"] = m
    ic = state["icon"]
    if ic is not None:
        try:
            ic.title = "모자이크 생성기 — " + m
        except Exception:
            pass


def notify(m):
    ic = state["icon"]
    if ic is not None:
        try:
            ic.notify(m, "모자이크 생성기")
        except Exception:
            pass


def kill_tree(p):
    if p is None or p.poll() is not None:
        return
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, creationflags=NOWIN, timeout=30)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def work():
    say("새 판 확인 중…")
    try:
        with open(ULOG, "w", encoding="utf-8") as f:
            subprocess.run([PY, os.path.join(ROOT, "tools", "update.py"), "update"], cwd=ROOT, env=ENV,
                           stdout=f, stderr=subprocess.STDOUT, creationflags=NOWIN, timeout=3600)
    except Exception:
        pass
    if state["quit"]:
        return
    say("켜는 중…")
    try:
        if os.path.exists(SLOG):
            os.replace(SLOG, os.path.join(LOGD, "server_이전.log"))
    except Exception:
        pass
    lf = open(SLOG, "w", encoding="utf-8", buffering=1)
    srv = subprocess.Popen([PY, os.path.join(ROOT, "app.py")], cwd=ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, creationflags=NOWIN)
    state["srv"] = srv
    t0 = time.time()
    while srv.poll() is None and not alive() and time.time() - t0 < 600:
        time.sleep(0.5)
    if srv.poll() is None:
        say("실행 중 (아이콘을 누르면 열기·끄기)")
        webbrowser.open(URL)
    srv.wait()
    if not state["quit"]:                         # 스스로 꺼졌으면 알리고 기록을 열어 줌
        notify("생성기가 멈췄어요. 기록 파일을 열어 드릴게요 — 캡처해서 보내 주세요")
        try:
            os.startfile(SLOG)
        except Exception:
            pass
        time.sleep(3)
    ic = state["icon"]
    if ic is not None:
        ic.stop()


def main():
    if not single() or alive():
        if not alive():
            time.sleep(4)
        webbrowser.open(URL)
        return
    def load_tray():
        try:
            import pystray as ps_
            from PIL import Image as im_
            return ps_, im_
        except Exception:
            return None, None
    pystray, Image = load_tray()
    if pystray is None:                           # 알림 영역 아이콘 부품이 아직 없으면 업데이트(부품 설치)부터 하고 다시
        try:
            with open(ULOG, "w", encoding="utf-8") as f:
                subprocess.run([PY, os.path.join(ROOT, "tools", "update.py"), "update"], cwd=ROOT, env=ENV,
                               stdout=f, stderr=subprocess.STDOUT, creationflags=NOWIN, timeout=3600)
        except Exception:
            pass
        pystray, Image = load_tray()
    if pystray is None:                           # 그래도 없으면 아이콘 없이 켜기만
        work()
        return

    def open_(icon=None, item=None):
        webbrowser.open(URL)

    def log_(icon=None, item=None):
        try:
            os.startfile(SLOG)
        except Exception:
            pass

    def quit_(icon=None, item=None):
        state["quit"] = True
        say("끄는 중…")
        kill_tree(state["srv"])
        icon.stop()

    try:
        img = Image.open(os.path.join(ROOT, "app.ico"))
    except Exception:
        img = Image.new("RGB", (64, 64), (232, 165, 75))
    icon = pystray.Icon("mosaic", img, "모자이크 생성기 — 켜는 중…", menu=pystray.Menu(
        pystray.MenuItem("열기", open_, default=True),
        pystray.MenuItem(lambda it: state["msg"], None, enabled=False),
        pystray.MenuItem("기록 보기", log_),
        pystray.MenuItem("끄기", quit_)))
    state["icon"] = icon
    threading.Thread(target=work, daemon=True).start()
    icon.run()
    state["quit"] = True
    kill_tree(state["srv"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        with open(os.path.join(LOGD, "launcher_error.log"), "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
