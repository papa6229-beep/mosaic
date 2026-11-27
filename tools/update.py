"""설치·업데이트 (Git 없이).
  install : 이 폴더에 프로그램을 받고, 필요한 패키지를 깔고, 블러/자막/둘 다 중 고르고, 바탕화면 바로가기를 만듦
  update  : 켤 때마다 — 새 판이 있으면 프로그램 파일만 덮어씀 (작업·설정·모델은 그대로)
모든 것은 이 폴더 안에만 들어감 (파이썬도 python 폴더에). 지우려면 폴더째 지우면 끝."""
import os, sys, io, json, zipfile, subprocess, urllib.request, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "papa6229-beep/mosaic"
ZIP = f"https://github.com/{REPO}/archive/refs/heads/main.zip"
VER = os.path.join(ROOT, "settings", "version.txt")
PY = sys.executable
UA = {"User-Agent": "mosaic-generator"}


def _get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def latest_sha():
    try:
        return json.loads(_get(f"https://api.github.com/repos/{REPO}/commits/main", 20))["sha"]
    except Exception:
        return None


def fetch(force=False):
    """최신 판 받아서 덮어쓰기. 새로 받았으면 True"""
    sha = latest_sha()
    cur = open(VER).read().strip() if os.path.exists(VER) else ""
    if not force and sha and sha == cur:
        return False
    data = _get(ZIP, 300)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.infolist():
            name = m.filename.split("/", 1)[1] if "/" in m.filename else ""
            if not name or m.is_dir():
                continue
            dst = os.path.join(ROOT, *name.split("/"))
            b = z.read(m)
            try:
                if open(dst, "rb").read() == b:    # 바뀐 파일만 씀 (지금 실행 중인 start.bat 을 건드리지 않게)
                    continue
            except OSError:
                pass
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".new"
            with open(tmp, "wb") as f:
                f.write(b)
            os.replace(tmp, dst)
    os.makedirs(os.path.dirname(VER), exist_ok=True)
    open(VER, "w").write(sha or time.strftime("%Y%m%d%H%M"))
    return True


def pip(*args):
    r = subprocess.run([PY, "-m", "pip", "install", "--no-warn-script-location", *args])
    if r.returncode != 0:
        raise RuntimeError("패키지 설치 실패")


def packages(force=False):
    """필요한 패키지 (처음 한 번 + requirements.txt 가 바뀌었을 때만)"""
    req = os.path.join(ROOT, "requirements.txt")
    mark = os.path.join(os.path.dirname(PY), "req.txt")
    new = open(req, encoding="utf-8").read()
    if not force and os.path.exists(mark) and open(mark, encoding="utf-8").read() == new:
        return
    print("\n필요한 프로그램 묶음을 설치하는 중이에요 (처음엔 10~20분, 그래픽카드용 묶음이 커요)…\n")
    pip("--upgrade", "pip")
    pip("torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")   # RTX 50(5060)까지 지원
    pip("-r", req)
    open(mark, "w", encoding="utf-8").write(new)


def ask_mode():
    p = os.path.join(ROOT, "settings", "install.json")
    if os.path.exists(p):
        return
    print("\n이 PC에서 무엇을 쓰실 건가요?")
    print("  1. 블러 + 자막 둘 다")
    print("  2. 블러만")
    print("  3. 자막만")
    m = None
    while m is None:
        a = input("번호를 누르고 엔터 (그냥 엔터 = 1) > ").strip() or "1"
        m = {"1": "both", "2": "blur", "3": "subs"}.get(a)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"mode": m}, open(p, "w", encoding="utf-8"))
    print({"both": "블러 + 자막", "blur": "블러만", "subs": "자막만"}[m], "— 처음 켤 때 필요한 모델을 알아서 받아요.")


def shortcut():
    """바탕화면·시작 메뉴에 '모자이크 생성기' 바로가기 — 검은 창 없는 실행기(launcher.pyw)로"""
    if os.name != "nt":
        return
    name = "모자이크 생성기"
    pyw = os.path.join(os.path.dirname(PY), "pythonw.exe")
    lau = os.path.join(ROOT, "launcher.pyw")
    ico = os.path.join(ROOT, "app.ico")
    if os.path.exists(pyw) and os.path.exists(lau):
        tgt, arg = pyw, f'"{lau}"'
    else:
        tgt, arg = os.path.join(ROOT, "start.bat"), ""
    for where in ("Desktop", "Programs"):
        ps = (f"$d=[Environment]::GetFolderPath('{where}');"
              f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d '{name}.lnk'));"
              f"$s.TargetPath='{tgt}';$s.Arguments='{arg}';$s.WorkingDirectory='{ROOT}';"
              + (f"$s.IconLocation='{ico}';" if os.path.exists(ico) else "") + "$s.Save()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], creationflags=0x08000000)


def check_gpu():
    """엔비디아 그래픽카드와 드라이버 버전 확인 (자막 받아쓰기·번역은 드라이버 580 이상 필요)"""
    if os.name != "nt":
        return
    import shutil
    smi = shutil.which("nvidia-smi") or os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvidia-smi.exe")
    try:
        out = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        name, ver = [x.strip() for x in out.split(",")]
    except Exception:
        print("\n[알림] 엔비디아 그래픽카드를 찾지 못했어요. 그래픽카드 드라이버를 설치한 뒤 다시 켜 주세요.")
        print("        드라이버 받는 곳: https://www.nvidia.com/ko-kr/drivers/")
        return
    print(f"\n그래픽카드: {name} (드라이버 {ver})")
    try:
        old = float(ver.split(".")[0]) < 580
    except ValueError:
        old = False
    if old:
        print("[알림] 그래픽카드 드라이버가 오래됐어요. 자막 기능은 드라이버 580 이상이 필요해요.")
        print("       드라이버 받는 곳(열어 드릴게요): https://www.nvidia.com/ko-kr/drivers/ — 설치 후 PC를 다시 켜 주세요.")
        try:
            os.startfile("https://www.nvidia.com/ko-kr/drivers/")
        except Exception:
            pass


def set_mode(m):
    p = os.path.join(ROOT, "settings", "install.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"mode": m}, open(p, "w", encoding="utf-8"))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "update"
    # 설치 프로그램(모자이크생성기_설치.exe)이 단계별로 부름 — 묻는 것 없이
    if what == "fetch":
        print("프로그램 받는 중…", flush=True)
        fetch(force=True)
        print("받았어요.", flush=True)
        return 0
    if what == "packages":
        packages(force=True)
        return 0
    if what == "finish":
        m = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ("both", "blur", "subs") else "both"
        set_mode(m)
        print({"both": "블러 + 자막", "blur": "블러만", "subs": "자막만"}[m], "— 처음 켤 때 필요한 모델을 알아서 받아요.", flush=True)
        check_gpu()
        return 0
    if what == "install":
        print("프로그램 받는 중…")
        fetch(force=True)
        packages(force=True)
        ask_mode()
        shortcut()
        check_gpu()
        print("\n설치 끝! 바탕화면 아이콘으로 켜요. (이 폴더의 start.bat 과 같아요)")
        return 0
    try:                                   # 켤 때: 새 판 확인 (인터넷이 안 되면 지금 판으로)
        if fetch():
            print("새 판으로 바꿨어요.")
        else:
            print("최신 판이에요.")
    except Exception as e:
        print("새 판 확인 실패 — 지금 판으로 켤게요:", e)
    try:
        packages()
    except Exception as e:
        print("패키지 확인 실패:", e)
    mk = os.path.join(ROOT, "settings", "shortcut2.txt")    # 예전 바로가기(검은 창)를 새 실행기로 한 번 바꿔 줌
    if os.name == "nt" and not os.path.exists(mk) and os.path.exists(os.path.join(os.path.dirname(PY), "pythonw.exe")):
        try:
            shortcut()
            open(mk, "w").write("ok")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n문제가 생겼어요:", e)
        print("이 창을 캡처해서 보내 주세요.")
        sys.exit(1)
