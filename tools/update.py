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
    """바탕화면에 '모자이크 생성기' 바로가기"""
    if os.name != "nt":
        return
    bat = os.path.join(ROOT, "start.bat")
    ps = ("$d=[Environment]::GetFolderPath('Desktop');"
          "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d '모자이크 생성기.lnk'));"
          f"$s.TargetPath='{bat}';$s.WorkingDirectory='{ROOT}';$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps])


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "update"
    if what == "install":
        print("프로그램 받는 중…")
        fetch(force=True)
        packages(force=True)
        ask_mode()
        shortcut()
        print("\n설치 끝! 바탕화면의 '모자이크 생성기'로 켜요. (이 폴더의 start.bat 과 같아요)")
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
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n문제가 생겼어요:", e)
        print("이 창을 캡처해서 보내 주세요.")
        sys.exit(1)
