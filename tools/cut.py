"""영상 일부를 원본 형식 그대로 잘라 냄 (다시 압축하지 않음 → 화질 그대로, 몇 분이면 끝)
사용: 30분자르기.bat (앞 30분) / 구간자르기.bat (시작 시간·길이를 물어봄) 위에 영상 파일을 끌어다 놓기"""
import os, sys, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_path(args):
    """끌어다 놓은 파일 경로 찾기. 띄어쓰기·한글이 든 경로가 여러 조각으로 나뉘어 들어와도 다시 붙여서 찾음"""
    cands = []
    if os.name == "nt":                                   # 윈도우가 넘겨준 원래 명령줄(유니코드)에서 직접 읽기
        try:
            import ctypes
            from ctypes import wintypes
            GetCommandLineW = ctypes.windll.kernel32.GetCommandLineW
            GetCommandLineW.restype = wintypes.LPCWSTR
            CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
            CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
            CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
            n = ctypes.c_int()
            av = CommandLineToArgvW(GetCommandLineW(), ctypes.byref(n))
            wargs = [av[k] for k in range(n.value)]
            k = next((k for k, a in enumerate(wargs) if a.lower().endswith("cut.py")), None)
            if k is not None:
                rest = wargs[k + 1:]
                cands += rest + [" ".join(rest)]
        except Exception:
            pass
    cands += list(args) + [" ".join(args)]
    for c in cands:
        c = c.strip().strip('"').strip()
        if c and os.path.isfile(c):
            return c
    return None


def main():
    minutes = float(os.environ.get("CUT_MIN", "30"))
    src = _find_path(sys.argv[1:])
    while not src:                                       # 경로를 못 받았으면 이 창에 직접 끌어다 놓게 함
        print("영상 파일을 이 창에 끌어다 놓은 뒤 엔터를 눌러 주세요.")
        try:
            line = input("> ")
        except Exception:
            return 1
        line = line.strip()
        if line.startswith("& "):                        # 파워셸 창에 끌어다 놓으면 앞에 & 가 붙음
            line = line[2:]
        src = _find_path([line.strip().strip("'")])
        if not src:
            print("그 경로에서 파일을 못 찾았어요:", line)
    import media
    base, ext = os.path.splitext(src)
    start = 0.0
    if os.environ.get("CUT_ASK"):                        # 구간자르기: 시작 시간과 길이를 물어봄
        while True:
            t_in = input("어디서부터 자를까요? (예: 25:00 = 25분, 1:05:30 = 1시간 5분 30초) > ").strip()
            try:
                start = sum(float(x) * 60 ** k for k, x in enumerate(reversed(t_in.split(":"))))
                break
            except ValueError:
                print("시간을 25:00 처럼 적어 주세요.")
        m_in = input(f"몇 분 자를까요? (그냥 엔터 = {int(minutes)}분) > ").strip()
        if m_in:
            try:
                minutes = float(m_in)
            except ValueError:
                pass
        dst = f"{base}_{int(start // 60)}분부터{int(minutes)}분{ext}"
        if os.environ.get("CUT_AUDIO"):                  # 소리만 (보내기 쉽게 작게, 소리 트랙·좌우 채널은 원본 그대로)
            dst = f"{base}_{int(start // 60)}분부터{int(minutes)}분_소리.mka"
            print(f"{int(start // 60)}분 {int(start % 60)}초부터 {int(minutes)}분 소리만 꺼내는 중...")
            r = subprocess.run([media.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-stats",
                                *(["-ss", str(start)] if start else []), "-i", src, "-t", str(minutes * 60),
                                "-map", "0:a", "-vn", "-c:a", "flac", "-ar", "24000", dst])
            if r.returncode != 0 or not os.path.exists(dst):
                print("소리 꺼내기에 실패했어요. 이 창을 캡처해서 보내 주세요.")
                return 1
            print(f"\n완료: {dst}  ({os.path.getsize(dst) / 1e6:.0f}MB) — 이 파일을 보내 주세요")
            return 0
    else:
        dst = f"{base}_앞{int(minutes)}분{ext}"
    info = media.probe(src)
    print(f"원본: {os.path.basename(src)}  ({info.get('codec')} {info.get('pix_fmt')}, 소리 트랙 {info.get('audio', 0)}개)")
    print(f"{int(start // 60)}분 {int(start % 60)}초부터 {int(minutes)}분을 원본 형식 그대로 잘라서 저장하는 중... (다시 압축하지 않아요)")
    t = time.time()
    cmd = [media.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-stats",
           *(["-ss", str(start)] if start else []), "-i", src, "-t", str(minutes * 60),
           "-map", "0:v:0", "-map", "0:a?", "-c", "copy"]
    if info.get("timecode"):
        cmd += ["-timecode", info["timecode"]]
    cmd += [dst]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.exists(dst):
        print("자르기에 실패했어요. 이 창을 캡처해서 보내 주세요.")
        return 1
    out = media.probe(dst)
    print()
    print(f"완료 ({int(time.time() - t)}초): {dst}")
    print(f"결과: {out.get('codec')} {out.get('pix_fmt')}, 소리 트랙 {out.get('audio', 0)}개")
    return 0


if __name__ == "__main__":
    code = main()
    try:
        input("\n엔터를 누르면 창이 닫혀요...")
    except Exception:
        pass
    sys.exit(code)
