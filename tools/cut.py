"""영상 앞부분만 원본 형식 그대로 잘라 냄 (다시 압축하지 않음 → 화질 그대로, 몇 분이면 끝)
사용: 30분자르기.bat 위에 영상 파일을 끌어다 놓기"""
import os, sys, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 2:
        print("영상 파일을 30분자르기.bat 위에 끌어다 놓아 주세요.")
        return 1
    src = sys.argv[1].strip('"')
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    if not os.path.exists(src):
        print("파일을 찾을 수 없어요:", src)
        return 1
    import media
    base, ext = os.path.splitext(src)
    dst = f"{base}_앞{int(minutes)}분{ext}"
    info = media.probe(src)
    print(f"원본: {os.path.basename(src)}  ({info.get('codec')} {info.get('pix_fmt')}, 소리 트랙 {info.get('audio', 0)}개)")
    print(f"앞 {int(minutes)}분을 원본 형식 그대로 잘라서 저장하는 중... (다시 압축하지 않아요)")
    t = time.time()
    cmd = [media.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-stats", "-i", src, "-t", str(minutes * 60),
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
