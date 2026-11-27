"""저장 단계 속도 점검 (2~3분): 영상 가운데 20초만 잘라서, 실제 생성기의 저장 단계를
새 방식과 예전 방식으로 각각 돌려 1초에 몇 장 저장하는지 비교함. 결과 영상은 지우고 숫자만 남김.
사용: 저장속도점검.bat 위에 영상 파일을 끌어다 놓기 (또는 실행 후 창에 끌어다 놓고 엔터)"""
import os, sys, time, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
SEC = 20
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def run_one(app, clip, n, W, H, fps, ells, polys, d, ovl):
    app.OVL_SAVE = ovl
    os.makedirs(d, exist_ok=True)
    t = {}
    t0 = time.time()
    app.paint_save(clip, ells, polys, os.path.join(d, "x_cover.mp4"), {}, n, W, H, fps, {}, 0, 99, times=t)
    el = time.time() - t0
    return el, t


def main():
    from cut import _find_path
    src = _find_path(sys.argv[1:])
    while not src:
        print("점검할 영상 파일을 이 창에 끌어다 놓은 뒤 엔터를 눌러 주세요.")
        try:
            line = input("> ").strip()
        except Exception:
            return 1
        if line.startswith("& "):
            line = line[2:]
        src = _find_path([line.strip().strip("'")])
    import cv2, media, app
    info = media.probe(src)
    W, H = info["W"], info["H"]
    cap = cv2.VideoCapture(src); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 3000; cap.release()
    fps = float(info["fps"])
    work = os.path.join(ROOT, "tmp", "bench_save")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    ext = os.path.splitext(src)[1]
    clip = os.path.join(work, "clip" + ext)
    mid = total / 2 / fps
    out("===== 저장 단계 속도 점검 =====")
    out(f"파일: {os.path.basename(src)} ({info.get('codec')} {info.get('pix_fmt')} {W}x{H}, {info['fps_str']})")
    out(f"영상 가운데 {SEC}초를 잘라서 새 방식 / 예전 방식으로 각각 저장해 봐요. 다른 프로그램은 쓰지 말고 기다려 주세요 (2~4분)...")
    subprocess.run([media.ffmpeg_exe(), "-y", "-v", "error", "-ss", f"{mid:.2f}", "-i", src, "-t", str(SEC), "-map", "0", "-c", "copy", clip],
                   capture_output=True)
    cap = cv2.VideoCapture(clip); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if n <= 0:
        out("영상을 자르지 못했어요.")
        return 1
    app._set_hw(clip)
    # 실제 영상처럼: 화면 가운데쯤 블러 하나가 천천히 움직임 (반지름은 화면 높이의 14%)
    import numpy as np
    r = H * 0.14
    ells, polys = [], []
    for f in range(n):
        cx, cy = W * (0.5 + 0.05 * np.sin(f / 40)), H * (0.55 + 0.03 * np.cos(f / 55))
        ells.append([[[cx, cy], [[1 / (r * 0.85) ** 2, 0], [0, 1 / r ** 2]], 1.0, 0]])
        polys.append([[[cx - r * 0.4, cy - r * 0.6], [cx + r * 0.4, cy - r * 0.6], [cx + r * 0.4, cy + r * 0.6], [cx - r * 0.4, cy + r * 0.6]]])
    res = {}
    for name, ovl in (("새 방식", True), ("예전 방식", False)):
        el, t = run_one(app, clip, n, W, H, fps, ells, polys, os.path.join(work, "new" if ovl else "old"), ovl)
        res[name] = n / el
        how = t.get("save_mode") or "예전 방식"
        out(f"{name}: {n}장을 {el:.1f}초에 저장 → 1초에 {n / el:.1f}장  ({how} · {(t.get('out_fmt') or '').split(' (')[0]})")
    out("")
    if res.get("예전 방식"):
        out(f"새 방식이 예전 방식보다 {res['새 방식'] / res['예전 방식']:.2f}배 빠름")
    out(f"참고: 이 영상은 1초에 {fps:.2f}장이라, 1초에 {fps:.0f}장을 저장하면 저장 시간이 영상 길이와 같아요.")
    out("끝. 이 창을 캡처하거나, 생성기 폴더의 저장속도점검결과.txt를 보내 주세요.")
    try:
        open(os.path.join(ROOT, "저장속도점검결과.txt"), "w", encoding="utf-8").write("\n".join(lines))
    except Exception:
        pass
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        r = main()
    except Exception:
        import traceback
        traceback.print_exc()
        out("점검 중 오류가 났어요. 이 창을 캡처해서 보내 주세요.")
        r = 1
    if os.name == "nt":
        input("\n엔터를 누르면 닫혀요")
    sys.exit(r)
