"""속도 점검: 생성기의 각 단계를 따로따로 돌려 이 PC에서 무엇이 느린지 숫자로 보여 줌 (결과물은 만들지 않음)
사용: 속도점검.bat 위에 영상 파일을 끌어다 놓기 (또는 실행 후 창에 끌어다 놓고 엔터)"""
import os, sys, time, subprocess, platform, multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

N = 240          # 단계마다 쓸 프레임 수 (약 8초 분량)
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def fake_entries(W, H):
    """가짜 모자이크 하나 (화면 가운데, 실제 작업과 비슷한 크기)"""
    x1, y1, x2, y2 = W * 0.42, H * 0.45, W * 0.58, H * 0.75
    return x1, y1, x2, y2


# ---------- 일꾼들 (여러 개 동시 점검용) ----------
def w_pipe(src, start, n, W, H, fmt, q):
    import app
    app._low_priority()
    import media
    fsz = media.frame_bytes(fmt, W, H) if fmt != "bgr24" else W * H * 3
    info = media.probe(src)
    d = app.Dec(src, start, n, W, H, info["fps"], fmt, fsz, t0=info.get("t0", 0.0), mi=info)
    t = time.time(); k = 0
    while d.read() is not None:
        k += 1
    d.close()
    q.put((k, time.time() - t))


def w_scan(src, start, n, W, H, q):
    import app, numpy as np
    app._low_priority()
    import cover2, media
    info = media.probe(src)
    d = app.Dec(src, start, n, W, H, info["fps"], t0=info.get("t0", 0.0), mi=info)
    x1, y1, x2, y2 = fake_entries(W, H)
    A = cover2.Analyzer()
    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    t = time.time(); k = 0
    while True:
        g = d.read()
        if g is None:
            break
        f = np.frombuffer(g[1], np.uint8).reshape(H, W, 3)
        A.analyze(f, g[0], [(1, [x1, y1, x2, y2], poly, True)], None)
        k += 1
    d.close()
    q.put((k, time.time() - t))


def w_paint(src, start, n, W, H, q, to_disk=None):
    import app, numpy as np
    app._low_priority()
    import cover2, media
    info = media.probe(src)
    fmt = media.compose_fmt(info)
    enc, _ = media.encode_args(info, fmt)
    fsz = media.frame_bytes(fmt, W, H)
    d = app.Dec(src, start, n, W, H, info["fps"], fmt, fsz, t0=info.get("t0", 0.0))
    dst = ["-f", "nut", to_disk] if to_disk else ["-f", "null", "-"]
    e = subprocess.Popen([media.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{W}x{H}",
                          "-r", info["fps_str"], *media.input_args(info), "-i", "-", *enc, "-threads", "2", *dst],
                         stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    x1, y1, x2, y2 = fake_entries(W, H)
    c = [(x1 + x2) / 2, (y1 + y2) / 2]
    rx, ry = (x2 - x1) * 0.8, (y2 - y1) * 0.8
    ell = [(c, [[1 / rx ** 2, 0], [0, 1 / ry ** 2]], 1.0, 1)]
    poly = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
    P = cover2.Painter()
    t = time.time(); k = 0
    while True:
        g = d.read()
        if g is None:
            break
        yf = media.YuvFrame(g[1], fmt, W, H, info["matrix"], info["range"])
        P.paint_yuv(yf, ell, poly)
        e.stdin.write(yf.tobytes())
        k += 1
    d.close(); e.stdin.close(); e.wait()
    if to_disk:
        try:
            os.remove(to_disk)
        except Exception:
            pass
    q.put((k, time.time() - t))


def _one_cpu_env():
    """실제 생성기(_run_parallel)와 똑같이: 일꾼들의 계산 도구가 CPU 하나만 쓰게"""
    for k_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k_] = "1"


def parallel(target, K, src, n, W, H, total, *extra):
    _one_cpu_env()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    starts = [int(total * (k + 0.5) / (K + 1)) for k in range(K)]
    ps = [ctx.Process(target=target, args=(src, s, n, W, H) + extra + (q,) if target is not w_paint else (src, s, n, W, H, q) + extra)
          for s in starts]
    t = time.time()
    for p in ps:
        p.start()
    res = [q.get(timeout=1800) for _ in ps]
    for p in ps:
        p.join()
    el = max(r[1] for r in res)                     # 프로그램 띄우는 시간은 빼고, 실제 일한 시간만
    frames = sum(r[0] for r in res)
    return frames / el, el


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
    import cv2, media, app, cover2
    info = media.probe(src)
    W, H = info["W"], info["H"]
    cap = cv2.VideoCapture(src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000
    cap.release()
    K = app._cpu_workers(10 ** 6)
    out("===== 모자이크 생성기 속도 점검 =====")
    out(f"파일: {os.path.basename(src)} ({info.get('codec')} {info.get('pix_fmt')} {W}x{H}, {total}프레임)")
    out(f"CPU {os.cpu_count()}개 · 동시 작업 {K}개 · 파이썬 {platform.python_version()} · OS {platform.platform()}")
    v = subprocess.run([media.ffmpeg_exe(), "-version"], capture_output=True, text=True).stdout.splitlines()
    out(f"ffmpeg: {v[0] if v else '?'}")
    out("다른 프로그램은 쓰지 말고 기다려 주세요 (약 5분)...")
    out("")

    # 1. 디스크 읽기
    t = time.time(); got = 0
    with open(src, "rb", buffering=0) as fp:
        fp.seek(int(os.path.getsize(src) * 0.6))
        while got < 400 * 1024 * 1024:
            b = fp.read(8 * 1024 * 1024)
            if not b:
                break
            got += len(b)
    out(f"1. 디스크에서 파일 읽기: {got / (time.time() - t) / 1e6:.0f} MB/초")

    # 2. ffmpeg 혼자 영상 풀기
    st = f"{total * 0.3 / info['fps']:.3f}"
    t = time.time()
    subprocess.run([media.ffmpeg_exe(), "-v", "error", "-ss", st, "-i", src, "-frames:v", str(N), "-f", "null", "-"])
    out(f"2. 영상 풀기만 (ffmpeg 혼자, CPU 다 씀): {N / (time.time() - t):.0f} 프레임/초")
    t = time.time()
    subprocess.run([media.ffmpeg_exe(), "-v", "error", "-threads", "2", "-ss", st, "-i", src, "-frames:v", str(N), "-f", "null", "-"])
    out(f"3. 영상 풀기만 (CPU 2개): {N / (time.time() - t):.0f} 프레임/초")

    # 4~5. 파이썬으로 넘겨받기
    for fmt in ("bgr24", media.compose_fmt(info)):
        fsz = media.frame_bytes(fmt, W, H) if fmt != "bgr24" else W * H * 3
        t = time.time()
        p = subprocess.Popen([media.ffmpeg_exe(), "-v", "error", "-threads", "2", "-ss", st, "-i", src, "-frames:v", str(N),
                              "-f", "rawvideo", "-pix_fmt", fmt, "-"], stdout=subprocess.PIPE, bufsize=fsz * 4)
        k = 0
        while len(p.stdout.read(fsz)) == fsz:
            k += 1
        p.wait()
        out(f"4. 풀어서 파이썬으로 넘기기 ({fmt}, 한 장 {fsz / 1e6:.1f}MB): {k / (time.time() - t):.0f} 프레임/초")
    import queue as _q
    q = _q.Queue()
    w_pipe(src, int(total * 0.3), N, W, H, "bgr24", q)
    k, el = q.get()
    out(f"5. 생성기 방식으로 풀어 넘기기 (프레임 번호 맞추기·색 바꾸기 포함): {k / el:.0f} 프레임/초")

    # 6. 털 찾기 / 블러 칠하기 / 저장 (한 개)
    w_scan(src, int(total * 0.3), N, W, H, q)
    k, el = q.get()
    out(f"6. 털 찾기 단계 1개 (풀기 포함): {k / el:.1f} 프레임/초 ({el / max(k, 1) * 1000:.0f}ms/장)")
    w_paint(src, int(total * 0.3), N, W, H, q)
    k, el = q.get()
    out(f"7. 블러 칠하고 압축 1개 (풀기 포함, 저장 안 함): {k / el:.1f} 프레임/초 ({el / max(k, 1) * 1000:.0f}ms/장)")
    import numpy as np
    raw = subprocess.run([media.ffmpeg_exe(), "-v", "error", "-ss", st, "-i", src, "-frames:v", "60", "-f", "rawvideo",
                          "-pix_fmt", media.compose_fmt(info), "-"], capture_output=True).stdout
    fmt = media.compose_fmt(info); fsz = media.frame_bytes(fmt, W, H)
    x1, y1, x2, y2 = fake_entries(W, H)
    c = [(x1 + x2) / 2, (y1 + y2) / 2]; rx, ry = (x2 - x1) * 0.8, (y2 - y1) * 0.8
    P = cover2.Painter(); t = time.time(); k = 0
    for i in range(len(raw) // fsz):
        yf = media.YuvFrame(raw[i * fsz:(i + 1) * fsz], fmt, W, H, info["matrix"], info["range"])
        P.paint_yuv(yf, [(c, [[1 / rx ** 2, 0], [0, 1 / ry ** 2]], 1.0, 1)], [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]); k += 1
    out(f"8. 블러 칠하기 계산만 1개: {(time.time() - t) / max(k, 1) * 1000:.0f}ms/장")
    enc, desc = media.encode_args(info, fmt)
    t = time.time()
    e = subprocess.Popen([media.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{W}x{H}",
                          "-r", info["fps_str"], *media.input_args(info), "-i", "-", *enc, "-threads", "2", "-f", "null", "-"],
                         stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for _ in range(3):
        for i in range(len(raw) // fsz):
            e.stdin.write(raw[i * fsz:(i + 1) * fsz])
    e.stdin.close(); e.wait()
    out(f"9. 원본 형식으로 압축만 1개 ({desc}): {3 * (len(raw) // fsz) / (time.time() - t):.0f} 프레임/초")
    out("")

    # 10~13. 여러 개 동시 (실제 작업과 같은 상황)
    fps, el = parallel(w_pipe, K, src, N, W, H, total, "bgr24")
    out(f"10. {K}개 동시에 풀어 넘기기: 합계 {fps:.0f} 프레임/초")
    fps, el = parallel(w_scan, K, src, N, W, H, total)
    out(f"11. {K}개 동시에 털 찾기: 합계 {fps:.1f} 프레임/초")
    fps, el = parallel(w_paint, K, src, N, W, H, total)
    out(f"12. {K}개 동시에 블러 칠하고 압축 (저장 안 함): 합계 {fps:.1f} 프레임/초")
    tmpd = os.path.join(ROOT, "jobs")
    os.makedirs(tmpd, exist_ok=True)
    _one_cpu_env()
    ctx = mp.get_context("spawn"); qq = ctx.Queue()
    ps = [ctx.Process(target=w_paint, args=(src, int(total * (k + 0.5) / (K + 1)), N, W, H, qq, os.path.join(tmpd, f"_bench{k}.nut")))
          for k in range(K)]
    t = time.time()
    for p in ps:
        p.start()
    res = [qq.get(timeout=1800) for _ in ps]
    for p in ps:
        p.join()
    out(f"13. {K}개 동시에 블러 칠하고 압축해서 디스크에 저장: 합계 {sum(r[0] for r in res) / max(r[1] for r in res):.1f} 프레임/초")
    out("")
    out("끝. 이 창을 캡처하거나, 생성기 폴더의 속도점검결과.txt를 보내 주세요.")
    try:
        open(os.path.join(ROOT, "속도점검결과.txt"), "w", encoding="utf-8").write("\n".join(lines))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        out("점검 중 오류가 났어요. 이 창을 캡처해서 보내 주세요.")
        code = 1
    try:
        input("\n엔터를 누르면 창이 닫혀요...")
    except Exception:
        pass
    sys.exit(code)
