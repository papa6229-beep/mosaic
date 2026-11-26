"""저장 단계 속도 점검 (2~3분): 실제 생성기의 저장 단계와 똑같은 방식으로 영상 몇 초 분량만 돌려
'영상 풀기 / 블러 그리기 / 압축' 중 어디가 막히는지 숫자로 보여 줌. 결과 영상은 만들지 않음.
사용: 저장속도점검.bat 위에 영상 파일을 끌어다 놓기 (또는 실행 후 창에 끌어다 놓고 엔터)"""
import os, sys, time, subprocess, multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
N = int(os.environ.get("BENCH_N", "300"))          # 점검에 쓰는 프레임 수 (약 10초 분량)
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def worker(src, start, n, W, H, enc, q, mode="all"):
    """생성기 저장 일꾼과 같은 일: 풀기(그래픽카드 확인된 경우 그래픽카드) → 블러 칠하기 → 압축 프로그램에 넘기기"""
    import app, media, cover2
    app._low_priority()
    info = media.probe(src)
    fmt = media.compose_fmt(info)
    fsz = media.frame_bytes(fmt, W, H)
    d = app.Dec(src, start, n, W, H, info["fps"], fmt, fsz, t0=info.get("t0", 0.0))
    e = None if mode == "read" else subprocess.Popen([media.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{W}x{H}",
                          "-r", info["fps_str"], *media.input_args(info), "-i", "-", *enc, "-threads", "2", "-f", "null", "-"],
                         stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    import numpy as np
    P = cover2.Painter()
    tm = {"read": 0.0, "paint": 0.0, "write": 0.0}
    t0 = time.time(); k = 0
    while True:
        a = time.time(); g = d.read(); tm["read"] += time.time() - a
        if g is None:
            break
        if mode == "read":
            k += 1
            continue
        a = time.time()
        sx_ = np.sin(k / 9) * W * 0.01                      # 실제 영상처럼 블러가 매 장 조금씩 움직임
        x1, y1, x2, y2 = W * 0.42 + sx_, H * 0.45, W * 0.58 + sx_, H * 0.75
        ell = [([(x1 + x2) / 2, (y1 + y2) / 2], [[1 / ((x2 - x1) * 0.8) ** 2, 0], [0, 1 / ((y2 - y1) * 0.8) ** 2]], 1.0, 1)]
        poly = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        yf = media.YuvFrame(g[1], fmt, W, H, info["matrix"], info["range"])
        P.paint_yuv(yf, ell, poly)
        b_ = yf.tobytes()
        tm["paint"] += time.time() - a
        a = time.time(); e.stdin.write(b_); tm["write"] += time.time() - a
        k += 1
    d.close()
    if e:
        e.stdin.close(); e.wait()
    q.put((k, time.time() - t0, tm))


def run(K, src, W, H, total, enc, mode="all"):
    for k_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k_] = "1"
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(src, int(total * (k + 0.5) / (K + 1)), N, W, H, enc, q, mode)) for k in range(K)]
    for p in ps:
        p.start()
    res = [q.get(timeout=1800) for _ in ps]
    for p in ps:
        p.join()
    el = max(r[1] for r in res)
    tm = {k: sum(r[2][k] for r in res) / len(res) for k in ("read", "paint", "write")}
    return sum(r[0] for r in res) / el, tm, el


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
    fmt = media.compose_fmt(info)
    enc, desc = media.encode_args(info, fmt)
    genc = media.gpu_encode_args(info, fmt)
    if genc:
        enc, desc = genc, desc + " · 그래픽카드 압축"
    elif media.GPU_ENC_WHY:
        desc += " · CPU 압축 (" + media.GPU_ENC_WHY + ")"
    out("===== 저장 단계 속도 점검 =====")
    out(f"파일: {os.path.basename(src)} ({info.get('codec')} {info.get('pix_fmt')} {W}x{H}, {info['fps_str']})")
    out(f"압축: {desc}")
    hw = app._set_hw(src)
    out(f"영상 풀기: {'그래픽카드' if hw else 'CPU (그래픽카드로 풀면 결과가 달라지는 영상이라)'}")
    smi = None
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        pass
    out(f"그래픽카드: {smi or '?'}")
    out("다른 프로그램은 쓰지 말고 기다려 주세요 (2~3분)...")
    out("")
    ff = media.ffmpeg_exe()
    s0 = f"{total / 2 / float(info['fps']):.2f}"

    t = time.time()
    subprocess.run([ff, "-v", "error", *(["-hwaccel", "cuda"] if hw else []), "-ss", s0, "-i", src, "-map", "0:v:0",
                    "-frames:v", str(N), "-f", "null", "-"], capture_output=True)
    out(f"A. 영상 풀기만 (ffmpeg 혼자): {N / (time.time() - t):.0f} 프레임/초")
    t = time.time()
    subprocess.run([ff, "-v", "error", *(["-hwaccel", "cuda"] if hw else []), "-ss", s0, "-i", src, "-map", "0:v:0",
                    "-frames:v", str(N), *enc, "-f", "null", "-"], capture_output=True)
    out(f"B. 풀고 바로 압축 (블러 없이, ffmpeg 혼자 — 이 PC의 최고 속도에 가까움): {N / (time.time() - t):.0f} 프레임/초")
    fps, _, _ = run(6, src, W, H, total, enc, "read")
    out(f"D. 생성기가 영상을 풀어서 넘겨받기만 6개 동시 (블러·압축 없이): 합계 {fps:.1f} 프레임/초")
    out("")
    for K in (1, 3, 6):
        fps, tm, el = run(K, src, W, H, total, enc)
        sm = tm["read"] + tm["paint"] + tm["write"] or 1
        out(f"C{K}. 생성기 방식 {K}개 동시: 합계 {fps:.1f} 프레임/초  "
            f"(일꾼 시간 나눔: 풀기 {tm['read'] / sm * 100:.0f}% · 블러 그리기 {tm['paint'] / sm * 100:.0f}% · "
            f"압축에 넘기기 {tm['write'] / sm * 100:.0f}%)")
    out("")
    out(f"참고: 원본 {float(info['fps']):.2f} 프레임/초 영상을 러닝타임만큼 걸려 저장하려면 합계 {float(info['fps']):.0f} 프레임/초가 필요해요.")
    out("끝. 이 창을 캡처하거나, 생성기 폴더의 저장속도점검결과.txt를 보내 주세요.")
    try:
        open(os.path.join(ROOT, "저장속도점검결과.txt"), "w", encoding="utf-8").write("\n".join(lines))
    except Exception:
        pass
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
